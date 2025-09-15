import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx
import optimistix as optx

from tqdm import tqdm

from ..coolprop import Fluid
from .. import helpers_props as jxp



# -------------------------------------------------------------------------
# 
# FluidBicubic: main features and implementation choices
#
# - Encapsulated in an Equinox module: handles table generation, loading,
#   and interpolation, with static vs dynamic fields separated.
# 
# - Uses NumPy for table generation and JAX for interpolation (JIT, autodiff, vmap).
# 
# - Table management: loads from pickle if available, otherwise generates
#   and saves to ./fluid_tables/{fluid_name}_{Nh}x{Np}.pkl (default path).
# 
# - Grid: uniform in enthalpy and log(pressure), enabling direct index
#   lookup without search for p-h function calls
# 
# - Properties stored in a table dictionary with:
#     value   : property value
#     grad_h  : df/dh (finite diff, forward)
#     grad_p  : df/dp w.r.t actual P (not logP)
#     grad_ph : mixed derivative d2f/(dhdP)
#     coeffs  : 16 bicubic coefficients per grid cell
# 
# - Derivatives computed via small adaptive steps (eps_h, eps_p).
# 
# - Progress bar shown during table generation (tqdm).
# 
# - Bicubic coefficients computed in one vectorized step using the fixed
#   16x16 transformation matrix A_MAT.
# 
# - Interpolation:
#     * Input (h, p) mapped to fractional indices in (h, logP) space.
#     * Basis = outer product of x-powers and y-powers.
#     * Dot product with stored coefficients gives interpolated values.
# 
# - Vectorization: scalar interpolant wrapped in jax.vmap for batch queries;
#   h and p broadcast to common shape.
# 
# - get_props returns a FluidState object with all canonical properties
#   (NaN if unavailable).
# 
# - Error handling:
#     * Failed table points are NaNs (no errors are raised), success rate reported.
#     * Out-of-bounds queries currently clipped.
#
# TODOs / open points:
# - Explain why interpolated pressure is not exact but enthalpy is.
# - Consider support for other input pairs (e.g. P-T, ρ-h) via root-finding
#   or a more general 2D search.
# 
# -------------------------------------------------------------------------


class FluidBicubic(eqx.Module):
    """
    Fluid model using bicubic property interpolation on an enthalpy-pressure grid.

    Workflow:
      * On construction, attempts to load a precomputed table from disk.
      * If not found, generates the table with finite-difference derivatives,
        saves it to disk, and stores it in memory.
      * On `get_props`, performs bicubic interpolation at the requested state.

    Parameters
    ----------
    fluid_name : str
        Fluid identifier for CoolProp.
    backend : str
        CoolProp backend string.
    h_min, h_max : float
        Min/max enthalpy [J/kg].
    p_min, p_max : float
        Min/max pressure [Pa].
    N_h, N_p : int
        Number of grid points in h and p.
    table_name : str, optional
        Name of the table pickle file (default: "{fluid_name}_{N_h}x{N_p}").
    table_dir : str, optional
        Directory for saving/loading table pickle (default: "fluid_tables").
    identifier : str, optional
        Tag stored in the returned FluidState objects.

    Notes
    -----
    * Table generation uses forward finite differences for first and mixed
      derivatives with adaptive step size.
    * Currently supports only HmassP_INPUTS as input pair.
    """
    # Attributes
    fluid_name: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    h_min: float = eqx.field(static=True)
    h_max: float = eqx.field(static=True)
    p_min: float = eqx.field(static=True)
    p_max: float = eqx.field(static=True)
    N_h: int = eqx.field(static=True)
    N_p: int = eqx.field(static=True)
    table_name: str = eqx.field(static=True)
    table_dir: str = eqx.field(static=True)
    table: dict = eqx.field(static=False)
    identifier: str = eqx.field(static=True)

    # New field: store batched interpolation function
    _batched_interp: callable = eqx.field(static=True)


    # ---------------------------
    # Constructor
    # ---------------------------
    def __init__(
        self,
        fluid_name: str,
        backend: str,
        h_min: float,
        h_max: float,
        p_min: float,
        p_max: float,
        N_h: int,
        N_p: int,
        identifier: str = None,
        table_name: str = None,
        table_dir: str = "fluid_tables",
    ):
        # Initialize variables
        self.fluid_name = fluid_name
        self.backend = backend
        self.h_min, self.h_max, self.N_h = h_min, h_max, N_h
        self.p_min, self.p_max, self.N_p = p_min, p_max, N_p
        self.identifier = identifier or fluid_name
        self.table_name = table_name or f"{fluid_name}_{N_h}x{N_p}"
        self.table_dir = table_dir

        # Create the table if it does not exist
        self.table = self._load_or_generate_table()
        self._batched_interp = jax.vmap(calculate_props, in_axes=(None, 0, 0, None))

    # ---------------------------
    # Table generation
    # ---------------------------
    def _load_or_generate_table(self):
        pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                table = pickle.load(f)
            print(f"Loaded property table from: {pkl_path}")
            return table

        print("No existing table found, generating new one...")
        return self._generate_property_table()

    def _generate_property_table(self):
        """Generate a property table on an enthalpy–pressure grid."""
        fluid = Fluid(self.fluid_name, self.backend)
        h_vals = np.linspace(self.h_min, self.h_max, self.N_h)
        logPvals = np.linspace(np.log(self.p_min), np.log(self.p_max), self.N_p)

        delta_h = h_vals[1] - h_vals[0]
        delta_logP = logPvals[1] - logPvals[0]

        table = {
            "h_vals": h_vals,
            "p_vals": np.exp(logPvals),
            "metadata": dict(
                fluid=self.fluid_name,
                backend=self.backend,
                h_range=(self.h_min, self.h_max),
                p_range=(self.p_min, self.p_max),
                N_h=self.N_h,
                N_p=self.N_p,
            ),
        }

        for k in jxp.PROPERTIES_CANONICAL:
            table[k] = {
                "value": np.zeros((self.N_h, self.N_p)),
                "grad_h": np.zeros((self.N_h, self.N_p)),
                "grad_p": np.zeros((self.N_h, self.N_p)),
                "grad_ph": np.zeros((self.N_h, self.N_p)),
                "coeffs": np.zeros((self.N_h - 1, self.N_p - 1, 16)),
            }

        total_points = self.N_h * self.N_p
        success_count = 0
        os.makedirs(self.table_dir, exist_ok=True)

        with tqdm(
            total=total_points,
            desc="Generating property table",
            ascii=True,
            ncols=80,
            bar_format="{l_bar}{bar}|",
        ) as pbar:
            for i, h in enumerate(h_vals):
                for j, logP in enumerate(logPvals):
                    # TODO: Ask Simone about pressure spacing
                    p = np.exp(logP)
                    eps_h = max(1e-6 * abs(h), 1e-3 * delta_h)
                    eps_p = max(1e-6 * abs(p), 1e-3 * (np.exp(delta_logP) - 1.0) * p)

                    try:
                        f_0 = fluid.get_state(jxp.HmassP_INPUTS, h, p)
                        f_h = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p)
                        f_p = fluid.get_state(jxp.HmassP_INPUTS, h, p + eps_p)
                        f_ph = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p + eps_p)
                        success_count += 1
                    except Exception:
                        f_0 = np.nan
                        f_h = np.nan
                        f_p = np.nan
                        f_ph = np.nan

                    for k in jxp.PROPERTIES_CANONICAL:
                        value = f_0[k]
                        grad_h = (f_h[k] - f_0[k]) / eps_h
                        grad_p = (f_p[k] - f_0[k]) / eps_p
                        grad_ph = (f_ph[k] - f_h[k] - f_p[k] + f_0[k]) / (eps_h * eps_p)
                        table[k]["value"][i, j] = value
                        table[k]["grad_h"][i, j] = grad_h
                        table[k]["grad_p"][i, j] = grad_p
                        table[k]["grad_ph"][i, j] = grad_ph

                    pbar.update(1)

        # Compute coefficients after filling values and derivatives for all i,j
        for k in jxp.PROPERTIES_CANONICAL:
            table[k]["coeffs"] = compute_coefficients(
                value=table[k]["value"],
                grad_h=table[k]["grad_h"],
                grad_p=table[k]["grad_p"],
                grad_hp=table[k]["grad_ph"],
                delta_h=delta_h,
                delta_logP=delta_logP,
            )

        frac_success = success_count / total_points * 100
        print(
            f"Successfully evaluated {success_count}/{total_points} points "
            f"({frac_success:.2f} %)"
        )

        # Convert all lists to jnp arrays
        table["h_vals"] = jnp.array(table["h_vals"])
        table["p_vals"] = jnp.array(table["p_vals"])
        for k in jxp.PROPERTIES_CANONICAL:
            for sub in ["value", "grad_h", "grad_p", "grad_ph", "coeffs"]:
                table[k][sub] = jnp.array(table[k][sub])

        # save to pickle
        pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(table, f)
        print(f"Saved property table to: {pkl_path}")

        return table


    # ---------------------------
    # Table interpolation
    # ---------------------------
    @eqx.filter_jit
    def get_props(self, input_pair: str, h, p) -> jxp.FluidState:
        """Return interpolated property state at (h, p).
        Supports scalar or broadcastable JAX arrays.
        Always returns arrays.
        """

        # Broadcast h and p to the same shape
        h_arr, p_arr = jnp.broadcast_arrays(h, p)

        # Vectorized version of the scalar interpolant
        props = self._batched_interp(input_pair, h_arr.ravel(), p_arr.ravel(), self.table)

        # Reshape back to broadcasted shape
        props = {k: v.reshape(h_arr.shape) for k, v in props.items()}

        return jxp.FluidState(
            fluid_name=self.fluid_name,
            identifier=self.identifier,
            **props,
        )


# TODO: it seems that having jax.lax.cond slows things a lot!
# A better aproach could be to do the mapping like in perect gas

# Taken from perfect gas
# PROPERTY_CALCULATORS = {
#     jxp.PT_INPUTS: calculate_properties_PT,
#     jxp.HmassSmass_INPUTS: calculate_properties_hs,
#     jxp.HmassP_INPUTS: calculate_properties_hP,
#     jxp.PSmass_INPUTS: calculate_properties_Ps,
#     jxp.DmassHmass_INPUTS: calculate_properties_rhoh,
#     jxp.DmassP_INPUTS: calculate_properties_rhop,
# }



# Unified entry point
def calculate_props(input_pair, x, y, table):
    # return interpolate_bicubic_hp(x, y, table)
    return jax.lax.cond(
        input_pair == jxp.HmassP_INPUTS,
        lambda _: interpolate_bicubic_hp(x, y, table),
        lambda _: interpolate_bicubic_xy(input_pair, x, y, table),
        operand=None,
    )


# fmt: off
# Bicubic interpolation matrix [TODO provide reference in docs]
# https://en.wikipedia.org/wiki/Bicubic_interpolation
A_MAT = np.array([
    [+1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [-3.0, +3.0, +0.0, +0.0, -2.0, -1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+2.0, -2.0, +0.0, +0.0, +1.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, -3.0, +3.0, +0.0, +0.0, -2.0, -1.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +2.0, -2.0, +0.0, +0.0, +1.0, +1.0, +0.0, +0.0],
    [-3.0, +0.0, +3.0, +0.0, +0.0, +0.0, +0.0, +0.0, -2.0, +0.0, -1.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, -3.0, +0.0, +3.0, +0.0, +0.0, +0.0, +0.0, +0.0, -2.0, +0.0, -1.0, +0.0],
    [+9.0, -9.0, -9.0, +9.0, +6.0, +3.0, -6.0, -3.0, +6.0, -6.0, +3.0, -3.0, +4.0, +2.0, +2.0, +1.0],
    [-6.0, +6.0, +6.0, -6.0, -3.0, -3.0, +3.0, +3.0, -4.0, +4.0, -2.0, +2.0, -2.0, -2.0, -1.0, -1.0],
    [+2.0, +0.0, -2.0, +0.0, +0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +2.0, +0.0, -2.0, +0.0, +0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +1.0, +0.0],
    [-6.0, +6.0, +6.0, -6.0, -4.0, -2.0, +4.0, +2.0, -3.0, +3.0, -3.0, +3.0, -2.0, -1.0, -2.0, -1.0],
    [+4.0, -4.0, -4.0, +4.0, +2.0, +2.0, -2.0, -2.0, +2.0, -2.0, +2.0, -2.0, +1.0, +1.0, +1.0, +1.0],
], dtype=np.float64)
# fmt: on


def compute_coefficients(value, grad_h, grad_p, grad_hp, delta_h, delta_logP):
    """
    Compute bicubic interpolation coefficients for one property on a uniform (h, logP) grid.

    Parameters
    ----------
    value : ndarray of shape (Nh, Np)
        Property values f(h, P) evaluated on the grid.
    grad_h : ndarray of shape (Nh, Np)
        Partial derivative with respect to enthalpy evaluated on the grid.
    grad_p : ndarray of shape (Nh, Np)
        Partial derivative with respect to pressure evaluated on the grid (with respect to P, not logP).
    grad_hp : ndarray of shape (Nh, Np)
        Mixed derivative with respect to enthalpy and pressure evaluated on the grid.
    delta_h : float
        Uniform grid spacing in h.
    delta_logP : float
        Uniform grid spacing in logP.

    Returns
    -------
    coeffs : ndarray of shape (Nh-1, Np-1, 16)
        Bicubic coefficients for each cell. The ordering is c[4*n + m],
        where m is the power of x (h-direction) and n is the power of y (logP-direction).
    """

    Nh, Np = value.shape
    coeffs = np.zeros((Nh - 1, Np - 1, 16), dtype=np.float64)
    # delta_logP = np.exp(delta_logP)  # TODO: what should be the delta_p used in this function?

    for i in range(Nh - 1):
        for j in range(Np - 1):
            # Assemble the 16 basis values for this cell
            xx = np.array(
                [
                    value[i, j],
                    value[i + 1, j],
                    value[i, j + 1],
                    value[i + 1, j + 1],
                    grad_h[i, j] * delta_h,
                    grad_h[i + 1, j] * delta_h,
                    grad_h[i, j + 1] * delta_h,
                    grad_h[i + 1, j + 1] * delta_h,
                    grad_p[i, j] * delta_logP,
                    grad_p[i + 1, j] * delta_logP,
                    grad_p[i, j + 1] * delta_logP,
                    grad_p[i + 1, j + 1] * delta_logP,
                    grad_hp[i, j] * delta_h * delta_logP,
                    grad_hp[i + 1, j] * delta_h * delta_logP,
                    grad_hp[i, j + 1] * delta_h * delta_logP,
                    grad_hp[i + 1, j + 1] * delta_h * delta_logP,
                ],
                dtype=np.float64,
            )

            # Dense matvec multiplication
            coeffs[i, j, :] = A_MAT @ xx

    return coeffs

    # # Alternative vectorized form with similar execution time
    # # Assemble the 16 basis values for each cell:
    # # [f00, f10, f01, f11, fx00, fx10, fx01, fx11,
    # #  fy00, fy10, fy01, fy11, fxy00, fxy10, fxy01, fxy11]
    # XX = np.stack(
    #     [
    #         value[:-1, :-1],
    #         value[1:, :-1],
    #         value[:-1, 1:],
    #         value[1:, 1:],
    #         grad_h[:-1, :-1] * delta_h,
    #         grad_h[1:, :-1] * delta_h,
    #         grad_h[:-1, 1:] * delta_h,
    #         grad_h[1:, 1:] * delta_h,
    #         grad_p[:-1, :-1] * delta_logP,
    #         grad_p[1:, :-1] * delta_logP,
    #         grad_p[:-1, 1:] * delta_logP,
    #         grad_p[1:, 1:] * delta_logP,
    #         grad_hp[:-1, :-1] * delta_h * delta_logP,
    #         grad_hp[1:, :-1] * delta_h * delta_logP,
    #         grad_hp[:-1, 1:] * delta_h * delta_logP,
    #         grad_hp[1:, 1:] * delta_h * delta_logP,
    #     ],
    #     axis=-1,
    # )

    # # Multiply by the fixed 16x16 matrix A_MAT to get coefficients
    # # einsum: 'ab,ijb->ija' → sum over b, output shape (Nh-1, Np-1, 16)
    # coeffs = np.einsum("ab,ijb->ija", A_MAT, XX)
    # return coeffs



def interpolate_bicubic_hp(h, p, table):
    """
    Bicubic interpolation for a single (h, p) query.

    Parameters
    ----------
    h : float
        Enthalpy [J/kg].
    p : float
        Pressure [Pa].
    table : dict
        Property table with fields "h_vals", "p_vals", and per-property coeffs.

    Returns
    -------
    dict
        Dictionary of interpolated property values at (h, p).
    """
             
    # Extract uniform enthalpy and pressure grids from table
    h_vals = table["h_vals"]          # (Nh,) enthalpy grid
    p_vals = table["p_vals"]          # (Np,) pressure grid
    logPvals = jnp.log(p_vals)       # work in logP for smoother interpolation

    Nh = h_vals.shape[0]
    Np = logPvals.shape[0]

    # Compute uniform grid spacing
    h_min, h_max = h_vals[0], h_vals[-1]
    logPmin, logPmax = logPvals[0], logPvals[-1]
    delta_h = (h_max - h_min) / (Nh - 1)
    delta_logP = (logPmax - logPmin) / (Np - 1)

    # Convert query point (h, logP) into continuous indices in grid space
    ii = (h - h_min) / delta_h
    jj = (jnp.log(p) - logPmin) / delta_logP

    # Clamp values to prevent extrapolation
    # Select lower-left cell index, clamped to table bounds
    # TODO: Decide and implement extrapolation behavior
    i = jnp.clip(jnp.floor(ii).astype(int), 0, Nh - 2)
    j = jnp.clip(jnp.floor(jj).astype(int), 0, Np - 2)

    # Compute fractional coordinates inside the cell
    x = ii - i   # [0,1) in enthalpy direction
    y = jj - j   # [0,1) in logP direction

    # Build cubic basis vectors in both directions
    xm = jnp.array([1.0, x, x*x, x*x*x])   # powers of x
    ym = jnp.array([1.0, y, y*y, y*y*y])   # powers of y

    # Outer product flattened into (16,) basis vector
    # Ordering: basis[4*n + m] = x^m * y^n
    basis = jnp.kron(ym, xm)

    # Evaluate bicubic polynomial for each property
    props = {
        name: jnp.dot(table[name]["coeffs"][i, j], basis)
        for name in jxp.PROPERTIES_CANONICAL
    }

    return jxp.FluidState(**props)




def interpolate_bicubic_xy(input_pair, val1, val2, table, coarse_step=1, scale=1.0):
    """
    Invert the bicubic interpolant for a given input pair.
    Finds (h, p) such that interpolate_bicubic_hp(h, p, table)
    matches the target values.

    Parameters
    ----------
    input_pair : int
        Identifier for the input pair (e.g. jxp.HmassSmass_INPUTS).
    val1, val2 : float
        Target values for the two properties.
    table : dict
        Property table with bicubic coefficients and node values.
    coarse_step : int, optional
        Step for coarse grid scan (default: 5).
    scale : float, optional
        Scaling factor for optimization variables (default: 1).
        Both h and p are mapped into [0, scale].

    Returns
    -------
    dict
        Interpolated fluid properties at the recovered (h, p).
    """
    # TODO: Fix this function as it does not have a reliable convergence behavior
    # The initial step with a Newton solver is too aggresive and crashes the problem
    # Should we change to a different solution strategy in 1D for more reliable convergence?

    # Map input_pair to property names
    prop1, prop2 = jxp.INPUT_PAIR_MAP[input_pair]
    prop1 = jxp.ALIAS_TO_CANONICAL[prop1]
    prop2 = jxp.ALIAS_TO_CANONICAL[prop2]

    # axis limits
    h_axis = table["h_vals"]          # (Nh,)
    p_axis = table["p_vals"]          # (Np,)
    h_min, h_max = h_axis[0], h_axis[-1]
    p_min, p_max = p_axis[0], p_axis[-1]

    # transforms
    def to_scaled(h, p):
        h_scaled = (h - h_min) / (h_max - h_min) * scale
        p_scaled = (p - p_min) / (p_max - p_min) * scale
        return h_scaled, p_scaled

    def from_scaled(h_scaled, p_scaled):
        h = (h_scaled / scale) * (h_max - h_min) + h_min
        p = (p_scaled / scale) * (p_max - p_min) + p_min
        return h, p

    # --- coarse grid scan for initial guess ---
    H_field = table["enthalpy"]["value"][::coarse_step, ::coarse_step]
    P_field = table["pressure"]["value"][::coarse_step, ::coarse_step]
    prop1_field = table[prop1]["value"][::coarse_step, ::coarse_step]
    prop2_field = table[prop2]["value"][::coarse_step, ::coarse_step]

    # scale by property ranges for consistency
    rng1 = jnp.maximum(table[prop1]["value"].ptp(), 1.0)
    rng2 = jnp.maximum(table[prop2]["value"].ptp(), 1.0)
    errs = ((prop1_field - val1)/rng1)**2 + ((prop2_field - val2)/rng2)**2

    idx = jnp.argmin(errs)
    i, j = jnp.unravel_index(idx, errs.shape)
    h0_node, p0_node = H_field[i, j], P_field[i, j]
    x0 = jnp.array(to_scaled(h0_node, p0_node))

    # --- residual in unit variables ---
    def residual(x, _):
        h, p = from_scaled(*x)
        props = interpolate_bicubic_hp(h, p, table)

        r1 = (props[prop1] - val1) / rng1
        r2 = (props[prop2] - val2) / rng2
        r = jnp.array([r1, r2])

        jax.debug.print(
            "residual(h={:.6e}, p={:.6e})\n"
            "  {p1}: props={:.6e}, target={:.6e}, normdiff={:.3e}\n"
            "  {p2}: props={:.6e}, target={:.6e}, normdiff={:.3e}",
            h, p,
            props[prop1], val1, r1,
            props[prop2], val2, r2,
            p1=prop1, p2=prop2
        )
        # return jnp.linalg.norm(r)
        return r

    # BFGS least-squares solver
    # TODO: this does not work well for Newton solver. Investigate why
    solver = optx.BFGS(rtol=1e-8, atol=1e-8)
    sol = optx.least_squares(residual, solver, x0, throw=True)

    # solver = optx.Newton(rtol=1e-8, atol=1e-8)
    # sol = optx.root_find(residual, solver, x0, throw=True)

    h_scaled, p_scaled = sol.value
    h, p = from_scaled(h_scaled, p_scaled)
    return interpolate_bicubic_hp(h, p, table)







# # @jax.jit
# def inverse_interpolant_scalar_hD(h, D):
#     # Find the real(float) index
#     ii = (h - hmin) / (hmax - hmin) * (N - 1)
#     # The integer part is the cell index
#     i = ii.astype(int)
#     # The remainder (for numerical stability better to use the difference)
#     # is instead the position within our interpolation cell.
#     x = ii - i
#     # find interval that contains the solution
#     xth = jnp.ones_like(h)  # initialize x to the 0th power
#     # First we compute the nodal values, that is the values of D(h,P) where
#     # h is the actual enthalpy and P are grid values.
#     # TODO: instead of computing all the nodal values and then use sortedsearch
#     # to find the correct interval, we could do a binary search. This would
#     # constraint M to be a power of 2.
#     # Possible example (to be refined) to compute the node. Start with the node
#     # corresponding to j=M/2, then compute new index j=j+M/4*(2*(Dj>D)-1)
#     # then j=j+M/8*(2*(Dj>D)-1) and so on ...
#     # after log2(M) iteration we converged to the index j.
#     D_nodal = jnp.zeros(M)
#     for m in range(4):
#         D_nodal += bicubic_coefficients[i, :, m] * xth
#         xth = xth * x
#     # We search more efficiently in which interval we have the solution
#     # if we assume a sorted vector.
#     # TODO: This assumes that P has a monotonic trend with respect to D
#     # at fixed h. This causes some problems and needs further investigation
#     if iD == cp.iSmass:
#         j = jax.numpy.searchsorted(-D_nodal, -D).astype(int) - 1
#     else:
#         j = jax.numpy.searchsorted(D_nodal, D).astype(int) - 1

#     # After we are in the unit square, that is for known i and j
#     # compute 1D cubic coefficients (as complex numbers to avoid promotion)
#     # Each coefficient is bj=sum(aij*x**i)
#     # Leading to the equation D=b0 + b1*y + b2*y**2 + b3*y**3
#     xth = jnp.ones_like(h)
#     b0 = jnp.zeros_like(h, dtype=complex128)
#     b1 = jnp.zeros_like(h, dtype=complex128)
#     b2 = jnp.zeros_like(h, dtype=complex128)
#     b3 = jnp.zeros_like(h, dtype=complex128)
#     for m in range(4):
#         b0 += bicubic_coefficients[i, j, 4 * 0 + m] * xth
#         b1 += bicubic_coefficients[i, j, 4 * 1 + m] * xth
#         b2 += bicubic_coefficients[i, j, 4 * 2 + m] * xth
#         b3 += bicubic_coefficients[i, j, 4 * 3 + m] * xth
#         xth = xth * x
#     # solve cubic equation - all three solutions
#     # TODO: if necessary, add solution for degenerate (quadratic and linear)
#     # For more information:https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
#     D0 = b2 * b2 - 3 * b3 * b1
#     D1 = 2 * b2 * b2 * b2 - 9 * b3 * b2 * b1 + 27 * b3 * b3 * (b0 - D)
#     C = ((D1 + (D1 * D1 - 4 * D0 * D0 * D0) ** 0.5) / 2) ** (1 / 3)
#     D0C = jax.lax.select(C == (0 + 0j), 0 + 0j, D0 / C)
#     z = jnp.array([1, -0.5 + 0.8660254037844386j, -0.5 - 0.8660254037844386j])
#     y = -1 / (3 * b3) * (b2 + C * z + D0C / z)
#     # To find our solution we have two criteria:
#     #   -0 imaginary part
#     #   -real part between 0 and 1, that are the bounds of our cell
#     # We define a "badness" as the deviation from these critera, and pick the
#     # solution with the lowest badness
#     badness = jax.nn.relu(4 * (jnp.real(y) - 0.5) ** 2 - 1) + jnp.imag(y) ** 2
#     yreal = jnp.real(y[jnp.argmin(badness)])
#     jj = j + yreal
#     L = Lmin + jj * (Lmax - Lmin) / (M - 1)
#     P = jnp.exp(L)
#     return P


# # @jax.jit
# def inverse_interpolant_scalar_DP(D, P):
#     # Convert pressure to log space
#     L = jnp.log(P)

#     # Compute index along pressure grid
#     jj = (L - Lmin) / (Lmax - Lmin) * (M - 1)
#     j = jj.astype(int)
#     y = jj - j  # fractional position in pressure direction

#     # Compute nodal D(h) values at fixed pressure (we'll search h index now)
#     yth = jnp.ones_like(D)
#     D_nodal = jnp.zeros(N)
#     for m in range(4):
#         D_nodal += bicubic_coefficients[:, j, m] * yth
#         yth = yth * y

#     # Search h-direction to find which cell to use
#     if iD == cp.iSmass:
#         i = jnp.searchsorted(-D_nodal, -D).astype(int) - 1
#     else:
#         i = jnp.searchsorted(D_nodal, D).astype(int) - 1

#     # Now build 1D cubic in x (h-direction) at fixed j
#     yth = jnp.ones_like(D)
#     b0 = jnp.zeros_like(D, dtype=complex128)
#     b1 = jnp.zeros_like(D, dtype=complex128)
#     b2 = jnp.zeros_like(D, dtype=complex128)
#     b3 = jnp.zeros_like(D, dtype=complex128)
#     for m in range(4):
#         b0 += bicubic_coefficients[i, j, m + 4 * 0] * yth
#         b1 += bicubic_coefficients[i, j, m + 4 * 1] * yth
#         b2 += bicubic_coefficients[i, j, m + 4 * 2] * yth
#         b3 += bicubic_coefficients[i, j, m + 4 * 3] * yth
#         yth = yth * y

#     # Solve cubic: D = b0 + b1*x + b2*x^2 + b3*x^3
#     D0 = b2 * b2 - 3 * b3 * b1
#     D1 = 2 * b2**3 - 9 * b3 * b2 * b1 + 27 * b3**2 * (b0 - D)
#     C = ((D1 + jnp.sqrt(D1**2 - 4 * D0**3)) / 2) ** (1 / 3)
#     D0C = jax.lax.select(C == 0, 0 + 0j, D0 / C)
#     z = jnp.array([1, -0.5 + 0.8660254037844386j, -0.5 - 0.8660254037844386j])
#     x = -1 / (3 * b3) * (b2 + C * z + D0C / z)

#     # Pick root with lowest badness
#     badness = jax.nn.relu(4 * (jnp.real(x) - 0.5) ** 2 - 1) + jnp.imag(x) ** 2
#     xreal = jnp.real(x[jnp.argmin(badness)])

#     # Final result: compute h from i + x
#     ii = i + xreal
#     h = hmin + ii * (hmax - hmin) / (N - 1)
#     return h
