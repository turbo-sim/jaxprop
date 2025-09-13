import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx

from tqdm import tqdm

from ..coolprop import Fluid
from .. import helpers_props as jxp

# To increase performance, this module uses numpy for table generation and jax.numpy for table intepolation


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
        self._batched_interp = jax.vmap(interpolate_bicubic_hp, in_axes=(0, 0, None))

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
        logP_vals = np.linspace(np.log(self.p_min), np.log(self.p_max), self.N_p)

        delta_h = h_vals[1] - h_vals[0]
        delta_logP = logP_vals[1] - logP_vals[0]

        table = {
            "h_vals": h_vals,
            "p_vals": np.exp(logP_vals),
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
                for j, logP in enumerate(logP_vals):
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
                        pbar.update(1)
                        continue

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
        if input_pair != jxp.HmassP_INPUTS:
            raise NotImplementedError(
                f"FluidBicubic currently supports only HmassP_INPUTS, got {input_pair}"
            )

        # Broadcast h and p to the same shape
        h_arr, p_arr = jnp.broadcast_arrays(h, p)

        # Vectorized version of the scalar interpolant
        props = self._batched_interp(h_arr.ravel(), p_arr.ravel(), self.table)

        # Reshape back to broadcasted shape
        props = {k: v.reshape(h_arr.shape) for k, v in props.items()}

        return jxp.FluidState(
            fluid_name=self.fluid_name,
            identifier=self.identifier,
            **props,
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
        Partial derivative ∂f/∂h evaluated on the grid.
    grad_p : ndarray of shape (Nh, Np)
        Partial derivative ∂f/∂P evaluated on the grid (with respect to P, not logP).
    grad_hp : ndarray of shape (Nh, Np)
        Mixed derivative ∂²f/(∂h∂P) evaluated on the grid.
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

    # Assemble the 16 basis values for each cell:
    # [f00, f10, f01, f11, fx00, fx10, fx01, fx11,
    #  fy00, fy10, fy01, fy11, fxy00, fxy10, fxy01, fxy11]
    XX = np.stack(
        [
            value[:-1, :-1],
            value[1:, :-1],
            value[:-1, 1:],
            value[1:, 1:],
            grad_h[:-1, :-1] * delta_h,
            grad_h[1:, :-1] * delta_h,
            grad_h[:-1, 1:] * delta_h,
            grad_h[1:, 1:] * delta_h,
            grad_p[:-1, :-1] * delta_logP,
            grad_p[1:, :-1] * delta_logP,
            grad_p[:-1, 1:] * delta_logP,
            grad_p[1:, 1:] * delta_logP,
            grad_hp[:-1, :-1] * delta_h * delta_logP,
            grad_hp[1:, :-1] * delta_h * delta_logP,
            grad_hp[:-1, 1:] * delta_h * delta_logP,
            grad_hp[1:, 1:] * delta_h * delta_logP,
        ],
        axis=-1,
    )

    # Multiply by the fixed 16x16 matrix A_MAT to get coefficients
    # einsum: 'ab,ijb->ija' → sum over b, output shape (Nh-1, Np-1, 16)
    coeffs = np.einsum("ab,ijb->ija", A_MAT, XX)
    return coeffs


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

    # TODO: decide extrapolation behavior

    # Extract uniform enthalpy and pressure grids from table
    h_vals = table["h_vals"]          # (Nh,) enthalpy grid
    p_vals = table["p_vals"]          # (Np,) pressure grid
    logP_vals = jnp.log(p_vals)       # work in logP for smoother interpolation

    Nh = h_vals.shape[0]
    Np = logP_vals.shape[0]

    # Compute uniform grid spacing
    h_min, h_max = h_vals[0], h_vals[-1]
    logP_min, logP_max = logP_vals[0], logP_vals[-1]
    delta_h = (h_max - h_min) / (Nh - 1)
    delta_logP = (logP_max - logP_min) / (Np - 1)

    # Convert query point (h, logP) into continuous indices in grid space
    ii = (h - h_min) / delta_h
    jj = (jnp.log(p) - logP_min) / delta_logP

    # Select lower-left cell index, clamped to table bounds
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
    out = {
        name: jnp.dot(table[name]["coeffs"][i, j], basis)
        for name in jxp.PROPERTIES_CANONICAL
    }

    return out
