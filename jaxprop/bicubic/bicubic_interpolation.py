import os
import time
import tqdm
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import jaxprop.coolprop as jxp



# TODO Add saturation curves look-up:
# # JAX-compatible cubic Hermite interpolation
# def jax_cubic_spline(x, x_vals, y_vals):
#     """
#     Piecewise cubic Hermite spline approximation (JAX-compatible)
#     """
#     # Find interval indices (clip to valid range)
#     idx = jnp.clip(jnp.searchsorted(x_vals, x) - 1, 0, len(x_vals)-2)
   
#     x0 = x_vals[idx]
#     x1 = x_vals[idx+1]
#     y0 = y_vals[idx]
#     y1 = y_vals[idx+1]
   
#     # Linear slope (simple Hermite approximation)
#     m = (y1 - y0) / (x1 - x0 + 1e-12)  # avoid div by zero
#     t = (x - x0) / (x1 - x0 + 1e-12)
   
#     # Cubic Hermite polynomial: h00 = 2t^3 - 3t^2 + 1, h10 = t^3 - 2t^2 + t
#     h00 = 2*t**3 - 3*t**2 + 1
#     h10 = t**3 - 2*t**2 + t
   
#     return h00*y0 + (x1 - x0)*h10*m + (1 - h00)*y1

# ================================================================
# FluidBicubic class
# ================================================================
class FluidBicubic(eqx.Module):
    r"""
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
    coarse_step : int
        Number of points to skip when determining the initial guess during 2D inverse interpolations
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
    coarse_step: int = eqx.field(static=True)
    delta_h: float = eqx.field(static=True)
    delta_logP: float = eqx.field(static=True)
    h_vals: jnp.ndarray = eqx.field(static=False)
    logP_vals: jnp.ndarray = eqx.field(static=False)
    table_name: str = eqx.field(static=True)
    table_dir: str = eqx.field(static=True)
    table: dict = eqx.field(static=False)
    identifier: str = eqx.field(static=True)
    grad_method: str = eqx.field(static=True)

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
        coarse_step: int = 10,
        gradient_method: str = "central",
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
        self.grad_method = gradient_method
        self.coarse_step = coarse_step

        # Initialize table dictionary
        self.h_vals = jnp.linspace(self.h_min, self.h_max, self.N_h)
        self.logP_vals = jnp.linspace(np.log(self.p_min), np.log(self.p_max), self.N_p)
        self.delta_h = float(self.h_vals[1] - self.h_vals[0])
        self.delta_logP = float(self.logP_vals[1] - self.logP_vals[0])

        # Create the table if it does not exist
        self.table = self._load_or_generate_table()

    # ------------------ Table generation ------------------
    def _load_or_generate_table(self):
        pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                table = pickle.load(f)
            print(f"Loaded property table from: {pkl_path}")
            return table

            # TODO do a check that all metadate of the loaded table matches, an dif it changes print message and recompute table

        print("No existing table found, generating new one...")
        return self._generate_property_table()

    def _generate_property_table(self):
        # Initialize fluid
        # Using jxp.Fluid is faster than jxp.FluidJAX
        fluid = jxp.Fluid(self.fluid_name, self.backend)

        # Initialize interpolation table
        table = {
            "metadata": dict(
                fluid=self.fluid_name,
                backend=self.backend,
                h_min=self.h_min,
                h_max=self.h_max,
                p_min=self.p_min,
                p_max=self.p_max,
                N_h=self.N_h,
                N_p=self.N_p,
                delta_h=self.delta_h,
                delta_logP=self.delta_logP,
            ),
        }

        for k in jxp.PROPERTIES_CANONICAL:
            table[k] = {
                "value": np.empty((self.N_h, self.N_p)),
                "grad_h": np.empty((self.N_h, self.N_p)),
                "grad_p": np.empty((self.N_h, self.N_p)),
                "grad_logP": np.empty((self.N_h, self.N_p)),
                "grad_ph": np.empty((self.N_h, self.N_p)),
                "grad_hlogP": np.empty((self.N_h, self.N_p)),
                "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16)),
            }

        # Start property calculations
        total_points = self.N_h * self.N_p
        success_count = 0
        start_time = time.perf_counter()
        with tqdm.tqdm(
            total=total_points,
            desc="Generating property table",
            ascii=True,
            ncols=80,
            bar_format="{l_bar}{bar}|",
        ) as pbar:
            for i, h in enumerate(self.h_vals):
                for j, logP in enumerate(self.logP_vals):

                    # Define finite difference step size
                    p = np.exp(logP)
                    eps_h = 1e-5 * abs(h)
                    eps_p = 1e-5 * abs(p)

                    # Compute gradients according to specific method
                    try:
                        if self.grad_method == "forward":
                            grads = self._gradients_forward(fluid, h, p, eps_h, eps_p)
                            success_count += 1
                        elif self.grad_method == "central":
                            grads = self._gradients_central(fluid, h, p, eps_h, eps_p)
                            success_count += 1
                        else:
                            m = self.grad_method
                            raise ValueError(f"Unknown gradient scheme: {m}")
                    except Exception:
                        continue

                    # Store gradients and values in table
                    for k, (val, grad_h, grad_p, grad_hp) in grads.items():
                        table[k]["value"][i, j] = val
                        table[k]["grad_h"][i, j] = grad_h
                        table[k]["grad_p"][i, j] = grad_p
                        table[k]["grad_logP"][i, j] = grad_p * p
                        table[k]["grad_ph"][i, j] = grad_hp
                        table[k]["grad_hlogP"][i, j] = grad_hp * p

                    # Update progress bar
                    pbar.update(1)

            # Compute polynomial coefficients ONLY AFTER computing all table values
            for k in jxp.PROPERTIES_CANONICAL:
                table[k]["coeffs"] = compute_coefficients(
                    table[k]["value"],
                    table[k]["grad_h"],
                    table[k]["grad_logP"],
                    table[k]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )

        # Print table generation report
        frac_success = success_count / total_points * 100
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(
            f"Successfully evaluated {success_count}/{total_points} points "
            f"({frac_success:.2f} %)"
        )
        print(f"Total table generation time: {elapsed:.2f} s")

        # Convert all numpy arrays to jax.numpy
        for k in jxp.PROPERTIES_CANONICAL:
            for sub in ["value", "grad_h", "grad_p", "grad_ph", "coeffs"]:
                table[k][sub] = jnp.array(table[k][sub])

        # Save to pickle file
        os.makedirs(self.table_dir, exist_ok=True)
        pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(table, f)
        print(f"Saved property table to: {pkl_path}")

        return table

    @staticmethod
    def _gradients_forward(fluid, h, p, eps_h, eps_p):
        """First-order forward finite differences for h and p gradients."""
        f0 = fluid.get_state(jxp.HmassP_INPUTS, h, p)
        fh = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p)
        fp = fluid.get_state(jxp.HmassP_INPUTS, h, p + eps_p)
        fhp = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p + eps_p)

        grads = {}
        for k in jxp.PROPERTIES_CANONICAL:
            val = f0[k]
            grad_h = (fh[k] - f0[k]) / eps_h
            grad_p = (fp[k] - f0[k]) / eps_p
            grad_hp = (fhp[k] - fh[k] - fp[k] + f0[k]) / (eps_h * eps_p)
            grads[k] = (val, grad_h, grad_p, grad_hp)
        return grads

    @staticmethod
    def _gradients_central(fluid, h, p, eps_h, eps_p):
        """Second-order central finite differences for h and p gradients."""
        f0 = fluid.get_state(jxp.HmassP_INPUTS, h, p)
        fhp = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p)
        fhm = fluid.get_state(jxp.HmassP_INPUTS, h - eps_h, p)
        fph = fluid.get_state(jxp.HmassP_INPUTS, h, p + eps_p)
        fpm = fluid.get_state(jxp.HmassP_INPUTS, h, p - eps_p)
        fh_p = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p + eps_p)
        fh_m = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p - eps_p)
        fm_p = fluid.get_state(jxp.HmassP_INPUTS, h - eps_h, p + eps_p)
        fm_m = fluid.get_state(jxp.HmassP_INPUTS, h - eps_h, p - eps_p)

        grads = {}
        for k in jxp.PROPERTIES_CANONICAL:
            val = f0[k]
            grad_h = (fhp[k] - fhm[k]) / (2 * eps_h)
            grad_p = (fph[k] - fpm[k]) / (2 * eps_p)
            grad_hp = (fh_p[k] - fh_m[k] - fm_p[k] + fm_m[k]) / (4 * eps_h * eps_p)
            grads[k] = (val, grad_h, grad_p, grad_hp)
        return grads

    # ------------------ Bicubic property interpolation ------------------
    PROPERTY_CALCULATORS = {
        jxp.HmassP_INPUTS: lambda self, h, p: self._interp_h_p(h, p),
        jxp.PT_INPUTS: lambda self, p, T: self._interp_x_p(p, T, "temperature"),
        jxp.DmassP_INPUTS: lambda self, d, p: self._interp_x_p(p, d, "density"),
        jxp.PSmass_INPUTS: lambda self, p, s: self._interp_x_p(p, s, "entropy"),
        jxp.HmassSmass_INPUTS: lambda self, h, s: self._interp_h_y(h, s, "entropy"),
        jxp.DmassHmass_INPUTS: lambda self, d, h: self._interp_h_y(h, d, "density"),
        jxp.DmassT_INPUTS: lambda self, d, T: self._interp_x_y(
            jxp.DmassT_INPUTS, d, T, coarse_step=self.coarse_step
        ),
        jxp.DmassSmass_INPUTS: lambda self, d, s: self._interp_x_y(
            jxp.DmassSmass_INPUTS, d, s, coarse_step=self.coarse_step
        ),
    }

    @eqx.filter_jit
    def get_state(self, input_type, val1, val2):
        """
        Compute thermodynamic states from any supported input pair using bicubic interpolation.

        This is the main user-facing entry point. It accepts either scalar or array
        inputs for `val1` and `val2`, automatically broadcasting them to a common shape
        and evaluating the corresponding fluid state at each point in a vectorized
        manner. Internally, the method selects one of several solvers depending on the
        specified `input_type`:

          • For `(h, p)` inputs, properties are obtained directly through bicubic
            interpolation on the enthalpy-log(pressure) grid.

          • For 1D inversion problems (e.g. `P,T` or `D,p`), a Newton root finder is
            used to solve for either enthalpy or pressure, using the closest table node
            as the initial guess.

          • For 2D inversion problems (e.g. `D, T`), a Newton solver in normalized
            `(h, p)` space is used to recover the unique thermodynamic state that
            matches both input properties simultaneously.

        All computations are compatible with JAX transformations such as `jit`, `vmap`,
        and automatic differentiation. Vectorization is handled internally using
        `vmap` over the scalar solver/interpolator. The output is a `FluidState`
        object containing all thermodynamic and transport properties defined in the
        table, either for a single point (scalar inputs) or for each element of the
        broadcasted input arrays.

        Parameters
        ----------
        input_type : int
            Identifier of the input pair (e.g. `jxp.HmassP_INPUTS`, `jxp.PT_INPUTS`).
        val1 : float or array_like
            First input variable (e.g. enthalpy, pressure, density).
        val2 : float or array_like
            Second input variable (e.g. pressure, temperature, entropy).

        Returns
        -------
        FluidState
            Interpolated fluid state(s) corresponding to the specified input pair.
            If inputs are arrays, each property is returned as an array with the
            broadcasted shape of `val1` and `val2`.
        """
        # Broadcast h and p to the same shape
        val1 = jnp.asarray(val1)
        val2 = jnp.asarray(val2)
        val1, val2 = jnp.broadcast_arrays(val1, val2)

        # Define vectorized mapping explicitly using jax.vmap
        batched_fn = jax.vmap(lambda v1, v2: self._get_state_scalar(input_type, v1, v2))

        # Apply to flattened arrays
        props_batched = batched_fn(val1.ravel(), val2.ravel())

        # Reshape each leaf in the pytree to the broadcasted shape
        props = jax.tree.map(lambda x: x.reshape(val1.shape), props_batched)

        return props

    @eqx.filter_jit
    def _get_state_scalar(self, input_type, val1, val2):
        """
        Compute a single FluidState for the given input pair (scalar inputs).
        """
        return self.PROPERTY_CALCULATORS.get(input_type)(self, val1, val2)

    def _interp_h_p(self, h, p):
        """
        Interpolate all fluid properties at a given (h, p) state using
        bicubic interpolation on a uniform enthalpy-log(pressure) grid.

        The method evaluates a bicubic polynomial for each property using
        precomputed coefficients stored in the table. The input enthalpy and
        pressure are first converted to continuous indices within the table
        grid. Their fractional coordinates are then used to construct local
        cubic basis vectors in both directions, and the property values are
        obtained through a dot product with the corresponding coefficients.

        The returned state contains thermodynamic and transport properties
        at the specified (h, p) point. Input enthalpy and pressure are enforced
        exactly in the output. Grid indices are clamped to the valid range, so
        queries outside the table will snap to the nearest valid cell rather
        than extrapolating.

        Parameters
        ----------
        h : float
            Enthalpy [J/kg] at which to interpolate.
        p : float
            Pressure [Pa] at which to interpolate.

        Returns
        -------
        FluidState
            Interpolated fluid state containing thermodynamic and transport
            properties at the specified (h, p).
        """

        # Convert (h,p) to grid indices
        ii = (h - self.h_min) / self.delta_h
        jj = (jnp.log(p) - jnp.log(self.p_min)) / self.delta_logP

        # Clamp indices
        i = jnp.clip(jnp.floor(ii).astype(int), 0, self.N_h - 2)
        j = jnp.clip(jnp.floor(jj).astype(int), 0, self.N_p - 2)

        # Fractional coords
        x = ii - i
        y = jj - j

        # Basis vectors
        xm = jnp.array([1.0, x, x * x, x * x * x])
        ym = jnp.array([1.0, y, y * y, y * y * y])
        basis = jnp.kron(ym, xm)

        # Interpolate all properties
        props = {
            name: jnp.dot(self.table[name]["coeffs"][i, j], basis)
            for name in jxp.PROPERTIES_CANONICAL
        }

        # Enforce exact inputs
        props["enthalpy"] = h
        props["pressure"] = p

        return jxp.FluidState(fluid_name=self.fluid_name, **props)

    def _interp_h_y(self, h, y_value, y_name, tol=1e-10, max_steps=64):
        """
        Solve for pressure at fixed enthalpy such that the specified property
        matches a target value, using Newton's method with a table-based initial guess.
        """
        # Use closest table value as initial guess
        i0 = jnp.argmin(jnp.abs(self.h_vals - h))
        j0 = jnp.argmin(jnp.abs(self.table[y_name]["value"][i0, :] - y_value))
        p0 = self.table["pressure"]["value"][i0, j0]

        # Define residual
        def residual(p, args):
            val = self._interp_h_p(h, p)[y_name] - y_value
            # jax.debug.print("[_interp_h_y] p={:.6e}, residual={:.3e}", p, val)
            return val

        # Solve root-finding problem
        solver = optx.Newton(rtol=tol, atol=tol)
        solution = optx.root_find(
            residual,
            solver,
            y0=p0,
            options={"lower": self.p_min, "upper": self.p_max, "max_steps": max_steps},
        )

        return self._interp_h_p(h, solution.value)

    def _interp_x_p(self, p, x_value, x_name, tol=1e-10, max_steps=64):
        """
        Solve for enthalpy at fixed pressure such that the specified property
        matches a target value, using Newton's method with a table-based initial guess.
        """
        # Use closest table value as initial guess
        j0 = jnp.argmin(jnp.abs(self.logP_vals - jnp.log(p)))
        i0 = jnp.argmin(jnp.abs(self.table[x_name]["value"][:, j0] - x_value))
        h0 = self.table["enthalpy"]["value"][i0, j0]

        # Define residual
        def residual(h, args):
            val = self._interp_h_p(h, p)[x_name] - x_value
            # jax.debug.print("[_interp_x_p] h={:.6e}, residual={:.3e}", h, val)
            return val

        # Solve root-finding problem
        solver = optx.Newton(rtol=tol, atol=tol)
        solution = optx.root_find(
            residual,
            solver,
            y0=h0,
            options={"lower": self.h_min, "upper": self.h_max, "max_steps": max_steps},
        )

        return self._interp_h_p(solution.value, p)

    def _interp_x_y(self, input_pair, x, y, coarse_step=8, tol=1e-12):
        r"""
        Solve for (h, p) corresponding to a given input pair (x, y) by inverting
        the bicubic enthalpy-pressure property interpolant.

        This function finds the enthalpy and pressure that simultaneously match
        two target property values (e.g. density-temperature, pressure-entropy).
        It performs a coarse table scan to initialize the search, followed by a
        Newton root-finding on scaled variables.

        The solver operates in nomalized p-h variables (0-1) to avoid ill-conditioning
        due to large enthalpy/pressure magnitudes. The residuals are normalized by the
        property ranges to improve solver conditioning.

        Parameters
        ----------
        input_pair : int
            Identifier for the input property pair (e.g. jxp.HmassSmass_INPUTS).
        x : float
            Target value of the first property.
        y : float
            Target value of the second property.
        coarse_step : int, optional
            Subsampling step used for the coarse grid scan to obtain an initial
            guess (default: 1, i.e. use all nodes).
        tol : float, optional
            Absolute and relative tolerance for the Newton solver (default: 1e-8).

        Returns
        -------
        FluidState
            Interpolated fluid state corresponding to the recovered (h, p).

        """
        # Map input_pair to property names
        prop1, prop2 = jxp.INPUT_PAIR_MAP[input_pair]
        prop1 = jxp.ALIAS_TO_CANONICAL[prop1]
        prop2 = jxp.ALIAS_TO_CANONICAL[prop2]

        # transforms
        def to_scaled(h, p):
            h_scaled = (h - self.h_min) / (self.h_max - self.h_min)
            p_scaled = (p - self.p_min) / (self.p_max - self.p_min)
            return h_scaled, p_scaled

        def from_scaled(h_scaled, p_scaled):
            h = h_scaled * (self.h_max - self.h_min) + self.h_min
            p = p_scaled * (self.p_max - self.p_min) + self.p_min
            return h, p

        # Scan the table to identify a suitable initial guess
        h_table = self.table["enthalpy"]["value"][::coarse_step, ::coarse_step]
        p_table = self.table["pressure"]["value"][::coarse_step, ::coarse_step]
        val_x = self.table[prop1]["value"][::coarse_step, ::coarse_step]
        val_y = self.table[prop2]["value"][::coarse_step, ::coarse_step]
        range_x = jnp.ptp(self.table[prop1]["value"])
        range_y = jnp.ptp(self.table[prop2]["value"])
        errors = ((val_x - x) / range_x) ** 2 + ((val_y - y) / range_y) ** 2
        idx = jnp.argmin(errors)
        i, j = jnp.unravel_index(idx, errors.shape)
        h0_node, p0_node = h_table[i, j], p_table[i, j]
        x0 = jnp.array(to_scaled(h0_node, p0_node))

        # Define residual function
        def residual(xy, _):
            h, p = from_scaled(*xy)
            props = self._interp_h_p(h, p)
            res_x = (props[prop1] - x) / range_x
            res_y = (props[prop2] - y) / range_y
            # jax.debug.print(
            #     "[_interp_x_y] h={:.6e}, p={:.6e}, res_x={:.3e}, res_y={:.3e}",
            #     h, p, res_x, res_y
            # )
            return jnp.array([res_x, res_y])

        # Solve root-finding problem
        solver = optx.Newton(rtol=tol, atol=tol)
        solution = optx.root_find(residual, solver, x0, throw=True)

        # Return the computed state
        h_scaled, p_scaled = solution.value
        h, p = from_scaled(h_scaled, p_scaled)

        return self._interp_h_p(h, p)


# ================================================================
# Bicubic coefficient computation
# ================================================================


def compute_coefficients(value, grad_h, grad_logP, grad_hlogP, delta_h, delta_logP):
    """
    Compute bicubic interpolation coefficients for one property on a uniform (h, logP) grid.

    Parameters
    ----------
    value : ndarray of shape (Nh, Np)
        Property values f(h, P) evaluated on the grid.
    grad_h : ndarray of shape (Nh, Np)
        Partial derivative with respect to enthalpy evaluated on the grid.
    grad_logP : ndarray of shape (Nh, Np)
        Partial derivative with respect to pressure evaluated on the grid (with respect to P, not logP).
    grad_hlogP : ndarray of shape (Nh, Np)
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
            grad_logP[:-1, :-1] * delta_logP,
            grad_logP[1:, :-1] * delta_logP,
            grad_logP[:-1, 1:] * delta_logP,
            grad_logP[1:, 1:] * delta_logP,
            grad_hlogP[:-1, :-1] * delta_h * delta_logP,
            grad_hlogP[1:, :-1] * delta_h * delta_logP,
            grad_hlogP[:-1, 1:] * delta_h * delta_logP,
            grad_hlogP[1:, 1:] * delta_h * delta_logP,
        ],
        axis=-1,
    )

    # Multiply by the fixed 16x16 matrix A_MAT to get coefficients
    # einsum: 'ab,ijb->ija' → sum over b, output shape (Nh-1, Np-1, 16)
    coeffs = np.einsum("ab,ijb->ija", A_MAT, XX)
    return coeffs

    # Alternative equivalent calculation with a double for-loop (slower)
    # Nh, Np = value.shape
    # coeffs = np.zeros((Nh, Np, 16), dtype=np.float64)
    # for i in range(Nh - 1):
    #     for j in range(Np - 1):
    #         # Assemble the 16 basis values for this cell
    #         xx = np.array(
    #             [
    #                 value[i, j],
    #                 value[i + 1, j],
    #                 value[i, j + 1],
    #                 value[i + 1, j + 1],
    #                 grad_h[i, j] * delta_h,
    #                 grad_h[i + 1, j] * delta_h,
    #                 grad_h[i, j + 1] * delta_h,
    #                 grad_h[i + 1, j + 1] * delta_h,
    #                 grad_logP[i, j] * delta_logP,
    #                 grad_logP[i + 1, j] * delta_logP,
    #                 grad_logP[i, j + 1] * delta_logP,
    #                 grad_logP[i + 1, j + 1] * delta_logP,
    #                 grad_hlogP[i, j] * delta_h * delta_logP,
    #                 grad_hlogP[i + 1, j] * delta_h * delta_logP,
    #                 grad_hlogP[i, j + 1] * delta_h * delta_logP,
    #                 grad_hlogP[i + 1, j + 1] * delta_h * delta_logP,
    #             ],
    #             dtype=np.float64,
    #         )

    #         # Dense matvec multiplication
    #         coeffs[i, j, :] = A_MAT @ xx

    # return coeffs


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
