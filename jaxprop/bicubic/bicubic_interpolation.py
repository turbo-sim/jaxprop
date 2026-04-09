import os
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import jaxprop.coolprop as jxp
from tqdm import tqdm


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

    # Attributes (static vs dynamic)
    fluid_name: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    h_min: float = eqx.field(static=True)
    h_max: float = eqx.field(static=True)
    p_min: float = eqx.field(static=True)
    p_max: float = eqx.field(static=True)
    N_h: int = eqx.field(static=True)
    N_p: int = eqx.field(static=True)
    N_p_sat: int = eqx.field(static=True)
    coarse_step: int = eqx.field(static=True)
    delta_h: float = eqx.field(static=True)
    delta_logP: float = eqx.field(static=True)
    delta_logP_sat: float = eqx.field(static=True)
    h_vals: jnp.ndarray = eqx.field(static=False)
    logP_vals: jnp.ndarray = eqx.field(static=False)
    logP_sat_vals: jnp.ndarray = eqx.field(static=False)
    table_name: str = eqx.field(static=True)
    table_dir: str = eqx.field(static=True)
    table: dict = eqx.field(static=False)
    identifier: str = eqx.field(static=True)
    grad_method: str = eqx.field(static=True)
    mixture_ratio: float = eqx.field(static=True)
    metastable_phase: str = eqx.field(static=True)
    critical_point: dict = eqx.field(static=True)

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
        N_p_sat: int=60,
        mixture_ratio: float = None,
        metastable_phase: str = None,
        coarse_step: int = 10,
        gradient_method: str = "central",
        identifier: str = None,
        table_name: str = None,
        table_dir: str = "fluid_tables",
    ):
        # store basic params
        self.fluid_name = fluid_name
        self.backend = backend
        self.h_min, self.h_max, self.N_h = h_min, h_max, N_h
        self.p_min, self.p_max, self.N_p = p_min, p_max, N_p
        self.mixture_ratio = mixture_ratio
        self.identifier = identifier or fluid_name
        self.metastable_phase = metastable_phase
        self.N_p_sat = N_p_sat
        

        # include mixture_ratio in filename if relevant (prevent collisions)
        if table_name is None:
            if mixture_ratio is not None or "mixture" in fluid_name.lower():
                self.table_name = f"{fluid_name}_{N_h}x{N_p}_mix_{mixture_ratio:.2f}"
            elif metastable_phase is not None:
                self.table_name = f"{fluid_name}_meta_{metastable_phase}_{N_h}x{N_p}"
            else:
                self.table_name = f"{fluid_name}_{N_h}x{N_p}"
        else:
            self.table_name = table_name

        self.table_dir = table_dir
        self.grad_method = gradient_method
        self.coarse_step = coarse_step

        # Create h and logP grids
        self.h_vals = jnp.linspace(self.h_min, self.h_max, self.N_h)
        self.logP_vals = jnp.linspace(np.log(self.p_min), np.log(self.p_max), self.N_p)
        self.delta_h = float(self.h_vals[1] - self.h_vals[0])
        self.delta_logP = float(self.logP_vals[1] - self.logP_vals[0])

        # Get critical and triple point properties
        if self.mixture_ratio is None:
            fluid = jxp.Fluid(self.fluid_name, self.backend)
            self.critical_point = fluid.critical_point
            self.logP_sat_vals = jnp.linspace(np.log(self.p_min), np.log(self.critical_point["pressure"]*0.98), self.N_p_sat)
            self.delta_logP_sat = float(self.logP_sat_vals[1] - self.logP_sat_vals[0])
        else:
            self.critical_point = None
            self.logP_sat_vals = None
            self.delta_logP_sat = None

        # Load or generate
        self.table = self._load_or_generate_table()

    # ------------------ Table loading / generation ------------------
    def _load_or_generate_table(self):
        os.makedirs(self.table_dir, exist_ok=True)
        pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                table = pickle.load(f)
            print(f"Loaded property table from: {pkl_path}")
            return table

        print("No existing table found, generating new one...")

        # choose mixture or single-fluid generator based on fluid_name containing "mixture"
        if "mixture" in self.fluid_name.lower():
            return self._generate_mixture_property_table()
        elif self.metastable_phase == "liquid":
            return self._generate_metastable_liquid_property_table()
        elif self.metastable_phase == "vapor":
            return self._generate_metastable_vapor_property_table()
        else:
            return self._generate_property_table()

    def _generate_property_table(self):
        # Initialize fluid (single component)
        fluid = jxp.Fluid(self.fluid_name, self.backend)

        # metadata
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

        # initialize property dicts for canonical properties
        for k in jxp.PROPERTIES_CANONICAL:
            table[k] = {
                "value": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_h": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_p": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_logP": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_ph": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_hlogP": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16), dtype=np.float64),
            }

        # Add enthalpy and pressure as explicit table entries (used for initial guesses)
        # Enthalpy: f(h,p) = h -> df/dh = 1, df/dp = 0
        enthalpy_vals = np.tile(np.asarray(self.h_vals)[:, None], (1, self.N_p))
        pressure_vals = np.tile(
            np.exp(np.asarray(self.logP_vals))[None, :], (self.N_h, 1)
        )
        table["enthalpy"] = {
            "value": enthalpy_vals,
            "grad_h": np.ones_like(enthalpy_vals),
            "grad_p": np.zeros_like(enthalpy_vals),
            "grad_logP": np.zeros_like(enthalpy_vals),
            "grad_ph": np.zeros_like(enthalpy_vals),
            "grad_hlogP": np.zeros_like(enthalpy_vals),
            "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16), dtype=np.float64),
        }
        # Pressure: f(h,p) = p -> df/dp = 1, df/dlogP = p
        table["pressure"] = {
            "value": pressure_vals,
            "grad_h": np.zeros_like(pressure_vals),
            "grad_p": np.ones_like(pressure_vals),
            "grad_logP": pressure_vals,  # grad wrt P times p => here 1 * p
            "grad_ph": np.zeros_like(pressure_vals),
            "grad_hlogP": np.zeros_like(pressure_vals),
            "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16), dtype=np.float64),
        }

        # Start evaluating properties
        total_points = self.N_h * self.N_p
        success_count = 0
        start_time = time.perf_counter()
        with tqdm(
            total=total_points,
            desc="Generating property table",
            ascii=True,
            ncols=80,
            bar_format="{l_bar}{bar}|",
        ) as pbar:
            for i, h in enumerate(np.asarray(self.h_vals)):
                for j, logP in enumerate(np.asarray(self.logP_vals)):
                    p = float(np.exp(logP))
                    # finite difference steps (avoid zero)
                    eps_h = 1e-5 * max(abs(h), 1.0)
                    eps_p = 1e-5 * max(abs(p), 1.0)

                    try:
                        if self.grad_method == "forward":
                            grads = self._gradients_forward(fluid, h, p, eps_h, eps_p)
                        elif self.grad_method == "central":
                            grads = self._gradients_central(fluid, h, p, eps_h, eps_p)
                        else:
                            raise ValueError(
                                f"Unknown gradient scheme: {self.grad_method}"
                            )
                        success_count += 1
                    except Exception:
                        # skip problematic grid points but keep going
                        pbar.update(1)
                        continue

                    # store computed properties
                    for k, (val, grad_h, grad_p, grad_hp) in grads.items():
                        table[k]["value"][i, j] = float(val)
                        table[k]["grad_h"][i, j] = float(grad_h)
                        table[k]["grad_p"][i, j] = float(grad_p)
                        # store derivatives wrt logP = (d/dP) * P
                        table[k]["grad_logP"][i, j] = float(grad_p * p)
                        table[k]["grad_ph"][i, j] = float(grad_hp)
                        table[k]["grad_hlogP"][i, j] = float(grad_hp * p)

                    pbar.update(1)

            # compute coefficients after full table computed
            for k in jxp.PROPERTIES_CANONICAL:
                table[k]["coeffs"] = compute_bicubic_coefficients(
                    table[k]["value"],
                    table[k]["grad_h"],
                    table[k]["grad_logP"],
                    table[k]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )

            # compute enthalpy/pressure coeffs too (they were created earlier)
            table["enthalpy"]["coeffs"] = compute_bicubic_coefficients(
                table["enthalpy"]["value"],
                table["enthalpy"]["grad_h"],
                table["enthalpy"]["grad_logP"],
                table["enthalpy"]["grad_hlogP"],
                self.delta_h,
                self.delta_logP,
            )
            table["pressure"]["coeffs"] = compute_bicubic_coefficients(
                table["pressure"]["value"],
                table["pressure"]["grad_h"],
                table["pressure"]["grad_logP"],
                table["pressure"]["grad_hlogP"],
                self.delta_h,
                self.delta_logP,
            )

        # report
        frac_success = success_count / total_points * 100.0
        elapsed = time.perf_counter() - start_time
        print(
            f"Successfully evaluated {success_count}/{total_points} points ({frac_success:.2f} %)"
        )
        print(f"Total table generation time: {elapsed:.2f} s")

        # convert to jax arrays for all relevant sub-keys
        for k in table.keys():
            # skip metadata dictionary
            if k == "metadata":
                continue
            for sub in [
                "value",
                "grad_h",
                "grad_p",
                "grad_logP",
                "grad_ph",
                "grad_hlogP",
                "coeffs",
            ]:
                # some entries (like coeffs) might already be jnp arrays but this is safe
                table[k][sub] = jnp.array(table[k][sub], dtype=jnp.float64)

        table["saturation_props"] = {}
        
        # save
        pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(table, f)
        print(f"Saved property table to: {pkl_path}")

        return table

    def _generate_mixture_property_table(self):
        # initialize mixture fluid object (expects mixture name / backend + ratio)\

        fluidmix = jxp.FluidMix(self.fluid_name, self.backend, self.mixture_ratio)

        table = {
            "metadata": dict(
                fluid=self.fluid_name,
                backend=self.backend,
                mixture_ratio=self.mixture_ratio,
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
                "value": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_h": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_p": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_logP": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_ph": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_hlogP": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16), dtype=np.float64),
            }

        # # enthalpy & pressure table entries (same logic as single-fluid)
        # enthalpy_vals = np.tile(np.asarray(self.h_vals)[:, None], (1, self.N_p))
        # pressure_vals = np.tile(
        #     np.exp(np.asarray(self.logP_vals))[None, :], (self.N_h, 1)
        # )
        # table["enthalpy"] = {
        #     "value": enthalpy_vals,
        #     "grad_h": np.ones_like(enthalpy_vals),
        #     "grad_p": np.zeros_like(enthalpy_vals),
        #     "grad_logP": np.zeros_like(enthalpy_vals),
        #     "grad_ph": np.zeros_like(enthalpy_vals),
        #     "grad_hlogP": np.zeros_like(enthalpy_vals),
        #     "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16), dtype=np.float64),
        # }
        # table["pressure"] = {
        #     "value": pressure_vals,
        #     "grad_h": np.zeros_like(pressure_vals),
        #     "grad_p": np.ones_like(pressure_vals),
        #     "grad_logP": pressure_vals,
        #     "grad_ph": np.zeros_like(pressure_vals),
        #     "grad_hlogP": np.zeros_like(pressure_vals),
        #     "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16), dtype=np.float64),
        # }

        total_points = self.N_h * self.N_p
        success_count = 0
        start_time = time.perf_counter()
        with tqdm(
            total=total_points,
            desc="Generating property table",
            ascii=True,
            ncols=80,
            bar_format="{l_bar}{bar}|",
        ) as pbar:
            for i, h in enumerate(np.asarray(self.h_vals)):
                for j, logP in enumerate(np.asarray(self.logP_vals)):
                    p = float(np.exp(logP))
                    eps_h = 1e-5 * max(abs(h), 1.0)
                    eps_p = 1e-5 * max(abs(p), 1.0)

                    try:
                        if self.grad_method == "forward":
                            grads = self._gradients_forward(
                                fluidmix, h, p, eps_h, eps_p
                            )
                        elif self.grad_method == "central":
                            grads = self._gradients_central(
                                fluidmix, h, p, eps_h, eps_p
                            )
                        else:
                            raise ValueError(
                                f"Unknown gradient scheme: {self.grad_method}"
                            )
                        success_count += 1
                    except Exception:
                        pbar.update(1)
                        continue

                    for k, (val, grad_h, grad_p, grad_hp) in grads.items():
                        table[k]["value"][i, j] = float(val)
                        table[k]["grad_h"][i, j] = float(grad_h)
                        table[k]["grad_p"][i, j] = float(grad_p)
                        table[k]["grad_logP"][i, j] = float(grad_p * p)
                        table[k]["grad_ph"][i, j] = float(grad_hp)
                        table[k]["grad_hlogP"][i, j] = float(grad_hp * p)

                    pbar.update(1)

            # compute coefficients
            for k in jxp.PROPERTIES_CANONICAL:
                table[k]["coeffs"] = compute_bicubic_coefficients(
                    table[k]["value"],
                    table[k]["grad_h"],
                    table[k]["grad_logP"],
                    table[k]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )

        frac_success = success_count / total_points * 100.0
        elapsed = time.perf_counter() - start_time
        print(
            f"Successfully evaluated {success_count}/{total_points} points ({frac_success:.2f} %)"
        )
        print(f"Total table generation time: {elapsed:.2f} s")

        # # convert to jax arrays
        # for k in table.keys():
        #     if k == "metadata":
        #         continue
        #     for sub in [
        #         "value",
        #         "grad_h",
        #         "grad_p",
        #         "grad_logP",
        #         "grad_ph",
        #         "grad_hlogP",
        #         "coeffs",
        #     ]:
        #         table[k][sub] = jnp.array(table[k][sub], dtype=jnp.float64)

        # Convert all numpy arrays to jax.numpy
        for k in jxp.PROPERTIES_CANONICAL:
            for sub in ["value", "grad_h", "grad_p", "grad_ph", "coeffs"]:
                table[k][sub] = jnp.array(table[k][sub])


        # # save
        # pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")
        # with open(pkl_path, "wb") as f:
        #     pickle.dump(table, f)

        table["saturation_props"] = {}
        
        # Save to pickle file
        os.makedirs(self.table_dir, exist_ok=True)
        pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(table, f)
            
        print(f"Saved property table to: {pkl_path}")

        return table
    
    def _generate_metastable_liquid_property_table(self):
        # Initialize fluid (single component)
        fluid = jxp.Fluid(self.fluid_name, self.backend)
        
        # metadata
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
                metastable_phase=self.metastable_phase,
            ),
        }

        # initialize property dicts for canonical properties
        for k in jxp.PROPERTIES_CANONICAL:
            table[k] = {
                "value": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_h": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_p": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_logP": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_ph": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_hlogP": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16), dtype=np.float64),
            }

        # Add enthalpy and pressure as explicit table entries (used for initial guesses)
        # Enthalpy: f(h,p) = h -> df/dh = 1, df/dp = 0
        enthalpy_vals = np.tile(np.asarray(self.h_vals)[:, None], (1, self.N_p))
        pressure_vals = np.tile(
            np.exp(np.asarray(self.logP_vals))[None, :], (self.N_h, 1)
        )
        table["enthalpy"] = {
            "value": enthalpy_vals,
            "grad_h": np.ones_like(enthalpy_vals),
            "grad_p": np.zeros_like(enthalpy_vals),
            "grad_logP": np.zeros_like(enthalpy_vals),
            "grad_ph": np.zeros_like(enthalpy_vals),
            "grad_hlogP": np.zeros_like(enthalpy_vals),
            "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16), dtype=np.float64),
        }
        # Pressure: f(h,p) = p -> df/dp = 1, df/dlogP = p
        table["pressure"] = {
            "value": pressure_vals,
            "grad_h": np.zeros_like(pressure_vals),
            "grad_p": np.ones_like(pressure_vals),
            "grad_logP": pressure_vals,  # grad wrt P times p => here 1 * p
            "grad_ph": np.zeros_like(pressure_vals),
            "grad_hlogP": np.zeros_like(pressure_vals),
            "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16), dtype=np.float64),
        }


        rho_guess = fluid.get_state(jxp.HmassP_INPUTS, enthalpy_vals[0][0], pressure_vals[-1][-1])["density"]
        T_guess = fluid.get_state(jxp.HmassP_INPUTS, enthalpy_vals[0][0], pressure_vals[-1][-1])["temperature"]
        isothermal_bulk_modulus_first = fluid.get_state(jxp.HmassP_INPUTS, enthalpy_vals[0][0], pressure_vals[-1][-1])["isothermal_bulk_modulus"]

        # Define the specific keys you want to protect/skip
        keys_to_skip = {'metadata', 'pressure', 'enthalpy'}
        

        # Start evaluating properties
        after_spinodal = False
        total_points = self.N_h * self.N_p
        success_count = 0
        start_time = time.perf_counter()
        with tqdm(
            total=total_points,
            desc="Generating property table",
            ascii=True,
            ncols=80,
            bar_format="{l_bar}{bar}|",
        ) as pbar:
            for j, logP in reversed(list(enumerate(np.asarray(self.logP_vals)))):
                ratio = 1
                ratio_old = ratio + 1
                p = float(np.exp(logP))

                # 3. Inner Loop: Standard (Min Enthalpy -> Max Enthalpy)
                for i, h in enumerate(np.asarray(self.h_vals)):
                    # print(f"({i},{j})")
                    # finite difference steps (avoid zero)
                    eps_h = 1e-5 * max(abs(h), 1.0)
                    eps_p = 1e-5 * max(abs(p), 1.0)
                    
                    
                    if after_spinodal == False:
                        try:
                            if self.grad_method == "forward":
                                grads = self._gradients_metastable_forward(fluid, h, p, eps_h, eps_p, rho_guess, T_guess)
                                # print("grads:",grads.items())
                                # print("\n")
                                # print("table:", table.keys())                            
                            elif self.grad_method == "central":
                                grads = self._gradients_metastable_central(fluid, h, p, eps_h, eps_p)
                            else:
                                raise ValueError(
                                    f"Unknown gradient scheme: {self.grad_method}"
                                )
                            success_count += 1

                        except:

                            # TODO: DO NOT DELETE!!!                            
                            for k in table.keys():
                                if k in keys_to_skip:
                                    continue  # Skip this iteration
                                # Fill the rest with fallback values
                                table[k]["value"][i, j] = table[k]["value"][i-1, j]
                                table[k]["grad_h"][i, j] = table[k]["grad_h"][i-1, j]
                                table[k]["grad_p"][i, j] = table[k]["grad_p"][i-1, j]
                                table[k]["grad_logP"][i, j] = table[k]["grad_logP"][i-1, j]
                                table[k]["grad_ph"][i, j] = table[k]["grad_ph"][i-1, j]
                                table[k]["grad_hlogP"][i, j] = table[k]["grad_hlogP"][i-1, j]

                            # for k in table.keys():
                            #     if k in keys_to_skip:
                            #         continue  # Skip this iteration
                            #     # Fill the rest with fallback values
                            #     table[k]["value"][i, j] = 1e-12
                            #     table[k]["grad_h"][i, j] = 1e-12
                            #     table[k]["grad_p"][i, j] = 1e-12
                            #     table[k]["grad_logP"][i, j] = 1e-12
                            #     table[k]["grad_ph"][i, j] = 1e-12
                            #     table[k]["grad_hlogP"][i, j] = 1e-12

                            pbar.update(1)
                            after_spinodal = True
                            continue

                        

                        
                        # Update rho guess and T guess for the next grid point
                        # rho_guess, T_guess = grads["density"][0], grads["temperature"][0]

                        rho_guess = grads["density"][0] + grads["density"][1] * self.delta_h
                        T_guess = grads["temperature"][0] + grads["temperature"][1] * self.delta_h
                        # print(f"point:({i},{j}) | {grads["isothermal_compressibility"][0]}")
                        # store computed properties
                        # NOTE: We use [i, j]. Even though we are looping P backwards, 
                        # 'j' is the correct index for that specific pressure in the matrix.
                        for k, (val, grad_h, grad_p, grad_hp) in grads.items():
                            table[k]["value"][i, j] = float(val)
                            table[k]["grad_h"][i, j] = float(grad_h)
                            table[k]["grad_p"][i, j] = float(grad_p)
                            # store derivatives wrt logP = (d/dP) * P
                            table[k]["grad_logP"][i, j] = float(grad_p * p)
                            table[k]["grad_ph"][i, j] = float(grad_hp)
                            table[k]["grad_hlogP"][i, j] = float(grad_hp * p)

                        if i > 0:
                            ratio = grads["isothermal_bulk_modulus"][0] / table["isothermal_bulk_modulus"]["value"][i-1, j] 
                            diff = ratio_old - ratio

                            # if ratio < 1e-1:
                            if diff < 0 or grads["isothermal_bulk_modulus"][0] < 0:
                                after_spinodal = True
                                
                                # for k in table.keys():
                                #     if k in keys_to_skip:
                                #         continue  # Skip this iteration
                                #     # Fill the rest with fallback values
                                #     table[k]["value"][i, j] = 1e-12
                                #     table[k]["grad_h"][i, j] = 1e-12
                                #     table[k]["grad_p"][i, j] = 1e-12
                                #     table[k]["grad_logP"][i, j] = 1e-12
                                #     table[k]["grad_ph"][i, j] = 1e-12
                                #     table[k]["grad_hlogP"][i, j] = 1e-12
                                
                                for k in table.keys():
                                    if k in keys_to_skip:
                                        continue  # Skip this iteration
                                    # Fill the rest with fallback values
                                    table[k]["value"][i, j] = table[k]["value"][i-1, j]
                                    table[k]["grad_h"][i, j] = table[k]["grad_h"][i-1, j]
                                    table[k]["grad_p"][i, j] = table[k]["grad_p"][i-1, j]
                                    table[k]["grad_logP"][i, j] = table[k]["grad_logP"][i-1, j]
                                    table[k]["grad_ph"][i, j] = table[k]["grad_ph"][i-1, j]
                                    table[k]["grad_hlogP"][i, j] = table[k]["grad_hlogP"][i-1, j]

                                    

                                # print("After spinodal")
                                
                        # print(f"point:{i,j}")
                        # print(f"IBM:{grads["isothermal_bulk_modulus"][0]}")
                        # print(f"IBM-ratio:{ratio}")

                        ratio_old = ratio

                        pbar.update(1) 

                    else:
                        # print("here")
                        for k in table.keys():
                            if k in keys_to_skip:
                                continue  # Skip this iteration
                            # Fill the rest with fallback values
                            table[k]["value"][i, j] = 1e-12
                            table[k]["grad_h"][i, j] = 1e-12
                            table[k]["grad_p"][i, j] = 1e-12
                            table[k]["grad_logP"][i, j] = 1e-12
                            table[k]["grad_ph"][i, j] = 1e-12
                            table[k]["grad_hlogP"][i, j] = 1e-12

                        pbar.update(1)
                        
                

                rho_guess = fluid.get_state(jxp.HmassP_INPUTS, enthalpy_vals[0][0], p)["density"]
                T_guess = fluid.get_state(jxp.HmassP_INPUTS, enthalpy_vals[0][0], p)["temperature"]
                isothermal_bulk_modulus_first = fluid.get_state(jxp.HmassP_INPUTS, enthalpy_vals[0][0], p)["isothermal_bulk_modulus"]
                after_spinodal = False


                # compute coefficients after full table computed
                for k in jxp.PROPERTIES_CANONICAL:
                    table[k]["coeffs"] = compute_bicubic_coefficients(
                        table[k]["value"],
                        table[k]["grad_h"],
                        table[k]["grad_logP"],
                        table[k]["grad_hlogP"],
                        self.delta_h,
                        self.delta_logP,
                    )

                # compute enthalpy/pressure coeffs too (they were created earlier)
                table["enthalpy"]["coeffs"] = compute_bicubic_coefficients(
                    table["enthalpy"]["value"],
                    table["enthalpy"]["grad_h"],
                    table["enthalpy"]["grad_logP"],
                    table["enthalpy"]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )
                table["pressure"]["coeffs"] = compute_bicubic_coefficients(
                    table["pressure"]["value"],
                    table["pressure"]["grad_h"],
                    table["pressure"]["grad_logP"],
                    table["pressure"]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )

        # report
        frac_success = success_count / total_points * 100.0
        elapsed = time.perf_counter() - start_time
        # print(
        #     f"Successfully evaluated {success_count}/{total_points} points ({frac_success:.2f} %)"
        # )
        # print(f"Total table generation time: {elapsed:.2f} s")

        # convert to jax arrays for all relevant sub-keys
        for k in table.keys():
            # skip metadata dictionary
            if k == "metadata":
                continue
            for sub in [
                "value",
                "grad_h",
                "grad_p",
                "grad_logP",
                "grad_ph",
                "grad_hlogP",
                "coeffs",
            ]:
                # some entries (like coeffs) might already be jnp arrays but this is safe
                table[k][sub] = jnp.array(table[k][sub], dtype=jnp.float64)

        print("Generating saturation properties table...")
        table["saturation_props"] = {}
        for k in jxp.PROPERTIES_CANONICAL:
            table["saturation_props"][k] = {
                "value": np.empty(self.N_p_sat, dtype=np.float64),
                "grad_p": np.empty( self.N_p_sat, dtype=np.float64),
            }
            
        
        # Loop from pmin to 0.99 p_crit
        # store sat props using gradients_saturation_forward and gradients_saturation_central
        # gradients_saturation_forward and gradients_saturation_central will use jxp.PQ_inputs of coolprop fluid object
        # calculate grad_p for calculation of coefficients

        for j, logP in list(enumerate(np.asarray(self.logP_sat_vals))):
            p = float(np.exp(logP))
            eps_p = 1e-5 * max(abs(p), 1.0)
            try:
                if self.grad_method == "forward":
                    grads = self._gradients_saturation_forward(fluid, p, 0.0, eps_p)                            
                elif self.grad_method == "central":
                    grads = self._gradients_saturation_central(fluid, p, 0.0, eps_p)
                else:
                    raise ValueError(
                        f"Unknown gradient scheme: {self.grad_method}"
                    )
                success_count += 1

            except Exception as e:
                        print(f"Error:{e}")
                        print(f"j:{j}")

            # store computed properties
            for k, (val, grad_p) in grads.items():
                table["saturation_props"][k]["value"][j] = float(val)
                table["saturation_props"][k]["grad_p"][j] = float(grad_p)
            

        # Convert saturation props to JAX arrays for JIT/vmap compatibility
        for k in jxp.PROPERTIES_CANONICAL:
            for sub in ["value", "grad_p"]:
                table["saturation_props"][k][sub] = jnp.array(
                    table["saturation_props"][k][sub], dtype=jnp.float64
                )

        # Printing outcome
        print(
            f"Successfully evaluated {success_count}/{total_points} points ({frac_success:.2f} %)"
        )
        print(f"Total table generation time: {elapsed:.2f} s")

        # save
        pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(table, f)
        print(f"Saved property table to: {pkl_path}")

        return table



    def _generate_metastable_vapor_property_table(self):
        # Initialize fluid (single component)
        fluid = jxp.Fluid(self.fluid_name, self.backend)
        
        # metadata
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
                metastable_phase=self.metastable_phase,
            ),
        }

        # initialize property dicts for canonical properties
        for k in jxp.PROPERTIES_CANONICAL:
            table[k] = {
                "value": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_h": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_p": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_logP": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_ph": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "grad_hlogP": np.empty((self.N_h, self.N_p), dtype=np.float64),
                "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16), dtype=np.float64),
            }

        # Add enthalpy and pressure as explicit table entries (used for initial guesses)
        # Enthalpy: f(h,p) = h -> df/dh = 1, df/dp = 0
        enthalpy_vals = np.tile(np.asarray(self.h_vals)[:, None], (1, self.N_p))
        pressure_vals = np.tile(
            np.exp(np.asarray(self.logP_vals))[None, :], (self.N_h, 1)
        )
        table["enthalpy"] = {
            "value": enthalpy_vals,
            "grad_h": np.ones_like(enthalpy_vals),
            "grad_p": np.zeros_like(enthalpy_vals),
            "grad_logP": np.zeros_like(enthalpy_vals),
            "grad_ph": np.zeros_like(enthalpy_vals),
            "grad_hlogP": np.zeros_like(enthalpy_vals),
            "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16), dtype=np.float64),
        }
        # Pressure: f(h,p) = p -> df/dp = 1, df/dlogP = p
        table["pressure"] = {
            "value": pressure_vals,
            "grad_h": np.zeros_like(pressure_vals),
            "grad_p": np.ones_like(pressure_vals),
            "grad_logP": pressure_vals,  # grad wrt P times p => here 1 * p
            "grad_ph": np.zeros_like(pressure_vals),
            "grad_hlogP": np.zeros_like(pressure_vals),
            "coeffs": np.empty((self.N_h - 1, self.N_p - 1, 16), dtype=np.float64),
        }


        rho_guess = fluid.get_state(jxp.HmassP_INPUTS, enthalpy_vals[-1][-1], pressure_vals[-1][-1])["density"]
        T_guess = fluid.get_state(jxp.HmassP_INPUTS, enthalpy_vals[-1][-1], pressure_vals[-1][-1])["temperature"]

        # Define the specific keys you want to protect/skip
        keys_to_skip = {'metadata', 'pressure', 'enthalpy'}
        
        # Start evaluating properties
        after_spinodal = False
        total_points = self.N_h * self.N_p
        success_count = 0
        start_time = time.perf_counter()
        with tqdm(
            total=total_points,
            desc="Generating property table",
            ascii=True,
            ncols=80,
            bar_format="{l_bar}{bar}|",
        ) as pbar:
            for j, logP in reversed(list(enumerate(np.asarray(self.logP_vals)))):
                ratio = 1
                p = float(np.exp(logP))

                # 3. Inner Loop: Standard (Min Enthalpy -> Max Enthalpy)
                for i, h in reversed(list(enumerate(np.asarray(self.h_vals)))):
                    # print(f"({i},{j})")
                    # finite difference steps (avoid zero)
                    eps_h = 1e-5 * max(abs(h), 1.0)
                    eps_p = 1e-5 * max(abs(p), 1.0)
                    
                    
                    if after_spinodal == False:
                        try:
                            if self.grad_method == "forward":
                                grads = self._gradients_metastable_forward(fluid, h, p, eps_h, eps_p, rho_guess, T_guess)
                                # print("grads:",grads.items())
                                # print("\n")
                                # print("table:", table.keys())                            
                            elif self.grad_method == "central":
                                grads = self._gradients_metastable_central(fluid, h, p, eps_h, eps_p)
                            else:
                                raise ValueError(
                                    f"Unknown gradient scheme: {self.grad_method}"
                                )
                            success_count += 1

                        except:

                            # TODO: DO NOT DELETE!!!                            
                            for k in table.keys():
                                if k in keys_to_skip:
                                    continue  # Skip this iteration
                                # Fill the rest with fallback values
                                table[k]["value"][i, j] = table[k]["value"][i+1, j]
                                table[k]["grad_h"][i, j] = table[k]["grad_h"][i+1, j]
                                table[k]["grad_p"][i, j] = table[k]["grad_p"][i+1, j]
                                table[k]["grad_logP"][i, j] = table[k]["grad_logP"][i+1, j]
                                table[k]["grad_ph"][i, j] = table[k]["grad_ph"][i+1, j]
                                table[k]["grad_hlogP"][i, j] = table[k]["grad_hlogP"][i+1, j]

                            # for k in table.keys():
                            #     if k in keys_to_skip:
                            #         continue  # Skip this iteration
                            #     # Fill the rest with fallback values
                            #     table[k]["value"][i, j] = 1e-12
                            #     table[k]["grad_h"][i, j] = 1e-12
                            #     table[k]["grad_p"][i, j] = 1e-12
                            #     table[k]["grad_logP"][i, j] = 1e-12
                            #     table[k]["grad_ph"][i, j] = 1e-12
                            #     table[k]["grad_hlogP"][i, j] = 1e-12

                            pbar.update(1)
                            after_spinodal = True
                            continue

                        

                        
                        # Update rho guess and T guess for the next grid point
                        # rho_guess, T_guess = grads["density"][0], grads["temperature"][0]

                        rho_guess = grads["density"][0] + grads["density"][1] * self.delta_h
                        T_guess = grads["temperature"][0] + grads["temperature"][1] * self.delta_h
                        # print(f"point:({i},{j}) | {grads["isothermal_compressibility"][0]}")
                        # store computed properties
                        # NOTE: We use [i, j]. Even though we are looping P backwards, 
                        # 'j' is the correct index for that specific pressure in the matrix.
                        for k, (val, grad_h, grad_p, grad_hp) in grads.items():
                            table[k]["value"][i, j] = float(val)
                            table[k]["grad_h"][i, j] = float(grad_h)
                            table[k]["grad_p"][i, j] = float(grad_p)
                            # store derivatives wrt logP = (d/dP) * P
                            table[k]["grad_logP"][i, j] = float(grad_p * p)
                            table[k]["grad_ph"][i, j] = float(grad_hp)
                            table[k]["grad_hlogP"][i, j] = float(grad_hp * p)

                        if grads["isothermal_bulk_modulus"][0] < 0:
                            after_spinodal = True
                            
                            # for k in table.keys():
                            #     if k in keys_to_skip:
                            #         continue  # Skip this iteration
                            #     # Fill the rest with fallback values
                            #     table[k]["value"][i, j] = 1e-12
                            #     table[k]["grad_h"][i, j] = 1e-12
                            #     table[k]["grad_p"][i, j] = 1e-12
                            #     table[k]["grad_logP"][i, j] = 1e-12
                            #     table[k]["grad_ph"][i, j] = 1e-12
                            #     table[k]["grad_hlogP"][i, j] = 1e-12

                            for k in table.keys():
                                if k in keys_to_skip:
                                    continue  # Skip this iteration
                                # Fill the rest with fallback values
                                table[k]["value"][i, j] = table[k]["value"][i+1, j]
                                table[k]["grad_h"][i, j] = table[k]["grad_h"][i+1, j]
                                table[k]["grad_p"][i, j] = table[k]["grad_p"][i+1, j]
                                table[k]["grad_logP"][i, j] = table[k]["grad_logP"][i+1, j]
                                table[k]["grad_ph"][i, j] = table[k]["grad_ph"][i+1, j]
                                table[k]["grad_hlogP"][i, j] = table[k]["grad_hlogP"][i+1, j]

                            # print("After spinodal")
                                
                        # print(f"point:{i,j}")
                        # print(f"IBM:{grads["isothermal_bulk_modulus"][0]}")
                        # print(f"IBM-ratio:{ratio}")

                        pbar.update(1) 

                    else:
                        # print("here")
                        for k in table.keys():
                            if k in keys_to_skip:
                                continue  # Skip this iteration
                            # Fill the rest with fallback values
                            table[k]["value"][i, j] = 1e-12
                            table[k]["grad_h"][i, j] = 1e-12
                            table[k]["grad_p"][i, j] = 1e-12
                            table[k]["grad_logP"][i, j] = 1e-12
                            table[k]["grad_ph"][i, j] = 1e-12
                            table[k]["grad_hlogP"][i, j] = 1e-12

                        pbar.update(1)
                        
                

                rho_guess = fluid.get_state(jxp.HmassP_INPUTS, enthalpy_vals[-1][-1], p)["density"]
                T_guess = fluid.get_state(jxp.HmassP_INPUTS, enthalpy_vals[-1][-1], p)["temperature"]
                after_spinodal = False


                # compute coefficients after full table computed
                for k in jxp.PROPERTIES_CANONICAL:
                    table[k]["coeffs"] = compute_bicubic_coefficients(
                        table[k]["value"],
                        table[k]["grad_h"],
                        table[k]["grad_logP"],
                        table[k]["grad_hlogP"],
                        self.delta_h,
                        self.delta_logP,
                    )

                # compute enthalpy/pressure coeffs too (they were created earlier)
                table["enthalpy"]["coeffs"] = compute_bicubic_coefficients(
                    table["enthalpy"]["value"],
                    table["enthalpy"]["grad_h"],
                    table["enthalpy"]["grad_logP"],
                    table["enthalpy"]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )
                table["pressure"]["coeffs"] = compute_bicubic_coefficients(
                    table["pressure"]["value"],
                    table["pressure"]["grad_h"],
                    table["pressure"]["grad_logP"],
                    table["pressure"]["grad_hlogP"],
                    self.delta_h,
                    self.delta_logP,
                )

        # report
        frac_success = success_count / total_points * 100.0
        elapsed = time.perf_counter() - start_time
        # print(
        #     f"Successfully evaluated {success_count}/{total_points} points ({frac_success:.2f} %)"
        # )
        # print(f"Total table generation time: {elapsed:.2f} s")

        # convert to jax arrays for all relevant sub-keys
        for k in table.keys():
            # skip metadata dictionary
            if k == "metadata":
                continue
            for sub in [
                "value",
                "grad_h",
                "grad_p",
                "grad_logP",
                "grad_ph",
                "grad_hlogP",
                "coeffs",
            ]:
                # some entries (like coeffs) might already be jnp arrays but this is safe
                table[k][sub] = jnp.array(table[k][sub], dtype=jnp.float64)

        print("Generating saturation properties table...")

        table["saturation_props"] = {}
        for k in jxp.PROPERTIES_CANONICAL:
            table["saturation_props"][k] = {
                "value": np.empty(self.N_p_sat, dtype=np.float64),
                "grad_p": np.empty( self.N_p_sat, dtype=np.float64),
            }
            
        
        # Loop from pmin to 0.99 p_crit
        # store sat props using gradients_saturation_forward and gradients_saturation_central
        # gradients_saturation_forward and gradients_saturation_central will use jxp.PQ_inputs of coolprop fluid object
        # calculate grad_p for calculation of coefficients

        for j, logP in list(enumerate(np.asarray(self.logP_sat_vals))):
            p = float(np.exp(logP))
            eps_p = 1e-5 * max(abs(p), 1.0)
            try:
                if self.grad_method == "forward":
                    grads = self._gradients_saturation_forward(fluid, p, 1.0, eps_p)                            
                elif self.grad_method == "central":
                    grads = self._gradients_saturation_central(fluid, p, 1.0, eps_p)
                else:
                    raise ValueError(
                        f"Unknown gradient scheme: {self.grad_method}"
                    )
                success_count += 1

            except Exception as e:
                        print(f"Error:{e}")
                        print(f"j:{j}")

            # store computed properties
            for k, (val, grad_p) in grads.items():
                table["saturation_props"][k]["value"][j] = float(val)
                table["saturation_props"][k]["grad_p"][j] = float(grad_p)


        # Convert saturation props to JAX arrays for JIT/vmap compatibility
        for k in jxp.PROPERTIES_CANONICAL:
            for sub in ["value", "grad_p"]:
                table["saturation_props"][k][sub] = jnp.array(
                    table["saturation_props"][k][sub], dtype=jnp.float64
                )

        # Printing generation results
        print(
            f"Successfully evaluated {success_count}/{total_points} points ({frac_success:.2f} %)"
        )
        print(f"Total table generation time: {elapsed:.2f} s")

        # save
        pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(table, f)
        print(f"Saved property table to: {pkl_path}")

        return table





    # ------------------ Gradient helpers ------------------
    @staticmethod
    def _gradients_forward(fluid, h, p, eps_h, eps_p):
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
            grad_h = (fhp[k] - fhm[k]) / (2.0 * eps_h)
            grad_p = (fph[k] - fpm[k]) / (2.0 * eps_p)
            grad_hp = (fh_p[k] - fh_m[k] - fm_p[k] + fm_m[k]) / (4.0 * eps_h * eps_p)
            grads[k] = (val, grad_h, grad_p, grad_hp)
        return grads

    @staticmethod
    def _gradients_metastable_forward(fluid, h, p, eps_h, eps_p, rho_guess, T_guess):
        rhoT_guess = np.array([rho_guess, T_guess])        

        f0 = fluid.get_state_metastable(
            prop_1 = "p", prop_1_value = p, prop_2 = "h", prop_2_value = h, rhoT_guess=rhoT_guess,)
        fh = fluid.get_state_metastable(prop_1 ="p", prop_1_value = p, prop_2 ="h", prop_2_value = h + eps_h, rhoT_guess=rhoT_guess,)
        fp = fluid.get_state_metastable(prop_1 ="p", prop_1_value = p + eps_p, prop_2 = "h", prop_2_value = h, rhoT_guess=rhoT_guess,)
        fhp = fluid.get_state_metastable(prop_1 ="p", prop_1_value = p + eps_p, prop_2 ="h", prop_2_value = h + eps_h, rhoT_guess=rhoT_guess,)

        grads = {}
        for k in jxp.PROPERTIES_CANONICAL:
            val = f0[k]
            grad_h = (fh[k] - f0[k]) / eps_h
            grad_p = (fp[k] - f0[k]) / eps_p
            grad_hp = (fhp[k] - fh[k] - fp[k] + f0[k]) / (eps_h * eps_p)
            grads[k] = (val, grad_h, grad_p, grad_hp)
        return grads

    @staticmethod
    def _gradients_saturation_forward(fluid, p,Q , eps_p):
        f0 = fluid.get_state(jxp.PQ_INPUTS, p, Q)
        fp = fluid.get_state(jxp.PQ_INPUTS, p + eps_p, Q)

        grads = {}
        for k in jxp.PROPERTIES_CANONICAL:
            val = f0[k]
            grad_p = (fp[k] - f0[k]) / eps_p
            grads[k] = (val, grad_p)
        return grads
    
    @staticmethod
    def _gradients_saturation_central(fluid, p,Q , eps_p):
        f0 = fluid.get_state(jxp.PQ_INPUTS, p, Q)
        fp_plus = fluid.get_state(jxp.PQ_INPUTS, p + eps_p, Q)
        fp_minus = fluid.get_state(jxp.PQ_INPUTS, p - eps_p, Q)

        grads = {}
        for k in jxp.PROPERTIES_CANONICAL:
            val = f0[k]
            grad_p = (fp_plus[k] - fp_minus[k]) / 2*eps_p
            grads[k] = (val, grad_p)
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
        # jax.debug.print("input:{ip}, v1:{v1}, v2:{v2}", ip=input_type, v1=val1, v2=val2)

        # TODO: add a warning for values behond the table limits!!!

        val1 = jnp.asarray(val1)
        val2 = jnp.asarray(val2)
        val1, val2 = jnp.broadcast_arrays(val1, val2)

        batched_fn = jax.vmap(lambda v1, v2: self._get_state_scalar(input_type, v1, v2))

        props_batched = batched_fn(val1.ravel(), val2.ravel())

        # reshape leaves to broadcasted shape using correct jax API
        props = jax.tree.map(lambda x: x.reshape(val1.shape), props_batched)

        return props
    
    def get_state_saturation(self, pressure):

        # TODO: add a warning for values behond the table limits!!!

        sat_props = self._interp_sat_p(pressure)

        return sat_props



    @eqx.filter_jit
    def _get_state_scalar(self, input_type, val1, val2):
        fn = self.PROPERTY_CALCULATORS.get(input_type)
        if fn is None:
            raise KeyError(
                f"Unsupported input_type: {input_type}. Available keys: {list(self.PROPERTY_CALCULATORS.keys())}"
            )
        return fn(self, val1, val2)

    def _interp_h_p(self, h, p):

        out_of_range = (h < self.h_min) | (h > self.h_max) | (p < self.p_min) | (p > self.p_max)
        jax.debug.callback(
            lambda oor, h_, p_: print(
                f"Extrapolation attempted in table {self.table_name}. Values out of interpolation range: h={h_}, p={p_}"
            ) if oor else None,
            out_of_range, h, p,
            ordered=True,
        )

        ii = (h - self.h_min) / self.delta_h
        jj = (jnp.log(p) - jnp.log(self.p_min)) / self.delta_logP

        i = jnp.clip(jnp.floor(ii).astype(int), 0, self.N_h - 2)
        j = jnp.clip(jnp.floor(jj).astype(int), 0, self.N_p - 2)

        x = ii - i
        y = jj - j

        xm = jnp.array([1.0, x, x * x, x * x * x])
        ym = jnp.array([1.0, y, y * y, y * y * y])
        basis = jnp.kron(ym, xm)

        props = {
            name: jnp.dot(self.table[name]["coeffs"][i, j], basis)
            for name in self.table.keys()
            if name != "metadata" and name != "saturation_props"
        }

        # enforce exact inputs
        props["enthalpy"] = h
        props["pressure"] = p

        return jxp.FluidState(fluid_name=self.fluid_name, **props)

    def _interp_h_y(self, h, y_value, y_name, tol=1e-10, max_steps=64):
        i0 = jnp.argmin(jnp.abs(self.h_vals - h))
        j0 = jnp.argmin(jnp.abs(self.table[y_name]["value"][i0, :] - y_value))
        p0 = self.table["pressure"]["value"][i0, j0]

        def residual(p, _):
            val = self._interp_h_p(h, p)[y_name] - y_value
            return val

        solver = optx.Newton(rtol=tol, atol=tol)
        solution = optx.root_find(
            residual,
            solver,
            y0=p0,
            options={"lower": self.p_min, "upper": self.p_max, "max_steps": max_steps},
        )

        return self._interp_h_p(h, solution.value)

    def _interp_x_p(self, p, x_value, x_name, tol=1e-10, max_steps=64):
        j0 = jnp.argmin(jnp.abs(self.logP_vals - jnp.log(p)))
        i0 = jnp.argmin(jnp.abs(self.table[x_name]["value"][:, j0] - x_value))
        h0 = self.table["enthalpy"]["value"][i0, j0]

        def residual(h, _):
            val = self._interp_h_p(h, p)[x_name] - x_value
            return val

        solver = optx.Newton(rtol=tol, atol=tol)
        solution = optx.root_find(
            residual,
            solver,
            y0=h0,
            options={"lower": self.h_min, "upper": self.h_max, "max_steps": max_steps},
        )

        return self._interp_h_p(solution.value, p)

    def _interp_x_y(self, input_pair, x, y, coarse_step=8, tol=1e-12):
        prop1, prop2 = jxp.INPUT_PAIR_MAP[input_pair]
        prop1 = jxp.ALIAS_TO_CANONICAL[prop1]
        prop2 = jxp.ALIAS_TO_CANONICAL[prop2]

        def to_scaled(h, p):
            h_scaled = (h - self.h_min) / (self.h_max - self.h_min)
            p_scaled = (p - self.p_min) / (self.p_max - self.p_min)
            return h_scaled, p_scaled

        def from_scaled(h_scaled, p_scaled):
            h = h_scaled * (self.h_max - self.h_min) + self.h_min
            p = p_scaled * (self.p_max - self.p_min) + self.p_min
            return h, p

        # coarse scan
        step = max(1, coarse_step)
        h_table = self.table["enthalpy"]["value"][::step, ::step]
        p_table = self.table["pressure"]["value"][::step, ::step]
        val_x = self.table[prop1]["value"][::step, ::step]
        val_y = self.table[prop2]["value"][::step, ::step]
        range_x = jnp.ptp(self.table[prop1]["value"])
        range_y = jnp.ptp(self.table[prop2]["value"])
        # avoid zero-range issues
        range_x = jnp.where(range_x == 0.0, 1.0, range_x)
        range_y = jnp.where(range_y == 0.0, 1.0, range_y)

        errors = ((val_x - x) / range_x) ** 2 + ((val_y - y) / range_y) ** 2
        idx = jnp.argmin(errors)
        i, j = jnp.unravel_index(idx, errors.shape)
        h0_node, p0_node = h_table[i, j], p_table[i, j]
        x0 = jnp.array(to_scaled(h0_node, p0_node))

        def residual(xy, _):
            h, p = from_scaled(*xy)
            props = self._interp_h_p(h, p)
            res_x = (props[prop1] - x) / range_x
            res_y = (props[prop2] - y) / range_y
            return jnp.array([res_x, res_y])

        solver = optx.Newton(rtol=tol, atol=tol)
        solution = optx.root_find(residual, solver, x0, throw=True)

        h_scaled, p_scaled = solution.value
        h, p = from_scaled(h_scaled, p_scaled)
        return self._interp_h_p(h, p)
    
    def _interp_sat_p(self, p):
        jj = (jnp.log(p) - jnp.log(self.p_min)) / self.delta_logP_sat
        j = jnp.clip(jnp.floor(jj).astype(int), 0, self.N_p_sat)

        # Use jnp.take for JAX-traced index compatibility (vmap / JIT)
        def _take(arr, idx):
            return jnp.take(arr, idx, mode="clip")

        # Raw calculations
        p_j  = _take(self.table["saturation_props"]["pressure"]["value"], j)
        p_j1 = _take(self.table["saturation_props"]["pressure"]["value"], j + 1)
        dx = p_j1 - p_j
        t = (p - p_j) / dx

        props = {}

        for k in jxp.PROPERTIES_CANONICAL:
            val_j  = _take(self.table["saturation_props"][k]["value"], j)
            val_j1 = _take(self.table["saturation_props"][k]["value"], j + 1)
            gp_j   = _take(self.table["saturation_props"][k]["grad_p"], j)
            gp_j1  = _take(self.table["saturation_props"][k]["grad_p"], j + 1)

            dy = val_j1 - val_j
            a = gp_j * dx - dy
            b = -gp_j1 * dx + dy

            # Equation (1): Symmetrical form of q(x)
            # q(x) = (1-t)y1 + t*y2 + t(1-t)((1-t)a + t*b)
            one_minus_t = 1 - t
            term1 = one_minus_t * val_j
            term2 = t * val_j1
            correction = t * one_minus_t * (one_minus_t * a + t * b)

            props[k] = term1 + term2 + correction

        props["pressure"] = p

        return props












# ------------------------------------------------------------------
# Bicubic coefficient computation function (unchanged logic)
def compute_bicubic_coefficients(value, grad_h, grad_logP, grad_hlogP, delta_h, delta_logP):
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
    coeffs = np.einsum("ab,ijb->ija", A_MAT, XX)
    return coeffs


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
