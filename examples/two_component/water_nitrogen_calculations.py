import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import METHODS as ODE_METHODS
import jaxprop as jxp
import matplotlib.pyplot as plt
import pandas as pd
import itertools

jxp.set_plot_options(grid=True)


def flatten_state_dict(prefix, data, out):
    """
    Recursively flattens a dict or BaseState into 'out'.
    Keys become 'prefix.key'.
    """
    if isinstance(data, dict):
        for k, v in data.items():
            flat_key = f"{prefix}.{k}" if prefix else k
            flatten_state_dict(flat_key, v, out)
        return

    # Base case: scalar/array -> store directly
    out[prefix] = data


def postprocess_ode(t, y, ode_handle):
    """
    Post-processes the output of an ordinary differential equation (ODE) solver.

    This function takes the time points and corresponding ODE solution matrix,
    and for each time point, it calls a user-defined ODE handling function to
    process the state of the ODE system. It collects the results into a
    dictionary where each key corresponds to a state variable and the values
    are numpy arrays of that state variable at each integration step

    Parameters
    ----------
    t : array_like
        Integration points at which the ODE was solved, as a 1D numpy array.
    y : array_like
        The solution of the ODE system, as a 2D numpy array with shape (n,m) where
        n is the number of points and m is the number of state variables.
    ode_handle : callable
        A function that takes in a integration point and state vector and returns a tuple,
        where the first element is ignored (can be None) and the second element
        is a dictionary representing the processed state of the system.

    Returns
    -------
    ode_out : dict
        A dictionary where each key corresponds to a state variable and each value
        is a numpy array containing the values of that state variable at each integration step.
    """
    # return ode_out
    ode_out = {}

    for t_i, y_i in zip(t, y.T):
        _, out_raw = ode_handle(t_i, y_i)

        # flatten nested dict/BaseState
        flat_out = {}
        for key, value in out_raw.items():
            flatten_state_dict(key, value, flat_out)

        # accumulate time series
        for key, value in flat_out.items():
            if key not in ode_out:
                ode_out[key] = []
            ode_out[key].append(value)

    # convert lists to numpy arrays where possible
    for key in ode_out:
        try:
            ode_out[key] = np.array(ode_out[key])
        except Exception:
            # fallback: keep as list of objects
            ode_out[key] = np.array(ode_out[key], dtype=object)

    return ode_out


def barotropic_model_two_component(
    fluid_name_1,
    fluid_name_2,
    mixture_ratio,
    T_in,
    p_in,
    p_out,
    efficiency,
    backend="HEOS",
    process_type=None,
    ODE_solver="lsoda",
    ODE_tolerance=1e-8,
):
    """
    Simulates a polytropic process for a mixture of two different fluids.

    TODO: add model equations and explanation

    Parameters
    ----------
    fluid_name_1 : str
        The name of the first component of the mixture.
    fluid_name_2 : str
        The name of the second component of the mixture.
    mixture_ratio : float
        Mass ratio of the first to the second fluid in the mixture.
    T_in : float
        Inlet temperature of the mixture in Kelvin.
    p_in : float
        Inlet pressure of the mixture in Pascals.
    p_out : float
        Outlet pressure of the mixture in Pascals.
    efficiency : float
        The efficiency of the polytropic process, (between zero and one).
    ODE_solver : str, optional
        The solver to use for the ODE integration. Valid options:

        .. list-table::
            :widths: 20 50
            :header-rows: 1

            * - Solver name
              - Description
            * - ``RK23``
              - Explicit Runge-Kutta method of order 3(2)
            * - ``RK45``
              - Explicit Runge-Kutta method of order 5(4)
            * - ``DOP853``
              - Explicit Runge-Kutta method of order 8
            * - ``Radau``
              - Implicit Runge-Kutta method of the Radau IIA family of order 5
            * - ``BDF``
              - Implicit multi-step variable-order (1 to 5) method based on a backward differentiation formula for the derivative approximation
            * - ``LSODA``
              - Adams/BDF method with automatic stiffness detection and switching

        See `Scipy solver_ivp() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_  for more info.
        Recommended solvers: ``BDF``, ``LSODA``, or ``Radau`` for stiff problems or ``RK45`` for non-stiff problems with smooth blending.

    ODE_tolerance : float, optional
        The relative and absolute tolerance for the ODE solver.

    Returns
    -------
    states : dictionary of arrays
        A dictionary of Numpy arrays representing the properties of the fluid at each evaluation point.
    solution : scipy.integrate.OdeResult
        The result of the ODE integration containing information about the solver process.
    """

    # Check if the provided ODE_solver is valid
    valid_solvers_ode = list(ODE_METHODS.keys())
    if ODE_solver not in valid_solvers_ode:
        error_message = (
            f"Invalid ODE solver '{ODE_solver}' provided. "
            f"Valid solver are: {', '.join(valid_solvers_ode)}."
        )
        raise ValueError(error_message)

    # Pre-process efficiency value
    if not (0.0 <= efficiency <= 1.0):
        raise ValueError(
            f"Efficiency must be between 0 and 1. Provided: {efficiency:.3f}"
        )

    if process_type == "compression":
        if efficiency == 0.0:
            raise ValueError(
                "Efficiency cannot be zero for compression (division by zero)."
            )
        efficiency = 1 / efficiency
    elif process_type != "expansion":
        raise ValueError(
            f"Invalid process_type='{process_type}'. Must be 'expansion' or 'compression'."
        )

    # Initialize fluid and compute inlet state
    fluid_1 = jxp.Fluid(name=fluid_name_1, backend=backend, exceptions=True)
    fluid_2 = jxp.Fluid(name=fluid_name_2, backend=backend, exceptions=True)

    # Compute the inlet enthalpy of the mixture (ODE initial value)
    state_in = jxp.get_mixture_state(fluid_1, fluid_2, p_in, T_in, mixture_ratio)
    h_in = state_in.enthalpy

    # Define the ODE system
    def odefun(t, y):

        # Compute mixture thermodynamic properties
        p = t
        h, T = y
        state = jxp.get_mixture_state(fluid_1, fluid_2, p, T, mixture_ratio).to_dict()

        # Compute derived properties
        state["velocity"] = np.sqrt(2*(h_in - state["enthalpy"] + 1e-6))
        state["Mach"] = state["velocity"] / state["speed_of_sound"]

        # Compute right-hand-side of the ODE
        dhdp = efficiency / state["density"]
        dTdp = dhdp / state["isobaric_heat_capacity"] - state["joule_thomson"]
        return [dhdp, dTdp], state

    # Solve polytropic expansion differential equation
    ode_sol = solve_ivp(
        fun=lambda p, h: odefun(p, h)[0],  # Get only first output
        t_span=[p_in, p_out],
        t_eval=np.linspace(p_in, p_out, 200),
        y0=[h_in, T_in],
        method=ODE_solver,
        rtol=ODE_tolerance,
        atol=ODE_tolerance,
    )
    if not ode_sol.success:
        raise Exception(ode_sol.message)

    # Postprocess solution
    states = postprocess_ode(ode_sol.t, ode_sol.y, odefun)

    return states, ode_sol


if __name__ == "__main__":

    # Define base inputs
    fluid_name_1 = "water"
    fluid_name_2 = "air"
    T_in = 25 + 273.15            # K
    p_out = 1e5              # Pa
    efficiency = 0.8
    process_type = "expansion"

    A_throat_mm2 = 250
    A_throat = A_throat_mm2 / 1e6  # mm² -> m²

    efficiency_pump = 0.7
    compressor_specific_power = 0.126  # kW/(m3/hr)

    # sensitivity parameters
    p_in_list = [7e5, 13e5]           # Pa
    R_list = [20, 30, 40, 50, 100, 150, 200, 300, 400, 500]      # mass ratio

    # containers
    results = []
    curves = []   # store for plotting

    for p_in in p_in_list:
        for R in R_list:

            # Run model
            states, sol = barotropic_model_two_component(
                fluid_name_1=fluid_name_1,
                fluid_name_2=fluid_name_2,
                mixture_ratio=R,
                T_in=T_in,
                p_in=p_in,
                p_out=p_out,
                efficiency=efficiency,
                process_type=process_type,
                ODE_solver="LSODA",
                ODE_tolerance=1e-6,
            )

            p_bar = states["pressure"] / 1e5

            # throat condition
            idx_throat = np.argmin(np.abs(states["Mach"] - 1.0))
            rho_throat = states["density"][idx_throat]
            u_throat = states["velocity"][idx_throat]
            mu_throat = states["viscosity"][idx_throat]
            Re_throat = u_throat*rho_throat*np.sqrt(A_throat)/mu_throat
            m_dot = rho_throat * u_throat * A_throat

            # component mass flows
            m_dot_water = m_dot * states["mass_fraction_1"][-1]
            m_dot_air = m_dot * states["mass_fraction_2"][-1]
            V_dot_air = m_dot_air / states["component_2.density"][-1]
            V_dot_water = m_dot_water / states["component_1.density"][0]
            power_pump = V_dot_water * (p_in - p_out) / efficiency_pump

            # outlet conditions
            v_out = states["velocity"][-1]
            Ma_out = states["Mach"][-1]
            vol_frac_in = states["volume_fraction_2"][0]
            vol_frac_out = states["volume_fraction_2"][-1]

            # 4 stationary blades
            n_blades = 4.
            aspect_ratio = 3.0
            gauge_angle = 50*np.pi/180
            spacing_to_chord = 1.1
            area_out = m_dot / states["velocity"][-1] /  states["density"][-1]
            chord = np.sqrt(area_out / (spacing_to_chord *  np.cos(gauge_angle) * aspect_ratio * n_blades))*1e3

            Re_blades = states["velocity"][-1]*states["density"][-1]*chord/states["viscosity"][-1]

            # store table data
            results.append({
                "T_in [degC]": T_in - 273.15,
                "p_in [bar]": p_in / 1e5,
                "R [-]": R,
                "mass_frac_air [%]": 1/(1+R)*100,
                "vol_frac_in [%]": vol_frac_in*100,
                "vol_frac_out [%]": vol_frac_out*100,
                # "A_throat [mm2]": A_throat_mm2,
                "m_total [kg/min]": m_dot*60,
                "V_water [m3/h]": V_dot_water*3600,
                "W_pump [kW]": power_pump/1000,
                "V_air [m3/h]": V_dot_air*3600,
                "W_comp [kW]": V_dot_air*3600 * compressor_specific_power,
                "thrust [N]": m_dot * v_out,
                "v_out [m/s]": v_out,
                "Ma_out [-]": Ma_out,
                "Re_throat": Re_throat/1e6,
                "Re_blades": Re_blades/1e6,
            })

            # store curve data for plots
            curves.append({
                "p_in": p_in/1e5,
                "R": R,
                "p_bar": p_bar,
                "density": states["density"],
                "volume_fraction_2": states["volume_fraction_2"],
                "speed_of_sound": states["speed_of_sound"],
                "Mach": states["Mach"],
            })

    # ---- Print report ----

    # Input parameters table
    inputs = [
        ("Fluid 1", fluid_name_1, "-"),
        ("Fluid 2", fluid_name_2, "-"),
        ("Nozzle inlet temperature", T_in - 273.15, "degC"),
        ("Nozzle inlet pressure", p_in_list[0]/1e5, "bar"),
        ("Nozle outlet pressure", p_out/1e5, "bar"),
        ("Nozzle throat area", A_throat_mm2, "mm2"),
        ("Nozzle efficiency", efficiency, "-"),
        ("Pump efficiency", efficiency_pump, "-"),
        ("Compressor specific power", compressor_specific_power, "kW/(m3/h)"),
    ]

    df_inputs = pd.DataFrame(inputs, columns=["Parameter", "Value", "Unit"])

    print("=============================================================")
    print("=================== Case input parameters ===================")
    print("=============================================================")
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
        print(df_inputs.to_string(index=False))

    # Results table
    df = pd.DataFrame(results)

    print("")
    print("=============================================================")
    print("==================== Case output results ====================")
    print("=============================================================")
    with pd.option_context('display.float_format', '{:.3f}'.format):
        print(df.to_string(index=False))

    # ---- Plot results ----
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    
    # define cyclers
    linestyle_cycler = itertools.cycle(['-', '--', '-.', ':'])

    # unique sets
    unique_R = sorted(set(c["R"] for c in curves))
    unique_p = sorted(set(c["p_in"] for c in curves))

    # sample magma colormap between 0.25 and 0.80
    cmap = plt.get_cmap("magma")
    color_positions = np.linspace(0.25, 0.80, len(unique_R))
    colors = {R: cmap(pos) for R, pos in zip(unique_R, color_positions)}

    # cycle linestyles for pressures
    linestyle_cycler = itertools.cycle(['-', '--', '-.', ':'])
    linestyles = {p: next(linestyle_cycler) for p in unique_p}

    for c in curves:
        color = colors[c["R"]]
        linestyle = linestyles[c["p_in"]]
        label = f"Mass ratio = {c['R']}"

        axs[0, 0].plot(c["p_bar"], c["density"], color=color, ls=linestyle, label=label)
        axs[0, 1].plot(c["p_bar"], c["volume_fraction_2"], color=color, ls=linestyle)
        axs[1, 0].plot(c["p_bar"], c["speed_of_sound"], color=color, ls=linestyle)
        axs[1, 1].plot(c["p_bar"], c["Mach"], color=color, ls=linestyle)

    axs[0, 0].set_ylabel("Density [kg/m³]")
    axs[0, 1].set_ylabel("Void fraction [-]")
    axs[1, 0].set_ylabel("Speed of sound [m/s]")
    axs[1, 0].set_xlabel("Pressure [bar]")
    axs[1, 1].set_ylabel("Mach number [-]")
    axs[1, 1].set_xlabel("Pressure [bar]")

    for ax in axs.ravel():
        ax.grid()
        ax.set_xlim(max(c["p_bar"]), min(c["p_bar"]))

    axs[0, 0].legend(fontsize=8, ncol=1)

    plt.tight_layout(pad=1)
    plt.savefig("results.png", dpi=300)
    plt.show()