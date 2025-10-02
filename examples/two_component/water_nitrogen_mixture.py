import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import METHODS as ODE_METHODS
import jaxprop as props
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import matplotlib as mpl
props.set_plot_options(grid=True)



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
    # Initialize ode_out as a dictionary
    ode_out = {}
    for t_i, y_i in zip(t, y.T):
        _, out = ode_handle(t_i, y_i)

        for key, value in out.items():
            # Initialize with an empty list
            if key not in ode_out:
                ode_out[key] = []
            # Append the value to list of current key
            ode_out[key].append(value)

    # Convert lists to numpy arrays
    for key in ode_out:
        ode_out[key] = np.array(ode_out[key])

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

    # Calculate mass fractions of each component (constant values)
    y_1 = mixture_ratio / (1 + mixture_ratio)
    y_2 = 1 / (1 + mixture_ratio)

    # Initialize fluid and compute inlet state
    fluid_1 = props.Fluid(name=fluid_name_1, backend=backend, exceptions=True)
    fluid_2 = props.Fluid(name=fluid_name_2, backend=backend, exceptions=True)

    # Compute the inlet enthalpy of the mixture (ODE initial value)
    props_in_1 = fluid_1.get_state(props.PT_INPUTS, p_in, T_in)
    props_in_2 = fluid_2.get_state(props.PT_INPUTS, p_in, T_in)
    h_in = y_1 * props_in_1.h + y_2 * props_in_2.h

    # Define the ODE system
    def odefun(t, y):

        # Rename arguments
        p = t
        h, T = y

        # Compute fluid states
        state_1 = fluid_1.get_state(props.PT_INPUTS, p, T)
        state_2 = fluid_2.get_state(props.PT_INPUTS, p, T)

        # Compute mixture thermodynamic properties
        state = props.coolprop.calculate_mixture_properties(state_1, state_2, y_1, y_2)

        # Add individual phases to the mixture properties
        for key, value in state_1.items():
            state[f"{key}_1"] = value
        for key, value in state_2.items():
            state[f"{key}_2"] = value

        state["velocity"] = np.sqrt(2*(h_in - state["enthalpy"] + 1e-6))
        state["Mach"] = state["velocity"] / state["speed_of_sound"]

        # Compute right-hand-side of the ODE
        dhdp = efficiency / state["density"]
        dTdp = (dhdp - state["dhdp_T"]) / state["isobaric_heat_capacity"]

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




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Define base inputs
    fluid_name_1 = "water"
    fluid_name_2 = "nitrogen"
    T_in = 300.0             # K
    p_out = 1e5              # Pa
    efficiency = 0.8
    process_type = "expansion"

    A_throat_mm2 = 250
    A_throat = A_throat_mm2 / 1e6  # mm² -> m²

    # sensitivity parameters
    p_in_list = [13e5]           # Pa
    R_list = [10, 25, 50, 100, 250, 500]      # mass ratio

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
            m_dot = rho_throat * u_throat * A_throat

            # component mass flows
            m_dot_water = m_dot * states["mass_frac_1"][-1]
            m_dot_nitrogen = m_dot * states["mass_frac_2"][-1]

            # outlet conditions
            v_out = states["velocity"][-1]
            Ma_out = states["Mach"][-1]
            vol_frac_in = states["vol_frac_2"][0]
            vol_frac_out = states["vol_frac_2"][-1]

            # 4 stationary blades
            n_blades = 4.
            aspect_ratio = 3.0
            gauge_angle = 50*np.pi/180

            spacing_to_chord = 1.1
            area_out = m_dot / states["velocity"][-1] /  states["density"][-1]

            chord = np.sqrt(area_out / (spacing_to_chord *  np.cos(gauge_angle) * aspect_ratio * n_blades))*1e3


            # store table data
            results.append({
                "T_in [degC]": T_in - 273.15,
                "p_in [bar]": p_in / 1e5,
                "R [-]": R,
                "A_throat [mm2]": A_throat_mm2,
                "mdot [kg/s]": m_dot,
                "mdot_water [kg/s]": m_dot_water,
                "mdot_nitrogen [kg/s]": m_dot_nitrogen,
                "v_out [m/s]": v_out,
                "Ma_out [-]": Ma_out,
                "vol_frac_in [-]": vol_frac_in,
                "vol_frac_out [-]": vol_frac_out,
                "chord [mm]": chord
            })

            # store curve data for plots
            curves.append({
                "p_in": p_in/1e5,
                "R": R,
                "p_bar": p_bar,
                "density": states["density"],
                "vol_frac_2": states["vol_frac_2"],
                "speed_of_sound": states["speed_of_sound"],
                "Mach": states["Mach"],
            })

    # ---- Print table ----
    df = pd.DataFrame(results)
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
        axs[0, 1].plot(c["p_bar"], c["vol_frac_2"], color=color, ls=linestyle)
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

    plt.tight_layout()
    plt.show()