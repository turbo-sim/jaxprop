import difflib
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import pysolver_view as psv

from functools import wraps

from . import core_calculations as core

from ..helpers_props import (
    FluidState,
    INPUT_TYPE_MAP,
    # MEANLINE_PROPERTIES,
    LABEL_MAPPING,
)


def _handle_computation_exceptions(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            # Perform the computation
            result = func(self, *args, **kwargs)
            self.converged_flag = True
            return result
        except Exception as e:
            self.converged_flag = False
            if self.exceptions:
                input_type = args[0]
                value_1 = args[1]
                value_2 = args[2]
                label = INPUT_TYPE_MAP.get(
                    input_type, f"Unknown input type ({input_type})"
                )

                msg = (
                    f"Thermodynamic property calculations failed:\n"
                    f"  Input type : {label}\n"
                    f"  Property 1 : {value_1}\n"
                    f"  Property 2 : {value_2}\n"
                    f"  Error      : {str(e)}"
                )
                raise RuntimeError(msg)
            return None

    return wrapper


class Fluid:
    """
    Represents a fluid with various thermodynamic properties computed via CoolProp.

    This class provides a convenient interface to CoolProp for various fluid property calculations.

    Critical and triple point properties are computed upon initialization and stored internally for convenience.

    Attributes
    ----------
    name : str
        Name of the fluid.
    backend : str
        Backend used for CoolProp, default is 'HEOS'.
    exceptions : bool
        Determines if exceptions should be raised during state calculations. Default is True.
    converged_flag : bool
        Flag indicating whether properties calculations converged.
    properties : dict
        Dictionary of various fluid properties. Accessible directly as attributes (e.g., `fluid.p` for pressure).
    critical_point : FluidState
        Properties at the fluid's critical point.
    triple_point_liquid : FluidState
        Properties at the fluid's triple point in the liquid state.
    triple_point_vapor : FluidState
        Properties at the fluid's triple point in the vapor state.

    Methods
    -------
    get_state(input_type, prop_1, prop_2):
        Set the thermodynamic state of the fluid using specified property inputs.

    Examples
    --------

    Calculating properties with Fluid.get_state()

    >>> fluid = bpy.Fluid(name="Water", backend="HEOS")
    >>> state = fluid.get_state(bpy.PT_INPUTS, 101325, 300)
    >>> print(f"Water density is {state.rho:0.2f} kg/m3 at p={state.p:0.2f} Pa and T={state.T:0.2f} K")
    Water density is 996.56 kg/m3 at p=101325.00 Pa and T=300.00 K

    >>> fluid = bpy.Fluid(name="Air", backend="HEOS")
    >>> state = fluid.get_state(bpy.PT_INPUTS, 101325, 300)
    >>> print(f"Air heat capacity ratio is {state.gamma:0.2f} at p={state.p:0.2f} Pa and T={state.T:0.2f} K")
    Air heat capacity ratio is 1.40 at p=101325.00 Pa and T=300.00 K


    Accessing critical point properties:

    >>> fluid.critical_point.p  # Retrieves critical pressure
    >>> fluid.critical_point['T']  # Retrieves critical temperature

    Accessing triple point properties:

    >>> fluid.triple_point_liquid.h  # Retrieves liquid enthalpy at the triple point
    >>> fluid.triple_point_vapor.s  # Retrieves vapor entropy at the triple point

    """

    def __getstate__(self):
        """Strip CoolProp AbstractState to make the object pickleable."""
        state = self.__dict__.copy()
        state["_AS"] = None  # Drop the unpickleable CoolProp object
        return state

    def __setstate__(self, state):
        """Restore Fluid without AbstractState (can be rebuilt later)."""
        self.__dict__.update(state)
        self._AS = None  # Leave it unset or rebuild later

    def __init__(
        self,
        name,
        backend="HEOS",
        exceptions=True,
        identifier=None,
        # initialize_critical=True,
        # initialize_triple=True,
    ):
        self.name = name
        self.backend = backend
        # self._AS = CP.AbstractState(backend, name)
        self._AS = try_initialize_fluid(name, backend)
        self.abstract_state = self._AS
        self.exceptions = exceptions
        self.converged_flag = False
        self.identifier = identifier if identifier is not None else name

        # Initialize variables
        self.sat_liq = None
        self.sat_vap = None
        self.spdl_liq = None
        self.spdl_vap = None
        self.pseudo_critical_line = None
        # self.quality_grid = None
        self.q_mesh = None
        self.graphic_elements = {}

        # Get critical and triple point properties
        if is_pure_substance(self._AS):
            self.critical_point = self._compute_critical_point()
            self.triple_point_liquid = self._compute_triple_point_liquid()
            self.triple_point_vapor = self._compute_triple_point_vapor()

        # Pressure and temperature limits
        self.p_min = 1
        self.p_max = self._AS.pmax()
        self.T_min = self._AS.Tmin()
        self.T_max = self._AS.Tmax()

        # # Define reference state
        # self.reference_state = self.critical_point

    def _compute_critical_point(self):
        """Calculate the properties at the critical point"""
        d_crit = self._AS.rhomass_critical()
        T_crit = self._AS.T_critical()
        p_crit = self._AS.p_critical()
        # state_crit = self.get_state(DmassT_INPUTS, rho_crit, T_crit-1e-12)
        try:
            state_crit = self.get_state(CP.DmassT_INPUTS, d_crit, T_crit - 1e-5)
        except:
            state_crit = self.get_state(CP.PT_INPUTS, p_crit, T_crit - 1e-5)
        return state_crit

    def _compute_triple_point_liquid(self):
        """Calculate the properties at the triple point (liquid state)"""
        return self.get_state(CP.QT_INPUTS, 0.00, self._AS.Ttriple())

    def _compute_triple_point_vapor(self):
        """Calculate the properties at the triple point (vapor state)"""
        return self.get_state(CP.QT_INPUTS, 1.00, self._AS.Ttriple())

    @_handle_computation_exceptions
    def get_state(
        self,
        input_type,
        prop_1,
        prop_2,
        generalize_quality=False,
        supersaturation=False,
    ):
        r"""
        Set the thermodynamic state of the fluid using the CoolProp low level interface.

        This method updates the thermodynamic state of the fluid in the CoolProp ``abstractstate`` object
        using the given input properties. It then calculates either single-phase or two-phase
        properties based on the current phase of the fluid.

        If the calculation of properties fails, `converged_flag` is set to False, indicating an issue with
        the property calculation. Otherwise, it's set to True.

        Parameters
        ----------
        input_type : int
            The variable pair used to define the thermodynamic state. This should be one of the
            predefined input pairs in CoolProp, such as ``PT_INPUTS`` for pressure and temperature.
        prop_1 : float
            The first property value corresponding to the input type.
        prop_2 : float
            The second property value corresponding to the input type.

        Returns
        -------
        barotropy.State
            A State object containing the fluid properties

        Raises
        ------
        Exception
            If `throw_exceptions` attribute is set to True and an error occurs during property calculation,
            the original exception is re-raised.


        """
        props = core.compute_properties_coolprop(
            self._AS,
            input_type,
            prop_1,
            prop_2,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
        )

        return FluidState(
            fluid_name=self.name,
            identifier=self.identifier,
            **props,
        )
    

    @_handle_computation_exceptions
    def get_state_equilibrium(
        self,
        prop_1,
        prop_1_value,
        prop_2,
        prop_2_value,
        rhoT_guess=None,
        supersaturation=True,
        generalize_quality=True,
        solver_algorithm="hybr",
        solver_tolerance=1e-6,
        solver_max_iterations=100,
        print_convergence=False,
    ):
        r"""
        Calculate fluid properties according to thermodynamic equilibrium.

        .. note::

            For a detailed description of input arguments and calculation methods, see the
            documentation of the function :ref:`compute_properties <compute_properties>`.

        """
        props = core.compute_properties(
            self._AS,
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            calculation_type="equilibrium",
            rhoT_guess_equilibrium=rhoT_guess,
            supersaturation=supersaturation,
            generalize_quality=generalize_quality,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
        )
        return FluidState(
            fluid_name=self.name,
            identifier=self.identifier,
            **props,
        )
    
    @_handle_computation_exceptions
    def get_state_metastable(
        self,
        prop_1,
        prop_1_value,
        prop_2,
        prop_2_value,
        rhoT_guess=None,
        supersaturation=True,
        generalize_quality=True,
        solver_algorithm="hybr",
        solver_tolerance=1e-6,
        solver_max_iterations=100,
        print_convergence=False,
    ):
        r"""
        Calculate fluid properties assuming phase metastability

        .. note::

            For a detailed description of input arguments and calculation methods, see the
            documentation of the function :ref:`compute_properties <compute_properties>`.

        """
        if prop_1 == "rho" and prop_2 == "T":
            props = core.compute_properties_metastable_rhoT(
                abstract_state=self._AS,
                rho=prop_1_value,
                T=prop_2_value,
                generalize_quality=generalize_quality,
                supersaturation=supersaturation,
            )
        elif prop_1 == "T" and prop_2 == "rho":
            props = core.compute_properties_metastable_rhoT(
                abstract_state=self._AS,
                rho=prop_2_value,
                T=prop_1_value,
                generalize_quality=generalize_quality,
                supersaturation=supersaturation,
            )
        else:
            props = core.compute_properties(
                self._AS,
                prop_1=prop_1,
                prop_1_value=prop_1_value,
                prop_2=prop_2,
                prop_2_value=prop_2_value,
                calculation_type="metastable",
                rhoT_guess_metastable=rhoT_guess,
                supersaturation=supersaturation,
                generalize_quality=generalize_quality,
                solver_algorithm=solver_algorithm,
                solver_tolerance=solver_tolerance,
                solver_max_iterations=solver_max_iterations,
                print_convergence=print_convergence,
            )
        
        return FluidState(
            fluid_name=self.name,
            identifier=self.identifier,
            **props,
        )

    @_handle_computation_exceptions
    def get_state_blending(
        self,
        prop_1,
        prop_1_value,
        prop_2,
        prop_2_value,
        rhoT_guess_equilibrium,
        rhoT_guess_metastable,
        blending_variable,
        blending_onset,
        blending_width,
        phase_change,
        supersaturation=True,
        generalize_quality=True,
        solver_algorithm="hybr",
        solver_tolerance=1e-6,
        solver_max_iterations=100,
        print_convergence=False,
    ):
        r"""
        Calculate fluid properties by blending equilibrium and metastable properties

        .. note::

            For a detailed description of input arguments and calculation methods, see the
            documentation of the function :ref:`compute_properties <compute_properties>`.

        """
        blended, equilibrium, metastable = core.compute_properties(
            self._AS,
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            calculation_type="blending",
            rhoT_guess_equilibrium=rhoT_guess_equilibrium,
            rhoT_guess_metastable=rhoT_guess_metastable,
            blending_variable=blending_variable,
            blending_onset=blending_onset,
            blending_width=blending_width,
            phase_change=phase_change,
            supersaturation=supersaturation,
            generalize_quality=generalize_quality,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
        )

        return (
            FluidState(fluid_name=self.name, identifier=self.identifier, **blended),
            FluidState(fluid_name=self.name, identifier=self.identifier, **equilibrium),
            FluidState(fluid_name=self.name, identifier=self.identifier, **metastable),
        )

    

    def plot_phase_diagram(
        self,
        x_prop="s",
        y_prop="T",
        axes=None,
        N=100,
        plot_saturation_line=True,
        plot_critical_point=True,
        plot_triple_point_liquid=False,
        plot_triple_point_vapor=False,
        plot_spinodal_line=False,
        spinodal_line_color=0.5 * np.array([1, 1, 1]),
        spinodal_line_width=1.25,
        spinodal_line_method="slsqp",  # Alternative is slsqp
        spinodal_line_use_previous=False,  # True is not as robust
        plot_quality_isolines=False,
        plot_two_phase_patch=False,
        plot_pseudocritical_line=False,
        quality_levels=np.linspace(0.1, 1.0, 10),
        quality_labels=False,
        show_in_legend=False,
        x_scale="linear",
        y_scale="linear",
        dT_crit=1.00,
    ):

        if axes is None:
            # axes = plt.gca()
            fig, axes = plt.subplots(figsize=(6, 5))
            axes.set_xlabel(LABEL_MAPPING.get(x_prop, x_prop))
            axes.set_ylabel(LABEL_MAPPING.get(y_prop, y_prop))
            axes.set_xscale(x_scale)
            axes.set_yscale(y_scale)

        # Saturation line
        if plot_saturation_line:
            if self.sat_liq is None or self.sat_vap is None:
                self.sat_liq, self.sat_vap = compute_saturation_line(
                    self, N, dT_crit=dT_crit
                )
            x = jnp.concatenate([jnp.flip(self.sat_liq[x_prop]), self.sat_vap[x_prop]])
            y = jnp.concatenate([jnp.flip(self.sat_liq[y_prop]), self.sat_vap[y_prop]])
            label = self._get_label("Saturation line", show_in_legend)
            params = {"label": label, "color": "black"}
            self._graphic_saturation_line = self._plot_or_update_line(
                axes,
                x,
                y,
                "saturation_line",
                **params,
            )
        else:
            self._set_visibility(axes, "saturation_line", False)

        # Spinodal line
        if plot_spinodal_line:
            if self.spdl_liq is None or self.spdl_vap is None:
                self.spdl_liq, self.spdl_vap = compute_spinodal_line(
                    self,
                    N=N,
                    method=spinodal_line_method,
                    use_previous_as_initial_guess=spinodal_line_use_previous,
                    supersaturation=False,
                    dT_crit=dT_crit,
                )
            x = jnp.concatenate([jnp.flip(self.spdl_liq[x_prop]), self.spdl_vap[x_prop]])
            y = jnp.concatenate([jnp.flip(self.spdl_liq[y_prop]), self.spdl_vap[y_prop]])
            label = self._get_label("Spinodal line", show_in_legend)
            params = {
                "label": label,
                "color": spinodal_line_color,
                "linewidth": spinodal_line_width,
            }
            self._graphic_spinodal_line = self._plot_or_update_line(
                axes,
                x,
                y,
                "spinodal_line",
                **params,
            )
        else:
            self._set_visibility(axes, "spinodal_line", False)

        # Plot pseudocritical line
        if plot_pseudocritical_line:
            if self.pseudo_critical_line is None:
                self.pseudo_critical_line = compute_pseudocritical_line(self)
            x = self.pseudo_critical_line[x_prop]
            y = self.pseudo_critical_line[y_prop]
            label = self._get_label("Pseudocritical line", show_in_legend)
            params = {
                "label": label,
                "color": "black",
                "linestyle": "--",
                "linewidth": 0.75,
            }
            self._graphic_pseudocritical_line = self._plot_or_update_line(
                axes,
                x,
                y,
                "pseudocritical_line",
                **params,
            )
        else:
            self._set_visibility(axes, "pseudocritical_line", False)

        # Plot quality isolines
        if plot_quality_isolines:
            if self.q_mesh is None:
                self.q_mesh = compute_quality_grid(
                    self, N, quality_levels, dT_crit=dT_crit
                )
            x = self.q_mesh[x_prop]
            y = self.q_mesh[y_prop]
            _, m = np.shape(x)
            z = np.tile(quality_levels, (m, 1)).T
            params = {"colors": "black", "linestyles": ":", "linewidths": 0.75}
            self._graphics_q_lines = self._plot_or_update_contours(
                axes,
                x,
                y,
                z,
                quality_levels,
                "quality_isolines",
                **params,
            )

            if quality_labels:
                axes.clabel(self._graphics_q_lines, fontsize=9, rightside_up=True)

        else:
            # Remove existing contour lines if they exist
            if "quality_isolines" in self.graphic_elements.get(axes, {}):
                self.graphic_elements[axes]["quality_isolines"].remove()
                del self.graphic_elements[axes]["quality_isolines"]

        # Plot critical point
        params = {
            "color": "black",
            "marker": "o",
            "markersize": 4.5,
            "markerfacecolor": "w",
        }
        if plot_critical_point:
            x = self.critical_point[x_prop]
            y = self.critical_point[y_prop]
            label = self._get_label("Critical point", show_in_legend)
            self._graphic_critical_point = self._plot_or_update_line(
                axes,
                x,
                y,
                "critical_point",
                label=label,
                **params,
            )
        else:
            self._set_visibility(axes, "critical_point", False)

        # Plot liquid triple point
        if plot_triple_point_liquid:
            x = self.triple_point_liquid[x_prop]
            y = self.triple_point_liquid[y_prop]
            label = self._get_label("Triple point liquid", show_in_legend)
            self._graphic_triple_point_liquid = self._plot_or_update_line(
                axes,
                x,
                y,
                "triple_point_liquid",
                label=label,
                **params,
            )
        else:
            self._set_visibility(axes, "triple_point_liquid", False)

        # Plot vapor triple point
        if plot_triple_point_vapor:
            x = self.triple_point_vapor[x_prop]
            y = self.triple_point_vapor[y_prop]
            label = self._get_label("Triple point vapor", show_in_legend)
            self._graphic_triple_point_vapor = self._plot_or_update_line(
                axes,
                x,
                y,
                "triple_point_vapor",
                label=label,
                **params,
            )
        else:
            self._set_visibility(axes, "triple_point_vapor", False)

        if plot_two_phase_patch:
            axes.fill_betweenx(
                y=np.concatenate([self.sat_liq[y_prop][::-1], self.sat_vap[y_prop]]),
                x1=np.concatenate([self.sat_liq[x_prop][::-1], self.sat_vap[x_prop]]),
                x2=np.concatenate([self.sat_vap[x_prop][::-1], self.sat_liq[x_prop]]),
                facecolor="white",
                edgecolor="none",
                # zorder=10
            )

        return axes.figure, axes

    def _get_label(self, label, show_in_legend):
        """Returns the appropriate label value based on whether it should be shown in the legend."""
        return label if show_in_legend else "_no_legend_"

    def _plot_or_update_line(self, axes, x_data, y_data, line_name, **plot_params):
        # Ensure there is a dictionary for this axes
        if axes not in self.graphic_elements:
            self.graphic_elements[axes] = {}

        # Make sure elements are arrays (avoid error when plotting a single point)
        x_data = np.atleast_1d(x_data)
        y_data = np.atleast_1d(y_data)

        # Check if the line exists for this axes
        if line_name in self.graphic_elements[axes]:
            line = self.graphic_elements[axes][line_name]
            line.set_data(x_data, y_data)
            # Update line properties
            for param, value in plot_params.items():
                setattr(line, param, value)
            line.set_visible(True)
        else:
            # Create a new line with the provided plot parameters
            (line,) = axes.plot(x_data, y_data, **plot_params)
            self.graphic_elements[axes][line_name] = line
        return line

    def _plot_or_update_contours(
        self, axes, x_data, y_data, z_data, contour_levels, line_name, **contour_params
    ):

        # Ensure there is a dictionary for this axes
        if axes not in self.graphic_elements:
            self.graphic_elements[axes] = {}

        # Check if the contour exists for this axes
        if line_name in self.graphic_elements[axes]:
            # Remove the old contour
            self.graphic_elements[axes][line_name].remove()

        # Create a new contour
        contour = axes.contour(x_data, y_data, z_data, contour_levels, **contour_params)
        self.graphic_elements[axes][line_name] = contour
        return contour

    def _set_visibility(self, axes, line_name, visible):
        if axes in self.graphic_elements and line_name in self.graphic_elements[axes]:
            self.graphic_elements[axes][line_name].set_visible(visible)


# ------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------ #


def compute_saturation_line(fluid, N=100, dT_crit=0.5):
    """
    Compute the saturation line for a given fluid.

    Parameters
    ----------
    fluid : object
        The fluid object containing thermodynamic properties and methods.
    N : int, optional
        Number of points to compute along the saturation line. Default is 100.

    Returns
    -------
    saturation_liq : FluidState
        Batched FluidState for the liquid saturation line.
    saturation_vap : FluidState
        Batched FluidState for the vapor saturation line.
    """
    # Define temperature array with refinement close to the critical point
    R = 1 - fluid.triple_point_liquid.T / fluid.critical_point.T
    R_crit = dT_crit / fluid.critical_point.T
    t1 = np.logspace(np.log10(R_crit), np.log10(R / 2), int(np.ceil(N / 2)))
    t2 = np.linspace(R / 2, R, int(np.floor(N / 2)))
    # t2 = np.asarray([])
    T_sat = (1 - np.concatenate([t1, t2])) * fluid.critical_point.T

    # Collect states in lists
    liq_states = [fluid.critical_point]  # start with critical point
    vap_states = [fluid.critical_point]

    for T in T_sat:
        liq_states.append(fluid.get_state(CP.QT_INPUTS, 0.00, T))
        vap_states.append(fluid.get_state(CP.QT_INPUTS, 1.00, T))

    # Reverse the liquid branch if needed
    # liq_states = list(reversed(liq_states))

    # Stack into batched FluidState
    saturation_liq = FluidState.stack(liq_states)
    saturation_vap = FluidState.stack(vap_states)

    return saturation_liq, saturation_vap

def compute_pseudocritical_line(fluid, N_points=100):
    """Compute pseudocritical line (approximate, defined here at critical density)."""
    # Define temperature array with refinement close to the critical point
    tau = np.logspace(np.log10(1e-3), np.log10(1), N_points)
    T_range = (1 + tau) * fluid.critical_point.T

    states = []
    for T in T_range:
        if T >= fluid.abstract_state.Tmax():
            break
        state = fluid.get_state(CP.DmassT_INPUTS, fluid.critical_point.d, T)
        states.append(state)

    # Stack all computed states into one batched FluidState
    return FluidState.stack(states)


def compute_spinodal_line(
    fluid,
    N=50,
    method="slsqp",
    use_previous_as_initial_guess=False,
    supersaturation=False,
    dT_crit=0.25,
    tolerance=1e-8,
):
    """
    Compute the spinodal line for a given fluid.

    Parameters
    ----------
    fluid : object
        The fluid object containing thermodynamic properties and methods.
    N : int, optional
        Number of points to compute along the spinodal line. Default is 50.
    method : str, optional
        The optimization method to solve the spinodal point problem ('bfgs' or 'slsqp'). Default is 'bfgs'.
    use_previous_as_initial_guess : bool, optional
        Whether to use the previous point as the initial guess for the next point. Default is False.
    supersaturation : bool, optional
        Whether to compute supersaturation properties. Default is False.

    Returns
    -------
    spinodal_liq : FluidState
        Batched FluidState for the liquid spinodal line.
    spinodal_vap : FluidState
        Batched FluidState for the vapor spinodal line.
    """

    # Temperature array with refinement close to the critical point
    alpha = 0.01
    T_max = fluid.critical_point.T - dT_crit
    T_min = alpha * T_max + (1 - alpha) * fluid.triple_point_liquid.T
    ratio = 1 - T_min / T_max
    t1 = np.logspace(np.log10(1 - 0.9999), np.log10(ratio / 10), int(np.ceil(N / 2)))
    t2 = np.logspace(np.log10(ratio / 10), np.log10(ratio), int(np.floor(N / 2)))
    T_spinodal = (1 - np.concatenate([t1, t2])) * T_max

    # Get limits of entropy to prevent points where EoS breaks down
    s_min, s_max = fluid.triple_point_liquid["s"], fluid.triple_point_vapor["s"]
    delta = (s_max - s_min) / 100
    s_min, s_max = s_min + delta, s_max - delta

    # --- liquid branch
    states_liq = []
    props_liq = compute_spinodal_point(
        T_spinodal[0],
        fluid,
        "liquid",
        method=method,
        supersaturation=supersaturation,
    )
    for T in T_spinodal:
        rho = props_liq["rho"] if use_previous_as_initial_guess else None
        props_liq = compute_spinodal_point(
            T,
            fluid,
            "liquid",
            rho_guess=rho,
            method=method,
            supersaturation=supersaturation,
            tolerance=tolerance,
        )
        if s_min < props_liq["s"] < s_max:
            states_liq.append(props_liq)
        else:
            break

    # --- vapor branch
    states_vap = []
    props_vap = compute_spinodal_point(
        T_spinodal[0], fluid, "vapor", method=method, supersaturation=supersaturation
    )
    for T in T_spinodal:
        rho = props_vap["rho"] if use_previous_as_initial_guess else None
        props_vap = compute_spinodal_point(
            T,
            fluid,
            "vapor",
            rho_guess=rho,
            method=method,
            supersaturation=supersaturation,
        )
        if s_min < props_vap["s"] < s_max:
            states_vap.append(props_vap)
        else:
            break

    # Convert to batched FluidState objects
    spinodal_liq = FluidState.stack(states_liq)
    spinodal_vap = FluidState.stack(states_vap)

    return spinodal_liq, spinodal_vap


# ------------------------------------------------------------------------------------ #
# Spinodal point calculations
# ------------------------------------------------------------------------------------ #

from scipy.optimize import root_scalar


def compute_spinodal_point_general(
    prop_type,
    prop_value,
    fluid,
    branch,
    rho_guess=None,
    N_trial=100,
    method="slsqp",
    tolerance=1e-8,
    print_convergence=False,
    supersaturation=False,
):
    """
    General function to compute the spinodal point for a given property name and value.

    This function uses the underlying `compute_spinodal_point` function and iterates
    on temperature until the specified property at the spinodal point matches the given value.

    Parameters
    ----------
    prop_type : str
        The type of property to match (e.g., 'rho', 'p').
    prop_value : float
        The value of the property to match at the spinodal point.
    fluid : object
        The fluid object containing thermodynamic properties and methods.
    branch : str
        The branch of the spinodal line. Options: 'liquid' or 'vapor'.
    rho_guess : float, optional
        Initial guess for the density. If provided, this value will be used directly.
        If not provided, the density initial guess will be generated based on a number of trial points.
    N_trial : int, optional
        Number of trial points to generate the density initial guess. Default is 100.
    method : str, optional
        The optimization method to solve the problem ('bfgs' or 'slsqp'). Default is 'bfgs'.
    tolerance : float, optional
        Tolerance for the solver termination. Defaults to 1e-6.
    print_convergence : bool, optional
        If True, displays the convergence progress. Defaults to False.

    Returns
    -------
    barotropy.State
        A State object containing the fluid properties

    Raises
    ------
    ValueError
        If the scalar root to determine the spinodal point fails to converge.

    Notes
    -----
    This function uses a root-finding algorithm to iterate on temperature until the
    property specified by `prop_type` at the spinodal point matches `prop_value`.
    The `compute_spinodal_point` function is called iteratively to evaluate the spinodal
    properties at different temperatures.

    Examples
    --------
    >>> state = compute_spinodal_point_general(
    ...     prop_type='density',
    ...     prop_value=500,
    ...     T_guess=300,
    ...     fluid=my_fluid,
    ...     branch='liquid',
    ...     rho_guess=10,
    ...     N_trial=150,
    ...     method='slsqp',
    ...     tolerance=1e-7,
    ...     print_convergence=True,
    ... )
    """

    # Define residual to be driven to zero
    def residual(T):
        props = compute_spinodal_point(
            T,
            fluid,
            branch,
            rho_guess=rho_guess,
            N_trial=N_trial,
            method=method,
            tolerance=tolerance,
            print_convergence=print_convergence,
            supersaturation=supersaturation,
        )
        return props[prop_type] - prop_value

    # Try to compute spinodal point
    try:
        bracket = [fluid.triple_point_vapor.T + 0.1, fluid.critical_point.T - 0.1]
        result = root_scalar(residual, method="toms748", bracket=bracket)
    except ValueError as e:
        raise ValueError(
            f"Failed to find a spinodal point for {prop_type}={prop_value:.2e}.\n"
            f"This might be because there is no spinodal point for the given fluid property value.\n"
            f"Check the value of {prop_type} to ensure there is a matching spinodal point."
        ) from e
    if not result.converged:
        raise ValueError("Scalar root to determine spinodal point failed")

    # Manually check the residual at the solution
    final_residual = residual(result.root)
    if abs(final_residual) > 5 * tolerance:
        raise ValueError(
            f"The solution converged but the residual {final_residual:.2e} is not within the tolerance.\n"
            f"Check the value of {prop_type}={prop_value:.2e} and branch={branch} to ensure there is a matching spinodal point."
        )

    # Compute state for the computed solution
    state = compute_spinodal_point(
        result.root,
        fluid,
        branch,
        rho_guess=rho_guess,
        N_trial=N_trial,
        method=method,
        tolerance=tolerance,
        print_convergence=print_convergence,
        supersaturation=supersaturation,
    )

    return state


def compute_spinodal_point(
    temperature,
    fluid,
    branch,
    rho_guess=None,
    N_trial=100,
    method="slsqp",
    tolerance=1e-8,
    supersaturation=False,
    print_convergence=False,
):
    r"""
    Compute the vapor or liquid spinodal point of a fluid at a given temperature.

    Parameters
    ----------
    temperature : float
        Temperature of the fluid (K).
    fluid : barotropy.Fluid
        The fluid for which the spinodal point is to be calculated.
    branch : str
        Branch of the spinodal line used to determine the density initial guess.
        Options: 'liquid' or 'vapor'.
    rho_guess : float, optional
        Initial guess for the density. If provided, this value will be used directly.
        If not provided, the density initial guess will be generated based on a number of trial points.
    N_trial : int, optional
        Number of trial points to generate the density initial guess. Default is 50.
    method : str, optional
        The optimization method to solve the problem ('bfgs' or 'slsqp').
    tolerance : float, optional
        Tolerance for the solver termination. Defaults to 1e-6.
    print_convergence : bool, optional
        If True, displays the convergence progress. Defaults to False.

    Returns
    -------
    barotropy.State
        A State object containing the fluid properties

    Raises
    ------
    ValueError
        If the input temperature is higher than the critical temperature or lower than
        the triple temperature.

    Notes
    -----
    When a single-phase fluid undergoes a thermodynamic process and enters the two-phase region it
    can exist in a single-phase state that is different from the equilibrium two-phase state.
    Such states are know as metastable states and they are only possible in the thermodynamic
    region between the saturation lines and the spinodal lines. If the thermodynamic process
    continues and crosses the spinodal lines metastable states become unstable and the transition
    to two-distinct phase occurs rapidly. Therefore, the spinodal line represents the locus of points
    that separates the region where a mixture is thermodynamically unstable and prone to phase separation
    from the region where metastable states are physically possible.

    In mathematical terms, the spinodal line is defined as the loci of thermodynamic states in which the isothermal bulk modulus of the fluid is zero:

    .. math::

        K_T = \rho \left( \frac{\partial p}{\partial \rho} \right)_T = 0

    More precisely, a vapor spinodal point is the first local maximum of a isotherm line in a pressure-density diagram as the density increases.
    Conversely, a liquid spinodal point is the first local minimum of a isotherm line in a pressure-density diagram as the density decreases.
    The spinodal lines and phase envelope of carbon dioxide according to the HEOS developed by :cite:`span_new_1996` are illustrated in the figure below

    .. image:: /_static/spinodal_points_CO2.svg
        :alt: Pressure-density diagram and spinodal points for carbon dioxide.


    Some equations of state are not well-posed and do not satisfy the condition :math:`K_T=0` within the two-phase region.
    This is exemplified by the nitrogen HEOS developed by :cite:`span_reference_2000`.

    .. image:: /_static/spinodal_points_nitrogen.svg
        :alt: Pressure-density diagram and "pseudo" spinodal points for carbon dioxide.

    As seen in the figure, this HEOS is not well-posed because there are isotherms that do not have a local minimum/maximum corresponding to a state with zero isothermal bulk modulus.
    In such cases, this function returns the inflection point of the isotherms (corresponding to the point closest to zero bulk modulus) as the spinodal point.

    """

    # Instantiate new abstract state to compute saturation properties without changing the state of the class

    # Check that the inlet temperature is lower than the critical temperature
    T_critical = fluid.critical_point.T
    if temperature >= T_critical:
        msg = f"T={temperature:.3f}K must be less than T_critical={T_critical:.3f}K"
        raise ValueError(msg)

    # Check that the inlet temperature is greater than the triple temperature
    T_triple = fluid.triple_point_vapor.T
    if temperature < T_triple:
        msg = f"T={temperature:.3f}K must be greater than T_triple={T_triple:.3f}K"
        raise ValueError(msg)

    # Create spinodal point optimization problem
    problem = _SpinodalPointProblem(temperature, fluid, branch, supersaturation)

    # Check solver method (deprecated)
    if method == "bfgs":
        # problem.get_bounds = lambda: ([-np.inf], [np.inf])
        pass
    elif method == "slsqp":
        pass
    else:
        raise ValueError(
            f"Solver method {method} is not valid. Valid options are 'slsqp' and 'bfgs'"
        )

    # Create solver object
    solver = psv.OptimizationSolver(
        problem=problem,
        library="scipy",
        method=method,  # "l-bfgs-b", "slsqp" "slsqp"
        tolerance=tolerance,
        print_convergence=print_convergence,
        problem_scale=50,
    )

    # Generate initial guess if not provided
    if rho_guess is None:
        rho_guess = problem.generate_density_guess(N_trial)

    # Solve the problem using the provided or generated initial guess
    rho_opt = solver.solve(rho_guess).item()
    state = fluid.get_state_metastable(
        "rho",
        rho_opt,
        "T",
        temperature,
        supersaturation=supersaturation,
    )

    # I tried different combinations of solvers and initial guesses
    # SLSQP is very fast, but also very aggresive and can lead to unpredictable results
    # BFGS is reliable when the initial guess is good
    # Achieving a good initial guess is possible by:
    #  1. Using previous point in the spinodal line and having high resolution on the spinodal line
    #  2. Generating an initial guess for each point with the initial guess strategy
    # Option 2. seems the most reliable, and even if the computational cost can be a bit hight
    # it seems to be the most effective way to get accurate spinodal lines.
    #
    # 11.07.2025
    # I improved the code with 2 modifications:
    #
    #  1. Use logspace instead of linspace to generate initial guess
    #  2. Apply problem scaling to SLSQP computation
    #
    # With this enhancements, SLSQP produces good spinodal line results
    #

    return state


class _SpinodalPointProblem(psv.OptimizationProblem):
    """Auxiliary class for the determination of the liquid and vapor spinodal points"""

    def __init__(self, temperature, fluid, branch, supersaturation):

        # Declare class attributes
        self.T = temperature
        self.branch = branch
        self.fluid = fluid
        self.supersaturation = supersaturation

        # Compute saturation liquid density (used to determine initial guess)
        state_vap = self.fluid.get_state(CP.QT_INPUTS, 0.00, self.T)
        self.rho_liq = state_vap.rho

        # Calculate saturation vapor density (used to determine initial guess)
        state_liq = self.fluid.get_state(CP.QT_INPUTS, 1.00, self.T)
        self.rho_vap = state_liq.rho

    def generate_density_guess(self, N):
        """Generate a density initial guess that is close to the first local minima of the absolute value of the bulk modulus"""

        # Generate candidate densities between saturation and the critical value
        if self.branch == "liquid":
            # rho_array = np.linspace(self.rho_liq, self.rho_vap, N)
            rho_array = np.logspace(np.log10(self.rho_liq), np.log10(self.rho_vap), N)
        elif self.branch == "vapor":
            # rho_array = np.linspace(self.rho_vap, self.rho_liq, N)
            rho_array = np.logspace(np.log10(self.rho_vap), np.log10(self.rho_liq), N)
        else:
            msg = f"Invalid value for parameter branch={self.branch}. Options: 'liquid' or 'vapor'"
            raise ValueError(msg)

        # Evaluate residual vector at trial densities
        residual = np.abs([self.fitness(rho) for rho in rho_array])

        # Return the first local minima in the residual vector
        for i in range(1, N - 1):
            if residual[i - 1] > residual[i] < residual[i + 1]:
                self.rho_guess = rho_array[i + 1]
                return self.rho_guess

    def fitness(self, rho):
        """
        Compute the objective function of the optimization problem: the absolute value of the isothermal bulk modulus.

        This function uses the absolute value of residual to solve an optimization problem
        rather than a non-linear equation having the bulk modulus as residual. This approach
        is adopted because some fluids (e.g., nitrogen) have ill-posed EoS that not have a well-defined
        spinodal points where the isothermal bulk modulus is zero.

        Not a good idea to scale the bulk modulus by pressure because it can take negative values or zero when evaluated with the HEOS.
        """
        state = self.fluid.get_state_metastable(
            "rho", rho, "T", self.T, supersaturation=self.supersaturation
        )
        return np.atleast_1d(np.abs(state["isothermal_bulk_modulus"])) / 1e6

    def get_bounds(self):
        """Compute the bounds of the optimization problem."""
        return [(self.rho_vap,), (self.rho_liq,)]
        # rho_limit = self.rho_guess
        # # rho_limit = self.rho_crit
        # if self.branch == "liquid":
        #     return [(rho_limit,), (self.rho_liq,)]
        # elif self.branch == "vapor":
        #     return [(self.rho_vap,), (rho_limit,)]
        # else:
        #     raise ValueError(f"Invalid value for branch={self.branch}")

    def get_nec(self):
        return 0

    def get_nic(self):
        return 0


# ------------------------------------------------------------------------------------ #
# Other calculations
# ------------------------------------------------------------------------------------ #


def states_to_dict(states):
    """
    Recursively convert a list of FluidState objects into a nested dictionary
    of NumPy arrays. Handles missing keys by inserting NaNs or None.

    Parameters
    ----------
    states : list of FluidState
        A list of FluidState objects. Keys may differ between objects.

    Returns
    -------
    dict
        Nested dictionary where leaf nodes are 1D numpy arrays of field values.
        Missing keys are filled with NaN or None.
    """
    # from .jax_import import jax, jnp, JAX_AVAILABLE
    # np = jnp

    from collections.abc import Mapping

    def extract_value(state, key):
        if hasattr(state, "get"):
            return state.get(key, None)
        return getattr(state, key, None)

    def is_nested(value):
        return isinstance(value, Mapping) or hasattr(value, "keys")

    # Collect the union of all keys across states
    all_keys = set()
    for state in states:
        all_keys.update(state.keys())

    result = {}

    for key in sorted(all_keys):
        values = [extract_value(s, key) for s in states]

        # Check if this field is nested (but skip None)
        nested_values = [v for v in values if v is not None]
        if nested_values and is_nested(nested_values[0]):
            result[key] = states_to_dict(nested_values)
        else:
            # If values are mixed (e.g., some None), fill with None or np.nan
            if all(isinstance(v, (int, float, np.number, type(None))) for v in values):
                result[key] = np.array([np.nan if v is None else v for v in values])
            else:
                result[key] = np.array(values, dtype=object)

    return result


def states_to_dict_2d(states):
    """
    Convert a 2D list (grid) of state objects into a dictionary.

    Each key is a field name of the state objects, and each value is a 2D Numpy array of all the values for that field.

    Parameters
    ----------
    states_grid : list of lists of FluidState
        A 2D grid where each element is a state object with the same keys.

    Returns
    -------
    dict
        A dictionary where keys are field names and values are 2D arrays of field values.
    """
    state_dict_2d = {}
    m, n = len(states), len(states[0])
    for i, row in enumerate(states):
        for j, state in enumerate(row):
            for field, value in state.items():
                if field not in state_dict_2d:
                    dtype = type(value)  # Determine dtype from the first occurrence
                    state_dict_2d[field] = np.empty((m, n), dtype=dtype)
                state_dict_2d[field][i, j] = value
    return state_dict_2d


def compute_quality_grid(fluid, num_points, quality_levels, dT_crit=1.0):
    # Define temperature levels
    R = 1 - fluid.triple_point_liquid.T / fluid.critical_point.T
    R_crit = dT_crit / fluid.critical_point.T
    t1 = np.logspace(np.log10(R_crit), np.log10(R / 2), int(np.ceil(num_points / 2)))
    t2 = np.linspace(R / 2, R, int(np.floor(num_points / 2)))
    temperature_levels = (1 - np.concatenate([t1, t2])) * fluid.critical_point.T

    # Calculate property grid
    quality_grid = []
    for q in quality_levels:
        row = []
        row.append(fluid.critical_point)
        for T in temperature_levels:
            row.append(fluid.get_state(CP.QT_INPUTS, q, T))

        quality_grid.append(row)

    return states_to_dict_2d(quality_grid)


def compute_property_grid(
    fluid,
    input_pair,
    range_1,
    range_2,
    generalize_quality=False,
    supersaturation=False,
):
    """
    Compute fluid properties over a specified range and store them in a dictionary.

    This function creates a meshgrid of property values based on the specified ranges and input pair,
    computes the properties of the fluid at each point on the grid, and stores the results in a
    dictionary where each key corresponds to a fluid property.

    Parameters
    ----------
    fluid : Fluid object
        An instance of the Fluid class.
    input_pair : tuple
        The input pair specifying the property type (e.g., PT_INPUTS for pressure-temperature).
    range1 : tuple
        The range linspace(min, max, n) for the first property of the input pair.
    range2 : tuple
        The range linspace(min, max, n) for the second property of the input pair.

    Returns
    -------
    properties_dict : dict
        A dictionary where keys are property names and values are 2D numpy arrays of computed properties.
    grid1, grid2 : numpy.ndarray
        The meshgrid arrays for the first and second properties.
    """

    # Create the meshgrid
    grid1, grid2 = np.meshgrid(range_1, range_2)

    # Initialize dictionary to store properties and pre-allocate storage
    properties_dict = {}
    n, m = len(range_1), len(range_2)

    # Compute properties at each point
    for i in range(m):
        for j in range(n):
            # Set state of the fluid
            state = fluid.get_state(
                input_pair,
                grid1[i, j],
                grid2[i, j],
                generalize_quality=generalize_quality,
                supersaturation=supersaturation,
            )

            # Store the properties (initialize as empty array if new key)
            for key, value in state.items():
                if key not in properties_dict.keys():
                    dtype = type(value)  # Determine dtype from the first occurrence
                    properties_dict[key] = np.empty((m, n), dtype=dtype)
                    # properties_dict[key] = np.zeros_like(grid1)
                properties_dict[key][i, j] = state[key]

    return properties_dict


def compute_property_grid_rhoT(
    fluid,
    rho_array,
    T_array,
):

    # Calculate property grid
    states_meta = []
    for T in T_array:
        row = []
        for rho in rho_array:
            row.append(core.compute_properties_metastable_rhoT(fluid._AS, rho, T))
        states_meta.append(row)

    # Convert nested list of dictionaries into dictionary of 2D arrays
    states_meta = states_to_dict_2d(states_meta)

    return states_meta



# def get_state_Qs(fluid, Q, s):
#     # Define the residual equation
#     def get_residual(p):
#         state = fluid.get_state(PSmass_INPUTS, p, s, generalize_quality=True)
#         residual = Q - state.Q
#         return residual

#     # Solve the scalar equation
#     p_triple = 1.0 * fluid.triple_point_liquid.p
#     p_critical = 1.25 * fluid.critical_point.p
#     bounds = [p_triple, p_critical]
#     sol = scipy.optimize.root_scalar(get_residual, bracket=bounds, method="brentq")

#     # Check if the solver has converged
#     if not sol.converged:
#         raise ValueError("The root-finding algorithm did not converge!")

#     # Compute the outlet state
#     state = fluid.get_state(PSmass_INPUTS, sol.root, s, generalize_quality=True)

#     return state


# def get_isentropic_saturation_state(fluid, s_in):
#     # Calculate saturation sate
#     if s_in < fluid.critical_point.s:
#         state_sat = get_state_Qs(fluid, Q=0.00, s=s_in)
#     else:
#         state_sat = get_state_Qs(fluid, Q=1.00, s=s_in)

#     return state_sat


# # ------------------------------------------------------------------------------------ #
# # Sonic point calculations
# # ------------------------------------------------------------------------------------ #


# class _SonicStateProblem(psv.NonlinearSystemProblem):
#     """ """

#     def __init__(self, fluid, property_pair, prop_1, prop_2):
#         # Calculate the thermodynamic state
#         self.fluid = fluid
#         self.state = fluid.get_state(property_pair, prop_1, prop_2)

#         # Initial guess based in input sstate
#         self.initial_guess = [self.state.d, self.state.T]

#         # # Initial guess based on perfect gass relations
#         # gamma = self.state.gamma
#         # d_star = (2/(gamma + 1)) ** (1/(gamma-1)) * self.state.rho
#         # T_star =  (2/(gamma + 1)) * self.state.T
#         # self.initial_guess = [d_star, T_star]

#     def get_values(self, x):
#         # Ensure x can be indexed and contains exactly two elements
#         if not hasattr(x, "__getitem__") or len(x) != 2:
#             raise ValueError(
#                 "Input x must be a list, tuple or numpy array containing exactly two elements: density and temperature."
#             )

#         # Calculate state for the current density-temperature pair
#         crit_state = self.fluid.get_state(DmassT_INPUTS, x[0], x[1])

#         # Calculate the sonic state residual
#         residual = np.asarray(
#             [
#                 1.0 - (crit_state.h + 0.5 * crit_state.a**2) / self.state.h,
#                 1.0 - crit_state.s / self.state.s,
#             ]
#         )

#         return residual


# class _SonicStateProblem2(psv.OptimizationProblem):
#     """ """

#     def __init__(self, fluid, property_pair, prop_1, prop_2):
#         # Calculate the thermodynamic state
#         self.fluid = fluid
#         self.state = fluid.get_state(property_pair, prop_1, prop_2)

#         # Initial guess based in input sstate
#         self.initial_guess = [self.state.d, self.state.T * 0.9]

#         # # Initial guess based on perfect gass relations
#         # gamma = self.state.gamma
#         # d_star = (2/(gamma + 1)) ** (1/(gamma-1)) * self.state.rho
#         # T_star =  (2/(gamma + 1)) * self.state.T
#         # self.initial_guess = [d_star, T_star]

#     def get_values(self, x):
#         """
#         Compute the residuals for the given density and temperature.

#         Parameters
#         ----------
#         x : list
#             List containing the values for density and temperature.

#         Returns
#         -------
#         np.ndarray
#             Array containing residuals (difference) for the two properties.
#         """

#         # Ensure x can be indexed and contains exactly two elements
#         if not hasattr(x, "__getitem__") or len(x) != 2:
#             raise ValueError(
#                 "Input x must be a list, tuple or numpy array containing exactly two elements: density and temperature."
#             )

#         # Calculate state for the current density-temperature pair
#         crit_state = self.fluid.get_state(DmassT_INPUTS, x[0], x[1])

#         # Calculate the sonic state residual
#         residual = [
#             1.0 - crit_state.s / self.state.s,
#         ]

#         # Objective function
#         self.f = crit_state.d**2 * (self.state.h - crit_state.h)
#         self.f = -self.f / (self.state.d * self.state.a) ** 2

#         # Equality constraints
#         self.c_eq = residual

#         # No inequality constraints given for this problem
#         self.c_ineq = []

#         # Combine objective function and constraints
#         objective_and_constraints = self.merge_objective_and_constraints(
#             self.f, self.c_eq, self.c_ineq
#         )

#         return objective_and_constraints

#     def get_bounds(self):
#         bound_density = (
#             self.fluid.triple_point_vapor.d * 1.5,
#             self.fluid.critical_point.d * 3,
#         )
#         bound_temperature = (
#             self.fluid.triple_point_vapor.T * 1,
#             self.fluid.critical_point.T * 3,
#         )
#         # return [bound_density, bound_temperature]
#         return None

#     def get_n_eq(self):
#         return self.get_number_of_constraints(self.c_eq)

#     def get_n_ineq(self):
#         return self.get_number_of_constraints(self.c_ineq)


# ------------------------------------------------------------------------------------ #
# Utility functions
# ------------------------------------------------------------------------------------ #


def safe_alias_split(raw_string):
    """
    Split a CoolProp alias string safely, preserving multi-part names
    that include commas (e.g., chemical names like '1,2-dichloroethane').

    Parameters
    ----------
    raw_string : str
        Comma-separated alias string from CoolProp.

    Returns
    -------
    list of str
        Cleaned list of alias names, with spurious splits recombined.
    """
    tokens = raw_string.split(",")
    merged = []
    buffer = ""

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            if buffer:
                buffer += "," + token
            else:
                buffer = token
        else:
            if buffer:
                buffer += "," + token
                merged.append(buffer)
                buffer = ""
            else:
                merged.append(token)
    if buffer:
        merged.append(buffer)
    return merged


def print_fluid_names():
    """
    Print all canonical fluids and their aliases in a structured format.
    """
    canonical_fluids = CP.get_global_param_string("FluidsList").split(",")

    for fluid in canonical_fluids:
        raw_aliases = CP.get_fluid_param_string(fluid, "aliases")
        aliases = safe_alias_split(raw_aliases)
        print(f"{fluid}:")
        for alias in aliases:
            print(f"  - {alias}")
        print()  # empty line between fluids


def build_alias_map():
    """
    Build a mapping from alias names to canonical CoolProp fluid names.

    Returns
    -------
    dict
        Dictionary where keys are aliases and values are canonical fluid names.
    """
    alias_map = {}
    canonical_fluids = CP.get_global_param_string("FluidsList").split(",")

    for fluid in canonical_fluids:
        raw_aliases = CP.get_fluid_param_string(fluid, "aliases")
        aliases = safe_alias_split(raw_aliases)
        if aliases:
            for alias in aliases:
                alias_map[alias] = fluid
        # Always include canonical name as valid alias
        alias_map[fluid] = fluid

    return alias_map


def try_initialize_fluid(name: str, backend: str = "HEOS"):
    """
    Attempt to initialize a CoolProp AbstractState for the given fluid name.
    If initialization fails, raise a ValueError with close name suggestions.

    Parameters
    ----------
    name : str
        Fluid name or alias to initialize.
    backend : str, optional
        CoolProp backend to use (default is 'HEOS').

    Returns
    -------
    CP.AbstractState
        Initialized CoolProp AbstractState object.

    Raises
    ------
    ValueError
        If the fluid name is not recognized by CoolProp, with close suggestions.
    """
    try:
        return CP.AbstractState(backend, name)
    except ValueError as e:
        alias_map = build_alias_map()
        all_aliases = list(alias_map.keys())
        suggestions = difflib.get_close_matches(name, all_aliases, n=3)

        suggestion_msg = (
            f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        )

        raise ValueError(
            f"CoolProp could not initialize fluid '{name}'.{suggestion_msg}"
        ) from e


def is_pure_substance(AS) -> bool:
    backend = AS.backend_name()

    if backend == "HelmholtzEOSBackend":  # Alias HEOS
        return AS.fluid_param_string("pure").lower() == "true"

    elif backend in ["REFPROPBackend", "BicubicBackend"]:
        return len(AS.fluid_names()) == 1 and AS.name().lower() not in [
            "air",
            "air (dry)",
            "r404a",
            "r407c",
            "r410a",
            "r507a",
        ]

    else:
        raise NotImplementedError(
            f"Purity check not implemented for backend '{backend}'"
        )
