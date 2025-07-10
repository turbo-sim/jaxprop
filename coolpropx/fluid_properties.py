import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import pysolver_view as psv

from functools import wraps

from . import core_calculations as props


MEANLINE_PROPERTIES = [
    "p",
    "T",
    "h",
    "s",
    "d",
    "Z",
    "a",
    "mu",
    "k",
    "cp",
    "cv",
    "gamma",
]

LABEL_MAPPING = {
    "density": "Density (kg/m$^3$)",
    "viscosity": "Viscosity (Pa·s)",
    "speed_sound": "Speed of sound (m/s)",
    "void_fraction": "Void fraction",
    "vapor_quality": "Vapor quality",
    "p": "Pressure (Pa)",
    "s": "Entropy (J/kg/K)",
    "T": "Temperature (K)",
    "h": "Enthalpy (J/kg)",
    "rho": r"Density (kg/m$^3$)",
}

# Dynamically add INPUTS fields to the module
# for attr in dir(CP):
#     if attr.endswith('_INPUTS'):
#         globals()[attr] = getattr(CP, attr)

# Statically add phase indices to the module (IDE autocomplete)
iphase_critical_point = CP.iphase_critical_point
iphase_gas = CP.iphase_gas
iphase_liquid = CP.iphase_liquid
iphase_not_imposed = CP.iphase_not_imposed
iphase_supercritical = CP.iphase_supercritical
iphase_supercritical_gas = CP.iphase_supercritical_gas
iphase_supercritical_liquid = CP.iphase_supercritical_liquid
iphase_twophase = CP.iphase_twophase
iphase_unknown = CP.iphase_unknown

# Statically add INPUT fields to the module (IDE autocomplete)
QT_INPUTS = CP.QT_INPUTS
PQ_INPUTS = CP.PQ_INPUTS
QSmolar_INPUTS = CP.QSmolar_INPUTS
QSmass_INPUTS = CP.QSmass_INPUTS
HmolarQ_INPUTS = CP.HmolarQ_INPUTS
HmassQ_INPUTS = CP.HmassQ_INPUTS
DmolarQ_INPUTS = CP.DmolarQ_INPUTS
DmassQ_INPUTS = CP.DmassQ_INPUTS
PT_INPUTS = CP.PT_INPUTS
DmassT_INPUTS = CP.DmassT_INPUTS
DmolarT_INPUTS = CP.DmolarT_INPUTS
HmolarT_INPUTS = CP.HmolarT_INPUTS
HmassT_INPUTS = CP.HmassT_INPUTS
SmolarT_INPUTS = CP.SmolarT_INPUTS
SmassT_INPUTS = CP.SmassT_INPUTS
TUmolar_INPUTS = CP.TUmolar_INPUTS
TUmass_INPUTS = CP.TUmass_INPUTS
DmassP_INPUTS = CP.DmassP_INPUTS
DmolarP_INPUTS = CP.DmolarP_INPUTS
HmassP_INPUTS = CP.HmassP_INPUTS
HmolarP_INPUTS = CP.HmolarP_INPUTS
PSmass_INPUTS = CP.PSmass_INPUTS
PSmolar_INPUTS = CP.PSmolar_INPUTS
PUmass_INPUTS = CP.PUmass_INPUTS
PUmolar_INPUTS = CP.PUmolar_INPUTS
HmassSmass_INPUTS = CP.HmassSmass_INPUTS
HmolarSmolar_INPUTS = CP.HmolarSmolar_INPUTS
SmassUmass_INPUTS = CP.SmassUmass_INPUTS
SmolarUmolar_INPUTS = CP.SmolarUmolar_INPUTS
DmassHmass_INPUTS = CP.DmassHmass_INPUTS
DmolarHmolar_INPUTS = CP.DmolarHmolar_INPUTS
DmassSmass_INPUTS = CP.DmassSmass_INPUTS
DmolarSmolar_INPUTS = CP.DmolarSmolar_INPUTS
DmassUmass_INPUTS = CP.DmassUmass_INPUTS
DmolarUmolar_INPUTS = CP.DmolarUmolar_INPUTS

# Define dictionary with dynamically generated fields
PHASE_INDEX = {attr: getattr(CP, attr) for attr in dir(CP) if attr.startswith("iphase")}
INPUT_PAIRS = {attr: getattr(CP, attr) for attr in dir(CP) if attr.endswith("_INPUTS")}
INPUT_PAIRS = sorted(INPUT_PAIRS.items(), key=lambda x: x[1])
INPUT_TYPE_MAP = {v: k for k, v in INPUT_PAIRS}


# Convert each input key to a tuple of FluidState variable names
# Capitalized names that should not be lowercased
preserve_case = {'T', 'Q'}

def extract_vars(name):
    base = name.replace("_INPUTS", "")
    parts = []
    current = base[0]
    for c in base[1:]:
        if c.isupper():
            parts.append(current)
            current = c
        else:
            current += c
    parts.append(current)
    return tuple(p if p in preserve_case else p.lower() for p in parts)

INPUT_PAIR_MAP = {k: extract_vars(v) for k, v in INPUT_TYPE_MAP.items()}



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
                label = INPUT_TYPE_MAP.get(input_type, f"Unknown input type ({input_type})")

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

    
def _generate_coolprop_input_table():
    """Create table of input pairs as string to be copy-pasted in Sphinx documentation"""
    inputs_table = ".. list-table:: CoolProp input mappings\n"
    inputs_table += "   :widths: 50 30\n"
    inputs_table += "   :header-rows: 1\n\n"
    inputs_table += "   * - Input pair name\n"
    inputs_table += "     - Input pair mapping\n"
    for name, value in INPUT_PAIRS:
        inputs_table += f"   * - {name}\n"
        inputs_table += f"     - {value}\n"

    return inputs_table

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
        self._AS = CP.AbstractState(backend, name)
        self.exceptions = exceptions
        self.converged_flag = False
        self._properties = {}
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
        if self._AS.fluid_param_string("pure") == "true":
            self.critical_point = self._compute_critical_point()
            self.triple_point_liquid = self._compute_triple_point_liquid()
            self.triple_point_vapor = self._compute_triple_point_vapor()

        # # Assign critical point properties
        # if initialize_critical:
        #     self.critical_point = self._compute_critical_point()

        # # Assign triple point properties
        # if initialize_triple:
        #     self.triple_point_liquid = self._compute_triple_point_liquid()
        #     self.triple_point_vapor = self._compute_triple_point_vapor()

        # Pressure and temperature limits
        self.p_min = 1
        self.p_max = self._AS.pmax()
        self.T_min = self._AS.Tmin()
        self.T_max = self._AS.Tmax()

    def _compute_critical_point(self):
        """Calculate the properties at the critical point"""
        rho_crit, T_crit = self._AS.rhomass_critical(), self._AS.T_critical()
        return self.get_state(DmassT_INPUTS, rho_crit, T_crit)

    def _compute_triple_point_liquid(self):
        """Calculate the properties at the triple point (liquid state)"""
        return self.get_state(QT_INPUTS, 0.00, self._AS.Ttriple())

    def _compute_triple_point_vapor(self):
        """Calculate the properties at the triple point (vapor state)"""
        return self.get_state(QT_INPUTS, 1.00, self._AS.Ttriple())

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
        self._properties = props.compute_properties_coolprop(
            self._AS,
            input_type,
            prop_1,
            prop_2,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
        )
        self._properties["identifier"] = self.identifier
        return FluidState(self._properties, self.name)

    def get_states(
        self,
        input_type,
        prop_1,
        prop_2,
        generalize_quality=False,
        supersaturation=False,
    ):
        """
        Vectorized version of get_state:
        - if both prop_1 and prop_2 are scalars, returns a single FluidState
        - if one is 1D array and the other scalar, returns ({prop: 1D array}, grid)
        - if both are 1D arrays, returns ({prop: 2D array}, grid1, grid2)
        """
        # Force into 1D arrays
        p1 = np.atleast_1d(prop_1)
        p2 = np.atleast_1d(prop_2)

        # Both properties are scalars
        if p1.size == 1 and p2.size == 1:
            return self.get_state(input_type, float(p1), float(p2),
                                  generalize_quality, supersaturation)

        # 1D array for property 1
        if p1.size > 1 and p2.size == 1:
            states = [self.get_state(input_type, float(p1i), float(p2),
                                     generalize_quality, supersaturation)
                      for p1i in p1]
            return states_to_dict(states), p1

        # 1D array for property 2
        if p1.size == 1 and p2.size > 1:
            states = [self.get_state(input_type, float(p1), float(p2j),
                                     generalize_quality, supersaturation)
                      for p2j in p2]
            return states_to_dict(states), p2

        # Both properties are 1D arrays
        # meshgrid with indexing='xy' yields shape (len(p2),len(p1))
        grid_1, grid_2 = np.meshgrid(p1, p2, indexing='ij')
        states_2d = []
        for i in range(grid_1.shape[0]):
            row = []
            for j in range(grid_1.shape[1]):
                row.append(
                    self.get_state(
                        input_type,
                        float(grid_1[i, j]),
                        float(grid_2[i, j]),
                        generalize_quality,
                        supersaturation,
                    )
                )
            states_2d.append(row)

        return states_to_dict_2d(states_2d)
    

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
        self._properties = props.compute_properties(
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
        return FluidState(self._properties, self.name)

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
            self._properties = props.compute_properties_metastable_rhoT(
                abstract_state=self._AS,
                rho=prop_1_value,
                T=prop_2_value,
                generalize_quality=generalize_quality,
                supersaturation=supersaturation,
            )
        elif prop_1 == "T" and prop_2 == "rho":
            self._properties = props.compute_properties_metastable_rhoT(
                abstract_state=self._AS,
                rho=prop_2_value,
                T=prop_1_value,
                generalize_quality=generalize_quality,
                supersaturation=supersaturation,
            )
        else:
            self._properties = props.compute_properties(
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
        return FluidState(self._properties, self.name)

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
        blended, equilibrium, metastable = props.compute_properties(
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
            FluidState(blended, self.name),
            FluidState(equilibrium, self.name),
            FluidState(metastable, self.name),
        )

    def get_property(self, propname):
        """Get the value of a single property"""
        if propname in self._properties:
            return self._properties[propname]
        else:
            valid_options = "\n\t".join(self._properties.keys())
            raise ValueError(
                f"The requested property '{propname}' is not available. The valid options are:\n\t{valid_options}"
            )

    def compute_properties_meanline(self, input_type, prop_1, prop_2):
        """Extract fluid properties for meanline model"""

        # Compute properties in the normal way
        self.get_state(input_type, prop_1, prop_2)

        # Store a subset of the properties in a dictionary
        fluid_properties = {}
        for item in MEANLINE_PROPERTIES:
            fluid_properties[item] = self._properties[item]

        return fluid_properties

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
        spinodal_line_method="bfgs",  # Alternative is slsqp
        spinodal_line_use_previous=False,  # True is not as robust
        plot_quality_isolines=False,
        plot_two_phase_patch=False,
        plot_pseudocritical_line=False,
        quality_levels=np.linspace(0.1, 1.0, 10),
        quality_labels=False,
        show_in_legend=False,
        x_scale="linear",
        y_scale="linear",
        eps_sat=1e-4
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
                self.sat_liq, self.sat_vap = compute_saturation_line(self, N, eps=eps_sat)
            x = list(reversed(self.sat_liq[x_prop])) + self.sat_vap[x_prop]
            y = list(reversed(self.sat_liq[y_prop])) + self.sat_vap[y_prop]
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
                )
            x = list(reversed(self.spdl_liq[x_prop])) + self.spdl_vap[x_prop]
            y = list(reversed(self.spdl_liq[y_prop])) + self.spdl_vap[y_prop]
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
                self.q_mesh = compute_quality_grid(self, N, quality_levels)
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


class FluidState:
    """
    An immutable class representing the thermodynamic state of a fluid.

    This class is designed to provide a read-only representation of a fluid's state,
    with properties accessible through both attribute-style and dictionary-style access.
    The state of a fluid is defined at initialization and cannot be altered thereafter,
    ensuring the integrity of the data.

    Instances of this class store fluid properties and the fluid's name, providing
    methods to access these properties but not to modify them.

    Parameters
    ----------
    properties : dict
        A dictionary where keys are property names (as strings) and values are the
        corresponding properties of the fluid. This dictionary is converted to an
        immutable internal representation.
    fluid_name : str
        The name of the fluid.

    Attributes
    ----------
    properties : dict
        An internal dictionary storing the properties of the fluid state. This attribute
        is immutable and cannot be modified after the object's initialization.
    fluid_name : str
        The name of the fluid. Immutable after initialization.

    Methods
    -------
    to_dict():
        Returns a copy of the fluid properties as a dictionary.
    keys():
        Returns the keys of the fluid properties.
    items():
        Returns the items (key-value pairs) of the fluid properties.
    """

    __slots__ = ("_properties", "fluid_name")  # Define allowed attributes


    def __init__(self, properties, fluid_name):
        # Convert keys to strings and store properties in an internal attribute
        # Ensure the properties dictionary is immutable (e.g., by using a frozendict if modifications are a concern)
        object.__setattr__(
            self, "_properties", {str(k): v for k, v in properties.items()}
        )
        object.__setattr__(self, "fluid_name", fluid_name)

    def __getattr__(self, name):
        # Allows attribute-style access to the fluid properties. If the property does not exist, raises AttributeError.
        try:
            props = object.__getattribute__(self, "_properties")
            return props[name]
        except KeyError:
            raise AttributeError(f"'{name}' not found in FluidState properties")

    def __getitem__(self, key):
        # Allows dictionary-style access to the fluid properties. If the key does not exist, raises KeyError.
        try:
            props = object.__getattribute__(self, "_properties")
            return props[str(key)]
        except KeyError:
            raise KeyError(f"'{key}' not found in FluidState properties")

    def __setattr__(self, key, value):
        # Prevents modifications to the instance attributes, ensuring immutability.
        raise AttributeError(
            "Cannot modify properties of an immutable FluidState class"
        )

    def __setitem__(self, key, value):
        # Prevents modifications to the fluid properties through dictionary-style access, ensuring immutability.
        raise AttributeError(
            "Cannot modify properties of an immutable FluidState class"
        )

    def __repr__(self):
        # Returns a string representation of the FluidState instance, including its class name, properties, and fluid name.
        return f"{self.__class__.__name__}({self._properties}, '{self.fluid_name}')"

    def __str__(self):
        # Make object print()-able
        prop_str = "\n   ".join(
            f"{k}: {v if isinstance(v, str) else f'{v: .6e}'}"
            for k, v in self._properties.items()
        )
        return f"FluidState:\n   {prop_str}"

    def __iter__(self):
        # Iterate over the object like a dictionary
        return iter(self._properties)

    def to_dict(self):
        # Return a copy of the properties to ensure immutability
        return dict(self._properties)

    def keys(self):
        # Return the keys of the properties
        return self._properties.keys()

    def values(self):
        # Return the values of the properties
        return self._properties.values()

    def items(self):
        # Return the items of the properties
        return self._properties.items()

    def get(self, key, default=None):
        return self._properties.get(key, default)

    def __getstate__(self):
        """Enable pickling by returning internal state without violating immutability."""
        return {
            "_properties": self._properties,
            "fluid_name": self.fluid_name,
        }

    def __setstate__(self, state):
        """Restore internal state during unpickling while bypassing immutability guard."""
        object.__setattr__(self, "_properties", state["_properties"])
        object.__setattr__(self, "fluid_name", state["fluid_name"])


def states_to_dict(states):
    """
    Convert a list of state objects into a dictionary.

    Each key is a field name of the state objects, and each value is a Numpy array of all the values for that field.
    
    Parameters
    ----------
    states_grid : list of FluidState
        A 1D grid where each element is a state object with the same keys.

    Returns
    -------
    dict
        A dictionary where keys are field names and values are 1D arrays of field values.
    """
    state_dict = {}
    for field in states[0].keys():
        state_dict[field] = np.array([getattr(state, field) for state in states])
    return state_dict


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
    


# ------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------ #


def compute_saturation_line(fluid, N=100, eps=1e-4):
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
    saturation_liq : dict of lists
        Dictionary containing the liquid saturation properties.
    saturation_vap : dict of lists
        Dictionary containing the vapor saturation properties.
    """
    # Define temperature array with refinement close to the critical point
    R = 1 - fluid.triple_point_liquid.T / fluid.critical_point.T
    t1 = np.logspace(np.log10(eps), np.log10(R / 10), int(np.ceil(N / 2)))
    t2 = np.linspace(R / 10, R, int(np.floor(N / 2)))
    T_sat = (1 - np.concatenate([t1, t2])) * fluid.critical_point.T

    # Initialize dictionaries for storing properties
    saturation_liq = {}
    saturation_vap = {}

    # Loop over temperatures for liquid and vapor states
    for T in T_sat:
        state_liquid = fluid.get_state(CP.QT_INPUTS, 0.00, T)
        state_vapor = fluid.get_state(CP.QT_INPUTS, 1.00, T)
        for name in state_liquid.keys():
            if name not in saturation_liq:
                saturation_liq[name] = [fluid.critical_point[name]]
            saturation_liq[name].append(state_liquid[name])

        for name in state_vapor.keys():
            if name not in saturation_vap:
                saturation_vap[name] = [fluid.critical_point[name]]
            saturation_vap[name].append(state_vapor[name])

    # # Reverse the liquid properties for easy concatenation
    # for name in saturation_liq.keys():
    #     saturation_liq[name] = list(reversed(saturation_liq[name]))

    return saturation_liq, saturation_vap


def compute_spinodal_line(
    fluid,
    N=50,
    method="bfgs",
    use_previous_as_initial_guess=False,
    supersaturation=False,
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
    spinodal_liq : dict of lists
        Dictionary containing the liquid spinodal properties.
    spinodal_vap : dict of lists
        Dictionary containing the vapor spinodal properties.
    """

    # Temperature array with refinement close to the critical point
    alpha = 0.00
    T_max = fluid.critical_point.T - 0.5
    T_min = alpha * T_max + (1 - alpha) * fluid.triple_point_liquid.T
    ratio = 1 - T_min / T_max
    t1 = np.logspace(np.log10(1 - 0.9999), np.log10(ratio / 10), int(np.ceil(N / 2)))
    t2 = np.logspace(np.log10(ratio / 10), np.log10(ratio), int(np.floor(N / 2)))
    T_spinodal = (1 - np.concatenate([t1, t2])) * T_max

    # Get limits of entropy to prevent points where EoS breaks down
    s_min, s_max = fluid.triple_point_liquid["s"], fluid.triple_point_vapor["s"]

    # Initialize dictionaries for storing properties
    spinodal_liq = {}
    spinodal_vap = {}

    # Compute liquid branch until calculations break down
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
        )

        if s_min < props_liq["s"] < s_max:
            for name in props_liq.keys():
                if name not in spinodal_liq:  # Initialize list if new property
                    spinodal_liq[name] = []
                spinodal_liq[name].append(props_liq.get(name, np.nan))
        else:
            break

    # Compute vapor branch until calculations break down
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

        if s_min < props_vap["s"] < s_max:  # If not satisfied the HEOS is blowing up
            for name in props_vap.keys():
                if name not in spinodal_vap:  # Initialize list if new property
                    spinodal_vap[name] = []
                spinodal_vap[name].append(props_vap.get(name, np.nan))
        else:
            break

    # # Reverse the liquid properties for easy concatenation
    # for name in spinodal_liq.keys():
    #     spinodal_liq[name] = list(reversed(spinodal_liq[name]))

    return spinodal_liq, spinodal_vap


def compute_pseudocritical_line(fluid, N_points=100):
    # Initialize objects to store properties
    prop_names = fluid._properties.keys()
    pseudocritical_line = {name: [] for name in prop_names}

    # Define temperature array with refinement close to the critical point
    tau = np.logspace(np.log10(1e-3), np.log10(1), N_points)
    T_range = (1 + tau) * fluid.critical_point.T

    # Loop over temperatures and compute pseudocritical properties
    for T in T_range:
        for name in prop_names:
            state = fluid.get_state(DmassT_INPUTS, fluid.critical_point.d, T)
            pseudocritical_line[name].append(state[name])

    return pseudocritical_line


def compute_quality_grid(fluid, num_points, quality_levels):
    # Define temperature levels
    t1 = np.logspace(np.log10(1 - 0.9999), np.log10(0.1), int(num_points / 2))
    t2 = np.logspace(
        np.log10(0.1),
        np.log10(1 - (fluid.triple_point_liquid.T) / fluid.critical_point.T),
        int(num_points / 2),
    )
    temperature_levels = (1 - np.hstack((t1, t2))) * fluid.critical_point.T

    # Calculate property grid
    quality_grid = []
    for q in quality_levels:
        row = []
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
    m, n = len(range_1), len(range_2)

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
            row.append(props.compute_properties_metastable_rhoT(fluid._AS, rho, T))
        states_meta.append(row)

    # Convert nested list of dictionaries into dictionary of 2D arrays
    states_meta = states_to_dict_2d(states_meta)

    return states_meta


# TODO  Implement class to calculate intersection with saturation line?


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
    method="bfgs",
    tolerance=1e-6,
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
    if abs(final_residual) > 1e-6:
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
    method="bfgs",
    tolerance=1e-6,
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

    # Check solver method
    if method == "slsqp":
        pass
    elif method == "bfgs":
        problem.get_bounds = lambda: None
    else:
        raise ValueError(
            f"Solver method {method} is not valid. Valid options are 'slsqp' and 'bfgs'"
        )

    # Create solver object
    solver = psv.OptimizationSolver(
        problem=problem,
        library="scipy",
        method=method,  # "l-bfgs-b", "slsqp" "bfgs"
        tolerance=tolerance,
        print_convergence=print_convergence,
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
            rho_array = np.linspace(self.rho_liq, self.rho_vap, N)
        elif self.branch == "vapor":
            rho_array = np.linspace(self.rho_vap, self.rho_liq, N)
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
