
import os
import numpy as np
import jaxprop as cpx

# Detect if running in GitHub Actions
IN_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"

# Define available backends
def get_available_backends():
    if os.environ.get("GITHUB_ACTIONS", "false").lower() == "true":
        return ["HEOS"]
    else:
        return ["HEOS", "REFPROP"]

# Define reference states dynamically
def get_reference_state(fluid, backend, label):
    fluid = cpx.Fluid(fluid, backend)
    p_crit = fluid.critical_point.p
    T_crit = fluid.critical_point.T
    s_crit = fluid.critical_point.s
    ds_crit = (fluid.triple_point_vapor.s - fluid.triple_point_liquid.s) / 10
    p_subcritical = 0.6 * p_crit
    T_sat_subcritical = fluid.get_state(cpx.PQ_INPUTS, p_subcritical, 0.0).T

    if label == "saturated_liquid":
        return fluid.get_state(cpx.PQ_INPUTS, p_subcritical, 0 + 1e-3)

    elif label == "saturated_vapor":
        return fluid.get_state(cpx.PQ_INPUTS, p_subcritical, 1 - 1e-3)

    elif label == "two_phase":
        return fluid.get_state(cpx.PQ_INPUTS, p_subcritical, 0.5)

    elif label == "subcooled_liquid":
        state = fluid.get_state(cpx.PT_INPUTS, p_subcritical, T_sat_subcritical - 5)
        return fluid.get_state(cpx.DmassT_INPUTS, state.rhomass, state.T)

    elif label == "superheated_vapor":
        state = fluid.get_state(cpx.PT_INPUTS, p_subcritical, T_sat_subcritical + 5)
        return fluid.get_state(cpx.DmassT_INPUTS, state.rhomass, state.T)

    elif label == "supercritical_liquid":
        state = fluid.get_state(cpx.PSmass_INPUTS, 1.5 * p_crit, s_crit - ds_crit)
        return fluid.get_state(cpx.DmassT_INPUTS, state.rhomass, state.T)

    elif label == "supercritical_gas":
        T = min(fluid.abstract_state.Tmax(), 1.1*T_crit)
        state = fluid.get_state(cpx.SmassT_INPUTS, s_crit + ds_crit, T)
        return fluid.get_state(cpx.DmassT_INPUTS, state.rhomass, state.T)

    raise ValueError(f"Unknown state label: {label}")


def assert_consistent_values(
    v_ref,
    v_new,
    prop_name,
    fluid_name,
    backend,
    state_label,
    input_type,
    prop1_name,
    prop1,
    prop2_name,
    prop2,
    tolerance,
    log_list,
    raise_error=True,
):
    """
    Compare a reference and computed value for thermodynamic consistency.

    Checks if values agree within specified absolute or relative tolerances.
    Logs errors and optionally raises an exception if they do not agree.

    Parameters
    ----------
    v_ref : float
        Reference value (e.g., from a known or validated solver).
    v_new : float
        New value to be validated.
    prop_name : str
        Name of the thermodynamic property.
    fluid_name : str
        Name of the working fluid.
    backend : str
        Thermodynamic backend used (e.g., HEOS, REFPROP).
    state_label : str
        Label of the thermodynamic state (e.g., 'supercritical_gas').
    input_type : str
        Input variable combination used to define the state.
    prop1_name, prop2_name : str
        Names of the input variables.
    prop1, prop2 : float
        Input values used to define the state.
    tolerance : float
        Tolerance for consistency check.
    log_list : list
        List to store detailed log of error metrics.
    raise_error : bool
        Whether to raise an error if consistency fails.

    Notes
    -----
    The consistency check passes if:

        abs_error < atol  OR  rel_error < rtol

    """

    # Skip boolean types
    if isinstance(v_ref, (bool, np.bool_)) or isinstance(v_new, (bool, np.bool_)):
        return
    
    # Skip non-numeric types
    if not isinstance(v_ref, (int, float, np.number)) or not isinstance(v_new, (int, float, np.number)):
        return

    # Skip NaNs (both-NaN treated as consistent)
    if np.isnan(v_ref) and np.isnan(v_new):
        return

    # Compute error metrics
    abs_err = abs(v_new - v_ref)
    rel_err = abs_err / abs(v_ref) if abs(v_ref) > 0 else np.inf
    min_error = min(abs_err, rel_err)

    # Append result to log
    log_list.append(
        {
            "fluid": fluid_name,
            "backend": backend,
            "state": state_label,
            "property": prop_name,
            "input_type": input_type,
            "input_1": prop1,
            "input_2": prop2,
            "ref_value": v_ref,
            "new_value": v_new,
            "abs_error": abs_err,
            "rel_error": rel_err,
            "min_error": min_error,
        }
    )

    # Logical OR check (interpretable and robust)
    if not (abs_err < tolerance or rel_err < tolerance) and raise_error:
        raise AssertionError(
            f"Inconsistency in '{prop_name}' for fluid '{backend}::{fluid_name}' "
            f"at state '{state_label}' using input '{input_type}'\n"
            f"  input = ({prop1_name} = {prop1:.6g}, {prop2_name} = {prop2:.6g})\n"
            f"  ref = {v_ref:.6g}, new = {v_new:.6g}\n"
            f"  abs_err = {abs_err:.2e}, rel_err = {rel_err:.2e} "
            f"(fails check: abs_err < {tolerance:.1e} or rel_err < {tolerance:.1e})"
        )

