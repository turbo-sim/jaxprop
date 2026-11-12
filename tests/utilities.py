
import os
import numpy as np
import jaxprop as jxp

def get_available_backends():
    """Select test backends, opting into REFPROP with JAXPROP_TEST_BACKENDS.

    HEOS is the default on both local machines and CI. For example, set
    ``JAXPROP_TEST_BACKENDS=HEOS,REFPROP`` to also collect REFPROP cases.
    Requested REFPROP cases are skipped by conftest when its library cannot
    be initialized. Each other collected calculation retains its own failures.
    """
    requested = os.environ.get("JAXPROP_TEST_BACKENDS", "HEOS")
    backends = list(dict.fromkeys(name.strip().upper() for name in requested.split(",")))
    if any(name not in {"HEOS", "REFPROP"} for name in backends):
        raise ValueError(
            "JAXPROP_TEST_BACKENDS must contain HEOS and/or REFPROP, "
            f"separated by commas. Received {requested!r}."
        )
    return backends

# Define reference states dynamically
def get_reference_state(fluid, backend, label):
    fluid = jxp.Fluid(fluid, backend)
    p_crit = fluid.critical_point.p
    T_crit = fluid.critical_point.T
    s_crit = fluid.critical_point.s
    ds_crit = (fluid.triple_point_vapor.s - fluid.triple_point_liquid.s) / 10
    p_subcritical = 0.6 * p_crit
    T_sat_subcritical = fluid.get_state(jxp.PQ_INPUTS, p_subcritical, 0.0).T

    if label == "saturated_liquid":
        return fluid.get_state(jxp.PQ_INPUTS, p_subcritical, 0 + 1e-3)

    elif label == "saturated_vapor":
        return fluid.get_state(jxp.PQ_INPUTS, p_subcritical, 1 - 1e-3)

    elif label == "two_phase":
        return fluid.get_state(jxp.PQ_INPUTS, p_subcritical, 0.5)

    elif label == "subcooled_liquid":
        state = fluid.get_state(jxp.PT_INPUTS, p_subcritical, T_sat_subcritical - 5)
        return fluid.get_state(jxp.DmassT_INPUTS, state.rhomass, state.T)

    elif label == "superheated_vapor":
        state = fluid.get_state(jxp.PT_INPUTS, p_subcritical, T_sat_subcritical + 5)
        return fluid.get_state(jxp.DmassT_INPUTS, state.rhomass, state.T)

    elif label == "supercritical_liquid":
        state = fluid.get_state(jxp.PSmass_INPUTS, 1.5 * p_crit, s_crit - ds_crit)
        return fluid.get_state(jxp.DmassT_INPUTS, state.rhomass, state.T)

    elif label == "supercritical_gas":
        T = min(fluid.abstract_state.Tmax(), 1.1*T_crit)
        state = fluid.get_state(jxp.SmassT_INPUTS, s_crit + ds_crit, T)
        return fluid.get_state(jxp.DmassT_INPUTS, state.rhomass, state.T)

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
    *,
    atol=None,
    rtol=None,
):
    """Compare every element of real scalar/array properties and record coverage.

    Inputs must have identical, nonempty shapes. Core thermodynamic properties
    must be finite. Paired NaNs represent unavailable optional properties.
    Infinities and unmatched NaNs fail. Each finite element must satisfy the
    absolute or relative tolerance, which default to ``tolerance``. ``atol``
    and ``rtol`` allow property-specific tolerances in their respective units.

    Return whether the comparison passed. ``raise_error=False`` records failed
    numerical comparisons for diagnostics. Invalid types, shapes and tolerance
    settings always raise, so a malformed comparison cannot silently pass.
    """
    reference = np.asarray(v_ref)
    calculated = np.asarray(v_new)
    if reference.dtype.kind not in "biuf" or calculated.dtype.kind not in "biuf":
        raise TypeError(f"{prop_name} comparison requires real numeric values")
    if reference.shape != calculated.shape:
        raise AssertionError(
            f"{prop_name} shapes differ: {reference.shape} != {calculated.shape}"
        )
    if reference.size == 0:
        raise AssertionError(f"{prop_name} comparison contains no values")
    absolute_tolerance = float(tolerance if atol is None else atol)
    relative_tolerance = float(tolerance if rtol is None else rtol)
    tolerances = np.array([absolute_tolerance, relative_tolerance])
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0):
        raise ValueError("Comparison tolerances must be finite and nonnegative")

    reference = reference.astype(np.float64)
    calculated = calculated.astype(np.float64)
    finite = np.isfinite(reference) & np.isfinite(calculated)
    paired_nan = np.isnan(reference) & np.isnan(calculated)
    core_properties = {
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    }
    canonical_name = jxp.ALIAS_TO_CANONICAL.get(prop_name, prop_name)
    unavailable = paired_nan & (canonical_name not in core_properties)
    absolute_errors = np.full(reference.shape, np.inf)
    relative_errors = np.full(reference.shape, np.inf)
    absolute_errors[finite] = np.abs(calculated[finite] - reference[finite])
    np.divide(
        absolute_errors,
        np.abs(reference),
        out=relative_errors,
        where=finite & (reference != 0),
    )
    absolute_errors[unavailable] = 0.0
    relative_errors[unavailable] = 0.0
    passed = bool(
        np.all(
            (absolute_errors <= absolute_tolerance)
            | (relative_errors <= relative_tolerance)
        )
    )
    abs_err = float(np.max(absolute_errors))
    rel_err = float(np.max(relative_errors))
    min_error = float(np.max(np.minimum(absolute_errors, relative_errors)))
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
            "finite_count": int(np.count_nonzero(finite)),
            "unavailable_count": int(np.count_nonzero(unavailable)),
            "value_count": int(reference.size),
            "passed": passed,
        }
    )
    if not passed and raise_error:
        raise AssertionError(
            f"Inconsistency in '{prop_name}' for fluid '{backend}::{fluid_name}' "
            f"at state '{state_label}' using input '{input_type}'\n"
            f"  input = ({prop1_name} = {prop1}, {prop2_name} = {prop2})\n"
            f"  ref = {reference}, new = {calculated}\n"
            f"  abs_err = {abs_err:.2e}, rel_err = {rel_err:.2e} "
            f"(required abs_err <= {absolute_tolerance:.1e} "
            f"or rel_err <= {relative_tolerance:.1e})"
        )
    return passed

