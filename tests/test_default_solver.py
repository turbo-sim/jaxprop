import pytest
import numpy as np
import coolpropx as cpx
import pandas as pd

# Accumulate all deviations here
log_solver_consistency = []  

# Loose tolerance for comparison
ATOL = 1e-6
RTOL = 1e-6

# Define list of calculation backends
BACKENDS = [
    "HEOS",
    "REFPROP",
]

# Define all solver combinations
SOLVERS_TWO_PHASE = [
    "QT_INPUTS",
    "PQ_INPUTS",
    # "DmassQ_INPUTS",  # Weird error for water
    "DmassT_INPUTS",
    "SmassT_INPUTS",
    "DmassP_INPUTS",
    "HmassP_INPUTS",
    # "PSmass_INPUTS",
    "HmassSmass_INPUTS",
    "DmassHmass_INPUTS",
    "DmassSmass_INPUTS",
]

SOLVERS_SINGLE_PHASE = [
    "DmassT_INPUTS",
    "DmassP_INPUTS",
    "DmassHmass_INPUTS",
    "DmassSmass_INPUTS",
    "DmassUmass_INPUTS",
    "PT_INPUTS",
    "SmassT_INPUTS",
    "HmassP_INPUTS",
    "PSmass_INPUTS",
    "HmassSmass_INPUTS",
]

# Convert to input_id form for lookup
SOLVERS_SINGLE_PHASE = {
    v: k for k, v in cpx.INPUT_TYPE_MAP.items() if v in SOLVERS_SINGLE_PHASE
}
SOLVERS_TWO_PHASE = {
    v: k for k, v in cpx.INPUT_TYPE_MAP.items() if v in SOLVERS_TWO_PHASE
}

# Define all working fluids
FLUID_NAMES = [
    "Water",
    "CO2",
    "Ammonia",
    "Nitrogen",
    "Pentane",
    # "Cyclopentane",
    "R134a",
    "R245fa",
    "R1233ZDE",
]

STATE_LABELS = [
    "subcooled_liquid",
    "saturated_liquid",
    "two_phase",
    "saturated_vapor",
    "superheated_vapor",
    "supercritical_liquid",
    "supercritical_gas",
]


QUALITY_NAMES = [
    "Q",
    "quality_mass",
    "vapor_quality",
    "void_fraction",
    "quality_volume",
]


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
        return fluid.get_state(cpx.PQ_INPUTS, p_subcritical, 0 + 1e-4)

    elif label == "saturated_vapor":
        return fluid.get_state(cpx.PQ_INPUTS, p_subcritical, 1 - 1e-4)

    elif label == "two_phase":
        return fluid.get_state(cpx.PQ_INPUTS, p_subcritical, 0.5)

    elif label == "subcooled_liquid":
        return fluid.get_state(cpx.PT_INPUTS, p_subcritical, T_sat_subcritical - 5)

    elif label == "superheated_vapor":
        return fluid.get_state(cpx.PT_INPUTS, p_subcritical, T_sat_subcritical + 5)

    elif label == "supercritical_liquid":
        return fluid.get_state(cpx.PSmass_INPUTS, 1.5 * p_crit, s_crit - ds_crit)

    elif label == "supercritical_gas":
        T = min(fluid.abstract_state.Tmax(), 1.1*T_crit)
        return fluid.get_state(cpx.SmassT_INPUTS, s_crit + ds_crit, T)

    raise ValueError(f"Unknown state label: {label}")


# ---------------------------------------------------------------------------- #
# Check default solver consistency
# ---------------------------------------------------------------------------- #

def assert_consistent_values(
    v_ref,
    v_new,
    prop_name,
    fluid_name,
    state_label,
    input_type,
    prop1_name,
    prop1,
    prop2_name,
    prop2,
):
    """
    Compares scalar reference and new thermodynamic state values.

    Skips known nested keys (like saturation_liquid), treats both-NaN as equal,
    logs the dominant error (abs or rel), and optionally asserts consistency.
    """

    # Skip non-numeric types (e.g., strings)
    if (
        isinstance(v_ref, (bool, np.bool_))
        or isinstance(v_new, (bool, np.bool_))
        or not isinstance(v_ref, (int, float, np.number))
        or not isinstance(v_new, (int, float, np.number))
    ):
        return

    # Treat both-NaN as consistent
    if np.isnan(v_ref):
        return

    # Compute errors
    abs_err = abs(v_new - v_ref)
    rel_err = abs_err / abs(v_ref) if abs(v_ref) > 1e-8 else np.inf
    dominant_err = abs_err if abs(v_ref) < 1e-8 else rel_err

    log_solver_consistency.append(
        {
            "fluid": fluid_name,
            "state": state_label,
            "property": prop_name,
            "input_type": input_type,
            "input_1": prop1,
            "input_2": prop2,
            "ref_value": v_ref,
            "new_value": v_new,
            "abs_error": abs_err,
            "rel_error": rel_err,
            "dominant_error": dominant_err,
        }
    )

    if not np.isclose(v_ref, v_new, rtol=RTOL, atol=ATOL, equal_nan=True):
        raise AssertionError(
            f"Inconsistency in '{prop_name}' for fluid '{fluid_name}' and state '{state_label}' "
            f"with input type '{input_type}'\n"
            f"  input = ({prop1_name}={prop1:.6g}, {prop2_name}={prop2:.6g})\n"
            f"  reference value: {v_ref:.6g}, new value: {v_new:.6g}, "
            f"abs_err: {abs_err:.2e}, rel_err: {rel_err:.2e}"
        )


@pytest.mark.parametrize("fluid_name", FLUID_NAMES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("state_label", STATE_LABELS)
def test_internal_consistency(fluid_name, backend, state_label):
    fluid = cpx.Fluid(fluid_name, backend)
    state_ref = get_reference_state(fluid_name, backend, state_label)

    input_map = SOLVERS_TWO_PHASE if state_ref.is_two_phase else SOLVERS_SINGLE_PHASE

    for input_type, input_idx in input_map.items():

        # Get the input values
        prop1_name, prop2_name = cpx.extract_vars(input_type)
        prop1 = state_ref[prop1_name]
        prop2 = state_ref[prop2_name]

        # Clip the quality since it must be exactly between [0, 1]
        # (some of the CoolProp solvers return values very close but not example in the bracket)
        for name, value in [(prop1_name, prop1), (prop2_name, prop2)]:
            if name == "Q":
                clipped = np.clip(value, 0.0, 1.0)
                if abs(clipped - value) > 1e-6:
                    print(
                        f"[Q clipping] {fluid_name} input_type: {input_type}, {name} = {value:.6e} clipped to {clipped:.6e}"
                    )
                if name == prop1_name:
                    prop1 = clipped
                else:
                    prop2 = clipped

        # Compute the new state
        state_new = fluid.get_state(input_idx, prop1, prop2, generalize_quality=True)
        # print()
        # print(fluid_name, backend, state_label, input_type)

        for key, v_ref in state_ref.items():

            # Skip Q check if not two-phase
            if key in QUALITY_NAMES and not state_ref.is_two_phase:
                continue

            # Skip nested keys
            if key in ("saturation_liquid", "saturation_vapor"):
                continue

            # Skip volume fraction
            if key in ("quality_volume", "void_fraction"):
                continue

            # Check values are consistent
            v_new = state_new[key]
            assert_consistent_values(
                v_ref,
                v_new,
                prop_name=key,
                fluid_name=fluid_name,
                state_label=state_label,
                input_type=input_type,
                prop1_name=prop1_name,
                prop1=prop1,
                prop2_name=prop2_name,
                prop2=prop2,
            )




if __name__ == "__main__":

    # Running pytest from Python
    pytest.main([__file__])
    # pytest.main([__file__, "-v"])
    # pytest.main([__file__, "-vv"])

    # for label in STATE_LABELS:
    #     print(label)
    #     test_internal_consistency("CO2", "HEOS", label)
    # 
    # if log_solver_consistency:
    #     df = pd.DataFrame(log_solver_consistency)
    #     df_sorted = df.sort_values(by="dominant_error", ascending=False)
    #     print("\nTop 10 largest deviations:")
    #     print(df_sorted.head(10).to_string(index=False, float_format=lambda x: f"{x:.3e}"))


