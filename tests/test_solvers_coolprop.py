import os
import pytest
import atexit
import numpy as np
import jaxprop as jxp
import pandas as pd

from utilities import get_reference_state, assert_consistent_values, get_available_backends

# import CoolProp.CoolProp as CP
# CP.set_config_bool(CP.ENABLE_SUPERANCILLARIES, True)


# Consistency statistics
CONSISTENCY_LOG = []  
PRINT_STATISTICS = os.environ.get("PRINT_STATISTICS") == "1"

# Tolerances for comparison
TOL = 1e-6

# Define list of calculation backends
BACKENDS = get_available_backends()

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
    v: k for k, v in jxp.INPUT_TYPE_MAP.items() if v in SOLVERS_SINGLE_PHASE
}
SOLVERS_TWO_PHASE = {
    v: k for k, v in jxp.INPUT_TYPE_MAP.items() if v in SOLVERS_TWO_PHASE
}

# Define all working fluids
FLUID_NAMES = [
    "Water",
    "CO2",
    "Ammonia",
    "Nitrogen",
    "Pentane",
    "Cyclopentane",
    "R134a",
    "R245fa",
    "R1233ZDE",
]

# Define names of reference states
STATE_LABELS = [
    "subcooled_liquid",
    "saturated_liquid",
    "two_phase",
    "saturated_vapor",
    "superheated_vapor",
    "supercritical_liquid",
    "supercritical_gas",
]

# Aliases for the vapor quality
QUALITY_NAMES = [
    "Q",
    "quality_mass",
    "vapor_quality",
    "void_fraction",
    "quality_volume",
]


@pytest.mark.parametrize("fluid_name", FLUID_NAMES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("state_label", STATE_LABELS)
def test_default_solver_consistency(fluid_name, backend, state_label):

    # Compute reference state
    fluid = jxp.Fluid(fluid_name, backend)
    state_ref = get_reference_state(fluid_name, backend, state_label)

    # Choose solver types based on current state
    input_map = SOLVERS_TWO_PHASE if state_ref.is_two_phase else SOLVERS_SINGLE_PHASE

    # Loop over the different CoolProp input types
    for input_type, input_idx in input_map.items():

        # Get the input values
        prop1_name, prop2_name = jxp.extract_vars(input_type)
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

        # Compute the new state and check consistency
        state_new = fluid.get_state(input_idx, prop1, prop2, generalize_quality=True)
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
                backend=backend,
                state_label=state_label,
                input_type=input_type,
                prop1_name=prop1_name,
                prop1=prop1,
                prop2_name=prop2_name,
                prop2=prop2,
                tolerance=TOL,
                log_list=CONSISTENCY_LOG,
                raise_error=True
            )


# Print consistency summary
log_name = os.path.splitext(os.path.basename(__file__))[0]

def print_log_summary():
    if PRINT_STATISTICS:
        if CONSISTENCY_LOG:
            df = pd.DataFrame(CONSISTENCY_LOG)
            df = df.sort_values(by="min_error", ascending=False)
            print(f"\n[INFO] {log_name} summary (top 10 deviations):")
            print(df.head(20).to_string(index=False, float_format=lambda x: f"{x:.3e}"))
        else:
            print(f"\n[INFO] {log_name} log is empty.")

atexit.register(print_log_summary)


if __name__ == "__main__":

    # Running pytest from this script
    os.environ["PRINT_STATISTICS"] = "1"
    # pytest.main([__file__])
    pytest.main([__file__, "-v"])
    # # pytest.main([__file__, "-vv"])

