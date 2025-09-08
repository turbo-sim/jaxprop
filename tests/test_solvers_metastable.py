import os
import pytest
import atexit
import numpy as np
import jaxprop as jaxp
import pandas as pd

from utilities import get_reference_state, assert_consistent_values, get_available_backends

# import CoolProp.CoolProp as CP
# CP.set_config_bool(CP.ENABLE_SUPERANCILLARIES, True)


# Consistency statistics
CONSISTENCY_LOG = []
PRINT_STATISTICS = os.environ.get("PRINT_STATISTICS") == "1"

# Define list of calculation backends
BACKENDS = get_available_backends()

# Define all solver combinations
SOLVERS_TWO_PHASE = [
    ("Q", "T"),
    ("Q", "p"),
    ("Q", "h"),
    ("Q", "s"),
    ("T", "s"),
    ("T", "h"),
    ("p", "s"),
    ("p", "h"),
    ("h", "s")
]

SOLVERS_SINGLE_PHASE = [
    ("T", "p"),
    ("T", "s"),
    ("T", "h"),
    ("p", "T"),
    ("p", "s"),
    ("p", "h"),
    ("h", "s")
]

# Aliases for the vapor quality
QUALITY_NAMES = [
    "Q",
    "quality_mass",
    "vapor_quality",
    "void_fraction",
    "quality_volume",
]

# Tolerances for comparison
# The Helmholtz solver needs looser tolerance
TOL = 1e-3

# Define all working fluids
FLUID_NAMES = [
    "Water",
    "CO2",
    "Ammonia",
    "Nitrogen",
    "Pentane",
    "R245fa",
    "R1233ZDE",
]

# Define names of reference states
STATE_LABELS = [
    "subcooled_liquid",
    "superheated_vapor",
    "supercritical_liquid",
    "supercritical_gas",
]

SOLVERS_SINGLE_PHASE = [
    ("T", "p"),
    ("T", "s"),
    ("T", "h"),
    ("p", "T"),
    ("p", "s"),
    ("p", "h"),
    ("h", "s")
]

 
@pytest.mark.parametrize("fluid_name", FLUID_NAMES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("state_label", STATE_LABELS)
def test_metastable_solver_consistency(fluid_name, backend, state_label):

    # Compute reference state
    fluid = jaxp.Fluid(fluid_name, backend)
    state_ref = get_reference_state(fluid_name, backend, state_label)
    
    # Choose solver types based on current state
    input_map = SOLVERS_TWO_PHASE if state_ref.is_two_phase else SOLVERS_SINGLE_PHASE

    # Provide the density-temperature initial guess
    # Small perturbation to ensure solver converges
    rho_in, T_in = 1.05 * state_ref.rhomass, state_ref.T + 2.5

    # Loop over the different custom input types
    for prop1_name, prop2_name in input_map:

        # Get the input values
        prop1 = state_ref[prop1_name]
        prop2 = state_ref[prop2_name]

        # Compute the new state and check consistency
        state_new = fluid.get_state_metastable(
            prop_1=prop1_name,
            prop_1_value=prop1,
            prop_2=prop2_name,
            prop_2_value=prop2,
            rhoT_guess=[rho_in, T_in],
            supersaturation=False,
            generalize_quality=True,
            solver_algorithm="lm",
            solver_tolerance=1e-6,
            solver_max_iterations=100,
            print_convergence=True,
        )

        for key, v_ref in state_ref.items():

            # Skip quality checks
            if key in QUALITY_NAMES:
                continue

            # Skip non-numeric keys
            if key in ("is_two_phase", "identifier"):
                continue

            # Skip nested keys
            if key in ("saturation_liquid", "saturation_vapor"):
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
                input_type=f"{prop1_name}-{prop2_name}",
                prop1_name=prop1_name,
                prop1=prop1,
                prop2_name=prop2_name,
                prop2=prop2,
                tolerance=TOL,
                log_list=CONSISTENCY_LOG,
                raise_error=True,
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

