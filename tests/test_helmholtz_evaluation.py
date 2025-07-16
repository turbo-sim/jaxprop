import os
import pytest
import atexit
import numpy as np
import coolpropx as cpx
import pandas as pd

from utilities import get_reference_state, assert_consistent_values

# Consistency statistics
CONSISTENCY_LOG = []  
PRINT_STATISTICS = os.environ.get("PRINT_STATISTICS") == "1"

# Tolerances for comparison
TOL = 1e-4

# Define list of calculation backends
BACKENDS = [
    "HEOS",
    "REFPROP",
]

# Aliases for the vapor quality
QUALITY_NAMES = [
    "Q",
    "quality_mass",
    "vapor_quality",
    "void_fraction",
    "quality_volume",
]

# Define names of reference states
STATE_LABELS = [
    "subcooled_liquid",
    # "saturated_liquid",
    # "two_phase",
    # "saturated_vapor",
    "superheated_vapor",
    "supercritical_liquid",
    "supercritical_gas",
]

# Define all working fluids
FLUID_NAMES = [
    "Water",
    "CO2",
    "Ammonia",
    "Nitrogen",
    # "Pentane",
    # "R134a",
    "R245fa",
    "R1233ZDE",
]


@pytest.mark.parametrize("fluid_name", FLUID_NAMES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("state_label", STATE_LABELS)
def test_hemholtz_solver(fluid_name, backend, state_label):

    # Compute reference state
    fluid = cpx.Fluid(fluid_name, backend)
    state_ref = get_reference_state(fluid_name, backend, state_label)

    # Recompute state with direct call to Helmholtz EoS
    rho_in, T_in = state_ref.rhomass, state_ref.T
    props = cpx.compute_properties_metastable_rhoT(fluid.abstract_state, rho_in, T_in)

    # Check all properties are consistency
    for key, v_ref in state_ref.items():

        # Skip quality checks
        if key in QUALITY_NAMES:
            continue

        # Skip non-numeric keys
        if key in ("is_two_phase", "identifier"):
            continue

        # Check values are consistent
        v_new = props[key]
        assert_consistent_values(
            v_ref,
            v_new,
            prop_name=key,
            fluid_name=fluid_name,
            backend=backend,
            state_label=state_label,
            input_type="Helmholtz rhoT",
            prop1_name="rho",
            prop1=rho_in,
            prop2_name="T",
            prop2=T_in,
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



 
