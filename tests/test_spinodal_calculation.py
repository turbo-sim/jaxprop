import os
import pytest
import atexit
import numpy as np
import coolpropx as cpx
import pandas as pd

from utilities import get_reference_state, assert_consistent_values

# Consistency statistics
SPINODAL_LOG = []
PRINT_STATISTICS = os.environ.get("PRINT_STATISTICS") == "1"

# Tolerances for comparison
# Loose tolerance as the isothermal bulk modulus takes large values
TOL = 1e-4

# Define list of calculation backends
from utilities import BACKENDS

# Define all working fluids
FLUID_NAMES = [
    "Water",
    "CO2",
    "Ammonia",
    # "Nitrogen",
    "Pentane",
    "R134a",
    "R245fa",
    "R1233ZDE",
]

@pytest.mark.parametrize("fluid_name", FLUID_NAMES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_spinodal_calculation(fluid_name, backend):

    # Compute spinodal points
    fluid = cpx.Fluid(fluid_name, backend)
    spinodal_liq, spinodal_vap = cpx.compute_spinodal_line(
        fluid,
        N=10,
        method="slsqp",
        tolerance=1e-12,
        eps=1.0
    )

    # Check both spinodal branches
    for label, spinodal in [("liquid", spinodal_liq), ("vapor", spinodal_vap)]:
        p = np.array(spinodal["p"])
        B = np.array(spinodal["isothermal_bulk_modulus"])
        B_over_p = B / p

        # Log values (optional)
        SPINODAL_LOG.append({
            "fluid": fluid_name,
            "backend": backend,
            "branch": label,
            "max(B/p)": np.max(np.abs(B_over_p)),
        })

        # Assert near-zero normalized bulk modulus
        assert np.allclose(
            B_over_p, 0.0, atol=TOL
        ), f"{label.capitalize()} spinodal: B/p not zero for {fluid_name} ({backend}) — max(B/p) = {np.max(np.abs(B_over_p)):.3e}"


# Print consistency summary
log_name = os.path.splitext(os.path.basename(__file__))[0]

def print_log_summary():
    if PRINT_STATISTICS:
        if SPINODAL_LOG:
            df = pd.DataFrame(SPINODAL_LOG)
            df = df.sort_values(by="max(B/p)", ascending=False)
            print(f"\n[INFO] {log_name} summary (top 20 highest normalized bulk moduli):")
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

