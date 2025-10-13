import os
import shutil
import atexit
import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jaxprop as jxp



# -------------------------------------------------------
# Settings
# -------------------------------------------------------
# Global log and printing flag
CONSISTENCY_LOG = []
PRINT_STATISTICS = os.environ.get("PRINT_STATISTICS") == "1"

# Import fluid cases from the bicubic test module
from test_bicubic_accuracy import FLUID_CASES

TABLE_DIR = "fluid_tables"
GRID_SIZE = 40
CONSISTENCY_TOL = 1e-10  # relative L² error tolerance

# Input types to test (derived from the class itself)
INPUT_TYPES = list(jxp.bicubic.FluidBicubic.PROPERTY_CALCULATORS.keys())


# -------------------------------------------------------
# Utility
# -------------------------------------------------------
def l2_relative_error(a, b):
    """Relative L² error between arrays a and b."""
    eps = 1e-16
    diff = np.ravel(a - b)
    ref = np.ravel(b)
    num = np.linalg.norm(diff, ord=2)
    den = np.linalg.norm(ref, ord=2)
    return num / max(den, eps)


# -------------------------------------------------------
# Fixtures
# -------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def clean_tables_start():
    """
    Delete all existing tables at the start of the test session if they exist.
    Does not regenerate tables.
    """
    if os.path.exists(TABLE_DIR):
        print(f"[INFO] Removing existing table directory: {TABLE_DIR}")
        shutil.rmtree(TABLE_DIR, ignore_errors=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    # Run the tests
    yield

# -------------------------------------------------------
# Test
# -------------------------------------------------------
@pytest.mark.parametrize("input_type", INPUT_TYPES)
@pytest.mark.parametrize("fluid_case", FLUID_CASES)
def test_solver_consistency(fluid_case, input_type):
    """
    Check that each solver input pair gives results consistent with H-P baseline,
    based on relative L² errors of all canonical properties.
    """
    fluid_name = fluid_case["fluid"]
    h_min, h_max = fluid_case["h_min"], fluid_case["h_max"]
    p_min, p_max = fluid_case["p_min"], fluid_case["p_max"]

    # Load bicubic fluid model (table already exists)
    fluid = jxp.FluidBicubic(
        fluid_name=fluid_name,
        backend="HEOS",
        h_min=h_min, h_max=h_max,
        p_min=p_min, p_max=p_max,
        N_h=GRID_SIZE, N_p=GRID_SIZE,
        table_dir=TABLE_DIR,
        coarse_step=1,
    )

    # Reference grid at midpoints
    h_vals = jnp.linspace(h_min, h_max, 25)
    logp_vals = jnp.linspace(jnp.log(p_max), jnp.log(p_max), 25)
    p_vals = jnp.exp(logp_vals)
    H_mesh, P_mesh = jnp.meshgrid(h_vals, p_vals, indexing="ij")

    # PH baseline
    state_ref = fluid.get_state(jxp.HmassP_INPUTS, H_mesh, P_mesh)

    # Extract required inputs from reference state
    var1_name, var2_name = jxp.INPUT_PAIR_MAP[input_type]
    v1 = state_ref[var1_name]
    v2 = state_ref[var2_name]

    # Interpolate using this input pair
    interp = fluid.get_state(input_type, v1, v2)

    # Compare property by property
    for prop in jxp.PROPERTIES_CANONICAL:
        ref_val = np.array(state_ref[prop])
        interp_val = np.array(interp[prop])

        mask = np.isfinite(ref_val)
        if not np.any(mask):
            continue
        
        rel_err = l2_relative_error(interp_val[mask], ref_val[mask])
        log_result(fluid_name, input_type, prop, rel_err, CONSISTENCY_TOL)


        assert rel_err < CONSISTENCY_TOL, (
            f"[{fluid_name}] input={jxp.INPUT_TYPE_MAP[input_type]} "
            f"prop={prop} rel L2 error={rel_err:.3e} > tol={CONSISTENCY_TOL:.1e}"
        )


def log_result(fluid_name, input_type, prop, rel_err, tol):
    """Append one result to the global consistency log."""
    CONSISTENCY_LOG.append({
        "fluid": fluid_name,
        "input_type": jxp.INPUT_TYPE_MAP[input_type],
        "property": prop,
        "rel_L2_error": rel_err,
        "tol": tol,
        "pass": rel_err < tol,
    })


def print_log_summary():
    """Print a summary of all logged consistency results at program exit."""
    if not PRINT_STATISTICS:
        return
    if CONSISTENCY_LOG:
        df = pd.DataFrame(CONSISTENCY_LOG)
        df = df.sort_values(by="rel_L2_error", ascending=False)
        print("\n[INFO] Solver consistency test summary (all results):")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    else:
        print("\n[INFO] Consistency log is empty.")

atexit.register(print_log_summary)


if __name__ == "__main__":

    # Running pytest from this script
    os.environ["PRINT_STATISTICS"] = "1"
    PRINT_STATISTICS = os.environ.get("PRINT_STATISTICS") == "1"
    pytest.main([__file__, "-v"])
    # test_solver_consistency(FLUID_CASES[1], INPUT_TYPES[4])

