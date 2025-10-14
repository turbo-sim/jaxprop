import os
import shutil
import pytest
import atexit
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Fluid-specific domains
# -------------------------------------------------------
# Global log and printing flag
INTERP_LOG = []
PRINT_STATISTICS = os.environ.get("PRINT_STATISTICS") == "1"

TABLE_DIR = "fluid_tables"

FLUID_CASES = [
    {
        "fluid": "CO2",
        "h_min": 600e3,
        "h_max": 2000e3,
        "p_min": 20e5,
        "p_max": 300e5,
    },
    {
        "fluid": "Water",
        "h_min": 3000e3,
        "h_max": 3500e3,
        "p_min": 2e5,
        "p_max": 200e5,
    },
    {
        "fluid": "R134a",
        "h_min": 500e3,
        "h_max": 800e3,
        "p_min": 2e3,
        "p_max": 20e5,
    },
]


GRID_CASES = [
    {"N": 10, "tol": 1e-1},   # coarse grid → lower accuracy expected
    {"N": 20, "tol": 1e-2},
    {"N": 40, "tol": 1e-3},   # fine grid → tight tolerance
]

# -------------------------------------------------------
# Utility
# -------------------------------------------------------

def mean_absolute_percentage_error(a, b):
    """Mean Absolute Percentage Error (MAPE) between arrays a and b."""
    a_flat = np.ravel(a)
    b_flat = np.ravel(b)
    rel_err = np.abs(a_flat - b_flat) / np.maximum(np.abs(b_flat), 1e-30)
    return np.mean(rel_err) * 100.0


def plot_table_domain(fluid_case):
    """
    Plot the thermodynamic domain and midpoint evaluation grid for a given fluid case.

    Parameters
    ----------
    case : dict
        Dictionary containing 'fluid', 'h_min', 'h_max', 'p_min', 'p_max'.
    N : int
        Number of grid points in each direction (used to compute midpoints).
    """
    fluid_name = fluid_case["fluid"]
    h_min, h_max = fluid_case["h_min"], fluid_case["h_max"]
    p_min, p_max = fluid_case["p_min"], fluid_case["p_max"]

    # Reference fluid (for plotting phase diagram)
    fluid_ref = jxp.FluidJAX(fluid_name, exceptions=False)

    # Plot phase diagram background
    fig, ax = fluid_ref.fluid.plot_phase_diagram(
        x_prop="enthalpy",
        y_prop="pressure",
        x_scale="linear",
        y_scale="log"
    )

    # Draw interpolation domain box
    h_box = [h_min, h_max, h_max, h_min, h_min]
    p_box = [p_min, p_min, p_max, p_max, p_min]
    ax.plot(h_box, p_box, "r-", linewidth=1.25, label="interpolation domain")
    fig.tight_layout(pad=1)

    return fig, ax

# -------------------------------------------------------
# Tests
# -------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_table_dir():
    """
    Automatically clean the fluid_tables directory before each test.
    This ensures no cached tables from previous runs affect current results.
    """
    shutil.rmtree(TABLE_DIR, ignore_errors=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    yield  # run the test
    # Optional: clean again after test
    shutil.rmtree(TABLE_DIR, ignore_errors=True)


@pytest.mark.parametrize("fluid_case", FLUID_CASES)
@pytest.mark.parametrize("grid_case", GRID_CASES)
def test_bicubic_midpoint_accuracy(fluid_case, grid_case):
    """
    Check that bicubic interpolation at midpoints matches CoolProp
    within prescribed L2 relative error thresholds for key properties.
    """

    # Fluid-specific domain
    fluid_name = fluid_case["fluid"]
    h_min, h_max = fluid_case["h_min"], fluid_case["h_max"]
    p_min, p_max = fluid_case["p_min"], fluid_case["p_max"]
    N = grid_case["N"]
    tol = grid_case["tol"]


    # Build bicubic interpolant
    fluid_bicubic = jxp.FluidBicubic(
        fluid_name=fluid_name,
        backend="HEOS",
        h_min=h_min, h_max=h_max,
        p_min=p_min, p_max=p_max,
        N_h=N, N_p=N,
        table_dir=TABLE_DIR,
    )

    # Reference CoolProp model
    fluid_cp = jxp.FluidJAX(fluid_name, exceptions=False)

    # Midpoint grid
    h_vals = jnp.linspace(h_min, h_max, 25)
    p_vals = jnp.exp(jnp.linspace(jnp.log(p_min), jnp.log(p_max), 25))
    H_mesh, P_mesh = np.meshgrid(h_vals, p_vals, indexing="ij")

    # Evaluate both models at midpoints
    interp_grid = fluid_bicubic.get_state(jxp.HmassP_INPUTS, H_mesh, P_mesh)
    true_grid = fluid_cp.get_state(jxp.HmassP_INPUTS, H_mesh, P_mesh)

    # Properties to check
    # properties = ["T", "d", "e", "s", "a", "G"]
    props_to_check = jxp.PROPERTIES_CANONICAL

    for prop in props_to_check:
        interp_val = np.array(interp_grid[prop])
        true_val = np.array(true_grid[prop])

        # Mask out NaNs in the reference (true) values
        mask = ~np.isnan(true_val)
        if not np.any(mask):
            continue
        
        # Calculate error and save info
        err = mean_absolute_percentage_error(interp_val[mask], true_val[mask])
        log_result(fluid_name, N, prop, err, tol) 


        assert err < tol, (
            f"{fluid_name} N={N} prop={prop}: MAPE={err:.3e} > tol={tol:.1e}"
        )


def log_result(fluid_name, N, prop, err, tol):
    """Append one result to the global log."""
    INTERP_LOG.append({
        "fluid": fluid_name,
        "N": N,
        "property": prop,
        "MAPE": err,
        "tol": tol,
        "pass": err < tol
    })

def print_log_summary():
    """Print summary of interpolation errors after all tests (even if pytest fails)."""
    if not PRINT_STATISTICS:
        return
    if INTERP_LOG:
        df = pd.DataFrame(INTERP_LOG)
        df = df.sort_values(by="MAPE", ascending=False)
        print("\n[INFO] Bicubic interpolation test summary (all results):")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    else:
        print("\n[INFO] Interpolation log is empty.")


# Register the print function at exit
atexit.register(print_log_summary)

if __name__ == "__main__":

    # # Run specific test cases manually
    # test_bicubic_midpoint_accuracy(FLUID_CASES[2], GRID_CASES[0])
    # test_bicubic_midpoint_accuracy(FLUID_CASES[2], GRID_CASES[1])
    # test_bicubic_midpoint_accuracy(FLUID_CASES[2], GRID_CASES[2])

    # Running pytest from this script
    os.environ["PRINT_STATISTICS"] = "1"
    pytest.main([__file__, "-v"])
    # # # pytest.main([__file__, "-vv"])