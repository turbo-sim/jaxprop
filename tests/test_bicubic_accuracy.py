import os
import pytest
import atexit
import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP
import jaxprop as jxp
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Fluid-specific domains
# -------------------------------------------------------
# Global log and printing flag
INTERP_LOG = []
PRINT_STATISTICS = os.environ.get("PRINT_STATISTICS") == "1"

CORE_PROPERTIES = (
    "pressure",
    "temperature",
    "density",
    "enthalpy",
    "entropy",
    "internal_energy",
    "isobaric_heat_capacity",
    "isochoric_heat_capacity",
    "speed_of_sound",
)

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
        # HEOS publishes Tmax=455 K. The former 500–800 kJ/kg domain
        # extrapolated up to 641 K; retain a broad vapor domain below Tmax.
        "h_min": 450e3,
        "h_max": 550e3,
        "p_min": 2e3,
        "p_max": 20e5,
    },
]


# True cell centers and maximum local errors need finer grids than the old
# arbitrary 25-point samples. Retain the original percentage error budgets.
GRID_CASES = [
    {"N": 16, "tol": 1e-1},
    {"N": 32, "tol": 1e-2},
    {"N": 64, "tol": 1e-3},
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
    fluid_ref = jxp.FluidJAX(fluid_name)

    # Plot phase diagram background
    fig, ax = fluid_ref.fluid.plot_phase_diagram(
        x_prop="enthalpy", y_prop="pressure", x_scale="linear", y_scale="log"
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
def sample_grids(fluid_case, num_nodes):
    """Return actual cell centers and boundary nodes/edge centers."""
    h_nodes = np.linspace(fluid_case["h_min"], fluid_case["h_max"], num_nodes)
    log_p_nodes = np.linspace(
        np.log(fluid_case["p_min"]), np.log(fluid_case["p_max"]), num_nodes
    )
    h_centers = (h_nodes[:-1] + h_nodes[1:]) / 2
    p_centers = np.exp((log_p_nodes[:-1] + log_p_nodes[1:]) / 2)
    center_grid = np.meshgrid(h_centers, p_centers, indexing="ij")

    h_edges = np.sort(np.concatenate((h_nodes, h_centers)))
    p_edges = np.sort(np.concatenate((np.exp(log_p_nodes), p_centers)))
    boundary_grid = (
        np.concatenate(
            (
                np.full(p_edges.shape, fluid_case["h_min"]),
                np.full(p_edges.shape, fluid_case["h_max"]),
                h_edges,
                h_edges,
            )
        ),
        np.concatenate(
            (
                p_edges,
                p_edges,
                np.full(h_edges.shape, fluid_case["p_min"]),
                np.full(h_edges.shape, fluid_case["p_max"]),
            )
        ),
    )
    return {"cell_centers": center_grid, "boundaries": boundary_grid}


def test_accuracy_sampling_uses_cell_centers_and_domain_boundaries():
    """Protect the intended cell geometry with independently known coordinates."""
    domain = {"h_min": 0.0, "h_max": 6.0, "p_min": 1.0, "p_max": 64.0}
    grids = sample_grids(domain, num_nodes=4)
    h_centers, p_centers = grids["cell_centers"]
    np.testing.assert_allclose(h_centers, [[1, 1, 1], [3, 3, 3], [5, 5, 5]])
    np.testing.assert_allclose(p_centers, [[2, 8, 32], [2, 8, 32], [2, 8, 32]])

    h_boundary, p_boundary = grids["boundaries"]
    assert h_boundary.shape == p_boundary.shape == (28,)
    assert np.all(
        (h_boundary == 0) | (h_boundary == 6) | (p_boundary == 1) | (p_boundary == 64)
    )
    for enthalpy, pressure in ((0, 1), (0, 64), (6, 1), (6, 64), (0, 2), (3, 64)):
        assert np.any(
            np.isclose(h_boundary, enthalpy) & np.isclose(p_boundary, pressure)
        )


@pytest.mark.parametrize("fluid_case", FLUID_CASES, ids=lambda case: case["fluid"])
def test_accuracy_fixture_boundaries_respect_published_source_temperature_limits(
    fluid_case,
):
    """Accuracy fixtures must test interpolation within the source EOS domain."""
    source = CP.AbstractState("HEOS", fluid_case["fluid"])
    minimum, maximum = source.Tmin(), source.Tmax()
    enthalpies, pressures = sample_grids(fluid_case, num_nodes=64)["boundaries"]
    for enthalpy, pressure in zip(enthalpies, pressures):
        # Raw CoolProp accepts some extrapolated states, so inspect its result
        # independently of the wrapper's domain-failure policy.
        source.update(CP.HmassP_INPUTS, float(enthalpy), float(pressure))
        temperature = source.T()
        assert minimum <= temperature <= maximum, (
            f"{fluid_case['fluid']}: h={enthalpy}, p={pressure} resolves to "
            f"T={temperature} K outside [{minimum}, {maximum}] K"
        )


@pytest.mark.parametrize("fluid_case", FLUID_CASES)
@pytest.mark.parametrize("grid_case", GRID_CASES)
def test_bicubic_midpoint_accuracy(fluid_case, grid_case, tmp_path):
    """
    Compare true cell centers and boundaries with the source properties.

    Preserve the mean percentage-error budgets and enforce the same budgets
    for the maximum local error of the required core properties.
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
        h_min=h_min,
        h_max=h_max,
        p_min=p_min,
        p_max=p_max,
        N_h=N,
        N_p=N,
        table_dir=str(tmp_path),
    )

    # Reference CoolProp model
    fluid_cp = jxp.FluidJAX(fluid_name)

    failures = []
    for sample_name, (H_mesh, P_mesh) in sample_grids(fluid_case, N).items():
        interp_grid = fluid_bicubic.get_state(jxp.HmassP_INPUTS, H_mesh, P_mesh)
        true_grid = fluid_cp.get_state(jxp.HmassP_INPUTS, H_mesh, P_mesh)
        for prop in jxp.PROPERTIES_CANONICAL:
            assert interp_grid.supports_property(prop) == true_grid.supports_property(
                prop
            ), prop
            if not interp_grid.supports_property(prop):
                assert prop not in CORE_PROPERTIES
                with pytest.raises(jxp.UnsupportedPropertyError):
                    interp_grid.raw_property(prop)
                with pytest.raises(jxp.UnsupportedPropertyError):
                    true_grid.raw_property(prop)
                continue
            interp_val = np.asarray(interp_grid.raw_property(prop))
            true_val = np.asarray(true_grid.raw_property(prop))
            context = f"{fluid_name} N={N} {sample_name} prop={prop}"
            assert interp_val.shape == true_val.shape == H_mesh.shape, context
            if prop in CORE_PROPERTIES:
                assert np.all(np.isfinite(true_val)), f"{context}: invalid reference"
                assert np.all(
                    np.isfinite(interp_val)
                ), f"{context}: invalid interpolant"
            np.testing.assert_array_equal(
                np.isnan(interp_val), np.isnan(true_val), err_msg=context
            )
            assert not np.any(np.isinf(true_val)), f"{context}: infinite reference"
            assert not np.any(np.isinf(interp_val)), f"{context}: infinite interpolant"
            mask = np.isfinite(true_val)
            if not np.any(mask):
                continue

            err = mean_absolute_percentage_error(interp_val[mask], true_val[mask])
            log_result(fluid_name, N, prop, err, tol)
            if err >= tol:
                failures.append(f"{context}: mean error {err:.6e}% >= {tol:.6e}%")
            if prop in CORE_PROPERTIES:
                local_error = (
                    100
                    * np.abs(interp_val[mask] - true_val[mask])
                    / np.maximum(np.abs(true_val[mask]), 1e-30)
                )
                maximum_error = float(np.max(local_error))
                if maximum_error >= tol:
                    failures.append(
                        f"{context}: maximum error {maximum_error:.6e}% >= {tol:.6e}%"
                    )
    assert not failures, "\n".join(failures)


def log_result(fluid_name, N, prop, err, tol):
    """Append one result to the global log."""
    INTERP_LOG.append(
        {
            "fluid": fluid_name,
            "N": N,
            "property": prop,
            "MAPE": err,
            "tol": tol,
            "pass": err < tol,
        }
    )


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
