"""Small scientific grids retain state identity and their actual coordinates."""

import CoolProp.CoolProp as CP
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.coolprop import fluid_properties as functions
from utilities import get_available_backends


@pytest.fixture(scope="module", params=get_available_backends())
def fluid(request):
    return jp.Fluid("CO2", backend=request.param, identifier="scientific-grid")


def test_homogeneous_density_temperature_grid_is_a_common_state(fluid):
    densities = np.array([50.0, 200.0, 400.0])
    temperatures = np.array([330.0, 350.0])
    grid = functions.compute_property_grid_rhoT(fluid, densities, temperatures)
    assert isinstance(grid, jp.FluidState) and grid.shape == (2, 3)
    assert (
        grid.identifier == fluid.identifier and grid.reference_id == fluid.reference_id
    )
    np.testing.assert_array_equal(grid.status, 0)
    np.testing.assert_allclose(grid.rho, np.broadcast_to(densities, (2, 3)))
    np.testing.assert_allclose(grid.T, np.broadcast_to(temperatures[:, None], (2, 3)))
    for row, T in enumerate(temperatures):
        for column, rho in enumerate(densities):
            state = fluid.get_state_metastable(
                "rho", rho, "T", T, supersaturation=False, generalize_quality=False
            )
            np.testing.assert_allclose(grid.h[row, column], state.h, rtol=2e-14)


def test_small_saturation_quality_and_pseudocritical_grids(fluid):
    liquid, vapor = functions.compute_saturation_line(fluid, N=3, dT_crit=2.0)
    assert liquid.shape == vapor.shape == (4,)
    assert liquid.identifier == vapor.identifier == fluid.identifier
    assert np.all(np.diff(liquid.T) < 0)
    np.testing.assert_allclose(liquid.raw_property("quality_mass")[1:], 0.0, atol=2e-14)
    np.testing.assert_allclose(vapor.raw_property("quality_mass")[1:], 1.0, atol=2e-14)
    quality = functions.compute_quality_grid(fluid, 3, [0.0, 0.3, 1.0], dT_crit=2.0)
    assert quality.shape == (3, 3)
    np.testing.assert_allclose(
        quality.Q,
        np.broadcast_to(np.array([0.0, 0.3, 1.0])[:, None], (3, 3)),
        atol=2e-12,
    )
    pseudo = functions.compute_pseudocritical_line(fluid, N_points=4)
    assert pseudo.shape == (4,)
    np.testing.assert_allclose(pseudo.rho, float(fluid.critical_point.rho), rtol=2e-12)
    assert np.all(pseudo.T > float(fluid.critical_point.T))


@pytest.mark.parametrize(
    "rho,T",
    (
        ([], [300.0]),
        ([1.0], []),
        ([0.0], [300.0]),
        ([1.0], [np.nan]),
        ([[1.0]], [300.0]),
    ),
)
def test_invalid_homogeneous_grid_axes_are_explicit_configuration_errors(fluid, rho, T):
    with pytest.raises(jp.InvalidConfigurationError):
        functions.compute_property_grid_rhoT(fluid, rho, T)


@pytest.mark.parametrize(
    "function,options",
    (
        (functions.compute_saturation_line, {"N": 1}),
        (functions.compute_saturation_line, {"dT_crit": 0.0}),
        (functions.compute_saturation_line, {"dT_crit": 1e6}),
        (functions.compute_pseudocritical_line, {"N_points": 0}),
        (functions.compute_quality_grid, {"num_points": 3, "quality_levels": []}),
        (functions.compute_quality_grid, {"num_points": 3, "quality_levels": [1.2]}),
    ),
)
def test_invalid_grid_sampling_settings_are_explicit(fluid, function, options):
    with pytest.raises(jp.InvalidConfigurationError):
        function(fluid, **options)


@pytest.mark.parametrize("name", ("Air", "R410A"))
@pytest.mark.parametrize("backend", get_available_backends())
@pytest.mark.parametrize("qualities", ((0.0, 1.0), (0.0, 0.3, 1.0)))
def test_pseudopure_quality_grid_declares_actual_qt_geometry(name, backend, qualities):
    fluid = jp.Fluid(name, backend=backend)
    if backend == "HEOS" and 0.3 in qualities:
        with pytest.raises(jp.UnsupportedOperationError, match="Use PQ"):
            functions.compute_quality_grid(fluid, 3, qualities, dT_crit=3.0)
    else:
        state = functions.compute_quality_grid(fluid, 3, qualities, dT_crit=3.0)
        assert state.shape == (len(qualities), 3)
        np.testing.assert_allclose(
            state.raw_property("quality_mass"),
            np.broadcast_to(np.asarray(qualities)[:, None], state.shape),
            atol=3e-11,
        )


@pytest.fixture(scope="module")
def spinodal_fluid():
    return jp.Fluid("CO2", backend="HEOS")


@pytest.mark.parametrize("branch", ("liquid", "vapor"))
def test_small_spinodal_point_recovers_actual_stability_limit(spinodal_fluid, branch):
    state = functions.compute_spinodal_point(
        280.0, spinodal_fluid, branch, N_trial=30, tolerance=1e-11
    )
    assert state.valid and state.T == 280.0
    assert abs(float(state.K_T / state.p)) < 1e-5
    saturation = spinodal_fluid.get_state_saturation(CP.QT_INPUTS, [0.0, 1.0], 280.0)
    assert saturation.rho[1] < state.rho < saturation.rho[0]


def test_small_spinodal_line_preserves_both_branches(spinodal_fluid):
    liquid, vapor = functions.compute_spinodal_line(
        spinodal_fluid, N=3, dT_crit=3.0, tolerance=1e-11
    )
    assert liquid.shape[0] > 0 and vapor.shape[0] > 0
    assert np.max(np.abs(liquid.K_T / liquid.p)) < 1e-4
    assert np.max(np.abs(vapor.K_T / vapor.p)) < 1e-4


def test_general_spinodal_pressure_inverse_uses_a_dimensionless_residual(
    spinodal_fluid,
):
    reference = functions.compute_spinodal_point(
        280.0, spinodal_fluid, "vapor", N_trial=30, tolerance=1e-11
    )
    recovered = functions.compute_spinodal_point_general(
        "p", float(reference.p), spinodal_fluid, "vapor", N_trial=30, tolerance=1e-8
    )
    np.testing.assert_allclose(recovered.T, 280.0, rtol=0.0, atol=3e-5)
    np.testing.assert_allclose(recovered.p, reference.p, rtol=5e-8)


@pytest.mark.parametrize("backend", get_available_backends())
def test_optimizer_failure_recovers_a_true_nearcritical_water_spinodal(
    backend, monkeypatch
):
    fluid = jp.Fluid("Water", backend=backend)

    class FailedOptimizer:
        success = False
        message = "inequality constraints incompatible"

        def __init__(self, *args, **kwargs):
            pass

        def solve(self, initial):
            return np.atleast_1d(initial)

    monkeypatch.setattr(functions.psv, "OptimizationSolver", FailedOptimizer)
    state = functions.compute_spinodal_point(
        float(fluid.critical_point.T) - 1.1,
        fluid,
        "liquid",
        N_trial=40,
        tolerance=1e-12,
    )
    assert state.valid
    assert abs(float(state.K_T / state.p)) < 1e-4
