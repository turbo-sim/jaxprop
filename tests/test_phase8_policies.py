"""Bicubic public-domain and failure policies, independent of solver internals."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import pytest

import jaxprop as jp
from jaxprop.bicubic.runtime import prepare_runtime


@pytest.fixture(scope="module")
def fluid(tmp_path_factory):
    return jp.Fluid(
        "CO2",
        method="bicubic",
        identifier="phase8-working-fluid",
        table_options=dict(
            h_min=600e3,
            h_max=850e3,
            p_min=5e6,
            p_max=12e6,
            N_h=8,
            N_p=8,
            N_p_sat=4,
            coarse_step=1,
            table_dir=str(tmp_path_factory.mktemp("phase8_policies")),
        ),
    )


def test_capabilities_describe_strict_natural_domain_and_policies(fluid):
    assert fluid.capabilities.error_modes == {"raise", "mask", "clip"}
    assert not fluid.capabilities.domain.extrapolation
    assert fluid.capabilities.domain.bounds["enthalpy"] == (600e3, 850e3)
    assert fluid.capabilities.domain.bounds["pressure"] == (5e6, 12e6)
    assert "saturation" not in fluid.capabilities.phase_operations
    assert len(jax.tree_util.tree_leaves(fluid)) < 100


@pytest.mark.parametrize("policy", ["raise", "mask", "clip"])
def test_successful_broadcast_states_preserve_identity(fluid, policy):
    h, p = jnp.asarray([[650e3], [750e3]]), jnp.asarray([[6e6, 8e6, 10e6]])
    state = fluid.get_state(jp.HmassP_INPUTS, h, p, errors=policy)
    assert state.shape == (2, 3) and state.error_policy == policy
    assert (
        state.identifier == fluid.identifier
        and state.reference_id == fluid.reference_id
    )
    assert state.available_properties == fluid.capabilities.properties
    np.testing.assert_array_equal(state.status, 0)
    np.testing.assert_array_equal(state.clipped, False)
    np.testing.assert_array_equal(state.h, np.broadcast_to(h, (2, 3)))
    np.testing.assert_array_equal(state.p, np.broadcast_to(p, (2, 3)))
    for name in state.available_properties:
        assert state.raw_property(name).shape == (2, 3)


@pytest.mark.parametrize(
    "h,p,code,error",
    [
        (590e3, 8e6, 2, jp.OutOfDomainError),
        (860e3, 8e6, 2, jp.OutOfDomainError),
        (700e3, 4e6, 2, jp.OutOfDomainError),
        (700e3, 13e6, 2, jp.OutOfDomainError),
        (np.nan, 8e6, 1, jp.InvalidStateError),
        (700e3, np.inf, 1, jp.InvalidStateError),
        (700e3, -1.0, 1, jp.InvalidStateError),
    ],
)
def test_eager_raise_and_mask_are_typed_indexed_and_elementwise(
    fluid, h, p, code, error
):
    with pytest.raises(error) as captured:
        fluid.get_state(jp.HmassP_INPUTS, [700e3, h], [8e6, p])
    assert captured.value.index == (1,) and captured.value.provider == "bicubic"
    assert captured.value.request == "HmassP_INPUTS"
    state = fluid.get_state(jp.HmassP_INPUTS, [700e3, h], [8e6, p], errors="mask")
    np.testing.assert_array_equal(state.status, [0, code])
    for name in state.available_properties:
        assert np.isnan(state.raw_property(name)[1])
    assert np.isfinite(state.T[0])


@pytest.mark.parametrize(
    "h,p", [(590e3, 8e6), (860e3, 8e6), (700e3, -1), (700e3, 13e6)]
)
def test_clip_returns_the_actual_projected_full_state(fluid, h, p):
    projected_h, projected_p = np.clip(h, 600e3, 850e3), np.clip(p, 5e6, 12e6)
    expected = fluid.get_state(jp.HmassP_INPUTS, projected_h, projected_p)
    actual = fluid.get_state(jp.HmassP_INPUTS, h, p, errors="clip")
    assert actual.status == 0 and actual.clipped
    for name in expected.available_properties:
        np.testing.assert_array_equal(
            actual.raw_property(name), expected.raw_property(name)
        )
    np.testing.assert_array_equal(actual.h, projected_h)
    np.testing.assert_array_equal(actual.p, projected_p)


def test_clip_preserves_unrecoverable_nonfinite_failures(fluid):
    actual = fluid.get_state(jp.HmassP_INPUTS, [590e3, np.nan], 8e6, errors="clip")
    np.testing.assert_array_equal(actual.status, [0, 1])
    np.testing.assert_array_equal(actual.clipped, [True, False])
    assert np.isnan(actual.raw_property("T")[1])


@pytest.mark.parametrize("policy", ["mask", "clip"])
def test_transformed_full_state_empty_and_failed_gradients(fluid, policy):
    evaluate = lambda h, p: fluid.get_state(jp.HmassP_INPUTS, h, p, errors=policy)
    h, p = jnp.asarray([700e3, np.nan, 590e3]), jnp.asarray([8e6, 8e6, 8e6])
    vector, batch = jax.jit(jax.vmap(evaluate))(h, p), jax.jit(evaluate)(h, p)
    for name in batch.available_properties:
        np.testing.assert_array_equal(
            vector.raw_property(name), batch.raw_property(name)
        )
    np.testing.assert_array_equal(vector.status, batch.status)
    np.testing.assert_array_equal(vector.clipped, batch.clipped)
    empty = jax.jit(evaluate)(jnp.empty((0, 1)), jnp.ones((1, 3)) * 8e6)
    assert empty.shape == (0, 3)
    gradient = jax.jit(
        jax.grad(lambda values: jnp.nan_to_num(evaluate(values, p).T).sum())
    )(h)
    assert np.all(np.isfinite(gradient)) and gradient[1] == 0 and gradient[2] == 0
    program = jax.jit(evaluate).lower(h, p).compile().as_text()
    assert "callback" not in program.lower()


def test_transformed_raise_is_rejected_even_for_closed_inputs(fluid):
    with pytest.raises(jp.UnsupportedExecutionModeError, match="explicit"):
        jax.jit(lambda h: fluid.get_state(jp.HmassP_INPUTS, h, 8e6))(700e3)
    with pytest.raises(jp.UnsupportedExecutionModeError):
        jax.jit(lambda: fluid.get_state(jp.HmassP_INPUTS, 700e3, 8e6))()


def test_invalid_phase_cells_are_never_silently_extrapolated(fluid):
    bad = eqx.tree_at(
        lambda model: model._cell_validity, fluid, jnp.zeros_like(fluid._cell_validity)
    )
    with pytest.raises(jp.OutOfDomainError, match="trusted"):
        bad.get_state(jp.HmassP_INPUTS, 700e3, 8e6)
    for policy in ("mask", "clip"):
        state = bad.get_state(jp.HmassP_INPUTS, 700e3, 8e6, errors=policy)
        assert (
            state.status == 2
            and not state.clipped
            and np.isnan(state.raw_property("T"))
        )


def test_no_saturation_payload_is_explicitly_unsupported(fluid):
    with pytest.raises(jp.UnsupportedOperationError, match="no saturation endpoint"):
        fluid.get_state_saturation(jp.PQ_INPUTS, 6e6, 0.0)


def test_unknown_requests_and_disabled_properties_remain_explicit(fluid):
    with pytest.raises(jp.UnsupportedInputPairError):
        fluid.get_state(jp.QT_INPUTS, 0.5, 300.0, errors="clip")
    with pytest.raises(jp.UnsupportedPropertyError):
        fluid.get_state(jp.HmassP_INPUTS, 700e3, 8e6).raw_property(
            "pressure_saturation"
        )


def test_prepared_phase_cells_reject_mixed_markers():
    phase = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    table = {
        "temperature": {"coeffs": np.zeros((2, 2, 16))},
        "is_two_phase": {"value": phase},
    }
    _, _, cells = prepare_runtime(table, frozenset(), "equilibrium")
    np.testing.assert_array_equal(cells, [[True, False], [False, False]])


@pytest.mark.parametrize("pressure,expected", [(2e6, False), (6e6, True)])
def test_coarse_subcritical_liquid_vapor_bridge_is_not_a_trusted_cell(
    pressure, expected
):
    table = {
        "temperature": {"coeffs": np.zeros((1, 1, 16))},
        "is_two_phase": {"value": np.zeros((2, 2))},
        "quality_mass": {"value": np.array([[0.0, 0.0], [1.0, 1.0]])},
        "pressure": {"value": np.full((2, 2), pressure)},
    }
    _, _, cells = prepare_runtime(
        table, frozenset(), "equilibrium", critical_pressure=4e6
    )
    assert bool(cells[0, 0]) == expected


def test_public_inverse_budget_is_wired_and_does_not_change_generation_identity(fluid):
    options = dict(fluid.configuration.table_options)
    options["inverse_max_steps"] = 0
    limited = jp.FluidBicubic(fluid.name, backend=fluid.backend, **options)
    assert limited.metadata == fluid.metadata
    assert limited.table_name == fluid.table_name
    reference = fluid.get_state(jp.HmassP_INPUTS, 711234.0, 8.1e6)
    with pytest.raises(jp.ConvergenceError) as captured:
        limited.get_state(jp.PT_INPUTS, 8.1e6, reference.T)
    assert captured.value.index == ()
    for policy in ("mask", "clip"):
        actual = limited.get_state(jp.PT_INPUTS, 8.1e6, reference.T, errors=policy)
        assert actual.status == 3 and not actual.clipped
        assert np.isnan(actual.raw_property("h"))
    derivative = jax.jit(
        jax.grad(
            lambda value: jnp.nan_to_num(
                limited.get_state(jp.PT_INPUTS, 8.1e6, value, errors="mask").h
            )
        )
    )(reference.T)
    assert derivative == 0


def test_monotone_inverse_clip_returns_a_consistent_boundary_state(fluid):
    pressure = 8e6
    boundary = fluid.get_state(jp.HmassP_INPUTS, fluid.h_min, pressure)
    target = float(boundary.T) - 50.0
    with pytest.raises(jp.OutOfDomainError):
        fluid.get_state(jp.PT_INPUTS, pressure, target)
    actual = fluid.get_state(jp.PT_INPUTS, pressure, target, errors="clip")
    assert actual.status == 0 and actual.clipped
    for name in actual.available_properties:
        np.testing.assert_allclose(
            actual.raw_property(name),
            boundary.raw_property(name),
            rtol=1e-10,
            atol=1e-8,
        )
    derivative = jax.jit(
        jax.grad(
            lambda value: fluid.get_state(
                jp.PT_INPUTS, pressure, value, errors="clip"
            ).h
        )
    )(target)
    assert derivative == 0


@pytest.mark.parametrize(
    "option,value",
    [
        ("inverse_max_steps", -1),
        ("inverse_max_steps", True),
        ("inverse_tolerance", "small"),
        ("inverse_tolerance", np.nan),
    ],
)
def test_inverse_configuration_is_validated_at_construction(fluid, option, value):
    settings = dict(fluid.configuration.table_options) | {option: value}
    with pytest.raises(jp.InvalidConfigurationError) as captured:
        jp.FluidBicubic(fluid.name, backend=fluid.backend, **settings)
    assert captured.value.request == option


def test_two_phase_pt_does_not_accept_a_tiny_spurious_temperature_slope(fluid):
    temperature = jnp.zeros_like(fluid._table["temperature"]["coeffs"])
    temperature = temperature.at[..., 0].set(300.0).at[..., 1].set(1e-14)
    phase = jnp.zeros_like(fluid._table["is_two_phase"]["coeffs"]).at[..., 0].set(1.0)
    view = eqx.tree_at(
        lambda model: (
            model._table["temperature"]["coeffs"],
            model._table["is_two_phase"]["coeffs"],
        ),
        fluid,
        (temperature, phase),
    )
    with pytest.raises(jp.ConvergenceError, match="PT does not identify"):
        view.get_state(jp.PT_INPUTS, 8e6, 300.0)
    state = view.get_state(jp.PT_INPUTS, 8e6, 300.0, errors="mask")
    assert state.status == 3 and np.isnan(state.raw_property("h"))
    derivative = jax.jit(
        jax.grad(
            lambda value: jnp.nan_to_num(
                view.get_state(jp.PT_INPUTS, 8e6, value, errors="mask").h
            )
        )
    )(300.0)
    assert derivative == 0


def test_lean_evaluator_pickle_preserves_complete_public_state_and_inspection(fluid):
    restored = pickle.loads(pickle.dumps(fluid))
    expected = fluid.get_state(jp.HmassP_INPUTS, 700e3, 8e6)
    actual = restored.get_state(jp.HmassP_INPUTS, 700e3, 8e6)
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(fluid)
    assert restored.metadata == fluid.metadata
    for name in expected.available_properties:
        np.testing.assert_array_equal(
            actual.raw_property(name), expected.raw_property(name)
        )
    for name in jp.PROPERTIES_CANONICAL:
        for key in fluid.table[name]:
            np.testing.assert_array_equal(
                restored.table[name][key], fluid.table[name][key]
            )


def test_prepared_inspection_identity_cannot_change_after_jit_caching(fluid):
    with pytest.raises(AttributeError, match="immutable"):
        fluid._inspection._identity = "changed"


def test_runtime_reuses_validated_finite_coefficient_arrays(fluid):
    for name in ("temperature", "density", "entropy"):
        assert fluid._table[name]["coeffs"] is fluid._inspection.table[name]["coeffs"]
