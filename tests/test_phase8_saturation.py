"""Typed endpoint saturation behavior for the retained metastable table modes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jp


@pytest.fixture(scope="module", params=["liquid", "vapor"])
def endpoint_table(request, tmp_path_factory):
    phase = request.param
    h_min, h_max, p_min, p_max = (
        (100e3, 110e3, 1e6, 2e6) if phase == "liquid" else (3e6, 3.1e6, 1e5, 2e5)
    )
    return jp.FluidBicubic(
        "Water",
        "HEOS",
        h_min,
        h_max,
        p_min,
        p_max,
        2,
        2,
        N_p_sat=5,
        coarse_step=1,
        gradient_method="forward",
        metastable_phase=phase,
        identifier="phase8-saturation",
        table_dir=str(tmp_path_factory.mktemp("phase8_sat")),
    )


def quality(fluid):
    return 0.0 if fluid.metastable_phase == "liquid" else 1.0


def test_endpoint_operation_is_typed_and_recovers_stored_source_nodes(endpoint_table):
    fluid = endpoint_table
    assert "saturation" in fluid.capabilities.phase_operations
    pressure = fluid.table["saturation_props"]["pressure"]["value"]
    state = fluid.get_state_saturation(jp.PQ_INPUTS, pressure, quality(fluid))
    assert isinstance(state, jp.FluidState)
    assert (
        state.identifier == fluid.identifier
        and state.reference_id == fluid.reference_id
    )
    np.testing.assert_array_equal(state.status, 0)
    np.testing.assert_array_equal(
        state.p, np.clip(np.asarray(pressure), fluid.p_min, fluid._saturation_p_max)
    )
    np.testing.assert_array_equal(state.Q, quality(fluid))
    for name in ("temperature", "density", "enthalpy", "entropy", "internal_energy"):
        np.testing.assert_allclose(
            state.raw_property(name),
            fluid.table["saturation_props"][name]["value"],
            rtol=2e-12,
            atol=1e-8,
        )


def test_endpoint_quality_and_pressure_limits_are_explicit(endpoint_table):
    fluid = endpoint_table
    with pytest.raises(jp.OutOfDomainError, match="contains only Q") as captured:
        fluid.get_state_saturation(jp.PQ_INPUTS, fluid.p_min, 0.5)
    assert captured.value.index == ()
    with pytest.raises(jp.UnsupportedOperationError, match="PQ_INPUTS"):
        fluid.get_state_saturation(jp.QT_INPUTS, quality(fluid), 300.0)
    with pytest.raises(jp.OutOfDomainError):
        fluid.get_state_saturation(jp.PQ_INPUTS, fluid.p_min * 0.5, quality(fluid))


@pytest.mark.parametrize("policy", ["mask", "clip"])
def test_endpoint_jit_and_per_element_failure_policies(endpoint_table, policy):
    fluid = endpoint_table
    pressure = jnp.asarray([fluid.p_min, fluid.p_min * 0.5, jnp.nan])
    evaluate = jax.jit(
        lambda p: fluid.get_state_saturation(
            jp.PQ_INPUTS, p, quality(fluid), errors=policy
        )
    )
    state = evaluate(pressure)
    np.testing.assert_array_equal(state.status, [0, 0 if policy == "clip" else 2, 1])
    np.testing.assert_array_equal(state.clipped, [False, policy == "clip", False])
    if policy == "clip":
        assert state.p[1] == fluid.p_min
    for name in state.available_properties:
        assert np.isnan(state.raw_property(name)[2])
    assert "callback" not in evaluate.lower(pressure).compile().as_text().lower()


def test_endpoint_pressure_derivative_and_clipping_are_consistent(endpoint_table):
    fluid = endpoint_table
    p = fluid.p_min * 1.05
    temperature = lambda value: fluid.get_state_saturation(
        jp.PQ_INPUTS, value, quality(fluid), errors="clip"
    ).T
    step = p * 1e-5
    difference = (temperature(p + step) - temperature(p - step)) / (2 * step)
    np.testing.assert_allclose(jax.jit(jax.grad(temperature))(p), difference, rtol=1e-7)
    assert jax.jit(jax.grad(temperature))(fluid.p_min * 0.5) == 0
    invalid = lambda value: jnp.nan_to_num(
        fluid.get_state_saturation(jp.PQ_INPUTS, value, quality(fluid), errors="mask").T
    )
    assert jax.jit(jax.grad(invalid))(fluid.p_min * 0.5) == 0


def test_saturation_transformed_raise_and_dynamic_pair_fail_informatively(
    endpoint_table,
):
    fluid = endpoint_table
    with pytest.raises(jp.UnsupportedExecutionModeError):
        jax.jit(lambda p: fluid.get_state_saturation(jp.PQ_INPUTS, p, quality(fluid)))(
            fluid.p_min
        )
    with pytest.raises(jp.UnsupportedOperationError):
        jax.jit(
            lambda pair: fluid.get_state_saturation(
                pair, fluid.p_min, quality(fluid), errors="mask"
            )
        )(jnp.asarray(jp.PQ_INPUTS))
