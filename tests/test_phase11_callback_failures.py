"""Failed primal rows cannot contaminate callback reverse-mode derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.coolprop import core_calculations


@pytest.fixture(scope="module", params=("source", "binary"))
def callback(request):
    source = jp.Fluid("CO2", method="coolprop_jax")
    if request.param == "source":
        return source
    return jp.Fluid(
        "callback-binary",
        method="binary_mixture",
        model_options=dict(
            components=(source, jp.Fluid("Nitrogen", method="coolprop_jax")),
            mass_fractions=(0.3, 0.7),
        ),
    )


def all_property_cotangents(state):
    def ones_or_float0(value):
        if np.issubdtype(value.dtype, np.inexact):
            return jnp.ones_like(value)
        return np.zeros(value.shape, dtype=jax.dtypes.float0)

    return jax.tree_util.tree_map(ones_or_float0, state)


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_mixed_failed_primal_rows_have_zero_full_state_and_consumer_vjp(
    callback, compiled
):
    def evaluate(p):
        return callback.get_state(jp.PT_INPUTS, p, 500.0, errors="mask")

    def pullback_all(p):
        state, pullback = jax.vjp(evaluate, p)
        return pullback(all_property_cotangents(state))[0]

    def enthalpy_consumer(p):
        state = evaluate(p)
        return jnp.sum(jnp.where(state.status == 0, state.raw_property("enthalpy"), 0))

    evaluate_call = jax.jit(evaluate) if compiled else evaluate
    transpose_call = jax.jit(pullback_all) if compiled else pullback_all
    gradient_call = jax.grad(enthalpy_consumer)
    if compiled:
        gradient_call = jax.jit(gradient_call)
    pressure = jnp.asarray([1e6, -1.0])
    state, tangent = jax.jvp(evaluate_call, (pressure,), (jnp.ones_like(pressure),))
    np.testing.assert_array_equal(state.status == 0, [True, False])
    expected_adjoint = 0.0
    for leaf in jax.tree_util.tree_leaves(tangent):
        if np.issubdtype(leaf.dtype, np.inexact):
            assert np.isfinite(leaf[0])
            assert leaf[1] == 0
            expected_adjoint += float(leaf[0])
    gradient = transpose_call(pressure)
    assert np.all(np.isfinite(gradient)) and gradient[1] == 0
    np.testing.assert_allclose(gradient[0], expected_adjoint, rtol=2e-12, atol=1e-9)
    selected_gradient = gradient_call(pressure)
    assert np.all(np.isfinite(selected_gradient)) and selected_gradient[1] == 0
    np.testing.assert_allclose(selected_gradient[0], tangent.h[0], rtol=2e-12)


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_undefined_optional_primal_property_has_zero_callback_ad_contribution(
    callback, compiled, monkeypatch
):
    original = core_calculations.compute_properties_1phase

    def no_transport(*args, **kwargs):
        properties = original(*args, **kwargs)
        properties["viscosity"] = np.nan
        return properties

    monkeypatch.setattr(core_calculations, "compute_properties_1phase", no_transport)

    def state_call(p):
        return callback.get_state(jp.PT_INPUTS, p, 500.0, errors="mask")

    def loss(p):
        state = state_call(p)
        return jnp.sum(
            jnp.where(
                state.property_valid("viscosity"), state.raw_property("viscosity"), 0
            )
        )

    state_function = jax.jit(state_call) if compiled else state_call
    gradient_function = jax.grad(loss)
    if compiled:
        gradient_function = jax.jit(gradient_function)
    pressure = jnp.asarray([1e6, -1.0])
    state, tangent = jax.jvp(state_function, (pressure,), (jnp.ones_like(pressure),))
    assert state.status[0] == 0
    assert not np.any(state.property_valid("viscosity"))
    np.testing.assert_array_equal(tangent.raw_property("viscosity"), [0.0, 0.0])
    np.testing.assert_array_equal(gradient_function(pressure), [0.0, 0.0])


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_valid_primal_with_failed_forward_stencil_keeps_derivative_unavailable(
    callback, compiled
):
    source = (
        callback.fluid
        if callback.provider == "coolprop_jax"
        else callback.components[0].fluid
    )
    maximum = source.T_max

    def evaluate(temperature):
        return callback.get_state(jp.PT_INPUTS, 1e6, temperature, errors="mask")

    call = jax.jit(evaluate) if compiled else evaluate
    state, tangent = jax.jvp(call, (jnp.asarray(maximum),), (jnp.asarray(1.0),))
    assert state.status == 0 and np.isfinite(state.h)
    assert np.isnan(tangent.h)
    assert np.isnan(tangent.T)
