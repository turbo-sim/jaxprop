"""Whole consumers preserve explicit state/status semantics under JAX."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jp


@pytest.fixture(scope="module", params=["perfect_gas", "bicubic"])
def native_model(request, tmp_path_factory):
    method = request.param
    options = (
        {"model_options": {"T_ref": 600.0, "p_ref": 1e5}}
        if method == "perfect_gas"
        else {
            "table_options": {
                "h_min": 600e3,
                "h_max": 850e3,
                "p_min": 5e6,
                "p_max": 12e6,
                "N_h": 6,
                "N_p": 6,
                "N_p_sat": 4,
                "coarse_step": 1,
                "table_dir": str(tmp_path_factory.mktemp("phase11_consumer")),
            }
        }
    )
    return jp.Fluid("CO2", method=method, identifier="consumer-stream", **options)


def consumer_inputs(fluid):
    if fluid.provider == "bicubic":
        # Independent fields share exact source identities at grid nodes;
        # interior interpolation has its separate resolution-dependent budget.
        return (
            jp.HmassP_INPUTS,
            jnp.asarray([fluid.h_vals[1], fluid.h_vals[2], fluid.h_vals[3]]),
            jnp.asarray(
                [jnp.exp(fluid.logP_vals[1]), jnp.exp(fluid.logP_vals[2]), jnp.nan]
            ),
        )
    return (
        jp.PT_INPUTS,
        jnp.asarray([1e5, 2e5, jnp.nan]),
        jnp.asarray([500.0, 650.0, 700.0]),
    )


def test_compiled_complete_state_consumer_reuses_trace_and_preserves_status(
    native_model,
):
    pair, first, second = consumer_inputs(native_model)
    traces = []

    def consumer(model, a, b):
        traces.append(None)
        state = model.get_state(pair, a, b, errors="mask")
        # Materialize complete properties/status, and carry status with the
        # application result rather than synchronizing inside the executable.
        return state, (state.h - state.u), state.status

    evaluate = jax.jit(consumer)
    state, pv, status = evaluate(native_model, first, second)
    state_again, pv_again, status_again = evaluate(native_model, first * 1.001, second)
    assert len(traces) == 1
    assert isinstance(state, jp.FluidState)
    assert state.identifier == "consumer-stream"
    np.testing.assert_array_equal(status[:2], 0)
    assert int(status[-1]) != 0
    np.testing.assert_array_equal(status, status_again)
    np.testing.assert_allclose(pv[:2], (state.p / state.rho)[:2], rtol=2e-12)
    assert np.isnan(pv[-1]) and np.isnan(pv_again[-1])
    assert not np.array_equal(
        np.asarray(state.T), np.asarray(state_again.T), equal_nan=True
    ) or not np.array_equal(
        np.asarray(state.p), np.asarray(state_again.p), equal_nan=True
    )
    executable = evaluate.lower(native_model, first, second).compile().as_text().lower()
    assert "callback" not in executable


def test_status_preserving_projection_and_masked_gradients(native_model):
    pair, first, second = consumer_inputs(native_model)

    def projection(a):
        state = native_model.get_state(pair, a, second, errors="mask")
        return state.h, state.status

    h, status = jax.jit(projection)(first)
    expected = native_model.get_state(pair, first, second, errors="mask")
    np.testing.assert_allclose(h, expected.h, equal_nan=True)
    np.testing.assert_array_equal(status, expected.status)
    derivative = jax.jit(jax.grad(lambda a: jnp.nansum(projection(a)[0])))(first)
    assert np.all(np.isfinite(derivative))
    assert float(derivative[-1]) == 0.0


def test_model_updates_are_dynamic_and_previous_states_remain_independent(native_model):
    pair, first, second = consumer_inputs(native_model)
    evaluate = jax.jit(
        lambda model: model.get_state(pair, first[:2], second[:2], errors="mask")
    )
    original = evaluate(native_model)
    snapshot = {
        name: np.array(original.raw_property(name))
        for name in original.available_properties
    }
    if native_model.provider == "bicubic":
        changed = eqx.tree_at(
            lambda model: model._table["temperature"]["coeffs"],
            native_model,
            native_model._table["temperature"]["coeffs"] * 1.001,
        )
    else:
        changed = eqx.tree_at(
            lambda model: model.constants.gamma,
            native_model,
            native_model.constants.gamma * 1.001,
        )
    updated = evaluate(changed)
    assert not np.array_equal(
        np.asarray(original.h), np.asarray(updated.h)
    ) or not np.array_equal(np.asarray(original.T), np.asarray(updated.T))
    for name, values in snapshot.items():
        np.testing.assert_array_equal(original.raw_property(name), values)


def test_callback_consumer_returns_complete_state_and_explicit_status():
    fluid = jp.Fluid("CO2", method="coolprop_jax", identifier="callback-consumer")

    @jax.jit
    def consumer(pressure):
        state = fluid.get_state(jp.PT_INPUTS, pressure, 500.0, errors="mask")
        return state, state.h - state.u, state.status

    state, pv, status = consumer(jnp.asarray([1e6, -1.0]))
    assert state.identifier == fluid.identifier
    assert int(status[0]) == 0 and int(status[1]) != 0
    np.testing.assert_allclose(pv[0], state.p[0] / state.rho[0], rtol=1e-12)
    assert np.isnan(pv[1])
