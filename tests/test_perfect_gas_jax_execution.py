"""Native perfect-gas calls compile for shapes/types, not changing values."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp
from jaxprop.perfect_gas import request_policy as pg_requests


@pytest.fixture
def fluid(request):
    # Distinct static identity isolates each compilation-cache assertion from
    # other tests without clearing process-wide JAX/Equinox caches.
    return jxp.FluidPerfectGas(
        "CO2", T_ref=600.0, p_ref=1e5, identifier=request.node.nodeid
    )


def test_varying_python_scalar_values_reuse_one_compilation(fluid, monkeypatch):
    traces = []
    original = pg_requests.resolve_pair

    def resolve_pair(
        input_type, first, second, constants, *, errors, return_success=False
    ):
        traces.append(True)
        return original(
            input_type,
            first,
            second,
            constants,
            errors=errors,
            return_success=return_success,
        )

    monkeypatch.setattr(pg_requests, "resolve_pair", resolve_pair)
    cp = float(fluid.constants.gamma * fluid.constants.R / (fluid.constants.gamma - 1))
    for h, p in ((700e3, 1e5), (710e3, 2e5), (720e3, 3e5)):
        state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
        expected = fluid.constants.T_ref + (h - fluid.constants.h_ref) / cp
        np.testing.assert_allclose(state.T, expected, rtol=1e-13)
        np.testing.assert_array_equal(state.p, p)
        assert state.identifier == fluid.identifier
    assert len(traces) == 1


def test_list_inputs_follow_the_same_numeric_state_path(fluid):
    h, p = [700e3, 710e3], [1e5, 2e5]
    from_lists = fluid.get_state(jxp.HmassP_INPUTS, h, p)
    from_arrays = fluid.get_state(jxp.HmassP_INPUTS, jnp.asarray(h), jnp.asarray(p))
    assert from_lists.shape == (2,)
    for name in from_arrays.available_properties:
        np.testing.assert_allclose(
            from_lists.raw_property(name),
            from_arrays.raw_property(name),
            rtol=1e-13,
            atol=0,
        )
        np.testing.assert_array_equal(
            from_lists.property_valid(name), from_arrays.property_valid(name)
        )


def test_broadcast_full_state_survives_outer_jit_and_projection_derivatives(fluid):
    h = jnp.asarray([[700e3], [710e3]])
    p = jnp.asarray([[1e5, 2e5, 3e5]])
    evaluate = lambda enthalpy, pressure: fluid.get_state(
        jxp.HmassP_INPUTS, enthalpy, pressure, errors="mask"
    )
    eager = evaluate(h, p)
    compiled = jax.jit(evaluate)(h, p)
    cp = float(fluid.constants.gamma * fluid.constants.R / (fluid.constants.gamma - 1))
    expected_T = np.broadcast_to(
        float(fluid.constants.T_ref)
        + (np.asarray(h) - float(fluid.constants.h_ref)) / cp,
        (2, 3),
    )
    assert eager.shape == compiled.shape == (2, 3)
    assert compiled.identifier == fluid.identifier
    np.testing.assert_allclose(compiled.T, expected_T, rtol=1e-13)
    np.testing.assert_array_equal(compiled.p, np.broadcast_to(p, (2, 3)))
    for name in eager.available_properties:
        # Outer fusion can reassociate the stable Sutherland expression by
        # three ULPs. Its separate formula check retains a 5e-15 error budget.
        np.testing.assert_array_max_ulp(
            np.asarray(compiled.raw_property(name)),
            np.asarray(eager.raw_property(name)),
            maxulp=4 if name in ("viscosity", "conductivity") else 2,
        )
        np.testing.assert_array_equal(
            compiled.property_valid(name), eager.property_valid(name)
        )
    derivative = jax.jit(
        jax.grad(
            lambda enthalpy: fluid.get_state(
                jxp.HmassP_INPUTS, enthalpy, 1e5, errors="mask"
            ).T
        )
    )(700e3)
    np.testing.assert_allclose(derivative, 1 / cp, rtol=1e-13)


def test_compiled_model_constants_remain_dynamic(fluid):
    traces = []

    @eqx.filter_jit
    def temperature(model, h, p):
        traces.append(True)
        return model.get_state(jxp.HmassP_INPUTS, h, p, errors="mask").T

    h, p = jnp.asarray(700e3), jnp.asarray(1e5)
    original_T = temperature(fluid, h, p)
    gamma = fluid.constants.gamma * 1.02
    modified = eqx.tree_at(lambda model: model.constants.gamma, fluid, gamma)
    modified_T = temperature(modified, h, p)
    expected = fluid.constants.T_ref + (h - fluid.constants.h_ref) / (
        gamma * fluid.constants.R / (gamma - 1)
    )
    np.testing.assert_allclose(modified_T, expected, rtol=1e-13)
    assert not np.isclose(original_T, modified_T)
    assert len(traces) == 1


def test_native_batch_agrees_with_vectorized_calls(fluid):
    h = jnp.asarray([680e3, 700e3, 720e3])
    p = jnp.asarray([1e5, 2e5, 3e5])
    batch = fluid.get_state(jxp.HmassP_INPUTS, h, p)
    vectorized = jax.jit(
        jax.vmap(
            lambda enthalpy, pressure: fluid.get_state(
                jxp.HmassP_INPUTS, enthalpy, pressure, errors="mask"
            )
        )
    )(h, p)
    for name in batch.available_properties:
        # Vectorized/array fusion can differ by a final floating-point bit.
        np.testing.assert_allclose(
            vectorized.raw_property(name),
            batch.raw_property(name),
            rtol=1e-13,
            atol=0,
        )
    np.testing.assert_array_equal(vectorized.status, batch.status)
