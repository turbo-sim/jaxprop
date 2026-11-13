"""Public bicubic calls must keep state values dynamic at the JIT boundary."""

from copy import deepcopy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp


@pytest.fixture(scope="module")
def bicubic(tmp_path_factory):
    return jxp.FluidBicubic(
        "CO2",
        "HEOS",
        600e3,
        850e3,
        5e6,
        12e6,
        4,
        4,
        N_p_sat=4,
        coarse_step=1,
        table_dir=str(tmp_path_factory.mktemp("bicubic_execution")),
    )


def test_changing_python_float_values_reuses_compilation(bicubic, monkeypatch):
    traces = []
    original = jxp.FluidBicubic._interp_h_p

    def traced(self, enthalpy, pressure):
        traces.append((enthalpy.shape, pressure.shape))
        return original(self, enthalpy, pressure)

    monkeypatch.setattr(jxp.FluidBicubic, "_interp_h_p", traced)
    # Give this test a separate static identity so a prior test's compilation
    # cannot hide tracing of the first call.
    model = deepcopy(bicubic)
    object.__setattr__(model, "identifier", "dynamic-python-floats")
    inputs = [(680e3, 7e6), (700e3, 8e6), (720e3, 9e6)]
    states = [model.get_state(jxp.HmassP_INPUTS, h, p) for h, p in inputs]
    jax.block_until_ready(states)

    assert len(traces) == 1
    for current, (enthalpy, pressure) in zip(states, inputs):
        assert current.shape == ()
        assert float(current.h) == enthalpy
        assert float(current.p) == pressure
    assert len({float(current.T) for current in states}) == len(inputs)


@pytest.mark.parametrize("storage", ["python", "numpy", "jax"])
def test_scalar_input_storage_has_same_values_and_metadata(bicubic, storage):
    convert = {"python": float, "numpy": np.asarray, "jax": jnp.asarray}[storage]
    current = bicubic.get_state(jxp.HmassP_INPUTS, convert(700e3), convert(8e6))
    expected = bicubic.get_state(
        jxp.HmassP_INPUTS, jnp.asarray([700e3]), jnp.asarray([8e6])
    )
    assert current.shape == ()
    assert current.reference_id == expected.reference_id
    assert current.provider == "bicubic"
    for name in current.available_properties:
        np.testing.assert_allclose(
            current.raw_property(name),
            expected.raw_property(name)[0],
            rtol=1e-13,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            current.property_valid(name), expected.property_valid(name)[0]
        )


def test_python_lists_broadcast_without_changing_singleton_axes(bicubic):
    current = bicubic.get_state(jxp.HmassP_INPUTS, [[680e3], [720e3]], [7e6, 8e6, 9e6])
    assert current.shape == (2, 3)
    np.testing.assert_array_equal(current.h, [[680e3] * 3, [720e3] * 3])
    np.testing.assert_array_equal(current.p, [[7e6, 8e6, 9e6]] * 2)
    for row, enthalpy in enumerate((680e3, 720e3)):
        for col, pressure in enumerate((7e6, 8e6, 9e6)):
            expected = bicubic.get_state(jxp.HmassP_INPUTS, enthalpy, pressure)
            np.testing.assert_allclose(current.T[row, col], expected.T, rtol=1e-13)


def test_outer_jit_full_state_and_gradient_use_dynamic_inputs(bicubic):
    compiled = eqx.filter_jit(
        lambda model, enthalpy, pressure: model.get_state(
            jxp.HmassP_INPUTS, enthalpy, pressure
        )
    )
    for enthalpy in (680e3, 720e3):
        result = compiled(bicubic, jnp.asarray(enthalpy), jnp.asarray(8e6))
        assert float(result.h) == enthalpy
        np.testing.assert_allclose(
            result.T,
            bicubic.get_state(jxp.HmassP_INPUTS, enthalpy, 8e6).T,
            rtol=0,
            atol=0,
        )
    derivative = jax.jit(
        jax.grad(lambda enthalpy: bicubic.get_state(jxp.HmassP_INPUTS, enthalpy, 8e6).T)
    )(700e3)
    difference = (
        bicubic.get_state(jxp.HmassP_INPUTS, 700e3 + 1.0, 8e6).T
        - bicubic.get_state(jxp.HmassP_INPUTS, 700e3 - 1.0, 8e6).T
    ) / 2.0
    np.testing.assert_allclose(derivative, difference, rtol=1e-7, atol=1e-11)
