"""Selected perfect-gas outputs must prune unrelated thermodynamic work."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp


@pytest.fixture(scope="module")
def gas():
    return jxp.Fluid(
        "CO2", method="perfect_gas", model_options={"T_ref": 600.0, "p_ref": 1e5}
    )


@pytest.mark.parametrize("pair", [jxp.PT_INPUTS, jxp.HmassP_INPUTS])
@pytest.mark.parametrize("errors", ["mask", "clip"])
def test_temperature_projection_does_not_retain_entropy_or_transport_work(
    gas, pair, errors
):
    cp = float(gas.constants.gamma * gas.constants.R / (gas.constants.gamma - 1))
    value = (
        500.0
        if pair == jxp.PT_INPUTS
        else float(gas.constants.h_ref + cp * (500.0 - gas.constants.T_ref))
    )

    def evaluate(value):
        first, second = (8e6, value) if pair == jxp.PT_INPUTS else (value, 8e6)
        return gas.get_state(pair, first, second, errors=errors).T

    compiled = jax.jit(evaluate)
    actual = compiled(jnp.asarray(value))
    np.testing.assert_allclose(actual, 500.0, rtol=1e-12)
    program = compiled.lower(jnp.asarray(value)).compile().as_text()
    # PT/HP temperature needs no logarithm, transport power, sound-speed root
    # or callback. A status check must not pull those unrequested fields back.
    for operation in ("log", "power", "sqrt"):
        assert re.search(rf"\b{operation}\(", program) is None, operation
    assert "callback" not in program.lower()


@pytest.mark.parametrize("errors", ["mask", "clip"])
def test_projected_derivative_stays_analytic_and_callback_free(gas, errors):
    cp = float(gas.constants.gamma * gas.constants.R / (gas.constants.gamma - 1))
    enthalpy = jnp.asarray(
        float(gas.constants.h_ref + cp * (500.0 - gas.constants.T_ref))
    )
    derivative = jax.jit(
        jax.grad(lambda h: gas.get_state(jxp.HmassP_INPUTS, h, 8e6, errors=errors).T)
    )
    np.testing.assert_allclose(derivative(enthalpy), 1 / cp, rtol=1e-12)
    program = derivative.lower(enthalpy).compile().as_text()
    assert re.search(r"\blog\(", program) is None
    assert "callback" not in program.lower()
