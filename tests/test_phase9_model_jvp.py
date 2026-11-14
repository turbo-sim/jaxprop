"""Forward/reverse model derivatives must survive inactive numerical bounds.

The independent reference below differentiates the ideal-gas core equations,
not the implementation's clipping or representability helpers. The high-pressure
cases deliberately permit optional response fields to become unavailable.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.perfect_gas.perfect_gas import specific_heat
from jaxprop.perfect_gas.request_policy import _temperature_limit

CORE = ("pressure", "temperature", "density", "enthalpy", "entropy", "internal_energy")
SCALES = np.asarray([1e7, 1e3, 1e3, 1e6, 1e3, 1e6])


@pytest.fixture(
    scope="module", params=[np.float64, np.float32], ids=["float64", "float32"]
)
def models(request):
    dtype = request.param
    components = []
    for name in ("Nitrogen", "Oxygen"):
        model = jp.Fluid(name, method="perfect_gas")
        if dtype == np.float32:
            model = model.with_constants(
                **{
                    key: jnp.asarray(value, dtype=jnp.float32)
                    for key, value in vars(model.constants).items()
                }
            )
        components.append(model)
    binary = jp.Fluid(
        "model-jvp",
        method="binary_mixture",
        model_options={"components": tuple(components), "mass_fractions": (0.4, 0.6)},
    )
    return tuple(components), binary, dtype


def constants_as_floats(component):
    return {name: float(value) for name, value in vars(component.constants).items()}


def independent_core_and_derivative(
    components, weights, pressure, temperature, parameter
):
    """Differentiate the fixed-composition ideal-gas equations analytically."""
    constants = [constants_as_floats(component) for component in components]
    T = np.asarray(temperature, dtype=np.float64)
    P = np.asarray(pressure, dtype=np.float64)
    gas_constant = sum(w * c["R"] for w, c in zip(weights, constants))
    density = (P / T) / gas_constant
    h = sum(
        w * (c["h_ref"] + c["gamma"] * c["R"] / (c["gamma"] - 1) * (T - c["T_ref"]))
        for w, c in zip(weights, constants)
    )
    s = sum(
        w
        * (
            c["s_ref"]
            + c["gamma"] * c["R"] / (c["gamma"] - 1) * np.log(T / c["T_ref"])
            - c["R"] * (np.log(P) - np.log(c["P_ref"]))
        )
        for w, c in zip(weights, constants)
    )
    primal = np.stack((P, T, density, h, s, h - gas_constant * T), axis=-1)

    active, weight = constants[0], weights[0]
    if parameter == "R":
        dR = weight
        dcp = weight * active["gamma"] / (active["gamma"] - 1)
    else:
        dR = 0.0
        dcp = -weight * active["R"] / (active["gamma"] - 1) ** 2
    zero = np.zeros_like(P)
    dh = dcp * (T - active["T_ref"])
    ds = dcp * np.log(T / active["T_ref"]) - dR * (np.log(P) - np.log(active["P_ref"]))
    derivative = np.stack(
        (zero, zero, -(density / gas_constant) * dR, dh, ds, dh - dR * T), axis=-1
    )
    return primal / SCALES, derivative / SCALES


@pytest.mark.parametrize("kind", ["pure", "binary"])
@pytest.mark.parametrize("parameter", ["R", "gamma"])
@pytest.mark.parametrize("pressure_case", ["ordinary", "failed_row", "floatmax"])
def test_complete_core_model_jvp_and_vjp_match_ideal_gas_equations(
    models, kind, parameter, pressure_case
):
    components, binary, dtype = models
    model = components[0] if kind == "pure" else binary
    pressure = np.asarray(
        [
            1e6,
            (
                np.nan
                if pressure_case == "failed_row"
                else np.finfo(dtype).max if pressure_case == "floatmax" else 2e6
            ),
        ],
        dtype=dtype,
    )
    temperature = np.asarray([500.0, 500.0], dtype=dtype)
    p, T = jnp.asarray(pressure), jnp.asarray(temperature)
    scales = jnp.asarray(SCALES, dtype=dtype)

    def selector(current):
        component = current if kind == "pure" else current.components[0]
        return getattr(component.constants, parameter)

    def evaluate(value):
        view = eqx.tree_at(selector, model, value)
        state = view.get_state(jp.PT_INPUTS, p, T, errors="mask")
        core = jnp.stack(tuple(state.raw_property(name) for name in CORE), axis=-1)
        return jnp.nan_to_num(core / scales)

    value = selector(model)
    primal, forward = jax.jit(
        lambda value: jax.jvp(evaluate, (value,), (jnp.ones_like(value),))
    )(value)
    reverse = jax.jit(jax.grad(lambda value: evaluate(value).sum()))(value)
    state = model.get_state(jp.PT_INPUTS, p, T, errors="mask")
    expected_status = [0, 1] if pressure_case == "failed_row" else [0, 0]
    np.testing.assert_array_equal(state.status, expected_status)
    # A failed request contributes no model derivative. Use a finite independent
    # reference there, then remove that row explicitly rather than differentiate
    # any NaN expressions in the analytic oracle itself.
    reference_pressure = np.where(np.isfinite(pressure), pressure, 1e6)
    expected_primal, expected_derivative = independent_core_and_derivative(
        components[:1] if kind == "pure" else components,
        (1.0,) if kind == "pure" else (0.4, 0.6),
        reference_pressure,
        temperature,
        parameter,
    )
    if pressure_case == "failed_row":
        expected_primal[1] = 0.0
        expected_derivative[1] = 0.0
    tolerance = 4e-5 if dtype == np.float32 else 3e-12
    absolute = 2e-8 if dtype == np.float32 else 3e-14
    assert np.all(np.isfinite(primal)) and np.all(np.isfinite(forward))
    assert np.isfinite(reverse)
    np.testing.assert_allclose(primal, expected_primal, rtol=tolerance, atol=absolute)
    np.testing.assert_allclose(
        forward, expected_derivative, rtol=tolerance, atol=absolute
    )
    np.testing.assert_allclose(
        reverse, np.sum(expected_derivative), rtol=tolerance, atol=absolute
    )


@pytest.mark.parametrize("parameter", ["R", "gamma"])
def test_representability_ceiling_has_a_finite_model_jvp(models, parameter):
    components, _, dtype = models
    component = components[0]

    def bound(value):
        constants = eqx.tree_at(
            lambda current: getattr(current.constants, parameter), component, value
        ).constants
        cp, cv = specific_heat(constants)
        return _temperature_limit(constants, cp, cv, jnp.finfo(dtype))

    value = getattr(component.constants, parameter)
    primal, forward = jax.jit(
        lambda value: jax.jvp(bound, (value,), (jnp.ones_like(value),))
    )(value)
    reverse = jax.jit(jax.grad(bound))(value)
    assert np.isfinite(primal) and primal > 500.0
    assert np.isfinite(forward) and np.isfinite(reverse)
    # For these moderate source-calibrated coefficients the limiting energy
    # bound is proportional to 1/cp; its derivative remains representable.
    expected = (
        -float(primal) / float(value)
        if parameter == "R"
        else float(primal) / (float(value) * (float(value) - 1))
    )
    tolerance = 4e-5 if dtype == np.float32 else 3e-12
    np.testing.assert_allclose(forward, expected, rtol=tolerance)
    np.testing.assert_allclose(reverse, expected, rtol=tolerance)


@pytest.mark.parametrize("kind", ["pure", "binary"])
@pytest.mark.parametrize("pressure_case", ["failed_row", "floatmax"])
def test_complete_core_input_jvps_and_vjps_remain_finite(models, kind, pressure_case):
    components, binary, dtype = models
    model = components[0] if kind == "pure" else binary
    pressure = np.asarray(
        [1e6, np.nan if pressure_case == "failed_row" else np.finfo(dtype).max],
        dtype=dtype,
    )
    temperature = np.asarray([500.0, 500.0], dtype=dtype)
    p, T = jnp.asarray(pressure), jnp.asarray(temperature)
    scales = jnp.asarray(SCALES, dtype=dtype)

    def evaluate(p, T):
        state = model.get_state(jp.PT_INPUTS, p, T, errors="mask")
        return jnp.nan_to_num(
            jnp.stack(tuple(state.raw_property(name) for name in CORE), axis=-1)
            / scales
        )

    _, forward_p = jax.jit(
        lambda values: jax.jvp(
            lambda p: evaluate(p, T), (values,), (jnp.ones_like(values),)
        )
    )(p)
    _, forward_T = jax.jit(
        lambda values: jax.jvp(
            lambda T: evaluate(p, T), (values,), (jnp.ones_like(values),)
        )
    )(T)
    reverse_p, reverse_T = jax.jit(
        jax.grad(lambda p, T: evaluate(p, T).sum(), argnums=(0, 1))
    )(p, T)

    active = components[:1] if kind == "pure" else components
    weights = (1.0,) if kind == "pure" else (0.4, 0.6)
    constants = [constants_as_floats(component) for component in active]
    R = sum(w * c["R"] for w, c in zip(weights, constants))
    cp = sum(
        w * c["gamma"] * c["R"] / (c["gamma"] - 1) for w, c in zip(weights, constants)
    )
    P = np.asarray(np.where(np.isfinite(pressure), pressure, 1e6), dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)
    rho = (P / temp) / R
    zero, one = np.zeros_like(P), np.ones_like(P)
    expected_p = (
        np.stack((one, zero, one / (R * temp), zero, -R / P, zero), axis=-1) / SCALES
    )
    expected_T = (
        np.stack((zero, one, -rho / temp, one * cp, cp / temp, one * (cp - R)), axis=-1)
        / SCALES
    )
    if pressure_case == "failed_row":
        expected_p[1] = expected_T[1] = 0.0
    tolerance = 4e-5 if dtype == np.float32 else 3e-12
    absolute = 2e-8 if dtype == np.float32 else 3e-14
    for actual, expected in ((forward_p, expected_p), (forward_T, expected_T)):
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=absolute)
    for actual, expected in (
        (reverse_p, expected_p.sum(axis=-1)),
        (reverse_T, expected_T.sum(axis=-1)),
    ):
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=absolute)
