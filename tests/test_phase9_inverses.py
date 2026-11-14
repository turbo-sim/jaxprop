"""Independent temperature-root and derivative checks for binary mixtures."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxprop.binary_mixture.inverses import solve_temperature


class CaloricConstants(eqx.Module):
    R: jax.Array
    gamma: jax.Array
    T_ref: jax.Array
    P_ref: jax.Array
    h_ref: jax.Array
    s_ref: jax.Array


class CaloricComponent(eqx.Module):
    constants: CaloricConstants


class PerfectGasPair(eqx.Module):
    components: tuple
    mass_fractions: tuple = eqx.field(static=True)
    _all_perfect_gas: bool = eqx.field(static=True, default=True)
    inverse_max_steps: int = eqx.field(static=True, default=64)
    inverse_tolerance: float = eqx.field(static=True, default=1e-10)
    lower: float = eqx.field(static=True, default=200.0)
    upper: float = eqx.field(static=True, default=1000.0)

    def _temperature_bounds(self, pressure):
        return jnp.asarray(self.lower), jnp.asarray(self.upper)


class SmoothBinaryResponse(eqx.Module):
    """Nonlinear caloric model with independently known local derivatives.

    h(T,p)=h0+a(T-300)+b(T²-300²)/2+d(p-1e5), and
    s(T,p)=s0+a log(T/300)+b(T-300)-r log(p/1e5).
    Consequently cp(T)=a+bT and both temperature inverses are increasing.
    """

    a: jax.Array
    b: jax.Array
    d: jax.Array
    r: jax.Array
    h0: jax.Array
    s0: jax.Array
    _all_perfect_gas: bool = eqx.field(static=True, default=False)
    inverse_max_steps: int = eqx.field(static=True, default=64)
    inverse_tolerance: float = eqx.field(static=True, default=1e-10)
    lower: float = eqx.field(static=True, default=200.0)
    upper: float = eqx.field(static=True, default=1000.0)
    monotone: bool = eqx.field(static=True, default=True)
    forced_status: int = eqx.field(static=True, default=0)

    def _temperature_bounds(self, pressure):
        return jnp.asarray(self.lower), jnp.asarray(self.upper)

    def _inverse_monotone(self, pressure, name):
        return self.monotone

    def _inverse_response(self, pressure, temperature, name):
        if name == "enthalpy":
            value = (
                self.h0
                + self.a * (temperature - 300)
                + 0.5 * self.b * (temperature**2 - 300**2)
                + self.d * (pressure - 1e5)
            )
        else:
            value = (
                self.s0
                + self.a * jnp.log(temperature / 300)
                + self.b * (temperature - 300)
                - self.r * jnp.log(pressure / 1e5)
            )
        return value, jnp.asarray(self.forced_status, dtype=jnp.int32)


@pytest.fixture
def smooth():
    return SmoothBinaryResponse(
        *(jnp.asarray(value) for value in (1200.0, 0.6, 1e-3, 287.0, 1e5, 5000.0))
    )


@pytest.fixture
def caloric():
    components = tuple(
        CaloricComponent(CaloricConstants(*(jnp.asarray(value) for value in row)))
        for row in (
            (280.0, 1.4, 315.0, 1.2e5, -2e5, 1100.0),
            (310.0, 1.3, 360.0, 2.0e5, 5e5, 7000.0),
        )
    )
    return PerfectGasPair(components, (0.25, 0.75))


def independent_smooth_value(model, pressure, temperature, name):
    a, b, d, r, h0, s0 = (
        float(value)
        for value in (model.a, model.b, model.d, model.r, model.h0, model.s0)
    )
    if name == "enthalpy":
        return (
            h0
            + a * (temperature - 300)
            + b * (temperature**2 - 300**2) / 2
            + d * (pressure - 1e5)
        )
    return (
        s0
        + a * np.log(temperature / 300)
        + b * (temperature - 300)
        - r * np.log(pressure / 1e5)
    )


def independent_caloric_value(model, pressure, temperature, name):
    total = 0.0
    for component, weight in zip(model.components, model.mass_fractions):
        c = component.constants
        cp = float(c.gamma) * float(c.R) / (float(c.gamma) - 1)
        if name == "enthalpy":
            value = float(c.h_ref) + cp * (temperature - float(c.T_ref))
        else:
            value = (
                float(c.s_ref)
                + cp * np.log(temperature / float(c.T_ref))
                - float(c.R) * np.log(pressure / float(c.P_ref))
            )
        total += weight * value
    return total


@pytest.mark.parametrize("name", ["enthalpy", "entropy"])
@pytest.mark.parametrize("temperature", [200.0, 237.0, 712.0, 1000.0])
def test_bounded_inverse_recovers_independent_nonlinear_response(
    smooth, name, temperature
):
    pressure = 7.3e5
    target = independent_smooth_value(smooth, pressure, temperature, name)
    actual, status, clipped = jax.jit(
        lambda model, p, value: solve_temperature(model, p, value, name)
    )(smooth, pressure, target)
    assert status == 0 and not clipped
    np.testing.assert_allclose(actual, temperature, rtol=2e-10, atol=2e-8)


@pytest.mark.parametrize("name", ["enthalpy", "entropy"])
def test_implicit_derivatives_match_independent_thermal_identities(smooth, name):
    pressure, temperature = 8.5e5, 712.0
    target = independent_smooth_value(smooth, pressure, temperature, name)
    function = lambda p, value: solve_temperature(smooth, p, value, name)[0]
    dp, dtarget = jax.jit(jax.grad(function, argnums=(0, 1)))(pressure, target)
    cp = float(smooth.a) + float(smooth.b) * temperature
    expected_target = 1 / cp if name == "enthalpy" else temperature / cp
    expected_pressure = (
        -float(smooth.d) / cp
        if name == "enthalpy"
        else float(smooth.r) * temperature / (pressure * cp)
    )
    np.testing.assert_allclose(dtarget, expected_target, rtol=2e-8, atol=1e-14)
    np.testing.assert_allclose(dp, expected_pressure, rtol=2e-8, atol=1e-14)

    def varied(a):
        model = eqx.tree_at(lambda item: item.a, smooth, a)
        return solve_temperature(model, pressure, target, name)[0]

    da = jax.jit(jax.grad(varied))(smooth.a)
    expected_a = (
        -(temperature - 300) / cp
        if name == "enthalpy"
        else -temperature * np.log(temperature / 300) / cp
    )
    np.testing.assert_allclose(da, expected_a, rtol=2e-8, atol=1e-12)


@pytest.mark.parametrize("name", ["enthalpy", "entropy"])
@pytest.mark.parametrize("target,expected", [(-1e300, 200.0), (1e300, 1000.0)])
def test_monotone_clipping_reaches_actual_boundary_and_zero_target_derivative(
    smooth, name, target, expected
):
    function = lambda value: solve_temperature(smooth, 8e5, value, name, clip=True)
    temperature, status, clipped = jax.jit(function)(target)
    assert status == 0 and clipped
    np.testing.assert_allclose(temperature, expected, rtol=2e-10, atol=2e-8)
    assert jax.jit(jax.grad(lambda value: function(value)[0]))(target) == 0


@pytest.mark.parametrize("name", ["enthalpy", "entropy"])
def test_domain_failures_have_zero_input_and_model_gradients(smooth, name):
    temperature, status, clipped = solve_temperature(smooth, 8e5, 1e300, name)
    assert status == 2 and not clipped and np.isfinite(temperature)
    function = lambda p, value: solve_temperature(smooth, p, value, name)[0]
    gradients = jax.jit(jax.grad(function, argnums=(0, 1)))(8e5, 1e300)
    np.testing.assert_array_equal(gradients, [0.0, 0.0])
    derivative = jax.jit(
        jax.grad(
            lambda a: solve_temperature(
                eqx.tree_at(lambda item: item.a, smooth, a), 8e5, 1e300, name
            )[0]
        )
    )(smooth.a)
    assert derivative == 0


@pytest.mark.parametrize(
    "pressure,target",
    [(np.nan, 1e5), (np.inf, 1e5), (-1.0, 1e5), (0.0, 1e5), (1e5, np.nan)],
)
def test_nonfinite_and_nonpositive_requests_return_explicit_invalid_status(
    smooth, pressure, target
):
    actual, status, clipped = jax.jit(
        lambda p, value: solve_temperature(smooth, p, value, "enthalpy", clip=True)
    )(pressure, target)
    assert status == 1 and not clipped and np.isfinite(actual)


def test_iteration_budget_is_real_and_failed_ad_is_zero(smooth):
    model = replace(smooth, inverse_max_steps=0)
    target = independent_smooth_value(model, 8e5, 712.0, "enthalpy")
    temperature, status, clipped = solve_temperature(model, 8e5, target, "enthalpy")
    assert status == 3 and not clipped and np.isfinite(temperature)
    derivative = jax.jit(
        jax.grad(lambda value: solve_temperature(model, 8e5, value, "enthalpy")[0])
    )(target)
    assert derivative == 0


def test_constant_response_cannot_be_accepted_as_an_identifiable_inverse(smooth):
    model = eqx.tree_at(
        lambda item: (item.a, item.b), smooth, (jnp.asarray(0.0), jnp.asarray(0.0))
    )
    target = independent_smooth_value(model, 8e5, 600.0, "enthalpy")
    _, status, _ = solve_temperature(model, 8e5, target, "enthalpy")
    assert status == 3


def test_missing_monotonicity_proof_does_not_fabricate_a_clipped_root(smooth):
    model = replace(smooth, monotone=False)
    _, status, clipped = solve_temperature(model, 8e5, 1e300, "enthalpy", clip=True)
    assert status == 3 and not clipped


@pytest.mark.parametrize("name,decrement", [("enthalpy", 1e-8), ("entropy", 1e-10)])
def test_generic_cancelled_lower_reference_uses_local_endpoint_precision(
    smooth, name, decrement
):
    pressure = 1e5
    lower_value = independent_smooth_value(smooth, pressure, smooth.lower, name)
    selector = (
        (lambda model: model.h0) if name == "enthalpy" else (lambda model: model.s0)
    )
    shifted = eqx.tree_at(selector, smooth, selector(smooth) - lower_value)
    boundary = independent_smooth_value(shifted, pressure, shifted.lower, name)
    # The upper response remains large after the lower reference is shifted
    # near zero. Its scale must not excuse a truly below-domain request.
    temperature, status, clipped = solve_temperature(
        shifted, pressure, boundary - decrement, name
    )
    assert status == 2 and not clipped and np.isfinite(temperature)


@pytest.mark.parametrize("limits", [(np.nan, 1000.0), (-1.0, 1000.0), (600.0, 500.0)])
def test_empty_or_invalid_common_temperature_domain_is_explicit(smooth, limits):
    model = replace(smooth, lower=limits[0], upper=limits[1])
    temperature, status, clipped = solve_temperature(model, 8e5, 1e5, "enthalpy")
    assert status == 2 and not clipped and np.isfinite(temperature)


def test_component_failure_status_survives_the_inverse(smooth):
    model = replace(smooth, forced_status=4)
    target = independent_smooth_value(model, 8e5, 712.0, "enthalpy")
    _, status, clipped = solve_temperature(model, 8e5, target, "enthalpy")
    assert status == 4 and not clipped


@pytest.mark.parametrize("name", ["enthalpy", "entropy"])
@pytest.mark.parametrize("temperature", [200.0, 237.0, 712.0, 1000.0])
def test_perfect_gas_pair_uses_consistent_offsets_without_iterations(
    caloric, name, temperature
):
    model = replace(caloric, inverse_max_steps=0)
    pressure = 7.3e5
    target = independent_caloric_value(model, pressure, temperature, name)
    actual, status, clipped = jax.jit(
        lambda model, p, value: solve_temperature(model, p, value, name)
    )(model, pressure, target)
    assert status == 0 and not clipped
    np.testing.assert_allclose(actual, temperature, rtol=2e-12, atol=1e-10)


@pytest.mark.parametrize("name", ["enthalpy", "entropy"])
def test_perfect_gas_pair_derivatives_match_analytical_weighted_heat_capacity(
    caloric, name
):
    pressure, temperature = 8e5, 712.0
    target = independent_caloric_value(caloric, pressure, temperature, name)
    cp = sum(
        weight
        * float(item.constants.gamma * item.constants.R / (item.constants.gamma - 1))
        for item, weight in zip(caloric.components, caloric.mass_fractions)
    )
    gas_constant = sum(
        weight * float(item.constants.R)
        for item, weight in zip(caloric.components, caloric.mass_fractions)
    )
    gradients = jax.jit(
        jax.grad(
            lambda p, value: solve_temperature(caloric, p, value, name)[0],
            argnums=(0, 1),
        )
    )(pressure, target)
    expected = (
        (0.0, 1 / cp)
        if name == "enthalpy"
        else (gas_constant * temperature / (pressure * cp), temperature / cp)
    )
    np.testing.assert_allclose(gradients, expected, rtol=2e-12, atol=1e-12)


@pytest.mark.parametrize("name", ["enthalpy", "entropy"])
def test_perfect_gas_extreme_clipping_and_failed_model_gradients_are_safe(
    caloric, name
):
    temperature, status, clipped = solve_temperature(
        caloric, 8e5, 1e308, name, clip=True
    )
    assert status == 0 and clipped and temperature == 1000.0
    failed = solve_temperature(caloric, 8e5, 1e308, name)
    assert failed[1] == 2 and not failed[2]
    derivative = jax.jit(
        jax.grad(
            lambda gas_constant: solve_temperature(
                eqx.tree_at(
                    lambda model: model.components[0].constants.R, caloric, gas_constant
                ),
                8e5,
                1e308,
                name,
            )[0]
        )
    )(caloric.components[0].constants.R)
    assert derivative == 0


def test_inverse_batches_are_callback_free_and_keep_model_precision(smooth):
    targets = jnp.asarray(
        [
            independent_smooth_value(smooth, 8e5, temperature, "enthalpy")
            for temperature in (237.0, 712.0)
        ]
    )
    function = jax.jit(
        jax.vmap(
            lambda value: solve_temperature(smooth, jnp.float32(8e5), value, "enthalpy")
        )
    )
    actual, status, _ = function(targets)
    np.testing.assert_allclose(actual, [237.0, 712.0], rtol=2e-10, atol=2e-8)
    np.testing.assert_array_equal(status, [0, 0])
    assert actual.dtype == jnp.float64
    assert "callback" not in function.lower(targets).compile().as_text().lower()
