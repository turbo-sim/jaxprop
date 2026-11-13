"""Setup guarantees and projection pruning in the native perfect-gas path."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp


@pytest.fixture(scope="module")
def gas():
    return jxp.Fluid(
        "CO2", method="perfect_gas", model_options={"T_ref": 600, "p_ref": 100000}
    )


@pytest.mark.parametrize("errors", ["mask", "clip"])
def test_temperature_projection_prunes_entropy_energy_and_transport_work(gas, errors):
    evaluate = jax.jit(
        lambda h: gas.get_state(jxp.HmassP_INPUTS, h, 8e6, errors=errors).T
    )
    h = jnp.asarray(700e3)
    program = evaluate.lower(h).compile().as_text()
    assert not re.search(r"\b(?:log|exp|power|sqrt)\(", program)
    assert "callback" not in program.lower()
    cp = gas.constants.gamma * gas.constants.R / (gas.constants.gamma - 1)
    np.testing.assert_allclose(
        evaluate(h), gas.constants.T_ref + (h - gas.constants.h_ref) / cp, rtol=1e-13
    )


@pytest.mark.parametrize(
    "changes, expected_request",
    [
        ({"R": jnp.float32(1e38), "gamma": jnp.float32(1.1)}, "gamma"),
        (
            {"R": jnp.float32(np.finfo(np.float32).tiny), "gamma": jnp.float32(1e38)},
            "gamma",
        ),
        ({"R": 1e305, "gamma": 2.0, "h_ref": -1e308}, "h_ref"),
        ({"R": 1e305, "gamma": 2.0}, "s_ref"),
        ({"R": np.nextafter(np.float32(0), np.float32(1))}, "R"),
        ({"h_ref": np.finfo(np.float64).max}, "h_ref"),
        (
            {
                "R": np.float32(1.5e36),
                "gamma": np.float32(2),
                "T_ref": np.float32(1),
                "P_ref": np.float32(1e5),
                "h_ref": np.float32(0),
                "s_ref": np.float32(0),
            },
            "s_ref",
        ),
    ],
)
def test_materialization_establishes_range_guarantees_in_actual_coefficient_dtype(
    gas, changes, expected_request
):
    with pytest.raises(jxp.InvalidConfigurationError) as captured:
        gas.with_constants(**changes)
    assert captured.value.request == expected_request
    assert captured.value.provider == "native"
    assert captured.value.reason


def test_large_finite_pressure_preserves_core_and_local_optional_unavailability(gas):
    pressure = np.finfo(np.float64).max
    state = gas.get_state(jxp.PT_INPUTS, pressure, 500.0, errors="mask")
    assert state.status == 0
    for name in ("T", "p", "rho", "h", "s", "u"):
        assert np.isfinite(state.raw_property(name)), name
    assert np.isnan(state.raw_property("K_s"))
    assert not state.property_valid("K_s")
    assert np.isfinite(state.kappa_s)


def test_dynamic_valid_parameter_views_keep_native_temperature_derivatives(gas):
    def evaluate(gamma):
        import equinox as eqx

        view = eqx.tree_at(lambda model: model.constants.gamma, gas, gamma)
        return view.get_state(jxp.HmassP_INPUTS, 700e3, 8e6, errors="mask").T

    gamma = float(gas.constants.gamma)
    expected = (700e3 - float(gas.constants.h_ref)) / (
        float(gas.constants.R) * gamma**2
    )
    np.testing.assert_allclose(jax.jit(jax.grad(evaluate))(gamma), expected, rtol=1e-12)


def test_inverse_independent_coordinates_are_preserved_exactly_without_clipping(gas):
    reference = gas.get_state(jxp.PT_INPUTS, 2e5, 650.0)
    for pair, first, second, names in (
        (jxp.HmassP_INPUTS, reference.h, reference.p, ("h", "p")),
        (jxp.HmassSmass_INPUTS, reference.h, reference.s, ("h", "s")),
        (jxp.PSmass_INPUTS, reference.p, reference.s, ("p", "s")),
        (jxp.DmassHmass_INPUTS, reference.rho, reference.h, ("rho", "h")),
    ):
        actual = gas.get_state(pair, first, second, errors="mask")
        np.testing.assert_array_equal(actual.raw_property(names[0]), first)
        np.testing.assert_array_equal(actual.raw_property(names[1]), second)


def test_default_model_precision_is_not_silently_downcast_by_float32_inputs(gas):
    state = gas.get_state(
        jxp.PT_INPUTS, jnp.float32(1e5), jnp.float32(500), errors="mask"
    )
    assert all(
        not getattr(value, "weak_type", False) for value in vars(gas.constants).values()
    )
    assert state.T.dtype == state.h.dtype == state.s.dtype == jnp.float64


def test_explicit_float32_materialized_model_retains_its_precision(gas):
    changes = {
        name: jnp.asarray(value, dtype=jnp.float32)
        for name, value in vars(gas.constants).items()
    }
    model = gas.with_constants(**changes)
    state = model.get_state(
        jxp.PT_INPUTS, jnp.float32(1e5), jnp.float32(500), errors="mask"
    )
    assert state.status == 0
    assert all(value.dtype == jnp.float32 for value in state.to_dict().values())


def test_materializing_weak_parameter_view_strengthens_it_once(gas):
    import equinox as eqx

    view = eqx.tree_at(lambda model: model.constants.gamma, gas, 1.4)
    materialized = view.materialize()
    assert not materialized.constants.gamma.weak_type
    assert materialized.materialize().constants is materialized.constants
    assert materialized.reference_id == gas.with_constants(gamma=1.4).reference_id


@pytest.mark.parametrize("invalid", [False, True])
def test_gamma_gradient_avoids_inactive_range_overflow(gas, invalid):
    import equinox as eqx

    model = gas if invalid else gas.with_constants(R=0.1, gamma=1.4)

    def enthalpy(gamma):
        view = eqx.tree_at(lambda current: current.constants.gamma, model, gamma)
        state = (
            view.get_state(jxp.PSmass_INPUTS, 1e5, 1e308, errors="mask")
            if invalid
            else view.get_state(jxp.PT_INPUTS, 1e5, 500.0, errors="mask")
        )
        return jnp.where(state.valid, state.h, 0.0)

    gamma = model.constants.gamma
    derivative = jax.jit(jax.grad(enthalpy))(gamma)
    assert np.isfinite(derivative)
    expected = (
        0.0
        if invalid
        else -(500.0 - float(model.constants.T_ref))
        * float(model.constants.R)
        / (float(gamma) - 1) ** 2
    )
    np.testing.assert_allclose(derivative, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pair", [jxp.PT_INPUTS, jxp.PSmass_INPUTS])
@pytest.mark.parametrize("name", ["R", "gamma", "T_ref", "h_ref"])
def test_full_state_invalid_rows_have_zero_model_parameter_gradient(gas, pair, name):
    import equinox as eqx

    def total(value):
        view = eqx.tree_at(lambda model: getattr(model.constants, name), gas, value)
        state = view.get_state(pair, 1e5, 1e308, errors="mask")
        return sum(
            jnp.nan_to_num(prop, nan=0.0, posinf=0.0, neginf=0.0)
            for prop in state.to_dict().values()
        )

    value = jnp.asarray(getattr(gas.constants, name), dtype=jnp.float64)
    assert jax.jit(jax.grad(total))(value) == 0


def test_stable_sutherland_algebra_retains_reference_accuracy_and_high_T_range(gas):
    from jaxprop.perfect_gas.perfect_gas import conductivity_from_T, viscosity_from_T

    temperature = jnp.asarray([1.0, 300.0, 600.0, 1500.0])
    for function, reference, coefficient in (
        (viscosity_from_T, gas.constants.mu_ref, gas.constants.S_mu),
        (conductivity_from_T, gas.constants.k_ref, gas.constants.S_k),
    ):
        original = (
            reference
            * (temperature / gas.constants.T_ref) ** 1.5
            * (gas.constants.T_ref + coefficient)
            / (temperature + coefficient)
        )
        np.testing.assert_allclose(
            function(temperature, gas.constants), original, rtol=5e-15
        )
        assert np.isfinite(function(jnp.asarray(1e305), gas.constants))


@pytest.mark.parametrize("errors", ["mask", "clip"])
@pytest.mark.parametrize("pattern", ["valid", "mixed", "invalid", "empty"])
def test_materialized_full_batch_matches_scalar_vmap_for_every_property(
    gas, errors, pattern
):
    temperatures = np.full((2, 64), 500.0)
    if pattern == "mixed":
        temperatures[:, 1:4] = [-10.0, np.nan, 1e308]
    elif pattern == "invalid":
        temperatures.fill(np.nan)
    elif pattern == "empty":
        temperatures = temperatures[:0]
    temperatures = jnp.asarray(temperatures)

    def evaluate(T):
        return gas.get_state(jxp.PT_INPUTS, 1e5, T, errors=errors)

    actual = jax.jit(evaluate)(temperatures)
    expected = jax.jit(jax.vmap(jax.vmap(evaluate)))(temperatures)
    inner_batches = jax.jit(jax.vmap(evaluate))(temperatures)
    for current in (actual, inner_batches):
        np.testing.assert_array_equal(current.status, expected.status)
        np.testing.assert_array_equal(current.clipped, expected.clipped)
        for name, value in current.to_dict().items():
            np.testing.assert_allclose(
                value,
                expected.raw_property(name),
                rtol=2e-13,
                atol=1e-12,
                equal_nan=True,
                err_msg=name,
            )

    def total(T):
        return sum(
            jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).sum()
            for value in evaluate(T).to_dict().values()
        )

    actual_gradient = jax.jit(jax.grad(total))(temperatures)
    expected_gradient = jax.jit(jax.vmap(jax.vmap(jax.grad(total))))(temperatures)
    np.testing.assert_allclose(
        actual_gradient, expected_gradient, rtol=2e-12, atol=1e-12
    )


@pytest.mark.parametrize("pair", [jxp.PT_INPUTS, jxp.HmassP_INPUTS])
@pytest.mark.parametrize("errors", ["mask", "clip"])
def test_successful_batch_preserves_tolerated_floors_and_clip_density_coupling(
    gas, pair, errors
):
    temperatures = jnp.asarray(
        np.resize([1 - 0.4e-12, 1.0, 1 + 0.4e-12, 500.0], (2, 64))
    )
    if errors == "clip":
        temperatures = temperatures.at[:, -1].set(1e20)
    pressures = jnp.full_like(temperatures, 0.1 * (1 - 0.4e-12))
    cp = gas.constants.gamma * gas.constants.R / (gas.constants.gamma - 1)
    enthalpies = gas.constants.h_ref + cp * (temperatures - gas.constants.T_ref)
    first, second = (
        (pressures, temperatures) if pair == jxp.PT_INPUTS else (enthalpies, pressures)
    )

    def evaluate(a, b):
        return gas.get_state(pair, a, b, errors=errors)

    actual = jax.jit(evaluate)(first, second)
    expected = jax.jit(jax.vmap(jax.vmap(evaluate)))(first, second)
    np.testing.assert_array_equal(actual.status, 0)
    np.testing.assert_array_equal(actual.status, expected.status)
    np.testing.assert_array_equal(actual.clipped, expected.clipped)
    for name, value in actual.to_dict().items():
        np.testing.assert_allclose(
            value, expected.raw_property(name), rtol=3e-13, atol=1e-12, equal_nan=True
        )


@pytest.mark.parametrize("pair", [jxp.PT_INPUTS, jxp.HmassP_INPUTS])
@pytest.mark.parametrize("errors", ["mask", "clip"])
@pytest.mark.parametrize("name", ["R", "gamma", "T_ref", "h_ref"])
def test_vmapped_inner_batch_invalid_rows_preserve_model_parameter_gradients(
    gas, pair, errors, name
):
    import equinox as eqx

    cp = gas.constants.gamma * gas.constants.R / (gas.constants.gamma - 1)
    h = gas.constants.h_ref + cp * (500.0 - gas.constants.T_ref)
    primary = jnp.full((2, 64), 500.0 if pair == jxp.PT_INPUTS else h)
    primary = primary.at[1, :].set(
        1e308 if pair == jxp.PT_INPUTS else np.finfo(np.float64).max
    )
    primary = primary.at[0, 0].set(jnp.nan)
    pressure = jnp.full_like(primary, 1e5)
    first, second = (
        (pressure, primary) if pair == jxp.PT_INPUTS else (primary, pressure)
    )

    def loss(value, *, scalar):
        view = eqx.tree_at(lambda model: getattr(model.constants, name), gas, value)

        def evaluate(a, b):
            return view.get_state(pair, a, b, errors=errors)

        mapped = jax.vmap(jax.vmap(evaluate)) if scalar else jax.vmap(evaluate)
        state = mapped(first, second)
        return sum(
            jnp.nan_to_num(prop, nan=0.0, posinf=0.0, neginf=0.0).sum()
            for prop in state.to_dict().values()
        )

    value = jnp.asarray(getattr(gas.constants, name), dtype=jnp.float64)
    actual = jax.jit(jax.grad(lambda value: loss(value, scalar=False)))(value)
    expected = jax.jit(jax.grad(lambda value: loss(value, scalar=True)))(value)
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-9)
