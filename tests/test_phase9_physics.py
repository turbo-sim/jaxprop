"""Independent thermodynamic checks for the fixed-composition binary kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from jaxprop.binary_mixture.physics import mix_properties


def gas_state(p, T, *, R=287.0, cp=1005.0, h_ref=1e5, s_ref=1000.0, xp=jnp):
    """Analytic ideal gas with explicit, nonzero enthalpy/entropy references."""
    p, T = xp.broadcast_arrays(xp.asarray(p), xp.asarray(T))
    one = xp.ones_like(T)
    rho, cv = p / (R * T), cp - R
    h = h_ref + cp * (T - 300.0)
    result = {
        "pressure": p,
        "temperature": T,
        "density": rho,
        "enthalpy": h,
        "internal_energy": h - R * T,
        "entropy": s_ref + cp * xp.log(T / 300.0) - R * xp.log(p / 1e5),
        "isobaric_heat_capacity": cp * one,
        "isochoric_heat_capacity": cv * one,
        "heat_capacity_ratio": (cp / cv) * one,
        "isobaric_expansion_coefficient": 1 / T,
        "isothermal_compressibility": 1 / p,
        "isentropic_compressibility": cv / (cp * p),
        "isothermal_bulk_modulus": p,
        "isentropic_bulk_modulus": (cp / cv) * p,
        "speed_of_sound": xp.sqrt(cp / cv * R * T),
        "gruneisen": (R / cv) * one,
        "isothermal_joule_thomson": xp.zeros_like(T),
        "joule_thomson": xp.zeros_like(T),
        "viscosity": 1.8e-5 * (T / 300.0) ** 0.7,
        "conductivity": 0.025 * (T / 300.0) ** 0.8,
    }
    return result


def other_gas(p, T, *, xp=jnp):
    return gas_state(p, T, R=461.5, cp=1850.0, h_ref=2e6, s_ref=5000.0, xp=xp)


def liquid_state(p, T, *, alpha=4e-4, xp=jnp):
    """Exact elastic-liquid EOS derived from an independent Gibbs potential.

    g = -v0/kappa*exp(alpha*(T-T0)-kappa*(p-p0))
        -c0*T*log(T/T0)+c0*T.

    This gives constant cv=c0, alpha and kappa, but pressure/temperature-dependent
    cp. It exercises the nontrivial correction missing in legacy averaged cv.
    """
    p, T = xp.broadcast_arrays(xp.asarray(p), xp.asarray(T))
    c0, kappa = 4000.0, 5e-10
    v = 1e-3 * xp.exp(alpha * (T - 300.0) - kappa * (p - 1e6))
    rho = 1 / v
    cp = c0 + T * alpha**2 * v / kappa
    h = (T * alpha - 1) * v / kappa + c0 * T
    s = alpha * v / kappa + c0 * xp.log(T / 300.0)
    kS = kappa * c0 / cp
    muT = (1 - T * alpha) * v
    one = xp.ones_like(T)
    return {
        "pressure": p,
        "temperature": T,
        "density": rho,
        "enthalpy": h,
        "internal_energy": h - p * v,
        "entropy": s,
        "isobaric_heat_capacity": cp,
        "isochoric_heat_capacity": c0 * one,
        "heat_capacity_ratio": cp / c0,
        "isobaric_expansion_coefficient": alpha * one,
        "isothermal_compressibility": kappa * one,
        "isentropic_compressibility": kS,
        "isothermal_bulk_modulus": one / kappa,
        "isentropic_bulk_modulus": 1 / kS,
        "speed_of_sound": xp.sqrt(1 / (rho * kS)),
        "gruneisen": alpha / (rho * c0 * kappa),
        "isothermal_joule_thomson": muT,
        "joule_thomson": -muT / cp,
        "viscosity": 0.001 * (300.0 / T),
        "conductivity": 0.6 * one,
    }


@pytest.mark.parametrize("w", [0.01, 0.25, 0.75, 0.99])
@pytest.mark.parametrize("p,T", [(1e5, 300.0), (4e6, 750.0)])
def test_ideal_gas_mixture_matches_analytic_complete_model(w, p, T):
    first, second = gas_state(p, T), other_gas(p, T)
    mixture = mix_properties(first, second, (w, 1 - w))
    R, cp = w * 287.0 + (1 - w) * 461.5, w * 1005.0 + (1 - w) * 1850.0
    cv, rho = cp - R, p / (R * T)
    gamma = cp / cv
    expected = {
        "density": rho,
        "isobaric_heat_capacity": cp,
        "isochoric_heat_capacity": cv,
        "heat_capacity_ratio": gamma,
        "isothermal_compressibility": 1 / p,
        "isentropic_compressibility": 1 / (gamma * p),
        "isothermal_bulk_modulus": p,
        "isentropic_bulk_modulus": gamma * p,
        "speed_of_sound": np.sqrt(gamma * R * T),
        "speed_of_sound_pT": np.sqrt(gamma * R * T),
        "isobaric_expansion_coefficient": 1 / T,
        "gruneisen": R / cv,
        "isothermal_joule_thomson": 0.0,
        "joule_thomson": 0.0,
    }
    for name, value in expected.items():
        np.testing.assert_allclose(mixture[name], value, rtol=3e-14, atol=1e-14)
    for name in ("enthalpy", "internal_energy", "entropy"):
        np.testing.assert_allclose(
            mixture[name], w * first[name] + (1 - w) * second[name], rtol=2e-15
        )
    np.testing.assert_allclose(mixture["vol_frac_1"] + mixture["vol_frac_2"], 1.0)
    np.testing.assert_allclose(
        mixture["internal_energy"], mixture["enthalpy"] - p / rho, rtol=3e-15
    )


def extensive_properties(p, T, w):
    """Independent additive state functions, with no use of the mixing kernel."""
    liquid, gas = liquid_state(p, T, xp=np), gas_state(p, T, xp=np)
    return {
        "volume": w / liquid["density"] + (1 - w) / gas["density"],
        **{
            name: w * liquid[name] + (1 - w) * gas[name]
            for name in ("enthalpy", "internal_energy", "entropy")
        },
    }


@pytest.mark.parametrize("w", [0.05, 0.5, 0.95])
@pytest.mark.parametrize("p,T", [(1e6, 320.0), (4e6, 450.0)])
def test_asymmetric_liquid_gas_responses_match_independent_state_derivatives(w, p, T):
    result = mix_properties(liquid_state(p, T), gas_state(p, T), (w, 1 - w))
    center = extensive_properties(p, T, w)
    dT, dp = 0.01, 100.0
    hot, cold = extensive_properties(p, T + dT, w), extensive_properties(p, T - dT, w)
    high, low = extensive_properties(p + dp, T, w), extensive_properties(p - dp, T, w)
    cp = (hot["enthalpy"] - cold["enthalpy"]) / (2 * dT)
    alpha = (hot["volume"] - cold["volume"]) / (2 * dT * center["volume"])
    kT = -(high["volume"] - low["volume"]) / (2 * dp * center["volume"])
    muT = (high["enthalpy"] - low["enthalpy"]) / (2 * dp)
    for name, value in (
        ("isobaric_heat_capacity", cp),
        ("isobaric_expansion_coefficient", alpha),
        ("isothermal_compressibility", kT),
        ("isothermal_joule_thomson", muT),
    ):
        np.testing.assert_allclose(result[name], value, rtol=2e-8, atol=1e-11)

    # Hold total volume fixed independently, allowing the constituent volumes
    # to exchange. This differentiates equilibrium cv from the legacy average.
    def energy_at_constant_volume(temperature):
        pressure = brentq(
            lambda pressure: extensive_properties(pressure, temperature, w)["volume"]
            - center["volume"],
            p * 0.5,
            p * 1.5,
            xtol=1e-7,
        )
        return extensive_properties(pressure, temperature, w)["internal_energy"]

    cv = (energy_at_constant_volume(T + dT) - energy_at_constant_volume(T - dT)) / (
        2 * dT
    )
    np.testing.assert_allclose(result["isochoric_heat_capacity"], cv, rtol=2e-8)

    # Follow the independently defined constant-entropy path and differentiate
    # pressure with respect to density to recover the equilibrium sound speed.
    def rho_at_constant_entropy(pressure):
        temperature = brentq(
            lambda temperature: extensive_properties(pressure, temperature, w)[
                "entropy"
            ]
            - center["entropy"],
            T * 0.9,
            T * 1.1,
            xtol=1e-11,
        )
        return 1 / extensive_properties(pressure, temperature, w)["volume"]

    a2 = 2 * dp / (rho_at_constant_entropy(p + dp) - rho_at_constant_entropy(p - dp))
    np.testing.assert_allclose(result["speed_of_sound"] ** 2, a2, rtol=2e-8)
    np.testing.assert_allclose(
        result["isentropic_bulk_modulus"], result["density"] * a2, rtol=2e-8
    )


@pytest.mark.parametrize("w", [0.01, 0.2, 0.8, 0.99])
def test_named_acoustic_limits_match_legacy_thermal_formula_and_wood(w):
    first, second = liquid_state(2e6, 400.0), gas_state(2e6, 400.0)
    result = mix_properties(first, second, (w, 1 - w))
    rho, vol1, vol2 = result["density"], result["vol_frac_1"], result["vol_frac_2"]
    r1, r2 = first["density"], second["density"]
    a1, a2 = first["speed_of_sound"], second["speed_of_sound"]
    C1, C2 = (
        vol1 * r1 * first["isobaric_heat_capacity"],
        vol2 * r2 * second["isobaric_heat_capacity"],
    )
    frozen_inverse_a2 = rho * (vol1 / (r1 * a1**2) + vol2 / (r2 * a2**2))
    relaxation = (
        rho
        * first["temperature"]
        * C1
        * C2
        / (C1 + C2)
        * (first["gruneisen"] / (r1 * a1**2) - second["gruneisen"] / (r2 * a2**2)) ** 2
    )
    np.testing.assert_allclose(
        result["speed_of_sound_p"], 1 / np.sqrt(frozen_inverse_a2), rtol=3e-15
    )
    np.testing.assert_allclose(
        result["speed_of_sound_pT"],
        1 / np.sqrt(frozen_inverse_a2 + relaxation),
        rtol=5e-15,
    )
    np.testing.assert_array_equal(result["speed_of_sound"], result["speed_of_sound_pT"])
    assert result["speed_of_sound_pT"] <= result["speed_of_sound_p"]


@pytest.mark.parametrize("fractions", [(1.0, 0.0), (0.0, 1.0)])
def test_pure_endpoints_recover_active_component_and_ignore_missing_inactive_data(
    fractions,
):
    state = gas_state(jnp.array([1e5, 2e6]), jnp.array([300.0, 600.0]))
    first, second = (state, None) if fractions[0] else (None, state)
    result = jax.jit(lambda: mix_properties(first, second, fractions))()
    for name, value in state.items():
        np.testing.assert_array_equal(result[name], value)
    np.testing.assert_array_equal(result["speed_of_sound_p"], state["speed_of_sound"])
    np.testing.assert_array_equal(result["speed_of_sound_pT"], state["speed_of_sound"])
    if fractions[1] == 0:
        assert np.isnan(result["mixture_ratio"]).all()
    else:
        np.testing.assert_array_equal(result["mixture_ratio"], 0.0)


def test_component_order_and_numpy_jax_broadcast_results_agree():
    p, T = np.array([[1e5], [2e6]]), np.array([[300.0, 400.0, 500.0]])
    first, second = liquid_state(p, T), gas_state(p, T)
    result = jax.jit(lambda a, b: mix_properties(a, b, (0.3, 0.7)))(first, second)
    swapped = mix_properties(second, first, (0.7, 0.3))
    host = mix_properties(
        liquid_state(p, T, xp=np), gas_state(p, T, xp=np), (0.3, 0.7), xp=np
    )
    for name, value in result.items():
        assert value.shape == (2, 3)
        # muT=v*(1-T*alpha) can subtract nearly equal terms in gas-dominated
        # states. Allow a few machine ulps of those terms across FMA/non-FMA
        # execution, rather than assigning an unrealistically relative budget
        # to the small cancellation remainder. Other fields keep zero atol.
        atol = 0.0
        if name in {"isothermal_joule_thomson", "joule_thomson"}:
            scale = 1 / np.asarray(result["density"])
            if name == "joule_thomson":
                scale = scale / np.asarray(result["isobaric_heat_capacity"])
            atol = float(8 * np.finfo(float).eps * np.max(scale))
        np.testing.assert_allclose(value, host[name], rtol=3e-14, atol=atol)
        if name not in {
            "mixture_ratio",
            "mass_frac_1",
            "mass_frac_2",
            "vol_frac_1",
            "vol_frac_2",
        }:
            np.testing.assert_allclose(value, swapped[name], rtol=3e-14, atol=atol)
    np.testing.assert_allclose(result["vol_frac_1"], swapped["vol_frac_2"])
    assert (
        not {"quality_mass", "quality_volume", "is_two_phase", "compressibility_factor"}
        & result.keys()
    )
    assert len(result) == 27


@pytest.mark.parametrize("missing", ["viscosity", "isobaric_expansion_coefficient"])
def test_locally_unavailable_optional_inputs_preserve_core_and_zero_failed_gradients(
    missing,
):
    name = "viscosity" if missing == "viscosity" else "isochoric_heat_capacity"

    def evaluate(T):
        first = liquid_state(2e6, T)
        first[missing] = jnp.asarray(jnp.nan)
        return mix_properties(first, gas_state(2e6, T), (0.9, 0.1))

    result = jax.jit(evaluate)(400.0)
    assert np.isnan(result[name])
    for field in ("density", "enthalpy", "entropy", "internal_energy"):
        assert np.isfinite(result[field])
    derivative = jax.jit(jax.grad(lambda T: jnp.nan_to_num(evaluate(T)[name])))(400.0)
    assert derivative == 0.0


def test_zero_mixture_expansion_has_finite_zero_gruneisen_and_smooth_derivatives():
    def evaluate(T):
        return mix_properties(
            liquid_state(1e6, T, alpha=4e-4),
            liquid_state(1e6, T, alpha=-4e-4),
            (0.5, 0.5),
        )

    result = jax.jit(evaluate)(300.0)
    assert result["isobaric_expansion_coefficient"] == 0.0
    assert result["gruneisen"] == 0.0
    np.testing.assert_array_equal(
        result["isochoric_heat_capacity"], result["isobaric_heat_capacity"]
    )
    np.testing.assert_array_equal(
        result["isentropic_compressibility"], result["isothermal_compressibility"]
    )
    np.testing.assert_allclose(
        result["isothermal_joule_thomson"], 1 / result["density"], rtol=2e-15
    )
    derivative = jax.jit(jax.grad(lambda T: evaluate(T)["gruneisen"]))(300.0)
    expected = (4e-4) ** 2 / (
        result["density"]
        * result["isochoric_heat_capacity"]
        * result["isothermal_compressibility"]
    )
    np.testing.assert_allclose(derivative, expected, rtol=3e-14)


def test_unrepresentable_mass_ratio_does_not_invalidate_finite_mass_fractions():
    fractions = 1.0, float(np.nextafter(0.0, 1.0))
    result = jax.jit(
        lambda: mix_properties(gas_state(1e5, 400.0), other_gas(1e5, 400.0), fractions)
    )()
    assert np.isnan(result["mixture_ratio"])
    for name in (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    ):
        assert np.isfinite(result[name])


def test_high_finite_pressure_density_values_and_forward_reverse_ad_are_stable():
    weights, temperature = (0.4, 0.6), 500.0
    pressure = jnp.asarray([1e6, np.finfo(float).max])
    gas_constant = weights[0] * 287.0 + weights[1] * 461.5

    def evaluate(p):
        return mix_properties(
            gas_state(p, temperature), other_gas(p, temperature), weights
        )

    def density(p):
        return evaluate(p)["density"]

    value, forward = jax.jit(lambda p: jax.jvp(density, (p,), (jnp.ones_like(p),)))(
        pressure
    )
    reverse = jax.jit(jax.grad(lambda p: density(p).sum()))(pressure)
    np.testing.assert_allclose(
        value, np.asarray(pressure) / (gas_constant * temperature), rtol=3e-15
    )
    expected = 1 / (gas_constant * temperature)
    np.testing.assert_allclose(forward, [expected, expected], rtol=3e-14)
    np.testing.assert_allclose(reverse, [expected, expected], rtol=3e-14)
    # Some optional bulk-modulus products at floatmax are unrepresentable;
    # their NaNs do not change the finite density value tested here.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        host = mix_properties(
            gas_state(np.asarray(pressure), temperature, xp=np),
            other_gas(np.asarray(pressure), temperature, xp=np),
            weights,
            xp=np,
        )
    np.testing.assert_allclose(value, host["density"], rtol=3e-15)
    _, volume_derivative = jax.jit(
        lambda p: jax.jvp(
            lambda p: jnp.stack((evaluate(p)["vol_frac_1"], evaluate(p)["vol_frac_2"])),
            (p,),
            (jnp.ones_like(p),),
        )
    )(pressure)
    np.testing.assert_allclose(volume_derivative, 0.0, atol=2e-22)


@pytest.mark.parametrize("densities", [(8e302, 1.2e303), (1e303, 1e303)])
def test_large_component_density_partials_match_analytic_harmonic_law(densities):
    weights = 0.4, 0.6
    first, second = gas_state(1e6, 500.0), other_gas(1e6, 500.0)

    def evaluate(values):
        left = dict(first) | {"density": values[0]}
        right = dict(second) | {"density": values[1]}
        return mix_properties(left, right, weights)["density"]

    values = jnp.asarray(densities)
    density = 1 / sum(weight / value for weight, value in zip(weights, densities))
    expected = [
        weight * (density / value) ** 2 for weight, value in zip(weights, densities)
    ]
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(evaluate))(values), expected, rtol=3e-14
    )
    np.testing.assert_allclose(
        jax.jit(jax.jacrev(evaluate))(values), expected, rtol=3e-14
    )
