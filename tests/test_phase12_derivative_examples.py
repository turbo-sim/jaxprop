"""Independent scientific acceptance for retained derivative/optimizer studies."""

import importlib

import CoolProp.CoolProp as CP
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

import jaxprop as jp
from examples.optimistix import demo_ideal_gas, demo_minimize_io
from examples.two_phase_derivatives.enthalpy_derivatives import (
    get_enthalpy_derivatives_numerical,
)
from examples.two_phase_derivatives.saturation_derivatives import (
    finite_difference_responses,
    saturation_reference,
)


@pytest.mark.parametrize("temperature", [293.15, 423.15, 573.15])
@pytest.mark.parametrize("quality", [0.1, 0.5, 0.9])
def test_equilibrium_jacobian_responses_match_independent_signed_source_flashes(
    temperature, quality
):
    source = CP.AbstractState("HEOS", "Water")
    source.update(CP.QT_INPUTS, quality, temperature)
    pressure = source.p()
    reference = saturation_reference(source, temperature, quality)
    numerical = finite_difference_responses(source, pressure, quality)
    for name in ("isochoric_heat_capacity", "speed_of_sound", "gruneisen"):
        assert np.isfinite(reference[name]) and np.isfinite(numerical[name])
        assert reference[name] > 0
        np.testing.assert_allclose(numerical[name], reference[name], rtol=2e-6)
    # This also protects the source-restoration contract of perturbation helpers.
    np.testing.assert_allclose(
        [source.p(), source.Q()], [pressure, quality], rtol=1e-12
    )


def test_enthalpy_derivatives_have_the_correct_order_sign_and_units():
    fluid = jp.Fluid("Water")
    temperature, quality = 423.15, 0.7
    state = fluid.get_state(jp.QT_INPUTS, quality, temperature)
    source = CP.AbstractState("HEOS", "Water")
    reference = saturation_reference(source, temperature, quality)
    numerical = get_enthalpy_derivatives_numerical(
        fluid, float(state.p), float(state.rho)
    )
    expected = np.array(
        [
            reference["enthalpy_pressure_derivative"],
            reference["enthalpy_density_derivative"],
        ]
    )
    assert numerical[0] > 0 > numerical[1]
    np.testing.assert_allclose(numerical, expected, rtol=2e-6)
    # At fixed pressure/temperature only mass quality changes, independently
    # giving dh/drho = delta_h/(d rho/dQ). This preserves dimensional meaning.
    source.update(CP.QT_INPUTS, 0, temperature)
    liquid_h, liquid_v = source.hmass(), 1 / source.rhomass()
    source.update(CP.QT_INPUTS, 1, temperature)
    density_derivative = -(source.hmass() - liquid_h) / (
        float(state.rho) ** 2 * (1 / source.rhomass() - liquid_v)
    )
    np.testing.assert_allclose(numerical[1], density_derivative, rtol=2e-6)


@pytest.fixture(scope="module")
def perfect_gas():
    return jp.Fluid(
        "Air", method="perfect_gas", model_options={"T_ref": 300.0, "p_ref": 101325.0}
    )


@pytest.mark.parametrize("mach", [0.0, 0.35, 1.0])
def test_normalized_static_state_solver_recovers_the_isentropic_law_and_energy(
    perfect_gas, mach
):
    pressure, density = jnp.asarray(1e5), jnp.asarray(1.2)
    total = perfect_gas.get_state(jp.DmassP_INPUTS, density, pressure)
    state = demo_ideal_gas.compute_static_state(
        pressure, density, jnp.asarray(mach), perfect_gas
    )
    result = demo_ideal_gas.verify_static_state(total, state, mach)
    assert (
        abs(result["energy_residual_J_per_kg"])
        <= result["energy_roundoff_budget_J_per_kg"]
    )
    if mach == 0:
        np.testing.assert_allclose([state.p, state.T], [total.p, total.T], rtol=1e-12)


def test_static_state_residual_scaling_is_independent_of_energy_reference(perfect_gas):
    shifted = perfect_gas.with_constants(
        h_ref=perfect_gas.constants.h_ref + 1e8,
        s_ref=perfect_gas.constants.s_ref - 5e4,
    )
    arguments = (jnp.asarray(1e5), jnp.asarray(1.2), jnp.asarray(0.7))
    original = demo_ideal_gas.compute_static_state(*arguments, perfect_gas)
    actual = demo_ideal_gas.compute_static_state(*arguments, shifted)
    np.testing.assert_allclose(
        [actual.p, actual.T, actual.rho],
        [original.p, original.T, original.rho],
        rtol=2e-11,
    )
    total = shifted.get_state(jp.DmassP_INPUTS, arguments[1], arguments[0])
    demo_ideal_gas.verify_static_state(total, actual, arguments[2])


@pytest.mark.parametrize("compiled", [False, True])
def test_logged_objective_counts_plain_calls_and_derivative_calls_once(compiled):
    function = demo_minimize_io.with_counters(demo_minimize_io.rosenbrock)
    objective = jax.jit(function) if compiled else function
    value_gradient = jax.value_and_grad(function)
    if compiled:
        value_gradient = jax.jit(value_gradient)
    point = jnp.asarray([-1.2, 1.0])
    demo_minimize_io.reset_counters()
    jax.block_until_ready(objective(point, None))
    jax.effects_barrier()
    assert demo_minimize_io.get_counters() == {"f_calls": 1, "g_calls": 0}
    value, gradient = jax.block_until_ready(value_gradient(point, None))
    jax.effects_barrier()
    assert demo_minimize_io.get_counters() == {"f_calls": 2, "g_calls": 1}
    np.testing.assert_allclose(value, 24.2, rtol=1e-13)
    np.testing.assert_allclose(gradient, [-215.6, -88], rtol=1e-13)


@pytest.mark.parametrize(
    "name",
    [
        "examples.two_phase_derivatives.check_gruneisen",
        "examples.two_phase_derivatives.check_speed_sound",
        "examples.two_phase_derivatives.enthalpy_derivatives",
        "examples.two_phase_derivatives.saturation_derivatives",
        "examples.two_phase_derivatives.gibbs_saturation",
        "examples.optimistix.demo_ideal_gas",
        "examples.optimistix.demo_root_find",
        "examples.optimistix.demo_minimize",
        "examples.optimistix.demo_minimize_io",
        "examples.optimistix.diffrax_rosenbrock",
    ],
)
def test_importing_repaired_studies_does_not_evaluate_fluids_or_create_figures(
    name, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "An imported example executed a physical/plotting workflow"
        )

    monkeypatch.setattr(jp, "Fluid", forbidden)
    monkeypatch.setattr(plt, "subplots", forbidden)
    module = importlib.reload(importlib.import_module(name))
    assert callable(module.main)
