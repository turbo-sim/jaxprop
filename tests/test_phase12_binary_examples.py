"""Scientific oracles for the repaired binary/ODE example applications."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import jaxprop as jp
from examples.two_component._model import mixture
from examples.two_component.water_nitrogen_calculations import (
    barotropic_model_two_component,
    blade_geometry,
    find_sonic_state,
)
from examples.two_component.water_nitrogen_speed_sound import isentropic_sound_speed
from examples.optimistix.diffrax_polytropic_expansion import (
    PARAMETERS,
    get_exit_temperature,
    get_exit_temperature_gradient,
)


@pytest.fixture(scope="module")
def water_air():
    return mixture(
        tuple(jp.Fluid(name) for name in ("Water", "Air")), 100.0, name="water_air"
    )


@pytest.mark.parametrize("efficiency", [1.0, 0.8])
def test_coupled_temperature_ODE_recovers_independent_HP_integration(
    water_air, efficiency
):
    """The old wrong JT sign violates this independent enthalpy-coordinate path."""
    p_in, p_out, temperature = 7e5, 1e5, 298.15
    inlet = water_air.get_state(jp.PT_INPUTS, p_in, temperature)
    values, solution = barotropic_model_two_component(
        "Water",
        "Air",
        100.0,
        temperature,
        p_in,
        p_out,
        efficiency,
        process_type="expansion",
        ODE_solver="LSODA",
        ODE_tolerance=1e-8,
        fluid=water_air,
        sample_points=41,
    )
    reference = solve_ivp(
        lambda p, delta_h: [
            efficiency
            / float(
                water_air.get_state(
                    jp.HmassP_INPUTS, float(inlet.h) + delta_h[0], p
                ).rho
            )
        ],
        (p_in, p_out),
        [0.0],
        rtol=1e-9,
        atol=1e-7,
    )
    assert reference.success
    expected = water_air.get_state(
        jp.HmassP_INPUTS, float(inlet.h) + reference.y[0, -1], p_out
    )
    np.testing.assert_allclose(values["temperature"][-1], expected.T, rtol=2e-8)
    np.testing.assert_allclose(values["integrated_enthalpy"][-1], expected.h, rtol=2e-8)
    np.testing.assert_allclose(
        values["velocity"] ** 2 / 2 + values["integrated_enthalpy"], inlet.h, rtol=2e-15
    )
    assert values["velocity"][0] == 0.0
    if efficiency == 1.0:
        np.testing.assert_allclose(values["entropy"], inlet.s, rtol=1e-8, atol=1e-6)
    else:
        assert values["entropy"][-1] > float(inlet.s)
    sonic = find_sonic_state(water_air, solution)
    assert sonic is not None
    np.testing.assert_allclose(
        sonic["velocity"] / sonic["state"].speed_of_sound, 1.0, rtol=1e-8
    )


def test_a_subsonic_pressure_interval_is_not_reported_as_a_sonic_throat(water_air):
    _, solution = barotropic_model_two_component(
        "Water",
        "Air",
        100.0,
        298.15,
        7e5,
        6.5e5,
        0.8,
        process_type="expansion",
        fluid=water_air,
        sample_points=21,
    )
    assert find_sonic_state(water_air, solution) is None


def test_enthalpy_origin_does_not_change_the_integrated_process():
    components = tuple(
        jp.Fluid(name, method="perfect_gas") for name in ("Air", "Nitrogen")
    )
    shifted = tuple(
        component.with_constants(h_ref=float(component.constants.h_ref) + 1e8)
        for component in components
    )
    ordinary, offset = mixture(components, 100.0), mixture(shifted, 100.0)
    trajectories = []
    for model in (ordinary, offset):
        values, solution = barotropic_model_two_component(
            "Air",
            "Nitrogen",
            100.0,
            298.15,
            7e5,
            1e5,
            0.8,
            process_type="expansion",
            fluid=model,
            sample_points=31,
        )
        trajectories.append((values, solution))
    np.testing.assert_allclose(
        trajectories[0][0]["temperature"], trajectories[1][0]["temperature"], rtol=1e-12
    )
    np.testing.assert_allclose(
        trajectories[0][1].enthalpy_increment,
        trajectories[1][1].enthalpy_increment,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        trajectories[0][0]["velocity"],
        trajectories[1][0]["velocity"],
        rtol=1e-10,
        atol=1e-8,
    )


def test_zero_expansion_efficiency_is_an_isenthalpic_zero_velocity_limit(water_air):
    values, solution = barotropic_model_two_component(
        "Water",
        "Air",
        100.0,
        298.15,
        7e5,
        1e5,
        0.0,
        process_type="expansion",
        fluid=water_air,
        sample_points=21,
    )
    np.testing.assert_array_equal(solution.enthalpy_increment, 0.0)
    np.testing.assert_array_equal(values["velocity"], 0.0)
    assert find_sonic_state(water_air, solution) is None


def test_blade_reynolds_uses_the_reported_chord_in_metres():
    # An independently specified outlet flow fixes the geometric area.
    mass_flow, rho, velocity, viscosity = 0.3, 10.0, 30.0, 1e-5
    chord_m, reynolds = blade_geometry(mass_flow, rho, velocity, viscosity)
    expected_area = mass_flow / (rho * velocity)
    expected_chord = np.sqrt(expected_area / (1.1 * np.cos(np.deg2rad(50)) * 3 * 4))
    np.testing.assert_allclose(chord_m, expected_chord, rtol=1e-15)
    np.testing.assert_allclose(
        reynolds, rho * velocity * expected_chord / viscosity, rtol=1e-15
    )
    # The old display-unit chord would produce a factor-1000 Reynolds error.
    assert not np.isclose(
        reynolds, rho * velocity * (expected_chord * 1000) / viscosity
    )


def test_acoustic_check_uses_a_supported_PS_route_for_a_fixed_mass_ratio():
    components = tuple(
        jp.Fluid(name, method="coolprop_jax") for name in ("Water", "Nitrogen")
    )
    model = mixture(components, 50.0)
    state = model.get_state(jp.PT_INPUTS, 5e5, 350.0)
    acoustic = isentropic_sound_speed(model, state.p, state.s)
    np.testing.assert_allclose(acoustic, state.speed_of_sound_pT, rtol=2e-5)
    assert state.speed_of_sound_pT <= state.speed_of_sound_p


@pytest.mark.parametrize("efficiency", [0.4, 1.0])
def test_diffrax_gradient_recovers_the_independent_perfect_gas_solution(efficiency):
    fluid = jp.Fluid(
        "Air", method="perfect_gas", model_options=dict(T_ref=300.0, p_ref=1e5)
    )
    initial = fluid.get_state(jp.PT_INPUTS, 2e6, 400.0)
    params = dict(
        h_in=jnp.asarray(initial.h),
        p_in=jnp.asarray(2e6),
        p_out=jnp.asarray(5e5),
        efficiency=jnp.asarray(efficiency),
    )
    cp, temperature = float(initial.cp), float(initial.T)
    gas_constant = float(initial.p / (initial.rho * initial.T))
    exponent = efficiency * gas_constant / cp
    expected = temperature * (5e5 / 2e6) ** exponent
    np.testing.assert_allclose(get_exit_temperature(params, fluid), expected, rtol=2e-6)
    # Tighten only this independent derivative-convergence oracle; the actual
    # example retains its original adaptive trajectory tolerances.
    accurate = lambda values: get_exit_temperature(values, fluid, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(accurate(params), expected, rtol=1e-8)
    forward = jax.jacfwd(accurate)(params)
    reverse = jax.jacrev(accurate)(params)
    analytical = dict(
        h_in=expected / (cp * temperature),
        p_in=-exponent * expected / 2e6,
        p_out=exponent * expected / 5e5,
        efficiency=expected * gas_constant / cp * np.log(5e5 / 2e6),
    )
    for name in PARAMETERS:
        np.testing.assert_allclose(
            forward[name], analytical[name], rtol=2e-5, atol=1e-10
        )
        np.testing.assert_allclose(
            reverse[name], analytical[name], rtol=2e-5, atol=1e-10
        )
    # Scaling both pressure endpoints preserves the pressure ratio and T_out.
    shifted = dict(params, p_in=2 * params["p_in"], p_out=2 * params["p_out"])
    np.testing.assert_allclose(
        get_exit_temperature(shifted, fluid), expected, rtol=2e-6
    )


def test_real_fluid_diffrax_first_derivatives_match_central_parameter_differences():
    fluid = jp.Fluid("CO2", method="coolprop_jax")
    inlet = fluid.get_state(jp.PT_INPUTS, 2e7, 400.0)
    params = dict(
        h_in=jnp.asarray(inlet.h),
        p_in=jnp.asarray(2e7),
        p_out=jnp.asarray(5e6),
        efficiency=jnp.asarray(0.8),
    )
    forward = jax.jacfwd(get_exit_temperature)(params, fluid)
    reverse = jax.jacrev(get_exit_temperature)(params, fluid)
    finite = get_exit_temperature_gradient(params, fluid)
    fwd = np.array([forward[key] for key in PARAMETERS])
    rev = np.array([reverse[key] for key in PARAMETERS])
    fd = np.array([finite[key] for key in PARAMETERS])
    assert (
        np.all(np.isfinite(fwd))
        and np.all(np.isfinite(rev))
        and np.all(np.isfinite(fd))
    )
    np.testing.assert_allclose(fwd, rev, rtol=2e-8, atol=1e-10)
    np.testing.assert_allclose(fwd, fd, rtol=2e-3, atol=1e-12)
    assert np.linalg.norm(fwd - fd) / np.linalg.norm(fd) < 2e-3
