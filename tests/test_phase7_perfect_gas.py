"""Independent perfect-gas equations, calibration, failures and JAX policies."""

import json
import os
from pathlib import Path
import subprocess
import sys

import CoolProp.CoolProp as CP
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp
from utilities import get_available_backends

CALIBRATIONS = [
    (backend, name, temperature, pressure)
    for backend in get_available_backends()
    for name, temperature, pressure in (
        ("Air", 300.0, 1e5),
        ("Nitrogen", 300.0, 1e5),
        ("CO2", 600.0, 1e5),
        ("Water", 500.0, 1e5),
    )
]
PAIRS = (
    (jxp.PT_INPUTS, "pressure", "temperature"),
    (jxp.HmassP_INPUTS, "enthalpy", "pressure"),
    (jxp.PSmass_INPUTS, "pressure", "entropy"),
    (jxp.HmassSmass_INPUTS, "enthalpy", "entropy"),
    (jxp.DmassP_INPUTS, "density", "pressure"),
    (jxp.DmassHmass_INPUTS, "density", "enthalpy"),
)
GAS_PROPERTIES = frozenset(
    {
        "temperature",
        "pressure",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
        "isobaric_heat_capacity",
        "isochoric_heat_capacity",
        "heat_capacity_ratio",
        "compressibility_factor",
        "speed_of_sound",
        "viscosity",
        "conductivity",
        "gruneisen",
        "is_two_phase",
        "isothermal_compressibility",
        "isentropic_compressibility",
        "isothermal_bulk_modulus",
        "isentropic_bulk_modulus",
        "isobaric_expansion_coefficient",
        "joule_thomson",
        "isothermal_joule_thomson",
    }
)
CORE = ("pressure", "temperature", "density", "enthalpy", "entropy", "internal_energy")


@pytest.fixture(
    scope="module", params=CALIBRATIONS, ids=lambda item: f"{item[0]}-{item[1]}"
)
def calibration(request):
    backend, name, temperature, pressure = request.param
    source = CP.AbstractState(backend, name)
    source.update(CP.PT_INPUTS, pressure, temperature)
    # Store the source oracle independently before creating the active model.
    oracle = {
        "T_ref": temperature,
        "p_ref": pressure,
        "h_ref": source.hmass(),
        "s_ref": source.smass(),
        "cp": source.cpmass(),
        "R": jxp.GAS_CONSTANT / source.molar_mass(),
        "viscosity": source.viscosity(),
        "conductivity": source.conductivity(),
    }
    fluid = jxp.Fluid(
        name,
        method="perfect_gas",
        backend=backend,
        identifier=f"gas-{name}",
        model_options={"T_ref": temperature, "p_ref": pressure},
    )
    return fluid, oracle


def ideal_values(oracle, pressure, temperature):
    """Independent mass-SI ideal-gas equations, including energy offsets."""
    pressure, temperature = np.broadcast_arrays(pressure, temperature)
    cp, gas_constant = oracle["cp"], oracle["R"]
    cv = cp - gas_constant
    gamma = cp / cv
    enthalpy = oracle["h_ref"] + cp * (temperature - oracle["T_ref"])
    return {
        "pressure": pressure,
        "temperature": temperature,
        "density": pressure / (gas_constant * temperature),
        "enthalpy": enthalpy,
        "entropy": oracle["s_ref"]
        + cp * np.log(temperature / oracle["T_ref"])
        - gas_constant * np.log(pressure / oracle["p_ref"]),
        "internal_energy": enthalpy - gas_constant * temperature,
        "isobaric_heat_capacity": cp,
        "isochoric_heat_capacity": cv,
        "heat_capacity_ratio": gamma,
        "compressibility_factor": 1.0,
        "speed_of_sound": np.sqrt(gamma * gas_constant * temperature),
        "gruneisen": gas_constant / cv,
        "is_two_phase": 0.0,
        "isothermal_compressibility": 1 / pressure,
        "isentropic_compressibility": 1 / (gamma * pressure),
        "isothermal_bulk_modulus": pressure,
        "isentropic_bulk_modulus": gamma * pressure,
        "isobaric_expansion_coefficient": 1 / temperature,
        "joule_thomson": 0.0,
        "isothermal_joule_thomson": 0.0,
    }


def assert_success(state, expected, fluid, shape):
    assert state.shape == shape
    assert state.fluid_name == fluid.name
    assert state.identifier == fluid.identifier
    assert state.reference_id == fluid.reference_id
    assert state.model == "perfect_gas" and state.provider == "native"
    assert state.available_properties == GAS_PROPERTIES
    np.testing.assert_array_equal(state.status, np.zeros(shape, dtype=np.int32))
    for key, value in expected.items():
        actual = state.raw_property(key)
        assert actual.shape == shape, key
        np.testing.assert_allclose(
            actual, np.broadcast_to(value, shape), rtol=2e-12, atol=1e-9, err_msg=key
        )
        assert np.all(np.asarray(state.property_valid(key))), key


def test_calibration_matches_independent_source_but_enforces_ideal_heat_capacity_identity(
    calibration,
):
    fluid, oracle = calibration
    state = fluid.get_state(jxp.PT_INPUTS, oracle["p_ref"], oracle["T_ref"])
    expected = ideal_values(oracle, oracle["p_ref"], oracle["T_ref"])
    assert_success(state, expected, fluid, ())
    np.testing.assert_allclose(state.h, oracle["h_ref"], rtol=1e-12)
    np.testing.assert_allclose(state.s, oracle["s_ref"], rtol=1e-12)
    np.testing.assert_allclose(state.cp, oracle["cp"], rtol=1e-12)
    np.testing.assert_allclose(state.cp - state.cv, oracle["R"], rtol=1e-12)
    np.testing.assert_allclose(state.h - state.u, oracle["R"] * state.T, rtol=1e-12)
    for name in ("viscosity", "conductivity"):
        np.testing.assert_allclose(state.raw_property(name), oracle[name], rtol=1e-12)


@pytest.mark.parametrize("pair,first_name,second_name", PAIRS)
def test_all_six_pairs_recover_independent_off_calibration_batch(
    calibration, pair, first_name, second_name
):
    fluid, oracle = calibration
    temperature = oracle["T_ref"] * np.asarray([[0.8], [1.25]])
    pressure = oracle["p_ref"] * np.asarray([[0.2, 2.0, 7.0]])
    expected = ideal_values(oracle, pressure, temperature)
    state = fluid.get_state(pair, expected[first_name], expected[second_name])
    assert_success(state, expected, fluid, (2, 3))
    assert not np.any(np.asarray(state.clipped))


@pytest.mark.parametrize(
    "field,input_axis",
    [
        ("enthalpy", 1),
        ("internal_energy", 1),
        ("entropy", 1),
        ("entropy", 0),
        ("density", 1),
        ("density", 0),
        ("speed_of_sound", 1),
    ],
)
def test_autodiff_matches_independent_analytic_partials(calibration, field, input_axis):
    fluid, oracle = calibration
    pressure, temperature = 2.5 * oracle["p_ref"], 1.2 * oracle["T_ref"]
    values = ideal_values(oracle, pressure, temperature)
    gas_constant, cp = oracle["R"], oracle["cp"]
    expected = {
        ("enthalpy", 1): cp,
        ("internal_energy", 1): cp - gas_constant,
        ("entropy", 1): cp / temperature,
        ("entropy", 0): -gas_constant / pressure,
        ("density", 1): -values["density"] / temperature,
        ("density", 0): values["density"] / pressure,
        ("speed_of_sound", 1): values["speed_of_sound"] / (2 * temperature),
    }[(field, input_axis)]

    def project(inputs):
        return fluid.get_state(
            jxp.PT_INPUTS, inputs[0], inputs[1], errors="mask"
        ).raw_property(field)

    actual = jax.jit(jax.grad(project))(jnp.asarray([pressure, temperature]))[
        input_axis
    ]
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)


def test_transport_uses_sutherland_law_and_its_analytic_slope(calibration):
    fluid, oracle = calibration
    temperature = 1.3 * oracle["T_ref"]
    state = fluid.get_state(jxp.PT_INPUTS, oracle["p_ref"], temperature)
    for field, constant_name in (("viscosity", "S_mu"), ("conductivity", "S_k")):
        coefficient = float(getattr(fluid.constants, constant_name))
        expected = (
            oracle[field]
            * (temperature / oracle["T_ref"]) ** 1.5
            * ((oracle["T_ref"] + coefficient) / (temperature + coefficient))
        )
        np.testing.assert_allclose(state.raw_property(field), expected, rtol=2e-12)
        derivative = jax.grad(
            lambda t: fluid.get_state(
                jxp.PT_INPUTS, oracle["p_ref"], t, errors="mask"
            ).raw_property(field)
        )(temperature)
        np.testing.assert_allclose(
            derivative,
            expected * (1.5 / temperature - 1 / (temperature + coefficient)),
            rtol=2e-12,
            atol=1e-18,
        )


@pytest.fixture(scope="module")
def gas():
    return jxp.Fluid(
        "CO2",
        method="perfect_gas",
        identifier="policy-gas",
        model_options={"T_ref": 600.0, "p_ref": 1e5},
    )


def test_eager_failure_reports_first_row_major_index(gas):
    with pytest.raises(jxp.InvalidStateError) as captured:
        gas.get_state(jxp.PT_INPUTS, np.asarray([[1e5, -1.0], [np.nan, 2e5]]), 500.0)
    assert captured.value.index == (0, 1)
    assert captured.value.request == "PT_INPUTS"
    assert captured.value.provider == "native"
    assert captured.value.model == "perfect_gas"
    assert captured.value.reason


@pytest.mark.parametrize("compile_call", [False, True])
def test_mask_policy_preserves_successful_rows_and_masks_failed_rows(gas, compile_call):
    pressure = jnp.asarray([[1e5, -1.0], [np.nan, 2e5]])
    evaluate = lambda p: gas.get_state(jxp.PT_INPUTS, p, 500.0, errors="mask")
    state = (jax.jit(evaluate) if compile_call else evaluate)(pressure)
    np.testing.assert_array_equal(state.status, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(state.valid, [[True, False], [False, True]])
    assert not np.any(np.asarray(state.clipped))
    for name in state.available_properties:
        values = np.asarray(state.raw_property(name))
        assert np.all(np.isnan(values[~np.asarray(state.valid)])), name
    for index in ((0, 0), (1, 1)):
        expected = gas.get_state(jxp.PT_INPUTS, float(pressure[index]), 500.0)
        for name in CORE:
            np.testing.assert_allclose(
                state.raw_property(name)[index], expected.raw_property(name), rtol=1e-12
            )


@pytest.mark.parametrize("compile_call", [False, True])
def test_clip_policy_reports_adjusted_consistent_states_and_unrecoverable_rows(
    gas, compile_call
):
    lower_temperature = gas.capabilities.domain.bounds["temperature"][0]
    lower_pressure = gas.capabilities.domain.bounds["pressure"][0]
    pressure = jnp.asarray([1e5, -2.0, np.nan, 1e5])
    temperature = jnp.asarray([500.0, 500.0, 500.0, -20.0])
    evaluate = lambda p, t: gas.get_state(jxp.PT_INPUTS, p, t, errors="clip")
    state = (jax.jit(evaluate) if compile_call else evaluate)(pressure, temperature)
    assert state.error_policy == "clip"
    np.testing.assert_array_equal(state.status, [0, 0, 1, 0])
    np.testing.assert_array_equal(state.clipped, [False, True, False, True])
    np.testing.assert_array_equal(state.valid, [True, True, False, True])
    np.testing.assert_allclose(
        np.asarray(state.T)[[0, 1, 3]], [500.0, 500.0, lower_temperature], rtol=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(state.p)[[0, 1, 3]], [1e5, lower_pressure, 1e5], rtol=1e-13
    )
    for name in CORE:
        values = np.asarray(state.raw_property(name))
        assert np.all(np.isfinite(values[[0, 1, 3]])), name
        assert np.isnan(values[2]), name
    np.testing.assert_allclose(
        np.asarray(state.p)[[0, 1, 3]],
        np.asarray(state.rho)[[0, 1, 3]]
        * float(gas.constants.R)
        * np.asarray(state.T)[[0, 1, 3]],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(state.h - state.u)[[0, 1, 3]],
        float(gas.constants.R) * np.asarray(state.T)[[0, 1, 3]],
        rtol=1e-12,
    )


def test_positive_subfloor_values_are_domain_failures_until_explicitly_clipped(gas):
    lower_temperature = gas.capabilities.domain.bounds["temperature"][0]
    temperature = 0.5 * lower_temperature
    with pytest.raises(jxp.OutOfDomainError) as captured:
        gas.get_state(jxp.PT_INPUTS, 1e5, temperature)
    assert captured.value.index == ()
    masked = gas.get_state(jxp.PT_INPUTS, 1e5, temperature, errors="mask")
    assert masked.status == jxp.StatusCode.OUT_OF_DOMAIN
    assert not masked.clipped
    assert all(np.isnan(masked.raw_property(name)) for name in CORE)
    clipped = gas.get_state(jxp.PT_INPUTS, 1e5, temperature, errors="clip")
    assert clipped.status == 0 and clipped.clipped
    np.testing.assert_allclose(clipped.T, lower_temperature, rtol=1e-13)


@pytest.mark.parametrize("pair", [pair for pair, _, _ in PAIRS])
def test_finite_invalid_pair_inputs_clip_to_a_consistent_declared_domain(gas, pair):
    reference = gas.get_state(jxp.PT_INPUTS, 1e5, 500.0)
    cp = float(reference.cp)
    h_below_domain = float(gas.constants.h_ref - cp * gas.constants.T_ref) - 100.0
    first, second = {
        jxp.PT_INPUTS: (-1.0, -1.0),
        jxp.HmassP_INPUTS: (h_below_domain, -1.0),
        jxp.PSmass_INPUTS: (-1.0, float(reference.s)),
        jxp.HmassSmass_INPUTS: (h_below_domain, float(reference.s)),
        jxp.DmassP_INPUTS: (-1.0, -1.0),
        jxp.DmassHmass_INPUTS: (-1.0, h_below_domain),
    }[pair]
    state = gas.get_state(pair, first, second, errors="clip")
    assert state.status == 0 and state.valid and state.clipped
    for field in CORE:
        assert np.isfinite(state.raw_property(field)), field
    for name, limits in gas.capabilities.domain.bounds.items():
        if name in {"temperature", "pressure", "density"}:
            lower, upper = limits
            value = float(state.raw_property(name))
            if lower is not None:
                assert value >= lower * (1 - 1e-12), name
            if upper is not None:
                assert value <= upper * (1 + 1e-12), name
    np.testing.assert_allclose(
        state.p, state.rho * gas.constants.R * state.T, rtol=1e-12
    )
    np.testing.assert_allclose(
        state.h - state.u, gas.constants.R * state.T, rtol=1e-9, atol=1e-8
    )


@pytest.mark.parametrize("pair", [jxp.PT_INPUTS, jxp.HmassP_INPUTS])
def test_clipped_coordinate_derivatives_are_zero_beyond_the_boundary(gas, pair):
    if pair == jxp.PT_INPUTS:
        evaluate = lambda value: gas.get_state(pair, 1e5, value, errors="clip").h
        outside = -10.0
    else:
        cp = float(gas.get_state(jxp.PT_INPUTS, 1e5, 500.0).cp)
        outside = float(gas.constants.h_ref - cp * gas.constants.T_ref) - 100.0
        evaluate = lambda value: gas.get_state(pair, value, 1e5, errors="clip").T
    derivative = jax.jit(jax.grad(evaluate))(outside)
    np.testing.assert_array_equal(derivative, 0.0)


def test_failed_rows_do_not_poison_gradients_of_successful_masked_rows(gas):
    cp = float(gas.get_state(jxp.PT_INPUTS, 1e5, 500.0).cp)

    def enthalpy_sum(temperature):
        state = gas.get_state(jxp.PT_INPUTS, 1e5, temperature, errors="mask")
        return jnp.nansum(state.raw_property("enthalpy"))

    actual = jax.jit(jax.grad(enthalpy_sum))(jnp.asarray([500.0, -10.0, np.nan]))
    np.testing.assert_allclose(actual, [cp, 0.0, 0.0], rtol=1e-12)


@pytest.mark.parametrize(
    "pair,bad_entropy",
    [
        (jxp.PSmass_INPUTS, 1e308),
        (jxp.HmassSmass_INPUTS, -1e308),
    ],
)
@pytest.mark.parametrize("replace_invalid", ["where", "nan_to_num"])
def test_inverse_overflow_is_a_provider_failure_without_poisoning_masked_gradients(
    gas,
    pair,
    bad_entropy,
    replace_invalid,
):
    reference = gas.get_state(jxp.PT_INPUTS, 1e5, 500.0)
    fixed = reference.p if pair == jxp.PSmass_INPUTS else reference.h
    entropy = jnp.asarray([float(reference.s), bad_entropy])

    def evaluate(values):
        return gas.get_state(pair, fixed, values, errors="mask")

    state = jax.jit(evaluate)(entropy)
    np.testing.assert_array_equal(state.status, [0, jxp.StatusCode.PROVIDER_FAILURE])
    assert np.all([np.isnan(state.raw_property(name)[1]) for name in CORE])
    np.testing.assert_allclose(state.h[0], reference.h, rtol=1e-12)

    def total_enthalpy(values):
        result = evaluate(values)
        enthalpy = result.raw_property("enthalpy")
        safe = (
            jnp.where(result.valid, enthalpy, 0.0)
            if replace_invalid == "where"
            else jnp.nan_to_num(enthalpy, nan=0.0)
        )
        return safe.sum()

    actual = jax.jit(jax.grad(total_enthalpy))(entropy)
    # dh/ds|p=T; with h fixed, this projection has zero entropy derivative.
    expected = [float(reference.T) if pair == jxp.PSmass_INPUTS else 0.0, 0.0]
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_float_max_temperature_reports_provider_failure_and_finite_zero_masked_gradient(
    gas,
):
    temperature = jnp.asarray([500.0, np.finfo(np.float64).max])

    def evaluate(values):
        return gas.get_state(jxp.PT_INPUTS, 1e5, values, errors="mask")

    state = jax.jit(evaluate)(temperature)
    np.testing.assert_array_equal(state.status, [0, jxp.StatusCode.PROVIDER_FAILURE])
    assert all(np.isnan(state.raw_property(name)[1]) for name in CORE)

    def total_enthalpy(values):
        result = evaluate(values)
        return jnp.where(result.valid, result.raw_property("enthalpy"), 0.0).sum()

    derivative = jax.jit(jax.grad(total_enthalpy))(temperature)
    cp = float(gas.get_state(jxp.PT_INPUTS, 1e5, 500.0).cp)
    np.testing.assert_allclose(derivative, [cp, 0.0], rtol=1e-12)


def test_clip_projects_underflowing_entropy_inverse_to_temperature_floor(gas):
    state = gas.get_state(jxp.PSmass_INPUTS, 1e5, -1e308, errors="clip")
    assert state.status == 0 and state.valid and state.clipped
    assert all(np.isfinite(state.raw_property(name)) for name in CORE)
    np.testing.assert_allclose(
        state.T, gas.capabilities.domain.bounds["temperature"][0], rtol=1e-12
    )
    np.testing.assert_allclose(
        state.p, state.rho * gas.constants.R * state.T, rtol=1e-12
    )


@pytest.mark.parametrize("pair,first_name,second_name", PAIRS)
def test_density_floor_projected_state_round_trips_all_input_pairs(
    gas,
    pair,
    first_name,
    second_name,
):
    # Coupled p/rho bounds can otherwise round a projected boundary state just
    # outside the accepted domain when it is reused through another pair.
    projected = gas.get_state(jxp.PT_INPUTS, 0.1, 1e11, errors="clip")
    assert projected.status == 0 and projected.clipped
    recovered = gas.get_state(
        pair,
        projected.raw_property(first_name),
        projected.raw_property(second_name),
        errors="mask",
    )
    assert recovered.status == 0 and recovered.valid
    assert not recovered.clipped
    for name in CORE:
        np.testing.assert_allclose(
            recovered.raw_property(name),
            projected.raw_property(name),
            rtol=2e-12,
            atol=1e-8,
            err_msg=name,
        )


def test_clip_domain_pressure_adjustment_enforces_density_floor(gas):
    lower_density = gas.capabilities.domain.bounds["density"][0]
    lower_pressure = gas.capabilities.domain.bounds["pressure"][0]
    temperature = 2 * lower_pressure / (lower_density * float(gas.constants.R))
    state = gas.get_state(jxp.PT_INPUTS, lower_pressure, temperature, errors="clip")
    assert state.status == 0 and state.clipped
    np.testing.assert_allclose(state.rho, lower_density, rtol=1e-12)
    np.testing.assert_allclose(state.T, temperature, rtol=1e-12)
    np.testing.assert_allclose(state.p, 2 * lower_pressure, rtol=1e-12)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_clip_does_not_invent_a_finite_state_from_nonfinite_explicit_inputs(gas, bad):
    state = gas.get_state(jxp.PT_INPUTS, bad, 500.0, errors="clip")
    assert state.status == jxp.StatusCode.INVALID_INPUT
    assert not state.clipped and not state.valid
    assert all(
        np.isnan(state.raw_property(name)) for name in state.available_properties
    )


@pytest.mark.parametrize("errors", ["mask", "clip"])
def test_native_evaluation_does_not_reenter_coolprop_or_add_host_callbacks(
    gas, monkeypatch, errors
):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Source construction entered native perfect-gas evaluation"
        )

    monkeypatch.setattr(CP, "AbstractState", forbidden)
    evaluate = jax.jit(
        lambda temperature: gas.get_state(
            jxp.PT_INPUTS, 1e5, temperature, errors=errors
        )
    )
    state = evaluate(jnp.asarray([450.0, 550.0]))
    np.testing.assert_array_equal(state.status, [0, 0])
    assert np.all(np.isfinite(state.h))
    program = evaluate.lower(jnp.asarray([450.0, 550.0])).compile().as_text()
    assert "callback" not in program.lower()


@pytest.mark.parametrize("transform", ["jit", "constant_jit", "vmap", "grad"])
def test_transformed_raise_is_rejected_with_explicit_policy_guidance(gas, transform):
    evaluate = lambda temperature: gas.get_state(jxp.PT_INPUTS, 1e5, temperature).h
    call = {
        "jit": lambda: jax.jit(evaluate)(500.0),
        "constant_jit": lambda: jax.jit(lambda: evaluate(500.0))(),
        "vmap": lambda: jax.vmap(evaluate)(jnp.asarray([500.0, 600.0])),
        "grad": lambda: jax.grad(evaluate)(500.0),
    }[transform]
    with pytest.raises(jxp.UnsupportedExecutionModeError) as captured:
        call()
    assert captured.value.request == "errors=raise"
    assert "mask" in str(captured.value)
    assert "clip" in str(captured.value)


@pytest.mark.parametrize("errors", ["raise", "mask", "clip"])
@pytest.mark.parametrize("pair", [jxp.PQ_INPUTS, jxp.QT_INPUTS, jxp.DmolarT_INPUTS])
def test_static_unsupported_inputs_are_rejected_in_every_policy(gas, errors, pair):
    with pytest.raises(jxp.UnsupportedInputPairError):
        gas.get_state(pair, 1.0, 300.0, errors=errors)


@pytest.mark.parametrize("errors", ["raise", "mask", "clip"])
def test_perfect_gas_phase_requests_remain_explicitly_unsupported(gas, errors):
    with pytest.raises(jxp.UnsupportedOperationError):
        gas.get_state_saturation(jxp.PQ_INPUTS, 1e5, 0.0, errors=errors)
    with pytest.raises(jxp.UnsupportedOptionError):
        gas.get_state(jxp.PT_INPUTS, 1e5, 500.0, supersaturation=True, errors=errors)


def test_negative_energy_and_entropy_reference_does_not_invalidate_inverse_inputs():
    program = r"""
import CoolProp.CoolProp as CP
import json
import numpy as np
import jaxprop as p
CP.set_reference_state('CO2',300.,500.,-1e5,-1e3)
f=p.Fluid('CO2',method='perfect_gas',model_options={'T_ref':600.,'p_ref':1e5})
state=f.get_state(p.PT_INPUTS,2e5,625.)
assert state.h<0 and state.s<0 and state.u<0
for pair,x,y in [(p.HmassP_INPUTS,state.h,state.p),(p.HmassSmass_INPUTS,state.h,state.s),(p.PSmass_INPUTS,state.p,state.s),(p.DmassHmass_INPUTS,state.rho,state.h)]:
 recovered=f.get_state(pair,x,y)
 assert int(recovered.status)==0
 np.testing.assert_allclose([recovered.T,recovered.p],[625.,2e5],rtol=2e-11)
print(json.dumps({'negative_h':bool(state.h<0),'negative_s':bool(state.s<0),'negative_u':bool(state.u<0)}))
"""
    repo = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(json.loads(result.stdout.strip().splitlines()[-1]).values())


@pytest.mark.parametrize("backend", get_available_backends())
def test_transport_fit_keeps_its_second_source_sample_within_published_limits(backend):
    # The old unconditional T_ref+100 sample would exceed R134a's 455 K limit.
    source = CP.AbstractState(backend, "R134a")
    temperature = source.Tmax() - 10.0
    source.update(CP.PT_INPUTS, 1e5, temperature)
    expected_h, expected_cp = source.hmass(), source.cpmass()
    fluid = jxp.Fluid(
        "R134a",
        method="perfect_gas",
        backend=backend,
        model_options={"T_ref": temperature, "p_ref": 1e5},
    )
    state = fluid.get_state(jxp.PT_INPUTS, 1e5, temperature)
    assert state.status == 0
    np.testing.assert_allclose(state.h, expected_h, rtol=1e-12)
    np.testing.assert_allclose(state.cp, expected_cp, rtol=1e-12)
    assert np.isfinite(state.viscosity) and state.viscosity > 0
    assert np.isfinite(state.conductivity) and state.conductivity > 0
