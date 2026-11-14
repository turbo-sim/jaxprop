"""Source-backed binary inverses, preserved references and host transactions."""

from concurrent.futures import ThreadPoolExecutor
import time

import CoolProp.CoolProp as CP
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.binary_mixture.host import HostEvaluator, legacy_hp
from jaxprop.coolprop import core_calculations as source_core


@pytest.fixture(scope="module")
def sources():
    return jp.FluidCoolProp("Water"), jp.FluidCoolProp("Nitrogen")


@pytest.fixture(scope="module")
def evaluator(sources):
    return HostEvaluator(sources, (0.3, 0.7))


def independent_core(pressure, temperature, weights=(0.3, 0.7), *, backend="HEOS"):
    """Direct source oracle; never invokes a mixture implementation."""
    states = [CP.AbstractState(backend, name) for name in ("Water", "Nitrogen")]
    for state in states:
        state.update(CP.PT_INPUTS, pressure, temperature)
    volume = sum(weight / state.rhomass() for weight, state in zip(weights, states))
    return {
        "pressure": pressure,
        "temperature": temperature,
        "density": 1 / volume,
        "enthalpy": sum(
            weight * state.hmass() for weight, state in zip(weights, states)
        ),
        "internal_energy": sum(
            weight * state.umass() for weight, state in zip(weights, states)
        ),
        "entropy": sum(
            weight * state.smass() for weight, state in zip(weights, states)
        ),
    }


@pytest.mark.parametrize(
    "pressure,temperature", [(1e6, 350.0), (5e6, 400.0), (5e6, 700.0)]
)
def test_pt_hp_ps_recover_independent_source_core(evaluator, pressure, temperature):
    expected = independent_core(pressure, temperature)
    for pair, first, second in (
        (CP.PT_INPUTS, pressure, temperature),
        (CP.HmassP_INPUTS, expected["enthalpy"], pressure),
        (CP.PSmass_INPUTS, pressure, expected["entropy"]),
    ):
        actual, status, causes = evaluator.evaluate(pair, first, second)
        assert status == 0 and causes == {}
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name], value, rtol=3e-11, atol=2e-8)
        np.testing.assert_allclose(
            actual["isentropic_bulk_modulus"],
            actual["density"] * actual["speed_of_sound"] ** 2,
            rtol=3e-15,
        )
        np.testing.assert_allclose(
            actual["isentropic_compressibility"],
            1 / actual["isentropic_bulk_modulus"],
            rtol=3e-15,
        )


def test_source_evaluation_does_not_rebuild_sources_or_use_full_residual_states(
    evaluator, monkeypatch
):
    expected = independent_core(5e6, 400.0)
    full_calls = []
    original = source_core.compute_properties_1phase

    def record_full(*args, **kwargs):
        full_calls.append(args[0])
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Source construction or public full-state residual call is forbidden."
        )

    monkeypatch.setattr(CP, "AbstractState", forbidden)
    monkeypatch.setattr(source_core, "compute_properties_1phase", record_full)
    for component in evaluator.components:
        monkeypatch.setattr(component, "get_state", forbidden)
    actual, status, _ = evaluator.evaluate(CP.HmassP_INPUTS, expected["enthalpy"], 5e6)
    assert status == 0
    np.testing.assert_allclose(actual["temperature"], 400.0, rtol=3e-11)
    assert len(full_calls) == 2  # Once per component, only after root validation.


def test_broadcast_numeric_failures_are_elementwise_and_callbacks_return_same_data(
    evaluator,
):
    pressure = np.array([[5e6], [np.nan], [0.0], [1e20]])
    temperature = np.array([[350.0, 400.0]])
    values, status, causes = evaluator.evaluate(CP.PT_INPUTS, pressure, temperature)
    assert status.shape == (4, 2)
    np.testing.assert_array_equal(status, [[0, 0], [1, 1], [1, 1], [2, 2]])
    assert set(causes) == {(i, j) for i in range(1, 4) for j in range(2)}
    for value in values.values():
        assert value.shape == (4, 2)
        assert np.isnan(value[1:]).all()
    callback_values, callback_status = evaluator.callback(
        CP.PT_INPUTS, pressure, temperature
    )
    np.testing.assert_array_equal(callback_status, status)
    for name in values:
        np.testing.assert_array_equal(callback_values[name], values[name])
        assert callback_values[name].dtype == np.float64
    assert callback_status.dtype == np.int32


def test_out_of_temperature_domain_and_invalid_nonfinite_targets_are_explicit(
    evaluator,
):
    for pair, first, second, expected in (
        (CP.PT_INPUTS, 5e6, 1.0, 2),
        (CP.PT_INPUTS, 5e6, -1.0, 1),
        (CP.HmassP_INPUTS, np.inf, 5e6, 1),
        (CP.PSmass_INPUTS, 5e6, np.nan, 1),
    ):
        values, status, causes = evaluator.evaluate(pair, first, second)
        assert status == expected and () in causes
        assert all(np.isnan(value) for value in values.values())


def test_iteration_budget_is_real_and_does_not_affect_direct_pt(sources):
    evaluator = HostEvaluator(sources, (0.3, 0.7), inverse_max_steps=0)
    direct, status, _ = evaluator.evaluate(CP.PT_INPUTS, 5e6, 400.0)
    assert status == 0
    for pair, first, second in (
        (CP.HmassP_INPUTS, direct["enthalpy"], 5e6),
        (CP.PSmass_INPUTS, 5e6, direct["entropy"]),
    ):
        values, status, causes = evaluator.evaluate(pair, first, second)
        assert status == 3 and () in causes
        assert all(np.isnan(value) for value in values.values())


def test_unreachable_isobaric_phase_gap_is_not_accepted_as_a_root(sources):
    pressure = 1e5
    water, nitrogen = CP.AbstractState("HEOS", "Water"), CP.AbstractState(
        "HEOS", "Nitrogen"
    )
    water.update(CP.PQ_INPUTS, pressure, 0.0)
    temperature, liquid_h = water.T(), water.hmass()
    water.update(CP.PQ_INPUTS, pressure, 1.0)
    vapor_h = water.hmass()
    nitrogen.update(CP.PT_INPUTS, pressure, temperature)
    target = 0.5 * ((liquid_h + vapor_h) / 2 + nitrogen.hmass())
    evaluator = HostEvaluator(sources, (0.5, 0.5))
    values, status, causes = evaluator.evaluate(CP.HmassP_INPUTS, target, pressure)
    assert status == 3 and () in causes
    assert all(np.isnan(value) for value in values.values())


@pytest.mark.parametrize("fractions", [(1.0, 0.0), (0.0, 1.0)])
def test_pure_endpoints_do_not_read_inactive_components(sources, fractions):
    active = sources[0]
    components = (active, object()) if fractions[0] else (object(), active)
    evaluator = HostEvaluator(components, fractions)
    reference = active.get_state(CP.PT_INPUTS, 5e6, 400.0)
    for pair, first, second in (
        (CP.PT_INPUTS, 5e6, 400.0),
        (CP.HmassP_INPUTS, float(reference.h), 5e6),
        (CP.PSmass_INPUTS, 5e6, float(reference.s)),
    ):
        values, status, causes = evaluator.evaluate(pair, first, second)
        assert status == 0 and causes == {}
        for name in (
            "temperature",
            "density",
            "enthalpy",
            "internal_energy",
            "entropy",
        ):
            np.testing.assert_allclose(
                values[name], reference.raw_property(name), rtol=2e-10
            )


def test_source_callback_components_use_their_existing_underlying_provider():
    components = jp.FluidJAX("Water"), jp.FluidJAX("Nitrogen")
    evaluator = HostEvaluator(components, (0.3, 0.7))
    values, status, _ = evaluator.evaluate(CP.PT_INPUTS, 5e6, 400.0)
    assert status == 0
    expected = independent_core(5e6, 400.0)
    for name, value in expected.items():
        np.testing.assert_allclose(values[name], value, rtol=3e-12)


@pytest.mark.parametrize("backend", ["HEOS", "REFPROP"])
def test_host_inverses_preserve_selected_source_backend(backend):
    components = tuple(
        jp.FluidCoolProp(name, backend=backend) for name in ("Water", "Nitrogen")
    )
    evaluator = HostEvaluator(components, (0.3, 0.7))
    expected = independent_core(5e6, 400.0, backend=backend)
    for pair, first, second in (
        (CP.PT_INPUTS, 5e6, 400.0),
        (CP.HmassP_INPUTS, expected["enthalpy"], 5e6),
        (CP.PSmass_INPUTS, 5e6, expected["entropy"]),
    ):
        values, status, causes = evaluator.evaluate(pair, first, second)
        assert status == 0 and causes == {}
        for name, value in expected.items():
            np.testing.assert_allclose(values[name], value, rtol=3e-11, atol=2e-8)


def test_host_perfect_gas_hybrid_matches_component_calibration(sources):
    gas = jp.FluidPerfectGas("Nitrogen")
    components = sources[0], gas
    evaluator = HostEvaluator(components, (0.3, 0.7))
    state1, state2 = (
        component.get_state(CP.PT_INPUTS, 5e6, 400.0) for component in components
    )
    expected_h = 0.3 * state1.h + 0.7 * state2.h
    expected_s = 0.3 * state1.s + 0.7 * state2.s
    for pair, first, second in (
        (CP.PT_INPUTS, 5e6, 400.0),
        (CP.HmassP_INPUTS, float(expected_h), 5e6),
        (CP.PSmass_INPUTS, 5e6, float(expected_s)),
    ):
        values, status, _ = evaluator.evaluate(pair, first, second)
        assert status == 0
        np.testing.assert_allclose(values["temperature"], 400.0, rtol=3e-11)
        np.testing.assert_allclose(values["enthalpy"], expected_h, rtol=3e-11)
        np.testing.assert_allclose(values["entropy"], expected_s, rtol=3e-11)


@pytest.mark.parametrize("temperature", [400.0, 500.0, 600.0])
def test_host_ideal_transport_matches_native_on_both_sides_of_sutherland_pole(
    sources, temperature
):
    gas = jp.FluidPerfectGas("Nitrogen").with_constants(S_mu=-500.0, S_k=-500.0)
    evaluator = HostEvaluator((sources[0], gas), (0.3, 0.7))
    water = sources[0].get_state(CP.PT_INPUTS, 5e6, temperature)
    ideal = gas.get_state(CP.PT_INPUTS, 5e6, temperature)
    density = 1 / (0.3 / float(water.rho) + 0.7 / float(ideal.rho))
    fractions = 0.3 * density / float(water.rho), 0.7 * density / float(ideal.rho)
    actual, status, _ = evaluator.evaluate(CP.PT_INPUTS, 5e6, temperature)
    assert status == 0
    for name in ("viscosity", "conductivity"):
        expected = fractions[0] * water.raw_property(name) + fractions[
            1
        ] * ideal.raw_property(name)
        np.testing.assert_allclose(actual[name], expected, rtol=3e-14)


def test_host_hybrid_inverse_respects_pressure_dependent_pg_density_floor(
    sources, monkeypatch
):
    pressure, temperature = 8e6, 400.0
    gas = jp.FluidPerfectGas("Nitrogen").with_constants(R=1e16, gamma=1.4)
    evaluator = HostEvaluator((sources[0], gas), (0.3, 0.7))
    entry = evaluator.entries[1]
    lower, upper = entry.temperature_bounds(pressure)
    assert lower == 1.0
    np.testing.assert_allclose(upper, 800.0, rtol=1e-15)
    selected_temperatures = []
    selected = entry.selected

    def check_sample(p, T, name):
        selected_temperatures.append(T)
        assert T <= upper
        return selected(p, T, name)

    monkeypatch.setattr(entry, "selected", check_sample)
    component_states = [
        component.get_state(CP.PT_INPUTS, pressure, temperature)
        for component in evaluator.components
    ]
    expected = {
        "pressure": pressure,
        "temperature": temperature,
        "density": 1
        / sum(w / float(state.rho) for w, state in zip((0.3, 0.7), component_states)),
        **{
            name: sum(
                w * float(state.raw_property(name))
                for w, state in zip((0.3, 0.7), component_states)
            )
            for name in ("enthalpy", "internal_energy", "entropy")
        },
    }
    for pair, first, second in (
        (CP.PT_INPUTS, pressure, temperature),
        (CP.HmassP_INPUTS, expected["enthalpy"], pressure),
        (CP.PSmass_INPUTS, pressure, expected["entropy"]),
    ):
        values, status, causes = evaluator.evaluate(pair, first, second)
        assert status == 0 and causes == {}
        for name, value in expected.items():
            np.testing.assert_allclose(values[name], value, rtol=3e-11, atol=2e-8)
    assert selected_temperatures
    assert max(selected_temperatures) == upper
    values, status, causes = evaluator.evaluate(CP.PT_INPUTS, pressure, 1000.0)
    assert status == 2 and () in causes
    assert all(np.isnan(value) for value in values.values())


def test_empty_batches_do_not_evaluate_providers(evaluator, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Empty batches must not evaluate a point.")

    monkeypatch.setattr(evaluator, "_scalar", forbidden)
    values, status, causes = evaluator.evaluate(
        CP.PT_INPUTS, np.empty((0, 2)), np.empty((0, 1))
    )
    assert status.shape == (0, 2) and causes == {}
    assert all(value.shape == (0, 2) for value in values.values())


def test_reversed_shared_component_transactions_remain_isolated(sources, monkeypatch):
    first = HostEvaluator(sources, (0.3, 0.7))
    reverse = HostEvaluator(tuple(reversed(sources)), (0.7, 0.3))
    expected = [independent_core(5e6, T) for T in (350.0, 700.0)]
    original = source_core.compute_properties_1phase

    def yield_during_full(*args, **kwargs):
        time.sleep(0.001)
        return original(*args, **kwargs)

    monkeypatch.setattr(source_core, "compute_properties_1phase", yield_during_full)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(engine.evaluate, CP.PT_INPUTS, 5e6, T)
            for engine, T in ((first, 350.0), (reverse, 700.0))
        ]
        results = [job.result(timeout=10) for job in jobs]
    for (values, status, causes), oracle in zip(results, expected):
        assert status == 0 and causes == {}
        for name, value in oracle.items():
            np.testing.assert_allclose(values[name], value, rtol=3e-12)


def test_legacy_raw_as_entry_uses_same_core_without_creating_sources(monkeypatch):
    AS1, AS2 = CP.AbstractState("HEOS", "Water"), CP.AbstractState("HEOS", "Nitrogen")
    expected = independent_core(5e6, 400.0)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Raw legacy evaluation must not construct an AbstractState."
        )

    monkeypatch.setattr(CP, "AbstractState", forbidden)
    values = legacy_hp(expected["enthalpy"], 5e6, AS1, AS2, 0.3 / 0.7)
    for name, value in expected.items():
        np.testing.assert_allclose(values[name], value, rtol=3e-11, atol=2e-8)
