"""Independent diagnostics and reference-compatible source helper execution."""

import json
import os
from pathlib import Path
import subprocess
import sys

import CoolProp.CoolProp as CP
import numpy as np
import pytest

import jaxprop as jxp
from jaxprop.coolprop import core_calculations as core
from jaxprop.coolprop.source_context import SourceEvaluationContext
from jaxprop.source_references import initialize_source_references
from utilities import get_available_backends

SOURCES = [
    (backend, name)
    for backend in get_available_backends()
    for name in ("CO2", "Water", "Air")
]


@pytest.fixture(scope="module", params=SOURCES, ids=lambda item: "-".join(item))
def source(request):
    backend, name = request.param
    state = CP.AbstractState(backend, name)
    references = initialize_source_references(name, backend, state=state)
    context = SourceEvaluationContext(state, references, name=name, backend=backend)
    return name, backend, state, context


def test_default_quality_uses_stored_reference_without_scratch_work(source):
    name, _, state, context = source
    updates = []
    actual_auxiliary = context.auxiliary_state

    class Counted:
        def update(self, *args):
            updates.append(args)
            return actual_auxiliary.update(*args)

        def __getattr__(self, name):
            return getattr(actual_auxiliary, name)

    context.auxiliary_state = Counted()
    try:
        temperature = {"CO2": 400.0, "Water": 500.0, "Air": 300.0}[name]
        props = core.compute_properties_coolprop(
            state, CP.PT_INPUTS, 1e5, temperature, source_context=context
        )
        expected = float(props["entropy"] > context.critical_entropy)
        assert props["quality_mass"] == props["quality_volume"] == expected
        assert updates == []
    finally:
        context.auxiliary_state = actual_auxiliary


def test_configured_helpers_never_construct_source_objects_during_evaluation(
    source, monkeypatch
):
    name, _, state, context = source

    def forbidden(*args, **kwargs):
        raise AssertionError("Source construction entered repeated property evaluation")

    monkeypatch.setattr(CP, "AbstractState", forbidden)
    temperature = {"CO2": 270.0, "Water": 450.0, "Air": 100.0}[name]
    ordinary = core.compute_properties_coolprop(
        state,
        CP.PT_INPUTS,
        1e5,
        temperature,
        generalize_quality=True,
        supersaturation=True,
        source_context=context,
    )
    assert np.isfinite(ordinary["enthalpy"])
    assert np.isfinite(ordinary["quality_mass"])
    two_phase = core.compute_properties_coolprop(
        state,
        CP.QT_INPUTS,
        0.0 if name == "Air" else 0.3,
        temperature,
        supersaturation=True,
        source_context=context,
    )
    assert np.isfinite(two_phase["enthalpy"])


def test_diagnostics_reuse_fixed_bounds_without_reflashing_triple_endpoints(source):
    name, _, state, context = source
    updates = []
    actual_auxiliary = context.auxiliary_state

    class Counted:
        def update(self, *args):
            updates.append(args)
            return actual_auxiliary.update(*args)

        def __getattr__(self, key):
            if key in {"Ttriple", "T_critical", "p_critical", "rhomass_critical"}:
                raise AssertionError(
                    "Fixed source bounds were queried during evaluation"
                )
            return getattr(actual_auxiliary, key)

    context.auxiliary_state = Counted()
    try:
        temperature = {"CO2": 270.0, "Water": 450.0, "Air": 100.0}[name]
        pressure = max(1e5, 1.1 * context.triple_pressure_liquid)
        props = core.compute_properties_coolprop(
            state,
            CP.PT_INPUTS,
            pressure,
            temperature,
            supersaturation=True,
            source_context=context,
        )
        assert np.isfinite(props["pressure_saturation"])
        # Two bubble and one dew flashes at the requested pressure, plus the
        # requested-temperature bubble pressure. No repeated lower endpoints.
        assert updates == [
            (CP.QT_INPUTS, 0.0, temperature),
            (CP.PQ_INPUTS, props["pressure"], 0.0),
            (CP.PQ_INPUTS, props["pressure"], 0.0),
            (CP.PQ_INPUTS, props["pressure"], 1.0),
        ]
        updates.clear()
        props = core.compute_properties_coolprop(
            state,
            CP.PT_INPUTS,
            1.1 * context.critical_pressure,
            1.1 * context.critical_temperature,
            supersaturation=True,
            source_context=context,
        )
        assert np.isnan(props["temperature_saturation"])
        assert np.isnan(props["pressure_saturation"])
        assert updates == [
            (CP.DmassP_INPUTS, context.critical_density, props["pressure"]),
            (CP.DmassP_INPUTS, context.critical_density, props["pressure"]),
        ]
    finally:
        context.auxiliary_state = actual_auxiliary


@pytest.mark.parametrize("backend", get_available_backends())
def test_pseudo_bubble_temperature_uses_liquid_lower_endpoint(backend, monkeypatch):
    fluid = jxp.Fluid("Air", backend=backend)
    oracle = CP.AbstractState(backend, "Air")
    oracle.update(CP.QT_INPUTS, 0.0, oracle.Ttriple())
    p_liquid = oracle.p()
    oracle.update(CP.QT_INPUTS, 1.0, oracle.Ttriple())
    p_vapor = oracle.p()
    assert p_vapor < p_liquid
    pressure = (p_liquid + p_vapor) / 2
    updates = []
    auxiliary = fluid._source_context.auxiliary_state

    class Counted:
        def update(self, *args):
            updates.append(args)
            return auxiliary.update(*args)

        def __getattr__(self, key):
            return getattr(auxiliary, key)

    monkeypatch.setattr(fluid._source_context, "auxiliary_state", Counted())
    state = fluid.get_state(jxp.PT_INPUTS, pressure, 100.0, supersaturation=True)
    assert int(state.status) == 0
    assert np.isfinite(state.pressure_saturation)
    assert np.isnan(state.raw_property("temperature_saturation"))
    assert np.isnan(state.raw_property("supersaturation_degree"))
    # Do not attempt a bubble flash below its own lower pressure. The signed
    # subcooling helper instead clamps that flash to the bubble lower endpoint.
    assert (CP.PQ_INPUTS, float(state.p), 0.0) not in updates
    assert (CP.PQ_INPUTS, p_liquid, 0.0) in updates
    # Dew availability is independent; HEOS Air cannot solve every PQ request
    # even within its reported endpoint bounds, while REFPROP can here.
    try:
        oracle.update(CP.PQ_INPUTS, pressure, 1.0)
    except ValueError:
        assert np.isnan(state.raw_property("superheating"))
    else:
        np.testing.assert_allclose(state.superheating, 100.0 - oracle.T(), rtol=1e-10)


def test_prepared_context_preserves_ordinary_standalone_numerical_values(source):
    name, backend, state, context = source
    standalone = CP.AbstractState(backend, name)
    temperature = {"CO2": 270.0, "Water": 450.0, "Air": 100.0}[name]
    for pair, first, second in (
        (CP.PT_INPUTS, 1e5, temperature),
        (CP.QT_INPUTS, 0.0 if name == "Air" else 0.3, temperature),
    ):
        expected = core.compute_properties_coolprop(standalone, pair, first, second)
        actual = core.compute_properties_coolprop(
            state, pair, first, second, source_context=context
        )
        assert actual.keys() == expected.keys()
        for key in expected:
            if key in {"quality_mass", "quality_volume"} and np.isnan(expected[key]):
                # A prepared, reported fallback reference can supply the
                # critical entropy when a fresh exact critical flash cannot.
                assert context.references.critical.available
                assert actual[key] == float(
                    actual["entropy"] > context.critical_entropy
                )
                continue
            # REFPROP's saturation/FD responses depend on flash residuals and
            # its starting state. Keep primal core values strict; response
            # agreement remains substantially tighter than the 1e-6 full
            # source-oracle budget rather than claiming bitwise FD equality.
            tolerance = (
                1e-12
                if key
                in {
                    "pressure",
                    "temperature",
                    "density",
                    "enthalpy",
                    "entropy",
                    "internal_energy",
                }
                else 1e-8
            )
            np.testing.assert_allclose(
                actual[key],
                expected[key],
                rtol=tolerance,
                atol=1e-12,
                equal_nan=True,
                err_msg=key,
            )


@pytest.mark.parametrize("backend", get_available_backends())
@pytest.mark.parametrize(
    "name,pressure,temperature",
    [
        ("Water", 1e5, 300.0),
        ("Water", 1e5, 450.0),
        ("CO2", 1e6, 270.0),
        ("Air", 1e5, 100.0),
    ],
)
def test_subcritical_diagnostics_match_independent_bubble_and_dew_requests(
    backend, name, pressure, temperature
):
    fluid = jxp.Fluid(name, backend=backend)
    actual = fluid.get_state(jxp.PT_INPUTS, pressure, temperature, supersaturation=True)
    assert int(actual.status) == 0
    assert (
        set(
            (
                "subcooling",
                "superheating",
                "pressure_saturation",
                "temperature_saturation",
                "supersaturation_degree",
                "supersaturation_ratio",
            )
        )
        <= actual.available_properties
    )

    oracle = CP.AbstractState(backend, name)
    oracle.update(CP.QT_INPUTS, 0.0, temperature)
    p_sat = oracle.p()
    oracle.update(CP.PQ_INPUTS, pressure, 0.0)
    bubble_t, bubble_h, bubble_cp = oracle.T(), oracle.hmass(), oracle.cpmass()
    oracle.update(CP.PQ_INPUTS, pressure, 1.0)
    dew_t, dew_h, dew_cp = oracle.T(), oracle.hmass(), oracle.cpmass()
    expected_subcool = (
        (bubble_h - float(actual.h)) / bubble_cp
        if actual.h > bubble_h
        else bubble_t - temperature
    )
    expected_superheat = (
        (float(actual.h) - dew_h) / dew_cp if actual.h < dew_h else temperature - dew_t
    )
    expected = dict(
        pressure_saturation=p_sat,
        temperature_saturation=bubble_t,
        supersaturation_ratio=pressure / p_sat,
        supersaturation_degree=temperature - bubble_t,
        subcooling=expected_subcool,
        superheating=expected_superheat,
    )
    for key, value in expected.items():
        np.testing.assert_allclose(
            actual.raw_property(key), value, rtol=1e-10, atol=1e-9, err_msg=key
        )


@pytest.mark.parametrize("backend", get_available_backends())
@pytest.mark.parametrize(
    "pressure,temperature,has_p_sat,has_t_sat",
    [(1e6, 400.0, False, True), (1e7, 270.0, True, False), (1e7, 400.0, False, False)],
)
def test_independent_saturation_departure_availability_preserves_core_state(
    backend, pressure, temperature, has_p_sat, has_t_sat
):
    fluid = jxp.Fluid("CO2", backend=backend)
    state = fluid.get_state(jxp.PT_INPUTS, pressure, temperature, supersaturation=True)
    assert int(state.status) == 0 and bool(state.valid)
    for key in (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    ):
        assert np.isfinite(state.raw_property(key)), key
    for key in ("pressure_saturation", "supersaturation_ratio"):
        assert bool(state.property_valid(key)) == has_p_sat
    for key in ("temperature_saturation", "supersaturation_degree"):
        assert bool(state.property_valid(key)) == has_t_sat
    if not has_p_sat:
        with pytest.raises(jxp.PropertyUnavailableError):
            _ = state.pressure_saturation


@pytest.mark.parametrize("backend", get_available_backends())
def test_two_phase_diagnostics_use_the_requested_state_not_last_fd_perturbation(
    backend,
):
    fluid = jxp.Fluid("Water", backend=backend)
    pressure, quality = 1e5, 0.3
    state = fluid.get_state(jxp.PQ_INPUTS, pressure, quality, supersaturation=True)
    oracle = CP.AbstractState(backend, "Water")
    oracle.update(CP.PQ_INPUTS, pressure, quality)
    temperature, enthalpy = oracle.T(), oracle.hmass()
    oracle.update(CP.PQ_INPUTS, pressure, 0.0)
    bubble_h, bubble_cp = oracle.hmass(), oracle.cpmass()
    oracle.update(CP.PQ_INPUTS, pressure, 1.0)
    dew_h, dew_cp = oracle.hmass(), oracle.cpmass()
    np.testing.assert_allclose(state.p, pressure, rtol=1e-12)
    np.testing.assert_allclose(state.T, temperature, rtol=1e-12)
    np.testing.assert_allclose(state.pressure_saturation, pressure, rtol=1e-10)
    np.testing.assert_allclose(state.supersaturation_degree, 0.0, atol=1e-8)
    np.testing.assert_allclose(
        state.subcooling, (bubble_h - enthalpy) / bubble_cp, rtol=1e-10
    )
    np.testing.assert_allclose(
        state.superheating, (enthalpy - dew_h) / dew_cp, rtol=1e-10
    )


def test_optional_heat_capacity_failure_does_not_invalidate_single_phase_core(source):
    name, _, state, context = source

    class WithoutHeatCapacity:
        def cpmass(self):
            raise ValueError("optional response is undefined")

        def __getattr__(self, key):
            return getattr(state, key)

    temperature = {"CO2": 400.0, "Water": 500.0, "Air": 300.0}[name]
    props = core.compute_properties_coolprop(
        WithoutHeatCapacity(), CP.PT_INPUTS, 1e5, temperature, source_context=context
    )
    assert np.isnan(props["isobaric_heat_capacity"])
    assert np.isnan(props["heat_capacity_ratio"])
    assert np.isnan(props["joule_thomson"])
    for key in (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    ):
        assert np.isfinite(props[key]), key


@pytest.mark.parametrize("backend", get_available_backends())
@pytest.mark.parametrize("method", ["coolprop", "coolprop_jax"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
@pytest.mark.parametrize(
    "getter,property_name",
    [
        ("cpmass", "isobaric_heat_capacity"),
        ("cvmass", "isochoric_heat_capacity"),
        ("speed_sound", "speed_of_sound"),
        ("conductivity", "conductivity"),
        ("viscosity", "viscosity"),
    ],
)
def test_missing_optional_endpoint_getter_preserves_two_phase_state(
    backend, method, error_type, getter, property_name
):
    fluid = jxp.Fluid("Water", method=method, backend=backend)
    source = fluid.fluid if method == "coolprop_jax" else fluid
    auxiliary = source._source_context.auxiliary_state

    class WithoutResponse:
        def __getattr__(self, key):
            if key == getter:

                def unavailable():
                    raise error_type("optional endpoint property is undefined")

                return unavailable
            return getattr(auxiliary, key)

    source._source_context.auxiliary_state = WithoutResponse()
    for errors in ("raise", "mask"):
        state = fluid.get_state(jxp.PQ_INPUTS, 1e5, 0.3, errors=errors)
        assert int(state.status) == 0
        assert not state.property_valid(property_name)
        assert np.isnan(state.raw_property(property_name))
        for name in (
            "pressure",
            "temperature",
            "density",
            "enthalpy",
            "entropy",
            "internal_energy",
        ):
            assert np.isfinite(state.raw_property(name)), name


@pytest.mark.parametrize("backend", get_available_backends())
@pytest.mark.parametrize("method", ["coolprop", "coolprop_jax"])
@pytest.mark.parametrize("getter", ["conductivity", "viscosity"])
@pytest.mark.parametrize(
    "error_type", [RuntimeError, NotImplementedError, AttributeError]
)
def test_missing_transport_getter_preserves_single_phase_state(
    backend, method, getter, error_type
):
    fluid = jxp.Fluid("Water", method=method, backend=backend)
    source = fluid.fluid if method == "coolprop_jax" else fluid
    primary = source._AS

    class WithoutTransport:
        def __getattr__(self, key):
            if key == getter:

                def unavailable():
                    raise error_type("transport model is unavailable")

                return unavailable
            return getattr(primary, key)

    source._AS = WithoutTransport()
    for errors in ("raise", "mask"):
        state = fluid.get_state(jxp.PT_INPUTS, 1e5, 450.0, errors=errors)
        assert int(state.status) == 0
        assert np.isnan(state.raw_property(getter))
        assert np.isfinite(state.h) and np.isfinite(state.s) and np.isfinite(state.d)


@pytest.mark.parametrize("backend", get_available_backends())
@pytest.mark.parametrize("method", ["coolprop", "coolprop_jax"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
@pytest.mark.parametrize(
    "helper,property_name",
    [
        ("compute_dsdp_q", "speed_of_sound"),
        ("compute_gruneisen", "gruneisen"),
        ("compute_isothermal_joule_thomson", "isothermal_joule_thomson"),
        ("compute_joule_thomson", "joule_thomson"),
    ],
)
def test_missing_two_phase_response_helper_preserves_core_and_call_order(
    backend, method, error_type, helper, property_name, monkeypatch
):
    fluid = jxp.Fluid("Water", method=method, backend=backend)
    calls = []
    for name in (
        "compute_dsdp_q",
        "compute_gruneisen",
        "compute_isothermal_joule_thomson",
        "compute_joule_thomson",
    ):
        original = getattr(core, name)

        def recorded(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            if _name == helper:
                raise error_type("optional two-phase response is unavailable")
            return _original(*args, **kwargs)

        monkeypatch.setattr(core, name, recorded)
    expected_order = [
        "compute_dsdp_q",
        "compute_dsdp_q",
        "compute_gruneisen",
        "compute_isothermal_joule_thomson",
        "compute_joule_thomson",
    ]
    for errors in ("raise", "mask"):
        calls.clear()
        state = fluid.get_state(jxp.PQ_INPUTS, 1e5, 0.3, errors=errors)
        assert int(state.status) == 0
        assert not state.property_valid(property_name)
        assert np.isnan(state.raw_property(property_name))
        for name in (
            "pressure",
            "temperature",
            "density",
            "enthalpy",
            "entropy",
            "internal_energy",
        ):
            assert np.isfinite(state.raw_property(name)), name
        assert calls == expected_order


def test_old_source_diagnostics_and_quality_keep_their_convention_after_global_changes():
    repo = Path(__file__).resolve().parents[1]
    script = r"""
import CoolProp.CoolProp as CP
import jax
import numpy as np
import jaxprop as p
source = p.Fluid('CO2')
callback = p.Fluid('CO2',method='coolprop_jax')
requests = [(p.PT_INPUTS,1e5,500.),(p.PT_INPUTS,1e6,270.),(p.PT_INPUTS,1e7,400.),(p.PQ_INPUTS,1e6,.3)]
before = [source.get_state(*request,generalize_quality=True,supersaturation=True) for request in requests]
compiled = [jax.jit(lambda a,b,pair=pair: callback.get_state(pair,a,b,generalize_quality=True,supersaturation=True,errors='mask')) for pair,_,_ in requests]
before_callback = [evaluate(a,b) for evaluate,(_,a,b) in zip(compiled,requests)]
jax.block_until_ready(before_callback)
identity = source.reference_id
CP.set_reference_state('CO2',300.,500.,12345.,1e6)
for expected,request in zip(before,requests):
 actual = source.get_state(*request,generalize_quality=True,supersaturation=True)
 assert actual.reference_id==identity
 assert actual.available_properties==expected.available_properties
 assert int(actual.status)==0
 for key in expected.available_properties:
  np.testing.assert_allclose(actual.raw_property(key),expected.raw_property(key),rtol=1e-12,atol=1e-10,equal_nan=True,err_msg=key)
for evaluate,expected,(_,a,b) in zip(compiled,before_callback,requests):
 actual = evaluate(a,b)
 assert actual.reference_id==identity
 assert actual.available_properties==expected.available_properties
 assert int(actual.status)==0
 for key in expected.available_properties:
  np.testing.assert_allclose(actual.raw_property(key),expected.raw_property(key),rtol=1e-12,atol=1e-10,equal_nan=True,err_msg=key)
assert p.Fluid('CO2').reference_id!=identity
print('reference-compatible diagnostics retained')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reference-compatible diagnostics retained" in result.stdout
