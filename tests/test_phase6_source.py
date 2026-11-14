"""Numerical policy for direct CoolProp calls, independent of native models."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import CoolProp.CoolProp as CP
import numpy as np
import pytest

import jaxprop as jxp
from utilities import get_available_backends, get_reference_state

module = importlib.import_module("jaxprop.coolprop.fluid_properties")
CORE = ("pressure", "temperature", "density", "enthalpy", "entropy", "internal_energy")


@pytest.fixture(scope="module", params=get_available_backends())
def fluid(request):
    return jxp.Fluid("CO2", backend=request.param, identifier="source-test")


def assert_success(state, fluid, shape):
    assert state.shape == shape
    assert state.fluid_name == fluid.name
    assert state.identifier == fluid.identifier
    assert state.provider == "coolprop"
    assert state.reference_id == fluid.reference_id
    assert state.status.dtype == np.dtype(np.int32)
    assert all(np.all(np.isfinite(state.raw_property(key))) for key in CORE)


@pytest.mark.parametrize("legacy", [False, np.bool_(False), 0, None, "yes"])
def test_failure_suppression_is_rejected_before_source_allocation(monkeypatch, legacy):
    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected configuration allocated CoolProp")

    monkeypatch.setattr(CP, "AbstractState", forbidden)
    with pytest.raises(jxp.InvalidConfigurationError) as captured:
        jxp.FluidCoolProp("CO2", exceptions=legacy)
    assert captured.value.request == "exceptions"
    assert "errors='mask'" in captured.value.reason


@pytest.mark.parametrize(
    "pressure,temperature,shape",
    [
        (1e5, 500.0, ()),
        ([1e5], 500.0, (1,)),
        ([[1e5], [2e5]], [450.0, 500.0, 550.0], (2, 3)),
        (np.empty((0, 1)), np.ones((1, 3)) * 500.0, (0, 3)),
    ],
)
@pytest.mark.parametrize("errors", ["raise", "mask"])
def test_ordinary_shapes_and_error_policy_preserve_identity(
    fluid, pressure, temperature, shape, errors
):
    state = fluid.get_state(jxp.PT_INPUTS, pressure, temperature, errors=errors)
    assert_success(state, fluid, shape)
    assert state.error_policy == errors
    np.testing.assert_array_equal(state.status, np.zeros(shape, dtype=np.int32))
    assert all(
        state.raw_property(key).shape == shape for key in state.available_properties
    )
    assert not state.status.flags.writeable


@pytest.mark.parametrize(
    "pair,first,second",
    [
        (jxp.PT_INPUTS, -1.0, 500.0),
        (jxp.PT_INPUTS, 1e5, 0.0),
        (jxp.HmassP_INPUTS, 700e3, -1.0),
        (jxp.PSmass_INPUTS, -1.0, 2000.0),
        (jxp.DmassP_INPUTS, 0.0, 1e5),
        (jxp.DmassT_INPUTS, 1.0, -1.0),
        (jxp.DmassSmass_INPUTS, -1.0, 2000.0),
        (jxp.DmassHmass_INPUTS, -1.0, 700e3),
        (jxp.DmassUmass_INPUTS, -1.0, -700e3),
        (jxp.SmassT_INPUTS, -1000.0, -1.0),
        (jxp.QT_INPUTS, -0.01, 270.0),
        (jxp.QT_INPUTS, 1.01, 270.0),
        (jxp.PQ_INPUTS, 1e6, -0.01),
        (jxp.HmassSmass_INPUTS, np.nan, -1000.0),
        (jxp.DmassUmass_INPUTS, 1.0, np.inf),
    ],
)
def test_pair_specific_invalid_values_raise_or_mask(fluid, pair, first, second):
    with pytest.raises(jxp.InvalidStateError) as captured:
        fluid.get_state(pair, first, second)
    assert type(captured.value) is jxp.InvalidStateError
    assert captured.value.index == ()
    assert captured.value.provider == "coolprop"
    assert captured.value.reason
    state = fluid.get_state(pair, first, second, errors="mask")
    assert state.status == jxp.StatusCode.INVALID_INPUT
    assert all(np.isnan(state.raw_property(key)) for key in state.available_properties)


def test_mask_mode_preserves_successful_elements(fluid):
    pressure = np.array([[1e5, -1.0], [np.nan, 2e5]])
    state = fluid.get_state(jxp.PT_INPUTS, pressure, 500.0, errors="mask")
    np.testing.assert_array_equal(state.status, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(state.valid, [[True, False], [False, True]])
    for key in state.available_properties:
        values = state.raw_property(key)
        assert np.all(np.isnan(values[~state.valid]))
    for index in ((0, 0), (1, 1)):
        reference = fluid.get_state(jxp.PT_INPUTS, pressure[index], 500.0)
        for key in CORE:
            np.testing.assert_array_equal(
                state.raw_property(key)[index], reference.raw_property(key)
            )
    with pytest.raises(jxp.InvalidStateError) as captured:
        fluid.get_state(jxp.PT_INPUTS, pressure, 500.0)
    assert captured.value.index == (0, 1)


def test_provider_failure_before_later_invalid_input_preserves_row_major_order(
    fluid, monkeypatch
):
    original = module.core.compute_properties_coolprop

    def fail_first(state, pair, first, second, **options):
        if first == 1e5:
            raise RuntimeError("injected source failure")
        return original(state, pair, first, second, **options)

    monkeypatch.setattr(module.core, "compute_properties_coolprop", fail_first)
    pressure = np.array([[1e5, -1.0], [2e5, 3e5]])
    with pytest.raises(jxp.CalculationError) as captured:
        fluid.get_state(jxp.PT_INPUTS, pressure, 500.0)
    assert captured.value.index == (0, 0)
    assert isinstance(captured.value.__cause__, RuntimeError)
    state = fluid.get_state(jxp.PT_INPUTS, pressure, 500.0, errors="mask")
    np.testing.assert_array_equal(state.status, [[4, 1], [0, 0]])
    assert all(np.all(np.isfinite(state.raw_property(key)[1])) for key in CORE)


def test_explicit_solver_failure_has_its_own_status(fluid, monkeypatch):
    def failed(*args, **kwargs):
        raise jxp.ConvergenceError(request="inner", reason="injected solver failure")

    monkeypatch.setattr(module.core, "compute_properties_coolprop", failed)
    with pytest.raises(jxp.ConvergenceError) as captured:
        fluid.get_state(jxp.PT_INPUTS, 1e5, 500.0)
    assert captured.value.index == ()
    assert isinstance(captured.value.__cause__, jxp.ConvergenceError)
    assert fluid.get_state(jxp.PT_INPUTS, 1e5, 500.0, errors="mask").status == 3


@pytest.mark.parametrize("field", ["temperature", "pressure", "density", "enthalpy"])
def test_invalid_core_output_is_a_provider_failure(fluid, monkeypatch, field):
    original = module.core.compute_properties_coolprop

    def invalid_core(*args, **kwargs):
        result = original(*args, **kwargs)
        result[field] = np.nan
        return result

    monkeypatch.setattr(module.core, "compute_properties_coolprop", invalid_core)
    with pytest.raises(jxp.CalculationError) as captured:
        fluid.get_state(jxp.PT_INPUTS, 1e5, 500.0)
    assert isinstance(captured.value.__cause__, ValueError)
    state = fluid.get_state(jxp.PT_INPUTS, 1e5, 500.0, errors="mask")
    assert state.status == 4
    assert all(np.isnan(state.raw_property(key)) for key in state.available_properties)


def test_optional_unavailability_does_not_fail_a_valid_state(fluid, monkeypatch):
    original = module.core.compute_properties_coolprop

    def missing_optional(*args, **kwargs):
        result = original(*args, **kwargs)
        result["conductivity"] = np.nan
        return result

    monkeypatch.setattr(module.core, "compute_properties_coolprop", missing_optional)
    state = fluid.get_state(jxp.PT_INPUTS, 1e5, 500.0)
    assert state.status == 0
    assert not state.property_valid("conductivity")
    assert all(np.isfinite(state.raw_property(key)) for key in CORE)


def test_known_source_domain_has_distinct_failure_status(fluid):
    cases = [
        (jxp.PT_INPUTS, fluid.p_max * 1.01, 500.0),
        (jxp.PT_INPUTS, 1e5, fluid.T_min - 1.0),
        (jxp.PT_INPUTS, 1e5, fluid.T_max + 1.0),
        (
            jxp.QT_INPUTS,
            0.0,
            fluid.references["critical_point"].reported_coordinates["temperature"]
            + 1.0,
        ),
        (
            jxp.PQ_INPUTS,
            fluid.references["critical_point"].reported_coordinates["pressure"] * 1.01,
            0.0,
        ),
    ]
    for pair, first, second in cases:
        with pytest.raises(jxp.OutOfDomainError) as captured:
            fluid.get_state(pair, first, second)
        assert captured.value.index == ()
        assert fluid.get_state(pair, first, second, errors="mask").status == 2


def test_source_boundary_inputs_accept_roundoff_and_preserve_coordinates(fluid):
    cases = [
        (1e5, np.nextafter(fluid.T_min, np.inf)),
        (1e5, np.nextafter(fluid.T_max, -np.inf)),
        (1e5, np.nextafter(fluid.T_max, np.inf)),
        (np.nextafter(fluid.p_max, -np.inf), 500.0),
        (np.nextafter(fluid.p_max, np.inf), 500.0),
    ]
    backend = CP.AbstractState(fluid.backend, fluid.name)
    for pressure, temperature in cases:
        state = fluid.get_state(jxp.PT_INPUTS, pressure, temperature)
        assert_success(state, fluid, ())
        assert state.status == 0
        # Compare with the raw provider, rather than silently replacing a
        # coordinate by its advertised boundary before the EOS evaluation.
        backend.update(CP.PT_INPUTS, pressure, temperature)
        np.testing.assert_allclose(float(state.p), backend.p(), rtol=1e-12)
        assert float(state.T) == backend.T() == temperature


def test_water_saturation_accepts_one_ulp_below_reported_triple_temperature():
    source = jxp.Fluid("Water", backend="HEOS")
    temperature = np.nextafter(source.T_min, -np.inf)
    state = source.get_state(jxp.QT_INPUTS, 0.0, temperature)
    assert_success(state, source, ())
    assert state.status == 0
    assert float(state.T) == temperature < source.T_min


@pytest.mark.parametrize("backend", get_available_backends())
@pytest.mark.parametrize(
    "name,pair", [("Cyclopentane", jxp.HmassP_INPUTS), ("R245fa", jxp.PSmass_INPUTS)]
)
def test_inverse_boundary_coordinates_can_be_reused_without_clipping(
    backend, name, pair
):
    source = jxp.Fluid(name, backend=backend)
    reference = get_reference_state(name, backend, "supercritical_gas")
    first = float(reference.h) if pair == jxp.HmassP_INPUTS else float(reference.p)
    second = float(reference.p) if pair == jxp.HmassP_INPUTS else float(reference.s)
    state = source.get_state(pair, first, second)
    assert_success(state, source, ())
    assert state.status == 0
    recovered = source.get_state(jxp.DmassT_INPUTS, float(state.d), float(state.T))
    assert_success(recovered, source, ())
    assert recovered.status == 0
    assert float(recovered.T) == float(state.T)
    np.testing.assert_allclose(float(recovered.T), float(reference.T), rtol=1e-9)


def test_boundary_acceptance_does_not_admit_meaningful_extrapolation(
    fluid, monkeypatch
):
    for temperature in (fluid.T_min - 10.0, fluid.T_max + 10.0):
        with pytest.raises(jxp.OutOfDomainError):
            fluid.get_state(jxp.PT_INPUTS, 1e5, temperature)
        assert (
            fluid.get_state(jxp.PT_INPUTS, 1e5, temperature, errors="mask").status == 2
        )

    original = module.core.compute_properties_coolprop

    def extrapolated_inverse(*args, **kwargs):
        result = original(*args, **kwargs)
        result["temperature"] = fluid.T_max + 10.0
        return result

    monkeypatch.setattr(
        module.core, "compute_properties_coolprop", extrapolated_inverse
    )
    with pytest.raises(jxp.OutOfDomainError):
        fluid.get_state(jxp.HmassP_INPUTS, 700e3, 1e5)
    state = fluid.get_state(jxp.HmassP_INPUTS, 700e3, 1e5, errors="mask")
    assert state.status == 2
    assert all(np.isnan(state.raw_property(key)) for key in state.available_properties)


@pytest.mark.parametrize(
    "pair,first,second",
    [(jxp.PQ_INPUTS, 1e6, [0.0, 1.0]), (jxp.QT_INPUTS, [0.0, 1.0], 270.0)],
)
def test_uniform_saturation_entry_point_matches_ordinary_evaluation(
    fluid, pair, first, second
):
    actual = fluid.get_state_saturation(pair, first, second)
    ordinary = fluid.get_state(pair, first, second)
    assert_success(actual, fluid, (2,))
    for key in CORE:
        np.testing.assert_array_equal(
            actual.raw_property(key), ordinary.raw_property(key)
        )
    assert "saturation" in fluid.capabilities.phase_operations


@pytest.mark.parametrize("pair", [jxp.PT_INPUTS, True, "QT_INPUTS", [jxp.QT_INPUTS]])
def test_wrong_saturation_pair_is_an_informative_operation_error(fluid, pair):
    with pytest.raises(jxp.UnsupportedOperationError) as captured:
        fluid.get_state_saturation(pair, 1e6, 0.0)
    assert captured.value.request == "saturation"
    assert set(captured.value.supported) == {"PQ_INPUTS", "QT_INPUTS"}


def test_private_batches_are_atomic_for_shared_source_objects(fluid):
    pressures = np.linspace(1e5, 4e5, 12)

    def evaluate(offset):
        p = pressures + offset
        props, status = fluid._evaluate_state_data(
            jxp.PT_INPUTS, p, np.full(p.shape, 500.0), errors="mask"
        )
        np.testing.assert_array_equal(status, np.zeros(p.shape, dtype=np.int32))
        np.testing.assert_allclose(props["pressure"], p, rtol=1e-10, atol=1e-6)
        return props["density"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(evaluate, (0.0, 1e4, 2e4, 3e4)))
    for offset, values in zip((0.0, 1e4, 2e4, 3e4), actual):
        expected = fluid.get_state(jxp.PT_INPUTS, pressures + offset, 500.0)
        np.testing.assert_allclose(values, expected.raw_property("density"), rtol=1e-12)


def test_negative_energy_and_entropy_inputs_remain_valid():
    repo = Path(__file__).resolve().parents[1]
    program = """
import json
import CoolProp.CoolProp as CP
import jaxprop as p
CP.set_reference_state('CO2', 300.0, 500.0, -10000.0, -100.0)
f=p.Fluid('CO2')
reference=f.get_state(p.DmassT_INPUTS,500.0,300.0)
states=[f.get_state(p.HmassP_INPUTS,float(reference.h),float(reference.p)),f.get_state(p.DmassUmass_INPUTS,float(reference.d),float(reference.u))]
print(json.dumps({'h':float(reference.h),'s':float(reference.s),'u':float(reference.u),'status':[int(state.status) for state in states],'temperatures':[float(state.T) for state in states]}))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    values = json.loads(result.stdout.strip().splitlines()[-1])
    assert values["h"] < 0 and values["s"] < 0 and values["u"] < 0
    assert values["status"] == [0, 0]
    np.testing.assert_allclose(values["temperatures"], [300.0, 300.0], rtol=1e-10)


@pytest.mark.parametrize("name", ["Air", "R410A"])
def test_heos_pseudo_pure_qt_endpoint_restriction_is_not_applied_to_pq(name):
    source = jxp.Fluid(name, backend="HEOS")
    with pytest.raises(jxp.OutOfDomainError) as captured:
        source.get_state(jxp.QT_INPUTS, 0.3, 100.0 if name == "Air" else 270.0)
    assert "quality_mass 0 or 1" in captured.value.reason
    temperature = 100.0 if name == "Air" else 270.0
    state = source.get_state(jxp.QT_INPUTS, [0.0, 0.3, 1.0], temperature, errors="mask")
    np.testing.assert_array_equal(state.status, [0, 2, 0])
    assert (
        "QT flashes accept only quality endpoints"
        in source.capabilities.domain.description
    )
    # This verifies the backend flash/core state, not glide-sensitive responses.
    state = source.get_state(jxp.PQ_INPUTS, 1e6, 0.3)
    assert_success(state, source, ())


@pytest.mark.parametrize("name", ["Air", "R410A"])
def test_heos_pseudo_pure_pq_recovers_backend_mass_quality(name):
    backend = CP.AbstractState("HEOS", name)
    backend.update(CP.PQ_INPUTS, 1e6, 0.3)
    expected_quality = backend.Q()
    np.testing.assert_allclose(expected_quality, 0.3, rtol=1e-12)
    source = jxp.Fluid(name, backend="HEOS")
    state = source.get_state(jxp.PQ_INPUTS, 1e6, 0.3)
    assert_success(state, source, ())
    assert state.status == 0
    np.testing.assert_allclose(float(state.quality_mass), expected_quality, rtol=1e-10)
