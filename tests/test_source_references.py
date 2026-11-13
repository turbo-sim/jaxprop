"""Source-reference records are independent of the numerical fluid evaluator."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import warnings

import CoolProp.CoolProp as CP
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp
from utilities import get_available_backends

references_module = importlib.import_module("jaxprop.source_references")
POINT_NAMES = ("critical_point", "triple_point_liquid", "triple_point_vapor")
PROVIDERS = ("coolprop", "coolprop_jax", "bicubic", "perfect_gas")
FLUID_NAMES = ("CO2", "Water", "Air")
CORE_PROPERTIES = (
    "pressure",
    "temperature",
    "density",
    "enthalpy",
    "entropy",
    "internal_energy",
)


def assert_primitive_metadata(value):
    """Static records must never retain NumPy/JAX arrays, even nested ones."""
    assert not isinstance(value, (np.ndarray, jax.Array)), type(value)
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_primitive_metadata(key)
            assert_primitive_metadata(item)
    elif is_dataclass(value):
        for field in fields(value):
            assert_primitive_metadata(getattr(value, field.name))
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            assert_primitive_metadata(item)


@pytest.fixture(scope="module")
def fluid_models(tmp_path_factory):
    directory = tmp_path_factory.mktemp("phase5_sources")
    models, construction_warnings = {}, {}
    for name, temperature in (("CO2", 600.0), ("Water", 850.0), ("Air", 450.0)):
        source = CP.AbstractState("HEOS", name)
        enthalpies = []
        for pressure in (5e6, 12e6):
            for sample_temperature in (temperature - 30, temperature + 30):
                source.update(CP.PT_INPUTS, pressure, sample_temperature)
                enthalpies.append(source.hmass())
        table_options = dict(
            h_min=min(enthalpies),
            h_max=max(enthalpies),
            p_min=5e6,
            p_max=12e6,
            N_h=4,
            N_p=4,
            N_p_sat=4,
            coarse_step=1,
            table_dir=str(directory / name),
        )
        for provider in PROVIDERS:
            options = dict(name=name, identifier="source-reference-test")
            if provider == "perfect_gas":
                options.update(
                    provider="native",
                    model="perfect_gas",
                    model_options={"T_ref": temperature, "p_ref": 1e5},
                )
            else:
                options["provider"] = provider
                if provider == "bicubic":
                    options["table_options"] = table_options
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                models[name, provider] = jxp.create_fluid(**options)
            construction_warnings[name, provider] = tuple(
                str(warning.message) for warning in caught
            )
    return models, construction_warnings


@pytest.mark.parametrize("name", FLUID_NAMES)
@pytest.mark.parametrize("provider", PROVIDERS)
def test_all_evaluators_publish_the_same_source_points(name, provider, fluid_models):
    models, _ = fluid_models
    fluid = models[name, provider]
    source = models[name, "coolprop"]
    assert fluid.references.reference_id == source.references.reference_id
    assert fluid.references == source.references
    assert hash(fluid.references) == hash(source.references)
    assert set(fluid.references) == set(POINT_NAMES)
    assert_primitive_metadata(fluid.references)
    for point_name in POINT_NAMES:
        record = fluid.references[point_name]
        assert record.available, (point_name, record.reason)
        assert record.backend == "HEOS"
        assert (
            record.reference_convention
            == source.references[point_name].reference_convention
        )
        assert record.source["coolprop_version"] == CP.get_global_param_string(
            "version"
        )
        state = getattr(fluid, point_name)
        assert isinstance(state, jxp.FluidState)
        assert state.model == "real_fluid" and state.provider == "coolprop"
        assert state.reference_id == fluid.references.reference_id
        for property_name in CORE_PROPERTIES:
            actual = state.raw_property(property_name)
            expected = getattr(source, point_name).raw_property(property_name)
            assert actual.shape == () and np.isfinite(actual)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name", FLUID_NAMES)
@pytest.mark.parametrize("provider", PROVIDERS[1:])
def test_reference_metadata_does_not_introduce_static_array_warnings(
    name, provider, fluid_models
):
    _, construction_warnings = fluid_models
    bad = [
        message
        for message in construction_warnings[name, provider]
        if "array" in message.lower() and "static" in message.lower()
    ]
    assert not bad, bad


@pytest.mark.parametrize("backend", get_available_backends())
@pytest.mark.parametrize("name", FLUID_NAMES)
def test_reported_coordinates_and_reference_values_match_the_selected_source(
    backend, name
):
    references = references_module.initialize_source_references(name, backend)
    source = CP.AbstractState(backend, name)
    critical = references["critical_point"]
    assert critical.available, critical.reason
    assert critical.classification == ("pseudo_pure" if name == "Air" else "pure")
    for property_name, expected in (
        ("pressure", source.p_critical()),
        ("temperature", source.T_critical()),
        ("density", source.rhomass_critical()),
    ):
        np.testing.assert_allclose(
            critical.reported_coordinates[property_name], expected, rtol=1e-12
        )
    for point_name in POINT_NAMES:
        point = references[point_name]
        assert point.available, point.reason
        state = point.state
        coordinates = point.evaluation_coordinates
        if point_name == "critical_point":
            source.update(
                CP.DmassT_INPUTS, coordinates["density"], coordinates["temperature"]
            )
        else:
            # Reproduce the independent saturation request. DmassT can choose
            # a different branch near an endpoint, and REFPROP pseudo-pure
            # models reject that two-phase input route altogether.
            source.update(
                CP.QT_INPUTS,
                0.0 if point_name == "triple_point_liquid" else 1.0,
                coordinates["temperature"],
            )
        for property_name, expected in (
            ("pressure", source.p()),
            ("temperature", source.T()),
            ("density", source.rhomass()),
            ("enthalpy", source.hmass()),
            ("entropy", source.smass()),
            ("internal_energy", source.umass()),
        ):
            np.testing.assert_allclose(
                state.raw_property(property_name), expected, rtol=1e-10
            )
        for coordinate, reported in point.reported_coordinates.items():
            if coordinate in coordinates and reported is not None:
                np.testing.assert_allclose(
                    point.evaluation_offsets[coordinate],
                    coordinates[coordinate] - reported,
                    rtol=1e-12,
                    atol=1e-12,
                )


@pytest.mark.parametrize("name", ["Air", "R410A"])
def test_pseudo_pure_endpoints_preserve_source_temperature_and_distinct_pressures(name):
    references = references_module.initialize_source_references(name)
    source = CP.AbstractState("HEOS", name)
    liquid, vapor = (references[key] for key in POINT_NAMES[1:])
    assert liquid.available and vapor.available
    assert liquid.classification == vapor.classification == "pseudo_pure"
    assert liquid.interpretation == vapor.interpretation == "lower_saturation_endpoint"
    for record, quality in ((liquid, 0.0), (vapor, 1.0)):
        source.update(CP.QT_INPUTS, quality, source.Ttriple())
        np.testing.assert_allclose(
            [record.state.p, record.state.T], [source.p(), source.T()], rtol=1e-12
        )
    assert float(liquid.state.T) == float(vapor.state.T)
    assert not np.isclose(liquid.state.p, vapor.state.p)


def test_reference_records_are_immutable_pickleable_and_return_isolated_states():
    references = references_module.initialize_source_references("CO2")
    record = references["critical_point"]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        record.reason = "changed"
    with pytest.raises(TypeError):
        record.properties["temperature"] = 1.0
    with pytest.raises(TypeError):
        references.source["backend"] = "other"
    restored = pickle.loads(pickle.dumps(references))
    assert restored == references and hash(restored) == hash(references)
    first, second = record.state, record.state
    assert first is not second
    for property_name in first.available_properties:
        value = first.raw_property(property_name)
        assert isinstance(value, np.ndarray) and not value.flags.writeable
        np.testing.assert_array_equal(value, second.raw_property(property_name))


def test_aliases_share_a_physical_reference_identity_without_rewriting_user_names():
    alias = jxp.Fluid("CO2", identifier="one")
    canonical = jxp.Fluid("CarbonDioxide", identifier="two")
    assert alias.name == "CO2" and canonical.name == "CarbonDioxide"
    assert alias.references.reference_id == canonical.references.reference_id
    assert alias.reference_id == canonical.reference_id
    assert alias.references.source["fluid_names"] == ("CarbonDioxide",)
    first = alias.get_state(jxp.PT_INPUTS, 1e5, 500.0)
    second = canonical.get_state(jxp.PT_INPUTS, 1e5, 500.0)
    assert first.identifier == "one" and second.identifier == "two"


def test_perfect_gas_identity_includes_calibration_without_splitting_source_identity():
    ordinary = jxp.FluidPerfectGas("CO2", T_ref=600.0, p_ref=1e5)
    alias = jxp.FluidPerfectGas("CarbonDioxide", T_ref=600.0, p_ref=1e5)
    changed_temperature = jxp.FluidPerfectGas("CO2", T_ref=650.0, p_ref=1e5)
    changed_pressure = jxp.FluidPerfectGas("CO2", T_ref=600.0, p_ref=2e5)
    assert ordinary.reference_id == alias.reference_id
    assert (
        len(
            {
                ordinary.reference_id,
                changed_temperature.reference_id,
                changed_pressure.reference_id,
            }
        )
        == 3
    )
    assert (
        len(
            {
                fluid.references.reference_id
                for fluid in (ordinary, alias, changed_temperature, changed_pressure)
            }
        )
        == 1
    )


@pytest.mark.parametrize("missing", POINT_NAMES)
def test_one_unavailable_reference_does_not_remove_the_other_points(
    monkeypatch, missing
):
    constructor = CP.AbstractState

    class MissingPoint:
        def __init__(self, backend, name):
            self.wrapped = constructor(backend, name)

        def T_critical(self):
            if missing == "critical_point":
                raise ValueError("test critical coordinates unavailable")
            return self.wrapped.T_critical()

        def update(self, input_pair, first, second):
            if input_pair == CP.QT_INPUTS and (
                (missing == "triple_point_liquid" and first == 0.0)
                or (missing == "triple_point_vapor" and first == 1.0)
            ):
                raise ValueError("test endpoint unavailable")
            return self.wrapped.update(input_pair, first, second)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    monkeypatch.setattr(CP, "AbstractState", MissingPoint)
    references = references_module.initialize_source_references("CO2")
    for name in POINT_NAMES:
        record = references[name]
        if name == missing:
            assert not record.available and "test" in record.reason
            with pytest.raises(jxp.UnavailableReferencePointError) as captured:
                _ = record.state
            assert captured.value.request == name
            assert captured.value.reason == record.reason
        else:
            assert record.available, (name, record.reason)
    fluid = jxp.Fluid("CO2")
    ordinary = fluid.get_state(jxp.PT_INPUTS, 1e5, 500.0)
    assert all(np.isfinite(ordinary.raw_property(key)) for key in CORE_PROPERTIES)
    if missing == "critical_point":
        assert ordinary.supports_property("quality_mass")
        assert np.isnan(ordinary.raw_property("quality_mass"))
        assert np.isnan(ordinary.raw_property("quality_volume"))
        with pytest.raises(jxp.PropertyUnavailableError):
            _ = ordinary.Q
    with pytest.raises(jxp.UnavailableReferencePointError) as captured:
        getattr(fluid, missing)
    assert captured.value.provider == "coolprop"
    assert captured.value.request == missing


def test_locally_undefined_optional_properties_do_not_hide_a_reference(monkeypatch):
    constructor = CP.AbstractState

    class WithoutHeatCapacity:
        def __init__(self, backend, name):
            self.wrapped = constructor(backend, name)

        def cpmass(self):
            raise ValueError("test optional property undefined")

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    monkeypatch.setattr(CP, "AbstractState", WithoutHeatCapacity)
    references = references_module.initialize_source_references("CO2")
    for point_name in POINT_NAMES:
        record = references[point_name]
        assert record.available, record.reason
        state = record.state
        assert np.isnan(state.raw_property("isobaric_heat_capacity"))
        assert not state.property_valid("isobaric_heat_capacity")
        with pytest.raises(jxp.PropertyUnavailableError):
            _ = state.cp
        assert all(np.isfinite(state.raw_property(key)) for key in CORE_PROPERTIES)


def test_critical_fallback_records_actual_offset_without_claiming_singular_properties(
    monkeypatch,
):
    constructor = CP.AbstractState
    source = constructor("HEOS", "CO2")
    reported_temperature = source.T_critical()
    reported_density = source.rhomass_critical()
    rejected_exact_requests = []

    class WithoutExactCriticalEvaluation:
        def __init__(self, backend, name):
            self.wrapped = constructor(backend, name)

        def update(self, input_pair, first, second):
            if (
                input_pair == CP.DmassT_INPUTS
                and first == reported_density
                and second == reported_temperature
            ):
                rejected_exact_requests.append((first, second))
                raise ValueError("test exact critical evaluation unavailable")
            return self.wrapped.update(input_pair, first, second)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    monkeypatch.setattr(CP, "AbstractState", WithoutExactCriticalEvaluation)
    record = references_module.initialize_source_references("CO2")["critical_point"]
    assert record.available and record.reason is None
    assert rejected_exact_requests == [(reported_density, reported_temperature)]
    assert record.reported_coordinates["temperature"] == reported_temperature
    assert record.reported_coordinates["density"] == reported_density
    np.testing.assert_allclose(
        record.evaluation_coordinates["temperature"],
        reported_temperature - 1e-5,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        record.evaluation_offsets["temperature"], -1e-5, rtol=0.0, atol=1e-12
    )
    source.update(CP.DmassT_INPUTS, reported_density, reported_temperature - 1e-5)
    state = record.state
    for name, expected in (
        ("pressure", source.p()),
        ("temperature", source.T()),
        ("density", source.rhomass()),
        ("enthalpy", source.hmass()),
        ("entropy", source.smass()),
        ("internal_energy", source.umass()),
    ):
        np.testing.assert_allclose(state.raw_property(name), expected, rtol=1e-12)
    for name in ("isobaric_heat_capacity", "isothermal_compressibility"):
        assert state.supports_property(name)
        assert np.isnan(state.raw_property(name))
        assert not state.property_valid(name)
        with pytest.raises(jxp.PropertyUnavailableError):
            state[name]


def test_source_and_references_share_the_same_constructed_reference_convention(
    monkeypatch,
):
    """Different constructor epochs simulate a concurrent global h/s change."""
    constructor = CP.AbstractState
    constructed = []

    class ReferenceEpoch:
        def __init__(self, backend, name):
            self.wrapped = constructor(backend, name)
            self.enthalpy_offset = (len(constructed) + 1) * 100000.0
            self.entropy_offset = (len(constructed) + 1) * 100.0
            constructed.append(self)

        def hmass(self):
            return self.wrapped.hmass() + self.enthalpy_offset

        def smass(self):
            return self.wrapped.smass() + self.entropy_offset

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    monkeypatch.setattr(CP, "AbstractState", ReferenceEpoch)
    source = CP.AbstractState("HEOS", "CO2")
    references = references_module.initialize_source_references("CO2", state=source)
    assert len(constructed) == 1
    anchor = references.anchor
    source.update(getattr(CP, anchor["input_pair"]), *anchor["inputs"])
    np.testing.assert_array_equal(anchor["enthalpy"], source.hmass())
    np.testing.assert_array_equal(anchor["entropy"], source.smass())
    # A separately prepared diagnostic helper must have the same actual h/s
    # convention. This fake changes it on every construction, so reject it
    # explicitly rather than silently attach incompatible auxiliary state.
    with pytest.raises(jxp.InvalidConfigurationError, match="[Rr]eference"):
        jxp.Fluid("CO2")


def test_binary_model_does_not_invent_unique_aggregate_phase_points():
    fluid = jxp.FluidMix("water_nitrogen_mixture", mixture_ratio=0.1)
    for point_name in POINT_NAMES:
        record = fluid.references[point_name]
        assert not record.available and record.reason
        with pytest.raises(jxp.UnavailableReferencePointError) as captured:
            getattr(fluid, point_name)
        assert captured.value.provider == fluid.provider
        assert captured.value.model == "binary_mixture"
        assert captured.value.request == point_name
    assert len(fluid.references.components) == 2
    assert all(
        component["critical_point"].available
        for component in fluid.references.components
    )
    assert np.isfinite(fluid.get_state(jxp.HmassP_INPUTS, 500e3, 8e6).T)


@pytest.mark.parametrize("backend", get_available_backends())
def test_perfect_gas_calibrates_from_the_selected_source_backend(backend):
    fluid = jxp.create_fluid(
        "CO2",
        provider="native",
        model="perfect_gas",
        backend=backend,
        model_options={"T_ref": 600.0, "p_ref": 1e5},
    )
    source = CP.AbstractState(backend, "CO2")
    source.update(CP.PT_INPUTS, 1e5, 600.0)
    assert fluid.backend == fluid.configuration.backend == backend
    assert fluid.references["critical_point"].backend == backend
    # Phase 5 preserves the current model calibration formula; correcting the
    # thermodynamic cp/cv/R relation and h offsets is explicitly Phase 7.
    np.testing.assert_allclose(
        fluid.constants.gamma, source.cpmass() / source.cvmass(), rtol=1e-12
    )
    np.testing.assert_allclose(fluid.constants.s_ref, source.smass(), rtol=1e-12)
    for input_pair, first, second in (
        (jxp.PQ_INPUTS, 1e6, 0.0),
        (jxp.QT_INPUTS, 1.0, 270.0),
    ):
        with pytest.raises(jxp.UnsupportedInputPairError):
            fluid.get_state(input_pair, first, second)
    with pytest.raises(jxp.UnsupportedOperationError):
        fluid.get_state_saturation(jxp.PQ_INPUTS, 1e6, 0.0)


@pytest.mark.parametrize("provider", ("bicubic", "perfect_gas"))
def test_native_execution_does_not_repeat_source_initialization(
    provider, fluid_models, monkeypatch
):
    fluid = fluid_models[0]["CO2", provider]

    def forbidden(*args, **kwargs):
        raise AssertionError("Source initialization entered repeated evaluation")

    monkeypatch.setattr(CP, "AbstractState", forbidden)
    traces = []

    @jax.jit
    def temperature(h):
        traces.append(None)
        return fluid.get_state(jxp.HmassP_INPUTS, h, 8e6).T

    values = [float(temperature(jnp.asarray(h))) for h in (680e3, 700e3, 720e3)]
    assert len(traces) == 1
    assert np.all(np.isfinite(values)) and np.all(np.diff(values) > 0)
    assert np.isfinite(jax.grad(temperature)(jnp.asarray(700e3)))


def test_changed_global_reference_is_detected_without_mutating_existing_objects():
    """Global reference mutations stay in a subprocess, preserving the parent."""
    repo = Path(__file__).resolve().parents[1]
    script = r"""
import json
import CoolProp.CoolProp as CP
import jaxprop as jxp
from jaxprop.bicubic.cache_metadata import source_signature, metadata_fingerprint

def ordinary():
    source = CP.AbstractState("HEOS", "CO2")
    source.update(CP.PT_INPUTS, 1e5, 500.0)
    return [source.hmass(), source.smass()]

before_values = ordinary()
old = jxp.Fluid("CO2")
old_id = old.reference_id
old_point_h = float(old.critical_point.h)
old_state_h = float(old.get_state(jxp.PT_INPUTS, 1e5, 500.0).h)
old_cache = source_signature("CO2", "HEOS")
unchanged_default = ordinary() == before_values
CP.set_reference_state("CO2", 300.0, 500.0, 12345.0, 67.0)
custom_values = ordinary()
new = jxp.Fluid("CO2")
callback = jxp.FluidJAX("CO2")
gas = jxp.FluidPerfectGas("CO2", T_ref=600.0)
new_cache = source_signature("CO2", "HEOS")
print(json.dumps({
    "unchanged_default": unchanged_default,
    "unchanged_custom": ordinary() == custom_values,
    "reference_changed": old_id != new.reference_id,
    "source_shared": new.references.reference_id == callback.references.reference_id == gas.references.reference_id,
    "cache_changed": metadata_fingerprint(old_cache) != metadata_fingerprint(new_cache),
    "old_identity_preserved": old.reference_id == old_id,
    "old_reference_preserved": float(old.critical_point.h) == old_point_h,
    "old_state_preserved": float(old.get_state(jxp.PT_INPUTS, 1e5, 500.0).h) == old_state_h,
    "new_state_changed": float(new.get_state(jxp.PT_INPUTS, 1e5, 500.0).h) != old_state_h,
    "new_reference_changed": float(new.critical_point.h) != old_point_h,
    "model_identity_distinct": gas.reference_id != new.reference_id,
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    outcomes = json.loads(result.stdout.strip().splitlines()[-1])
    assert all(outcomes.values()), outcomes


def test_configured_source_quality_keeps_its_original_entropy_convention():
    """Configured quality and core values retain their original source gauge."""
    repo = Path(__file__).resolve().parents[1]
    script = r"""
import json
import CoolProp.CoolProp as CP
import numpy as np
import jaxprop as jxp

fluid = jxp.Fluid("CO2")
before = fluid.get_state(jxp.PT_INPUTS, 1e5, 500.0)
reference_id = fluid.reference_id
critical_entropy = float(fluid.critical_point.s)
CP.set_reference_state("CO2", 300.0, 500.0, 12345.0, 1e6)
after = fluid.get_state(jxp.PT_INPUTS, 1e5, 500.0)
core = ("pressure", "temperature", "density", "enthalpy", "entropy", "internal_energy")
print(json.dumps({
    "core_preserved": all(np.array_equal(before.raw_property(key), after.raw_property(key)) for key in core),
    "identity_preserved": reference_id == fluid.reference_id == before.reference_id == after.reference_id,
    "critical_reference_preserved": float(fluid.critical_point.s) == critical_entropy,
    "quality_before": float(before.Q),
    "quality_after": float(after.Q),
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    outcomes = json.loads(result.stdout.strip().splitlines()[-1])
    if not all(
        outcomes[key]
        for key in (
            "core_preserved",
            "identity_preserved",
            "critical_reference_preserved",
        )
    ):
        raise RuntimeError(
            f"Unexpected configured-source invariant failure: {outcomes}"
        )
    if outcomes["quality_before"] not in (0.0, 1.0) or outcomes[
        "quality_after"
    ] not in (
        0.0,
        1.0,
    ):
        raise RuntimeError(f"Unexpected ordinary quality availability: {outcomes}")
    assert outcomes["quality_before"] == outcomes["quality_after"], outcomes
