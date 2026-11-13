"""Construction routing and immutable metadata are independent of numerical kernels."""

from dataclasses import FrozenInstanceError
import importlib
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from jaxprop.configuration import (
    FluidCapabilities,
    FluidDomain,
    FrozenMapping,
    build_configuration,
)
from jaxprop.errors import CalculationError, InvalidConfigurationError
from jaxprop.factory import create_fluid

TABLE_OPTIONS = dict(h_min=600e3, h_max=850e3, p_min=5e6, p_max=12e6, N_h=4, N_p=4)


@pytest.mark.parametrize(
    "provider,model,module,class_name,extra,expected",
    [
        (
            "coolprop",
            "real_fluid",
            "jaxprop.coolprop.fluid_properties",
            "Fluid",
            {},
            {},
        ),
        (
            "coolprop_jax",
            "real_fluid",
            "jaxprop.coolprop.jax_wrapper",
            "FluidJAX",
            {},
            {},
        ),
        (
            "bicubic",
            "real_fluid",
            "jaxprop.bicubic.bicubic_interpolation",
            "FluidBicubic",
            {"table_options": TABLE_OPTIONS},
            TABLE_OPTIONS,
        ),
        (
            "native",
            "perfect_gas",
            "jaxprop.perfect_gas.perfect_gas",
            "FluidPerfectGas",
            {"model_options": {"T_ref": 600.0, "p_ref": 1e5}},
            {"T_ref": 600.0, "p_ref": 1e5},
        ),
    ],
)
def test_factory_returns_actual_constructor_result_without_an_adapter(
    monkeypatch, provider, model, module, class_name, extra, expected
):
    constructor_module = importlib.import_module(module)
    received = []
    result = object()

    def construct(**kwargs):
        received.append(kwargs)
        return result

    monkeypatch.setattr(constructor_module, class_name, construct)
    actual = create_fluid("CO2", provider=provider, model=model, identifier="", **extra)
    assert actual is result
    assert len(received) == 1
    arguments = received[0]
    assert arguments.get("name", arguments.get("fluid_name")) == "CO2"
    assert arguments["identifier"] == ""
    if provider != "native":
        assert arguments["backend"] == "HEOS"
    assert all(arguments[key] == value for key, value in expected.items())


@pytest.mark.parametrize(
    "arguments,expected_request",
    [
        ({"name": ""}, "name"),
        ({"name": "   "}, "name"),
        ({"name": 1}, "name"),
        ({"identifier": []}, "identifier"),
        ({"backend": ""}, "backend"),
        ({"backend": True}, "backend"),
        ({"provider": "CoolProp"}, "provider"),
        ({"provider": "unknown"}, "provider"),
        ({"model": "unknown"}, "model"),
        ({"provider": "native"}, "provider/model"),
        ({"model": "perfect_gas"}, "provider/model"),
        ({"table_options": {"N_h": 4}}, "table_options"),
        ({"model_options": {"T_ref": 600.0}}, "model_options"),
        ({"model_options": []}, "model_options"),
        ({"table_options": "table"}, "table_options"),
        ({"model_options": {1: 2}}, "model_options"),
    ],
)
def test_invalid_configuration_is_rejected_before_source_construction(
    monkeypatch, arguments, expected_request
):
    module = importlib.import_module("jaxprop.coolprop.fluid_properties")

    def forbidden(**kwargs):
        raise AssertionError("Invalid configuration must not initialize a source")

    monkeypatch.setattr(module, "Fluid", forbidden)
    with pytest.raises(InvalidConfigurationError) as captured:
        create_fluid(**({"name": "CO2"} | arguments))
    assert captured.value.request == expected_request
    assert captured.value.reason


@pytest.mark.parametrize(
    "changed,expected_request",
    [
        ({"h_max": 600e3}, "h_min/h_max"),
        ({"h_min": np.nan}, "h_min"),
        ({"p_min": 0.0}, "p_min/p_max"),
        ({"p_max": np.inf}, "p_max"),
        ({"N_h": True}, "N_h"),
        ({"N_p": 4.5}, "N_p"),
        ({"N_p_sat": 1}, "N_p_sat"),
        ({"coarse_step": 0}, "coarse_step"),
        ({"gradient_method": {}}, "gradient_method"),
        ({"gradient_method": ["central"]}, "gradient_method"),
        ({"gradient_method": "unknown"}, "gradient_method"),
        ({"mixture_ratio": 0.1}, "table_options"),
        ({"metastable_phase": "vapor"}, "table_options"),
        ({"table_dir": None}, "table_dir"),
        ({"table_name": ""}, "table_name"),
    ],
)
def test_invalid_or_unvalidated_table_options_do_not_construct_a_table(
    monkeypatch, changed, expected_request
):
    module = importlib.import_module("jaxprop.bicubic.bicubic_interpolation")

    def forbidden(**kwargs):
        raise AssertionError("Invalid settings must not allocate/generate a table")

    monkeypatch.setattr(module, "FluidBicubic", forbidden)
    with pytest.raises(InvalidConfigurationError) as captured:
        create_fluid("CO2", provider="bicubic", table_options=TABLE_OPTIONS | changed)
    assert captured.value.request == expected_request


@pytest.mark.parametrize(
    "value", [True, None, -1.0, 0.0, np.inf, np.nan, "600", np.array([600.0])]
)
def test_invalid_perfect_gas_calibration_is_rejected(value):
    with pytest.raises(InvalidConfigurationError) as captured:
        create_fluid(
            "CO2",
            provider="native",
            model="perfect_gas",
            model_options={"T_ref": value},
        )
    assert captured.value.request == "T_ref"


def test_perfect_gas_does_not_claim_unimplemented_source_backend_selection():
    with pytest.raises(InvalidConfigurationError) as captured:
        create_fluid("CO2", provider="native", model="perfect_gas", backend="REFPROP")
    assert captured.value.request == "backend"
    assert captured.value.supported == ("HEOS",)


def test_constructor_source_failure_preserves_the_original_cause(monkeypatch):
    module = importlib.import_module("jaxprop.coolprop.fluid_properties")
    cause = ValueError("requested source backend is unavailable")

    def fail(**kwargs):
        raise cause

    monkeypatch.setattr(module, "Fluid", fail)
    with pytest.raises(InvalidConfigurationError) as captured:
        create_fluid("CO2", backend="UNAVAILABLE")
    assert captured.value.__cause__ is cause
    assert captured.value.fluid_name == "CO2"
    assert captured.value.provider == "coolprop"


@pytest.mark.parametrize(
    "failure",
    [
        CalculationError(request="table", reason="source failed"),
        PermissionError("cache directory is read-only"),
    ],
)
def test_factory_does_not_disguise_bicubic_calculation_or_cache_io_failures(
    monkeypatch, failure
):
    module = importlib.import_module("jaxprop.bicubic.bicubic_interpolation")

    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(module, "FluidBicubic", fail)
    with pytest.raises(type(failure)) as captured:
        create_fluid("CO2", provider="bicubic", table_options=TABLE_OPTIONS)
    assert captured.value is failure


@pytest.mark.parametrize(
    "fractions,expected_request",
    [
        ((0.2, 0.2), "mass_fractions"),
        ((-0.1, 1.1), "mass_fractions"),
        ((np.nan, 1.0), "mass_fractions"),
        ((0.0, 1.0), "model"),
        ((1.0, 0.0), "model"),
    ],
)
def test_binary_requests_are_validated_without_faking_the_component_model(
    fractions, expected_request
):
    component = SimpleNamespace(get_state=lambda *args: None)
    with pytest.raises(InvalidConfigurationError) as captured:
        create_fluid(
            "binary",
            provider="native",
            model="binary_mixture",
            model_options={
                "components": (component, component),
                "mass_fractions": fractions,
            },
        )
    assert captured.value.request == expected_request
    if expected_request == "model":
        assert "consolidation" in captured.value.reason


def test_configuration_is_a_recursive_snapshot_and_preserves_explicit_identity():
    options = {"nested": {"values": [1.0, 2.0]}, "labels": {"gas", "liquid"}}
    config = build_configuration(
        "CarbonDioxide", "bicubic", "real_fluid", "HEOS", "", table_options=options
    )
    options["nested"]["values"][0] = 100.0
    options["labels"].add("other")
    assert config.name == "CarbonDioxide" and config.identifier == ""
    assert config.table_options["nested"]["values"] == (1.0, 2.0)
    assert config.table_options["labels"] == frozenset({"gas", "liquid"})
    with pytest.raises(FrozenInstanceError):
        config.name = "other"
    with pytest.raises(TypeError):
        config.table_options["new"] = 1
    restored = pickle.loads(pickle.dumps(config))
    assert restored == config and hash(restored) == hash(config)


def test_mapping_hash_and_equality_ignore_insertion_order():
    first = FrozenMapping({"a": [1, 2], "b": {"name": "CO2"}})
    second = FrozenMapping({"b": {"name": "CO2"}, "a": [1, 2]})
    assert first == second
    assert hash(first) == hash(second)


def test_capabilities_domain_and_collections_are_immutable_and_pickleable():
    pairs, properties, phases = {20}, {"pressure", "temperature"}, ["gas"]
    bounds = {"temperature": [0.0, None]}
    domain = FluidDomain(description="gas", bounds=bounds, phases=iter(phases))
    capability = FluidCapabilities(
        pairs,
        properties,
        domain=domain,
        jax_modes={"eager", "jit"},
        derivative_order=1,
        derivative_method="autodiff",
    )
    pairs.add(9)
    properties.add("enthalpy")
    bounds["temperature"][0] = -1.0
    phases.append("liquid")
    assert capability.input_pairs == {20}
    assert capability.properties == {"pressure", "temperature"}
    assert domain.phases == {"gas"}
    assert domain.bounds["temperature"] == (0.0, None)
    with pytest.raises(FrozenInstanceError):
        capability.derivative_order = 2
    restored = pickle.loads(pickle.dumps(capability))
    assert restored == capability and hash(restored) == hash(capability)


@pytest.mark.parametrize(
    "changed",
    [
        {"description": []},
        {"extrapolation": "no"},
        {"bounds": {"pressure": (2, 1)}},
        {"bounds": {"pressure": (np.nan, 1)}},
        {"phases": "gas"},
    ],
)
def test_domain_rejects_mutable_or_malformed_scalar_metadata(changed):
    with pytest.raises((TypeError, ValueError)):
        FluidDomain(**changed)


@pytest.mark.parametrize(
    "changed",
    [
        {"input_pairs": {True}},
        {"properties": "pressure"},
        {"jax_modes": {1}},
        {"error_modes": {"ignore"}},
        {"derivative_order": True},
        {"derivative_order": -1},
        {"derivative_method": []},
    ],
)
def test_capabilities_reject_malformed_metadata(changed):
    with pytest.raises((TypeError, ValueError)):
        FluidCapabilities(
            **({"input_pairs": {20}, "properties": {"pressure"}} | changed)
        )
