"""Construction and static-request contracts independent of provider physics."""

from dataclasses import FrozenInstanceError

import CoolProp.CoolProp as CP
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp

from fluid_contract_cases import CONTRACT_CASES, build_current_fluid, factory_arguments


@pytest.fixture(scope="module")
def fluids(tmp_path_factory):
    directory = tmp_path_factory.mktemp("phase4_api")
    return {case.id: build_current_fluid(case, directory) for case in CONTRACT_CASES}


def assert_context(error, fluid, request):
    assert error.fluid_name == fluid.name
    assert error.provider == fluid.provider
    assert error.model == fluid.model
    assert str(request) in error.request
    assert error.reason
    assert isinstance(error.supported, tuple)
    assert str(request) in str(error)


@pytest.mark.parametrize("case", CONTRACT_CASES[:-1], ids=lambda case: case.id)
def test_factory_returns_native_class_and_resolves_source_defaults(case, tmp_path):
    settings = factory_arguments(case, tmp_path)
    settings["backend"] = None
    settings["identifier"] = ""
    fluid = jxp.create_fluid(**settings)
    assert type(fluid) is getattr(jxp, case.current_class)
    assert fluid.name == case.name
    assert fluid.identifier == ""
    assert fluid.backend == "HEOS"
    assert fluid.configuration.name == case.name
    assert fluid.configuration.provider == case.provider
    assert fluid.configuration.model == case.model
    assert fluid.configuration.identifier == ""


@pytest.mark.parametrize(
    "settings",
    [
        {"name": None},
        {"name": True},
        {"name": "   "},
        {"identifier": 123},
        {"provider": "COOLPROP"},
        {"provider": " coolprop"},
        {"provider": None},
        {"model": "Real_Fluid"},
        {"backend": ""},
        {"backend": False},
        {"table_options": []},
        {"model_options": []},
        {"provider": "coolprop_jax", "table_options": {"N_h": 4}},
        {"provider": "native", "model": "perfect_gas", "model_options": {"typo": 1}},
    ],
)
def test_factory_rejects_invalid_types_and_inapplicable_settings(settings):
    with pytest.raises(jxp.InvalidConfigurationError) as captured:
        jxp.create_fluid(**({"name": "CO2"} | settings))
    assert captured.value.request and captured.value.reason


def test_factory_retains_original_source_failure_as_cause():
    with pytest.raises(jxp.InvalidConfigurationError) as captured:
        jxp.create_fluid("this_fluid_does_not_exist")
    assert captured.value.__cause__ is not None
    assert captured.value.fluid_name == "this_fluid_does_not_exist"
    assert captured.value.reason


def test_perfect_gas_cannot_silently_substitute_heos_for_an_unavailable_backend():
    with pytest.raises(jxp.InvalidConfigurationError) as captured:
        jxp.create_fluid(
            "CO2", provider="native", model="perfect_gas", backend="MISSING_BACKEND"
        )
    assert captured.value.request in {"backend", "source", "source_reference"}
    assert captured.value.__cause__ is not None
    assert captured.value.reason


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_configuration_and_capabilities_are_immutable(case, fluids):
    fluid = fluids[case.id]
    configuration = fluid.configuration
    capability = fluid.capabilities
    assert isinstance(configuration, jxp.FluidConfiguration)
    assert isinstance(capability, jxp.FluidCapabilities)
    assert isinstance(capability.domain, jxp.FluidDomain)
    for value in (
        capability.input_pairs,
        capability.properties,
        capability.options,
        capability.phase_operations,
        capability.jax_modes,
        capability.error_modes,
        capability.domain.phases,
    ):
        assert isinstance(value, frozenset)
    expected_modes = (
        {"raise", "mask"} if case.id in {"coolprop", "coolprop_jax"} else {"raise"}
    )
    if case.id in {"perfect_gas", "bicubic"}:
        expected_modes = {"raise", "mask", "clip"}
    assert capability.error_modes == frozenset(expected_modes)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        configuration.name = "Water"
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        capability.input_pairs = frozenset()
    with pytest.raises(TypeError):
        capability.domain.bounds["unexpected"] = 1
    with pytest.raises(TypeError):
        configuration.model_options["unexpected"] = 1
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        fluid.name = "Water"
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        del fluid.name
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        fluid.reference_id = "different-reference"
    if case.id == "binary_mixture":
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            fluid.mixture_ratio = 1.0
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            del fluid.fluid_1
    assert fluid.name == case.name and fluid.configuration is configuration


@pytest.mark.parametrize("ratio", [None, -1.0, True, np.nan, np.inf, 1j])
def test_legacy_mixture_configuration_rejects_invalid_mass_ratios(ratio):
    with pytest.raises(jxp.InvalidConfigurationError) as captured:
        jxp.FluidMix("water_nitrogen_mixture", mixture_ratio=ratio)
    assert captured.value.request == "mixture_ratio"
    assert captured.value.reason


@pytest.mark.parametrize(
    "pair,properties",
    [
        (jxp.SmassT_INPUTS, ("entropy", "temperature")),
        (jxp.DmassUmass_INPUTS, ("density", "internal_energy")),
    ],
)
def test_additional_advertised_callback_pairs_match_the_source(pair, properties):
    source = jxp.Fluid("CO2")
    callback = jxp.FluidJAX("CO2")
    reference = source.get_state(jxp.HmassP_INPUTS, 700e3, 8e6)
    inputs = [reference.raw_property(name) for name in properties]
    expected = source.get_state(pair, *inputs)
    actual = callback.get_state(pair, *inputs)
    for name in (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    ):
        np.testing.assert_allclose(
            actual.raw_property(name),
            expected.raw_property(name),
            rtol=1e-10,
            atol=1e-8,
        )


def test_factory_copies_configuration_mappings_before_evaluation(tmp_path):
    options = {"T_ref": 600.0, "p_ref": 1e5}
    gas = jxp.create_fluid(
        "CO2", provider="native", model="perfect_gas", model_options=options
    )
    original = gas.get_state(jxp.PT_INPUTS, 1e5, 600.0)
    options["T_ref"] = 350.0
    options["unknown_setting"] = True
    assert gas.configuration.model_options["T_ref"] == 600.0
    assert "unknown_setting" not in gas.configuration.model_options
    np.testing.assert_array_equal(
        gas.get_state(jxp.PT_INPUTS, 1e5, 600.0).h, original.h
    )

    case = next(case for case in CONTRACT_CASES if case.id == "bicubic")
    arguments = factory_arguments(case, tmp_path)
    table_options = arguments["table_options"]
    table = jxp.create_fluid(**arguments)
    table_options["N_h"] = 99
    assert table.N_h == 12
    assert table.configuration.table_options["N_h"] == 12


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("pair", ["PT_INPUTS", True, -12345])
def test_unknown_or_noninteger_pairs_raise_typed_errors(case, pair, fluids):
    fluid = fluids[case.id]
    with pytest.raises(jxp.UnsupportedInputPairError) as captured:
        fluid.get_state(pair, 700e3, 8e6)
    assert_context(captured.value, fluid, str(pair))


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize(
    "pair_name",
    ["DmolarP_INPUTS", "HmolarP_INPUTS", "PSmolar_INPUTS", "HmolarSmolar_INPUTS"],
)
def test_additional_molar_pairs_never_acquire_mass_basis_by_conversion(
    case, pair_name, fluids
):
    fluid = fluids[case.id]
    with pytest.raises(jxp.UnsupportedInputPairError) as captured:
        fluid.get_state(getattr(CP, pair_name), 100.0, 500.0)
    assert_context(captured.value, fluid, pair_name)


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize(
    "options,option_request",
    [
        ({"unknown_option": True}, "unknown_option"),
        ({"generalize_quality": "yes"}, "generalize_quality"),
        ({"supersaturation": 1}, "supersaturation"),
        ({"errors": "ignore"}, "errors"),
    ],
)
def test_invalid_options_are_typed_for_every_evaluator(
    case, options, option_request, fluids
):
    fluid = fluids[case.id]
    enthalpy = 500e3 if case.id == "binary_mixture" else 700e3
    with pytest.raises(jxp.UnsupportedOptionError) as captured:
        fluid.get_state(jxp.HmassP_INPUTS, enthalpy, 8e6, **options)
    assert_context(captured.value, fluid, option_request)


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("errors", ["raise", "mask"])
@pytest.mark.parametrize(
    "first,second",
    [
        (True, 8e6),
        ("700000", 8e6),
        (700e3 + 1j, 8e6),
        (np.array([700e3], dtype=object), 8e6),
        (np.ones((2,)), np.ones((3,))),
    ],
)
def test_malformed_arrays_fail_even_with_requested_numerical_masking(
    case, errors, first, second, fluids
):
    fluid = fluids[case.id]
    with pytest.raises(jxp.InvalidStateError) as captured:
        fluid.get_state(jxp.HmassP_INPUTS, first, second, errors=errors)
    assert_context(captured.value, fluid, "HmassP_INPUTS")


@pytest.mark.parametrize(
    "case",
    [case for case in CONTRACT_CASES if case.id == "binary_mixture"],
    ids=lambda case: case.id,
)
def test_unimplemented_numerical_mask_mode_is_rejected_explicitly(case, fluids):
    fluid = fluids[case.id]
    enthalpy = 500e3 if case.id == "binary_mixture" else 700e3
    with pytest.raises(jxp.UnsupportedExecutionModeError) as captured:
        fluid.get_state(jxp.HmassP_INPUTS, enthalpy, 8e6, errors="mask")
    assert_context(captured.value, fluid, "errors=mask")


@pytest.mark.parametrize("case", CONTRACT_CASES[:4], ids=lambda case: case.id)
def test_disabled_diagnostics_are_unsupported_rather_than_available_nan_slots(
    case, fluids
):
    state = fluids[case.id].get_state(jxp.HmassP_INPUTS, 700e3, 8e6)
    for name in (
        "subcooling",
        "superheating",
        "pressure_saturation",
        "temperature_saturation",
        "supersaturation_ratio",
        "supersaturation_degree",
    ):
        assert name not in state.available_properties
        with pytest.raises(jxp.UnsupportedPropertyError):
            state.raw_property(name)


@pytest.mark.parametrize("case", CONTRACT_CASES[1:4], ids=lambda case: case.id)
def test_static_request_errors_remain_typed_inside_outer_jit(case, fluids):
    fluid = fluids[case.id]
    with pytest.raises(jxp.UnsupportedInputPairError) as captured:
        jax.jit(lambda density: fluid.get_state(jxp.DmolarT_INPUTS, density, 500.0).T)(
            jnp.asarray(100.0)
        )
    assert_context(captured.value, fluid, "DmolarT_INPUTS")
    with pytest.raises(jxp.UnsupportedOptionError) as captured:
        jax.jit(
            lambda enthalpy: fluid.get_state(
                jxp.HmassP_INPUTS, enthalpy, 8e6, unknown_option=True
            ).T
        )(jnp.asarray(700e3))
    assert_context(captured.value, fluid, "unknown_option")


@pytest.mark.parametrize("case", CONTRACT_CASES[1:4], ids=lambda case: case.id)
def test_dynamic_pair_selectors_are_rejected_without_tracer_conversion(case, fluids):
    fluid = fluids[case.id]
    with pytest.raises(jxp.UnsupportedInputPairError) as captured:
        jax.jit(lambda selector: fluid.get_state(selector, 700e3, 8e6).T)(
            jnp.asarray(jxp.HmassP_INPUTS)
        )
    assert captured.value.fluid_name == fluid.name
    assert captured.value.provider == fluid.provider
    assert captured.value.request and captured.value.reason
    assert captured.value.supported


@pytest.mark.parametrize("case", CONTRACT_CASES[1:4], ids=lambda case: case.id)
def test_dynamic_option_flags_raise_typed_errors_instead_of_tracer_conversion(
    case, fluids
):
    fluid = fluids[case.id]
    with pytest.raises(jxp.UnsupportedOptionError) as captured:
        jax.jit(
            lambda flag: fluid.get_state(
                jxp.HmassP_INPUTS, 700e3, 8e6, generalize_quality=flag
            ).T
        )(jnp.asarray(False))
    assert_context(captured.value, fluid, "generalize_quality")


def test_named_status_codes_match_independent_public_numeric_contract():
    expected = {
        "SUCCESS": 0,
        "INVALID_INPUT": 1,
        "OUT_OF_DOMAIN": 2,
        "NONCONVERGENCE": 3,
        "PROVIDER_FAILURE": 4,
    }
    for name, value in expected.items():
        assert int(getattr(jxp.StatusCode, name)) == value
    successful = jxp.FluidState(pressure=[1e5, 2e5])
    np.testing.assert_array_equal(successful.status, [jxp.StatusCode.SUCCESS] * 2)
    np.testing.assert_array_equal(successful.valid, [True, True])
