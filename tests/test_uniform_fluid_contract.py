"""Executable target contract for the staged public fluid-interface refactor.

The fixed cases keep current capabilities from silently narrowing the required
coverage. Each test exercises one concern through the public constructor and
native evaluator, including the consolidated component-object binary model.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp

from fluid_contract_cases import (
    BICUBIC_PAIRS,
    BINARY_PAIRS,
    CONTRACT_CASES,
    CORE_PROPERTIES,
    PAIR_FIELDS,
    PERFECT_GAS_PAIRS,
    SHAPE_PROPERTIES,
    SOURCE_PAIRS,
    build_current_fluid,
    constructor_arguments,
)


@pytest.fixture(scope="module")
def fluid_objects(tmp_path_factory):
    table_dir = tmp_path_factory.mktemp("uniform_fluid_contract")
    return {case.id: build_current_fluid(case, table_dir) for case in CONTRACT_CASES}


def baseline(fluid, case_id=None):
    # The binary model's valid single-phase component branches differ from CO2.
    # 700 kJ/kg lies in a water phase-change gap at this composition/pressure.
    enthalpy = 500e3 if case_id == "binary_mixture" else 700e3
    return fluid.get_state(jxp.HmassP_INPUTS, enthalpy, 8e6)


def assert_typed_error(call, expected_name, request):
    """Check actual rejection before checking the promised public exception type."""
    try:
        call()
    except Exception as error:
        expected_type = getattr(jxp, expected_name, None)
        assert (
            expected_type is not None
        ), f"Missing public error class {expected_name}: {error!r}"
        assert isinstance(
            error, expected_type
        ), f"Expected {expected_name}, received {error!r}"
        for attribute in (
            "fluid_name",
            "provider",
            "model",
            "request",
            "reason",
            "supported",
        ):
            assert hasattr(error, attribute), f"Missing error context {attribute}"
        assert error.fluid_name
        assert error.provider
        assert error.reason
        assert request in str(error.request)
        assert request in str(error)
    else:
        assert False, f"{expected_name} was not raised for {request}"


def test_fixed_conformance_matrix_covers_current_and_consolidated_models():
    """Changing runtime dispatch cannot silently reduce expected coverage."""
    assert {case.current_class for case in CONTRACT_CASES} == {
        "FluidCoolProp",
        "FluidJAX",
        "FluidBicubic",
        "FluidPerfectGas",
        "FluidBinaryMixture",
    }
    assert {(case.provider, case.model) for case in CONTRACT_CASES} == {
        ("coolprop", "real_fluid"),
        ("coolprop_jax", "real_fluid"),
        ("bicubic", "real_fluid"),
        ("native", "perfect_gas"),
        ("native", "binary_mixture"),
    }
    assert len(PERFECT_GAS_PAIRS) == 6 and len(BICUBIC_PAIRS) == 8
    assert len(SOURCE_PAIRS) == 10 and len(BINARY_PAIRS) == 3


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_common_constructor_constructs_each_advertised_model(case, tmp_path):
    fluid = jxp.Fluid(**constructor_arguments(case, tmp_path))
    assert fluid.name == case.name
    assert fluid.provider == case.provider and fluid.model == case.model
    assert fluid.backend == (None if case.model == "binary_mixture" else "HEOS")
    assert fluid.identifier == "working_fluid"


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_default_identifier_is_none(case, fluid_objects):
    assert fluid_objects[case.id].identifier is None


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_metadata_and_capability_queries(case, fluid_objects):
    fluid = fluid_objects[case.id]
    assert fluid.name == case.name
    assert fluid.provider == case.provider and fluid.model == case.model
    capability = fluid.capabilities
    # Metadata must describe implemented functionality rather than advertise
    # the later model-completion targets below before they exist.
    assert jxp.HmassP_INPUTS in capability.input_pairs
    assert set(SHAPE_PROPERTIES) <= set(capability.properties)
    assert set(capability.properties) <= set(jxp.PROPERTIES_CANONICAL)
    assert baseline(fluid, case.id).available_properties <= set(capability.properties)
    for attribute in (
        "options",
        "phase_operations",
        "jax_modes",
        "domain",
        "error_modes",
    ):
        assert hasattr(capability, attribute)
    if case.model == "perfect_gas":
        assert jxp.QT_INPUTS not in capability.input_pairs
        assert jxp.PQ_INPUTS not in capability.input_pairs
        assert "quality_mass" not in capability.properties
        assert "saturation" not in capability.phase_operations


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_target_capability_minimums_remain_explicit(case, fluid_objects):
    capability = fluid_objects[case.id].capabilities
    assert set(getattr(jxp, name) for name in case.required_pairs) <= set(
        capability.input_pairs
    )
    assert set(CORE_PROPERTIES) <= set(capability.properties)


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_state_preserves_source_identity(case, fluid_objects):
    assert baseline(fluid_objects[case.id], case.id).fluid_name == case.name


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_explicit_identifier_reaches_returned_state(case, tmp_path):
    fluid = build_current_fluid(case, tmp_path, identifier="working_fluid")
    assert baseline(fluid, case.id).identifier == "working_fluid"


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_common_property_keyword_vocabulary(case, fluid_objects):
    fluid = fluid_objects[case.id]
    enthalpy = 500e3 if case.id == "binary_mixture" else 700e3
    result = fluid.get_state(input_type=jxp.HmassP_INPUTS, prop_1=enthalpy, prop_2=8e6)
    np.testing.assert_allclose(result.h, enthalpy, rtol=1e-10)
    np.testing.assert_allclose(result.p, 8e6, rtol=1e-10)


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_disabled_diagnostics_are_noops_for_every_model(case, fluid_objects):
    fluid = fluid_objects[case.id]
    reference = baseline(fluid, case.id)
    enthalpy = 500e3 if case.id == "binary_mixture" else 700e3
    state = fluid.get_state(
        jxp.HmassP_INPUTS,
        enthalpy,
        8e6,
        generalize_quality=False,
        supersaturation=False,
    )
    for property_name in SHAPE_PROPERTIES:
        np.testing.assert_array_equal(
            getattr(state, property_name),
            getattr(reference, property_name),
        )


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_successful_scalar_states_have_finite_mass_based_core_fields(
    case, fluid_objects
):
    state = baseline(fluid_objects[case.id], case.id)
    assert set(CORE_PROPERTIES) <= state.available_properties
    for property_name in CORE_PROPERTIES:
        value = np.asarray(state.raw_property(property_name))
        assert value.shape == () and np.isfinite(value), property_name
    # Independent first-law identity checks the units and specific-energy basis.
    tolerance = 1e-5 if case.id == "bicubic" else 1e-10
    np.testing.assert_allclose(state.h - state.p / state.rho, state.u, rtol=tolerance)


SHAPES = (
    ("scalar", 700e3, 8e6, ()),
    ("singleton", np.array([700e3]), np.array([8e6]), (1,)),
    ("singleton_matrix", np.array([[700e3], [710e3]]), 8e6, (2, 1)),
    (
        "broadcast_matrix",
        np.array([[700e3], [710e3]]),
        np.array([[7e6, 8e6, 9e6]]),
        (2, 3),
    ),
)
SHAPE_PARAMETERS = []
for _case in CONTRACT_CASES:
    for _label, _h, _p, _shape in SHAPES:
        if _case.id == "binary_mixture":
            _h = np.asarray(_h) - 200e3
        SHAPE_PARAMETERS.append(
            pytest.param(_case, _h, _p, _shape, id=f"{_case.id}-{_label}")
        )


@pytest.mark.parametrize("case,h,p,shape", SHAPE_PARAMETERS)
def test_numeric_fields_preserve_exact_broadcast_shape(
    case, h, p, shape, fluid_objects
):
    state = fluid_objects[case.id].get_state(jxp.HmassP_INPUTS, h, p)
    for property_name in SHAPE_PROPERTIES:
        value = np.asarray(getattr(state, property_name))
        assert value.shape == shape, f"{property_name}: {value.shape} != {shape}"
        assert np.all(np.isfinite(value)), property_name


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_enabled_numeric_fields_status_and_property_masks_share_broadcast_shape(
    case, fluid_objects
):
    enthalpy = 500e3 if case.id == "binary_mixture" else 700e3
    state = fluid_objects[case.id].get_state(
        jxp.HmassP_INPUTS,
        np.array([[enthalpy], [enthalpy + 1e3]]),
        8e6,
    )
    status = np.asarray(state.status)
    valid = np.asarray(state.valid)
    assert status.shape == valid.shape == (2, 1)
    assert np.issubdtype(status.dtype, np.integer)
    assert valid.dtype == np.dtype(bool)
    np.testing.assert_array_equal(status, np.zeros((2, 1), dtype=int))
    assert np.all(valid)
    assert set(CORE_PROPERTIES) <= set(state.available_properties)
    raw = state.to_dict()
    for name in state.available_properties:
        assert name in raw
        assert np.asarray(raw[name]).shape == (2, 1), name
        mask = np.asarray(state.property_valid(name))
        assert mask.shape == (2, 1) and mask.dtype == np.dtype(bool), name
        if name in CORE_PROPERTIES:
            assert np.all(mask) and np.all(np.isfinite(raw[name])), name


ARRAY_PARAMETERS = []
for _case in CONTRACT_CASES:
    for _kind in ("list", "numpy", "jax"):
        ARRAY_PARAMETERS.append(pytest.param(_case, _kind, id=f"{_case.id}-{_kind}"))


@pytest.mark.parametrize("case,kind", ARRAY_PARAMETERS)
def test_python_numpy_and_jax_array_inputs(case, kind, fluid_objects):
    convert = {"list": list, "numpy": np.asarray, "jax": jnp.asarray}[kind]
    h = convert([500e3, 510e3] if case.id == "binary_mixture" else [700e3, 710e3])
    p = convert([8e6, 9e6])
    state = fluid_objects[case.id].get_state(jxp.HmassP_INPUTS, h, p)
    np.testing.assert_allclose(state.h, h, rtol=1e-10)
    np.testing.assert_allclose(state.p, p, rtol=1e-10)


PAIR_PARAMETERS = []
for _case in CONTRACT_CASES:
    for _pair in _case.required_pairs:
        if _pair not in PAIR_FIELDS:
            continue  # Saturation endpoints are checked separately below.
        PAIR_PARAMETERS.append(pytest.param(_case, _pair, id=f"{_case.id}-{_pair}"))


@pytest.mark.parametrize("case,pair_name", PAIR_PARAMETERS)
def test_fixed_required_input_pairs_recover_the_same_state(
    case, pair_name, fluid_objects
):
    fluid = fluid_objects[case.id]
    reference = baseline(fluid, case.id)
    first, second = PAIR_FIELDS[pair_name]
    result = fluid.get_state(
        getattr(jxp, pair_name), getattr(reference, first), getattr(reference, second)
    )
    for property_name in SHAPE_PROPERTIES:
        np.testing.assert_allclose(
            getattr(result, property_name),
            getattr(reference, property_name),
            rtol=1e-8,
            atol=1e-8,
        )


@pytest.mark.parametrize("case", CONTRACT_CASES[:2], ids=lambda case: case.id)
@pytest.mark.parametrize(
    "pair_name,first,second", [("PQ_INPUTS", 1e6, 0.0), ("QT_INPUTS", 1.0, 270.0)]
)
def test_source_evaluators_accept_quality_endpoint_pairs(
    case, pair_name, first, second, fluid_objects
):
    state = fluid_objects[case.id].get_state(getattr(jxp, pair_name), first, second)
    for property_name in CORE_PROPERTIES:
        assert np.isfinite(np.asarray(getattr(state, property_name))), property_name
    assert float(state.Q) == (first if pair_name == "QT_INPUTS" else second)


@pytest.mark.parametrize(
    "pair_name,first,second", [("QT_INPUTS", 0.0, 270.0), ("PQ_INPUTS", 1e6, 1.0)]
)
def test_perfect_gas_quality_inputs_raise_informative_typed_errors(
    pair_name, first, second, fluid_objects
):
    fluid = fluid_objects["perfect_gas"]
    assert_typed_error(
        lambda: fluid.get_state(getattr(jxp, pair_name), first, second),
        "UnsupportedInputPairError",
        pair_name,
    )


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_molar_input_pairs_are_rejected_without_implicit_conversion(
    case, fluid_objects
):
    fluid = fluid_objects[case.id]
    assert_typed_error(
        lambda: fluid.get_state(jxp.DmolarT_INPUTS, 100.0, 600.0),
        "UnsupportedInputPairError",
        "DmolarT_INPUTS",
    )


@pytest.mark.parametrize("access", ["canonical", "alias", "item"])
def test_perfect_gas_quality_property_access_is_explicit(access, fluid_objects):
    state = baseline(fluid_objects["perfect_gas"])
    read = {
        "canonical": lambda: state.quality_mass,
        "alias": lambda: state.Q,
        "item": lambda: state["Q"],
    }[access]
    assert_typed_error(read, "UnsupportedPropertyError", "quality_mass")


def test_perfect_gas_saturation_operation_is_explicitly_rejected(fluid_objects):
    fluid = fluid_objects["perfect_gas"]
    assert_typed_error(
        lambda: fluid.get_state_saturation(jxp.PQ_INPUTS, 1e6, 0.0),
        "UnsupportedOperationError",
        "saturation",
    )


@pytest.mark.parametrize("case", CONTRACT_CASES[2:], ids=lambda case: case.id)
@pytest.mark.parametrize("option", ["generalize_quality", "supersaturation"])
def test_unsupported_diagnostic_options_have_explicit_errors(
    case, option, fluid_objects
):
    fluid = fluid_objects[case.id]
    assert_typed_error(
        lambda: fluid.get_state(jxp.HmassP_INPUTS, 700e3, 8e6, **{option: True}),
        "UnsupportedOptionError",
        option,
    )
