"""Target failure, reference, and execution policies for the uniform API.

These cases intentionally exercise existing objects without depending on the
not-yet-implemented factory. Every xfail is a confirmed, phase-owned gap.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp


@pytest.fixture(scope="module")
def perfect_gas():
    return jxp.FluidPerfectGas("CO2", identifier="working_fluid")


@pytest.fixture(scope="module")
def source_fluid():
    return jxp.Fluid("CO2", identifier="working_fluid")


ERROR_BASES = (
    ("JaxPropError", Exception),
    ("InvalidConfigurationError", ValueError),
    ("UnsupportedRequestError", ValueError),
    ("UnsupportedInputPairError", "UnsupportedRequestError"),
    ("UnsupportedPropertyError", "UnsupportedRequestError"),
    ("UnsupportedOptionError", "UnsupportedRequestError"),
    ("UnsupportedOperationError", "UnsupportedRequestError"),
    ("UnsupportedExecutionModeError", "UnsupportedRequestError"),
    ("UnavailableReferencePointError", LookupError),
    ("PropertyUnavailableError", ValueError),
    ("InvalidStateError", ValueError),
    ("OutOfDomainError", "InvalidStateError"),
    ("CalculationError", RuntimeError),
    ("ConvergenceError", "CalculationError"),
)


@pytest.mark.parametrize("name,base", ERROR_BASES)
def test_public_errors_preserve_typed_categories_and_one_library_base(name, base):
    error_type = getattr(jxp, name)
    expected_base = getattr(jxp, base) if isinstance(base, str) else base
    assert issubclass(error_type, expected_base)
    assert issubclass(error_type, jxp.JaxPropError)


def _assert_typed_failure(call, expected_type, *, request):
    """A generic exception or silent clipping cannot satisfy the contract."""
    try:
        call()
    except Exception as error:
        public_type = getattr(jxp, expected_type, None)
        assert public_type is not None, f"missing public error {expected_type}: {error}"
        assert isinstance(error, public_type), repr(error)
        assert error.fluid_name == "CO2"
        assert error.provider in {"coolprop", "coolprop_jax", "native"}
        assert error.model in {"real_fluid", "perfect_gas"}
        assert error.request == request
        assert error.reason
        assert hasattr(error, "supported")
        message = str(error)
        assert (
            "CO2" in message and error.provider in message and error.reason in message
        )
        assert str(request) in message
        return error
    assert False, f"expected {expected_type}; invalid request was accepted"


@pytest.mark.parametrize(
    "pressure,temperature",
    [
        (-1.0, 500.0),
        (0.0, 500.0),
        (1e5, -1.0),
        (1e5, 0.0),
        (np.nan, 500.0),
        (1e5, np.inf),
    ],
)
def test_invalid_perfect_gas_states_are_rejected_without_clipping(
    perfect_gas, pressure, temperature
):
    _assert_typed_failure(
        lambda: perfect_gas.get_state(jxp.PT_INPUTS, pressure, temperature),
        "InvalidStateError",
        request="PT_INPUTS",
    )


@pytest.mark.parametrize(
    "pressure,temperature",
    [
        (True, 500.0),
        ("100000", 500.0),
        (1e5 + 1j, 500.0),
        (np.array([1e5], dtype=object), 500.0),
        (np.ones((2,)), np.ones((3,))),
    ],
)
def test_malformed_numeric_inputs_raise_informative_validation_errors(
    source_fluid, pressure, temperature
):
    _assert_typed_failure(
        lambda: source_fluid.get_state(jxp.PT_INPUTS, pressure, temperature),
        "InvalidStateError",
        request="PT_INPUTS",
    )


def test_default_batch_failure_is_typed_and_reports_first_row_major_index(source_fluid):
    error = _assert_typed_failure(
        lambda: source_fluid.get_state(
            jxp.PT_INPUTS, np.array([[1e5, -1.0], [-2.0, 2e5]]), 500.0
        ),
        "InvalidStateError",
        request="PT_INPUTS",
    )
    assert tuple(error.index) == (0, 1)


def test_mask_mode_preserves_successful_elements_and_masks_only_failed_elements(
    perfect_gas,
):
    pressure = jnp.array([[1e5, -1.0], [np.nan, 2e5]])
    state = perfect_gas.get_state(jxp.PT_INPUTS, pressure, 500.0, errors="mask")
    np.testing.assert_array_equal(state.status, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(state.valid, [[True, False], [False, True]])
    for name in (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    ):
        value = np.asarray(state.raw_property(name))
        assert value.shape == pressure.shape
        assert np.all(np.isfinite(value[np.asarray(state.valid)]))
        assert np.all(np.isnan(value[~np.asarray(state.valid)]))
    np.testing.assert_allclose(np.asarray(state.p)[[0, 1], [0, 1]], [1e5, 2e5])


def test_outer_jit_requires_explicit_mask_policy():
    fluid = jxp.FluidJAX("CO2")
    _assert_typed_failure(
        lambda: jax.jit(
            lambda pressure: fluid.get_state(jxp.PT_INPUTS, pressure, 500.0).T
        )(1e5),
        "UnsupportedExecutionModeError",
        request="errors=raise",
    )


def test_jit_returns_full_state_identity_and_per_element_status(perfect_gas):
    evaluate = jax.jit(
        lambda pressure: perfect_gas.get_state(
            jxp.PT_INPUTS, pressure, 500.0, errors="mask"
        )
    )
    state = evaluate(jnp.array([1e5, -1.0]))
    assert state.fluid_name == "CO2" and state.identifier == "working_fluid"
    np.testing.assert_array_equal(state.status, [0, 1])
    np.testing.assert_array_equal(state.valid, [True, False])
    assert np.isfinite(state.raw_property("density")[0])
    assert np.isnan(state.raw_property("density")[1])


def test_supported_gradient_with_mask_policy_matches_independent_perfect_gas_identity(
    perfect_gas,
):
    def enthalpy(temperature):
        return perfect_gas.get_state(jxp.PT_INPUTS, 1e5, temperature, errors="mask").h

    cp = float(perfect_gas.get_state(jxp.PT_INPUTS, 1e5, 500.0).cp)
    np.testing.assert_allclose(jax.grad(enthalpy)(500.0), cp, rtol=1e-12)


def test_source_reference_records_match_across_real_fluid_and_perfect_gas(
    source_fluid, perfect_gas
):
    for name in ("critical_point", "triple_point_liquid", "triple_point_vapor"):
        source = source_fluid.references[name]
        approximate = perfect_gas.references[name]
        assert source.available and approximate.available
        assert source.classification == approximate.classification == "pure"
        assert source.backend == approximate.backend == "HEOS"
        assert source.reference_convention == approximate.reference_convention
        np.testing.assert_allclose(
            [getattr(source_fluid, name).p, getattr(source_fluid, name).T],
            [getattr(perfect_gas, name).p, getattr(perfect_gas, name).T],
            rtol=1e-12,
        )


def test_air_preserves_distinct_lower_saturation_endpoints_and_evaluable_ordinary_states():
    fluid = jxp.Fluid("Air")
    ordinary = fluid.get_state(jxp.PT_INPUTS, 1e5, 300.0)
    assert np.isfinite(ordinary.rho)
    liquid = fluid.references["triple_point_liquid"]
    vapor = fluid.references["triple_point_vapor"]
    assert fluid.references["critical_point"].available
    assert liquid.available and vapor.available
    assert liquid.classification == vapor.classification == "pseudo_pure"
    assert liquid.interpretation == vapor.interpretation == "lower_saturation_endpoint"
    assert fluid.triple_point_liquid.T == fluid.triple_point_vapor.T
    assert not np.isclose(fluid.triple_point_liquid.p, fluid.triple_point_vapor.p)


def test_state_collections_reject_same_name_and_stream_with_different_physical_models(
    source_fluid, perfect_gas
):
    source = source_fluid.get_state(jxp.PT_INPUTS, 1e5, 500.0)
    approximate = perfect_gas.get_state(jxp.PT_INPUTS, 1e5, 500.0)
    assert source.model != approximate.model
    assert source.reference_id != approximate.reference_id
    with pytest.raises(ValueError, match="model|reference"):
        source + approximate
    with pytest.raises(ValueError, match="model|reference"):
        jxp.FluidState.stack([source, approximate])


def test_unavailable_at_a_valid_state_is_distinct_from_an_unsupported_property(
    source_fluid,
):
    state = source_fluid.get_state(jxp.PT_INPUTS, 1e5, 500.0)
    assert state.supports_property("surface_tension")
    assert not bool(state.property_valid("surface_tension"))
    assert np.isnan(state.raw_property("surface_tension"))
    with pytest.raises(jxp.PropertyUnavailableError):
        state.surface_tension
    assert bool(state.valid)


@pytest.mark.parametrize(
    "settings",
    [
        {"name": ""},
        {"provider": "unknown"},
        {"model": "unknown"},
        {"model": "perfect_gas", "provider": "coolprop"},
        {"model": "real_fluid", "provider": "native"},
        {"table_options": {"N_h": 8}},
        {"model_options": {"unknown_option": 1}},
        {"provider": "bicubic", "table_options": {"h_min": 0.0}},
    ],
)
def test_incompatible_or_unknown_factory_settings_raise_configuration_errors(settings):
    arguments = {"name": "CO2"} | settings
    factory = jxp.create_fluid
    with pytest.raises(jxp.InvalidConfigurationError) as captured:
        factory(**arguments)
    assert captured.value.reason
    assert captured.value.request


def test_factory_defaults_select_direct_real_fluid_heos_without_a_stream_identifier():
    fluid = jxp.create_fluid("CO2")
    assert fluid.name == "CO2"
    assert fluid.provider == "coolprop"
    assert fluid.model == "real_fluid"
    assert fluid.backend == "HEOS"
    assert fluid.identifier is None


@pytest.mark.parametrize(
    "options,option_request",
    [
        ({"unknown_option": True}, "unknown_option"),
        ({"generalize_quality": "yes"}, "generalize_quality"),
        ({"errors": "ignore"}, "errors"),
    ],
)
def test_unknown_or_invalid_evaluation_options_are_explicit(
    source_fluid, options, option_request
):
    _assert_typed_failure(
        lambda: source_fluid.get_state(jxp.PT_INPUTS, 1e5, 500.0, **options),
        "UnsupportedOptionError",
        request=option_request,
    )


def test_perfect_gas_calibration_matches_source_enthalpy_entropy_and_heat_capacity():
    source = jxp.Fluid("CO2").get_state(jxp.PT_INPUTS, 1e5, 600.0)
    gas = jxp.FluidPerfectGas("CO2", T_ref=600.0, p_ref=1e5)
    state = gas.get_state(jxp.PT_INPUTS, 1e5, 600.0)
    for name in ("enthalpy", "entropy", "isobaric_heat_capacity"):
        np.testing.assert_allclose(
            getattr(state, name), getattr(source, name), rtol=1e-12
        )
    np.testing.assert_allclose(state.cp - state.cv, gas.constants.R, rtol=1e-12)
    np.testing.assert_allclose(state.gamma, state.cp / state.cv, rtol=1e-12)


def test_empty_broadcast_batch_preserves_shape_without_source_evaluation(
    source_fluid, monkeypatch
):
    from jaxprop.coolprop import core_calculations

    def forbidden_evaluation(*args, **kwargs):
        raise AssertionError("empty batch must not evaluate a numerical state")

    monkeypatch.setattr(
        core_calculations, "compute_properties_coolprop", forbidden_evaluation
    )
    state = source_fluid.get_state(jxp.PT_INPUTS, np.empty((0, 1)), 500.0)
    for name in (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    ):
        assert np.shape(getattr(state, name)) == (0, 1)
