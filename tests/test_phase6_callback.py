"""Source callback masks, eager errors and independently checked derivatives."""

import CoolProp.CoolProp as CP
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp
from utilities import get_available_backends


@pytest.fixture(scope="module")
def callback():
    return jxp.Fluid("CO2", method="coolprop_jax", identifier="callback-stream")


def assert_identity(state, fluid, errors):
    assert state.fluid_name == fluid.name
    assert state.identifier == fluid.identifier
    assert state.provider == "coolprop_jax"
    assert state.providers == frozenset({"coolprop_jax"})
    assert state.model == "real_fluid"
    assert state.reference_id == fluid.reference_id
    assert state.error_policy == errors


def assert_same_values(first, second):
    assert first.available_properties == second.available_properties
    np.testing.assert_array_equal(first.status, second.status)
    for name in first.available_properties:
        np.testing.assert_allclose(
            first.raw_property(name),
            second.raw_property(name),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            first.property_valid(name), second.property_valid(name)
        )


def test_eager_raising_failures_keep_callback_context_and_row_major_index(callback):
    with pytest.raises(jxp.InvalidStateError) as captured:
        callback.get_state(jxp.PT_INPUTS, [[1e5, -1.0], [-2.0, 2e5]], 500.0)
    error = captured.value
    assert error.fluid_name == "CO2"
    assert error.provider == "coolprop_jax"
    assert error.model == "real_fluid"
    assert error.request == "PT_INPUTS"
    assert error.index == (0, 1)
    assert error.reason


def test_eager_provider_error_preserves_original_cause(callback, monkeypatch):
    source_error = RuntimeError("controlled source failure")

    def fail(*args, **kwargs):
        raise jxp.CalculationError(
            fluid_name="CO2",
            provider="coolprop",
            model="real_fluid",
            request="PT_INPUTS",
            reason="controlled source failure",
            index=(1,),
        ) from source_error

    monkeypatch.setattr(callback.fluid, "_evaluate_state_data", fail)
    with pytest.raises(jxp.CalculationError) as captured:
        callback.get_state(jxp.PT_INPUTS, [1e5, 2e5], 500.0)
    assert captured.value.provider == "coolprop_jax"
    assert captured.value.index == (1,)
    assert captured.value.__cause__ is source_error


@pytest.mark.parametrize("mode", ["eager", "jit", "vmap"])
def test_mask_policy_preserves_successes_and_masks_only_invalid_elements(
    callback, mode
):
    pressure = jnp.asarray([[1e5, -1.0], [jnp.nan, 2e5]])
    evaluate = lambda p: callback.get_state(jxp.PT_INPUTS, p, 500.0, errors="mask")
    if mode == "jit":
        evaluate = jax.jit(evaluate)
    elif mode == "vmap":
        evaluate = jax.vmap(evaluate)
    state = evaluate(pressure)
    jax.block_until_ready(state)
    assert_identity(state, callback, "mask")
    assert state.shape == (2, 2)
    np.testing.assert_array_equal(state.status, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(state.valid, [[True, False], [False, True]])
    valid = np.asarray(state.valid)
    for name in state.available_properties:
        value = np.asarray(state.raw_property(name))
        assert value.shape == pressure.shape
        assert np.all(np.isnan(value[~valid])), name
    for name in (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    ):
        assert np.all(np.isfinite(np.asarray(state.raw_property(name))[valid])), name
    np.testing.assert_array_equal(
        np.asarray(state.raw_property("pressure"))[valid], [1e5, 2e5]
    )


@pytest.mark.parametrize("shape", [(), (1,), (2, 1), (0, 1)])
def test_broadcast_shapes_and_full_identity_survive_jit(callback, shape):
    enthalpy = jnp.full(shape, 700e3)
    pressure = jnp.asarray([7e6, 8e6, 9e6]) if len(shape) > 1 else 8e6
    expected_shape = np.broadcast_shapes(shape, np.shape(pressure))
    evaluate = lambda h, p: callback.get_state(jxp.HmassP_INPUTS, h, p, errors="mask")
    eager, compiled = evaluate(enthalpy, pressure), jax.jit(evaluate)(
        enthalpy, pressure
    )
    jax.block_until_ready((eager, compiled))
    assert_identity(compiled, callback, "mask")
    assert compiled.shape == expected_shape
    for name in compiled.available_properties:
        assert compiled.raw_property(name).shape == expected_shape
        assert compiled.property_valid(name).shape == expected_shape
    assert_same_values(eager, compiled)


@pytest.mark.parametrize("mode", ["jit", "closed_jit", "vmap", "grad"])
def test_transformed_raise_policy_is_explicit_even_with_constant_inputs(callback, mode):
    evaluate = lambda p: callback.get_state(jxp.PT_INPUTS, p, 500.0).T
    call = {
        "jit": lambda: jax.jit(evaluate)(1e5),
        "closed_jit": lambda: jax.jit(lambda: evaluate(1e5))(),
        "vmap": lambda: jax.vmap(evaluate)(jnp.asarray([1e5, 2e5])),
        "grad": lambda: jax.grad(evaluate)(1e5),
    }[mode]
    with pytest.raises(jxp.UnsupportedExecutionModeError) as captured:
        call()
    assert captured.value.provider == "coolprop_jax"
    assert captured.value.request == "errors=raise"
    assert "mask" in captured.value.reason


@pytest.mark.parametrize("errors", ["raise", "mask"])
def test_static_request_errors_precede_execution_mode(callback, errors):
    with pytest.raises(jxp.UnsupportedInputPairError):
        jax.jit(lambda x: callback.get_state(-999, x, 500.0, errors=errors).T)(1e5)
    with pytest.raises(jxp.UnsupportedOptionError):
        callback.get_state(jxp.PT_INPUTS, 1e5, 500.0, typo=True, errors=errors)
    with pytest.raises(jxp.InvalidStateError):
        callback.get_state(
            jxp.PT_INPUTS, [1e5, 2e5], [400.0, 500.0, 600.0], errors=errors
        )


def test_empty_mask_batch_performs_no_host_property_evaluation(callback, monkeypatch):
    from jaxprop.coolprop import core_calculations

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("Empty batch must not evaluate a thermodynamic point")

    monkeypatch.setattr(core_calculations, "compute_properties_coolprop", forbidden)
    state = jax.jit(
        lambda h: callback.get_state(jxp.HmassP_INPUTS, h, 8e6, errors="mask")
    )(jnp.empty((0,)))
    jax.block_until_ready(state)
    assert state.shape == (0,)
    assert state.status.size == 0
    assert not calls


@pytest.mark.parametrize("supersaturation", [False, True])
@pytest.mark.parametrize("generalize_quality", [False, True])
def test_diagnostic_options_forward_to_the_same_source(
    callback, supersaturation, generalize_quality
):
    source = callback.fluid
    options = dict(
        supersaturation=supersaturation, generalize_quality=generalize_quality
    )
    expected = source.get_state(jxp.PT_INPUTS, 1e5, 500.0, **options)
    eager = callback.get_state(jxp.PT_INPUTS, 1e5, 500.0, **options)
    compiled = jax.jit(
        lambda t: callback.get_state(jxp.PT_INPUTS, 1e5, t, errors="mask", **options)
    )(500.0)
    assert_identity(eager, callback, "raise")
    assert_same_values(eager, expected)
    assert_same_values(compiled, expected)
    assert ("pressure_saturation" in eager.available_properties) == supersaturation


@pytest.mark.parametrize(
    "input_type, first, second",
    [(jxp.PQ_INPUTS, 1e6, 0.0), (jxp.QT_INPUTS, 1.0, 270.0)],
)
def test_uniform_saturation_operation_matches_ordinary_source(
    callback, input_type, first, second
):
    assert "saturation" in callback.capabilities.phase_operations
    expected = callback.fluid.get_state(input_type, first, second)
    actual = callback.get_state_saturation(input_type, first, second)
    compiled = jax.jit(
        lambda value: callback.get_state_saturation(
            input_type, first, value, errors="mask"
        )
    )(second)
    assert_same_values(actual, expected)
    assert_same_values(compiled, expected)


def test_saturation_rejects_other_pairs_before_numerical_work(callback):
    with pytest.raises(jxp.UnsupportedOperationError) as captured:
        callback.get_state_saturation(jxp.PT_INPUTS, 1e5, 500.0, errors="mask")
    assert captured.value.request == "saturation"
    assert set(captured.value.supported) == {"PQ_INPUTS", "QT_INPUTS"}


def test_callback_preserves_source_domain_description_and_bounds(callback):
    assert callback.capabilities.domain == callback.fluid.capabilities.domain
    pseudo = jxp.Fluid("Air", method="coolprop_jax")
    assert pseudo.capabilities.domain == pseudo.fluid.capabilities.domain
    assert (
        "QT flashes accept only quality endpoints"
        in pseudo.capabilities.domain.description
    )


def test_compiled_two_phase_diagnostics_keep_requested_pressure_and_temperature():
    fluid = jxp.Fluid("Water", method="coolprop_jax")
    state = jax.jit(
        lambda q: fluid.get_state(
            jxp.PQ_INPUTS, 1e5, q, supersaturation=True, errors="mask"
        )
    )(0.3)
    reference = CP.AbstractState("HEOS", "Water")
    reference.update(CP.PQ_INPUTS, 1e5, 0.3)
    assert state.status == 0
    np.testing.assert_allclose(state.p, reference.p(), rtol=1e-12)
    np.testing.assert_allclose(state.T, reference.T(), rtol=1e-12)
    np.testing.assert_allclose(state.pressure_saturation, reference.p(), rtol=1e-10)
    np.testing.assert_allclose(state.supersaturation_degree, 0.0, atol=1e-8)
    assert np.isfinite(state.subcooling)
    assert np.isfinite(state.superheating)


@pytest.mark.parametrize("backend", get_available_backends())
@pytest.mark.parametrize(
    "name, pressure, temperature",
    [
        ("CO2", 1e5, 500.0),
        ("CO2", 8e6, 450.0),
        ("Water", 6e6, 700.0),
        ("Air", 1e6, 350.0),
    ],
)
@pytest.mark.parametrize(
    "derivative", ["dh_dT_P", "ds_dT_P", "drho_dP_T", "dT_dh_P", "dT_dP_s"]
)
def test_first_derivatives_match_independent_source_partial_derivatives(
    backend, name, pressure, temperature, derivative
):
    fluid = jxp.Fluid(name, method="coolprop_jax", backend=backend)
    state = CP.AbstractState(backend, name)
    state.update(CP.PT_INPUTS, pressure, temperature)
    if derivative == "dh_dT_P":
        function = lambda value: fluid.get_state(
            jxp.PT_INPUTS, pressure, value, errors="mask"
        ).h
        value = temperature
        expected = state.first_partial_deriv(CP.iHmass, CP.iT, CP.iP)
    elif derivative == "ds_dT_P":
        function = lambda value: fluid.get_state(
            jxp.PT_INPUTS, pressure, value, errors="mask"
        ).s
        value = temperature
        expected = state.first_partial_deriv(CP.iSmass, CP.iT, CP.iP)
    elif derivative == "drho_dP_T":
        function = lambda value: fluid.get_state(
            jxp.PT_INPUTS, value, temperature, errors="mask"
        ).d
        value = pressure
        expected = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
    elif derivative == "dT_dh_P":
        function = lambda value: fluid.get_state(
            jxp.HmassP_INPUTS, value, pressure, errors="mask"
        ).T
        value = state.hmass()
        expected = state.first_partial_deriv(CP.iT, CP.iHmass, CP.iP)
    else:
        function = lambda value: fluid.get_state(
            jxp.PSmass_INPUTS, value, state.smass(), errors="mask"
        ).T
        value = pressure
        expected = state.first_partial_deriv(CP.iT, CP.iP, CP.iSmass)
    actual = jax.jit(jax.grad(function))(value)
    assert np.isfinite(actual) and np.isfinite(expected)
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-9)


def test_integer_constant_input_has_zero_tangent_without_extra_difference(
    callback, monkeypatch
):
    evaluations = []
    evaluate = callback.fluid._evaluate_state_data

    def record(*args, **kwargs):
        evaluations.append(None)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(callback.fluid, "_evaluate_state_data", record)
    derivative = jax.grad(
        lambda t: callback.get_state(jxp.PT_INPUTS, 100000, t, errors="mask").h
    )(500.0)
    jax.block_until_ready(derivative)
    assert len(evaluations) == 2
    reference = CP.AbstractState("HEOS", "CO2")
    reference.update(CP.PT_INPUTS, 100000, 500.0)
    np.testing.assert_allclose(derivative, reference.cpmass(), rtol=5e-5)


def test_joint_input_jvp_matches_analytic_direction_and_integer_status(callback):
    enthalpy, pressure = 700e3, 8e6
    enthalpy_dot, pressure_dot = 350.0, -5000.0

    def evaluate(h, p):
        state = callback.get_state(jxp.HmassP_INPUTS, h, p, errors="mask")
        return state.T, state.status

    (temperature, status), (temperature_dot, status_dot) = jax.jit(
        lambda h, p: jax.jvp(evaluate, (h, p), (enthalpy_dot, pressure_dot))
    )(enthalpy, pressure)
    reference = CP.AbstractState("HEOS", "CO2")
    reference.update(CP.HmassP_INPUTS, enthalpy, pressure)
    expected = (
        reference.first_partial_deriv(CP.iT, CP.iHmass, CP.iP) * enthalpy_dot
        + reference.first_partial_deriv(CP.iT, CP.iP, CP.iHmass) * pressure_dot
    )
    np.testing.assert_allclose(temperature, reference.T(), rtol=1e-12)
    np.testing.assert_allclose(temperature_dot, expected, rtol=5e-5)
    assert status == 0
    assert status.dtype == jnp.int32
    assert status_dot.dtype == jax.dtypes.float0


def test_higher_derivatives_raise_informative_execution_error(callback):
    function = lambda h: callback.get_state(jxp.HmassP_INPUTS, h, 8e6, errors="mask").T
    with pytest.raises(jxp.UnsupportedExecutionModeError) as captured:
        jax.grad(jax.grad(function))(700e3)
    assert captured.value.request == "derivative_order"
    assert "first-order" in captured.value.reason
    assert "higher-order" in captured.value.reason


def test_legacy_false_constructor_is_rejected_before_backend_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected constructor option reached CoolProp")

    monkeypatch.setattr(CP, "AbstractState", forbidden)
    with pytest.raises(jxp.InvalidConfigurationError) as captured:
        jxp.FluidJAX("CO2", exceptions=False)
    assert captured.value.provider == "coolprop_jax"
    assert captured.value.request == "exceptions"
    assert "mask" in captured.value.reason


def test_two_independent_callback_calls_in_one_compiled_consumer_are_stable(callback):
    @jax.jit
    def evaluate(h):
        cold = callback.get_state(jxp.HmassP_INPUTS, h, 8e6, errors="mask")
        hot = callback.get_state(jxp.HmassP_INPUTS, h + 50e3, 8e6, errors="mask")
        return cold.T, hot.T

    first, second = evaluate(jnp.asarray([650e3, 700e3, 750e3]))
    assert np.all(np.asarray(second) > np.asarray(first))
    expected = callback.fluid.get_state(
        jxp.HmassP_INPUTS, np.asarray([650e3, 700e3, 750e3]), 8e6
    )
    np.testing.assert_allclose(first, expected.T, rtol=1e-12)
