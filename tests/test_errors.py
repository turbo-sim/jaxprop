"""Public exception hierarchy, diagnostics, and transport between workers."""

import pickle

import numpy as np
import pytest

from jaxprop import errors

ERROR_BASES = (
    (errors.JaxPropError, Exception),
    (errors.InvalidConfigurationError, ValueError),
    (errors.UnsupportedRequestError, ValueError),
    (errors.UnsupportedInputPairError, errors.UnsupportedRequestError),
    (errors.UnsupportedPropertyError, errors.UnsupportedRequestError),
    (errors.UnsupportedOptionError, errors.UnsupportedRequestError),
    (errors.UnsupportedOperationError, errors.UnsupportedRequestError),
    (errors.UnsupportedExecutionModeError, errors.UnsupportedRequestError),
    (errors.UnavailableReferencePointError, LookupError),
    (errors.PropertyUnavailableError, ValueError),
    (errors.InvalidStateError, ValueError),
    (errors.OutOfDomainError, errors.InvalidStateError),
    (errors.CalculationError, RuntimeError),
    (errors.ConvergenceError, errors.CalculationError),
)


@pytest.mark.parametrize("error_type,base_type", ERROR_BASES)
def test_errors_have_one_library_base_and_the_promised_python_category(
    error_type, base_type
):
    assert issubclass(error_type, base_type)
    assert issubclass(error_type, errors.JaxPropError)
    assert error_type.__name__ in errors.__all__


def test_error_context_and_message_explain_rejected_request_and_alternatives():
    error = errors.UnsupportedInputPairError(
        fluid_name="CO2",
        provider="native",
        model="perfect_gas",
        request="QT_INPUTS",
        reason="perfect gas has no phase equilibrium",
        supported=("PT_INPUTS", "HmassP_INPUTS"),
    )
    assert error.fluid_name == "CO2"
    assert error.provider == "native"
    assert error.model == "perfect_gas"
    assert error.request == "QT_INPUTS"
    assert error.reason == "perfect gas has no phase equilibrium"
    assert error.supported == ("PT_INPUTS", "HmassP_INPUTS")
    assert error.index is None and error.indices is None
    for part in (
        "CO2",
        "native",
        "perfect_gas",
        "QT_INPUTS",
        error.reason,
        *error.supported,
    ):
        assert part in str(error)


@pytest.mark.parametrize("index", [(), (1,), (2, 1)])
def test_numerical_failure_preserves_scalar_and_batched_indices(index):
    error = errors.InvalidStateError(
        request="PT_INPUTS", reason="pressure must be positive", index=index
    )
    assert error.index == index
    assert f"index={index}" in str(error)


def test_property_failure_preserves_all_indices_with_a_bounded_message():
    indices = np.arange(40).reshape(20, 2)
    error = errors.PropertyUnavailableError(
        request="surface_tension", reason="not on saturation", indices=indices
    )
    assert error.indices == tuple(tuple(row) for row in indices)
    assert "20 total" in str(error)
    assert len(str(error)) < 400


def test_context_is_complete_even_when_the_message_limits_long_source_diagnostics():
    reason = "source detail " * 1000
    supported = tuple(f"property_{index}" for index in range(100))
    error = errors.UnsupportedPropertyError(
        request="quality_mass", reason=reason, supported=supported
    )
    assert error.reason == reason
    assert error.supported == supported
    assert len(str(error)) < 500
    assert "100 total" in str(error)


def test_one_supported_name_is_preserved_as_one_choice():
    error = errors.UnsupportedInputPairError(
        request="QT_INPUTS", reason="gas only", supported="PT_INPUTS"
    )
    assert error.supported == ("PT_INPUTS",)
    assert "supported: PT_INPUTS" in str(error)


def test_underlying_provider_exception_is_retained_by_explicit_chaining():
    original = RuntimeError("source EOS rejected this state")
    with pytest.raises(errors.CalculationError) as captured:
        try:
            raise original
        except RuntimeError as cause:
            raise errors.CalculationError(
                fluid_name="CO2",
                provider="coolprop",
                model="real_fluid",
                request="PT_INPUTS",
                reason="source evaluation failed",
            ) from cause
    assert captured.value.__cause__ is original
    assert captured.value.__suppress_context__


@pytest.mark.parametrize("error_type,base_type", ERROR_BASES)
def test_errors_round_trip_through_pickle_with_context(error_type, base_type):
    original = error_type(
        fluid_name="CO2",
        provider="bicubic",
        model="real_fluid",
        request="HmassP_INPUTS",
        reason="outside enthalpy bounds",
        supported=("PT_INPUTS",),
        index=(2, 1),
        indices=((2, 1), (3, 0)),
    )
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is error_type
    assert isinstance(restored, base_type)
    assert restored.__dict__ == original.__dict__
    assert str(restored) == str(original)
    assert restored.args == original.args
