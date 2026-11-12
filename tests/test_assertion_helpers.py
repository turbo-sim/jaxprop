"""Ensure numerical test assertions actually inspect every returned value."""

import jax.numpy as jnp
import numpy as np
import pytest

from utilities import assert_consistent_values


@pytest.fixture
def comparison():
    return dict(
        prop_name="temperature",
        fluid_name="CO2",
        backend="HEOS",
        state_label="assertion_test",
        input_type="PT_INPUTS",
        prop1_name="p",
        prop1=1e5,
        prop2_name="T",
        prop2=300.0,
        tolerance=1e-6,
        log_list=[],
    )


@pytest.mark.parametrize("convert", [float, np.float64, np.asarray, jnp.asarray])
def test_scalar_numeric_types_are_compared_and_mismatches_rejected(convert, comparison):
    assert assert_consistent_values(convert(300), convert(300), **comparison)
    with pytest.raises(AssertionError, match="temperature"):
        assert_consistent_values(convert(300), convert(9999), **comparison)
    assert len(comparison["log_list"]) == 2
    assert [entry["finite_count"] for entry in comparison["log_list"]] == [1, 1]
    assert not comparison["log_list"][-1]["passed"]


@pytest.mark.parametrize("shape", [(1,), (3,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("convert", [np.asarray, jnp.asarray])
def test_all_array_elements_are_checked(shape, convert, comparison):
    reference = np.full(shape, 300.0)
    wrong = reference.copy()
    wrong.flat[-1] += 1000
    assert assert_consistent_values(convert(reference), convert(reference), **comparison)
    with pytest.raises(AssertionError):
        assert_consistent_values(convert(reference), convert(wrong), **comparison)
    assert comparison["log_list"][-1]["finite_count"] == reference.size


@pytest.mark.parametrize(
    "reference,calculated",
    [(np.nan, 1.0), (1.0, np.nan), (np.inf, np.inf), (1.0, -np.inf)],
)
def test_unexpected_nonfinite_values_fail(reference, calculated, comparison):
    with pytest.raises(AssertionError):
        assert_consistent_values(reference, calculated, **comparison)


@pytest.mark.parametrize(
    "property_name", ["temperature", "T", "pressure", "h", "s", "rho", "u"]
)
def test_required_properties_cannot_pass_as_paired_nan(property_name, comparison):
    comparison["prop_name"] = property_name
    with pytest.raises(AssertionError):
        assert_consistent_values(np.nan, np.nan, **comparison)


def test_unavailable_optional_fields_are_counted_without_hiding_finite_entries(comparison):
    comparison["prop_name"] = "surface_tension"
    assert assert_consistent_values([np.nan, 0.07], [np.nan, 0.07], **comparison)
    assert comparison["log_list"][-1]["finite_count"] == 1
    assert comparison["log_list"][-1]["unavailable_count"] == 1
    with pytest.raises(AssertionError):
        assert_consistent_values([np.nan, 0.07], [np.nan, np.nan], **comparison)


@pytest.mark.parametrize(
    "reference,calculated",
    [([300.0], 300.0), (np.ones((2, 1)), np.ones((1, 2))), ([], [])],
)
def test_malformed_or_empty_shapes_cannot_pass(reference, calculated, comparison):
    with pytest.raises(AssertionError):
        assert_consistent_values(reference, calculated, **comparison)


@pytest.mark.parametrize("value", ["300", None, {"temperature": 300}, 300 + 1j])
def test_nonnumeric_inputs_cannot_be_silently_skipped(value, comparison):
    with pytest.raises(TypeError):
        assert_consistent_values(value, value, **comparison)


@pytest.mark.parametrize("setting", [-1.0, np.nan, np.inf])
def test_invalid_tolerances_are_rejected(setting, comparison):
    comparison["tolerance"] = setting
    with pytest.raises(ValueError):
        assert_consistent_values(300.0, 300.0, **comparison)


def test_absolute_and_relative_tolerances_apply_elementwise(comparison):
    assert assert_consistent_values(
        [0.0, 1e9], [1e-5, 1e9 + 10], atol=1e-4, rtol=1e-7, **comparison
    )
    with pytest.raises(AssertionError):
        assert_consistent_values(
            [0.0, 1e9], [1e-3, 1e9 + 1000], atol=1e-4, rtol=1e-7, **comparison
        )


def test_exact_zero_tolerances_and_boolean_properties(comparison):
    assert assert_consistent_values(
        [0.0, 300.0], [0.0, 300.0], atol=0, rtol=0, **comparison
    )
    comparison["prop_name"] = "is_two_phase"
    assert assert_consistent_values([True, False], [True, False], **comparison)
    with pytest.raises(AssertionError):
        assert_consistent_values([True, False], [False, False], **comparison)


def test_diagnostic_mode_records_failed_comparisons(comparison):
    assert not assert_consistent_values(300.0, 9999.0, raise_error=False, **comparison)
    assert not comparison["log_list"][-1]["passed"]
