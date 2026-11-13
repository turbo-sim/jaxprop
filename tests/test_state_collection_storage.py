"""Collection storage, immutability and transformation regression checks."""

import pickle

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxprop import FluidState, UnsupportedPropertyError


def state(values, **kwargs):
    return FluidState(
        fluid_name="CO2",
        identifier="stream",
        model="real_fluid",
        reference_id=("source", "reference"),
        **({"provider": "coolprop", "error_policy": "mask"} | kwargs),
        pressure=values,
    )


def combine(first, second, operation):
    return FluidState.stack([first, second]) if operation == "stack" else first + second


def assert_readonly_numpy(current):
    leaves = (
        *current.to_dict().values(),
        current.status,
        *(current.property_valid(name) for name in current.available_properties),
    )
    for value in leaves:
        assert isinstance(value, np.ndarray)
        assert not value.flags.writeable


@pytest.mark.parametrize("operation", ["stack", "concatenate"])
def test_numpy_collections_preserve_masks_status_and_readonly_storage(operation):
    first = state(
        [[1.0], [2.0]],
        property_validity={"pressure": [[True], [False]]},
    )
    second = state([[3.0], [4.0]], provider="bicubic", status=[[0], [4]])
    result = combine(first, second, operation)
    expected = (
        np.array([[[1.0], [np.nan]], [[3.0], [np.nan]]])
        if operation == "stack"
        else np.array([[1.0], [np.nan], [3.0], [np.nan]])
    )
    expected_status = (
        np.array([[[0], [0]], [[0], [4]]])
        if operation == "stack"
        else np.array([[0], [0], [0], [4]])
    )
    assert_readonly_numpy(result)
    np.testing.assert_array_equal(result.raw_property("p"), expected)
    np.testing.assert_array_equal(result.status, expected_status)
    np.testing.assert_array_equal(result.property_valid("p"), np.isfinite(expected))
    assert result.provider == "mixed"
    assert result.providers == frozenset({"coolprop", "bicubic"})
    assert result.reference_id == first.reference_id
    assert result.identifier == first.identifier
    np.testing.assert_array_equal(first.raw_property("p"), [[1.0], [np.nan]])
    np.testing.assert_array_equal(second.raw_property("p"), [[3.0], [np.nan]])
    with pytest.raises(ValueError):
        result.to_dict()["pressure"][...] = 99.0

    for restored in (
        FluidState.from_data(result.to_data()),
        pickle.loads(pickle.dumps(result)),
    ):
        assert_readonly_numpy(restored)
        np.testing.assert_array_equal(restored.raw_property("p"), expected)
        np.testing.assert_array_equal(restored.status, expected_status)
        np.testing.assert_array_equal(
            restored.property_valid("p"), np.isfinite(expected)
        )
        assert restored.providers == result.providers


@pytest.mark.parametrize("operation", ["stack", "concatenate"])
@pytest.mark.parametrize("first_backend", ["numpy", "jax"])
def test_mixed_storage_collections_keep_jax_arrays(operation, first_backend):
    numpy_state = state(np.array([1.0, 2.0]))
    jax_state = state(jnp.array([3.0, 4.0]))
    first, second = (
        (numpy_state, jax_state)
        if first_backend == "numpy"
        else (jax_state, numpy_state)
    )
    result = combine(first, second, operation)
    expected = (
        np.stack([np.asarray(first.p), np.asarray(second.p)])
        if operation == "stack"
        else np.concatenate([np.asarray(first.p), np.asarray(second.p)])
    )
    for value in jax.tree_util.tree_leaves(result):
        assert isinstance(value, jax.Array)
    np.testing.assert_array_equal(result.p, expected)
    np.testing.assert_array_equal(result.property_valid("p"), np.ones_like(expected))


@pytest.mark.parametrize("operation", ["stack", "concatenate"])
@pytest.mark.parametrize("storage", ["numpy", "jax", "mixed"])
def test_signed_unsigned_status_combinations_remain_integer_codes(operation, storage):
    first_xp = np if storage in ("numpy", "mixed") else jnp
    second_xp = np if storage == "numpy" else jnp
    first = state(
        first_xp.asarray([1.0, 2.0]),
        status=first_xp.asarray([0, 4], dtype=np.int64),
    )
    second = state(
        second_xp.asarray([3.0, 4.0]),
        status=second_xp.asarray([1, 0], dtype=np.uint64),
    )
    result = combine(first, second, operation)
    expected_status = [[0, 4], [1, 0]] if operation == "stack" else [0, 4, 1, 0]
    expected_values = (
        [[1.0, np.nan], [np.nan, 4.0]]
        if operation == "stack"
        else [1.0, np.nan, np.nan, 4.0]
    )
    assert np.issubdtype(result.status.dtype, np.integer)
    np.testing.assert_array_equal(result.status, expected_status)
    np.testing.assert_array_equal(result.raw_property("p"), expected_values)
    np.testing.assert_array_equal(
        result.property_valid("p"), np.isfinite(expected_values)
    )
    assert isinstance(result.status, np.ndarray if storage == "numpy" else jax.Array)
    if storage == "numpy":
        assert_readonly_numpy(result)
    else:
        compiled = jax.jit(lambda a, b: combine(a, b, operation))(first, second)
        assert np.issubdtype(compiled.status.dtype, np.integer)
        np.testing.assert_array_equal(compiled.status, expected_status)
        np.testing.assert_array_equal(compiled.raw_property("p"), expected_values)


def test_numpy_reversal_and_indexing_keep_storage_and_readonly_status():
    original = state([1.0, 2.0, 3.0], status=[0, 4, 0])
    reversed_state = original.flipped()
    assert_readonly_numpy(reversed_state)
    np.testing.assert_array_equal(reversed_state.raw_property("p"), [3, np.nan, 1])
    np.testing.assert_array_equal(reversed_state.status, [0, 4, 0])
    assert reversed_state.at_index(0).raw_property("p") == 3.0
    np.testing.assert_array_equal(original.raw_property("p"), [1, np.nan, 3])


def test_empty_numpy_collections_and_scalar_promotion_preserve_exact_axes():
    empty = state(np.empty((0, 1)))
    stacked = FluidState.stack([empty, empty])
    assert stacked.shape == (2, 0, 1)
    assert_readonly_numpy(stacked)
    assert (empty + state([[2.0]])).shape == (1, 1)

    promoted = state(1.0) + state([2.0, 3.0])
    assert_readonly_numpy(promoted)
    assert promoted.shape == (3,)
    np.testing.assert_array_equal(promoted.p, [1.0, 2.0, 3.0])


def test_collection_property_intersection_can_be_empty_without_losing_status():
    first = state([1.0, 2.0], status=[0, 4])
    second = FluidState(
        **(first.to_data()["metadata"] | {"available_properties": {"temperature"}}),
        temperature=[300.0],
        status=[2],
    )
    result = first + second
    assert result.available_properties == frozenset()
    assert result.to_dict() == {}
    assert_readonly_numpy(result)
    np.testing.assert_array_equal(result.status, [0, 4, 2])
    with pytest.raises(UnsupportedPropertyError):
        result.p


@pytest.mark.parametrize("operation", ["stack", "concatenate"])
def test_collection_fast_assembly_keeps_error_policy_mismatch_explicit(operation):
    with pytest.raises(ValueError, match="error_policy"):
        combine(state([1.0]), state([2.0], error_policy="raise"), operation)


@pytest.mark.parametrize("transform", [jax.jit, eqx.filter_jit])
def test_transformed_collections_return_full_state_with_masks_and_status(transform):
    def evaluate(values):
        first = state(values, status=jnp.array([0, 4]))
        second = state(3 * values, provider="bicubic")
        return (first + second).flipped()

    result = transform(evaluate)(jnp.array([1.0, 2.0]))
    np.testing.assert_array_equal(result.raw_property("p"), [6, 3, np.nan, 1])
    np.testing.assert_array_equal(result.status, [0, 0, 4, 0])
    np.testing.assert_array_equal(result.property_valid("p"), [True, True, False, True])
    assert result.providers == frozenset({"coolprop", "bicubic"})


def test_vectorized_collection_and_gradient_preserve_values_and_derivatives():
    def evaluate(value):
        return FluidState.stack([state(value), state(3 * value)]).flipped()

    values = jnp.array([1.0, 2.0, 3.0])
    result = jax.jit(jax.vmap(evaluate))(values)
    np.testing.assert_array_equal(result.p, [[3.0, 1.0], [6.0, 2.0], [9.0, 3.0]])
    np.testing.assert_array_equal(result.status, np.zeros((3, 2), dtype=int))
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda x: evaluate(x).raw_property("p").sum()))(2.0),
        4.0,
    )
