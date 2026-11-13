"""Clipping is recoverable state provenance, distinct from numerical failure."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp


def state(pressure, **kwargs):
    return jxp.FluidState(
        fluid_name="Air",
        identifier="stream",
        provider="native",
        model="perfect_gas",
        reference_id=("perfect_gas", "calibration"),
        error_policy="clip",
        pressure=pressure,
        **kwargs,
    )


@pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
@pytest.mark.parametrize("shape", [(), (2, 1), (0, 3)])
def test_absent_clipping_stays_lazy_without_numerical_leaves(xp, shape):
    current = state(xp.full(shape, 1e5))
    assert current._clipped is None
    leaves_before = jax.tree_util.tree_leaves(current)
    assert len(leaves_before) == 2  # pressure and integer status only
    flags = current.clipped
    assert flags.shape == current.shape == shape
    assert flags.dtype == np.bool_
    np.testing.assert_array_equal(flags, np.zeros(shape, dtype=bool))
    assert current._clipped is None
    assert len(jax.tree_util.tree_leaves(current)) == len(leaves_before)
    if xp is np:
        assert not flags.flags.writeable


@pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
def test_clipping_broadcasts_with_state_and_does_not_mark_failed_rows(xp):
    current = state(
        xp.asarray([[1.0, 2.0, 3.0]]),
        clipped=xp.asarray([[True], [False]]),
        status=xp.asarray([0, 4, 0]),
    )
    assert current.shape == (2, 3)
    np.testing.assert_array_equal(
        current.clipped, [[True, False, True], [False, False, False]]
    )
    np.testing.assert_array_equal(current.valid, [[True, False, True]] * 2)
    np.testing.assert_array_equal(current.raw_property("p"), [[1.0, np.nan, 3.0]] * 2)
    assert current.error_policy == "clip"
    assert "clipped" not in current.to_dict()
    assert "clipped" not in current.available_properties
    if xp is np:
        assert not current.clipped.flags.writeable


def test_clipping_input_does_not_share_mutable_numpy_owners():
    owner = np.array([True, False, True, False])
    flags = owner[::2]
    flags.setflags(write=False)
    current = state([1.0, 2.0], clipped=flags)
    owner[:] = False
    np.testing.assert_array_equal(current.clipped, [True, True])
    assert not current.clipped.flags.writeable


@pytest.mark.parametrize(
    "flags", [1, [0, 1], [0.0, 1.0], ["True", "False"], [[True], [False, True]]]
)
def test_nonboolean_or_malformed_clipping_flags_raise_typed_errors(flags):
    with pytest.raises(jxp.InvalidStateError) as captured:
        state([1.0, 2.0], clipped=flags)
    assert captured.value.request == "clipped"
    assert "boolean" in captured.value.reason
    assert captured.value.fluid_name == "Air"


def test_incompatible_clipping_shape_raises_typed_shape_error():
    with pytest.raises(jxp.InvalidStateError) as captured:
        state(np.ones((2, 3)), clipped=np.ones((4,), dtype=bool))
    assert captured.value.request == "shape"


@pytest.mark.parametrize("operation", ["stack", "concatenate"])
@pytest.mark.parametrize("storage", ["numpy", "jax", "mixed"])
def test_collections_preserve_explicit_and_absent_clipping(operation, storage):
    first_xp = np if storage in ("numpy", "mixed") else jnp
    second_xp = np if storage == "numpy" else jnp
    first = state(
        first_xp.asarray([[1.0], [2.0]]),
        clipped=first_xp.asarray([[True], [True]]),
        status=first_xp.asarray([[0], [2]]),
    )
    second = state(second_xp.asarray([[3.0], [4.0]]))
    combine = (
        (lambda a, b: jxp.FluidState.stack([a, b]))
        if operation == "stack"
        else (lambda a, b: a + b)
    )
    actual = combine(first, second)
    expected_flags = (
        [[[True], [False]], [[False], [False]]]
        if operation == "stack"
        else [[True], [False], [False], [False]]
    )
    np.testing.assert_array_equal(actual.clipped, expected_flags)
    assert actual.clipped.shape == actual.shape
    assert second._clipped is None
    if storage == "numpy":
        assert isinstance(actual.clipped, np.ndarray)
        assert not actual.clipped.flags.writeable
    else:
        compiled = jax.jit(combine)(first, second)
        assert isinstance(compiled.clipped, jax.Array)
        np.testing.assert_array_equal(compiled.clipped, expected_flags)


@pytest.mark.parametrize("operation", ["stack", "concatenate"])
def test_unclipped_collections_keep_absent_storage(operation):
    first, second = state([1.0, 2.0]), state([3.0, 4.0])
    actual = (
        jxp.FluidState.stack([first, second])
        if operation == "stack"
        else first + second
    )
    assert actual._clipped is None
    assert len(jax.tree_util.tree_leaves(actual)) == 2
    assert not np.any(actual.clipped)


@pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
@pytest.mark.parametrize("has_flags", [False, True])
def test_roundtrips_preserve_clipping_storage_and_state_values(xp, has_flags):
    original = state(
        xp.asarray([1.0, 2.0, 3.0]),
        clipped=xp.asarray([True, False, True]) if has_flags else None,
        status=xp.asarray([0, 0, 1]),
    )
    for restored in (
        jxp.FluidState.from_data(original.to_data()),
        pickle.loads(pickle.dumps(original)),
    ):
        np.testing.assert_array_equal(restored.clipped, original.clipped)
        np.testing.assert_array_equal(restored.status, original.status)
        np.testing.assert_array_equal(restored.p, original.p)
        assert restored.error_policy == "clip"
        assert (restored._clipped is None) == (not has_flags)
        if xp is np:
            assert not restored.clipped.flags.writeable


def test_from_data_accepts_prior_unclipped_payload_without_clipping_key():
    exported = state([1.0, 2.0]).to_data()
    exported.pop("clipped")
    restored = jxp.FluidState.from_data(exported)
    assert restored._clipped is None
    np.testing.assert_array_equal(restored.clipped, [False, False])


@pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
def test_index_and_reverse_preserve_clipping_and_storage(xp):
    original = state(
        xp.asarray([[1.0], [2.0], [3.0]]),
        clipped=xp.asarray([[True], [False], [False]]),
    )
    np.testing.assert_array_equal(
        original.flipped().clipped, [[False], [False], [True]]
    )
    np.testing.assert_array_equal(original.at_index(0).clipped, [True])
    np.testing.assert_array_equal(original.at_index(1).clipped, [False])
    if xp is np:
        assert not original.flipped().clipped.flags.writeable
        assert not original.at_index(0).clipped.flags.writeable


def test_jit_vmap_and_grad_keep_clipping_distinct_from_success():
    def evaluate(values):
        return state(jnp.maximum(values, 1.0), clipped=values < 1.0)

    inputs = jnp.asarray([-2.0, 2.0, 4.0])
    compiled = jax.jit(evaluate)(inputs)
    vectorized = jax.jit(jax.vmap(evaluate))(inputs)
    for current in (compiled, vectorized):
        np.testing.assert_array_equal(current.p, [1.0, 2.0, 4.0])
        np.testing.assert_array_equal(current.clipped, [True, False, False])
        np.testing.assert_array_equal(current.status, [0, 0, 0])
        np.testing.assert_array_equal(current.property_valid("p"), [True] * 3)
    gradient = jax.jit(jax.grad(lambda values: evaluate(values).p.sum()))(inputs)
    np.testing.assert_array_equal(gradient, [0.0, 1.0, 1.0])


def test_supplied_false_flags_keep_the_clip_output_structure_stable():
    @jax.jit
    def evaluate(values):
        return state(jnp.maximum(values, 1.0), clipped=values < 1.0)

    recovered = evaluate(jnp.asarray([-2.0, 2.0]))
    unchanged = evaluate(jnp.asarray([3.0, 4.0]))
    assert unchanged._clipped is not None
    assert jax.tree_util.tree_structure(recovered) == jax.tree_util.tree_structure(
        unchanged
    )
    np.testing.assert_array_equal(unchanged.clipped, [False, False])


def test_empty_clipping_batches_preserve_all_axes_and_roundtrip():
    current = state(np.empty((0, 2)), clipped=np.empty((0, 2), dtype=bool))
    assert current.shape == current.clipped.shape == (0, 2)
    assert jxp.FluidState.stack([current, current]).clipped.shape == (2, 0, 2)
    assert jxp.FluidState.from_data(current.to_data()).clipped.shape == (0, 2)
    assert not current.clipped.flags.writeable
