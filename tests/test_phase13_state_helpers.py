"""Public state method bindings stay dynamic across JAX transformations."""

from dataclasses import FrozenInstanceError
import inspect
import pickle
import weakref

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jp


def assert_equal_tree(actual, expected):
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    assert actual_tree == expected_tree
    for actual_value, expected_value in zip(actual_leaves, expected_leaves):
        np.testing.assert_array_equal(actual_value, expected_value)


def state(values, *, storage="jax", policy="mask", clipped=None):
    xp = jnp if storage == "jax" else np
    values = xp.asarray(values, dtype=float)
    return jp.FluidState(
        fluid_name="Air",
        identifier="state-helper-check",
        provider="native",
        model="perfect_gas",
        reference_id=("test", "fixed-reference"),
        error_policy=policy,
        pressure=values,
        temperature=values + 300.0,
        conductivity=xp.full_like(values, np.nan),
        clipped=clipped,
    )


@pytest.mark.parametrize(
    "method,arguments", [("raw_property", ("p",)), ("at_index", (1,))]
)
def test_public_method_introspection_class_calls_and_immutable_binding(
    method, arguments
):
    original = state([1.0, 2.0, 3.0])
    bound = getattr(original, method)
    unbound = getattr(jp.FluidState, method)
    assert bound.__self__ is original
    assert bound.__func__ is unbound
    assert bound.__wrapped__.__self__ is original
    assert bound.__name__ == unbound.__name__
    assert bound.__qualname__ == unbound.__qualname__
    assert bound.__module__ == unbound.__module__
    assert bound.__doc__ == unbound.__doc__
    expected_parameters = list(inspect.signature(unbound).parameters)[1:]
    assert list(inspect.signature(bound).parameters) == expected_parameters
    assert_equal_tree(bound, getattr(original, method))
    assert_equal_tree(bound(*arguments), unbound(original, *arguments))
    with pytest.raises(FrozenInstanceError):
        bound.__self__ = state([4.0, 5.0, 6.0])


@pytest.mark.parametrize("transform", [jax.jit, eqx.filter_jit])
def test_passed_raw_method_retains_dynamic_state_and_reuses_trace(transform):
    """A plain Python bound method would capture state as a static callable."""
    traces = []

    def read(method):
        traces.append(None)
        return method("pressure")

    evaluate = transform(read)
    first, second = state([1.0, 2.0]), state([3.0, 4.0])
    np.testing.assert_array_equal(evaluate(first.raw_property), [1.0, 2.0])
    np.testing.assert_array_equal(evaluate(second.raw_property), [3.0, 4.0])
    assert len(traces) == 1


def test_passed_raw_method_supports_grad_vmap_and_tree_updates():
    original = state([1.0, 2.0, 3.0])
    bound = original.raw_property
    gradient = eqx.filter_grad(lambda method: jnp.sum(method("p") ** 2))(bound)
    np.testing.assert_array_equal(gradient.__self__._data["pressure"], [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(
        jax.vmap(lambda method: method("p"))(bound), [1.0, 2.0, 3.0]
    )
    updated = eqx.tree_at(
        lambda method: method.__self__._data["pressure"],
        bound,
        jnp.array([4.0, 5.0, 6.0]),
    )
    np.testing.assert_array_equal(updated("p"), [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(bound("p"), [1.0, 2.0, 3.0])


def test_direct_jit_accepts_bound_methods_as_weakrefable_callable_objects():
    original = state([1.0, 2.0, 3.0])
    getter = original.raw_property
    assert weakref.ref(getter)() is getter
    assert isinstance(hash(getter), int)
    np.testing.assert_array_equal(
        jax.jit(getter, static_argnums=0)("p"), [1.0, 2.0, 3.0]
    )
    selected = jax.jit(original.at_index, static_argnums=0)(1)
    np.testing.assert_array_equal(selected.raw_property("p"), 2.0)


@pytest.mark.parametrize(
    "method,arguments", [("raw_property", ("T",)), ("at_index", (-1,))]
)
def test_bound_method_pytree_and_pickle_roundtrips_preserve_callable(method, arguments):
    original = state([1.0, 2.0, 3.0], storage="numpy")
    bound = getattr(original, method)
    leaves, definition = jax.tree_util.tree_flatten(bound)
    for restored in (
        jax.tree_util.tree_unflatten(definition, leaves),
        pickle.loads(pickle.dumps(bound)),
    ):
        assert restored.__func__ is bound.__func__
        assert restored.__self__.reference_id == original.reference_id
        assert_equal_tree(restored(*arguments), bound(*arguments))
        assert not restored.__self__.raw_property("p").flags.writeable


def test_raw_property_retains_enabled_schema_and_unavailable_value_errors():
    original = state([1.0, 2.0], policy="raise")
    assert original.raw_property("T") is original.raw_property("temperature")
    assert np.all(np.isnan(original.raw_property("conductivity")))
    with pytest.raises(jp.PropertyUnavailableError):
        _ = original.conductivity
    with pytest.raises(jp.UnsupportedPropertyError):
        original.raw_property("density")
    with pytest.raises(KeyError, match="Unknown property alias"):
        original.raw_property("not_a_property")


@pytest.mark.parametrize("storage", ["numpy", "jax"])
@pytest.mark.parametrize("clipping", [False, True])
def test_index_preserves_complete_values_status_masks_metadata_and_storage(
    storage, clipping
):
    xp = np if storage == "numpy" else jnp
    values = xp.arange(12.0).reshape(3, 4) + 1
    status = xp.asarray([[0, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0]], dtype=xp.int32)
    flags = xp.asarray([[False] * 4, [True] * 4, [False] * 4]) if clipping else None
    original = jp.FluidState(
        fluid_name="CO2",
        identifier="path",
        provider="native",
        model="test",
        reference_id=("reference", 1),
        error_policy="clip" if clipping else "mask",
        pressure=values,
        temperature=values + 300,
        conductivity=xp.full_like(values, np.nan),
        status=status,
        clipped=flags,
    )
    selected = original.at_index(1)
    assert selected.shape == (4,)
    assert selected._metadata() == original._metadata()
    assert (selected._clipped is None) == (original._clipped is None)
    for name in original.available_properties:
        np.testing.assert_array_equal(
            selected.raw_property(name), np.asarray(original.raw_property(name))[1]
        )
        np.testing.assert_array_equal(
            selected.property_valid(name), np.asarray(original.property_valid(name))[1]
        )
    np.testing.assert_array_equal(selected.status, [0, 4, 0, 0])
    np.testing.assert_array_equal(
        selected.clipped, [clipping, False, clipping, clipping]
    )
    if storage == "numpy":
        for value in (*selected.to_dict().values(), selected.status, selected.clipped):
            assert isinstance(value, np.ndarray)
            assert not value.flags.writeable
        assert np.shares_memory(
            selected.raw_property("pressure"), original.raw_property("pressure")
        )
    else:
        assert all(
            isinstance(value, jax.Array) for value in selected.to_dict().values()
        )
    with pytest.raises(FrozenInstanceError):
        selected.identifier = "changed"


def test_index_nested_transforms_and_static_bounds_retain_gather_derivatives():
    values = jnp.arange(12.0).reshape(3, 4)
    original = state(values)
    selected = jax.jit(lambda current: current.at_index(-1))(original)
    np.testing.assert_array_equal(selected.raw_property("p"), values[-1])
    derivative = jax.jit(
        jax.grad(lambda x: jnp.sum(state(x * x).at_index(1).raw_property("p")))
    )(values)
    np.testing.assert_array_equal(
        derivative, np.asarray([[0.0] * 4, [8.0, 10.0, 12.0, 14.0], [0.0] * 4])
    )
    vectorized = jax.jit(
        jax.vmap(lambda row: state(row).at_index(1).raw_property("p"))
    )(values)
    np.testing.assert_array_equal(vectorized, values[:, 1])
    with pytest.raises(jp.UnsupportedExecutionModeError, match="static integer"):
        jax.jit(lambda current, index: current.at_index(index))(
            original, jnp.asarray(1)
        )


def test_index_scalar_identity_and_bounds_are_enforced_before_jax_gather():
    scalar = state(3.0)
    assert scalar.at_index(0) is scalar and scalar.at_index(-1) is scalar
    for invalid in (1, -2):
        with pytest.raises(IndexError):
            scalar.at_index(invalid)
    batch = state([1.0, 2.0, 3.0])
    for invalid in (3, -4):
        with pytest.raises(IndexError):
            batch.at_index(invalid)
    for invalid in (True, np.bool_(False), 1.0, slice(0, 1)):
        with pytest.raises(TypeError):
            batch.at_index(invalid)


def test_index_tangent_state_retains_float0_bookkeeping():
    _, tangent = jax.jvp(
        lambda x: state(x), (jnp.array([1.0, 2.0, 3.0]),), (jnp.ones(3),)
    )
    selected = tangent.at_index(1)
    assert selected.status.dtype == jax.dtypes.float0
    np.testing.assert_array_equal(selected.raw_property("p"), 1.0)


@pytest.mark.parametrize("size", [16, 128])
def test_small_and_large_native_batches_preserve_failed_rows_clipping_and_gradients(
    size,
):
    """Both native assembly paths retain failed-row masks and derivatives."""
    template = jp.FluidState(pressure=1e5, temperature=300.0, error_policy="clip")
    metadata = template._metadata()

    def evaluate(values, codes):
        return jp.FluidState._from_native(
            metadata,
            {"pressure": values, "temperature": values + 300.0},
            codes,
            clipped=values < 0,
        )

    values = jnp.linspace(-2.0, 2.0, size)
    codes = jnp.where(jnp.arange(size) % 3 == 0, 4, 0).astype(jnp.int32)
    actual = jax.jit(evaluate)(values, codes)
    np.testing.assert_array_equal(
        actual.raw_property("p"), np.where(np.asarray(codes) == 0, values, np.nan)
    )
    np.testing.assert_array_equal(
        actual.clipped, (np.asarray(values) < 0) & (np.asarray(codes) == 0)
    )
    derivative = jax.jit(
        jax.grad(lambda x: jnp.nansum(evaluate(x, codes).raw_property("p")))
    )(values)
    np.testing.assert_array_equal(derivative, np.asarray(codes) == 0)
