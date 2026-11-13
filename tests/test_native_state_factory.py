"""Trusted native state assembly preserves the public state contract."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp


def metadata(policy="mask"):
    return dict(
        fluid_name="Air",
        identifier="native-stream",
        provider="native",
        providers=frozenset({"native"}),
        model="perfect_gas",
        reference_id=("perfect_gas", "prepared-constants"),
        available_properties=frozenset({"pressure", "temperature", "conductivity"}),
        error_policy=policy,
    )


def native(values, status, clipped=None, *, policy="mask"):
    xp = jnp if isinstance(values, (jax.Array, jax.core.Tracer)) else np
    return jxp.FluidState._from_native(
        metadata(policy),
        {
            "pressure": values,
            "temperature": values + 300.0,
            "conductivity": xp.full_like(values, np.nan),
        },
        status,
        clipped,
    )


@pytest.mark.parametrize("storage", ["numpy", "jax", "compiled"])
@pytest.mark.parametrize("policy", ["raise", "mask", "clip"])
def test_trusted_native_matches_public_constructor_values_and_masks(storage, policy):
    xp = np if storage == "numpy" else jnp
    values = xp.asarray([[1.0], [2.0], [3.0]])
    status = xp.asarray([[0], [4], [0]], dtype=np.int32)
    clipped = xp.asarray([[False], [True], [True]]) if policy == "clip" else None
    evaluate = lambda x, codes, flags: native(x, codes, flags, policy=policy)
    actual = (
        jax.jit(evaluate)(values, status, clipped)
        if storage == "compiled"
        else evaluate(values, status, clipped)
    )
    expected = jxp.FluidState(
        **metadata(policy),
        pressure=values,
        temperature=values + 300.0,
        conductivity=xp.full_like(values, np.nan),
        status=status,
        clipped=clipped,
    )
    assert actual._metadata() == expected._metadata()
    assert actual.shape == (3, 1)
    assert (actual._clipped is None) == (clipped is None)
    np.testing.assert_array_equal(actual.status, expected.status)
    np.testing.assert_array_equal(actual.clipped, expected.clipped)
    for name in actual.available_properties:
        np.testing.assert_array_equal(
            actual.raw_property(name), expected.raw_property(name)
        )
        np.testing.assert_array_equal(
            actual.property_valid(name), expected.property_valid(name)
        )
    with pytest.raises(jxp.UnsupportedPropertyError):
        actual.raw_property("Q")
    if policy == "raise":
        with pytest.raises(jxp.PropertyUnavailableError) as captured:
            _ = actual.p
        assert captured.value.indices == ((1, 0),)
    else:
        np.testing.assert_array_equal(actual["p"], actual.raw_property("pressure"))
    if storage == "numpy":
        for value in (*actual.to_dict().values(), actual.status, actual.clipped):
            assert isinstance(value, np.ndarray)
            assert not value.flags.writeable


def test_trusted_numpy_output_does_not_share_mutable_owners_or_freeze_inputs():
    values = np.array([1.0, 2.0])
    status = np.array([0, 4], dtype=np.int32)
    flags = np.array([True, True])
    current = native(values, status, flags, policy="clip")
    assert values.flags.writeable and status.flags.writeable and flags.flags.writeable
    values[:] = 99.0
    status[:] = 0
    flags[:] = False
    np.testing.assert_array_equal(current.p, [1.0, np.nan])
    np.testing.assert_array_equal(current.status, [0, 4])
    np.testing.assert_array_equal(current.clipped, [True, False])


@pytest.mark.parametrize("storage", ["numpy", "jax"])
def test_native_collections_indexing_and_roundtrips_preserve_public_data(storage):
    xp = np if storage == "numpy" else jnp
    first = native(xp.asarray([1.0, 2.0]), xp.asarray([0, 1], dtype=np.int32))
    second = native(xp.asarray([3.0, 4.0]), xp.asarray([0, 0], dtype=np.int32))
    combined = jxp.FluidState.stack([first, second]).flipped()
    np.testing.assert_array_equal(
        combined.raw_property("p"), [[3.0, 4.0], [1.0, np.nan]]
    )
    np.testing.assert_array_equal(combined.at_index(1).status, [0, 1])
    concatenated = first + second
    np.testing.assert_array_equal(
        concatenated.raw_property("p"), [1.0, np.nan, 3.0, 4.0]
    )
    for restored in (
        jxp.FluidState.from_data(combined.to_data()),
        pickle.loads(pickle.dumps(combined)),
    ):
        assert restored._metadata() == combined._metadata()
        assert restored._clipped is None
        np.testing.assert_array_equal(
            restored.raw_property("p"), combined.raw_property("p")
        )
        np.testing.assert_array_equal(restored.status, combined.status)


def test_trusted_native_jit_vectorization_and_gradients_preserve_failed_rows():
    def evaluate(x, codes):
        return native(x * x, codes, x < 0, policy="clip")

    values = jnp.asarray([-2.0, 3.0, 4.0])
    status = jnp.asarray([0, 4, 0], dtype=jnp.int32)
    for current in (
        jax.jit(evaluate)(values, status),
        jax.jit(jax.vmap(evaluate))(values, status),
    ):
        np.testing.assert_array_equal(current.p, [4.0, np.nan, 16.0])
        np.testing.assert_array_equal(current.clipped, [True, False, False])
        np.testing.assert_array_equal(current.valid, [True, False, True])
    derivative = jax.jit(
        jax.grad(lambda x: jnp.where(status == 0, evaluate(x, status).p, 0.0).sum())
    )(values)
    np.testing.assert_array_equal(derivative, [-4.0, 0.0, 8.0])


def test_native_assembly_avoids_public_schema_and_per_property_finite_checks(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Native assembly entered arbitrary-input validation")

    monkeypatch.setattr(jxp.FluidState, "_numeric_array", staticmethod(forbidden))
    values = jnp.asarray([1.0, 2.0])
    status = jnp.asarray([0, 1], dtype=jnp.int32)
    evaluate = lambda x, codes: native(x, codes)
    current = jax.jit(evaluate)(values, status)
    np.testing.assert_array_equal(current.raw_property("p"), [1.0, np.nan])
    program = str(jax.make_jaxpr(evaluate)(values, status))
    assert "is_finite" not in program
    optimized = jax.jit(evaluate).lower(values, status).compile().as_text()
    assert "callback" not in optimized.lower()


@pytest.mark.parametrize("policy", ["mask", "clip"])
def test_unmasked_factory_is_restricted_to_guarded_raise_entries(policy):
    with pytest.raises(ValueError, match="guarded raise"):
        jxp.FluidState._from_native(
            metadata(policy),
            {"pressure": jnp.asarray([1.0, 2.0])},
            jnp.asarray([0, 4], dtype=jnp.int32),
            mask_failed=False,
        )


@pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
def test_guarded_raise_assembly_keeps_internal_values_without_selection(xp):
    values = xp.asarray([1.0, 2.0])
    codes = xp.asarray([0, 4], dtype=np.int32)
    current = jxp.FluidState._from_native(
        metadata("raise"),
        {
            "pressure": values,
            "temperature": values + 300.0,
            "conductivity": xp.full_like(values, np.nan),
        },
        codes,
        mask_failed=False,
    )
    # This failed intermediate is private: the raising entry must inspect its
    # status before exposing a state. Public guarded access still rejects it.
    np.testing.assert_array_equal(current.raw_property("p"), [1.0, 2.0])
    np.testing.assert_array_equal(current.status, [0, 4])
    with pytest.raises(jxp.PropertyUnavailableError):
        _ = current.p
    if xp is np:
        values[:] = 7.0
        codes[:] = 0
        np.testing.assert_array_equal(current.raw_property("p"), [1.0, 2.0])
        np.testing.assert_array_equal(current.status, [0, 4])
        assert not current.raw_property("p").flags.writeable


def test_guarded_raise_assembly_traces_no_masks_or_finite_validation():
    def evaluate(values, status):
        return jxp.FluidState._from_native(
            metadata("raise"),
            {
                "pressure": values,
                "temperature": values + 300.0,
                "conductivity": jnp.full_like(values, np.nan),
            },
            status,
            mask_failed=False,
        )

    values = jnp.asarray([1.0, 2.0])
    status = jnp.asarray([0, 4], dtype=jnp.int32)
    program = str(jax.make_jaxpr(evaluate)(values, status))
    assert "is_finite" not in program
    assert "select_n" not in program
    assert "callback" not in program.lower()


def test_perfect_gas_raise_checks_status_before_returning_raw_native_state(
    monkeypatch, request
):
    observed = []
    original = jxp.FluidState._from_native.__func__

    def record(
        cls, metadata, properties, status, clipped=None, *, mask_failed=True, **options
    ):
        observed.append((metadata["error_policy"], mask_failed))
        return original(
            cls,
            metadata,
            properties,
            status,
            clipped,
            mask_failed=mask_failed,
            **options,
        )

    monkeypatch.setattr(jxp.FluidState, "_from_native", classmethod(record))
    gas = jxp.FluidPerfectGas(
        "CO2", T_ref=600.0, p_ref=1e5, identifier=request.node.nodeid
    )
    with pytest.raises(jxp.InvalidStateError) as captured:
        gas.get_state(jxp.PT_INPUTS, jnp.asarray([1e5, -1.0]), 500.0)
    assert captured.value.index == (1,)
    assert ("raise", False) in observed
    successful = gas.get_state(jxp.PT_INPUTS, jnp.asarray([1e5, 2e5]), 500.0)
    np.testing.assert_array_equal(successful.p, [1e5, 2e5])
    masked = gas.get_state(
        jxp.PT_INPUTS, jnp.asarray([1e5, -1.0]), 500.0, errors="mask"
    )
    assert ("mask", True) in observed
    np.testing.assert_array_equal(masked.raw_property("p"), [1e5, np.nan])


@pytest.mark.parametrize("shape", [(), (0, 3), (2, 1, 3)])
def test_native_exact_shapes_and_lazy_absent_clipping_are_preserved(shape):
    values = jnp.ones(shape)
    status = jnp.zeros(shape, dtype=jnp.int32)
    current = jax.jit(native)(values, status)
    assert current.shape == current.p.shape == current.clipped.shape == shape
    assert current._clipped is None
    assert len(jax.tree_util.tree_leaves(current)) == 4


@pytest.mark.parametrize("shape", [(64,), (2, 64), (0, 64), (1,)])
@pytest.mark.parametrize("outcome", ["success", "mixed", "failed"])
def test_native_bulk_masking_preserves_all_rows_and_clipping(shape, outcome):
    size = int(np.prod(shape))
    values = jnp.arange(size, dtype=jnp.float64).reshape(shape) + 1.0
    codes = np.zeros(size, dtype=np.int32)
    if outcome == "mixed":
        codes[::2] = jxp.StatusCode.PROVIDER_FAILURE
    elif outcome == "failed":
        codes[:] = jxp.StatusCode.INVALID_INPUT
    codes = jnp.asarray(codes.reshape(shape))
    flags = jnp.ones(shape, dtype=bool)
    actual = jax.jit(
        lambda x, status, clipped: native(x, status, clipped, policy="clip")
    )(values, codes, flags)
    expected_pressure = np.where(np.asarray(codes) == 0, np.asarray(values), np.nan)
    np.testing.assert_array_equal(actual.raw_property("p"), expected_pressure)
    np.testing.assert_array_equal(actual.raw_property("T"), expected_pressure + 300.0)
    np.testing.assert_array_equal(actual.status, codes)
    np.testing.assert_array_equal(actual.clipped, np.asarray(codes) == 0)
    assert actual.shape == shape
    assert np.all(np.isnan(actual.raw_property("conductivity")))
    exported = actual.to_dict(include_aliases=True)
    np.testing.assert_array_equal(exported["p"], expected_pressure)


def test_native_bulk_vmap_and_complete_export_gradient_match_scalar_rows():
    values = jnp.linspace(-2.0, 3.0, 128).reshape(2, 64)
    codes = jnp.zeros((2, 64), dtype=jnp.int32).at[1, ::3].set(4)

    def evaluate(x, status):
        return native(x * x, status, x < 0, policy="clip")

    whole = jax.jit(evaluate)(values, codes)
    mapped = jax.jit(jax.vmap(evaluate))(values, codes)
    scalar = jax.jit(jax.vmap(jax.vmap(evaluate)))(values, codes)
    for current in (mapped, scalar):
        for name in whole.available_properties:
            np.testing.assert_array_equal(
                current.raw_property(name), whole.raw_property(name)
            )
        np.testing.assert_array_equal(current.status, whole.status)
        np.testing.assert_array_equal(current.clipped, whole.clipped)

    def exported_sum(x):
        result = jax.vmap(evaluate)(x, codes)
        return sum(jnp.nan_to_num(value).sum() for value in result.to_dict().values())

    gradient = jax.jit(jax.grad(exported_sum))(values)
    np.testing.assert_allclose(
        gradient, jnp.where(codes == 0, 4 * values, 0.0), rtol=1e-13
    )


def test_native_bulk_masking_keeps_nan_masks_as_actual_numerical_leaves():
    values = jnp.arange(64.0)
    codes = jnp.zeros(64, dtype=jnp.int32).at[3].set(4)
    current = jax.jit(native)(values, codes)
    assert not hasattr(current, "_mask_on_read")
    for leaf in current.to_dict().values():
        assert np.isnan(leaf[3])
    # Masking has happened before the state is exposed, including generic
    # numerical PyTree operations rather than only property getter calls.
    for leaf in jax.tree_util.tree_leaves(current):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert np.isnan(leaf[3])
