"""Availability survives compact state returns and JAX control flow."""

import pickle

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp


def state(**properties):
    return jxp.FluidState(
        fluid_name="CO2",
        identifier="stream",
        provider="native",
        model="real_fluid",
        reference_id=("same", "reference"),
        **({"error_policy": "mask"} | properties),
    )


@pytest.mark.parametrize("choose_constant", [False, True])
def test_conditional_constant_and_traced_states_have_compatible_structure(
    choose_constant,
):
    @jax.jit
    def evaluate(values, constant):
        # Construct this branch from NumPy values inside the trace. Its storage
        # implementation must have the same PyTree structure as traced states.
        fixed = state(
            pressure=np.array([2.0, 4.0, 6.0]),
            status=np.array([0, 0, 4]),
            property_validity={"pressure": np.array([True, False, True])},
        )
        return jax.lax.cond(
            constant,
            lambda _: fixed,
            lambda _: state(pressure=values, status=jnp.array([0, 1, 0])),
            operand=None,
        )

    current = evaluate(jnp.array([1.0, 3.0, 5.0]), choose_constant)
    expected = [2.0, np.nan, np.nan] if choose_constant else [1.0, np.nan, 5.0]
    expected_status = [0, 0, 4] if choose_constant else [0, 1, 0]
    np.testing.assert_array_equal(current.raw_property("p"), expected)
    np.testing.assert_array_equal(current.status, expected_status)
    np.testing.assert_array_equal(current.property_valid("p"), np.isfinite(expected))


def test_scan_accepts_numpy_initial_state_and_returns_traced_state_trajectory():
    initial = state(pressure=np.array([1.0, 2.0]), status=np.zeros(2, dtype=np.int32))

    def advance(previous, increment):
        current = state(
            pressure=previous.raw_property("p") + increment,
            status=jnp.zeros(2, dtype=jnp.int32),
        )
        return current, current

    final, trajectory = jax.jit(
        lambda increments: jax.lax.scan(advance, initial, increments)
    )(jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(final.raw_property("p"), [7.0, 8.0])
    np.testing.assert_array_equal(
        trajectory.raw_property("p"), [[2.0, 3.0], [4.0, 5.0], [7.0, 8.0]]
    )
    np.testing.assert_array_equal(trajectory.property_valid("p"), np.ones((3, 2)))
    assert trajectory.identifier == final.identifier == "stream"


def test_traced_status_does_not_trigger_host_validation_of_captured_numpy_values():
    fixed = state(pressure=np.array([1.0, 2.0]), error_policy="raise")

    @jax.jit
    def evaluate(status):
        updated = eqx.tree_at(lambda current: current.status, fixed, status)
        return updated.p, updated.property_valid("p")

    pressure, valid = evaluate(jnp.array([0, 0]))
    np.testing.assert_array_equal(pressure, [1.0, 2.0])
    np.testing.assert_array_equal(valid, [True, True])


@pytest.mark.parametrize("storage", ["numpy", "jax", "compiled"])
def test_validity_preserves_explicit_masks_nonfinite_values_and_failed_status(storage):
    xp = np if storage == "numpy" else jnp

    def evaluate(values):
        return state(
            pressure=values,
            status=xp.array([0, 4, 0, 0, 0]),
            property_validity={"p": xp.array([False, True, True, True, True])},
            available_properties={"pressure", "temperature"},
        )

    values = xp.array([1.0, 2.0, np.inf, np.nan, 5.0])
    current = jax.jit(evaluate)(values) if storage == "compiled" else evaluate(values)
    expected = [False, False, False, False, True]
    np.testing.assert_array_equal(current.property_valid("p"), expected)
    np.testing.assert_array_equal(
        current.raw_property("p"), [np.nan, np.nan, np.nan, np.nan, 5.0]
    )
    np.testing.assert_array_equal(current.property_valid("T"), np.zeros(5, dtype=bool))
    assert current.property_valid("p").dtype == np.bool_
    if storage == "numpy":
        assert isinstance(current.property_valid("p"), np.ndarray)
        assert not current.property_valid("p").flags.writeable
    else:
        assert isinstance(current.property_valid("p"), jax.Array)

    # Serialization must not recover a finite value that was explicitly masked.
    exported = current.to_data()
    np.testing.assert_array_equal(exported["property_validity"]["pressure"], expected)
    for restored in (
        jxp.FluidState.from_data(exported),
        pickle.loads(pickle.dumps(current)),
    ):
        np.testing.assert_array_equal(restored.property_valid("p"), expected)
        np.testing.assert_array_equal(restored.status, [0, 4, 0, 0, 0])
        np.testing.assert_array_equal(
            restored.raw_property("p"), current.raw_property("p")
        )
        assert restored.available_properties == current.available_properties


@pytest.mark.parametrize("storage", ["numpy", "jax"])
def test_eager_availability_errors_keep_failing_indices_and_unsupported_errors(storage):
    xp = np if storage == "numpy" else jnp
    current = state(
        pressure=xp.array([1.0, 2.0, 3.0]),
        status=xp.array([0, 0, 4]),
        property_validity={"p": xp.array([True, False, True])},
        error_policy="raise",
    )
    for read in (
        lambda: current.pressure,
        lambda: current.p,
        lambda: current["p"],
    ):
        with pytest.raises(jxp.PropertyUnavailableError) as captured:
            read()
        assert captured.value.request == "pressure"
        assert set(captured.value.indices) == {(1,), (2,)}
    for read in (
        lambda: current.Q,
        lambda: current.property_valid("Q"),
    ):
        with pytest.raises(jxp.UnsupportedPropertyError):
            read()


def test_projected_gradient_and_validity_agree_after_compiled_collections():
    def objective(values):
        first = state(
            pressure=values,
            status=jnp.array([0, 0, 1, 0]),
            property_validity={"p": jnp.array([True, True, True, False])},
        )
        combined = jxp.FluidState.stack([first, first]).flipped()
        raw = combined.raw_property("p")
        return jnp.where(combined.property_valid("p"), raw, 0.0).sum()

    value, gradient = jax.jit(jax.value_and_grad(objective))(
        jnp.array([1.0, np.inf, 3.0, 4.0])
    )
    assert value == 2.0
    np.testing.assert_array_equal(gradient, [2.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("shape", [(), (0, 3)])
def test_numpy_derived_validity_is_readonly_at_scalar_and_empty_shapes(shape):
    current = state(pressure=np.ones(shape))
    valid = current.property_valid("p")
    assert isinstance(valid, np.ndarray)
    assert valid.shape == shape
    assert valid.dtype == np.bool_
    assert not valid.flags.writeable
    np.testing.assert_array_equal(valid, np.ones(shape, dtype=bool))
