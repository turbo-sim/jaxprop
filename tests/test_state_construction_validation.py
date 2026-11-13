"""Construction edge cases retained while reducing repeated assembly work."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp


def state(**properties):
    return jxp.FluidState(
        fluid_name="CO2",
        identifier="working_fluid",
        provider="native",
        model="real_fluid",
        reference_id="source-reference",
        **properties,
    )


@pytest.mark.parametrize("array", [np.asarray, jnp.asarray], ids=["numpy", "jax"])
def test_missing_enabled_values_remain_unavailable_with_mixed_broadcast_inputs(array):
    current = state(
        pressure=array([[1e5], [2e5]]),
        enthalpy=array([1.0, np.inf, 3.0]),
        available_properties={"pressure", "enthalpy", "temperature"},
        property_validity={
            "pressure": array([[False, True, True]]),
            "temperature": array(True),
        },
        status=array([[0], [2]]),
    )

    assert current.shape == (2, 3)
    np.testing.assert_array_equal(current.status, [[0, 0, 0], [2, 2, 2]])
    np.testing.assert_array_equal(
        current.raw_property("pressure"),
        [[np.nan, 1e5, 1e5], [np.nan, np.nan, np.nan]],
    )
    np.testing.assert_array_equal(
        current.raw_property("enthalpy"),
        [[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]],
    )
    assert current.supports_property("T")
    assert np.all(np.isnan(current.raw_property("T")))
    assert not np.any(current.property_valid("T"))
    with pytest.raises(jxp.PropertyUnavailableError) as captured:
        current.T
    assert captured.value.request == "temperature"
    assert set(captured.value.indices) == {
        (row, column) for row in range(2) for column in range(3)
    }
    with pytest.raises(jxp.UnsupportedPropertyError):
        current.Q


@pytest.mark.parametrize("array", [np.asarray, jnp.asarray], ids=["numpy", "jax"])
def test_empty_validity_mask_can_supply_batch_shape_for_scalar_and_missing_values(
    array,
):
    current = state(
        pressure=1e5,
        available_properties={"pressure", "temperature"},
        property_validity={"pressure": array(np.empty((0, 3), dtype=bool))},
    )

    assert current.shape == (0, 3)
    assert current.p.shape == current.T.shape == (0, 3)
    for name in ("pressure", "temperature"):
        assert current.property_valid(name).shape == (0, 3)
    assert current.valid.shape == (0, 3)


def test_readonly_strided_inputs_do_not_allow_mutable_owners_to_change_numpy_state():
    pressure_owner = np.array([1e5, 9e5, 2e5, 9e5])
    status_owner = np.array([0, 4, 0, 4])
    mask_owner = np.array([True, False, True, False])
    pressure = pressure_owner[::2]
    status = status_owner[::2]
    mask = mask_owner[::2]
    for values in (pressure, status, mask):
        values.setflags(write=False)
    current = state(
        pressure=pressure,
        status=status,
        property_validity={"pressure": mask},
    )

    pressure_owner[:] = -1.0
    status_owner[:] = 4
    mask_owner[:] = False
    np.testing.assert_array_equal(current.p, [1e5, 2e5])
    np.testing.assert_array_equal(current.status, [0, 0])
    np.testing.assert_array_equal(current.property_valid("p"), [True, True])
    for values in (current.p, current.status, current.property_valid("p")):
        assert isinstance(values, np.ndarray)
        assert not values.flags.writeable


def test_finite_zero_dimensional_numpy_inputs_remain_isolated_from_later_mutation():
    pressure = np.array(1e5)
    status = np.array(0)
    mask = np.array(True)
    current = state(
        pressure=pressure,
        status=status,
        property_validity={"pressure": mask},
    )
    pressure[...] = 2e5
    status[...] = 4
    mask[...] = False

    assert current.shape == ()
    assert current.p == 1e5
    assert current.valid
    assert current.property_valid("p")
    assert not current.p.flags.writeable


@pytest.mark.parametrize(
    "status",
    [-1, 5, np.array([0, -1]), jnp.array([[0, 5]])],
    ids=["negative-scalar", "large-scalar", "negative-numpy-batch", "large-jax-batch"],
)
def test_eager_status_range_errors_retain_structured_context(status):
    with pytest.raises(jxp.InvalidStateError) as captured:
        state(pressure=1e5, status=status, error_policy="mask")
    error = captured.value
    assert error.request == "status"
    assert error.fluid_name == "CO2"
    assert error.provider == "native"
    assert error.model == "real_fluid"
    assert "0 through 4" in error.reason


@pytest.mark.parametrize("status", [0.0, np.array([0.0, 1.5])])
def test_floating_status_codes_are_rejected_even_when_values_are_integral(status):
    with pytest.raises(jxp.InvalidStateError) as captured:
        state(pressure=1e5, status=status)
    assert captured.value.request == "status"
    assert "integer" in captured.value.reason


def test_jitted_masked_projection_gradient_excludes_explicit_and_status_failures():
    def evaluate(values):
        return state(
            pressure=values**2,
            status=jnp.array([0, 0, 3]),
            available_properties={"pressure", "temperature"},
            property_validity={"pressure": jnp.array([True, False, True])},
            error_policy="mask",
        )

    values = jnp.array([2.0, 3.0, 4.0])
    current = jax.jit(evaluate)(values)
    np.testing.assert_array_equal(current.p, [4.0, np.nan, np.nan])
    np.testing.assert_array_equal(current.property_valid("p"), [True, False, False])
    assert np.all(np.isnan(current.T))
    assert current.identifier == "working_fluid"
    gradient = jax.jit(jax.grad(lambda x: evaluate(x).p[0]))(values)
    np.testing.assert_array_equal(gradient, [4.0, 0.0, 0.0])
