"""Complete bicubic batch semantics and AD across native batch assembly."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.bicubic.request_policy import evaluate_scalar


@pytest.fixture(scope="module")
def fluid(tmp_path_factory):
    return jp.Fluid(
        "CO2",
        method="bicubic",
        table_options=dict(
            h_min=600e3,
            h_max=850e3,
            p_min=5e6,
            p_max=12e6,
            N_h=8,
            N_p=8,
            N_p_sat=4,
            coarse_step=1,
            table_dir=str(tmp_path_factory.mktemp("phase14_bicubic")),
        ),
    )


def scalar_reference(fluid, enthalpy, pressure, policy):
    """Retain the scalar policy/interpolation route as a batch-shape oracle."""
    enthalpy, pressure = jnp.broadcast_arrays(enthalpy, pressure)
    result = jax.vmap(
        lambda h, p: evaluate_scalar(fluid, jp.HmassP_INPUTS, h, p, errors=policy)
    )(enthalpy.ravel(), pressure.ravel())
    return jax.tree_util.tree_map(lambda value: value.reshape(enthalpy.shape), result)


def assert_complete_state_matches(actual, expected):
    assert actual.shape == expected.shape
    assert actual.reference_id == expected.reference_id
    assert actual.available_properties == expected.available_properties
    np.testing.assert_array_equal(actual.status, expected.status)
    np.testing.assert_array_equal(actual.clipped, expected.clipped)
    for name in actual.available_properties:
        value, reference = actual.raw_property(name), expected.raw_property(name)
        assert value.dtype == reference.dtype
        np.testing.assert_array_equal(jnp.isnan(value), jnp.isnan(reference))
        np.testing.assert_allclose(value, reference, rtol=3e-14, atol=0, equal_nan=True)


@pytest.mark.parametrize("size", [1, 16, 63, 64, 65, 128, 1024])
@pytest.mark.parametrize("policy", ["mask", "clip"])
def test_complete_batches_preserve_scalar_policy_at_factory_boundaries(
    fluid, size, policy
):
    h = jnp.linspace(650e3, 800e3, size)
    p = jnp.linspace(6e6, 11e6, size)
    if size >= 16:
        h = h.at[1].set(jnp.nan).at[2].set(590e3).at[3].set(860e3)
        p = p.at[4].set(-1).at[5].set(jnp.inf)
    actual = jax.jit(
        lambda h, p: fluid.get_state(jp.HmassP_INPUTS, h, p, errors=policy)
    )(h, p)
    reference = jax.jit(lambda h, p: scalar_reference(fluid, h, p, policy))(h, p)
    assert_complete_state_matches(actual, reference)


@pytest.mark.parametrize("shape", [(8, 16), (16, 8), (1, 16), (0, 3)])
def test_independent_axis_broadcast_preserves_every_field(fluid, shape):
    h = jnp.linspace(650e3, 800e3, shape[0])[:, None]
    p = jnp.linspace(6e6, 11e6, shape[1])[None, :]
    actual = fluid.get_state(jp.HmassP_INPUTS, h, p, errors="mask")
    expected = jax.jit(lambda h, p: scalar_reference(fluid, h, p, "mask"))(h, p)
    assert_complete_state_matches(actual, expected)


@pytest.mark.parametrize("policy", ["mask", "clip"])
def test_broadcast_input_and_dynamic_coefficient_gradients_match_scalar_route(
    fluid, policy
):
    h = jnp.asarray([[680e3], [jnp.nan], [590e3], [750e3]])
    p = jnp.asarray([[7e6, 9e6]])

    def total(scale, enthalpy, *, scalar):
        coefficients = fluid._table["temperature"]["coeffs"] * scale
        view = eqx.tree_at(
            lambda model: model._table["temperature"]["coeffs"], fluid, coefficients
        )
        state = (
            scalar_reference(view, enthalpy, p, policy)
            if scalar
            else view.get_state(jp.HmassP_INPUTS, enthalpy, p, errors=policy)
        )
        return jnp.nan_to_num(state.raw_property("temperature")).sum()

    native = jax.jit(jax.value_and_grad(lambda s, h: total(s, h, scalar=False), (0, 1)))
    scalar = jax.jit(jax.value_and_grad(lambda s, h: total(s, h, scalar=True), (0, 1)))
    for scale in (1.0, 1.001):
        actual, expected = native(scale, h), scalar(scale, h)
        for value, reference in zip(
            jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)
        ):
            assert np.all(np.isfinite(value))
            np.testing.assert_allclose(value, reference, rtol=2e-13, atol=1e-12)
        assert actual[1][1][1, 0] == 0
        assert actual[1][1][2, 0] == 0


@pytest.mark.parametrize("density", [-1.0, np.inf])
def test_dynamic_core_failure_still_masks_complete_batch(fluid, density):
    coefficients = jnp.zeros_like(fluid._table["density"]["coeffs"])
    coefficients = coefficients.at[..., 0].set(density)
    view = eqx.tree_at(
        lambda model: model._table["density"]["coeffs"], fluid, coefficients
    )
    h, p = jnp.linspace(650e3, 750e3, 128), jnp.asarray(8e6)
    actual = view.get_state(jp.HmassP_INPUTS, h, p, errors="mask")
    expected = jax.jit(lambda h, p: scalar_reference(view, h, p, "mask"))(h, p)
    assert_complete_state_matches(actual, expected)
    np.testing.assert_array_equal(actual.status, jp.StatusCode.PROVIDER_FAILURE)
    for name in actual.available_properties:
        assert np.all(np.isnan(actual.raw_property(name)))


@pytest.mark.parametrize("policy", ["mask", "clip"])
def test_mixed_batch_input_and_model_jvp_matches_scalar_and_finite_difference(
    fluid, policy
):
    h = jnp.asarray([[680e3], [jnp.nan], [590e3], [750e3]])
    p = jnp.asarray([[7e6, 9e6]])
    h_direction = jnp.asarray([[3.0], [0.0], [2.0], [5.0]])
    scale_direction = jnp.asarray(0.01)

    def values(scale, enthalpy, *, scalar):
        view = eqx.tree_at(
            lambda model: model._table["temperature"]["coeffs"],
            fluid,
            fluid._table["temperature"]["coeffs"] * scale,
        )
        state = (
            scalar_reference(view, enthalpy, p, policy)
            if scalar
            else view.get_state(jp.HmassP_INPUTS, enthalpy, p, errors=policy)
        )
        # Optional unavailability and failed states represent zero contributions
        # to this differentiable full-property consumer.
        return {
            name: jnp.nan_to_num(state.raw_property(name))
            for name in state.available_properties
        }

    primals = (jnp.asarray(1.0), h)
    directions = (scale_direction, h_direction)
    native_values, native_tangents = jax.jit(
        lambda scale, h: jax.jvp(
            lambda s, enthalpy: values(s, enthalpy, scalar=False),
            (scale, h),
            directions,
        )
    )(*primals)
    scalar_values, scalar_tangents = jax.jit(
        lambda scale, h: jax.jvp(
            lambda s, enthalpy: values(s, enthalpy, scalar=True),
            (scale, h),
            directions,
        )
    )(*primals)
    step = 0.01
    plus = values(1.0 + step * scale_direction, h + step * h_direction, scalar=False)
    minus = values(1.0 - step * scale_direction, h - step * h_direction, scalar=False)
    for name, tangent in native_tangents.items():
        assert np.all(np.isfinite(tangent))
        np.testing.assert_allclose(
            native_values[name], scalar_values[name], rtol=3e-14, atol=0
        )
        np.testing.assert_allclose(
            tangent, scalar_tangents[name], rtol=2e-12, atol=1e-12
        )
        np.testing.assert_array_equal(tangent[1], 0)
        if policy == "mask":
            np.testing.assert_array_equal(tangent[2], 0)
        difference = (plus[name] - minus[name]) / (2 * step)
        np.testing.assert_allclose(
            tangent[jnp.asarray([0, 3])],
            difference[jnp.asarray([0, 3])],
            rtol=3e-6,
            atol=1e-9,
        )


def test_strided_numpy_single_precision_inputs_with_scalar_pressure(fluid):
    source = np.linspace(650e3, 800e3, 32, dtype=np.float32).reshape(4, 8)
    h = source[:, ::2]
    p = np.float32(8e6)
    assert not h.flags.c_contiguous
    actual = fluid.get_state(jp.HmassP_INPUTS, h, p, errors="mask")
    dtype = jnp.result_type(h, p, fluid.h_vals, fluid.logP_vals, 0.0)
    expected = jax.jit(lambda h, p: scalar_reference(fluid, h, p, "mask"))(
        jnp.asarray(h, dtype=dtype), jnp.asarray(p, dtype=dtype)
    )
    assert_complete_state_matches(actual, expected)
    assert actual.raw_property("temperature").dtype == dtype
    np.testing.assert_array_equal(actual.status, 0)
    np.testing.assert_array_equal(actual.raw_property("enthalpy"), h)
    np.testing.assert_array_equal(actual.raw_property("pressure"), np.full(h.shape, p))


def test_optional_cell_unavailability_preserves_successful_complete_batch(fluid):
    masks = dict(fluid._coefficient_validity)
    masks["viscosity"] = (
        jnp.ones((fluid.N_h - 1, fluid.N_p - 1), dtype=bool).at[2, 3].set(False)
    )
    view = eqx.tree_at(lambda model: model._coefficient_validity, fluid, masks)
    h, p = jnp.linspace(650e3, 800e3, 128), jnp.asarray(8e6)
    actual = view.get_state(jp.HmassP_INPUTS, h, p, errors="mask")
    expected = jax.jit(lambda h, p: scalar_reference(view, h, p, "mask"))(h, p)
    assert_complete_state_matches(actual, expected)
    np.testing.assert_array_equal(actual.status, jp.StatusCode.SUCCESS)
    unavailable = np.isnan(actual.raw_property("viscosity"))
    assert np.any(unavailable) and not np.all(unavailable)
    np.testing.assert_array_equal(actual.property_valid("viscosity"), ~unavailable)
    baseline = fluid.get_state(jp.HmassP_INPUTS, h, p, errors="mask")
    for name in actual.available_properties - {"viscosity"}:
        np.testing.assert_array_equal(
            actual.raw_property(name), baseline.raw_property(name)
        )


def test_eager_broadcast_raise_reports_first_failed_index_and_scalar_error(fluid):
    # Different failure types must retain row-major first-failure selection,
    # rather than choosing the globally highest-priority status in the batch.
    for first, later, error_type in (
        (590e3, np.nan, jp.OutOfDomainError),
        (np.nan, 590e3, jp.InvalidStateError),
    ):
        h = np.full((8, 16), 700e3)
        h[1, 7], h[4, 2] = first, later
        with pytest.raises(error_type) as batch_error:
            fluid.get_state(jp.HmassP_INPUTS, h, 8e6, errors="raise")
        assert batch_error.value.index == (1, 7)
        assert batch_error.value.request == "HmassP_INPUTS"
        assert batch_error.value.provider == "bicubic"
        with pytest.raises(error_type) as scalar_error:
            fluid.get_state(jp.HmassP_INPUTS, first, 8e6, errors="raise")
        assert scalar_error.value.index == ()
        assert type(batch_error.value) is type(scalar_error.value)
