"""Derivative values have explicit units, local axes and primal provenance."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp
from jaxprop.property_derivatives import PropertyDerivatives, get_props_gradient


@pytest.fixture(scope="module")
def fluid():
    return jxp.Fluid(
        "CO2",
        method="perfect_gas",
        identifier="derivative-stream",
        model_options={"T_ref": 600.0, "p_ref": 1e5},
    )


def primal(**kwargs):
    return jxp.FluidState(
        fluid_name="Air",
        provider="native",
        model="perfect_gas",
        reference_id="reference",
        error_policy="clip",
        **kwargs,
    )


def test_result_has_distinct_quantity_kind_units_and_primal_axes():
    base = primal(pressure=[1.0, 2.0], temperature=[300.0, 400.0])
    result = PropertyDerivatives(
        base,
        {"p": [[1.0, 0.0], [1.0, 0.0]], "T": [[0.0, 1.0], [0.0, 1.0]]},
        ("p", "T"),
    )
    assert not isinstance(result, jxp.FluidState)
    assert result.quantity_kind == "property_derivatives"
    assert result.primal is base
    assert result.shape == result.primal_shape == (2,)
    assert result.derivative_shape == (2, 2)
    assert result.input_names == ("pressure", "temperature")
    assert result.input_units == ("Pa", "K")
    assert result.property_units["temperature"] == "K"
    assert result.derivative_units("T") == ("(K)/(Pa)", "(K)/(K)")
    np.testing.assert_array_equal(result.T, result["temperature"])
    np.testing.assert_array_equal(result.to_dict(include_aliases=True)["p"], result.p)
    assert "status" not in result.to_dict()
    with pytest.raises(AttributeError):
        result.input_names = ("density", "enthalpy")


def test_derivatives_preserve_primal_status_and_clipping_without_false_validity():
    base = primal(
        pressure=[1.0, 2.0, 3.0],
        temperature=[300.0, np.nan, 400.0],
        clipped=[True, False, True],
        status=[0, 0, 1],
    )
    result = PropertyDerivatives(
        base,
        {"p": [[1.0, 0.0]] * 3, "T": [[0.0, 1.0]] * 3},
        ("p", "T"),
    )
    np.testing.assert_array_equal(result.status, [0, 0, 1])
    np.testing.assert_array_equal(result.valid, [True, True, False])
    np.testing.assert_array_equal(result.clipped, [True, False, False])
    np.testing.assert_array_equal(
        result.property_valid("T"), [[True, True], [False, False], [False, False]]
    )
    np.testing.assert_array_equal(result.T[0], [0.0, 1.0])
    assert np.isnan(result.T[1:]).all()
    assert result.error_policy == "clip"
    assert result.reference_id == base.reference_id


def test_derivatives_cannot_be_combined_with_ordinary_states():
    base = primal(pressure=1.0)
    result = PropertyDerivatives(base, {"p": [1.0, 0.0]}, ("p", "T"))
    with pytest.raises(TypeError):
        base + result
    with pytest.raises(TypeError):
        result + base
    with pytest.raises(TypeError, match="FluidState"):
        jxp.FluidState.stack([base, result])


@pytest.mark.parametrize(
    "derivatives,names,error_type,request_name",
    [
        ({"p": [1.0, 0.0]}, ("p",), jxp.InvalidConfigurationError, "input_names"),
        (
            {"p": [1.0, 0.0]},
            ("p", "typo"),
            jxp.InvalidConfigurationError,
            "input_names",
        ),
        ({"p": [1.0, 0.0]}, None, jxp.InvalidConfigurationError, "input_names"),
        (
            {"p": [1.0, 0.0], "pressure": [1.0, 0.0]},
            ("p", "T"),
            jxp.InvalidConfigurationError,
            "pressure",
        ),
        ({"p": [1, 0]}, ("p", "T"), jxp.InvalidStateError, "pressure"),
        ({"p": [1.0]}, ("p", "T"), jxp.InvalidStateError, "shape"),
        ({"T": [0.0, 1.0]}, ("p", "T"), jxp.UnsupportedPropertyError, "T"),
    ],
)
def test_derivative_schema_rejects_malformed_metadata_and_values(
    derivatives, names, error_type, request_name
):
    with pytest.raises(error_type) as captured:
        PropertyDerivatives(primal(pressure=1.0), derivatives, names)
    assert captured.value.request == request_name


@pytest.mark.parametrize("method", ["auto", "jax", "fd", "finite_difference"])
def test_pt_partials_match_independent_ideal_gas_identities(fluid, method):
    pressure, temperature = 2e5, 700.0
    result = get_props_gradient(fluid, jxp.PT_INPUTS, pressure, temperature, method)
    R, gamma = float(fluid.constants.R), float(fluid.constants.gamma)
    cp = gamma * R / (gamma - 1)
    cv = R / (gamma - 1)
    expected = {
        "pressure": [1.0, 0.0],
        "temperature": [0.0, 1.0],
        "density": [1 / (R * temperature), -pressure / (R * temperature**2)],
        "enthalpy": [0.0, cp],
        "internal_energy": [0.0, cv],
        "entropy": [-R / pressure, cp / temperature],
        "speed_of_sound": [0.0, 0.5 * (gamma * R / temperature) ** 0.5],
        "isothermal_bulk_modulus": [1.0, 0.0],
        "isentropic_bulk_modulus": [gamma, 0.0],
        "joule_thomson": [0.0, 0.0],
    }
    for name, partials in expected.items():
        np.testing.assert_allclose(result[name], partials, rtol=1e-5, atol=1e-9)
    assert result.primal.identifier == fluid.identifier
    assert result.status.shape == result.primal.shape == ()
    assert result.reference_id == fluid.reference_id
    assert result.property_units["enthalpy"] == "J/kg"
    assert result.method == (
        "autodiff" if method in ("auto", "jax") else "finite_difference"
    )


@pytest.mark.parametrize("method", ["auto", "finite_difference"])
def test_broadcast_gradients_use_local_partials_not_cross_element_jacobians(
    fluid, method
):
    pressure = jnp.asarray([[1e5], [2e5]])
    temperature = jnp.asarray([[500.0, 600.0, 700.0]])
    result = get_props_gradient(fluid, jxp.PT_INPUTS, pressure, temperature, method)
    R = float(fluid.constants.R)
    assert result.shape == (2, 3)
    assert result.derivative_shape == (2, 3, 2)
    assert result.status.shape == (2, 3)
    for values in result.to_dict().values():
        assert values.shape == (2, 3, 2)
    np.testing.assert_allclose(
        result.rho[..., 0], np.broadcast_to(1 / (R * temperature), (2, 3)), rtol=1e-5
    )
    np.testing.assert_allclose(
        result.rho[..., 1], -pressure / (R * temperature**2), rtol=1e-5
    )


@pytest.mark.parametrize("method", ["auto", "finite_difference"])
def test_jit_and_vmap_preserve_partial_axis_metadata_and_values(fluid, method):
    evaluate = lambda p, t: get_props_gradient(
        fluid, jxp.PT_INPUTS, p, t, method=method
    )
    p, t = jnp.asarray([1e5, 2e5]), jnp.asarray([500.0, 700.0])
    compiled = jax.jit(evaluate)(p, t)
    vectorized = jax.jit(jax.vmap(evaluate))(p, t)
    assert compiled.shape == vectorized.shape == (2,)
    assert compiled.derivative_shape == vectorized.derivative_shape == (2, 2)
    assert compiled.input_names == vectorized.input_names == ("pressure", "temperature")
    for name, values in compiled.items():
        np.testing.assert_allclose(values, vectorized[name], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ["auto", "finite_difference"])
def test_clip_derivatives_are_zero_outside_bounds_and_keep_primal_flags(fluid, method):
    result = get_props_gradient(
        fluid, jxp.PT_INPUTS, 1e5, -100.0, method=method, errors="clip"
    )
    assert result.primal.clipped
    assert result.status == 0
    np.testing.assert_array_equal(result.T, [0.0, 0.0])
    np.testing.assert_allclose(result.h, [0.0, 0.0], atol=1e-9)
    assert result.valid


def test_invalid_primal_rows_do_not_gain_formal_zero_derivatives(fluid):
    result = get_props_gradient(fluid, jxp.PT_INPUTS, jnp.asarray([1e5, -1.0]), 500.0)
    np.testing.assert_array_equal(result.valid, [True, False])
    for name in result.available_properties:
        assert np.isnan(result[name][1]).all(), name
    assert result.property_valid("T")[0].all()


def test_finite_difference_boundary_failure_keeps_primal_and_axis_validity_separate(
    fluid,
):
    lower_temperature = float(fluid.capabilities.domain.bounds["temperature"][0])
    result = get_props_gradient(
        fluid,
        jxp.PT_INPUTS,
        1e5,
        lower_temperature + 1e-10,
        method="finite_difference",
    )
    assert result.valid
    assert result.status == 0
    np.testing.assert_array_equal(result.property_valid("T"), [True, False])
    assert result.T[0] == 0.0
    assert np.isnan(result.T[1])


def test_derivative_quantity_pickle_roundtrip_preserves_units_and_primal(fluid):
    original = get_props_gradient(fluid, jxp.PT_INPUTS, 1e5, 600.0)
    restored = pickle.loads(pickle.dumps(original))
    assert restored.quantity_kind == original.quantity_kind
    assert restored.input_names == original.input_names
    assert restored.property_units == original.property_units
    assert restored.reference_id == original.reference_id
    np.testing.assert_array_equal(restored.T, original.T)
    np.testing.assert_array_equal(restored.status, original.status)


@pytest.mark.parametrize("method", ["typo", None, True, []])
def test_unknown_differentiation_method_is_rejected_before_evaluation(fluid, method):
    with pytest.raises(jxp.UnsupportedOptionError) as captured:
        get_props_gradient(fluid, jxp.PT_INPUTS, 1e5, 500.0, method=method)
    assert captured.value.request == "method"


@pytest.mark.parametrize(
    "options,request_name",
    [
        ({"eps_rel": -1.0}, "eps_rel"),
        ({"eps_abs": np.inf}, "eps_abs"),
        ({"eps_abs": True}, "eps_abs"),
        ({"eps_rel": 0.0, "eps_abs": 0.0}, "eps_rel/eps_abs"),
    ],
)
def test_invalid_finite_difference_options_are_not_silently_ignored(
    fluid, options, request_name
):
    with pytest.raises(jxp.UnsupportedOptionError) as captured:
        get_props_gradient(fluid, jxp.PT_INPUTS, 1e5, 500.0, **options)
    assert captured.value.request == request_name


def test_helper_rejects_transformed_raising_policy_explicitly(fluid):
    with pytest.raises(jxp.UnsupportedExecutionModeError) as captured:
        get_props_gradient(fluid, jxp.PT_INPUTS, 1e5, 500.0, errors="raise")
    assert captured.value.request == "errors=raise"
