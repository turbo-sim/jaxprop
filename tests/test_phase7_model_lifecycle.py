"""Materialized perfect-gas models retain physical and source identities."""

import CoolProp.CoolProp as CP
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import pytest

import jaxprop as jxp


@pytest.fixture
def gas():
    return jxp.Fluid(
        "CO2", method="perfect_gas", model_options={"T_ref": 600.0, "p_ref": 1e5}
    )


def test_materializing_unchanged_model_keeps_identity_without_source_work(
    gas, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Materialization must not initialize source states")

    monkeypatch.setattr(CP, "AbstractState", forbidden)
    result = gas.materialize()
    assert result is not gas
    assert result.reference_id == gas.reference_id
    assert result.configuration == gas.configuration
    assert result.references is gas.references
    assert result.constants is gas.constants
    assert isinstance(result, jxp.Fluid)
    expected = gas.get_state(jxp.PT_INPUTS, 2e5, 650.0)
    actual = result.get_state(jxp.PT_INPUTS, 2e5, 650.0)
    for name in expected.available_properties:
        np.testing.assert_array_equal(
            actual.raw_property(name), expected.raw_property(name)
        )


def test_materialize_refreshes_dynamic_view_identity_and_configuration(gas):
    gamma = jnp.asarray(1.35, dtype=jnp.float32)
    view = eqx.tree_at(lambda model: model.constants.gamma, gas, gamma)
    assert view.reference_id == gas.reference_id
    actual = view.materialize()
    direct = gas.with_constants(gamma=gamma)
    assert actual.reference_id == direct.reference_id != gas.reference_id
    assert actual.constants.gamma.dtype == jnp.float32
    assert actual.constants.gamma is view.constants.gamma
    assert actual.references is gas.references
    assert actual.configuration.model_options["constants"]["gamma"] == float(gamma)
    expected = actual.get_state(jxp.PT_INPUTS, 1e5, 700.0)
    np.testing.assert_allclose(expected.gamma, float(gamma))
    np.testing.assert_allclose(expected.cp - expected.cv, actual.constants.R, rtol=1e-6)
    assert gas.configuration.model_options["constants"]["gamma"] != float(gamma)


def test_reference_offset_update_preserves_thermodynamic_differences(gas):
    enthalpy_shift, entropy_shift = -2e6, 1000.0
    changed = gas.with_constants(
        h_ref=gas.constants.h_ref + enthalpy_shift,
        s_ref=gas.constants.s_ref + entropy_shift,
    )
    initial = gas.get_state(jxp.PT_INPUTS, 3e5, 700.0)
    actual = changed.get_state(jxp.PT_INPUTS, 3e5, 700.0)
    np.testing.assert_allclose(actual.h - initial.h, enthalpy_shift)
    np.testing.assert_allclose(actual.u - initial.u, enthalpy_shift)
    np.testing.assert_allclose(actual.s - initial.s, entropy_shift)
    np.testing.assert_allclose(actual.h - actual.u, actual.p / actual.rho, rtol=1e-12)
    assert changed.reference_id != gas.reference_id
    assert changed.references is gas.references
    with pytest.raises(ValueError, match="physical model or reference_id"):
        jxp.FluidState.stack([initial, actual])


@pytest.mark.parametrize(
    "changes, expected_request",
    [
        ({"unknown": 1.0}, "unknown"),
        ({"gamma": 1.0}, "gamma"),
        ({"gamma": float("inf")}, "gamma"),
        ({"gamma": True}, "gamma"),
        ({"gamma": [1.4]}, "gamma"),
        ({"gamma": "invalid"}, "gamma"),
        ({"R": 0.0}, "R"),
        ({"T_ref": -1.0}, "T_ref"),
        ({"P_ref": float("nan")}, "P_ref"),
        ({"h_ref": float("nan")}, "h_ref"),
        ({"s_ref": complex(1, 2)}, "s_ref"),
        ({"mu_ref": -1.0}, "mu_ref"),
        ({"S_k": float("inf")}, "k_ref"),
    ],
)
def test_materialization_rejects_invalid_model_constants(
    gas, changes, expected_request
):
    with pytest.raises(jxp.InvalidConfigurationError) as caught:
        gas.with_constants(**changes)
    assert caught.value.provider == "native"
    assert caught.value.model == "perfect_gas"
    assert caught.value.request == expected_request


def test_materialization_requires_eager_concrete_constants(gas):
    with pytest.raises(jxp.InvalidConfigurationError) as caught:
        jax.jit(lambda gamma: gas.with_constants(gamma=gamma).constants.gamma)(1.4)
    assert caught.value.request == "gamma"
    assert "eager" in caught.value.reason


def test_unavailable_transport_uses_stable_model_identity(gas):
    constants = dict(mu_ref=jnp.nan, S_mu=jnp.nan, k_ref=jnp.nan, S_k=jnp.nan)
    first, second = gas.with_constants(**constants), gas.with_constants(**constants)
    assert first.reference_id == second.reference_id
    assert first.configuration == second.configuration
    assert first.configuration.model_options["constants"]["mu_ref"] is None
    state = first.get_state(jxp.PT_INPUTS, 1e5, 600.0)
    assert state.status == 0
    assert np.isfinite(state.h)
    assert not state.property_valid("mu")
    assert not state.property_valid("k")
    restored = jxp.FluidState.from_data(state.to_data())
    assert restored.reference_id == state.reference_id


def test_materialized_model_pickle_round_trip_preserves_parameter_identity(gas):
    model = gas.with_constants(gamma=1.4, h_ref=-1000.0)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.reference_id == model.reference_id != gas.reference_id
    assert restored.configuration == model.configuration
    assert restored.references == model.references
    expected = model.get_state(jxp.PT_INPUTS, 2e5, 650.0)
    actual = restored.get_state(jxp.PT_INPUTS, 2e5, 650.0)
    for name in expected.available_properties:
        np.testing.assert_array_equal(
            actual.raw_property(name), expected.raw_property(name)
        )
