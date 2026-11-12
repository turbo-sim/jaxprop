"""Evaluator wiring for the shared state identity and derivative schema."""

import numpy as np
import pytest

import jaxprop as jxp
from jaxprop.perfect_gas import get_props_gradient

from fluid_contract_cases import CONTRACT_CASES, build_current_fluid

IDENTIFIERS = (None, "", "working_fluid")


@pytest.fixture(scope="module")
def configured_fluids(tmp_path_factory):
    """Reuse one generated table across the three bicubic identity variants."""
    table_dir = tmp_path_factory.mktemp("evaluator_state_metadata")
    return {
        (case.id, identifier): build_current_fluid(case, table_dir, identifier)
        for case in CONTRACT_CASES
        for identifier in IDENTIFIERS
    }


def _state(fluid, case_id):
    enthalpy = 500e3 if case_id == "binary_mixture" else 700e3
    return fluid.get_state(jxp.HmassP_INPUTS, enthalpy, 8e6)


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize(
    "identifier", IDENTIFIERS, ids=("default", "empty", "explicit")
)
def test_every_evaluator_preserves_identifier_and_model_metadata(
    case, identifier, configured_fluids
):
    fluid = configured_fluids[(case.id, identifier)]
    state = _state(fluid, case.id)

    assert fluid.identifier == identifier
    assert state.identifier == identifier
    assert state.fluid_name == case.name
    assert state.provider == case.provider
    assert state.providers == frozenset({case.provider})
    assert state.model == case.model
    assert state.reference_id == fluid.reference_id
    hash(state.reference_id)
    assert state.shape == ()


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.id)
def test_state_collection_operations_preserve_evaluator_identity(
    case, configured_fluids
):
    state = _state(configured_fluids[(case.id, "working_fluid")], case.id)
    combined = state + state
    stacked = jxp.FluidState.stack([state])
    states = (combined, stacked, combined.flipped(), combined.at_index(-1))

    for result in states:
        assert result.fluid_name == state.fluid_name
        assert result.identifier == state.identifier
        assert result.provider == state.provider
        assert result.providers == state.providers
        assert result.model == state.model
        assert result.reference_id == state.reference_id
        assert result.available_properties == state.available_properties
        assert np.all(result.valid)
    assert combined.shape == (2,)
    assert stacked.shape == (1,)
    assert combined.at_index(-1).shape == ()
    assert state.shape == ()
    np.testing.assert_allclose(combined.p, [state.p, state.p])


def test_reference_identity_is_provider_independent_and_model_specific(
    configured_fluids,
):
    source_states = [
        _state(configured_fluids[(case_id, "working_fluid")], case_id)
        for case_id in ("coolprop", "coolprop_jax", "bicubic")
    ]
    assert len({state.reference_id for state in source_states}) == 1
    assert len({state.provider for state in source_states}) == 3

    mixed = jxp.FluidState.stack(source_states)
    assert mixed.provider == "mixed"
    assert mixed.providers == frozenset({"coolprop", "coolprop_jax", "bicubic"})
    assert mixed.reference_id == source_states[0].reference_id
    assert mixed.identifier == "working_fluid"
    np.testing.assert_allclose(mixed.p, [8e6, 8e6, 8e6])

    gas = _state(configured_fluids[("perfect_gas", "working_fluid")], "perfect_gas")
    assert gas.reference_id != source_states[0].reference_id
    with pytest.raises(ValueError, match="model|reference"):
        gas + source_states[0]


@pytest.mark.parametrize("method", ["auto", "finite_difference"])
def test_perfect_gas_gradient_helper_differentiates_numeric_properties_only(
    method, configured_fluids
):
    fluid = configured_fluids[("perfect_gas", "working_fluid")]
    pressure, temperature = 1e5, 600.0
    primal = fluid.get_state(jxp.PT_INPUTS, pressure, temperature)
    derivative = get_props_gradient(
        fluid, jxp.PT_INPUTS, pressure, temperature, method=method
    )

    assert derivative.fluid_name == primal.fluid_name
    assert derivative.identifier == primal.identifier
    assert derivative.provider == primal.provider
    assert derivative.model == primal.model
    assert derivative.reference_id == primal.reference_id
    assert derivative.available_properties == primal.available_properties
    assert derivative.error_policy == "mask"
    assert derivative.shape == (2,)
    np.testing.assert_array_equal(derivative.status, [0, 0])
    assert np.issubdtype(np.asarray(derivative.status).dtype, np.integer)
    for value in derivative.to_dict().values():
        assert np.shape(value) == (2,)
        assert np.issubdtype(np.asarray(value).dtype, np.floating)

    np.testing.assert_allclose(derivative.p, [1.0, 0.0], rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(derivative.T, [0.0, 1.0], rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(derivative.h, [0.0, primal.cp], rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(
        derivative.rho,
        [primal.rho / pressure, -primal.rho / temperature],
        rtol=1e-5,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        derivative.s,
        [-fluid.constants.R / pressure, primal.cp / temperature],
        rtol=1e-5,
        atol=1e-9,
    )
