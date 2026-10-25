"""Compatibility of state collection operations used by thermodynamic cycles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxprop import FluidState


def test_scalar_states_combine_reverse_and_extract_without_mutating_inputs():
    """Component endpoints retain their ordering, aliases and stream metadata."""
    inlet = FluidState(fluid_name="CO2", identifier="cycle", pressure=1e7, enthalpy=2e5)
    outlet = FluidState(
        fluid_name="CO2", identifier="cycle", pressure=2e7, enthalpy=3e5
    )
    states = inlet + outlet
    np.testing.assert_allclose(states.p, [1e7, 2e7])
    np.testing.assert_allclose(states.flipped().h, [3e5, 2e5])
    assert states.at_index(-1).h == outlet.h
    assert states.flipped().at_index(0).p == outlet.p
    assert states.fluid_name == "CO2" and states.identifier == "cycle"
    assert states.at_index(0).identifier == "cycle"
    assert inlet.p == 1e7 and outlet.p == 2e7
    assert inlet.at_index(0).h == inlet.h


def test_batched_states_concatenate_along_the_first_axis():
    """Combining an existing path with an endpoint extends its property arrays."""
    path = FluidState(fluid_name="CO2", pressure=jnp.array([1e7, 2e7]), enthalpy=2e5)
    endpoint = FluidState(fluid_name="CO2", pressure=3e7, enthalpy=3e5)
    combined = path + endpoint
    np.testing.assert_allclose(combined.p, [1e7, 2e7, 3e7])
    assert path.flipped().h == path.h  # Scalar properties remain scalar.
    np.testing.assert_allclose(path.p, [1e7, 2e7])


@pytest.mark.parametrize("metadata", [{"fluid_name": "Water"}, {"identifier": "other"}])
def test_combining_incompatible_state_metadata_is_rejected(metadata):
    """A component path cannot silently mix fluid identities or stream labels."""
    state = FluidState(fluid_name="CO2", identifier="cycle")
    other = FluidState(**({"fluid_name": "CO2", "identifier": "cycle"} | metadata))
    with pytest.raises(ValueError, match="different fluid_name or identifier"):
        state + other


def test_state_collection_operations_preserve_jax_derivatives():
    """Restored collection helpers remain usable with traced property values."""

    def outlet_enthalpy(value):
        inlet = FluidState(fluid_name="CO2", enthalpy=value)
        outlet = FluidState(fluid_name="CO2", enthalpy=3 * value)
        return (inlet + outlet).flipped().at_index(0).h

    assert jax.grad(outlet_enthalpy)(2.0) == 3.0
