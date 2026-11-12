"""Fixed public state expectations for the uniform fluid interface.

Strict expected failures identify confirmed phase-3 gaps. Remove each marker
when its intended behavior is implemented; an unexpected pass fails the suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxprop import FluidState

# Keep this public vocabulary independent of PROPERTY_ALIASES and dispatch code.
PUBLIC_ALIASES = (
    ("pressure", "p"),
    ("pressure", "P"),
    ("temperature", "T"),
    ("density", "rho"),
    ("enthalpy", "h"),
    ("entropy", "s"),
    ("internal_energy", "u"),
    ("isobaric_heat_capacity", "cp"),
    ("isochoric_heat_capacity", "cv"),
    ("heat_capacity_ratio", "gamma"),
    ("speed_of_sound", "a"),
    ("viscosity", "mu"),
    ("conductivity", "k"),
    pytest.param(
        "isobaric_expansion_coefficient",
        "alpha_p",
        marks=pytest.mark.xfail(
            strict=True,
            raises=KeyError,
            reason="phase3: duplicate alias definition removes alpha_p",
        ),
    ),
)


@pytest.mark.parametrize("canonical,alias", PUBLIC_ALIASES)
def test_public_aliases_match_canonical_attribute_and_mapping_access(canonical, alias):
    values = jnp.array([[1.25], [2.5]])
    state = FluidState(**{canonical: values})

    np.testing.assert_array_equal(state[canonical], values)
    np.testing.assert_array_equal(state[alias], values)
    np.testing.assert_array_equal(getattr(state, alias), values)


def _state(values, *, fluid_name="CO2", identifier="working_fluid"):
    """Use explicit arrays so these tests exercise state operations alone."""
    values = jnp.asarray(values, dtype=float)
    return FluidState(
        fluid_name=fluid_name,
        identifier=identifier,
        pressure=values,
        enthalpy=2.0 * values,
    )


def _stack_case(shape, count):
    if shape == () and count == 2:
        return pytest.param(shape, count, id=f"scalar-{count}")
    return pytest.param(
        shape,
        count,
        id=f"shape{shape}-{count}",
        marks=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason="phase3: stack squeezes the added axis or existing singleton axes",
        ),
    )


@pytest.mark.parametrize(
    "shape,count",
    [
        _stack_case(shape, count)
        for shape in ((), (1,), (1, 1), (2, 1), (1, 2, 1))
        for count in (1, 2)
    ],
)
def test_stack_adds_one_leading_axis_and_preserves_every_input_axis(shape, count):
    states = [_state(jnp.full(shape, index + 1.0)) for index in range(count)]
    result = FluidState.stack(states)

    assert np.shape(result.p) == (count,) + shape
    assert np.shape(result.h) == (count,) + shape
    np.testing.assert_array_equal(
        result.p, np.stack([np.asarray(state.p) for state in states])
    )
    assert result.fluid_name == "CO2"
    assert result.identifier == "working_fluid"


def test_stack_preserves_values_metadata_and_does_not_mutate_inputs():
    first = _state([1.0, 2.0])
    second = _state([3.0, 4.0])
    result = FluidState.stack([first, second])

    np.testing.assert_array_equal(result.p, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(result.h, [[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_array_equal(first.p, [1.0, 2.0])
    np.testing.assert_array_equal(second.p, [3.0, 4.0])
    assert result.fluid_name == first.fluid_name == second.fluid_name == "CO2"
    assert result.identifier == first.identifier == second.identifier == "working_fluid"


def test_index_and_reverse_preserve_trailing_singleton_axes_and_identity():
    original = _state([[[1.0], [2.0]], [[3.0], [4.0]]])
    reversed_state = original.flipped()
    selected = reversed_state.at_index(0)

    assert np.shape(reversed_state.p) == (2, 2, 1)
    assert np.shape(selected.p) == (2, 1)
    np.testing.assert_array_equal(selected.p, [[3.0], [4.0]])
    np.testing.assert_array_equal(original.at_index(-1).p, selected.p)
    np.testing.assert_array_equal(original.p, [[[1.0], [2.0]], [[3.0], [4.0]]])
    for state in (original, reversed_state, selected):
        assert state.fluid_name == "CO2"
        assert state.identifier == "working_fluid"


def test_concatenation_extends_first_axis_without_losing_trailing_axes():
    first = _state([[1.0], [2.0]])
    second = _state([[3.0]])
    result = first + second

    assert np.shape(result.p) == (3, 1)
    np.testing.assert_array_equal(result.p, [[1.0], [2.0], [3.0]])
    np.testing.assert_array_equal(result.h, [[2.0], [4.0], [6.0]])
    np.testing.assert_array_equal(first.p, [[1.0], [2.0]])
    np.testing.assert_array_equal(second.p, [[3.0]])
    assert result.fluid_name == "CO2"
    assert result.identifier == "working_fluid"


def test_empty_stack_raises_informative_error():
    with pytest.raises(ValueError, match="[Nn]o states|[Ee]mpty"):
        FluidState.stack([])


@pytest.mark.parametrize("operation", ["stack", "concatenate"])
@pytest.mark.parametrize(
    "metadata", [{"fluid_name": "Water"}, {"identifier": "other_stream"}]
)
def test_collection_operations_reject_fluid_or_stream_identity_mismatch(
    operation, metadata
):
    first = _state([1.0])
    second = _state([2.0], **metadata)

    with pytest.raises(ValueError, match="fluid_name|identifier"):
        if operation == "stack":
            FluidState.stack([first, second])
        else:
            first + second


def test_dictionary_contains_numeric_data_and_optional_aliases_without_identity():
    state = _state([[1.0], [2.0]])
    canonical = state.to_dict()
    aliased = state.to_dict(include_aliases=True)

    assert "fluid_name" not in canonical
    assert "identifier" not in canonical
    assert "p" not in canonical
    assert "h" not in canonical
    assert "pressure" in canonical
    np.testing.assert_array_equal(aliased["p"], canonical["pressure"])
    np.testing.assert_array_equal(aliased["h"], canonical["enthalpy"])
    for value in canonical.values():
        dtype = np.asarray(value).dtype
        assert np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)


def test_dictionary_round_trip_preserves_arrays_and_explicit_identity():
    original = _state([[1.0], [2.0]])
    restored = FluidState(
        fluid_name=original.fluid_name,
        identifier=original.identifier,
        **original.to_dict(),
    )

    assert restored.fluid_name == original.fluid_name
    assert restored.identifier == original.identifier
    for name, value in original.to_dict().items():
        np.testing.assert_array_equal(restored[name], value)


def test_state_dictionary_and_collection_operations_preserve_jax_transformations():
    def evaluate(value):
        state = _state(jnp.reshape(value, (2, 1)))
        combined = state + _state(jnp.reshape(3 * value, (2, 1)))
        return combined.flipped().at_index(0).to_dict()["enthalpy"][0]

    values = jnp.array([1.0, 2.0])
    assert jax.jit(evaluate)(values) == 12.0
    np.testing.assert_array_equal(jax.grad(evaluate)(values), [0.0, 6.0])
