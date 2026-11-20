"""Immutable metadata equality preserves mapping semantics and JIT cache keys."""

from collections import OrderedDict
from copy import copy, deepcopy
from functools import partial
import pickle
from types import MappingProxyType

import jax
import numpy as np
import pytest

from jaxprop import FluidConfiguration, FrozenMapping


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"name": "CO2", "grid": (100, 200)},
        {"nested": {"bounds": {"pressure": [1e5, 3e7]}}, "flags": {"jit", "eager"}},
        {1: "one", "two": (2, 3), (3, 4): None},
    ],
)
def test_equivalent_frozen_mappings_preserve_order_independent_hash_and_equality(
    values,
):
    first = FrozenMapping(values)
    second = FrozenMapping(dict(values))
    reordered = FrozenMapping(OrderedDict(reversed(list(values.items()))))
    restored = pickle.loads(pickle.dumps(first))
    for other in (second, reordered, restored, copy(first), deepcopy(first)):
        assert first == other and other == first
        assert not (first != other)
        assert hash(first) == hash(other)


def test_aligned_equal_metadata_needs_no_mapping_field_materialization(monkeypatch):
    first = FrozenMapping({"nested": {"bounds": (1.0, 2.0)}, "name": "CO2"})
    second = FrozenMapping({"nested": {"bounds": (1.0, 2.0)}, "name": "CO2"})

    def unexpected(*args):
        raise AssertionError(
            "equal immutable metadata must not build dicts through field lookup"
        )

    monkeypatch.setattr(FrozenMapping, "__getitem__", unexpected)
    assert first == second
    assert first == first


@pytest.mark.parametrize(
    "other",
    [
        {"a": (1, 2), "b": 3},
        OrderedDict([("b", 3), ("a", (1, 2))]),
        MappingProxyType({"b": 3.0, "a": (1.0, 2.0)}),
    ],
)
def test_other_mapping_types_keep_symmetric_public_equality(other):
    frozen = FrozenMapping({"a": [1, 2], "b": 3})
    assert frozen == other and other == frozen
    assert frozen != dict(other, b=4)


@pytest.mark.parametrize("other", [None, 1, "metadata", [1, 2], (1, 2)])
def test_non_mappings_compare_unequal_without_raising(other):
    frozen = FrozenMapping({"a": 1})
    assert frozen != other and other != frozen


def test_colliding_hashes_do_not_hide_changed_metadata():
    class CollidingInt(int):
        def __hash__(self):
            return 123

    first = FrozenMapping({"scale": CollidingInt(2)})
    second = FrozenMapping({"scale": CollidingInt(3)})
    assert hash(first) == hash(second)
    assert first != second


def test_nan_identity_and_distinct_nan_semantics_are_preserved():
    value = float("nan")
    first = FrozenMapping({"value": value})
    assert first == FrozenMapping({"value": value})
    assert first != FrozenMapping({"value": float("nan")})


def test_frozen_mapping_subclass_keeps_its_public_lookup_semantics():
    class Doubled(FrozenMapping):
        def __getitem__(self, key):
            return 2 * super().__getitem__(key)

    subclass = Doubled({"scale": 1})
    ordinary = FrozenMapping({"scale": 1})
    assert subclass != ordinary and ordinary != subclass
    assert subclass == {"scale": 2}


def test_jit_reuses_equivalent_metadata_and_retraces_a_real_setting_change():
    traces = []

    @partial(jax.jit, static_argnames="configuration")
    def evaluate(value, configuration):
        traces.append(configuration)
        return value * configuration.model_options["scale"]

    def config(options):
        return FluidConfiguration(
            "Air", "perfect_gas", "perfect_gas", "HEOS", model_options=options
        )

    first = config({"scale": 2.0, "nested": {"label": "gas", "values": [1, 2]}})
    equal = config({"nested": {"values": (1, 2), "label": "gas"}, "scale": 2.0})
    changed = config({"scale": 3.0, "nested": {"label": "gas", "values": [1, 2]}})
    assert first == equal and hash(first) == hash(equal)
    np.testing.assert_array_equal(np.asarray(evaluate(4.0, first)), 8.0)
    np.testing.assert_array_equal(np.asarray(evaluate(5.0, equal)), 10.0)
    assert len(traces) == 1
    np.testing.assert_array_equal(np.asarray(evaluate(4.0, changed)), 12.0)
    assert len(traces) == 2
