"""Small JAX interface regressions for empty batches and input-pair reuse."""

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp
from jaxprop.perfect_gas import request_policy as pg_requests

from fluid_contract_cases import CONTRACT_CASES, build_current_fluid

JAX_CASES = tuple(
    case
    for case in CONTRACT_CASES
    if case.id in {"coolprop_jax", "bicubic", "perfect_gas"}
)
NATIVE_CASES = tuple(case for case in JAX_CASES if case.id != "coolprop_jax")


@pytest.fixture(scope="module")
def table_directory(tmp_path_factory):
    return tmp_path_factory.mktemp("phase4_jax_edges")


@pytest.fixture
def fluid(case, table_directory, request):
    # A unique static identity prevents earlier tests from hiding first traces.
    return build_current_fluid(case, table_directory, identifier=request.node.nodeid)


def assert_metadata(state, fluid):
    assert state.fluid_name == fluid.name
    assert state.identifier == fluid.identifier
    assert state.provider == fluid.provider
    assert state.providers == frozenset({fluid.provider})
    assert state.model == fluid.model
    assert state.reference_id == fluid.reference_id
    diagnostics = {
        "subcooling",
        "superheating",
        "pressure_saturation",
        "temperature_saturation",
        "supersaturation_ratio",
        "supersaturation_degree",
    }
    assert state.available_properties == fluid.capabilities.properties - diagnostics
    assert state.error_policy == "mask"


@pytest.mark.parametrize("case", JAX_CASES, ids=lambda case: case.id)
def test_empty_broadcast_batches_preserve_full_state_eager_and_jitted(case, fluid):
    enthalpy = jnp.empty((0, 1))
    pressure = jnp.asarray([[7e6, 8e6, 9e6]])
    evaluate = lambda h, p: fluid.get_state(jxp.HmassP_INPUTS, h, p, errors="mask")
    eager = evaluate(enthalpy, pressure)
    compiled = jax.jit(evaluate)(enthalpy, pressure)
    jax.block_until_ready((eager, compiled))

    for state in (eager, compiled):
        assert_metadata(state, fluid)
        assert state.shape == (0, 3)
        assert state.status.shape == state.valid.shape == (0, 3)
        assert np.issubdtype(state.status.dtype, np.integer)
        assert state.valid.dtype == np.dtype(bool)
        for name in state.available_properties:
            value = state.raw_property(name)
            mask = state.property_valid(name)
            assert value.shape == mask.shape == (0, 3), name
            assert mask.dtype == np.dtype(bool), name
            np.testing.assert_array_equal(value, compiled.raw_property(name))
            np.testing.assert_array_equal(mask, compiled.property_valid(name))
        np.testing.assert_array_equal(state.status, compiled.status)
        np.testing.assert_array_equal(state.valid, compiled.valid)


@pytest.mark.parametrize("case", NATIVE_CASES, ids=lambda case: case.id)
def test_alternating_input_pairs_reuses_each_compilation(case, fluid, monkeypatch):
    traces = []
    if case.id == "perfect_gas":
        original = pg_requests.resolve_pair

        def counted(pair, *args, **kwargs):
            traces.append(pair)
            return original(pair, *args, **kwargs)

        monkeypatch.setattr(pg_requests, "resolve_pair", counted)
    else:
        original = type(fluid)._get_state_scalar

        def counted(model, pair, first, second, **kwargs):
            traces.append(pair)
            return original(model, pair, first, second, **kwargs)

        monkeypatch.setattr(type(fluid), "_get_state_scalar", counted)

    temperatures = []
    policy = "mask"
    for enthalpy, pressure in ((680e3, 7e6), (720e3, 9e6)):
        from_hp = fluid.get_state(jxp.HmassP_INPUTS, enthalpy, pressure, errors=policy)
        temperature = float(from_hp.T)
        from_pt = fluid.get_state(jxp.PT_INPUTS, pressure, temperature, errors=policy)
        jax.block_until_ready((from_hp, from_pt))
        temperatures.append(temperature)

        np.testing.assert_array_equal(from_hp.h, enthalpy)
        np.testing.assert_array_equal(from_hp.p, pressure)
        for state in (from_hp, from_pt):
            assert_metadata(state, fluid)
            assert state.shape == ()
        # Match the existing input-pair conformance tolerance for the bicubic
        # inverse solve, rather than adding a new interpolation accuracy target.
        for name in from_hp.available_properties:
            np.testing.assert_allclose(
                from_pt.raw_property(name), from_hp.raw_property(name), rtol=1e-8
            )
            np.testing.assert_array_equal(
                from_pt.property_valid(name), from_hp.property_valid(name)
            )
        np.testing.assert_array_equal(from_pt.status, from_hp.status)

    assert not np.isclose(*temperatures)
    assert Counter(traces) == {jxp.HmassP_INPUTS: 1, jxp.PT_INPUTS: 1}
