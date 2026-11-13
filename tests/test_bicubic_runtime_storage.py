"""Keep cache provenance out of repeated numerical PyTree traversal."""

from copy import deepcopy
from pathlib import Path
import pickle

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp
from jaxprop.bicubic.table_cache import (
    normalize_generated_arrays,
    table_validation_error,
)


@pytest.fixture(scope="module")
def runtime_fluid(tmp_path_factory):
    return jxp.FluidBicubic(
        fluid_name="CO2",
        backend="HEOS",
        h_min=600e3,
        h_max=850e3,
        p_min=5e6,
        p_max=12e6,
        N_h=4,
        N_p=4,
        N_p_sat=4,
        coarse_step=1,
        table_dir=str(tmp_path_factory.mktemp("runtime_table")),
    )


def test_generation_manifest_is_not_a_dynamic_runtime_tree(runtime_fluid):
    fluid = runtime_fluid
    paths_and_leaves = jax.tree_util.tree_flatten_with_path(fluid)[0]
    assert paths_and_leaves
    assert all(isinstance(value, jax.Array) for _, value in paths_and_leaves)
    assert all("metadata" not in str(path) for path, _ in paths_and_leaves)
    assert "metadata" not in fluid._table

    # Adding construction-only provenance must not add warmed-call leaves.
    expanded = deepcopy(fluid.metadata)
    expanded["extra_provenance"] = {str(i): f"source {i}" for i in range(200)}
    clone = deepcopy(fluid)
    object.__setattr__(clone, "metadata", expanded)
    assert len(jax.tree_util.tree_leaves(clone)) == len(paths_and_leaves)
    np.testing.assert_array_equal(
        fluid.get_state(jxp.HmassP_INPUTS, 700e3, 8e6).T,
        clone.get_state(jxp.HmassP_INPUTS, 700e3, 8e6).T,
    )


def test_public_and_persisted_payload_retain_complete_manifest(runtime_fluid):
    fluid = runtime_fluid
    with (Path(fluid.table_dir) / f"{fluid.table_name}.pkl").open("rb") as stream:
        persisted = pickle.load(stream)
    public = fluid.table
    assert public["metadata"] == persisted["metadata"] == fluid.metadata
    assert set(public) == set(persisted)
    assert table_validation_error(public, fluid.metadata) is None
    assert table_validation_error(persisted, fluid.metadata) is None
    for name in jxp.PROPERTIES_CANONICAL:
        for key, value in persisted[name].items():
            np.testing.assert_array_equal(public[name][key], value)


def test_full_state_jit_and_derivative_preserve_values(runtime_fluid):
    fluid = runtime_fluid
    h, p = 700e3, 8e6
    eager = fluid.get_state(jxp.HmassP_INPUTS, h, p)
    compiled = eqx.filter_jit(
        lambda model, enthalpy, pressure: model.get_state(
            jxp.HmassP_INPUTS, enthalpy, pressure, errors="mask"
        )
    )(fluid, jnp.asarray(h), jnp.asarray(p))
    for name in eager.available_properties:
        np.testing.assert_array_equal(
            compiled.raw_property(name), eager.raw_property(name)
        )
        np.testing.assert_array_equal(
            compiled.property_valid(name), eager.property_valid(name)
        )
    derivative = jax.jit(
        jax.grad(
            lambda enthalpy: fluid.get_state(
                jxp.HmassP_INPUTS, enthalpy, p, errors="mask"
            ).T
        )
    )(h)
    step = 1.0
    difference = (
        fluid.get_state(jxp.HmassP_INPUTS, h + step, p).T
        - fluid.get_state(jxp.HmassP_INPUTS, h - step, p).T
    ) / (2 * step)
    np.testing.assert_allclose(derivative, difference, rtol=1e-7, atol=1e-11)


def test_coefficients_remain_dynamic_inputs_to_compiled_evaluation(runtime_fluid):
    traces = []

    @eqx.filter_jit
    def temperature(model, enthalpy, pressure):
        traces.append(True)
        return model.get_state(jxp.HmassP_INPUTS, enthalpy, pressure, errors="mask").T

    h, p = jnp.asarray(700e3), jnp.asarray(8e6)
    original = temperature(runtime_fluid, h, p)
    coefficients = runtime_fluid._table["temperature"]["coeffs"]
    shifted = eqx.tree_at(
        lambda model: model._table["temperature"]["coeffs"],
        runtime_fluid,
        coefficients.at[..., 0].add(10.0),
    )
    updated = temperature(shifted, h, p)
    np.testing.assert_allclose(updated - original, 10.0, rtol=0, atol=1e-12)
    assert len(traces) == 1


def test_normalization_reuses_matching_jax_storage(monkeypatch):
    array = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    table = {
        "temperature": {"value": array},
        "saturation_props": {"temperature": {"value": array, "grad_p": array}},
    }

    def unexpected_conversion(*args, **kwargs):
        raise AssertionError("matching JAX storage must not be reconverted")

    monkeypatch.setattr(jnp, "asarray", unexpected_conversion)
    normalized = normalize_generated_arrays(table, {"numeric_dtype": "float64"})
    assert normalized["temperature"]["value"] is array
    assert normalized["saturation_props"]["temperature"]["value"] is array


def test_normalization_still_converts_numpy_and_wrong_dtype():
    table = {
        "temperature": {
            "value": np.asarray([1.0, 2.0], dtype=np.float32),
            "grad_h": jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        },
        "saturation_props": {},
    }
    normalized = normalize_generated_arrays(table, {"numeric_dtype": "float64"})
    for value in normalized["temperature"].values():
        assert isinstance(value, jax.Array)
        assert value.dtype == np.float64
    np.testing.assert_array_equal(normalized["temperature"]["value"], [1.0, 2.0])
