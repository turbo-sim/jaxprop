"""Lossless configured snapshots, source identity and worker/state isolation.

Configuration metadata is an inspection record rather than a factory recipe.
Pickle preserves the complete configured evaluator, including materialized
native parameters/tables; source reconstruction additionally verifies the
original reference, native definitions/files and backend configuration.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import pickle
import subprocess
import sys

import CoolProp.CoolProp as CP
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.bicubic import cache_metadata
from jaxprop.coolprop import source_lifecycle

BACKENDS = os.environ.get("JAXPROP_TEST_BACKENDS", "HEOS").split(",")


def assert_same_state(actual, expected):
    assert actual.available_properties == expected.available_properties
    assert actual.reference_id == expected.reference_id
    assert actual.provider == expected.provider
    assert actual.model == expected.model
    assert actual.identifier == expected.identifier
    np.testing.assert_array_equal(actual.status, expected.status)
    for name in expected.available_properties:
        np.testing.assert_allclose(
            actual.raw_property(name),
            expected.raw_property(name),
            rtol=2e-12,
            atol=1e-9,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            actual.property_valid(name), expected.property_valid(name)
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ("coolprop", "coolprop_jax"))
def test_source_snapshot_rebuilds_both_native_handles_and_preserves_full_states(
    backend, method
):
    original = jp.Fluid("CO2", backend=backend, method=method, identifier="stream-a")
    initial = original.get_state(jp.PT_INPUTS, [6e6, 8e6], 500.0)
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is type(original)
    assert restored.configuration == original.configuration
    assert restored.capabilities == original.capabilities
    assert restored.references == original.references
    old_source = original.fluid if method == "coolprop_jax" else original
    new_source = restored.fluid if method == "coolprop_jax" else restored
    assert new_source._AS is new_source.abstract_state
    assert new_source._AS is not old_source._AS
    assert (
        new_source._source_context.auxiliary_state
        is not old_source._source_context.auxiliary_state
    )
    assert new_source._source_context.auxiliary_state is not new_source._AS
    assert new_source._state_lock is not old_source._state_lock
    for pair, first, second in (
        (jp.PT_INPUTS, [6e6, 8e6], 500.0),
        (jp.HmassP_INPUTS, initial.h, initial.p),
        (jp.PQ_INPUTS, [1e6, 2e6], [0.0, 1.0]),
    ):
        assert_same_state(
            restored.get_state(pair, first, second),
            original.get_state(pair, first, second),
        )


@pytest.mark.parametrize("method", ("coolprop", "coolprop_jax"))
def test_restored_source_uses_prepared_helpers_without_runtime_identity_probes(
    method, monkeypatch
):
    fluid = jp.Fluid("CO2", method=method)
    expected = fluid.get_state(jp.PT_INPUTS, 1e6, 350.0, supersaturation=True)
    restored = pickle.loads(pickle.dumps(fluid))

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Repeated evaluation must reuse native handles and identities"
        )

    monkeypatch.setattr(CP, "AbstractState", forbidden)
    monkeypatch.setattr(source_lifecycle, "source_serialization_identity", forbidden)
    for _ in range(3):
        actual = restored.get_state(jp.PT_INPUTS, 1e6, 350.0, supersaturation=True)
        assert_same_state(actual, expected)
    source = restored.fluid if method == "coolprop_jax" else restored
    homogeneous = source.get_state_metastable(
        "rho", float(expected.rho), "T", float(expected.T), supersaturation=False
    )
    np.testing.assert_allclose(homogeneous.p, expected.p, rtol=2e-12)
    if method == "coolprop_jax":
        compiled = jax.jit(
            lambda pressure: restored.get_state(
                jp.PT_INPUTS, pressure, 350.0, errors="mask"
            )
        )
        actual = compiled(jnp.asarray(1e6))
        np.testing.assert_allclose(actual.h, expected.h, rtol=2e-12)
        assert int(actual.status) == 0
        derivative = jax.jit(jax.grad(lambda pressure: compiled(pressure).h))(1e6)
        assert np.isfinite(derivative)


@pytest.mark.parametrize("backend", BACKENDS)
def test_restore_rejects_changed_native_definition_or_files(backend, monkeypatch):
    fluid = jp.Fluid("CO2", backend=backend)
    payload = pickle.dumps(fluid)
    original = cache_metadata.source_signature

    def changed(*args, **kwargs):
        signature = original(*args, **kwargs)
        component = signature["components"][0]
        if backend == "REFPROP":
            component["native_files"]["interaction"]["sha256"] = "changed-native-file"
        else:
            component["native_definition"]["sha256"] = "changed-native-definition"
        return signature

    monkeypatch.setattr(cache_metadata, "source_signature", changed)
    with pytest.raises(jp.InvalidConfigurationError) as caught:
        pickle.loads(payload)
    assert caught.value.request == "serialization"
    assert "native definitions or files" in caught.value.reason
    assert fluid.get_state(jp.PT_INPUTS, 1e6, 500.0).valid


def test_unverifiable_backend_still_evaluates_but_rejects_lossless_serialization(
    monkeypatch,
):
    monkeypatch.setattr(
        source_lifecycle,
        "source_serialization_identity",
        lambda *args: (None, "The test source has no verifiable native definition."),
    )
    fluid = jp.Fluid("CO2")
    assert fluid.get_state(jp.PT_INPUTS, 1e6, 500.0).valid
    with pytest.raises(jp.InvalidConfigurationError) as caught:
        pickle.dumps(fluid)
    assert caught.value.request == "serialization"
    assert "no verifiable native definition" in caught.value.reason


def test_legacy_unverified_source_payload_is_rejected_explicitly():
    fluid = jp.Fluid("CO2")
    state = fluid.__getstate__()
    state.pop("_serialized_source_identity")
    with pytest.raises(
        jp.InvalidConfigurationError, match="no verifiable native-source identity"
    ):
        jp.FluidCoolProp.__new__(jp.FluidCoolProp).__setstate__(state)


@pytest.mark.parametrize(
    "methods",
    [
        ("coolprop", "coolprop"),
        ("coolprop_jax", "coolprop_jax"),
        ("coolprop", "perfect_gas"),
        ("perfect_gas", "perfect_gas"),
    ],
)
def test_binary_snapshot_rebuilds_prepared_accessors_from_restored_components(methods):
    components = tuple(
        jp.Fluid(name, method=method)
        for name, method in zip(("CO2", "Nitrogen"), methods)
    )
    fluid = jp.Fluid(
        "binary",
        method="binary_mixture",
        identifier="mixed-stream",
        model_options=dict(components=components, mass_fractions=(0.3, 0.7)),
    )
    reference = fluid.get_state(jp.PT_INPUTS, 2e6, 500.0)
    restored = pickle.loads(pickle.dumps(fluid))
    assert restored.configuration == fluid.configuration
    assert restored.references == fluid.references
    assert restored._execution == fluid._execution
    if restored._host_evaluator is not None:
        for component, entry in zip(
            restored.components, restored._host_evaluator.entries
        ):
            assert entry.original is component
            base = (
                component.fluid if component.provider == "coolprop_jax" else component
            )
            if getattr(base, "_AS", None) is not None:
                assert entry.source is base._AS
                assert entry.lock is base._state_lock
                assert entry.context is base._source_context
        assert all(
            fresh is not previous
            for fresh in restored._host_evaluator.locks
            for previous in fluid._host_evaluator.locks
        )
    for pair, first, second in (
        (jp.PT_INPUTS, [2e6, 2e6], [500.0, 500.0]),
        (jp.HmassP_INPUTS, reference.h, 2e6),
        (jp.PSmass_INPUTS, 2e6, reference.s),
    ):
        assert_same_state(
            restored.get_state(pair, first, second),
            fluid.get_state(pair, first, second),
        )
    if restored._execution == "callback":
        compiled = jax.jit(
            lambda p: restored.get_state(jp.PT_INPUTS, p, 500.0, errors="mask")
        )
        assert_same_state(compiled(jnp.asarray(2e6)), reference)


def test_shared_source_aliases_survive_binary_snapshot_and_lock_preparation():
    component = jp.Fluid("CO2")
    fluid = jp.Fluid(
        "same-source",
        method="binary_mixture",
        model_options=dict(
            components=(component, component), mass_fractions=(0.4, 0.6)
        ),
    )
    restored = pickle.loads(pickle.dumps(fluid))
    assert restored.components[0] is restored.components[1]
    assert restored._host_evaluator.same_component
    assert len(restored._host_evaluator.locks) == 1
    assert_same_state(
        restored.get_state(jp.PT_INPUTS, 1e6, 500.0),
        fluid.get_state(jp.PT_INPUTS, 1e6, 500.0),
    )


def test_restored_source_batches_and_previously_returned_states_are_isolated():
    fluid = pickle.loads(pickle.dumps(jp.Fluid("CO2")))
    initial = fluid.get_state(jp.PT_INPUTS, 1e6, 500.0)
    before = {
        name: np.array(initial.raw_property(name))
        for name in initial.available_properties
    }

    def evaluate(offset):
        pressure = np.linspace(1e6, 2e6, 8) + offset
        result = fluid.get_state(jp.PT_INPUTS, pressure, 500.0, supersaturation=True)
        assert np.all(result.valid)
        np.testing.assert_allclose(result.p, pressure, rtol=2e-12)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(evaluate, (0, 1e4, 2e4, 3e4)))
    for offset, actual in zip((0, 1e4, 2e4, 3e4), results):
        assert_same_state(actual, evaluate(offset))
    for name, values in before.items():
        np.testing.assert_array_equal(initial.raw_property(name), values)


def test_native_model_snapshots_preserve_materialized_constants_and_table_updates(
    tmp_path,
):
    gas = jp.Fluid("CO2", method="perfect_gas").with_constants(
        gamma=1.37, h_ref=-1234.0
    )
    table = jp.Fluid(
        "CO2",
        method="bicubic",
        table_options=dict(
            h_min=600e3,
            h_max=850e3,
            p_min=5e6,
            p_max=12e6,
            N_h=4,
            N_p=4,
            N_p_sat=4,
            coarse_step=1,
            table_dir=str(tmp_path),
        ),
    )
    coefficients = table._table["temperature"]["coeffs"] * 1.01
    view = eqx.tree_at(
        lambda model: model._table["temperature"]["coeffs"], table, coefficients
    )
    for model in (gas.materialize(), view):
        restored = pickle.loads(pickle.dumps(model))
        assert restored.configuration == model.configuration
        assert restored.reference_id == model.reference_id
        assert restored.references == model.references
        assert_same_state(
            restored.get_state(jp.HmassP_INPUTS, 700e3, 8e6),
            model.get_state(jp.HmassP_INPUTS, 700e3, 8e6),
        )


def test_shifted_reference_snapshots_never_mutate_or_adopt_global_conventions():
    program = r"""
import json, pickle
import CoolProp.CoolProp as CP
import jaxprop as jp
from jaxprop.source_references import source_reference_metadata
old = [jp.Fluid('CO2', method=method) for method in ('coolprop', 'coolprop_jax')]
states = [f.get_state(jp.PT_INPUTS, 1e6, 500.0) for f in old]
CP.set_reference_state('CO2', 300.0, 500.0, 12345.0, 67.0)
shifted = source_reference_metadata('CO2')['reference_id']
# Serialization must keep the configured old convention even after the global
# convention changes; it must not relabel it with a freshly probed identity.
payloads = [pickle.dumps(f) for f in old]
rejected = 0
for payload in payloads:
    try:
        pickle.loads(payload)
    except jp.InvalidConfigurationError as error:
        assert error.request == 'source_reference'
        rejected += 1
    else:
        raise AssertionError('A different energy reference must reject restoration')
assert source_reference_metadata('CO2')['reference_id'] == shifted
shifted_fluid = jp.Fluid('CO2')
shifted_copy = pickle.loads(pickle.dumps(shifted_fluid))
assert shifted_copy.get_state(jp.PT_INPUTS, 1e6, 500.0).h == shifted_fluid.get_state(jp.PT_INPUTS, 1e6, 500.0).h
for f, state in zip(old, states):
    assert f.get_state(jp.PT_INPUTS, 1e6, 500.0).h == state.h
CP.set_reference_state('CO2', 'DEF')
for payload, state in zip(payloads, states):
    restored = pickle.loads(payload)
    assert restored.get_state(jp.PT_INPUTS, 1e6, 500.0).h == state.h
print(json.dumps({'rejected': rejected, 'global_reference_preserved': True, 'matching_restore': True}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == dict(
        rejected=2, global_reference_preserved=True, matching_restore=True
    )


def test_callbacks_respect_explicit_float32_execution_after_snapshot_restore():
    program = r"""
import json, pickle
import CoolProp.CoolProp as CP
import jax
import jax.numpy as jnp
import numpy as np
import jaxprop as jp
assert not jax.config.x64_enabled
components = tuple(jp.Fluid(name, method='coolprop_jax') for name in ('CO2', 'Nitrogen'))
binary = jp.Fluid('binary', method='binary_mixture',
    model_options=dict(components=components, mass_fractions=(0.3, 0.7)))
sources = [CP.AbstractState('HEOS', name) for name in ('CO2', 'Nitrogen')]
for source in sources:
    source.update(CP.PT_INPUTS, 2e6, 500.0)
density = [s.rhomass() for s in sources]
derivative = [s.first_partial_deriv(CP.iDmass, CP.iP, CP.iT) for s in sources]
mixed_density = 1 / (0.3/density[0] + 0.7/density[1])
mixed_derivative = mixed_density**2 * sum(w*dr/r**2 for w, dr, r in zip((0.3,0.7), derivative, density))
for fluid, expected_rho, expected_drho in ((components[0], density[0], derivative[0]), (binary, mixed_density, mixed_derivative)):
    restored = pickle.loads(pickle.dumps(fluid))
    call = lambda p: restored.get_state(jp.PT_INPUTS, p, jnp.asarray(500.0), errors='mask')
    compiled = jax.jit(call)
    actual = compiled(jnp.asarray(2e6))
    eager = call(jnp.asarray(2e6))
    for name in actual.available_properties:
        assert actual.raw_property(name).dtype == np.dtype('float32'), name
        np.testing.assert_allclose(actual.raw_property(name), eager.raw_property(name), rtol=2e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(actual.rho, expected_rho, rtol=2e-6)
    actual_derivative = jax.jit(jax.grad(lambda p: compiled(p).rho))(jnp.asarray(2e6))
    np.testing.assert_allclose(actual_derivative, expected_drho, rtol=3e-3, atol=1e-10)
    failed = compiled(jnp.asarray([2e6, -1.0]))
    assert list(np.asarray(failed.status) == 0) == [True, False]
    assert np.isfinite(failed.rho[0]) and np.isnan(failed.rho[1])
print(json.dumps({'precision': 'float32', 'source_and_binary': True, 'derivatives_checked': True}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "JAX_ENABLE_X64": "false"},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == dict(
        precision="float32", source_and_binary=True, derivatives_checked=True
    )


def test_fresh_worker_reconstructs_source_callback_and_binary_snapshots(tmp_path):
    """A new process must rebuild native objects, not borrow parent handles."""
    components = tuple(
        jp.Fluid(name, method="coolprop_jax") for name in ("CO2", "Nitrogen")
    )
    fluids = (
        jp.Fluid("CO2"),
        components[0],
        jp.Fluid(
            "worker-binary",
            method="binary_mixture",
            model_options=dict(components=components, mass_fractions=(0.3, 0.7)),
        ),
    )
    snapshot = tmp_path / "configured_models.pkl"
    snapshot.write_bytes(pickle.dumps(fluids))
    program = r"""
import json, pickle, sys
from pathlib import Path
import jax
import jax.numpy as jnp
import jaxprop as jp
fluids = pickle.loads(Path(sys.argv[1]).read_bytes())
result = []
for fluid in fluids:
    if fluid.provider == 'coolprop':
        state = fluid.get_state(jp.PT_INPUTS, 2e6, 500.0)
    else:
        state = jax.jit(lambda p: fluid.get_state(jp.PT_INPUTS, p, 500.0, errors='mask'))(jnp.asarray(2e6))
    assert int(state.status) == 0
    result.append({key:float(state.raw_property(key)) for key in ('pressure', 'temperature', 'density', 'enthalpy', 'entropy', 'internal_energy')})
print(json.dumps(result))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(snapshot)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "JAX_ENABLE_X64": "true"},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    for fluid, actual in zip(fluids, result):
        expected = fluid.get_state(jp.PT_INPUTS, 2e6, 500.0)
        for name, value in actual.items():
            np.testing.assert_allclose(value, expected.raw_property(name), rtol=2e-12)
