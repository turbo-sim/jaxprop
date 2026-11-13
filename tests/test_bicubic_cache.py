"""Table-cache identity, validation and publication through public construction."""

from copy import deepcopy
import builtins
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import pickle
from threading import Barrier, Event
import time

import numpy as np
import pytest

import jaxprop as jxp
import jaxprop.bicubic.bicubic_interpolation as interpolation


def _polynomial_table(fluid, generation):
    """Complete deterministic arrays; no backend/table-generator work is mocked in the loader."""
    h = np.broadcast_to(np.asarray(fluid.h_vals)[:, None], (fluid.N_h, fluid.N_p))
    p = np.broadcast_to(np.exp(np.asarray(fluid.logP_vals))[None, :], h.shape)
    table = {"metadata": deepcopy(fluid.metadata)}
    for name in jxp.PROPERTIES_CANONICAL:
        value = np.full(h.shape, 1.0)
        grad_h = np.zeros(h.shape)
        grad_p = np.zeros(h.shape)
        if name == "pressure":
            value, grad_p = p.copy(), np.ones(h.shape)
        elif name == "enthalpy":
            value, grad_h = h.copy(), np.ones(h.shape)
        elif name == "internal_energy":
            value = h - p / 100.0
            grad_h, grad_p = np.ones(h.shape), np.full(h.shape, -0.01)
        elif name == "density":
            value = np.full(h.shape, 100.0)
        elif name == "temperature":
            value = np.full(h.shape, 500.0 + generation)
        elif name == "entropy":
            value = np.full(h.shape, 2000.0)
        grad_logp = grad_p * p
        mixed = np.zeros(h.shape)
        table[name] = {
            "value": value,
            "grad_h": grad_h,
            "grad_p": grad_p,
            "grad_logP": grad_logp,
            "grad_ph": mixed.copy(),
            "grad_hlogP": mixed.copy(),
            "coeffs": interpolation.compute_bicubic_coefficients(
                value,
                grad_h,
                grad_logp,
                mixed,
                fluid.delta_h,
                fluid.delta_logP,
            ),
        }
    table["saturation_props"] = {}
    if fluid.metastable_phase is not None:
        sat_p = np.exp(np.asarray(fluid.logP_sat_vals))
        for name in jxp.PROPERTIES_CANONICAL:
            value = np.full(fluid.N_p_sat, 1.0)
            if name == "pressure":
                value = sat_p
            elif name == "enthalpy":
                value = np.full(fluid.N_p_sat, float(np.mean(h)))
            elif name == "internal_energy":
                value = np.full(fluid.N_p_sat, float(np.mean(h))) - sat_p / 100.0
            elif name == "density":
                value = np.full(fluid.N_p_sat, 100.0)
            elif name == "temperature":
                value = np.full(fluid.N_p_sat, 500.0)
            table["saturation_props"][name] = {
                "value": value,
                "grad_p": (
                    np.ones(fluid.N_p_sat)
                    if name == "pressure"
                    else np.zeros(fluid.N_p_sat)
                ),
            }
    return table


@pytest.fixture
def cache_factory(tmp_path, monkeypatch):
    generated = []

    def generator(mode):
        def generate(fluid):
            generated.append(mode)
            return _polynomial_table(fluid, len(generated))

        return generate

    for method, mode in (
        ("_generate_property_table", "equilibrium"),
        ("_generate_mixture_property_table", "mixture"),
        ("_generate_metastable_liquid_property_table", "metastable_liquid"),
        ("_generate_metastable_vapor_property_table", "metastable_vapor"),
    ):
        monkeypatch.setattr(jxp.FluidBicubic, method, generator(mode))

    def create(**overrides):
        settings = dict(
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
            table_dir=str(tmp_path),
            table_name="cache",
        )
        settings.update(overrides)
        return jxp.FluidBicubic(**settings)

    return create, generated, tmp_path


def _cache_path(fluid):
    return Path(fluid.table_dir) / f"{fluid.table_name}.pkl"


def _edit_cache(fluid, mutate):
    path = _cache_path(fluid)
    with path.open("rb") as stream:
        table = pickle.load(stream)
    mutate(table)
    with path.open("wb") as stream:
        pickle.dump(table, stream)


def test_matching_generation_manifest_loads_without_generating_again(cache_factory):
    create, generated, _ = cache_factory
    first = create()
    second = create()
    assert generated == ["equilibrium"]
    assert second.metadata == second.table["metadata"] == first.metadata
    for name in jxp.PROPERTIES_CANONICAL:
        for field in first.table[name]:
            np.testing.assert_array_equal(
                first.table[name][field], second.table[name][field]
            )


def test_generation_manifest_explicitly_records_modes_numerics_schema_and_source(
    cache_factory,
):
    create, _, _ = cache_factory
    fluid = create()
    required = {
        "fluid",
        "backend",
        "h_min",
        "h_max",
        "p_min",
        "p_max",
        "N_h",
        "N_p",
        "delta_h",
        "delta_logP",
        "gradient_method",
        "N_p_sat",
        "mixture_ratio",
        "metastable_phase",
        "table_mode",
        "cache_format_version",
        "generator_revision",
        "generator_code_fingerprint",
        "source_signature",
        "properties",
        "numeric_dtype",
        "jax_enable_x64",
        "generation_settings",
    }
    assert required <= fluid.metadata.keys()
    assert fluid.metadata["gradient_method"] == "central"
    assert fluid.metadata["N_p_sat"] == 4
    assert fluid.metadata["mixture_ratio"] is None
    assert fluid.metadata["metastable_phase"] is None
    assert fluid.metadata["table_mode"] == "equilibrium"
    assert set(fluid.metadata["properties"]) == set(jxp.PROPERTIES_CANONICAL)
    assert isinstance(fluid.metadata["source_signature"], dict)
    for irrelevant in ("identifier", "coarse_step", "table_dir", "table_name"):
        assert irrelevant not in fluid.metadata


@pytest.mark.parametrize(
    "identity", ["cache_format_version", "generator_revision", "source_signature"]
)
def test_changed_format_generator_or_source_identity_invalidates_the_cache(
    cache_factory,
    identity,
):
    create, generated, _ = cache_factory
    first = create()

    def mutate(table):
        if identity == "source_signature":
            table["metadata"][identity]["deliberate_source_change"] = True
        else:
            table["metadata"][identity] = "deliberately obsolete"

    _edit_cache(first, mutate)
    restored = create()
    assert generated == ["equilibrium", "equilibrium"]
    assert restored.table["metadata"] == restored.metadata


@pytest.mark.parametrize(
    "changed",
    [
        {"h_min": 610e3},
        {"h_max": 860e3},
        {"p_min": 6e6},
        {"p_max": 13e6},
        {"N_h": 5},
        {"N_p": 5},
        {"N_p_sat": 5},
        {"gradient_method": "forward"},
    ],
)
def test_changed_generation_settings_regenerate_an_explicit_cache_name(
    cache_factory, changed
):
    create, generated, _ = cache_factory
    first = create()
    changed_fluid = create(**changed)
    assert generated == ["equilibrium", "equilibrium"]
    assert changed_fluid.metadata != first.metadata
    assert changed_fluid.table["metadata"] == changed_fluid.metadata
    np.testing.assert_array_equal(changed_fluid.table["temperature"]["value"], 502.0)
    create(**changed)
    assert len(generated) == 2


def test_runtime_identity_and_inverse_guess_stride_do_not_invalidate_table(
    cache_factory,
):
    create, generated, _ = cache_factory
    first = create(identifier="first", coarse_step=1)
    second = create(identifier="second", coarse_step=2)
    assert generated == ["equilibrium"]
    assert first.metadata == second.metadata
    assert second.identifier == "second" and second.coarse_step == 2


@pytest.mark.parametrize("mutation", ["extra", "missing", "changed"])
def test_metadata_matching_checks_the_complete_manifest(cache_factory, mutation):
    create, generated, _ = cache_factory
    first = create()

    def mutate(table):
        metadata = table["metadata"]
        if mutation == "extra":
            metadata["obsolete_generation_option"] = "must not be ignored"
        elif mutation == "missing":
            metadata.pop(next(iter(metadata)))
        else:
            key = next(iter(metadata))
            metadata[key] = "deliberately changed"

    _edit_cache(first, mutate)
    restored = create()
    assert len(generated) == 2
    assert restored.table["metadata"] == restored.metadata


def test_mode_transition_invalidates_and_dispatches_the_requested_generator(
    cache_factory,
):
    create, generated, _ = cache_factory
    create()
    create(metastable_phase="liquid", gradient_method="forward")
    create(metastable_phase="vapor", gradient_method="forward")
    final = create()
    assert generated == [
        "equilibrium",
        "metastable_liquid",
        "metastable_vapor",
        "equilibrium",
    ]
    assert final.table["saturation_props"] == {}


def test_mixture_composition_is_not_rounded_in_cache_identity(cache_factory):
    create, generated, _ = cache_factory
    first = create(
        fluid_name="water_nitrogen_mixture", mixture_ratio=0.101, table_name=None
    )
    second = create(
        fluid_name="water_nitrogen_mixture", mixture_ratio=0.102, table_name=None
    )
    assert generated == ["mixture", "mixture"]
    assert first.table_name != second.table_name
    assert _cache_path(first).exists() and _cache_path(second).exists()


def test_automatic_cache_names_distinguish_generation_schemes(cache_factory):
    create, generated, _ = cache_factory
    first = create(table_name=None, gradient_method="central")
    second = create(table_name=None, gradient_method="forward")
    assert generated == ["equilibrium", "equilibrium"]
    assert first.table_name != second.table_name


@pytest.mark.parametrize(
    "payload",
    [b"", b"not a pickle", b"\x80\x05\x95", b"cmissing_table_cache_module\nMissing\n."],
)
def test_truncated_or_unreadable_cache_is_regenerated(cache_factory, payload):
    create, generated, _ = cache_factory
    first = create()
    _cache_path(first).write_bytes(payload)
    loaded = create()
    assert generated == ["equilibrium", "equilibrium"]
    assert loaded.table["metadata"] == loaded.metadata


def test_cache_permission_errors_are_not_misreported_as_invalid_content(
    cache_factory, monkeypatch
):
    create, generated, _ = cache_factory
    first = create()
    target = _cache_path(first)
    original_open = builtins.open
    original_path_open = Path.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file) == target and mode == "rb":
            raise PermissionError("deliberate cache read denial")
        return original_open(file, mode, *args, **kwargs)

    def guarded_path_open(path, mode="r", *args, **kwargs):
        if path == target and mode == "rb":
            raise PermissionError("deliberate cache read denial")
        return original_path_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    with pytest.raises(PermissionError, match="deliberate cache read denial"):
        create()
    assert generated == ["equilibrium"]


@pytest.mark.parametrize(
    "damage",
    [
        "not_a_table",
        "missing_property",
        "missing_array",
        "wrong_shape",
        "wrong_dtype",
        "nan_core",
        "infinite_core",
        "infinite_optional",
        "nan_derivative",
        "nan_coefficients",
    ],
)
def test_malformed_or_incomplete_cache_arrays_are_regenerated(cache_factory, damage):
    create, generated, _ = cache_factory
    first = create()

    def mutate(table):
        if damage == "missing_property":
            table.pop("density")
        elif damage == "missing_array":
            table["temperature"].pop("coeffs")
        elif damage == "wrong_shape":
            table["temperature"]["value"] = np.ones((2, 2))
        elif damage == "wrong_dtype":
            table["temperature"]["value"] = np.ones((4, 4), dtype=np.float32)
        elif damage == "nan_core":
            table["temperature"]["value"] = np.full((4, 4), np.nan)
        elif damage == "infinite_core":
            table["density"]["value"] = np.full((4, 4), np.inf)
        elif damage == "infinite_optional":
            table["surface_tension"]["value"] = np.full((4, 4), np.inf)
        elif damage == "nan_derivative":
            table["temperature"]["grad_h"] = np.full((4, 4), np.nan)
        elif damage == "nan_coefficients":
            table["temperature"]["coeffs"] = np.full((3, 3, 16), np.nan)

    if damage == "not_a_table":
        with _cache_path(first).open("wb") as stream:
            pickle.dump(["not", "a", "table"], stream)
    else:
        _edit_cache(first, mutate)
    loaded = create()
    assert generated == ["equilibrium", "equilibrium"]
    assert loaded.table["temperature"]["value"].shape == (4, 4)


def test_undefined_optional_properties_are_valid_cache_data(cache_factory):
    create, generated, _ = cache_factory
    first = create()

    def mutate(table):
        for field, value in table["surface_tension"].items():
            table["surface_tension"][field] = np.full(np.shape(value), np.nan)

    _edit_cache(first, mutate)
    loaded = create()
    assert generated == ["equilibrium"]
    assert np.all(np.isnan(loaded.table["surface_tension"]["value"]))


def test_generation_failure_does_not_publish_or_replace_a_cache(
    cache_factory, monkeypatch
):
    create, generated, directory = cache_factory
    first = create()
    previous = _cache_path(first).read_bytes()

    def failed_generation(fluid):
        raise RuntimeError("deliberate generation failure")

    monkeypatch.setattr(jxp.FluidBicubic, "_generate_property_table", failed_generation)
    with pytest.raises(RuntimeError, match="deliberate generation failure"):
        create(gradient_method="forward")
    assert _cache_path(first).read_bytes() == previous
    assert sorted(path.name for path in directory.iterdir()) == ["cache.pkl"]


def test_invalid_generated_arrays_are_rejected_without_publishing(
    cache_factory, monkeypatch
):
    create, generated, directory = cache_factory
    first = create()
    previous = _cache_path(first).read_bytes()

    def invalid_generation(fluid):
        table = _polynomial_table(fluid, 99)
        table["density"]["value"][:] = np.nan
        return table

    monkeypatch.setattr(
        jxp.FluidBicubic, "_generate_property_table", invalid_generation
    )
    with pytest.raises(jxp.CalculationError):
        create(gradient_method="forward")
    assert _cache_path(first).read_bytes() == previous
    assert sorted(path.name for path in directory.iterdir()) == ["cache.pkl"]


def test_failed_atomic_publication_keeps_the_previous_complete_cache(
    cache_factory, monkeypatch
):
    create, generated, directory = cache_factory
    first = create()
    previous = _cache_path(first).read_bytes()

    def failed_replace(*args, **kwargs):
        raise PermissionError("deliberate atomic replace failure")

    monkeypatch.setattr(interpolation.os, "replace", failed_replace)
    with pytest.raises(PermissionError, match="deliberate atomic replace failure"):
        create(gradient_method="forward")
    assert _cache_path(first).read_bytes() == previous
    assert sorted(path.name for path in directory.iterdir()) == ["cache.pkl"]


def test_failed_pickle_write_keeps_the_previous_complete_cache(
    cache_factory, monkeypatch
):
    create, generated, directory = cache_factory
    first = create()
    previous = _cache_path(first).read_bytes()

    def failed_dump(*args, **kwargs):
        raise OSError("deliberate serialization failure")

    monkeypatch.setattr(pickle, "dump", failed_dump)
    with pytest.raises(OSError, match="deliberate serialization failure"):
        create(gradient_method="forward")
    assert _cache_path(first).read_bytes() == previous
    assert sorted(path.name for path in directory.iterdir()) == ["cache.pkl"]


def test_metastable_central_scheme_is_explicitly_unsupported(cache_factory):
    create, generated, _ = cache_factory
    with pytest.raises(jxp.UnsupportedOptionError):
        create(metastable_phase="liquid", gradient_method="central")
    assert generated == []


@pytest.mark.parametrize(
    "invalid",
    [
        {"N_h": 1},
        {"N_p": 1},
        {"N_h": 2.5},
        {"N_p": True},
        {"N_p_sat": 1},
        {"gradient_method": "not_a_method"},
        {"coarse_step": 0},
        {"h_min": 850e3},
        {"h_max": np.inf},
        {"p_min": 0.0},
        {"p_max": 5e6},
        {"metastable_phase": "not_a_phase"},
        {"mixture_ratio": 0.1},
        {"fluid_name": "water_nitrogen_mixture"},
        {"fluid_name": "water_nitrogen_mixture", "mixture_ratio": -1.0},
        {"fluid_name": "water_nitrogen_mixture", "mixture_ratio": np.nan},
        {
            "fluid_name": "water_nitrogen_mixture",
            "mixture_ratio": 0.1,
            "metastable_phase": "liquid",
        },
    ],
)
def test_invalid_generation_configuration_is_rejected_before_generating(
    cache_factory, invalid
):
    create, generated, _ = cache_factory
    with pytest.raises(jxp.InvalidConfigurationError):
        create(**invalid)
    assert generated == []


def test_unverifiable_source_never_reads_or_publishes_a_persistent_cache(
    cache_factory,
    monkeypatch,
):
    create, generated, directory = cache_factory
    original = create()
    target = _cache_path(original)
    previous = target.read_bytes()
    signature = deepcopy(original.metadata["source_signature"])
    signature["deterministic_cache_identity"] = False
    monkeypatch.setattr(
        interpolation, "source_signature", lambda *args: deepcopy(signature)
    )
    original_open = builtins.open

    def prohibit_cached_read(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file) == target and mode == "rb":
            raise AssertionError("an unverifiable source must not read its old cache")
        return original_open(file, mode, *args, **kwargs)

    def prohibit_publication(*args, **kwargs):
        raise AssertionError("an unverifiable source must not publish a cache")

    monkeypatch.setattr(builtins, "open", prohibit_cached_read)
    monkeypatch.setattr(interpolation, "save_table_atomic", prohibit_publication)
    for _ in range(2):
        with pytest.warns(UserWarning, match="persistent table caching is disabled"):
            current = create()
        assert (
            current.metadata["source_signature"]["deterministic_cache_identity"]
            is False
        )
    unpublished_directory = directory / "unpublished"
    with pytest.warns(UserWarning, match="persistent table caching is disabled"):
        create(table_dir=str(unpublished_directory))
    assert not unpublished_directory.exists()
    assert generated == ["equilibrium"] * 4
    assert target.read_bytes() == previous
    assert sorted(path.name for path in directory.iterdir()) == ["cache.pkl"]


def test_reference_change_during_initialization_rejects_mislabeled_cache(
    cache_factory, monkeypatch
):
    create, generated, _ = cache_factory
    original = create()
    target = _cache_path(original)
    previous = target.read_bytes()
    signature = deepcopy(original.metadata["source_signature"])
    signature["source_reference_id"] = ("real_fluid", "different_reference")
    monkeypatch.setattr(
        interpolation, "source_signature", lambda *args: deepcopy(signature)
    )

    with pytest.raises(
        jxp.InvalidConfigurationError, match="changed during table initialization"
    ) as captured:
        create()
    assert captured.value.request == "source_reference"
    assert generated == ["equilibrium"]
    assert target.read_bytes() == previous


def test_source_change_during_generation_rejects_publication_and_preserves_old_cache(
    cache_factory,
    monkeypatch,
):
    create, generated, directory = cache_factory
    original = create()
    target = _cache_path(original)
    previous = target.read_bytes()
    signature = deepcopy(original.metadata["source_signature"])
    calls = 0

    def changing_source(*args):
        nonlocal calls
        calls += 1
        result = deepcopy(signature)
        if calls > 1:
            result["deliberate_reference_change"] = True
        return result

    monkeypatch.setattr(interpolation, "source_signature", changing_source)
    with pytest.warns(UserWarning, match="Regenerating property table"):
        with pytest.raises(
            jxp.CalculationError, match="changed during table generation"
        ):
            create(gradient_method="forward")
    assert calls == 2
    assert generated == ["equilibrium", "equilibrium"]
    assert target.read_bytes() == previous
    assert sorted(path.name for path in directory.iterdir()) == ["cache.pkl"]


def test_concurrent_atomic_writers_expose_only_complete_pickles_to_readers(
    tmp_path,
    monkeypatch,
):
    from jaxprop.bicubic import table_cache

    target = tmp_path / "atomic.pkl"
    payloads = {
        number: {
            "writer": number,
            "values": np.arange(4096, dtype=float) + number,
            "complete": True,
        }
        for number in (0, 1, 2)
    }
    table_cache.save_table_atomic(payloads[0], target)
    original_dump = pickle.dump
    partial_writes = Barrier(3)
    release_writers = Event()
    stop_reading = Event()
    reader_started = Event()
    writers_finished = Event()
    read_after_publication = Event()
    observations = []
    read_denials = []

    def read_complete():
        with target.open("rb") as stream:
            actual = pickle.load(stream)
        number = actual["writer"]
        assert number in payloads and actual["complete"] is True
        np.testing.assert_array_equal(actual["values"], payloads[number]["values"])
        return number

    def reader():
        denial_deadline = None
        while not stop_reading.is_set():
            try:
                number = read_complete()
            except PermissionError:
                if os.name != "nt":
                    raise
                # Windows can briefly deny opening a destination while another
                # thread replaces it. Retry sharing denials, not corrupt data.
                if denial_deadline is None:
                    denial_deadline = time.monotonic() + 1.0
                if time.monotonic() >= denial_deadline:
                    raise
                read_denials.append(True)
                stop_reading.wait(0.001)
                continue
            denial_deadline = None
            observations.append(number)
            reader_started.set()
            if writers_finished.is_set():
                read_after_publication.set()

    def pause_halfway_through_temporary_pickle(table, stream, protocol):
        serialized = pickle.dumps(table, protocol=protocol)
        midpoint = len(serialized) // 2
        stream.write(serialized[:midpoint])
        stream.flush()
        partial_writes.wait(timeout=10)
        assert release_writers.wait(timeout=10)
        stream.write(serialized[midpoint:])

    monkeypatch.setattr(pickle, "dump", pause_halfway_through_temporary_pickle)
    write_denials = []
    with ThreadPoolExecutor(max_workers=3) as workers:
        reading = workers.submit(reader)
        try:
            if not reader_started.wait(timeout=5):
                reading.result(timeout=0)
                pytest.fail("reader did not observe the initial complete cache")
            writing = [
                workers.submit(table_cache.save_table_atomic, payloads[number], target)
                for number in (1, 2)
            ]
            partial_writes.wait(timeout=10)
            # Both temporary files are incomplete while the public file stays valid.
            assert [read_complete() for _ in range(20)] == [0] * 20
            release_writers.set()
            for number, future in zip((1, 2), writing):
                try:
                    future.result(timeout=10)
                except PermissionError:
                    if os.name != "nt":
                        raise
                    # Windows can deny replacement while a reader holds a handle.
                    # An explicit I/O failure must leave a complete previous file.
                    write_denials.append(number)
            writers_finished.set()
            if not read_after_publication.wait(timeout=5):
                reading.result(timeout=0)
                pytest.fail("reader did not observe a complete cache after publication")
        finally:
            release_writers.set()
            stop_reading.set()
        reading.result(timeout=10)
    monkeypatch.setattr(pickle, "dump", original_dump)
    for number in write_denials:
        table_cache.save_table_atomic(payloads[number], target)
    assert observations
    assert reader_started.is_set() and read_after_publication.is_set()
    assert read_complete() in (1, 2)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["atomic.pkl"]
