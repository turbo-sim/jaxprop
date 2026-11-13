"""Source provenance protects tables against global and backend changes."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import CoolProp.CoolProp as CP
import pytest

from jaxprop import InvalidConfigurationError
from jaxprop.bicubic import cache_metadata
from jaxprop.bicubic.cache_metadata import metadata_fingerprint, source_signature
from jaxprop.source_references import source_reference_metadata
from utilities import get_available_backends


def test_metadata_fingerprint_is_order_independent_and_sensitive_to_nested_values():
    first = {"source": {"version": "1", "reference": [1.0, 2.0]}, "size": 16}
    reordered = {"size": 16, "source": {"reference": [1.0, 2.0], "version": "1"}}
    assert metadata_fingerprint(first) == metadata_fingerprint(reordered)
    assert len(metadata_fingerprint(first)) == 64
    changed = {"source": {"version": "1", "reference": [1.0, 2.001]}, "size": 16}
    assert metadata_fingerprint(first) != metadata_fingerprint(changed)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_metadata_cannot_create_a_cache_fingerprint(bad_value):
    with pytest.raises(ValueError):
        metadata_fingerprint({"nested": {"value": bad_value}})


def test_aliases_produce_the_same_canonical_source_signature():
    alias = source_signature("CO2", "HEOS")
    canonical = source_signature("CarbonDioxide", "HEOS")
    assert alias == canonical
    assert alias["components"][0]["fluid_names"] == ["CarbonDioxide"]
    assert alias["components"][0]["native_definition"]["available"]
    assert alias["deterministic_cache_identity"] is True
    assert alias["coolprop_version"] == CP.get_global_param_string("version")
    assert alias["coolprop_git_revision"] == CP.get_global_param_string("gitrevision")
    assert (
        alias["source_reference_id"]
        == source_reference_metadata("CO2", "HEOS")["reference_id"]
    )
    json.dumps(alias, allow_nan=False)


@pytest.mark.parametrize("backend", get_available_backends())
@pytest.mark.parametrize("name", ["CO2", "Water", "Air"])
def test_sources_have_a_reproducible_finite_reference_anchor(backend, name):
    first = source_signature(name, backend)
    assert first == source_signature(name, backend)
    component = first["components"][0]
    reference = source_reference_metadata(name, backend)
    assert first["source_reference_id"] == component["source_reference_id"]
    assert first["source_reference_id"] == reference["reference_id"]
    assert component["reference_anchor"] == reference["reference_anchor"]
    hash(first["source_reference_id"])
    assert component["composition"] == {"basis": "mole", "fractions": [1.0]}
    anchor = component["reference_anchor"]
    assert anchor["input_pair"] in {"DmassT_INPUTS", "PT_INPUTS"}
    assert all(
        math.isfinite(anchor[key])
        for key in ("density", "temperature", "pressure", "enthalpy", "entropy")
    )
    assert all(anchor[key] > 0 for key in ("density", "temperature", "pressure"))
    if backend == "REFPROP":
        assert first["backend_version"] == CP.get_global_param_string("REFPROP_version")
        assert component["native_definition"]["available"] is False
        assert (
            first["deterministic_cache_identity"]
            is component["native_files"]["available"]
        )
        if component["native_files"]["available"]:
            assert len(component["native_files"]["library"]["sha256"]) == 64
            assert len(component["native_files"]["interaction"]["sha256"]) == 64
    else:
        assert first["backend_version"] == first["coolprop_version"]
        assert component["native_definition"]["available"] is True


@pytest.fixture
def native_files(tmp_path, monkeypatch):
    root = tmp_path / "refprop"
    fluids = root / "FLUIDS"
    fluids.mkdir(parents=True)
    library = root / ("REFPRP64.DLL" if os.name == "nt" else "librefprop.so")
    library.write_bytes(b"library-v1")
    (fluids / "WATER.FLD").write_bytes(b"water-v1")
    (fluids / "HMX.BNC").write_bytes(b"interaction-v1")
    monkeypatch.delenv("COOLPROP_REFPROP_ROOT", raising=False)
    monkeypatch.delenv("RPPREFIX", raising=False)
    monkeypatch.setattr(cache_metadata, "_loaded_refprop_library", lambda: library)
    return root, library, {"ALTERNATIVE_REFPROP_PATH": str(root)}


def test_refprop_native_files_include_actual_library_components_and_interactions(
    native_files,
):
    root, library, configuration = native_files
    identity = cache_metadata._refprop_native_files(["Water"], configuration)
    assert identity["available"]
    assert Path(identity["library"]["path"]) == library
    assert Path(identity["interaction"]["path"]) == root / "FLUIDS/HMX.BNC"
    assert [Path(item["path"]).name for item in identity["components"]] == ["WATER.FLD"]


@pytest.mark.parametrize("filename", ["WATER.FLD", "HMX.BNC", "library"])
def test_native_content_changes_invalidate_even_with_original_size_and_timestamp(
    native_files, filename
):
    root, library, configuration = native_files
    before = cache_metadata._refprop_native_files(["Water"], configuration)
    path = library if filename == "library" else root / "FLUIDS" / filename
    original_stat = path.stat()
    original = path.read_bytes()
    path.write_bytes(original[:-1] + b"2")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    after = cache_metadata._refprop_native_files(["Water"], configuration)
    assert before["available"] and after["available"]
    assert path.stat().st_size == original_stat.st_size
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert metadata_fingerprint(before) != metadata_fingerprint(after)


def test_native_timestamp_only_changes_do_not_invalidate_identical_content(
    native_files,
):
    root, _, configuration = native_files
    before = cache_metadata._refprop_native_files(["Water"], configuration)
    path = root / "FLUIDS/WATER.FLD"
    original_stat = path.stat()
    os.utime(
        path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 2_000_000_000)
    )
    assert before == cache_metadata._refprop_native_files(["Water"], configuration)


def test_configured_interaction_file_overrides_default(native_files, tmp_path):
    root, _, configuration = native_files
    alternate = tmp_path / "custom-interaction.BNC"
    alternate.write_bytes(b"custom-interactions")
    configuration["ALTERNATIVE_REFPROP_HMX_BNC_PATH"] = str(alternate)
    before = cache_metadata._refprop_native_files(["Water"], configuration)
    assert before["available"]
    assert Path(before["interaction"]["path"]) == alternate
    (root / "FLUIDS/HMX.BNC").write_bytes(b"changed unused default")
    assert before == cache_metadata._refprop_native_files(["Water"], configuration)


def test_pseudo_pure_ppf_and_predefined_mix_dependencies_are_distinct(native_files):
    root, _, configuration = native_files
    fluids = root / "FLUIDS"
    (fluids / "AIR.PPF").write_bytes(b"pseudo-pure-air")
    for name in ("NITROGEN", "ARGON", "OXYGEN"):
        (fluids / f"{name}.FLD").write_bytes(name.encode())
    mixtures = root / "MIXTURES"
    mixtures.mkdir()
    (mixtures / "AIR.MIX").write_text(
        "Air (dry)\n 28.9 132.8\n3\nNITROGEN.FLD\nARGON.FLD\nOXYGEN.FLD\n.78\n.01\n.21\n",
        encoding="ascii",
    )
    pseudo = cache_metadata._refprop_native_files(["Air"], configuration)
    mixture = cache_metadata._refprop_native_files(["AIR.MIX"], configuration)
    assert pseudo["available"] and mixture["available"]
    assert {Path(item["path"]).name for item in pseudo["components"]} == {"AIR.PPF"}
    assert {Path(item["path"]).name for item in mixture["components"]} == {
        "AIR.MIX",
        "NITROGEN.FLD",
        "ARGON.FLD",
        "OXYGEN.FLD",
    }


@pytest.mark.parametrize("missing", ["WATER.FLD", "HMX.BNC", "library"])
def test_unresolved_native_files_are_explicitly_noncacheable(native_files, missing):
    root, library, configuration = native_files
    path = library if missing == "library" else root / "FLUIDS" / missing
    path.unlink()
    identity = cache_metadata._refprop_native_files(["Water"], configuration)
    assert identity["available"] is False
    assert identity["reason"]


def test_ambiguous_unconfigured_data_roots_are_not_assumed_to_be_the_source(
    native_files, tmp_path, monkeypatch
):
    root, _, _ = native_files
    other = tmp_path / "other"
    (other / "FLUIDS").mkdir(parents=True)
    for name in ("WATER.FLD", "HMX.BNC"):
        (other / "FLUIDS" / name).write_bytes(b"other-data")
    monkeypatch.setenv("RPPREFIX", str(other))
    result = cache_metadata._refprop_native_files(["Water"], {})
    assert result["available"] is False
    assert "multiple" in result["reason"]


def test_explicit_environment_root_has_priority_over_alternative_data_path(
    native_files, tmp_path, monkeypatch
):
    _, _, configuration = native_files
    other = tmp_path / "explicit-root"
    (other / "FLUIDS").mkdir(parents=True)
    for name in ("WATER.FLD", "HMX.BNC"):
        (other / "FLUIDS" / name).write_bytes(b"explicit-root-data")
    monkeypatch.setenv("COOLPROP_REFPROP_ROOT", str(other))
    result = cache_metadata._refprop_native_files(["Water"], configuration)
    assert result["available"]
    assert Path(result["data_root"]) == other


def test_configured_native_library_is_resolved_if_loaded_path_is_unobservable(
    native_files, monkeypatch
):
    _, library, configuration = native_files
    monkeypatch.setattr(cache_metadata, "_loaded_refprop_library", lambda: None)
    configuration["ALTERNATIVE_REFPROP_LIBRARY_PATH"] = str(library)
    result = cache_metadata._refprop_native_files(["Water"], configuration)
    assert result["available"]
    assert Path(result["library"]["path"]) == library


def test_binary_source_identity_contains_both_sources_and_the_unrounded_mass_ratio():
    first = source_signature("water_nitrogen_mixture", "HEOS", 0.104)
    second = source_signature("water_nitrogen_mixture", "HEOS", 0.105)
    assert [item["fluid_names"] for item in first["components"]] == [
        ["Water"],
        ["Nitrogen"],
    ]
    assert first["composition"]["component_1_to_2"] == 0.104
    assert first["composition"]["basis"] == "mass_ratio"
    assert sum(first["composition"]["mass_fractions"]) == pytest.approx(1.0)
    assert first["source_reference_id"] == (
        "binary_mixture",
        tuple(item["source_reference_id"] for item in first["components"]),
        ("mass_ratio", 0.104),
    )
    assert first["source_reference_id"] != second["source_reference_id"]
    assert metadata_fingerprint(first) != metadata_fingerprint(second)


def test_explicit_environment_library_has_priority_when_loaded_path_is_unobservable(
    native_files, tmp_path, monkeypatch
):
    _, library, configuration = native_files
    other = tmp_path / "explicit-root"
    (other / "FLUIDS").mkdir(parents=True)
    for name in ("WATER.FLD", "HMX.BNC"):
        (other / "FLUIDS" / name).write_bytes(b"explicit-root-data")
    selected_library = other / library.name
    selected_library.write_bytes(b"explicit-root-library")
    configuration["ALTERNATIVE_REFPROP_LIBRARY_PATH"] = str(library)
    monkeypatch.setattr(cache_metadata, "_loaded_refprop_library", lambda: None)
    monkeypatch.setenv("COOLPROP_REFPROP_ROOT", str(other))
    result = cache_metadata._refprop_native_files(["Water"], configuration)
    assert result["available"]
    assert Path(result["library"]["path"]) == selected_library


@pytest.mark.parametrize(
    "name,backend,ratio",
    [
        ("", "HEOS", None),
        ("CO2", "", None),
        ("no_such_fluid", "HEOS", None),
        ("CO2", "NO_BACKEND", None),
        ("CO2", "HEOS", 0.1),
        ("water_nitrogen_mixture", "HEOS", None),
        ("water_nitrogen_mixture", "HEOS", -0.1),
        ("water_nitrogen_mixture", "HEOS", math.inf),
        ("water_nitrogen_mixture", "HEOS", True),
    ],
)
def test_unverifiable_or_invalid_sources_fail_with_structured_configuration_errors(
    name, backend, ratio
):
    with pytest.raises(InvalidConfigurationError) as captured:
        source_signature(name, backend, ratio)
    assert captured.value.provider == "bicubic"
    assert captured.value.request
    assert captured.value.reason


def test_coolprop_version_changes_are_part_of_source_identity(monkeypatch):
    previous = source_signature("CO2", "HEOS")
    actual = CP.get_global_param_string

    def changed_version(key):
        return "test-other-version" if key == "version" else actual(key)

    monkeypatch.setattr(CP, "get_global_param_string", changed_version)
    changed = source_signature("CO2", "HEOS")
    before_component, after_component = (
        previous["components"][0],
        changed["components"][0],
    )
    assert before_component["native_definition"] == after_component["native_definition"]
    assert before_component["reference_anchor"] == after_component["reference_anchor"]
    assert (
        before_component["source_reference_id"]
        != after_component["source_reference_id"]
    )
    assert previous["source_reference_id"] != changed["source_reference_id"]
    assert metadata_fingerprint(previous) != metadata_fingerprint(changed)


def test_reference_anchor_falls_back_without_assuming_critical_data_exist(monkeypatch):
    actual_constructor = CP.AbstractState

    class WithoutCriticalData:
        def __init__(self, backend, name):
            self.state = actual_constructor(backend, name)

        def T_critical(self):
            raise ValueError("Critical data unavailable")

        def __getattr__(self, name):
            return getattr(self.state, name)

    monkeypatch.setattr(CP, "AbstractState", WithoutCriticalData)
    anchor = source_signature("CO2", "HEOS")["components"][0]["reference_anchor"]
    assert anchor["input_pair"] == "PT_INPUTS"
    assert anchor["inputs"] == [101325.0, 300.0]
    assert math.isfinite(anchor["enthalpy"])


def test_no_evaluable_anchor_rejects_unverifiable_cache_identity(monkeypatch):
    actual_constructor = CP.AbstractState

    class NoEvaluableState:
        def __init__(self, backend, name):
            self.state = actual_constructor(backend, name)

        def update(self, *args):
            raise ValueError("Source evaluation failed")

        def __getattr__(self, name):
            return getattr(self.state, name)

    monkeypatch.setattr(CP, "AbstractState", NoEvaluableState)
    with pytest.raises(InvalidConfigurationError, match="verifiable") as captured:
        source_signature("CO2", "HEOS")
    assert captured.value.request == "source_reference"
    assert isinstance(captured.value.__cause__, ValueError)


def test_invalid_native_definition_is_not_treated_as_an_unsupported_definition(
    monkeypatch,
):
    actual_constructor = CP.AbstractState

    class InvalidNativeDefinition:
        def __init__(self, backend, name):
            self.state = actual_constructor(backend, name)

        def fluid_param_string(self, parameter):
            return (
                "not JSON"
                if parameter == "JSON"
                else self.state.fluid_param_string(parameter)
            )

        def __getattr__(self, name):
            return getattr(self.state, name)

    monkeypatch.setattr(CP, "AbstractState", InvalidNativeDefinition)
    with pytest.raises(InvalidConfigurationError) as captured:
        source_signature("CO2", "HEOS")
    assert captured.value.request == "source_definition"


def test_reference_and_configuration_changes_invalidate_in_an_isolated_process():
    """Never reset an application's unknown global reference convention."""
    repo = Path(__file__).resolve().parents[1]
    script = r"""
import json
import CoolProp.CoolProp as CP
from jaxprop.bicubic.cache_metadata import source_signature, metadata_fingerprint
from jaxprop.source_references import source_reference_metadata

def probe():
    state = CP.AbstractState("HEOS", "CO2")
    state.update(CP.PT_INPUTS, 1e5, 400.0)
    return [state.hmass(), state.smass()]

ordinary_before = probe()
before = source_signature("CO2", "HEOS")
ordinary_after = probe()
CP.set_reference_state("CO2", 300.0, 500.0, 12345.0, 67.0)
changed_reference = source_signature("CO2", "HEOS")
custom_before = probe()
source_signature("CO2", "HEOS")
custom_after = probe()
CP.set_config_bool(CP.CRITICAL_SPLINES_ENABLED, not CP.get_config_bool(CP.CRITICAL_SPLINES_ENABLED))
changed_configuration = source_signature("CO2", "HEOS")
print(json.dumps({
    "unchanged_default": ordinary_before == ordinary_after,
    "unchanged_custom": custom_before == custom_after,
    "reference_changed": before["components"][0]["reference_anchor"] != changed_reference["components"][0]["reference_anchor"],
    "shared_reference_id_changed": before["source_reference_id"] != changed_reference["source_reference_id"],
    "shared_reference_id_matches": changed_reference["source_reference_id"] == source_reference_metadata("CO2", "HEOS")["reference_id"],
    "reference_fingerprint_changed": metadata_fingerprint(before) != metadata_fingerprint(changed_reference),
    "configuration_fingerprint_changed": metadata_fingerprint(changed_reference) != metadata_fingerprint(changed_configuration),
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    outcomes = json.loads(result.stdout.strip().splitlines()[-1])
    assert all(outcomes.values()), outcomes
