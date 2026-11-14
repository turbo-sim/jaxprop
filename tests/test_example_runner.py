"""Example import boundaries, profile/result isolation and honest execution status."""

from pathlib import Path
import importlib
import json
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import run_examples as runner


def test_catalogue_covers_each_original_program_and_classifies_nonexecutables():
    entries = runner.load_catalogue()
    original = {
        path.relative_to(ROOT / "examples").as_posix()
        for path in (ROOT / "examples").rglob("*.py")
        if not path.name.startswith("_") and path.name != "run_examples.py"
    }
    assert len(entries) == 65
    assert {entry["path"] for entry in entries} == original
    assert sum(entry["kind"] == "reference" for entry in entries) == 3
    assert sum(entry["kind"] == "input-tool" for entry in entries) == 3
    assert all(entry["purpose"] and entry["validation"] for entry in entries)
    assert all(set(entry["profiles"]) == {"quick", "full"} for entry in entries)


@pytest.fixture
def quiet_runner(monkeypatch):
    monkeypatch.setattr(runner, "package_hash", lambda: "verified-source")
    monkeypatch.setattr(runner, "framework_hash", lambda: "verified-framework")
    return next(
        entry
        for entry in runner.load_catalogue()
        if entry["path"] == "basic_calculations/solver_coolprop.py"
    )


def test_missing_input_and_reference_files_are_never_executed_or_passed(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Non-executable entry reached subprocess execution")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    for kind, expected in (
        ("reference", "reference_only"),
        ("input-tool", "input_required"),
    ):
        entry = next(
            entry for entry in runner.load_catalogue() if entry["kind"] == kind
        )
        result = runner.run_example(
            entry, profile="quick", output_dir=tmp_path, table_dir=tmp_path / "tables"
        )
        assert result["outcome"] == expected and not result["executed"]
    assert list(tmp_path.iterdir()) == []


def test_stale_success_manifest_cannot_make_a_new_empty_run_pass(
    tmp_path, monkeypatch, quiet_runner
):
    folder = tmp_path / "basic_calculations/solver_coolprop"
    folder.mkdir(parents=True)
    (folder / "example_result.json").write_text(
        json.dumps(dict(completed=True, profile="quick", checks={"old": True}))
    )
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0)
    )
    result = runner.run_example(
        quiet_runner,
        profile="quick",
        output_dir=tmp_path,
        table_dir=tmp_path / "tables",
    )
    assert result["outcome"] == "failed"
    assert "without its execution" in result["reason"]


def test_timed_out_program_is_not_a_quick_or_full_pass(
    tmp_path, monkeypatch, quiet_runner
):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    result = runner.run_example(
        quiet_runner,
        profile="full",
        output_dir=tmp_path,
        table_dir=tmp_path / "tables",
        timeout=0.1,
    )
    assert result["outcome"] == "timeout" and result["profile"] == "full"


def test_wrong_profile_manifest_is_rejected(tmp_path, monkeypatch, quiet_runner):
    def wrong(command, **kwargs):
        (kwargs["cwd"] / "example_result.json").write_text(
            json.dumps(dict(completed=True, profile="quick", checks={"sample": 3}))
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", wrong)
    result = runner.run_example(
        quiet_runner, profile="full", output_dir=tmp_path, table_dir=tmp_path / "tables"
    )
    assert result["outcome"] == "failed" and "requested profile" in result["reason"]


def test_real_basic_example_runs_from_an_independent_working_directory(tmp_path):
    entry = next(
        entry
        for entry in runner.load_catalogue()
        if entry["path"] == "basic_calculations/solver_coolprop.py"
    )
    result = runner.run_example(
        entry, profile="quick", output_dir=tmp_path, table_dir=tmp_path / "tables"
    )
    assert result["outcome"] == "completed", result
    assert result["execution"]["profile"] == "quick"
    assert result["execution"]["checks"]["state_checks"]
    assert (
        result["example_unchanged"]
        and result["package_unchanged"]
        and result["framework_unchanged"]
    )


def test_importing_all_examples_does_not_evaluate_fluids_plot_or_write_outputs(
    tmp_path,
):
    code = r"""
from pathlib import Path
import importlib, json, os
import CoolProp
import CoolProp.CoolProp as CP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jaxprop as jp
from examples import _common, run_examples

def forbidden(*args, **kwargs):
    raise AssertionError("Example executed work while being imported")

_common.example_options = forbidden
for cls in (jp.FluidCoolProp, jp.FluidJAX, jp.FluidPerfectGas, jp.FluidBicubic, jp.FluidBinaryMixture):
    cls.__init__ = forbidden
CP.AbstractState = CoolProp.AbstractState = forbidden
plt.subplots = plt.show = plt.savefig = forbidden
before = set(Path.cwd().iterdir())
for entry in run_examples.load_catalogue():
    module = importlib.import_module("examples." + entry["path"][:-3].replace("/", "."))
    if entry["kind"] != "reference":
        assert callable(getattr(module, "main", None)), entry["path"]
assert not plt.get_fignums()
assert set(Path.cwd().iterdir()) == before
print(json.dumps({"imported":65,"evaluations":0,"figures":0,"output_writes":0}))
"""
    environment = os.environ.copy()
    environment.update(
        JAX_ENABLE_X64="true", MPLBACKEND="Agg", PYTHONPATH=str(ROOT), PYTHONUTF8="1"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["imported"] == 65
