"""Discovery, process status, and explicit optional-backend selection."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import conftest as backend_configuration
import run_tests
from utilities import get_available_backends


TESTS_DIR = Path(__file__).resolve().parent


def test_runner_import_does_not_execute_tests(monkeypatch):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("Importing the runner executed pytest")

    monkeypatch.setattr(pytest, "main", unexpected_run)
    specification = importlib.util.spec_from_file_location(
        "imported_test_runner", TESTS_DIR / "run_tests.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    assert callable(module.main)


def test_runner_forwards_arguments_and_failed_exit_status(monkeypatch, tmp_path):
    observed = []

    def failed_run(arguments):
        observed.extend(arguments)
        return pytest.ExitCode.TESTS_FAILED

    monkeypatch.setattr(pytest, "main", failed_run)
    monkeypatch.chdir(tmp_path)
    assert run_tests.main(["-k", "fluid_state", "--strict-markers"]) == 1
    assert Path(observed[0]) == TESTS_DIR
    assert observed[1:3] == ["-c", str(TESTS_DIR / "pytest.ini")]
    assert observed[3:] == ["-k", "fluid_state", "--strict-markers"]


def test_runner_collects_state_tests_from_another_directory(tmp_path):
    environment = dict(os.environ, JAXPROP_TEST_BACKENDS="HEOS", MPLBACKEND="Agg")
    result = subprocess.run(
        [
            sys.executable,
            str(TESTS_DIR / "run_tests.py"),
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-k",
            "test_scalar_states_combine_reverse_and_extract_without_mutating_inputs",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "test_fluid_state.py::"
        "test_scalar_states_combine_reverse_and_extract_without_mutating_inputs"
    ) in result.stdout


def test_runner_cli_propagates_pytest_usage_error(tmp_path):
    result = subprocess.run(
        [sys.executable, str(TESTS_DIR / "run_tests.py"), "--unknown-runner-option"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "unrecognized arguments" in result.stderr


def test_default_backend_selection_is_heos(monkeypatch):
    monkeypatch.delenv("JAXPROP_TEST_BACKENDS", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    assert get_available_backends() == ["HEOS"]


def test_optional_backend_selection_is_explicit_and_normalized(monkeypatch):
    monkeypatch.setenv("JAXPROP_TEST_BACKENDS", " heos,refprop, HEOS ")
    assert get_available_backends() == ["HEOS", "REFPROP"]


@pytest.mark.parametrize("selection", ["", "HEOS,", "UNKNOWN", "HEOS,UNKNOWN"])
def test_invalid_backend_selection_is_rejected(monkeypatch, selection):
    monkeypatch.setenv("JAXPROP_TEST_BACKENDS", selection)
    with pytest.raises(ValueError, match="JAXPROP_TEST_BACKENDS"):
        get_available_backends()


def test_refprop_availability_probe_is_cached(monkeypatch):
    calls = []

    def available_state(backend, fluid):
        calls.append((backend, fluid))
        return object()

    backend_configuration._refprop_unavailability_reason.cache_clear()
    monkeypatch.setattr(backend_configuration.CP, "AbstractState", available_state)
    try:
        assert backend_configuration._refprop_unavailability_reason() is None
        assert backend_configuration._refprop_unavailability_reason() is None
        assert calls == [("REFPROP", "Water")]
    finally:
        backend_configuration._refprop_unavailability_reason.cache_clear()


@pytest.mark.parametrize("error_type", [ValueError, OSError])
def test_refprop_availability_probe_explains_missing_library(monkeypatch, error_type):
    def unavailable_state(backend, fluid):
        raise error_type("Missing optional REFPROP library.\nAdditional detail.")

    backend_configuration._refprop_unavailability_reason.cache_clear()
    monkeypatch.setattr(backend_configuration.CP, "AbstractState", unavailable_state)
    try:
        reason = backend_configuration._refprop_unavailability_reason()
        assert reason == (
            "Requested REFPROP backend is unavailable. Missing optional REFPROP library."
        )
    finally:
        backend_configuration._refprop_unavailability_reason.cache_clear()


def test_missing_refprop_marks_only_requested_refprop_cases(monkeypatch):
    heos = SimpleNamespace(callspec=SimpleNamespace(params={"backend": "HEOS"}))
    refprop = SimpleNamespace(callspec=SimpleNamespace(params={"backend": "REFPROP"}))
    marks = []
    heos.add_marker = lambda marker: pytest.fail("HEOS was skipped")
    refprop.add_marker = marks.append
    monkeypatch.setattr(
        backend_configuration,
        "_refprop_unavailability_reason",
        lambda: "Requested REFPROP backend is unavailable. Missing test library.",
    )
    backend_configuration.pytest_collection_modifyitems(None, [heos, refprop])
    assert len(marks) == 1
    assert marks[0].name == "skip"
    assert "Missing test library" in marks[0].kwargs["reason"]


def test_heos_collection_does_not_probe_refprop(monkeypatch):
    def unexpected_probe():
        raise AssertionError("HEOS-only collection probed REFPROP")

    monkeypatch.setattr(
        backend_configuration, "_refprop_unavailability_reason", unexpected_probe
    )
    heos = SimpleNamespace(callspec=SimpleNamespace(params={"backend": "HEOS"}))
    backend_configuration.pytest_collection_modifyitems(None, [heos])
