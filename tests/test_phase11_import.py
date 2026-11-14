"""Importing the library respects application-owned JAX and exception settings."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def import_probe(setup="", **settings):
    environment = os.environ.copy()
    for name in ("JAX_ENABLE_X64", "JAX_PLATFORM_NAME", "JAX_PLATFORMS"):
        environment.pop(name, None)
    environment.update(settings)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            setup + "\n" + """
import json, os, sys
previous_hook = sys.excepthook
import jaxprop
import jax
print(json.dumps({
    'x64': bool(jax.config.jax_enable_x64),
    'platform_environment': os.environ.get('JAX_PLATFORM_NAME'),
    'platforms_environment': os.environ.get('JAX_PLATFORMS'),
    'configured_platform': jax.config.read('jax_platform_name'),
    'same_exception_hook': sys.excepthook is previous_hook,
}))
""",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_fresh_import_keeps_scientific_double_precision_default():
    result = import_probe()
    assert result["x64"]
    assert result["platform_environment"] is None
    assert result["platforms_environment"] is None
    assert result["same_exception_hook"]


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_environment_precision_takes_priority(enabled):
    result = import_probe(JAX_ENABLE_X64=str(enabled).lower())
    assert result["x64"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_preimported_jax_precision_is_preserved(enabled):
    result = import_probe(
        f"import jax\njax.config.update('jax_enable_x64', {enabled!r})"
    )
    assert result["x64"] is enabled


def test_platform_environment_and_custom_exception_hook_are_preserved():
    # No accelerator is required to check that the consumer's preference is
    # retained. No device is requested by this import-only probe.
    result = import_probe(
        "import IPython.core.ultratb\nimport sys\nsys.excepthook = lambda *args: None",
        JAX_PLATFORM_NAME="gpu",
        JAX_PLATFORMS="cpu",
    )
    assert result["platform_environment"] == "gpu"
    assert result["platforms_environment"] == "cpu"
    assert result["same_exception_hook"]


def test_programmatic_platform_selection_is_preserved():
    result = import_probe("import jax\njax.config.update('jax_platform_name', 'cpu')")
    assert result["configured_platform"] == "cpu"
    assert result["platform_environment"] is None
