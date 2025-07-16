import os
import sys
import pytest
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

# Base path relative to this script's location
# Robust regardless from where the script is executed
THIS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = THIS_DIR.parent / "examples"

# Manually define relative paths from examples/ root
example_relative_paths = [
    "basic_calculations.py",
    "solver_metastable.py",
    "solver_equilibrium.py",
    "subcoling_superheating.py",
    "contour_pressure.py",
    "contour_compressibility.py",
    "generalized_saturation_properties/quality.py",
    "generalized_saturation_properties/subcooling.py",
    "generalized_saturation_properties/superheating.py",
    "spinodal_lines/spinodal_lines.py",
    "spinodal_lines/spinodal_lines_PT.py",
]

# Full absolute paths
EXAMPLE_SCRIPTS = [EXAMPLES_DIR / rel_path for rel_path in example_relative_paths]

# Optional: make test output more readable
example_ids = [p.name for p in EXAMPLE_SCRIPTS]

@pytest.mark.parametrize("script_path", EXAMPLE_SCRIPTS, ids=example_ids)
def test_examples(script_path):
    # Use sys.executable instead of just 'python' to run correctly in GitHub actions (Windows)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "DISABLE_PLOTS": "1"}
    )

    assert result.returncode == 0, f"Failed: {script_path}\n{result.stderr}"