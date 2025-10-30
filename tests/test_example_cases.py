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
    # #
    # "basic_calculations/solver_coolprop.py",
    # "basic_calculations/solver_equilibrium.py",
    # "basic_calculations/solver_metastable.py",
    # #
    # "contour_plots/compressibility_factor.py",
    # "contour_plots/pressure_isolines.py",
    #
    # "props_supersaturation/supersaturation_along_isentrope.py",
    # "props_supersaturation/supersaturation_along_isobar.py",
    # "props_supersaturation/supersaturation_along_spinodal.py",
    #
    # "props_undersaturation/demo_calculations.py",
    # "props_undersaturation/quality_along_isolines.py",
    # "props_undersaturation/subcooling_along_isobar.py",
    # "props_undersaturation/superheating_along_isobar.py",
    # #
    "spinodal_lines/spinodal_lines.py",
    "spinodal_lines/spinodal_lines_PT.py",
    # #
    # "blending_calculations/blending_condensing.py",
    # "blending_calculations/blending_flashing.py",
    # #
    # "perfect_gas/demo_perfect_gas_eos.py",
    # "perfect_gas/demo_perfect_gas_gradients.py",
    # "perfect_gas/polytropic_compression.py",
    # #
    # "coolprop_jax/coolprop_broadcast.py",
    # "coolprop_jax/derivative_consistency_1.py",
    # "coolprop_jax/derivative_consistency_2.py",
    # "coolprop_jax/derivative_consistency_3.py",
    # #
    # "bicubic_interpolation/demo_interpolation_midpoints.py",
    # "bicubic_interpolation/demo_solver_consistency.py",
    # "bicubic_interpolation/demo_derivatives_jax.py",
]

# Full absolute paths
EXAMPLE_SCRIPTS = [EXAMPLES_DIR / rel_path for rel_path in example_relative_paths]

# Optional: make test output more readable
example_ids = [p.name for p in EXAMPLE_SCRIPTS]


@pytest.mark.parametrize("script_path", EXAMPLE_SCRIPTS, ids=example_ids)
def test_examples(script_path):
    # Use sys.executable instead of just 'python' to run correctly in GitHub actions (Windows)
    working_dir = script_path.parent
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=working_dir,
        capture_output=True,
        text=True,
        env={**os.environ, "DISABLE_PLOTS": "1"},
    )

    assert result.returncode == 0, f"Failed: {script_path}\n{result.stderr}"


if __name__ == "__main__":

    # Running pytest from this script
    pytest.main([__file__, "-v"])
