#!/usr/bin/env python3
import pytest

# Define the list of tests
tests_list = [
    "test_bicubic_accuracy.py",
    "test_bicubic_consistency.py",
    # "test_example_cases.py",
    "test_helmholtz_eos.py",
    "test_perfect_gas_eos.py",
    # "test_phase_diagrams.py",
    "test_solvers_coolprop.py",
    "test_solvers_equilibrium.py",
    "test_solvers_metastable.py",
    "test_spinodal_calculation.py",
]

# Run pytest when this python script is executed
# pytest.main(tests_list)
# pytest.main(tests_list + [""])
pytest.main(tests_list + ["-v"])


