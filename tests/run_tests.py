#!/usr/bin/env python3
import pytest

# Define the list of tests
tests_list = [
    # "test_example_cases.py",
    "test_property_solvers.py",
    # "test_helmholtz_evaluation.py",
    # "test_phase_diagrams.py",
]

# Run pytest when this python script is executed
# pytest.main(tests_list)
pytest.main(tests_list + ["-v"])



# TODO it would be good to use TOX or NOX to test my installation across multiple python versions.
# Maybe it is even better to do so through github actions


# I have to do one function to check spinodal calculation without errors and near zero bulk modulus
# I have to do tests for the superheating and supersaturation functions?
