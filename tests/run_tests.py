#!/usr/bin/env python3
import pytest

# Define the list of tests
tests_list = ["test_example_cases.py", "test_default_solver.py", "test_phase_diagrams.py"]
# tests_list = ["test_example_cases.py"]

# Run pytest when the python script is executed
# pytest.main(tests_list)
pytest.main(tests_list + ["-v"])
# pytest.main([__file__, "-vv"])
# pytest.main([__file__])



# TODO it would be good to use TOX or NOX to test my installation across multiple python versions.
# Maybe it is even better to do so through github actions


# I have to do one function to check spinodal calculation without errors and near zero bulk modulus
# I have to do tests for the superheating and supersaturation functions?

