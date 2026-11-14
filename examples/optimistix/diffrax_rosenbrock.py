"""Inspect an Optimistix BFGS line search on Rosenbrock's objective.

The historical filename mentions Diffrax; this program demonstrates Optimistix
optimization and its solver-specific state. Keep the original 5,000-step budget,
1e-8 tolerances and Armijo settings. Report the minimizer and scalar objective
under their correct names.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx


def rosenbrock(x, args):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    solver = optx.BFGS(rtol=1e-8, atol=1e-8)
    solver = eqx.tree_at(
        lambda item: item.search,
        solver,
        optx.BacktrackingArmijo(step_init=0.5, slope=1e-8),
    )
    solution = optx.minimise(
        rosenbrock, solver, jnp.array([-1.2, 1.0]), args=None, max_steps=5000
    )
    jax.block_until_ready(solution.value)
    objective = float(rosenbrock(solution.value, None))
    assert solution.result == optx.RESULTS.successful and objective < 1e-12
    np.testing.assert_allclose(solution.value, [1, 1], rtol=0, atol=1e-6)
    state = solution.state
    print(f"Minimizer: {solution.value}")
    print(f"Final scalar objective: {objective:.10e}")
    print(
        f"Steps: {solution.stats['num_steps']}; accepted steps: {state.num_accepted_steps}"
    )
    print(f"Last line-search step: {state.search_state.step_size}")
    print(f"Last evaluated point: {state.y_eval}")
    print(f"Last gradient: {state.f_info.grad}")
    print(f"Inverse Hessian:\n{state.f_info.hessian_inv.pytree}")
    options.record(
        "scientific_checks",
        {
            "minimizer": np.asarray(solution.value).tolist(),
            "objective": objective,
            "steps": int(solution.stats["num_steps"]),
            "accepted_steps": int(state.num_accepted_steps),
            "successful": True,
        },
    )
    options.finish()
    return solution


if __name__ == "__main__":
    main()
