"""Compare Optimistix minimizers on the original Rosenbrock problem.

Both retain rtol=atol=1e-6 and the 1,000-step budget. A step-based stopping
criterion does not promise six decimal places in the minimizer of an
ill-conditioned valley; independently check objective reduction and position.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx


def rosenbrock(x, args):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    initial = jnp.array([-1.2, 1.0])
    solvers = {
        "BFGS": optx.BFGS(rtol=1e-6, atol=1e-6),
        "Conjugate Gradient": optx.NonlinearCG(rtol=1e-6, atol=1e-6),
    }
    results = {}
    for name, solver in solvers.items():
        solution = optx.minimise(
            rosenbrock, solver, initial, max_steps=1000, args=None, throw=False
        )
        jax.block_until_ready(solution.value)
        loss = float(rosenbrock(solution.value, None))
        assert solution.result == optx.RESULTS.successful
        assert np.isfinite(loss) and loss <= 2e-6
        np.testing.assert_allclose(solution.value, [1, 1], rtol=0, atol=3e-3)
        print(
            f"{name}: steps={solution.stats['num_steps']}, minimizer={solution.value}, objective={loss:.8e}"
        )
        results[name] = {
            "minimizer": np.asarray(solution.value).tolist(),
            "objective": loss,
            "steps": int(solution.stats["num_steps"]),
            "successful": True,
        }
    options.record("scientific_checks", results)
    options.finish()
    return results


if __name__ == "__main__":
    main()
