"""Compare scalar Optimistix solvers on cos(pi*x/2), whose root is x=1."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx


def f(x, args):
    return jnp.cos(0.5 * jnp.pi * x)


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    initial = jnp.array(0.5)
    solvers = {
        "Bisection": optx.Bisection(rtol=1e-6, atol=1e-6),
        "Newton": optx.Newton(rtol=1e-6, atol=1e-6),
        "Chord": optx.Chord(rtol=1e-6, atol=1e-6),
    }
    results = {}
    for name, solver in solvers.items():
        solution = optx.root_find(
            f,
            solver,
            initial,
            args=None,
            throw=False,
            options={"lower": 0.0, "upper": 2.0},
        )
        jax.block_until_ready(solution.value)
        residual = float(f(solution.value, None))
        assert solution.result == optx.RESULTS.successful
        np.testing.assert_allclose(solution.value, 1.0, rtol=0, atol=3e-6)
        assert abs(residual) < 5e-6
        print(
            f"{name}: steps={solution.stats['num_steps']}, root={solution.value:.10e}, residual={residual:.3e}"
        )
        results[name] = {
            "root": float(solution.value),
            "residual": residual,
            "steps": int(solution.stats["num_steps"]),
            "successful": True,
        }
    options.record("scientific_checks", results)
    options.finish()
    return results


if __name__ == "__main__":
    main()
