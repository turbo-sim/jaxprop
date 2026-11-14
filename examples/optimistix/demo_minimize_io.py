"""Count objective/gradient evaluations in the original Rosenbrock example.

These ordered host callbacks are deliberate instrumentation, not a numerical
performance benchmark. Synchronize their effects before reporting counters.
The original solver tolerances and 500-step limits are retained.
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

_eval_counters = {"f_calls": 0, "g_calls": 0}


def reset_counters():
    _eval_counters.update(f_calls=0, g_calls=0)


def get_counters():
    return dict(_eval_counters)


def _make_counter(kind):
    def callback(value):
        _eval_counters[kind] += 1

    return callback


def with_counters(function):
    """Count both plain calls and custom-VJP calls exactly once each."""

    def evaluate(x, args):
        jax.experimental.io_callback(_make_counter("f_calls"), None, x, ordered=True)
        return function(x, args)

    @jax.custom_vjp
    def wrapped(x, args):
        return evaluate(x, args)

    def forward(x, args):
        return evaluate(x, args), (x, args)

    def backward(residuals, cotangent):
        x, args = residuals
        jax.experimental.io_callback(_make_counter("g_calls"), None, x, ordered=True)
        return cotangent * jax.grad(function)(x, args), None

    wrapped.defvjp(forward, backward)
    return wrapped


def rosenbrock(x, args):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    initial = jnp.array([-1.2, 1.0])
    objective = with_counters(rosenbrock)
    solvers = {
        "BFGS": optx.BFGS(rtol=1e-6, atol=1e-6),
        "Conjugate Gradient": optx.NonlinearCG(rtol=1e-6, atol=1e-6),
    }
    results = {}
    for name, solver in solvers.items():
        reset_counters()
        solution = optx.minimise(
            objective, solver, initial, max_steps=500, args=None, throw=False
        )
        jax.block_until_ready(solution.value)
        jax.effects_barrier()
        counts = get_counters()
        loss = float(rosenbrock(solution.value, None))
        assert solution.result == optx.RESULTS.successful
        assert np.isfinite(loss) and loss <= 2e-6
        np.testing.assert_allclose(solution.value, [1, 1], rtol=0, atol=3e-3)
        assert counts["f_calls"] >= counts["g_calls"] > 0
        print(
            f"{name}: minimizer={solution.value}, objective={loss:.8e}, steps={solution.stats['num_steps']}, counters={counts}"
        )
        results[name] = {
            **counts,
            "objective": loss,
            "minimizer": np.asarray(solution.value).tolist(),
            "steps": int(solution.stats["num_steps"]),
            "successful": True,
        }
    options.record("scientific_checks", results)
    options.record("instrumentation_is_benchmark", False)
    options.finish()
    return results


if __name__ == "__main__":
    main()
