import jax
import jax.numpy as jnp
import optimistix as optx

# Global counters (Python side)
_eval_counters = {"f_calls": 0, "g_calls": 0}

def reset_counters():
    _eval_counters["f_calls"] = 0
    _eval_counters["g_calls"] = 0

def get_counters():
    return dict(_eval_counters)

# Helper to increment counters from inside JAX
def _make_counter(kind):
    def _cb(x):
        _eval_counters[kind] += 1
    return _cb

def with_counters(fun):
    """Wrap a scalar objective with counters and custom VJP."""
    @jax.custom_vjp
    def wrapped(x, args):
        return fun(x, args)

    def fwd(x, args):
        # Count every function evaluation
        jax.experimental.io_callback(_make_counter("f_calls"), None, x)
        y = fun(x, args)
        return y, (x, args)

    def bwd(residuals, g):
        x, args = residuals
        # Count every gradient evaluation
        jax.experimental.io_callback(_make_counter("g_calls"), None, x)
        grad = jax.grad(fun)(x, args)
        return (g * grad, None)

    wrapped.defvjp(fwd, bwd)
    return wrapped

# Example: Rosenbrock
def rosenbrock(x, args):
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

rosenbrock_logged = with_counters(rosenbrock)

# Initial guess
x0 = jnp.array([-1.2, 1.0])

# Define solvers
solvers = {
    "BFGS": optx.BFGS(rtol=1e-6, atol=1e-6),
    "Conjugate Gradient": optx.NonlinearCG(rtol=1e-6, atol=1e-6),
}

# Run loop
for name, solver in solvers.items():
    print(f"\n=== {name} solver ===")
    reset_counters()  # reset before each run

    sol = optx.minimise(
        rosenbrock_logged,
        solver,
        x0,
        max_steps=500,
        args=None,
        throw=False,
    )

    counts = get_counters()
    print(f"Success: {sol.result == optx.RESULTS.successful}")
    print(f"Minimizer: {sol.value}")
    print(f"Steps: {sol.stats['num_steps']}")
    print(f"Minimum value: {rosenbrock(sol.value, None):0.6e}")
    print(f"Function calls: {counts['f_calls']}")
    print(f"Gradient calls: {counts['g_calls']}")
    print(f"Function-only calls: {counts['f_calls'] - counts['g_calls']}")
