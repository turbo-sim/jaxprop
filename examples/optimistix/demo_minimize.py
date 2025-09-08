import jax.numpy as jnp
import optimistix as optx

# Define the Rosenbrock function
def rosenbrock(x, args):
    # x is a vector [x0, x1]
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

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
    sol = optx.minimise(
        rosenbrock,
        solver,
        x0,
        max_steps=1000,
        args=None,
        throw=False,
    )

    print(f"Success: {sol.result == optx.RESULTS.successful}")
    print(f"Steps: {sol.stats['num_steps']}")
    print(f"Solution: {sol.value}")
    print(f"Objective value: {rosenbrock(sol.value, None):0.6e}")
