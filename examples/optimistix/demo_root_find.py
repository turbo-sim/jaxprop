import jax.numpy as jnp
import optimistix as optx

# Define the function cos(πx/2) with root at x=1
def f(x, args):
    return jnp.cos(0.5 * jnp.pi * x)

# Initial guess
x0 = jnp.array(0.5)

# Define solvers
solvers = {
    "Bisection": optx.Bisection(rtol=1e-6, atol=1e-6),
    "Newton": optx.Newton(rtol=1e-6, atol=1e-6),
    "Chord": optx.Chord(rtol=1e-6, atol=1e-6),
}

# Run loop
for name, solver in solvers.items():
    print(f"\n=== {name} solver ===")
    sol = optx.root_find(
        f,
        solver,
        x0,
        args=None,
        throw=False,
        options={"lower": 0.0, "upper": 2.0},
    )

    print(f"Success: {sol.result == optx.RESULTS.successful}")
    print(f"Steps: {sol.stats["num_steps"]}")
    print(f"Root: {sol.value:0.6e}")
    print(f"Residual: {f(sol.value, None):0.6e}")
    

