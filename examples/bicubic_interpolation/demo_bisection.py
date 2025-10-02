import jax.numpy as jnp
from jax import lax

def bisection_root_scalar(func, lower, upper, tol=1e-8, max_iter=100):
    """
    Simple scalar root finder using bisection method in JAX.
    
    Parameters:
        func: function f(x), must be scalar-valued and monotonic over [lower, upper]
        lower: lower bound of root bracket
        upper: upper bound of root bracket
        tol: convergence tolerance
        max_iter: maximum number of iterations

    Returns:
        x_root: root such that f(x_root) ≈ 0
    """
    def cond_fun(state):
        lower, upper, mid, f_mid, i = state
        return jnp.logical_and(jnp.abs(f_mid) > tol, i < max_iter)

    def body_fun(state):
        lower, upper, mid, f_mid, i = state
        mid = 0.5 * (lower + upper)
        f_mid = func(mid)
        f_low = func(lower)

        # Check sign of f at mid and lower to determine new interval
        update_lower = jnp.sign(f_mid) == jnp.sign(f_low)
        new_lower = jnp.where(update_lower, mid, lower)
        new_upper = jnp.where(update_lower, upper, mid)

        return new_lower, new_upper, mid, f_mid, i + 1

    f_init = func(0.5 * (lower + upper))
    init_state = (lower, upper, 0.5 * (lower + upper), f_init, 0)

    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    _, _, x_root, _, _ = final_state
    return x_root
