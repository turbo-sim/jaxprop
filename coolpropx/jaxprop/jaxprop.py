from functools import partial
import numpy as np

from ..fluid_properties import INPUT_PAIR_MAP
from ..jax_import import jax, jnp, JAX_AVAILABLE


def _sanitize(props, sentinel=-1.0):
    """
    Convert a dict of CoolProp FluidState values to JAX-compatible scalars.

    Parameters
    ----------
    props : dict
        Dictionary of fluid properties, typically from `FluidState.to_dict()`.
    sentinel : float, optional
        Value used to replace non-numeric or invalid entries. Default is -1.0.

    Returns
    -------
    dict
        Dictionary with the same keys, but all values converted to
        `jnp.float64` scalars. Any invalid values are replaced by `sentinel`.
    """
    out = {}
    for k, v in props.items():
        if not isinstance(v, (float, int)) or not np.isfinite(v):
            out[k] = jnp.asarray(sentinel, dtype=jnp.float64)
        else:
            out[k] = jnp.asarray(float(v), dtype=jnp.float64)
    return out



if JAX_AVAILABLE:
    @partial(jax.custom_jvp, nondiff_argnums=(0, 3))
    def get_props(input_state, prop1, prop2, fluid):
        """CoolProp-backed properties; JAX-differentiable w.r.t. (prop1, prop2)."""
        state_dict = fluid.get_state(input_state, prop1, prop2).to_dict()
        return _sanitize(state_dict)

    @get_props.defjvp
    def _get_props_jvp(input_state, fluid, primals, tangents):
        """
        Custom JVP rule for get_props, using finite differences for derivatives.

        This approach sanitizes all states first so that derivative computations
        do not need to handle NaN, None, strings, or bools.
        """
        p1, p2 = primals
        p1_dot, p2_dot = tangents

        delta_p1 = 1e-6 * p1
        delta_p2 = 1e-6 * p2

        # Sanitize all states right after retrieval
        base_state = _sanitize(fluid.get_state(input_state, p1, p2).to_dict())
        state_p1   = _sanitize(fluid.get_state(input_state, p1 + delta_p1, p2).to_dict())
        state_p2   = _sanitize(fluid.get_state(input_state, p1, p2 + delta_p2).to_dict())

        # Compute partial derivatives (safe because all bad values are replaced)
        df_dprop1 = {k: (state_p1[k] - base_state[k]) / delta_p1 for k in base_state}
        df_dprop2 = {k: (state_p2[k] - base_state[k]) / delta_p2 for k in base_state}

        # Directional derivative
        jvp = {k: df_dprop1[k] * p1_dot + df_dprop2[k] * p2_dot for k in base_state}

        return base_state, jvp


    # It seems that the nice custom derivative with alpha will not work because of how JAX tracing works
    # p1_dot and p2_dot are tracers, not numerical values, and therefore cannot be passed to coolprop

    # @get_props.defjvp
    # def _get_props_jvp(input_state, fluid, primals, tangents):
    #     """
    #     Custom JVP (forward-mode) using a directional, scale-aware finite difference:

    #         alpha = sqrt( (p1_dot/d1)^2 + (p2_dot/d2)^2 )   with d1,d2 ~ reference scales
    #         step  = (p1_dot/alpha, p2_dot/alpha)
    #         jvp   = alpha * (f(p + step) - f(p))

    #     This keeps the step aligned with the tangent, with size set by the
    #     reference magnitudes to avoid poor conditioning.
    #     """
    #     p1, p2 = primals
    #     p1_dot, p2_dot = tangents

    #     # Try to pull per-variable scales from reference_state, fallback to |p| or 1.0
    #     name_1, name_2 = INPUT_PAIR_MAP[input_state]
    #     delta_p1 = 1e-6 * fluid.reference_state[name_1]
    #     delta_p2 = 1e-6 * fluid.reference_state[name_2]
    #     alpha = jnp.sqrt((p1_dot/delta_p1)**2 + (p2_dot/delta_p2)**2)

    #     # Base and "plus" states (sanitize once so diffs are safe)
    #     p1f = p1 + (p1_dot / alpha)
    #     p2f = p2 + (p2_dot / alpha)
    #     base_state = _sanitize(fluid.get_state(input_state, p1, p2).to_dict())
    #     plus_state = _sanitize(fluid.get_state(input_state, p1f, p2f).to_dict())

    #     # Directional 
    #     jvp = {k: alpha * (plus_state[k] - base_state[k]) for k in base_state}

    #     return base_state, jvp

