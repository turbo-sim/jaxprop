import math
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from functools import partial


from .. import helpers_props as jxp


# output template for pure_callback (shapes/dtypes must be static)

def _make_template(shape):
    """Return a pure_callback template dict with the given output shape."""
    return {k: jax.ShapeDtypeStruct(shape, jnp.float64) for k in jxp.PROPERTIES_CANONICAL}


def _get_props_python(input_type, x, y, fluid):
    """Host-side evaluation. Accepts scalars or arrays and returns dict of ndarrays.
    If the state is None or a property is missing/non-finite, fills with np.nan.
    """
    x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    results = {name: np.empty(x.shape, dtype=np.float64) for name in jxp.PROPERTIES_CANONICAL}

    for i in np.ndindex(x.shape):
        state = fluid.get_state(input_type, float(x[i]), float(y[i]))
        if state is None:
            # Fill all properties with NaN if state is invalid
            for name in jxp.PROPERTIES_CANONICAL:
                results[name][i] = np.nan
        else:
            state_dict = state.to_dict()
            for name in jxp.PROPERTIES_CANONICAL:
                val = state_dict.get(name, np.nan)
                # Convert only if it's a finite number; otherwise force NaN
                try:
                    val_f = np.float64(val)
                    results[name][i] = val_f if np.isfinite(val_f) else np.nan
                except Exception:
                    results[name][i] = np.nan

    return results

@partial(jax.custom_jvp, nondiff_argnums=(0, 3))
def get_state(input_pair, x, y, fluid):
    """
    JAX-callable wrapper around CoolProp returning a dictionary of properties

    Internally calls CoolProp via a host callback, using STATE_TEMPLATE
    for shape/dtype specification. Supports JIT but not direct autodiff.
    A custom JVP rule is registered to provide finite-difference gradients.
    """

    # determine broadcasted output shape at trace time
    shape1 = jnp.shape(x)
    shape2 = jnp.shape(y)
    out_shape = np.broadcast_shapes(shape1, shape2)
    template = _make_template(out_shape)

    def local_eval(p1, p2):
        # returns dict[str, float]; pure_callback will map to jnp scalars using _TEMPLATE
        return _get_props_python(input_pair, p1, p2, fluid)

    return jax.pure_callback(local_eval, template, x, y, vmap_method="broadcast_all")


@get_state.defjvp
def _get_state_jvp(input_state, fluid, primals, tangents):
    """Custom JVP rule for get_props() using finite differences.

    Approximates partial derivatives of each state property with respect
    to the inputs (x, y) by evaluating CoolProp at slightly perturbed
    states, then combines with tangents (x_dot, y_dot).

    Returns
    -------
    base : dict
        State evaluated at the primal inputs.
    jvp : dict
        Tangent state containing directional derivatives.
    """
    x, y = primals
    x_dot, y_dot = tangents

    # shapes and template (same logic as in primal)
    out_shape = np.broadcast_shapes(jnp.shape(x), jnp.shape(y))
    template = _make_template(out_shape)

    # elementwise FD steps (broadcasted)
    eps1 = 1e-6 * (jnp.abs(x) + 1.0)
    eps2 = 1e-6 * (jnp.abs(y) + 1.0)

    def local_eval(xx, yy):
        return _get_props_python(input_state, xx, yy, fluid)

    # base and perturbed evaluations (each returns dict of arrays)
    props_base = jax.pure_callback(local_eval, template, x, y, vmap_method="broadcast_all")
    props_x = jax.pure_callback(local_eval, template, x + eps1, y, vmap_method="broadcast_all")
    props_y = jax.pure_callback(local_eval, template, x, y + eps2, vmap_method="broadcast_all")

    # directional derivatives (broadcasted arithmetic)
    df_dp1 = {k: (props_x[k] - props_base[k]) / eps1 for k in jxp.PROPERTIES_CANONICAL}
    df_dp2 = {k: (props_y[k] - props_base[k]) / eps2 for k in jxp.PROPERTIES_CANONICAL}
    jvp    = {k: df_dp1[k] * x_dot + df_dp2[k] * y_dot for k in jxp.PROPERTIES_CANONICAL}

    return props_base, jvp

class FluidJAX(eqx.Module):
    name: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    fluid: object = eqx.field(static=True)  # raw CoolProp.AbstractState object
    exceptions: bool = eqx.field(static=True)

    def __init__(self, name: str, backend: str="HEOS", exceptions: bool=True):
        from .fluid_properties import Fluid
        self.name = name
        self.backend = backend
        self.exceptions = exceptions
        self.fluid = Fluid(name=name, backend=backend, exceptions=exceptions)

    @eqx.filter_jit
    def get_state(self, input_pair, x, y):
        """Return a CoolPropState with fields shaped like broadcast(x, y)."""
        raw = get_state(input_pair, x, y, self.fluid)
        return jxp.FluidState(**{k: raw[k] for k in jxp.PROPERTIES_CANONICAL})