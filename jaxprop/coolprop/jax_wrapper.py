import math
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from functools import partial
from dataclasses import fields

from .. import helpers_coolprop as jxp


# ----------------------------------------------------------------------------- #
# state container (single source of truth for fields)
# ----------------------------------------------------------------------------- #

# class CoolPropState(eqx.Module):
#     """Container for a thermodynamic state from CoolProp.

#     Fields correspond to common mass-based properties (p, T, rho, u, h, s, …),
#     stored as JAX arrays for compatibility with JIT and autodiff.

#     Provides attribute-style access (state.p) and alias-aware dict-style access
#     (state["P"], state["rhomass"], …).
#     """
#     p: jnp.ndarray
#     T: jnp.ndarray
#     rho: jnp.ndarray
#     u: jnp.ndarray
#     h: jnp.ndarray
#     s: jnp.ndarray
#     a: jnp.ndarray
#     gruneisen: jnp.ndarray
#     mu: jnp.ndarray
#     k: jnp.ndarray
#     cp: jnp.ndarray
#     cv: jnp.ndarray
#     gamma: jnp.ndarray

#     def __getitem__(self, key: str):
#         """Allow dictionary-style access to state variables.
#         Returns the attribute matching `key` or its alias in PROPERTY_ALIAS."""
#         if hasattr(self, key):
#             return getattr(self, key)
#         if key in PROPERTY_ALIAS:
#             return getattr(self, PROPERTY_ALIAS[key])
#         raise KeyError(f"Unknown property alias: {key}")

#     def __repr__(self):
#         """Return a readable string representation of the state,
#         listing all field names and their scalar values."""
#         lines = []
#         for name, val in self.__dict__.items():
#             try:
#                 val = jnp.array(val).item()
#             except Exception:
#                 pass
#             lines.append(f"  {name}={val}")
#         return "CoolPropState(\n" + ",\n".join(lines) + "\n)"


# output template for pure_callback (shapes/dtypes must be static)
_NAMES = [f.name for f in fields(jxp.FluidState)]

def _make_template(shape):
    """Return a pure_callback template dict with the given output shape."""
    return {k: jax.ShapeDtypeStruct(shape, jnp.float64) for k in _NAMES}


# def _get_props_python(input_pair, x, y, fluid):
#     """Host-side evaluation. Accepts scalars or arrays and returns dict of ndarrays."""
#     x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
#     results = {name: np.empty(x.shape, dtype=np.float64) for name in _NAMES}

#     for i in np.ndindex(x.shape):
#         state = fluid.get_state(input_pair, float(x[i]), float(y[i])).to_dict()
#         for name in _NAMES:
#             val = state.get(name)
#             results[name][i] = np.float64(val) if np.isfinite(val) else np.nan

#     return results


def _get_props_python(input_pair, x, y, fluid):
    """Host-side evaluation. Accepts scalars or arrays and returns dict of ndarrays."""
    x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    results = {name: np.empty(x.shape, dtype=np.float64) for name in _NAMES}

    for i in np.ndindex(x.shape):
        state = fluid.get_state(input_pair, float(x[i]), float(y[i])).to_dict()

        for name in _NAMES:
            val = state.get(name, None)

            if val is None:
                results[name][i] = np.nan
                continue

            # convert jax/numpy arrays to scalars if possible
            if isinstance(val, (jnp.ndarray, np.ndarray)):
                try:
                    val = val.item()
                except Exception:
                    val = float(val.ravel()[0])

            # convert booleans to {0.0, 1.0}
            if isinstance(val, (bool, np.bool_)):
                results[name][i] = 1.0 if val else 0.0
                continue

            try:
                v = float(val)
                results[name][i] = v if math.isfinite(v) else np.nan
            except Exception:
                results[name][i] = np.nan

    return results


@partial(jax.custom_jvp, nondiff_argnums=(0, 3))
def get_props(input_pair, x, y, fluid):
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


@get_props.defjvp
def _get_props_jvp(input_state, fluid, primals, tangents):
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
    df_dp1 = {k: (props_x[k] - props_base[k]) / eps1 for k in _NAMES}
    df_dp2 = {k: (props_y[k] - props_base[k]) / eps2 for k in _NAMES}
    jvp    = {k: df_dp1[k] * x_dot + df_dp2[k] * y_dot for k in _NAMES}

    return props_base, jvp

class FluidJAX(eqx.Module):
    name: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    fluid: object = eqx.field(static=True)  # raw CoolProp.AbstractState object

    def __init__(self, name: str, backend: str = "HEOS"):
        from .fluid_properties import Fluid
        self.name = name
        self.backend = backend
        self.fluid = Fluid(name=name, backend=backend)

    def get_props(self, input_pair, x, y):
        """Return a CoolPropState with fields shaped like broadcast(x, y)."""
        raw = get_props(input_pair, x, y, self.fluid)
        return jxp.FluidState(**{k: raw[k] for k in _NAMES})