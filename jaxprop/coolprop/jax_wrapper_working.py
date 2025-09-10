import jax
import jax.numpy as jnp
import equinox as eqx
import math

from typing import Any
from functools import partial
from dataclasses import fields

# aliases, to mirror perfect-gas behavior
PROPERTY_ALIAS = {
    "P": "p",
    "pressure": "p",
    "density": "rho",
    "d": "rho",
    "rhomass": "rho",
    "dmass": "rho",
    "hmass": "h",
    "smass": "s",
    "speed_sound": "a",
    "viscosity": "mu",
    "conductivity": "k",
}


# ----------------------------------------------------------------------------- #
# state container (single source of truth for fields)
# ----------------------------------------------------------------------------- #


class CoolPropState(eqx.Module):
    """Container for a thermodynamic state from CoolProp.

    Fields correspond to common mass-based properties (p, T, rho, u, h, s, …),
    stored as JAX arrays for compatibility with JIT and autodiff.

    Provides attribute-style access (state.p) and alias-aware dict-style access
    (state["P"], state["rhomass"], …).
    """

    p: jnp.ndarray
    T: jnp.ndarray
    rho: jnp.ndarray
    u: jnp.ndarray
    h: jnp.ndarray
    s: jnp.ndarray
    a: jnp.ndarray
    gruneisen: jnp.ndarray
    mu: jnp.ndarray
    k: jnp.ndarray
    cp: jnp.ndarray
    cv: jnp.ndarray
    gamma: jnp.ndarray

    def __getitem__(self, key: str):
        """Allow dictionary-style access to state variables.
        Returns the attribute matching `key` or its alias in PROPERTY_ALIAS."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in PROPERTY_ALIAS:
            return getattr(self, PROPERTY_ALIAS[key])
        raise KeyError(f"Unknown property alias: {key}")

    def __repr__(self):
        """Return a readable string representation of the state,
        listing all field names and their scalar values."""
        lines = []
        for name, val in self.__dict__.items():
            try:
                val = jnp.array(val).item()
            except Exception:
                pass
            lines.append(f"  {name}={val}")
        return "CoolPropState(\n" + ",\n".join(lines) + "\n)"


# output template for pure_callback (shapes/dtypes must be static)
_NAMES = [f.name for f in fields(CoolPropState)]
_TEMPLATE = {k: jax.ShapeDtypeStruct((), jnp.float64) for k in _NAMES}


def _to_scalar64(x) -> float:
    """robust cast to finite float; otherwise NaN"""
    try:
        xf = float(x)
        return xf if math.isfinite(xf) else float("nan")
    except Exception:
        return float("nan")


def _get_props_python(input_state, x, y, fluid):
    """Runs on host (Python). Returns plain floats."""
    x_float = float(jnp.squeeze(x))
    y_float = float(jnp.squeeze(y))
    st = fluid.get_state(input_state, x_float, y_float).to_dict()
    return {k: _to_scalar64(st.get(k)) for k in _NAMES}


@partial(jax.custom_jvp, nondiff_argnums=(0, 3))
def get_props(input_state, prop1, prop2, fluid):
    """
    JAX-callable wrapper around CoolProp returning a dictionary of properties

    Internally calls CoolProp via a host callback, using STATE_TEMPLATE
    for shape/dtype specification. Supports JIT but not direct autodiff.
    A custom JVP rule is registered to provide finite-difference gradients.
    """

    def local_eval(p1, p2):
        # returns dict[str, float]; pure_callback will map to jnp scalars using _TEMPLATE
        return _get_props_python(input_state, p1, p2, fluid)

    return jax.pure_callback(local_eval, _TEMPLATE, prop1, prop2)


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

    # relative step with floor for stability
    eps1 = 1e-6 * (jnp.abs(x) + 1.0)
    eps2 = 1e-6 * (jnp.abs(y) + 1.0)

    # single generic python callback; no tracers captured
    def local_eval(xx, yy):
        return _get_props_python(input_state, xx, yy, fluid)

    # primal at (p1, p2)
    props_base = jax.pure_callback(local_eval, _TEMPLATE, x, y)

    # forward-diff samples: pass the already-perturbed args to pure_callback
    props_x = jax.pure_callback(local_eval, _TEMPLATE, x + eps1, y)
    props_y = jax.pure_callback(local_eval, _TEMPLATE, x, y + eps2)

    # directional derivative
    df_dp1 = {k: (props_x[k] - props_base[k]) / eps1 for k in _NAMES}
    df_dp2 = {k: (props_y[k] - props_base[k]) / eps2 for k in _NAMES}
    jvp = {k: df_dp1[k] * x_dot + df_dp2[k] * y_dot for k in _NAMES}

    return props_base, jvp


class FluidJAX(eqx.Module):
    name: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    fluid: object = eqx.field(static=True)  # raw CoolProp.AbstractState object

    def __init__(self, name: str, backend: str = "HEOS"):
        from .fluid_properties import Fluid

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "fluid", Fluid(name=name, backend=backend))

    def get_props(self, input_pair, x, y):
        """Return a CoolPropState with a flexible set of props, JAX-compatible."""

        raw = get_props(input_pair, x, y, self.fluid)
        # return CoolPropState(raw)

        # raw = get_props(input_pair, x, y, self.fluid)
        # # wrap jnp arrays into a CoolPropState
        return CoolPropState(**{k: raw[k] for k in _NAMES})
