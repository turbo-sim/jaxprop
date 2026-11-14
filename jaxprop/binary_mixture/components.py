"""Selected native component equations for binary inverses.

These helpers are private: users obtain complete states from configured fluid
objects. Inverse iterations need only one response, not two complete states.
Source-backed host transactions live separately in :mod:`.host`.
"""

import jax.numpy as jnp

from .. import helpers_props as pairs
from ..fluid_api import StatusCode


def component_temperature_bounds(component, pressure):
    """Return the component's actual PT interval at the supplied pressure."""
    if component.provider == "native" and component.model == "perfect_gas":
        from ..perfect_gas.perfect_gas import (
            MIN_DENSITY,
            MIN_TEMPERATURE,
            specific_heat,
        )
        from ..perfect_gas.request_policy import _temperature_limit

        constants = component.constants
        cp, cv = specific_heat(constants)
        numerical = jnp.finfo(jnp.result_type(pressure, constants.R))
        upper = _temperature_limit(constants, cp, cv, numerical)
        # Only form p/(R*rho_min) when it is the active bound. Otherwise that
        # discarded quantity can overflow and poison model/input derivatives.
        # upper*R is representable because upper already enforces R*T limits.
        density_active = pressure < (upper * constants.R) * MIN_DENSITY
        safe_pressure = jnp.where(density_active, pressure, 0.0)
        safe_R = jnp.where(density_active, constants.R, 1.0)
        density_limit = (safe_pressure / safe_R) / MIN_DENSITY
        lower = jnp.asarray(
            MIN_TEMPERATURE, dtype=jnp.result_type(pressure, constants.R, cp, cv)
        )
        return lower, jnp.where(density_active, density_limit, upper)
    if component.provider == "bicubic":
        from ..bicubic.numerics import interpolate_property

        low = interpolate_property(component, component.h_min, pressure, "temperature")
        high = interpolate_property(component, component.h_max, pressure, "temperature")
        return low, high
    raise TypeError("Native binary equations require perfect-gas or bicubic components")


def temperature_bounds(fluid, pressure):
    """Intersect active component intervals; zero-weight domains do not restrict it."""
    lower, upper = None, None
    for component, weight in zip(fluid.components, fluid.mass_fractions):
        if weight == 0:
            continue
        low, high = component_temperature_bounds(component, pressure)
        lower = low if lower is None else jnp.maximum(lower, low)
        upper = high if upper is None else jnp.minimum(upper, high)
    return lower, upper


def inverse_monotone(fluid, pressure, name):
    """Prove a connected increasing response before projecting inverse targets.

    PG heat capacity is setup-validated positive. For a table, inspect the exact
    cubic derivative extrema along its fixed-pressure h slice; corner sampling
    alone cannot establish monotonicity. Invalid phase cells invalidate proof.
    """
    from ..bicubic.numerics import _monotone_target_bounds, interpolate_property

    monotone = jnp.asarray(True)
    for component, weight in zip(fluid.components, fluid.mass_fractions):
        if weight == 0 or component.provider == "native":
            continue
        for field in (
            ("temperature", "entropy") if name == "entropy" else ("temperature",)
        ):
            _, _, proven = _monotone_target_bounds(
                component, pressure, "fixed_pressure", field
            )
            low = interpolate_property(component, component.h_min, pressure, field)
            high = interpolate_property(component, component.h_max, pressure, field)
            monotone = monotone & proven & (high > low)
    return monotone


def component_response(component, pressure, temperature, name):
    """Return one PT response and status without public state construction."""
    if component.provider == "native":
        from ..perfect_gas.perfect_gas import specific_heat

        constants = component.constants
        cp, _ = specific_heat(constants)
        if name == "enthalpy":
            value = constants.h_ref + cp * (temperature - constants.T_ref)
        else:
            value = (
                constants.s_ref
                + cp * (jnp.log(temperature) - jnp.log(constants.T_ref))
                - constants.R * (jnp.log(pressure) - jnp.log(constants.P_ref))
            )
        code = jnp.where(jnp.isfinite(value), 0, StatusCode.PROVIDER_FAILURE)
        return jnp.where(jnp.isfinite(value), value, 0.0), code.astype(jnp.int32)

    from ..bicubic.numerics import _cell_basis, interpolate_property, solve_inverse

    h, p, converged, _, code = solve_inverse(
        component,
        pairs.PT_INPUTS,
        pressure,
        temperature,
        return_status=True,
        max_steps=component.inverse_max_steps,
        tol=component.inverse_tolerance,
    )
    h = jnp.where(converged, h, (component.h_min + component.h_max) / 2)
    p = jnp.where(converged, p, jnp.sqrt(component.p_min) * jnp.sqrt(component.p_max))
    i, j, _, _ = _cell_basis(component, h, p)
    trusted = component._cell_validity[i, j]
    phase = interpolate_property(component, h, p, "is_two_phase")
    code = jnp.where((code == 0) & ~trusted, StatusCode.OUT_OF_DOMAIN, code)
    code = jnp.where((code == 0) & (phase > 0.5), StatusCode.NONCONVERGENCE, code)
    value = h if name == "enthalpy" else interpolate_property(component, h, p, name)
    code = jnp.where(
        (code == 0) & ~jnp.isfinite(value), StatusCode.PROVIDER_FAILURE, code
    )
    return jnp.where(jnp.isfinite(value), value, 0.0), code.astype(jnp.int32)


def inverse_response(fluid, pressure, temperature, name):
    total, status = jnp.zeros_like(temperature), jnp.zeros_like(
        temperature, dtype=jnp.int32
    )
    for component, weight in zip(fluid.components, fluid.mass_fractions):
        if weight == 0:
            continue
        value, code = component_response(component, pressure, temperature, name)
        total = total + weight * value
        status = jnp.where(status == 0, code, status)
    return total, status
