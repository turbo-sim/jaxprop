"""Bounded temperature inverses for a fixed-composition binary model.

At fixed pressure both supported inverse pairs require one scalar temperature,
not the legacy two-variable pressure/temperature solve. Calorically perfect
components have explicit inverses. Other native components use selected raw
responses, and a custom implicit derivative avoids differentiating iterations.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

from ..fluid_api import StatusCode

_NAMES = frozenset(("enthalpy", "entropy"))
_BOUNDARY_RTOL = 1e-12


def _relative_precision(dtype):
    return max(_BOUNDARY_RTOL, 64 * jnp.finfo(dtype).eps)


def _outside_response(target, lower, upper, precision):
    """Use each endpoint's own relative allowance, including near-zero bounds.

    A distant representability ceiling must not widen the lower-bound tolerance
    (or conversely). Both divisors are at least one, so even a finite target at
    the floating-point maximum remains safe without subtracting large values.
    """
    lower_scale = jnp.maximum(jnp.abs(lower), 1.0)
    upper_scale = jnp.maximum(jnp.abs(upper), 1.0)
    return (target / lower_scale < lower / lower_scale - precision) | (
        target / upper_scale > upper / upper_scale + precision
    )


def _request(fluid, pressure, target):
    """Prepare finite arithmetic without changing the original request status."""
    pressure, target = jnp.asarray(pressure), jnp.asarray(target)
    dtype = jnp.result_type(pressure, target, 0.0)
    pressure, target = pressure.astype(dtype), target.astype(dtype)
    input_valid = jnp.isfinite(pressure) & (pressure > 0) & jnp.isfinite(target)
    fallback_pressure = getattr(fluid, "p_min", 1.0)
    pressure = jnp.where(
        jnp.isfinite(pressure) & (pressure > 0), pressure, fallback_pressure
    )
    target = jnp.where(jnp.isfinite(target), target, 0.0)
    lower, upper = fluid._temperature_bounds(pressure)
    dtype = jnp.result_type(pressure, target, lower, upper, 0.0)
    pressure, target = pressure.astype(dtype), target.astype(dtype)
    lower, upper = jnp.asarray(lower, dtype=dtype), jnp.asarray(upper, dtype=dtype)
    bounds_valid = (
        jnp.isfinite(lower) & jnp.isfinite(upper) & (lower > 0) & (lower < upper)
    )
    # Failed rows still need finite log arguments and a nonzero coordinate span.
    lower = jnp.where(bounds_valid, lower, 1.0)
    upper = jnp.where(bounds_valid, upper, 1000.0)
    return pressure, target, lower, upper, input_valid, bounds_valid


@jax.custom_jvp
def _zero_failed_tangents(temperature, successful):
    """Keep a protected primal temperature but zero all failed contributions."""
    return temperature


@_zero_failed_tangents.defjvp
def _zero_failed_tangents_jvp(primals, tangents):
    temperature, successful = primals
    temperature_tangent, _ = tangents
    return temperature, jnp.where(successful, temperature_tangent, 0.0)


def _perfect_gas_temperature(fluid, pressure, target, name, *, clip):
    """Invert weighted caloric equations directly, retaining reference offsets.

    An anchor inside the common temperature domain avoids forming an artificial
    zero-reference intercept. The component mass fractions are fixed at setup;
    their heat capacities and reference constants remain numerical model data.
    """
    pressure, target, lower, upper, input_valid, bounds_valid = _request(
        fluid, pressure, target
    )
    anchor = jnp.clip(jnp.asarray(300.0, dtype=pressure.dtype), lower, upper)
    heat_capacity = jnp.zeros((), dtype=pressure.dtype)
    reference = jnp.zeros((), dtype=pressure.dtype)
    for component, weight in zip(fluid.components, fluid.mass_fractions):
        constants = component.constants
        cp = constants.gamma * constants.R / (constants.gamma - 1)
        heat_capacity = heat_capacity + weight * cp
        if name == "enthalpy":
            value = constants.h_ref + cp * (anchor - constants.T_ref)
        else:
            value = (
                constants.s_ref
                + cp * (jnp.log(anchor) - jnp.log(constants.T_ref))
                - constants.R * (jnp.log(pressure) - jnp.log(constants.P_ref))
            )
        reference = reference + weight * value

    caloric_valid = jnp.isfinite(heat_capacity) & (heat_capacity > 0)
    denominator = jnp.where(caloric_valid, heat_capacity, 1.0)
    precision = _relative_precision(jnp.result_type(heat_capacity, target))
    if name == "enthalpy":
        low_value = reference + heat_capacity * (lower - anchor)
        high_value = reference + heat_capacity * (upper - anchor)
    else:
        log_lower, log_upper = jnp.log(lower), jnp.log(upper)
        low_value = reference + heat_capacity * (log_lower - jnp.log(anchor))
        high_value = reference + heat_capacity * (log_upper - jnp.log(anchor))
    representable = (
        jnp.isfinite(reference) & jnp.isfinite(low_value) & jnp.isfinite(high_value)
    )
    low_value = jnp.where(jnp.isfinite(low_value), low_value, 0.0)
    high_value = jnp.where(jnp.isfinite(high_value), high_value, low_value)
    outside = (
        caloric_valid
        & representable
        & _outside_response(target, low_value, high_value, precision)
    )
    # Project before inverse arithmetic even in mask mode. A finite but extreme
    # target must not overflow subtraction or poison zero failed gradients.
    adjusted = jnp.clip(target, low_value, high_value)
    protected_reference = jnp.where(jnp.isfinite(reference), reference, 0.0)
    displacement = adjusted / denominator - protected_reference / denominator
    if name == "enthalpy":
        temperature = jnp.clip(anchor + displacement, lower, upper)
    else:
        log_temperature = jnp.log(anchor) + displacement
        temperature = jnp.clip(
            jnp.exp(jnp.clip(log_temperature, log_lower, log_upper)), lower, upper
        )
    temperature = jnp.where(target < low_value, lower, temperature)
    temperature = jnp.where(target > high_value, upper, temperature)

    successful = caloric_valid & representable & jnp.isfinite(temperature)
    status = jnp.where(successful, StatusCode.SUCCESS, StatusCode.PROVIDER_FAILURE)
    if not clip:
        status = jnp.where(outside, StatusCode.OUT_OF_DOMAIN, status)
    status = jnp.where(bounds_valid, status, StatusCode.OUT_OF_DOMAIN)
    status = jnp.where(input_valid, status, StatusCode.INVALID_INPUT).astype(jnp.int32)
    successful = status == StatusCode.SUCCESS
    temperature = _zero_failed_tangents(temperature, successful)
    return temperature, status, successful & outside & clip


def _physical_residual(temperature, context):
    fluid, pressure, target, name, _, _, scale, _ = context
    value, status = fluid._inverse_response(pressure, temperature, name)
    finite = jnp.isfinite(value)
    protected = jnp.where(finite, value, 0.0)
    # Divide before subtracting so opposite, very large finite targets cannot
    # overflow an otherwise useful normalized residual.
    residual = protected / scale - target / scale
    return jnp.where(finite & (status == StatusCode.SUCCESS), residual, 1.0)


def _normalized_residual(point, context):
    lower, upper = context[4:6]
    temperature = lower + point[0] * (upper - lower)
    return jnp.reshape(_physical_residual(temperature, context), (1,))


@eqx.filter_custom_jvp
def _bounded_temperature(context, *, max_steps, tolerance):
    """Solve on a unit interval, then validate the physical root independently."""
    fluid, pressure, _, name, lower, upper, _, request_permitted = context
    initial = jnp.asarray([0.5], dtype=jnp.result_type(lower, upper))
    solution = optx.root_find(
        _normalized_residual,
        optx.Newton(rtol=tolerance, atol=tolerance),
        initial,
        args=context,
        options={"lower": jnp.zeros_like(initial), "upper": jnp.ones_like(initial)},
        max_steps=max_steps,
        throw=False,
    )
    finite_point = jnp.all(jnp.isfinite(solution.value))
    point = jnp.clip(
        jnp.where(jnp.isfinite(solution.value), solution.value, initial), 0, 1
    )
    temperature = lower + point[0] * (upper - lower)
    value, component_status = fluid._inverse_response(pressure, temperature, name)
    residual = _physical_residual(temperature, context)
    derivative = jax.jacfwd(_physical_residual)(temperature, context)
    precision = jnp.finfo(temperature.dtype)
    successful = (
        request_permitted
        & (solution.result == optx.RESULTS.successful)
        & finite_point
        & jnp.isfinite(value)
        & jnp.isfinite(residual)
        & (jnp.abs(residual) <= jnp.maximum(tolerance, 64 * precision.eps))
        & jnp.isfinite(derivative)
        & (derivative > precision.tiny)
        & (component_status == StatusCode.SUCCESS)
    )
    status = jnp.where(successful, StatusCode.SUCCESS, StatusCode.NONCONVERGENCE)
    status = jnp.where(component_status == StatusCode.SUCCESS, status, component_status)
    return temperature, status.astype(jnp.int32)


@_bounded_temperature.def_jvp
def _bounded_temperature_jvp(primals, tangents, *, max_steps, tolerance):
    (context,), (context_tangent,) = primals, tangents
    temperature, status = _bounded_temperature(
        context, max_steps=max_steps, tolerance=tolerance
    )
    derivative = jax.jacfwd(_physical_residual)(temperature, context)
    _, residual_tangent = eqx.filter_jvp(
        lambda args: _physical_residual(temperature, args),
        (context,),
        (context_tangent,),
    )
    # R(T,theta)=0 implies dT=-R_theta/R_T. Use the actual selected response,
    # not stored cp: a bicubic table's interpolated cp need not exactly equal
    # the derivative of its separately interpolated and inverted enthalpy.
    active = (
        (status == StatusCode.SUCCESS) & jnp.isfinite(derivative) & (derivative > 0)
    )
    denominator = jnp.where(active, derivative, 1.0)
    tangent = jnp.where(active, -residual_tangent / denominator, 0.0)
    return (temperature, status), (tangent, None)


def _general_temperature(fluid, pressure, target, name, *, clip):
    pressure, target, lower, upper, input_valid, bounds_valid = _request(
        fluid, pressure, target
    )
    low_value, low_status = fluid._inverse_response(pressure, lower, name)
    high_value, high_status = fluid._inverse_response(pressure, upper, name)
    endpoints_valid = (
        (low_status == StatusCode.SUCCESS)
        & (high_status == StatusCode.SUCCESS)
        & jnp.isfinite(low_value)
        & jnp.isfinite(high_value)
        & (low_value < high_value)
    )
    proof = getattr(fluid, "_inverse_monotone", None)
    monotone = endpoints_valid & (proof(pressure, name) if proof is not None else False)
    low_value = jnp.where(jnp.isfinite(low_value), low_value, 0.0)
    high_value = jnp.where(jnp.isfinite(high_value), high_value, 0.0)
    endpoint_scale = jnp.maximum(
        jnp.maximum(jnp.abs(low_value), jnp.abs(high_value)), 1.0
    )
    precision = _relative_precision(jnp.result_type(target, endpoint_scale))
    outside = monotone & _outside_response(target, low_value, high_value, precision)
    adjusted = jnp.where(monotone, jnp.clip(target, low_value, high_value), target)
    # Scale the adjusted target, so a huge projected request does not make an
    # inaccurate interior point appear converged merely through normalization.
    scale = jnp.maximum(endpoint_scale, jnp.abs(adjusted))
    permitted = input_valid & bounds_valid & (~outside | clip)
    context = fluid, pressure, adjusted, name, lower, upper, scale, permitted
    temperature, status = _bounded_temperature(
        context,
        max_steps=fluid.inverse_max_steps,
        tolerance=fluid.inverse_tolerance,
    )
    if not clip:
        status = jnp.where(outside, StatusCode.OUT_OF_DOMAIN, status)
    status = jnp.where(bounds_valid, status, StatusCode.OUT_OF_DOMAIN)
    status = jnp.where(input_valid, status, StatusCode.INVALID_INPUT).astype(jnp.int32)
    return temperature, status, (status == StatusCode.SUCCESS) & outside & clip


def solve_temperature(fluid, pressure, target, name, *, clip=False):
    """Return protected temperature, public numerical status and projection flag.

    ``name`` selects ``enthalpy`` (HP) or ``entropy`` (PS). The caller owns the
    pressure policy and final complete-state assembly. Temperature bounds and
    selected component responses are supplied by the constructed binary model.
    Generic target clipping additionally needs its connected monotonicity proof;
    finite endpoint values alone do not justify inventing an unreachable root.
    """
    if name not in _NAMES:
        raise ValueError(f"Unsupported internal binary inverse property: {name!r}")
    if fluid._all_perfect_gas:
        return _perfect_gas_temperature(fluid, pressure, target, name, clip=clip)
    return _general_temperature(fluid, pressure, target, name, clip=clip)
