"""Native perfect-gas numerical policies with one eager status inspection."""

from functools import partial, update_wrapper

import jax
import jax.numpy as jnp
import numpy as np

from ..errors import (
    CalculationError,
    InvalidStateError,
    OutOfDomainError,
    UnsupportedExecutionModeError,
)
from ..fluid_api import PAIR_NAMES, StatusCode, validate_request
from ..fluid_state import FluidState
from .. import helpers_props as pairs

_EAGER_TRACE_STATE = jax.core.get_opaque_trace_state()
_BOUNDARY_RTOL = 1e-12
INPUT_NAMES = {
    pairs.PT_INPUTS: ("pressure", "temperature"),
    pairs.HmassSmass_INPUTS: ("enthalpy", "entropy"),
    pairs.HmassP_INPUTS: ("enthalpy", "pressure"),
    pairs.PSmass_INPUTS: ("pressure", "entropy"),
    pairs.DmassHmass_INPUTS: ("density", "enthalpy"),
    pairs.DmassP_INPUTS: ("density", "pressure"),
}


def _temperature_limit(constants, cp, cv, numerical):
    """Scalar representability bound; coefficients stay dynamic under JAX.

    Construction validates the energy intercept and the entropy envelope, so
    these bounds imply finite h/u/s without evaluating those fields here.
    """
    maximum = numerical.max
    intercept = constants.h_ref - cp * constants.T_ref
    remaining = 1 - intercept / maximum
    # Factor out the representability ceiling before differentiation. A direct
    # maximum/cp has a finite primal/derivative, but its forward-AD rule can form
    # maximum*dcp before division and overflow a discarded clipping branch.
    fraction = jnp.minimum(
        1.0,
        jnp.minimum(
            1.0 / jnp.maximum(cp, 1.0),
            jnp.minimum(
                1.0 / jnp.maximum(constants.R, 1.0),
                jnp.minimum(
                    remaining / jnp.maximum(cp, remaining),
                    remaining / jnp.maximum(cv, remaining),
                ),
            ),
        ),
    )
    return (maximum * fraction) * (1 - 64 * numerical.eps)


def resolve_pair(input_type, first, second, constants, *, errors, return_success=False):
    """Resolve one input pair to safe canonical T/p and numerical status.

    Only the selected pair's domain/arithmetic guards are staged. In particular,
    PT/HP temperature projections do not evaluate entropy or energy merely to
    validate a state. Invalid rows use finite intermediates before masking, so
    their zero cotangents cannot contaminate valid derivatives.
    """
    from .perfect_gas import MIN_DENSITY, MIN_PRESSURE, MIN_TEMPERATURE, specific_heat

    dtype = jnp.result_type(
        first,
        second,
        constants.R,
        constants.gamma,
        constants.h_ref,
        constants.s_ref,
        constants.T_ref,
        constants.P_ref,
        0.0,
    )
    first, second = jnp.broadcast_arrays(
        jnp.asarray(first, dtype=dtype), jnp.asarray(second, dtype=dtype)
    )
    numerical = jnp.finfo(
        jnp.result_type(first, second, constants.R, constants.h_ref, constants.T_ref)
    )
    maximum, tiny = numerical.max, numerical.tiny
    cp, cv = specific_heat(constants)
    upper_T = _temperature_limit(constants, cp, cv, numerical)
    finite_first, finite_second = jnp.isfinite(first), jnp.isfinite(second)
    nonfinite = ~finite_first | ~finite_second
    invalid = nonfinite
    outside = jnp.zeros_like(nonfinite)
    safe = []
    floors = {
        "pressure": MIN_PRESSURE,
        "temperature": MIN_TEMPERATURE,
        "density": MIN_DENSITY,
    }
    for name, value, finite in zip(
        INPUT_NAMES[input_type], (first, second), (finite_first, finite_second)
    ):
        if name in floors:
            floor = floors[name]
            invalid = invalid | (value <= 0)
            outside = outside | (value < floor * (1 - _BOUNDARY_RTOL))
            safe.append(jnp.maximum(jnp.where(finite, value, floor), floor))
        else:
            fallback = constants.h_ref if name == "enthalpy" else constants.s_ref
            safe.append(jnp.where(finite, value, fallback))

    calculation_failed = jnp.zeros_like(nonfinite)
    resolved_invalid = jnp.zeros_like(nonfinite)
    log_T = None
    if input_type == pairs.PSmass_INPUTS:
        pressure = safe[0]
        log_T = (
            jnp.log(constants.T_ref)
            + (
                (safe[1] - constants.s_ref)
                + constants.R * (jnp.log(pressure) - jnp.log(constants.P_ref))
            )
            / cp
        )
        log_upper_T = jnp.log(upper_T)
        calculation_failed = ~jnp.isfinite(log_T) | (log_T > log_upper_T)
        resolved_invalid = log_T < jnp.log(tiny)
        outside = outside | (log_T < jnp.log(MIN_TEMPERATURE) - _BOUNDARY_RTOL)
        safe_log_T = jnp.clip(
            jnp.where(jnp.isfinite(log_T), log_T, 0.0),
            jnp.log(MIN_TEMPERATURE),
            log_upper_T,
        )
        temperature = jnp.minimum(jnp.exp(safe_log_T), upper_T)
    else:
        if input_type == pairs.PT_INPUTS:
            raw_T = second
        elif input_type in (pairs.HmassP_INPUTS, pairs.HmassSmass_INPUTS):
            raw_T = constants.T_ref + (safe[0] - constants.h_ref) / cp
        elif input_type == pairs.DmassHmass_INPUTS:
            raw_T = constants.T_ref + (safe[1] - constants.h_ref) / cp
        else:
            denominator = safe[0] * constants.R
            denominator_finite = jnp.isfinite(denominator)
            denominator_positive = denominator > 0
            safe_denominator = jnp.where(
                denominator_finite & denominator_positive, denominator, 1.0
            )
            raw_T = safe[1] / safe_denominator
            # An overflowing rho*R implies a temperature below the domain;
            # an underflowing product cannot represent the requested high T.
            raw_T = jnp.where(
                ~denominator_finite,
                0.0,
                jnp.where(denominator_positive, raw_T, jnp.inf),
            )
        calculation_failed = ~jnp.isfinite(raw_T) | (raw_T > upper_T)
        resolved_invalid = raw_T <= 0
        outside = outside | (raw_T < MIN_TEMPERATURE * (1 - _BOUNDARY_RTOL))
        temperature = jnp.clip(
            jnp.where(jnp.isfinite(raw_T), raw_T, MIN_TEMPERATURE),
            MIN_TEMPERATURE,
            upper_T,
        )

    RT = constants.R * temperature
    if input_type == pairs.HmassSmass_INPUTS:
        log_p = (
            jnp.log(constants.P_ref)
            + (
                cp * (jnp.log(temperature) - jnp.log(constants.T_ref))
                - (safe[1] - constants.s_ref)
            )
            / constants.R
        )
        log_upper_p = jnp.log(maximum) + jnp.log(jnp.minimum(1.0, RT))
        calculation_failed = (
            calculation_failed | ~jnp.isfinite(log_p) | (log_p > log_upper_p)
        )
        resolved_invalid = resolved_invalid | (log_p < jnp.log(tiny))
        outside = outside | (log_p < jnp.log(MIN_PRESSURE) - _BOUNDARY_RTOL)
        safe_log_p = jnp.clip(
            jnp.where(jnp.isfinite(log_p), log_p, jnp.log(MIN_PRESSURE)),
            jnp.log(MIN_PRESSURE),
            log_upper_p - 64 * numerical.eps,
        )
        pressure = jnp.exp(safe_log_p)
    elif input_type == pairs.DmassHmass_INPUTS:
        upper_rho = (maximum / RT) * (1 - 64 * numerical.eps)
        calculation_failed = calculation_failed | (safe[0] > upper_rho)
        pressure = jnp.minimum(safe[0], upper_rho) * RT
    elif input_type != pairs.PSmass_INPUTS:
        pressure = safe[0] if input_type == pairs.PT_INPUTS else safe[1]

    # This condition proves finite density without computing it for projections.
    upper_pressure = maximum * jnp.minimum(1.0, RT)
    pressure_failed = ~jnp.isfinite(pressure) | (pressure > upper_pressure)
    calculation_failed = calculation_failed | pressure_failed
    pressure = jnp.where(pressure_failed, MIN_PRESSURE, pressure)
    outside = outside | (pressure < MIN_PRESSURE * (1 - _BOUNDARY_RTOL))
    outside = outside | (pressure < MIN_DENSITY * (1 - _BOUNDARY_RTOL) * RT)

    clipped = None
    if errors == "clip":
        pressure = jnp.maximum(
            pressure,
            jnp.maximum(MIN_PRESSURE, MIN_DENSITY * (1 + 8 * numerical.eps) * RT),
        )
        clipped = outside | resolved_invalid | invalid
        # Build status selections directly in the public int32 dtype. Bare
        # IntEnum/Python integers otherwise create int64 intermediates when
        # the consuming application enables JAX double precision.
        status = jnp.where(
            calculation_failed,
            np.int32(StatusCode.PROVIDER_FAILURE),
            np.int32(StatusCode.SUCCESS),
        )
        status = jnp.where(nonfinite, np.int32(StatusCode.INVALID_INPUT), status)
        successful = ~(nonfinite | calculation_failed)
    else:
        status = jnp.where(
            outside, np.int32(StatusCode.OUT_OF_DOMAIN), np.int32(StatusCode.SUCCESS)
        )
        status = jnp.where(resolved_invalid, np.int32(StatusCode.INVALID_INPUT), status)
        status = jnp.where(
            calculation_failed, np.int32(StatusCode.PROVIDER_FAILURE), status
        )
        status = jnp.where(invalid, np.int32(StatusCode.INVALID_INPUT), status)
        successful = ~(invalid | outside | resolved_invalid | calculation_failed)
    result = temperature, pressure, status.astype(jnp.int32), clipped
    return (*result, successful) if return_success else result


def evaluate_state(fluid, input_type, first, second, *, errors):
    """Resolve once, apply policy, then assemble one trusted native state."""
    from .perfect_gas import calculate_properties_PT

    temperature, pressure, status, clipped, successful = resolve_pair(
        input_type, first, second, fluid.constants, errors=errors, return_success=True
    )

    def properties_for_state():
        properties = calculate_properties_PT(pressure, temperature, fluid.constants)
        if errors != "clip":
            # Preserve independent coordinates in successful non-clipping requests.
            # Overwriting before JAX lowering prunes the redundant forward h/s
            # equations, including log(exp(...)) work in HS/PS evaluations.
            for name, value in zip(INPUT_NAMES[input_type], (first, second)):
                if name == "enthalpy":
                    properties["enthalpy"] = jnp.broadcast_to(
                        jnp.asarray(value, dtype=temperature.dtype), status.shape
                    )
                    properties["internal_energy"] = (
                        properties["enthalpy"] - fluid.constants.R * temperature
                    )
                elif name == "entropy":
                    properties["entropy"] = jnp.broadcast_to(
                        jnp.asarray(value, dtype=temperature.dtype), status.shape
                    )
                elif name == "density":
                    properties["density"] = jnp.broadcast_to(
                        jnp.asarray(value, dtype=temperature.dtype), status.shape
                    )
        return properties

    metadata = dict(
        fluid_name=fluid.name,
        identifier=fluid.identifier,
        provider=fluid.provider,
        providers=frozenset((fluid.provider,)),
        model=fluid.model,
        reference_id=fluid.reference_id,
        available_properties=fluid.capabilities.properties,
        error_policy=errors,
    )

    def native_state(_):
        return FluidState._from_native(
            metadata,
            properties_for_state(),
            status,
            clipped=clipped,
            mask_failed=errors != "raise",
            success_mask=successful,
        )

    if errors != "raise" and status.size >= 64:
        # Stage numerical policy before property assembly. On a fully valid
        # batch this shares the resolved coordinates once, and prevents XLA
        # from cloning validation into every property/mask fusion.
        return jax.lax.cond(
            jnp.all(successful),
            lambda _: FluidState._from_validated(
                metadata, properties_for_state(), status, clipped
            ),
            native_state,
            operand=None,
        )
    return native_state(None)


class _PerfectGasStateCall:
    """Preserve native JIT; typed eager errors inspect status once per call."""

    def __init__(self, function):
        update_wrapper(self, function)
        self.compiled = jax.jit(
            function,
            static_argnames=(
                "input_type",
                "generalize_quality",
                "supersaturation",
                "errors",
            ),
        )

    def __get__(self, instance, owner=None):
        return self if instance is None else partial(self, instance)

    def __call__(
        self,
        fluid,
        input_type,
        prop_1,
        prop_2,
        *,
        generalize_quality=False,
        supersaturation=False,
        errors="raise",
        **options
    ):
        first, second = validate_request(
            fluid,
            input_type,
            prop_1,
            prop_2,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
            errors=errors,
            array_backend="jax",
            options=options,
        )
        if errors == "raise" and (
            isinstance(first, jax.core.Tracer)
            or isinstance(second, jax.core.Tracer)
            or jax.core.get_opaque_trace_state() != _EAGER_TRACE_STATE
        ):
            raise UnsupportedExecutionModeError(
                fluid_name=fluid.name,
                provider=fluid.provider,
                model=fluid.model,
                request="errors=raise",
                reason="JAX transformations require explicit errors='mask' or errors='clip'; eager raise preserves indexed numerical errors.",
                supported=("errors=mask", "errors=clip"),
            )
        result = self.compiled(
            fluid,
            input_type,
            first,
            second,
            generalize_quality=bool(generalize_quality),
            supersaturation=bool(supersaturation),
            errors=errors,
        )
        if errors == "raise":
            status = np.asarray(result.status)
            if status.ndim == 0:
                code, index = int(status), ()
            else:
                failures = np.flatnonzero(status)
                code = int(status.flat[failures[0]]) if failures.size else 0
                index = (
                    tuple(int(i) for i in np.unravel_index(failures[0], status.shape))
                    if failures.size
                    else ()
                )
            if code:
                error_type = {
                    StatusCode.INVALID_INPUT: InvalidStateError,
                    StatusCode.OUT_OF_DOMAIN: OutOfDomainError,
                }.get(code, CalculationError)
                reason = {
                    StatusCode.INVALID_INPUT: "Inputs must be finite and imply positive temperature, pressure and density; enthalpy/entropy may use either sign.",
                    StatusCode.OUT_OF_DOMAIN: "The state lies below a declared perfect-gas numerical bound; use errors='clip' to project onto the declared domain.",
                }.get(
                    code,
                    "The perfect-gas calculation cannot represent finite core values at these inputs.",
                )
                raise error_type(
                    fluid_name=fluid.name,
                    provider=fluid.provider,
                    model=fluid.model,
                    request=PAIR_NAMES[input_type],
                    reason=reason,
                    index=index,
                )
        return result


def checked_perfect_gas_jit(function):
    return _PerfectGasStateCall(function)
