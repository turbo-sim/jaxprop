"""Bicubic numerical policies and a single eager status inspection."""

from functools import partial, update_wrapper
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .. import helpers_props as pairs
from ..errors import (
    CalculationError,
    ConvergenceError,
    InvalidStateError,
    OutOfDomainError,
    UnsupportedExecutionModeError,
    UnsupportedOperationError,
)
from ..fluid_api import PAIR_NAMES, StatusCode, validate_request, DIAGNOSTIC_PROPERTIES
from ..fluid_state import FluidState
from .numerics import _cell_basis, interpolate_properties, solve_inverse

_EAGER_TRACE_STATE = jax.core.get_opaque_trace_state()
_BOUNDARY_RTOL = 1e-12
_INPUT_NAMES = {
    pairs.HmassP_INPUTS: ("enthalpy", "pressure"),
    pairs.PT_INPUTS: ("pressure", "temperature"),
    pairs.DmassP_INPUTS: ("density", "pressure"),
    pairs.PSmass_INPUTS: ("pressure", "entropy"),
    pairs.HmassSmass_INPUTS: ("enthalpy", "entropy"),
    pairs.DmassHmass_INPUTS: ("density", "enthalpy"),
    pairs.DmassT_INPUTS: ("density", "temperature"),
    pairs.DmassSmass_INPUTS: ("density", "entropy"),
}
_CORE = ("temperature", "density", "enthalpy", "pressure", "entropy", "internal_energy")


def outside(value, lower, upper):
    relative = max(_BOUNDARY_RTOL, 64 * jnp.finfo(jnp.result_type(value, 0.0)).eps)
    tolerance = relative * max(abs(lower), abs(upper), 1.0)
    return (value < lower - tolerance) | (value > upper + tolerance)


def _metadata(fluid, errors, supersaturation=False):
    return dict(
        fluid_name=fluid.name,
        identifier=fluid.identifier,
        provider=fluid.provider,
        providers=frozenset((fluid.provider,)),
        model=fluid.model,
        reference_id=fluid.reference_id,
        available_properties=fluid.capabilities.properties if supersaturation else fluid.capabilities.properties - DIAGNOSTIC_PROPERTIES,
        error_policy=errors,
    )


def _evaluate_coordinates(fluid, input_type, first, second, *, errors, supersaturation=False):
    """Apply pointwise policy to a scalar or a direct natural-coordinate batch.

    Inverse solves remain scalar and are vectorized by ``evaluate_state``.
    Direct HP batches share protected cell coordinates and assemble one state,
    avoiding a scalar state-construction graph inside ``vmap``.
    """
    finite_first, finite_second = jnp.isfinite(first), jnp.isfinite(second)
    nonfinite = ~finite_first | ~finite_second
    invalid = nonfinite
    natural_outside = jnp.asarray(False)
    supplied = []
    for name, value, finite in zip(
        _INPUT_NAMES[input_type], (first, second), (finite_first, finite_second)
    ):
        if name in ("pressure", "temperature", "density"):
            invalid = invalid | (value <= 0)
        value = jnp.where(finite, value, fluid._input_fallbacks[name])
        if name == "pressure":
            natural_outside = natural_outside | outside(value, fluid.p_min, fluid.p_max)
            value = jnp.clip(value, fluid.p_min, fluid.p_max)
        elif name == "enthalpy":
            natural_outside = natural_outside | outside(value, fluid.h_min, fluid.h_max)
            value = jnp.clip(value, fluid.h_min, fluid.h_max)
        supplied.append(value)

    if input_type == pairs.HmassP_INPUTS:
        h, p = supplied
        status = jnp.asarray(StatusCode.SUCCESS, dtype=jnp.int32)
        projected = natural_outside
    else:
        h, p, converged, projected, status = solve_inverse(
            fluid,
            input_type,
            supplied[0],
            supplied[1],
            clip=errors == "clip",
            return_status=True,
            max_steps=fluid.inverse_max_steps,
            tol=fluid.inverse_tolerance,
        )
        # The protected interpolation coordinates keep failed inverse outputs
        # finite. Its custom implicit derivative supplies zero failed tangents.
        middle_h = (fluid.h_min + fluid.h_max) / 2
        middle_p = jnp.sqrt(fluid.p_min * fluid.p_max)
        h = jnp.where(converged & jnp.isfinite(h), h, middle_h)
        p = jnp.where(converged & jnp.isfinite(p), p, middle_p)

    if errors != "clip":
        status = jnp.where(natural_outside, StatusCode.OUT_OF_DOMAIN, status)
        status = jnp.where(invalid, StatusCode.INVALID_INPUT, status)
    else:
        status = jnp.where(nonfinite, StatusCode.INVALID_INPUT, status)

    # Interpolation indices are always protected, including failed requests.
    h, p = jnp.clip(h, fluid.h_min, fluid.h_max), jnp.clip(p, fluid.p_min, fluid.p_max)
    native_batch = first.ndim != 0
    if native_batch:
        cell_basis = _cell_basis(fluid, h, p)
        i, j = cell_basis[:2]
    else:
        i = jnp.clip(
            jnp.floor((h - fluid.h_min) / fluid.delta_h).astype(jnp.int32),
            0,
            fluid.N_h - 2,
        )
        j = jnp.clip(
            jnp.floor((jnp.log(p) - jnp.log(fluid.p_min)) / fluid.delta_logP).astype(
                jnp.int32
            ),
            0,
            fluid.N_p - 2,
        )
    properties = (
        interpolate_properties(fluid, h, p, cell_basis=cell_basis)
        if native_batch
        else fluid._interp_h_p(h, p)
    )
    from .phase_boundary import point_valid
    boundary_valid = point_valid(fluid, h, p)
    status = jnp.where(
        (status == StatusCode.SUCCESS) & ~(fluid._cell_validity[i, j] | boundary_valid),
        StatusCode.OUT_OF_DOMAIN,
        status,
    )
    if input_type == pairs.PT_INPUTS and "is_two_phase" in properties:
        # At saturation T depends on p, not on quality/enthalpy. Numerical
        # derivative noise must not manufacture a unique PT inverse there.
        status = jnp.where(
            (status == StatusCode.SUCCESS) & (properties["is_two_phase"] > 0.5),
            StatusCode.NONCONVERGENCE,
            status,
        )
    finite_core = jnp.asarray(True)
    for name in _CORE:
        finite_core = finite_core & jnp.isfinite(properties[name])
    finite_core = (
        finite_core & (properties["temperature"] > 0) & (properties["density"] > 0)
    )
    status = jnp.where(
        (status == StatusCode.SUCCESS) & ~finite_core,
        StatusCode.PROVIDER_FAILURE,
        status,
    ).astype(jnp.int32)
    clipped = (natural_outside | projected | invalid) if errors == "clip" else None
    if getattr(fluid, "phase_support", False) and not supersaturation:
        properties = {name: value for name, value in properties.items() if name not in DIAGNOSTIC_PROPERTIES}
    metadata = _metadata(fluid, errors, supersaturation)
    if native_batch:
        # Keep natural-coordinate batches pointwise. The generic all-success
        # batch conditional can force tuple-branch copies of every field; a
        # per-row mask lets a compiled consumer fuse only the fields it uses.
        # Input/domain/core checks above establish the same trusted invariants
        # as scalar native assembly, including locally unavailable optional data.
        successful = status == StatusCode.SUCCESS
        if errors != "raise":
            properties = {
                name: jnp.where(successful, value, jnp.nan)
                for name, value in properties.items()
            }
        if clipped is not None:
            clipped = clipped & successful
        return FluidState._from_validated(metadata, properties, status, clipped)
    return FluidState._from_native(
        metadata,
        properties,
        status,
        clipped=clipped,
        mask_failed=errors != "raise",
    )


def evaluate_scalar(fluid, input_type, first, second, *, errors, supersaturation=False):
    """Evaluate one scalar request through the shared coordinate policy."""
    return _evaluate_coordinates(fluid, input_type, first, second, errors=errors, supersaturation=supersaturation)


def evaluate_state(fluid, input_type, first, second, *, errors, supersaturation=False):
    dtype = jnp.result_type(first, second, fluid.h_vals, fluid.logP_vals, 0.0)
    first, second = jnp.broadcast_arrays(
        jnp.asarray(first, dtype=dtype), jnp.asarray(second, dtype=dtype)
    )
    if first.ndim == 0:
        return fluid._get_state_scalar(input_type, first, second, errors=errors, supersaturation=supersaturation)
    if input_type == pairs.HmassP_INPUTS:
        return _evaluate_coordinates(fluid, input_type, first, second, errors=errors, supersaturation=supersaturation)
    result = jax.vmap(
        lambda a, b: fluid._get_state_scalar(input_type, a, b, errors=errors, supersaturation=supersaturation)
    )(first.ravel(), second.ravel())
    return jax.tree_util.tree_map(lambda value: value.reshape(first.shape), result)


def _saturation_scalar(fluid, pressure, quality, *, errors):
    endpoint = 0.0 if fluid.metastable_phase == "liquid" else 1.0
    upper = jnp.exp(fluid.logP_sat_vals[-1])
    nonfinite = ~jnp.isfinite(pressure) | ~jnp.isfinite(quality)
    invalid = nonfinite | (pressure <= 0)
    pressure_outside = outside(pressure, fluid.p_min, fluid._saturation_p_max)
    quality_outside = jnp.abs(quality - endpoint) > _BOUNDARY_RTOL
    projected = pressure_outside | quality_outside
    safe_pressure = jnp.clip(
        jnp.where(jnp.isfinite(pressure), pressure, fluid.p_min), fluid.p_min, upper
    )
    status = jnp.asarray(StatusCode.SUCCESS, dtype=jnp.int32)
    if errors == "clip":
        status = jnp.where(nonfinite, StatusCode.INVALID_INPUT, status)
    else:
        status = jnp.where(projected, StatusCode.OUT_OF_DOMAIN, status)
        status = jnp.where(invalid, StatusCode.INVALID_INPUT, status)
    properties = fluid._interp_sat_p(safe_pressure)
    finite = jnp.asarray(True)
    for name in _CORE:
        finite = finite & jnp.isfinite(properties[name])
    finite = finite & (properties["temperature"] > 0) & (properties["density"] > 0)
    status = jnp.where((status == 0) & ~finite, StatusCode.PROVIDER_FAILURE, status)
    return FluidState._from_native(
        _metadata(fluid, errors),
        properties,
        status.astype(jnp.int32),
        clipped=projected | invalid if errors == "clip" else None,
        mask_failed=errors != "raise",
    )


def evaluate_saturation(fluid, pressure, quality, *, errors, supersaturation=False):
    if getattr(fluid, "phase_support", False):
        from .phase_support import evaluate_endpoints
        return evaluate_endpoints(fluid, pressure, quality, errors=errors, supersaturation=supersaturation)
    dtype = jnp.result_type(pressure, quality, fluid.h_vals, 0.0)
    pressure, quality = jnp.broadcast_arrays(
        jnp.asarray(pressure, dtype=dtype), jnp.asarray(quality, dtype=dtype)
    )
    if pressure.ndim == 0:
        return _saturation_scalar(fluid, pressure, quality, errors=errors)
    result = jax.vmap(lambda p, q: _saturation_scalar(fluid, p, q, errors=errors))(
        pressure.ravel(), quality.ravel()
    )
    return jax.tree_util.tree_map(lambda value: value.reshape(pressure.shape), result)


class _SaturationRequest:
    def __init__(self, fluid):
        self.name, self.provider, self.model = fluid.name, fluid.provider, fluid.model
        self.capabilities = fluid._saturation_capabilities


class _BicubicStateCall:
    def __init__(self, function, saturation=False):
        update_wrapper(self, function)
        self.saturation = saturation
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
        **options,
    ):
        if self.saturation:
            if "saturation" not in fluid.capabilities.phase_operations:
                raise UnsupportedOperationError(
                    fluid_name=fluid.name,
                    provider=fluid.provider,
                    model=fluid.model,
                    request="saturation",
                    reason="This table has no saturation endpoint payload.",
                )
            if (
                isinstance(input_type, (bool, np.bool_))
                or not isinstance(input_type, Integral)
                or input_type != pairs.PQ_INPUTS
            ):
                raise UnsupportedOperationError(
                    fluid_name=fluid.name,
                    provider=fluid.provider,
                    model=fluid.model,
                    request="saturation",
                    reason="Saturation evaluation requires PQ_INPUTS at the declared endpoint qualities.",
                    supported=("PQ_INPUTS",),
                )
        context = _SaturationRequest(fluid) if self.saturation else fluid
        first, second = validate_request(
            context,
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
                reason="JAX transformations require explicit errors='mask' or errors='clip'; eager raise provides indexed failures.",
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
            codes = np.asarray(result.status)
            failed = np.flatnonzero(codes)
            if failed.size:
                position = int(failed[0])
                code = int(codes.flat[position])
                index = (
                    tuple(
                        int(value) for value in np.unravel_index(position, codes.shape)
                    )
                    if codes.ndim
                    else ()
                )
                error_type = {
                    StatusCode.INVALID_INPUT: InvalidStateError,
                    StatusCode.OUT_OF_DOMAIN: OutOfDomainError,
                    StatusCode.NONCONVERGENCE: ConvergenceError,
                }.get(code, CalculationError)
                reason = {
                    StatusCode.INVALID_INPUT: "Inputs must be finite, with positive temperature, pressure and density.",
                    StatusCode.OUT_OF_DOMAIN: "The request lies outside the table domain, a trusted interpolation cell, or the declared saturation endpoint; use errors='clip' for recoverable domain projection.",
                    StatusCode.NONCONVERGENCE: "No unique, residual-validated inverse root was obtained inside the table bounds.",
                }.get(
                    code, "The table produced nonfinite or nonphysical core properties."
                )
                if (self.saturation or input_type == pairs.PQ_INPUTS) and getattr(fluid, "phase_support", False) and code == StatusCode.OUT_OF_DOMAIN:
                    reason = f"Phase-support saturation accepts Q=0 or Q=1, with pressure from {fluid._saturation_p_min:g} to {fluid._saturation_p_max:g} Pa."
                elif self.saturation and code == StatusCode.OUT_OF_DOMAIN:
                    endpoint = 0 if fluid.metastable_phase == "liquid" else 1
                    reason = (
                        f"This saturation table contains only Q={endpoint}, "
                        f"at pressures from {fluid.p_min:g} to {fluid._saturation_p_max:g} Pa."
                    )
                elif (
                    input_type == pairs.PT_INPUTS and code == StatusCode.NONCONVERGENCE
                ):
                    reason += " PT does not identify enthalpy in a two-phase cell."
                raise error_type(
                    fluid_name=fluid.name,
                    provider=fluid.provider,
                    model=fluid.model,
                    request=PAIR_NAMES[input_type],
                    reason=reason,
                    index=index,
                )
        return result


def checked_bicubic_jit(function):
    return _BicubicStateCall(function)


def checked_saturation_jit(function):
    return _BicubicStateCall(function, saturation=True)
