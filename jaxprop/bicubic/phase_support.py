"""Optional source-generated saturation endpoints for ordinary bicubic models.

The source EOS is used only in construction. Runtime endpoint evaluation is
native array interpolation; unsupported qualities are rejected, not mixed into
a new thermodynamic model. Diagnostic 2D coefficients use the existing source
definitions rather than new temperature/enthalpy approximations here.
"""

import numpy as np
import jax
import jax.numpy as jnp

from ..fluid_api import DIAGNOSTIC_PROPERTIES, StatusCode
from ..fluid_state import FluidState
from ..property_schema import PROPERTIES_CANONICAL


def generate_endpoints(model):
    from ..coolprop import FluidCoolProp
    from ..helpers_props import PQ_INPUTS

    if model.logP_sat_vals is None:
        return {}
    source = FluidCoolProp(model.name, model.backend)
    pressures = np.exp(np.asarray(model.logP_sat_vals))
    result = {}
    for label, quality in (("liquid", 0.0), ("vapor", 1.0)):
        minimum = float(
            (
                source.triple_point_liquid
                if quality == 0
                else source.triple_point_vapor
            ).pressure
        )
        maximum = float(source.critical_point.pressure)
        payload = {
            name: {key: np.full(len(pressures), np.nan) for key in ("value", "grad_p")}
            for name in PROPERTIES_CANONICAL
        }
        for index, pressure in enumerate(pressures):
            step = 1e-5 * max(pressure, 1.0)
            lower, upper = max(minimum, pressure - step), min(
                np.nextafter(maximum, 0.0), pressure + step
            )
            center = source.get_state(
                PQ_INPUTS, float(pressure), quality, supersaturation=True
            ).to_dict()
            below = source.get_state(
                PQ_INPUTS, float(lower), quality, supersaturation=True
            ).to_dict()
            above = source.get_state(
                PQ_INPUTS, float(upper), quality, supersaturation=True
            ).to_dict()
            for name, entry in payload.items():
                entry["value"][index] = center.get(name, np.nan)
                entry["grad_p"][index] = (
                    above.get(name, np.nan) - below.get(name, np.nan)
                ) / (upper - lower)
        result[label] = payload
    return result


def validation_error(payload, metadata, core):
    settings = metadata["generation_settings"]
    low, high = (
        settings["saturation_log_pressure_min"],
        settings["saturation_log_pressure_max"],
    )
    if low is None or high is None:
        return (
            None
            if payload == {}
            else "phase support: undeclared endpoint pressure grid"
        )
    if not np.isfinite(low) or not np.isfinite(high) or not low < high:
        return "phase support: invalid endpoint pressure grid"
    if set(payload) != {"liquid", "vapor"}:
        return "phase support: both saturation endpoints are required"
    dtype = np.dtype(metadata["numeric_dtype"])
    expected_p = np.exp(np.linspace(low, high, metadata["N_p_sat"], dtype=dtype))
    for side in ("liquid", "vapor"):
        fields = payload[side]
        if not isinstance(fields, dict) or set(fields) != set(PROPERTIES_CANONICAL):
            return f"phase support {side}: incomplete property schema"
        for name, entry in fields.items():
            if not isinstance(entry, dict) or set(entry) != {"value", "grad_p"}:
                return f"phase support {side}.{name}: incomplete arrays"
            for key, values in entry.items():
                array = np.asarray(values)
                if array.shape != expected_p.shape or array.dtype != dtype:
                    return f"phase support {side}.{name}.{key}: incorrect shape/dtype"
                if np.any(np.isinf(array)) or (
                    name in core and not np.all(np.isfinite(array))
                ):
                    return f"phase support {side}.{name}.{key}: nonfinite data"
        if not np.allclose(
            fields["pressure"]["value"], expected_p, rtol=1e-9, atol=1e-6
        ):
            return f"phase support {side}: pressure grid mismatch"
        if np.any(np.asarray(fields["temperature"]["value"]) <= 0) or np.any(
            np.asarray(fields["density"]["value"]) <= 0
        ):
            return f"phase support {side}: nonpositive core data"
        expected_quality = 0.0 if side == "liquid" else 1.0
        if not np.all(np.asarray(fields["quality_mass"]["value"]) == expected_quality):
            return f"phase support {side}: incorrect endpoint quality"
    return None


def interpolate_endpoint(fluid, pressure, quality, *, supersaturation=False):
    """Hermite interpolation in pressure with the declared log-pressure nodes."""
    coordinate = (jnp.log(pressure) - fluid.logP_sat_vals[0]) / fluid.delta_logP_sat
    index = jnp.clip(jnp.floor(coordinate).astype(jnp.int32), 0, fluid.N_p_sat - 2)
    fields = fluid.capabilities.properties
    if not supersaturation:
        fields = fields - DIAGNOSTIC_PROPERTIES
    branches = {}
    for side in ("liquid", "vapor"):
        table, masks = fluid._saturation_table[side], fluid._saturation_validity[side]
        left, right = (
            table["pressure"]["value"][index],
            table["pressure"]["value"][index + 1],
        )
        width = right - left
        x = (pressure - left) / width
        values = {}
        for name in fields:
            entry = table[name]
            a, b = entry["value"][index], entry["value"][index + 1]
            ga, gb = entry["grad_p"][index], entry["grad_p"][index + 1]
            delta = b - a
            value = (
                (1 - x) * a
                + x * b
                + x
                * (1 - x)
                * ((1 - x) * (ga * width - delta) + x * (delta - gb * width))
            )
            mask = masks[name]
            available = True if mask is None else mask[index]
            values[name] = jnp.where(available & jnp.isfinite(value), value, jnp.nan)
        branches[side] = values
    vapor = quality > 0.5
    result = {
        name: jnp.where(vapor, branches["vapor"][name], branches["liquid"][name])
        for name in fields
    }
    result["pressure"] = pressure
    result["quality_mass"] = jnp.where(vapor, 1.0, 0.0).astype(pressure.dtype)
    return result


def evaluate_endpoints(fluid, pressure, quality, *, errors, supersaturation=False):
    from .request_policy import outside

    dtype = jnp.result_type(pressure, quality, fluid.h_vals, 0.0)
    pressure, quality = jnp.broadcast_arrays(
        jnp.asarray(pressure, dtype=dtype), jnp.asarray(quality, dtype=dtype)
    )
    nonfinite = ~jnp.isfinite(pressure) | ~jnp.isfinite(quality)
    invalid = nonfinite | (pressure <= 0)
    pressure_outside = outside(
        pressure, fluid._saturation_p_min, fluid._saturation_p_max
    )
    quality_outside = (jnp.abs(quality) > 1e-12) & (jnp.abs(quality - 1) > 1e-12)
    projected = pressure_outside | quality_outside
    safe_p = jnp.clip(
        jnp.where(jnp.isfinite(pressure), pressure, fluid._saturation_p_min),
        fluid._saturation_p_min,
        fluid._saturation_p_max,
    )
    safe_q = jnp.where(jnp.isfinite(quality), quality, 0.0)
    props = interpolate_endpoint(fluid, safe_p, safe_q, supersaturation=supersaturation)
    status = jnp.zeros_like(pressure, dtype=jnp.int32)
    if errors == "clip":
        status = jnp.where(nonfinite, StatusCode.INVALID_INPUT, status)
    else:
        status = jnp.where(projected, StatusCode.OUT_OF_DOMAIN, status)
        status = jnp.where(invalid, StatusCode.INVALID_INPUT, status)
    finite = (props["temperature"] > 0) & (props["density"] > 0)
    for name in (
        "temperature",
        "density",
        "enthalpy",
        "pressure",
        "entropy",
        "internal_energy",
    ):
        finite &= jnp.isfinite(props[name])
    status = jnp.where(
        (status == 0) & ~finite, StatusCode.PROVIDER_FAILURE, status
    ).astype(jnp.int32)
    metadata = dict(
        fluid_name=fluid.name,
        identifier=fluid.identifier,
        provider=fluid.provider,
        providers=frozenset((fluid.provider,)),
        model=fluid.model,
        reference_id=fluid.reference_id,
        available_properties=(
            fluid.capabilities.properties
            if supersaturation
            else fluid.capabilities.properties - DIAGNOSTIC_PROPERTIES
        ),
        error_policy=errors,
    )
    return FluidState._from_native(
        metadata,
        props,
        status,
        clipped=(projected | invalid) if errors == "clip" else None,
        mask_failed=errors != "raise",
    )
