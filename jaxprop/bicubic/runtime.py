"""Construction-only preparation of the lean bicubic numerical payload."""

import jax.numpy as jnp
import numpy as np

from ..property_schema import PROPERTIES_CANONICAL
from .cache_metadata import metadata_fingerprint


class InspectionPayload:
    """Opaque, instance-owned inspection data, excluded from numerical PyTrees.

    The complete cache remains available through ``fluid.table``. Only runtime
    coefficients and inverse-guess fields participate in JAX transformations.
    """

    __slots__ = ("_table", "_identity")

    def __init__(self, table, identity=None):
        object.__setattr__(
            self,
            "_identity",
            metadata_fingerprint(table["metadata"]) if identity is None else identity,
        )
        object.__setattr__(
            self,
            "_table",
            {name: value for name, value in table.items() if name != "metadata"},
        )

    def __setattr__(self, name, value):
        raise AttributeError("Inspection provenance is immutable")

    @property
    def table(self):
        return self._table

    def __hash__(self):
        return hash(self._identity)

    def __eq__(self, other):
        return (
            isinstance(other, InspectionPayload) and self._identity == other._identity
        )

    def __deepcopy__(self, memo):
        # Payload arrays are immutable. Dynamic evaluator arrays are copied by
        # the containing fluid's ordinary PyTree/deepcopy machinery.
        return self

    def __reduce__(self):
        return type(self), (self._table, self._identity)


def prepare_runtime(table, properties, table_mode, critical_pressure=None):
    """Prepare availability and phase masks once, after complete cache validation."""
    numerical, availability = {}, {}
    guess_fields = {"enthalpy", "pressure", "temperature", "density", "entropy"}
    for name in PROPERTIES_CANONICAL:
        if name not in properties:
            continue
        entry = table[name]
        coefficients = np.asarray(entry["coeffs"])
        valid = np.all(np.isfinite(coefficients), axis=-1)
        all_available = bool(np.all(valid))
        # Absence of a mask is a setup-proven, static all-available pattern.
        # Do not send/gather a redundant boolean grid at every evaluation.
        availability[name] = None if all_available else jnp.asarray(valid)
        # Unavailable optional cells use finite arithmetic even under AD. The
        # public inspection view retains their original NaN coefficient pattern.
        numerical[name] = {
            "coeffs": (
                entry["coeffs"]
                if all_available
                else jnp.asarray(np.where(valid[..., None], coefficients, 0.0))
            )
        }
        if name in guess_fields:
            numerical[name]["value"] = entry["value"]

    shape = np.asarray(table["temperature"]["coeffs"]).shape[:2]
    # Public cache validation requires the persisted source-envelope mask.
    # The fallback supports deliberately minimal private polynomial fixtures;
    # it never permits an old/incomplete persisted cache to pass validation.
    valid_cells = np.asarray(
        table.get("phase_cell_validity", np.ones(shape, dtype=bool))
    ).copy()
    if table_mode == "equilibrium":
        phase = np.asarray(table["is_two_phase"]["value"])
        corners = np.stack(
            [phase[:-1, :-1], phase[1:, :-1], phase[:-1, 1:], phase[1:, 1:]]
        )
        known = np.all(np.isfinite(corners), axis=0)
        binary = np.all((corners == 0) | (corners == 1), axis=0)
        same_phase = np.all(corners == corners[0], axis=0)
        valid_cells &= known & binary & same_phase
        # A coarse cell can span the entire dome while all four corners are
        # single phase. Detect liquid-to-vapor edges below the critical point;
        # supercritical quality labels do not define a phase discontinuity.
        if critical_pressure is not None and "quality_mass" in table:
            quality = np.asarray(table["quality_mass"]["value"])
            pressure = np.asarray(table["pressure"]["value"])
            low, high = quality[:-1], quality[1:]
            endpoints = ((low == 0) | (low == 1)) & ((high == 0) | (high == 1))
            crosses = endpoints & (low != high) & (pressure[:-1] < critical_pressure)
            valid_cells &= ~(crosses[:, :-1] | crosses[:, 1:])
    return numerical, availability, jnp.asarray(valid_cells)


def inspection_view(fluid):
    """Return independent dictionaries with current authoritative coefficients."""
    original = fluid._inspection.table
    result = {"metadata": fluid.metadata}
    for name, entry in original.items():
        if name == "phase_cell_validity":
            result[name] = entry
            continue
        if name == "saturation_props":
            result[name] = (
                {
                    side: {key: dict(value) for key, value in fields.items()}
                    for side, fields in entry.items()
                }
                if set(entry) == {"liquid", "vapor"}
                else {key: dict(value) for key, value in entry.items()}
            )
            continue
        current = dict(entry)
        if name in fluid._table:
            runtime = fluid._table[name]
            valid = fluid._coefficient_validity[name]
            current["coeffs"] = (
                runtime["coeffs"]
                if valid is None
                else jnp.where(valid[..., None], runtime["coeffs"], entry["coeffs"])
            )
            if "value" in runtime:
                current["value"] = runtime["value"]
        result[name] = current
    return result


def prepare_saturation(table, properties):
    payload = table["saturation_props"]
    if set(payload) == {"liquid", "vapor"}:
        prepared = {
            side: prepare_saturation({"saturation_props": entry}, properties)
            for side, entry in payload.items()
        }
        return {side: entry[0] for side, entry in prepared.items()}, {
            side: entry[1] for side, entry in prepared.items()
        }
    numerical, availability = {}, {}
    for name, entry in table["saturation_props"].items():
        if name not in properties:
            continue
        values, slopes = np.asarray(entry["value"]), np.asarray(entry["grad_p"])
        finite = np.isfinite(values) & np.isfinite(slopes)
        all_available = bool(np.all(finite))
        availability[name] = (
            None if all_available else jnp.asarray(finite[:-1] & finite[1:])
        )
        numerical[name] = {
            "value": (
                entry["value"]
                if all_available
                else jnp.asarray(np.where(finite, values, 0.0))
            ),
            "grad_p": (
                entry["grad_p"]
                if all_available
                else jnp.asarray(np.where(finite, slopes, 0.0))
            ),
        }
    return numerical, availability
