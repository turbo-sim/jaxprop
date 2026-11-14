"""Shared structural validation and atomic persistence for bicubic tables."""

from pathlib import Path
import os
import pickle
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from ..property_schema import PROPERTIES_CANONICAL
from .cache_metadata import metadata_fingerprint
from .thermal_stability import metastable_table_error

CACHE_FORMAT_VERSION = 2
GENERATOR_REVISION = 3
TABLE_ARRAYS = (
    "value",
    "grad_h",
    "grad_p",
    "grad_logP",
    "grad_ph",
    "grad_hlogP",
    "coeffs",
)
CORE_PROPERTIES = (
    "pressure",
    "temperature",
    "density",
    "enthalpy",
    "entropy",
    "internal_energy",
)


def table_validation_error(table, metadata):
    """Return a readable rejection reason, preserving optional NaN properties."""
    if not isinstance(table, dict):
        return "table payload is not a dictionary"
    try:
        matching_metadata = metadata_fingerprint(
            table.get("metadata")
        ) == metadata_fingerprint(metadata)
    except (TypeError, ValueError):
        return "generation metadata is not a valid canonical manifest"
    if not matching_metadata:
        return "generation metadata does not match exactly"
    expected_keys = set(PROPERTIES_CANONICAL) | {
        "metadata",
        "saturation_props",
        "phase_cell_validity",
    }
    if set(table) != expected_keys:
        return "table property schema is incomplete or contains unknown entries"
    shape = (metadata["N_h"], metadata["N_p"])
    coefficient_shape = (shape[0] - 1, shape[1] - 1, 16)
    dtype = np.dtype(metadata["numeric_dtype"])
    phase_cells = np.asarray(table["phase_cell_validity"])
    if phase_cells.shape != coefficient_shape[:2] or phase_cells.dtype != np.bool_:
        return "phase-cell validity: incorrect shape or boolean dtype"
    for name in PROPERTIES_CANONICAL:
        entry = table[name]
        if not isinstance(entry, dict) or set(entry) != set(TABLE_ARRAYS):
            return f"{name}: missing or unknown numerical arrays"
        for key in TABLE_ARRAYS:
            try:
                array = np.asarray(entry[key])
            except (TypeError, ValueError):
                return f"{name}.{key}: not a numerical array"
            expected_shape = coefficient_shape if key == "coeffs" else shape
            if array.shape != expected_shape or array.dtype != dtype:
                return f"{name}.{key}: incorrect shape or dtype"
            if np.any(np.isinf(array)):
                return f"{name}.{key}: contains infinity"
            if name in CORE_PROPERTIES and not np.all(np.isfinite(array)):
                return f"{name}.{key}: incomplete or nonfinite core data"
    h = np.asarray(table["enthalpy"]["value"])
    p = np.asarray(table["pressure"]["value"])
    expected_h = np.linspace(
        metadata["h_min"], metadata["h_max"], shape[0], dtype=dtype
    )
    expected_p = np.exp(
        np.linspace(
            np.log(metadata["p_min"]), np.log(metadata["p_max"]), shape[1], dtype=dtype
        )
    )
    tolerance = max(1e-9, 64 * np.finfo(dtype).eps)
    if not np.allclose(h, expected_h[:, None], rtol=tolerance, atol=1e-6):
        return "enthalpy nodes do not match the requested grid"
    if not np.allclose(p, expected_p[None, :], rtol=tolerance, atol=1e-6):
        return "pressure nodes do not match the requested grid"
    if np.any(np.asarray(table["temperature"]["value"]) <= 0) or np.any(
        np.asarray(table["density"]["value"]) <= 0
    ):
        return "temperature and density must be positive"
    source_domain = metadata.get("source_domain")
    if source_domain is not None:
        temperature = np.asarray(table["temperature"]["value"])
        boundary_precision = 1e-9
        if (
            np.any(temperature < source_domain["T_min"] * (1 - boundary_precision))
            or np.any(temperature > source_domain["T_max"] * (1 + boundary_precision))
            or np.any(p > source_domain["p_max"] * (1 + boundary_precision))
        ):
            return "table nodes exceed the declared source EOS domain"
    saturation = table["saturation_props"]
    if not isinstance(saturation, dict):
        return "saturation payload is not a dictionary"
    if metadata["table_mode"] in ("equilibrium", "mixture"):
        if saturation:
            return "ordinary tables must not contain an undeclared saturation mode"
    else:
        if set(saturation) != set(PROPERTIES_CANONICAL):
            return "saturation property schema is incomplete"
        for name, entry in saturation.items():
            if not isinstance(entry, dict) or set(entry) != {"value", "grad_p"}:
                return f"saturation {name}: incomplete arrays"
            for key, value in entry.items():
                array = np.asarray(value)
                if array.shape != (metadata["N_p_sat"],) or array.dtype != dtype:
                    return f"saturation {name}.{key}: incorrect shape or dtype"
                if np.any(np.isinf(array)) or (
                    name in CORE_PROPERTIES and not np.all(np.isfinite(array))
                ):
                    return f"saturation {name}.{key}: incomplete or nonfinite data"
        settings = metadata["generation_settings"]
        log_min = settings["saturation_log_pressure_min"]
        log_max = settings["saturation_log_pressure_max"]
        if (
            log_min is None
            or log_max is None
            or not np.isfinite(log_min)
            or not np.isfinite(log_max)
            or not log_min < log_max
        ):
            return "saturation pressure grid is not declared correctly"
        expected_sat_p = np.exp(
            np.linspace(log_min, log_max, metadata["N_p_sat"], dtype=dtype)
        )
        sat_p = np.asarray(saturation["pressure"]["value"])
        if not np.allclose(sat_p, expected_sat_p, rtol=tolerance, atol=1e-6):
            return (
                "saturation pressure nodes do not match the declared log-pressure grid"
            )
        if source_domain is not None:
            sat_temperature = np.asarray(saturation["temperature"]["value"])
            if (
                np.any(
                    sat_temperature < source_domain["T_min"] * (1 - boundary_precision)
                )
                or np.any(
                    sat_temperature > source_domain["T_max"] * (1 + boundary_precision)
                )
                or np.any(sat_p > source_domain["p_max"] * (1 + boundary_precision))
            ):
                return "saturation nodes exceed the declared source EOS domain"
    if metadata["table_mode"] in ("metastable_liquid", "metastable_vapor"):
        return metastable_table_error(table, metadata)
    return None


def normalize_generated_arrays(table, metadata):
    """Give every generation mode the same declared floating storage dtype."""
    dtype = np.dtype(metadata["numeric_dtype"])

    def normalized(value):
        # A validated pickle normally restores correctly typed JAX arrays.
        # Avoid redispatching an asarray conversion for every cached field.
        if isinstance(value, jax.Array) and value.dtype == dtype:
            return value
        return jnp.asarray(value, dtype=dtype)

    # Do not repair a missing schema: validation must still reject that table.
    if not isinstance(table, dict):
        return table
    for name in PROPERTIES_CANONICAL:
        if isinstance(table.get(name), dict):
            for key in TABLE_ARRAYS:
                if key in table[name]:
                    table[name][key] = normalized(table[name][key])
    if isinstance(table.get("saturation_props"), dict):
        for entry in table["saturation_props"].values():
            if isinstance(entry, dict):
                for key in ("value", "grad_p"):
                    if key in entry:
                        entry[key] = normalized(entry[key])
    if "phase_cell_validity" in table:
        mask = table["phase_cell_validity"]
        if not isinstance(mask, jax.Array) or mask.dtype != np.bool_:
            table["phase_cell_validity"] = jnp.asarray(mask, dtype=bool)
    return table


def save_table_atomic(table, path):
    """Publish a complete pickle in one replacement; preserve any prior file."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(table, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
