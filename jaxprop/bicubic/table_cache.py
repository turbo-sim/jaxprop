"""Shared structural validation and atomic persistence for bicubic tables."""

from pathlib import Path
import os
import pickle
import tempfile

import jax.numpy as jnp
import numpy as np

from ..property_schema import PROPERTIES_CANONICAL
from .cache_metadata import metadata_fingerprint

CACHE_FORMAT_VERSION = 1
GENERATOR_REVISION = 1
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
    expected_keys = set(PROPERTIES_CANONICAL) | {"metadata", "saturation_props"}
    if set(table) != expected_keys:
        return "table property schema is incomplete or contains unknown entries"
    shape = (metadata["N_h"], metadata["N_p"])
    coefficient_shape = (shape[0] - 1, shape[1] - 1, 16)
    dtype = np.dtype(metadata["numeric_dtype"])
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
    return None


def normalize_generated_arrays(table, metadata):
    """Give every generation mode the same declared floating storage dtype."""
    dtype = np.dtype(metadata["numeric_dtype"])
    # Do not repair a missing schema: validation must still reject that table.
    if not isinstance(table, dict):
        return table
    for name in PROPERTIES_CANONICAL:
        if isinstance(table.get(name), dict):
            for key in TABLE_ARRAYS:
                if key in table[name]:
                    table[name][key] = jnp.asarray(table[name][key], dtype=dtype)
    if isinstance(table.get("saturation_props"), dict):
        for entry in table["saturation_props"].values():
            if isinstance(entry, dict):
                for key in ("value", "grad_p"):
                    if key in entry:
                        entry[key] = jnp.asarray(entry[key], dtype=dtype)
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
