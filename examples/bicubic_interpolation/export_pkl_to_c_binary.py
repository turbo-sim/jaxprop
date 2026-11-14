"""Export a validated JaxProp table to a versioned C-friendly binary payload.

An explicit --input is required. Version 2 retains all node/derivative/bicubic
arrays, the phase-cell guard, complete generation metadata and saturation data.
The raw arrays use little-endian float64 in C order. JSON provenance follows the
boolean phase mask; optional saturation NaNs are represented as JSON null.

The format is a data interchange example, not a replacement fluid evaluator.
A C consumer must honor phase validity, optional NaNs and the declared domain.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options

import json
import pickle
import re
import struct
import numpy as np

MAGIC = b"JXTBIN2\0"
VERSION = 2
ENDIAN_TAG = 0x01020304
NODE_FIELDS = ("value", "grad_h", "grad_p", "grad_logP", "grad_ph", "grad_hlogP")
COEFF_FIELD = "coeffs"
COEFFS_PER_CELL = 16
# Eight uint32 fields followed by mask offset, JSON offset and JSON size.
HEADER_FMT = "<8sIIIIIIIIQQQ"
META_FMT = "<6d"
RECORD_FMT = "<64sQQQQQQQ"


def validate_table(table):
    """Reject incomplete/legacy payloads instead of manufacturing NaN arrays."""
    from jaxprop.bicubic.table_cache import CACHE_FORMAT_VERSION, table_validation_error

    if not isinstance(table, dict) or not isinstance(table.get("metadata"), dict):
        raise ValueError(
            "Input must be a current JaxProp table dictionary with metadata."
        )
    metadata = table["metadata"]
    if metadata.get("cache_format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported table cache schema: {metadata.get('cache_format_version')!r}; expected {CACHE_FORMAT_VERSION}."
        )
    try:
        reason = table_validation_error(table, metadata)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"Incomplete or malformed table schema: {error}") from error
    if reason is not None:
        raise ValueError(f"Invalid table payload: {reason}")
    return metadata


def load_table(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required --input table does not exist: {path}")
    with path.open("rb") as stream:
        table = pickle.load(stream)
    validate_table(table)
    return table


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray) or hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sanitize_c_identifier(name):
    result = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    return "_" + result if result[:1].isdigit() else result


def write_cbin(table, out_bin, out_header):
    """Write complete v2 data and its matching C indexing declarations."""
    metadata = validate_table(table)
    nh, np_ = int(metadata["N_h"]), int(metadata["N_p"])
    props = sorted(
        name
        for name, value in table.items()
        if name not in {"metadata", "saturation_props", "phase_cell_validity"}
    )
    arrays = {
        name: {
            field: np.ascontiguousarray(table[name][field], dtype="<f8")
            for field in (*NODE_FIELDS, COEFF_FIELD)
        }
        for name in props
    }
    cursor = (
        struct.calcsize(HEADER_FMT)
        + struct.calcsize(META_FMT)
        + len(props) * struct.calcsize(RECORD_FMT)
    )
    records = []
    for name in props:
        offsets = []
        for field in (*NODE_FIELDS, COEFF_FIELD):
            offsets.append(cursor)
            cursor += arrays[name][field].nbytes
        records.append((name, offsets))
    phase_mask = np.ascontiguousarray(table["phase_cell_validity"], dtype=np.uint8)
    phase_offset = cursor
    json_offset = phase_offset + phase_mask.nbytes
    provenance = json.dumps(
        _json_value(
            dict(metadata=metadata, saturation_props=table["saturation_props"])
        ),
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    out_bin, out_header = Path(out_bin), Path(out_header)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_header.parent.mkdir(parents=True, exist_ok=True)
    with out_bin.open("wb") as stream:
        stream.write(
            struct.pack(
                HEADER_FMT,
                MAGIC,
                VERSION,
                ENDIAN_TAG,
                nh,
                np_,
                len(props),
                len(NODE_FIELDS),
                COEFFS_PER_CELL,
                0,
                phase_offset,
                json_offset,
                len(provenance),
            )
        )
        stream.write(
            struct.pack(
                META_FMT,
                *(
                    float(metadata[key])
                    for key in (
                        "h_min",
                        "h_max",
                        "p_min",
                        "p_max",
                        "delta_h",
                        "delta_logP",
                    )
                ),
            )
        )
        for name, offsets in records:
            encoded = name.encode("ascii")
            if len(encoded) >= 64:
                raise ValueError(f"Property name exceeds C directory width: {name}")
            stream.write(struct.pack(RECORD_FMT, encoded.ljust(64, b"\0"), *offsets))
        for name, _ in records:
            for field in (*NODE_FIELDS, COEFF_FIELD):
                stream.write(arrays[name][field].tobytes(order="C"))
        stream.write(phase_mask.tobytes(order="C"))
        stream.write(provenance)
    enums = "\n".join(
        f"    JX_PROP_{_sanitize_c_identifier(name)} = {index},"
        for index, name in enumerate(props)
    )
    header = f"""#ifndef JAXPROP_TABLE_LAYOUT_H
#define JAXPROP_TABLE_LAYOUT_H
#include <stdint.h>
#include <stddef.h>
#define JX_MAGIC_EXPECTED "JXTBIN2"
#define JX_VERSION_EXPECTED {VERSION}u
#define JX_NODE_FIELDS {len(NODE_FIELDS)}u
#define JX_COEFFS_PER_CELL 16u
/* Arrays are little-endian float64. Never evaluate a cell whose mask is zero. */
#define JX_NODE_IDX(i,j,NP) ((size_t)(i)*(size_t)(NP)+(size_t)(j))
#define JX_COEFF_IDX(i,j,k,NP_MINUS_1) ((((size_t)(i)*(size_t)(NP_MINUS_1)+(size_t)(j))*16u)+(size_t)(k))
#define JX_CELL_IDX(i,j,NP_MINUS_1) ((size_t)(i)*(size_t)(NP_MINUS_1)+(size_t)(j))
typedef struct {{
    char name[64];
    uint64_t off_value, off_grad_h, off_grad_p, off_grad_logP;
    uint64_t off_grad_ph, off_grad_hlogP, off_coeffs;
}} jx_prop_record_t;
enum jx_property_id {{
{enums}
    JX_PROP_COUNT = {len(props)}
}};
#endif
"""
    out_header.write_text(header, encoding="ascii")
    return dict(
        format_version=VERSION,
        nodes=[nh, np_],
        properties=len(props),
        binary_bytes=out_bin.stat().st_size,
        phase_mask_offset=phase_offset,
        provenance_offset=json_offset,
    )


def read_cbin(path):
    """Read v2 back into its declared original numerical dtype and schema."""
    payload = Path(path).read_bytes()
    header_size, meta_size, rec_size = map(
        struct.calcsize, (HEADER_FMT, META_FMT, RECORD_FMT)
    )
    if len(payload) < header_size + meta_size:
        raise ValueError("Truncated C binary header")
    (
        magic,
        version,
        endian,
        nh,
        np_,
        nprops,
        nfields,
        ncoeff,
        _,
        phase_offset,
        json_offset,
        json_size,
    ) = struct.unpack_from(HEADER_FMT, payload)
    if magic != MAGIC or version != VERSION or endian != ENDIAN_TAG:
        raise ValueError("Unsupported C binary magic, version or endian tag")
    if nh < 2 or np_ < 2 or nfields != len(NODE_FIELDS) or ncoeff != 16:
        raise ValueError("Invalid C binary grid or field schema")
    if (
        json_offset + json_size != len(payload)
        or phase_offset + (nh - 1) * (np_ - 1) != json_offset
    ):
        raise ValueError("Truncated or inconsistent C binary payload offsets")
    provenance = json.loads(payload[json_offset:].decode("utf-8"))
    metadata = provenance["metadata"]
    # C consumers use these packed coordinates directly, so the JSON record
    # must describe the same grid even when the array payload remains intact.
    packed_coordinates = struct.unpack_from(META_FMT, payload, header_size)
    expected_coordinates = tuple(
        float(metadata[key])
        for key in ("h_min", "h_max", "p_min", "p_max", "delta_h", "delta_logP")
    )
    if packed_coordinates != expected_coordinates:
        raise ValueError("C binary coordinate header disagrees with JSON metadata")
    dtype = np.dtype(metadata["numeric_dtype"])
    table = {
        "metadata": metadata,
        "saturation_props": {
            name: {
                field: np.asarray(value, dtype=dtype) for field, value in entry.items()
            }
            for name, entry in provenance["saturation_props"].items()
        },
    }
    mask = np.frombuffer(
        payload, dtype=np.uint8, count=(nh - 1) * (np_ - 1), offset=phase_offset
    )
    if np.any((mask != 0) & (mask != 1)):
        raise ValueError("Invalid C binary phase-cell flags")
    table["phase_cell_validity"] = mask.astype(bool).reshape(nh - 1, np_ - 1)
    directory = header_size + meta_size
    data_start = directory + nprops * rec_size
    if nprops < 1 or nprops > 1000 or data_start > phase_offset:
        raise ValueError("Invalid or truncated C property directory")
    for index in range(nprops):
        record = struct.unpack_from(RECORD_FMT, payload, directory + index * rec_size)
        name = record[0].split(b"\0", 1)[0].decode("ascii")
        entry = {}
        for field, offset in zip((*NODE_FIELDS, COEFF_FIELD), record[1:]):
            shape = (nh - 1, np_ - 1, 16) if field == COEFF_FIELD else (nh, np_)
            count = int(np.prod(shape))
            if offset < data_start or offset + count * 8 > phase_offset:
                raise ValueError(f"Invalid array offset: {name}.{field}")
            entry[field] = (
                np.frombuffer(payload, dtype="<f8", count=count, offset=offset)
                .astype(dtype)
                .reshape(shape)
            )
        if name in table:
            raise ValueError(f"Duplicate C property directory entry: {name}")
        table[name] = entry
    validate_table(table)
    return table


def assert_roundtrip(original, reconstructed):
    from jaxprop.bicubic.cache_metadata import metadata_fingerprint

    assert metadata_fingerprint(original["metadata"]) == metadata_fingerprint(
        reconstructed["metadata"]
    )
    np.testing.assert_array_equal(
        original["phase_cell_validity"], reconstructed["phase_cell_validity"]
    )
    count = 1
    for name, value in original.items():
        if name in {"metadata", "phase_cell_validity"}:
            continue
        if name == "saturation_props":
            for prop, entry in value.items():
                for field, array in entry.items():
                    np.testing.assert_array_equal(
                        array, reconstructed[name][prop][field]
                    )
                    count += 1
        else:
            for field, array in value.items():
                np.testing.assert_array_equal(array, reconstructed[name][field])
                count += 1
    return count


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    if options.input_path is None:
        raise ValueError(
            "This export tool requires --input PATH to an existing current-schema table."
        )
    table = load_table(options.input_path)
    out_bin = options.output_dir / (options.input_path.stem + ".cbin")
    out_header = options.output_dir / (options.input_path.stem + "_layout.h")
    result = write_cbin(table, out_bin, out_header)
    result["roundtrip_arrays_checked"] = assert_roundtrip(table, read_cbin(out_bin))
    options.record("export", result)
    print(f"Wrote and round-trip verified {out_bin}")
    return options.finish()


if __name__ == "__main__":
    main()
