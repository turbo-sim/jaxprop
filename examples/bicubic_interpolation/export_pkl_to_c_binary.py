import argparse
import pickle
import re
import struct
from pathlib import Path

import numpy as np


"""
Export jaxprop bicubic table pickle to a C-friendly binary format.

Output files (in fluid_tables/ by default):
- <table_name>.cbin       : binary table
- <table_name>_layout.h   : C header with indexing macros + property enum

Binary layout (little-endian):
1) Fixed header
2) Metadata block
3) Property directory (offset table)
4) Raw array payload (float64, row-major)

Indexing in C (exact):
- node fields (Nh x Np):      idx = i * Np + j
- coeff fields ((Nh-1)x(Np-1)x16): idx = ((i * (Np-1) + j) * 16) + k
"""

MAGIC = b"JXTBIN1\0"  # 8 bytes
VERSION = 1
ENDIAN_TAG = 0x01020304

NODE_FIELDS = [
    "value",
    "grad_h",
    "grad_p",
    "grad_logP",
    "grad_ph",
    "grad_hlogP",
]
COEFF_FIELD = "coeffs"
COEFFS_PER_CELL = 16

HEADER_FMT = "<8sIIIIIIII"  # magic, version, endian, Nh, Np, n_props, n_node_fields, n_coeff, reserved
META_FMT = "<6d"            # h_min, h_max, p_min, p_max, delta_h, delta_logP
# name[64] + 7 offsets (node fields + coeffs)
RECORD_FMT = "<64sQQQQQQQ"


def _sanitize_c_identifier(name: str) -> str:
    x = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if re.match(r"^[0-9]", x):
        x = "_" + x
    return x


def _infer_node_shape(table: dict) -> tuple[int, int]:
    # Prefer metadata if available
    md = table.get("metadata", {}) if isinstance(table.get("metadata", {}), dict) else {}
    nh = md.get("N_h")
    np_ = md.get("N_p")
    if isinstance(nh, (int, np.integer)) and isinstance(np_, (int, np.integer)):
        return int(nh), int(np_)

    # Fallback to enthalpy.value shape
    enthalpy = table.get("enthalpy")
    if isinstance(enthalpy, dict) and "value" in enthalpy:
        arr = np.asarray(enthalpy["value"])
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])

    # Generic fallback
    for v in table.values():
        if isinstance(v, dict) and "value" in v:
            arr = np.asarray(v["value"])
            if arr.ndim == 2:
                return int(arr.shape[0]), int(arr.shape[1])

    raise ValueError("Could not infer (N_h, N_p) from pickle table.")


def _collect_properties(table: dict, node_shape: tuple[int, int]) -> list[str]:
    props = []
    for key, val in table.items():
        if key == "metadata" or not isinstance(val, dict):
            continue
        if "value" not in val:
            continue
        arr = np.asarray(val["value"])
        if arr.shape == node_shape:
            props.append(key)
    return sorted(props)


def _as_f64_c(arr, shape):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"Unexpected shape {arr.shape}, expected {shape}")
    return np.ascontiguousarray(arr)


def _build_arrays(table: dict, props: list[str], nh: int, np_: int):
    cell_shape = (nh - 1, np_ - 1, COEFFS_PER_CELL)
    node_shape = (nh, np_)

    arrays = {}
    for prop in props:
        p = table[prop]
        arrays[prop] = {}

        for field in NODE_FIELDS:
            if field in p:
                arr = np.asarray(p[field], dtype=np.float64)
                if arr.shape == node_shape:
                    arrays[prop][field] = np.ascontiguousarray(arr)
                    continue
            # Missing/mismatched -> fill NaN
            arrays[prop][field] = np.full(node_shape, np.nan, dtype=np.float64)

        if COEFF_FIELD in p:
            carr = np.asarray(p[COEFF_FIELD], dtype=np.float64)
            if carr.shape == cell_shape:
                arrays[prop][COEFF_FIELD] = np.ascontiguousarray(carr)
            else:
                arrays[prop][COEFF_FIELD] = np.full(cell_shape, np.nan, dtype=np.float64)
        else:
            arrays[prop][COEFF_FIELD] = np.full(cell_shape, np.nan, dtype=np.float64)

    return arrays


def _encode_name64(name: str) -> bytes:
    b = name.encode("ascii", errors="ignore")[:63]
    return b + b"\0" * (64 - len(b))


def write_cbin(table: dict, out_bin: Path, out_header: Path):
    nh, np_ = _infer_node_shape(table)
    props = _collect_properties(table, (nh, np_))
    if not props:
        raise ValueError("No properties with node-grid shape found.")

    arrays = _build_arrays(table, props, nh, np_)

    header_size = struct.calcsize(HEADER_FMT)
    meta_size = struct.calcsize(META_FMT)
    rec_size = struct.calcsize(RECORD_FMT)
    n_props = len(props)
    data_offset = header_size + meta_size + n_props * rec_size

    # Build directory with byte offsets
    records = []
    cursor = data_offset
    for prop in props:
        rec = {"name": prop}
        for fld in NODE_FIELDS + [COEFF_FIELD]:
            rec[fld] = cursor
            cursor += arrays[prop][fld].nbytes
        records.append(rec)

    md = table.get("metadata", {}) if isinstance(table.get("metadata", {}), dict) else {}
    h_min = float(md.get("h_min", np.nan))
    h_max = float(md.get("h_max", np.nan))
    p_min = float(md.get("p_min", np.nan))
    p_max = float(md.get("p_max", np.nan))
    delta_h = float(md.get("delta_h", np.nan))
    delta_logp = float(md.get("delta_logP", np.nan))

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with out_bin.open("wb") as f:
        f.write(
            struct.pack(
                HEADER_FMT,
                MAGIC,
                VERSION,
                ENDIAN_TAG,
                nh,
                np_,
                n_props,
                len(NODE_FIELDS),
                COEFFS_PER_CELL,
                0,
            )
        )
        f.write(struct.pack(META_FMT, h_min, h_max, p_min, p_max, delta_h, delta_logp))

        for rec in records:
            f.write(
                struct.pack(
                    RECORD_FMT,
                    _encode_name64(rec["name"]),
                    rec["value"],
                    rec["grad_h"],
                    rec["grad_p"],
                    rec["grad_logP"],
                    rec["grad_ph"],
                    rec["grad_hlogP"],
                    rec["coeffs"],
                )
            )

        for rec in records:
            prop = rec["name"]
            for fld in NODE_FIELDS + [COEFF_FIELD]:
                f.write(arrays[prop][fld].tobytes(order="C"))

    # Emit C header companion
    enum_lines = []
    for i, p in enumerate(props):
        enum_lines.append(f"    JX_PROP_{_sanitize_c_identifier(p)} = {i},")

    header_text = f"""#ifndef JAXPROP_TABLE_LAYOUT_H
#define JAXPROP_TABLE_LAYOUT_H

#include <stdint.h>
#include <stddef.h>

#define JX_MAGIC_EXPECTED \"JXTBIN1\"
#define JX_VERSION_EXPECTED {VERSION}u
#define JX_NODE_FIELDS {len(NODE_FIELDS)}u
#define JX_COEFFS_PER_CELL {COEFFS_PER_CELL}u

/* Exact indexing macros (row-major, C-order) */
#define JX_NODE_IDX(i,j,NP) ((size_t)(i) * (size_t)(NP) + (size_t)(j))
#define JX_COEFF_IDX(i,j,k,NP_MINUS_1) ((((size_t)(i) * (size_t)(NP_MINUS_1) + (size_t)(j)) * (size_t)JX_COEFFS_PER_CELL) + (size_t)(k))

typedef struct {{
    char name[64];
    uint64_t off_value;
    uint64_t off_grad_h;
    uint64_t off_grad_p;
    uint64_t off_grad_logP;
    uint64_t off_grad_ph;
    uint64_t off_grad_hlogP;
    uint64_t off_coeffs;
}} jx_prop_record_t;

enum jx_property_id {{
{chr(10).join(enum_lines)}
    JX_PROP_COUNT = {len(props)}
}};

#endif /* JAXPROP_TABLE_LAYOUT_H */
"""
    out_header.write_text(header_text, encoding="ascii")

    print(f"Wrote binary table : {out_bin}")
    print(f"Wrote C header     : {out_header}")
    print(f"N_h={nh}, N_p={np_}, properties={len(props)}")


def _resolve_input_pickle(table_dir: Path, requested_input: str | None) -> Path:
    available = sorted(table_dir.glob("*.pkl")) if table_dir.exists() else []

    if requested_input:
        candidate = table_dir / requested_input
        if candidate.exists():
            return candidate

        direct = Path(requested_input)
        if direct.is_file():
            return direct

        avail = ", ".join(p.name for p in available) if available else "<none>"
        raise FileNotFoundError(
            f"Input pickle not found: {requested_input}\n"
            f"Available .pkl in {table_dir}: {avail}"
        )

    if not available:
        raise FileNotFoundError(f"No .pkl files found in: {table_dir}")

    print("Available pickle tables:")
    for idx, p in enumerate(available, start=1):
        print(f"  {idx:2d}. {p.name}")

    while True:
        raw = input("Select table by number or filename [1]: ").strip()
        if raw == "":
            return available[0]

        if raw.isdigit():
            k = int(raw)
            if 1 <= k <= len(available):
                return available[k - 1]
            print(f"Invalid index: {k}. Choose 1..{len(available)}")
            continue

        candidate = table_dir / raw
        if candidate.exists():
            return candidate

        print("Invalid selection. Enter a valid index or filename.")


def main():
    parser = argparse.ArgumentParser(description="Export jaxprop pickle table to C-friendly binary.")
    parser.add_argument(
        "--input",
        type=str,
        default="CO2_50x50.pkl",
        help="Input pickle filename. If omitted, script prompts interactively.",
    )
    parser.add_argument("--table-dir", type=str, default=None, help="Directory containing pickle and outputs")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    table_dir = Path(args.table_dir) if args.table_dir else (script_dir / "fluid_tables")

    input_pkl = _resolve_input_pickle(table_dir, args.input)

    with input_pkl.open("rb") as f:
        table = pickle.load(f)

    if not isinstance(table, dict):
        raise TypeError(f"Pickle must contain dict, got {type(table)}")

    out_dir = input_pkl.parent
    out_bin = out_dir / f"{input_pkl.stem}.cbin"
    out_header = out_dir / f"{input_pkl.stem}_layout.h"
    write_cbin(table, out_bin, out_header)


if __name__ == "__main__":
    main()

