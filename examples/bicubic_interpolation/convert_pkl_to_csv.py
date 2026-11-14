"""Export an explicit current-schema table to CSV or an optional Excel workbook.

CSV is the default and preserves complete numeric arrays, cell validity,
saturation data and generation metadata. The tool verifies a CSV round trip.
Use --format xlsx for inspection in Excel (requires pandas and openpyxl);
spreadsheet floating-point formatting is checked at a declared 1e-12 tolerance.
No input file or output directory is inferred from the source-code location.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
from examples.bicubic_interpolation.export_pkl_to_c_binary import (
    NODE_FIELDS,
    _json_value,
    assert_roundtrip,
    load_table,
    validate_table,
)

import csv
import json
import numpy as np


def table_columns(table):
    metadata = validate_table(table)
    nh, np_ = metadata["N_h"], metadata["N_p"]
    properties = sorted(
        name
        for name in table
        if name not in {"metadata", "saturation_props", "phase_cell_validity"}
    )
    i, j = np.indices((nh, np_))
    nodes = {"i": i.ravel(), "j": j.ravel()}
    for name in properties:
        for field in NODE_FIELDS:
            nodes[name + "__" + field] = np.asarray(table[name][field]).ravel()
    i, j = np.indices((nh - 1, np_ - 1))
    cells = {
        "i": i.ravel(),
        "j": j.ravel(),
        "phase_valid": np.asarray(table["phase_cell_validity"], dtype=int).ravel(),
    }
    for name in properties:
        for k in range(16):
            cells[name + f"__c{k:02d}"] = np.asarray(table[name]["coeffs"])[
                ..., k
            ].ravel()
    saturation = {
        "i": (
            np.arange(metadata["N_p_sat"])
            if table["saturation_props"]
            else np.array([], dtype=int)
        )
    }
    for name, entry in table["saturation_props"].items():
        for field, array in entry.items():
            saturation[name + "__" + field] = np.asarray(array)
    return dict(nodes=nodes, cells=cells, saturation=saturation), properties


def _write_csv(path, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(
            zip(*(np.asarray(value).tolist() for value in columns.values()))
        )


def write_csv_tables(table, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    columns, properties = table_columns(table)
    for name, values in columns.items():
        _write_csv(folder / (name + ".csv"), values)
    manifest = dict(
        export_schema=1,
        metadata=table["metadata"],
        properties=properties,
        saturation_properties=list(table["saturation_props"]),
    )
    (folder / "metadata.json").write_text(
        json.dumps(_json_value(manifest), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return columns


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        rows = list(reader)
    return {name: np.asarray([float(row[name]) for row in rows]) for name in names}


def read_csv_tables(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if manifest.get("export_schema") != 1:
        raise ValueError("Unsupported CSV export schema")
    metadata = manifest["metadata"]
    dtype = np.dtype(metadata["numeric_dtype"])
    nh, np_ = metadata["N_h"], metadata["N_p"]
    nodes, cells, saturation = (
        _read_csv(folder / (name + ".csv")) for name in ("nodes", "cells", "saturation")
    )
    node_i, node_j = np.indices((nh, np_))
    cell_i, cell_j = np.indices((nh - 1, np_ - 1))
    for columns, expected_i, expected_j in (
        (nodes, node_i, node_j),
        (cells, cell_i, cell_j),
    ):
        if not np.array_equal(columns["i"], expected_i.ravel()) or not np.array_equal(
            columns["j"], expected_j.ravel()
        ):
            raise ValueError("CSV row indices do not match the declared C-order grid")
    flags = cells["phase_valid"]
    if np.any((flags != 0) & (flags != 1)):
        raise ValueError("CSV phase-cell flags must be exactly zero or one")
    table = {
        "metadata": metadata,
        "saturation_props": {},
        "phase_cell_validity": cells["phase_valid"]
        .astype(bool)
        .reshape(nh - 1, np_ - 1),
    }
    for name in manifest["properties"]:
        table[name] = {
            field: nodes[name + "__" + field].astype(dtype).reshape(nh, np_)
            for field in NODE_FIELDS
        }
        table[name]["coeffs"] = (
            np.stack([cells[name + f"__c{k:02d}"] for k in range(16)], axis=-1)
            .astype(dtype)
            .reshape(nh - 1, np_ - 1, 16)
        )
    for name in manifest["saturation_properties"]:
        table["saturation_props"][name] = {
            field: saturation[name + "__" + field].astype(dtype)
            for field in ("value", "grad_p")
        }
    validate_table(table)
    return table


def write_excel(table, path):
    try:
        import pandas as pd
        import openpyxl  # noqa: F401 -- make the optional dependency explicit
    except ImportError as error:
        raise ImportError(
            "Excel export requires the optional pandas and openpyxl packages; CSV export uses the standard library and NumPy."
        ) from error
    columns, _ = table_columns(table)
    frames = {name: pd.DataFrame(values) for name, values in columns.items()}
    metadata = json.dumps(_json_value(table["metadata"]), allow_nan=False)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        pd.DataFrame({"json": [metadata]}).to_excel(
            writer, sheet_name="metadata", index=False
        )
    for name, frame in frames.items():
        restored = pd.read_excel(path, sheet_name=name)
        assert list(restored.columns) == list(frame.columns)
        np.testing.assert_allclose(
            restored.to_numpy(dtype=float),
            frame.to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    assert json.loads(
        pd.read_excel(path, sheet_name="metadata")["json"][0]
    ) == json.loads(metadata)
    return {name: list(frame.shape) for name, frame in frames.items()}


def configure_parser(parser):
    parser.add_argument(
        "--format",
        choices=("csv", "xlsx"),
        default="csv",
        help="CSV is a lossless numeric round trip; XLSX is an optional inspection workbook.",
    )


def main(argv=None):
    options = example_options(
        __file__, argv, description=__doc__, configure_parser=configure_parser
    )
    if options.input_path is None:
        raise ValueError(
            "This converter requires --input PATH to a current-schema property table."
        )
    table = load_table(options.input_path)
    if options.args.format == "csv":
        columns = write_csv_tables(table, options.output_dir)
        arrays = assert_roundtrip(table, read_csv_tables(options.output_dir))
        result = dict(
            format="csv",
            roundtrip_arrays_checked=arrays,
            rows={
                name: len(next(iter(values.values())))
                for name, values in columns.items()
            },
        )
    else:
        path = options.output_dir / (options.input_path.stem + ".xlsx")
        result = dict(
            format="xlsx",
            sheets=write_excel(table, path),
            roundtrip_relative_tolerance=1e-12,
        )
    options.record("export", result)
    return options.finish()


if __name__ == "__main__":
    main()
