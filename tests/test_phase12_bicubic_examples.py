"""Scientific example domains and complete table interchange regressions."""

from copy import deepcopy
import ctypes
import csv
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

import CoolProp.CoolProp as CP
import numpy as np
import pytest

import jaxprop as jp
from examples.bicubic_interpolation import convert_pkl_to_csv as csv_export
from examples.bicubic_interpolation import export_pkl_to_c_binary as binary_export


@pytest.fixture(scope="module", params=("equilibrium", "metastable_liquid"))
def actual_table(request, tmp_path_factory):
    options = dict(
        N_h=4,
        N_p=4,
        N_p_sat=4,
        coarse_step=1,
        table_dir=str(tmp_path_factory.mktemp(request.param)),
    )
    if request.param == "equilibrium":
        name = "CO2"
        options.update(h_min=500e3, h_max=700e3, p_min=2e6, p_max=10e6)
    else:
        name = "Nitrogen"
        options.update(
            h_min=-120e3,
            h_max=-110e3,
            p_min=8e4,
            p_max=2e5,
            metastable_phase="liquid",
            gradient_method="forward",
        )
    table = deepcopy(jp.Fluid(name, method="bicubic", table_options=options).table)
    # An additional conservative exclusion must survive the interchange too.
    validity = np.asarray(table["phase_cell_validity"]).copy()
    validity[0, 0] = False
    table["phase_cell_validity"] = validity
    return table


def test_c_binary_roundtrip_preserves_every_array_guard_and_provenance(
    actual_table, tmp_path
):
    path, header = tmp_path / "table.cbin", tmp_path / "table_layout.h"
    summary = binary_export.write_cbin(actual_table, path, header)
    restored = binary_export.read_cbin(path)
    arrays = binary_export.assert_roundtrip(actual_table, restored)
    assert summary["format_version"] == 2
    assert arrays == 1 + 7 * len(jp.PROPERTIES_CANONICAL) + 2 * len(
        actual_table["saturation_props"]
    )
    assert not restored["phase_cell_validity"][0, 0]
    assert b"JXTBIN2\0" == path.read_bytes()[:8]
    assert "never evaluate a cell" in header.read_text().lower()


def test_binary_offsets_and_host_c_header_layout_are_independently_consistent(
    actual_table, tmp_path
):
    class CHeader(ctypes.LittleEndianStructure):
        _fields_ = (
            [("magic", ctypes.c_char * 8)]
            + [
                (name, ctypes.c_uint32)
                for name in (
                    "version",
                    "endian",
                    "nh",
                    "np",
                    "properties",
                    "node_fields",
                    "coefficients",
                    "reserved",
                )
            ]
            + [
                (name, ctypes.c_uint64)
                for name in ("phase_offset", "json_offset", "json_size")
            ]
        )

    path = tmp_path / "table.cbin"
    binary_export.write_cbin(actual_table, path, tmp_path / "layout.h")
    payload = path.read_bytes()
    header = CHeader.from_buffer_copy(payload[: ctypes.sizeof(CHeader)])
    assert ctypes.sizeof(CHeader) == 64
    assert header.phase_offset + (header.nh - 1) * (header.np - 1) == header.json_offset
    assert header.json_offset + header.json_size == len(payload)
    assert payload[header.phase_offset] == 0
    provenance = json.loads(payload[header.json_offset :])
    assert provenance["metadata"]["fluid"] == actual_table["metadata"]["fluid"]
    # Read one directory/array entry independently of the Python reader.
    directory = ctypes.sizeof(CHeader) + 6 * 8
    name, *offsets = struct.unpack_from("<64s7Q", payload, directory)
    name = name.split(b"\0", 1)[0].decode()
    shape = (header.nh, header.np)
    values = np.frombuffer(
        payload, dtype="<f8", count=header.nh * header.np, offset=offsets[0]
    ).reshape(shape)
    np.testing.assert_array_equal(values, actual_table[name]["value"])


def test_actual_c_reader_matches_exported_values_and_phase_mask(actual_table, tmp_path):
    requested = os.environ.get("JAXPROP_TEST_CC")
    compiler = (
        Path(requested)
        if requested
        else shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    )
    if compiler is None:
        pytest.skip(
            "Optional native C interoperability check: set JAXPROP_TEST_CC or provide cc/gcc/clang"
        )
    compiler = Path(compiler).resolve()
    assert compiler.is_file(), f"Requested C compiler is unavailable: {compiler}"
    binary, header = tmp_path / "fixture.cbin", tmp_path / "fixture_layout.h"
    binary_export.write_cbin(actual_table, binary, header)
    source = (
        Path(__file__).resolve().parents[1]
        / "examples/bicubic_interpolation/test_cbin_reader.c"
    )
    executable = tmp_path / ("reader.exe" if os.name == "nt" else "reader")
    environment = os.environ.copy()
    # Restrict compiler/DLL lookup changes to the native subprocesses; changing
    # the Python process PATH can otherwise alter source backend DLL resolution.
    environment["PATH"] = (
        str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    )
    built = subprocess.run(
        [
            str(compiler),
            "-O2",
            "-Wall",
            "-Wextra",
            "-std=c11",
            '-DJX_LAYOUT_HEADER="fixture_layout.h"',
            "-I",
            str(tmp_path),
            str(source),
            "-o",
            str(executable),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert built.returncode == 0, built.stderr
    result = subprocess.run(
        [str(executable), str(binary)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    row = json.loads(
        re.search(r"^JX_SAMPLE (.+)$", result.stdout, flags=re.MULTILINE).group(1)
    )
    name, i, j, ci, cj, k = (
        row[key] for key in ("property", "node_i", "node_j", "cell_i", "cell_j", "k")
    )
    for field in ("value", "grad_h", "grad_logP"):
        assert row[field] == float(actual_table[name][field][i, j])
    assert row["coefficient"] == float(actual_table[name]["coeffs"][ci, cj, k])
    assert bool(row["cell_valid"]) == bool(actual_table["phase_cell_validity"][ci, cj])


def test_csv_roundtrip_preserves_optional_nans_saturation_and_phase_mask(
    actual_table, tmp_path
):
    columns = csv_export.write_csv_tables(actual_table, tmp_path)
    restored = csv_export.read_csv_tables(tmp_path)
    binary_export.assert_roundtrip(actual_table, restored)
    assert (
        len(columns["nodes"]["i"])
        == actual_table["metadata"]["N_h"] * actual_table["metadata"]["N_p"]
    )
    assert (tmp_path / "metadata.json").is_file()
    assert not restored["phase_cell_validity"][0, 0]


def test_binary_and_csv_preserve_declared_float32_storage(actual_table, tmp_path):
    table = deepcopy(actual_table)
    table["metadata"]["numeric_dtype"] = "float32"
    for name, entry in table.items():
        if name in {"metadata", "phase_cell_validity"}:
            continue
        if name == "saturation_props":
            for prop in entry:
                entry[prop] = {
                    field: np.asarray(value, dtype=np.float32)
                    for field, value in entry[prop].items()
                }
        else:
            table[name] = {
                field: np.asarray(value, dtype=np.float32)
                for field, value in entry.items()
            }
    binary_export.validate_table(table)
    path = tmp_path / "single_precision.cbin"
    binary_export.write_cbin(table, path, tmp_path / "layout.h")
    restored = binary_export.read_cbin(path)
    assert restored["temperature"]["value"].dtype == np.float32
    binary_export.assert_roundtrip(table, restored)
    csv_export.write_csv_tables(table, tmp_path)
    binary_export.assert_roundtrip(table, csv_export.read_csv_tables(tmp_path))


def test_optional_excel_export_checks_sheet_and_metadata_roundtrip(
    actual_table, tmp_path
):
    pytest.importorskip("pandas", reason="Optional spreadsheet export dependency")
    pytest.importorskip("openpyxl", reason="Optional spreadsheet export dependency")
    shapes = csv_export.write_excel(actual_table, tmp_path / "inspection.xlsx")
    assert (
        shapes["nodes"][0]
        == actual_table["metadata"]["N_h"] * actual_table["metadata"]["N_p"]
    )
    assert shapes["cells"][0] == (actual_table["metadata"]["N_h"] - 1) * (
        actual_table["metadata"]["N_p"] - 1
    )


@pytest.mark.parametrize("damage", ("missing", "shape", "infinity", "version"))
def test_export_rejects_malformed_input_instead_of_filling_missing_data(
    actual_table, tmp_path, damage
):
    table = deepcopy(actual_table)
    if damage == "missing":
        del table["temperature"]["grad_h"]
    elif damage == "shape":
        table["temperature"]["coeffs"] = np.zeros((1, 1, 16))
    elif damage == "infinity":
        values = np.asarray(table["temperature"]["value"]).copy()
        values[0, 0] = np.inf
        table["temperature"]["value"] = values
    else:
        table["metadata"]["cache_format_version"] = 0
    with pytest.raises(ValueError, match="schema|payload"):
        binary_export.write_cbin(
            table, tmp_path / "invalid.cbin", tmp_path / "invalid.h"
        )
    assert not (tmp_path / "invalid.cbin").exists()


@pytest.mark.parametrize("damage", ("version", "truncated", "guard", "directory"))
def test_binary_reader_rejects_unsupported_or_corrupt_payload(
    actual_table, tmp_path, damage
):
    path = tmp_path / "table.cbin"
    binary_export.write_cbin(actual_table, path, tmp_path / "layout.h")
    payload = bytearray(path.read_bytes())
    if damage == "version":
        struct.pack_into("<I", payload, 8, 1)
    elif damage == "truncated":
        payload = payload[:-1]
    elif damage == "directory":
        struct.pack_into("<I", payload, 24, 0)
    else:
        header = struct.unpack_from(binary_export.HEADER_FMT, payload)
        payload[header[-3]] = 2
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="version|[Tt]runcated|flags|directory"):
        binary_export.read_cbin(path)


@pytest.mark.parametrize("actual_table", ("equilibrium",), indirect=True)
def test_binary_reader_rejects_coordinate_header_that_disagrees_with_metadata(
    actual_table, tmp_path
):
    path = tmp_path / "table.cbin"
    binary_export.write_cbin(actual_table, path, tmp_path / "layout.h")
    original = path.read_bytes()
    coordinate_offset = struct.calcsize(binary_export.HEADER_FMT)
    coordinates = struct.unpack_from(
        binary_export.META_FMT, original, coordinate_offset
    )
    for index, coordinate in enumerate(coordinates):
        payload = bytearray(original)
        # A one-ULP alteration must not be masked by a relative tolerance: the
        # writer stores the exact same six doubles in both representations.
        struct.pack_into(
            "<d",
            payload,
            coordinate_offset + index * 8,
            np.nextafter(coordinate, np.inf),
        )
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="coordinate header.*JSON metadata"):
            binary_export.read_cbin(path)


@pytest.mark.parametrize("damage", ("row_index", "phase_flag"))
def test_csv_reader_rejects_misordered_rows_or_nonbinary_flags(
    actual_table, tmp_path, damage
):
    csv_export.write_csv_tables(actual_table, tmp_path)
    target = tmp_path / ("nodes.csv" if damage == "row_index" else "cells.csv")
    with target.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    column = rows[0].index("i" if damage == "row_index" else "phase_valid")
    rows[1][column] = "1" if damage == "row_index" else "2"
    with target.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError, match="grid|flags"):
        csv_export.read_csv_tables(tmp_path)


@pytest.mark.parametrize(
    "name", ("convert_pkl_to_csv", "export_pkl_to_c_binary", "plot_metastable_point")
)
def test_input_dependent_tools_require_an_explicit_input(name, tmp_path):
    module = importlib.import_module("examples.bicubic_interpolation." + name)
    with pytest.raises(ValueError, match="requires? --input"):
        module.main(["--headless", "--output-dir", str(tmp_path / name)])


@pytest.mark.parametrize("pressure", np.geomspace(1e5, 3e6, 7))
def test_revised_nitrogen_liquid_domain_has_independent_physical_margin(pressure):
    fluid = jp.Fluid("Nitrogen")
    guess = [800.0, 78.0]
    raw = CP.AbstractState("HEOS", "Nitrogen")
    raw.specify_phase(CP.iphase_liquid)
    for enthalpy in np.linspace(-120e3, -40e3, 9):
        state = fluid.get_state_metastable(
            "p",
            pressure,
            "h",
            enthalpy,
            rhoT_guess=guess,
            solver_tolerance=1e-10,
            supersaturation=False,
            generalize_quality=False,
        )
        guess = [float(state.rho), float(state.T)]
        raw.update(CP.DmassT_INPUTS, *guess)
        assert raw.cvmass() > 900.0
        assert 1 / raw.isothermal_compressibility() > 1e7
        np.testing.assert_allclose(raw.p(), pressure, rtol=1e-8)
        np.testing.assert_allclose(raw.hmass(), enthalpy, rtol=1e-9, atol=1e-5)


@pytest.mark.parametrize("pressure", np.geomspace(1e5, 13e6, 5))
def test_revised_water_rectangle_has_a_stable_connected_liquid_branch(pressure):
    # Follow the connected branch in small physical increments. A single jump
    # from saturation to a distant enthalpy is not an independent branch proof.
    fluid = jp.Fluid("Water")
    guess = [990.0, 345.0]
    raw = CP.AbstractState("HEOS", "Water")
    raw.specify_phase(CP.iphase_liquid)
    for enthalpy in np.linspace(300e3, 1400e3, 23):
        state = fluid.get_state_metastable(
            "p",
            pressure,
            "h",
            enthalpy,
            rhoT_guess=guess,
            solver_tolerance=1e-10,
            supersaturation=False,
            generalize_quality=False,
        )
        guess = [float(state.rho), float(state.T)]
        raw.update(CP.DmassT_INPUTS, *guess)
        assert raw.cvmass() > 2000
        assert 1 / raw.isothermal_compressibility() > 1e7
        np.testing.assert_allclose(raw.p(), pressure, rtol=1e-7)
