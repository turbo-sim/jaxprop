import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1) FILE CONFIGURATION
# ============================================================
# Set only the pickle file name. The script reads from fluid_tables/
# and writes one Excel workbook back into fluid_tables/.
INPUT_PKL_NAME = "CO2_200x200.pkl"

SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR / "fluid_tables"

input_pkl = TABLE_DIR / INPUT_PKL_NAME
output_excel = TABLE_DIR / f"{input_pkl.stem}.xlsx"


# ============================================================
# 2) HELPERS
# ============================================================
def find_node_shape(table_dict):
    """Find the 2D node-grid shape (N_h, N_p) from property value arrays."""
    if (
        "enthalpy" in table_dict
        and isinstance(table_dict["enthalpy"], dict)
        and "value" in table_dict["enthalpy"]
    ):
        arr = np.asarray(table_dict["enthalpy"]["value"])
        if arr.ndim == 2:
            return arr.shape

    for v in table_dict.values():
        if isinstance(v, dict) and "value" in v:
            arr = np.asarray(v["value"])
            if arr.ndim == 2:
                return arr.shape

    raise ValueError("Could not infer node-grid shape from pickle data.")


def collect_properties_with_shape(table_dict, node_shape):
    """Return top-level property keys whose value arrays match node_shape."""
    props = []
    for key, val in table_dict.items():
        if key == "metadata" or not isinstance(val, dict) or "value" not in val:
            continue
        arr = np.asarray(val["value"])
        if arr.shape == node_shape:
            props.append(key)
    return props


# ============================================================
# 3) LOAD PICKLE
# ============================================================
if not input_pkl.exists():
    available = sorted(TABLE_DIR.glob("*.pkl")) if TABLE_DIR.exists() else []
    available_names = ", ".join(p.name for p in available) if available else "<none>"
    raise FileNotFoundError(
        f"Input file not found: {input_pkl}\n"
        f"Available .pkl files in {TABLE_DIR}: {available_names}"
    )

with input_pkl.open("rb") as f:
    data = pickle.load(f)

if not isinstance(data, dict):
    raise TypeError(f"Expected pickle to contain dict, got {type(data)}")

print("Input pickle:", input_pkl)
print("Top-level keys:", list(data.keys()))


# ============================================================
# 4) BUILD NODE TABLE (values + gradients at grid nodes)
# ============================================================
node_shape = find_node_shape(data)
N_h, N_p = node_shape
print(f"Node-grid shape: (N_h, N_p) = ({N_h}, {N_p})")

node_props = collect_properties_with_shape(data, node_shape)
node_fields = ["value", "grad_h", "grad_p", "grad_logP", "grad_ph", "grad_hlogP"]

ii, jj = np.indices(node_shape)
node_columns = {
    "i": ii.ravel(),
    "j": jj.ravel(),
}

if "enthalpy" in data and isinstance(data["enthalpy"], dict) and "value" in data["enthalpy"]:
    h_arr = np.asarray(data["enthalpy"]["value"])
    if h_arr.shape == node_shape:
        node_columns["h"] = h_arr.ravel()

if "pressure" in data and isinstance(data["pressure"], dict) and "value" in data["pressure"]:
    p_arr = np.asarray(data["pressure"]["value"])
    if p_arr.shape == node_shape:
        node_columns["p"] = p_arr.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            node_columns["logP"] = np.log(p_arr).ravel()

for prop in node_props:
    p_dict = data[prop]
    for field in node_fields:
        if field not in p_dict:
            continue
        arr = np.asarray(p_dict[field])
        if arr.shape != node_shape:
            continue
        node_columns[f"{prop}_{field}"] = arr.ravel()

nodes_df = pd.DataFrame(node_columns)
print("Nodes table shape:", nodes_df.shape)


# ============================================================
# 5) BUILD CELL TABLE (16 bicubic coeffs per cell)
# ============================================================
cell_shape = (N_h - 1, N_p - 1)
ci, cj = np.indices(cell_shape)
cell_columns = {
    "cell_i": ci.ravel(),
    "cell_j": cj.ravel(),
}

coeff_prop_count = 0
for prop, p_dict in data.items():
    if prop == "metadata" or not isinstance(p_dict, dict) or "coeffs" not in p_dict:
        continue

    coeffs = np.asarray(p_dict["coeffs"])
    if coeffs.ndim != 3:
        continue
    if coeffs.shape[0] != cell_shape[0] or coeffs.shape[1] != cell_shape[1]:
        continue

    coeff_prop_count += 1
    n_coeff = coeffs.shape[2]
    for m in range(n_coeff):
        cell_columns[f"{prop}_c{m:02d}"] = coeffs[:, :, m].ravel()

cells_df = pd.DataFrame(cell_columns)
print("Cells table shape:", cells_df.shape)
print("Properties with coeffs exported:", coeff_prop_count)


# ============================================================
# 6) SAVE ONE EXCEL FILE (nodes + cells sheets)
# ============================================================
TABLE_DIR.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    nodes_df.to_excel(writer, sheet_name="nodes", index=False)
    cells_df.to_excel(writer, sheet_name="cells", index=False)

print(f"Done! Saved Excel workbook with sheets [nodes, cells]: {output_excel}")
