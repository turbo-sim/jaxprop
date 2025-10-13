import os
import math
import pprint as pp
import numpy as np
import jaxprop as jxp


# ---------------------------
# Helpers
# ---------------------------
def rel_err(a, b):
    """relative error (a - b)/b with safe zero handling."""
    if b == 0.0:
        return 0.0 if a == 0.0 else np.inf * np.sign(a)
    return (a - b) / b


CONSISTENCY_TOLERANCE = 1e-12

# ---------------------------
# Create fluid object
# ---------------------------
outdir = "fluid_tables"
fluid_name = "CO2"
h_min = 500e3  # J/kg
h_max = 1500e3  # J/kg
p_min = 2e6  # Pa
p_max = 20e6  # Pa
N_h = 50
N_p = 50

fluid = jxp.FluidBicubic(
    fluid_name=fluid_name,
    backend="HEOS",
    h_min=h_min,
    h_max=h_max,
    p_min=p_min,
    p_max=p_max,
    N_h=N_h,
    N_p=N_p,
    table_dir=outdir,
)


# ---------------------------
# Compute reference state
# ---------------------------
h_ref = 1205e3  # J/kg
P_ref = 7.75e6  # Pa
ref = fluid.get_state(jxp.HmassP_INPUTS, h_ref, P_ref)


# ------------------------------------------
# Compute properties for different inputs
# ------------------------------------------
tests = [
    (jxp.HmassP_INPUTS, ref["enthalpy"], ref["pressure"]),
    (jxp.PT_INPUTS, ref["pressure"], ref["temperature"]),
    (jxp.HmassSmass_INPUTS, ref["enthalpy"], ref["entropy"]),
    (jxp.PSmass_INPUTS, ref["pressure"], ref["entropy"]),
    (jxp.DmassHmass_INPUTS, ref["density"], ref["enthalpy"]),
    (jxp.DmassP_INPUTS, ref["density"], ref["pressure"]),
    (jxp.DmassT_INPUTS, ref["density"], ref["temperature"]),
    (jxp.DmassSmass_INPUTS, ref["density"], ref["entropy"])
]

col_names = [jxp.INPUT_TYPE_MAP[it] for it, _, _ in tests]

props_to_check = jxp.PROPERTIES_CANONICAL
values_table = {prop: [] for prop in props_to_check}
errors_table = {prop: [] for prop in props_to_check}

for input_type, v1, v2 in tests:
    st = fluid.get_state(input_type, v1, v2)
    for prop in props_to_check:
        values_table[prop].append(st[prop])
        errors_table[prop].append(rel_err(st[prop], ref[prop]))


# ---------------------------
# Print consistency check
# ---------------------------

# Define common utilities
clean_col_names = [name.replace("_INPUTS", "") for name in col_names]
col_w = max(len(c) for c in clean_col_names) + 1
prop_w = max(len(p) for p in props_to_check)
header = f"{'property':<{prop_w}} | " + " | ".join(
    f"{c:>{col_w}s}" for c in clean_col_names
)
width = len(header) + 2


def print_property_matrix(table_dict, fmt=".4e"):
    """Print a table of property values or errors."""
    print(header)
    print("-" * width)
    for prop in props_to_check:
        row = " | ".join(f"{val:+{col_w}{fmt}}" for val in table_dict[prop])
        print(f"{prop:<{prop_w}} | {row}")
    print("-" * width)


# Print calculated values
print("\n" + "-" * width)
print(" Property values")
print("-" * width)
print_property_matrix(values_table, fmt=".4e")

# Print consistency errors
print("\n" + "-" * width)
print(" Relative error")
print("-" * width)
print_property_matrix(errors_table, fmt=".3e")

# Check consistency (ignore NaNs)
violations = [
    (prop, col, err)
    for prop in props_to_check
    for col, err in zip(clean_col_names, errors_table[prop])
    if np.isfinite(err) and abs(err) > CONSISTENCY_TOLERANCE
]

n_total = sum(np.isfinite(errors_table[prop]).sum() for prop in props_to_check)
n_fail = len(violations)

if n_fail == 0:
    print(
        f"\nAll {n_total} finite relative errors are within tolerance (tol={CONSISTENCY_TOLERANCE:g})."
    )
else:
    msg = [
        f"\nConsistency check failed: {n_fail}/{n_total} finite values exceed tol={CONSISTENCY_TOLERANCE:g}",
        "Violations:",
    ]
    msg += [f"  property={p}, input={c}, error={e:.3e}" for p, c, e in violations]
    raise RuntimeError("\n".join(msg))
