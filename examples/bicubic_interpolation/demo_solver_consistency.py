import os
import time
import numpy as np
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt


# ---------------------------
# Helpers
# ---------------------------
def l2_absolute_error(a, b):
    """RMS absolute error over the entire mesh."""
    diff = np.ravel(a - b)
    return np.linalg.norm(diff, ord=2) / np.sqrt(diff.size)

def l2_relative_error(a, b):
    """Relative L² error over the entire mesh."""
    eps = 1e-16
    diff = np.ravel(a - b)
    denom = np.ravel(b)
    num = np.linalg.norm(diff, ord=2)
    den = np.linalg.norm(denom, ord=2)
    return num / max(den, eps)

CONSISTENCY_TOLERANCE = 1e-12

# ---------------------------
# Create fluid object
# ---------------------------
outdir = "fluid_tables"
fluid_name = "CO2"
backend = "HEOS"
h_min = 500e3  # J/kg
h_max = 1500e3  # J/kg
p_min = 2e6  # Pa
p_max = 20e6  # Pa
N_h = 32
N_p = 32

fluid = jxp.FluidBicubic(
    fluid_name=fluid_name,
    backend=backend,
    h_min=h_min,
    h_max=h_max,
    p_min=p_min,
    p_max=p_max,
    N_h=N_h,
    N_p=N_p,
    table_dir=outdir,
    table_name="solver_consistency"
)

# ---------------------------
# Reference values at midpoints
# ---------------------------
h_nodes = jnp.linspace(h_min, h_max, N_h + 1)
logp_nodes = jnp.linspace(jnp.log(p_min), jnp.log(p_max), N_p + 1)
h_val = 0.5 * (h_nodes[:-1] + h_nodes[1:])
logp_val = 0.5 * (logp_nodes[:-1] + logp_nodes[1:])
p_val = jnp.exp(logp_val)
h_ref, p_ref = jnp.meshgrid(h_val, p_val, indexing="ij")
state_ref = fluid.get_state(jxp.HmassP_INPUTS, h_ref, p_ref)

# h_ref = jnp.asarray(1205e3)  # J/kg
# p_ref = jnp.asarray(7.75e6)  # Pa
# state_ref = fluid.get_state(jxp.HmassP_INPUTS, h_ref, p_ref)


# ------------------------------------------
# Compute properties for different inputs
# ------------------------------------------
tests = [
    (jxp.HmassP_INPUTS, state_ref["enthalpy"], state_ref["pressure"]),
    (jxp.PT_INPUTS, state_ref["pressure"], state_ref["temperature"]),
    (jxp.HmassSmass_INPUTS, state_ref["enthalpy"], state_ref["entropy"]),
    (jxp.PSmass_INPUTS, state_ref["pressure"], state_ref["entropy"]),
    (jxp.DmassHmass_INPUTS, state_ref["density"], state_ref["enthalpy"]),
    (jxp.DmassP_INPUTS, state_ref["density"], state_ref["pressure"]),
    (jxp.DmassT_INPUTS, state_ref["density"], state_ref["temperature"]),
    (jxp.DmassSmass_INPUTS, state_ref["density"], state_ref["entropy"])
]

col_names = [jxp.INPUT_TYPE_MAP[it] for it, _, _ in tests]
props_to_check = jxp.PROPERTIES_CANONICAL


abs_errors_table = {prop: [] for prop in props_to_check}
rel_errors_table = {prop: [] for prop in props_to_check}
abs_norm_table = {prop: [] for prop in props_to_check}
rel_norm_table = {prop: [] for prop in props_to_check}  # ← new
timings = []


# Pre-warm (mock run) for each input type to exclude compilation time
for input_type, v1, v2 in tests:
    _ = fluid.get_state(input_type, v1, v2)  # not timed

# Actual timed runs
for input_type, v1, v2 in tests:
    start = time.perf_counter()
    interp = fluid.get_state(input_type, v1, v2)
    timings.append(time.perf_counter() - start)

    for prop in props_to_check:
        abs_errors_table[prop].append(interp[prop]-state_ref[prop])
        rel_errors_table[prop].append(interp[prop]/state_ref[prop] - 1.0)
        abs_norm_table[prop].append(l2_absolute_error(interp[prop], state_ref[prop]))
        rel_norm_table[prop].append(l2_relative_error(interp[prop], state_ref[prop]))


# ---------------------------
# Print consistency check
# ---------------------------
col_names = [name.replace("_INPUTS", "") for name in col_names]
col_w = max(len(c) for c in col_names) + 1
prop_w = max(len(p) for p in props_to_check)
header = f"{'property':<{prop_w}} | " + " | ".join(
    f"{c:>{col_w}s}" for c in col_names
)
width = len(header) + 2

def print_property_matrix(table_dict, fmt=".4e"):
    """Print a table of property values or errors."""
    print(header)
    print("-" * width)
    for prop in props_to_check:
        row = " | ".join(f"{val.mean():+{col_w}{fmt}}" for val in table_dict[prop])
        print(f"{prop:<{prop_w}} | {row}")
    print("-" * width)

# Print timings
print("\nEvaluation times per input type:")
for name, t in zip(col_names, timings):
    print(f"  {name:<20}: {t*1e3:8.3f} ms")

# Print absolute L² errors
print("\n" + "-" * width)
print(" Absolute two-norm error across mesh")
print("-" * width)
print_property_matrix(abs_norm_table, fmt=".3e")

# Print relative L² errors
print("\n" + "-" * width)
print(" Relative two-norm error across mesh")
print("-" * width)
print_property_matrix(rel_norm_table, fmt=".3e")


# ---------------------------
# Consistency check (relative error)
# ---------------------------
violations = []
n_total = 0

for prop in props_to_check:
    for col_name, rel_err_array in zip(col_names, rel_errors_table[prop]):
        # Flatten and filter NaNs/Infs
        rel_err_flat = np.ravel(rel_err_array)
        rel_err_flat = rel_err_flat[np.isfinite(rel_err_flat)]

        n_total += rel_err_flat.size
        if rel_err_flat.size == 0:
            continue

        max_err = np.max(np.abs(rel_err_flat))
        if max_err > CONSISTENCY_TOLERANCE:
            violations.append((prop, col_name, max_err))

n_fail = len(violations)

if n_fail == 0:
    print(
        f"\nConsistency check passed: all {n_total} relative error values are within "
        f"tolerance (tol = {CONSISTENCY_TOLERANCE:g})."
    )
else:
    print(
        f"\nConsistency check failed: {n_fail}/{n_total} relative error values exceed "
        f"tolerance (tol = {CONSISTENCY_TOLERANCE:g})."
    )
    print("Violations (property, input type, max relative error):")
    for prop, col_name, max_err in violations:
        print(f"  {prop:<20} | {col_name:<20} | max error = {max_err:.3e}")

    raise RuntimeError("Consistency check failed.")


# ---------------------------
# Plot thermodynamic region
# ---------------------------
fluid = jxp.Fluid(name=fluid_name, backend=backend)
x_prop, y_prop = "enthalpy", "pressure"
fig, ax = fluid.plot_phase_diagram(
    x_prop=x_prop, y_prop=y_prop, x_scale="linear", y_scale="log"
)
ax.scatter(state_ref[x_prop], state_ref[y_prop], s=5, c="tab:orange")
fig.tight_layout(pad=1)


# Show figures
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()
