import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt


# ---------------------------
# Configuration
# ---------------------------
outdir = "fluid_tables"
fluid_name = "CO2"
h_min = 200e3  # J/kg
h_max = 600e3  # J/kg
p_min = 2e6    # Pa
p_max = 20e6   # Pa
N_h = 120
N_p = 120

fluid_bicubic = jxp.FluidBicubic(
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

fluid_jax = jxp.FluidJAX(name=fluid_name, backend="HEOS")

# ---------------------------
# Step 1: Interpolate at (h, P)
# ---------------------------
test_h = 1000e3   # J/kg
test_P = 10e6    # Pa
props_hp = fluid_bicubic.get_props(jxp.HmassP_INPUTS, test_h, test_P)

print(props_hp["density"] - fluid_bicubic.get_props(jxp.HmassP_INPUTS, test_h, p_min)["density"])
print(props_hp["density"] - fluid_bicubic.get_props(jxp.HmassP_INPUTS, test_h, p_max)["density"])

print(props_hp["density"] - fluid_bicubic.get_props(jxp.HmassP_INPUTS, h_max, test_P)["density"])
print(props_hp["density"] - fluid_bicubic.get_props(jxp.HmassP_INPUTS, h_min, test_P)["density"])

# ---------------------------
# Step 2: Reconstruct properties using all inverse input pairs
# ---------------------------

input_tests = {
    "PT_INPUTS": jxp.PT_INPUTS,
    "HmassSmass_INPUTS": jxp.HmassSmass_INPUTS,
    "PSmass_INPUTS": jxp.PSmass_INPUTS,
    "DmassHmass_INPUTS": jxp.DmassHmass_INPUTS,
    "DmassP_INPUTS": jxp.DmassP_INPUTS,
}

results = {}
reference_h = test_h
reference_p = test_P

print("\n" + "=" * 60)
print(f"Reference state from (h={test_h:.1f}, P={test_P:.1f}):")
print(f"  T = {props_hp['T']:.2f}, s = {props_hp['entropy']:.2f}, d = {props_hp['density']:.2f}")
print("=" * 60)

for label, input_pair in input_tests.items():
    x_alias, y_alias = jxp.INPUT_PAIR_MAP[input_pair]
    x_key = jxp.ALIAS_TO_CANONICAL[x_alias]
    y_key = jxp.ALIAS_TO_CANONICAL[y_alias]
    x_val = float(props_hp[x_key])
    y_val = float(props_hp[y_key])

    try:
        props_back = fluid_bicubic.get_props(input_pair, x_val, y_val)
        dh = props_back["enthalpy"] - reference_h
        dp = props_back["pressure"] - reference_p
        rel_dh = dh / reference_h
        rel_dp = dp / reference_p

        results[label] = {
            "props": props_back,
            "dh": dh,
            "dp": dp,
            "rel_dh": rel_dh,
            "rel_dp": rel_dp,
        }

        print(f"[{label:20s}]")
        print(f"  Inputs: {x_key} = {x_val:.4e}, {y_key} = {y_val:.4e}")
        print(f"  Recovered: h = {props_back['enthalpy']:.4f}, p = {props_back['pressure']:.4f}")
        print(f"  dh = {dh:+.4e}  (rel = {rel_dh:+.2e})")
        print(f"  dp = {dp:+.4e}  (rel = {rel_dp:+.2e})")
    except Exception as e:
        print(f"[{label:20s}] Error during reconstruction: {e}")

    print("-" * 60)

# ---------------------------
# Step 3: Plot T/s/d vs P (for fixed h) and vs h (for fixed P) → Check Monotonicity
# ---------------------------

# Define ranges
P_range = jnp.linspace(p_min, p_max, 1000)
h_range = jnp.linspace(h_min, h_max, 1000)

# Properties to test
prop_labels = {
    "T": "Temperature [K]",
    "entropy": "Entropy [J/kg-K]",
    "density": "Density [kg/m³]",
}

# Helper: detect monotonicity
def check_monotonicity(diffs, label):
    is_increasing = jnp.all(diffs > 0)
    is_decreasing = jnp.all(diffs < 0)

    if is_increasing:
        print(f"{label} is strictly increasing.")
    elif is_decreasing:
        print(f"{label} is strictly decreasing.")
    else:
        print(f"{label} is NOT monotonic.")
        num_violations = jnp.sum(jnp.sign(diffs[1:]) != jnp.sign(diffs[:-1]))
        print(f" → Number of sign changes in derivative: {num_violations}")

# Loop over each property
for prop, ylabel in prop_labels.items():
    # Evaluate prop(h_fixed, P)
    vals_vs_P = jnp.array([
        fluid_bicubic.get_props(jxp.HmassP_INPUTS, test_h, P)[prop]
        for P in P_range
    ])

    # Evaluate prop(h, P_fixed)
    vals_vs_h = jnp.array([
        fluid_bicubic.get_props(jxp.HmassP_INPUTS, h, test_P)[prop]
        for h in h_range
    ])

    # Plot 1: prop vs Pressure at fixed enthalpy
    plt.figure(figsize=(6, 4))
    plt.plot(P_range / 1e6, vals_vs_P)
    plt.xlabel("Pressure [MPa]")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel.split()[0]} vs Pressure at h = {test_h/1e3:.1f} kJ/kg")
    plt.grid(True)
    plt.tight_layout()

    # Plot 2: prop vs Enthalpy at fixed pressure
    plt.figure(figsize=(6, 4))
    plt.plot(h_range / 1e3, vals_vs_h)
    plt.xlabel("Enthalpy [kJ/kg]")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel.split()[0]} vs Enthalpy at P = {test_P/1e5:.1f} bar")
    plt.grid(True)
    plt.tight_layout()

    # Check monotonicity
    check_monotonicity(jnp.diff(vals_vs_P), f"{ylabel.split()[0]} vs Pressure (at fixed h)")
    check_monotonicity(jnp.diff(vals_vs_h), f"{ylabel.split()[0]} vs Enthalpy (at fixed P)")

plt.show()


# # ---------------------------
# # Step 2: Reconstruct (h, P) using PT_INPUTS, HmassSmass_INPUTS, PSmass_INPUTS, DmassHmass_INPUTS, DmassP_INPUTS
# # ---------------------------

# # ---------------------------
# # Organize inverse input tests
# # ---------------------------

# input_tests = {
#     "PT_INPUTS": (jxp.PT_INPUTS, test_P, props_hp["T"]),
#     "HmassSmass_INPUTS": (jxp.HmassSmass_INPUTS, test_h, props_hp["entropy"]),
#     "PSmass_INPUTS": (jxp.PSmass_INPUTS, test_P, props_hp["entropy"]),
#     "DmassHmass_INPUTS": (jxp.DmassHmass_INPUTS, props_hp["density"], test_h),
#     "DmassP_INPUTS": (jxp.DmassP_INPUTS, props_hp["density"], test_P),
# }

# # ---------------------------
# # Compute inverse property results
# # ---------------------------
# results = {}
# for label, (input_pair, x_val, y_val) in input_tests.items():
#     try:
#         results[label] = fluid_bicubic.get_props(input_pair, x_val, y_val)
#     except Exception as e:
#         print(f"Error in {label}: {e}")

# # ---------------------------
# # Step 3: Compare recovered values
# # ---------------------------

# print("\n" + "="*60)
# print("Consistency check (reconstructed vs direct HmassP):")
# print(f"Reference point: h = {props_hp['enthalpy']:.2f}, p = {props_hp['pressure']:.2f}")
# print("-" * 60)

# for label, props_back in results.items():
#     dh = props_back["enthalpy"] - props_hp["enthalpy"]
#     dp = props_back["pressure"] - props_hp["pressure"]

#     rel_dh = dh / props_hp["enthalpy"]
#     rel_dp = dp / props_hp["pressure"]

#     print(f"[{label:20s}]")
#     print(f"  Recovered: h = {props_back['enthalpy']:.4f}, p = {props_back['pressure']:.4f}")
#     print(f"  dh = {dh:+.4e}  (rel = {rel_dh:+.2e})")
#     print(f"  dp = {dp:+.4e}  (rel = {rel_dp:+.2e})")
#     print("-" * 60)

# ---------------------------
# Step 4: Residual surface scan for PT inversion
# # ---------------------------

# import numpy as np
# import jax.numpy as jnp
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa


# # Target values (from HP state)
# target_p = float(test_P)
# target_T = float(props_hp["T"])  # pick one state if vectorized

# # Get property names for PT inversion
# prop1, prop2 = jxp.INPUT_PAIR_MAP[jxp.PT_INPUTS]
# prop1 = jxp.ALIAS_TO_CANONICAL[prop1]
# prop2 = jxp.ALIAS_TO_CANONICAL[prop2]

# # Property ranges for scaling
# rng1 = jnp.maximum(fluid_bicubic.table[prop1]["value"].ptp(), 1.0)
# rng2 = jnp.maximum(fluid_bicubic.table[prop2]["value"].ptp(), 1.0)

# # Grid definition
# h_axis = jnp.linspace(h_min, h_max, 60)
# p_axis = jnp.linspace(p_min, p_max, 60)
# H, P = jnp.meshgrid(h_axis, p_axis, indexing="ij")

# # Property evaluation
# props_grid = fluid_bicubic.get_props(jxp.HmassP_INPUTS, H, P)
# val1_grid = props_grid[prop1]
# val2_grid = props_grid[prop2]

# # Scaled residual fields (same as solver)
# f1_grid = (val1_grid - target_p) / rng1
# f2_grid = (val2_grid - target_T) / rng2


# # 3D surface plot (f1, f2, and zero plane)
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection="3d")

# ax.plot_surface(
#     np.asarray(H / 1e3), np.asarray(P / 1e5), np.asarray(f1_grid),
#     cmap="viridis", alpha=0.7, linewidth=0
# )
# ax.plot_surface(
#     np.asarray(H / 1e3), np.asarray(P / 1e5), np.asarray(f2_grid),
#     cmap="plasma", alpha=0.7, linewidth=0
# )

# # Zero plane
# ax.plot_surface(
#     np.asarray(H / 1e3), np.asarray(P / 1e5), np.zeros_like(f1_grid),
#     color="red", alpha=0.3
# )

# ax.plot(props_hp["h"]/1e3, props_hp["p"]/1e5, 0, "ko")
# ax.plot(props_hp["h"]/1e3, props_hp["p"]/1e5, 0, "r+")

# ax.set_xlabel("Enthalpy [kJ/kg]")
# ax.set_ylabel("Pressure [bar]")
# ax.set_zlabel("Scaled residual")
# ax.set_title("3D residual surfaces (scaled f1 and f2)")
# plt.tight_layout()

# # 2D contour plot (zero lines)
# fig2, ax2 = plt.subplots(figsize=(7, 6))

# ax2.plot(props_hp["h"]/1e3, props_hp["p"]/1e5, "ko")
# ax2.plot(props_pt["h"]/1e3, props_pt["p"]/1e5, "r+")

# ax2.contour(
#     np.asarray(H / 1e3), np.asarray(P / 1e5), np.asarray(f1_grid),
#     levels=[0.0], colors="blue", linewidths=2, linestyles="--"
# )
# ax2.contour(
#     np.asarray(H / 1e3), np.asarray(P / 1e5), np.asarray(f2_grid),
#     levels=[0.0], colors="green", linewidths=2, linestyles="-."
# )

# ax2.set_xlabel("Enthalpy [kJ/kg]")
# ax2.set_ylabel("Pressure [bar]")
# ax2.set_title("Zero-contours of scaled residuals f1=0 and f2=0")
# plt.tight_layout()
# plt.show()