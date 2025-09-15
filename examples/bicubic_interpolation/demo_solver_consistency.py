import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt


# ---------------------------
# Configuration
# ---------------------------
outdir = "fluid_tables"
fluid_name = "CO2"
h_min = 500e3  # J/kg
h_max = 1500e3  # J/kg
p_min = 2e6    # Pa
p_max = 20e6   # Pa
N_h = 80
N_p = 80

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


# ---------------------------
# Step 1: Interpolate at (h, P)
# ---------------------------
test_h = 1000e3   # J/kg
test_P = 12e6    # Pa
props_hp = fluid_bicubic.get_props(jxp.HmassP_INPUTS, test_h, test_P)

# ---------------------------
# Step 2: Reconstruct (h, P) using PT_INPUTS
# ---------------------------
test_T = props_hp["T"]
print(test_T)
props_pt = fluid_bicubic.get_props(jxp.PT_INPUTS, test_P, test_T)

# ---------------------------
# Step 3: Compare recovered values
# ---------------------------

print("\nConsistency check:")
print(f"Direct H-P: h = {props_hp['enthalpy']}, p = {props_hp['pressure']}")
print(f"From P-T:   h = {props_pt['enthalpy']}, p = {props_pt['pressure']}")

dh = props_pt['enthalpy'] - props_hp['enthalpy']
dp = props_pt['pressure'] - props_hp['pressure']

rel_dh = dh / props_hp['enthalpy']
rel_dp = dp / props_hp['pressure']

print(f"dh = {dh}  (rel = {rel_dh})")
print(f"dp = {dp}  (rel = {rel_dp})")



# ---------------------------
# Step 4: Residual surface scan for PT inversion
# ---------------------------

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


# Target values (from HP state)
target_p = float(test_P)
target_T = float(test_T)  # pick one state if vectorized

# Get property names for PT inversion
prop1, prop2 = jxp.INPUT_PAIR_MAP[jxp.PT_INPUTS]
prop1 = jxp.ALIAS_TO_CANONICAL[prop1]
prop2 = jxp.ALIAS_TO_CANONICAL[prop2]

# Property ranges for scaling
rng1 = jnp.maximum(fluid_bicubic.table[prop1]["value"].ptp(), 1.0)
rng2 = jnp.maximum(fluid_bicubic.table[prop2]["value"].ptp(), 1.0)

# Grid definition
h_axis = jnp.linspace(h_min, h_max, 60)
p_axis = jnp.linspace(p_min, p_max, 60)
H, P = jnp.meshgrid(h_axis, p_axis, indexing="ij")

# Property evaluation
props_grid = fluid_bicubic.get_props(jxp.HmassP_INPUTS, H, P)
val1_grid = props_grid[prop1]
val2_grid = props_grid[prop2]

# Scaled residual fields (same as solver)
f1_grid = (val1_grid - target_p) / rng1
f2_grid = (val2_grid - target_T) / rng2


# 3D surface plot (f1, f2, and zero plane)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    np.asarray(H / 1e3), np.asarray(P / 1e5), np.asarray(f1_grid),
    cmap="viridis", alpha=0.7, linewidth=0
)
ax.plot_surface(
    np.asarray(H / 1e3), np.asarray(P / 1e5), np.asarray(f2_grid),
    cmap="plasma", alpha=0.7, linewidth=0
)

# Zero plane
ax.plot_surface(
    np.asarray(H / 1e3), np.asarray(P / 1e5), np.zeros_like(f1_grid),
    color="red", alpha=0.3
)

ax.plot(props_hp["h"]/1e3, props_hp["p"]/1e5, 0, "ko")
ax.plot(props_pt["h"]/1e3, props_pt["p"]/1e5, 0, "r+")

ax.set_xlabel("Enthalpy [kJ/kg]")
ax.set_ylabel("Pressure [bar]")
ax.set_zlabel("Scaled residual")
ax.set_title("3D residual surfaces (scaled f1 and f2)")
plt.tight_layout()

# 2D contour plot (zero lines)
fig2, ax2 = plt.subplots(figsize=(7, 6))

ax2.plot(props_hp["h"]/1e3, props_hp["p"]/1e5, "ko")
ax2.plot(props_pt["h"]/1e3, props_pt["p"]/1e5, "r+")

ax2.contour(
    np.asarray(H / 1e3), np.asarray(P / 1e5), np.asarray(f1_grid),
    levels=[0.0], colors="blue", linewidths=2, linestyles="--"
)
ax2.contour(
    np.asarray(H / 1e3), np.asarray(P / 1e5), np.asarray(f2_grid),
    levels=[0.0], colors="green", linewidths=2, linestyles="-."
)

ax2.set_xlabel("Enthalpy [kJ/kg]")
ax2.set_ylabel("Pressure [bar]")
ax2.set_title("Zero-contours of scaled residuals f1=0 and f2=0")
plt.tight_layout()
plt.show()