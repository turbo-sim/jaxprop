import os
import numpy as np
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt

jxp.set_plot_options()

# ---------------------------
# Configuration
# ---------------------------

fluid_name = "nitrogen"
h_min = -140e3  # J/kg
h_max = -10e3  # J/kg
p_min = 2e4  # Pa
p_max = 1e6  # Pa
N_p = 160 # Number of pressure points
N_h = 160 # Number of enthalpy points
metastable_phase = "liquid"

outdir = "demo_metastable_table_generation"

# ---------------------------
# Build models
# ---------------------------
fluid_bicubic = jxp.FluidBicubic(
    fluid_name=fluid_name,
    backend="HEOS",
    h_min=h_min, h_max=h_max,
    p_min=p_min, p_max=p_max,
    N_h=N_h, N_p=N_p,
    table_dir=outdir,
    metastable_phase=metastable_phase,
    gradient_method="forward",
)

fluid_cp = jxp.Fluid(fluid_name, exceptions=False)

# ---------------------------
# Saturation quality for comparison
# ---------------------------
Q_sat = 0.0 if metastable_phase == "liquid" else 1.0

# ---------------------------
# Saturation pressure midpoints
# ---------------------------
# Use the same log-spaced pressure nodes as the table, then take midpoints
logP_sat_nodes = np.asarray(fluid_bicubic.logP_sat_vals)
logP_midpoints = 0.5 * (logP_sat_nodes[:-1] + logP_sat_nodes[1:])
p_midpoints = np.exp(logP_midpoints)

# ---------------------------
# Evaluate saturation at midpoints
# ---------------------------
properties = ["temperature", "density", "enthalpy", "gruneisen"]

error_data = {prop: [] for prop in properties}
interp_vals = {prop: [] for prop in properties}
cp_vals = {prop: [] for prop in properties}

for p in p_midpoints:
    sat_interp = fluid_bicubic.get_state_saturation(float(p))
    # print(sat_interp)
    sat_cp = fluid_cp.get_state(jxp.PQ_INPUTS, float(p), Q_sat)

    for prop in properties:
        val_interp = float(sat_interp[prop])
        val_cp = float(sat_cp[prop])
        interp_vals[prop].append(val_interp)
        cp_vals[prop].append(val_cp)

        if abs(val_cp) > 1e-30:
            rel_err = abs(val_interp - val_cp) / abs(val_cp) * 100.0
        else:
            rel_err = float("nan")
        error_data[prop].append(rel_err)

# Convert to arrays
for prop in properties:
    error_data[prop] = np.array(error_data[prop])
    interp_vals[prop] = np.array(interp_vals[prop])
    cp_vals[prop] = np.array(cp_vals[prop])

# ---------------------------
# Print error summary
# ---------------------------
print("\nSaturation midpoint interpolation error summary:")
header = f"{'Prop':<10} {'% min':>12} {'% max':>12} {'% mean':>12}"
print("-" * len(header))
print(header)
print("-" * len(header))
for prop in properties:
    err = error_data[prop]
    print(f"{prop:<10} {np.nanmin(err):>12.3e} {np.nanmax(err):>12.3e} {np.nanmean(err):>12.3e}")

# ---------------------------
# Plot: relative error vs pressure
# ---------------------------
fig, axes = plt.subplots(len(properties), 1, figsize=(8, 3 * len(properties)), sharex=True)
if len(properties) == 1:
    axes = [axes]

for ax, prop in zip(axes, properties):
    ax.semilogy(p_midpoints / 1e6, error_data[prop], "o-", markersize=3)
    ax.set_ylabel(f"{prop} error [%]")
    ax.grid(True, which="both", ls="--", alpha=0.5)

axes[-1].set_xlabel("Pressure [MPa]")
axes[0].set_title(f"Saturation interpolation error at midpoints ({fluid_name}, Q={Q_sat})")
plt.tight_layout()

# ---------------------------
# Plot: interpolated vs CoolProp values
# ---------------------------
fig2, axes2 = plt.subplots(len(properties), 1, figsize=(8, 3 * len(properties)), sharex=True)
if len(properties) == 1:
    axes2 = [axes2]

for ax, prop in zip(axes2, properties):
    ax.plot(p_midpoints / 1e6, cp_vals[prop], "k-", label="CoolProp", linewidth=1.5)
    ax.plot(p_midpoints / 1e6, interp_vals[prop], "r--", label="Bicubic", linewidth=1.5)
    ax.set_ylabel(prop)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=8)

axes2[-1].set_xlabel("Pressure [MPa]")
axes2[0].set_title(f"Saturation properties comparison ({fluid_name}, Q={Q_sat})")
plt.tight_layout()

# Show figures
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()
