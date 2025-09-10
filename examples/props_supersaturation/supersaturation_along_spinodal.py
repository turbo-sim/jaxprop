import os
import numpy as np
import matplotlib.pyplot as plt

import jaxprop.coolprop as jxp


# Create the folder to save figures
jxp.set_plot_options(grid=False)
colors = jxp.COLORS_MATLAB
outdir = "output"
os.makedirs(outdir, exist_ok=True)

# Define fluid
fluid = jxp.Fluid("CO2", backend="HEOS")

# Compute spinodal lines and store in fluid to avoid redundant calculation
spinodal_liq, spinodal_vap = jxp.compute_spinodal_line(fluid, N=100, supersaturation=True)
fluid.spdl_liq = spinodal_liq
fluid.spdl_vap = spinodal_vap


# -------------------------------------------------------------------- #
# Plot supersaturation temperature along spinodal line
# -------------------------------------------------------------------- #

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'wspace': 0.25})

# Plot phase diagram and thermodynamic points on the first subplot
x = "s"
y = "T"
ax1.set_xlabel("Entropy (J/kg/K)")
ax1.set_ylabel("Temperature (K)")
ax1.set_xlim([fluid.triple_point_liquid[x], fluid.triple_point_vapor[x]])
ax1.set_ylim([fluid.triple_point_liquid[y], 1.1 * fluid.critical_point[y]])
fluid.plot_phase_diagram(
    x_prop=x,
    y_prop=y,
    plot_saturation_line=True,
    plot_spinodal_line=False,
    plot_quality_isolines=True,
    axes=ax1
)

ax1.plot(spinodal_liq[x], spinodal_liq["T"], color=colors[0], linestyle="-", label=r"$T_\text{spinodal,liquid}$")
ax1.plot(spinodal_liq[x], spinodal_liq["T_saturation"], color=colors[0], linestyle="--", label=r"$T_\text{sat}(p_\text{spinodal,liquid})$")
ax1.plot(spinodal_vap[x], spinodal_vap["T"], color=colors[1], linestyle="-", label=r"$T_\text{spinodal,vapor}$")
ax1.plot(spinodal_vap[x], spinodal_vap["T_saturation"], color=colors[1], linestyle="--", label=r"$T_\text{sat}(p_\text{spinodal,vapor})$")
ax1.legend(loc="upper left")

# Plot the degree of supersaturation on the second subplot
y1 = "supersaturation_degree"
ax2.set_xlabel("Entropy (J/kg/K)")
ax2.set_ylabel("Supersaturation degree (K)")
ax2.axhline(y=0, color="black")
ax2.plot(spinodal_liq[x], spinodal_liq[y1], color=colors[0], label="Liquid supersaturation degree")
ax2.plot(spinodal_vap[x], spinodal_vap[y1], color=colors[1], label="Vapor supersaturation degree")
ax2.legend(loc="lower right")

jxp.savefig_in_formats(fig, os.path.join(outdir, f"supersaturation_along_spinodal_Ts_{fluid.name}"))


# -------------------------------------------------------------------- #
# Plot supersaturation ratio along spinodal line
# -------------------------------------------------------------------- #

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'wspace': 0.25})

# Plot phase diagram and thermodynamic points on the first subplot
x = "s"
y = "p"
ax1.set_xlabel("Entropy (J/kg/K)")
ax1.set_ylabel("Pressure (Pa)")
ax1.set_xlim([fluid.triple_point_liquid[x], fluid.triple_point_vapor[x]])
ax1.set_ylim([fluid.triple_point_liquid[y], 1.1 * fluid.critical_point[y]])
fluid.plot_phase_diagram(
    x_prop=x,
    y_prop=y,
    plot_saturation_line=True,
    plot_spinodal_line=False,
    plot_quality_isolines=True,
    axes=ax1
)

ax1.plot(spinodal_liq[x], spinodal_liq["p"], color=colors[0], linestyle="-", label=r"$p_\text{spinodal,liquid}$")
ax1.plot(spinodal_liq[x], spinodal_liq["p_saturation"], color=colors[0], linestyle="--", label=r"$p_\text{sat}(T_\text{spinodal,liquid})$")
ax1.plot(spinodal_vap[x], spinodal_vap["p"], color=colors[1], linestyle="-", label=r"$p_\text{spinodal,vapor}$")
ax1.plot(spinodal_vap[x], spinodal_vap["p_saturation"], color=colors[1], linestyle="--", label=r"$p_\text{sat}(T_\text{spinodal,vapor})$")
ax1.legend(loc="upper left")

# Plot the supersaturation on the second subplot
y1 = "supersaturation_ratio"
ax2.set_xlabel("Entropy (J/kg/K)")
ax2.set_ylabel("Supersaturation ratio")
ax2.set_ylim([0, 2])
ax2.axhline(y=1, color="black")
ax2.plot(spinodal_liq[x], spinodal_liq[y1], color=colors[0], label="Liquid supersaturation ratio")
ax2.plot(spinodal_vap[x], spinodal_vap[y1], color=colors[1], label="Vapor supersaturation ratio")
ax2.legend(loc="upper left")
jxp.savefig_in_formats(fig, os.path.join(outdir, f"supersaturation_along_spinodal_ps_{fluid.name}"))


# Show figures
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()


