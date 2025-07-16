import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import coolpropx as cpx

# Setup
cpx.set_plot_options(grid=False)
fig_dir = "output"
os.makedirs(fig_dir, exist_ok=True)

# Fluids to plot
fluid_names = ["Water", "CarbonDioxide", "R134a", "R245fa", "R1233ZDE"]
colors = mpl.cm.magma(np.linspace(0.1, 0.9, len(fluid_names)))  # Colormap

# Create the plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Surface tension (N/m)")
N = 200
T_buffer = 1  # K margin above triple and below critical to avoid undefined values

for i, name in enumerate(fluid_names):
    color = colors[i]

    # --- REFPROP fluid ---
    fluid_refprop = cpx.Fluid(name=name, backend="REFPROP", exceptions=True)
    T_min = fluid_refprop.triple_point_liquid.T + T_buffer
    T_max = fluid_refprop.critical_point.T - T_buffer
    T_array = np.linspace(T_min, T_max, N)
    states_rp = fluid_refprop.get_states(cpx.QT_INPUTS, 0.0, T_array)

    ax.plot(
        states_rp["T"],
        states_rp["surface_tension"],
        label=f"{name}",
        linewidth=1.25,
        color=color
    )

    # --- HEOS fluid ---
    fluid_heos = cpx.Fluid(name=name, backend="HEOS", exceptions=True)
    T_array_markers = np.linspace(T_min, T_max, N)
    states_heos = fluid_heos.get_states(cpx.QT_INPUTS, 0.0, T_array_markers)

    ax.plot(
        states_heos["T"][::10],
        states_heos["surface_tension"][::10],
        linestyle="none",
        marker="o",
        markersize=4,
        # label=f"{name} (HEOS)",
        color=colors[i]
    )

# Finalize plot
ax.legend(loc="upper right", fontsize=12)
fig.tight_layout(pad=1)
cpx.savefig_in_formats(fig, os.path.join(fig_dir, f"surface_tension_comparison"))
plt.show()