import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import jaxprop.coolprop as jxp

jxp.set_plot_options(grid=False)

outdir = "output"
os.makedirs(outdir, exist_ok=True)

# Define nice colormap
colormap = mcolors.LinearSegmentedColormap.from_list(
    "truncated_blues", plt.cm.Blues(np.linspace(0.3, 1.0, 256))
)

# Create fluid object
fluid_name = "CO2"
fluid = jxp.Fluid(name=fluid_name, backend="HEOS", exceptions=True)

# Create figure for plotting
fig, ax = plt.subplots(figsize=(6.0, 5.0))
ax.set_xlabel(r"Entropy (kJ/kg/K)")
ax.set_ylabel(r"Temperature ($^\circ$C)")
s_min, s_max = 1.0, 2
T_min, T_max = 00, 80
ax.set_xlim([s_min, s_max])
ax.set_ylim([T_min, T_max])


import time

# Plot phase diagram
start = time.perf_counter()

fig, ax = fluid.plot_phase_diagram("entropy", "temperature", axes=ax, plot_quality_isolines=True, plot_pseudocritical_line=True)

end = time.perf_counter()
print(f"Run {0}: {end - start:.6f} s")

for i in range(5):
    start = time.perf_counter()

    fluid_name = "CO2"
    fluid = jxp.Fluid(name=fluid_name, backend="HEOS", exceptions=True)


    # Plot phase diagram
    fluid.plot_phase_diagram(
        "entropy", "temperature",
        axes=None,  # set to ax if you want to reuse an existing axis
        plot_quality_isolines=True,
        plot_pseudocritical_line=True
    )

    end = time.perf_counter()
    print(f"Run {i+1}: {end - start:.6f} s")


# # Plot pressure isobars
# range_x = np.linspace(s_min, s_max, 51) * 1e3
# range_y = np.linspace(T_min, T_max, 51) + 273.15
# prop_dict = fluid.get_states(jxp.SmassT_INPUTS, range_x, range_y, generalize_quality=False)
# contour = ax.contour(
#     prop_dict["entropy"],
#     prop_dict["temperature"],
#     prop_dict["pressure"] / 1e5,
#     levels=np.asarray(np.linspace(60, 100, 11)),
#     cmap=colormap,
#     zorder=1,
# )

# contourf = ax.contourf(
#     prop_dict["entropy"],
#     prop_dict["temperature"],
#     prop_dict["pressure"]/1e5,
#     levels=np.asarray(np.linspace(60, 100, 11)),
#     cmap=colormap,
#     zorder=1,
# )
# contourf.set_visible(False)

# # Add a colorbar with a label
# colorbar = fig.colorbar(contourf, ax=ax, label="Pressure (bar)")

# # Set plot scale
# jxp.scale_graphics_x(fig, +1e-3, mode="multiply")
# jxp.scale_graphics_y(fig, -273.15, mode="add")
# # ax.relim(visible_only=False)
# # ax.autoscale_view()

# # Add legend with heat capacity definitions
# ax.plot([], [], label=r"$c_p = \left( \frac{\partial h}{\partial T} \right)_p = T \left( \frac{\partial s}{\partial T} \right)_p$")
# ax.legend(loc="lower right")

# # Adjust pad
# fig.tight_layout(pad=1)

# # Save figure
# filename = os.path.join(outdir, "Ts_diagram_isobars")
# jxp.savefig_in_formats(fig, filename)

# # Show figures
# if not os.environ.get("DISABLE_PLOTS"):
#     plt.show()
