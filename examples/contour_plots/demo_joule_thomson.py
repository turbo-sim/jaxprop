import numpy as np
import matplotlib.pyplot as plt
import jaxprop as jxp

jxp.set_plot_options(grid=False)

# --- fluid and property ranges
fluid = jxp.FluidJAX(name="CO2")
x_prop = "h"
y_prop = "p"

x_range = np.linspace(200e3, 600e3, 50)   # enthalpy [J/kg]
y_range = np.logspace(np.log10(30e5), np.log10(100e5), 50)  # pressure [Pa]
x_grid, y_grid = np.meshgrid(x_range, y_range)

# --- prepare figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
props_to_plot = ["joule_thomson", "isothermal_joule_thomson"]
titles = [r"Joule–Thomson $\mu_{JT} = (\partial T/\partial p)_h$",
          r"Isothermal Joule–Thomson $\mu_T = (\partial h/\partial p)_T$"]

for ax, z_prop, title in zip(axes, props_to_plot, titles):
    # compute properties
    states = fluid.get_props(jxp.HmassP_INPUTS, x_grid, y_grid)
    
    # contour plot
    levels = 12
    cs = ax.contourf(states[x_prop], states[y_prop], states[z_prop], levels=levels)
    cbar = fig.colorbar(cs, ax=ax, pad=0.02)
    cbar.set_label(jxp.LABEL_MAPPING.get(z_prop, z_prop))

    # labels, limits, title
    ax.set_xlabel(jxp.LABEL_MAPPING.get(x_prop, x_prop))
    ax.set_xlim([x_range.min(), x_range.max()])
    ax.set_title(title, fontsize=11)
    ax.grid(False)

    # add phase diagram
    fluid.fluid.plot_phase_diagram(x_prop, y_prop, dT_crit=0.01, axes=ax)

# shared y-label
axes[0].set_ylabel(jxp.LABEL_MAPPING.get(y_prop, y_prop))
axes[0].set_ylim([y_range.min(), y_range.max()])

plt.tight_layout(pad=1.2)
plt.show()
