from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
import jaxprop as jp

import numpy as np
import matplotlib.pyplot as plt
import jaxprop as jxp


def main(argv=None):
    """Run the declared scientific case; quick profiles reduce sampling only."""
    options = example_options(__file__, argv, description=__doc__)
    jxp.set_plot_options(grid=False)

    # --- fluid and property ranges
    fluid = jxp.Fluid(method="coolprop_jax", name="CO2")
    x_prop = "h"
    y_prop = "p"

    x_range = np.linspace(200e3, 600e3, options.points(50))  # enthalpy [J/kg]
    y_range = np.logspace(
        np.log10(30e5), np.log10(100e5), options.points(50)
    )  # pressure [Pa]
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # --- prepare figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    props_to_plot = ["joule_thomson", "isothermal_joule_thomson"]
    titles = [
        r"Joule–Thomson $\mu_{JT} = (\partial T/\partial p)_h$",
        r"Isothermal Joule–Thomson $\mu_T = (\partial h/\partial p)_T$",
    ]

    for ax, z_prop, title in zip(axes, props_to_plot, titles):
        # compute properties
        states = fluid.get_state(jxp.HmassP_INPUTS, x_grid, y_grid, errors="mask")

        # contour plot
        levels = 12
        # Leave locally undefined response properties as gaps in the contour.
        values = np.ma.masked_where(
            ~np.asarray(states.property_valid(z_prop)), states.raw_property(z_prop)
        )
        cs = ax.contourf(x_grid, y_grid, values, levels=levels)
        cbar = fig.colorbar(cs, ax=ax, pad=0.02)
        cbar.set_label(jxp.LABEL_MAPPING.get(z_prop, z_prop))

        # labels, limits, title
        ax.set_xlabel(jxp.LABEL_MAPPING.get(x_prop, x_prop))
        ax.set_xlim([x_range.min(), x_range.max()])
        ax.set_title(title, fontsize=11)
        ax.grid(False)

        # add phase diagram
        jp.plot_phase_diagram(
            fluid, x_prop, y_prop, dT_crit=0.01, axes=ax, N=options.points(100)
        )

    # shared y-label
    axes[0].set_ylabel(jxp.LABEL_MAPPING.get(y_prop, y_prop))
    axes[0].set_ylim([y_range.min(), y_range.max()])

    plt.tight_layout(pad=1.2)
    options.check_states(locals(), allow_failures=True)
    options.record(
        "execution", "Scientific example completed; see numerical-check records."
    )
    return options.finish()


if __name__ == "__main__":
    main()
