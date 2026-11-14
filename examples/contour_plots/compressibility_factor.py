from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
import jaxprop as jp

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import jaxprop.coolprop as jxp
from jaxprop import Fluid


def main(argv=None):
    """Run the declared scientific case; quick profiles reduce sampling only."""
    options = example_options(__file__, argv, description=__doc__)
    jxp.set_plot_options(grid=False)

    # Define fluid and property ranges
    fluid = Fluid(name="Cyclopentane")
    x_prop = "s"
    y_prop = "T"
    z_prop = "Z"
    x_range = np.linspace(0, 2000, options.points(100))
    y_range = np.linspace(300, 600, options.points(100))

    # Create figure
    fig_1, ax_1 = plt.subplots(figsize=(6, 5))
    ax_1.grid(False)
    ax_1.set_xlabel(jxp.LABEL_MAPPING.get(x_prop, x_prop))
    ax_1.set_ylabel(jxp.LABEL_MAPPING.get(y_prop, y_prop))
    ax_1.set_xlim([x_range.min(), x_range.max()])
    ax_1.set_ylim([y_range.min(), y_range.max()])

    # Compute properties and plot contour
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    states = fluid.get_state(jxp.SmassT_INPUTS, x_grid, y_grid, errors="mask")
    levels = np.concatenate(
        (np.linspace(0.0, 0.95, 20), [0.99])
    )  # 0.0 to 1.0 in 0.1 steps
    colors = plt.get_cmap("Greys")(
        np.linspace(1.0, 0.4, len(levels) - 1)
    )  # custom grey scale
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(boundaries=levels, ncolors=len(colors))
    contour = ax_1.contourf(
        x_grid,
        y_grid,
        np.ma.masked_invalid(states.raw_property(z_prop)),
        levels=levels,
        cmap=cmap,
        norm=norm,
    )  # , extend="both")
    cbar = fig_1.colorbar(contour, ax=ax_1, pad=0.02)
    cbar.set_label("Compressibility factor Z")
    cbar.set_ticks(levels[::2])  # every second tick

    # Plot phase diagram and experimental data
    jp.plot_phase_diagram(
        fluid,
        x_prop,
        y_prop,
        axes=ax_1,
        dT_crit=2.0,
        plot_two_phase_patch=True,
        N=options.points(100),
    )
    plt.tight_layout(pad=1)

    # Show figures
    options.check_states(locals(), allow_failures=True)
    options.record(
        "execution", "Scientific example completed; see numerical-check records."
    )
    return options.finish()


if __name__ == "__main__":
    main()
