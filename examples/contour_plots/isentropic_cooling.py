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

    # ----------------------------------------------------------------------------------------------- #
    # Plot sensitivity of temperature with respect to pressure at constant entropy for water
    # ----------------------------------------------------------------------------------------------- #
    # Define fluid and property ranges
    fluid = Fluid(name="water")
    x_prop = "s"
    y_prop = "p"
    z_prop = "Z"
    x_range = np.linspace(500, 9000, options.points(50))
    y_range = np.logspace(np.log10(1e3), np.log10(1e8), options.points(50))
    p_grid, s_grid = np.meshgrid(y_range, x_range, indexing="ij")
    states1 = fluid.get_state(jxp.PSmass_INPUTS, p_grid, s_grid, errors="mask")
    states2 = fluid.get_state(
        jxp.PSmass_INPUTS, p_grid + 1e-3 * p_grid, s_grid, errors="mask"
    )
    prop = (states2.raw_property("T") - states1.raw_property("T")) / (1e-3 * p_grid)

    # Create figure
    fig_1, ax_1 = plt.subplots(figsize=(6, 5))
    ax_1.grid(False)
    ax_1.set_yscale("log")
    ax_1.set_xlabel(jxp.LABEL_MAPPING.get(x_prop, x_prop))
    ax_1.set_ylabel(jxp.LABEL_MAPPING.get(y_prop, y_prop))
    ax_1.set_xlim([x_range.min(), x_range.max()])
    ax_1.set_ylim([y_range.min(), y_range.max()])

    # Plot phase diagram and contour
    jp.plot_phase_diagram(fluid, x_prop, y_prop, axes=ax_1, N=options.points(100))
    vmin = np.nanmin(prop)
    vmax = np.nanmax(prop)
    levels = np.linspace(vmin, vmax, 20)
    colors = plt.get_cmap("Greys")(np.linspace(0.8, 0.2, len(levels) - 1))
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    contour = ax_1.contourf(s_grid, p_grid, prop, levels=levels, cmap=cmap, norm=norm)
    cbar = fig_1.colorbar(contour, ax=ax_1, pad=0.02)
    cbar.set_label(r"$(\partial T / \partial p)_s$")
    cbar.set_ticks(levels[::2])
    cbar.set_ticklabels([f"{v:.2e}" for v in levels[::2]])
    plt.tight_layout(pad=1)

    # ----------------------------------------------------------------------------------------------- #
    # Plot sensitivity of temperature with respect to pressure at constant entropy for nitrogen
    # ----------------------------------------------------------------------------------------------- #
    # Define fluid and property ranges
    fluid = Fluid(name="nitrogen")
    x_prop = "s"
    y_prop = "p"
    z_prop = "Z"
    x_range = np.linspace(2500, 5750, options.points(100))
    y_range = np.logspace(np.log10(2e4), np.log10(2e7), options.points(100))
    p_grid, s_grid = np.meshgrid(y_range, x_range, indexing="ij")
    states1 = fluid.get_state(jxp.PSmass_INPUTS, p_grid, s_grid, errors="mask")
    states2 = fluid.get_state(
        jxp.PSmass_INPUTS, p_grid + 1e-3 * p_grid, s_grid, errors="mask"
    )
    prop = (states2.raw_property("T") - states1.raw_property("T")) / (1e-3 * p_grid)

    # Create figure
    fig_1, ax_1 = plt.subplots(figsize=(6, 5))
    ax_1.grid(False)
    ax_1.set_yscale("log")
    ax_1.set_xlabel(jxp.LABEL_MAPPING.get(x_prop, x_prop))
    ax_1.set_ylabel(jxp.LABEL_MAPPING.get(y_prop, y_prop))
    ax_1.set_xlim([x_range.min(), x_range.max()])
    ax_1.set_ylim([y_range.min(), y_range.max()])

    # Plot phase diagram and contour
    jp.plot_phase_diagram(fluid, x_prop, y_prop, axes=ax_1, N=options.points(100))
    vmin = np.nanmin(prop)
    vmax = np.nanmax(prop)
    levels = np.linspace(vmin, vmax, 20)
    colors = plt.get_cmap("Greys")(np.linspace(0.8, 0.2, len(levels) - 1))
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    contour = ax_1.contourf(s_grid, p_grid, prop, levels=levels, cmap=cmap, norm=norm)
    cbar = fig_1.colorbar(contour, ax=ax_1, pad=0.02)
    cbar.set_label(r"$(\partial T / \partial p)_s$")
    cbar.set_ticks(levels[::2])
    cbar.set_ticklabels([f"{v:.2e}" for v in levels[::2]])
    plt.tight_layout(pad=1)

    # Show figures
    options.check_states(locals(), allow_failures=True)
    options.record(
        "execution", "Scientific example completed; see numerical-check records."
    )
    return options.finish()


if __name__ == "__main__":
    main()
