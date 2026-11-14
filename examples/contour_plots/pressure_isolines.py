from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
import jaxprop as jp

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import jaxprop.coolprop as jxp
from jaxprop import Fluid


def main(argv=None):
    """Run the declared scientific case; quick profiles reduce sampling only."""
    options = example_options(__file__, argv, description=__doc__)
    jxp.set_plot_options(grid=False)

    outdir = options.output_dir
    os.makedirs(outdir, exist_ok=True)

    # Define nice colormap
    colormap = mcolors.LinearSegmentedColormap.from_list(
        "truncated_blues", plt.cm.Blues(np.linspace(0.3, 1.0, options.points(256)))
    )

    # Create fluid object
    fluid_name = "CO2"
    fluid = Fluid(name=fluid_name, backend="HEOS")

    # Create figure for plotting
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.set_xlabel(r"Entropy (kJ/kg/K)")
    ax.set_ylabel(r"Temperature ($^\circ$C)")
    s_min, s_max = 1.0, 2
    T_min, T_max = 00, 80
    ax.set_xlim([s_min * 1e3, s_max * 1e3])
    ax.set_ylim([T_min + 273.15, T_max + 273.15])

    # Plot phase diagram
    fig, ax = jp.plot_phase_diagram(
        fluid,
        "s",
        "T",
        axes=ax,
        plot_quality_isolines=True,
        plot_pseudocritical_line=True,
        N=options.points(100),
    )

    # Plot pressure isobars
    range_x = np.linspace(s_min, s_max, options.points(51)) * 1e3
    range_y = np.linspace(T_min, T_max, options.points(51)) + 273.15
    x_grid, y_grid = np.meshgrid(range_x, range_y)
    prop_dict = fluid.get_state(
        jxp.SmassT_INPUTS, x_grid, y_grid, generalize_quality=False
    )
    contour = ax.contour(
        prop_dict["s"],
        prop_dict["T"],
        prop_dict["p"] / 1e5,
        levels=np.asarray(np.linspace(60, 100, 11)),
        cmap=colormap,
        zorder=1,
    )

    contourf = ax.contourf(
        prop_dict["s"],
        prop_dict["T"],
        prop_dict["p"] / 1e5,
        levels=np.asarray(np.linspace(60, 100, 11)),
        cmap=colormap,
        zorder=1,
    )
    contourf.set_visible(False)

    # Convert only the thermodynamic axes before adding the pressure colorbar.
    # The graphics helpers transform artists, but do not update axis limits.
    jxp.scale_graphics_x(fig, +1e-3, mode="multiply")
    jxp.scale_graphics_y(fig, -273.15, mode="add")
    ax.set_xlim([s_min, s_max])
    ax.set_ylim([T_min, T_max])
    colorbar = fig.colorbar(contourf, ax=ax, label="Pressure (bar)")

    # Keep displayed units and the independent pressure scale explicit.
    np.testing.assert_allclose(ax.get_xlim(), [s_min, s_max])
    np.testing.assert_allclose(ax.get_ylim(), [T_min, T_max])
    np.testing.assert_allclose(colorbar.ax.get_ylim(), [60, 100])
    options.record(
        "display_coordinates",
        dict(
            entropy_kJ_per_kg_K=ax.get_xlim(),
            temperature_C=ax.get_ylim(),
            pressure_bar=colorbar.ax.get_ylim(),
            contour_vertices=sum(len(path.vertices) for path in contour.get_paths()),
        ),
    )

    # Label the pressure contours actually shown in this figure.
    ax.clabel(contour, inline=True, fmt=lambda value: f"{value:g} bar", fontsize=8)

    # Adjust pad
    fig.tight_layout(pad=1)

    # Save figure
    filename = os.path.join(outdir, "Ts_diagram_isobars")
    jxp.savefig_in_formats(fig, filename)

    # Show figures
    options.check_states(locals(), allow_failures=True)
    options.record(
        "execution", "Scientific example completed; see numerical-check records."
    )
    return options.finish()


if __name__ == "__main__":
    main()
