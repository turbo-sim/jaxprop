import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import perf_counter

import jaxprop as jxp

# Create the folder to save figures
jxp.set_plot_options(grid=False)
outdir = "output"
os.makedirs(outdir, exist_ok=True)

# Solver to compute spinodal points
# method = "bfgs"
method = "slsqp"
# names = ["CO2", "water", "nitrogen", "ammonia", "butane", "R134a"]
# names = ["water"]
names = ["R134a"]
# names = ["R1233ZDE"]
# names = ["Carbon dioxide"]

for fluid_name in names:

    # Create fluid
    fluid = jxp.Fluid(
        name=fluid_name,
        backend="HEOS",
        exceptions=True,
    )

    # ---------------------------------------------------------------------------------- #
    # Density-pressure diagram and bulk modulus
    # ---------------------------------------------------------------------------------- #

    # Create figure
    prop_x = "rhomass"
    prop_y = "p"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_xlabel(r"Density (kg/m$^3$)")
    ax1.set_ylabel(r"Pressure (Pa)")
    ax2.set_xlabel(r"Density (kg/m$^3$)")
    ax2.set_ylabel(r"Isothermal bulk modulus (Pa)")
    ax1.set_xlim(
        sorted([fluid.triple_point_liquid[prop_x], fluid.triple_point_vapor[prop_x]])
    )
    ax1.set_ylim([fluid.triple_point_liquid[prop_y], 2 * fluid.critical_point[prop_y]])
    ax2.set_ylim([-2 * fluid.critical_point.p, 2 * fluid.critical_point.p])
    ax2.axhline(y=0, color="black")

    # Plot phase diagram
    time_start = perf_counter()
    fluid.plot_phase_diagram(
        prop_x,
        prop_y,
        axes=ax1,
        plot_saturation_line=True,
        plot_spinodal_line=True,
        spinodal_line_use_previous=False,
        spinodal_line_method=method,
        N=50,
    )
    time_end = perf_counter()
    print(
        f"Elapsed time for {fluid_name} phase diagram calculations {time_end-time_start:0.2f} seconds."
    )

    # Create entropy range
    rho_1 = 1.00*fluid.triple_point_vapor.rho
    rho_2 = 0.80*fluid.triple_point_liquid.rho
    rho_array = np.linspace(rho_1, rho_2, 500)

    # Plot metastable states
    T_array = fluid.critical_point.T - np.asarray([5, 10, 15, 20, 25, 30])
    states_meta = jxp.compute_property_grid_rhoT(fluid, rho_array, T_array)
    colormap = cm.magma(np.linspace(0.7, 0.1, len(T_array)))
    for i, T in enumerate(T_array):
        ax1.plot(
            states_meta[prop_x][i, :],
            states_meta[prop_y][i, :],
            color=colormap[i],
            label=f"$\\Delta T_{{crit}}={fluid.critical_point.T - T:0.0f}$ K"
        )
        ax2.plot(
            states_meta["rho"][i, :],
            states_meta["isothermal_bulk_modulus"][i, :],
            color=colormap[i],
            label=f"$\\Delta T_{{crit}}={fluid.critical_point.T-T:0.0f}$ K",
        )

    # # Plot equilibrium states
    # states_eq = bpy.compute_property_grid(fluid, bpy.DmassT_INPUTS, rho_array, T_array)
    # for i, T in enumerate(T_array):
    #     ax1.plot(
    #         states_eq[prop_x][i, :],
    #         states_eq[prop_y][i, :],
    #         color=colormap[i],
    #         linestyle="--"
    #     )

    # Plot spinodal points
    spinodal_liq, spinodal_vap = [], []
    for i, T in enumerate(T_array):

        # Liquid branch
        spinodal_liq.append(
            jxp.compute_spinodal_point(
                T,
                fluid,
                branch="liquid",
                method=method,
            )
        )
        ax1.plot(
            spinodal_liq[i][prop_x],
            spinodal_liq[i][prop_y],
            marker="o",
            markerfacecolor="w",
            color=colormap[i],
        )
        ax2.plot(
            spinodal_liq[i]["rho"],
            spinodal_liq[i]["isothermal_bulk_modulus"],
            marker="o",
            markerfacecolor="w",
            color=colormap[i],
        )

        # Vapor branch
        spinodal_vap.append(
            jxp.compute_spinodal_point(T, fluid, branch="vapor", method=method)
        )
        ax1.plot(
            spinodal_vap[i][prop_x],
            spinodal_vap[i][prop_y],
            marker="o",
            markerfacecolor="w",
            color=colormap[i],
        )
        ax2.plot(
            spinodal_vap[i]["rho"],
            spinodal_vap[i]["isothermal_bulk_modulus"],
            marker="o",
            markerfacecolor="w",
            color=colormap[i],
        )

    ax1.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="upper right", fontsize=10)
    fig.tight_layout(pad=1)
    jxp.savefig_in_formats(
        fig, os.path.join(outdir, f"spinodal_points_density_pressure_{fluid.name}")
    )

    # ---------------------------------------------------------------------------------- #
    # Temperature-entropy diagram
    # ---------------------------------------------------------------------------------- #

    # Create figure
    prop_x = "s"
    prop_y = "T"
    fig, ax3 = plt.subplots(figsize=(6, 5))
    ax3.set_xlabel("Entropy (J/kg/K))")
    ax3.set_ylabel("Temperature (K)")
    ax3.set_xlim(
        sorted([fluid.triple_point_liquid[prop_x], fluid.triple_point_vapor[prop_x]])
    )
    ax3.set_ylim(
        [fluid.triple_point_liquid[prop_y], 1.2 * fluid.critical_point[prop_y]]
    )

    # Plot phase diagram
    fluid.plot_phase_diagram(
        prop_x,
        prop_y,
        axes=ax3,
        plot_saturation_line=True,
        plot_spinodal_line=True,
    )

    # Plot isotherms and spinodal points
    colormap = cm.magma(np.linspace(0.7, 0.1, len(T_array)))
    for i, T in enumerate(T_array):

        # Isotherms
        ax3.plot(
            states_meta[prop_x][i, :],
            states_meta[prop_y][i, :],
            color=colormap[i],
            label=f"$\\Delta T_{{crit}}={fluid.critical_point.T-T:0.0f}$ K",
        )

        # Liquid spinodal points
        ax3.plot(
            spinodal_liq[i][prop_x],
            spinodal_liq[i][prop_y],
            marker="o",
            markerfacecolor="w",
            color=colormap[i],
        )

        # Vapor spinodal points
        ax3.plot(
            spinodal_vap[i][prop_x],
            spinodal_vap[i][prop_y],
            marker="o",
            markerfacecolor="w",
            color=colormap[i],
        )

    ax3.legend(loc="upper left", fontsize=10)
    fig.tight_layout(pad=1)
    jxp.savefig_in_formats(
        fig, os.path.join(outdir, f"spinodal_points_temperature_entropy_{fluid.name}")
    )


# Show figures
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()
