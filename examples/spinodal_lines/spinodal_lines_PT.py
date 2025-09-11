import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import cm

import jaxprop.coolprop as jxp

# Create the folder to save figures
jxp.set_plot_options(grid=False)
colors = jxp.COLORS_MATLAB
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)


# Solver to compute spinodal points
# names = ["CO2", "water", "nitrogen", "ammonia", "butane", "R134a"]
names = ["CO2"]


for fluid_name in names:

    # Create fluid
    fluid = jxp.Fluid(
        name=fluid_name,
        backend="HEOS",
        exceptions=True,
    )

    # Compute saturation and spinodal lines
    spinodal_liq, spinodal_vap = jxp.compute_spinodal_line(fluid, N=50)
    saturation_liq, saturation_vap = jxp.compute_saturation_line(fluid, N=50)

    # ------------------------------------------------------------- #
    # Plot metastable liquid region
    # ------------------------------------------------------------- #
    prop_x = "T"
    prop_y = "p"
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlabel(r"$T/T_\text{crit}$ - Reduced temperature")
    ax.set_ylabel(r"$p/p_\text{crit}$ - Reduced pressure")
    ax.set_xlim([fluid.triple_point_liquid[prop_x]/fluid.critical_point[prop_x], 1])
    ax.set_ylim([0, 1])

    # Plot subcooled liquid region
    x = jnp.concatenate([
        saturation_liq[prop_x],
        jnp.array([fluid.triple_point_liquid[prop_x],
                1.1 * fluid.critical_point[prop_x]])
    ])
    y = jnp.concatenate([
        saturation_liq[prop_y],
        jnp.array([2 * fluid.critical_point[prop_y],
                2 * fluid.critical_point[prop_y]])
    ])
    ax.fill(x, y, colors[0], alpha=0.7, label="Subcooled liquid")

    # Plot metastable liquid region
    x = jnp.concatenate([jnp.flip(saturation_liq[prop_x]), spinodal_liq[prop_x]])
    y = jnp.concatenate([jnp.flip(saturation_liq[prop_y]), spinodal_liq[prop_y]])
    ax.fill(x, y, colors[0], alpha=0.5, label="Metastable liquid")

    # Plot superheated vapor region
    x = jnp.concatenate([
        spinodal_liq[prop_x],
        jnp.array([fluid.triple_point_liquid[prop_x],
                1.1 * fluid.critical_point[prop_x]])
    ])
    y = jnp.concatenate([
        spinodal_liq[prop_y],
        jnp.array([-2 * fluid.critical_point[prop_y],
                -2 * fluid.critical_point[prop_y]])
    ])
    ax.fill(x, y, colors[1], alpha=0.7, label="Superheated vapor")

    # Plot spinodal lines
    ax.plot(
        spinodal_liq[prop_x],
        spinodal_liq[prop_y],
        color=colors[0],
        label="Liquid spinodal line"
    )
    ax.plot(
        spinodal_vap[prop_x],
        spinodal_vap[prop_y],
        color=colors[1],
        label="Vapor spinodal line"
    )

    # Plot saturation line
    ax.plot(
        saturation_liq[prop_x],
        saturation_liq[prop_y],
        color="black",
        label="Saturation line"
    )

    # Plot critical point
    ax.plot(
        fluid.critical_point[prop_x],
        fluid.critical_point[prop_y],
        color="black",
        marker="o",
    )

    ax.legend(loc="upper left", fontsize=12)
    jxp.scale_graphics_x(fig, 1/fluid.critical_point[prop_x], mode="multiply")
    jxp.scale_graphics_y(fig, 1/fluid.critical_point[prop_y], mode="multiply")
    fig.tight_layout(pad=1)
    jxp.savefig_in_formats(fig, os.path.join(out_dir, f"metastable_liquid_{fluid.name}"))


    # ------------------------------------------------------------- #
    # Plot metastable vapor region
    # ------------------------------------------------------------- #
    prop_x = "T"
    prop_y = "p"
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlabel(r"$T/T_\text{crit}$ - Reduced temperature")
    ax.set_ylabel(r"$p/p_\text{crit}$ - Reduced pressure")
    ax.set_xlim([fluid.triple_point_liquid[prop_x]/fluid.critical_point[prop_x], 1])
    ax.set_ylim([0, 1])

    # Plot subcooled liquid region
    x = jnp.concatenate([
        spinodal_vap[prop_x],
        jnp.array([fluid.triple_point_liquid[prop_x], 1.1 * fluid.critical_point[prop_x]])
    ])
    y = jnp.concatenate([
        spinodal_vap[prop_y],
        jnp.array([2 * fluid.critical_point[prop_y], 2 * fluid.critical_point[prop_y]])
    ])
    ax.fill(x, y, colors[0], alpha=0.7, label="Subcooled liquid")

    # Plot metastable vapor region
    x = jnp.concatenate([
        jnp.flip(saturation_vap[prop_x]),
        spinodal_vap[prop_x]
    ])
    y = jnp.concatenate([
        jnp.flip(saturation_vap[prop_y]),
        spinodal_vap[prop_y]
    ])
    ax.fill(x, y, colors[1], alpha=0.5, label="Metastable vapor")

    # Plot superheated vapor region
    x = jnp.concatenate([
        saturation_vap[prop_x],
        jnp.array([fluid.triple_point_liquid[prop_x], 1.1 * fluid.critical_point[prop_x]])
    ])
    y = jnp.concatenate([
        saturation_vap[prop_y],
        jnp.array([-2 * fluid.critical_point[prop_y], -2 * fluid.critical_point[prop_y]])
    ])
    ax.fill(x, y, colors[1], alpha=0.7, label="Superheated vapor")

    # Plot spinodal lines
    ax.plot(
        spinodal_liq[prop_x],
        spinodal_liq[prop_y],
        color=colors[0],
        label="Liquid spinodal line"
    )
    ax.plot(
        spinodal_vap[prop_x],
        spinodal_vap[prop_y],
        color=colors[1],
        label="Vapor spinodal line"
    )

    # Plot saturation line
    ax.plot(
        saturation_liq[prop_x],
        saturation_liq[prop_y],
        color="black",
        label="Saturation line"
    )

    # Plot critical point
    ax.plot(
        fluid.critical_point[prop_x],
        fluid.critical_point[prop_y],
        color="black",
        marker="o",
    )

    ax.legend(loc="upper left", fontsize=12)
    jxp.scale_graphics_x(fig, 1/fluid.critical_point[prop_x], mode="multiply")
    jxp.scale_graphics_y(fig, 1/fluid.critical_point[prop_y], mode="multiply")
    fig.tight_layout(pad=1)
    jxp.savefig_in_formats(fig, os.path.join(out_dir, f"metastable_vapor_{fluid.name}"))


# Show figures
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()