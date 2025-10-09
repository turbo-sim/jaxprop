import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import CoolProp.CoolProp as cp

import jaxprop as jxp

# ---------------------------
# Configuration
# ---------------------------
outdir = "fluid_tables"
fluid_name = "CO2"
h_min = 200e3  # J/kg
h_max = 600e3  # J/kg
p_min = 2e6    # Pa
p_max = 20e6   # Pa
N = 80

fluid_bicubic = jxp.FluidBicubic(
    fluid_name=fluid_name,
    h_min=h_min,
    h_max=h_max,
    p_min=p_min,
    p_max=p_max,
    N=N,
    table_dir=outdir,
)

# Reference CoolProp fluid
fluid_cp = jxp.FluidJAX(fluid_name)

# ---------------------------
# Midpoint grid
# ---------------------------
h_nodes = jnp.linspace(h_min, h_max, N)
p_nodes = jnp.exp(jnp.linspace(jnp.log(p_min), jnp.log(p_max), N))

h_vals = 0.5 * (h_nodes[:-1] + h_nodes[1:])
p_vals = 0.5 * (p_nodes[:-1] + p_nodes[1:])

H_mesh, P_mesh = np.meshgrid(h_vals, p_vals, indexing="ij")

# ---------------------------
# Properties to test
# ---------------------------
properties = ["p", "h", "T", "mu", "d"]

# ---------------------------
# Loop over properties
# ---------------------------
for prop in properties:
    print(f"\nTesting: {prop}")

    interp_grid = np.zeros_like(H_mesh)
    true_grid = np.zeros_like(H_mesh)

    for i in range(H_mesh.shape[0]):
        for j in range(H_mesh.shape[1]):
            h = float(H_mesh[i, j])
            P = float(P_mesh[i, j])

            try:
                interp_props = fluid_bicubic.get_props(jxp.HmassP_INPUTS, h, P)
                interp_val = interp_props[prop]

                cp_props = fluid_cp.get_props(jxp.HmassP_INPUTS, h, P)
                true_val = cp_props[prop]

                interp_grid[i, j] = interp_val
                true_grid[i, j] = true_val
            except Exception:
                interp_grid[i, j] = np.nan
                true_grid[i, j] = np.nan

    rel_error = np.abs((interp_grid - true_grid) / true_grid)
    percent_error = rel_error * 100

    print(f"   - Abs error: min={np.nanmin(np.abs(interp_grid - true_grid)):.3e}, "
          f"max={np.nanmax(np.abs(interp_grid - true_grid)):.3e}")
    print(f"   - % error: min={np.nanmin(percent_error):.3e}%, "
          f"max={np.nanmax(percent_error):.3e}%, mean={np.nanmean(percent_error):.3e}%")

    # ---------------------------
    # Plot error contour
    # ---------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    fluid_cp.fluid.plot_phase_diagram(
        x_prop="enthalpy", y_prop="pressure", axes=ax, x_scale="linear", y_scale="log"
    )

    levels = np.logspace(-6, 2, 9)
    masked_error = np.clip(percent_error, levels[0], levels[-1])

    contour = ax.contourf(
        H_mesh, P_mesh, masked_error,
        levels=levels,
        norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
        cmap="viridis", extend="both"
    )

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_ticks(levels)
    cbar.set_ticklabels([f"$10^{{{int(np.log10(l))}}}$" for l in levels])
    cbar.set_label("Percentage error [%] (log scale)")

    ax.set_xlabel("Enthalpy [J/kg]")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_title(f"Percentage Error (Midpoints): {prop}")
    fig.tight_layout(pad=1)
    plt.show()
