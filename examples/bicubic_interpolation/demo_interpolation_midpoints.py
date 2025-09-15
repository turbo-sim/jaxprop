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
fluid_name = "CO2"
hmin_ = 200e3     # J/kg
h_max = 600e3     # J/kg
p_min = 2e6       # Pa
p_max = 20e6      # Pa
N_h = 60
N_p = 40
SAVE_FIGURES = False

# ---------------------------
# Build bicubic fluid object
# ---------------------------
fluid_bicubic = jxp.FluidBicubic(
    fluid_name=fluid_name,
    backend="HEOS",
    h_min=hmin_,
    h_max=h_max,
    p_min=p_min,
    p_max=p_max,
    N_h=N_h,
    N_p=N_p,
    table_dir="fluid_tables",
)

# Reference CoolProp fluid
fluid_cp = jxp.FluidJAX(fluid_name)

# ---------------------------
# Midpoint grid
# ---------------------------
h_nodes = jnp.linspace(hmin_, h_max, N_h)
p_nodes = jnp.exp(jnp.linspace(jnp.log(p_min), jnp.log(p_max), N_p))

h_vals = 0.5 * (h_nodes[:-1] + h_nodes[1:])
p_vals = 0.5 * (p_nodes[:-1] + p_nodes[1:])

# h_vals = h_nodes
# p_vals = p_nodes 
H_mesh, P_mesh = jnp.meshgrid(h_vals, p_vals, indexing="ij")

# ---------------------------
# Properties to test
# ---------------------------
# Bicubic interpolation (vectorized)
interp_props = fluid_bicubic.get_props(jxp.HmassP_INPUTS, H_mesh, P_mesh)

# Reference CoolProp values (vectorized via FluidJAX)
coolprop_props = fluid_cp.get_props(jxp.HmassP_INPUTS, H_mesh, P_mesh)

properties = ["p", "h", "a", "T", "mu"]

# ---------------------------
# Loop over properties
# ---------------------------
for prop in properties:
    print(f"\nTesting: {prop}")

    interp_grid = interp_props[prop]
    true_grid = coolprop_props[prop]

    # Relative error
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
    fluid_cp.fluid.plot_phase_diagram(x_prop="enthalpy", y_prop="pressure", axes=ax)

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
    ax.set_ylabel("Pressure [bar]")
    ax.set_title(f"Percentage Error (Midpoints): {prop}")
    fig.tight_layout(pad=1)
    plt.show()
        # # Save or show
    # if SAVE_FIGURES:
    #     os.makedirs("verification_figures", exist_ok=True)
    #     fig_path = os.path.join("verification_figures", f"{prop}_midpoint_interp_error_contour.png")
    #     plt.savefig(fig_path, dpi=300)
    #     print(f"Saved figure: {fig_path}")
    #     plt.close()
    # else:
plt.show()
