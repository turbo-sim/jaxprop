import os
import numpy as np
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

jxp.set_plot_options()

# ---------------------------
# Configuration
# ---------------------------
outdir = "fluid_tables"
# fluid_name = "CO2"
# h_min = 200e3  # J/kg
# h_max = 600e3  # J/kg
# p_min = 2e6    # Pa
# p_max = 20e6   # Pa
# N_h = 32
# N_p = 32

fluid_name = "air"
h_min = 50e3  # J/kg
h_max = 600e3  # J/kg
p_min = 0.6e5    # Pa
p_max = 1.5e5   # Pa
N_h = 32
N_p = 32

# ---------------------------
# Build models
# ---------------------------
fluid_bicubic = jxp.FluidBicubic(
    fluid_name=fluid_name,
    backend="HEOS",
    h_min=h_min, h_max=h_max,
    p_min=p_min, p_max=p_max,
    N_h=N_h, N_p=N_p,
    table_dir=outdir,
    table_name="interpolation_midpoints"
)

fluid_cp = jxp.FluidJAX(fluid_name, exceptions=False)  # Mind exceptions!

# ---------------------------
# Midpoint grid (evaluation points)
# ---------------------------
h_nodes = jnp.linspace(h_min, h_max, N_h)
p_nodes = jnp.exp(jnp.linspace(jnp.log(p_min), jnp.log(p_max), N_p))
h_vals = 0.5 * (h_nodes[:-1] + h_nodes[1:])
p_vals = 0.5 * (p_nodes[:-1] + p_nodes[1:])
H_mesh, P_mesh = np.meshgrid(h_vals, p_vals, indexing="ij")

# ---------------------------
# Evaluate states once (vectorized)
# ---------------------------
interp_grid = fluid_bicubic.get_state(jxp.HmassP_INPUTS, H_mesh, P_mesh)
true_grid   = fluid_cp.get_state(jxp.HmassP_INPUTS, H_mesh, P_mesh)

# ---------------------------
# Collect error statistics
# ---------------------------
error_summary = []
properties = [ "T", "d", "e", "s", "a", "G"]

for prop in properties:
    interp_val = np.array(interp_grid[prop])
    true_val   = np.array(true_grid[prop])

    abs_err = np.abs(interp_val - true_val)
    rel_err = abs_err / np.maximum(np.abs(true_val), 1e-30) * 100.0  # %

    error_summary.append({
        "property": prop,
        "abs_min": np.nanmin(abs_err),
        "abs_max": np.nanmax(abs_err),
        "rel_min": np.nanmin(rel_err),
        "rel_max": np.nanmax(rel_err),
        "rel_mean": np.nanmean(rel_err),
        "field": rel_err,
    })

# ---------------------------
# Print error summary table
# ---------------------------
print("\nProperty interpolation error summary (evaluated at cell midpoints):")
print(f"{'Prop':<5} {'| Abs min':>12} {'Abs max':>12} {'% min':>12} {'% max':>12} {'% mean':>12}")
print("-" * 70)
for e in error_summary:
    print(f"{e['property']:<5} | {e['abs_min']:>12.3e} {e['abs_max']:>12.3e} "
          f"{e['rel_min']:>12.3e} {e['rel_max']:>12.3e} {e['rel_mean']:>12.3e}")



# ---------------------------
# Plot subplot grid for selected properties
# ---------------------------
n_props = len(properties)
ncols = 3
nrows = int(np.ceil(n_props / ncols))

# Use GridSpec to reserve a narrow column for the colorbar
fig = plt.figure(figsize=(4*ncols, 3*nrows))
gs = gridspec.GridSpec(nrows, ncols+1, width_ratios=[1]*ncols + [0.05], figure=fig)

axes = []
for i in range(n_props):
    r, c = divmod(i, ncols)
    ax = fig.add_subplot(gs[r, c])
    axes.append(ax)

# Color levels
levels = np.logspace(-6, 2, 9)

# Plot each property
for ax, prop in zip(axes, properties):
    e = next(x for x in error_summary if x["property"] == prop)
    rel_err_field = np.clip(e["field"], levels[0], levels[-1])

    # Phase diagram background
    fluid_cp.fluid.plot_phase_diagram(
        x_prop="enthalpy", y_prop="pressure", axes=ax,
        x_scale="linear", y_scale="log"
    )

    # Contour
    c = ax.contourf(
        H_mesh, P_mesh, rel_err_field,
        levels=levels,
        norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
        cmap="viridis", extend="both"
    )

    ax.set_title(f"{prop} [% error]", fontsize=11)
    ax.set_xlabel("Enthalpy [J/kg]")
    ax.set_ylabel("Pressure [Pa]")

# Colorbar axis on the right
cax = fig.add_subplot(gs[:, -1])
cb = fig.colorbar(c, cax=cax)
cb.set_ticks(levels)
cb.set_label("Relative error [%] (log scale)")

# Remove unused axes (if any)
total_axes = nrows * ncols
for i in range(len(properties), total_axes):
    r, c = divmod(i, ncols)
    fig.delaxes(fig.add_subplot(gs[r, c]))

plt.tight_layout(pad=1)


# Show figures
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()
