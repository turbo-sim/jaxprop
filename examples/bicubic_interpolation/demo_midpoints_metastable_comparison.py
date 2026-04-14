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
# fluid_name = "nitrogen"
# h_min = -145e3  # J/kg
# h_max = -10e3  # J/kg
# p_min = 3e4  # Pa
# p_max = 1e6  # Pa
# N_p = 165 # Number of pressure points
# N_h = 165 # Number of enthalpy points
# metastable_phase = "liquid"

fluid_name = "nitrogen"
h_min = 20e3  # J/kg
h_max = 200e3  # J/kg
p_min = 3e4  # Pa
p_max = 1e6  # Pa
N_p = 165 # Number of pressure points
N_h = 165 # Number of enthalpy points
metastable_phase = "vapor"

outdir = "demo_metastable_table_generation"

# ---------------------------
# Build bicubic table
# ---------------------------
fluid_bicubic = jxp.FluidBicubic(
    fluid_name=fluid_name,
    backend="HEOS",
    h_min=h_min, h_max=h_max,
    p_min=p_min, p_max=p_max,
    N_h=N_h, N_p=N_p,
    table_dir=outdir,
    metastable_phase=metastable_phase,
    gradient_method="forward",
)

# ---------------------------
# FluidJax reference with get_state_metastable
# ---------------------------
fluid_ref = jxp.Fluid(fluid_name, backend="HEOS")
print(f"Metastable {metastable_phase}")

# ---------------------------
# Midpoint grid (evaluation points)
# ---------------------------
h_nodes = np.linspace(h_min, h_max, int(N_h*0.5))
p_nodes = np.exp(np.linspace(np.log(p_min), np.log(p_max), int(N_p*0.5)))
h_vals = 0.5 * (h_nodes[:-1] + h_nodes[1:])
p_vals = 0.5 * (p_nodes[:-1] + p_nodes[1:])
H_mesh, P_mesh = np.meshgrid(h_vals, p_vals, indexing="ij")

# ---------------------------
# Evaluate bicubic interpolation (vectorized)
# ---------------------------
interp_grid = {prop: np.full(H_mesh.shape, np.nan) for prop in jxp.PROPERTIES_CANONICAL}
total = H_mesh.size
print(f"Evaluating bicubic interpolation ({total} points)...")
for count, idx in enumerate(np.ndindex(H_mesh.shape)):
    if count % 1000 == 0:
        print(f"  {count}/{total} points evaluated", flush=True)
    state = fluid_bicubic.get_state(jxp.HmassP_INPUTS, float(H_mesh[idx]), float(P_mesh[idx]))
    for prop in jxp.PROPERTIES_CANONICAL:
        interp_grid[prop][idx] = float(state[prop])
print(f"  {total}/{total} points evaluated")
# ---------------------------
# Evaluate FluidJax reference (sweep left-to-right per pressure)
# ---------------------------
properties = ["temperature", "density", "entropy", "gruneisen"]

true_grid = {prop: np.full(H_mesh.shape, np.nan) for prop in properties}

# H_mesh has shape (N_h_mid, N_p_mid) with indexing="ij"
# Rows = enthalpy, Columns = pressure
# Sweep left-to-right in enthalpy (row index) for each pressure (column index)
n_h_mid, n_p_mid = H_mesh.shape
total = H_mesh.size
count = 0
print(f"Evaluating FluidJax get_state_metastable ({total} points)...")

for j in range(n_p_mid):
    p_val = float(P_mesh[0, j])

    # Get initial guess from single-phase state at leftmost enthalpy
    h_start = float(H_mesh[0, j])
    try:
        state_init = fluid_ref.get_state(jxp.HmassP_INPUTS, h_start, p_val)
        rho_guess = float(state_init.rho)
        T_guess = float(state_init.T)
    except Exception:
        rho_guess = None
        T_guess = None

    for i in range(n_h_mid):
        if count % 1000 == 0:
            print(f"  {count}/{total} points evaluated", flush=True)
        h_val = float(H_mesh[i, j])

        rhoT_guess = [rho_guess, T_guess] if (rho_guess is not None) else None
        try:
            state = fluid_ref.get_state_metastable(
                prop_1="h",
                prop_1_value=h_val,
                prop_2="p",
                prop_2_value=p_val,
                rhoT_guess=rhoT_guess,
                supersaturation=True,
                generalize_quality=True,
            )
            for prop in properties:
                true_grid[prop][i, j] = float(state[prop])
            # Update guess for next enthalpy point
            rho_guess = float(state.rho)
            T_guess = float(state.T)
        except Exception:
            pass  # leave as NaN, guess unchanged

        count += 1

print(f"  {total}/{total} points evaluated")

# ---------------------------
# Collect error statistics
# ---------------------------
error_summary = []

for prop in properties:
    interp_val = np.array(interp_grid[prop])
    true_val = true_grid[prop]

    abs_err = np.abs(interp_val - true_val)
    rel_err = abs_err / np.maximum(np.abs(true_val), 1e-30) * 100.0

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
print("\nProperty interpolation error summary (bicubic vs CoolProp specify_phase, midpoints):")
print(f"{'Prop':<25} {'| Abs min':>12} {'Abs max':>12} {'% min':>12} {'% max':>12} {'% mean':>12}")
print("-" * 80)
for e in error_summary:
    print(f"{e['property']:<25} | {e['abs_min']:>12.3e} {e['abs_max']:>12.3e} "
          f"{e['rel_min']:>12.3e} {e['rel_max']:>12.3e} {e['rel_mean']:>12.3e}")

# ---------------------------
# Plot subplot grid for selected properties
# ---------------------------
n_props = len(properties)
ncols = 3
nrows = int(np.ceil(n_props / ncols))

fig = plt.figure(figsize=(4 * ncols, 3 * nrows))
gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.05], figure=fig)

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
    fluid_bg = jxp.Fluid(fluid_name)
    fluid_bg.plot_phase_diagram(
        x_prop="enthalpy", y_prop="pressure", axes=ax,
        x_scale="linear", y_scale="log"
    )

    # Contour
    cf = ax.contourf(
        H_mesh, P_mesh, rel_err_field,
        levels=levels,
        norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
        cmap="viridis", extend="both"
    )

    # Interpolation domain box
    ax.plot(
        [h_min, h_max, h_max, h_min, h_min],
        [p_min, p_min, p_max, p_max, p_min],
        "r--", linewidth=1.5,
    )

    ax.set_title(f"{prop} [% error]", fontsize=11)
    ax.set_xlabel("Enthalpy [J/kg]")
    ax.set_ylabel("Pressure [Pa]")

# Colorbar axis on the right
cax = fig.add_subplot(gs[:, -1])
cb = fig.colorbar(cf, cax=cax)
cb.set_ticks(levels)
cb.set_label("Relative error [%] (log scale)")

# Remove unused axes
total_axes = nrows * ncols
for i in range(n_props, total_axes):
    r, c = divmod(i, ncols)
    fig.delaxes(fig.add_subplot(gs[r, c]))

plt.tight_layout(pad=1)

# Show figures
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()
