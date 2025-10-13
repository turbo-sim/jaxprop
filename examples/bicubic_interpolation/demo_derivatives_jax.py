import os
import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec


jxp.set_plot_options()

# ---------------------------
# Configuration
# ---------------------------
outdir = "fluid_tables"
fluid_name = "CO2"
h_min = 500e3  # J/kg
h_max = 1000e3  # J/kg
p_min = 2e6    # Pa
p_max = 20e6   # Pa
N_h = 32
N_p = 32
Nh_chk = Np_chk = 50

# ------------------------------------------
# Fluid objects
# ------------------------------------------
fluid_coolprop = jxp.FluidJAX(fluid_name, backend="HEOS")
fluid_bicubic = jxp.FluidBicubic(
    fluid_name=fluid_name,
    backend="HEOS",
    h_min=h_min, h_max=h_max,
    p_min=p_min, p_max=p_max,
    N_h=N_h, N_p=N_p,
    table_dir=outdir,
    table_name="derivative_check"
)

# ------------------------------------------
# Grid
# ------------------------------------------
h_vals = jnp.linspace(h_min, h_max, Nh_chk)
p_vals = jnp.linspace(p_min, p_max, Np_chk)
h_grid, p_grid = jnp.meshgrid(h_vals, p_vals, indexing="ij")

# Base state on (H,P)
states_grid = fluid_bicubic.get_state(jxp.HmassP_INPUTS, h_grid, p_grid)
rho_grid = states_grid["density"]
s_grid = states_grid["entropy"]

# ------------------------------------------
# 1. Check dT/dh|p vs 1/cp
# ------------------------------------------
def T_of_h_scalar(h, p):
    return fluid_bicubic.get_state(jxp.HmassP_INPUTS, h, p)["temperature"]

def dTdh_scalar(h, p):
    return jax.grad(lambda hh: T_of_h_scalar(hh, p))(h)

dTdh_flat = jax.vmap(dTdh_scalar)(h_grid.ravel(), p_grid.ravel())
cp_jax = 1/dTdh_flat.reshape(h_grid.shape)
cp_table = states_grid["isobaric_heat_capacity"]

rel_err_cp = (cp_jax - cp_table) / cp_table * 100.0

print("\n[Heat capacity check]")
print(f"  min = {np.nanmin(rel_err_cp):.3e} %, max = {np.nanmax(rel_err_cp):.3e} %, mean = {np.nanmean(rel_err_cp):.3e} %")

tol_cp = 1e-2  # tolerance in %
max_err_cp = np.nanmax(rel_err_cp)
if max_err_cp > tol_cp:
    raise ValueError(f"[Heat capacity check failed] max relative error = {max_err_cp:.3e} % > {tol_cp:.3e} %")


# ------------------------------------------
# 2. Check a**2 = (∂p/∂ρ)_s
# ------------------------------------------
def p_of_rho_s_scalar(rho, s):
    return fluid_bicubic.get_state(jxp.DmassSmass_INPUTS, rho, s)["pressure"]

def a_from_derivative_scalar(rho, s):
    dp_drho = jax.grad(lambda rr: p_of_rho_s_scalar(rr, s))(rho)
    return jnp.sqrt(jnp.maximum(dp_drho, 0.0))

a_deriv_flat = jax.vmap(a_from_derivative_scalar)(rho_grid.ravel(), s_grid.ravel())
a_jax = a_deriv_flat.reshape(rho_grid.shape)
a_table = states_grid["speed_sound"]

rel_err_a = (a_jax - a_table) / jnp.maximum(a_table, 1e-16) * 100.0

print("\n[Speed of sound check]")
print(f"  min = {np.nanmin(rel_err_a):.3e} %, max = {np.nanmax(rel_err_a):.3e} %, mean = {np.nanmean(rel_err_a):.3e} %")


tol_a = 1e-2  # tolerance in %
max_err_a = np.nanmax(rel_err_a)
if max_err_a > tol_a:
    raise ValueError(f"[Speed of sound check failed] max relative error = {max_err_a:.3e} % > {tol_a:.3e} %")

# ------------------------------------------
# Plot derivative consistency errors
# ------------------------------------------

# Data to plot
error_fields = [
    np.abs(rel_err_cp),
    np.abs(rel_err_a),
]
titles = ["Heat capacity comparison", "Speed of sound comparison"]
n_props = len(error_fields)
ncols = 2
nrows = 1

# Levels for log contour
levels = np.logspace(-6, 2, 9)

fig = plt.figure(figsize=(6*ncols, 4*nrows))
gs = gridspec.GridSpec(nrows, ncols+1, width_ratios=[1]*ncols + [0.05], figure=fig)

axes = []
for i in range(n_props):
    r, c = divmod(i, ncols)
    ax = fig.add_subplot(gs[r, c])
    ax.set_yscale("log")
    axes.append(ax)

# Reference phase diagram
for ax, err_field, title in zip(axes, error_fields, titles):
    # Clip error to avoid issues in LogNorm
    err_field = np.clip(err_field, levels[0], levels[-1])

    fluid_coolprop.fluid.plot_phase_diagram(
        x_prop="enthalpy", y_prop="pressure", axes=ax,
        x_scale="linear", y_scale="log"
    )

    c = ax.contourf(
        h_grid, p_grid, err_field,
        levels=levels,
        norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
        cmap="viridis", extend="both"
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Enthalpy [J/kg]")
    ax.set_ylabel("Pressure [Pa]")

# Colorbar axis on the right
cax = fig.add_subplot(gs[:, -1])
cb = fig.colorbar(c, cax=cax)
cb.set_ticks(levels)
cb.set_label("Relative error [%] (log scale)")

plt.tight_layout(pad=1)


# Show figures
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()
