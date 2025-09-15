

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import jaxprop as jxp

# ---------------------------
# Configuration
# ---------------------------
outdir = "fluid_tables"
fluid_name = "CO2"
h_min = 500e3  # J/kg
h_max = 1000e3  # J/kg
p_min = 2e6    # Pa
p_max = 20e6   # Pa
N_h = 80
N_p = 80

# ---------------------------
# Build bicubic fluid object
# ---------------------------
fluid_coolprop = jxp.FluidJAX(fluid_name, backend="HEOS")
fluid_bicubic = jxp.FluidBicubic(
    fluid_name=fluid_name,
    backend="HEOS",
    h_min=h_min,
    h_max=h_max,
    p_min=p_min,
    p_max=p_max,
    N_h=N_h,
    N_p=N_p,
    table_dir=outdir,
)

# ---------------------------
# Define derivative function
# ---------------------------
def T_of_h(h, p, table):
    props = jxp.bicubic.interpolate_bicubic_hp(h, p, table)
    return props["temperature"]

# ∂T/∂h at constant p
dTdh_fun = jax.grad(lambda h, p, table: T_of_h(h, p, table))

# ---------------------------
# Grid consistency check
# ---------------------------
h_test = jnp.linspace(h_min, h_max, 40)
p_test = jnp.linspace(p_min, p_max, 40)
H, P = jnp.meshgrid(h_test, p_test, indexing="ij")

# Vectorized evaluation
dTdh_grid = jax.vmap(
    lambda h, p: dTdh_fun(h, p, fluid_bicubic.table)
)(H.ravel(), P.ravel()).reshape(H.shape)

props_grid = fluid_bicubic.get_props(jxp.HmassP_INPUTS, H, P)
inv_cp_grid = 1.0 / props_grid["isobaric_heat_capacity"]

rel_err_grid = np.array((dTdh_grid - inv_cp_grid) / inv_cp_grid * 100)

print("\nGrid consistency stats:")
print("  min err = %.3e %%  max err = %.3e %%  mean err = %.3e %%"
      % (np.nanmin(rel_err_grid), np.nanmax(rel_err_grid), np.nanmean(rel_err_grid)))

# ---------------------------
# Plot p–h diagram with error markers
# ---------------------------
fig, ax =fluid_coolprop.fluid.plot_phase_diagram(
    x_prop="enthalpy", y_prop="pressure", x_scale="linear", y_scale="log"
)

sc = ax.scatter(
    H, P,
    c=np.abs(rel_err_grid),
    cmap="viridis",
    norm=LogNorm(vmin=1e-6, vmax=100),
    s=30,
    edgecolor="k",
    linewidth=0.3,
    label=r"$\partial T / \partial h|_p$ vs $1/c_p$"
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Relative error [%] (log scale)")

# draw interpolation domain
ax.plot(
    [h_min, h_max, h_max, h_min, h_min],
    [p_min, p_min, p_max, p_max, p_min],
    "r--", linewidth=1.2, label="Interpolation domain"
)

ax.legend()
plt.tight_layout()
plt.show()
