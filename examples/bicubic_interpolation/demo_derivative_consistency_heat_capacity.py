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
# Scalar T(h,p) and ∂T/∂h|p
# ---------------------------
def T_of_h_scalar(h, p):
    """Temperature from table at scalar (h, p)."""
    return fluid_bicubic.get_state(jxp.HmassP_INPUTS, h, p)["temperature"]

def dTdh_scalar(h, p):
    """Derivative ∂T/∂h at constant p, evaluated with grad on scalars."""
    return jax.grad(lambda hh: T_of_h_scalar(hh, p))(h)

# Vectorized version over flattened arrays
dTdh_vectorized = jax.jit(jax.vmap(dTdh_scalar, in_axes=(0, 0)))

# ---------------------------
# Grid consistency check
# ---------------------------
h_test = jnp.linspace(h_min, h_max, 25)
p_test = jnp.linspace(p_min, p_max, 25)
H, P = jnp.meshgrid(h_test, p_test, indexing="ij")

# Flatten meshgrid
H_flat = H.ravel()
P_flat = P.ravel()

# Evaluate ∂T/∂h|p at each grid point
dTdh_flat = dTdh_vectorized(H_flat, P_flat)
dTdh_grid = dTdh_flat.reshape(H.shape)

# Get 1/cp on the same grid
props_grid = fluid_bicubic.get_state(jxp.HmassP_INPUTS, H, P)
inv_cp_grid = 1.0 / props_grid["isobaric_heat_capacity"]

# Relative error (%)
rel_err_grid = (dTdh_grid - inv_cp_grid) / inv_cp_grid * 100.0
rel_err_grid = np.array(rel_err_grid)

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
