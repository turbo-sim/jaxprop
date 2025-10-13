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
h_max = 1000e3 # J/kg
p_min = 2e6    # Pa
p_max = 20e6   # Pa
N_h = 80
N_p = 80

# ---------------------------
# Build models
# ---------------------------
fluid_coolprop = jxp.FluidJAX(fluid_name, backend="HEOS")
fluid_bicubic = jxp.FluidBicubic(
    fluid_name=fluid_name,
    backend="HEOS",
    h_min=h_min, h_max=h_max,
    p_min=p_min, p_max=p_max,
    N_h=N_h, N_p=N_p,
    table_dir=outdir,
)

# ---------------------------
# Scalar p(rho, s) and a(rho, s)
# ---------------------------
def p_of_rho_s_scalar(rho, s):
    """Pressure from bicubic model at scalar (rho, s)."""
    return fluid_bicubic.get_state(jxp.DmassSmass_INPUTS, rho, s)["pressure"]

def a_from_derivative_scalar(rho, s):
    """Speed of sound sqrt((∂p/∂rho)_s) at scalar (rho, s)."""
    dp_drho = jax.grad(lambda rr: p_of_rho_s_scalar(rr, s))(rho)
    return jnp.sqrt(jnp.maximum(dp_drho, 0.0))

# Vectorized version over flattened arrays
a_from_derivative_vectorized = jax.jit(jax.vmap(a_from_derivative_scalar, in_axes=(0, 0)))

# ---------------------------
# Grid consistency check
# ---------------------------
Nh_chk = 25
Np_chk = 25
H = jnp.linspace(h_min, h_max, Nh_chk)
P = jnp.linspace(p_min, p_max, Np_chk)
HH, PP = jnp.meshgrid(H, P, indexing="ij")

# States on the HP-grid via bicubic (guaranteed inside domain)
state_hp = fluid_bicubic.get_state(jxp.HmassP_INPUTS, HH, PP)
RHO = state_hp["density"]
S   = state_hp["entropy"]

# Flatten (rho, s)
RHO_flat = RHO.ravel()
S_flat   = S.ravel()

# 1) From derivative sqrt((∂p/∂rho)_s)
a_deriv_flat = a_from_derivative_vectorized(RHO_flat, S_flat)
a_deriv = a_deriv_flat.reshape(RHO.shape)

# 2) From bicubic's interpolated property at the same (H,P)
a_table = state_hp["speed_sound"]

# ---------------------------
# Errors (%)
# ---------------------------
eps = 1e-16
rel_err_table = (a_deriv - a_table) / jnp.maximum(a_table, eps) * 100.0
rel_err_table_np = np.array(rel_err_table)

print("\nSpeed of sound check (log % error magnitudes recommended near phase boundaries):")
print("vs table a        : min = %.3e %%  max = %.3e %%  mean = %.3e %%"
      % (np.nanmin(rel_err_table_np), np.nanmax(rel_err_table_np), np.nanmean(rel_err_table_np)))


# ---------------------------
# Plot p–h diagram with error markers
# ---------------------------
fig, ax = fluid_coolprop.fluid.plot_phase_diagram(
    x_prop="enthalpy", y_prop="pressure", x_scale="linear", y_scale="log"
)

# Scatter errors for derivative vs table
sc = ax.scatter(
    HH, PP,
    c=jnp.abs(rel_err_table),
    cmap="viridis",
    norm=LogNorm(vmin=1e-6, vmax=100),
    s=30,
    edgecolor="k",
    linewidth=0.3,
    label=r"$a=\sqrt{(\partial p/\partial \rho)_s}$ vs table $a$"
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Relative error to table a [%] (log scale)")

# Draw interpolation domain
ax.plot(
    [h_min, h_max, h_max, h_min, h_min],
    [p_min, p_min, p_max, p_max, p_min],
    "r--", linewidth=1.2, label="Interpolation domain"
)

ax.legend()
plt.tight_layout()
plt.show()
