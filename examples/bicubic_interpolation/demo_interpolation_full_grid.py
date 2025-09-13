import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import CoolProp.CoolProp as cp


import jaxprop as jxp


fig_dir = "output"
save_prefix = "test_table"
SAVE_FIGURES = True  # Set to False to just display plots


# ---------------------------
# Configuration
# ---------------------------
outdir = "fluid_tables"
fluid_name = "CO2"
h_min = 200e3     # J/kg
h_max = 600e3     # J/kg
p_min = 2e6       # Pa
p_max = 20e6      # Pa
N_h = 30           # Grid size
N_p = 30

# Generate Table
table = jxp.bicubic.get_or_create_property_table(
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

fluid = jxp.Fluid(fluid_name)


# ---------------------------
# Loop over grid and compute errors
# ---------------------------
prop_names = jxp.PROPERTIES_CANONICAL
Nh, Np = len(table["h_vals"]), len(table["p_vals"])

# storage: one array per property
values_bicubic = {prop: np.full((Nh, Np), np.nan) for prop in prop_names}
values_coolprop = {prop: np.full((Nh, Np), np.nan) for prop in prop_names}
rel_error      = {prop: np.full((Nh, Np), np.nan) for prop in prop_names}

from tqdm import tqdm

Nh, Np = len(table["h_vals"]), len(table["p_vals"])
total_points = Nh * Np

with tqdm(total=total_points, desc="Testing grid", ncols=80, ascii=True) as pbar:
    for i, h in enumerate(table["h_vals"]):
        for j, p in enumerate(table["p_vals"]):
            try:
                interp_vals = jxp.bicubic.interpolate_bicubic_hp(h, p, table)
                cp_vals     = fluid.get_state(cp.HmassP_INPUTS, h, p)

                for prop in prop_names:
                    interp_val = interp_vals[prop]
                    cp_val     = cp_vals[prop]

                    if cp_val != 0 and np.isfinite(cp_val):
                        err = abs((interp_val - cp_val) / cp_val)
                    else:
                        err = np.nan

                    values_bicubic[prop][i, j] = interp_val
                    values_coolprop[prop][i, j] = cp_val
                    rel_error[prop][i, j] = err

            except Exception:
                # leave NaNs in arrays if state evaluation fails
                pass
            finally:
                pbar.update(1)
# ---------------------------
# Report statistics
# ---------------------------
for prop in prop_names:
    percent_error = rel_error[prop] * 100
    print(f"\nTesting: {prop}")
    print(f"   → Abs error: min={np.nanmin(np.abs(values_bicubic[prop] - values_coolprop[prop])):.3e}, "
          f"max={np.nanmax(np.abs(values_bicubic[prop] - values_coolprop[prop])):.3e}")
    print(f"   → % error:  min={np.nanmin(percent_error):.3e}%, "
          f"max={np.nanmax(percent_error):.3e}%, mean={np.nanmean(percent_error):.3e}%")
    

# # # ---------------------------
# # # Load grid values
# # # ---------------------------
# # h_vals = jnp.linspace(h_min, h_max, N_h)[:-1]         # Include left boundary, exclude right
# # logP_vals = jnp.linspace(jnp.log(p_min), jnp.log(p_max), N_p)[:-1]
# # P_vals = jnp.exp(logP_vals)


# fluid = jxp.Fluid(fluid_name)


# # ---------------------------
# # Loop over grid and compute errors
# # ---------------------------
# for prop in jxp.PROPERTIES_CANONICAL:
#     values_bicubic = np.zeros((len(table["h_vals"]), len(table["p_vals"])))
#     values_coolprop = np.zeros_like(values_bicubic)
#     rel_error = np.zeros_like(values_bicubic)

#     print(f"\nTesting: {prop}")

#     for i, h in enumerate(table["h_vals"]):
#         for j, p in enumerate(table["p_vals"]):

#             try:
#                 interp_val = jxp.bicubic.bicubic_interpolant_property(h, p, table)[prop]
#                 cp_val = fluid.get_state(cp.HmassP_INPUTS, h, p)[prop]
#                 values_bicubic[i, j] = interp_val
#                 values_coolprop[i, j] = cp_val
#                 rel_error[i, j] = np.abs((interp_val - cp_val) / cp_val)
#             except Exception:
#                 values_bicubic[i, j] = np.nan
#                 values_coolprop[i, j] = np.nan
#                 rel_error[i, j] = np.nan

#     percent_error = rel_error * 100

#     print(f"   → Abs error: min={np.nanmin(np.abs(values_bicubic - values_coolprop)):.3e}, "
#           f"max={np.nanmax(np.abs(values_bicubic - values_coolprop)):.3e}")
#     print(f"   → % error: min={np.nanmin(percent_error):.3e}%, "
#           f"max={np.nanmax(percent_error):.3e}%, mean={np.nanmean(percent_error):.3e}%")





    # # ---------------------------
    # # Plot error contour
    # # ---------------------------
    # H_mesh, P_mesh = np.meshgrid(h_vals, P_vals / 1e5, indexing='ij')  # P in bar

    # plt.figure(figsize=(8, 5))
    # levels = np.logspace(-12, 2, 15)
    # masked_error = np.clip(percent_error, levels[0], levels[-1])  # avoid 0 or NaNs
    # contour = plt.contourf(
    #     H_mesh, P_mesh, masked_error,
    #     levels=levels,
    #     norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
    #     cmap='viridis', extend='both'
    # )

    # cbar = plt.colorbar(contour)
    # cbar.set_ticks(levels)
    # cbar.set_ticklabels([f"$10^{{{int(np.log10(l))}}}$" for l in levels])
    # cbar.set_label("Percentage error [%] (log scale)")

    # plt.xlabel("Enthalpy [J/kg]")
    # plt.ylabel("Pressure [bar]")
    # plt.title(f"Percentage Error: {prop}")

    # # ---------------------------
    # # Plot saturation lines
    # # ---------------------------
    # P_sats = np.linspace(p_min, p_max, 200)
    # h_l, h_v, P_sats_bar = [], [], []

    # for P_sat in P_sats:
    #     try:
    #         hl = cp.PropsSI('H', 'P', P_sat, 'Q', 0, fluid_name)
    #         hv = cp.PropsSI('H', 'P', P_sat, 'Q', 1, fluid_name)
    #         h_l.append(hl)
    #         h_v.append(hv)
    #         P_sats_bar.append(P_sat / 1e5)
    #     except:
    #         continue

    # plt.plot(h_l, P_sats_bar, 'w--', lw=1.5, label='Saturation Liquid')
    # plt.plot(h_v, P_sats_bar, 'w--', lw=1.5, label='Saturation Vapor')
    # plt.legend()

    # plt.tight_layout()

    # # ---------------------------
    # # Save or show
    # # ---------------------------
    # if SAVE_FIGURES:
    #     fig_dir = os.path.join(fig_dir)
    #     os.makedirs(fig_dir, exist_ok=True)
    #     fig_path = os.path.join(fig_dir, f"{prop}_interp_error_contour.png")
    #     plt.savefig(fig_path, dpi=300)
    #     print(f"Saved figure: {fig_path}")
    #     plt.close()
    # else:
    #     plt.show()
