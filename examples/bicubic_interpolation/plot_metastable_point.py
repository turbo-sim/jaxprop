# %%
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import CoolProp.CoolProp as CP
import jaxprop.coolprop.fluid_properties as props
import jaxprop as jxp
import jax.numpy as jnp

# ---------------------------
# 1. Load Data
# ---------------------------
fluid_name = "water"
outdir = "demo_metastable_table_generation"
filename = f"{fluid_name}_meta_liquid_20x20.pkl"
file_path = os.path.join(outdir, filename)

with open(file_path, "rb") as f:
    data = pickle.load(f)

# %%

def get_values(data_dict, key):
    obj = data_dict.get(key)
    if isinstance(obj, dict):
        return obj.get('values', list(obj.values())[0])
    return obj

h_raw = get_values(data, "enthalpy")
p_raw = get_values(data, "pressure")
kappa_raw = get_values(data, "isothermal_bulk_modulus")
# kappa_raw = get_values(data, "temperature")

saturation_table = data.get("saturation_props")


# ---------------------------
# 2. Reshape & Flatten Logic
# ---------------------------
# Flatten kappa first to know the target size (N_total)
kappa_flat = np.array(kappa_raw).flatten()
N_total = kappa_flat.size

# for i, k in enumerate(kappa_flat):
#     if k > 0:
#         kappa_flat[i] = 1.0
#     else:
#         kappa_flat[i] = -1.0

print(kappa_flat)

# Flatten H and P
h_flat = np.array(h_raw).flatten()
p_flat = np.array(p_raw).flatten()

print(f"Shapes -> Kappa: {kappa_flat.shape}, H: {h_flat.shape}, P: {p_flat.shape}")

# Reconstruct grid if necessary
if h_flat.size != N_total or p_flat.size != N_total:
    print("Coordinate vectors are unique. creating Meshgrid...")
    h_unique = np.unique(h_flat)
    p_unique = np.unique(p_flat)
    
    if h_unique.size * p_unique.size == N_total:
        H_grid, P_grid = np.meshgrid(h_unique, p_unique, indexing='ij')
        if H_grid.size != kappa_raw.size: 
             H_grid, P_grid = np.meshgrid(h_unique, p_unique, indexing='xy')
        h_flat = H_grid.flatten()
        p_flat = P_grid.flatten()
    else:
        raise ValueError(f"Cannot reconstruct grid. Data size {N_total} vs H({h_flat.size})*P({p_flat.size})")
else:
    print("Coordinates are already fully expanded. Skipping Meshgrid.")

# ---------------------------
# 3. Sanitize Data (SKIPPED)
# ---------------------------
# Request: Do not mask anything. Plot all values.
print(f"Plotting all {N_total} points (including negative/unstable values).")

# Calculate global limits for the colorbar (ignoring NaNs so plot doesn't break)
# vmin = -1.0
# vmax =  1.0

vmin = jnp.min(kappa_flat)
vmax = jnp.max(kappa_flat)

# ---------------------------
# 4. Saturation Dome (Context)
# ---------------------------
T_min = 220
T_crit = CP.PropsSI("Tcrit", fluid_name) - 0.1
T_sat_range = np.linspace(T_min, T_crit, 200)
h_liq_sat, h_vap_sat, p_sat_curve = [], [], []

for T in T_sat_range:
    try:
        p = CP.PropsSI("P", "T", T, "Q", 0, fluid_name)
        hl = CP.PropsSI("H", "T", T, "Q", 0, fluid_name)
        hv = CP.PropsSI("H", "T", T, "Q", 1, fluid_name)
        p_sat_curve.append(p)
        h_liq_sat.append(hl)
        h_vap_sat.append(hv)
    except:
        pass

# ---------------------------
# 5. Plotting
# ---------------------------
plt.figure(figsize=(10, 7))

# Saturation Lines
# plt.plot(np.array(h_liq_sat)/1e3, p_sat_curve, 'k--', linewidth=1.5, label="Saturation")
# plt.plot(np.array(h_vap_sat)/1e3, p_sat_curve, 'k--', linewidth=1.5)

# Data Points
# Use SymLogNorm to handle negative values (unstable regions) while keeping log scale logic.
# linthresh=1e-9 means values between -1e-9 and 1e-9 are plotted linearly.
# norm = colors.SymLogNorm(linthresh=1e-9, vmin=vmin, vmax=vmax, base=10)

# sc = plt.scatter(h_flat/1e3, p_flat, 
#                  c=kappa_flat, 
#                  cmap='viridis', 
#                  norm=norm,
#                  s=20, edgecolors='none')

plt.yscale('log')
plt.title(f"P-h Diagram: Isothermal Compressibility ({N_total} points)")
plt.xlabel("Enthalpy [kJ/kg]")
plt.ylabel("Pressure [Pa] (Log Scale)")
# plt.colorbar(sc, label=r"$\kappa_T$ [Pa$^{-1}$]")

# # Spinodal Line
fluid = jxp.Fluid(fluid_name)
spinodal_liquid, _  = props.compute_spinodal_line(fluid)
plt.plot(spinodal_liquid["enthalpy"]*1e-3, spinodal_liquid["pressure"], "-", label="Spinodal")

plt.plot(saturation_table["enthalpy"]["value"]*1e-3, saturation_table["pressure"]["value"], "o")


plt.grid(True, which="both", linestyle='--', alpha=0.4)
plt.legend(loc='upper left')
plt.tight_layout()




fluid.plot_phase_diagram(x_prop="s", y_prop="T", plot_quality_isolines=True, plot_spinodal_line=True)


fluid.plot_phase_diagram(x_prop="h", y_prop="p", plot_quality_isolines=True, plot_spinodal_line=True, y_scale="log")


plt.show()
