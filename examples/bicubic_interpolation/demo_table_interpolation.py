import jaxprop as jxp
import matplotlib.pyplot as plt




# ---------------------------
# Configuration
# ---------------------------
outdir = "fluid_tables"
fluid_name = "CO2"
h_min = 200e3  # J/kg
h_max = 600e3  # J/kg
p_min = 2e6    # Pa
p_max = 20e6   # Pa
N_p = 80       # Grid size for quick test
N_h = 80

fluid = jxp.FluidBicubic(
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
# Step 2: Interpolate at (h, P)
# ---------------------------
test_h = 500e3   # Test enthalpy [J/kg]
test_P = 12e6     # Test pressure [Pa]

interp_props = fluid.get_props(jxp.HmassP_INPUTS, test_h, test_P)
fluid = jxp.Fluid(fluid_name)
coolprop_props = fluid.get_state(jxp.HmassP_INPUTS, test_h, test_P)






# -----------------------------------
# Step 3: Compare with CoolProp
# -----------------------------------
header = f"{'Property':<35} {'Bicubic':>14} {'CoolProp':>14} {'Error [%]':>14}"
print("-" * len(header))
print("Comparison between interpolated and CoolProp values:")
print("-" * len(header))

print(header)
print("-" * len(header))

for prop in jxp.PROPERTIES_CANONICAL:
    bicubic_val = interp_props[prop]
    coolprop_val = coolprop_props[prop]
    if coolprop_val != 0 and not (bicubic_val is None or coolprop_val is None):
        rel_err = (bicubic_val - coolprop_val) / coolprop_val * 100
    else:
        rel_err = float("nan")

    print(f"{prop:<35} {bicubic_val:14.6e} {coolprop_val:14.6e} {rel_err:14.6f}")


# Plot the state
prop_x = "enthalpy"
prop_y = "pressure"
fig, ax = fluid.plot_phase_diagram(x_prop=prop_x, y_prop=prop_y, x_scale="linear", y_scale="log")
ax.plot(interp_props[prop_x], interp_props[prop_y], "ko")
ax.plot(coolprop_props[prop_x], coolprop_props[prop_y], "b+")

# add interpolation domain box (simple line version)
ax.plot(
    [h_min, h_max, h_max, h_min, h_min],
    [p_min, p_min, p_max, p_max, p_min],
    "r--",
    linewidth=1.5,
    label="Interpolation domain"
)
ax.legend()
plt.show()


#TODO, why is p not exaclty recovered?
# Comparison between interpolated and CoolProp values:
# Property                                   Bicubic       CoolProp      Error [%]
# --------------------------------------------------------------------------------
# pressure                              1.201237e+07   1.200000e+07       0.103112
# temperature                           3.806271e+02   3.805691e+02       0.015249
# density                               2.288502e+02   2.286297e+02       0.096424
# enthalpy                              5.000000e+05   5.000000e+05      -0.000000
# entropy                               1.893554e+03   1.893695e+03      -0.007476
# internal_energy                       4.475099e+05   4.475134e+05      -0.000779



# Time the cost of precomputing coefficients for different number of points
# 1. Save only prop tables
# 2. Save prop and coefficient tables
# See which cost dociminates for 10, 100, 1000 function evaluations


# Avoid converting to int and use pure jax?


# 2. Coefficient allocation inefficiency

# You allocate a full (Nh, Np, 16) coefficient array but only fill one cell:

# coeffs = jnp.zeros((Nh, Np, 16), dtype=jnp.float64)
# coeffs = coeffs.at[i, j, :].set(coeffs_local)


# That’s a lot of memory churn, especially for large grids.