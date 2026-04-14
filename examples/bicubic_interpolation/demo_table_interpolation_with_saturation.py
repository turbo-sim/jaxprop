import jaxprop as jxp
import matplotlib.pyplot as plt


fluid_name = "nitrogen"
h_min = -120e3  # J/kg
h_max = 0.001e3  # J/kg
p_min = 1e5  # Pa
p_max = 1e6  # Pa
N_p = 20 # Number of pressure points
N_h = 20 # Number of enthalpy points

# fl = jxp.Fluid(fluid_name)
# print(fl.critical_point["pressure"])


# ---------------------------
# Delete existing tables3
# ---------------------------
outdir = "demo_metastable_table_generation"
# if os.path.exists(outdir):
#     shutil.rmtree(outdir, ignore_errors=True)


# # ---------------------------
# # First call: generate table
# # ---------------------------
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
    metastable_phase="liquid",
    gradient_method = "forward",
)


# ---------------------------
# Step 2: Interpolate at (h, P)
# ---------------------------
test_h = 100e3   # Test enthalpy [J/kg]
test_P = 1.1e6     # Test pressure [Pa]
# test_h = 315e3  # Test enthalpy [J/kg]
# test_P = 5e6  # Test pressure [Pa]
props_bicubic = fluid_bicubic.get_state(jxp.HmassP_INPUTS, test_h, test_P)

# -----------------------------------
# Step 3: Compare with CoolProp
# -----------------------------------
fluid_coolprop = jxp.FluidJAX(fluid_name)
props_coolprop = fluid_coolprop.get_state(jxp.HmassP_INPUTS, test_h, test_P)

header = f"{'Property':<35} {'Bicubic':>14} {'CoolProp':>14} {'Error [%]':>14}"
print("-" * len(header))
print("Comparison between interpolated and CoolProp values:")
print("-" * len(header))
print(header)
print("-" * len(header))

for prop in jxp.PROPERTIES_CANONICAL:
    bicubic_val = props_bicubic[prop]
    coolprop_val = props_coolprop[prop]
    if coolprop_val != 0 and not (bicubic_val is None or coolprop_val is None):
        rel_err = (bicubic_val - coolprop_val) / coolprop_val * 100
    else:
        rel_err = float("nan")

    print(f"{prop:<35} {bicubic_val:14.6e} {coolprop_val:14.6e} {rel_err:14.6e}")


# Plot the state
prop_x = "enthalpy"
prop_y = "pressure"
fig, ax = fluid_coolprop.fluid.plot_phase_diagram(
    x_prop=prop_x, y_prop=prop_y, x_scale="linear", y_scale="log"
)
ax.plot(props_bicubic[prop_x], props_bicubic[prop_y], "ko")
ax.plot(props_coolprop[prop_x], props_coolprop[prop_y], "b+")

# add interpolation domain box (simple line version)
ax.plot(
    [h_min, h_max, h_max, h_min, h_min],
    [p_min, p_min, p_max, p_max, p_min],
    "r--",
    linewidth=1.5,
    label="Interpolation domain",
)

sat_bicubic = fluid_bicubic.get_state_saturation(test_P)

sat_coolprop = fluid_coolprop.get_state(jxp.PQ_INPUTS, test_P, 0.0)
print(60*'-')
print(60*'-')
header = f"{'Property':<35} {'Cubic':>14} {'CoolProp':>14} {'Error [%]':>14}"
print("-" * len(header))
print("Comparison between interpolated and CoolProp values for saturation curve:")
print("-" * len(header))
print(header)
print("-" * len(header))

for prop in jxp.PROPERTIES_CANONICAL:
    bicubic_val = sat_bicubic[prop]
    coolprop_val = sat_coolprop[prop]
    if coolprop_val != 0 and not (bicubic_val is None or coolprop_val is None):
        rel_err = (bicubic_val - coolprop_val) / coolprop_val * 100
    else:
        rel_err = float("nan")

    print(f"{prop:<35} {bicubic_val:14.6e} {coolprop_val:14.6e} {rel_err:14.6e}")


# Plot the state
prop_x = "enthalpy"
prop_y = "pressure"
fig, ax = fluid_coolprop.fluid.plot_phase_diagram(
    x_prop=prop_x, y_prop=prop_y, x_scale="linear", y_scale="log"
)
ax.plot(props_bicubic[prop_x], props_bicubic[prop_y], "ko")
ax.plot(props_coolprop[prop_x], props_coolprop[prop_y], "b+")

# add interpolation domain box (simple line version)
ax.plot(
    [h_min, h_max, h_max, h_min, h_min],
    [p_min, p_min, p_max, p_max, p_min],
    "r--",
    linewidth=1.5,
    label="Interpolation domain",
)

ax.legend()
plt.show()
