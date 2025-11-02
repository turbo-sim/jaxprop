import jaxprop as jxp
import matplotlib.pyplot as plt


# ---------------------------
# Configuration
# ---------------------------
outdir = "fluid_tables"
# fluid_name = "CO2"
# h_min = 200e3  # J/kg
# h_max = 600e3  # J/kg
# p_min = 2e6  # Pa
# p_max = 20e6  # Pa
# N_p = 30  # Grid size for quick test
# N_h = 30

fluid_name = "air"
h_min = 50e3  # J/kg
h_max = 600e3  # J/kg
p_min = 0.6e5    # Pa
p_max = 1.5e5   # Pa
N_h = 32
N_p = 32

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
# Step 2: Interpolate at (h, P)
# ---------------------------
test_h = 100e3   # Test enthalpy [J/kg]
test_P = 1.0e5     # Test pressure [Pa]
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
ax.legend()
plt.show()
