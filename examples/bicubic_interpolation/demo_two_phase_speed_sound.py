import jaxprop as jxp
import jax.numpy as jnp
import matplotlib.pyplot as plt

jxp.set_plot_options()


# ---------------------------
# Configuration
# ---------------------------
outdir = "fluid_tables"
fluid_name = "CO2"
h_min = 100e3  # J/kg
h_max = 600e3  # J/kg
p_min = 1e6    # Pa
p_max = 20e6   # Pa
N_p = 100       # Grid size for quick test
N_h = 100

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
test_h = 500e3   # Test enthalpy [J/kg]
test_p = 2e6     # Test pressure [Pa]

test_h = jnp.linspace(h_min, h_max, 3000)

props_bicubic = fluid_bicubic.get_state(jxp.HmassP_INPUTS, test_h, test_p)


# -----------------------------------
# Step 3: Compare with CoolProp
# -----------------------------------
fluid_coolprop = jxp.FluidJAX(fluid_name)
props_coolprop = fluid_coolprop.get_state(jxp.HmassP_INPUTS, test_h, test_p)


# Plot the state
prop_x = "enthalpy"
prop_y = "pressure"
fig, ax = fluid_coolprop.fluid.plot_phase_diagram(x_prop=prop_x, y_prop=prop_y, x_scale="linear", y_scale="log")
ax.plot(props_bicubic[prop_x], props_bicubic[prop_y], "k+")

# add interpolation domain box (simple line version)
ax.plot(
    [h_min, h_max, h_max, h_min, h_min],
    [p_min, p_min, p_max, p_max, p_min],
    "r--",
    linewidth=1.5,
    label="Interpolation domain"
)
ax.legend()
fig.tight_layout(pad=1)


fig, ax = plt.subplots()
ax.set_xlabel("Enthalpy")
ax.set_ylabel("Speed of sound")
ax.plot(props_coolprop["h"], props_coolprop["a"], "b+", label="coolprop")
ax.plot(props_bicubic["h"], props_bicubic["a"], color="k", label="bicubic")
ax.legend()
fig.tight_layout(pad=1)

plt.show()


