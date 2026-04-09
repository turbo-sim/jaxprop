import os
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt

jxp.set_plot_options()

# Compute viscosity over the temperature range
fluid = jxp.FluidPerfectGas("air", 298.15, 101_325.0)
T_range = jnp.linspace(300.0, 1000.0, 200)
mu_vals = fluid.get_state(jxp.PT_INPUTS, 101325, T_range).viscosity

# Create figure
fig, ax = plt.subplots(figsize=(6, 5))
ax.grid(False)
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("Dynamic viscosity [Pa·s]")
ax.set_xlim([T_range.min(), T_range.max()])

# Plot viscosity
ax.plot(T_range, mu_vals)
plt.tight_layout(pad=1)

# Show plot if not disabled
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()
