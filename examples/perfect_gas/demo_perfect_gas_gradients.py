import jax.numpy as jnp
import jaxprop as jxp
import jaxprop.perfect_gas as pg

# config
input_pair = jxp.HmassP_INPUTS
h = 3.00e5    # J/kg
p = 101325.0  # Pa


# base state
fluid = jxp.FluidPerfectGas("air", T_ref=298.15, p_ref=101325.0)
base = fluid.get_state(input_pair, h, p)
print("base state:")
print(f"  T   : {float(base['T']):+0.6f} K")
print(f"  p   : {float(base['p']):+0.6f} Pa")
print(f"  rho : {float(base['d']):+0.6f} kg/m^3")
print(f"  h   : {float(base['h']):+0.6f} J/kg")
print(f"  s   : {float(base['s']):+0.6f} J/(kg K)")

# finite differences jacobian
grad_fd = jxp.perfect_gas.get_props_gradient(fluid, input_pair, h, p, method="fd")

# jax jacobian (will crash if jax is not installed)
grad_jax = jxp.perfect_gas.get_props_gradient(fluid, input_pair, h, p, method="jax")


print("\npartials (finite differences):")
print(f"  dT/dh|p   : {float(grad_fd['T'][0]):+0.10e} K*kg/J")
print(f"  dT/dp|h   : {float(grad_fd['T'][1]):+0.10e} K/Pa")
print(f"  drho/dh|p : {float(grad_fd['d'][0]):+0.10e} kg*m^-3*kg/J")
print(f"  drho/dp|h : {float(grad_fd['d'][1]):+0.10e} kg*m^-3/Pa")

print("\npartials (jax):")
print(f"  dT/dh|p   : {float(grad_jax['T'][0]):+0.10e} K*kg/J")
print(f"  dT/dp|h   : {float(grad_jax['T'][1]):+0.10e} K/Pa")
print(f"  drho/dh|p : {float(grad_jax['d'][0]):+0.10e} kg*m^-3*kg/J")
print(f"  drho/dp|h : {float(grad_jax['d'][1]):+0.10e} kg*m^-3/Pa")

# relative errors
def rel(a, b):
    a = float(a); b = float(b)
    return abs(a - b) / max(1.0, abs(b))

print("\nrelative errors (fd vs jax):")
print(f"  dT/dh|p   : {rel(grad_fd['T'][0], grad_jax['T'][0]):+0.10e}")
print(f"  dT/dp|h   : {rel(grad_fd['T'][1], grad_jax['T'][1]):+0.10e}")
print(f"  drho/dh|p : {rel(grad_fd['d'][0], grad_jax['d'][0]):+0.10e}")
print(f"  drho/dp|h : {rel(grad_fd['d'][1], grad_jax['d'][1]):+0.10e}")

# identity checks using FD results
cp, _ = jxp.perfect_gas.specific_heat(fluid.constants)
R = fluid.constants.R
T = base["T"]

print("\nidentity checks (value, expected, rel. error):")
val_dT_dh = grad_jax['T'][0]
exp_dT_dh = 1.0 / cp
err_dT_dh = jnp.abs(val_dT_dh - exp_dT_dh) / jnp.abs(exp_dT_dh)
print(f"  dT/dh|p   : {val_dT_dh:+0.6f}   {exp_dT_dh:+0.6f}   {err_dT_dh:+0.6e}")

val_drho_dp = grad_jax["rho"][1]
exp_drho_dp = 1.0 / (R * T)
err_drho_dp = jnp.abs(val_drho_dp - exp_drho_dp) / jnp.abs(exp_drho_dp)
print(f"  drho/dp|h : {val_drho_dp:+0.6f}   {exp_drho_dp:+0.6f}   {err_drho_dp:+0.6e}")