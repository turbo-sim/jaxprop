import numpy as np
import jax
import jaxprop as jxp

# baseline (PT)
T_ref=300.0
p_ref=101325.0
fluid = jxp.FluidJAX(name="nitrogen", backend="HEOS")
ref = fluid.get_state(jxp.PT_INPUTS, p_ref, T_ref)

# input pair HmassP
input_pair = jxp.HmassP_INPUTS
h0, p0 = ref["h"], ref["p"]

# primals
st = fluid.get_state(input_pair, h0, p0)
print("state (HmassP_INPUTS):")
for k in ("T", "p", "d", "h", "s", "a", "mu", "k", "gamma"):
    print(f"  {k:>5s}: {float(np.asarray(st[k])):+0.6f}")

# scalar extractors
def T_of(h, p): return fluid.get_state(input_pair, h, p)["T"]
def rho_of(h, p): return fluid.get_state(input_pair, h, p)["d"]

# forward-mode
dT_dh = jax.jacfwd(T_of, argnums=0)(h0, p0)
dT_dp = jax.jacfwd(T_of, argnums=1)(h0, p0)
dd_dh = jax.jacfwd(rho_of, argnums=0)(h0, p0)
dd_dp = jax.jacfwd(rho_of, argnums=1)(h0, p0)
print("\npartials (forward-mode):")
print(f"  dT/dh|p   : {float(dT_dh):+0.6e} K*kg/J")
print(f"  dT/dp|h   : {float(dT_dp):+0.6e} K/Pa")
print(f"  dd/dh|p   : {float(dd_dh):+0.6e} kg*m^-3*kg/J")
print(f"  dd/dp|h   : {float(dd_dp):+0.6e} kg*m^-3/Pa")

# reverse-mode
dT_dh_r = jax.jacrev(T_of, argnums=0)(h0, p0)
dT_dp_r = jax.jacrev(T_of, argnums=1)(h0, p0)
dd_dh = jax.jacrev(rho_of, argnums=0)(h0, p0)
dd_dp = jax.jacrev(rho_of, argnums=1)(h0, p0)
print("\npartials (reverse-mode):")
print(f"  dT/dh|p   : {float(dT_dh_r):+0.6e} K*kg/J")
print(f"  dT/dp|h   : {float(dT_dp_r):+0.6e} K/Pa")
print(f"  dd/dh|p   : {float(dd_dh):+0.6e} kg*m^-3*kg/J")
print(f"  dd/dp|h   : {float(dd_dp):+0.6e} kg*m^-3/Pa")


import numpy as np
from scipy.optimize._numdiff import approx_derivative

# finite difference with relative step 1e-6
rel_step = 1e-6

# T_of derivatives
dT_dh_fd = approx_derivative(lambda h: T_of(h, p0),
                             x0=np.atleast_1d(h0),
                             rel_step=rel_step,
                             method='2-point')[0]

dT_dp_fd = approx_derivative(lambda p: T_of(h0, p),
                             x0=np.atleast_1d(p0),
                             rel_step=rel_step,
                             method='2-point')[0]

# rho_of derivatives
dd_dh_fd = approx_derivative(lambda h: rho_of(h, p0),
                             x0=np.atleast_1d(h0),
                             rel_step=rel_step,
                             method='2-point')[0]

dd_dp_fd = approx_derivative(lambda p: rho_of(h0, p),
                             x0=np.atleast_1d(p0),
                             rel_step=rel_step,
                             method='2-point')[0]

print("\nfinite-difference (SciPy, rel_step=1e-6):")
print(f"  dT/dh|p   : {dT_dh_fd:+0.6e} K*kg/J")
print(f"  dT/dp|h   : {dT_dp_fd:+0.6e} K/Pa")
print(f"  dd/dh|p   : {dd_dh_fd:+0.6e} kg*m^-3*kg/J")
print(f"  dd/dp|h   : {dd_dp_fd:+0.6e} kg*m^-3/Pa")