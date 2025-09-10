# demo: derivative checks vs CoolProp (CO2)
# - (dh/dT)_p  ≈ cp
# - (du/dT)_ρ  ≈ cv
# - a^2        ≈ (dp/dρ)_s

import jax
import jaxprop.coolprop as jxp

def rel_err(val, ref):
    denom = max(1e-6, abs(ref))
    return (abs(val) - abs(ref)) / denom

# setup state: CO2
# fluid = fluid.Fluid(name="CO2", backend="HEOS")
fluid = jxp.FluidJAX(name="CO2", backend="HEOS")
p0 = 100e5      # Pa
T0 = 400.0      # K
st = fluid.get_props(jxp.PT_INPUTS, p0, T0)
print(st)

rho0 = st["d"]
s0   = st["s"]
cp0  = st["cp"]
cv0  = st["cv"]
a0   = st["a"]
v0   = 1.0 / rho0
a2   = a0**2

print("base state (CO2):")
print(f"  T   : {T0:+0.6e} K")
print(f"  p   : {p0:+0.6e} Pa")
print(f"  rho : {rho0:+0.6e} kg/m^3")
print(f"  s   : {s0:+0.6e} J/(kg K)")
print(f"  cp  : {cp0:+0.6e} J/(kg K)")
print(f"  cv  : {cv0:+0.6e} J/(kg K)")
print(f"  a   : {a0:+0.6e} m/s")


# ---------------- basic heat-capacity / speed-of-sound checks ---------------- #
# (∂h/∂T)_p  via PT
h_of_T_at_p = lambda T: fluid.get_props(jxp.PT_INPUTS, p0, T)["h"]
dhdT_p = jax.jacfwd(h_of_T_at_p)(T0)

# (∂u/∂T)_rho via (rho, T)
u_of_T_at_rho = lambda T: fluid.get_props(jxp.DmassT_INPUTS, rho0, T)["u"]
dudT_rho = jax.jacfwd(u_of_T_at_rho)(T0)

# (∂p/∂rho)_s via (rho, s)
p_of_rho_at_s = lambda rho: fluid.get_props(jxp.DmassSmass_INPUTS, rho, s0)["p"]
dpdrho_s = jax.jacfwd(p_of_rho_at_s)(rho0)

print("\nchecks (value, expected, rel. error):")
print(f"  cp=(dh/dT)|p   : {dhdT_p:+0.6e}  {cp0:+0.6e}  {rel_err(dhdT_p, cp0):+0.6e}   J/(kg K)")
print(f"  cv=(du/dT)|rho : {dudT_rho:+0.6e}  {cv0:+0.6e}  {rel_err(dudT_rho, cv0):+0.6e}   J/(kg K) ")
print(f"  c2=(dp/drho)|s : {dpdrho_s:+0.6e}  {a2:+0.6e}  {rel_err(dpdrho_s, a2):+0.6e}   m^2/s^2")


# -------------------- exact differential identities -------------------- #
# T = (∂u/∂s)|v : hold rho constant (v = 1/rho)
u_of_s_at_rho = lambda s: fluid.get_props(jxp.DmassSmass_INPUTS, rho0, s)["u"]
du_ds_at_v = jax.jacfwd(u_of_s_at_rho)(s0)
T_from_u = fluid.get_props(jxp.DmassSmass_INPUTS, rho0, s0)["T"]

# -p = (∂u/∂v)|s : parameterize with rho at fixed s; use chain rule:
# du/dv|s = du/d(rho)|s * d(rho)/dv = du/d(rho)|s * (-rho^2)
u_of_rho_at_s = lambda rho: fluid.get_props(jxp.DmassSmass_INPUTS, rho, s0)["u"]
du_drho_at_s = jax.jacfwd(u_of_rho_at_s)(rho0)
du_dv_at_s = du_drho_at_s * (-(rho0**2))
p_from_state = fluid.get_props(jxp.DmassSmass_INPUTS, rho0, s0)["p"]

# T = (∂h/∂s)|p : use (p, s)
h_of_s_at_p = lambda s: fluid.get_props(jxp.PSmass_INPUTS, p0, s)["h"]
dh_ds_at_p = jax.jacfwd(h_of_s_at_p)(s0)
T_from_h = fluid.get_props(jxp.PSmass_INPUTS, p0, s0)["T"]

# v = (∂h/∂p)|s : derivative wrt p at const s; compare to 1/rho
h_of_p_at_s = lambda p: fluid.get_props(jxp.PSmass_INPUTS, p, s0)["h"]
dh_dp_at_s = jax.jacfwd(h_of_p_at_s)(p0)
v_from_state = 1.0 / fluid.get_props(jxp.PSmass_INPUTS, p0, s0)["d"]


print("\nchecks (value, expected, rel. error):")
print(f"  T = +(du/ds)|v  : {du_ds_at_v:+0.6e}  {T_from_u:+0.6e}  {rel_err(du_ds_at_v, T_from_u):+0.6e}   K")
print(f"  p = -(du/dv)|s  : {du_dv_at_s:+0.6e}  {-p_from_state:+0.6e}  {rel_err(du_dv_at_s, -p_from_state):+0.6e}   Pa")
print(f"  T = +(dh/ds)|p  : {dh_ds_at_p:+0.6e}  {T_from_h:+0.6e}  {rel_err(dh_ds_at_p, T_from_h):+0.6e}   K")
print(f"  v = +(dh/dp)|s  : {dh_dp_at_s:+0.6e}  {v_from_state:+0.6e}  {rel_err(dh_dp_at_s, v_from_state):+0.6e}   m^3/kg")


# --------------------------- Maxwell relations --------------------------- #
# 1) (∂s/∂p)|T = - (∂v/∂T)|p
s_of_p_at_T = lambda p: fluid.get_props(jxp.PT_INPUTS, p, T0)["s"]
ds_dp_T = jax.jacfwd(s_of_p_at_T)(p0)

v_of_T_at_p = lambda T: 1.0 / fluid.get_props(jxp.PT_INPUTS, p0, T)["d"]
dv_dT_p = jax.jacfwd(v_of_T_at_p)(T0)

print("\nMaxwell 1: (ds/dp)|T = -(dv/dT)|p")
print(f"  (ds/dp)|T       : {ds_dp_T:+0.6e} J/(kg K Pa)")
print(f"  -(dv/dT)|p      : {-dv_dT_p:+0.6e} J/(kg K Pa)")
print(f"  rel. error      : {rel_err(ds_dp_T, -dv_dT_p):+0.6e}")

# 2) (∂s/∂v)|T = (∂p/∂T)|v  -> convert to rho form:
#    (∂s/∂rho)|T = (∂s/∂v)|T * dv/d(rho) = (∂p/∂T)|v * (-1/rho^2)
s_of_rho_at_T = lambda rho: fluid.get_props(jxp.DmassT_INPUTS, rho, T0)["s"]
ds_drho_T = jax.jacfwd(s_of_rho_at_T)(rho0)

p_of_T_at_rho = lambda T: fluid.get_props(jxp.DmassT_INPUTS, rho0, T)["p"]
dp_dT_rho = jax.jacfwd(p_of_T_at_rho)(T0)

rhs = -(1.0 / rho0**2) * dp_dT_rho

print("\nMaxwell 2: (ds/dv)|T = (dp/dT)|v   ->   (ds/drho)|T = -(1/rho^2)(dp/dT)|rho")
print(f"  (ds/drho)|T      : {ds_drho_T:+0.6e} J/(kg K) / (kg/m^3)")
print(f"  RHS              : {rhs:+0.6e} J/(kg K) / (kg/m^3)")
print(f"  rel. error       : {rel_err(ds_drho_T, rhs):+0.6e}")