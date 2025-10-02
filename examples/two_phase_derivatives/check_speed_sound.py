import CoolProp.CoolProp as CP
import numpy as np

fluid = "Water"
AS = CP.AbstractState("HEOS", fluid)

# --- saturation state at given p and Q
p = 1e5   # Pa
Q = 0.5
AS.update(CP.PQ_INPUTS, p, Q)

rho = AS.rhomass()
s = AS.smass()
T = AS.T()
print(f"State: p={p:.3e} Pa, Q={Q:.2f}, T={T:.2f} K, rho={rho:.3f} kg/m3, s={s:.3f} J/kg-K")

# --- finite difference function
def speed_of_sound_fd(AS, rel_drho=1e-5):
    rho0 = AS.rhomass()
    s0 = AS.smass()
    p0 = AS.p()
    drho = rel_drho * abs(rho0)

    # central difference at constant entropy
    AS.update(CP.DmassSmass_INPUTS, rho0 + drho, s0)
    p_plus = AS.p()
    AS.update(CP.DmassSmass_INPUTS, rho0 - drho, s0)
    p_minus = AS.p()
    dpdrho_s = (p_plus - p_minus) / (2 * drho)

    # restore original state
    AS.update(CP.DmassSmass_INPUTS, rho0, s0)

    return dpdrho_s  # this is a^2

# --- 1. finite difference
a2_fd = speed_of_sound_fd(AS)
a_fd = np.sqrt(a2_fd) if a2_fd > 0 else np.nan

# --- 2. analytic two-phase derivative
try:
    dpdrho_two_phase = AS.first_two_phase_deriv(CP.iP, CP.iDmass, CP.iSmass)
    a2_two_phase = dpdrho_two_phase
    a_two_phase = np.sqrt(a2_two_phase) if a2_two_phase > 0 else np.nan
except Exception as e:
    a2_two_phase = np.nan
    a_two_phase = np.nan
    print("first_two_phase_deriv not available:", e)

# --- 3. analytic partial derivative (single-phase only)
try:
    dpdrho_s = AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iSmass)
    a2_partial = dpdrho_s
    a_partial = np.sqrt(a2_partial) if a2_partial > 0 else np.nan
except Exception as e:
    a2_partial = np.nan
    a_partial = np.nan
    print("first_partial_deriv not available:", e)


# --- results
print(f"Speed of sound FD:        {a_fd:.6e} m/s  (a^2={a2_fd:.6e})")
print(f"Speed of sound two-phase: {a_two_phase:.6e} m/s  (a^2={a2_two_phase:.6e})")
print(f"Speed of sound partial:   {a_partial:.6e} m/s  (a^2={a2_partial:.6e})")



import jaxprop as jxp
fluid = jxp.Fluid(name=fluid, backend="HEOS")

state = fluid.get_state(jxp.PQ_INPUTS, p, Q)
print("Speed of sound HEM", state.a)