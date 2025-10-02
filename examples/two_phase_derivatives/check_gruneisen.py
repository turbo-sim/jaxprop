import CoolProp.CoolProp as CP
import numpy as np

fluid = "Water"
AS = CP.AbstractState("HEOS", fluid)

# --- saturation state at given p and Q
p = 1e5   # Pa
Q = 0.5
AS.update(CP.PQ_INPUTS, p, Q)

rho = AS.rhomass()
u = AS.umass()
T = AS.T()
print(f"State: p={p:.3e} Pa, Q={Q:.2f}, T={T:.2f} K, rho={rho:.3f} kg/m3, u={u:.3f} J/kg")

# --- finite difference function
def gruneisen_fd(AS, rel_du=1e-5):
    rho = AS.rhomass()
    u0 = AS.umass()
    p0 = AS.p()
    du = rel_du * abs(u0)

    # central difference
    AS.update(CP.DmassUmass_INPUTS, rho, u0 + du)
    p_plus = AS.p()
    AS.update(CP.DmassUmass_INPUTS, rho, u0 - du)
    p_minus = AS.p()
    dpdu = (p_plus - p_minus) / (2 * du)

    # restore original state
    AS.update(CP.DmassUmass_INPUTS, rho, u0)

    return (1.0 / rho) * dpdu

# --- 1. finite difference
Gamma_fd = gruneisen_fd(AS)

# --- 2. analytic two-phase derivative
try:
    dpdu_two_phase = AS.first_two_phase_deriv(CP.iP, CP.iUmass, CP.iDmass)
    Gamma_two_phase = dpdu_two_phase / rho
except Exception as e:
    Gamma_two_phase = np.nan
    print("first_two_phase_deriv not available:", e)

# --- 3. analytic via partial derivatives (using dp/dT|rho over cv)
try:
    dpdT_rho = AS.first_partial_deriv(CP.iP, CP.iT, CP.iDmass)
    cv = AS.cvmass()
    dpdu_partial = dpdT_rho / cv
    Gamma_partial = (1.0 / rho) * dpdu_partial
except Exception as e:
    Gamma_partial = np.nan
    print("first_partial_deriv not available:", e)

# --- results
print(f"Gruneisen FD:        {Gamma_fd:.6e}")
print(f"Gruneisen two-phase: {Gamma_two_phase:.6e}")
print(f"Gruneisen partial:   {Gamma_partial:.6e}")
