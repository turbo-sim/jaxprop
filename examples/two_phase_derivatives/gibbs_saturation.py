
import CoolProp as CP
from coolpropx import compute_properties_metastable_rhoT
from scipy.optimize import brentq

def gibbs_difference_rhoT(p_trial, T, fluid):
    # Bracket densities
    rho_liq_bracket = [900, 1100]
    rho_vap_bracket = [1e-6, 0.2]

    # Liquid root
    rho_liq = find_rho_for_p(fluid, T, p_trial, bracket=rho_liq_bracket)

    # Vapor root
    rho_vap = find_rho_for_p(fluid, T, p_trial, bracket=rho_vap_bracket)

    # Compute root
    props_liq = compute_properties_metastable_rhoT(fluid, rho_liq, T)
    props_vap = compute_properties_metastable_rhoT(fluid, rho_vap, T)
    g_liq = props_liq["hmass"] - props_liq["T"]* props_liq["smass"]
    g_vap = props_vap["hmass"] - props_vap["T"]* props_vap["smass"]
    
    return g_liq - g_vap


def find_rho_for_p(AS, T, p_target, bracket):
    def residual(rho):
        props = compute_properties_metastable_rhoT(AS, rho, T)
        return props["p"] - p_target

    return brentq(residual, *bracket, xtol=1e-12)


# -------------------------------------------
# Main
fluid = "Water"
T = 310.0  # K
AS = CP.AbstractState("HEOS", fluid)

AS.update(CP.QT_INPUTS, 1, T)
rho_vap = AS.rhomass()

# Initial guess
AS.update(CP.QT_INPUTS, 0, T)
rho_liq = AS.rhomass()
p_sat = AS.p()
p_min = 0.9 * p_sat
p_max = 1.1 * p_sat

# Root finding on pressure
p_sat_found = brentq(gibbs_difference_rhoT, p_min, p_max, args=(T, AS), xtol=1e-9)

# Compare with CoolProp
AS.update(CP.QT_INPUTS, 0, T)
p_sat_CP = AS.p()

print(f"Saturation pressure via rho-T method  : {p_sat_found:.6f} Pa")
print(f"Saturation pressure from CoolProp     : {p_sat_CP:.6f} Pa")
print(f"Relative difference                   : {abs(p_sat_found - p_sat_CP)/ p_sat_CP *100:.6e} %")