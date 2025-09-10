
import numpy as np
import jaxprop as jxp


from pysolver_view import approx_derivative


import CoolProp as CP
from jaxprop import get_conductivity, get_viscosity, compute_dsdp_q, GAS_CONSTANT, calculate_subcooling, calculate_superheating, calculate_supersaturation, PROPERTY_ALIAS


# TODO:

# First check if the computation of speed of sound is correct against finite differences
# Then check if the derivatives of entropy are correct against finite difference
# Then check if the derivatives of enthalpy are corrrect against finite difference
# Finally chekc if the formulas for the speed of sound for derivatives of entropy and enthalpy check out

def compute_properties_2phase(abstract_state, supersaturation=False):
    """Compute two-phase fluid properties from CoolProp abstract state

    Get two-phase properties from mixing rules and single-phase CoolProp properties

    Homogeneous equilibrium model

    State formulas for T=T, p=p, mfrac/vfrac(rho), h-s-g-u-cp-cv, mu-k, a

    """

    # Instantiate new AbstractState to compute saturation properties without changing the state of the class
    AS = abstract_state
    fluids = AS.fluid_names()
    if len(fluids) != 1:
        raise ValueError(f"Expected one fluid, got {fluids}")
    cloned_AS = CP.AbstractState(AS.backend_name(), fluids[0])

    # Basic properties of the two-phase mixture
    T_mix = AS.T()
    p_mix = AS.p()
    rho_mix = AS.rhomass()
    u_mix = AS.umass()
    h_mix = AS.hmass()
    s_mix = AS.smass()
    surface_tension = AS.surface_tension()
    # gibbs_mix = AS.gibbsmass()

    # Saturated liquid properties
    cloned_AS.update(CP.QT_INPUTS, 0.00, T_mix)
    s_L = cloned_AS.smass()
    rho_L = cloned_AS.rhomass()
    cp_L = cloned_AS.cpmass()
    cv_L = cloned_AS.cvmass()
    k_L = get_conductivity(cloned_AS)
    mu_L = get_viscosity(cloned_AS)
    a_L = cloned_AS.speed_sound()
    # dsdp_L = cloned_AS.first_saturation_deriv(CP.iSmass, CP.iP)
    dsdp_L = compute_dsdp_q(cloned_AS, p_mix, quality=0.0)

    # Saturated vapor properties
    cloned_AS.update(CP.QT_INPUTS, 1.00, T_mix)
    s_V = cloned_AS.smass()
    rho_V = cloned_AS.rhomass()
    cp_V = cloned_AS.cpmass()
    cv_V = cloned_AS.cvmass()
    k_V = get_conductivity(cloned_AS)
    mu_V = get_viscosity(cloned_AS)
    a_V = cloned_AS.speed_sound()
    # dsdp_V = cloned_AS.first_saturation_deriv(CP.iSmass, CP.iP)
    dsdp_V = compute_dsdp_q(cloned_AS, p_mix, quality=1.0)

    # Volume fractions of vapor and liquid
    vfrac_V = (rho_mix - rho_L) / (rho_V - rho_L)
    vfrac_L = 1.00 - vfrac_V

    # Mass fractions of vapor and liquid
    mfrac_V = (1 / rho_mix - 1 / rho_L) / (1 / rho_V - 1 / rho_L)
    mfrac_L = 1.00 - mfrac_V

    # Heat capacities of the two-phase mixture
    cp_mix = mfrac_L * cp_L + mfrac_V * cp_V
    cv_mix = mfrac_L * cv_L + mfrac_V * cv_V

    # Transport properties of the two-phase mixture
    k_mix = vfrac_L * k_L + vfrac_V * k_V
    mu_mix = vfrac_L * mu_L + vfrac_V * mu_V

    # Compressibility factor of the two-phase mixture
    M = AS.molar_mass()
    Z_mix = p_mix / (rho_mix * (GAS_CONSTANT / M) * T_mix)

    # Speed of sound of the two-phase mixture
    B1 = vfrac_L / (rho_L * a_L**2) + vfrac_V / (rho_V * a_V**2)
    B2 = vfrac_L * rho_L / cp_L * dsdp_L**2 + vfrac_V * rho_V / cp_V * dsdp_V**2
    compressibility_HEM = B1 + T_mix * B2
    if mfrac_V < 1e-6:  # Avoid discontinuity when Q_v=0
        a_HEM = a_L
    elif mfrac_V > 1.0 - 1e-6:  # Avoid discontinuity when Q_v=1
        a_HEM = a_V
    else:
        a_HEM = (1 / rho_mix / compressibility_HEM) ** 0.5


    # Get saturation line slope from Clapeyron equation
    dpdT = (s_V - s_L) / (1/rho_V - 1/rho_L)

    # Entropy derivatives

    # dp/dT at saturation

    # dpdT_sat = cloned_AS.first_partial_deriv(CP.iP, CP.iT, CP.iQ)



    # # What is going on at the saturation line?

    # # Saturated liquid
    # cloned_AS = CP.AbstractState(AS.backend_name(), fluids[0])
    # cloned_AS.update(CP.QT_INPUTS, 0.0, T_mix) 

    # drhodp_sigma = AS.first_saturation_deriv(CP.iDmass, CP.iP)

    # drhodp_T = cloned_AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
    # drhodT_p = cloned_AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)

    # print("drhodp_T", drhodp_T)
    # print("drhodp_p", drhodT_p)
    # drhodp_custom = drhodp_T + drhodT_p/dpdT
    # print("drhodp_custom", drhodp_custom)
    # print("drhodp_sigma", drhodp_sigma)
    # print("speed_sound liquid  a_L", a_L)
    # print("speed_sound liquid  1/drhodp_sigma", 1/drhodp_sigma)
    # print("speed_sound liquid  1/drhodp_custom", 1/drhodp_custom)


    # # Vapor
    # cloned_AS = CP.AbstractState(AS.backend_name(), fluids[0])
    # cloned_AS.update(CP.QT_INPUTS, 1.0, T_mix) 

    # drhodp_sigma = AS.first_saturation_deriv(CP.iDmass, CP.iP)

    # drhodp_T = cloned_AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
    # drhodT_p = cloned_AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)

    # print("drhodp_T", drhodp_T)
    # print("drhodp_p", drhodT_p)
    # drhodp_custom = drhodp_T + drhodT_p/dpdT
    # print("drhodp_custom", drhodp_custom)
    # print("drhodp_sigma", drhodp_sigma)
    # print("speed_sound vapor  a_V", a_V)
    # print("speed_sound vapor  1/drhodp_sigma", 1/drhodp_sigma)
    # print("speed_sound vapor  1/drhodp_custom", 1/drhodp_custom)





    # cloned_AS.update(CP.QT_INPUTS, 0.00001, T_mix)  # Any quality in 2-phase
    # correction = cloned_AS.first_partial_deriv(CP.iSmass, CP.iP, CP.iT)

    # print(correction)
    # print("dsdp_L", dsdp_L, cp_L/T_mix/dpdT, dsdp_L-cp_L/T_mix/dpdT-correction)

    # cloned_AS.update(CP.QT_INPUTS, 1, T_mix)  # Any quality in 2-phase
    # correction = cloned_AS.first_partial_deriv(CP.iSmass, CP.iP, CP.iT)

    # print("dsdp_V", dsdp_V, cp_V/T_mix/dpdT+correction, dsdp_V-cp_V/T_mix/dpdT-correction)






    # # Entropy derivative wrt density at constant pressure
    # dsdrho_p = -dpdT_sat / rho_mix**2

    # # Entropy derivative wrt pressure at constant density
    # term1 = (vfrac_L / (rho_L * a_L**2) + vfrac_V / (rho_V * a_V**2)) * dpdT_sat
    # term2 = (vfrac_L * rho_L * cp_L + vfrac_V * rho_V * cp_V) / dpdT_sat
    # dsdp_rho = (term1 + term2) / rho_mix

    # Store properties in dictionary
    props = {}
    props["T"] = T_mix
    props["p"] = p_mix
    props["rhomass"] = rho_mix
    props["umass"] = u_mix
    props["hmass"] = h_mix
    props["smass"] = s_mix
    # props["gibbsmass"] = gibbs_mix
    props["cvmass"] = cv_mix
    props["cpmass"] = cp_mix
    props["gamma"] = props["cpmass"] / props["cvmass"]
    props["compressibility_factor"] = Z_mix
    props["speed_sound"] = a_HEM
    props["isentropic_bulk_modulus"] = rho_mix * a_HEM**2
    props["isentropic_compressibility"] = (rho_mix * a_HEM**2) ** -1
    props["isothermal_bulk_modulus"] = np.nan
    props["isothermal_compressibility"] = np.nan
    props["isobaric_expansion_coefficient"] = np.nan
    props["viscosity"] = mu_mix
    props["conductivity"] = k_mix
    props["Q"] = mfrac_V
    props["quality_mass"] = mfrac_V
    props["quality_volume"] = vfrac_V
    props["surface_tension"] = surface_tension

    if supersaturation:
        props["subcooling"] = calculate_subcooling(AS)
        props["superheating"] = calculate_superheating(AS)
        props = calculate_supersaturation(AS, props)

    # Add properties as aliases
    for key, value in PROPERTY_ALIAS.items():
        props[key] = props[value]

    # Add saturation properties as subdictionaries
    props["saturation_liquid"] = {
        "rhomass": rho_L,
        "cpmass": cp_L,
        "cvmass": cv_L,
        "conductivity": k_L,
        "viscosity": mu_L,
        "speed_sound": a_L,
        "dsdp": dsdp_L,
    }

    props["saturation_vapor"] = {
        "rhomass": rho_V,
        "cpmass": cp_V,
        "cvmass": cv_V,
        "conductivity": k_V,
        "viscosity": mu_V,
        "speed_sound": a_V,
        "dsdp": dsdp_V,
    }

    for key, value in PROPERTY_ALIAS.items():
        if value in props["saturation_liquid"]:
            props["saturation_liquid"][key] = props["saturation_liquid"][value]
        if value in props["saturation_vapor"]:
            props["saturation_vapor"][key] = props["saturation_vapor"][value]
            
    return props






def get_enthalpy_derivatives_numerical(fluid, pressure, density, eps=None):

    def get_h(x):
        p, d = x
        return fluid.get_state(jxp.DmassP_INPUTS, d, p).h
    
    x0 = np.asarray([pressure, density])
    grad_h = approx_derivative(get_h, x0, method='2-point', rel_step=eps)

    return grad_h



def get_enthalpy_derivatives_analytic(fluid, pressure, density):

    state = fluid.get_state(jxp.DmassP_INPUTS, density, pressure)
    temperature = state.T
    density = state.d

    eps = 1e-6*temperature
    p1 = fluid.get_state(jxp.QT_INPUTS, 0.0, temperature-eps).p
    p2 = fluid.get_state(jxp.QT_INPUTS, 0.0, temperature+eps).p
    dpdT = (p2 - p1) / (2*eps)

    grad_h = -temperature*dpdT/density**2

    return grad_h   


fluid = jxp.Fluid("water", backend="HEOS")


Q = 0.7
T = 150+273.15

state = fluid.get_state(jxp.QT_INPUTS, Q, T)

print(state)


grad_h = get_enthalpy_derivatives_numerical(fluid, state.p, state.rho)

print(grad_h)

grad_h = get_enthalpy_derivatives_analytic(fluid, state.p, state.rho)

print(grad_h)



