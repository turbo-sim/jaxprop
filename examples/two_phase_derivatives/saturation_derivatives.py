
import numpy as np
import jaxprop as jxp
import matplotlib.pyplot as plt

from pysolver_view import approx_derivative, print_dict, set_plot_options


import CoolProp as CP
from jaxprop import get_conductivity, get_viscosity, compute_dsdp_q, GAS_CONSTANT, calculate_subcooling, calculate_superheating, calculate_supersaturation, PROPERTY_ALIAS


# TODO:

# First check if the computation of speed of sound is correct against finite differences
# Then check if the derivatives of entropy are correct against finite difference
# Then check if the derivatives of enthalpy are corrrect against finite difference
# Finally chekc if the formulas for the speed of sound for derivatives of entropy and enthalpy check out

set_plot_options()

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
    e_L = cloned_AS.umass()
    rho_L = cloned_AS.rhomass()
    cp_L = cloned_AS.cpmass()
    cv_L = cloned_AS.cvmass()
    k_L = get_conductivity(cloned_AS)
    mu_L = get_viscosity(cloned_AS)
    a_L = cloned_AS.speed_sound()
    # dsdp_L = cloned_AS.first_saturation_deriv(CP.iSmass, CP.iP)
    # dsdp_L = compute_dsdp_q(cloned_AS, p_mix, quality=0.0)
    dsdp_L = cloned_AS.first_saturation_deriv(CP.iSmass, CP.iP)
    dvdp_L = - cloned_AS.first_saturation_deriv(CP.iDmass, CP.iP) / rho_L ** 2
    dvdT_L = - cloned_AS.first_saturation_deriv(CP.iDmass, CP.iT) / rho_L ** 2
    dedT_L = cloned_AS.first_saturation_deriv(CP.iUmass, CP.iT)



    drhodp_L = cloned_AS.first_saturation_deriv(CP.iDmass, CP.iP)

    eps = p_mix*1e-5
    cloned_AS.update(CP.PQ_INPUTS, p_mix - eps/2, 0.00)
    rho1 = cloned_AS.rhomass()
    cloned_AS.update(CP.PQ_INPUTS, p_mix + eps/2, 0.00)
    rho2 = cloned_AS.rhomass()

    drho_dp_L_FD = (rho2 - rho1) / eps


    cloned_AS.update(CP.PT_INPUTS, p_mix, T_mix-1e-2)
    a_L_bis = cloned_AS.speed_sound()

    print("Check drhodp_L", drhodp_L, drho_dp_L_FD,  1/a_L**2)


    # Saturated vapor properties
    cloned_AS.update(CP.QT_INPUTS, 1.00, T_mix)
    s_V = cloned_AS.smass()
    e_V = cloned_AS.umass()
    rho_V = cloned_AS.rhomass()
    cp_V = cloned_AS.cpmass()
    cv_V = cloned_AS.cvmass()
    k_V = get_conductivity(cloned_AS)
    mu_V = get_viscosity(cloned_AS)
    a_V = cloned_AS.speed_sound()
    # dsdp_V = cloned_AS.first_saturation_deriv(CP.iSmass, CP.iP)
    # dsdp_V = compute_dsdp_q(cloned_AS, p_mix, quality=1.0)

    # dsdp_V = compute_dsdp_q(cloned_AS, p_mix, quality=1.0)
    dsdp_V = cloned_AS.first_saturation_deriv(CP.iSmass, CP.iP)
    dvdp_V = - cloned_AS.first_saturation_deriv(CP.iDmass, CP.iP) / rho_V ** 2
    dvdT_V = - cloned_AS.first_saturation_deriv(CP.iDmass, CP.iT) / rho_V ** 2
    dedT_V = cloned_AS.first_saturation_deriv(CP.iUmass, CP.iT)

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
    dpdT_sat = (s_V - s_L) / (1/rho_V - 1/rho_L)
    dpdT_coolprop = cloned_AS.first_saturation_deriv(CP.iP, CP.iT)
    dTdp_sat = (1/rho_V - 1/rho_L) / (s_V - s_L)
    dTdp_coolprop = cloned_AS.first_saturation_deriv(CP.iT, CP.iP)





    cloned_AS.update(CP.QT_INPUTS, 0.00, T_mix)

    drho_dp_satL = cloned_AS.first_saturation_deriv(CP.iDmass, CP.iP)

    drho_dp_T = cloned_AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
    drho_dT_p = cloned_AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)

    drho_dp_s = cloned_AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iSmass)

    # print(T_mix, drho_dp_satL, drho_dp_T + drho_dT_p*dTdp_sat)

    # print(a_L, np.sqrt(1/drho_dp_s), np.sqrt(1/drho_dp_T))




    # Calculation of the speed of sound:
    temp_L =  mfrac_L *(dvdp_L  - dTdp_sat*dsdp_L)
    temp_V =  mfrac_V *(dvdp_V  - dTdp_sat*dsdp_V)
    speed_sound = np.sqrt(1 / (temp_L + temp_V) / rho_mix)

    eps = 1e-5 *rho_mix
    cloned_AS.update(CP.DmassSmass_INPUTS, rho_mix-eps/2, s_mix)
    p_1 = cloned_AS.p()
    cloned_AS.update(CP.DmassSmass_INPUTS, rho_mix+eps/2, s_mix)
    p_2 = cloned_AS.p()
    a_HEM_FD = np.sqrt((p_2 - p_1) / eps)

    # print("speed_sound HEM", f"{a_HEM:10.6f}", "Finite diff", f"{a_HEM_FD:10.6f}", "new", f"{speed_sound:10.6f}")


    # Calculation of Cv in the two-phase region
    cv_formula = mfrac_L*dedT_L + mfrac_V*dedT_V - (e_V - e_L)/ (1/rho_V - 1/rho_L) * (mfrac_L*dvdT_L +mfrac_V*dvdp_V)
    
    eps = 1e-5 *T_mix
    cloned_AS.update(CP.DmassT_INPUTS, rho_mix, T_mix-eps/2)
    e_1 = cloned_AS.p()
    cloned_AS.update(CP.DmassT_INPUTS, rho_mix, T_mix+eps/2)
    e_2 = cloned_AS.p()
    cv_FD = np.sqrt((e_2 - e_1) / eps)


    cv_wrong = mfrac_L*cv_L + mfrac_V*cv_V #- (e_V - e_L)/ (1/rho_V - 1/rho_L) * (mfrac_L*dvdT_L +mfrac_V*dvdp_V)
    


    print("FD", cv_FD, "cv", cv_formula, "cvWRONG", cv_wrong)



    # Entropy derivatives
    cloned_AS.update(CP.QT_INPUTS, 1.00, T_mix)


    drho_dp_satV = cloned_AS.first_saturation_deriv(CP.iDmass, CP.iP)





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

    # New props:
    props["dpdT"] = dpdT_sat
    props["dTdp"] = dTdp_sat
    props["dpdT_coolprop"] = dpdT_coolprop
    props["dTdp_coolprop"] = dTdp_coolprop

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
        "drho_dp": drho_dp_satL
    }

    props["saturation_vapor"] = {
        "rhomass": rho_V,
        "cpmass": cp_V,
        "cvmass": cv_V,
        "conductivity": k_V,
        "viscosity": mu_V,
        "speed_sound": a_V,
        "dsdp": dsdp_V,
        "drho_dp": drho_dp_satV
    }

    for key, value in PROPERTY_ALIAS.items():
        if value in props["saturation_liquid"]:
            props["saturation_liquid"][key] = props["saturation_liquid"][value]
        if value in props["saturation_vapor"]:
            props["saturation_vapor"][key] = props["saturation_vapor"][value]
            
    return props


Q = 0.5
T = 150+273.15
fluid = CP.AbstractState("HEOS", "water")
# fluid.update(CP.QT_INPUTS, Q, T)
# props = compute_properties_2phase(fluid)



T_list = np.linspace(20, 300, 11) + 273.15
props = []
for T in T_list:
    fluid.update(CP.QT_INPUTS, Q, T)
    props.append(compute_properties_2phase(fluid)) 

props = jxp.states_to_dict(props)





# # Create a figure with two subplots side by side
# fig, ax=  plt.subplots(figsize=(6, 5))
# ax.set_xlabel("Temperature (K)")
# ax.set_ylabel("Derivative dpdT saturation")
# ax.set_yscale("log")
# ax.plot(props["T"], props["dpdT"], "ko")
# ax.plot(props["T"], props["dpdT_coolprop"], "+")
# fig.tight_layout(pad=1)

# # Create a figure with two subplots side by side
# fig, ax=  plt.subplots(figsize=(6, 5))
# ax.set_xlabel("Temperature (K)")
# ax.set_ylabel("Derivative dTdp saturation")
# ax.set_yscale("log")
# ax.plot(props["T"], props["dTdp"], "ko")
# ax.plot(props["T"], props["dTdp_coolprop"], "+")
# fig.tight_layout(pad=1)


# Create a figure with two subplots side by side
fig, ax=  plt.subplots(figsize=(6, 5))
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Density")
ax.set_yscale("log")
ax.plot(props["T"], props["saturation_liquid"]["d"], "bo")
ax.plot(props["T"], props["saturation_vapor"]["d"], "ro")
fig.tight_layout(pad=1)


# Create a figure with two subplots side by side
fig, ax=  plt.subplots(figsize=(6, 5))
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Property")
z = 1/props["saturation_liquid"]["drho_dp"]
ax.plot(props["T"], props["saturation_liquid"]["speed_sound"], "ko")
ax.plot(props["T"], z, "+")
fig.tight_layout(pad=1)



# Show figures
# plt.show()
