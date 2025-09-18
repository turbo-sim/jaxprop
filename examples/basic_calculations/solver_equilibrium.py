"""
This script shows how to do basic function calculations used the custom Flash solver
"""

import jaxprop.coolprop as jxp

# State calculation using coolprop
fluid = jxp.Fluid(name="Air", backend="HEOS")
state = fluid.get_state(jxp.PT_INPUTS, 101325, 300)

# Define an initial guess
T_guess = 25 + state.T
rho_guess = 1.1 * state.rho
rhoT_guess = [rho_guess, T_guess]

# h-s function call
properties_pT = fluid.get_state_equilibrium(
    prop_1="p",
    prop_1_value=state.p,
    prop_2="T",
    prop_2_value=state.T,
    rhoT_guess=rhoT_guess,
    print_convergence=True,
)

# h-p function call
properties_hp = fluid.get_state_equilibrium(
    prop_1="h",
    prop_1_value=state.h,
    prop_2="p",
    prop_2_value=state.p,
    rhoT_guess=rhoT_guess,
    print_convergence=True,
)

# h-s function call
properties_hs = fluid.get_state_equilibrium(
    prop_1="h",
    prop_1_value=state.h,
    prop_2="s",
    prop_2_value=state.s,
    rhoT_guess=rhoT_guess,
    print_convergence=True,
)

# Print properties
print(
    f"Air density is {state.rho:0.4f} kg/m3 at p={state.p:0.4f} Pa and T={state.T:0.4f} K"
)
print(
    f"Air density is {properties_pT['rho']:0.4f} kg/m3 at p={properties_pT['p']:0.4f} Pa and T={properties_pT['T']:0.4f} K"
)
print(
    f"Air density is {properties_hp['rho']:0.4f} kg/m3 at p={properties_hp['p']:0.4f} Pa and T={properties_hp['T']:0.4f} K"
)
print(
    f"Air density is {properties_hs['rho']:0.4f} kg/m3 at p={properties_hs['p']:0.4f} Pa and T={properties_hs['T']:0.4f} K"
)
