"""
This demo illustrates how to do thermodynamic state calculations using both
CoolProp flash calculations and with the barotropy.fluid.get_state_metastable()
method using different types of input pairs.

The computed properties are illustrated in a T-s diagram and also printed to show
that the results of the different methods are consistent.
"""

import os
import matplotlib.pyplot as plt

import coolpropx as cp
cp.set_plot_options(grid=False)

# Create fluid
fluid = cp.Fluid(name="CO2", exceptions=True)

# Create figure
fig, ax = plt.subplots(figsize=(6.0, 5.0))
ax.set_xlabel("Entropy (J/kg/K)")
ax.set_ylabel("Temperature (K)")
prop_x = "s"
prop_y = "T"

# Plot phase diagram
fluid.plot_phase_diagram(
    prop_x,
    prop_y,
    axes=ax,
    plot_critical_point=True,
    plot_quality_isolines=False,
    plot_pseudocritical_line=False,
)

# Compute fluid state with a CoolProp function call
p = fluid.critical_point.p * 0.6
T = fluid.critical_point.T * 1.05
state = fluid.get_state(cp.PT_INPUTS, p, T)
ax.plot(state[prop_x], state[prop_y], "o", markersize=12, label="CoolProp call")


# Compute state with a (rho,T) call to the HEOS (no need for initial guess)
state_rhoT = fluid.get_state_metastable(
    prop_1="rho",
    prop_1_value=state.rho,
    prop_2="T",
    prop_2_value=state.T,
)
ax.plot(state_rhoT[prop_x], state_rhoT[prop_y], "o", markersize=8, label="HEOS rho-T call")


# Compute state with a (p,T) call to the HEOS
rho_guess = state.rho * 1.05
T_guess = state.T - 2
rhoT_guess = [rho_guess, T_guess]
state_pT = fluid.get_state_metastable(
    prop_1="p",
    prop_1_value=state.p,
    prop_2="T",
    prop_2_value=state.T,
    rhoT_guess=rhoT_guess,
    print_convergence=False,
)
ax.plot(state_pT[prop_x], state_pT[prop_y], "o", markersize=4, label="HEOS p-T call")

# Compute state with a (h,s) call to the HEOS
rho_guess = state.rho * 1.05
T_guess = state.T - 2
state_hs = fluid.get_state_metastable(
    prop_1="h",
    prop_1_value=state.h,
    prop_2="s",
    prop_2_value=state.s,
    rhoT_guess=rhoT_guess,
    print_convergence=False,
)
ax.plot(state_hs[prop_x], state_hs[prop_y], "+", label="HEOS h-s call")
fig.tight_layout(pad=1)

# Compare different function calls
print(f"{'Property':>15} {'CoolProp':>15} {'rho-T call':>15} {'p-T call':>15} {'h-s call':>15}")
props = ["p", "T", "rho", "h", "s", "cp", "speed_sound"]
for prop in props:
    print(f"{prop:>15} {state[prop]:15.04e} {state_rhoT[prop]:15.04e} {state_pT[prop]:15.04e} {state_hs[prop]:15.04e}")

# Show figures
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()
