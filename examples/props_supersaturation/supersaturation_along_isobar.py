import os
import numpy as np
import matplotlib.pyplot as plt

import jaxprop.coolpropx as jxp


# Create the folder to save figures
jxp.set_plot_options(grid=False)
colors = jxp.COLORS_MATLAB
outdir = "output"
os.makedirs(outdir, exist_ok=True)

    

# -------------------------------------------------------------------- #
# Compute degree of supersaturation along isobar
# -------------------------------------------------------------------- #

# Define fluid
fluid = jxp.Fluid("CO2", backend="HEOS")

# Compute spinodal points for a given pressure
p = 50e5
spndl_liq = jxp.compute_spinodal_point_general(
    "p", p, fluid, branch="liquid", supersaturation=True, tolerance=1e-6
)
spndl_vap = jxp.compute_spinodal_point_general(
    "p", p, fluid, branch="vapor", supersaturation=True, tolerance=1e-6
)

# Compute states from subcooled liquid to spinodal point
s1 = fluid.triple_point_liquid.s
s2 = spndl_liq.s
ds = s2 - s1
s_array = np.linspace(s1, s2, 100)
state = fluid.get_state(jxp.PSmass_INPUTS, p, s1)
states_liquid = []
for i, s in enumerate(s_array):
    state = fluid.get_state_metastable(
        prop_1="p",
        prop_1_value=p,
        prop_2="s",
        prop_2_value=s,
        rhoT_guess=[state.rho, state.T],
        supersaturation=True,
        print_convergence=False,
        solver_algorithm="lm",
    )
    states_liquid.append(state)
states_liquid = jxp.states_to_dict(states_liquid)

# Compute states from superheated vapor to spinodal point
s1 = fluid.triple_point_vapor.s
s2 = spndl_vap.s
ds = s2 - s1
s_array = np.linspace(s1, s2, 100)
state = fluid.get_state(jxp.PSmass_INPUTS, p, s1)
states_vapor = []
for i, s in enumerate(s_array):
    state = fluid.get_state_metastable(
        prop_1="p",
        prop_1_value=p,
        prop_2="s",
        prop_2_value=s,
        rhoT_guess=[state.rho, state.T],
        supersaturation=True,
        print_convergence=False,
        solver_algorithm="lm",
    )
    states_vapor.append(state)
states_vapor = jxp.states_to_dict(states_vapor)

# Compute equilibrium process
s1 = fluid.triple_point_liquid.s
s2 = fluid.triple_point_vapor.s
s_array = np.linspace(s1, s2, 200)

states_equilibrium = []
for i, s in enumerate(s_array):
    state = fluid.get_state(jxp.PSmass_INPUTS, p, s)
    states_equilibrium.append(state)
states_equilibrium = jxp.states_to_dict(states_equilibrium)


# -------------------------------------------------------------------- #
# Plot supersaturation degree along isobar
# -------------------------------------------------------------------- #

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'wspace': 0.25})

# Plot phase diagram and thermodynamic points on the first subplot
x = "s"
y = "T"
ax1.set_xlabel("Entropy (J/kg/K)")
ax1.set_ylabel("Temperature (K)")
ax1.set_xlim([fluid.triple_point_liquid[x], fluid.triple_point_vapor[x]])
ax1.set_ylim([fluid.triple_point_liquid[y], 1.1 * fluid.critical_point[y]])
fluid.plot_phase_diagram(
    x_prop=x,
    y_prop=y,
    plot_saturation_line=True,
    plot_spinodal_line=True,
    plot_quality_isolines=True,
    N=50,
    axes=ax1
)

# Plot equilibrium states
ax1.plot(
    states_equilibrium[x],
    states_equilibrium[y],
    color="black",
    linestyle="--",
    label="Equilibrium temperature",
)

# Plot metastable states
ax1.plot(states_liquid[x], states_liquid[y], color=colors[0])
ax1.plot(spndl_liq[x], spndl_liq[y], marker="o", color=colors[0], label="Metastable liquid")
ax1.plot(states_vapor[x], states_vapor[y], color=colors[1])
ax1.plot(spndl_vap[x], spndl_vap[y], marker="o", color=colors[1], label="Metastable vapor")
ax1.legend(loc="upper left")

# Define x and y variables for the second subplot
y1 = "supersaturation_degree"

# Plot the degree of supersaturation on the second subplot
ax2.set_xlabel("Entropy (J/kg/K)")
ax2.set_ylabel("Supersaturation degree (K)")
limits1 = 2 * np.asarray([-1, 1]) * np.maximum(np.abs(spndl_vap[y1]), np.abs(spndl_liq[y1]))
ax2.set_ylim(limits1)
ax2.plot(states_liquid[x], states_liquid[y1], color=colors[0])
ax2.plot(spndl_liq[x], spndl_liq[y1], marker="o", color=colors[0], label="Metastable liquid")
ax2.plot(states_vapor[x], states_vapor[y1], color=colors[1])
ax2.plot(spndl_vap[x], spndl_vap[y1], marker="o", color=colors[1], label="Metastable vapor")
ax2.legend(loc="upper right")
ax2.axhline(y=0, color="black")
jxp.savefig_in_formats(fig, os.path.join(outdir, f"supersaturation_along_isobar_Ts_{fluid.name}"))



# -------------------------------------------------------------------- #
# Plot supersaturation ratio along isobar
# -------------------------------------------------------------------- #

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'wspace': 0.25})

# Plot phase diagram and thermodynamic points on the first subplot
x = "s"
y = "p"
ax1.set_xlabel("Entropy (J/kg/K)")
ax1.set_ylabel("Pressure (Pa)")
ax1.set_xlim([fluid.triple_point_liquid[x], fluid.triple_point_vapor[x]])
ax1.set_ylim([fluid.triple_point_liquid[y], 1.1 * fluid.critical_point[y]])
fluid.plot_phase_diagram(
    x_prop=x,
    y_prop=y,
    plot_saturation_line=True,
    plot_spinodal_line=True,
    plot_quality_isolines=True,
    N=50,
    axes=ax1
)

# Plot equilibrium states
print(spndl_liq[x], )
print(spndl_liq["p_saturation"])
ax1.plot(
    states_liquid[x],
    states_liquid["p_saturation"],
    color="black",
    linestyle="--",
    label="Equilibrium pressure",
)
ax1.plot(
    states_vapor[x],
    states_vapor["p_saturation"],
    color="black",
    linestyle="--",

)

# # Plot metastable states
ax1.plot(states_liquid[x], states_liquid[y], color=colors[0])
ax1.plot(spndl_liq[x], spndl_liq[y], marker="o", color=colors[0], label="Metastable liquid")
ax1.plot(states_vapor[x], states_vapor[y], color=colors[1])
ax1.plot(spndl_vap[x], spndl_vap[y], marker="o", color=colors[1], label="Metastable vapor")
ax1.legend(loc="upper left")

# Define x and y variables for the second subplot
y1 = "supersaturation_ratio"

# Plot the degree of supersaturation on the second subplot
ax2.set_xlabel("Entropy (J/kg/K)")
ax2.set_ylabel("Supersaturation ratio")
ax2.set_ylim([0, 2])
ax2.plot(states_liquid[x], states_liquid[y1], color=colors[0])
ax2.plot(spndl_liq[x], spndl_liq[y1], marker="o", color=colors[0], label="Metastable liquid")
ax2.plot(states_vapor[x], states_vapor[y1], color=colors[1])
ax2.plot(spndl_vap[x], spndl_vap[y1], marker="o", color=colors[1], label="Metastable vapor")
ax2.legend(loc="upper right")
ax2.axhline(y=1, color="black")
jxp.savefig_in_formats(fig, os.path.join(outdir, f"supersaturation_along_isobar_ps_{fluid.name}"))


# Show figures
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()
