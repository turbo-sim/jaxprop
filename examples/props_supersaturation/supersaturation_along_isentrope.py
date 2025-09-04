import os
import numpy as np
import matplotlib.pyplot as plt

import jaxprop as jxp


# Create the folder to save figures
jxp.set_plot_options(grid=False)
colors = jxp.COLORS_MATLAB
outdir = "output"
os.makedirs(outdir, exist_ok=True)

# Define fluid
fluid = jxp.Fluid("CO2", backend="HEOS")
N_points = 100

# -------------------------------------------------------------------- #
# Compute degree of supersaturation along liquid isentrope
# -------------------------------------------------------------------- #

# Compute subcooled inlet state
p_in, dT_subcooling = 60e5, 5
p_out = 30e5
Q_onset = 0.10
dQ_transition = 0.03
state_in = fluid.get_state(jxp.PQ_INPUTS, p_in, 0)
state_in = fluid.get_state(jxp.PT_INPUTS, p_in, state_in.T - dT_subcooling)

# Define initial guess
rhoT_guess_equilibrium = [state_in.rho, state_in.T]
rhoT_guess_metastable = [state_in.rho, state_in.T]

# Compute equilibrium, metastable and blending states
states_equilibrium, states_metastable, states_blended = [], [], []
p_array = np.linspace(p_in, p_out, N_points)
for p in p_array:
    state_blended, state_equilibrium, state_metastable = fluid.get_state_blending(
        prop_1="p",
        prop_1_value=p,
        prop_2="s",
        prop_2_value=state_in.s,
        rhoT_guess_equilibrium=rhoT_guess_equilibrium,
        rhoT_guess_metastable=rhoT_guess_metastable,
        phase_change="flashing",
        blending_variable="Q",
        blending_onset=Q_onset,
        blending_width=dQ_transition,
        print_convergence=False,
        supersaturation=True,
    )

    states_metastable.append(state_metastable)
    states_blended.append(state_blended)
    states_equilibrium.append(state_equilibrium)

    # Update initial guess
    rhoT_guess_equilibrium = [state_equilibrium.rho, state_equilibrium.T]
    rhoT_guess_metastable = [state_metastable.rho, state_metastable.T]

# Save last state
state_out_eq = states_equilibrium[-1]
state_out_meta = states_metastable[-1]
state_out_blend = states_blended[-1]

# Convert lists to dicts of arrays
states_equilibrium = jxp.states_to_dict(states_equilibrium)
states_metastable = jxp.states_to_dict(states_metastable)
states_blended = jxp.states_to_dict(states_blended)

# Print properties at the inlet and outlet
msgs = [
    f"{'State':>10} {'Temperature (K)':>20} {'Pressure (Pa)':>20} {'Density (kg/m3)':>20} {'Enthalpy (J/kg)':>20} {'Entropy (J/kg/K)':>20} {'Supersaturation (K)':>20} {'Supersaturation ratio':>20}",
    f"{'1':>10} {state_in.T:20.4f} {state_in.p:20.1f} {state_in.rho:20.1f} {state_in.h:20.1f} {state_in.s:20.1f} {np.nan:>20} {np.nan:>20}",
    f"{'2-equi':>10} {state_out_eq.T:20.4f} {state_out_eq.p:20.1f} {state_out_eq.rho:20.1f} {state_out_eq.h:20.1f} {state_out_eq.s:20.1f} {np.nan:>20} {np.nan:>20}",
    f"{'2-blend':>10} {state_out_blend.T:20.4f} {state_out_blend.p:20.1f} {state_out_blend.rho:20.1f} {state_out_blend.h:20.1f} {state_out_blend.s:20.1f} {state_out_blend.supersaturation_degree:>20.03f} {state_out_blend.supersaturation_ratio:>20.03f}",
    f"{'2-meta':>10} {state_out_meta.T:20.4f} {state_out_meta.p:20.1f} {state_out_meta.rho:20.1f} {state_out_meta.h:20.1f} {state_out_meta.s:20.1f} {state_out_meta.supersaturation_degree:>20.03f} {state_out_meta.supersaturation_ratio:>20.03f}",
    "",
]
for msg in msgs:
    print(msg)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.25})

# Plot phase diagram and thermodynamic points on the first subplot
x1 = "s"
y1 = "T"
ax1.set_xlabel("Entropy (J/kg/K))")
ax1.set_ylabel("Temperature (K)")
ax1.set_xlim([1.5 * fluid.triple_point_liquid[x1], 0.9 * fluid.triple_point_vapor[x1]])
ax1.set_ylim([1.2 * fluid.triple_point_liquid[y1], 1.1 * fluid.critical_point[y1]])
fluid.plot_phase_diagram(
    x_prop=x1,
    y_prop=y1,
    plot_saturation_line=True,
    plot_spinodal_line=True,
    plot_quality_isolines=True,
    N=N_points,
    axes=ax1,
)

# Plot equilibrium and metastable states
ax1.plot(
    states_equilibrium[x1][[0, -1]] * 1.005,
    states_equilibrium[y1][[0, -1]],
    color=colors[0],
    linestyle="-",
    marker="o",
    label="Equilibrium properties",
)
ax1.plot(
    states_metastable[x1][[0, -1]],
    states_metastable[y1][[0, -1]],
    color=colors[1],
    linestyle="-",
    marker="o",
    label="Metastable properties",
)
ax1.legend(loc="upper right")

# Define x and y variables for the second subplot
x2 = "p"
y2 = "rho"
y2_bis = "supersaturation_degree"

# Plot the equilibrium and metastable states on the primary y-axis of the second subplot
ax2.set_xlabel("Pressure (Pa)")
ax2.set_ylabel("Density (kg/m$^3$)")
ax2.set_ylim([0, 1.2 * states_equilibrium[y2][0]])
(line1,) = ax2.plot(
    states_equilibrium[x2],
    states_equilibrium[y2],
    color=colors[0],
    linestyle="-",
    label="Equilibrium properties",
)
(line2,) = ax2.plot(
    states_metastable[x2],
    states_metastable[y2],
    color=colors[1],
    linestyle="-",
    label="Metastable properties",
)
(line3,) = ax2.plot(
    states_blended[x2],
    states_blended[y2],
    color="black",
    linestyle="-",
    label="Blended properties",
)

# Create a secondary y-axis for the supersaturation degree
ax2_secondary = ax2.twinx()
ax2_secondary.set_ylabel("Supersaturation degree (K)")
ax2_secondary.set_ylim(1.2 * np.asarray([-1, 1]) * np.max(states_metastable[y2_bis]))
(line4,) = ax2_secondary.plot(
    states_equilibrium[x2],
    states_equilibrium[y2_bis],
    color=colors[0],
    linestyle="--",
)
(line4,) = ax2_secondary.plot(
    states_metastable[x2],
    states_metastable[y2_bis],
    color=colors[1],
    linestyle="--",
)
(line5,) = ax2_secondary.plot(
    states_blended[x2], states_blended[y2_bis], color="black", linestyle="--"
)

# Combine legends from both y-axes
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax2.legend(lines, labels, loc="lower right")

jxp.savefig_in_formats(
    fig, os.path.join(outdir, f"supersaturation_along_liquid_isentrope_{fluid.name}")
)


# -------------------------------------------------------------------- #
# Compute degree of supersaturation along vapor isentrope
# -------------------------------------------------------------------- #

# Compute subcooled inlet state
p_in, dT_superheating = 70e5, 5
p_out = 40e5
Q_onset = 0.90
dQ_transition = 0.01
state_in = fluid.get_state(jxp.PQ_INPUTS, p_in, 0)
state_in = fluid.get_state(jxp.PT_INPUTS, p_in, state_in.T + dT_superheating)

# Define initial guess
rhoT_guess_equilibrium = [state_in.rho, state_in.T]
rhoT_guess_metastable = [state_in.rho, state_in.T]

# Compute equilibrium, metastable and blending states
states_equilibrium, states_metastable, states_blended = [], [], []
p_array = np.linspace(p_in, p_out, N_points)
for p in p_array:

    state_blended, state_equilibrium, state_metastable = fluid.get_state_blending(
        prop_1="p",
        prop_1_value=p,
        prop_2="s",
        prop_2_value=state_in.s,
        rhoT_guess_equilibrium=rhoT_guess_equilibrium,
        rhoT_guess_metastable=rhoT_guess_metastable,
        phase_change="condensation",
        blending_variable="Q",
        blending_onset=Q_onset,
        blending_width=dQ_transition,
        print_convergence=False,
        supersaturation=True,
    )

    states_metastable.append(state_metastable)
    states_blended.append(state_blended)
    states_equilibrium.append(state_equilibrium)

    # Update initial guess
    rhoT_guess_equilibrium = [state_equilibrium.rho, state_equilibrium.T]
    rhoT_guess_metastable = [state_metastable.rho, state_metastable.T]

# Save last state
state_out_eq = states_equilibrium[-1]
state_out_meta = states_metastable[-1]
state_out_blend = states_blended[-1]

# Convert lists to dicts of arrays
states_equilibrium = jxp.states_to_dict(states_equilibrium)
states_metastable = jxp.states_to_dict(states_metastable)
states_blended = jxp.states_to_dict(states_blended)

# Print properties at the inlet and outlet
msgs = [
    f"{'State':>10} {'Temperature (K)':>20} {'Pressure (Pa)':>20} {'Density (kg/m3)':>20} {'Enthalpy (J/kg)':>20} {'Entropy (J/kg/K)':>20} {'Supersaturation (K)':>20} {'Supersaturation ratio':>20}",
    f"{'1':>10} {state_in.T:20.4f} {state_in.p:20.1f} {state_in.rho:20.1f} {state_in.h:20.1f} {state_in.s:20.1f} {np.nan:>20} {np.nan:>20}",
    f"{'2-equi':>10} {state_out_eq.T:20.4f} {state_out_eq.p:20.1f} {state_out_eq.rho:20.1f} {state_out_eq.h:20.1f} {state_out_eq.s:20.1f} {np.nan:>20} {np.nan:>20}",
    f"{'2-blend':>10} {state_out_blend.T:20.4f} {state_out_blend.p:20.1f} {state_out_blend.rho:20.1f} {state_out_blend.h:20.1f} {state_out_blend.s:20.1f} {state_out_blend.supersaturation_degree:>20.03f} {state_out_blend.supersaturation_ratio:>20.03f}",
    f"{'2-meta':>10} {state_out_meta.T:20.4f} {state_out_meta.p:20.1f} {state_out_meta.rho:20.1f} {state_out_meta.h:20.1f} {state_out_meta.s:20.1f} {state_out_meta.supersaturation_degree:>20.03f} {state_out_meta.supersaturation_ratio:>20.03f}",
    "",
]
for msg in msgs:
    print(msg)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.25})

# Plot phase diagram and thermodynamic points on the first subplot
x1 = "s"
y1 = "T"
ax1.set_xlabel("Entropy (J/kg/K))")
ax1.set_ylabel("Temperature (K)")
ax1.set_xlim([1.5 * fluid.triple_point_liquid[x1], 0.9 * fluid.triple_point_vapor[x1]])
ax1.set_ylim([1.2 * fluid.triple_point_liquid[y1], 1.1 * fluid.critical_point[y1]])
fluid.plot_phase_diagram(
    x_prop=x1,
    y_prop=y1,
    plot_saturation_line=True,
    plot_spinodal_line=True,
    plot_quality_isolines=True,
    N=N_points,
    axes=ax1,
)

# Plot equilibrium and metastable states
ax1.plot(
    states_equilibrium[x1][[0, -1]] * 1.005,
    states_equilibrium[y1][[0, -1]],
    color=colors[0],
    linestyle="-",
    marker="o",
    label="Equilibrium properties",
)
ax1.plot(
    states_metastable[x1][[0, -1]],
    states_metastable[y1][[0, -1]],
    color=colors[1],
    linestyle="-",
    marker="o",
    label="Metastable properties",
)
ax1.legend(loc="upper right")

# Define x and y variables for the second subplot
x2 = "p"
y2 = "rho"
y2_bis = "supersaturation_degree"

# Plot the equilibrium and metastable states on the primary y-axis of the second subplot
ax2.set_xlabel("Pressure (Pa)")
ax2.set_ylabel("Density (kg/m$^3$)")
ax2.set_ylim([0, 1.2 * states_equilibrium[y2][0]])
(line1,) = ax2.plot(
    states_equilibrium[x2],
    states_equilibrium[y2],
    color=colors[0],
    linestyle="-",
    label="Equilibrium properties",
)
(line2,) = ax2.plot(
    states_metastable[x2],
    states_metastable[y2],
    color=colors[1],
    linestyle="-",
    label="Metastable properties",
)
(line3,) = ax2.plot(
    states_blended[x2],
    states_blended[y2],
    color="black",
    linestyle="-",
    label="Blended properties",
)

# Create a secondary y-axis for the supersaturation degree
ax2_secondary = ax2.twinx()
ax2_secondary.set_ylabel("Supersaturation degree (K)")
ax2_secondary.set_ylim(1.2 * np.asarray([-1, 1]) * np.max(states_metastable[y2_bis]))
(line4,) = ax2_secondary.plot(
    states_equilibrium[x2],
    states_equilibrium[y2_bis],
    color=colors[0],
    linestyle="--",
)
(line4,) = ax2_secondary.plot(
    states_metastable[x2],
    states_metastable[y2_bis],
    color=colors[1],
    linestyle="--",
)
(line5,) = ax2_secondary.plot(
    states_blended[x2], states_blended[y2_bis], color="black", linestyle="--"
)

# Combine legends from both y-axes
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax2.legend(lines, labels, loc="lower right")

jxp.savefig_in_formats(
    fig, os.path.join(outdir, f"supersaturation_along_vapor_isentrope_{fluid.name}")
)


# Show figures
if not os.environ.get("DISABLE_PLOTS"):
    plt.show()

