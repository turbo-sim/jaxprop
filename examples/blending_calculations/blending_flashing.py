from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
import jaxprop as jp

import os
import numpy as np
import matplotlib.pyplot as plt
import jaxprop.coolprop as jxp
from jaxprop import Fluid


def main(argv=None):
    """Run the declared scientific case; quick profiles reduce sampling only."""
    options = example_options(__file__, argv, description=__doc__)

    # Plot options and colors
    jxp.set_plot_options(grid=False)
    colors = jxp.COLORS_MATLAB
    fig_dir = options.output_dir
    os.makedirs(fig_dir, exist_ok=True)

    # Define fluid
    fluid = Fluid(name="nitrogen", backend="HEOS")

    # Create figure and labels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.0, 4.0))
    ax1.set_xlabel("Entropy (J/kg/K)")
    ax1.set_ylabel("Temperature (K)")
    ax2.set_xlabel("Pressure (Pa)")
    ax2.set_ylabel("Density (kg/m$^3$)")
    ax3.set_xlabel("Pressure (Pa)")
    ax3.set_ylabel("Generalized quality (-)")
    prop_x1, prop_y1 = "entropy", "temperature"
    prop_x2, prop_y2 = "pressure", "density"
    prop_x3, prop_y3 = "pressure", "quality_mass"
    axes = [ax1, ax2, ax3]
    prop_pairs = [
        (prop_x1, prop_y1),
        (prop_x2, prop_y2),
        (prop_x3, prop_y3),
    ]

    # Plot phase diagram
    jp.plot_phase_diagram(
        fluid,
        prop_x1,
        prop_y1,
        axes=ax1,
        plot_critical_point=True,
        plot_quality_isolines=True,
        plot_pseudocritical_line=False,
        plot_spinodal_line=False,
        N=options.points(100),
    )

    # Define inlet state
    p_in = 0.6 * fluid.critical_point.p
    dT_subcooling = 10
    props_in = fluid.get_state(jxp.PQ_INPUTS, p_in, 0)
    props_in = fluid.get_state(
        jxp.PT_INPUTS, p_in, props_in["temperature"] - dT_subcooling
    )

    # Determine phase change direction
    if props_in["entropy"] > fluid.critical_point.s:
        phase_change = "condensation"
        Q_onset = 0.99
        Q_width = 0.01
    else:
        phase_change = "evaporation"
        Q_onset = 0.01
        Q_width = 0.01

    # Initial guesses
    rhoT_guess_metastable = [props_in["density"], props_in["temperature"]]
    rhoT_guess_equilibrium = [props_in["density"], props_in["temperature"]]

    # Keep each physical model in its own collection. A constitutively blended
    # state has a different model/reference identity from either source-EOS branch.
    blended_states, equilibrium_states, metastable_states = [], [], []

    # Loop over pressures
    p_values = np.linspace(p_in, p_in / 5, options.points(100))
    for i, p_out in enumerate(p_values):
        # Get properties
        props_blend, props_eq, props_meta = fluid.get_state_blending(
            prop_1="p",
            prop_1_value=p_out,
            prop_2="s",
            prop_2_value=props_in["entropy"],
            rhoT_guess_equilibrium=rhoT_guess_equilibrium,
            rhoT_guess_metastable=rhoT_guess_metastable,
            phase_change=phase_change,
            blending_variable="Q",
            blending_onset=Q_onset,
            blending_width=Q_width,
            generalize_quality=True,
            supersaturation=True,
            print_convergence=False,
            solver_algorithm="lm",
        )

        # Update guesses
        rhoT_guess_metastable = [props_meta["density"], props_meta["temperature"]]
        rhoT_guess_equilibrium = [props_eq["density"], props_eq["temperature"]]
        blended_states.append(props_blend)
        equilibrium_states.append(props_eq)
        metastable_states.append(props_meta)

        # Plot with labels only in first iteration
        for ax, (x, y) in zip(axes, prop_pairs):
            label_eq = "Equilibrium" if i == 0 else None
            label_meta = "Metastable" if i == 0 else None
            label_blend = "Blended" if i == 0 else None
            ax.plot(
                props_eq[x], props_eq[y], marker="o", color=colors[0], label=label_eq
            )
            ax.plot(
                props_meta[x],
                props_meta[y],
                marker="o",
                color=colors[1],
                label=label_meta,
            )
            ax.plot(
                props_blend[x],
                props_blend[y],
                marker="o",
                color=colors[3],
                label=label_blend,
            )

    blended, equilibrium, metastable = (
        jxp.FluidState.stack(states)
        for states in (blended_states, equilibrium_states, metastable_states)
    )
    assert blended.model == "blended_real_fluid"
    assert blended.reference_id != equilibrium.reference_id == metastable.reference_id
    core_properties = (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    )
    for states in (blended, equilibrium, metastable):
        assert np.all(states.valid)
        for name in core_properties:
            assert np.all(np.isfinite(states.raw_property(name))), name
        np.testing.assert_allclose(states.p, p_values, rtol=1e-5)
        np.testing.assert_allclose(states.s, props_in["entropy"], rtol=1e-5)
    # A smooth convex transition must not overshoot either endpoint state.
    # It need not satisfy identities derived from a single EOS potential.
    for name in core_properties:
        value = blended.raw_property(name)
        first, second = equilibrium.raw_property(name), metastable.raw_property(name)
        margin = 2e-12 * np.maximum(np.maximum(np.abs(first), np.abs(second)), 1.0)
        assert np.all(value >= np.minimum(first, second) - margin)
        assert np.all(value <= np.maximum(first, second) + margin)

    # Add legend to one axis only
    ax3.legend(loc="upper right")
    plt.tight_layout(pad=1)

    # Show figures
    options.check_states(locals())
    options.record(
        "execution", "Scientific example completed; see numerical-check records."
    )
    return options.finish()


if __name__ == "__main__":
    main()
