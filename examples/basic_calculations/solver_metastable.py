"""
This demo illustrates how to do thermodynamic state calculations using both
CoolProp flash calculations and with the configured JaxProp get_state_metastable()
method using different types of input pairs.

The computed properties are illustrated in a T-s diagram and also printed to show
that the results of the different methods are consistent.
"""

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
    jxp.set_plot_options(grid=False)

    # Create fluid
    fluid = Fluid(name="CO2")

    # Create figure
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.set_xlabel("Entropy (J/kg/K)")
    ax.set_ylabel("Temperature (K)")
    prop_x = "s"
    prop_y = "T"

    # Plot phase diagram
    jp.plot_phase_diagram(
        fluid,
        prop_x,
        prop_y,
        axes=ax,
        plot_critical_point=True,
        plot_quality_isolines=False,
        plot_pseudocritical_line=False,
        N=options.points(100),
    )

    # Compute fluid state with a CoolProp function call
    p = fluid.critical_point.p * 0.6
    T = fluid.critical_point.T * 1.05
    state = fluid.get_state(jxp.PT_INPUTS, p, T)
    ax.plot(state[prop_x], state[prop_y], "o", markersize=12, label="CoolProp call")

    # Compute state with a (rho,T) call to the HEOS (no need for initial guess)
    state_rhoT = fluid.get_state_metastable(
        prop_1="rho",
        prop_1_value=state.rho,
        prop_2="T",
        prop_2_value=state.T,
    )
    ax.plot(
        state_rhoT[prop_x],
        state_rhoT[prop_y],
        "o",
        markersize=8,
        label="HEOS rho-T call",
    )

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
    ax.plot(
        state_pT[prop_x], state_pT[prop_y], "o", markersize=4, label="HEOS p-T call"
    )

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
    ax.legend()
    fig.tight_layout(pad=1)

    # Compare different function calls
    print(
        f"{'Property':>15} {'CoolProp':>15} {'rho-T call':>15} {'p-T call':>15} {'h-s call':>15}"
    )
    props = ["p", "T", "rho", "h", "s", "cp", "speed_sound"]
    # The chosen reference is a single-phase state, so these caloric/acoustic
    # responses are defined. A pure-fluid two-phase interior does not have a finite
    # equilibrium cp and must be handled through its declared property availability.
    # Verify complete core recovery and these defined responses at the original
    # default solver settings, instead of treating a printed table as acceptance.
    checked_properties = (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
        "isobaric_heat_capacity",
        "speed_of_sound",
    )
    for recovered in (state_rhoT, state_pT, state_hs):
        assert recovered.valid
        for name in checked_properties:
            np.testing.assert_allclose(
                recovered.raw_property(name),
                state.raw_property(name),
                rtol=1e-5,
                atol=1e-7,
            )
    for prop in props:
        print(
            f"{prop:>15} {state[prop]:15.04e} {state_rhoT[prop]:15.04e} {state_pT[prop]:15.04e} {state_hs[prop]:15.04e}"
        )

    # Show figures
    options.check_states(locals())
    options.record(
        "execution", "Scientific example completed; see numerical-check records."
    )
    return options.finish()


if __name__ == "__main__":
    main()
