"""
Recover one equilibrium state from different property pairs with the custom
density/temperature flash solver, then verify the complete core state.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
import jaxprop as jp

import numpy as np
import jaxprop.coolprop as jxp
from jaxprop import Fluid


def main(argv=None):
    """Run the declared scientific case; quick profiles reduce sampling only."""
    options = example_options(__file__, argv, description=__doc__)
    # State calculation using coolprop
    fluid = Fluid(name="Air", backend="HEOS")
    state = fluid.get_state(jxp.PT_INPUTS, 101325, 300)

    # Define an initial guess
    T_guess = 25 + state.T
    rho_guess = 1.1 * state.rho
    rhoT_guess = [rho_guess, T_guess]

    # p-T function call
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

    # Check recovery, rather than relying only on a solver termination message.
    # This independent comparison budget is consistent with the original default
    # root tolerance (1e-6); no solver settings or input state are changed.
    core_properties = (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    )
    for recovered in (properties_pT, properties_hp, properties_hs):
        assert recovered.valid
        for name in core_properties:
            np.testing.assert_allclose(
                recovered.raw_property(name),
                state.raw_property(name),
                rtol=1e-5,
                atol=1e-7,
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
    options.check_states(locals())
    options.record(
        "execution", "Scientific example completed; see numerical-check records."
    )
    return options.finish()


if __name__ == "__main__":
    main()
