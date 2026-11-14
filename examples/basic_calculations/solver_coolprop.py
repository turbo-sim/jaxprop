"""
This demo illustrates how to do basic thermodynamic state calculations using
the configured JaxProp fluid.get_state() method using different types of input pairs.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
import jaxprop as jp

import jaxprop.coolprop as jxp
from jaxprop import Fluid
import numpy as np


def main(argv=None):
    """Run the declared scientific case; quick profiles reduce sampling only."""
    options = example_options(__file__, argv, description=__doc__)
    # State calculation
    water = Fluid(name="Water", method="coolprop", backend="HEOS")
    air = Fluid(name="Air", method="coolprop", backend="HEOS")
    state = water.get_state(jxp.PT_INPUTS, 101325, 300)
    assert 990 < float(state.rho) < 1005
    print(
        f"Water density is {state.rho:0.2f} kg/m3 at p={state.p:0.2f} Pa and T={state.T:0.2f} K"
    )

    # State calculation
    state = air.get_state(jxp.PT_INPUTS, 101325, 300)
    assert 1.35 < float(state.gamma) < 1.45
    print(
        f"Air heat capacity ratio is {state.gamma:0.2f} at p={state.p:0.2f} Pa and T={state.T:0.2f} K"
    )

    # Properties of liquid water
    props_stable = water.get_state(jxp.PT_INPUTS, 101325, 300)
    print("\nProperties of liquid water")
    print(props_stable)

    # Properties of water/steam mixture
    props = water.get_state(jxp.QT_INPUTS, 0.5, 300)
    liquid = water.get_state(jxp.QT_INPUTS, 0.0, 300)
    vapor = water.get_state(jxp.QT_INPUTS, 1.0, 300)
    np.testing.assert_allclose(
        1 / props.rho, 0.5 / liquid.rho + 0.5 / vapor.rho, rtol=1e-12
    )
    np.testing.assert_allclose(props.h, 0.5 * (liquid.h + vapor.h), rtol=1e-12)
    print("\nProperties of water/steam mixture")
    print(props)
    options.check_states(locals())
    options.record(
        "execution", "Scientific example completed; see numerical-check records."
    )
    return options.finish()


if __name__ == "__main__":
    main()
