"""
This demo illustrates how to do calculate the generalized degrees of superheating and subcooling
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
import jaxprop as jp

import CoolProp as cp
import jaxprop.coolprop as jxp
from jaxprop import Fluid
import numpy as np


def main(argv=None):
    """Run the declared scientific case; quick profiles reduce sampling only."""
    options = example_options(__file__, argv, description=__doc__)
    # Create high-level Fluid object
    fluid = Fluid(name="water", backend="HEOS")

    # --- Superheating: low-level (only once) ---
    AS_superheat = cp.AbstractState("HEOS", "water")
    AS_superheat.update(cp.PT_INPUTS, 101325, 120 + 273.15)
    superheat_low = jxp.calculate_superheating(AS_superheat)
    print(f"[low-level] Superheating: {superheat_low:+0.3f} K")

    # --- Superheating: high-level ---
    state = fluid.get_state(cp.PT_INPUTS, 101325, 120 + 273.15, supersaturation=True)
    np.testing.assert_allclose(
        state.superheating, superheat_low, rtol=1e-12, atol=1e-10
    )
    print(f"[high-level] Superheating (PQ): {state.superheating:+0.3f} K")

    # --- Superheating: high-level ---
    state = fluid.get_state(cp.PQ_INPUTS, 101325, 0.95, supersaturation=True)
    print(f"[high-level] Superheating (PQ): {state.superheating:+0.3f} K")

    # --- Subcooling: high-level ---
    state_2 = fluid.get_state(cp.PT_INPUTS, 101325, 25 + 273.15, supersaturation=True)
    print(f"[high-level] Subcooling (PT): {state_2.subcooling:+0.3f} K")

    # --- Subcooling: high-level ---
    state_3 = fluid.get_state(cp.PQ_INPUTS, 101325, 0.05, supersaturation=True)
    assert state_2.subcooling > 0
    # Interior two-phase diagnostics are generalized signed extensions, not
    # literal temperature offsets. Preserve their negative values explicitly.
    assert state.superheating < 0 and state_3.subcooling < 0
    AS_superheat.update(cp.PQ_INPUTS, 101325, 0.0)
    saturation_temperature = AS_superheat.T()
    np.testing.assert_allclose(
        superheat_low, 393.15 - saturation_temperature, atol=1e-10
    )
    np.testing.assert_allclose(
        state_2.subcooling, saturation_temperature - 298.15, atol=1e-10
    )
    AS_superheat.update(cp.PQ_INPUTS, 101325, 0.95)
    np.testing.assert_allclose(
        state.superheating, jxp.calculate_superheating(AS_superheat), atol=1e-10
    )
    AS_superheat.update(cp.PQ_INPUTS, 101325, 0.05)
    np.testing.assert_allclose(
        state_3.subcooling, jxp.calculate_subcooling(AS_superheat), atol=1e-10
    )
    print(f"[high-level] Subcooling (PQ): {state_3.subcooling:+0.3f} K")
    options.check_states(locals())
    options.record(
        "execution", "Scientific example completed; see numerical-check records."
    )
    return options.finish()


if __name__ == "__main__":
    main()
