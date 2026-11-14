"""
Demo: perfect gas API usage with air

What this shows:
- how to obtain constants (precomputed vs computed)
- how to evaluate states from different input pairs
- how to evaluate transport properties as vectorized functions of temperature
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
import jaxprop as jp

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import jaxprop as jxp
import jaxprop.perfect_gas as pg


def main(argv=None):
    """Run the declared scientific case; quick profiles reduce sampling only."""
    options = example_options(__file__, argv, description=__doc__)
    jxp.set_plot_options()

    # --------------------------- get constants -------------------------- #
    fluid = jxp.Fluid(
        "air", method="perfect_gas", model_options={"T_ref": 298.15, "p_ref": 101_325.0}
    )
    print(f"\nPerfect-gas constants:")
    print(fluid.constants)

    # -------------------------- basic evaluations ----------------------- #
    # Property calculation using (p, T)
    P0 = 101_325.0  # Pa
    T0 = 300.0  # K
    state_PT = fluid.get_state(jxp.PT_INPUTS, P0, T0)
    print(f"\nState from PT (p = {P0:.0f} Pa, T = {T0:.2f} K)")
    print(state_PT)

    # Property calculation using (h, s)
    state_hs = fluid.get_state(jxp.HmassSmass_INPUTS, state_PT.h, state_PT.s)
    print("\nState from HmassSmass (h, s)")
    print(state_hs)

    # Property calculation using (h, p)
    state_hP = fluid.get_state(jxp.HmassP_INPUTS, state_PT.h, state_PT.p)
    print("\nState from HmassP (h, p)")
    print(state_hP)

    # Property calculation using (p, s)
    state_Ps = fluid.get_state(jxp.PSmass_INPUTS, state_PT.p, state_PT.s)
    print("\nState from PSmass (p, s)")
    print(state_Ps)

    # Property calculation using (rho, h)
    state_rhoh = fluid.get_state(jxp.DmassHmass_INPUTS, state_PT.d, state_PT.h)
    print("\nState from DmassHmass (rho, h)")
    print(state_rhoh)

    # --------------------------- state consistency checks --------------------------- #

    def rel_err(a, b):
        denom = max(1e-14, abs(b))
        return abs(a - b) / denom

    # Collect states
    states = {
        "PT": state_PT,
        "hs": state_hs,
        "hP": state_hP,
        "Ps": state_Ps,
        "rhoh": state_rhoh,
    }

    # Choose reference
    ref_state = state_PT
    # Inverse-pair round trips accumulate a few double-precision rounding units.
    tol = 8 * np.finfo(np.float64).eps
    # Unsupported properties are explicit errors, rather than NaN placeholders.
    # Compare every property enabled by this physical model.
    props_to_check = sorted(ref_state.available_properties)

    violations = []

    for name, state in states.items():
        if state.available_properties != ref_state.available_properties:
            raise ValueError(f"{name}: property availability differs from PT")
        for prop in props_to_check:

            # Get values
            val_ref = ref_state[prop]
            val = state[prop]

            if not np.isfinite(val_ref) or not np.isfinite(val):
                violations.append((name, prop, "nonfinite"))
                continue

            # Check error
            err = rel_err(val, val_ref)
            if err > tol:
                violations.append((name, prop, err))

    if violations:
        msg_lines = ["The following state property checks failed:"]
        for name, prop, err in violations:
            if err == "nonfinite":
                msg_lines.append(f"  - {name}: {prop} or its PT reference is nonfinite")
            else:
                msg_lines.append(f"  - {name}: {prop} relative error = {err:.3e}")
        raise ValueError("\n".join(msg_lines))
    else:
        print(
            f"\nAll states are thermodynamically consistent (relative errors < {tol:.1e})."
        )
    options.check_states(locals())
    options.record(
        "execution", "Scientific example completed; see numerical-check records."
    )
    return options.finish()


if __name__ == "__main__":
    main()
