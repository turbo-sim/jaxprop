"""
Demo: perfect gas API usage with air

What this shows:
- how to obtain constants (precomputed vs computed)
- how to evaluate states from different input pairs
- how to evaluate transport properties as vectorized functions of temperature
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import jaxprop as jxp
import jaxprop.perfect_gas as pg

jxp.set_plot_options()


# --------------------------- get constants -------------------------- #
fluid = jxp.FluidPerfectGas("air", 298.15, 101_325.0)
print(f"\nPerfect-gas constants:")
print(fluid.constants)


# -------------------------- basic evaluations ----------------------- #
# Property calculation using (p, T)
P0 = 101_325.0  # Pa
T0 = 300.0      # K
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
import numpy as np

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
tol = 1e-16
props_to_check = jxp.PROPERTIES_CANONICAL

violations = []

for name, state in states.items():
    for prop in props_to_check:

        # Get values
        val_ref = ref_state[prop]
        val = state[prop]

        # Skip if reference value is NaN → nothing to check
        if np.isnan(val_ref):
            continue

        # Check error
        err = rel_err(val, val_ref)
        if err > tol:
            violations.append((name, prop, err))

if violations:
    msg_lines = ["The following state property checks failed:"]
    for name, prop, err in violations:
        if err == "NaN":
            msg_lines.append(f"  - {name}: {prop} is NaN but reference is not")
        else:
            msg_lines.append(f"  - {name}: {prop} relative error = {err:.3e}")
    raise ValueError("\n".join(msg_lines))
else:
    print(f"\nAll states are thermodynamically consistent (relative errors < {tol:.1e}).")
