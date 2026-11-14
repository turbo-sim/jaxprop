"""Recover Water's saturation pressure from two homogeneous Gibbs energies.

Reuse one configured fluid for the density/temperature evaluations. The
phase-specific density brackets below belong to the fixed 310 K Water case.
"""

import CoolProp as CP
import numpy as np
import jaxprop as jxp
from scipy.optimize import brentq


def gibbs_difference_rhoT(p_trial, T, fluid):
    # Bracket densities
    rho_liq_bracket = [900, 1100]
    rho_vap_bracket = [1e-6, 0.2]

    # Liquid root
    rho_liq = find_rho_for_p(fluid, T, p_trial, bracket=rho_liq_bracket)

    # Vapor root
    rho_vap = find_rho_for_p(fluid, T, p_trial, bracket=rho_vap_bracket)

    # Compute root
    props_liq = fluid.get_state_metastable(
        "rho", rho_liq, "T", T, supersaturation=False, generalize_quality=False
    )
    props_vap = fluid.get_state_metastable(
        "rho", rho_vap, "T", T, supersaturation=False, generalize_quality=False
    )
    g_liq = props_liq["enthalpy"] - props_liq["temperature"] * props_liq["entropy"]
    g_vap = props_vap["enthalpy"] - props_vap["temperature"] * props_vap["entropy"]

    return g_liq - g_vap


def find_rho_for_p(fluid, T, p_target, bracket):
    """Solve on the selected density branch using complete fluid states."""

    def residual(rho):
        props = fluid.get_state_metastable(
            "rho", rho, "T", T, supersaturation=False, generalize_quality=False
        )
        return props["pressure"] - p_target

    return brentq(residual, *bracket, xtol=1e-12)


# -------------------------------------------
# Main
fluid_name = "Water"
fluid = jxp.Fluid(fluid_name, method="coolprop")
T = 310.0  # K
# Keep an independent source handle for the reference saturation pressure.
AS = CP.AbstractState("HEOS", fluid_name)

AS.update(CP.QT_INPUTS, 1, T)
rho_vap = AS.rhomass()

# Initial guess
AS.update(CP.QT_INPUTS, 0, T)
rho_liq = AS.rhomass()
p_sat = AS.p()
p_min = 0.9 * p_sat
p_max = 1.1 * p_sat

# Root finding on pressure
p_sat_found = brentq(gibbs_difference_rhoT, p_min, p_max, args=(T, fluid), xtol=1e-9)

# Compare with CoolProp
AS.update(CP.QT_INPUTS, 0, T)
p_sat_CP = AS.p()

# Enforce the agreement that the example previously only printed. Keep the
# density/pressure root tolerances and the physical case above unchanged.
np.testing.assert_allclose(p_sat_found, p_sat_CP, rtol=1e-8, atol=1e-7)
assert abs(gibbs_difference_rhoT(p_sat_found, T, fluid)) < 1e-5

print(f"Saturation pressure via rho-T method  : {p_sat_found:.6f} Pa")
print(f"Saturation pressure from CoolProp     : {p_sat_CP:.6f} Pa")
print(
    f"Relative difference                   : {abs(p_sat_found - p_sat_CP)/ p_sat_CP *100:.6e} %"
)
