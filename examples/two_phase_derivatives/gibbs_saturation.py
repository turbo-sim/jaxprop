"""Recover Water's saturation pressure from two homogeneous Gibbs energies.

Reuse one configured fluid for the density/temperature evaluations. The
phase-specific density brackets below belong to the fixed 310 K Water case.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options

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


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    fluid_name = "Water"
    fluid = jxp.Fluid(fluid_name, method="coolprop")
    temperature = 310.0  # K; keep the original physical case and root tolerances.
    source = CP.AbstractState("HEOS", fluid_name)
    source.update(CP.QT_INPUTS, 0, temperature)
    pressure = source.p()
    found = brentq(
        gibbs_difference_rhoT,
        0.9 * pressure,
        1.1 * pressure,
        args=(temperature, fluid),
        xtol=1e-9,
    )
    np.testing.assert_allclose(found, pressure, rtol=1e-8, atol=1e-7)
    gibbs_residual = abs(gibbs_difference_rhoT(found, temperature, fluid))
    assert gibbs_residual < 1e-5
    print(f"Saturation pressure via rho-T method  : {found:.6f} Pa")
    print(f"Saturation pressure from CoolProp     : {pressure:.6f} Pa")
    print(
        f"Relative difference                   : {abs(found-pressure)/pressure*100:.6e} %"
    )
    options.record(
        "scientific_checks",
        {
            "temperature_K": temperature,
            "source_pressure_Pa": pressure,
            "recovered_pressure_Pa": found,
            "relative_pressure_error": abs(found / pressure - 1),
            "gibbs_residual_J_per_kg": gibbs_residual,
        },
    )
    options.finish()
    return found, pressure


if __name__ == "__main__":
    main()
