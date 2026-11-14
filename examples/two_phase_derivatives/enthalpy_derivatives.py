"""Check signed enthalpy derivatives of a Water two-phase equilibrium.

At fixed saturation pressure, changing density changes vapor mass fraction;
dh/drho|p=-T*(dp_sat/dT)/rho**2. At fixed density, h=u+p/rho gives
dh/dp|rho=cv/(dp_sat/dT)+1/rho. Units are J m³/kg² and m³/kg respectively.
This keeps the original Q=0.7, T=423.15 K case.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options

import CoolProp.CoolProp as CP
import numpy as np
import jaxprop as jxp
from examples.two_phase_derivatives.saturation_derivatives import saturation_reference


def get_enthalpy_derivatives_numerical(fluid, pressure, density, relative_step=1e-5):
    """Central differences in (pressure,density) order, with each other fixed."""
    dp, drho = relative_step * pressure, relative_step * density
    hp = fluid.get_state(jxp.DmassP_INPUTS, density, pressure + dp).h
    hm = fluid.get_state(jxp.DmassP_INPUTS, density, pressure - dp).h
    hrp = fluid.get_state(jxp.DmassP_INPUTS, density + drho, pressure).h
    hrm = fluid.get_state(jxp.DmassP_INPUTS, density - drho, pressure).h
    return np.asarray([(hp - hm) / (2 * dp), (hrp - hrm) / (2 * drho)])


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    fluid = jxp.Fluid("Water", method="coolprop")
    source = CP.AbstractState("HEOS", "Water")
    quality, temperature = 0.7, 423.15
    state = fluid.get_state(jxp.QT_INPUTS, quality, temperature)
    reference = saturation_reference(source, temperature, quality)
    numerical = get_enthalpy_derivatives_numerical(
        fluid, float(state.p), float(state.rho)
    )
    analytical = np.array(
        [
            reference["enthalpy_pressure_derivative"],
            reference["enthalpy_density_derivative"],
        ]
    )
    assert state.valid and np.all(np.isfinite(numerical))
    assert analytical[0] > 0 and analytical[1] < 0
    np.testing.assert_allclose(numerical, analytical, rtol=2e-5, atol=1e-10)
    print(
        f"dh/dp|rho [m³/kg]: numerical={numerical[0]:.10e}, analytical={analytical[0]:.10e}"
    )
    print(
        f"dh/drho|p [J m³/kg²]: numerical={numerical[1]:.10e}, analytical={analytical[1]:.10e}"
    )
    options.record(
        "scientific_checks",
        {
            "quality": quality,
            "temperature_K": temperature,
            "numerical": numerical.tolist(),
            "analytical": analytical.tolist(),
            "relative_errors": np.abs(numerical / analytical - 1).tolist(),
        },
    )
    options.finish()
    return numerical, analytical


if __name__ == "__main__":
    main()
