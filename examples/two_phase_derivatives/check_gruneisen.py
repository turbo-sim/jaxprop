"""Check Gamma=(dp/du)|rho/rho at Water's original 1 bar, Q=0.5 state.

The source's single-phase partial-derivative API is not an equilibrium
two-phase Jacobian. Compare the configured JaxProp state with an independent
signed equilibrium finite difference and the saturation-coordinate identity.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options

import CoolProp.CoolProp as CP
import numpy as np
import jaxprop as jxp
from examples.two_phase_derivatives.saturation_derivatives import (
    finite_difference_responses,
    saturation_reference,
)


def gruneisen_fd(source, rel_du=1e-5):
    """Signed central equilibrium derivative, restoring the original PQ state."""
    return finite_difference_responses(source, source.p(), source.Q(), rel_du)[
        "gruneisen"
    ]


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    pressure, quality = 1e5, 0.5
    fluid = jxp.Fluid("Water", method="coolprop")
    state = fluid.get_state(jxp.PQ_INPUTS, pressure, quality)
    source = CP.AbstractState("HEOS", "Water")
    source.update(CP.PQ_INPUTS, pressure, quality)
    before = np.array([source.p(), source.Q(), source.rhomass(), source.umass()])
    finite_difference = gruneisen_fd(source)
    after = np.array([source.p(), source.Q(), source.rhomass(), source.umass()])
    np.testing.assert_allclose(after, before, rtol=2e-12, atol=1e-9)
    analytical = saturation_reference(source, float(state.T), quality)["gruneisen"]
    assert state.valid and np.isfinite(finite_difference)
    np.testing.assert_allclose(
        state.gruneisen, finite_difference, rtol=2e-5, atol=1e-10
    )
    np.testing.assert_allclose(state.gruneisen, analytical, rtol=2e-6, atol=1e-10)
    print(
        f"State: p={pressure:.3e} Pa, Q={quality:.2f}, T={state.T:.2f} K, rho={state.rho:.3f} kg/m³"
    )
    print(
        f"Gruneisen: JaxProp={state.gruneisen:.10e}, signed FD={finite_difference:.10e}, saturation Jacobian={analytical:.10e}"
    )
    options.record(
        "scientific_checks",
        {
            "pressure_Pa": pressure,
            "quality": quality,
            "library": float(state.gruneisen),
            "finite_difference": finite_difference,
            "analytical": analytical,
            "source_state_restored": True,
        },
    )
    options.finish()
    return float(state.gruneisen), finite_difference, analytical


if __name__ == "__main__":
    main()
