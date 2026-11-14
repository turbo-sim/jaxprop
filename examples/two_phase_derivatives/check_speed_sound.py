"""Check equilibrium sound speed at Water's original 1 bar, Q=0.5 state.

a**2=(dp/drho)|s is evaluated by signed constant-entropy source flashes.
A saturation-line density slope has a different constraint and cannot replace
this derivative. The comparison includes an independent T/Q Jacobian reference.
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


def speed_of_sound_fd(source, rel_drho=1e-5):
    """Return a**2 from central equilibrium flashes, restoring the PQ state."""
    sound = finite_difference_responses(source, source.p(), source.Q(), rel_drho)[
        "speed_of_sound"
    ]
    return sound**2


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    pressure, quality = 1e5, 0.5
    fluid = jxp.Fluid("Water", method="coolprop")
    state = fluid.get_state(jxp.PQ_INPUTS, pressure, quality)
    source = CP.AbstractState("HEOS", "Water")
    source.update(CP.PQ_INPUTS, pressure, quality)
    before = np.array([source.p(), source.Q(), source.rhomass(), source.smass()])
    sound_squared = speed_of_sound_fd(source)
    after = np.array([source.p(), source.Q(), source.rhomass(), source.smass()])
    np.testing.assert_allclose(after, before, rtol=2e-12, atol=1e-9)
    finite_difference = np.sqrt(sound_squared)
    analytical = saturation_reference(source, float(state.T), quality)["speed_of_sound"]
    assert state.valid and sound_squared > 0
    np.testing.assert_allclose(state.a, finite_difference, rtol=2e-5, atol=1e-9)
    np.testing.assert_allclose(state.a, analytical, rtol=2e-6, atol=1e-9)
    print(
        f"Equilibrium sound speed: JaxProp={state.a:.10e}, signed FD={finite_difference:.10e}, saturation Jacobian={analytical:.10e} m/s"
    )
    options.record(
        "scientific_checks",
        {
            "pressure_Pa": pressure,
            "quality": quality,
            "library_sound_m_per_s": float(state.a),
            "finite_difference_sound_m_per_s": finite_difference,
            "analytical_sound_m_per_s": analytical,
            "source_state_restored": True,
        },
    )
    options.finish()
    return float(state.a), finite_difference, analytical


if __name__ == "__main__":
    main()
