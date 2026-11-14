"""A configured water/nitrogen parcel model at common pressure and temperature."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    import numpy as np
    import jaxprop as jp
    from examples.two_component._model import mixture, check_state

    # Original pressure, temperature and first/second-component mass ratio.
    p, T, ratio = 5e5, 350.0, 50.0
    components = tuple(
        jp.Fluid(name, method="coolprop_jax") for name in ("water", "nitrogen")
    )
    fluid = mixture(components, ratio)
    state = fluid.get_state(jp.PT_INPUTS, p, T)
    check_state(state)
    # Wood's frozen-thermal speed is distinct from the thermally relaxed result.
    assert np.isfinite(state.speed_of_sound_p) and np.isfinite(state.speed_of_sound_pT)
    assert state.speed_of_sound_pT <= state.speed_of_sound_p * (1 + 1e-12)
    print(f"Wood speed: {float(state.speed_of_sound_p):.8f} m/s")
    print(f"Common-p/T equilibrium speed: {float(state.speed_of_sound_pT):.8f} m/s")
    options.record("settings", dict(p=p, T=T, mass_ratio=ratio))
    options.record("wood_speed_m_per_s", float(state.speed_of_sound_p))
    options.record("equilibrium_speed_m_per_s", float(state.speed_of_sound_pT))
    options.finish()


if __name__ == "__main__":
    main()
