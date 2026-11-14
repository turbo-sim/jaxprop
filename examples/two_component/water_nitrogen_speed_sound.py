"""Compare relaxed water/nitrogen sound speeds with an isentropic AD check.

The ratio configures each fixed-composition model. Its equilibrium speed obeys
a**2 = 1/(d rho/d p at fixed s), providing a check through the supported PS
inverse rather than pretending that a density-based inverse exists.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options


def isentropic_sound_speed(fluid, pressure, entropy):
    """Differentiate density at fixed entropy through the public PS operation."""
    import jax
    import jax.numpy as jnp
    import jaxprop as jp

    slope = jax.jacfwd(
        lambda p: fluid.get_state(
            jp.PSmass_INPUTS, p, entropy, errors="mask"
        ).raw_property("density")
    )(jnp.asarray(pressure))
    return jnp.sqrt(1 / slope)


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    import numpy as np
    import jaxprop as jp
    import matplotlib.pyplot as plt
    from examples.two_component._model import mixture, check_state

    jp.set_plot_options()
    p, T = 5e5, 350.0
    ratios = np.linspace(50, 500, 51 if options.profile == "full" else 7)
    components = tuple(
        jp.Fluid(name, method="coolprop_jax") for name in ("water", "nitrogen")
    )
    wood, equilibrium, derivative = [], [], []
    for ratio in ratios:
        fluid = mixture(components, ratio)
        state = fluid.get_state(jp.PT_INPUTS, p, T)
        check_state(state)
        actual = float(isentropic_sound_speed(fluid, p, state.s))
        np.testing.assert_allclose(actual, state.speed_of_sound_pT, rtol=2e-5)
        assert state.speed_of_sound_pT <= state.speed_of_sound_p * (1 + 1e-12)
        wood.append(float(state.speed_of_sound_p))
        equilibrium.append(float(state.speed_of_sound_pT))
        derivative.append(actual)
    fig, ax = plt.subplots()
    ax.plot(ratios, equilibrium, label="$pT$ equilibrium")
    ax.plot(ratios, wood, label="$p$ equilibrium (Wood)")
    ax.plot(ratios, derivative, "ko", markersize=3, label="Isentropic AD")
    ax.set_xlabel("Component mass ratio R = m1/m2")
    ax.set_ylabel("Speed of sound [m/s]")
    ax.legend()
    fig.tight_layout(pad=1)
    relative = np.abs(np.asarray(derivative) / np.asarray(equilibrium) - 1)
    options.record(
        "settings",
        dict(p=p, T=T, mass_ratio_bounds=[50, 500], sample_points=int(ratios.size)),
    )
    options.record("maximum_relative_acoustic_error", float(np.max(relative)))
    options.finish()


if __name__ == "__main__":
    main()
