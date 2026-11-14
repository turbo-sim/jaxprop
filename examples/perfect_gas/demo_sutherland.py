from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
import jaxprop as jp

import os
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt
import numpy as np


def main(argv=None):
    """Run the declared scientific case; quick profiles reduce sampling only."""
    options = example_options(__file__, argv, description=__doc__)
    jxp.set_plot_options()

    # Compute viscosity over the temperature range
    fluid = jxp.Fluid(
        "air", method="perfect_gas", model_options={"T_ref": 298.15, "p_ref": 101_325.0}
    )
    T_range = jnp.linspace(300.0, 1000.0, options.points(200))
    states = fluid.get_state(jxp.PT_INPUTS, 101325, T_range)
    mu_vals = states.viscosity
    constants = fluid.constants
    expected = (
        constants.mu_ref
        * (T_range / constants.T_ref) ** 1.5
        * (constants.T_ref + constants.S_mu)
        / (T_range + constants.S_mu)
    )
    np.testing.assert_allclose(mu_vals, expected, rtol=1e-12)
    assert np.all(np.diff(mu_vals) > 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(False)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Dynamic viscosity [Pa·s]")
    ax.set_xlim([T_range.min(), T_range.max()])

    # Plot viscosity
    ax.plot(T_range, mu_vals)
    plt.tight_layout(pad=1)

    # Show plot if not disabled
    options.check_states(locals())
    options.record(
        "execution", "Scientific example completed; see numerical-check records."
    )
    return options.finish()


if __name__ == "__main__":
    main()
