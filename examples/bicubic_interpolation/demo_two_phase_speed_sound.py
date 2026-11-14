"""Demo two phase speed sound.

Quick/full profiles have distinct declared grid sizes. Use explicit output and
table-cache paths; full retains the scientific resolution.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options

import jaxprop as jxp
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    jxp.set_plot_options()

    # ---------------------------
    # Configuration
    # ---------------------------
    outdir = str(options.output_dir)
    fluid_name = "CO2"
    h_min = 100e3  # J/kg
    h_max = 600e3  # J/kg
    p_min = 1e6  # Pa
    p_max = 20e6  # Pa
    N_p = 100 if options.profile == "full" else 24  # Grid size for quick test
    N_h = 100 if options.profile == "full" else 24

    fluid_bicubic = jxp.Fluid(
        name=fluid_name,
        method="bicubic",
        backend="HEOS",
        table_options={
            "h_min": h_min,
            "h_max": h_max,
            "p_min": p_min,
            "p_max": p_max,
            "N_h": N_h,
            "N_p": N_p,
            "table_dir": str(options.table_dir),
        },
    )

    # ---------------------------
    # Step 2: Interpolate at (h, P)
    # ---------------------------
    test_h = 500e3  # Test enthalpy [J/kg]
    test_p = 2e6  # Test pressure [Pa]

    test_h = jnp.linspace(h_min, h_max, 3000 if options.profile == "full" else 160)

    # A rectangular table deliberately excludes cells crossed by a phase
    # boundary. Preserve their gaps and report coverage instead of extrapolating.
    props_bicubic = fluid_bicubic.get_state(
        jxp.HmassP_INPUTS, test_h, test_p, errors="mask"
    )

    # -----------------------------------
    # Step 3: Compare with CoolProp
    # -----------------------------------
    fluid_coolprop = jxp.Fluid(fluid_name, method="coolprop_jax")
    props_coolprop = fluid_coolprop.get_state(
        jxp.HmassP_INPUTS, test_h, test_p, errors="mask"
    )

    # Plot the state
    prop_x = "enthalpy"
    prop_y = "pressure"
    fig, ax = jxp.plot_phase_diagram(
        fluid_coolprop, x_prop=prop_x, y_prop=prop_y, x_scale="linear", y_scale="log"
    )
    ax.plot(
        props_bicubic.raw_property(prop_x), props_bicubic.raw_property(prop_y), "k+"
    )

    # add interpolation domain box (simple line version)
    ax.plot(
        [h_min, h_max, h_max, h_min, h_min],
        [p_min, p_min, p_max, p_max, p_min],
        "r--",
        linewidth=1.5,
        label="Interpolation domain",
    )
    ax.legend()
    fig.tight_layout(pad=1)

    fig, ax = plt.subplots()
    ax.set_xlabel("Enthalpy")
    ax.set_ylabel("Speed of sound")
    ax.plot(
        props_coolprop.raw_property("h"),
        props_coolprop.raw_property("a"),
        "b+",
        label="coolprop",
    )
    ax.plot(
        props_bicubic.raw_property("h"),
        props_bicubic.raw_property("a"),
        color="k",
        label="bicubic",
    )
    ax.legend()
    fig.tight_layout(pad=1)

    # Figure rendering/display belongs to options.finish().

    options.record(
        "settings",
        {
            "fluid_name": fluid_name,
            "h_min": h_min,
            "h_max": h_max,
            "p_min": p_min,
            "p_max": p_max,
            "N_h": N_h,
            "N_p": N_p,
        },
    )
    eligible = np.asarray(props_bicubic.valid) & np.asarray(props_coolprop.valid)
    assert np.any(eligible)
    options.record(
        "coverage",
        dict(
            requested=int(test_h.size),
            bicubic_valid=int(np.count_nonzero(props_bicubic.valid)),
            source_valid=int(np.count_nonzero(props_coolprop.valid)),
            finite_sound_comparisons=int(
                np.count_nonzero(
                    np.isfinite(np.asarray(props_bicubic.raw_property("a")))
                    & np.isfinite(np.asarray(props_coolprop.raw_property("a")))
                )
            ),
            phase_boundary_gaps_preserved=True,
        ),
    )
    return options.finish()


if __name__ == "__main__":
    main()
