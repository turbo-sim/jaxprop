"""Demo table interpolation.

Quick/full profiles have distinct declared grid sizes. Use explicit output and
table-cache paths; full retains the scientific resolution.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options

import jaxprop as jxp
import numpy as np
import matplotlib.pyplot as plt


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    # ---------------------------
    # Configuration
    # ---------------------------
    outdir = str(options.output_dir)
    fluid_name = "CO2"
    h_min = 200e3  # J/kg
    h_max = 600e3  # J/kg
    p_min = 2e6  # Pa
    p_max = 20e6  # Pa
    N_p = 30 if options.profile == "full" else 12  # Grid size for quick test
    N_h = 30 if options.profile == "full" else 12

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
    test_P = 12e6  # Test pressure [Pa]
    # test_h = 315e3  # Test enthalpy [J/kg]
    # test_P = 5e6  # Test pressure [Pa]
    props_bicubic = fluid_bicubic.get_state(jxp.HmassP_INPUTS, test_h, test_P)

    # # -----------------------------------
    # # Step 3: Compare with CoolProp
    # # -----------------------------------
    # fluid_coolprop = jxp.FluidJAX(fluid_name)
    # props_coolprop = fluid_coolprop.get_state(jxp.HmassP_INPUTS, test_h, test_P)

    # header = f"{'Property':<35} {'Bicubic':>14} {'CoolProp':>14} {'Error [%]':>14}"
    # print("-" * len(header))
    # print("Comparison between interpolated and CoolProp values:")
    # print("-" * len(header))
    # print(header)
    # print("-" * len(header))

    # for prop in jxp.PROPERTIES_CANONICAL:
    #     bicubic_val = props_bicubic[prop]
    #     coolprop_val = props_coolprop[prop]
    #     if coolprop_val != 0 and not (bicubic_val is None or coolprop_val is None):
    #         rel_err = (bicubic_val - coolprop_val) / coolprop_val * 100
    #     else:
    #         rel_err = float("nan")

    #     print(f"{prop:<35} {bicubic_val:14.6e} {coolprop_val:14.6e} {rel_err:14.6e}")

    # # Plot the state
    # prop_x = "enthalpy"
    # prop_y = "pressure"
    # fig, ax = jxp.plot_phase_diagram(fluid_coolprop,
    #     x_prop=prop_x, y_prop=prop_y, x_scale="linear", y_scale="log"
    # )
    # ax.plot(props_bicubic[prop_x], props_bicubic[prop_y], "ko")
    # ax.plot(props_coolprop[prop_x], props_coolprop[prop_y], "b+")

    # # add interpolation domain box (simple line version)
    # ax.plot(
    #     [h_min, h_max, h_max, h_min, h_min],
    #     [p_min, p_min, p_max, p_max, p_min],
    #     "r--",
    #     linewidth=1.5,
    #     label="Interpolation domain",
    # )
    # ax.legend()
    # # Figure rendering/display belongs to options.finish().

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
    assert props_bicubic.valid
    np.testing.assert_allclose(
        [props_bicubic.h, props_bicubic.p], [test_h, test_P], rtol=1e-12
    )
    for name in ("temperature", "density", "internal_energy", "entropy"):
        assert np.isfinite(props_bicubic.raw_property(name))
    options.record(
        "selected_state",
        dict(
            valid=True,
            enthalpy_J_per_kg=float(props_bicubic.h),
            pressure_Pa=float(props_bicubic.p),
            temperature_K=float(props_bicubic.T),
            density_kg_per_m3=float(props_bicubic.rho),
        ),
    )
    return options.finish()


if __name__ == "__main__":
    main()
