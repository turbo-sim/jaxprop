"""Demo table interpolation with saturation.

Quick/full profiles have distinct declared grid sizes. Use explicit output and
table-cache paths; full retains the scientific resolution.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options

import jaxprop as jxp
import matplotlib.pyplot as plt


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    fluid_name = "nitrogen"
    h_min = -120e3  # J/kg
    h_max = -40e3  # J/kg: physical liquid branch with positive cv/K_T margin
    p_min = 1e5  # Pa
    p_max = 1e6  # Pa
    N_p = 60 if options.profile == "full" else 12  # Number of pressure points
    N_h = 60 if options.profile == "full" else 12  # Number of enthalpy points

    # fl = jxp.Fluid(fluid_name)
    # print(fl.critical_point["pressure"])

    # ---------------------------
    # Delete existing tables3
    # ---------------------------
    outdir = str(options.output_dir)
    # if os.path.exists(outdir):
    #     shutil.rmtree(outdir, ignore_errors=True)

    # # ---------------------------
    # # First call: generate table
    # # ---------------------------
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
            "metastable_phase": "liquid",
            "gradient_method": "forward",
        },
    )

    # ---------------------------
    # Step 2: Interpolate at (h, P)
    # ---------------------------
    # test_h = 100e3   # Test enthalpy [J/kg]
    # test_P = 1.1e6     # Test pressure [Pa]
    test_h = -60e3  # Test enthalpy [J/kg]
    test_P = 5e5  # Test pressure [Pa]
    props_bicubic = fluid_bicubic.get_state(jxp.HmassP_INPUTS, test_h, test_P)

    # -----------------------------------
    # Step 3: Compare with CoolProp
    # -----------------------------------
    fluid_coolprop = jxp.Fluid(fluid_name, method="coolprop")
    props_coolprop = fluid_coolprop.get_state_metastable(
        "h",
        test_h,
        "p",
        test_P,
        rhoT_guess=[float(props_bicubic.rho), float(props_bicubic.T)],
        supersaturation=False,
        # Metastable tables store the same generalized quality indicator.
        generalize_quality=True,
    )

    header = f"{'Property':<35} {'Bicubic':>14} {'CoolProp':>14} {'Error [%]':>14}"
    print("-" * len(header))
    print("Comparison between interpolated and CoolProp values:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for prop in sorted(
        props_bicubic.available_properties & props_coolprop.available_properties
    ):
        bicubic_val = props_bicubic.raw_property(prop)
        coolprop_val = props_coolprop.raw_property(prop)
        if coolprop_val != 0 and not (bicubic_val is None or coolprop_val is None):
            rel_err = (bicubic_val - coolprop_val) / coolprop_val * 100
        else:
            rel_err = float("nan")

        print(f"{prop:<35} {bicubic_val:14.6e} {coolprop_val:14.6e} {rel_err:14.6e}")

    # Plot the state
    prop_x = "enthalpy"
    prop_y = "pressure"
    fig, ax = jxp.plot_phase_diagram(
        fluid_coolprop, x_prop=prop_x, y_prop=prop_y, x_scale="linear", y_scale="log"
    )
    ax.plot(props_bicubic[prop_x], props_bicubic[prop_y], "ko")
    ax.plot(props_coolprop[prop_x], props_coolprop[prop_y], "b+")

    # add interpolation domain box (simple line version)
    ax.plot(
        [h_min, h_max, h_max, h_min, h_min],
        [p_min, p_min, p_max, p_max, p_min],
        "r--",
        linewidth=1.5,
        label="Interpolation domain",
    )

    sat_bicubic = fluid_bicubic.get_state_saturation(jxp.PQ_INPUTS, test_P, 0.0)

    sat_coolprop = fluid_coolprop.get_state(jxp.PQ_INPUTS, test_P, 0.0)
    print(60 * "-")
    print(60 * "-")
    header = f"{'Property':<35} {'Cubic':>14} {'CoolProp':>14} {'Error [%]':>14}"
    print("-" * len(header))
    print("Comparison between interpolated and CoolProp values for saturation curve:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for prop in sorted(
        sat_bicubic.available_properties & sat_coolprop.available_properties
    ):
        bicubic_val = sat_bicubic.raw_property(prop)
        coolprop_val = sat_coolprop.raw_property(prop)
        if coolprop_val != 0 and not (bicubic_val is None or coolprop_val is None):
            rel_err = (bicubic_val - coolprop_val) / coolprop_val * 100
        else:
            rel_err = float("nan")

        print(f"{prop:<35} {bicubic_val:14.6e} {coolprop_val:14.6e} {rel_err:14.6e}")

    # Plot the state
    prop_x = "enthalpy"
    prop_y = "pressure"
    fig, ax = jxp.plot_phase_diagram(
        fluid_coolprop, x_prop=prop_x, y_prop=prop_y, x_scale="linear", y_scale="log"
    )
    ax.plot(props_bicubic[prop_x], props_bicubic[prop_y], "ko")
    ax.plot(props_coolprop[prop_x], props_coolprop[prop_y], "b+")

    # add interpolation domain box (simple line version)
    ax.plot(
        [h_min, h_max, h_max, h_min, h_min],
        [p_min, p_min, p_max, p_max, p_min],
        "r--",
        linewidth=1.5,
        label="Interpolation domain",
    )

    ax.legend()
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
    assert props_bicubic.valid and props_coolprop.valid
    assert sat_bicubic.valid and sat_coolprop.valid
    options.record(
        "domain_revision",
        dict(
            original_h_max=1.0,
            physical_h_max=h_max,
            reason="The former rectangle included negative-cv Nitrogen states. The revised bounded liquid domain retains the original selected HP point and full 60-by-60 resolution.",
        ),
    )
    options.record(
        "checked_states",
        dict(
            homogeneous_HP_core_valid=True,
            saturation_endpoint_valid=True,
            selected_h=float(props_bicubic.h),
            selected_p=float(props_bicubic.p),
            source_temperature_K=float(props_coolprop.T),
            interpolated_temperature_K=float(props_bicubic.T),
        ),
    )
    return options.finish()


if __name__ == "__main__":
    main()
