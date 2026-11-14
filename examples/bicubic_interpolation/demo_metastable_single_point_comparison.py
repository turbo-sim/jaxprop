"""Demo metastable single point comparison.

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


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    fluid_name = "nitrogen"
    h_min = -110e3  # J/kg
    h_max = -10e3  # J/kg
    p_min = 3e4  # Pa
    p_max = 1e6  # Pa
    N_p = 165 if options.profile == "full" else 24  # Number of pressure points
    N_h = 165 if options.profile == "full" else 24  # Number of enthalpy points
    metastable_phase = "liquid"

    outdir = str(options.output_dir)

    # ---------------------------
    # Build bicubic table
    # ---------------------------
    # Preserve the original research rectangle explicitly. The physical mode
    # rejects its nonpositive-cv nodes; formal mode masks possible response poles.
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
            "metastable_phase": metastable_phase,
            "metastable_domain": "formal",
            "gradient_method": "forward",
        },
    )

    # This local HP/PT/saturation check does not validate the whole rectangular
    # domain. Preserve that domain and report its caloric instability explicitly.
    cv_nodes = np.asarray(fluid_bicubic.table["isochoric_heat_capacity"]["value"])
    calorically_unstable_nodes = np.count_nonzero(cv_nodes <= 0)
    if calorically_unstable_nodes:
        print(
            "Caloric-domain notice: "
            f"{calorically_unstable_nodes}/{cv_nodes.size} nodes have cv <= 0. "
            "Gamma has a pole at cv=0; review the physical domain/model. "
            "Agreement at the selected point does not establish whole-table accuracy."
        )

    hl = -85528.8
    p = 5.0719e05

    T_test = fluid_bicubic.get_state(jxp.HmassP_INPUTS, hl, p)["T"]
    print(T_test)

    h_test = fluid_bicubic.get_state(jxp.PT_INPUTS, p, T_test)["h"]
    print(h_test)
    np.testing.assert_allclose(h_test, hl, rtol=1e-10, atol=1e-7)

    quality = 0.0 if metastable_phase == "liquid" else 1.0
    sat = fluid_bicubic.get_state_saturation(jxp.PQ_INPUTS, p, quality)
    print(sat)

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
            "metastable_phase": metastable_phase,
        },
    )
    options.record(
        "formal_study",
        dict(
            research_only=True,
            whole_domain_accuracy_certified=False,
            calorically_unstable_nodes=int(calorically_unstable_nodes),
            selected_HP_PT_enthalpy_absolute_error_J_per_kg=float(abs(h_test - hl)),
        ),
    )
    return options.finish()


if __name__ == "__main__":
    main()
