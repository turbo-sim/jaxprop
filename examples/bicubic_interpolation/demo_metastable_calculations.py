"""Build and independently check a physical metastable Water liquid table.

The original h=[0.3, 1.8] MJ/kg, p=[0.1, 13] MPa rectangle requested
h=1.711764706 MJ/kg at 13 MPa beyond the connected liquid spinodal
h=1.710150809 MJ/kg. Its rejection remains recorded in z_EXAMPLES_AUDIT.md.

This revised liquid study uses h=[0.3, 1.4] MJ/kg at the original pressures,
with positive cv/K_T margins. Full mode retains 120-by-120 nodes and 150
saturation points; quick mode is a separately labeled 12-by-12 study.
The original 12 MPa, 578.95 K demonstration point remains in the new domain.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options

import CoolProp.CoolProp as CP
import numpy as np
import jaxprop as jxp

ORIGINAL_REJECTED_DOMAIN = dict(
    h_min=300e3,
    h_max=1800e3,
    p_min=1e5,
    p_max=13e6,
    first_recorded_failure_h=1711764.705882353,
    connected_liquid_spinodal_h_at_13MPa=1710150.8088949814,
)


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    full = options.profile == "full"
    settings = dict(
        h_min=300e3,
        h_max=1400e3,
        p_min=1e5,
        p_max=13e6,
        N_h=120 if full else 12,
        N_p=120 if full else 12,
        N_p_sat=150 if full else 24,
        coarse_step=1,
        metastable_phase="liquid",
        metastable_domain="physical",
        gradient_method="forward",
        table_dir=str(options.table_dir),
    )
    fluid = jxp.Fluid("Water", method="bicubic", table_options=settings)
    table = fluid.table
    cv = np.asarray(table["isochoric_heat_capacity"]["value"])
    bulk = np.asarray(table["isothermal_bulk_modulus"]["value"])
    assert np.all(cv > 0) and np.all(bulk > 0)
    assert np.all(np.asarray(table["phase_cell_validity"]))

    # Independently flash homogeneous liquid rho/T at nine representative
    # nodes. An equilibrium HP bulk is not a metastable liquid reference.
    source = CP.AbstractState("HEOS", "Water")
    source.specify_phase(CP.iphase_liquid)
    errors = []
    for i in (0, settings["N_h"] // 2, settings["N_h"] - 1):
        for j in (0, settings["N_p"] // 2, settings["N_p"] - 1):
            rho = float(table["density"]["value"][i, j])
            temperature = float(table["temperature"]["value"][i, j])
            source.update(CP.DmassT_INPUTS, rho, temperature)
            pressure = float(table["pressure"]["value"][i, j])
            enthalpy = float(table["enthalpy"]["value"][i, j])
            assert source.cvmass() > 0 and source.isothermal_compressibility() > 0
            error = max(
                abs(source.p() - pressure) / pressure,
                abs(source.hmass() - enthalpy)
                / (source.gas_constant() / source.molar_mass() * source.T_critical()),
            )
            assert error < 2e-8
            errors.append(error)

    state = fluid.get_state(jxp.PT_INPUTS, 12e6, 578.95)
    assert state.valid and np.isfinite(state.rho)
    recovered = fluid.get_state(jxp.HmassP_INPUTS, state.h, state.p)
    np.testing.assert_allclose(recovered.T, 578.95, rtol=1e-9)
    print(f"Physical Water table: {settings['N_h']} x {settings['N_p']}")
    print(f"Minimum sampled cv: {cv.min():.6g} J/(kg K)")
    print(f"Minimum sampled K_T: {bulk.min():.6g} Pa")
    print(f"Density at 12 MPa, 578.95 K: {float(state.rho):.9g} kg/m^3")
    options.record("settings", settings)
    options.record(
        "domain_revision",
        dict(
            original_rejected=ORIGINAL_REJECTED_DOMAIN,
            justification="Bounded liquid enthalpy interval with positive sampled/interpolated cv/K_T margins; original pressure range and full resolutions retained.",
        ),
    )
    options.record(
        "checks",
        dict(
            all_nodes_positive_cv_and_bulk=True,
            all_interpolation_cells_verified=True,
            independent_source_points=len(errors),
            maximum_scaled_source_residual=max(errors),
            minimum_cv=float(cv.min()),
            minimum_bulk_modulus=float(bulk.min()),
            pt_hp_temperature_roundtrip_K=float(recovered.T),
        ),
    )
    return options.finish()


if __name__ == "__main__":
    main()
