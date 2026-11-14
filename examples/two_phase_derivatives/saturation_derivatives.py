"""Verify pure-fluid equilibrium derivatives in saturation coordinates.

An equilibrium mixture is parameterized by saturation temperature T and vapor
mass fraction Q. Its specific volume, energy and entropy are mass-weighted
endpoint values. Differentiating these functions under the *specified*
constraint gives cv, sound speed and Gruneisen. A derivative along a saturation
endpoint is not an isentropic derivative; its inverse cannot be plotted as a
speed of sound. This keeps the original Water, Q=0.5 and 20--300 C range.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options

import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np

import jaxprop as jxp


def saturation_reference(source, temperature, quality):
    """Independent analytical T/Q Jacobian built from raw source endpoints.

    For x=(v,u,s,h), x_T=(1-Q)x'_L+Q*x'_V and x_Q=x_V-x_L.
    At constant density, Q_T=-v_T/v_Q; at constant entropy,
    Q_T=-s_T/s_Q. Thus cv=u_T-u_Q*v_T/v_Q and
    a**2=p'_sat/[-rho**2*(v_T-v_Q*s_T/s_Q)]. No copied fluid
    property kernel or single-phase partial derivative is used in the interior.
    """
    endpoints = []
    for endpoint in (0.0, 1.0):
        source.update(CP.QT_INPUTS, endpoint, float(temperature))
        density = source.rhomass()
        endpoints.append(
            dict(
                density=density,
                v=1 / density,
                u=source.umass(),
                s=source.smass(),
                h=source.hmass(),
                v_T=-source.first_saturation_deriv(CP.iDmass, CP.iT) / density**2,
                u_T=source.first_saturation_deriv(CP.iUmass, CP.iT),
                s_T=source.first_saturation_deriv(CP.iSmass, CP.iT),
                h_T=source.first_saturation_deriv(CP.iHmass, CP.iT),
                p_T=source.first_saturation_deriv(CP.iP, CP.iT),
            )
        )
    liquid, vapor = endpoints
    interpolate = lambda name: (1 - quality) * liquid[name] + quality * vapor[name]
    difference = lambda name: vapor[name] - liquid[name]
    density = 1 / interpolate("v")
    p_T = difference("s") / difference("v")  # Clapeyron equation
    cv = interpolate("u_T") - difference("u") * interpolate("v_T") / difference("v")
    drho_dT_s = -(density**2) * (
        interpolate("v_T") - difference("v") * interpolate("s_T") / difference("s")
    )
    sound_squared = p_T / drho_dT_s
    assert cv > 0 and sound_squared > 0
    return dict(
        density=density,
        isochoric_heat_capacity=cv,
        speed_of_sound=np.sqrt(sound_squared),
        gruneisen=p_T / (density * cv),
        saturation_pressure_slope=p_T,
        source_saturation_pressure_slope=liquid["p_T"],
        liquid_density=liquid["density"],
        vapor_density=vapor["density"],
        enthalpy_pressure_derivative=cv / p_T + 1 / density,
        enthalpy_density_derivative=-temperature * p_T / density**2,
    )


def finite_difference_responses(source, pressure, quality, relative_step=1e-5):
    """Compute signed equilibrium responses through independent source flashes.

    The old exploratory cv check differentiated pressure and took a square root.
    Here cv=(du/dT)|rho in J/(kg K), a**2=(dp/drho)|s in m**2/s**2,
    and Gamma=(dp/du)|rho/rho is dimensionless. Perturbation energy uses a
    physical R*T/p/rho scale rather than the arbitrary absolute energy zero.
    Raw CoolProp cv at a two-phase point is a homogeneous surrogate and can be
    enormous, so it must not determine a supposedly local energy stencil.
    """
    source.update(CP.PQ_INPUTS, float(pressure), float(quality))
    rho, temperature = source.rhomass(), source.T()
    entropy, energy = source.smass(), source.umass()
    energy_scale = max(
        abs(source.gas_constant() / source.molar_mass() * temperature),
        abs(pressure / rho),
        1,
    )
    dT, drho, du = (
        relative_step * temperature,
        relative_step * rho,
        relative_step * energy_scale,
    )
    try:
        source.update(CP.DmassT_INPUTS, rho, temperature + dT)
        up = source.umass()
        source.update(CP.DmassT_INPUTS, rho, temperature - dT)
        um = source.umass()
        source.update(CP.DmassSmass_INPUTS, rho + drho, entropy)
        pp = source.p()
        source.update(CP.DmassSmass_INPUTS, rho - drho, entropy)
        pm = source.p()
        source.update(CP.DmassUmass_INPUTS, rho, energy + du)
        p_up = source.p()
        source.update(CP.DmassUmass_INPUTS, rho, energy - du)
        p_um = source.p()
    finally:
        source.update(CP.PQ_INPUTS, float(pressure), float(quality))
    sound_squared = (pp - pm) / (2 * drho)
    assert sound_squared > 0
    return dict(
        isochoric_heat_capacity=(up - um) / (2 * dT),
        speed_of_sound=np.sqrt(sound_squared),
        gruneisen=(p_up - p_um) / (2 * du * rho),
    )


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    jxp.set_plot_options()
    fluid = jxp.Fluid("Water", method="coolprop")
    source = CP.AbstractState("HEOS", "Water")
    quality = 0.5
    temperatures = np.linspace(20, 300, options.points(11, 3)) + 273.15
    records = []
    relative_errors = {
        name: [] for name in ("isochoric_heat_capacity", "speed_of_sound", "gruneisen")
    }
    for temperature in temperatures:
        state = fluid.get_state(jxp.QT_INPUTS, quality, temperature)
        reference = saturation_reference(source, temperature, quality)
        finite_difference = finite_difference_responses(source, float(state.p), quality)
        assert state.valid
        assert not state.property_valid("isobaric_heat_capacity")
        np.testing.assert_allclose(
            reference["saturation_pressure_slope"],
            reference["source_saturation_pressure_slope"],
            rtol=2e-10,
        )
        for name in relative_errors:
            value = float(state.raw_property(name))
            assert np.isfinite(value)
            np.testing.assert_allclose(value, reference[name], rtol=2e-6, atol=1e-10)
            np.testing.assert_allclose(
                value, finite_difference[name], rtol=2e-4, atol=1e-10
            )
            relative_errors[name].append(abs(value / finite_difference[name] - 1))
        records.append(
            dict(
                temperature=float(temperature),
                library_sound=float(state.a),
                fd_sound=finite_difference["speed_of_sound"],
                **reference
            )
        )
    figure, axes = plt.subplots(figsize=(6, 5))
    axes.semilogy(
        temperatures,
        [row["liquid_density"] for row in records],
        "bo-",
        label="Saturated liquid",
    )
    axes.semilogy(
        temperatures,
        [row["vapor_density"] for row in records],
        "ro-",
        label="Saturated vapor",
    )
    axes.set(xlabel="Temperature (K)", ylabel="Density (kg/m³)")
    axes.legend()
    figure.tight_layout()
    figure, axes = plt.subplots(figsize=(6, 5))
    axes.plot(
        temperatures,
        [row["library_sound"] for row in records],
        "ko",
        label="JaxProp equilibrium",
    )
    axes.plot(
        temperatures,
        [row["fd_sound"] for row in records],
        "+",
        label="Signed constant-entropy finite difference",
    )
    axes.set(xlabel="Temperature (K)", ylabel="Speed of sound (m/s)")
    axes.legend()
    figure.tight_layout()
    options.record(
        "scientific_checks",
        {
            "fluid": "Water",
            "quality": quality,
            "samples": len(records),
            "max_relative_fd_errors": {
                name: max(values) for name, values in relative_errors.items()
            },
            "cp_unavailable_as_expected": True,
        },
    )
    options.finish()
    return records


if __name__ == "__main__":
    main()
