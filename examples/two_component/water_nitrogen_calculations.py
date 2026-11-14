"""Polytropic water/air parcel expansion and a preliminary nozzle estimate.

The fixed-composition parcels share p and T but exchange no mass. For expansion,
dh/dp=eta/rho and dh=cp*dT-cp*mu_JT*dp, hence dT/dp=(dh/dp)/cp+mu_JT.
The temperature sign is important: subtracting mu_JT integrates a different,
inconsistent enthalpy path. Configured component/mixture objects are reused.

Velocity follows the adiabatic stagnation-enthalpy balance. No small artificial
energy is added at the inlet. A sonic section is reported only if the integrated
trajectory crosses Mach 1; a nearest sample is not evidence of choking. The
subsequent throat/blade geometry remains a preliminary geometric assumption,
not a resolved nozzle-flow or loss model.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options


def barotropic_model_two_component(
    fluid_name_1,
    fluid_name_2,
    mixture_ratio,
    T_in,
    p_in,
    p_out,
    efficiency,
    backend="HEOS",
    process_type=None,
    ODE_solver="LSODA",
    ODE_tolerance=1e-8,
    *,
    fluid=None,
    sample_points=200,
):
    """Integrate the thermodynamic p/h/T path and return numerical series/solution.

    ``mixture_ratio`` means m1/m2. A prepared ``fluid`` avoids reconstruction
    across related process runs. Expansion uses dh/dp=eta/rho; compression uses
    1/(eta*rho). The kinetic diagnostics are defined only for the expansion
    stagnation-energy model. Both component fluids must remain in their declared
    source domains. The original solver tolerances and full sampling are retained.
    """
    import numpy as np
    from scipy.integrate import solve_ivp
    import jaxprop as jp
    from examples.two_component._model import mixture, check_state

    if process_type not in ("compression", "expansion"):
        raise ValueError("process_type must be 'compression' or 'expansion'.")
    if not 0 <= efficiency <= 1:
        raise ValueError("The process efficiency must be in [0,1].")
    if process_type == "compression" and efficiency == 0:
        raise ValueError("Compression requires positive efficiency.")
    if process_type == "expansion" and not p_in > p_out > 0:
        raise ValueError("Expansion requires p_in > p_out > 0.")
    if process_type == "compression" and not 0 < p_in < p_out:
        raise ValueError("Compression requires 0 < p_in < p_out.")
    factor = efficiency if process_type == "expansion" else 1 / efficiency
    if fluid is None:
        components = tuple(
            jp.Fluid(name, backend=backend) for name in (fluid_name_1, fluid_name_2)
        )
        fluid = mixture(
            components, mixture_ratio, name=f"{fluid_name_1}_{fluid_name_2}"
        )
    inlet = fluid.get_state(jp.PT_INPUTS, p_in, T_in)
    check_state(inlet)
    h_in = float(inlet.h)

    def rhs(p, y):
        state = fluid.get_state(jp.PT_INPUTS, p, y[1])
        dhdp = factor / state.rho
        return [dhdp, dhdp / state.cp + state.joule_thomson]

    # Integrate the enthalpy CHANGE: an arbitrary source energy zero must not
    # enlarge relative-error control. Return physical h coordinates below.
    solution = solve_ivp(
        rhs,
        (p_in, p_out),
        (0.0, T_in),
        method=ODE_solver,
        t_eval=np.linspace(p_in, p_out, sample_points),
        dense_output=True,
        rtol=ODE_tolerance,
        atol=ODE_tolerance,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    solution.enthalpy_increment = solution.y[0].copy()
    solution.y[0] += h_in
    increment_solution = solution.sol

    def physical_coordinates(pressure):
        values = increment_solution(pressure).copy()
        values[0] += h_in
        return values

    solution.sol = physical_coordinates
    state = fluid.get_state(jp.PT_INPUTS, solution.t, solution.y[1])
    check_state(state)
    values = {
        name: np.asarray(state.raw_property(name))
        for name in state.available_properties
    }
    values["integrated_enthalpy"] = solution.y[0]
    caloric_scale = max(float(np.max(state.cp * state.T)), 1.0)
    residual = np.asarray(state.h) - solution.y[0]
    np.testing.assert_allclose(
        residual, 0.0, atol=max(1e-4, 4 * ODE_tolerance * caloric_scale), rtol=0
    )
    values["enthalpy_residual"] = residual
    if process_type == "expansion":
        energy_drop = h_in - solution.y[0]
        assert np.min(energy_drop) >= -64 * np.finfo(float).eps * caloric_scale
        values["velocity"] = np.sqrt(2 * np.maximum(energy_drop, 0))
        values["Mach"] = values["velocity"] / np.asarray(state.speed_of_sound)
        # ds/dp=(eta-1)/(rho*T), so pressure decline cannot reduce entropy.
        assert np.min(
            np.asarray(state.s) - float(inlet.s)
        ) >= -4 * ODE_tolerance * float(np.max(state.cp))
    return values, solution


def find_sonic_state(fluid, solution):
    """Locate a genuine Mach-one crossing on the continuous ODE trajectory.

    Return None if the solved pressure interval never reaches a sonic state.
    The area/mass-flow estimate must not silently use an endpoint in that case.
    """
    import numpy as np
    from scipy.optimize import brentq
    import jaxprop as jp

    h_in = float(solution.y[0, 0])

    def at_pressure(pressure):
        h, T = solution.sol(pressure)
        state = fluid.get_state(jp.PT_INPUTS, pressure, T)
        velocity = np.sqrt(2 * max(h_in - float(h), 0.0))
        return state, velocity

    def residual(pressure):
        state, velocity = at_pressure(pressure)
        return velocity / float(state.speed_of_sound) - 1

    values = np.asarray([residual(p) for p in solution.t])
    crossings = np.flatnonzero(values[:-1] * values[1:] <= 0)
    if not crossings.size:
        return None
    index = int(crossings[0])
    a, b = sorted((float(solution.t[index]), float(solution.t[index + 1])))
    pressure = brentq(residual, a, b, xtol=1e-6, rtol=1e-10)
    state, velocity = at_pressure(pressure)
    np.testing.assert_allclose(velocity / state.speed_of_sound, 1.0, rtol=1e-8)
    return dict(pressure=pressure, state=state, velocity=velocity)


def blade_geometry(mass_flow, density, velocity, viscosity):
    """Return outlet chord [m] and dimensionless Reynolds number.

    Preserve the original four-blade, aspect-ratio/angle assumptions. Converting
    chord to millimetres is for reporting only; Reynolds uses metres.
    """
    import numpy as np

    n_blades, aspect_ratio, gauge_angle, spacing_to_chord = (
        4.0,
        3.0,
        50 * np.pi / 180,
        1.1,
    )
    area = mass_flow / (density * velocity)
    chord_m = np.sqrt(
        area / (spacing_to_chord * np.cos(gauge_angle) * aspect_ratio * n_blades)
    )
    return chord_m, density * velocity * chord_m / viscosity


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    import csv
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    import jaxprop as jp
    from examples.two_component._model import mixture

    jp.set_plot_options(grid=True)
    # Original full study: water/air, 25 C, p_out=1 bar, eta=0.8,
    # two inlet pressures, ten mass ratios and 200 samples per ODE path.
    fluid_names = ("water", "air")
    T_in, p_out, efficiency = 298.15, 1e5, 0.8
    inlet_pressures = [7e5, 13e5] if options.profile == "full" else [7e5]
    ratios = (
        [20, 30, 40, 50, 100, 150, 200, 300, 400, 500]
        if options.profile == "full"
        else [20, 100, 500]
    )
    sample_points = 200 if options.profile == "full" else 31
    area_throat, efficiency_pump, compressor_specific_power = 250e-6, 0.7, 0.126
    components = tuple(jp.Fluid(name) for name in fluid_names)
    models = {ratio: mixture(components, ratio, name="water_air") for ratio in ratios}
    rows, curves, maximum_energy_residual = [], [], 0.0
    sonic_count = 0
    for p_in in inlet_pressures:
        for ratio in ratios:
            fluid = models[ratio]
            values, solution = barotropic_model_two_component(
                *fluid_names,
                ratio,
                T_in,
                p_in,
                p_out,
                efficiency,
                process_type="expansion",
                ODE_solver="LSODA",
                ODE_tolerance=1e-6,
                fluid=fluid,
                sample_points=sample_points,
            )
            maximum_energy_residual = max(
                maximum_energy_residual,
                float(np.max(np.abs(values["enthalpy_residual"]))),
            )
            sonic = find_sonic_state(fluid, solution)
            row = dict(
                p_in_bar=p_in / 1e5,
                mass_ratio=ratio,
                sonic_crossing=sonic is not None,
                outlet_velocity_m_per_s=float(values["velocity"][-1]),
                outlet_Mach=float(values["Mach"][-1]),
            )
            if sonic is not None:
                sonic_count += 1
                state, speed = sonic["state"], sonic["velocity"]
                mass_flow = float(state.rho) * speed * area_throat
                first = components[0].get_state(jp.PT_INPUTS, p_in, T_in)
                second = components[1].get_state(
                    jp.PT_INPUTS, p_out, float(values["temperature"][-1])
                )
                mass_water, mass_air = mass_flow * np.asarray(fluid.mass_fractions)
                water_volume, air_volume = mass_water / float(
                    first.rho
                ), mass_air / float(second.rho)
                chord, reynolds = blade_geometry(
                    mass_flow,
                    values["density"][-1],
                    values["velocity"][-1],
                    values["viscosity"][-1],
                )
                row.update(
                    sonic_pressure_bar=sonic["pressure"] / 1e5,
                    mass_flow_kg_per_min=mass_flow * 60,
                    water_m3_per_hour=water_volume * 3600,
                    air_m3_per_hour=air_volume * 3600,
                    pump_power_kW=water_volume
                    * (p_in - p_out)
                    / efficiency_pump
                    / 1000,
                    compressor_power_kW=air_volume * 3600 * compressor_specific_power,
                    thrust_N=mass_flow * values["velocity"][-1],
                    chord_mm=chord * 1000,
                    Re_throat=float(state.rho)
                    * speed
                    * np.sqrt(area_throat)
                    / float(state.viscosity),
                    Re_blades=float(reynolds),
                )
            rows.append(row)
            curves.append((p_in, ratio, values))
            print(
                f"p_in={p_in/1e5:.0f} bar, R={ratio:3d}, Ma_out={row['outlet_Mach']:.6f}, sonic crossing={row['sonic_crossing']}"
            )
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (options.output_dir / "water_air_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    colours = dict(zip(ratios, plt.cm.magma(np.linspace(0.25, 0.8, len(ratios)))))
    styles = dict(zip(inlet_pressures, itertools.cycle(("-", "--", "-.", ":"))))
    for pressure, ratio, values in curves:
        for axis, name in zip(
            axes.ravel(), ("density", "vol_frac_2", "speed_of_sound", "Mach")
        ):
            axis.plot(
                values["pressure"] / 1e5,
                values[name],
                color=colours[ratio],
                ls=styles[pressure],
                label=f"R={ratio}, inlet={pressure/1e5:g} bar",
            )
    for axis, label in zip(
        axes.ravel(),
        (
            "Density [kg/m3]",
            "Air parcel volume fraction",
            "Speed of sound [m/s]",
            "Mach number",
        ),
    ):
        axis.set_ylabel(label)
        axis.set_xlim(max(inlet_pressures) / 1e5, p_out / 1e5)
    axes[1, 0].set_xlabel("Pressure [bar]")
    axes[1, 1].set_xlabel("Pressure [bar]")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout(pad=1)
    options.record(
        "settings",
        dict(
            components=list(fluid_names),
            T_in=T_in,
            p_out=p_out,
            efficiency=efficiency,
            inlet_pressures=inlet_pressures,
            mass_ratios=ratios,
            sample_points=sample_points,
            ODE_solver="LSODA",
            ODE_tolerance=1e-6,
            sonic_area_m2=area_throat,
            enthalpy_error_control="integrated change from inlet, independent of source energy zero",
        ),
    )
    options.record("trajectory_count", len(rows))
    options.record("sonic_crossings", sonic_count)
    options.record(
        "maximum_enthalpy_consistency_residual_J_per_kg", maximum_energy_residual
    )
    options.record(
        "geometry_status",
        "Reported only for genuine integrated Mach-one crossings; preliminary area assumptions",
    )
    options.finish()


if __name__ == "__main__":
    main()
