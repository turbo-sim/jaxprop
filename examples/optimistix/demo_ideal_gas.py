"""Recover the static state from stagnation density/pressure and Mach number.

The original dimensional residual stalled at approximately 1.16e-10 J/kg for
Mach zero against an absolute 1e-10 tolerance. Scale the pressure coordinate by
p0 and energy by cp*T0, which is independent of the enthalpy reference zero.
Verify the dimensional energy balance against representable double-precision
roundoff and compare pressure/temperature to the independent ideal-gas formula.
The physical case, 21-point full sweep and Newton tolerances remain unchanged.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options

import time
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx
import jaxprop as jxp


@eqx.filter_jit
def compute_static_state(p0, d0, mach, fluid):
    """Solve the dimensionless energy equation in log(p/p0) coordinates."""
    total = fluid.get_state(jxp.DmassP_INPUTS, d0, p0, errors="mask")
    entropy, enthalpy = total.raw_property("entropy"), total.raw_property("enthalpy")
    energy_scale = total.raw_property("isobaric_heat_capacity") * total.raw_property(
        "temperature"
    )

    def residual(coordinates, args):
        pressure = p0 * jnp.exp(coordinates[0])
        state = fluid.get_state(jxp.PSmass_INPUTS, pressure, entropy, errors="mask")
        velocity = state.raw_property("speed_of_sound") * mach
        return jnp.array(
            [
                (enthalpy - state.raw_property("enthalpy") - 0.5 * velocity**2)
                / energy_scale
            ]
        )

    solution = optx.root_find(
        residual,
        optx.Newton(rtol=1e-10, atol=1e-10),
        y0=jnp.array([jnp.log(0.9)]),
        args=(),
    )
    return fluid.get_state(
        jxp.PSmass_INPUTS, p0 * jnp.exp(solution.value[0]), entropy, errors="mask"
    )


def verify_static_state(total, state, mach):
    """Check physical energy and the independent constant-cp isentropic law."""
    assert total.valid and state.valid
    cp, cv = float(total.cp), float(total.cv)
    gamma = cp / cv
    ratio = 1 / (1 + 0.5 * (gamma - 1) * float(mach) ** 2)
    expected_temperature = float(total.T) * ratio
    expected_pressure = float(total.p) * ratio ** (gamma / (gamma - 1))
    np.testing.assert_allclose(
        [state.T, state.p],
        [expected_temperature, expected_pressure],
        rtol=2e-10,
        atol=1e-10,
    )
    velocity = float(state.a) * float(mach)
    energy_residual = float(total.h) - float(state.h) - 0.5 * velocity**2
    # Cancellation near Mach zero is limited by floating energy spacing. This
    # is a physical rounding budget, not a relative-to-enthalpy-zero tolerance.
    energy_budget = 32 * np.finfo(np.float64).eps * cp * float(total.T) + 4 * abs(
        np.spacing(float(total.h))
    )
    assert abs(energy_residual) <= energy_budget, (energy_residual, energy_budget)
    return dict(
        mach=float(mach),
        pressure_Pa=float(state.p),
        temperature_K=float(state.T),
        energy_residual_J_per_kg=energy_residual,
        energy_roundoff_budget_J_per_kg=energy_budget,
        relative_pressure_error=abs(float(state.p) / expected_pressure - 1),
    )


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    jxp.set_plot_options(grid=False)
    fluid = jxp.Fluid(
        "Air", method="perfect_gas", model_options={"T_ref": 300.0, "p_ref": 101325.0}
    )
    pressure, density = jnp.asarray(1e5), jnp.asarray(1.2)
    total = fluid.get_state(jxp.DmassP_INPUTS, density, pressure)
    mach_values = jnp.linspace(0, 1, options.points(21, 5))
    # Compile once, then synchronize every complete return. Per-point elapsed
    # values remain single observations, not a statistical speed comparison.
    jax.block_until_ready(
        compute_static_state(pressure, density, mach_values[0], fluid)
    )
    records = []
    for mach in mach_values:
        started = time.perf_counter()
        state = jax.block_until_ready(
            compute_static_state(pressure, density, mach, fluid)
        )
        elapsed = (time.perf_counter() - started) * 1e3
        record = verify_static_state(total, state, mach)
        records.append(record)
        print(
            f"Ma={mach:.3f}, p={state.p*1e-5:.6f} bar, synchronized elapsed={elapsed:.3f} ms"
        )
    figure, axes = plt.subplots(figsize=(5, 4))
    axes.plot(
        [row["mach"] for row in records],
        [row["pressure_Pa"] * 1e-5 for row in records],
        "o-",
        color="tab:blue",
    )
    axes.set(xlabel="Mach number (-)", ylabel="Static pressure (bar)")
    figure.tight_layout()
    options.record(
        "scientific_checks",
        {
            "samples": len(records),
            "solver_rtol": 1e-10,
            "solver_atol": 1e-10,
            "coordinate": "log(p/p0)",
            "residual_scale": "cp*T0",
            "max_absolute_energy_residual_J_per_kg": max(
                abs(row["energy_residual_J_per_kg"]) for row in records
            ),
            "max_relative_pressure_error": max(
                row["relative_pressure_error"] for row in records
            ),
            "states": records,
        },
    )
    options.record("elapsed_values_are_benchmarks", False)
    options.finish()
    return records


if __name__ == "__main__":
    main()
