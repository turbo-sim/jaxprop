"""Differentiate a real-fluid polytropic expansion through Diffrax.

The ODE is dh/dp=eta/rho(h,p). JaxProp's configured callback model supplies the
source bridge and first-order JVP; this example does not duplicate its kernel.
Pressure is parameterized by x in [0,1], p=p_in+x*(p_out-p_in), so both endpoint
pressures remain genuine differentiable parameters rather than global values.

Save only floating h/p arrays in Diffrax. Full thermodynamic states, including
integer status and optional-property metadata, are reconstructed from those
arrays for validation/plotting. DirectAdjoint supports both forward and reverse
first derivatives; central finite differences independently check all four
inputs (h_in, p_in, p_out, efficiency). Timings are synchronized demonstrations,
not substitutes for the interleaved package performance gate.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxprop as jp

PARAMETERS = ("h_in", "p_in", "p_out", "efficiency")


def _polytropic_odefun(x, enthalpy, args):
    params, fluid = args
    delta_p = params["p_out"] - params["p_in"]
    pressure = params["p_in"] + x * delta_p
    state = fluid.get_state(jp.HmassP_INPUTS, enthalpy, pressure, errors="mask")
    return delta_p * params["efficiency"] / state.raw_property("density")


def postprocess_ode(x, enthalpy, args):
    """Save numerical coordinates; reconstruct complete states after integration."""
    params, _ = args
    pressure = params["p_in"] + x * (params["p_out"] - params["p_in"])
    return dict(pressure=pressure, enthalpy=enthalpy)


@eqx.filter_jit
def polytropic_expansion(params, fluid, num_steps=None, *, rtol=1e-6, atol=1e-9):
    """Return a Diffrax Solution with h/p series for a polytropic expansion.

    Parameters are a dictionary of scalar numerical arrays named in PARAMETERS.
    All four remain dynamic. ``num_steps`` selects saved samples, not integration
    accuracy. The original Dopri5/PID tolerances (rtol=1e-6, atol=1e-9), adjoint
    and maximum step budget are retained. Explicit tighter tolerances support
    independent convergence studies without changing the example defaults.
    """
    if num_steps is None:
        saveat = dfx.SaveAt(t1=True, fn=postprocess_ode)
    else:
        saveat = dfx.SaveAt(ts=jnp.linspace(0.0, 1.0, num_steps), fn=postprocess_ode)
    return dfx.diffeqsolve(
        dfx.ODETerm(_polytropic_odefun),
        dfx.Dopri5(),
        t0=0.0,
        t1=1.0,
        dt0=None,
        y0=params["h_in"],
        args=(params, fluid),
        saveat=saveat,
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
        adjoint=dfx.DirectAdjoint(),
        max_steps=10_000,
    )


def get_exit_temperature(params, fluid, *, rtol=1e-6, atol=1e-9):
    """Return the exit temperature without relying on global pressure values."""
    solution = polytropic_expansion(params, fluid, rtol=rtol, atol=atol)
    state = fluid.get_state(
        jp.HmassP_INPUTS,
        solution.ys["enthalpy"][-1],
        solution.ys["pressure"][-1],
        errors="mask",
    )
    return state.raw_property("temperature")


def get_exit_temperature_gradient(params, fluid, rel_eps=1e-4):
    """Central finite differences in each physical parameter's own units."""
    gradient = {}
    for name in PARAMETERS:
        step = rel_eps * jnp.maximum(jnp.abs(params[name]), 1.0)
        plus, minus = dict(params), dict(params)
        plus[name], minus[name] = params[name] + step, params[name] - step
        gradient[name] = (
            get_exit_temperature(plus, fluid) - get_exit_temperature(minus, fluid)
        ) / (2 * step)
    return gradient


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    import time
    import numpy as np
    import matplotlib.pyplot as plt

    jp.set_plot_options(grid=False)
    # Original real-fluid conditions, ten efficiencies and 50 saved samples.
    T_in, p_in, p_out, efficiency = 400.0, 200e5, 50e5, 0.8
    fluid = jp.Fluid("CO2", method="coolprop_jax", backend="HEOS")
    inlet = fluid.get_state(jp.PT_INPUTS, p_in, T_in)
    params = {
        name: jnp.asarray(value)
        for name, value in dict(
            h_in=inlet.h, p_in=p_in, p_out=p_out, efficiency=efficiency
        ).items()
    }
    efficiencies = np.linspace(1.0, 0.1, 10 if options.profile == "full" else 3)
    num_steps = 50 if options.profile == "full" else 15
    forward = eqx.filter_jit(jax.jacfwd(get_exit_temperature))
    reverse = eqx.filter_jit(jax.jacrev(get_exit_temperature))
    # Separate first compilation from subsequent synchronized evaluations.
    initial = jax.block_until_ready(polytropic_expansion(params, fluid, num_steps))
    jax.block_until_ready(forward(params, fluid))
    jax.block_until_ready(reverse(params, fluid))
    fig, axis = plt.subplots(figsize=(6, 5))
    maximum_ad_error, maximum_fd_error = 0.0, 0.0
    rows = []
    for eta, colour in zip(
        efficiencies, plt.cm.magma(np.linspace(0.2, 0.8, len(efficiencies)))
    ):
        current = dict(params, efficiency=jnp.asarray(eta))
        started = time.perf_counter()
        solution = jax.block_until_ready(
            polytropic_expansion(current, fluid, num_steps)
        )
        trajectory_ms = (time.perf_counter() - started) * 1e3
        states = fluid.get_state(
            jp.HmassP_INPUTS, solution.ys["enthalpy"], solution.ys["pressure"]
        )
        assert np.all(np.asarray(states.valid))
        for name in (
            "pressure",
            "temperature",
            "density",
            "enthalpy",
            "entropy",
            "internal_energy",
        ):
            assert np.all(np.isfinite(states.raw_property(name))), name
        assert np.all(np.diff(np.asarray(states.h)) <= 1e-7)
        # The isentropic eta=1 limit is checked against an independent source PS flash.
        if eta == 1.0:
            reference = fluid.get_state(jp.PSmass_INPUTS, p_out, inlet.s)
            np.testing.assert_allclose(states.T[-1], reference.T, rtol=5e-6)
        values = []
        for function in (forward, reverse, get_exit_temperature_gradient):
            started = time.perf_counter()
            gradient = jax.block_until_ready(function(current, fluid))
            elapsed_ms = (time.perf_counter() - started) * 1e3
            values.append(
                (np.array([float(gradient[name]) for name in PARAMETERS]), elapsed_ms)
            )
        fwd, rev, finite = [item[0] for item in values]
        assert np.all(np.isfinite(np.concatenate((fwd, rev, finite))))
        np.testing.assert_allclose(fwd, rev, rtol=2e-8, atol=1e-10)
        ad_error = float(np.linalg.norm(fwd - rev) / np.linalg.norm(finite))
        fd_norm_error = float(np.linalg.norm(fwd - finite) / np.linalg.norm(finite))
        component_errors = np.abs((fwd - finite) / finite)
        fd_error = float(np.max(component_errors))
        # Source callbacks use first-order finite differences and the ODE has
        # adaptive rtol=1e-6. Check EACH physical parameter at 0.2%: an unscaled
        # mixed-unit norm would hide small pressure derivatives behind eta's
        # much larger dimensional sensitivity. Keep that norm as a diagnostic.
        np.testing.assert_allclose(fwd, finite, rtol=2e-3, atol=1e-12)
        maximum_ad_error, maximum_fd_error = max(maximum_ad_error, ad_error), max(
            maximum_fd_error, fd_error
        )
        axis.plot(
            np.asarray(states.s),
            np.asarray(states.T),
            color=colour,
            label=rf"$\eta_p={eta:.2f}$",
        )
        row = dict(
            efficiency=float(eta),
            exit_temperature_K=float(states.T[-1]),
            trajectory_ms=trajectory_ms,
            forward_ms=values[0][1],
            reverse_ms=values[1][1],
            finite_difference_ms=values[2][1],
            forward_reverse_relative_error=ad_error,
            AD_central_difference_relative_error=fd_error,
            AD_central_difference_relative_norm_error=fd_norm_error,
            parameter_relative_errors=dict(zip(PARAMETERS, component_errors.tolist())),
        )
        rows.append(row)
        print(
            f"eta={eta:.2f}, T_out={row['exit_temperature_K']:.8f} K, AD/FD error={fd_error:.3e}"
        )
    axis.set_xlabel("Entropy [J/(kg K)]")
    axis.set_ylabel("Temperature [K]")
    axis.legend(fontsize=7)
    fig.tight_layout(pad=1)
    options.record(
        "settings",
        dict(
            fluid="CO2",
            backend="HEOS",
            T_in=T_in,
            p_in=p_in,
            p_out=p_out,
            efficiencies=efficiencies,
            saved_points=num_steps,
            ODE_rtol=1e-6,
            ODE_atol=1e-9,
            max_steps=10000,
            finite_difference_relative_step=1e-4,
            gradient_component_relative_budget=2e-3,
        ),
    )
    options.record("maximum_forward_reverse_relative_error", maximum_ad_error)
    options.record("maximum_AD_central_difference_relative_error", maximum_fd_error)
    options.record("synchronized_demonstration_observations", rows)
    options.record(
        "timing_scope",
        "Synchronized single observations after warmup; formal repeat/interleave benchmark is Phase 13",
    )
    options.finish()


if __name__ == "__main__":
    main()
