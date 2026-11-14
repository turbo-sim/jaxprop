"""Demo derivatives jax.

Use --profile quick for a smaller declared study; --profile full retains the
full scientific resolution. Outputs and table caches use explicit CLI paths.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options

import os
import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    jxp.set_plot_options()

    # ---------------------------
    # Configuration
    # ---------------------------
    outdir = str(options.output_dir)
    fluid_name = "CO2"
    h_min = 500e3  # J/kg
    h_max = 1000e3  # J/kg
    p_min = 2e6  # Pa
    p_max = 20e6  # Pa
    N_h = 32
    N_p = 32
    Nh_chk = Np_chk = 50 if options.profile == "full" else 8

    # ------------------------------------------
    # Fluid objects
    # ------------------------------------------
    fluid_coolprop = jxp.Fluid(fluid_name, method="coolprop_jax", backend="HEOS")
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
            "table_name": "derivative_check",
        },
    )

    # ------------------------------------------
    # Grid
    # ------------------------------------------
    # Thermodynamic derivative identities apply inside the interpolation
    # domain. At a hard clipping boundary JAX differentiates the projection,
    # whose endpoint slope is not the one-sided source EOS derivative.
    # Retain 50-by-50 full checks, now at interior validation-cell midpoints.
    h_edges = jnp.linspace(h_min, h_max, Nh_chk + 1)
    p_edges = jnp.linspace(p_min, p_max, Np_chk + 1)
    h_vals = 0.5 * (h_edges[:-1] + h_edges[1:])
    p_vals = 0.5 * (p_edges[:-1] + p_edges[1:])
    h_grid, p_grid = jnp.meshgrid(h_vals, p_vals, indexing="ij")

    # Base state on (H,P)
    states_grid = fluid_bicubic.get_state(jxp.HmassP_INPUTS, h_grid, p_grid)
    rho_grid = states_grid["density"]
    s_grid = states_grid["entropy"]

    # ------------------------------------------
    # 1. Check dT/dh|p vs 1/cp
    # ------------------------------------------
    def T_of_h_scalar(h, p):
        return fluid_bicubic.get_state(jxp.HmassP_INPUTS, h, p, errors="mask")[
            "temperature"
        ]

    def dTdh_scalar(h, p):
        return jax.grad(lambda hh: T_of_h_scalar(hh, p))(h)

    dTdh_flat = jax.vmap(dTdh_scalar)(h_grid.ravel(), p_grid.ravel())
    cp_jax = 1 / dTdh_flat.reshape(h_grid.shape)
    cp_table = states_grid["isobaric_heat_capacity"]

    rel_err_cp = (cp_jax - cp_table) / cp_table * 100.0

    print("\n[Heat capacity check]")
    print(
        f"  min = {np.nanmin(rel_err_cp):.3e} %, max = {np.nanmax(rel_err_cp):.3e} %, mean = {np.nanmean(rel_err_cp):.3e} %"
    )

    tol_cp = 1e-2  # tolerance in %
    assert np.all(
        np.isfinite(rel_err_cp)
    ), "Every requested cp derivative must be defined"
    max_err_cp = np.max(np.abs(rel_err_cp))
    if max_err_cp > tol_cp:
        raise ValueError(
            f"[Heat capacity check failed] max relative error = {max_err_cp:.3e} % > {tol_cp:.3e} %"
        )

    # ------------------------------------------
    # 2. Check a**2 = (∂p/∂ρ)_s
    # ------------------------------------------
    def p_of_rho_s_scalar(rho, s):
        return fluid_bicubic.get_state(jxp.DmassSmass_INPUTS, rho, s, errors="mask")[
            "pressure"
        ]

    def a_from_derivative_scalar(rho, s):
        dp_drho = jax.grad(lambda rr: p_of_rho_s_scalar(rr, s))(rho)
        return dp_drho

    a_deriv_flat = jax.vmap(a_from_derivative_scalar)(rho_grid.ravel(), s_grid.ravel())
    dp_drho = a_deriv_flat.reshape(rho_grid.shape)
    assert np.all(np.isfinite(dp_drho)) and np.all(dp_drho > 0)
    a_jax = jnp.sqrt(dp_drho)
    a_table = states_grid["speed_sound"]

    rel_err_a = (a_jax - a_table) / jnp.maximum(a_table, 1e-16) * 100.0

    print("\n[Speed of sound check]")
    print(
        f"  min = {np.nanmin(rel_err_a):.3e} %, max = {np.nanmax(rel_err_a):.3e} %, mean = {np.nanmean(rel_err_a):.3e} %"
    )

    tol_a = 1e-2  # tolerance in %
    assert np.all(np.isfinite(rel_err_a)), "Every acoustic derivative must be defined"
    max_err_a = np.max(np.abs(rel_err_a))
    if max_err_a > tol_a:
        raise ValueError(
            f"[Speed of sound check failed] max relative error = {max_err_a:.3e} % > {tol_a:.3e} %"
        )

    # ------------------------------------------
    # Plot derivative consistency errors
    # ------------------------------------------

    # Data to plot
    error_fields = [
        np.abs(rel_err_cp),
        np.abs(rel_err_a),
    ]
    titles = ["Heat capacity comparison", "Speed of sound comparison"]
    n_props = len(error_fields)
    ncols = 2
    nrows = 1

    # Levels for log contour
    levels = np.logspace(-6, 2, 9)

    fig = plt.figure(figsize=(6 * ncols, 4 * nrows))
    gs = gridspec.GridSpec(
        nrows, ncols + 1, width_ratios=[1] * ncols + [0.05], figure=fig
    )

    axes = []
    for i in range(n_props):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_yscale("log")
        axes.append(ax)

    # Reference phase diagram
    for ax, err_field, title in zip(axes, error_fields, titles):
        # Clip error to avoid issues in LogNorm
        err_field = np.clip(err_field, levels[0], levels[-1])

        jxp.plot_phase_diagram(
            fluid_coolprop,
            x_prop="enthalpy",
            y_prop="pressure",
            axes=ax,
            x_scale="linear",
            y_scale="log",
        )

        c = ax.contourf(
            h_grid,
            p_grid,
            err_field,
            levels=levels,
            norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
            cmap="viridis",
            extend="both",
        )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Enthalpy [J/kg]")
        ax.set_ylabel("Pressure [Pa]")

    # Colorbar axis on the right
    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(c, cax=cax)
    cb.set_ticks(levels)
    cb.set_label("Relative error [%] (log scale)")

    plt.tight_layout(pad=1)

    # Show figures

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
            "Nh_chk": Nh_chk,
            "Np_chk": Np_chk,
        },
    )
    options.record(
        "signed_derivative_checks",
        dict(
            maximum_absolute_relative_cp_error_percent=float(max_err_cp),
            maximum_absolute_relative_sound_error_percent=float(max_err_a),
            unchanged_tolerance_percent=float(tol_cp),
            all_points_finite_and_positive=True,
            validation_grid=[Nh_chk, Np_chk],
        ),
    )
    return options.finish()


if __name__ == "__main__":
    main()
