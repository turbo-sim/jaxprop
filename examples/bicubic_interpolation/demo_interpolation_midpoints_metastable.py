"""Compare a physical Nitrogen liquid table at cell midpoints.

The original h_max=10 kJ/kg rectangle crosses the connected-liquid spinodal
at 3 MPa (h_sp=7.250624 kJ/kg), and at lower pressures enters a nonpositive-cv
formal EOS region even earlier. This deliberately revised physical study uses
h=[-120,-40] kJ/kg at the original p=[0.1,3] MPa, retaining 40-by-40 full nodes.
Independent homogeneous probes have positive cv and K_T throughout the new
sampled rectangle; generation also validates every derivative stencil/cell.

Quick/full profiles have distinct declared grid sizes. Use explicit output and
table-cache paths; full retains the scientific resolution.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options

import os
import numpy as np
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    jxp.set_plot_options()

    # ---------------------------
    # Configuration
    # ---------------------------
    # outdir = "fluid_tables"
    # fluid_name = "water"
    # h_min = 400e3  # J/kg
    # h_max = 800e3  # J/kg
    # p_min = 2e6    # Pa
    # p_max = 20e6   # Pa
    # N_h = 80
    # N_p = 80

    # fluid_name = "nitrogen"
    # h_min = 40e3  # J/kg
    # h_max = 100e3  # J/kg
    # p_min = 1.1e5  # Pa
    # p_max = 3e6  # Pa
    # N_p = 100 # Number of pressure points
    # N_h = 100 # Number of enthalpy points
    # metastable_phase = "vapor"

    fluid_name = "nitrogen"
    h_min = -120e3  # J/kg
    h_max = -40e3  # J/kg: physical liquid branch with positive cv/K_T margin
    p_min = 1e5  # Pa
    p_max = 3e6  # Pa
    N_p = 40 if options.profile == "full" else 12  # Number of pressure points
    N_h = 40 if options.profile == "full" else 12  # Number of enthalpy points
    metastable_phase = "liquid"

    outdir = str(options.output_dir)

    # ---------------------------
    # Build models
    # ---------------------------
    # table_name="interpolation_midpoints_metastable",
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
            "gradient_method": "forward",
        },
    )

    fluid_cp = jxp.Fluid(fluid_name, method="coolprop")
    table = fluid_bicubic.table
    cv = np.asarray(table["isochoric_heat_capacity"]["value"])
    bulk = np.asarray(table["isothermal_bulk_modulus"]["value"])
    assert np.all(cv > 0) and np.all(bulk > 0)
    assert np.all(np.asarray(table["phase_cell_validity"]))

    # ---------------------------
    # Midpoint grid (evaluation points)
    # ---------------------------
    h_nodes = jnp.linspace(h_min, h_max, N_h)
    p_nodes = jnp.exp(jnp.linspace(jnp.log(p_min), jnp.log(p_max), N_p))
    h_vals = 0.5 * (h_nodes[:-1] + h_nodes[1:])
    p_vals = 0.5 * (p_nodes[:-1] + p_nodes[1:])
    H_mesh, P_mesh = np.meshgrid(h_vals, p_vals, indexing="ij")

    # ---------------------------
    # Evaluate states once (vectorized)
    # ---------------------------
    interp_grid = fluid_bicubic.get_state(jxp.HmassP_INPUTS, H_mesh, P_mesh)
    properties = ["T", "d", "e", "s", "a", "G"]
    true_grid = {prop: np.full(H_mesh.shape, np.nan) for prop in properties}
    reference_failures = []
    for index in np.ndindex(H_mesh.shape):
        # Compare the same homogeneous branch. An ordinary HP flash can instead
        # return an equilibrium mixture and is not a metastable error reference.
        try:
            reference = fluid_cp.get_state_metastable(
                "h",
                float(H_mesh[index]),
                "p",
                float(P_mesh[index]),
                rhoT_guess=[float(interp_grid.rho[index]), float(interp_grid.T[index])],
                supersaturation=False,
                generalize_quality=False,
            )
            assert reference.cv > 0 and reference["isothermal_bulk_modulus"] > 0
            for prop in properties:
                true_grid[prop][index] = float(reference.raw_property(prop))
        except jxp.JaxPropError as error:
            reference_failures.append((index, str(error)))
    print(
        "Metastable source reference calls: "
        f"{H_mesh.size - len(reference_failures)}/{H_mesh.size} succeeded"
    )
    if reference_failures:
        print("First reference failure:", reference_failures[0])
    assert (
        not reference_failures
    ), "A physical midpoint comparison needs every source reference"

    # ---------------------------
    # Collect error statistics
    # ---------------------------
    error_summary = []

    for prop in properties:
        interp_val = np.array(interp_grid.raw_property(prop))
        true_val = np.array(true_grid[prop])

        abs_err = np.abs(interp_val - true_val)
        rel_err = abs_err / np.maximum(np.abs(true_val), 1e-30) * 100.0  # %

        error_summary.append(
            {
                "property": prop,
                "abs_min": np.nanmin(abs_err),
                "abs_max": np.nanmax(abs_err),
                "rel_min": np.nanmin(rel_err),
                "rel_max": np.nanmax(rel_err),
                "rel_mean": np.nanmean(rel_err),
                "field": rel_err,
            }
        )

    # ---------------------------
    # Print error summary table
    # ---------------------------
    print("\nProperty interpolation error summary (evaluated at cell midpoints):")
    print(
        f"{'Prop':<5} {'| Abs min':>12} {'Abs max':>12} {'% min':>12} {'% max':>12} {'% mean':>12}"
    )
    print("-" * 70)
    for e in error_summary:
        print(
            f"{e['property']:<5} | {e['abs_min']:>12.3e} {e['abs_max']:>12.3e} "
            f"{e['rel_min']:>12.3e} {e['rel_max']:>12.3e} {e['rel_mean']:>12.3e}"
        )

    # ---------------------------
    # Plot subplot grid for selected properties
    # ---------------------------
    n_props = len(properties)
    ncols = 3
    nrows = int(np.ceil(n_props / ncols))

    # Use GridSpec to reserve a narrow column for the colorbar
    fig = plt.figure(figsize=(4 * ncols, 3 * nrows))
    gs = gridspec.GridSpec(
        nrows, ncols + 1, width_ratios=[1] * ncols + [0.05], figure=fig
    )

    axes = []
    for i in range(n_props):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

    # Color levels
    levels = np.logspace(-6, 2, 9)

    # Plot each property
    for ax, prop in zip(axes, properties):
        e = next(x for x in error_summary if x["property"] == prop)
        rel_err_field = np.clip(e["field"], levels[0], levels[-1])

        # Phase diagram background
        fluid_cp.plot_phase_diagram(
            x_prop="enthalpy",
            y_prop="pressure",
            axes=ax,
            x_scale="linear",
            y_scale="log",
        )

        # Contour
        c = ax.contourf(
            H_mesh,
            P_mesh,
            rel_err_field,
            levels=levels,
            norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
            cmap="viridis",
            extend="both",
        )

        ax.set_title(f"{prop} [% error]", fontsize=11)
        ax.set_xlabel("Enthalpy [J/kg]")
        ax.set_ylabel("Pressure [Pa]")

    # Colorbar axis on the right
    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(c, cax=cax)
    cb.set_ticks(levels)
    cb.set_label("Relative error [%] (log scale)")

    # Remove unused axes (if any)
    total_axes = nrows * ncols
    for i in range(len(properties), total_axes):
        r, c = divmod(i, ncols)
        fig.delaxes(fig.add_subplot(gs[r, c]))

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
            "metastable_phase": metastable_phase,
        },
    )
    options.record(
        "domain_revision",
        dict(
            original_rejected_h_max=10e3,
            recorded_spinodal_h_at_3MPa=7250.624424239421,
            physical_h_max=h_max,
            minimum_cv=float(cv.min()),
            minimum_K_T=float(bulk.min()),
            source_reference_points=int(H_mesh.size),
            source_reference_failures=len(reference_failures),
        ),
    )
    options.record(
        "error_summary",
        [{k: v for k, v in row.items() if k != "field"} for row in error_summary],
    )
    return options.finish()


if __name__ == "__main__":
    main()
