"""Plot every stored table node, response value and excluded interpolation cell.

Provide --input PATH to a current-schema table. This is an offline inspection
study: negative values remain visible, optional NaNs are gray, and excluded
interpolation cells are marked separately. It does not infer a missing legacy
Water table or silently reinterpret a historical nested schema.

--source-curves optionally adds the matching source-EOS phase diagram. That
requires the original native/reference convention to be verifiable locally.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
from examples.bicubic_interpolation.export_pkl_to_c_binary import load_table

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np


def configure_parser(parser):
    parser.add_argument(
        "--source-curves",
        action="store_true",
        help="Overlay a verified matching source-EOS phase diagram.",
    )


def main(argv=None):
    options = example_options(
        __file__, argv, description=__doc__, configure_parser=configure_parser
    )
    if options.input_path is None:
        raise ValueError(
            "This plotting study requires --input PATH to a current-schema table."
        )
    table = load_table(options.input_path)
    h = np.asarray(table["enthalpy"]["value"])
    p = np.asarray(table["pressure"]["value"])
    bulk = np.asarray(table["isothermal_bulk_modulus"]["value"])
    finite = np.isfinite(bulk)
    scale = max(float(np.max(np.abs(bulk[finite]))) if np.any(finite) else 1.0, 1.0)
    norm = SymLogNorm(
        linthresh=max(scale * 1e-4, 1.0), vmin=-scale, vmax=scale, base=10
    )
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("0.6")
    fig, ax = plt.subplots(figsize=(9, 6))
    if options.args.source_curves:
        import jaxprop as jp
        from jaxprop.bicubic.cache_metadata import source_signature

        metadata = table["metadata"]
        if (
            source_signature(
                metadata["fluid"], metadata["backend"], metadata["mixture_ratio"]
            )
            != metadata["source_signature"]
        ):
            raise ValueError(
                "Current native source/reference convention does not match this table's provenance."
            )
        if metadata["mixture_ratio"] is not None:
            raise ValueError(
                "A binary parcel table has no own source saturation envelope."
            )
        source = jp.Fluid(metadata["fluid"], backend=metadata["backend"])
        jp.plot_phase_diagram(
            source,
            axes=ax,
            x_prop="enthalpy",
            y_prop="pressure",
            y_scale="log",
            N=50 if options.profile == "full" else 16,
        )
        # Source axes use J/kg; the table below does too, so overlays share units.
    scatter = ax.scatter(
        h.ravel(),
        p.ravel(),
        c=bulk.ravel(),
        norm=norm,
        cmap=cmap,
        s=14,
        plotnonfinite=True,
        label="Stored nodes",
    )
    valid = np.asarray(table["phase_cell_validity"])
    h_center = 0.5 * (h[:-1, :-1] + h[1:, 1:])
    p_center = np.sqrt(p[:-1, :-1] * p[1:, 1:])
    if np.any(~valid):
        ax.scatter(
            h_center[~valid],
            p_center[~valid],
            marker="x",
            s=20,
            c="black",
            label="Excluded interpolation cells",
        )
    saturation = table["saturation_props"]
    if saturation:
        ax.plot(
            np.asarray(saturation["enthalpy"]["value"]),
            np.asarray(saturation["pressure"]["value"]),
            "k--",
            label="Stored saturation endpoint",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Enthalpy [J/kg]")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_title(f"Stored isothermal bulk modulus: {h.size} nodes")
    colorbar = fig.colorbar(scatter, ax=ax, label=r"$K_T$ [Pa]")
    # Explicit signed ticks keep labels around the linear zero region from
    # colliding when the full modulus range spans many decades.
    ticks = [-scale, -0.01 * scale, 0.0, 0.01 * scale, scale]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(["0" if value == 0 else f"{value:.1e}" for value in ticks])
    ax.legend()
    fig.tight_layout()
    assert h.shape == p.shape == bulk.shape
    assert len(scatter.get_offsets()) == h.size
    options.record(
        "inspection",
        dict(
            node_shape=list(h.shape),
            all_nodes_plotted=int(h.size),
            finite_bulk_values=int(np.count_nonzero(finite)),
            negative_bulk_values=int(np.count_nonzero(bulk < 0)),
            optional_missing_values=int(np.count_nonzero(~finite)),
            excluded_interpolation_cells=int(np.count_nonzero(~valid)),
            source_curves=bool(options.args.source_curves),
            table_mode=table["metadata"]["table_mode"],
            claim="Table-data inspection only; no whole-domain EOS/interpolation accuracy certification.",
        ),
    )
    return options.finish()


if __name__ == "__main__":
    main()
