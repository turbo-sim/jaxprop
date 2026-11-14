"""Separate fresh source/curve drawing from cached phase-diagram updates.

Wall time includes synchronous figure drawing. This is a plotting application
study, not a property-kernel or old/new package performance comparison.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
import time
import numpy as np
import jaxprop as jp


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    N = options.points(100)
    repeats = 2 if options.profile == "quick" else 5
    source = jp.Fluid("CO2", method="coolprop")
    start = time.perf_counter()
    figure, axes = jp.plot_phase_diagram(
        source,
        "entropy",
        "temperature",
        N=N,
        plot_quality_isolines=True,
        plot_pseudocritical_line=True,
    )
    figure.canvas.draw()
    cold = time.perf_counter() - start
    artist_count = len(axes.lines) + len(axes.collections)
    cached, fresh = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        jp.plot_phase_diagram(
            source,
            "entropy",
            "temperature",
            axes=axes,
            N=N,
            plot_quality_isolines=True,
            plot_pseudocritical_line=True,
        )
        figure.canvas.draw()
        cached.append(time.perf_counter() - start)
        assert len(axes.lines) + len(axes.collections) == artist_count
        start = time.perf_counter()
        other = jp.Fluid("CO2", method="coolprop")
        other_figure, _ = jp.plot_phase_diagram(
            other,
            "entropy",
            "temperature",
            N=N,
            plot_quality_isolines=True,
            plot_pseudocritical_line=True,
        )
        other_figure.canvas.draw()
        fresh.append(time.perf_counter() - start)
    assert np.all(np.isfinite([cold, *cached, *fresh]))
    assert source.sat_liq.valid.all() and source.sat_vap.valid.all()
    options.record(
        "settings",
        dict(
            N=N,
            repetitions=repeats,
            units="SI axes",
            timed_scope="source/setup/curve/figure drawing",
        ),
    )
    options.record(
        "timing_illustration",
        dict(
            first_draw_seconds=cold,
            cached_update_seconds=cached,
            fresh_source_draw_seconds=fresh,
        ),
    )
    print(f"First draw: {cold:.6f} s; cached: {cached}; fresh: {fresh}")
    return options.finish()


if __name__ == "__main__":
    main()
