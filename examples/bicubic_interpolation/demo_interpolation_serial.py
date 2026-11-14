"""Serial full-state calls with explicit status, warmup and synchronization.

This timing demonstration uses changing physical inputs and complete states.
Table phase-boundary gaps are reported, not extrapolated. Perfect gas is a
different physical model, so its values are not an EOS-accuracy comparison.
These exploratory timings do not replace the isolated Phase 13 release gate.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options

import time
import jax
import numpy as np
from scipy.stats import qmc
import jaxprop as jp


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    full = options.profile == "full"
    table_options = dict(
        h_min=200e3,
        h_max=600e3,
        p_min=2e6,
        p_max=20e6,
        N_h=30 if full else 12,
        N_p=30 if full else 12,
        table_dir=str(options.table_dir),
    )
    models = {
        "bicubic": jp.Fluid("CO2", method="bicubic", table_options=table_options),
        "coolprop_jax": jp.Fluid("CO2", method="coolprop_jax"),
        "perfect_gas": jp.Fluid(
            "CO2", method="perfect_gas", model_options=dict(p_ref=20e6, T_ref=600)
        ),
    }
    count = 50 if full else 12
    unit = qmc.LatinHypercube(d=2, seed=42).random(count)
    points = qmc.scale(unit, [200e3, 2e6], [600e3, 20e6])
    enthalpy, pressure = points[:, 0], points[:, 1]
    first_calls = {}
    values = {}
    times = {}
    for name, model in models.items():
        started = time.perf_counter()
        warm = jax.block_until_ready(
            model.get_state(
                jp.HmassP_INPUTS, float(enthalpy[0]), float(pressure[0]), errors="mask"
            )
        )
        first_calls[name] = time.perf_counter() - started
        # The complete return, including numerical status, is synchronized.
        # Repeated samples below preserve all fields rather than projecting T.
        samples = []
        for _ in range(5 if full else 3):
            started = time.perf_counter()
            states = [
                jax.block_until_ready(
                    model.get_state(jp.HmassP_INPUTS, float(h), float(p), errors="mask")
                )
                for h, p in zip(enthalpy, pressure)
            ]
            samples.append(time.perf_counter() - started)
        values[name] = states
        times[name] = samples
    coverage = {}
    for name, states in values.items():
        rows = states if isinstance(states, list) else [states]
        successful = 0
        for row in rows:
            mask = np.asarray(row.valid)
            successful += int(np.count_nonzero(mask))
            for prop in (
                "temperature",
                "density",
                "enthalpy",
                "pressure",
                "entropy",
                "internal_energy",
            ):
                assert np.all(np.isfinite(np.asarray(row.raw_property(prop))[mask]))
        coverage[name] = dict(
            requested=count, successful=successful, failed=count - successful
        )
    assert all(row["successful"] > 0 for row in coverage.values())
    print("Complete-state synchronized exploratory timings; first calls are separate.")
    for name, samples in times.items():
        print(
            f"{name:14}: first={first_calls[name]*1e3:.3f} ms; median repeated={np.median(samples)*1e3:.3f} ms; valid={coverage[name]['successful']}/{count}"
        )
    options.record(
        "settings",
        dict(
            table=table_options,
            samples=count,
            repetitions=5 if full else 3,
            physical_models="source EOS and perfect gas are distinct",
        ),
    )
    options.record("coverage", coverage)
    options.record(
        "timing",
        dict(
            first_call_s=first_calls,
            raw_repeated_samples_s=times,
            scope="Exploratory complete-state calls; not the release performance gate",
            synchronized=True,
        ),
    )
    return options.finish()


if __name__ == "__main__":
    main()
