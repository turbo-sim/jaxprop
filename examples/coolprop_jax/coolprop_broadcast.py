"""Synchronized complete-state broadcasting: first-call and warmed costs.

The two physical models intentionally differ. This illustration is not an
old/new performance gate; Phase 13 uses matched model/version workloads.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options
import time
import statistics
import jax
import jax.numpy as jnp
import numpy as np
import jaxprop as jp


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    p0, T0 = 101325.0, 300.0
    models = {
        "perfect_gas": jp.Fluid(
            "Air", method="perfect_gas", model_options=dict(T_ref=T0, p_ref=p0)
        ),
        "source_callback": jp.Fluid("Air", method="coolprop_jax"),
    }
    sizes = [10, 100] if options.profile == "quick" else [10, 100, 1000]
    repeats = 2 if options.profile == "quick" else 3
    results = []
    for shape in ("array_array", "array_scalar"):
        for n in sizes:
            kP, kT = jax.random.split(jax.random.key(0))
            pressures = p0 * (1 + 0.1 * jax.random.normal(kP, (n,)))
            temperatures = (
                T0 * (1 + 0.1 * jax.random.normal(kT, (n,)))
                if shape == "array_array"
                else jnp.asarray(T0)
            )
            for name, fluid in models.items():
                evaluate = jax.jit(
                    lambda p, T: fluid.get_state(jp.PT_INPUTS, p, T, errors="mask")
                )
                start = time.perf_counter()
                state = jax.block_until_ready(evaluate(pressures, temperatures))
                first_seconds = time.perf_counter() - start
                assert state.shape == (n,) and np.all(np.asarray(state.status) == 0)
                if name == "perfect_gas":
                    np.testing.assert_allclose(
                        state.rho,
                        pressures / (fluid.constants.R * temperatures),
                        rtol=1e-12,
                    )
                samples = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    state = jax.block_until_ready(evaluate(pressures, temperatures))
                    samples.append(time.perf_counter() - start)
                options.check_states({f"{name}_{shape}_{n}": state})
                results.append(
                    dict(
                        model=name,
                        shape=shape,
                        size=n,
                        first_call_seconds=first_seconds,
                        warm_samples_seconds=samples,
                        warm_median_seconds=statistics.median(samples),
                        materialized_properties=len(state.available_properties),
                    )
                )
                print(
                    f"{name:16s} {shape:12s} n={n:4d}: first={first_seconds*1e3:.3f} ms, warm={statistics.median(samples)*1e3:.3f} ms"
                )
    options.record(
        "settings",
        dict(
            sizes=sizes,
            repeats=repeats,
            synchronization="complete returned state",
            comparable_physical_models=False,
        ),
    )
    options.record("timing_illustration", results)
    return options.finish()


if __name__ == "__main__":
    main()
