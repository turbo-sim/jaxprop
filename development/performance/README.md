# Native JAX performance checkpoint

Measured on 13 November 2026 against the unchanged published JaxProp 0.5.8 wheel, before Phase 4. This compact checkpoint retains that result; disposable intermediate workers, extracted packages, profiles, lowered compiler output, and exploratory scripts have been removed. The complete release performance gate remains Phase 13 of the [implementation plan](../../z_IMPLEMENTATION_PLAN.md).

A later compact [Phase 4 guard](phase4_validation.json) compares the shared-validation implementation with committed native-JAX snapshot `a9e2c92`. Four independent workers per version ran serially in ABBAABBA order, with nine interleaved samples per workload and synchronized CPU/x64 outputs. Small independent HEOS tables and sampled temperature checks keep this a focused interface regression check. Temporary workers/source/tables were removed; the formal release comparison remains Phase 13.

The longer follow-up did not reproduce the initial short guard's perfect-gas slowdown. Full-state scalar medians were 41.2 versus 42.3 microseconds (+2.5%); 1,024-state batches were 142.9 versus 145.8 microseconds (+2.0%), within the observed worker variation. Bicubic scalar/batch medians improved from 206.2/1,424.7 to 135.4/823.4 microseconds; current full states intentionally exclude unsupported placeholders. Short compiled temperature/derivative timings remain noisy, with no repeatable regression. The normalized optimized temperature-projection instruction structure is unchanged (10 perfect-gas instructions and 37 bicubic instructions), without callbacks. Separate current public/kernel measurements suggest approximately 6 microseconds of perfect-gas and 10 microseconds of bicubic scalar request/binding overhead. These results support no material repeatable slowdown for the tested warmed workloads; they do not establish all devices, construction/loading costs, or application performance.

## Method and scope

Six separate workers ran serially in baseline/current/current/baseline/baseline/current order. Each implementation has three workers and five samples per warmed workload, with multiple calls per sample and synchronized JAX outputs. Cold table construction has one sample per worker. Import paths, source hashes, dependencies, device/precision, and thread settings were verified.

Both versions used Python 3.13.2, JAX/jaxlib 0.6.2, CoolProp 7.2.0, NumPy 2.4.4, Equinox 0.12.2, and SciPy 1.17.1 on the same Windows CPU with double precision. Tables were independent 16-by-16 equilibrium CO2 tables over 600–850 kJ/kg and 5–12 MPa, using central differences. This scope does not establish full ThermOpt optimization, REFPROP performance, GPU behavior, or all phase/table modes.

The [portable summary](summary.json) retains all 37 comparable case medians, interquartile ranges, paired ratios, environment details, distribution/source hashes, and two new-only API measurements. It is a results record; the formal Phase 13 benchmark should use a maintained workload harness and resolve its own published baseline when that phase starts.

## Representative results

Warmed medians are microseconds per complete public call or per indicated batch.

| Workload | PyPI 0.5.8 | Optimized | New / old |
| --- | ---: | ---: | ---: |
| Perfect-gas scalar full state | 156.6 | 38.1 | 0.244 |
| Perfect-gas full-state batch, 1,024 | 273.2 | 72.7 | 0.266 |
| Bicubic scalar full state | 365.6 | 157.9 | 0.432 |
| Bicubic full-state batch, 1,024 | 1,297.1 | 1,280.3 | 0.987 |
| Bicubic three-state compressor, eager | 3,497.0 | 3,740.8 | 1.070 |
| Bicubic three-state compressor, compiled | 98.1 | 95.9 | 0.978 |

Perfect-gas scalar/batch and bicubic scalar improvements repeat in all three paired comparisons. Bicubic batch and compiled property/derivative projections are broadly comparable; short operations and solver workloads remain variable. Eager NumPy property availability checks retain a small absolute cost of about 2.4 microseconds versus the published unguarded field access near 0.05 microseconds.

Changing Python enthalpy values improve from 28.82 ms to 79.4 microseconds for perfect gas and 106.77 ms to 278.4 microseconds for bicubic. These figures include avoided per-value compilation and must not be presented as arithmetic-kernel speedups. Trace-count tests independently verify compilation reuse.

Cached loading remains slower: 31.6 versus 22.6 ms (+40%), retaining source/manifest/payload validation. Fresh small-table construction is approximately comparable at 2.20 versus 2.26 seconds. Lean runtime/loading improvements remain in Phase 8; validation is not disabled to improve timings.

The two new-only measurements are complete JAX perfect-gas state export at 157.6 microseconds and explicit requested validity at 55.0 microseconds. Published 0.5.8 has no equivalent API, so no old/new ratio is assigned.

## Correctness and validation

All 42 sampled common value/shape groups agree at relative/absolute tolerance $10^{-7}$, and the two sampled temperature derivatives match exactly across workers. The final full HEOS + REFPROP suite reports **936 passed, 78 strict expected failures, zero unexpected failures and zero skips**, retaining all **84,950 numerical comparisons with zero failures**: 50,189 finite values and 34,761 explicitly unavailable values. [Full verification](../test_audit/results/jax_efficiency/full_20261113_094055/summary.json).

The [efficiency audit](../../z_JAX_EFFICIENCY_AUDIT.md) records the implemented fixes, execution semantics, regression tests, and larger optimization ideas. The measured source is the optimized working tree based on commit `61728b0`; exact source and distribution hashes remain in the portable summary.
