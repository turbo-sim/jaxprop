# Native JAX performance checkpoint

The latest [perfect-gas repair report](pg_hotpath/README.md) supersedes the older PG timing conclusions below for the current 22-property model. Eager full-state calls are faster than PyPI 0.5.8, while compiled full batches remain 31–38% slower. The user accepted this tradeoff on 13 November 2026, completing Phase 7 acceptance and allowing Phase 8 to proceed. Broader Phase 13 comparisons remain pending. Historical checkpoints are retained as measurements of their own source versions.

Measured on 13 November 2026 against the unchanged published JaxProp 0.5.8 wheel, before Phase 4. This compact checkpoint retains that result; disposable intermediate workers, extracted packages, profiles, lowered compiler output, and exploratory scripts have been removed. The complete release performance gate remains Phase 13 of the [implementation plan](../../z_IMPLEMENTATION_PLAN.md).

A later compact [Phase 4 guard](phase4_validation.json) compares the shared-validation implementation with committed native-JAX snapshot `a9e2c92`. Four independent workers per version ran serially in ABBAABBA order, with nine interleaved samples per workload and synchronized CPU/x64 outputs. Small independent HEOS tables and sampled temperature checks keep this a focused interface regression check. Temporary workers/source/tables were removed; the formal release comparison remains Phase 13.

The longer follow-up did not reproduce the initial short guard's perfect-gas slowdown. Full-state scalar medians were 41.2 versus 42.3 microseconds (+2.5%); 1,024-state batches were 142.9 versus 145.8 microseconds (+2.0%), within the observed worker variation. Bicubic scalar/batch medians improved from 206.2/1,424.7 to 135.4/823.4 microseconds; current full states intentionally exclude unsupported placeholders. Short compiled temperature/derivative timings remain noisy, with no repeatable regression. The normalized optimized temperature-projection instruction structure is unchanged (10 perfect-gas instructions and 37 bicubic instructions), without callbacks. Separate current public/kernel measurements suggest approximately 6 microseconds of perfect-gas and 10 microseconds of bicubic scalar request/binding overhead. These results support no material repeatable slowdown for the tested warmed workloads; they do not establish all devices, construction/loading costs, or application performance.

The [Phase 5 guard](phase5_validation.json) compares shared source-reference initialization with Phase 4 commit `49aea3d`, using the same four-worker/ABBAABBA/nine-sample native-call method. Supported scalar outputs and temperature derivatives match, and normalized optimized temperature-projection instructions are unchanged. Perfect-gas scalar/batch medians are 36.0/109.6 versus 34.3/111.6 microseconds; bicubic scalar/batch medians are 118.5/793.1 versus 117.3/808.8 microseconds. Batch differences near +2% are within observed worker variation; there is no material repeatable regression in these workloads. Compare versions within each matched run, rather than treating absolute times from separate checkpoints as speedups.

Separate warmed setup samples use five samples per worker after two warm calls. Direct-source construction improves from 4.72 to 0.97 milliseconds and perfect-gas construction from 6.68 to 3.12 milliseconds, consistently across paired workers. The dedicated raw source-reference path avoids routing setup through repeated ordinary full-state calls and their cloned helper states. Bicubic cached-loading medians are 25.73 versus 22.08 milliseconds, with one slower paired worker; the record preserves that variability. Complete cache validation remains enabled. These setup measurements exclude imports, first JAX initialization and fresh table generation. Temporary source copies, workers and tables were removed; only the compact record remains. Full application/device and release comparisons still belong to Phase 13.

## Common-constructor check

The [constructor guard](constructor_validation.json) compares direct native construction at commit `32e18c9` with the common `Fluid(method=...)` initializer. The returned objects remain their native classes; scalar, bicubic batch and compiled projection times were broadly comparable. Perfect-gas full batches initially measured 89.4 versus 98.6 microseconds (+10.4%), so that result was investigated rather than dismissed.

The [longer perfect-gas follow-up](constructor_batch_followup.json) uses four independent workers per version in ABBAABBA order, 24 samples per case and 300 synchronized calls per sample, balancing all six public/private/direct-control order permutations. Public batch medians are 98.3 versus 99.0 microseconds (+0.7%); private compiled medians are 87.8 versus 88.6 microseconds (+0.9%). Interquartile ranges and paired worker ratios show substantial variability. The earlier 10% shift did not repeat; both records remain visible.

All nine model leaves have the same values, dtypes and weak typing across versions. All 13 full-batch floating fields and integer statuses match bitwise, and the normalized full-batch optimized program is identical across eight workers (189 instructions, no callbacks). Same-process public/direct objects also have equal native types, actual inheritance, PyTrees and outputs. These checks found no constructor-specific numerical/runtime change or material repeatable CPU slowdown; they do not establish all workloads/devices. Initial construction costs are reported separately in the first record. Temporary source/worker/table copies were removed; the formal performance gate remains Phase 13.

## Phase 6 source/callback contract check

The [Phase 6 guard](phase6_validation.json) compares the completed source/callback policies with constructor commit `f03443a`. Four independent workers per version ran **serially** in ABBAABBA order after all correctness tests finished, with nine interleaved samples per warmed case, synchronized outputs, one-thread BLAS settings and independent 8-by-8 HEOS CO2 table caches. Setup is measured separately. Current compiled callback mask mode is compared with the previous valid-state default; optional diagnostics have no supported old equivalent and receive no old/new ratio.

| Warmed complete call | Before | Phase 6 |
| --- | ---: | ---: |
| Perfect-gas scalar | 36.3 us | 35.9 us |
| Perfect-gas batch, 1,024 | 92.1 us | 81.6 us |
| Bicubic scalar | 113.6 us | 112.4 us |
| Bicubic batch, 1,024 | 743.0 us | 733.9 us |
| Direct source scalar | 573.0 us | 399.9 us |
| Direct source batch, 32 | 8.876 ms | 3.931 ms |
| Eager callback scalar | 1.139 ms | 0.244 ms |
| Eager callback batch, 32 | 22.085 ms | 3.681 ms |
| Compiled callback full batch, 32 | 22.044 ms | 4.006 ms |
| Compiled callback temperature derivative | 1.910 ms | 0.674 ms |

No material repeatable warmed slowdown appears in these workloads. Native temperature, derivative and full-batch optimized instruction programs have equal normalized hashes across all eight workers, with no callbacks: Phase 6 adds no native numerical work. Do not interpret the lower perfect-gas batch median as a kernel optimization; its program is unchanged and worker variation remains visible in the JSON. Native scalar/projection/bicubic differences are small. The source/callback improvements repeat across paired workers: constructor-prepared helper state and critical entropy remove repeated source initialization, eager callbacks use one compiled state assembly, and derivative callbacks skip inactive input perturbations. These are complete public-call timings, not raw EOS arithmetic timings.

The cost moves to setup: direct source construction is **0.902 versus 1.291 ms** (+0.389 ms), and callback construction **1.097 versus 1.495 ms** (+0.398 ms), preparing and verifying a reference-compatible auxiliary state once. Perfect-gas construction is 2.860 versus 3.028 ms with mixed paired-worker variation; cached loading is 20.959 versus 20.620 ms, broadly comparable. No required validation is disabled.

All **20,896 sampled scalar/status comparisons** match, including sampled common property fields and derivatives; unavailable fields match explicitly. Worker environments and per-version source hashes agree, and the current package matches the measured source hash. The final full HEOS/REFPROP suite passes **1,618 tests with 28 strict expected failures**, retaining all **54,272 numerical comparisons**, including **50,189 finite values**, with zero failures. [Correctness run](../test_audit/results/phase6/full_20261113_173309/summary.json). Temporary source copies, workers and tables were removed after preserving this record. This focused CPU/x64 guard does not complete the Phase 13 published-release/device/application gate.

## Phase 7 perfect-gas equations and numerical policies

The initial Phase 7 guard compared the 22-property model with Phase 6 commit
`385c977`, including unchanged bicubic controls. It identified PG batch
regressions and compiler duplication; a targeted status-output experiment
confirmed that numerical validity was being copied into property loops.
These findings prompted the guarded native-state repair.

The [accepted final record](pg_hotpath/README.md) supersedes those intermediate
runs and retains the comparison against untouched PyPI 0.5.8 and the pre-repair
Phase 7 source. The user accepted the remaining compiled tradeoff on
13 November 2026. Phase 7 is complete; further optimization remains optional
in Phases 11/13. [Final case summaries](pg_hotpath/results.json) and
[correctness verification](pg_hotpath/verification.json) are retained;
intermediate timing records, workers and scripts have been removed.
## Method and scope of the original PyPI checkpoint

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
