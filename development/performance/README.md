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

## Phase 8 bicubic policies and bounded inverses

The [Phase 8 guard](phase8_validation.json) compares the bicubic implementation
with accepted Phase 7 commit `ec80b96`. Four workers per version ran serially in
ABBAABBA order, with nine synchronized samples per case and independent 16-by-16
HEOS CO2 tables. All timings below return complete states; compilation is
recorded separately. These are comparisons within this run, not against PyPI.

| Complete call | Phase 7 | Phase 8 |
| --- | ---: | ---: |
| Eager HP scalar, raise | 119.5 us | 105.6 us |
| Eager HP batch, 1,024, raise | 737.7 us | 726.5 us |
| Compiled HP scalar | 56.4 us | 60.1 us |
| Compiled HP batch, 1,024 | 647.8 us | 614.5 us |
| Eager PT inverse | 2,204.4 us | 126.0 us |
| Compiled PT inverse | 205.1 us | 70.0 us |
| Eager PS inverse | 2,214.0 us | 126.3 us |
| Compiled PS inverse | 228.6 us | 70.7 us |

Eager inverse calls are about 14–18 times faster; compiled inverse calls about
2.5–3.2 times faster. Residuals calculate only their selected properties without
assembling states during iterations, and PS now uses a one-dimensional solve.
The representative numerical model has 35 array leaves rather than 269.
Setup-proven globally available properties carry no repeated availability grid;
partial optional masks and safe finite arithmetic remain. Domain, root,
finite-core and status checks are retained, without native host callbacks.

The short compiled HP scalar case is 6.5% slower in this guard, with paired
ratios from 0.995 to 1.130. The unchanged PG temperature control also shifts
about 6%; both PG optimized-program hashes are identical across versions. The
[longer focused comparison](phase8_scalar_followup.json) uses 24 samples and
four times as many calls per sample in each of four workers per version.
Compiled HP medians are 68.5 versus 66.3 us, but paired ratios range from 0.551
to 1.127; unchanged PG controls show similarly large worker shifts. The scalar
result is therefore **inconclusive**, not evidence of either a reliable speedup
or universal absence of overhead. Retain this case in the Phase 13 gate. The
ordinary eager batch and compiled batch cases show no repeatable slowdown
after removing redundant all-available masks.

Cached loading is 23.42 versus 27.33 ms, an additional 3.91 ms at setup.
Complete source/manifest/payload validation remains enabled. Construction and
loading are separate from repeated state evaluation; profiling their remaining
cost is optional Phase 11/13 work. Metastable generation now uses seven source
flashes rather than four per node to improve mixed derivatives; that additional
construction work is outside this ordinary-table timing comparison.

The [final HEOS/REFPROP verification](phase8_verification.json) passes **2,182
tests with 15 existing strict expected failures**, zero unexpected failures and
zero skips. It retains **54,272 numerical comparisons**, including **50,189
finite values** and **4,083 paired unavailable values**, with zero failures.
Phase 8 adds 200 passing cases and removes one obsolete unsupported-mask
parameter. All 884 sampled benchmark comparisons pass; core values compare
without adjustment. Only the old compressibility factor is normalized to the
selected EOS gas constant, matching the independently tested source correction.
The exact package hash is retained with both correctness and timing evidence.
A later documentation-only follow-up restores source explanations and option
documentation. The verification record retains both source-byte hashes and
confirms identical executable ASTs after excluding docstrings; numerical code
and the timing claims are unchanged.

This CPU/x64 guard does not establish all devices, phase envelopes, example
studies or ThermOpt application performance. The full published-release gate
remains Phase 13, and example execution acceptance remains Phase 12.
Temporary timing workers, extracted source copies, tables and helper scripts
were removed after preserving these compact records and the full test run.

## Phase 9 consolidated binary model

The [Phase 9 guard](phase9_validation.json) passes its warmed-call checkpoint:
all 13 comparable medians remain below the 10% regression budget. Source binary
calls are substantially faster; pure PG/bicubic control changes range from
-5.7% to +2.8%. This result has the numerical and execution scope below.

The baseline is the unchanged committed Phase 8 package at `4c6ae84`. Four
independent workers per version run **serially** in ABBAABBA order, with nine
samples per case, explicit warmup and `jax.block_until_ready` after every
measured call. Package paths/source hashes, dependencies, CPU/x64 settings,
thread settings and returned full-state fields are recorded. Construction and
first-call compilation are separate from warmed complete-state timings. This
is a phase-to-phase guard, not the Phase 13 comparison against a PyPI release.

The workload matrix has **13 comparable cases**: four legacy scalar source-HP
mixtures, one 16-state source-HP batch, and eight scalar/batch eager/compiled
PG/bicubic full-state controls. The source cases cover Water/Nitrogen at two
mass fractions, CO2/Nitrogen and Nitrogen/Oxygen. The old batch counterpart is
explicitly a scalar loop followed by public `FluidState.stack`; the new version
uses its public broadcast call. Each implementation returns its complete state,
including the new model's additional supported fields.

The timing sources use HEOS. Bicubic controls use separate per-worker 16-by-16
CO2 tables over 600–850 kJ/kg and 5–12 MPa. Warmed medians below are microseconds
per complete call; grouped control rows list scalar/batch values.

| Comparable complete call | Phase 8 | Phase 9 | New / old |
| --- | ---: | ---: | ---: |
| Source HP Water/Nitrogen, water mass fraction 0.8 | 1,117.1 | 536.1 | 0.480 |
| Source HP Water/Nitrogen, water mass fraction 0.2 | 1,017.3 | 468.3 | 0.460 |
| Source HP CO2/Nitrogen | 954.3 | 495.2 | 0.519 |
| Source HP Nitrogen/Oxygen | 934.0 | 434.5 | 0.465 |
| Source HP Water/Nitrogen, batch 16 | 16,784.4 | 5,705.7 | 0.340 |
| PG eager HP full scalar / batch 1,024 | 50.0 / 134.3 | 49.8 / 136.2 | 0.998 / 1.014 |
| PG compiled HP full scalar / batch 1,024 | 35.0 / 94.7 | 35.0 / 96.4 | 0.999 / 1.018 |
| Bicubic eager HP full scalar / batch 1,024 | 109.5 / 761.8 | 108.9 / 759.9 | 0.994 / 0.998 |
| Bicubic compiled HP full scalar / batch 1,024 | 65.3 / 607.1 | 61.6 / 624.0 | 0.943 / 1.028 |

Source HP scalar calls improve 1.93–2.17 times and batch16 improves 2.94 times.
The inverse iterations query only selected component enthalpies, preparing
source helpers at construction and assembling complete states after the root
passes its residual check. No numerical tolerance is relaxed for timing.
Individual paired control ratios remain variable: compiled PG batch ratios
range from 0.887 to 1.162. The bicubic optimized program is identical across
versions (703 instructions, no callback); PG retains 821 instructions and no
callback, with changed stable representability-bound algebra. The small control
shifts do not demonstrate a general kernel speedup or absence of all overhead.

The numerical comparison retains **19 binary fields**: the six core variables,
cp, expansion, isothermal compressibility/bulk modulus, sound speed, transport,
Gruneisen and composition fields. Inverse targets come from independent source
PT states in valid single-phase domains; matching an unchecked root across a
phase discontinuity is not an accuracy requirement. The old mass-averaged cv
and mass-weighted isentropic compressibility/bulk modulus are intentionally
corrected. Their new values are checked against
$c_v=c_p-T\alpha_p^2/(\rho\kappa_T)$,
$\kappa_s=1/(\rho a^2)$ and $K_s=\rho a^2$; they are not silently normalized
into equality with the old inconsistent output labels. The complete-call
timing includes these corrections and the larger 27-property state.

The **20 new-only cases** measure native PG/PG and bicubic/PG PT/HP/PS states
(eager scalar, compiled scalar and compiled batch), plus compiled whole-host
callback PT/HP states. The baseline has no working equivalent component-object
native/callback binary API. Their absolute complete-state medians are:

| New binary call | Eager scalar | Compiled scalar | Compiled batch |
| --- | ---: | ---: | ---: |
| PG/PG PT | 190.6 | 50.9 | 420.0, batch 1,024 |
| PG/PG HP | 206.7 | 53.6 | 492.4, batch 1,024 |
| PG/PG PS | 205.7 | 52.1 | 455.2, batch 1,024 |
| Bicubic/PG PT | 304.5 | 92.5 | 274.9, batch 128 |
| Bicubic/PG HP | 319.9 | 123.3 | 771.5, batch 128 |
| Bicubic/PG PS | 343.4 | 141.1 | 980.4, batch 128 |

Whole-host compiled callback PT/HP scalar medians are 529.8/717.6 us. All 27
enabled properties match their PT round trips, including paired unavailable
fields; the PG and bicubic/PG HP derivative checks pass. These new-only rows
receive no legacy speedup ratio. The larger native binary graphs remain a
profiling opportunity in Phase 11, particularly nested bicubic PT inversions.

Construction is separate from repeated evaluation. Four initial construction
samples per version give case medians of 4.58–8.43 ms for configured source binaries versus
2.11–3.84 ms for the old constructor: approximately 2.2–4.6 ms additional setup.
These are initial per-worker samples, not a dedicated warmed setup benchmark.
Reference/configuration/helper preparation remains enabled; Phase 11 tracks
profiling and reuse without moving these checks into state calls.

The [final HEOS/REFPROP verification](phase9_verification.json) passes **2,463
tests with four existing strict expected failures**, zero unexpected failures
and zero skips. Phase 9 adds **267 passing cases**, resolves 11 binary expected
failures and adds three schema/factory cases. All **54,272 numerical comparisons**
remain: **50,189 finite values and 4,083 paired unavailable values**, zero
failures. All **5,256 sampled old/new benchmark comparisons** pass across the
four worker pairs; every worker passes its own physical/round-trip checks.
The current timing package exactly matches the final full-suite source hash.
All 65 examples parse; one binary example actually runs, with larger studies
explicitly deferred in the [example audit](../../z_EXAMPLES_AUDIT.md).

The historical PG and Phase 8 caveats above remain valid for their measured
source versions. Broader devices, phase domains, ThermOpt consumers and the
latest published-release acceptance remain Phases 12–14 work. Temporary timing
workers, extracted source copies, generated tables and probe scripts were
removed after preserving these compact records and the final correctness run.

## Phase 10 phase calculations and source diagnostics

The [Phase 10 guard](phase10_validation.json) passes all 23 comparable warmed
cases within the 10% checkpoint budget. Native control medians range from
-4.6% to +3.4%, with identical optimized programs across all eight workers.
The repaired source phase/diagnostic paths are substantially faster.

The baseline is the exact accepted Phase 9 source snapshot, SHA256
`5f36b74dd8879e7c3921a5366bb7f6036d3c19f77cf560e3c3d1e9b31a7c4c01`.
That snapshot includes the uncommitted Phase 9 implementation; a Git commit
alone would not identify these measured source bytes. The comparison
uses four independent workers per version, **serially in ABBAABBA order**, with
eight warm calls and nine rotated/reversed samples per case. Every measured
call returns a complete state and synchronizes its numerical outputs.
Construction, cached loading and first-call compilation are recorded separately.
Import locations, exact package hashes, dependencies, CPU/x64/thread settings,
returned state fields, optimized native programs and paired-worker variability
are retained with the result. This remains a phase-to-phase guard; the formal
comparison against the latest published release belongs to Phase 13.

The exact [compressed Phase 9 source](phase10_baseline_phase9.zip) is retained
for reproducibility, with its archive/package hashes in the verification
record. It contains the package Python sources and can be extracted into an
isolated import directory alongside the recorded dependencies.

There are **23 comparable timing cases when REFPROP is available**, or 20 with
HEOS alone. Eight are eager/compiled PG and pure-bicubic HP scalar/batch controls.
Independent 16-by-16 HEOS CO2 tables cover 600–850 kJ/kg and 5–12 MPa. The other
cases cover ordinary source CO2 PT scalar/batch16, requested scalar diagnostics,
ordinary Water PT, compiled CO2 callback scalar/batch16, Water/Air/R410A PQ
states on HEOS and optional REFPROP, an actual homogeneous metastable Water
liquid evaluated by rhoT and PH inverse, and a legacy equilibrium PH inverse.
The metastable point is Water at 450 K and 0.7 MPa, with an independently
specified homogeneous liquid source as its reference. Both inverse versions
use the same requested tolerance; numerical budgets are not relaxed for speed.

Warmed complete-call medians below are microseconds. Grouped entries list
scalar/batch or HEOS/REFPROP values in that order.

| Comparable complete call | Phase 9 | Phase 10 | New / old |
| --- | ---: | ---: | ---: |
| PG eager HP scalar / batch 1,024 | 49.5 / 139.9 | 49.0 / 133.4 | 0.989 / 0.954 |
| PG compiled HP scalar / batch 1,024 | 34.9 / 93.7 | 33.3 / 92.8 | 0.954 / 0.990 |
| Bicubic eager HP scalar / batch 1,024 | 107.8 / 721.8 | 111.2 / 738.5 | 1.032 / 1.023 |
| Bicubic compiled HP scalar / batch 1,024 | 61.6 / 591.0 | 62.3 / 610.8 | 1.012 / 1.034 |
| Source CO2 PT scalar / batch 16 | 348.3 / 870.2 | 350.4 / 871.5 | 1.006 / 1.001 |
| Source CO2 PT with requested diagnostics | 987.9 | 433.2 | 0.439 |
| Source Water PT scalar | 354.2 | 353.0 | 0.997 |
| Compiled CO2 callback PT scalar / batch 16 | 426.0 / 1,008.1 | 426.1 / 965.1 | 1.000 / 0.957 |
| Source Water PQ scalar, HEOS / REFPROP | 585.4 / 876.9 | 405.2 / 445.8 | 0.692 / 0.508 |
| Source Air PQ scalar, HEOS / REFPROP | 524.5 / 1,313.4 | 465.4 / 518.4 | 0.887 / 0.395 |
| Source R410A PQ scalar, HEOS / REFPROP | 600.8 / 1,355.9 | 447.4 / 549.8 | 0.745 / 0.406 |
| Metastable Water rhoT scalar | 1,234.7 | 368.2 | 0.298 |
| Metastable Water PH inverse | 12,783.7 | 1,696.6 | 0.133 |
| Equilibrium Water PH inverse | 3,621.3 | 1,078.0 | 0.298 |

Requested diagnostics improve 2.28 times; two-phase calls improve 1.13–2.53
times; the tested metastable/equilibrium calls improve 3.35–7.54 times. Prepared
reference contexts avoid repeated source construction, related coefficients
share PQ samples, diagnostic transactions share saturation flashes, and host
quality limiters use NumPy rather than dispatching scalar JAX operations.
Numerical settings and comparison tolerances are retained. Small native shifts
are within paired-worker variation; for example, one eager bicubic batch pair
is 17.7% slower while its aggregate is +2.3%. The PG/bicubic optimized programs
are identical across both versions and all eight workers (821/703 instructions,
no callbacks). These controls add no native numerical work in this workload;
they do not establish every device, domain or application timing.

Preserved ordinary single-phase fields and core quantities must match within
the recorded numerical tolerance, including paired unavailable values. Native
controls compare all enabled fields at scalar states and sampled batch rows;
ordinary source/callback batches compare all 16 rows of their preserved common
fields.
Two-phase comparisons intentionally use a narrower physical scope. Pure Water
compares core quantities, compressibility factor and transport. Air/R410A
compare core quantities and compressibility factor. Their corrected actual-PQ
quality and phase volumes also change the empirical volume-averaged transport;
those transport values are independently checked rather than forced equal to
the old common-temperature endpoint construction.

The corrected two-phase heat capacities, acoustic, Gruneisen and Joule–Thomson
responses follow derivatives of the actual source PQ map. They do not preserve
the old phase-weighted response labels. Singular or undefined pure-fluid
coexistence responses remain unavailable. Current quality/phase-volume values
are checked against the actual source flash, transport against its actual
phase volumes and source PQ endpoint coefficients, and sound speed against an
independent fixed-entropy pressure difference. The latter solves quality from
raw PQ flashes with Brent's method at neighboring pressures; it does not reuse
the package's derivative helper or assume every pseudo-pure backend supports
an interior PS flash. These explicit corrections must remain visible in the
retained comparison record.

The legacy metastable timing cases select `generalize_quality=True` in **both**
versions. Phase 9's `False` route assigned a returned quality tuple to both
fields, accidentally broadcasting scalar core values into duplicated length-two
arrays. Treating that malformed result as an old scalar timing baseline would
be misleading. The new `False` route has its own scalar correctness tests;
quality indicators are excluded from cross-version metastable equivalence
checks. The legacy enabled-property correction is also explicit: old phase
states advertised the whole canonical schema, including unsupported fields;
new phase states enable only the applicable 25 ordinary or 31 diagnostic
properties. Both versions still return their complete states during timing.

Quality extensions outside coexistence are documented indicators, not physical
vapor fractions. The supercritical extension now uses entropy differences and
a gas-constant scale, preserving the result under permitted reference shifts.
Configured phase solvers retain the source's prepared reference context and
hold its lock through the complete calculation. Finite-difference helpers
preserve the caller's state. Requested diagnostic transactions share bubble/dew
data; default state calls do not perform that extra diagnostic work.

Bicubic phase-envelope validation is paid during generation, with a validated
boolean cell mask persisted in cache format 2. Adaptive source PQ envelope
probes catch interior curvature missed by nodal flags while retaining cells
wholly within a valid two-phase region. This is a finite, conservative policy,
not a proof of the exact EOS envelope throughout every retained cell. Cached
loading reuses the stored guard, and native evaluation uses its existing mask
lookup without new source queries. Metastable generation steps scale with the
table's enthalpy span, not absolute enthalpy; reference-shift tests and explicit
unresolvable-stencil errors protect derivative conditioning. Their policy is
captured in the generation manifest. Setup/cache costs must be reported
separately from repeated property evaluation, without claiming zero setup cost.

Setup remains separate. Case medians across four initial per-worker samples
measure fresh table construction at 1.382 versus 1.340 seconds and cached
loading at 94.44 versus 95.36 ms. Initial callback construction is 2.86 versus
3.24 ms (+0.38 ms), with variable individual samples; this is retained as a
Phase 11 setup-profiling opportunity. These are not dedicated repeated warmed
setup benchmarks, and no setup validation is disabled.

The [final verification](phase10_verification.json) passes **2,733 HEOS/REFPROP
tests**, with **two existing Phase 11 serialization expected failures**, zero
unexpected failures and zero skips. Phase 10 adds **268 passing tests** and
resolves the two pseudo-pure quality expected failures. All **54,272 numerical
comparisons** remain: **47,867 finite values and 6,405 paired unavailable values**,
with zero failures. Relative to Phase 9, 2,322 finite surrogate-label assertions
become explicit paired-unavailable checks: 774 each for pure two-phase Cp,
heat-capacity ratio and isothermal Joule–Thomson. The comparisons and their
tolerances are retained; availability changes are recorded rather than hidden
as unchanged finite coverage.

All **6,992 old/new benchmark comparisons** pass (6,768 finite and 224 paired
unavailable), and every worker passes its independent source/geometry/sound
checks. The final full-suite source hash exactly matches all current timing
workers. All 65 examples parse; no example study or figure execution is claimed
for Phase 10. [Example audit](../../z_EXAMPLES_AUDIT.md).

Temporary workers, extracted source snapshots, generated tables and probes were removed
after preserving the compact records and final full-suite evidence. Broader
devices, full phase domains, ThermOpt application performance and complete
example studies remain Phases 11–14 work. Historical Phase 7–9 measurements
above retain their original source versions and acceptance limits.

The subsequent [metastable example follow-up](../example_audit/metastable_solver_followup.json)
fixes a construction-only continuation seed defect, verified by four new tests
and a full run of **2,737 passing cases with two existing expected failures**.
Its package hash is `851d8c9…`; the timings above remain measurements of the
recorded `9e30ce5…` version, whose exact [source archive](phase10_measured_source.zip)
is retained. All other package files and runtime evaluator functions are
identical; the constructor differs only by two generation-policy manifest
strings, and the two metastable generation methods use corrected seeds.
No repeated state-evaluation checks or arithmetic were added by this repair.
Example execution times under concurrent correctness workloads are not used
as a new timing comparison.

## Phase 11 lifecycle, consumers and table stability

The required Phase 11 scope passes **2,852 HEOS/REFPROP tests**, with no failures,
skips or remaining expected failures. All **54,272 numerical comparisons** remain:
47,867 finite values and 6,405 paired unavailable values, with zero failures.
The frozen package hash `cb75ac4…` matches the complete suite and every current
timing worker. [Verification](phase11_verification.json),
[full run](../test_audit/results/phase11/full_20261114_114730/summary.json).

The guard compares accepted commit `632b7fd` (package hash `851d8c9…`) with the
current implementation, using four independent serial workers per version in
ABBAABBA order, nine synchronized samples and four times the base call count.
Both use CPU, double precision, matched dependencies and independent table
caches. Each complete return is synchronized. Object setup, fresh generation,
cached loading and first-call compilation are recorded separately from warmed
calls. This checkpoint is against the preceding accepted implementation; the
latest-PyPI comparison remains Phase 13.

All **26 comparable warmed cases meet the 10% budget**. New/old median ratios
range from **0.868 to 1.027**. All **98,688 sampled cross-version comparisons**
pass. Every worker also matches all **640 numerical table-array digests**
bitwise with paired NaNs: 280 ordinary CO2 arrays and 360 metastable Nitrogen
node/derivative/coefficient/saturation arrays. Generation manifests intentionally
differ and are not treated as numerical arrays. [Complete timings/raw samples](phase11_validation.json),
[retained runner](phase11_benchmark.py), [exact baseline source](phase11_baseline.zip).

Representative complete-state medians are:

| Workload | Previous | Phase 11 | Change |
| --- | ---: | ---: | ---: |
| PG eager scalar HP | 46.38 us | 45.82 us | -1.2% |
| PG eager HP batch 1,024 | 130.08 us | 114.05 us | -12.3% |
| PG compiled scalar HP | 48.32 us | 44.27 us | -8.4% |
| PG compiled HP batch 1,024 | 123.01 us | 107.21 us | -12.8% |
| Bicubic eager scalar HP | 88.41 us | 82.59 us | -6.6% |
| Bicubic eager HP batch 1,024 | 826.13 us | 779.07 us | -5.7% |
| Bicubic compiled scalar HP | 86.96 us | 87.81 us | +1.0% |
| Bicubic compiled HP batch 1,024 | 795.24 us | 767.89 us | -3.4% |
| Source callback full state + dh/dT, scalar | 955.92 us | 943.97 us | -1.3% |
| Source callback full state + dh/dT, batch 16 | 2,819.91 us | 2,590.67 us | -8.1% |
| Binary callback full state + dh/dT, scalar | 1,271.09 us | 1,167.13 us | -8.2% |
| Binary callback full state + dh/dT, batch 16 | 6,922.17 us | 6,493.01 us | -6.2% |
| Metastable Water direct rho/T | 393.41 us | 403.93 us | +2.7% |
| Metastable Water PH inverse | 2,066.69 us | 2,057.84 us | -0.4% |

The twelve compiled control programs (including native AD/consumers and primal
callbacks) are identical across all eight workers. Their apparent gains reflect
host variability rather than demonstrated arithmetic changes. Only the four
corrected callback derivative programs differ: failed-primal/unavailable-property
coefficients are zeroed before tangent multiplication to prevent reverse-mode
NaN contamination. Their sampled valid-state values and derivatives are retained.

Initial setup observations show a real separate cost:

| Initial setup, median of four workers | Previous | Phase 11 |
| --- | ---: | ---: |
| Direct source construction | 2.74 ms | 7.03 ms |
| Callback construction | 3.17 ms | 7.00 ms |
| Ordinary cached loading | 93.64 ms | 92.00 ms |
| Metastable liquid fresh 16-by-16 generation | 4.21 s | 4.08 s |

Source construction adds approximately **4 ms** for the verified serialization
provenance snapshot. No identity probes enter repeated state calls. Setup has
only four initial observations per version, not a dedicated repeated warmed
setup benchmark; ranges remain in the portable record. The generation ranges
overlap (previous 3.89-5.62 s, current 3.89-5.45 s), so the coefficient-loop repair
removes redundant work without a claimed measured generation speedup. Cached
loading remains comparable. Reusing prepared immutable setup information safely
is an explicitly deferred profiling opportunity.

Physical/formal metastable stability and optional pole masks are prepared during
construction and reuse the existing native masks. The four direct default
solver/blending examples retain all 3,636 compared core values bitwise. Both
large Nitrogen studies execute their original full resolutions under explicit
formal mode; finite response coverage and the remaining large near-pole errors
remain visible. Their exact earlier frozen execution source and unchanged
numerical-path proof are retained separately; they are not relabeled as timing
runs or as whole-domain accuracy acceptance. [Example audit](../../z_EXAMPLES_AUDIT.md#phase-11-lifecycle-and-formal-table-follow-up),
[lifecycle guide](../../docs/fluid_lifecycle.md).

Reproduce the guard with:

```powershell
python development/performance/phase11_benchmark.py --baseline-source development/performance/phase11_baseline.zip --output-directory development/performance/results/phase11/repeated_guard --calls-scale 4
```

## Phase 12 examples and source-inverse checkpoint

The matched guard compares accepted Phase 11 commit `588c144` (exact source
`cb75ac4…`) with Phase 12 source `21202218…`. It uses the retained unchanged
runner, four workers per version in serial ABBAABBA order, nine synchronized
samples per workload and `--calls-scale 4`. All repository scientific jobs had
finished first; the two unrelated local servers were left running and each used
only 0.03125 CPU seconds during the guard. This observation is not a complete
system-load profile.

All **26 warmed workloads meet the 10% budget**. All **98,688 cross-version
sampled checks** pass: 97,976 finite and 712 paired unavailable values. All 640
numeric-array digests per worker match. The eight native optimized programs and
all 16 inspected compiled graphs are identical across their matching workers;
native PG/bicubic graphs contain no callbacks. Individual worker ratios remain
visible because unchanged controls show substantial variability. Apparent native
speedups are not attributed to changed arithmetic.
[Full samples and environment](phase12_validation.json),
[correctness/provenance](phase12_verification.json).

Complete-state medians below are microseconds per scalar call or 1,024-state
batch. They are a Phase 11 checkpoint comparison, not the Phase 13 PyPI gate.

| Workload | Phase 11 | Phase 12 | New / old |
| --- | ---: | ---: | ---: |
| Perfect gas, eager scalar | 53.26 | 50.02 | 0.939 |
| Perfect gas, eager batch | 140.20 | 134.55 | 0.960 |
| Perfect gas, compiled scalar | 52.27 | 48.69 | 0.931 |
| Perfect gas, compiled batch | 132.18 | 122.87 | 0.930 |
| Bicubic, eager scalar | 118.89 | 118.33 | 0.995 |
| Bicubic, eager batch | 997.68 | 910.72 | 0.913 |
| Bicubic, compiled scalar | 153.31 | 107.36 | 0.700 |
| Bicubic, compiled batch | 971.69 | 860.59 | 0.886 |
| Water metastable density/temperature | 417.18 | 413.26 | 0.991 |
| Water metastable pressure/enthalpy inverse | 2,226.26 | 2,027.90 | 0.911 |

The only numerical package change removes a duplicate final source flash and
validates the exact returned inverse state. A pressure-only stiffness-scaled
floating-roundoff floor addresses a valid low-net-pressure Water root; nonlinear
iteration tolerances and other residual budgets remain unchanged. The inverse
shows an observed 9% improvement, with worker variability retained. Ordinary
source PT scalar/batch medians rise 4.95%/5.87%, within the checkpoint budget.
PG/bicubic numerical code is unchanged. The later smoothing docstring correction
is proven numerical-AST-equivalent to the exact full-suite/example execution
source in the [documentation proof](../example_audit/phase12_docstring_equivalence.json).

Four **initial setup observations** per version remain separate from warmed
evaluation: source construction 6.99→7.02 ms, callback construction 7.13→6.95 ms,
cached loading 92.41→92.04 ms, and fresh 16-by-16 metastable generation
5.12→4.33 s. Their ranges overlap; these are not dedicated repeated setup
benchmarks or evidence of a general construction speedup.

The full HEOS/REFPROP suite passes 2,951 tests and preserves all 54,272 numerical
comparisons. One final export-header regression passes with the other 47
table-tool tests, yielding 2,952 distinct verified cases. All 62 executable
examples pass quick and full profiles, with three reference-only files kept
separate. Examples' concurrent wall times are not benchmark measurements.

Reproduce the guard with:

```powershell
python development/performance/phase11_benchmark.py --baseline-source development/performance/phase12_baseline.zip --output-directory development/performance/results/phase12/repeated_guard --calls-scale 4
```

Phase 13 must still resolve and pin the latest published release and run the
broader standalone comparison. Final ThermOpt verification and application
timings remain Phase 14.

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
