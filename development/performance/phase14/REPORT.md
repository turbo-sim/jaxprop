# Phase 14 native performance investigation

**Historical C3 report, 14 November 2026.** The
[15 November follow-up](followup_20261115.md) records the newer PG tracing fix,
passing four-profile cold confirmation and controlled 119-case matrix with
five remaining timing flags. The [attribution report](mixed_attribution_20261115.md)
and [new portable measurements](followup_validation.json) retain the processor,
provider-history and GC evidence. Original observations below remain unchanged;
the performance gate is still open.

**The formal performance gate remains OPEN.** Phase 13 was committed as
`b282ff0`; this phase retains its original budgets. The completed broad matrix
has **17 warmed flags**, one setup flag and the unresolved PG first-call flag.
Those results are not waived by the narrower diagnostics. Numerical verification
covers an audited union of **3,027 distinct tests** and all benchmark comparisons
pass. Matched native-only diagnostics support useful batch improvements against
frozen source, but do not establish broad-workload performance acceptance.

## References and experiment provenance

The unchanged published reference is the pinned PyPI **0.5.8** wheel, verified
against the [retained artifact manifest](../phase13/baseline/baseline_manifest.json).
Its package source hash is
`858de0f7945b8fcb21e6af9c146b0c150e7101e2e722c4328511a3b3d2a88da0`.
The frozen Phase 13 package hash is
`9fc27b79041931d98cbcb8349cdab7d81fe31000a49cd3fe14dc7cd85ebe3651`.
The [frozen archive](frozen_source.zip) preserves that exact numerical source.

Candidate **C1** has package hash
`71048e47fb1e9b40db89ab2bd4159e569ecb5db3ae0a600396315f0accced199`:
the initial PG shared-context work and native bicubic HP batch evaluation.
Candidate **C2** has package hash
`d1a1156a595a6b75d158e667f1dfb3f3860a2498ef2cda04a2598287d4e68aa5`:
the same PG candidate with isolated branchless bicubic batch assembly.
Later docstring/test additions do not relabel these earlier tested sources.
Candidate **C3**, the chosen numerical checkpoint, has package hash
`ce94b70d2ddc89fe602a52a2fac8fb71ff5e7f095588431e7d6a5e065611cde3`:
the bicubic candidate plus the isolated PG failure-branch simplification and
final source documentation. Test-only additions do not change this package hash.

| Completed artifact | Actual candidate | Measurement scope |
| --- | --- | --- |
| [first_native](../results/phase14/first_native/summary.json) | C1 | One independent worker/version, nine samples, 13 native workloads; exploratory only |
| [balanced_native](../results/phase14/balanced_native/summary.json) | C1 | Four independent workers/version, 30 samples, published/frozen/current; 13 workloads |
| [branchless](../results/phase14/branchless/summary.json) | C2 | Four independent workers/version, 30 samples, frozen/current; 17 workloads |
| [cold_stages](../results/phase14/cold_stages/summary.json) | C2 | Two fresh workers/version/profile/mode; whole public call and separate stage diagnostics |
| [paired_models](../results/phase14/paired_models/summary.json) | C2 | Same-process, interleaved full returns and diagnostic projections; 30 samples |
| [paired_final](../results/phase14/paired_final/summary.json) | C3 | Same-process, interleaved full returns and diagnostic projections, including all-status packing; 30 samples |
| [cold_pg_branch](../results/phase14/cold_pg_branch/summary.json) | C3 | Two fresh workers/version/mode for PG HP 1,024-state whole calls and separate stage diagnostics |
| [full_matrix](../results/phase14/full_matrix/summary.json) | C3 | Four independent workers/version; 119 comparable workloads and four current-only capabilities |
| [cold_remaining](../results/phase14/cold_remaining/summary.json) | C3 | Two fresh workers/version/profile/mode for the other three original cold profiles |
| [null_control](../results/phase14/null_control/summary.json) | Frozen/frozen, both `9fc27b79…` | Six identical-source cases, four independent workers per label |

Each artifact records its actual source and harness hashes, imported package
paths, numerical checks and host observations. The [protocol](protocol.json)
declares matched fixed calls, balanced worker/case blocks, complete-return
synchronization and fresh compiler caches. Independent workers use the same
Python 3.13.2, JAX/jaxlib 0.6.2, Equinox 0.12.2, NumPy 2.4.4 and CoolProp 7.2.0
CPU/x64 environment. Timed PG full states retain the actual 22 current fields
versus 13 computed legacy fields and original placeholders. Legacy status is
never manufactured. Candidate/frozen verification compares every actual field,
shape, dtype, status, clipping flag and NaN/availability pattern.

## Completed formal matrix and unresolved gates

The [maintained measurements](../phase14_validation.json) retain the complete
matrix, original limits and every remaining flag. The full matrix includes
the **111 original comparable workloads plus eight added controls**, with
**four current-only capabilities**. Of 119 comparable warmed cases, **102 meet
their original budgets and 17 exceed them**: 16 original cases and one added
16-by-8 bicubic broadcast. The preaccepted PG 40% envelope stays confined to
its original full-batch cohort; its as-shipped/default HP case exceeds even
that limit in this run.

| Warmed case above its original budget | C3 slowdown versus PyPI |
| --- | ---: |
| PG dynamic-model HP batch, 1,024 | 37.2% |
| Bicubic compiled HP scalar | 12.3% |
| Bicubic compiled HP singleton batch | 11.6% |
| Bicubic compiled HP batch, 1,024 | 15.1% |
| Bicubic compiled PT batch, 1,024 | 21.5% |
| Bicubic compiled HP broadcast, 8-by-16 | 26.5% |
| Bicubic compiled full state plus dT/dh | 13.9% |
| PG common-13 property projection, 1,024 | 39.2% |
| PG as-shipped/default full HP batch, 1,024 | 44.8% |
| PG compiled compressor with actual status | 111.6% |
| PG compiled expander with actual status | 113.3% |
| PG compiled HX, 128 cells, with actual status | 70.1% |
| Bicubic compiled HX, 128 cells, with actual status | 48.3% |
| REFPROP Water liquid source PT scalar | 34.3% |
| REFPROP CO2 supercritical source HP scalar | 25.7% |
| REFPROP Water metastable liquid HP inverse | 13.1% |
| Added bicubic HP broadcast, 16-by-8 | 11.1% |

The aggregate setup comparison has **22 comparable operations and one flag**:
REFPROP Air source initialization takes **12.77 ms**, above **10.09 ms** allowed.
This phase did not alter that source-initialization numerical path. The other
21 setup operations pass; this is a specific unresolved setup observation, not
a reason to move checks into repeated state calls or disable source validation.

Four original first-call profiles are now measured on C3 in fresh processes:

| First synchronized public call | PyPI / C3, ms | Original allowance, ms | Result |
| --- | ---: | ---: | --- |
| PG HP batch, 1,024 | 54.67 / 200.37 | 182.01 | Above budget |
| Bicubic HP batch, 1,024 | 322.04 / 235.92 | 583.06 | Within budget |
| PG HX, 128 cells | 63.34 / 160.75 | 195.00 | Within budget |
| Bicubic compressor | 1,044.10 / 728.69 | 1,666.16 | Within budget |

The broader matrix measures current/frozen shifts of approximately 1.2–1.5
in several cases, including scalar/source consumers with unchanged numerical
paths or optimized programs. Native-only paired diagnostics below do not
reproduce all those shifts. Their disagreement requires further attribution;
it does not make the broad results invalid or passing.

The six-case null experiment loads the **same frozen package hash under both
labels**, uses four workers per label and measures ratios **1.003–1.085**.
It demonstrates measurable process/context variation, but is too small to
explain quantitatively the larger 20–50% shifts in the full workload mix.
All null samples remain recorded; no host-load explanation is assumed proven.

## What the earlier independent workers establish

All ten completed experiments report successful numerical verification.
Their timing distributions also expose significant process/context variation.
The first one-worker run showed broad regressions, including PG cases whose
optimized programs were unchanged by the subsequent bicubic experiment. It
does not establish an intrinsic penalty for the candidate.

The balanced C1 run measures bicubic full HP batches of 1,024 at
684.91→645.10 microseconds against frozen source, broadcast HP at
154.57→145.64, dynamic full batches at 779.82→672.03 and status-preserving HX
at 97.64→95.56. However, C1 still exceeds the original published-release
broadcast and HX budgets by 23.6% and 35.0%, respectively. Its full 1,024-state
case is 4.6% slower than published, inside the 10% limit.

The separate C2 worker run instead measures candidate/frozen ratios of 1.046
for full HP batches, 1.029 for broadcast, 1.008 for dynamic batches and 1.133
for HX. Unchanged PG compressor/expander programs also shift from approximately
3% faster than frozen in the earlier run to 12% slower. These observations are
retained rather than discarded; cross-run aggregate medians cannot isolate
the effect of removing the bicubic conditional. They motivated the separate
same-process attribution experiment.

## Bicubic matched diagnostics

The paired diagnostic co-loads unchanged source files under separate package
names and interleaves the versions using the same numerical inputs. It reduces
between-process confounding but is **diagnostic only**: it does not replace
the independent-worker published-release gate. PG includes the published
package; bicubic compares frozen/current only because its legacy absolute
state-constructor import prevents equivalent alias loading. No published
bicubic ratio is invented from this experiment.

| Bicubic workload | Frozen, microseconds | C2, microseconds | Change |
| --- | ---: | ---: | ---: |
| Full HP, 1,024 states | 884.10 | 843.95 | 4.5% faster |
| Full HP, flat 128 states | 149.93 | 135.63 | 9.5% faster |
| Full HP, 8-by-16 broadcast | 162.90 | 139.02 | 14.7% faster |
| Full HP, 16-by-8 broadcast | 160.17 | 136.62 | 14.7% faster |
| Full HP, independent broadcast axes | 156.33 | 139.53 | 10.7% faster |
| HX, numerical outputs plus actual status | 91.83 | 90.92 | 1.0% faster |
| HX, numerical-only diagnostic | 86.98 | 87.16 | 0.2% slower |

The evidence supports a real batch/broadcast improvement against the frozen
implementation, with no measured HX improvement of comparable size. Numerical
projections and status-only ablations explain surviving work; they cannot
replace the full-state or actual-status acceptance cases.

The repeated C3 diagnostic supports that result: full 1,024-state batches are
**6.1% faster**, flat 128 **13.0% faster**, 8-by-16 broadcast **11.9% faster**,
16-by-8 **15.5% faster** and independent broadcast axes **17.5% faster** than
frozen source in that matched run. Actual-status HX is **3.1% slower**; its
numerical-only projection is **12.8% slower** in this repeat, versus approximately
unchanged in C2. These diagnostic projections remain visible rather than being
used to claim a universal HX speedup or to replace the original workload mix.

## Bicubic source and compiler findings

Direct HP evaluation now computes protected coordinates and cell basis once
for the native batch, performs batched interpolation and assembles one batch
state. Inverse routes retain their scalar bounded solvers and vectorization;
scalar ordinary calls retain their previous numerical path. The implementation
keeps all core finite/positive checks, optional availability, phase/cell checks,
status precedence, clipping and dynamic coefficient derivatives.

The original scalar-state construction inside `vmap` led the optimized broadcast
program to repeat policy/masking expressions in multiple fusion bodies. Native
batch policy greatly reduces that duplication. The first candidate also made
the batch size visible to the generic state factory's all-success conditional.
On this CPU compilation it added a global reduction and copied successful
result arrays through the conditional boundary. C2 applies equivalent per-row
masking directly and uses trusted state assembly, avoiding that new boundary.

The table counts all instruction lines, including `ROOT`, in the retained
`first_native` and `branchless` worker-1 optimized HLO files. These static
counts include fusion bodies and both branches, not executed operation counts
or measured CPU/kernel-launch costs.

| Program | Frozen instructions | C1 instructions | C2 instructions |
| --- | ---: | ---: | ---: |
| Full HP, 1,024 states | 703 | 791 | 656 |
| HP broadcast, 8-by-16 | 3,011 | 837 | 704 |
| HX, 128 cells | 677 | 689 | 621 |

For broadcast, finite/select/clamp nodes fall from **323/303/118** in frozen
source to **35/59/18** in C2. C1 introduces one conditional and 25 array copies
in full/broadcast states; C2 restores zero conditionals and one copy. HX changes
from zero conditionals/copies to two/four in C1, then back to zero/zero in C2.
The full-batch and HX dot layouts otherwise remain unchanged. Dot/gather counts
remain **21/23** for complete HP states and **8/10** for HX; entry fusion counts
remain similar. A 77% instruction-count reduction therefore does not imply a
77% runtime reduction.

The remaining HX cost versus PyPI includes required failure detection. Current
status needs temperature, density, entropy and internal-energy interpolation
for each stream, whereas the old selected consumer retained temperature alone:
eight versus two interpolation dots. This is distinct from returning unused
properties, which XLA already prunes. A future setup certificate could remove
checks only after proving whole-cell finite/positive interpolation, including
overflow, interior undershoot, dtype/boundary slack and dynamic-update rules;
no such proof or relaxation has been implemented here.

## Cold public calls and separate stage diagnostics

Each mode/profile uses fresh workers and caches, with model construction and
device startup outside the timed region. Public and staged modes are different
processes. Their stage medians must not be summed and presented as the original
whole-call observation.

The measured bicubic profile is **HP broadcast 8-by-16**, not the 1,024-state
profile. Its first synchronized public call improves from **440.28 to 284.12
ms** against frozen source, a **35.5% reduction**. Published is 294.59 ms; C2
passes this run's original first-call allowance of 541.89 ms. Separate staged
diagnostics measure tracing/lowering at **145.05→99.06 ms**, compilation at
**261.70→174.90 ms**, and first executable dispatch at **0.60→0.74 ms**.
The improvement is predominantly before warmed execution in this experiment.

C2's PG first-call result was unfavorable: **229.29→276.74 ms** against
frozen source, versus 51.56 ms published. The separate C3 comparison measures
**203.66→200.37 ms** against frozen source, versus 54.67 ms published. C3 is
approximately unchanged in whole public-call latency and still exceeds that
run's **182.01 ms** original allowance. The cold public regression is unresolved.
Separate C3 stage diagnostics improve tracing/lowering **87.11→66.55 ms** and
compilation **137.87→113.11 ms**, with first executable dispatch **0.57→0.47
ms**. Those altered-path stage improvements do not establish a comparable
whole-public-call gain and must not be summed into a replacement acceptance time.

## Perfect-gas findings and final branch experiment

C2 same-process diagnostics find actual-status compressor/expander/HX ratios
against frozen source of **0.998/0.997/1.008**. Their consumer programs and
timings show no useful gain from that initial shared-context candidate. The
ordinary full HP batch ratio is **1.053**; dynamic full-batch ratio is **0.946**.
Published/current consumer times remain materially different. For example,
compressor medians are 11.01/14.88 microseconds and expander medians are
10.72/15.01. PG full-state output remains broader than legacy, but selected
consumer regressions cannot be attributed to returning all 22 fields because
unused fields have already been pruned.

C3 isolates the duplicated all-success work in the failure arm: once the outer
large-batch branch has already determined that a batch contains failures, its
failure arm applies the shared row mask directly instead of asking the generic
factory to repeat the success reduction and conditional. The successful batch
arm is retained. Required equations, all 22 fields, statuses, clipping and
dynamic-model derivatives remain unchanged.

| PG workload, C3 paired diagnostic | Frozen, microseconds | C3, microseconds | Change |
| --- | ---: | ---: | ---: |
| Complete HP, 1,024 states | 133.75 | 127.51 | 4.7% faster |
| Dynamic model, 1,024 full states | 161.08 | 153.99 | 4.4% faster |
| Compressor, numerical outputs plus actual status | 15.50 | 15.74 | 1.6% slower |
| Expander, numerical outputs plus actual status | 16.12 | 16.17 | 0.3% slower |
| HX, numerical outputs plus actual status | 30.93 | 27.85 | 9.9% faster |

Consumer costs relative to PyPI remain material in this diagnostic: current
compressor/expander/HX take 15.74/16.17/27.85 microseconds versus published
11.10/10.84/17.69. The actual broader current status/failure contract stays in
the comparison; numerical-only ablations do not waive its original 10% budget.

A separate diagnostic stacks all real consumer statuses into one output array,
preserving every status and numerical output. PG compressor time becomes
**12.51 microseconds**, versus **15.74** with three separate scalar status
buffers. This identifies output layout as another relevant cost alongside
policy arithmetic; it is not an implementation change or a replacement gate.
Application Phase 15 can assess such status collection without changing
ordinary public full-state returns. Further independent repetitions and
application measurements are needed before generalizing that result.

## Verification and remaining acceptance

The bicubic focused run passes **90 tests**, including **25 new cases**
covering complete scalar-reference parity at sizes 1/16/63/64/65/128/1,024,
broadcast/singleton/empty axes, scalar pressure and strided inputs, mixed
mask/clip failures, core failure handling, dynamic coefficients, VJP, JVP and
centered finite-difference agreement. Two additional public batch regressions
separately pass: local optional-cell unavailability preserves successful core
states and every other field, and mixed eager failures preserve typed scalar
errors and the first row-major broadcast index. This gives **27 new bicubic
cases**. **244 focused PG tests pass** at the reported PG checkpoint.

The complete chosen-source run initially records **3,026 passes and one
failure**: an outdated private-resolver test spy did not forward the new
`heat_capacities` keyword and raised `TypeError` before reaching its numerical
or compilation-reuse assertions. The spy now forwards `**options`, and the
entire affected five-test module passes. An AST audit verifies that every
original assertion and the rest of that test module are unchanged; numerical
source remains exactly C3 (`ce94b70d…`). This establishes an audited **3,027
distinct-case verified union**, with zero remaining failures/skips, while
preserving the original failed run rather than relabeling it as one all-green
execution. All **54,272 numerical comparisons** remain: **47,867 finite** and
**6,405 paired unavailable**, with zero failed comparisons.
[Union audit and original-run evidence](../../test_audit/results/phase14/verification_union.json),
[maintained verification](../phase14_verification.json).

All **six affected quick-profile examples** pass their execution/numerical
checks; this is the selected affected set, not a new full-resolution run of all
examples. Sphinx builds **28 content pages with zero warnings**. The final
independent matrix and original cold profiles have completed without numerical
failures. Their performance results above leave the formal gate open.

Original limits remain unchanged: 10% warmed overhead except the already accepted narrow 40%
closed-model PG compiled 1,024-state cohort, with separately declared setup and
first-call allowances. Dynamic models, broadcasts and status-preserving
consumers receive no new exception. ThermOpt remains final Phase 15, after
standalone acceptance, and has not been run in this phase.

## Required follow-up and optional research

The next attribution work must reproduce the **full workload mix** as well as
the native-only controls, then separate allocator/cache pressure, compilation
cache growth, Python dispatch/GC and provider activation/source-call history.
Profile actual allocations and executable/host work; add balanced same-source
controls with the same full registration and execution history. In particular,
inspect REFPROP activation/setup and the unchanged source cases alongside the
compiled consumers. Retain every case/sample and the existing invalid-run
criteria; do not discard the larger matrix shifts as generic ambient noise.

The packed-status consumer result remains a Phase 15 application experiment,
with every status retained and ordinary public full states unchanged. The
[core-certificate research note](certificate_research.md) is explicitly
**unimplemented optional research**: whole-cell polynomial/rounding proofs and
sound dynamic/lifecycle fallback are prerequisites. Neither idea closes the
current gate or permits weaker runtime behavior. Further fixes require the
same complete numerical/AD checks and a repeat of the original performance
matrix; material residual costs need an explicit future decision.
