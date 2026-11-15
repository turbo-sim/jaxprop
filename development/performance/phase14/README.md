# Phase 14 native performance investigation

The Phase 13 checkpoint was committed as `b282ff0`. This phase investigates its
remaining costs without changing public state contents, failure policies,
derivatives or the original [acceptance limits](../phase13/acceptance.json).
ThermOpt remains final Phase 15.

Reusable runners, workload modules, baseline preparation and run instructions
are now in [benchmarks](../../../benchmarks/README.md). Run
`python benchmarks/prepare.py` before a new comparison. The configurations and
archives retained here describe the recorded historical experiments.

The [visual timing report](timing_report.html) includes bar plots, every
comparable timing ratio, setup/first-call costs and searchable measurements.
Its figures and case data are embedded for offline viewing. Regenerate it with
`python benchmarks/phase14_report.py` from the project environment.

The latest [native reduction follow-up](reduction_followup.md) covers source
`d0fa73ec…`. Large bicubic batches use fused polynomial reductions, scalar/small
requests keep their dot kernels, and redundant PG predicates are removed.
The 61-case native comparison retains two PG scalar consumer flags. Its
[measurements](reduction_validation.json) and
[full verification](reduction_verification.json) record the scoped results.

The earlier [follow-up report](followup_20261115.md) covers source `6f93cb82…`:
shared PG property tracing, passing original cold gates, and a matched-placement
matrix with five of 119 warmed cases still above budget. The
[attribution report](mixed_attribution_20261115.md) explains the observed core
context and provider history; the [portable measurements](followup_validation.json)
retain the raw samples. The original [C3 report](REPORT.md) and unrestricted
17-flag observation remain historical evidence.

The [correctness record](followup_verification.json) preserves the complete
3,029-test run and all 54,272 numerical comparisons. Bulk raw vectors and
compiler output remain in ignored results; the maintained
[archive manifest](followup_evidence_manifest.json) records their location and
integrity hashes. The compact portable measurements remain directly readable.

Three distinct implementations are compared: the unchanged pinned PyPI 0.5.8
wheel, the exact Phase 13 package (`9fc27b79…`), and each candidate's recorded
source hash. [frozen_source.zip](frozen_source.zip) retains the checkpoint;
its SHA256 is `202f3c8dadfa920ab1876b9b4f2da97b237607d6f0a2c0ff54efff41a995b6ad`.
Unpack it under `development/performance/results/phase14/frozen/` to reproduce
the default frozen import root.

The [helper source archive](helper_source.zip) retains the latest numerical
package (`6f93cb822c058f40fe1e1503f7a92dd0ee185de0eb4b84014c24e919251cec56`);
archive SHA256 is `319a93f6473c0b283c1b898cee33030b0114d1c764911c33f6b4d99ab34dc7e0`.
The pre-affinity harness is retained in
[harness_20261115_pre_affinity.zip](harness_20261115_pre_affinity.zip), alongside
the earlier [harness snapshot](harness_before_20261115.zip). They preserve the
actual code used by the earlier experiments.

The [protocol](protocol.json) declares fixed, identical repetitions, balanced
worker/case orders, complete-return synchronization, fresh compilation caches
and numerical requirements. Candidate/frozen checks compare every field,
status, clipping flag, shape, dtype and availability pattern. Legacy comparisons
retain the separately documented physical-model and reference conventions.

```powershell
python benchmarks/phase14_benchmark.py --output-directory development/performance/results/phase14/native --workers 4 --samples 30
python benchmarks/phase14_benchmark.py --output-directory development/performance/results/phase14/full --suite full --workers 4 --samples 12
python benchmarks/phase14_cold.py --output-directory development/performance/results/phase14/cold --workers 4 --modes public
python benchmarks/phase14_paired.py --output-directory development/performance/results/phase14/paired --models pg bicubic --samples 30
```

The [follow-up protocol](followup_protocol.json) declares the controlled Windows
run. Its full command adds `--cpu-affinity 0xffff` on the observed machine, whose
performance-class logical processors are 0–15. Determine the actual topology
before selecting a mask on another host. The option narrows only each benchmark
worker before numerical imports; readback is verified and recorded for all
versions. Omission preserves inherited placement. The package itself sets no
affinity or machine power policy. Twelve samples balance both case directions
and all three numerical variants; the original nine-sample records remain.

The separate `phase14_attribution.py` runner records full preparation, GC,
Windows counters and provider predecessors for a declared six-case diagnostic.
Its memory/disk arms test live snapshot retention; neither demonstrated a
snapshot-release speedup. Keep its declared scope: wildcard/current-only
aggregation limitations are documented in the attribution report. It is not a
replacement gate.

Every invocation needs a new output directory. Raw workers, tables and compiler
files belong under ignored `results/`; retained summaries must identify their
actual candidate and harness hashes. Diagnostic projections and same-process
experiments explain costs; the public full-state/status workloads remain the
acceptance gates.

Cold public calls and separate tracing/lowering, executable compilation and
first-execution probes use different fresh processes. Do not sum the latter and
present them as the original public-call measurement. Ordinary shared-worker
first encounters may have related inner kernels warm.

Initial source/IR inspection identifies duplicate bicubic broadcast policy work
and PG nested all-success branches. The first native bicubic batch prototype
also exposed a generic batch conditional that copied successful result arrays;
the second candidate removes that new branch while preserving per-row masking.
Smaller IR alone does not establish a speedup. Identical PG consumer programs
have materially different timings between workers, so distributions and paired
diagnostics are needed before assigning cause.

The paired diagnostic loads unchanged package files under separate module
names in one JAX process and interleaves versions using the same input buffers.
PG includes the published implementation. Bicubic compares only frozen/current:
the legacy implementation has an absolute import of its state constructor, so
alias-loading it would alter its timed behavior. Current/frozen bicubic source
helpers are verified identical and used only at setup; their native evaluator
classes and complete states remain distinct. The independent-process wheel
comparison remains authoritative.
