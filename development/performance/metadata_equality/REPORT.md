# Immutable metadata equality follow-up

ThermOpt's downstream verification exposed a warm JIT-dispatch cost when
equivalent independent bicubic model instances compared their static metadata.
An identical-source control showed a 28.8% second-model penalty; profiling
20 complete polytropic cycles found 13,480 dataclass and 67,400 Mapping equality
calls in the disadvantaged path. This is metadata lookup work, outside the
compiled thermodynamic equations.

`FrozenMapping.__eq__` now recognizes identity and aligned equal frozen tuples
without materializing dictionaries through linear field lookup. Reordered or
changed data, foreign mappings and subclasses retain the general Mapping
comparison. Iteration order, hashing, serialization, validation and numerical
models remain unchanged. Equality uses actual values, never hash equality alone.
There is no global model interning or mutable evaluator sharing.

## Verification

- 17 new tests cover nested/reordered metadata, foreign mappings, subclasses,
  NaN identity, colliding hashes, copying/pickling and real JIT reuse/retracing.
  The allocation/lookup regression failed before the fix.
- The complete HEOS/REFPROP run passes 3,044 tests, initially skipping two optional
  native C reader checks. Both pass in a separate compiler-enabled rerun:
  **3,046 distinct library tests covered, zero remaining skipped cases**.
- ThermOpt's isolated packaged-wheel run passes **396 tests** plus four existing
  strict expected failures. Only `configuration.py` differs from the official
  0.6.0 wheel, and all 43 installed Python source files match the local wheel.
- Fourteen targeted standalone full-state scalar/batch and compiled-consumer
  cases meet their original budgets, comparing the unchanged published 0.6.0
  package with this source in two fresh workers per version.
- The identical-source ThermOpt bicubic polytropic control now has a 1.070
  second/first ratio. All twelve independent-model application comparisons
  meet 1.10 with the packaged fix.

Three balanced before/after wheel cohorts meet 1.10 in nine of ten complete-cycle
workloads; nitrogen bicubic is initially 1.117. Its separate three-pair repeat
uses 36 complete samples and `PYTHONHASHSEED=0`, yielding 0.999 source and 1.004
bicubic ratios. Original samples and the flag remain retained; changing seed,
sample count and measurement time does not establish one cause for the shift.
All six original cohorts and the follow-up pairs have identical physical outputs.
Older absolute ThermOpt checkpoint flags remain separately identified.

## Artifact and reproducibility

This is an **unpublished local test wheel**, retaining the source version 0.6.0.
It must be identified by SHA256, not confused with the official PyPI artifact:

- Official wheel: `0a8dfb68ca4d21cf97b7483e1da776b946cec108a94d50ebb770bb9848127d3c`.
- Local fixed wheel: `bd9d3cdf7ee73ffb096a615f1ed34abd4ced141a18105f09e29c256101e40d71`.

[verification.json](verification.json) retains package identities, unit-test XML
hashes, raw targeted samples, standalone results and interpretation. Detailed
logs, source snapshots, wheels and worker records remain under ignored
`benchmarks/results/metadata_equality/`. ThermOpt retains its application
controls and checkpoint in `benchmark/METADATA_FIX.json`.

The existing phase benchmark runner can reproduce the standalone comparison:
select `frozen current`, point `--frozen-source` and `--frozen-sha256` to the
retained 0.6.0 source, and select HP scalar/batch1024 plus the compressor,
expander and 128-cell exchanger cases for PG and bicubic. Application comparisons
use ThermOpt's maintained benchmark entry points with separately installed wheels.
No new release has been published by this follow-up.
