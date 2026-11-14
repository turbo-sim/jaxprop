# Published-release performance verification

The baseline is the unchanged **0.5.8** wheel resolved from official live PyPI
metadata on 14 November 2026. The [pinned manifest](baseline/baseline_manifest.json)
records the timestamp, upload, dependencies and SHA256. The wheel is retained
under `baseline/`, including for performance follow-up Phase 14 and final
ThermOpt Phase 15. An existing baseline
is never silently advanced by the resolver.

This harness belongs to the benchmark. `api.py` translates each version's public
signatures without altering the wheel or adding adapters to JaxProp/ThermOpt.

The Phase 13 measurement checkpoint is complete. Six warmed workloads and one
first-call profile remain above the original budgets, without user acceptance.
The user requested continued analysis and assessment in a new Phase 14; see the
[report](REPORT.md), [follow-up analysis](FOLLOWUP_ANALYSIS.md) and
[implementation checklist](../../../z_IMPLEMENTATION_PLAN.md#phase-14-assess-and-reduce-remaining-native-jax-performance-costs).
Frozen measurement records and the predeclared acceptance manifest retain their
original contents, including the then-current Phase 14 name for ThermOpt.

## Reproduce

Use the checkout's numerical environment for both independent processes:

```powershell
python development/performance/phase13/resolve_baseline.py --output-dir development/performance/phase13/baseline
python -m pip install --no-deps --no-compile --no-index --target development/performance/results/phase13/published development/performance/phase13/baseline/jaxprop-0.5.8-py3-none-any.whl
python development/performance/phase13_benchmark.py --validate-only --output-directory development/performance/results/phase13/preflight
python development/performance/phase13_benchmark.py --output-directory development/performance/results/phase13/acceptance
```

The public interpreter/dependency environment is shared, while imported package
directories, table caches, JIT caches and worker processes are separate. Every
worker verifies its actual imported path, source hash, precision and device.
Imports and initial JAX device startup are outside setup measurements. First
public calls retain actual compilation/binding costs and are separate from the
warmed samples. Case registration and independent checks must not prewarm those
first calls; source oracles run outside the timed functions.

## Workload and numerical contract

The [acceptance manifest](acceptance.json) is declared before acceptance timing.
The initial matrix includes full states from perfect gas, bicubic, direct source,
callback and shared fixed-composition source-mixture operations. It covers PT,
HP and PS routes, scalar/singleton/small/large batches, broadcast matrices,
homogeneous/saturation/near-critical source regions and available REFPROP.
Complete compiled compressor, expander and discretized heat-exchanger kernels
exercise whole consumers, with numerical outputs and new status retained.
State construction/access/export/collection helpers are measured separately.

Full actual states are always the primary thermodynamic result: the old PG model
computes 13 properties with additional scalar-NaN schema placeholders, while the
new model materializes 22 properties and numerical status. Actual leaves/shapes
are recorded. Complete arrays and unavailable masks are compared, with equal
floating dtypes. Legacy PG scalar/partial-axis coordinates may broadcast in the
explicit broadcast cases for physical-value verification only; their timed
output is never padded. Common-13 projections
are secondary diagnostics. New-only functionality has independent checks and no
fabricated old-version ratio.

The matched PG cohort uses the new public `with_constants` API to reproduce all
nine old reference/transport constants, its actual gamma/R, and its zero-offset
enthalpy convention. The wheel remains unchanged. A separate as-shipped cohort
preserves the intentional calibration/model differences; equal HP inputs do
not imply equal temperature under different energy references.

Source comparisons use independent raw CoolProp oracles and property-specific
absolute tolerances. Pure two-phase optional responses and legacy parcel-model
response errors are not forced into equivalence with their repaired definitions.
Physical PQ quality is compared where common. Generalized quality conventions
are not assumed interchangeable. If a published request fails, retain that
failure and classify a successful new route separately rather than repair the
published package or invent its timing.

## Timing interpretation

Four independent workers per version run serially in ABBAABBA order, nine
samples per warmed case after warmup, with complete-return synchronization and
per-worker samples/ratios retained. Numerical validation runs in separate fresh
processes first. No unrelated user process is stopped for measurements.
Eight untimed warmup calls cap slow-workload repetitions to target 20 ms per
sample, with at least one call. Fast native workloads retain their requested
counts. Actual per-worker repetitions are recorded, times are normalized per
call, and alternating inputs span samples even when a slow case uses one call.

Warmed evaluation/consumer medians target at most 10% slowdown. The explicitly
accepted Phase 7 closed-model compiled PG 1,024-state full-return cases retain a declared
40% envelope; their strict 10% flags stay visible. Construction, loading,
generation, first-call compilation and state-inspection/validation have separate
declared bounds with required new work explained. These do not change the
thermodynamic hot-path target.

First encounter of a wrapper in the shared worker is labeled separately from
true cold compilation. Selected cold public workloads run in fresh independent
processes with no preceding thermodynamic evaluation. Their budgets include
the complete first synchronized public call (tracing, lowering, compilation,
binding and first execution), with JAX device startup excluded. Phase 14 will
decompose these stages in separate diagnostics while retaining this primary gate.

Changing Python-number cases use **unseen** numerical values, distinct from
warmup and independent validation. Their timings include any per-value
specialization/compilation and record compiler events. They are separate from
warmed JAX-array cases and are not described as arithmetic-kernel speedups.

Results apply to CPU/x64 and the recorded installed providers. GPU claims,
ThermOpt application timing and publication remain outside this phase.
