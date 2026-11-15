# JaxProp benchmarks

Run these tools from the repository root in its numerical Python environment.
The suite compares the current checkout with an isolated published package and
the retained Phase 13 source checkpoint. Workloads cover full fluid states,
input pairs, scalar/batch/broadcast shapes, differentiation, compiled consumers,
source/callback/metastable/binary evaluators, and state utilities.

## Prepare the references

```powershell
python benchmarks/prepare.py
```

This restores and verifies the retained PyPI 0.5.8 wheel and frozen source under
ignored `benchmarks/results/`. Both versions share your installed numerical
dependencies. Each timing worker receives a fresh compiler/table cache.

## Run the suite

Choose a new output directory for each invocation. Validation checks every
returned field, shape, status, clipping flag and availability pattern, with
independent numerical checks and the recorded comparison tolerances.

```powershell
python benchmarks/phase14_benchmark.py --suite full --validate-only --workers 1 --samples 12 --output-directory benchmarks/results/preflight
python benchmarks/phase14_benchmark.py --suite full --workers 4 --samples 12 --output-directory benchmarks/results/full
python benchmarks/phase14_cold.py --workers 4 --modes public --output-directory benchmarks/results/cold
```

Warm, setup and first-call measurements have separate allowances. Each complete
return is synchronized. Headline warm timings use the median of worker medians.
Twelve samples balance forward/reverse order and the three numerical variants.
Run numerical jobs sequentially to avoid competing for CPU resources.

On the measured Windows host, `--cpu-affinity 0xffff` selects performance-class
logical processors 0–15 for every warm worker. Check your machine's topology
before selecting a mask. Omitting this option uses inherited scheduling.
Current results and the original limits are documented in
[the visual report](../development/performance/phase14/timing_report.html).

For a shorter development check, use `--suite native` and repeat `--case REGEX`
to select workloads. Registered validators can also compile unselected native
functions. This changes the workload mix and should be labelled as a targeted
check in any timing conclusions.

```powershell
python benchmarks/phase14_benchmark.py --suite native --workers 1 --samples 12 --case '^pg_matched_compiled_HP_scalar$' --output-directory benchmarks/results/pg-smoke
```

## Other tools

| Tool | Purpose |
| --- | --- |
| `phase14_cold.py --modes public stages` | Whole first public calls and separately measured lowering/compilation stages. |
| `phase14_paired.py` | Interleaved native evaluator comparisons within one process. |
| `phase14_attribution.py` | Six-case GC, memory and provider-history diagnostics. |
| `phase13_benchmark.py` | Original two-version matrix and unseen-Python-value checks. |
| `phase11_benchmark.py` | Earlier source-archive component comparison. |
| `phase14_report.py` | Rebuild the saved HTML report from the verified portable measurement record. |

The HTML generator defaults to the saved Phase 14 data. Its `--data`,
`--verification` and `--output` options accept matching portable records using
that schema. Raw timing-run summaries and the portable report input have
different schemas.

For an explicitly retained source checkpoint, the main and paired runners accept
`--frozen-source PATH --frozen-sha256 SHA256`. The paired runner also accepts
`--cpu-affinity MASK` and records the verified process mask with its results.

## Compare with a future PyPI release

Resolve each new baseline into a fresh directory to record its wheel and SHA256.
The resolver retains an existing pin when its directory already has a manifest.

```powershell
python benchmarks/phase13/resolve_baseline.py --output-dir benchmarks/results/new-baseline
python benchmarks/prepare.py --baseline-directory benchmarks/results/new-baseline --output-root benchmarks/results/new-references
python benchmarks/phase14_benchmark.py --suite full --workers 4 --samples 12 --baseline-manifest benchmarks/results/new-baseline/baseline_manifest.json --published-source benchmarks/results/new-references/published --frozen-source benchmarks/results/new-references/frozen --output-directory benchmarks/results/new-comparison
```

Version-specific API translation and numerical conventions live in
`phase13/api.py`. Review those translations and the acceptance limits when a
new release changes the public contract. Source, dependency, wheel and harness
identities are recorded in each experiment manifest.

## Evidence and maintenance

Executable runners and active workload/configuration files live here.
Historical measurements, reports, source/harness archives and their original
configuration snapshots live in `development/performance/`. The preserved
harness archives reproduce the exact older tool layout. Moving these runners
changes their harness hashes for future experiments.

The current source passed 3,029 HEOS/REFPROP tests and all 54,272 audited property
comparisons. Five warm timing flags remain under assessment in the
[implementation checklist](../z_IMPLEMENTATION_PLAN.md).
