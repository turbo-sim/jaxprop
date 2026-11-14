# Perfect-gas and bicubic JAX efficiency

Reviewed on 13 November 2026 using the installed JAX/jaxlib 0.6.2 and Equinox 0.12.2. The goals are fast repeated evaluations with dynamic state values, preserved public state/error behavior, and trustworthy comparisons with the pinned published JaxProp 0.5.8 baseline.

## Findings and implemented changes

### Keep numerical values dynamic at the JIT boundary

Both original `get_state` entry points used `eqx.filter_jit`. Its default filtering treats arrays dynamically and Python numbers statically. Converting a Python float with `jnp.asarray` inside the decorated function does not change its outer compilation key. The audit reproduced three traces for three changing Python enthalpies, versus one trace for equivalent array inputs. Fixed-value timing loops hid this issue. This agrees with the [Equinox transformation contract](https://docs.kidger.site/equinox/api/transformations/).

The numerical model objects already form JAX-compatible PyTrees with static identity/configuration metadata. Both native evaluators now use `jax.jit`, with only the property-pair selector explicitly static. State values and numerical model parameters remain dynamic. List/array conversion happens inside the traced function. This removes unintended per-value compilation and avoids an extra Python normalization/method-binding layer. Dynamic model constants and bicubic coefficients remain tested compiled inputs.

Native JIT also avoids the method-binding and input-conversion overhead of an additional public wrapper around a private filtered kernel.

### Avoid redundant state outputs

The shared state constructor still applies explicit masks, nonfinite-value checks, and unsuccessful status before storing values. If the original validity is $v$ and storage is $y=\mathrm{where}(v,x,\mathrm{NaN})$, then $\mathrm{isfinite}(y)\land(\mathrm{status}=0)$ recovers the original property validity. Separate persistent boolean arrays duplicate that information.

`FluidState` now stores masked numerical values and status, materializing validity on demand. All storage backends share the same tree structure, so `lax.cond` and `lax.scan` can combine NumPy-constant and traced states. Eager raising access inspects concrete values/status directly, preserving informative failing indices without dispatching a small JAX mask kernel just to synchronize it. Explicit JAX mask access and complete state export use fused array-only JIT helpers. Serialization still exposes the complete explicit-mask schema and preserves round trips.

For current full-state evaluators, this reduces bicubic array outputs from 77 to 39 and perfect-gas outputs from 27 to 14. Published perfect-gas array counts depend on the input path: with dynamic array inputs it returns 13 arrays; with Python scalars some input fields become static constants. This distinction must remain visible in comparisons.

### Simplify bicubic scalar execution

Scalar calls directly evaluate the scalar kernel. Batch calls retain broadcast normalization, flattened `vmap`, and shape restoration. The redundant nested filtered JIT on the private scalar dispatcher was removed. No interpolation equations, input-pair routing, solver tolerances, reference conventions, or table schema were changed in this pass.

## Numerical implementation review

The perfect-gas equations are vectorized JAX arithmetic. Inspected optimized full-state and temperature-projection HLO contains no host callbacks. The temperature projection removes unused pressure, density, entropy, and transport work. Source-level repetition of simple heat-capacity expressions is therefore not evidence that an algebraic rewrite will improve runtime; fusion and dead-code elimination already remove substantial unnecessary work.

Bicubic forward kernels are compiled and fused, with no host/CoolProp callbacks in the inspected optimized HLO. Full-state output bookkeeping and per-call model-tree handling were material costs. A selected-temperature projection produces one array and prunes unused property calculations.

Inverse calculations use compiled Optimistix control flow. The local dependency source shows that `throw=True` uses Equinox's conditional error branch: failed solves invoke a callback to report failure, while successful solves take a device-side pass-through branch. There are no per-iteration CoolProp/Python calls. Keep this distinction when describing callback-free forward kernels or measuring inverse failure behavior.

The final native JIT entry points support JAX asynchronous dispatch. Benchmark complete outputs with explicit synchronization, and validate the intended eager/transformed failure-delivery policy as the common API is completed. For peak throughput, compile compatible complete consuming calculations; the [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html) recommends outermost JIT and synchronized timing.

## Tests

This pass adds 23 regression cases:

| Tests | Cases | Coverage |
| --- | ---: | --- |
| [Perfect-gas execution](tests/test_perfect_gas_jax_execution.py) | 5 | Changing numeric values reuse compilation; lists; broadcasting; full-state outer JIT; derivatives; dynamic model constants; vectorization. |
| [Bicubic execution](tests/test_bicubic_jax_execution.py) | 6 | Changing numeric values reuse compilation; Python/NumPy/JAX scalars; lists/singleton/broadcast shapes; full-state outer JIT; derivatives. |
| [Lazy state validity](tests/test_state_lazy_validity.py) | 12 | Explicit masks/nonfinite values/failed status/missing values; cond/scan; captured NumPy data with traced status; serialization; error indices; unsupported properties; collection gradients; read-only/empty NumPy masks. |

Perfect-gas list input now satisfies its shared contract case, removing one strict expected failure. Vectorized floating calculations can differ by a final rounding bit; the new execution comparison uses relative tolerance $10^{-13}$, while status/validity comparisons remain exact. Existing independent numerical-oracle tolerances are unchanged.

The final native-JIT full HEOS + REFPROP run completed with **936 passed, 78 strict expected failures, zero unexpected failures, and zero skips** in 339.21 seconds. It retains all **84,950 numerical comparisons**, including 50,189 finite values and 34,761 explicitly unavailable values, with zero failed comparisons. [Summary](development/test_audit/results/jax_efficiency/full_20261113_094055/summary.json), [pytest log](development/test_audit/results/jax_efficiency/full_20261113_094055/pytest.log), [JUnit results](development/test_audit/results/jax_efficiency/full_20261113_094055/junit.xml). Existing static-reference and input-validation warnings remain assigned to later phases.

The [compact performance checkpoint](development/performance/README.md) and [portable case summary](development/performance/summary.json) retain final timings, variability, environment details, and source/distribution hashes. Disposable intermediate benchmarking scripts, workers, packages, profiles, and compiler dumps were removed after preserving that record. The [implementation plan](z_IMPLEMENTATION_PLAN.md) tracks the complete release gate separately.

## Final matched timings

Three independent workers per implementation ran 37 comparable workloads, with synchronized JAX results and matched CPU/x64/dependency/thread settings. The comparison used the unchanged published PyPI 0.5.8 wheel; its identity/hash is retained. All 42 sampled value/shape groups passed, and both sampled derivatives matched exactly. [Results and method](development/performance/README.md), [all case results and variability](development/performance/summary.json).

| Full-state workload | PyPI | Final | Result |
| --- | ---: | ---: | --- |
| Perfect-gas scalar | 156.6 us | 38.1 us | About 4.1 times faster |
| Perfect-gas batch, 1,024 | 273.2 us | 72.7 us | About 3.8 times faster |
| Bicubic scalar | 365.6 us | 157.9 us | About 2.3 times faster |
| Bicubic batch, 1,024 | 1,297.1 us | 1,280.3 us | Approximately comparable |

Compiled temperature/derivative projections and the compiled compressor remain broadly comparable. Changing Python scalar inputs improve from 28.82 ms to 79.4 us for perfect gas and 106.77 ms to 278.4 us for bicubic, chiefly because the original implementation recompiles for each numeric value. Those ratios include compilation avoidance rather than arithmetic alone.

The earlier native-evaluation regressions are resolved in this CPU workload. Cached loading remains slower at 31.6 versus 22.6 ms (+40%); complete validation remains enabled. The full device/backend/application performance gate remains open. Larger loading/evaluator/state ideas remain assigned to phases below.

## Larger follow-ups by phase

- **Phase 4 (implemented):** Validate immutable construction/capability metadata and static requests while preserving dynamic values, native JIT, compilation reuse, and selected-property pruning. Remaining trusted internal construction or packed-output ideas are tracked in Phase 11 and require profiling first.
- **Phase 5 (implemented):** Replace array-valued static critical/reference states with immutable primitive source-reference records. Explicit access constructs read-only NumPy states; native evaluation does not initialize sources. HEOS/REFPROP source and JAX checks verify the former bicubic static-array warnings are gone.
- **Phase 7:** Complete perfect-gas properties, physical references, and failure/domain behavior while preserving native JIT/vectorization and projection pruning. Avoid speculative algebraic rewrites without compiler/timing evidence.
- **Phase 8:** Build a lean runtime evaluator separate from full cache/inspection data; evaluate grouped coefficients/shared gathers; avoid full state assembly in selected-property inverse residuals; precompute reusable coarse-grid work safely. Treat a one-dimensional fixed-pressure entropy solve as a numerical change needing independent robustness/derivative validation. Profile NumPy-first loading/bulk placement while retaining cache provenance and integrity checks.
- **Phase 11:** Give consumers control over device/precision instead of forcing CPU at import. Measure GPU host transfers, eager availability checks, mask materialization, serialization, and concurrency separately.
- **Phase 14:** Compile compatible complete ThermOpt component/cycle/objective functions with reusable models and dynamic numerical inputs, then verify actual application behavior and trace counts.
- **Phase 13:** Include changing Python scalars, array inputs, static selectors, dynamic parameters, full-state versus projected outputs, explicit validity/export, larger batches, and whole consumers. Keep setup/compilation/evaluation separate and retain required validation. CPU measurements do not establish GPU behavior or complete the formal release gate.

Each item is tracked in its corresponding phase; these broader changes remain unimplemented until its numerical, behavioral, and performance checks pass.
