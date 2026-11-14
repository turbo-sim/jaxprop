# Phase 14 performance follow-up

Phase 13 measurements are complete; performance acceptance remains open. The
[validation record](../phase13_validation.json) reports 105/111 warmed cases
within their declared budgets, all 22 setup comparisons within budget, six
warmed flags and one cold-call flag. Numerical comparisons passed, and the
final suite passed 2,970 tests with 54,272 property comparisons. Phase 14 will
investigate these shortcomings; the final ThermOpt comparison is Phase 15.
These results
compare unchanged PyPI 0.5.8 with source `9fc27b79041931d98cbcb8349cdab7d81fe31000a49cd3fe14dc7cd85ebe3651`
on CPU/x64; they do not establish GPU or ThermOpt performance.

## Remaining measurements

Times below are medians of four independent worker medians, in microseconds.
The full matrix used nine samples per worker; the targeted repeat used 30.
Both runs retained complete actual state returns and real current statuses.

| Workload | Full matrix old → current | Slowdown | Targeted repeat old → current | Slowdown |
|---|---:|---:|---:|---:|
| Dynamic-model PG HP, 1,024 states | 83.960 → 95.334 | 13.5% | 108.659 → 141.202 | 29.9% |
| Bicubic HP broadcast, 8 × 16 | 84.086 → 92.988 | 10.6% | 135.431 → 156.086 | 15.3% |
| PG compressor | 9.013 → 11.538 | 28.0% | 11.262 → 15.717 | 39.6% |
| PG expander | 8.565 → 11.196 | 30.7% | 11.225 → 15.823 | 41.0% |
| PG heat exchanger, 128 cells | 12.094 → 16.780 | 38.7% | 16.799 → 23.274 | 38.5% |
| Bicubic heat exchanger, 128 cells | 41.643 → 52.097 | 25.1% | 62.531 → 86.105 | 37.7% |

The targeted dynamic-PG and broadcast workers drift substantially: their old
worker medians range from 69.777–179.639 and 81.140–209.763 microseconds,
respectively, with overlapping current ranges. The flagged measurements warrant
follow-up; their percentages are not stable intrinsic penalties. Preserve raw
worker distributions and host-load observations when evaluating improvements.

True-cold observations used two fresh workers per version and case, with fresh
compilation caches and no preceding thermodynamic evaluation. These measure the
**whole first synchronized public call**, including binding, tracing, lowering,
compilation and execution; device startup and model construction are excluded.

| Cold public workload | Old → current, ms | Declared limit, ms | Result |
|---|---:|---:|---|
| PG HP, 1,024 states | 42.075 → 184.270 | 163.113 | Above budget |
| Bicubic HP, 1,024 states | 231.195 → 316.684 | 446.793 | Within budget |
| PG heat exchanger, 128 cells | 59.659 → 127.501 | 189.489 | Within budget |
| Bicubic compressor | 906.993 → 523.448 | 1,460.489 | Within budget |

## What the retained graphs show

The following counts come from optimized HLO files under
`development/performance/results/phase13/targeted_repeat/workers/baseline1`
and `current1`. The [raw evidence archive](raw_results.json.gz) records their
normalized graph hashes and harness summaries, rather than the HLO text.
Each pair is **ENTRY computation
instructions / total instructions across all computation definitions**.
Totals include fusion bodies and both conditional branches; they are static
graph counts, not executed operations, kernel launches, or proportional timing
estimates. This table includes `ROOT` instructions; the harness summary's
`instructions` counter excludes those lines and therefore reports lower totals.

| Workload | Old ENTRY / total | Current ENTRY / total |
|---|---:|---:|
| Dynamic PG HP, 1,024 | 54 / 161 | 51 / 897 |
| PG compressor | 9 / 55 | 16 / 669 |
| PG expander | 9 / 65 | 15 / 973 |
| PG heat exchanger | 11 / 76 | 35 / 594 |
| Bicubic HP broadcast | 108 / 299 | 96 / 3,011 |
| Bicubic heat exchanger | 21 / 260 | 42 / 677 |
| Bicubic HP flat 1,024 | 86 / 258 | 99 / 703 |

- Both versions' PG compressor and expander retain three logarithm nodes and
  one exponential node. Both PG heat-exchanger graphs contain no logarithm or
  square-root nodes. Unused entropy/transport work already disappears from the
  latter consumer; removing public properties is not the supported diagnosis.
- Current bicubic broadcast retains 323 `is-finite`, 303 `select` and 118
  `clamp` nodes, versus 36, 62 and 22 for current flat 1,024 states. Both retain
  21 `dot` nodes. This points to policy/broadcast fusion duplication as a useful
  investigation, rather than establishing slow cubic arithmetic.
- Bicubic heat exchange retains two old versus eight current `dot` nodes.
  Current status checks retain four interpolated core fields per stream
  (temperature, density, entropy and internal energy); enthalpy/pressure are
  direct coordinates. That required failure detection is distinct from
  avoidable duplicate interpolation or masking.
- PG's outer all-success branch calls `_from_native` on failure, where another
  all-success reduction remains. Bicubic request policy and `_cell_basis`
  separately protect/locate coordinates; consecutive clamps survive in the
  heat-exchanger HLO. These are concrete investigation points, not demonstrated
  wall-time fixes.

The `final_acceptance` entry layouts also explain genuine output/interface
differences. PG machine consumers add three int32 statuses: logical output size
24 → 36 bytes. Both heat exchangers add two int32[128] statuses: 1,048 → 2,072
bytes. Old PG heat exchange prunes pressure entirely; current pressure is needed
for status. Full PG 1,024-state output changes from 13 floating arrays plus 25
scalar NaN placeholders to 22 floating arrays plus status: 106,696 → 184,320
logical bytes. These are logical payload sizes, not unique allocations or
measured memory traffic. Dynamic PG also has seven old versus ten current
numerical scalar model inputs: current T_ref, P_ref and h_ref remain dynamic.
The same viscosity update therefore exercises a broader current model contract.

## Prioritized experiments

1. **Attribute required work before changing policy.** Compare complete current
   states, real consumers with status, and separate diagnostic projections that
   retain only numerical outputs or status. Also diagnose the matched dynamic
   parameter subset versus the actual broader current model. Such ablations
   explain marginal costs only: they must never replace full-state/status
   acceptance workloads, fabricate legacy status, or narrow the public API.

2. **Share PG policy intermediates without freezing model values.** Investigate
   reuse of cp/cv, representability limits, safe coordinates and classification
   through resolution, assembly and masking. Isolate the redundant inner
   reduction from unrelated threshold changes, and check actual optimized
   fusion/copy changes. Derived values must remain functions of dynamic
   parameters, with correct model derivatives; construction-time static caching
   cannot silently make them stale.

3. **Investigate bicubic batch assembly and shared coordinates.** Trial one
   protected broadcast-coordinate/cell/basis context, vectorized raw
   interpolation, then one batch state assembly instead of scalar state
   assembly inside `vmap` followed by reshaping. Compare flat 128, 8 × 16,
   16 × 8, singleton axes, scalar pressure and flat 1,024. A harder optional
   study is a conservative whole-cell polynomial proof of finite/positive core
   interpolation, including dtype and tolerated boundary slack. Positive nodes
   or finite coefficients alone are insufficient: interiors can undershoot or
   overflow. This is research, not an immediate removal of checks. Preserve
   phase/domain/inverse checks and a guarded fallback for uncertified or changed
   coefficients/dtypes/model views; do not hash coefficients or query sources
   every call, or freeze dynamic parameters to keep a certificate valid.

4. **Decompose cold-call cost diagnostically.** In fresh processes measure host
   binding/tracing, lowering, backend compilation and first execution, alongside
   the unchanged whole-call metric. Expanded graph totals suggest investigation
   but do not identify the dominant stage. Preserve static pair/policy selectors,
   dynamic numbers/model parameters and current success with unseen Python
   numerical values. Do not describe 184.270 ms as pure backend compilation.

## Invariants and completion gate

Each candidate must preserve all 22 PG fields and complete bicubic states,
actual shapes/dtypes, property availability and NaN patterns, exact status
precedence, clipping flags and indexed eager errors. Keep optional unavailable
data from contaminating otherwise valid derivatives. Exercise mixed valid,
nonfinite, nonpositive, out-of-domain and overflow rows, including zero failed
tangents/cotangents without reverse-mode NaN leakage.

Controls must include PT/HP/PS, scalar/array pressure and sizes
1/16/63/64/65/128/1,024, complete states and compressor/expander/heat-exchanger
consumers. Check input and model JVP/VJP against analytical/finite-difference
oracles, including R, gamma, h_ref, reference/transport parameters and bicubic
coefficients. Verify changed parameters alter results without needless retracing.

Retain candidates only after matched independent-worker timings improve the
target while preserving full-state controls and passing meaningful tests.
Repeat the full suite and published-release matrix before closing Phase 14.
Warmed consumers/dynamic/broadcast calls keep the 10% target; the existing 40%
exception remains restricted to previously accepted closed-model PG 1,024-state
full returns. Cold calls retain the declared 1.5× old + 100 ms bound. Do not widen
budgets after observing results.

## Experiments already rejected

The [PG investigation](pg_policy_investigation.json) retains exploratory evidence:

- Int32 status intermediates removed unnecessary int64 selections and were
  retained, but showed no reliable wall-time gain.
- Raising the PG threshold to 256/512 while disabling the inner factory fast
  path reduced some graph branches but worsened full-state timings; it was
  reverted. Both cutoffs produced the same programs at the tested sizes
  1/128/1,024, so this was not an independent cutoff-tuning result.
- `jax.jit(inline=True)` gave no useful consumer graph reduction and generally
  increased full-state/consumer times; it was rejected.

Do not repeat these combined changes without a distinct, testable hypothesis.
Smaller graphs alone are insufficient evidence of an improvement.
