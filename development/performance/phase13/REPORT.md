# Phase 13 published-release comparison

**Measurement checkpoint complete; unresolved performance acceptance moves to Phase 14.** On 14 November 2026 the user requested continued analysis and a new phase to assess the shortcomings. The remaining costs have not been accepted and the original budgets are unchanged. Phase 12 was committed as `f1198fd`. The pinned official PyPI baseline is **0.5.8**, downloaded on 14 November 2026 with SHA256 recorded in its retained manifest. The current tested source is `9fc27b79…`.

All **2,970 HEOS/REFPROP tests pass**, zero failures/skips/expected failures; all **54,272 numerical comparisons** remain. The full performance matrix checks **1,394,944 finite values** and the longer repeat checks **325,728**, with zero numerical mismatches. Eight independent serial workers cover **111 comparable workloads**, plus four explicit current-only capabilities. Complete actual returns are synchronized; selected physical fields are verified as full arrays. All 22 setup comparisons pass their separately declared bounds.

The [portable record](../phase13_validation.json) includes per-case/raw worker samples, paired ratios, unchanged declared budgets and all remaining flags. [Compressed full evidence](raw_results.json.gz) preserves detailed numerical checks, payloads, setup, first calls, graphs and actual unique-input results. The [baseline manifest](baseline/baseline_manifest.json), wheel and metadata remain pinned for the Phase 14 performance follow-up and final ThermOpt Phase 15.

## Primary complete-state results

Full-matrix medians are microseconds per public call or indicated batch. The current PG backend returns all 22 materialized properties and status. The published backend retains its 13 computed properties and original placeholders. Matched parameters use the current public calibration API; ordinary as-shipped calibration cases are recorded separately.

| Workload | PyPI | Current | New / old |
| --- | ---: | ---: | ---: |
| PG eager scalar | 139.40 | 42.73 | 0.307 |
| PG compiled scalar | 46.35 | 29.77 | 0.642 |
| PG eager batch, 1,024 | 228.23 | 119.99 | 0.526 |
| PG compiled batch, 1,024 | 67.80 | 72.20 | 1.065 |
| Bicubic eager scalar | 317.42 | 71.20 | 0.224 |
| Bicubic compiled scalar | 56.90 | 47.11 | 0.828 |
| Bicubic eager batch, 1,024 | 1111.17 | 635.18 | 0.572 |
| Bicubic compiled batch, 1,024 | 527.71 | 546.24 | 1.035 |

The longer repeat uses a sustained native workload mix: PG full batches of 1,024 take 89.48→121.63 µs (+35.9%, within the explicitly preaccepted 40% cohort), and bicubic batches take 713.88→754.06 µs (+5.6%). Both runs remain visible; timings depend on workload mix.

The public dictionary of 13 common raw properties improves from the initial current implementation's 539.33 µs to 20.10 µs, within its predeclared absolute inspection allowance. Complete-state indexing improves from 906.88 µs to 44.10 µs, versus the published version's 642.64 µs. All 18 new compatibility tests and existing state tests pass; dynamic bound-method PyTrees, signatures, pickle, NumPy immutability, failures and clipping are preserved.

Changing 12 previously unseen Python inputs causes 12 compilations per published native backend and zero additional current compilations after warmup. These are including-compilation diagnostics, not arithmetic speedups; complete resulting states and inputs were checked. Broad eager/source/callback and inverse cases are mostly faster; individual case results remain in the record.

## Remaining budget deviations

The full run passes 105 of 111 warmed budgets. A longer repeat (four workers/version, 30 samples, same cases/code/tolerances) retains all six aggregate flags. Repeats below are microseconds per operation; no budget was relaxed. Dynamic-PG and bicubic-broadcast paired worker ratios span below and above parity, so their aggregate percentages are not established intrinsic kernel penalties. Both full-matrix and sustained-repeat records remain visible.

| Workload | PyPI | Current | Slowdown |
| --- | ---: | ---: | ---: |
| `pg_matched_dynamic_model_HP_batch1024` | 108.66 | 141.20 | 29.9% |
| `bicubic_CO2_compiled_HP_broadcast8x16` | 135.43 | 156.09 | 15.3% |
| `pg_matched_compiled_compressor` | 11.26 | 15.72 | 39.6% |
| `pg_matched_compiled_expander` | 11.23 | 15.82 | 41.0% |
| `pg_matched_compiled_heat_exchanger_128cells` | 16.80 | 23.27 | 38.5% |
| `bicubic_CO2_compiled_heat_exchanger_128cells` | 62.53 | 86.11 | 37.7% |

The new compiled consumers retain actual numerical status and failure masks; the published package has no status output. Existing HLO shows real finite/domain/status work after unused properties are pruned. This identifies added work, without proving its elapsed-time contribution or waiving the 10% budget. Integer-status intermediates now use int32; rejected batch-threshold and inlining experiments were reverted because they did not give reliable gains and sometimes slowed full states.

One of four cold first-public-call profiles exceeds its separate bound: PG HP batches of 1,024 take 42.08→184.27 ms, against 163.11 ms allowed. This synchronized call includes tracing, lowering, compilation and first dispatch/execution; it is not an isolated XLA compilation measurement. This cost occurs once per compiled signature; changing numerical values reuses compilation. Other cold profiles and all 22 setup comparisons pass. Cold processes use fresh independent caches, one selected thermodynamic workload and compact Windows paths. The initial deep benchmark paths were corrected without changing library numerics.

The user selected a new assessment phase. [Phase 14 in the implementation plan](../../../z_IMPLEMENTATION_PLAN.md#phase-14-assess-and-reduce-remaining-native-jax-performance-costs) now owns the remaining gates. The [continued analysis](FOLLOWUP_ANALYSIS.md) prioritizes shared PG policy dataflow, bicubic coordinate/broadcast reuse, dynamic-model work and cold-stage attribution, with numerical/status/AD invariants and measured controls. No additional production rewrite or speedup is claimed by this planning update. Final ThermOpt correctness/performance moves to Phase 15, under `thermopt/development/bicubic/`, and has not been run here.

## Scope and reproducibility

See [commands and semantic cohorts](README.md), [predeclared acceptance](acceptance.json), [full-suite record](../../test_audit/results/phase13/full_20261114_153004/summary.json), [state-helper investigation](state_helper_investigation.json) and [rejected policy experiments](pg_policy_investigation.json). Reference-quality and source gas-constant differences are explicit. The PyPI wheel is unmodified and installed separately; every worker verifies its imported package, source, dependency/device/precision/thread environment and wheel bytes.

No unrelated user processes were stopped. Host observations and every worker distribution remain available; absolute timing varies with workload mix. No GPU/application/publication or hosted-CI acceptance is inferred.
