# Phase 14 mixed-workload timing attribution

The final matched-placement full workload passed every numerical comparison and retains five timing flags across 119 comparable cases. Its 22 setup gates and four separate fresh-process public-call gates pass their original formulas. The performance gate remains open; no budget waiver or user acceptance is inferred. The [portable follow-up validation](followup_validation.json) preserves every comparable timing sample, worker graph identity, setup observation, cold profile and experiment identity.

The original mixed-workload result remains evidence: 17 timing flags across 119 comparable cases. The instrumented experiments below diagnose that result; they do not change a limit, remove samples, divide timings by a control, or establish that all historical shifts were caused by scheduling. The original workload was repeated with 12 balanced samples and four independent workers per version, retaining its complete workload and normal provider switching while allowing every version's worker the same processor class.

The diagnostic compares the frozen Phase 13 source, SHA-256 `9fc27b79041931d98cbcb8349cdab7d81fe31000a49cd3fe14dc7cd85ebe3651`, with the earlier Phase 14 C3 source, SHA-256 `ce94b70d2ddc89fe602a52a2fac8fb71ff5e7f095588431e7d6a5e065611cde3`. The C3 source is copied under `results/phase14/followup_20261115/c3`; this experiment must not be described as a measurement of a later candidate.

The final full workload and controlled crossover use the later source SHA-256 `6f93cb822c058f40fe1e1503f7a92dd0ee185de0eb4b84014c24e919251cec56`. The published reference remains the unmodified pinned PyPI 0.5.8 wheel.

## What the original workers show

The [original full matrix](../results/phase14/full_matrix/summary.json) has a strong time-dependent effect. For the compiled PG compressor, worker medians in microseconds were:

| Source | Worker 1 | Worker 2 | Worker 3 | Worker 4 |
| --- | ---: | ---: | ---: | ---: |
| Published | 11.454 | 10.100 | 9.273 | 12.327 |
| Frozen | 16.942 | 15.629 | 14.887 | 14.565 |
| C3 | 15.502 | 15.496 | 30.102 | 49.023 |

The C3 expander likewise changed from 15.154 and 15.200 to 29.654 and 42.919 microseconds. C3 compressor worker 4 retained samples from 12.115 to 127.321 microseconds; those samples are not discarded. Frozen/current scalar compressor and expander programs had previously been verified equivalent after normalization of compiler naming, so this large shift is not evidence of additional scalar mathematical operations.

The affected original timing windows were 17:34:26–17:34:55 UTC for C3 worker 3 and 17:36:06–17:36:34 UTC for C3 worker 4 on 14 November 2026. Published worker 4 was also affected at 17:37:44–17:38:23; frozen worker 4 recovered at 17:39:34–17:39:52. This establishes nonstationarity in the run. It does not identify external load, CPU frequency, processor type, garbage collection, or a particular background process as its cause.

The original [identical-source control](../results/phase14/null_control/summary.json) produced current/frozen ratios of approximately 1.003–1.085 across its six cases. That control does not explain every 20–50% full-matrix shift. The original workers lacked usable CPU, working-set and GC observations, so causal attribution cannot be reconstructed from those records alone.

## Provider and predecessor history

REFPROP scalar cases have two reproducible history modes. In the original frozen worker 2, Water liquid PT alternated approximately 3.8–4.1 milliseconds in forward blocks and 0.47–0.56 milliseconds in reverse blocks. CO2 supercritical HP had the same pattern. The immediately preceding cases explain which mode is exercised:

| Timed case | Forward predecessor | Reverse predecessor |
| --- | --- | --- |
| REFPROP Water liquid PT | CO2 near-critical PT | Water gas PT |
| REFPROP CO2 supercritical HP | Air gas PT | CO2 supercritical PS |

Forward blocks switch fluid; reverse blocks reuse the fluid. There are three numerical input variants, so this two-block alternation is not input-variant parity. Nine samples give five forward and four reverse observations, allowing the median to select the slower mode. Twelve samples give six of each and balance all three variants against both orders. Switching remains inside the measured public call. Mode-specific medians should accompany the aggregate when diagnosing a bimodal case: an aggregate median between two modes is not a measurement of either mode's latency.

The new diagnostic also exposes a predecessor effect for bicubic broadcast. Its forward predecessor is the eager calculation of the same HP broadcast; its reverse predecessor is the compiled full-state calculation with $dT/dh$. The effect occurs in frozen and C3. It warrants preserving the workload order and reporting both modes, rather than treating every sample as an independent measurement of an isolated kernel.

Another preparation issue matters when interpreting earlier targeted controls. The original filtered worker registers all native and source validators, filters timed cases, then still executes every validator. Several validators invoke or compile public functions for unselected cases. Consequently, a short timed selection is not necessarily a small compilation-cache or heap experiment. The diagnostic explicitly distinguishes full preparation from selected-case preparation; the original runner is unchanged.

## Full preparation with live verification snapshots

The [a_mem experiment](../results/phase14/a_mem/summary.json) uses full original preparation, the six declared timed cases, normal GC, live common/complete snapshots, original immediate-predecessor replay outside each selected timer, 12 samples, and two workers per version in frozen/current/current/frozen order. Fixed repetitions are 48 scalar calls, 16 batch calls and one source call per sample. It checks selected frozen/current complete output fields, status, clipping, shape and availability; correctness passed. It runs all preparation validators. The original full prepared snapshot sidecar is retained.

| Case | Frozen median, microseconds | C3 median, microseconds | C3/frozen |
| --- | ---: | ---: | ---: |
| PG compiled HP batch 1024 | 81.091 | 88.197 | 1.088 |
| Bicubic compiled HP broadcast 8 × 16 | 115.295 | 103.048 | 0.894 |
| PG compiled compressor | 11.052 | 10.870 | 0.984 |
| PG compiled expander | 10.769 | 10.789 | 1.002 |
| REFPROP Water liquid PT | 1674.025 | 1575.075 | 0.941 |
| REFPROP CO2 supercritical HP | 1671.300 | 1690.300 | 1.011 |

PG batch worker ratios differ materially: C3 worker 1/frozen worker 1 is 1.174, whereas C3 worker 2/frozen worker 2 is 1.007. Two workers do not establish a stable 8.8% source regression. Conversely, this diagnostic also cannot certify that the batch is faster under the full acceptance workload.

The observed history modes, in microseconds, were:

| Worker | Bicubic forward | Bicubic reverse | Water forward | Water reverse | CO2 HP forward | CO2 HP reverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen 1 | 82.5 | 132.4 | 2573.1 | 400.3 | 2613.6 | 406.8 |
| C3 1 | 88.3 | 130.9 | 2477.1 | 389.8 | 2564.4 | 425.0 |
| C3 2 | 86.9 | 137.3 | 3129.6 | 495.2 | 3265.8 | 529.3 |
| Frozen 2 | 103.9 | 145.7 | 3183.6 | 471.0 | 3268.2 | 497.2 |

Both versions' second workers have slower PG scalar controls and slower REFPROP source modes. The compressor control medians change from approximately 10.3 to 12.0–12.2 microseconds. That is further evidence of source-independent time dependence, not a basis for correcting a reported ratio.

GC callbacks show no overlap with any timed PG or bicubic measurement. Each C3 worker has one generation-0 collection overlapping REFPROP CO2 HP sample 7, lasting 0.129 or 0.150 milliseconds. Frozen workers have a generation-1 collection during the broader timing window but outside the individual case/control timers. No worker has a generation-2 collection during timing. The live large-snapshot GC-scan hypothesis is therefore not supported as the explanation of these PG/bicubic samples.

Working sets are approximately 1.50–1.59 GB after full preparation and increase by about 1 MB during timing. Private memory is approximately 1.71–1.80 GB and similarly stable. Each worker incurs approximately 228–231 page faults during timing. These observations do not establish memory pressure, nor do they rule out cache effects. The common/complete sidecar is approximately 45.6 MB; serialized size does not directly measure its live Python heap footprint.

The complete timed window is only 0.194–0.241 seconds. Windows process and thread CPU deltas in these records occur in 15.625-millisecond steps, which is too coarse to interpret an individual microsecond-scale row. `GetSystemTimes` deltas sometimes produce negative per-row busy fractions or no advancing total at all; retain the raw values but do not interpret those fractions as physical CPU utilization. CPU accounting can be assessed only over a sufficiently long aggregate interval. The observed logical processor is the calling thread's processor at the observation boundary; it does not identify where all JAX execution threads ran or prove a performance-core/efficiency-core effect.

## Matched snapshot-storage comparison

The matched [a_disk experiment](../results/phase14/a_disk/summary.json) completed and passed correctness. It uses the same sources, samples, predecessor replay, preparation, default GC and instrumentation. Both arms write identical verification sidecars before timing; disk mode then releases only `Context.snapshots` and `Context.complete_snapshots`, retaining models, buffers, numerical closures and compiled executables. It restores and verifies the sidecar after timing before output comparison. This tests live verification-data retention without removing validation or thermodynamic workload.

| Case | Disk frozen median, microseconds | Disk C3 median, microseconds | Disk C3/frozen | Disk/memory frozen | Disk/memory C3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PG compiled HP batch 1024 | 147.267 | 153.033 | 1.039 | 1.816 | 1.735 |
| Bicubic compiled HP broadcast 8 × 16 | 185.497 | 188.748 | 1.018 | 1.609 | 1.832 |
| PG compiled compressor | 13.467 | 16.460 | 1.222 | 1.219 | 1.514 |
| PG compiled expander | 13.632 | 14.790 | 1.085 | 1.266 | 1.371 |
| REFPROP Water liquid PT | 2057.850 | 2245.650 | 1.091 | 1.229 | 1.426 |
| REFPROP CO2 supercritical HP | 2086.750 | 2165.850 | 1.038 | 1.249 | 1.281 |

There is no measured speedup from releasing snapshots. This is not proof that snapshot release itself makes execution slower: the arms ran consecutively and the version-independent arithmetic control also slowed. Its medians in frozen 1/current 1/current 2/frozen 2 order were 6.13/6.65/7.59/7.64 microseconds for memory and 7.87/8.30/14.41/11.46 for disk. In particular, the second disk workers show large changes in an unchanged arithmetic workload. The current-source compressor control is slower than the frozen control in this arm too, despite the equivalent frozen/current scalar mathematical program. The data do not isolate the machine-level cause.

Clearing snapshot dictionaries changes the GC allocation counters and shifts a generation-0 collection into compressor sample 5 for every disk worker. The overlapping collection durations were 0.490, 0.494, 0.647 and 0.713 milliseconds, equivalent to approximately 10–15 microseconds per call within those particular 48-call samples. All those samples remain in the record. No disk PG-batch or bicubic timer overlapped GC, and no generation-1 or generation-2 collection occurred during the timing window. Thus these observed GC events explain a component of four individual compressor samples, not the broad disk batch slowdown or the original full-matrix result.

Working sets changed by 0 to approximately −1 MB when the snapshot dictionaries were cleared, with private memory behaving similarly. Freed Python objects need not immediately return their arenas to the operating system; the observations do not establish a 45.6 MB physical-memory reduction. Disk workers have zero to two page faults during timing, and complete timing windows of 0.282–0.386 seconds. There is no demonstrated memory-pressure mechanism.

A collected/enabled versus collected/disabled GC comparison would isolate the known collection events, but it is not the smallest next test for the open batch gate: neither disk nor memory PG-batch/bicubic samples overlapped a collection. Disabling GC would also change normal application behavior and cannot replace the original default-GC acceptance result.

## Processor placement and current power context

A subsequent read-only `GetSystemCpuSetInformation` query reports efficiency class 1 for logical processors 0–15, sharing eight core indices, and class 0 for logical processors 16–27, with twelve distinct core indices. Microsoft documents higher [EfficiencyClass](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information) values as faster and less power-efficient. The query uses each variable record's declared size and makes no affinity or power change. The [raw topology](../results/phase14/followup_20261115/windows_cpu_topology.json), [boundary counts](../results/phase14/followup_20261115/windows_cpu_topology_correlation.json) and [per-case associations with every sample retained](../results/phase14/followup_20261115/windows_cpu_case_associations.json) are retained.

All 480 memory-arm timed/control rows have class-1 caller observations at both boundaries. Disk current worker 2 has 47 of 120 rows with both boundaries in class 0, and frozen worker 2 has 39 of 120; migration observations are retained separately. Within each worker, stable class-0 boundaries are associated with slower unchanged controls and native calculations:

| Disk worker and case | Class 1 → 1 median, microseconds (rows) | Class 0 → 0 median, microseconds (rows) |
| --- | ---: | ---: |
| C3 2 arithmetic control | 10.82 (14) | 20.74 (9) |
| Frozen 2 arithmetic control | 7.83 (15) | 18.51 (9) |
| C3 2 PG expander | 10.98 (6) | 29.15 (5) |
| Frozen 2 PG expander | 13.01 (8) | 30.48 (4) |
| C3 2 PG HP batch 1024 | 107.67 (6) | 192.10 (5) |
| Frozen 2 PG HP batch 1024 | 117.49 (7) | 228.78 (3) |

None of the rows in this table overlap GC. No sample is removed from acceptance or divided by a class-specific correction. This is a concrete association between caller placement and a substantial part of the disk-arm drift, including a version-independent arithmetic control. It is not causal isolation: boundary observations can miss intermediate migrations; numerical execution can use other JAX threads; order, predecessor and frequency can covary with placement. It does not establish historical placement in the original uninstrumented run. A separately declared, identical-affinity diagnostic across all compared sources could test that mechanism if the unchanged full repeat remains unstable, while preserving the unconstrained results.

The [one-time power query](../results/phase14/followup_20261115/windows_power_context.json) at 23:57:41 UTC on 14 November reports AC online, battery 69% and charging, battery saver off, and the active scheme `HP Optimized (Modern Standby)` (`fb5220ff-7e1a-47aa-9a42-50ffbf45c673`). [GetSystemPowerStatus](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getsystempowerstatus) and `powercfg /getactivescheme` are read-only. This is current context for the later full repeat, not evidence of a power-state change during the earlier workers or a frequency/thermal measurement.

## Controlled mask crossover

The [completed crossover](../results/phase14/aff_cross/record.json) changes only its own process's allowed mask in alternating class-1/class-0 blocks. It compiles every program under inherited placement before intervention, warms each program after every mask change and retains 30 samples per mask for each of five programs, 300 samples altogether. It compares real complete states and the original status-preserving compressor. Complete outputs pass checks against pre-intervention outputs; inherited affinity `0xfffffff` is restored. The diagnostic collects once and disables GC for both arms, restoring its original state afterward; this differs from the normal-GC acceptance run.

| Program | Class 1 median, microseconds | Class 0 median, microseconds | Class 0 / class 1 |
| --- | ---: | ---: | ---: |
| Unchanged arithmetic control | 6.059 | 56.406 | 9.309 |
| Frozen PG full HP batch 1024 | 72.659 | 207.156 | 2.851 |
| Current PG full HP batch 1024 | 75.906 | 200.719 | 2.644 |
| Frozen PG compressor with actual status | 10.041 | 25.133 | 2.503 |
| Current PG compressor with actual status | 9.874 | 25.289 | 2.561 |

This establishes that changing allowed execution context can substantially change these unchanged programs, including the arithmetic control. It does not isolate microarchitecture: class 1 allows 16 logical processors on eight SMT cores, while class 0 allows twelve logical processors on twelve cores. The masks also change permitted worker scheduling, and programs share JAX pools created under inherited placement. Do not call the 9.309 arithmetic ratio a core-speed ratio or attribute all 17 historical flags to scheduling.

## Final full workload and remaining costs

The [completed full run](../results/phase14/pc_full/summary.json) uses 12 samples and four independent workers per source, explicitly matching allowed worker processors to class 1 (`0xffff`) before numerical imports. All 12 workers verify that exact mask before and after execution. The run preserves all cases, switching, normal GC, live verification snapshots, full output/status validation, samples and declared budgets. Twelve is the smallest sample count that balances both orders and all three input variants. The [follow-up protocol](followup_protocol.json) declares the placement scope and preserves the [exact earlier harness archive](harness_20261115_pre_affinity.zip).

There are 123 recorded workloads: the original 111 comparable cases, eight added comparable shape controls and four capabilities without a published equivalent. All numerical comparisons pass. Four of the original 111 and one of the eight shape controls exceed their original category budget:

| Remaining case | Published median, microseconds | Current median, microseconds | Original limit, microseconds | Current / published | Current / frozen |
| --- | ---: | ---: | ---: | ---: | ---: |
| PG compressor | 7.946 | 10.717 | 8.741 | 1.349 | 1.029 |
| PG expander | 7.223 | 10.284 | 7.946 | 1.424 | 1.024 |
| PG heat exchanger, 128 cells | 14.667 | 18.200 | 16.134 | 1.241 | 0.947 |
| Bicubic heat exchanger, 128 cells | 42.508 | 52.019 | 46.759 | 1.224 | 1.051 |
| Added bicubic HP shape 8 × 16 | 74.442 | 83.106 | 81.886 | 1.116 | 0.938 |

The four consumer costs were already present in frozen Phase 13. None of the five is more than 5.1% slower than frozen; the added shape is 6.2% faster than frozen. That does not waive the published-version limits. The different added shape flag in the original run was 16 × 8; the new 8 × 16 flag is also retained explicitly.

Previously affected complete-state batches now pass: PG dynamic-model HP batch 1024 is 88.389 versus 93.091 microseconds published, 5.1% faster; PG closed-model HP batch 1024 is 73.936 versus 69.541, 6.3% slower; bicubic HP batch 1024 is 540.575 versus 530.194, 2.0% slower; and the original bicubic HP broadcast is 91.438 versus 89.916, 1.7% slower. These are genuine complete returns with actual status, not projections substituted for the gate.

Eleven cases exceed a strict multiplicative 10% diagnostic, while only five exceed their declared category budgets. The difference includes the already declared closed-model PG scalar-pressure batch allowance and the state-inspection/construction absolute-time allowances. Those original formulas are unchanged and every strict flag remains in the portable record. The sole comparable current/frozen shift above 10% is the unchanged-source HEOS CO2 gas PT control, 483.425 versus 433.650 microseconds; it remains faster than its 792.000-microsecond published reference.

Across the 119 comparable cases, the median ratio of largest to smallest worker median is 1.077 for published, 1.095 for frozen and 1.090 for current. Some workloads still vary materially: the largest ratios are 1.723, 1.248 and 1.399 respectively. All samples and worker distributions remain in the export; no spikes are removed. Improved consistency does not make every remaining percentage difference precise or explain each original spike.

All 22 comparable setups pass the original category formula: published worker-median aggregate multiplied by 1.25 plus 10,000 microseconds for construction, or 100,000 for cached loading/fresh generation. Cached CO2 HEOS loading is 25.070 versus 19.695 milliseconds, within its original 124.619-millisecond limit. This retains the required metadata/payload validation and does not claim equal loading speed.

The separate four-worker fresh-process public-call repeat also passes all four original gates, using $1.5\,t_\mathrm{published}+100\,\mathrm{ms}$:

| Cold public-call profile | Published median, milliseconds | Current median, milliseconds | Original limit, milliseconds |
| --- | ---: | ---: | ---: |
| PG full HP batch 1024 | 48.764 | 153.622 | 173.146 |
| Bicubic full HP batch 1024 | 191.586 | 183.215 | 387.379 |
| PG heat exchanger, 128 cells | 49.065 | 106.335 | 173.598 |
| Bicubic compressor | 819.429 | 605.276 | 1329.144 |

The PG median is below the declared cold allowance and remains materially above the published value. This cold repeat used unrestricted inherited placement and a distinct recorded harness identity; it is not relabeled as the class-1 full-run condition. Shared-worker first encounters in the full run are retained separately and are not substituted for these cold measurements.

The portable export records source/harness/protocol identities, all 17,136 comparable timing samples, all 22 setup gates with their raw observations, worker graph hashes and all 48 fresh-process records. The historical `phase14_validation.json` remains byte-for-byte unchanged as dated C3 evidence. The 17-to-five change combines code, placement and order/sample changes; it is not a claim that code alone removed twelve regressions.

## Remaining decision

An additional unrestricted four-worker full repeat is not essential to this hypothesis. The controlled full comparison removes the identified processor-class variable across sources, while the immutable original four-worker full run and existing within-worker unchanged-control associations retain evidence of unrestricted behavior. Repeating a second uncontrolled full arm would not by itself isolate a new cause. An optional two-worker unrestricted 12-sample diagnostic is warranted only for a specifically declared question remaining after the controlled result. Restricting worker placement is an experiment condition, not a production performance improvement or a guarantee about arbitrary application scheduling.

Do not adopt disk snapshot storage as a performance fix: the matched arm demonstrates no benefit and is itself affected by source-independent temporal changes. Keep the original 9-sample evidence alongside the repeat. The useful attribution data are the exact source/variant/predecessor histories, caller processor classes and raw controls. The observations do not justify changing package GC policy or machine power settings. Any future placement condition must be separately declared and applied identically across versions.

The open performance decision concerns the five retained costs above. A further refactor must target those workloads and preserve actual status and complete outputs, or the user must explicitly accept those costs before the gate is closed. No additional benchmark is justified without a concrete unanswered question. The root task records the independently running full correctness-suite result and the overall implementation checklist.

Before extending the diagnostic itself to a three-source wildcard run, address its tooling limitations in a separately recorded revision: skip absent published cases when aggregating current-only capabilities; build predecessor mappings for state helpers appended during preparation; export `Context.checks` as `independent_checks`; and assert matching complete-snapshot case sets before comparison. These do not invalidate the declared six-case experiments and are not changes to frozen instrumentation.
