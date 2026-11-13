# JaxProp tests before the interface refactor

Uniform behavior across fluid classes is the primary refactoring goal. Tests should specify that intended contract and protect valid physical calculations. Backward compatibility is secondary, and intentional changes to signatures, array shapes, identity, reference conventions, and failures should be reflected in the new expectations. The [fluid interface audit](z_FLUID_INTERFACE_AUDIT.md) describes the proposed interface work.

The initial audit found useful numerical tests, an assertion-helper defect that bypassed many comparisons, and important public operations without active tests. Phase 1 repaired the assertion and execution foundation. Phase 2 added the fixed shared fluid contract suite. Broader functionality tests and implementation of the remaining contract gaps continue in the subsequent refactor phases.

The [implementation checklist](z_IMPLEMENTATION_PLAN.md) sequences these test repairs and the fluid-interface changes, with completion tracked against passing acceptance checks.

## Phase 1 implementation status

Phase 1 passed the complete 480-case HEOS and REFPROP suite on 12 November 2026. It added 37 numerical-assertion regressions, 15 runner/backend-selection regressions, a sampling regression, and a fixed input-pair contract test. The existing scalar solver cases now execute their numerical comparisons. Temporary fixtures, full pressure coverage, true cell centers/boundaries, finite-core assertions, and local inverse/accuracy checks are in place.

Actual cell-center checks exposed five failures at the old 10/20/40 accuracy grids. Using 16/32/64 grids meets the retained 0.1/0.01/0.001 percent error budgets, including maximum core-property errors. The full run checked 50,189 finite values with zero failed property comparisons. Twelve existing static-JAX-metadata warnings remain tracked in the implementation plan. [Full verification](development/test_audit/results/phase1/full_20261112_203206/summary.json).

## Phase 2 implementation status

The [public fluid contract](z_PUBLIC_FLUID_CONTRACT.md) now specifies the common construction/call vocabulary, model/provider/source distinction, identity, broadcast shapes, availability, reference conventions, errors, and JAX policy. Its independent five-row matrix covers all advertised evaluator/model families, including the binary consolidation target.

Phase 2 added 218 cases across [state operations](tests/test_state_contract.py), [shared fluid conformance](tests/test_uniform_fluid_contract.py), and [failure/reference/execution behavior](tests/test_contract_failures.py). Of these, 99 pass and 119 are confirmed gaps marked with strict expected failures and implementation ownership. A fixed gap must have its marker removed; unexpected passes fail the suite. Factory absence does not prevent direct tests of the existing classes.

The full HEOS suite passed 405 cases with 119 strict expected failures, zero unexpected failures, and zero skips. All 306 preexisting HEOS cases still pass. [Full verification](development/test_audit/results/phase2/full_heos_20261112_205840/summary.json). These new cases specify intended functionality; the numerical evaluator implementation remains unchanged in Phase 2.

## Phase 3 implementation status

The shared state foundation is implemented and adds 145 passing tests for [typed errors](tests/test_errors.py), [state availability/collections/serialization/JAX behavior](tests/test_state_foundation.py), and [evaluator identity/gradient integration](tests/test_evaluator_state_metadata.py). Forty earlier contract gaps are resolved. The final full HEOS + REFPROP suite reports **764 passed, 79 strict expected failures, zero unexpected failures, and zero skips**. [Full verification](development/test_audit/results/phase3/full_20261112_214051/summary.json).

The numerical oracles use raw enabled properties to inspect unavailable slots without triggering public-access errors. Finiteness, shape, availability-pattern, and tolerance checks are retained. Comparison coverage exactly matches Phase 1: 50,189 finite values and 34,761 explicitly unavailable values across 84,950 comparisons, with zero failed comparisons. The remaining expected failures continue to identify constructor/capability/reference and evaluator work in later phases.

## Table cache validation follow-up

The cache lifecycle portion of Phase 8 now adds 109 passing cases across [cache policy and atomic publication](tests/test_bicubic_cache.py), [source/reference/native-file fingerprints](tests/test_cache_metadata.py), and [real generation/reload integration](tests/test_bicubic_cache_integration.py). Actual equilibrium, mixture, and both metastable generators are covered. The complete HEOS + REFPROP run reports **873 passed, 79 strict expected failures, zero unexpected failures, and zero skips**, with the same numerical comparison counts as Phase 1. [Full verification](development/test_audit/results/cache_validation/full_20261112_223133/summary.json).

The [implementation checklist](z_IMPLEMENTATION_PLAN.md) distinguishes completed cache/configuration checks from remaining interpolation-cell, saturation-grid integrity, and metastable numerical-accuracy work. Metadata identity and finite payloads alone do not establish derivative accuracy.

## Performance and native JAX regression coverage

The performance work adds 62 passing regression cases: constructor/collection/runtime-storage behavior and native-JAX execution/lazy validity. These cover NumPy/JAX storage, immutability, missing values, masks/status, empty batches, serialization, dynamic coefficients/model parameters, compilation reuse, broadcasting, gradients, and constant/traced control flow. Perfect-gas list input now satisfies its contract case and its strict expected-failure marker was removed. The [JAX efficiency audit](z_JAX_EFFICIENCY_AUDIT.md) describes the implementation and phased follow-ups.

The complete HEOS + REFPROP suite reports **936 passed, 78 strict expected failures, zero unexpected failures, and zero skips**, retaining all 84,950 numerical comparisons with zero failures. [Full verification](development/test_audit/results/jax_efficiency/full_20261113_094055/summary.json). The compact [performance report](development/performance/README.md) preserves the final matched comparison; disposable benchmarking scaffolding and intermediate artifacts have been removed. The formal performance gate and remaining capability/reference/model/domain work stay open in the implementation plan.

## Phase 4 shared API coverage

The shared factory/configuration/request layer is implemented. New [factory/configuration tests](tests/test_fluid_factory.py) add 67 cases and [shared API tests](tests/test_phase4_api.py) add 158, including immutable snapshots/identity/composition, typed selectors/options/dtypes/shapes, honest capabilities, unsupported diagnostics/modes, named statuses, invalid legacy ratios, and extra callback input pairs. Target minimums remain fixed separately: missing PG core properties belong to Phase 7 and component-based binary construction/inversions to Phase 9.

The complete HEOS + REFPROP run passed **1,208 tests with 28 strict expected failures, zero unexpected failures, and zero skips**. [Full result](development/test_audit/results/phase4/full_20261113_111610/summary.json). The final contract/native execution run passed **396 cases with 28 expected failures**, including eight review cases added after full-suite collection and the final static-validation path. [Final checks](development/test_audit/results/phase4/final_contract.xml).

A further [JAX interface review](tests/test_phase4_jax_edges.py) added **five passing cases**: callback/bicubic/perfect-gas empty broadcast batches preserve full states in eager and outer-JIT execution, and native bicubic/perfect-gas HP/PT alternation reuses one compilation per pair while values change. No production fix was needed. Timing thresholds are deliberately separate from these deterministic compilation/behavior tests; the compact [Phase 4 performance check](development/performance/phase4_validation.json) records the independent serial measurements.

All **50,189 finite numerical comparisons are preserved**, with zero failures. Paired-unavailable checks decrease from 34,761 to 4,083, for 54,272 total comparisons: the removed entries were NaN placeholders for now-unsupported properties. Capability/unsupported-property assertions cover that distinction; tests do not restore fake availability to keep a historical counter constant. Source and bicubic numerical tests continue to enforce their existing finite-value, shape, inverse, and local-error tolerances.

## Common constructor revision coverage

The [public initializer tests](tests/test_public_fluid_constructor.py) add 32 passing cases and two narrow Phase 11 expected failures. They verify actual objects for all four implemented methods, the public class/runtime type, exactly-once initialization, typed invalid selectors/options before source allocation, defaults and identifiers, shared metadata/references, JIT/vmap/grad behavior, unchanged native model trees, dynamic coefficient/calibration parameters with compilation reuse, and PG/bicubic pickle execution. The expected failures isolate pre-existing unpickleable source handles rather than weakening object round-trip expectations.

Conformance fixtures now construct implemented routes through `Fluid(method=...)`, while lower-level `create_fluid` coverage remains. Source-specific spies and tests use the explicitly named `FluidCoolProp`; selector and shape/numerical expectations are unchanged. Full HEOS + REFPROP verification passes **1,304 cases with 29 strict expected failures, zero unexpected failures, and zero skips**, retaining all **54,272 numerical comparisons**, including **50,189 finite values and 4,083 paired-unavailable values**, with zero failures. [Full result](development/test_audit/results/public_constructor/full_20261113_160742/summary.json).

The example migration changes 40 files, checks all 65 for syntax, and verifies all 74 constructor argument sets against the committed pre-migration expressions. Nineteen representative programs complete their unchanged default workloads and figure rendering. [Migration record](development/example_audit/constructor_migration.json). The earlier 65-file execution audit remains a historical checkpoint; it is not silently relabeled as a full rerun after the constructor changes.

The initial constructor timing guard flagged a 10.4% perfect-gas batch shift. A longer repeat measured +0.7% public/+0.9% private batch medians with substantial worker variation, identical model leaf typing/values, bitwise-equal full batch outputs and an identical callback-free optimized batch program. [Initial guard](development/performance/constructor_validation.json), [follow-up](development/performance/constructor_batch_followup.json). Both records remain; no production code was changed to hide the initial flag.

## Phase 5 source-reference coverage

The shared [source-reference tests](tests/test_source_references.py) add 43 passing HEOS + REFPROP cases for CO2/Water/Air metadata across all four pure-fluid evaluators, independent source coordinate/core-property oracles, pseudo-pure endpoints, immutable array-free records, optional/individually unavailable properties, canonical aliases, selected-backend perfect-gas calibration, and construction/reference consistency. Subprocess tests detect changed global conventions without mutating existing objects or the parent process. Native execution is checked for compilation reuse, gradients, no repeated source initialization, and removal of static-array warnings. Existing Phase 5 expected failures were removed; previously unsupported PG REFPROP construction now verifies the selected source.

Cache tests additionally assert the shared source ID, version/alias/composition behavior, global h/s changes, and rejection of a source-reference mismatch before generation/publication, while preserving complete native definition/file/configuration/payload validation. Focused checks passed **304 cases with 12 existing expected failures**, plus **165 native/shared-contract cases with 14 expected failures** and the cache suites. The complete HEOS + REFPROP run passed **1,267 cases with 26 strict expected failures**, zero unexpected failures/skips, and all **54,272 comparisons unchanged** (50,189 finite and 4,083 paired-unavailable). [Full result](development/test_audit/results/phase5/full_20261113_133321/summary.json). Two final reference-edge cases added after collection passed as intended: the forced critical fallback passes, and the existing quality/reference mismatch is one strict Phase 6 expected failure. [Final checks](development/test_audit/results/phase5/final_reference_edges.xml).

A new narrow Phase 6 expected failure records the runtime quality helper's use of a newly constructed source under a later global entropy convention. Its subprocess first verifies that core values, identity, and stored reference entropy remain unchanged; only quality stability may satisfy the expected failure. Dynamically modified model-view identities are separately tracked in Phase 7. These findings do not weaken the implemented source-reference or finite-value assertions.

The findings and baseline below describe the starting state before these test repairs. The implementation checklist records current completion and the remaining coverage work.

## Executed baseline

The audit examined the local JaxProp 0.5.8 source at commit `ed66d108555e71d0d8543a1b670d05237a834221` and ran the active tests on 12 November 2026 using Python 3.13.2 and pytest 8.4.2 in ThermOpt's virtual environment. The tests' existing `GITHUB_ACTIONS=true` switch selected HEOS. REFPROP cases remain outside this runtime baseline.

Tests ran from an isolated directory under `development/test_audit/results/`, because the bicubic fixtures delete the working directory's `fluid_tables` folder. Source inspection and runtime probes establish functionality and assertion coverage here. The counts below are collected test cases and executed comparisons.

| Test module | HEOS cases | Current coverage |
| --- | ---: | --- |
| `test_solvers_coolprop.py` | 63 | Nine fluids, seven reference-state labels, multiple input pairs. |
| `test_solvers_equilibrium.py` | 48 | Eight fluids, six labels, multiple property pairs, one solver and nearby initial guess. |
| `test_solvers_metastable.py` | 28 | Seven fluids, four stable reference-state labels evaluated through the metastable solver. |
| `test_helmholtz_eos.py` | 28 | Direct density/temperature evaluation at stable liquid, vapor, and supercritical states. |
| `test_spinodal_calculation.py` | 7 | Seven fluids, two branches with ten points each, near-zero normalized bulk modulus. |
| `test_bicubic_accuracy.py` | 9 | Three fluids and three table resolutions, compared with source property values. |
| `test_bicubic_consistency.py` | 24 | Three fluids and eight implemented input pairs, compared with the same interpolant's HP result. |
| `test_perfect_gas_eos.py` | 40 | Four scalar calibration conditions, five inverse pairs, scalar values and gradients. |
| `test_fluid_state.py` | 5 | Addition, reversal, indexing, metadata rejection, nonmutation, and one gradient path. |
| **Total** | **252** | Nine active test modules. |

The existing suite reported **252 passed, 33 warnings in approximately 175 seconds**. Instrumentation recorded **42,475 calls to the shared numerical assertion helper, with every call bypassing its numerical check**. The affected four solver/Helmholtz modules contain 167 cases. Other modules have their own active assertions.

A second diagnostic run replaced that helper in memory with an array-aware comparison using the existing absolute-or-relative tolerance rule and the existing treatment of paired NaNs. All **167 cases passed**, with **25,022 numerical comparisons** and **17,453 paired-NaN comparisons**. This supports the covered scalar calculations once the oracle is repaired. Required-property finiteness and the missing behavior cases still need explicit tests.

The diagnostic helper and instrumentation were confined to the test processes. Detailed evidence is available in the [baseline summary](development/test_audit/results/baseline_20261112_195233/summary.json), [pytest log](development/test_audit/results/baseline_20261112_195233/pytest.log), [array-aware comparison run](development/test_audit/results/baseline_20261112_195233/array_oracle_summary.json), and [root collection](development/test_audit/results/baseline_20261112_195233/root_collection.json).

## Repair the existing test protection

### 1 Make numerical assertions inspect returned arrays

**First priority.** `assert_consistent_values` accepts only Python numeric scalars and NumPy scalar types. `FluidState.items()` passes values through `to_dict()`, which converts them to JAX arrays. Even a scalar property therefore reaches the helper as an array and is skipped. A probe with a valid state's temperature changed by 1000 K was accepted with zero comparison-log entries. Python floats representing the same kind of mismatch correctly failed. [Helper type filter](tests/utilities.py#L117), [state dictionary conversion](jaxprop/helpers_props.py#L97), [probe results](development/test_audit/results/baseline_20261112_195233/oracle_probe.json).

Make the helper array-aware, assert shapes, and distinguish finite values, expected unavailable values, and unexpected NaNs/infinities. Use property-specific absolute and relative tolerances. Test the helper itself with matching and deliberately wrong Python scalars, NumPy scalars/arrays, JAX scalars/arrays, singleton/matrix shapes, and finite/NaN mismatches. Assert that each numerical test actually compared its required properties. Test identity and phase flags explicitly.

The affected call sites are in the [CoolProp](tests/test_solvers_coolprop.py#L153), [equilibrium](tests/test_solvers_equilibrium.py#L137), [metastable](tests/test_solvers_metastable.py#L144), and [Helmholtz](tests/test_helmholtz_eos.py#L88) tests. The diagnostic run demonstrates that this repair can restore useful checks without requiring a fluid implementation change first.

### 2 Correct sampling and make required coverage explicit

**First priority.** The bicubic consistency grid uses `log(p_max)` for both pressure endpoints, so every inverse case is exercised at the upper pressure boundary. Its input-pair list is derived from `PROPERTY_CALCULATORS`, so removing an implementation entry also removes its test. Define the expected input contract independently and check it against declared capabilities. Sample multiple pressures, cell interiors, and boundaries. [Input-case derivation](tests/test_bicubic_consistency.py#L27), [pressure grid](tests/test_bicubic_consistency.py#L87).

The accuracy test's fixed 25-point grids include endpoints and do not construct actual interpolation-cell midpoints. Its mean percentage error and the consistency test's global relative norm can dilute a localized error. Add true cell centers, edges/corners, and maximum pointwise errors with sensible absolute tolerances near zero. First require finite core properties at valid states, then compare values. Existing finite-reference/NaN-result mismatches already fail these tests, while properties with entirely unavailable reference values are skipped. [Accuracy sampling](tests/test_bicubic_accuracy.py#L144), [accuracy masks and metric](tests/test_bicubic_accuracy.py#L159), [consistency mask and norm](tests/test_bicubic_consistency.py#L105).

Add application-relevant liquid, near-critical, pseudocritical, and applicable two-phase cases. The current CO2 domain starts at 600 kJ/kg, while the recuperated-cycle compressor inlet is around 283 kJ/kg. Exact phase endpoints also need coverage. The solver reference fixture labels saturated liquid/vapor using qualities 0.001 and 0.999, and the equilibrium suite excludes its saturated-liquid label. [Bicubic domains](tests/test_bicubic_accuracy.py#L21), [phase fixtures](tests/utilities.py#L26), [equilibrium labels](tests/test_solvers_equilibrium.py#L68).

### 3 Make test execution reliable and isolated

**First priority.** Replace relative shared table paths with `tmp_path` or `tmp_path_factory`. Share expensive valid tables through fixtures where appropriate, with separate tests for regeneration and cache reuse. Current bicubic fixtures delete `fluid_tables` before/after accuracy cases and at consistency-session setup. [Accuracy cleanup](tests/test_bicubic_accuracy.py#L101), [consistency cleanup](tests/test_bicubic_consistency.py#L46).

Align the custom runner with normal pytest discovery and propagate pytest's exit code. It currently lists basenames relative to the working directory, omits the state tests, and ignores `pytest.main`'s result. Running it from the isolated directory produced a missing-file error and zero collected tests, while the Python process returned exit code 0. CI invokes the pytest CLI directly, which propagates its result. [Custom runner](tests/run_tests.py#L5), [runner probe](development/test_audit/results/baseline_20261112_195233/custom_runner_probe.json), [CI invocation](.github/workflows/publish_to_pypi.yaml#L34).

Root collection and explicit `tests/` collection both produced 252 HEOS cases in this audit. Keep one documented invocation and test its discovery and failure status. Select or detect optional REFPROP availability explicitly. The current helper includes REFPROP whenever `GITHUB_ACTIONS` is false. [Backend selection](tests/utilities.py#L10), [pytest configuration](tests/pytest.ini#L1).

## Important missing functionality tests

### 4 One uniform fluid behavior suite

**High priority.** Parameterize one fixed contract over every advertised fluid configuration. Cover common construction and argument names, supported input pairs, returned state type, identity defaults and propagation, canonical property names, units, available fields, options, and unsupported requests. Verify scalar, singleton, vector, matrix-broadcast, and supported empty inputs using Python, NumPy, and JAX representations.

Include invalid pressure/temperature/enthalpy, NaN/infinite input, incompatible shapes, mixed-validity batches, out-of-domain queries, and consistent eager/JAX failure outcomes. Encode the chosen validity policy, identity rules, and exact shape contract. Current clipping, missing metadata, and squeezed dimensions should be exposed by those expectations. Existing numerical tests construct each class through its different interface and therefore provide little protection for this goal. [Class entry points](z_FLUID_INTERFACE_AUDIT.md#current-interfaces), [current failure differences](z_FLUID_INTERFACE_AUDIT.md#6-unify-invalid-state-and-convergence-handling).

Require explicit unsupported-request tests. Perfect gas must reject quality input pairs, saturation options, and requested saturation/quality properties with informative error types and messages. Verify the fluid, evaluator, requested pair/property, reason, and supported alternatives where applicable. Cover direct property requests, `state.quality_mass`, `state.Q`, and `state["Q"]` consistently. Static unsupported requests should produce the declared error before numerical evaluation/JIT, including when numerical failures use a non-throwing policy. Separately test a supported property that is undefined at the particular state, such as an unavailable saturation quantity above the applicable critical boundary.

### 5 Dedicated JAX evaluation and derivative tests

**High priority.** `FluidJAX` is used as a reference evaluator by bicubic accuracy tests, but has no dedicated contract or derivative tests. Check eager/outer-JIT agreement, supported `vmap` behavior, scalar and batched Jacobians, JVP/VJP accuracy where supported, identity, list/array acceptance, invalid callback inputs, and failure status. Verify the claimed transformation set for each provider.

For the callback wrapper, compare finite-difference JVPs with independent perturbations over several step sizes and input scales. For native bicubic and perfect gas, compare AD with finite differences of the same numerical model, and separately compare values/derivatives with an independent physical reference where applicable. Add inverse-derivative checks for the supported HP/PT/PS paths. [Callback/JVP implementation](jaxprop/coolprop/jax_wrapper.py#L47), [bicubic evaluator](jaxprop/bicubic/bicubic_interpolation.py#L1366), [existing perfect-gas derivative tests](tests/test_perfect_gas_eos.py#L95).

### 6 Bicubic construction, interpolation, and cache lifecycle

**High priority.** Add a small deterministic polynomial fixture in enthalpy and log-pressure. Verify bicubic values, first/mixed derivatives, coordinate scaling, and continuity across neighboring cells using known coefficients. This gives fast, independent protection for interpolation logic. [Coefficient construction](jaxprop/bicubic/bicubic_interpolation.py#L1586).

Test valid and invalid domains/grid sizes, exact boundaries, explicit extrapolation policy, inverse residuals, convergence failures, and effective iteration limits. Inject a failed source evaluation and verify incomplete cells/tables are rejected or explicitly masked according to the chosen policy. Assert generation versus reuse, settings-based invalidation, corrupt/truncated-cache handling, reload equivalence, and concurrent construction/atomic replacement. Current tests incidentally reload tables but do not assert these outcomes. [Skipped generation failures](jaxprop/bicubic/bicubic_interpolation.py#L317), [cache loading](jaxprop/bicubic/bicubic_interpolation.py#L164), [inverse solvers](jaxprop/bicubic/bicubic_interpolation.py#L1442).

Add targeted tests for saturation, mixture, and metastable-liquid/vapor table modes that will remain advertised. Ordinary single-fluid saturation data are empty, the saturation central derivative has a division-precedence defect, and these paths have no active coverage in the current bicubic modules. [Saturation data](jaxprop/bicubic/bicubic_interpolation.py#L398), [saturation derivative](jaxprop/bicubic/bicubic_interpolation.py#L1334).

### 7 Independent physical and reference checks

**High priority.** Scalar round trips can agree when their reference and inverse share an error. Add independent thermodynamic identities with applicability and units defined for each model, including $h-u=p/\rho$, $a^2=1/(\rho\kappa_s)$, and applicable heat-capacity/compressibility relations. Compare selected source-fluid states directly with CoolProp or validated reference values, including exact phase endpoints and difficult regimes.

For perfect gas, test off-calibration temperatures/pressures and independently verify $p=\rho RT$, $c_p-c_v=R$, $\gamma=c_p/c_v$, $a^2=\gamma RT$, internal energy, entropy differences, transport relations, and the selected enthalpy/entropy reference convention. Existing tests mainly compare five inversions with the same model's scalar PT output and check nine property keys. Core completeness and independently specified equations need their own checks. [Perfect-gas test cases and fields](tests/test_perfect_gas_eos.py#L14), [same-model reference](tests/test_perfect_gas_eos.py#L40), [state assembly](jaxprop/perfect_gas/perfect_gas.py#L241).

### 8 Phase diagnostics, metastability, and blending

**High priority.** Verify actual generalized/trimmed quality, subcooling, superheating, saturation pressure/temperature, departure and ratio values, and defined supercritical/unavailable outcomes. Existing solver tests skip quality/phase comparisons in many cases, and leave supersaturation disabled. [Skipped quality checks](tests/test_solvers_coolprop.py#L132), [equilibrium options](tests/test_solvers_equilibrium.py#L104), [diagnostic functions](jaxprop/coolprop/core_calculations.py#L1146).

Test metastable liquid and vapor branches between saturation and spinodal, branch selection, direct density/temperature paths, and approach to stability limits. Current metastable tests use ordinary stable reference states, a nearby initial guess, and the LM solver. Add malformed inputs/guesses, other advertised algorithms, iteration limits, unsuccessful solves, and recovered-input assertions. Extend spinodal checks to branch ordering/distinctness, physical validity, endpoints, and convergence failures. [Metastable labels and solver](tests/test_solvers_metastable.py#L69), [flash validation/convergence](jaxprop/coolprop/core_calculations.py#L839), [current spinodal assertion](tests/test_spinodal_calculation.py#L62).

Add public blending smoke tests, equilibrium/metastable endpoint recovery, transition continuity, aliases, invalid widths, and output schema. A normal `Fluid.get_state_blending` probe currently fails because dictionary results are passed to a helper that calls `.to_dict()`. That helper also adds `x` and `sigma` keys that the current `FluidState` constructor cannot accept. [Public blending](jaxprop/coolprop/fluid_properties.py#L429), [dictionary mismatch](jaxprop/coolprop/core_calculations.py#L1023), [extra output fields](jaxprop/coolprop/core_calculations.py#L1058), [public-path probe](development/test_audit/results/baseline_20261112_195233/uncovered_function_probes.json).

### 9 Mixture construction, composition, and conservation

**High priority for supported mixtures.** No active tests instantiate `FluidMix` or `FluidTwoComponent`. Add import/construction tests, component-order and composition-basis checks, validation, pure-component limits where defined, mass/enthalpy conservation, state recovery, output schema/identity, arrays, failure behavior, and claimed JAX support. Use the intended mixture physics as the oracle.

`FluidTwoComponent` currently fails to import because `MixtureState` is missing. `FluidMix` has a scalar HP-only evaluator and defaults composition to `None`. Make supported functionality and any repaired/replaced implementation visible through explicit tests. [FluidMix](jaxprop/coolprop/fluid_properties.py#L766), [mixture model](jaxprop/coolprop/core_calculations.py#L1633), [two-component import failure](jaxprop/two_component/mixture_calculations.py#L209), [probe](development/test_audit/results/baseline_20261112_195233/uncovered_function_probes.json).

### 10 Shared states, metadata, phase data, and plotting

**High priority for state operations, medium for rendering.** Extend the state tests to cover complete canonical/alias enumeration, `to_dict` and alias inclusion, scalar/singleton/matrix stacking, empty collections, metadata mismatch, and JAX behavior. `stack` currently squeezes singleton dimensions and a duplicated dictionary key removes the `alpha_p` alias. The existing state tests exercise addition/reversal/indexing but do not cover these paths. [State tests](tests/test_fluid_state.py#L11), [stack](jaxprop/helpers_props.py#L168), [overwritten alias](jaxprop/helpers_props.py#L251).

Test critical/triple reference initialization independently of the active evaluator. CO2 and Water should expose matching CoolProp source points through ordinary, JAX, bicubic, and perfect-gas objects. Perfect-gas reference metadata must be available while its quality/saturation evaluation requests produce the expected unsupported-request errors. Check initialization once outside JIT, source backend/version and reference conventions, identity, and explicit outcomes for unavailable points.

Include pseudo-pure Air and a refrigerant blend such as R410A. Require their CoolProp critical and lower-temperature liquid/vapor references to remain accessible with explicit pseudo-pure/model-boundary interpretation. Preserve different bubble/dew endpoint pressures when supplied. Test a genuinely unavailable or ambiguous mixture point with a recorded reason and an informative error on explicit access, while ordinary supported state evaluation remains usable. The [reference-point probes](development/test_audit/results/baseline_20261112_195233/reference_point_probe.json) provide concrete cases, including an ambiguous Nitrogen/Oxygen critical-point lookup.

Use small saturation, pseudocritical, quality, and density/temperature grids and verify shape, endpoints, finite required properties, and metadata. A one-point `compute_property_grid_rhoT` call currently raises `NameError` for the missing `states_to_dict_2d` helper. [Grid implementation](jaxprop/coolprop/fluid_properties.py#L1558), [probe](development/test_audit/results/baseline_20261112_195233/uncovered_function_probes.json).

Restore headless plotting checks for data/artist creation, repeated updates, stream grouping, visibility options, and reference-data access. Both the phase-diagram and example test modules are entirely commented out and collect zero cases. Small functional assertions can cover these behaviors without relying on image snapshots. [Disabled phase tests](tests/test_phase_diagrams.py#L1), [disabled example tests](tests/test_example_cases.py#L1), [phase plotting](jaxprop/coolprop/fluid_properties.py#L486).

### 11 Lifecycle, application configuration, and numerical helpers

**Medium priority.** Test supported serialization by evaluating properties after a round trip. Check repeated calls for state leakage and document/test any supported concurrency model. Use subprocess tests for import-time device/precision settings and public import/export behavior. Current serialization and application-setting defects are described in the [interface audit](z_FLUID_INTERFACE_AUDIT.md#10-document-jax-behavior-and-application-settings).

Cover numerical helpers used by blending and solver selection, especially sigmoid endpoints/derivatives, stable large inputs, broadcasting, and invalid selectors. Add independent perturbation checks for saturation/Gruneisen/Joule-Thomson derivative helpers, with controlled source states. General formatting and media-export utilities can follow the core numerical and interface work. [Smoothing functions](jaxprop/math.py#L3), [solver/adjoint selectors](jaxprop/helpers_jax.py#L30), [derivative helpers](jaxprop/coolprop/core_calculations.py#L246).

### 12 ThermOpt integration through the common contract

**High priority before declaring interchangeability.** Exercise common construction, setup/reference data, an isentropic compressor and expander, and a discretized heat exchanger through the same consumer entry points for every compatible evaluator. Check energy balances, input/output state recovery, finite derivatives where promised, identity, and constraint arrays.

Keep a representative recuperated CO2 calculation as an integration reference for source-compatible real-fluid evaluators, with table-resolution-appropriate tolerances. Include a phase-changing case when saturation support is advertised. Perfect-gas integration should use expected results for its chosen physical model. Existing example coverage is disabled, and the current numerical tests do not exercise this complete consumer path. [ThermOpt component calls](../thermopt/thermopt/components/basic_components.py#L329), [recuperated cycle](../thermopt/thermopt/cycles/cycle_power_recuperated.py#L15).

## Suggested implementation order

1. Repair and test the numerical oracle, table isolation, sampling, and runner exit/discovery behavior. Rerun the scalar numerical baseline with real assertions.
2. Specify the uniform identity, shape, option, capability, and failure contract. Write fixed expected-contract cases against every advertised evaluator. Track confirmed existing defects with explicit cases and intended outcomes.
3. Add fast independent physical invariants, state tests, JAX checks, polynomial interpolation tests, and public-path smoke tests for blending, grids, and mixtures. Add targeted failure injection for table generation and flash convergence.
4. Add the intended physical regimes, table lifecycle cases, and consumer integration checks. Reuse isolated valid fixtures for expensive source-generated tables and mark optional external-backend/slow cases explicitly.
5. Refactor against these expectations, updating consumers and examples together. Remove obsolete inconsistent expectations as the uniform contract becomes authoritative and document the intentional behavior changes.

Tests for known defects should describe the desired result, with a clear issue or temporary strict expected-failure marker where needed during incremental work. Independent physical checks protect valid calculations while the public behavior is deliberately standardized.
