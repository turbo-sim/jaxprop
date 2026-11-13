# JaxProp uniform fluid interface implementation plan

Uniform, consistent behavior across fluid objects is the primary goal. Backward compatibility is secondary. The implementation should protect valid physical calculations while establishing consistent construction, property calls, identity, array shapes, reference metadata, capabilities, and errors.

This checklist starts from the local 0.5.8 source at commit `ed66d108555e71d0d8543a1b670d05237a834221`, reviewed on 12 November 2026. The [interface audit](z_FLUID_INTERFACE_AUDIT.md) supplies the source findings and design requirements. The [test audit](z_PRE_REFACTOR_TEST_AUDIT.md) supplies the coverage gaps, runtime evidence, and proposed tests.

The [native JAX efficiency audit](z_JAX_EFFICIENCY_AUDIT.md), added on 13 November 2026, records compiler/source inspection, implemented execution/state fixes, and larger optimization ideas assigned to the phases below.

## Tracking changes

Check an implementation item when its source/test changes are present and its relevant acceptance checks pass. Record a short evidence entry beside the item or in the verification log, including the command, scope, result, and any explicit expected failures. A phase is complete when its checklist is complete.

Implement and verify one coherent change at a time. Start with tests for its intended behavior, implement the change, run the affected tests, and then run the shared contract/numerical checks relevant to its callers. Use the full HEOS suite at the phase boundaries. Run the declared optional-backend and slow integration cases before completing their corresponding release scope. Revisit completed items when a later change alters their contract.

The checked preparation items record completed audit work. Phases 1 through 5 are complete: the numerical test foundation is repaired, the [public fluid contract](z_PUBLIC_FLUID_CONTRACT.md) fixes the intended interface, the shared state/construction/request layers are verified, and source-reference initialization is consistent across evaluators. Remaining model, integration, and performance verification work is in Phases 6 through 13. The earlier array-aware diagnostic experiment is retained as historical evidence.

## Completed preparation

- [x] Audit all fluid classes and shared states, recording interface differences and source locations.
- [x] Audit active tests, assertion behavior, test discovery, and uncovered public functionality.
- [x] Execute the existing HEOS suite. Result recorded as 252 passed, with the shared comparison helper bypassing all 42,475 calls.
- [x] Execute an array-aware diagnostic comparison run. All 167 affected cases passed, with 25,022 numerical comparisons and 17,453 paired-NaN comparisons.
- [x] Record the priority of uniform behavior, common CoolProp source-reference initialization, explicit unsupported-request errors, and graceful pseudo-pure handling.
- [x] Compare the two mixture implementations and identify their common base model, complementary outputs, and derivative consistency problems.

Existing tests provide partial coverage of scalar source-fluid calculations, perfect-gas values/gradients, bicubic values/inverses, spinodal curves, and state collection operations. Missing contract, regime, failure, and lifecycle tests are included below.

## Phase 1 Restore trustworthy numerical tests

- [x] Persist the array-aware numerical assertion repair. Check shapes, finite values, unexpected NaNs/infinities, and expected unavailable fields using documented tolerances.
- [x] Test the assertion helper with deliberately incorrect Python, NumPy, and JAX values, including scalar, singleton, and matrix cases. Require real comparisons of the expected properties.
- [x] Isolate all generated tables and test outputs with pytest temporary-directory fixtures. Reuse expensive valid fixtures where appropriate.
- [x] Correct the bicubic consistency pressure grid and construct actual interpolation-cell midpoints. Add boundary and maximum local-error checks.
- [x] Define expected core input pairs independently of implementation dispatch dictionaries, and assert finite required properties before numerical comparison.
- [x] Fix custom-runner path resolution, test discovery, and exit-code propagation. Make optional REFPROP selection explicit and report its availability.
- [x] Run the existing HEOS suite with the repaired assertions and record the numerical baseline and any failures requiring investigation.

Acceptance verified on 12 November 2026. The complete suite passed 480 cases across HEOS and REFPROP with zero failures or skips, including 54 new regression cases in the HEOS selection. The repaired helper checked 50,189 finite values and 34,761 explicitly unavailable optional values across both backends. [Full result](development/test_audit/results/phase1/full_20261112_203206/summary.json).

True-center sampling exposed five accuracy failures with the old 10/20/40 grids. Accuracy fixtures now use 16/32/64 nodes while retaining the original 0.1/0.01/0.001 percent budgets, with those budgets also enforced on maximum core-property error. Inverse cases span the full pressure domain. The 12 existing static-JAX-metadata warnings are recorded for the later JAX work.

The test suite defaults to HEOS. Set `JAXPROP_TEST_BACKENDS=HEOS,REFPROP` to explicitly include REFPROP. Unavailable requested REFPROP cases are reported as skipped. The verified full run found REFPROP available.

The test runner can be invoked from any working directory as `python path/to/jaxprop/tests/run_tests.py -q`. This verification used normal Windows execution permissions because the restricted token could not access pytest private temporary-directory ACLs. Test code uses standard pytest temporary fixtures.

## Phase 2 Define and test the uniform public contract

- [x] Specify one construction API and provider selector, following the audit's proposed `create_fluid(name, provider, backend, identifier, ...)` pattern. Distinguish the numerical provider, physical model, and CoolProp source backend.
- [x] Specify one `get_state(input_type, prop_1, prop_2, ...)` argument vocabulary, common identity/defaults, SI and mass/molar conventions, and reference conventions.
- [x] Specify exact scalar and broadcast shapes, core state fields, optional property availability, supported input pairs/options, and batch failure behavior.
- [x] Specify typed errors for unsupported input pairs/properties/options, unavailable reference points, invalid states, and failed calculations. Define eager and supported JAX outcomes.
- [x] Add a fixed shared conformance matrix for every advertised evaluator and physical model, with explicit applicable and unsupported cases.

Acceptance requires tests to express the intended common behavior, including perfect-gas quality/saturation rejection. During incremental implementation, track confirmed failures explicitly so the remaining work stays visible.

Acceptance verified on 12 November 2026. The [public contract](z_PUBLIC_FLUID_CONTRACT.md) specifies construction/model/provider selection, mass-based SI inputs, preserved identity and broadcast axes, property availability, source references, typed errors, and eager/JAX failure policies. The fixed matrix includes direct CoolProp, its JAX wrapper, bicubic, perfect gas, and the consolidated binary target represented by current `FluidMix`. The broken unexported `FluidTwoComponent` is an input to consolidation rather than a promised separate public evaluator.

| New contract tests | Passed | Strict expected failures |
| --- | ---: | ---: |
| [State foundation](tests/test_state_contract.py) | 25 | 10 |
| [Shared evaluator conformance](tests/test_uniform_fluid_contract.py) | 74 | 61 |
| [Failures, references, and execution modes](tests/test_contract_failures.py) | 0 | 48 |
| **Total new cases** | **99** | **119** |

The full HEOS run completed with **405 passed, 119 strict expected failures, zero unexpected failures, and zero skips** in 279.61 seconds. All 306 existing HEOS cases passed. [Summary](development/test_audit/results/phase2/full_heos_20261112_205840/summary.json), [pytest log](development/test_audit/results/phase2/full_heos_20261112_205840/pytest.log), and [JUnit results](development/test_audit/results/phase2/full_heos_20261112_205840/junit.xml) record the run. The final uniform string-valued error-request assertions were separately reverified: [48 strict expected failures](development/test_audit/results/phase2/final_failures.xml).

Each expected failure identifies a confirmed gap and its responsible phase; remove its marker when implemented. `strict=True` makes an unexpected pass fail the suite. The missing factory does not gate the existing evaluators' numerical, identity, shape, or rejection tests. Phase 2 changes tests and specifications only. Its 119 expected failures are pending implementation requirements, not completed evaluator functionality.

Verification command: `JAXPROP_TEST_BACKENDS=HEOS python tests/run_tests.py -q --tb=short --junitxml=<run>/junit.xml` (set the environment variable using the local shell). Normal Windows execution permissions were used for the pytest fixtures. The run emitted 14 static-JAX-metadata warnings, one complex-input conversion warning, and one list-input JAX deprecation warning. The latter two expose current validation gaps covered by strict expected failures. REFPROP is outside this phase's HEOS verification scope; the Phase 1 optional-backend result remains recorded above.

## Phase 3 Implement the shared state foundation

- [x] Implement/export the shared typed-error hierarchy and structured context required by controlled state access; wire construction/evaluator request validation in the following phases.
- [x] Standardize and preserve fluid/stream identity in every returned state and through collection operations.
- [x] Fix the overwritten `alpha_p` alias and validate all canonical names and aliases.
- [x] Fix stacking to add a state axis while preserving existing singleton dimensions. Cover indexing, reversal, concatenation, empty inputs, metadata mismatch, and nonmutation. Reject out-of-range static indices instead of accepting JAX's implicit clamping.
- [x] Add property availability and controlled public lookup so explicit unsupported requests raise informative errors through attribute, canonical-name, and alias access.
- [x] Keep numerical storage and state operations compatible with the declared JAX transformations. Test dictionary conversion and serialization of state data.

Acceptance includes consistent behavior for `state.quality_mass`, `state.Q`, and `state["Q"]`, plus metadata and shape preservation through every state operation.

Acceptance verified on 12 November 2026. [FluidState](jaxprop/fluid_state.py) uses private numerical storage, immutable identity/provenance, explicit enabled properties, and per-element status/validity. It broadcasts constant properties, preserves singleton axes, checks model/reference compatibility, intersects collection property availability, and preserves mixed-provider provenance. Public canonical/alias lookups distinguish unsupported properties from locally unavailable values; perfect-gas quality access now raises the same informative typed error through all three access paths.

The [validated property schema](jaxprop/property_schema.py) restores `alpha_p` and rejects duplicate canonical definitions, alias collisions, and shadowing. The [typed errors](jaxprop/errors.py) expose structured context and support pickle round trips. State `to_data()` / `from_data()` and pickle preserve values, masks, identity, and access policy; NumPy inputs and exports cannot mutate existing states. Native/JAX numerical storage remains compatible with full-state JIT and floating-property derivatives. The perfect-gas gradient helper now differentiates numerical dictionaries rather than integer status/bool validity leaves.

| New Phase 3 tests | Passed |
| --- | ---: |
| [Typed error foundation](tests/test_errors.py) | 36 |
| [State foundation](tests/test_state_foundation.py) | 86 |
| [Evaluator metadata and gradient integration](tests/test_evaluator_state_metadata.py) | 23 |
| **Total new cases** | **145** |

Forty prior strict expected failures now pass and their markers were removed. The final full HEOS + REFPROP run completed with **764 passed, 79 strict expected failures, zero unexpected failures, and zero skips** in 366.65 seconds. The 79 remaining gaps belong to later phases. Numerical comparison coverage exactly matches Phase 1: **84,950 comparisons, 50,189 finite values, 34,761 explicitly unavailable values, and zero failed comparisons**. [Summary](development/test_audit/results/phase3/full_20261112_214051/summary.json), [pytest log](development/test_audit/results/phase3/full_20261112_214051/pytest.log), [JUnit results](development/test_audit/results/phase3/full_20261112_214051/junit.xml).

Verification used `JAXPROP_TEST_BACKENDS=HEOS,REFPROP` with the same complete pytest discovery as `python tests/run_tests.py -q`; a local recorder captured comparison counts and JUnit results. Tables were generated in fresh isolated pytest fixtures with normal Windows execution permissions. Seventeen warnings remain: 15 static-array metadata warnings and the two known input-validation warnings. Their fixes remain in the later reference/JAX and evaluator work.

Internal numerical comparisons now read raw enabled properties while retaining the repaired finite-core, shape, availability-pattern, and tolerance assertions. Profiling identified expensive Equinox method binding when bicubic finite differences repeatedly called `raw_property`. The gradient helpers now snapshot one numerical dictionary per sampled state. A ten-node probe improved from 0.480 s to 0.130 s with all 1,520 value/gradient entries and NaN patterns exactly equal. [Scoped benchmark](development/test_audit/results/phase3/snapshot_performance_probe.json). The full fresh-table rerun above verifies the optimized path; no finite-difference equations or numerical budgets were changed.

Reference IDs provisionally identify the model, supplied source/backend, and applicable construction parameters; actual source-reference fingerprints and alias canonicalization remain Phase 5. Phase 4 replaces ordinary calls' provisional full-schema declarations with honest per-call availability and typed static-request validation. Specialized equilibrium/metastable helpers remain in their evaluator/phase scopes. Perfect-gas quality-input and saturation-operation rejection now pass; physical calibration, missing properties, and numerical failure behavior remain Phase 7.

## Phase 4 Standardize construction and evaluator capabilities

- [x] Implement/export `create_fluid` with validated provider/model/backend routes and actual evaluator classes, without application-side adapters. Validate component-based binary settings but reject construction explicitly until Phase 9 supplies that model.
- [x] Expose immutable `FluidConfiguration`, `FluidCapabilities`, and `FluidDomain` snapshots. Preserve names/identifiers/backend selections; reject unsupported perfect-gas REFPROP calibration instead of mislabeling HEOS results. Protect source identity, reference IDs, and legacy composition against mutation/deletion.
- [x] Publish actual supported pairs, properties, diagnostics, phase operations, execution modes, and domain restrictions. Keep target model completeness separate from current capability publication; do not advertise missing PG properties or binary inversions.
- [x] Export `StatusCode` with the established values: success 0, invalid input 1, out of domain 2, nonconvergence 3, provider failure 4. Derive ordinary-call enabled properties from actual evaluator capabilities, removing disabled/inapplicable NaN slots.
- [x] Validate unknown/molar/noninteger pairs, unknown/invalid/unsupported options, and malformed real inputs before numerical evaluation/JIT. Give fluid/provider/model/request/reason/supported alternatives in typed errors.
- [x] Apply common `get_state(input_type, prop_1, prop_2, *, generalize_quality=False, supersaturation=False, errors="raise")` vocabulary and real scalar/list/NumPy/JAX broadcasting, preserving singleton/empty axes. Validate error-mode requests; reject not-yet-implemented numerical mask policies explicitly rather than claiming support.
- [x] Reuse configured evaluator objects and immutable setup outside repeated calls. Keep native JIT/dynamic inputs and use a lightweight checked descriptor rather than Equinox method wrappers. Cache only immutable dtype classification; validate shapes and requests each call without transferring traced numerical data to the host.
- [x] Retain the earlier state-constructor and lazy-validity improvements, with public validation, availability errors, serialization, collections, and JAX control-flow/derivative behavior covered.
- [x] Verify empty broadcast batches preserve full-state shapes, metadata, status, and validity in eager/outer-JIT execution for callback, bicubic, and perfect gas. Verify alternating HP/PT requests with changing values compile once per pair and reuse both native compiled paths.

The shared Phase 4 layer is complete. The target numerical failure policies remain to be implemented in the provider/model phases: source/callback behavior in Phase 6, perfect-gas clipping/completeness/reference equations in Phase 7, bicubic bounds/inverse/failure behavior in Phase 8, and consolidated binary construction/inversions in Phase 9. Current capabilities expose `error_modes={"raise"}`; mask requests are explicitly unsupported until those implementations land. Valid-state JAX execution remains usable while the target transformed-failure policy stays a strict expected failure.

The source and callback publish twelve tested mass pairs; bicubic eight; perfect gas six; the legacy direct binary object only scalar HP. Ordinary source/callback/pure-table states enable 25 properties, perfect gas 13, and legacy binary states 22. Generalized quality remains supported by direct CoolProp where numerically verified. Unsupported diagnostics are rejected, and disabled properties raise `UnsupportedPropertyError`; supported but locally undefined fields still use `PropertyUnavailableError`/validity masks. Factory construction of the component-object binary target remains explicitly owned by Phase 9.

Verification on 13 November 2026: full HEOS + REFPROP **1,208 passed, 28 strict expected failures, zero unexpected failures/skips**, with **50,189 finite comparisons unchanged** and zero failures. There are now 4,083 paired-unavailable comparisons (54,272 total). The 30,678 removed comparisons were NaN placeholders for properties now explicitly unsupported; separate capability/unsupported-property assertions cover those semantics. No finite numerical-oracle tolerance or coverage was removed. [Full result](development/test_audit/results/phase4/full_20261113_111610/summary.json).

The final reviewed contract/native-JAX run passed **396 cases with 28 expected failures**, including deletion/reference/composition guards, six invalid legacy-ratio cases, two additional callback-pair comparisons, and the final lightweight validation path. [Final contract result](development/test_audit/results/phase4/final_contract.xml). New files add 67 factory/configuration cases and 158 shared API cases; separately added target-minimum checks retain missing PG/binary features under their real owner phases. Earlier expected failures fell from 78 to 28.

A subsequent [JAX edge-case review](tests/test_phase4_jax_edges.py) added five passing cases for empty broadcast batches and alternating-pair compilation reuse. No production changes were required; the two existing bicubic static-reference warnings remain owned by Phase 5.

A longer serial guard compared the committed fast-native snapshot `a9e2c92` with Phase 4 using four workers per version, nine interleaved samples per workload, and synchronized CPU/x64 outputs. Perfect-gas scalar/batch medians changed by +2.5%/+2.0%, within worker variability; bicubic scalar/batch medians improved by 34%/42%. Initial short-run perfect-gas regressions did not repeat. Normalized optimized temperature-projection instruction structures are unchanged and callback-free; eager scalar request/binding overhead is approximately 6 microseconds for perfect gas and 10 for bicubic. The compact [result](development/performance/phase4_validation.json) retains per-case paired ratios and compiler evidence. Temporary source copies, workers, and tables were removed. This establishes no material repeatable regression in the sampled warmed calls, while construction/loading, application/device coverage, and the formal release gate remain Phase 13.

Acceptance covers provider selection, immutable metadata/capabilities, common ordinary-call syntax, dtype/shape behavior, honest availability, and typed static errors over the currently implemented model/domain. It does not claim completion of later physical/reference/phase/failure functionality.

## Phase 5 Initialize CoolProp source-reference data for every fluid object

- [x] Implement one dedicated CoolProp initialization path for critical and liquid/vapor triple-reference points, independently of the selected evaluator.
- [x] Expose source fluid/composition, backend/version, reference convention, point interpretation, availability, and failure reason. Reuse immutable source-reference data consistently.
- [x] Replace Phase 3's provisional construction/reference IDs with source-reference fingerprints, including actual source enthalpy/entropy anchor values, model parameters, and component reference identities. Detect changed global CoolProp reference conventions and carry the fingerprint into table cache validation.
- [x] Separate array-valued reference states from Equinox static metadata. Reference records retain only immutable scalar metadata; explicit state access materializes NumPy arrays outside the static evaluator records. Verify warning removal without weakening JAX behavior or cache identity.
- [x] Populate applicable source-reference points for perfect-gas and bicubic objects while enforcing their own evaluation capabilities. Extend perfect-gas calibration to the selected source backend, with HEOS and REFPROP tests.
- [x] Support pseudo-pure Air and refrigerant blends with explicit classification and interpretation of their lower saturation endpoints, preserving distinct liquid/vapor pressures.
- [x] Handle unavailable or ambiguous reference points individually, with informative errors on explicit access and usable ordinary calculations where supported.
- [x] Record reported reference coordinates and any numerical evaluation offsets. Define availability of singular/undefined properties at critical states.

Acceptance includes matching source-reference metadata across evaluators for CO2 and Water, graceful Air initialization, and reference availability alongside perfect-gas rejection of quality/saturation evaluation.

The shared [source-reference implementation](jaxprop/source_references.py) initializes each point independently, resolves source aliases, fingerprints actual h/s anchors, and stores frozen primitive records. Supplied fresh source objects keep ordinary source evaluation and reference snapshots on the same construction convention. Critical coordinates are attempted exactly before explicit recorded fallbacks; singular response properties remain unavailable. Composite binary records deliberately have no aggregate critical/triple points and expose their component reference records. Full native cache provenance remains enforced in addition to the shared source ID, with an initialization-race guard before table generation/publication.

Focused HEOS + REFPROP verification passed **304 cases with 12 existing strict expected failures**, including 43 new source-reference cases. The native/shared-contract run passed **165 cases with 14 expected failures**, and metadata/cache checks passed **40 + 69 cases plus one new initialization-race case**. No static-reference warnings remain.

The full HEOS + REFPROP run on 13 November 2026 passed **1,267 tests with 26 strict expected failures, zero unexpected failures, and zero skips**. All **54,272 numerical comparisons** passed, retaining **50,189 finite values and 4,083 paired-unavailable values** exactly as in Phase 4. Its 33 warnings are deliberately exercised cache-regeneration diagnostics. [Full result](development/test_audit/results/phase5/full_20261113_133321/summary.json). Two cases added after collection separately verified the forced critical-coordinate fallback (**1 passed**) and recorded the existing runtime quality/reference gap (**1 strict expected failure**). [Final reference edges](development/test_audit/results/phase5/final_reference_edges.xml). The source-reference module now has 44 passing cases plus that narrow Phase 6 expected failure.

The matched [Phase 5 timing guard](development/performance/phase5_validation.json) uses four independent workers per version against Phase 4 commit `49aea3d`, nine interleaved samples per native workload and synchronized CPU/x64 results. No material repeatable warmed-call regression appeared: perfect-gas scalar/batch medians changed from 36.0/109.6 to 34.3/111.6 microseconds; bicubic from 118.5/793.1 to 117.3/808.8 microseconds. Normalized temperature-projection instructions are unchanged; all 38 sampled supported fields and two derivatives agree. Five-sample warmed construction checks improve direct source from 4.72 to 0.97 milliseconds and perfect gas from 6.68 to 3.12 milliseconds. Cached-loading medians are 25.73 versus 22.08 milliseconds, with substantial paired variability retained. Full cache validation stays enabled. Temporary benchmark source/workers/tables were removed. Application/device coverage and the formal release gate remain Phase 13.

## Phase 6 Bring ordinary CoolProp and FluidJAX onto the contract

- [ ] Apply uniform names, identifiers, options, shapes, property availability, and errors to `Fluid` and `FluidJAX`.
- [ ] Resolve diagnostic option semantics, including generalized quality, subcooling, superheating, and saturation departures, with explicit support declarations. Add the uniform source/callback saturation entry point and its capability only after validation; ordinary quality input pairs already work.
- [ ] Give `FluidJAX` dedicated eager, outer-JIT, vectorization, batch failure, and derivative tests, including list/array acceptance and identity preservation.
- [ ] Validate callback derivative accuracy over representative input scales and document supported transformation modes and derivative order.
- [ ] Preserve a configured source's reference convention in runtime quality/phase helper clones. `calculate_trimmed_quality` currently compares an old object's entropy with a newly constructed critical entropy under the current process-global convention. The isolated strict regression `test_configured_source_quality_keeps_its_original_entropy_convention` exposes a changed quality after an external entropy shift while core values and stored references remain stable. Reconcile clone-based helpers without changing global settings or adding repeated source initialization to native evaluators.

Acceptance requires the same source-compatible values and common behavior through both evaluators, with capabilities explaining the supported execution modes.

## Phase 7 Complete the perfect-gas evaluator

- [x] Use native `jax.jit` for the numerical perfect-gas PyTree, keeping only the input-pair selector static. Python scalar values and model parameters remain dynamic, without per-value compilation or an additional filtered-method wrapper. Convert list/array inputs inside the traced body, preserving Python weak typing. Tests cover changing values, lists, broadcasts, outer JIT/AD, vectorization, and changing dynamic fluid constants; remove the resolved list-input strict expected failure.
- [x] Inspect compiled numerical/projection kernels before rewriting algebra. The existing perfect-gas formulas are vectorized JAX arithmetic; inspected full/temperature kernels contain no host callbacks, and projected temperature removes unused density, entropy, and transport work. Preserve this pruning in later property/model changes.
- [ ] Apply the common construction, identity, call, broadcast, property-availability, and failure contracts.
- [ ] Complete available core properties, including internal energy and a defined gas-state marker.
- [ ] Complete the perfect-gas derivative properties defined by the model, including expansion/compressibility, bulk moduli, and Joule-Thomson behavior. Test their consistency with the gas constant, heat capacities, and sound speed before declaring them available.
- [ ] Adopt and document the chosen enthalpy/entropy reference conventions, updating forward and inverse equations consistently.
- [ ] Define identity for dynamically modified model views. Constructor fingerprints include actual calibrated constants, but low-level `eqx.tree_at` edits used in differentiation/optimization retain the original static identity. Provide a supported reconfiguration/materialization path that refreshes identity before persisting or combining changed physical models, while keeping numerical parameters dynamic under JIT.
- [ ] Replace silent invalid-input clipping with the selected domain/failure policy, and explicitly reject phase-equilibrium inputs and properties.
- [ ] Add independent ideal-gas identities, off-calibration conditions, transport checks, batch cases, and advertised JAX derivative tests.
- [ ] Define a derivative-result schema or explicit quantity-kind metadata for property-gradient helpers, including derivative units and primal validity. Prevent derivative-valued fields from being combined with ordinary thermodynamic state values.

Acceptance requires thermodynamic consistency and common API behavior for the explicitly selected perfect-gas physical model.

## Phase 8 Harden bicubic interpolation and table lifecycle

- [x] Use native `jax.jit` for the numerical bicubic PyTree, with a static input-pair selector and dynamic state values/coefficients. Varying Python floats reuse compilation without additional method wrappers. Use a direct scalar path, retain broadcast batching, and remove redundant nested filtered JIT from the private scalar dispatcher. Regression tests cover trace reuse, scalar input forms, singleton/broadcast shapes, outer full-state JIT, and derivatives.
- [ ] Apply the common metadata, identity, call, shape, capability, and error contracts to `FluidBicubic`.
- [x] Validate the current table constructor's bounds, integer grid sizes, gradient scheme, and source/phase/composition combinations. Reject the unimplemented metastable central scheme explicitly. Revisit these checks when integrating the common factory/configuration API.
- [x] Reject failed source evaluations and incomplete core arrays before publication. Initialize unfilled buffers with NaNs and remove copied-neighbor/spinodal-clipping fallbacks; preserve an existing cache if construction fails.
- [ ] Complete phase-aware interpolation-cell validity and applicable masking/domain behavior beyond the current structural and finite-core checks.
- [x] Use one complete generation manifest across all four table modes. Include settings, source/reference and native-file fingerprints, numerical precision, algorithm/library versions, and schema; require exact manifest matching. Test reuse, invalidation, malformed caches, and atomic concurrent publication.
- [x] Separate the persisted generation manifest from the numerical runtime PyTree while preserving public provenance and exact cache validation. The private numerical table excludes metadata; `.metadata` and the public `.table["metadata"]` inspection view retain the full manifest. Tests verify that adding provenance does not add dynamic leaves, and that coefficients remain dynamic compiled inputs. The `.table` outer dictionary is now a snapshot.
- [x] Avoid redundant cached-array conversion when storage and dtype already match, while retaining manifest, structure, dtype, and finite-core validation. Existing correctly typed JAX arrays are reused; NumPy and wrong-dtype arrays still convert. Cached-load timing is recorded separately below.
- [ ] Separate a lean numerical evaluator from the full inspection/cache payload. Forward evaluation currently carries 266 table arrays in the runtime PyTree, although it principally needs coefficients and axes. Keep complete persisted validation and public table inspection, retain dynamic model parameters/gradients, and measure JIT tree traversal as well as compiled execution. Separating generation and evaluation modules can also avoid invalidating tables for runtime-only code changes without weakening generation-code fingerprints.
- [ ] Profile cached loading as unpickling/device placement, source fingerprinting, payload validation, and runtime setup separately. Consider NumPy-first persisted storage with validated bulk placement of only the numerical evaluator arrays. Preserve complete manifests, global reference-change detection, dtype/shape/finite-core checks, atomic publication, and public inspection behavior; do not cache mutable source signatures without a valid invalidation mechanism.
- [ ] Prototype coefficients grouped along a property axis to share gathers and interpolation basis work. Preserve optional NaNs, output shapes/dtypes, dynamic coefficient updates, and gradients. Require projection-specific benchmarks so calculating one property does not unnecessarily evaluate all properties; keep runtime representation changes distinct from the persisted cache schema unless a schema change is justified.
- [ ] Introduce raw selected-property numerical evaluation for inverse residuals, avoiding full-state assembly during tracing. Profile/precompute reusable coarse-grid scaling/initial-guess data where safe, with explicit refresh rules for dynamic model updates. Verify values, derivatives, compile time, and failure behavior for every supported inverse pair.
- [ ] Evaluate a one-dimensional fixed-pressure entropy inverse instead of the present two-dimensional `PSmass` solve. Treat this as a numerical solver change: independently validate roots, domains, initial guesses, near-critical behavior, convergence/failure policy, and derivatives before adopting it.
- [x] Preserve exact independent pressure/enthalpy grids and analytic derivatives in every mode. Correct the liquid instability criterion, descending vapor predictor sign, and main-grid/saturation success accounting exposed by real cache integration tests.
- [ ] Validate metastable solver residuals relative to the finite-difference perturbations, and improve solve accuracy or derivative construction where residual errors can dominate those steps. Cache identity and finite arrays alone do not establish derivative accuracy.
- [ ] Extend saturation payload integrity checks to compare pressure nodes with the declared log-pressure grid. Decide whether payload checksums are required to detect finite-value file corruption that structural checks cannot identify.
- [ ] Enforce the explicit domain policy and validate inverse roots/residuals. Correct iteration-limit wiring and bound the inverse problems where appropriate.
- [ ] Verify the selected eager/transformed failure policy at inverse-solver boundaries, including native JAX asynchronous error delivery and explicit synchronization. Preserve informative failure diagnostics without adding host callbacks to successful iterations; Optimistix currently uses conditional failure callbacks.
- [ ] Correct the saturation central-derivative denominator and complete the saturation, mixture, and metastable table modes retained in the public capability set. Replace the legacy pressure-only raw-dictionary saturation helper with the uniform typed operation before advertising it as a common phase capability.
- [ ] Add independent polynomial interpolation tests, cell-boundary checks, near-critical/application-relevant domains, and value/derivative comparisons.

Acceptance requires trustworthy table construction/reuse, informative domain and solver failures, and source-compatible values/derivatives within documented resolution-dependent tolerances.

The cache lifecycle subset was implemented early at the user's request on 12 November 2026. All four generators now use one manifest and array initializer, exact independent pressure/enthalpy coordinates, one validated loader, and one atomic publisher. The manifest covers mode/composition (including explicit `None` selectors), bounds/grid sizes, gradient scheme, saturation settings, dtype/JAX precision, package/numerical-library versions, generation-code fingerprints, source configuration and enthalpy/entropy anchors. HEOS definitions and resolved REFPROP native DLL/FLD/PPF/MIX/HMX data have content fingerprints. An unverifiable source disables persistent reuse/publication rather than claiming a verified identity. Runtime identifiers, file locations, and inverse initial-guess stride are excluded from generation identity.

Exact canonical manifest comparison rejects extra, missing, or changed fields. Default filenames include a manifest digest, removing rounded-mixture-ratio collisions and allowing configurations to coexist. Legacy caches are not trusted under the new format. Structural validation checks property schema, shape, dtype, independent grid coordinates, finite core values/derivatives/coefficients, and saturation array structure; legitimate optional NaNs remain allowed. Corrupt/incompatible files regenerate, while I/O errors propagate. Failed evaluations, invalid generated data, reference changes during construction, serialization failures, and failed atomic replacements preserve the previous file and do not publish partial data.

Real integration checks cover equilibrium forward/central tables, a binary mixture, and both metastable Water branches. These tests also exposed and verified fixes for exact pressure/enthalpy axes being overwritten by solver residuals, a spurious liquid instability criterion, descending vapor predictor direction, and saturation/main-grid counting. The missing metastable central implementation now raises an explicit unsupported-option error. Broader metastable residual/derivative accuracy and cell validity remain unchecked above; matching metadata is not an accuracy proof.

| New cache tests | Passed |
| --- | ---: |
| [Cache policy, validation, and atomic I/O](tests/test_bicubic_cache.py) | 61 |
| [Source/reference/native-file identity](tests/test_cache_metadata.py) | 40 |
| [Real numerical generation and reload](tests/test_bicubic_cache_integration.py) | 8 |
| **Total** | **109** |

Full HEOS + REFPROP verification: **873 passed, 79 strict expected failures, zero unexpected failures, and zero skips** in 391.04 seconds. Numerical coverage is unchanged: 84,950 comparisons, 50,189 finite values, 34,761 explicitly unavailable values, and zero failed comparisons. [Summary](development/test_audit/results/cache_validation/full_20261112_223133/summary.json), [pytest log](development/test_audit/results/cache_validation/full_20261112_223133/pytest.log), [JUnit results](development/test_audit/results/cache_validation/full_20261112_223133/junit.xml). The same recorded full-suite procedure as Phase 3 used `JAXPROP_TEST_BACKENDS=HEOS,REFPROP` and fresh isolated pytest fixtures. Expected cache-regeneration diagnostics and the existing static-array/input-validation warnings remain visible. The Windows concurrent reader/writer test additionally passed 20 repeated runs; it permits bounded explicit sharing denials while still rejecting partial, truncated, or corrupt pickle reads.

## Phase 9 Consolidate the two-component mixture implementations

- [ ] Define one public binary-mixture object, composition convention, shared pressure/temperature mixing kernel, and common state schema. Enable the Phase 4 factory component-object/mass-fraction route only when this real model is implemented; retain its strict target test meanwhile.
- [ ] Accept component fluid objects through the common interface so component evaluation can use supported numerical providers.
- [ ] Reconcile complementary outputs, repair the missing mixture-state type/schema, and preserve requested stream identifiers.
- [ ] Validate density, enthalpy, entropy, heat capacities, expansion/compressibility, Gruneisen, Joule-Thomson, and transport mixing relations against the selected physical model.
- [ ] Reconcile isentropic compressibility with sound speed and retain explicitly named acoustic relaxation assumptions/outputs.
- [ ] Provide the supported property-pair inversions, scalar/broadcast composition handling, validation, and claimed JAX transformations.
- [ ] Add composition-limit, conservation, thermodynamic-identity, inverse, derivative, failure, and import/construction tests. Consolidate obsolete duplicate entry points as part of the selected uniform interface.

Acceptance requires one validated mixture model with consistent derived properties and common external behavior. The existing implementations supply useful pieces of this model and need their equations checked during consolidation.

## Phase 10 Repair and test phase calculations and uncovered public paths

- [ ] Repair public blending input/output schemas and test endpoint recovery, transition continuity, aliases, invalid settings, and availability.
- [ ] Verify actual generalized/trimmed quality and saturation-departure diagnostics, including defined outcomes outside their applicable state region.
- [ ] Test actual metastable branches, direct density/temperature paths, stability limits, solver guesses, algorithms, and unsuccessful solves.
- [ ] Repair density/temperature property-grid generation and validate small saturation, pseudocritical, quality, and spinodal grids.
- [ ] Add independent thermodynamic identities and checks for derivative helpers used by these paths.

Acceptance requires every retained advertised public path to produce the specified state/schema or its informative declared failure outcome.

## Phase 11 Verify plotting, lifecycle, and application integration

- [ ] Provide phase/reference plotting through the common supported path, with headless artist creation/update, option, and stream-grouping tests.
- [ ] Repair supported object serialization and verify property evaluation after a round trip.
- [ ] Make application-wide JAX platform, precision, and exception-hook configuration explicit and test import behavior in subprocesses. In particular, stop overriding the consumer's device choice by forcing CPU at import; preserve explicit precision settings and verify numerical budgets on supported configurations before changing defaults.
- [ ] Document and test compiled consumer functions and explicit mask-policy execution alongside informative eager errors. Eager raising property access necessarily inspects concrete availability; avoid repeated host synchronization in JIT/vectorized application paths. Measure GPU transfers/reductions separately, including lazy-mask materialization and complete state export.
- [ ] Document/test repeated evaluation, state isolation, and the supported concurrency model.
- [x] Preserve NumPy storage when combining NumPy states and avoid reconstructing already-validated collection data through redundant eager JAX masking/broadcast operations. Retain compatibility, shape, identity, status, and availability checks. Mixed storage remains JAX; reversal preserves storage; NumPy results remain read-only. Combining signed/unsigned status arrays retains integer codes, including under JIT. Provisional two-state concatenation was 3.9 times slower than PyPI 0.5.8 before this fix.
- [ ] Prepare repeatable workloads for construction, table generation, repeated states, and transformed calls. The formal comparison with the downloaded PyPI release and its performance acceptance gate belong to Phase 13.
- [ ] Profile remaining trusted internal state-assembly/identity checks if source evaluation remains material; keep the public constructor fully validating and do not cache mutable physical/reference state.
- [ ] If full-state output handling is still material, evaluate packed numerical storage by dtype/trailing property axis. Preserve property dtypes, all state axes, availability, serialization, JIT/vmap/AD, and eager projection performance before adopting it.
- [ ] Test numerical smoothing and solver-selection helpers used by the retained calculation paths.

Acceptance requires reusable objects that behave predictably in a consuming application, including the advertised plotting, worker, and JAX execution modes.

## Phase 12 Validate ThermOpt and prepare interface publication

- [ ] Adapt ThermOpt construction, setup/reference access, and plotting to the common JaxProp API, reusing configured fluid objects.
- [ ] Compile compatible complete component/cycle/objective functions with dynamic numerical inputs and reusable fluid models, rather than relying only on individually compiled state calls. Check trace counts and preserve solver/result behavior. The native PG/bicubic entry points now avoid filtered-method overhead, while outer JIT also removes repeated Python property access/host boundaries and enables whole-consumer optimization. Retain explicit synchronization when benchmarking or delivering eager numerical failures.
- [ ] Test compressor, expander, and discretized heat-exchanger entry points across compatible evaluators, including energy balance, state recovery, identity, and constraint arrays.
- [ ] Rerun the recuperated CO2 reference case across source-compatible real-fluid evaluators with resolution-appropriate tolerances.
- [ ] Add an applicable phase-changing integration case and a perfect-gas integration case with expectations for its physical model.
- [ ] Align CI and documented test commands, including explicit optional REFPROP and slow-case selection.
- [ ] Update examples, API documentation, and release/migration notes to the selected consistent behavior and verified capability matrix.

Acceptance requires the intended consumer workflows to pass through the common API, with every claimed capability covered by the corresponding tests and the verification evidence recorded. Prepare the release documentation here; publish the refactored release after the Phase 13 performance gate passes.

## Phase 13 Compare performance with the latest PyPI release

Download the latest stable JaxProp release available on PyPI when this phase starts, before publishing the refactored implementation. Use that unchanged published package as the old implementation baseline. The aim is to preserve practical performance across representative uses of the new consistent interface, while retaining its correctness checks and complete state behavior.

- [ ] Resolve and download the latest stable PyPI release. Record the version, download date, distribution artifact/hash, and dependency versions. Pin this baseline for the complete comparison and retain the artifact so results can be reproduced after PyPI changes.
- [ ] Install the published baseline and the new checkout in isolated environments/processes. Verify their imported package paths so the local checkout cannot shadow the PyPI installation. Match hardware, Python, compatible dependency versions, CoolProp source backend, JAX device/precision, and thread settings; record any unavoidable differences separately.
- [ ] Fix the workload matrix, numerical accuracy requirements, and regression budgets before measuring. Use an initial working target of at most 10% higher median time for comparable warmed evaluations, derivatives, and application workloads. Define separate budgets for construction, compilation, and table generation, accounting for the additional required reference initialization. Record any budget revisions before collecting acceptance timings.
- [ ] Implement equivalent calls through each version's own public API, using benchmark-side argument translation where signatures differ. Verify comparable physical states, outputs, reference conventions, solver tolerances, and table domains/resolutions before timing. Account explicitly for intentional perfect-gas reference/model changes. Mark missing legacy functionality as not comparable; do not present it as a speedup or disable new validation/metadata to improve the primary comparison.
- [ ] Measure object/reference initialization, fresh table generation, cached table loading, first-call JAX compilation, and warmed repeated evaluation separately. Give each version independent table/compilation caches, with explicit cold/warm conditions. Include the complete public call cost; use kernel-only timings to diagnose overhead rather than replace the API comparison.
- [ ] Include changing-value Python scalars and pre-created JAX arrays as distinct workloads, with compilation/trace counts. Fixed-value calls alone concealed per-value static recompilation in both original native evaluators. Distinguish expected shape/dtype/static-selector compilation from unintended specialization on numeric values; include dynamic model-parameter updates.
- [ ] Compare full-state return, selected-property projection, explicit validity/export, and complete compiled consumers at multiple batch sizes. Inspect optimized IR for pruning/fusion and conditional error callbacks where helpful. Extend device-specific measurements after consumer-controlled platform setup; do not infer GPU behavior from CPU timings.
- [ ] Run the representative cases below wherever both versions provide the relevant functionality, including comparable HEOS and optional available REFPROP configurations. For changed mixture formulations, compare the common validated operations and report additional new capabilities separately.
- [ ] Warm up, repeat and interleave measurements in independent worker processes; synchronize all JAX outputs with `block_until_ready` before stopping timers. Use enough calls/repetitions to resolve short operations and report median, variability/uncertainty, absolute time, throughput, and new/old timing ratios for every case. For optimizers, report iteration/property-call counts alongside wall time so solver behavior can be distinguished from evaluation cost.
- [ ] Save runnable benchmark scripts, environment/configuration manifests, raw samples, correctness results, and a comparison report. Profile and fix repeatable regressions beyond the declared budgets, rerun the affected cases and numerical checks, and add any remaining findings to this checklist. Keep individual case results visible rather than hiding regressions in an overall average.

| Workload | Representative cases |
| --- | --- |
| State construction and access | Construct states; canonical/alias lookups; raw/dictionary export; indexing, stacking, and concatenation. |
| Fluid property evaluation | Direct CoolProp, callback/JAX, bicubic, perfect gas, and common binary-mixture operations; scalar/singleton inputs, small and large batches, and broadcast matrices. |
| States and input pairs | CO2, Water, and Air/Nitrogen; applicable liquid, gas, supercritical/near-critical, saturation, and metastable conditions; forward `PT`/`HmassP` and inverse `PSmass`/other supported pairs. |
| Differentiation and execution | Eager, JIT, vectorized calls, and supported first derivatives of matching floating properties; separate compilation from warmed execution. |
| ThermOpt consumers | Compressor, expander, discretized heat exchanger, repeated cycle evaluations, and the recuperated sCO2 optimization; include applicable perfect-gas and phase-changing cases. |

Acceptance requires numerically valid comparisons and no repeatable material slowdown beyond the predeclared budgets for comparable workloads. Resolve inconclusive timings with additional measurements. Publish the per-case report and retain explanations for unavoidable new setup work or non-comparable capabilities. The refactored release remains unpublished until this phase passes.

### Validated JAX performance snapshot

The native-JAX efficiency work was verified on 13 November 2026. Native `jax.jit` entry points keep numerical values/model parameters dynamic with only the input-pair selector static. Lazy validity preserves masked values/status while avoiding duplicate boolean outputs. The direct bicubic scalar path avoids unnecessary batching scaffolding. The [efficiency audit](z_JAX_EFFICIENCY_AUDIT.md) records source/compiler findings and remaining ideas by phase.

The complete HEOS + REFPROP suite passed **936 cases with 78 strict expected failures, zero unexpected failures, and zero skips**. All **84,950 comparisons** remain, including 50,189 finite values and 34,761 unavailable values, with zero failed comparisons. [Verification](development/test_audit/results/jax_efficiency/full_20261113_094055/summary.json).

The final matched CPU/x64 comparison used three independent workers per version, 37 comparable workloads, and five samples per warmed case. All 42 sampled value/shape groups passed, with exact sampled derivative agreement. Full-state perfect-gas scalar/batch calls were about 4.1/3.8 times faster than PyPI 0.5.8; bicubic scalar calls about 2.3 times faster and the 1,024-state batch approximately comparable. Changing Python values now reuse compilation. Cached loading remained 31.6 ms versus 22.6 ms, with complete validation retained. The full formal gate remains pending broader workloads/backends/devices/application cases and setup budgets.

Disposable benchmark workers, extracted distributions, tables, profiles, lowering dumps, and intermediate runs were removed at the user's request. The compact [final report](development/performance/README.md) and [case summaries](development/performance/summary.json) preserve final timings, variability, environment, distribution/source hashes, scope, and correctness results. This snapshot does not complete Phase 13; its eventual comparison should resolve/download the then-latest published baseline again.

## Verification log

| Work | Status | Evidence |
| --- | --- | --- |
| Pre-phase-1 HEOS suite | Completed baseline | 252 passed. [Recorded run](development/test_audit/results/baseline_20261112_195233/summary.json). Shared assertion-helper limitation is documented. |
| Array-aware diagnostic comparison | Completed experiment | 167 passed with 25,022 numerical comparisons. [Recorded run](development/test_audit/results/baseline_20261112_195233/array_oracle_summary.json). |
| Persistent test repairs | Completed phase 1 | 480 passed across HEOS and REFPROP, zero failures/skips, 50,189 finite values compared. [Summary](development/test_audit/results/phase1/full_20261112_203206/summary.json), [pytest log](development/test_audit/results/phase1/full_20261112_203206/pytest.log). |
| Uniform public contract | Completed phase 2 | 218 new cases: 99 passed, 119 strict expected failures. Full HEOS: 405 passed, zero unexpected failures/skips. [Summary](development/test_audit/results/phase2/full_heos_20261112_205840/summary.json). |
| Shared state foundation | Completed phase 3 | 145 new passing tests, 40 prior gaps resolved. HEOS + REFPROP: 764 passed, 79 strict expected failures, zero unexpected failures/skips; Phase 1 numerical comparison counts preserved. [Summary](development/test_audit/results/phase3/full_20261112_214051/summary.json). |
| Table cache identity and lifecycle | Completed subset of phase 8 | 109 new passing tests. HEOS + REFPROP: 873 passed, 79 strict expected failures, zero unexpected failures/skips; numerical comparison counts preserved. [Summary](development/test_audit/results/cache_validation/full_20261112_223133/summary.json). Other Phase 8 work remains open. |
| Shared construction and static request API | Completed phase 4 | HEOS + REFPROP: 1,208 passed, 28 expected failures; finite coverage preserved. Final contract/native checks: 396 passed, 28 expected failures. [Result](development/test_audit/results/phase4/full_20261113_111610/summary.json). |
| References and evaluator completion | Pending | Phases 5 through 11; implement/remove the corresponding strict expected failures. |
| Consumer integration and publication preparation | Pending | Phase 12. |
| Native JAX efficiency follow-up | Implemented and measured; broader optimizations phased | HEOS + REFPROP: 936 passed, 78 expected failures; zero failed numerical comparisons. Earlier native-evaluation regressions resolved; cached loading remains slower. [Audit](z_JAX_EFFICIENCY_AUDIT.md), [final report](development/performance/README.md). |
| Formal performance gate against the published PyPI baseline | Pending | Phase 13; resolve/pin the latest stable release at phase start and pass the complete comparison gate before publication. The provisional run does not replace this acceptance gate. |
