# JaxProp uniform fluid interface implementation plan

Uniform, consistent behavior across fluid objects is the primary goal. Backward compatibility is secondary. The implementation should protect valid physical calculations while establishing consistent construction, property calls, identity, array shapes, reference metadata, capabilities, and errors.

This checklist starts from the local 0.5.8 source at commit `ed66d108555e71d0d8543a1b670d05237a834221`, reviewed on 12 November 2026. The [interface audit](z_FLUID_INTERFACE_AUDIT.md) supplies the source findings and design requirements. The [test audit](z_PRE_REFACTOR_TEST_AUDIT.md) supplies the coverage gaps, runtime evidence, and proposed tests.

## Tracking changes

Check an implementation item when its source/test changes are present and its relevant acceptance checks pass. Record a short evidence entry beside the item or in the verification log, including the command, scope, result, and any explicit expected failures. A phase is complete when its checklist is complete.

Implement and verify one coherent change at a time. Start with tests for its intended behavior, implement the change, run the affected tests, and then run the shared contract/numerical checks relevant to its callers. Use the full HEOS suite at the phase boundaries. Run the declared optional-backend and slow integration cases before completing their corresponding release scope. Revisit completed items when a later change alters their contract.

The checked preparation items record completed audit work. Phases 1 and 2 are complete: the numerical test foundation is repaired, and the [public fluid contract](z_PUBLIC_FLUID_CONTRACT.md) fixes the intended interface with executable conformance cases. Production evaluator changes remain in Phases 3 through 12. The earlier array-aware diagnostic experiment is retained as historical evidence.

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

- [ ] Implement/export the shared typed-error hierarchy and structured context required by controlled state access; wire construction/evaluator request validation in the following phases.
- [ ] Standardize and preserve fluid/stream identity in every returned state and through collection operations.
- [ ] Fix the overwritten `alpha_p` alias and validate all canonical names and aliases.
- [ ] Fix stacking to add a state axis while preserving existing singleton dimensions. Cover indexing, reversal, concatenation, empty inputs, metadata mismatch, and nonmutation.
- [ ] Add property availability and controlled public lookup so explicit unsupported requests raise informative errors through attribute, canonical-name, and alias access.
- [ ] Keep numerical storage and state operations compatible with the declared JAX transformations. Test dictionary conversion and serialization of state data.

Acceptance includes consistent behavior for `state.quality_mass`, `state.Q`, and `state["Q"]`, plus metadata and shape preservation through every state operation.

## Phase 4 Standardize construction and evaluator capabilities

- [ ] Implement shared construction/configuration and consistent object metadata for the supported evaluators.
- [ ] Publish supported input pairs, properties, options, phase operations, JAX modes, and applicable domains through the common capability contract.
- [ ] Validate static unsupported requests before numerical evaluation or JIT entry. Include fluid, evaluator, request, reason, and relevant supported alternatives in errors.
- [ ] Apply one scalar/list/NumPy/JAX broadcasting policy and the selected per-element numerical failure policy.
- [ ] Create reusable fluid objects outside repeated cycle evaluations and verify that reuse preserves results and identity.

Acceptance requires changing the numerical provider through configuration while preserving the common call/output contract over its supported domain.

## Phase 5 Initialize CoolProp source-reference data for every fluid object

- [ ] Implement one dedicated CoolProp initialization path for critical and liquid/vapor triple-reference points, independently of the selected evaluator.
- [ ] Expose source fluid/composition, backend/version, reference convention, point interpretation, availability, and failure reason. Reuse immutable source-reference data consistently.
- [ ] Populate applicable source-reference points for perfect-gas and bicubic objects while enforcing their own evaluation capabilities.
- [ ] Support pseudo-pure Air and refrigerant blends with explicit classification and interpretation of their lower saturation endpoints, preserving distinct liquid/vapor pressures.
- [ ] Handle unavailable or ambiguous reference points individually, with informative errors on explicit access and usable ordinary calculations where supported.
- [ ] Record reported reference coordinates and any numerical evaluation offsets. Define availability of singular/undefined properties at critical states.

Acceptance includes matching source-reference metadata across evaluators for CO2 and Water, graceful Air initialization, and reference availability alongside perfect-gas rejection of quality/saturation evaluation.

## Phase 6 Bring ordinary CoolProp and FluidJAX onto the contract

- [ ] Apply uniform names, identifiers, options, shapes, property availability, and errors to `Fluid` and `FluidJAX`.
- [ ] Resolve diagnostic option semantics, including generalized quality, subcooling, superheating, and saturation departures, with explicit support declarations.
- [ ] Give `FluidJAX` dedicated eager, outer-JIT, vectorization, batch failure, and derivative tests, including list/array acceptance and identity preservation.
- [ ] Validate callback derivative accuracy over representative input scales and document supported transformation modes and derivative order.

Acceptance requires the same source-compatible values and common behavior through both evaluators, with capabilities explaining the supported execution modes.

## Phase 7 Complete the perfect-gas evaluator

- [ ] Apply the common construction, identity, call, broadcast, property-availability, and failure contracts.
- [ ] Complete available core properties, including internal energy and a defined gas-state marker.
- [ ] Adopt and document the chosen enthalpy/entropy reference conventions, updating forward and inverse equations consistently.
- [ ] Replace silent invalid-input clipping with the selected domain/failure policy, and explicitly reject phase-equilibrium inputs and properties.
- [ ] Add independent ideal-gas identities, off-calibration conditions, transport checks, batch cases, and advertised JAX derivative tests.

Acceptance requires thermodynamic consistency and common API behavior for the explicitly selected perfect-gas physical model.

## Phase 8 Harden bicubic interpolation and table lifecycle

- [ ] Apply the common metadata, identity, call, shape, capability, and error contracts to `FluidBicubic`.
- [ ] Validate constructor bounds/grid sizes and source/phase/composition configuration.
- [ ] Detect failed reference nodes and invalid interpolation cells. Reject or explicitly mask incomplete tables according to the selected policy.
- [ ] Expand cache identity and validation to cover generation settings, reference conventions, format, and source versions. Test reuse, invalidation, corruption, and atomic concurrent construction.
- [ ] Enforce the explicit domain policy and validate inverse roots/residuals. Correct iteration-limit wiring and bound the inverse problems where appropriate.
- [ ] Correct the saturation central-derivative denominator and complete the saturation, mixture, and metastable table modes retained in the public capability set.
- [ ] Add independent polynomial interpolation tests, cell-boundary checks, near-critical/application-relevant domains, and value/derivative comparisons.

Acceptance requires trustworthy table construction/reuse, informative domain and solver failures, and source-compatible values/derivatives within documented resolution-dependent tolerances.

## Phase 9 Consolidate the two-component mixture implementations

- [ ] Define one public binary-mixture object, composition convention, shared pressure/temperature mixing kernel, and common state schema.
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
- [ ] Make application-wide JAX platform, precision, and exception-hook configuration explicit and test import behavior in subprocesses.
- [ ] Document/test repeated evaluation, state isolation, and the supported concurrency model.
- [ ] Test numerical smoothing and solver-selection helpers used by the retained calculation paths.

Acceptance requires reusable objects that behave predictably in a consuming application, including the advertised plotting, worker, and JAX execution modes.

## Phase 12 Validate ThermOpt and publish the interface

- [ ] Adapt ThermOpt construction, setup/reference access, and plotting to the common JaxProp API, reusing configured fluid objects.
- [ ] Test compressor, expander, and discretized heat-exchanger entry points across compatible evaluators, including energy balance, state recovery, identity, and constraint arrays.
- [ ] Rerun the recuperated CO2 reference case across source-compatible real-fluid evaluators with resolution-appropriate tolerances.
- [ ] Add an applicable phase-changing integration case and a perfect-gas integration case with expectations for its physical model.
- [ ] Align CI and documented test commands, including explicit optional REFPROP and slow-case selection.
- [ ] Update examples, API documentation, and release/migration notes to the selected consistent behavior and verified capability matrix.

Acceptance requires the intended consumer workflows to pass through the common API, with every claimed capability covered by the corresponding tests and the verification evidence recorded.

## Verification log

| Work | Status | Evidence |
| --- | --- | --- |
| Pre-phase-1 HEOS suite | Completed baseline | 252 passed. [Recorded run](development/test_audit/results/baseline_20261112_195233/summary.json). Shared assertion-helper limitation is documented. |
| Array-aware diagnostic comparison | Completed experiment | 167 passed with 25,022 numerical comparisons. [Recorded run](development/test_audit/results/baseline_20261112_195233/array_oracle_summary.json). |
| Persistent test repairs | Completed phase 1 | 480 passed across HEOS and REFPROP, zero failures/skips, 50,189 finite values compared. [Summary](development/test_audit/results/phase1/full_20261112_203206/summary.json), [pytest log](development/test_audit/results/phase1/full_20261112_203206/pytest.log). |
| Uniform public contract | Completed phase 2 | 218 new cases: 99 passed, 119 strict expected failures. Full HEOS: 405 passed, zero unexpected failures/skips. [Summary](development/test_audit/results/phase2/full_heos_20261112_205840/summary.json). |
| Shared state and evaluator refactor | Pending | Phases 3 through 11; implement/remove the corresponding strict expected failures. |
| Consumer integration and publication checks | Pending | Phase 12. |
