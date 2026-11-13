# JaxProp uniform public fluid contract

This is the target interface for the fluid refactor. Applications select a physical model and an evaluator, then use the same construction, property-call, state, and error conventions. Evaluators are interchangeable over their declared domain when they represent the same physical model and reference convention. A perfect-gas approximation and a real-fluid equation of state can give different physical results.

Phase 2 fixes these requirements and adds executable conformance cases. The current implementation does not yet satisfy the complete contract. Strict expected failures in the tests identify the implementation phases that remain; the [implementation checklist](z_IMPLEMENTATION_PLAN.md) tracks their completion.

## Implemented shared layer after Phase 5

`create_fluid`, immutable `configuration`/`capabilities` records, the common ordinary `get_state` keywords, typed static-request validation, real-input broadcasting, honest property enabling, and named `StatusCode` values are implemented. Factories return actual evaluator classes; setup stays outside repeated calls. Configuration snapshots cannot be changed through the caller's option dictionaries or public identity/composition fields.

Capabilities report current implementation, rather than the target minimums below. Direct/callback source evaluators support twelve tested mass pairs, pure bicubic tables eight, perfect gas six, and the direct legacy binary model only scalar HP. Ordinary source/callback/pure-table states enable 25 properties, perfect gas 13, and legacy binary 22. Direct source generalized quality is retained where verified; disabled diagnostics and inapplicable properties are explicitly unsupported. The current bicubic domain record truthfully reports legacy extrapolation until Phase 8 enforces the target bounds policy.

The component-object binary factory route remains unavailable until Phase 9 implements that model; it raises an informative configuration error after validating settings. Perfect-gas calibration uses the selected CoolProp backend, including tested HEOS and REFPROP. Missing PG properties, the cp/cv/R correction, and enthalpy offsets remain Phase 7; reference initialization does not silently change those numerical equations.

`error_modes={"raise"}` describes the accepted current call mode. The target numerical/per-element mask policies and transformed error restrictions remain implementation requirements in Phases 6–9: `errors="mask"` is rejected explicitly instead of silently returning unvalidated statuses, while valid-state transformed calls remain usable. Legacy `exceptions=False` behavior is outside the factory path and is reconciled in those provider phases. Standalone uniform saturation operations are not advertised before validation; the bicubic pressure-only raw-dictionary helper remains a legacy path for Phase 8, and perfect gas explicitly rejects saturation.

Source references now use one dedicated CoolProp initialization path. `fluid.references` contains immutable, array-free records for `critical_point`, `triple_point_liquid`, and `triple_point_vapor`; explicit state access materializes read-only NumPy `FluidState` arrays. These are source-EOS states with canonical source names and `provider="coolprop"`, `model="real_fluid"`, independently of the active evaluator. Records expose source/version/composition metadata, actual reference anchors, point-specific availability, reported/evaluation coordinates, and offsets. Source-equivalent evaluators share a fingerprint derived from actual enthalpy/entropy anchor values, while perfect-gas and binary model identities also include their construction parameters/component identities. Tables carry the same source fingerprint alongside their complete provenance manifest. Reference records are reused only after a fresh source probe confirms their full immutable contents.

Air and pseudo-pure refrigerants retain distinct liquid/vapor lower saturation endpoints. A missing point leaves supported ordinary calculations usable; explicit access raises `UnavailableReferencePointError`. Critical singular response properties are deliberately unavailable, even if a backend returns a large finite near-singular number. A source with no evaluable deterministic anchor is rejected with a configuration error rather than assigned a fabricated shared identity. Existing quality-helper dependence on later global reference changes remains an explicit Phase 6 regression; actual stored references and ordinary core h/s values retain the construction convention.

The remaining sections specify the complete target. Their numerical/reference/model requirements are not weakened by this staged boundary.

The revised construction target below uses `Fluid(..., method=...)` as the common object initializer. This syntax is planned before Phase 6: the current package still exports a CoolProp-specific `Fluid` class and the generic `create_fluid` function. The validated routing, metadata, and reference implementation will be reused.

## Construction and identity

```python
fluid = jxp.Fluid(
    name="CO2",
    method="bicubic",
    backend="HEOS",
    identifier="working_fluid",
    table_options={
        "h_min": 400e3, "h_max": 800e3,
        "p_min": 8e6, "p_max": 30e6,
        "N_h": 200, "N_p": 200,
    },
)
state = fluid.get_state(jxp.HmassP_INPUTS, 500e3, 10e6)
```

The target public initializer is `Fluid(name, method="coolprop", backend=None, identifier=None, *, table_options=None, model_options=None)`. `name` is a nonempty source-fluid name, or a user label for a composite model. `method` is one exact lowercase selection below; it maps to separate physical-model and numerical-provider metadata internally. Consumers do not need to specify the implementation value `provider="native"` to request a perfect gas. Unknown selections, incompatible combinations, unrecognized settings, and settings for an inapplicable method raise `InvalidConfigurationError` before evaluation.

| `method` | Physical `model` metadata | Numerical `provider` metadata | Meaning |
| --- | --- | --- | --- |
| `coolprop` | `real_fluid` | `coolprop` | Direct CoolProp source-EOS evaluation. |
| `coolprop_jax` | `real_fluid` | `coolprop_jax` | The same source EOS through the JAX host-callback evaluator. |
| `bicubic` | `real_fluid` | `bicubic` | JaxProp interpolation tables generated from that source EOS. |
| `perfect_gas` | `perfect_gas` | `native` | Constant-heat-capacity perfect gas, calibrated from the named CoolProp source. |
| `binary_mixture` | `binary_mixture` | `native` | Future common-pressure/common-temperature binary model, evaluated using its component fluid objects in Phase 9. |

`backend` selects the CoolProp **source**, such as `HEOS` or explicitly requested `REFPROP`. For a single source, `None` resolves to `HEOS`. It does not select JaxProp's interpolation method. `method="bicubic"` is independent of CoolProp's own tabular backend names. The selected source also supplies bicubic tables and perfect-gas calibration/reference data. An unavailable requested source raises a configuration error with its original cause; it must not silently fall back to another source.

All fluid objects expose immutable configuration metadata: `name`, `provider`, `model`, `backend`, and `identifier`. `name` preserves the supplied fluid name, and returned states preserve it as `fluid_name`. An omitted identifier is `None`; an explicit string, including the empty string, is preserved exactly. Source canonicalization is recorded separately in source-reference metadata. Table creation/loading belongs to JaxProp and occurs during construction, outside repeated numerical calls.

For `perfect_gas`, `model_options` accepts `T_ref` (default 300 K) and `p_ref` (default 101325 Pa). For `binary_mixture`, it requires `components=(fluid_1, fluid_2)` and `mass_fractions=(w_1, w_2)`. The fractions are finite, nonnegative scalars with $w_1+w_2=1$ to absolute tolerance $10^{-12}$; endpoints are supported. No implicit mass-ratio convention or normalization is applied. Composition is fixed for the lifetime of an object in this first contract. Components retain their own providers, source backends, and reference conventions. The composite has `backend=None`; passing a backend override is an error. Its capabilities are limited by both component evaluators and by the mixture model. Existing `FluidMix` and `FluidTwoComponent` are consolidated into this single model in Phase 9.

For `bicubic`, `table_options` requires finite `h_min < h_max`, positive `p_min < p_max`, and integer `N_h, N_p >= 2`. `gradient_method` selects generation; `table_dir` and `table_name` select storage. `coarse_step` is a positive integer stride for inverse initial guesses, not a generation setting. Validate supported values during construction. Bounds are inclusive. Extrapolation and source-EOS fallback are disabled. Additional phase/table modes become public only after their validation in Phase 8.

Table reuse requires an exact complete generation manifest, including mode/composition, numerical settings, versions, source/reference fingerprints, and property schema. Identifiers, storage locations, and the inverse-guess stride do not alter table identity. Default filenames include a manifest digest so different settings coexist. Validate payload shapes, dtype, and finite core data before reuse, retaining legitimate optional NaNs. Corrupt or incompatible caches are regenerated; a failed generation/write preserves the prior file. Publication is atomic. If source provenance cannot be verified, generation may proceed but persistent cache reuse/publication is disabled with an explicit diagnostic. A matching cache manifest does not replace independent interpolation/derivative accuracy checks.

`Fluid(...)` is the recommended common object initializer. Selection, validation, source-reference initialization, and table creation/loading happen once during construction. Repeated `get_state` calls use the selected numerical implementation, preserving its native JAX PyTree, dynamic numerical inputs, compilation reuse, and supported transformations. Public concrete constructors retain the same applicable identity/configuration vocabulary. Returned-object type/introspection and compatibility-alias behavior are specified and tested during the constructor revision; they must not accidentally preserve the old meaning that every `Fluid` is a direct CoolProp object. Retaining legacy signatures is optional.

## Property calls and input units

```python
state = fluid.get_state(
    input_type=jxp.PT_INPUTS,
    prop_1=10e6,
    prop_2=500.0,
    generalize_quality=False,
    supersaturation=False,
    errors="raise",
)
```

The common signature is `get_state(input_type, prop_1, prop_2, *, generalize_quality=False, supersaturation=False, errors="raise")`. The first three arguments accept positional or the illustrated keyword form. `input_type` is a static CoolProp integer input-pair constant. Its constant defines argument order: `PT` is pressure then temperature, `HmassP` is enthalpy then pressure, and `PSmass` is pressure then entropy. Unknown constants and molar input pairs are explicitly unsupported. Strings are not alternate pair selectors.

All input and output values use SI and a mass basis: pressure Pa, temperature K, density kg/m$^3$, enthalpy/internal energy J/kg, entropy/heat capacities J/(kg K), dynamic viscosity Pa s, conductivity W/(m K), sound speed m/s, and dimensionless mass/volume fractions. Enthalpy and entropy may be negative; their validity depends on the declared reference convention. No automatic unit, mass/molar, or gauge/absolute conversion occurs.

The source EOS providers and their tables retain the same source enthalpy/entropy convention, recorded with source/backend/version metadata. Perfect-gas calibration uses the source gas constant divided by source molar mass for $R$, source $c_p$ at the calibration state, $c_v=c_p-R$, and $\gamma=c_p/c_v$. Require finite positive $R$ and $c_p>R$. It adopts the source enthalpy and entropy at that same state:

$$h=h_{\mathrm{ref}}+c_p(T-T_{\mathrm{ref}}),\qquad u=h-RT,$$

$$s=s_{\mathrm{ref}}+c_p\log(T/T_{\mathrm{ref}})-R\log(p/p_{\mathrm{ref}}).$$

Every perfect-gas forward/inverse equation uses these offsets. A composite uses the mass-weighted component reference conventions and records them. Changing the numerical provider must preserve the reference convention for a given model. Provider comparisons use source-compatible models; perfect-gas tests use its thermodynamic identities and calibration convention.

`generalize_quality=True` requests the documented extension of reported quality outside the ordinary two-phase interval. `supersaturation=True` requests saturation-departure diagnostics, including applicable subcooling/superheating outputs, after ordinary state evaluation. It does not select a metastable branch. False is accepted as a no-op by every evaluator. True must be declared supported and implemented; otherwise raise `UnsupportedOptionError`. Unrecognized options and invalid option types also receive informative typed errors. Diagnostic values and their regions of applicability are verified before advertising support.

## Shapes and state fields

Accept real Python scalars, lists, NumPy arrays, and JAX arrays. Reject booleans, complex values, strings, and object arrays. Normalize once and broadcast `prop_1` and `prop_2` using NumPy/JAX broadcasting. Every numerical state field and every status/validity array has **exactly** that broadcast shape:

| Input shapes | Output shape |
| --- | --- |
| `()`, `()` | `()` |
| `(1,)`, `()` | `(1,)` |
| `(2, 1)`, `(1, 3)` | `(2, 3)` |
| `(2, 1)`, `()` | `(2, 1)` |
| `(0, 1)`, `(1, 3)` | `(0, 3)` |

Do not squeeze singleton axes or keep constant-valued properties scalar inside a batch. An empty compatible batch returns correctly shaped empty arrays and performs no numerical evaluations. Incompatible shapes and invalid array types raise `InvalidStateError` before numerical work, even with `errors="mask"`.

Successful ordinary states have finite `pressure`, `temperature`, `density`, `enthalpy`, `entropy`, and `internal_energy`. Derivative/transport/phase properties are optional: they are not required to exist at singular points or for every model. The public numeric vocabulary and aliases are the canonical `FluidState` properties, with `alpha_p` restored for `isobaric_expansion_coefficient`. Each property has a single canonical name and collision-free aliases. `state.T`, `state["T"]`, and `state.temperature` agree, as do all other declared aliases.

State metadata includes `fluid_name`, `identifier`, `provider`, `model`, and an immutable `reference_id` identifying the physical/source model, composition, and energy/entropy convention. Model/reference compatibility is checked when combining states. Different providers for the same model and reference can be combined: `providers` retains the immutable set of originating providers and `provider` is `"mixed"` when that set has more than one member. Collections preserve the intersection of enabled properties and per-element validity; an unsupported property must not become available by filling missing entries with NaNs. `stack` adds one leading state axis; indexing, reversal, concatenation, and stacking preserve existing axes, compatible metadata, statuses, and property availability without mutating inputs. Empty stacking and incompatible metadata are rejected explicitly.

`reference_id` includes physical-model parameters and an actual source energy/entropy reference fingerprint, including source values at a fixed valid anchor state. Source name/backend/version alone cannot detect a changed CoolProp global reference convention. Table cache identities must include the same reference fingerprint. Initialization must not silently change the process-wide CoolProp reference convention. Perfect-gas parameters and component reference identities are included in their respective model fingerprints.

`at_index` accepts a static integer, checks positive/negative bounds, and raises `IndexError` outside the first state axis. A scalar state accepts indices `0` and `-1`. Traced indices raise `UnsupportedExecutionModeError`; this prevents JAX's implicit out-of-range clamping. Stacking requires identical state shapes. Concatenation promotes scalar states to a length-one axis and requires matching trailing axes. States with different error policies are rejected when combined, so collection operations cannot silently change access semantics.

### Property availability

`fluid.capabilities.properties` declares evaluator-supported canonical properties, including option-dependent diagnostics. `state.available_properties` is a static set of properties enabled for that evaluation. `state.supports_property(name)` accepts canonical names and aliases and returns a static boolean. Unsupported known properties, including disabled diagnostics, raise `UnsupportedPropertyError` through attribute, canonical-name, and dictionary access alike. An unknown attribute retains `AttributeError`; an unknown dictionary key retains `KeyError`.

A supported property may be undefined at an individual valid state, such as a singular critical-point derivative or an inapplicable saturation departure. `state.property_valid(name)` returns a boolean array of the state shape and is false at every numerically failed element. Its raw numerical storage is NaN where unavailable. On an eager state evaluated with `errors="raise"`, requesting an undefined supported property raises `PropertyUnavailableError` with failing indices. On a state evaluated with `errors="mask"`, ordinary access returns raw arrays, including NaNs, in eager and transformed execution alike. `state.raw_property(name)` always returns raw storage for an enabled property; use it with `state.property_valid(name)` when availability must be array-valued. No transformed access may convert traced booleans to Python. Raw access still rejects statically unsupported properties. `to_dict()` exports enabled numeric properties without triggering dynamic availability errors; unsupported slots are excluded, and optional aliases are explicitly requested. Numeric status arrays are separate state attributes.

The state constructor accepts `error_policy="raise"` or `"mask"`, `status`, and optional `property_validity` masks. `to_data()` / `FluidState.from_data(...)` preserve identity, enabled properties, raw values, statuses, masks, and access policy in a complete round trip. `to_dict()` remains a numerical-property export; reconstructing from it alone does not recover metadata or availability semantics. Pickle round trips preserve the same complete state behavior. Ordinary source states may keep read-only NumPy storage; JAX inputs/tracers remain JAX arrays. Input mutation must not change an existing state.

For a perfect gas, `quality_mass`, `quality_volume`, surface tension, and saturation properties are unsupported. In particular, `state.quality_mass`, `state.Q`, and `state["Q"]` must raise the same informative `UnsupportedPropertyError`. `is_two_phase` is supported and false; it is not an undefined quality value.

## Capabilities and the fixed conformance matrix

Every object exposes immutable `capabilities` with `input_pairs`, `properties`, `options`, `phase_operations`, `jax_modes`, and `domain`. Input sets use integer constants; property/option sets use canonical names. `domain` describes applicable phases, validity bounds, and the absence of extrapolation. JAX modes identify supported transformations, derivative order/method, and pair/property restrictions. A capability may be advertised only after its applicable numerical and failure tests pass. Configuration can restrict capabilities but cannot silently expand them.

The test matrix is written independently of implementation dispatch and runtime capabilities. It establishes these minimum requirements for pure HEOS source test fluids in applicable single-phase domains:

| Evaluator/model | Required pairs | Phase requests | JAX target |
| --- | --- | --- | --- |
| CoolProp real fluid | Eight pairs below, plus `PQ` and `QT` | Applicable saturation and source diagnostics | Eager only. |
| CoolProp JAX real fluid | Same source-compatible pairs | Applicable saturation and source diagnostics | Eager, outer JIT, vmap, first-order JVP/gradient by callback finite differences. |
| Bicubic real fluid, ordinary single-phase table | Eight pairs below | Quality pairs and saturation evaluation unsupported for this table mode. | Eager, outer JIT, vmap, first derivatives within valid smooth cells. |
| Native perfect gas | Six pairs below | Quality pairs, saturation evaluation, and phase diagnostics unsupported. | Eager, outer JIT, vmap, first derivatives. |
| Native binary mixture | `PT`, `HmassP`, `PSmass` | Composite saturation/quality unsupported; component references remain accessible. | Eager required; transformations only when both components and the validated inversion path support them. |

The six perfect-gas pairs are `PT_INPUTS`, `HmassSmass_INPUTS`, `HmassP_INPUTS`, `PSmass_INPUTS`, `DmassHmass_INPUTS`, and `DmassP_INPUTS`. The eight real-fluid pairs add `DmassT_INPUTS` and `DmassSmass_INPUTS`. These minimums do not claim that every CoolProp backend, mixture, or source fluid accepts every pair or phase operation. Source-specific restrictions must be declared and rejected explicitly. Additional pairs or transformations require their own tests.

The uniform phase entry point is `get_state_saturation(input_type, prop_1, prop_2, *, errors="raise")`, with `PQ_INPUTS` or `QT_INPUTS` in the same argument order and mass-quality convention as `get_state`. It returns `FluidState`. Unsupported models/table modes raise `UnsupportedOperationError` before evaluation. Perfect gas cannot acquire this capability merely by possessing source-fluid reference points.

The legacy `FluidMix` remains a conformance row while Phase 9 consolidates it; its current scalar-HP limitation is recorded as an implementation gap against the binary target. The unexported, currently unimportable `FluidTwoComponent` is a consolidation input, not a separately promised public evaluator. Conformance cases must not skip missing target functionality by querying current capabilities.

## Reference data

Construction initializes applicable source critical, liquid triple, and vapor triple reference records through a dedicated CoolProp path for every evaluator. Their availability is independent of the active evaluator's phase capabilities. `fluid.references` exposes point-specific records; the convenience accessors `critical_point`, `triple_point_liquid`, and `triple_point_vapor` return their state when available and raise `UnavailableReferencePointError` otherwise.

Each record carries `available`, `reason`, `classification`, `interpretation`, source fluid/composition, backend/CoolProp version, reference convention, reported coordinates, numerical evaluation coordinates, and evaluation offsets. Records are immutable and shared only for identical source configurations. Each point is evaluated independently. Undefined optional properties do not make otherwise well-defined reference coordinates disappear.

Pure-fluid triples are identified as triple references. Pseudo-pure models such as Air retain available critical data and distinct liquid/vapor lower-saturation endpoints, labeled `pseudo_pure` with interpretation `lower_saturation_endpoint`; their pressures must not be forced equal. Unknown or unavailable points record the source reason without preventing ordinary supported property calls. A composite binary model has no assumed unique critical/triple point: its aggregate records are explicitly unavailable, and source reference data remain accessible through its components. This does not preclude a separately validated mixture-EOS model in future work.

## Errors and numerical failures

All public typed errors inherit `JaxPropError` and are exported from `jaxprop`. They retain structured `fluid_name`, `provider`, `model`, `request`, `reason`, and `supported` attributes, plus failing indices where applicable. Their readable message identifies the fluid/evaluator, rejected request, reason, and relevant alternatives. Preserve the original exception as the cause for underlying provider/solver failures.

`request` is a string: the canonical input-pair name (for example `"PT_INPUTS"`) for pair/state failures, canonical property name for property access, option name for an invalid option, `"saturation"` for that operation, and `"errors=raise"` for rejection of the eager failure policy under a transformation. Unknown pairs use their supplied value's representation. `supported` contains canonical supported alternatives, or an empty tuple when none apply. Numerical errors expose the first failing `index` as a tuple; property-availability errors can report multiple `indices`.

| Type | Meaning |
| --- | --- |
| `InvalidConfigurationError` | Unknown/incompatible construction settings or unavailable requested source. |
| `UnsupportedInputPairError` | A known/unknown pair is unsupported by this evaluator/model. |
| `UnsupportedPropertyError` | A known property or alias is unsupported or not enabled. |
| `UnsupportedOptionError` | A requested diagnostic/keyword is unsupported or invalid. |
| `UnsupportedOperationError` | A phase or other public operation is unsupported. |
| `UnsupportedExecutionModeError` | The requested JAX/error mode is unsupported. |
| `UnavailableReferencePointError` | A specifically requested source reference is unavailable. |
| `PropertyUnavailableError` | An enabled property is undefined at one or more otherwise valid states. |
| `InvalidStateError` | Invalid numerical inputs, types, or broadcast shape. |
| `OutOfDomainError` | A state lies outside the declared evaluator/table domain. |
| `CalculationError` | Evaluation failed after valid inputs; retain the underlying cause. |
| `ConvergenceError` | A numerical inverse did not meet its declared convergence/residual criteria. |

`UnsupportedInputPairError`, `UnsupportedPropertyError`, `UnsupportedOptionError`, `UnsupportedOperationError`, and `UnsupportedExecutionModeError` share `UnsupportedRequestError`. Configuration, unsupported-request, property-unavailable, and invalid-state errors also derive from `ValueError`; reference errors from `LookupError`; calculation errors from `RuntimeError`. `OutOfDomainError` derives from `InvalidStateError`, and `ConvergenceError` from `CalculationError`.

With `errors="raise"` (the eager default), any failing numerical element raises the applicable typed error, with the first failing index in row-major order and no partial state returned. Never silently clamp invalid values, extrapolate tables, return `None`, or replace an entire partially valid batch with NaNs.

With `errors="mask"`, numerical failures are recorded independently per element. `state.status` is an integer array: `0` success, `1` invalid input, `2` outside domain, `3` nonconvergence, `4` provider failure. `state.valid` is exactly `state.status == 0`. All enabled numerical properties at a failed element are NaN; successful elements retain their values. A locally unavailable optional property does not change a successful state status. Static configuration/unsupported requests and malformed arrays still raise immediately in either policy. Status values are public constants rather than caller-specific magic numbers.

## JAX execution

Construction, capability validation, source-reference initialization, and table I/O happen outside transformed code. Input pairs, requested options, identity, and schema are static. Eager use of JAX-capable evaluators follows the same `errors="raise"` default as ordinary CoolProp.

Outer JIT, vmap, JVP, or gradient execution requires explicitly selecting `errors="mask"`; requesting `raise` under a transformation raises `UnsupportedExecutionModeError` during tracing. Transformed numerical failures return the same status/mask arrays as eager mask mode, without relying on Python exceptions from inside host callbacks. A statically unsupported pair/property/option raises its typed error before numerical execution even under JIT.

Derivative claims apply to successful differentiable states and enabled properties inside the declared domain, away from phase boundaries, interpolation-cell discontinuities in higher derivatives, and failure boundaries. Masks/statuses are nondifferentiable. The callback evaluator declares first-order finite-difference derivatives; higher derivatives are unsupported until independently validated. Bicubic/native providers may advertise further modes only after verification. JAX capability tests include returned-state identity and status, not just a scalar extracted property. Applications control platform and precision settings; the package's current import side effects are removed in Phase 11.

Differentiate a floating property projection, or the enabled numerical property dictionary, rather than the complete state with its integer status and boolean validity leaves. JIT/vectorization can return the full state. Gradient helpers must carry primal identity/validity separately from numerical derivatives; a dedicated derivative-result schema is tracked in Phase 7.

## Verification and implementation ownership

The shared tests exercise existing public classes directly while the factory is implemented, so a missing factory cannot hide all remaining differences. Passing numerical behavior remains ordinary passing tests. Known contract gaps have narrow `xfail(strict=True)` markers with the responsible phase and reason; a fixed gap produces a failing unexpected pass until its marker is removed. Unexpected failures remain failures.

Phase 3 owns the shared typed-error foundation, state aliases, shapes, identity/reference compatibility, and controlled access. Phase 4 owns construction, capability publication, common input validation, and integration of typed static-request errors. Phase 5 owns source references. Phases 6–9 apply the same contract to source/callback, perfect-gas, bicubic, and consolidated binary evaluators, including numerical failures and JAX behavior. Later phases validate diagnostics, lifecycle, and consuming applications. Completing Phase 2 establishes these requirements; it does not claim those later implementations already pass.
