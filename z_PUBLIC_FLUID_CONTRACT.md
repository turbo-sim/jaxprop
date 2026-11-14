# JaxProp uniform public fluid contract

This is the target interface for the fluid refactor. Applications select a physical model and an evaluator, then use the same construction, property-call, state, and error conventions. Evaluators are interchangeable over their declared domain when they represent the same physical model and reference convention. A perfect-gas approximation and a real-fluid equation of state can give different physical results.

Phase 2 fixes these requirements and adds executable conformance cases. The current implementation does not yet satisfy the complete contract. Strict expected failures in the tests identify the implementation phases that remain; the [implementation checklist](z_IMPLEMENTATION_PLAN.md) tracks their completion.

## Implemented shared layer

`Fluid(..., method=...)`, immutable `configuration`/`capabilities` records, the common ordinary `get_state` keywords, typed static-request validation, real-input broadcasting, honest property enabling, and named `StatusCode` values are implemented. Construction returns actual native evaluator objects; setup stays outside repeated calls. Configuration snapshots cannot be changed through the caller's option dictionaries or public identity/composition fields. The lower-level `create_fluid(provider=..., model=...)` function remains available for compatibility.

Capabilities report current implementation, rather than the target minimums below. Direct/callback source evaluators support twelve tested mass pairs, pure bicubic tables eight, perfect gas six, and the consolidated binary model `PT`, `HmassP` and `PSmass`. Ordinary source/callback/pure-table states enable 25 properties, perfect gas 22, and binary states 27. Source/callback support generalized quality and six additional diagnostics when requested; disabled diagnostics and inapplicable properties are explicitly unsupported. Their capability property set describes the supported union rather than every default state. Bicubic tables declare bounded enthalpy/pressure domains with extrapolation disabled; untrusted interpolation cells also produce declared numerical failures.

The component-object binary factory route returns `FluidBinaryMixture`. Its full-suite and timing evidence is recorded in the implementation plan. Perfect-gas calibration uses the selected CoolProp backend, including tested HEOS and REFPROP. Phase 7 completes core/response properties and consistent heat capacities and reference offsets; its ideal-gas equations intentionally replace the earlier inconsistent calibration equations.

Source/callback capabilities declare `error_modes={"raise", "mask"}`. Both implement per-element numerical failures, and transformed callbacks require explicit mask mode. Perfect gas and bicubic declare `raise`, `mask` and `clip`; their transformed calls require explicit mask or clip mode. Binary execution/policies depend on the active component evaluators, as described below. Concrete source/callback constructors and the legacy binary constructor reject `exceptions=False` with per-call mask guidance. Both source evaluators advertise the uniform saturation operation. Direct source objects also declare eager scalar equilibrium, metastable and blending operations; the callback's phase-operation set contains saturation only. Bicubic tables advertise typed saturation only where a stored liquid or vapor endpoint payload exists; arbitrary quality and QT evaluation remain unsupported. Perfect gas and the fixed-composition binary model explicitly reject saturation.

Source references now use one dedicated CoolProp initialization path. `fluid.references` contains immutable, array-free records for `critical_point`, `triple_point_liquid`, and `triple_point_vapor`; explicit state access materializes read-only NumPy `FluidState` arrays. These are source-EOS states with canonical source names and `provider="coolprop"`, `model="real_fluid"`, independently of the active evaluator. Records expose source/version/composition metadata, actual reference anchors, point-specific availability, reported/evaluation coordinates, and offsets. Source-equivalent evaluators share a fingerprint derived from actual enthalpy/entropy anchor values, while perfect-gas and binary model identities also include their construction parameters/component identities. Tables carry the same source fingerprint alongside their complete provenance manifest. Reference records are reused only after a fresh source probe confirms their full immutable contents.

Air and pseudo-pure refrigerants retain distinct liquid/vapor lower saturation endpoints. A missing point leaves supported ordinary calculations usable; explicit access raises `UnavailableReferencePointError`. Critical singular response properties are deliberately unavailable, even if a backend returns a large finite near-singular number. A source with no evaluable deterministic anchor is rejected with a configuration error rather than assigned a fabricated shared identity. Configured ordinary and legacy phase operations reuse constructor-prepared reference-compatible helpers. Phase 10 implements the equilibrium response, diagnostic and generalized-quality definitions below; final phase acceptance is recorded separately in the implementation checklist.

The remaining sections specify the complete target. Their numerical/reference/model requirements are not weakened by this staged boundary.

`Fluid(..., method=...)` is now the common object initializer in the development checkout. The direct source implementation is explicitly named `FluidCoolProp`; `jaxprop.coolprop.Fluid` remains its source-only compatibility alias. This constructor revision reuses the validated routing, metadata and reference implementation without changing evaluator equations or numerical failure policies.

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

The public initializer is `Fluid(name, method="coolprop", backend=None, identifier=None, *, table_options=None, model_options=None)`. `name` is a nonempty source-fluid name, or a user label for a composite model. `method` is one exact lowercase selection below; it maps to separate physical-model and numerical-provider metadata internally. Consumers do not need to specify the implementation value `provider="native"` to request a perfect gas. Unknown selections, incompatible combinations, unrecognized settings, and settings for an inapplicable method raise `InvalidConfigurationError` before evaluation. The implementation accepts unknown keywords only to issue these typed errors; they are not additional supported options.

| `method` | Physical `model` metadata | Numerical `provider` metadata | Meaning |
| --- | --- | --- | --- |
| `coolprop` | `real_fluid` | `coolprop` | Direct CoolProp source-EOS evaluation. |
| `coolprop_jax` | `real_fluid` | `coolprop_jax` | The same source EOS through the JAX host-callback evaluator. |
| `bicubic` | `real_fluid` | `bicubic` | JaxProp interpolation tables generated from that source EOS. |
| `perfect_gas` | `perfect_gas` | `native` | Constant-heat-capacity perfect gas, calibrated from the named CoolProp source. |
| `binary_mixture` | `binary_mixture` | `native` | Fixed-composition common-pressure/common-temperature parcels, evaluated using their component fluid objects. |

`backend` selects the CoolProp **source**, such as `HEOS` or explicitly requested `REFPROP`. For a single source, `None` resolves to `HEOS`. It does not select JaxProp's interpolation method. `method="bicubic"` is independent of CoolProp's own tabular backend names. The selected source also supplies bicubic tables and perfect-gas calibration/reference data. An unavailable requested source raises a configuration error with its original cause; it must not silently fall back to another source.

All fluid objects expose immutable configuration metadata: `name`, `provider`, `model`, `backend`, and `identifier`. `name` preserves the supplied fluid name, and returned states preserve it as `fluid_name`. An omitted identifier is `None`; an explicit string, including the empty string, is preserved exactly. Source canonicalization is recorded separately in source-reference metadata. Table creation/loading belongs to JaxProp and occurs during construction, outside repeated numerical calls.

For `perfect_gas`, `model_options` accepts `T_ref` (default 300 K) and `p_ref` (default 101325 Pa). For `binary_mixture`, it requires `components=(fluid_1, fluid_2)` and `mass_fractions=(w_1, w_2)`, with optional `inverse_max_steps=64` and `inverse_tolerance=1e-10`. The fractions are finite, nonnegative scalars with $w_1+w_2=1$ to absolute tolerance $10^{-12}$; endpoints are supported. No implicit mass-ratio convention or normalization is applied. Composition is fixed for the lifetime of an object, never an extra positional state input. Components retain their own providers, source backends, and reference conventions. The composite has `backend=None`; passing a backend override is an error. The iteration budget and scaled residual tolerance govern numerical HP/PS inverses where needed; analytic perfect-gas mixtures avoid an iterative solve.

For `bicubic`, `table_options` requires finite `h_min < h_max`, positive `p_min < p_max`, and integer `N_h, N_p >= 2`. `gradient_method` selects generation; `table_dir` and `table_name` select storage. `coarse_step` is a positive integer stride for inverse initial guesses, not a generation setting. Validate supported values during construction. Bounds are inclusive. Extrapolation and source-EOS fallback are disabled. Each declared table mode retains its validated phase and operation restrictions; metastable central first differences remain unsupported.

The common bicubic constructor accepts `metastable_phase='liquid'/'vapor'` and `metastable_domain='physical'/'formal'`. Physical mode is the default and requires positive sampled $c_v$ and $K_T$, including derivative stencils. Formal mode explicitly retains calorically unstable homogeneous EOS states for research; it still requires mechanical stability. Construction-time Bernstein bounds reject cells whose interpolated stability is unverified and mask optional responses in cells that may cross their heat-capacity denominators. These guards certify the interpolation polynomials rather than the exact source between samples. The domain and captured policy are part of the generation manifest; native state calls reuse existing masks without new source queries or stability arithmetic. The raw homogeneous source solver retains its separate scientific unstable-state capability.

Table reuse requires an exact complete generation manifest, including mode/composition, numerical settings, versions, source/reference fingerprints, and property schema. Identifiers, storage locations, and the inverse-guess stride do not alter table identity. Default filenames include a manifest digest so different settings coexist. Validate payload shapes, dtype, and finite core data before reuse, retaining legitimate optional NaNs. Corrupt or incompatible caches are regenerated; a failed generation/write preserves the prior file. Publication is atomic. If source provenance cannot be verified, generation may proceed but persistent cache reuse/publication is disabled with an explicit diagnostic. A matching cache manifest does not replace independent interpolation/derivative accuracy checks.

`Fluid(...)` is the recommended common object initializer. Selection, validation, source-reference initialization, and table creation/loading happen once during construction. Repeated `get_state` calls use the selected numerical implementation, preserving its native JAX PyTree, dynamic numerical inputs, compilation reuse, and supported transformations. Public concrete constructors retain the same applicable identity/configuration vocabulary.

`Fluid` is a real class whose native implementations are registered virtual subclasses. Consequently `isinstance(fluid, Fluid)` and `issubclass(FluidBicubic, Fluid)` are true, while `type(fluid)` identifies the actual evaluator (`FluidCoolProp`, `FluidJAX`, `FluidBicubic`, `FluidPerfectGas`, or `FluidBinaryMixture`). There is no instance wrapper and the common class is not inserted into the native inheritance/PyTree structure. The selected concrete constructor runs exactly once. Physical subclassing of the dispatcher is rejected explicitly; extend a concrete evaluator instead. `create_fluid` remains the lower-level provider/model compatibility function, and the source-only namespace retains its concrete `Fluid` alias.

Legacy `exceptions` construction flags belong to explicit concrete source classes and the binary compatibility constructor; the common initializer rejects them with guidance to the per-call `errors` policy. Source/callback implement raise/mask, while perfect gas and bicubic implement raise/mask/clip. Binary policies follow the active component execution path below. Configured evaluator pickle snapshots preserve native models and nested binary components. Source/callback snapshots reconstruct all native handles and locks only under a verified matching reference, backend configuration and native-definition/file identity; they never change process-global conventions. An unverifiable source explicitly rejects lossless serialization while remaining usable for evaluation. Transient plotting artists are discarded, numerical curves retained. Configuration records are inspection metadata, not general lossless factory arguments. See the [lifecycle guide](docs/fluid_lifecycle.md).

### Perfect-gas calibration and model updates

Perfect gas preserves source $c_p$, $h_\mathrm{ref}$ and $s_\mathrm{ref}$ at the calibration coordinates. With $R=R_u/M$, it defines $c_v=c_p-R>0$ and $\gamma=c_p/c_v$. Its equations are

$$h=h_\mathrm{ref}+c_p(T-T_\mathrm{ref}),\qquad u=h-RT,$$

$$s=s_\mathrm{ref}+c_p\log(T/T_\mathrm{ref})-R\log(p/p_\mathrm{ref}),\qquad p=\rho RT.$$

All six inverses use these same offsets. Real-EOS density, $c_v$ and response properties need not match the ideal approximation at calibration. The 22 enabled fields include internal energy, a zero two-phase marker, expansion/compressibility, bulk moduli and zero Joule–Thomson coefficients. Quality, surface tension and saturation remain explicitly unsupported. Optional viscosity/conductivity use Sutherland fits from in-domain source temperatures; unavailable fits or locally undefined transport do not invalidate thermodynamic core values.

`model.with_constants(**changes)` returns an eagerly validated materialized model with refreshed immutable identity/configuration. `model.materialize()` does the same for a differentiation view made using `eqx.tree_at`. Neither operation initializes a source. Low-level differentiable views retain the original static identity and are ephemeral until materialized; numerical evaluation does not inspect or regenerate fingerprints. The configuration records actual constants for inspection. Validated materialization and complete pickle snapshots are the supported reconstruction recipes; a declarative factory/JSON export of every model is separate optional work.

Model coefficients now carry explicit numerical dtypes, so strong float32 inputs cannot silently downcast a default double-precision model. Explicit float32 coefficient models are supported and tested. Construction/materialization validates actual coefficient arithmetic, energy range and entropy bounds in those dtypes. Valid dynamic coefficient views remain differentiable; state evaluation does not repeat the coefficient checks.

### Fixed-composition binary model

```python
nitrogen = jxp.Fluid("Nitrogen", method="perfect_gas")
oxygen = jxp.Fluid("Oxygen", method="perfect_gas")
mixture = jxp.Fluid(
    "working_mix",
    method="binary_mixture",
    identifier="mixed_stream",
    model_options={
        "components": (nitrogen, oxygen),
        "mass_fractions": (0.4, 0.6),
    },
)
state = mixture.get_state(jxp.PT_INPUTS, 1e6, 500.0)
```

This model represents fixed masses of two constituent parcels sharing $p$ and $T$, with no mass exchange or molecular mixing entropy. It is not a replacement for a chemical mixture EOS. Specific volume, $h$, $u$ and $s$ are mass additive. With volume fractions $\phi_i=w_i\rho/\rho_i$, the consistent response functions are

$$\frac{1}{\rho}=\sum_i\frac{w_i}{\rho_i},\qquad
c_p=\sum_i w_i c_{p,i},\qquad
\alpha_p=\sum_i\phi_i\alpha_{p,i},\qquad
\kappa_T=\sum_i\phi_i\kappa_{T,i},$$

$$c_v=c_p-\frac{T\alpha_p^2}{\rho\kappa_T},\qquad
\kappa_s=\kappa_T\frac{c_v}{c_p},\qquad
a_{pT}^2=\frac{1}{\rho\kappa_s}.$$

The former mass averages of $c_v$ and isentropic compressibility were inconsistent with the common-$p,T$ model and its sound speed; they are intentionally corrected. The canonical `speed_of_sound` and explicit `speed_of_sound_pT` use pressure/temperature relaxation. `speed_of_sound_p` retains the distinct thermally frozen Wood response, $a_p^{-2}=\rho\sum_i\phi_i\kappa_{s,i}$. These are different physical acoustic limits, not interchangeable aliases for arbitrary calculations. Transport retains the existing empirical volume averages.

Complete binary states enable 27 fields: six core variables; `cp`, `cv`, `gamma`, expansion; isothermal/isentropic compressibilities and bulk moduli; the three sound-speed fields; Gruneisen and both Joule–Thomson coefficients; viscosity/conductivity; and `mass_frac_1`, `mass_frac_2`, `vol_frac_1`, `vol_frac_2`, `mixture_ratio`. `mixture_ratio` is an optional numerical ratio, with NaN when $w_1/w_2$ is unrepresentable, including $w_2=0$. The finite mass fractions remain authoritative. No density ordering is used to label a constituent as vapor: composite quality, surface tension, compressibility factor and `is_two_phase` are unsupported, and saturation calls raise `UnsupportedOperationError`.

Execution is selected once from **active** (nonzero-weight) components:

| Active components | Execution and error policies |
| --- | --- |
| Perfect gas and/or bicubic | Native JAX, dynamic component numerical data, eager raise/mask/clip and transformed mask/clip; first-order state/model derivatives in their valid domains. |
| All `FluidJAX` | One callback for the complete host mixture transaction, eager raise/mask and transformed mask; first-order finite-difference state-input derivatives only. |
| Any direct source, or a source/callback/native hybrid | Eager host evaluation with raise/mask; compiled hybrid execution is deferred to Phase 11 and rejected explicitly. |

The `provider="native"` metadata names the composition model; it does not override the execution restrictions of its components. Zero-weight components are not evaluated and do not restrict the active numerical domain. Native clipping projects only onto a defined common component domain; inverse target clipping requires a connected monotone response proof. Nonfinite requests, invalid phase cells and unverified roots remain failed elements.

Native component parameters remain dynamic JAX inputs. Host/callback configurations are fixed snapshots of the configured component models/handles, stored outside dynamic numerical PyTrees; they do not promise model-parameter AD or live edits through differentiation views. Construct a new mixture from changed components to change that model. Source inverse iterations evaluate selected h/s values on prepared source states and assemble full component properties only after root validation, under one consistently ordered transaction of source locks. A phase discontinuity with no matching common-$p,T$ root reports nonconvergence rather than accepting a small temperature step as success.

`FluidMix(name, backend=..., mixture_ratio=..., identifier=...)` is a construction-only compatibility function that translates the legacy name/mass ratio and returns an actual `FluidBinaryMixture`. `FluidTwoComponent` is an alias of that same class, using the common fixed-composition constructor. The obsolete per-call fourth positional ratio and missing `MixtureState` schema are not retained. Configure and reuse a mixture object in simulations; the legacy `get_mixture_state` convenience helper constructs one per call.

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

The source EOS providers and their tables retain the same source enthalpy/entropy convention, recorded with source/backend/version metadata. Perfect-gas calibration uses the universal molar gas constant $R_u$ divided by source molar mass for $R$, source $c_p$ at the calibration state, $c_v=c_p-R$, and $\gamma=c_p/c_v$. Source-EOS calculations use the selected backend's own gas constant. Require finite positive $R$ and $c_p>R$ for the perfect-gas model. It adopts the source enthalpy and entropy at that same state:

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
| CoolProp real fluid | Eight pairs below, plus `PQ` and `QT` | Saturation/diagnostics; separate scalar equilibrium, metastable and blending operations | Eager only. |
| CoolProp JAX real fluid | Same source-compatible pairs | Applicable saturation and source diagnostics | Eager, outer JIT, vmap, first-order JVP/gradient by callback finite differences. |
| Bicubic real fluid, ordinary single-phase table | Eight pairs below | Quality pairs and saturation evaluation unsupported for this table mode. | Eager, outer JIT, vmap, first derivatives within valid smooth cells. |
| Native perfect gas | Six pairs below | Quality pairs, saturation evaluation, and phase diagnostics unsupported. | Eager, outer JIT, vmap, first derivatives. |
| Native binary mixture | `PT`, `HmassP`, `PSmass` | Composite saturation/quality unsupported; component references remain accessible. | Native PG/bicubic paths support JAX mask/clip; all-active `FluidJAX` paths use first-order mask callbacks; direct-source/hybrid paths are eager. |

The six perfect-gas pairs are `PT_INPUTS`, `HmassSmass_INPUTS`, `HmassP_INPUTS`, `PSmass_INPUTS`, `DmassHmass_INPUTS`, and `DmassP_INPUTS`. The eight real-fluid pairs add `DmassT_INPUTS` and `DmassSmass_INPUTS`. These minimums do not claim that every CoolProp backend, mixture, or source fluid accepts every pair or phase operation. Source-specific restrictions must be declared and rejected explicitly. Additional pairs or transformations require their own tests.

The uniform phase entry point is `get_state_saturation(input_type, prop_1, prop_2, *, errors="raise")`, with `PQ_INPUTS` or `QT_INPUTS` in the same argument order and mass-quality convention as `get_state`. It returns `FluidState`. Unsupported models/table modes raise `UnsupportedOperationError` before evaluation. Perfect gas cannot acquire this capability merely by possessing source-fluid reference points.

Ordinary CoolProp and its JAX callback implement this entry point. Their immutable capabilities describe the same ordinary source domain and supported property union, while the callback does not advertise the direct source's additional phase solvers. Default ordinary calls enable 25 fields; `supersaturation=True` additionally enables six saturation-departure fields. Undefined optional values remain NaN with successful core status; eager raising access reports `PropertyUnavailableError`. Generalized quality is independent of saturation diagnostics and does not select a metastable branch. HEOS pseudo-pure Air/R410A permit fractional `PQ` flashes but only endpoint `QT` flashes; REFPROP supports both fractional routes. The definitions below account for their temperature glide and actual flashed phase densities.

The binary conformance row now exercises `FluidBinaryMixture` directly, with the legacy constructors covered separately as setup-only compatibility paths. `FluidTwoComponent` imports successfully as an alias of the same class. Conformance cases must not skip missing target functionality by querying current capabilities; provider-specific transformation restrictions remain explicit tests.

### Source equilibrium phase responses

Inside the two-phase region, mass quality is the source flash's actual $q=Q$. Its volume quality is

$$\beta_v=\frac{q\rho}{\rho_v^{\mathrm{flash}}},\qquad
1-\beta_v=\frac{(1-q)\rho}{\rho_l^{\mathrm{flash}}}.$$

The phase densities belong to that flash. Reconstructing endpoints at the bulk temperature is incorrect for gliding pseudo-pure models. Even pressure-endpoint densities can differ from the actual quality-dependent phase densities returned by REFPROP. Quality recovery must not change a valid source core state to force a different lever rule.

Interior equilibrium response functions differentiate the source's actual map $(p,q)\mapsto(T,\rho,h,s,u)$. For $J(f,g)=f_p g_q-f_q g_p$, the equilibrium heat capacity and representative response definitions are

$$c_v=\frac{J(u,\rho)}{J(T,\rho)},\qquad
a^2=\left(\frac{\partial p}{\partial\rho}\right)_s,\qquad
\Gamma=\frac{1}{\rho}\left(\frac{\partial p}{\partial u}\right)_\rho,\qquad
\mu_{JT}=\left(\frac{\partial T}{\partial p}\right)_h.$$

Phase-change mass transfer is included: equilibrium $c_v$ is not the mass average of the phase heat capacities. For pure fluids, saturation pressure fixes temperature in the dome. Consequently `isobaric_heat_capacity`, `heat_capacity_ratio`, `isothermal_compressibility`, `isothermal_bulk_modulus`, `isobaric_expansion_coefficient` and `isothermal_joule_thomson` are unavailable there, rather than fabricated finite one-phase or phase-average responses. Finite $c_v$, Gruneisen, Joule–Thomson and isentropic responses remain enabled where their source derivatives are evaluable. The sound model is homogeneous equilibrium; it is not a claim that every two-phase acoustic regime has that speed. Exact $q=0$ and $q=1$ recover the corresponding homogeneous endpoint coefficients, while arbitrarily nearby interior states retain equilibrium responses without an artificial quality cutoff. Endpoint states remain labeled `is_two_phase=1`.

Pseudo-pure temperature glide can make the corresponding constant-pressure/isothermal derivatives finite. These fields describe the actual source PQ interpolation and are checked against independently constrained flashes. Empirical source maps need not satisfy every Maxwell identity or agree across every pair algorithm inside a gliding dome: HEOS pseudo-pure `DmassT` can use different geometry from `PQ`, and REFPROP can reject that density/temperature two-phase route. Supported single-phase pair tests do not establish interchangeability of those algorithms throughout that region. This follows the source's limitations rather than relabeling its core values; [CoolProp's derivative guidance](https://coolprop.org/coolprop/HighLevelAPI.html#partial-derivatives) likewise distinguishes homogeneous derivatives from two-phase calculations.

Viscosity and conductivity retain a declared empirical volume average of the pressure's bubble/dew endpoint transport correlations. These are not exact mixture laws or a claim of quality-dependent phase transport from a backend that does not expose it. Missing optional transport or response data leaves the six valid core quantities usable.

### Quality extensions and requested diagnostics

Within the physical dome, generalized and trimmed quality preserve the source's exact mass quality. Outside it, trimmed quality is a conventional 0/1 label obtained by comparing the state's entropy with the configured critical entropy; it is not a physical vapor fraction. Subcritical generalized quality extrapolates saturated entropies at the state's temperature. Above the critical temperature it uses the entropy on the critical-density isochore, $s_c^\rho(T)=s(\rho_c,T)$, and the physical source gas constant $R_s$:

$$w=R_s\max\!\left(\frac{T}{T_c}-1,\sqrt{\epsilon_{64}}\right),\qquad
q_g=\frac12+\frac{s-s_c^\rho(T)}{2w}.$$

The small reduced-temperature floor defines the exact critical limit. Existing log-sum-exp limiters smoothly bound the coordinate near $[-1,2]$. Its positive width is independent of the arbitrary entropy zero and increases quality with entropy; multiplying absolute critical entropy to obtain a width is no longer used. This is an explicitly conventional supercritical indicator, not a physical phase fraction. Unavailable source references/endpoints leave the optional result unavailable.

`supersaturation=True` enables exactly six additional fields: `pressure_saturation`, `temperature_saturation`, `supersaturation_ratio`, `supersaturation_degree`, `subcooling` and `superheating`. Bubble pressure at the state's temperature and bubble temperature at its pressure have independent source-domain checks. Subcooling/superheating retain their signed endpoint-enthalpy extensions and critical-density continuation. Bubble/dew data are shared only within the requested transaction, including one shared critical-density sample above the critical pressure; failed optional requests do not suppress independently available diagnostics. Unrequested diagnostics add no auxiliary flashes to ordinary evaluation.

### Direct-source phase solvers and constitutive blending

`FluidCoolProp` exposes `get_state_equilibrium`, `get_state_metastable` and `get_state_blending` as separate eager scalar operations using property names/aliases. They validate real scalar inputs and static solver settings, then hold the configured source lock for the entire calculation and state assembly. Density/temperature requests have a direct evaluation path; other inverse pairs require the applicable `rhoT_guess`, with separate equilibrium/metastable guesses for blending. A solver's success flag alone does not establish convergence: the returned state must pass its independent physical residual checks. These legacy phase methods default both diagnostic flags to true; callers can explicitly disable unneeded diagnostics. Their signatures do not replace the ordinary broadcasting/JAX `get_state` interface.

Numerical policies are `errors="raise"` or `"mask"`; masked failures return a scalar state with its enabled schema, failure status and NaN properties. Blending returns three such states in `(blended, equilibrium, metastable)` order. Malformed settings and unsupported requests still raise typed errors under mask mode. `clip` is explicitly unsupported because a unique physical projection of an arbitrary phase inverse is not defined; JAX tracing and array inputs are also rejected explicitly. The callback advertises only its implemented saturation operation, not these eager phase solvers.

Metastable evaluation uses the homogeneous Helmholtz branch, including formal unstable states used by stability studies. Its raw source handle can report equilibrium flash data at the same density/temperature, so returned homogeneous scalars govern its diagnostics and `is_two_phase` remains zero. Configured calls reuse their original source convention. Standalone helpers needing a new source object verify its reference basis against the underlying raw source and report an incompatible global convention rather than silently mixing entropy/energy gauges. Finite-difference helpers use prepared scratch or restore the caller's source state even after unsuccessful perturbations.

Blending is a constitutive interpolation with a cubic smoothstep, not another equilibrium EOS. It recovers the selected branch exactly at a transition endpoint, including when an optional value on the inactive branch is unavailable. Interior optional properties need both branches; interior `is_two_phase` is unavailable because a fractional flag has no physical meaning. Algorithm coordinates and weights are not exported as thermodynamic properties: in particular, `sigma` retains its surface-tension meaning. The blended state has `model="blended_real_fluid"` and its own immutable source/settings-dependent `reference_id`. Equilibrium and metastable output identities, and the source object's identity, remain unchanged. Blended states therefore cannot be combined with source-EOS states merely because their names match, nor are source thermodynamic response identities assumed for a constitutive blend.

## Reference data

Construction initializes applicable source critical, liquid triple, and vapor triple reference records through a dedicated CoolProp path for every evaluator. Their availability is independent of the active evaluator's phase capabilities. `fluid.references` exposes point-specific records; the convenience accessors `critical_point`, `triple_point_liquid`, and `triple_point_vapor` return their state when available and raise `UnavailableReferencePointError` otherwise.

Each record carries `available`, `reason`, `classification`, `interpretation`, source fluid/composition, backend/CoolProp version, reference convention, reported coordinates, numerical evaluation coordinates, and evaluation offsets. Records are immutable and shared only for identical source configurations. Each point is evaluated independently. Undefined optional properties do not make otherwise well-defined reference coordinates disappear.

Pure-fluid triples are identified as triple references. Pseudo-pure models such as Air retain available critical data and distinct liquid/vapor lower-saturation endpoints, labeled `pseudo_pure` with interpretation `lower_saturation_endpoint`; their pressures must not be forced equal. Unknown or unavailable points record the source reason without preventing ordinary supported property calls. A composite binary model has no assumed unique critical/triple point: its aggregate records are explicitly unavailable with `classification="composite"` and `interpretation="not_defined_for_composite"`; explicit point access raises `UnavailableReferencePointError`. Source records remain accessible through `mixture.references.components` and the component fluid objects. This does not preclude a separately validated mixture-EOS model in future work.

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

With `errors="clip"`, finite out-of-domain requests are projected onto a declared evaluator domain and the returned properties describe the adjusted, internally consistent state. `state.clipped` is a boolean array with the broadcast state shape. Recovered elements retain status 0 and valid properties; unrecoverable nonfinite inputs or floating-point failures retain failure statuses and NaNs. Unsupported requests and malformed arrays still raise. A zero throwing flag is not a replacement selector: callers choose the explicit policy string.

Phase 7 implements clipping for perfect gas with declared floors **1 K**, **0.1 Pa** and **$10^{-12}$ kg/m$^3$**. Projection uses a consistent PT state, adjusting pressure too when needed for the density floor; inverse inputs can therefore change together. A relative floor acceptance precision $10^{-12}$ handles arithmetic reuse. Returned flags distinguish actual domain recovery from successful ordinary evaluation. Clipped input directions have zero derivatives outside the clipping boundaries; derivatives at the boundary itself are not claimed. Finite but unrepresentable inverses are masked with provider-failure status using safe intermediates, preserving zero contributions from masked rows under differentiation. Ordinary states store no clipping array unless supplied; querying `.clipped` on such a state lazily returns false flags. Collections and serialization preserve supplied flags.

The optimized PG path resolves the input pair once, applies pair-specific numerical guards/projection, then assembles a complete state. Setup-validated range guarantees replace the scan of six calculated core fields; a PT/HP temperature projection therefore does not calculate entropy or energy for validation. A trusted native assembly path omits repeated schema/dtype/broadcast and per-property finiteness checks. Optional responses are normalized at their definition. Native mask/clip states contain the fully materialized, failed-row-masked properties. Large all-success batches use one shared fast branch to return the trusted arrays directly; batches containing failures apply per-element NaN masks. Public reads and exports require no deferred property masking or extra numerical state leaf. Eager raise skips failed-row masks because the entry point checks status and throws before exposing a failed state. The public `FluidState` constructor retains validation for arbitrary caller-supplied data. Stable square-root Sutherland algebra avoids an overflowing intermediate and preserves zero contributions from failed rows under model-parameter differentiation.

Phase 8 applies raise/mask/clip to bicubic tables. The numerical domain is the configured enthalpy/pressure rectangle, with pressure represented logarithmically during interpolation and inverse solves. Direct HP clipping evaluates the full state at the projected coordinates. Fixed-pressure or fixed-enthalpy inverses may also project a target only when an analytic cubic derivative test proves a monotone response along a connected valid slice. Unreachable or ambiguous two-property inverses retain failure statuses; clipping does not manufacture a root. Iteration limits and residual tolerances are setup options (`inverse_max_steps=64`, `inverse_tolerance=1e-10`), not repeated request validation or table-generation identity.

Bicubic inverse acceptance requires a bounded root, successful solve, independently checked scaled residual and invertible final Jacobian. PT rejects cells classified as two-phase: pure-fluid saturation makes that inverse nonunique, and a small interpolated slope is not proof of a validated gliding pseudo-pure inverse. A custom implicit JVP supplies validated first derivatives or zero failed contributions without throwing linear-solver callbacks. Optional unavailable coefficients use finite prepared arithmetic and property masks. The numerical PyTree contains coefficients and inverse-guess data; the complete persisted payload remains available through `.table` for inspection. Coefficient AD views retain the constructed source/availability pattern; changing a table's structural validity requires reconstructing a validated evaluator.

Cell checks reject mixed two-phase corner labels and subcritical liquid-to-vapor edges spanning a coarse cell. Where source saturation references are evaluable, Phase 10 additionally samples the configured source's bubble/dew enthalpy curves adaptively at construction and excludes cells intersecting padded boundary ranges, including curved crossings missed by corner flags. Unresolved curvature widens the exclusion and failed source probes invalidate the affected strip. This persisted finite-resolution guard is conservative, not a proof of exact EOS phase validity at every real pressure; it adds no native EOS or phase-classification work beyond the existing cell-mask lookup. Bounds, core finite/positive values and local property availability remain enforced. Ordinary equilibrium/mixture tables do not advertise saturation without a payload. Metastable endpoint tables expose `get_state_saturation(PQ_INPUTS, p, Q, errors=...)` at their stored liquid or vapor quality; unsupported QT/interior-quality evaluation is not advertised. Clip mode projects finite quality/pressure requests to that declared endpoint domain and reports the actual adjusted state.

Phase 6 implements raise/mask for ordinary CoolProp and the callback. Constructor-level `exceptions=False` is explicitly rejected with guidance to use per-call `errors="mask"`. Source errors retain their cause and failing index; unknown backend flash failures use provider-failure status rather than guessing nonconvergence from exception text. The binary model uses the same status meanings and indexed eager errors, retaining host causes without a mutable last-error transaction. Native binary paths may clip where their common domain/projection is defined; source/callback/hybrid paths reject clip explicitly. Every evaluator rejects a known but unimplemented policy through its published capabilities; shared request/state types recognizing `clip` does not fabricate clipping support for a source EOS.

Source temperature/pressure bounds allow relative numerical boundary precision $10^{-9}$, cached at setup and applied consistently to explicit and recovered coordinates. Returned values are never clipped. This handles observed inverse residuals near $2\times10^{-7}$ K and one-ULP saturation arithmetic; it is a numerical acceptance policy, not an EOS accuracy guarantee. Finite/positive input requirements and quality limits remain strict.

## JAX execution

Construction, capability validation, source-reference initialization, and table I/O happen outside transformed code. Input pairs, requested options, identity, and schema are static. Eager use of JAX-capable evaluators follows the same `errors="raise"` default as ordinary CoolProp.

For the CoolProp callback, outer JIT, vmap, JVP, or gradient execution requires explicitly selecting `errors="mask"`; perfect gas and bicubic accept explicit `mask` or `clip`. Binary objects follow their active component route: native PG/bicubic uses mask/clip; all-active `FluidJAX` uses a whole-host-transaction callback with mask only; direct-source/hybrid objects reject transformed execution. Requesting `raise` under these transformations raises `UnsupportedExecutionModeError` during tracing. Eager native raising calls inspect one returned status array to deliver indexed errors; native compiled mask/clip execution contains no host callback or eager status synchronization. A statically unsupported pair/property/option raises its typed error before numerical execution even under JIT.

Derivative claims apply to successful differentiable states and enabled properties inside the declared domain, away from phase boundaries, interpolation-cell discontinuities in higher derivatives, and failure boundaries. Masks/statuses are nondifferentiable. The callback evaluator declares first-order finite-difference derivatives; higher derivatives are unsupported until independently validated. Bicubic/native providers may advertise further modes only after verification. JAX capability tests include returned-state identity and status, not just a scalar extracted property. Applications control platform and precision settings: importing JaxProp selects no device and installs no exception hook, honors explicit environment/preloaded JAX settings, and retains the double-precision default only when it imports JAX first without an explicit precision choice.

The callback uses forward differences with step $10^{-6}(|x|+1)$ in double precision, skipping symbolic-zero tangents. Lower-precision callback outputs use a relative step of at least $\sqrt{\epsilon}$ to resolve rounded differences. The host calculation remains double precision and the returned dtype matches the active JAX precision. Tests compare first derivatives against independent CoolProp partial derivatives over pressure, temperature, enthalpy, entropy and density scales for HEOS and REFPROP. Numerical validity applies to the primal and perturbed states; crossing a failure boundary can make the derivative unavailable. Second and higher callback derivatives raise an informative execution-mode error. The explicit transformed-policy requirements also cover constant-input closures.

Each configured source prepares its reference-compatible auxiliary state, critical entropy, physical entropy scale and diagnostic bounds once. Runtime helpers reuse this data, so a later external CoolProp reference change cannot change an existing object's convention. One object lock serializes an entire host source batch or eager phase solve, including helper use; native JAX evaluators acquire no source lock. Source reconstruction verifies captured provenance only during setup/restore, outside numerical state calls. The raw provider handle remains an explicit advanced escape hatch whose external mutation is outside the managed evaluation contract.

Source and binary callback derivative coefficients are zeroed for failed primal rows and unavailable primal properties before tangent multiplication. This preserves zero failed contributions in both forward and reverse mode; masking only the final tangent would let `0 * NaN` contaminate a VJP. A valid primal whose required forward probe fails still has an unavailable NaN derivative. No zero slope is invented at that boundary.

Differentiate a floating property projection, or the enabled numerical property dictionary, rather than the complete state with its integer status and boolean flags. JIT/vectorization can return the full state. `perfect_gas.get_props_gradient` returns `PropertyDerivatives`, a separate quantity type carrying `.primal`, identity, status/clipping, canonical `.input_names`, SI `.input_units`/`.property_units`, and `.derivative_units(name)`. Its state shape is the primal broadcast shape and each derivative array adds a final axis of length two in input order. Autodiff computes local broadcast partials rather than a dense cross-element Jacobian; centered finite differences are an independently tested alternative. Derivative quantities cannot be stacked or combined as ordinary `FluidState` values.

## Verification and implementation ownership

The shared tests exercise existing public classes directly while the factory is implemented, so a missing factory cannot hide all remaining differences. Passing numerical behavior remains ordinary passing tests. Known contract gaps have narrow `xfail(strict=True)` markers with the responsible phase and reason; a fixed gap produces a failing unexpected pass until its marker is removed. Unexpected failures remain failures.

Phase 3 owns the shared typed-error foundation, state aliases, shapes, identity/reference compatibility, and controlled access. Phase 4 owns construction, capability publication, common input validation, and integration of typed static-request errors. Phase 5 owns source references. Phases 6–9 apply the same contract to source/callback, perfect-gas, bicubic, and consolidated binary evaluators, including numerical failures and JAX behavior. Phase 10 implements the phase/diagnostic repairs documented here; its final correctness and timing acceptance belongs to the implementation checklist. Lifecycle and consuming-application work remain in later phases. Completing Phase 2 establishes requirements, not automatic acceptance of subsequent implementations.
