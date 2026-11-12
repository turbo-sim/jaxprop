# JaxProp fluid interface audit

The fluid classes already share useful numerical operations and the `FluidState` container. Their constructors, keyword arguments, stream identity, array shapes, supported features, and failure handling differ. These differences prevent ThermOpt from changing evaluators through configuration alone.

Uniform, consistent behavior is the primary design priority. Define the intended interface first and adapt every evaluator and consuming package to that contract. Backward compatibility is secondary. Migration notes should explain intentional changes to signatures, shapes, references, and errors.

Track implementation, acceptance checks, and completed work in the [implementation checklist](z_IMPLEMENTATION_PLAN.md).

Interchangeability should provide one construction and property-call contract over each model's declared domain. CoolProp evaluation, its JAX callback wrapper, and bicubic interpolation can evaluate the same source fluid model. Selecting a perfect gas or a two-component formulation also selects a physical model, whose applicable phase features and reference conventions need to be explicit.

This audit covers the local JaxProp 0.5.8 checkout at commit `ed66d108555e71d0d8543a1b670d05237a834221` on 12 November 2026. Short runtime probes imported this checkout using ThermOpt's virtual environment, with Python 3.13.2, JAX 0.6.2, Equinox 0.12.2, CoolProp 7.2.0, NumPy 2.4.4, and SciPy 1.17.1. Bicubic probes reused an existing 4 by 4 CO2 table. These runtime results establish API behavior at the sampled points.

## Current interfaces

| Fluid object | Construction differences | Property-call differences | State and feature differences |
| --- | --- | --- | --- |
| `Fluid` | Accepts `name`, `backend`, `exceptions`, and `identifier`. | Accepts `input_type`, `prop_1`, `prop_2`, `generalize_quality`, and `supersaturation`. Input-pair support depends on the underlying CoolProp fluid and backend. | Returns identity in `FluidState`. Pure fluids expose critical/triple states and phase plotting. Singleton axes are squeezed. |
| `FluidJAX` | Accepts `name`, `backend`, and `exceptions`. Rejects `identifier`. | Uses `input_pair`, `x`, and `y`. Rejects the two diagnostic options. | Omits both state identity fields. Reference states and plotting are accessed through `.fluid`. Preserves NumPy/JAX input broadcast shapes. |
| `FluidBicubic` | Uses `fluid_name` and requires pressure/enthalpy bounds and grid sizes. Accepts `identifier`, with its own table options. | Uses `input_type`, `val1`, and `val2`. Supports eight input pairs. Rejects the diagnostic options and quality input pairs. | Returns `FluidState` with `fluid_name` but drops `identifier`. Exposes a critical point for supported pure source fluids. Triple states and phase plotting are absent. |
| `FluidPerfectGas` | Accepts `name`, reference temperature/pressure, and `identifier`. Stores the name as `fluid_name`. | Uses `input_pair`, `x`, and `y`. Supports six input pairs. Rejects the diagnostic options. | Returns identity, with `identifier=None` by default. Fields can have different broadcast shapes. Applicable gas properties are partly omitted. |
| `FluidMix` | Accepts common identity arguments plus a fixed `mixture_ratio`. Requires a component name string such as `water_nitrogen_mixture`. | Supports scalar `HmassP_INPUTS` only. Rejects diagnostic options. | Returns identity in `FluidState`. Reference phase metadata and plotting are absent. |
| `FluidTwoComponent` | Takes two component names/backends and an `identifier`. Its exports are commented out. | Requires a fourth positional composition ratio `R`. | Direct module import currently fails on the missing `MixtureState` type. The implementation also uses an obsolete mixture-state schema. |

The exported classes are defined in the [package initializer](jaxprop/__init__.py#L30). Their entry points are in [CoolProp Fluid](jaxprop/coolprop/fluid_properties.py#L167), [FluidJAX](jaxprop/coolprop/jax_wrapper.py#L110), [FluidBicubic](jaxprop/bicubic/bicubic_interpolation.py#L75), [FluidPerfectGas](jaxprop/perfect_gas/perfect_gas.py#L280), [FluidMix](jaxprop/coolprop/fluid_properties.py#L766), and [FluidTwoComponent](jaxprop/two_component/mixture_calculations.py#L47).

## Proposed public interface

A public constructor can select the evaluation implementation and give every evaluator the same construction contract. Consolidate existing entry points where that improves consistency. Keep the implementation selector distinct from the existing CoolProp `backend` argument, whose values include `HEOS` and CoolProp's own tabular `BICUBIC&HEOS`.

The following is a proposed API example. The function name and configuration keys can be finalized during implementation.

```python
fluid = jxp.create_fluid(
    name="CO2",
    provider="bicubic",
    backend="HEOS",
    identifier="working_fluid",
    table_options={
        "h_min": 110e3,
        "h_max": 1.3e6,
        "p_min": 8e6,
        "p_max": 39e6,
        "N_h": 200,
        "N_p": 200,
    },
)
state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
temperature = state.T
```

JaxProp should own table creation, loading, validation, and provider selection. The consumer should receive a fluid object with the common public contract and use its `get_state` method. Existing CoolProp input constants and `FluidState` aliases provide a practical starting point.

## Changes required for direct ThermOpt use

### 1 Standardize construction and identity

**Priority high.** Give every usable fluid object a common `name` and `identifier`, with a documented default for `identifier`. Accept the same identity keywords at construction and use one documented identity vocabulary across evaluators and returned states. Expose the evaluation implementation separately from the physical/source model and its EOS backend. Keep table, perfect-gas, and composition settings in explicit model-specific options.

`FluidJAX` currently rejects ThermOpt's `identifier` constructor keyword. Bicubic and perfect gas lack `.name`, which ThermOpt reads when assembling component results. `FluidJAX` returns states with `fluid_name=None` and `identifier=None`. Bicubic receives an identifier but omits it when constructing the returned state. ThermOpt uses the state identifier to recognize the working-fluid stream in its plots. [JAX construction and output](jaxprop/coolprop/jax_wrapper.py#L116), [bicubic output](jaxprop/bicubic/bicubic_interpolation.py#L1440), [ThermOpt component metadata](../thermopt/thermopt/components/basic_components.py#L281), [ThermOpt stream identification](../thermopt/thermopt/optimize_cycle.py#L893).

Acceptance requires construction with `identifier="working_fluid"` and preservation of that value through scalar evaluation, array evaluation, addition, stacking, reversal, indexing, and plotting.

### 2 Standardize property arguments and diagnostic options

**Priority high.** Choose one set of public argument names and types for `get_state` and apply it consistently to every evaluator. Update consumers and examples to use those names. The current `FluidPerfectGas` annotation says `input_pair: str`, although dispatch uses the integer input constants.

ThermOpt's compression and expansion routines request `supersaturation=True`, and expansion also requests `generalize_quality=True`. Only `Fluid` currently accepts these options. Implement the requested outputs for supporting providers and expose option support before evaluation. A requested unsupported diagnostic needs a specific, documented outcome.

The semantics also need correction in the current `Fluid.get_state` documentation. Its `supersaturation` option adds saturation-departure, subcooling, and superheating outputs following the normal CoolProp state update. Metastable evaluation has dedicated methods. Generalized quality changes the reported quality. These options carry meaningful output semantics and need explicit handling across providers. [Fluid options and documentation](jaxprop/coolprop/fluid_properties.py#L238), [normal CoolProp evaluation](jaxprop/coolprop/core_calculations.py#L575), [single-phase diagnostics](jaxprop/coolprop/core_calculations.py#L79), [generalized quality](jaxprop/coolprop/core_calculations.py#L1146), [ThermOpt expansion calls](../thermopt/thermopt/components/basic_components.py#L438).

Acceptance requires the same supported call signatures, tests of actual diagnostic values, and an explicit unsupported-option result. Changing only the signature would leave the semantic mismatch unresolved.

### 3 Publish supported inputs and capabilities

**Priority high.** Expose supported input pairs, available properties, optional diagnostics, phase operations, and JAX transformation support through public capability information. Validate a consumer's requirements before running a cycle or optimizer.

The common isentropic cycle path needs `HmassP_INPUTS`, `PT_INPUTS`, and `PSmass_INPUTS`. Bicubic and perfect gas support all three. `FluidMix` currently supports only scalar `HmassP_INPUTS`, selected by the literal integer `20`. Replace that literal with the named constant and declare its limited support. Bicubic has eight implemented pairs and perfect gas has six. Derive the available set from each implementation. [Bicubic dispatch](jaxprop/bicubic/bicubic_interpolation.py#L1347), [perfect-gas dispatch](jaxprop/perfect_gas/perfect_gas.py#L270), [mixture dispatch](jaxprop/coolprop/fluid_properties.py#L841).

Acceptance requires a capability query for each advertised evaluator, successful calls for its declared pairs, and informative rejection of unsupported pairs before an optimization starts.

### 4 Initialize common CoolProp reference points and declare phase capabilities

**Priority high.** Every fluid object should expose `critical_point`, `triple_point_liquid`, and `triple_point_vapor` through one common reference-data contract. Initialize applicable source-fluid points using a dedicated CoolProp object during construction, independently of the active numerical evaluator. This applies to ordinary CoolProp, its JAX wrapper, bicubic, and perfect gas. A perfect-gas object can therefore expose the real source fluid's CoolProp reference points while its own evaluator rejects quality and saturation calculations.

Use a shared initialization routine and record source fluid/composition, CoolProp backend/version, reference convention, point interpretation, availability, and any failure reason. Reuse immutable reference records across evaluators. Evaluate each point independently. When CoolProp cannot supply a unique or meaningful point, keep the availability result and reason accessible and raise an informative reference-point error when the unavailable point is requested. A composite model needs a matching reference definition, with component reference data accessible through its component objects.

The current `is_pure_substance` gate suppresses all reference points for pseudo-pure fluids such as Air, including available critical data. Replace that blanket gate with source classification and point-specific availability. Air's CoolProp model is explicitly marked pseudo-pure and supplies a model critical point plus liquid/vapor endpoints at its lower saturation temperature. These endpoints need a `pseudo_pure`/lower-saturation-reference interpretation in metadata. In a CoolProp 7.2.0 HEOS probe, both endpoints were at 59.75 K, with liquid and vapor pressures about 5.265 and 2.432 kPa. This pressure difference identifies distinct model endpoints. Use the source-provided values with their interpretation recorded. CoolProp's versioned fluid definition also stores these distinct reference endpoints. [Current purity gate](jaxprop/coolprop/fluid_properties.py#L195), [purity classification](jaxprop/coolprop/fluid_properties.py#L1858), [CoolProp Air definition](https://raw.githubusercontent.com/CoolProp/CoolProp/v7.2.0/dev/fluids/Air.json), [local probe](development/test_audit/results/baseline_20261112_195233/reference_point_probe.json).

Preserve the distinction between reported reference coordinates and the coordinates used for numerical evaluation. The current critical-state routine perturbs temperature by `1e-5` K. Report such evaluation details explicitly and give singular/undefined critical properties their own availability outcome. A reference-point record should remain usable for its well-defined primary properties. [Critical-state evaluation](jaxprop/coolprop/fluid_properties.py#L210).

ThermOpt setup accesses these reference points and also makes `QT`, `DmassT`, and dilute-gas `PT` queries. A supercritical cycle table can exclude those conditions, so setup/reference evaluation needs an explicit supported path alongside the interpolation-domain policy. Accessing CoolProp reference metadata on a perfect-gas object must preserve the perfect-gas evaluator's declared input/property capabilities. Likewise, Air's pseudo-pure classification and its evaluator's actual phase support need independent declarations. [ThermOpt setup](../thermopt/thermopt/optimize_cycle.py#L521), [bicubic reference setup](jaxprop/bicubic/bicubic_interpolation.py#L144).

For phase-capable evaluators, standardize `PQ`/`QT` access and the returned state schema. Bicubic's `get_state_saturation` currently returns a dictionary, and ordinary single-fluid table generation saves an empty saturation dictionary. A normal cached-table call therefore raises `KeyError("pressure")`. ThermOpt's optional heat-exchanger saturation nodes require `PQ` queries. [Empty saturation data](jaxprop/bicubic/bicubic_interpolation.py#L398), [saturation entry point](jaxprop/bicubic/bicubic_interpolation.py#L1384), [ThermOpt saturation queries](../thermopt/thermopt/components/basic_components.py#L42).

Acceptance requires equivalent CoolProp reference metadata across evaluators for the same source fluid, working Air initialization with explicit pseudo-pure interpretation, and informative outcomes for unavailable or ambiguous mixture reference points. Perfect-gas quality/saturation requests must raise the declared unsupported-request error even when source reference points are available. Advertised phase evaluation also requires working liquid/vapor endpoint queries.

### 5 Specify array shapes and complete the common state contract

**Priority high.** Normalize scalar and array-like inputs once and preserve their exact broadcast shape in numerical state fields. Define scalar output as shape `()`, singleton-vector output as `(1,)`, and broadcasted matrices as their full broadcast shape. Represent unavailable numerical properties consistently, including their shape and availability information.

`Fluid` squeezes singleton dimensions. `FluidJAX` preserves NumPy/JAX array shapes but fails on Python-list inputs in the tested call. Perfect-gas PT evaluation can return `p` with shape `(2, 1)`, `T` and `h` with `(1, 3)`, and `rho` with `(2, 3)` from the same query. Its omitted fields remain scalar NaNs. Bicubic already preserves the tested scalar and array shapes. `FluidMix` lacks an array evaluation path. [Fluid broadcasting and squeeze](jaxprop/coolprop/fluid_properties.py#L278), [JAX callback entry](jaxprop/coolprop/jax_wrapper.py#L47), [perfect-gas PT evaluation](jaxprop/perfect_gas/perfect_gas.py#L191), [bicubic broadcasting](jaxprop/bicubic/bicubic_interpolation.py#L1366).

Complete the common core state where the model supplies it. For example, perfect-gas assembly omits computable internal energy and leaves the phase marker as NaN. Keep SI units, mass/molar conventions, canonical field names, and availability semantics documented together. [Perfect-gas state assembly](jaxprop/perfect_gas/perfect_gas.py#L241), [shared state schema](jaxprop/helpers_props.py#L11).

Acceptance requires the same shape contract for Python scalars/lists, NumPy arrays, JAX arrays, singleton axes, broadcast matrices, and supported empty batches. Test both available and unavailable fields.

### 6 Unify invalid-state and convergence handling

**Priority high.** Define consistent outcomes for invalid inputs, unsupported requests, out-of-domain states, and failed inversions. Provide an eager exception policy and a JAX-compatible validity/status policy. For batched evaluation, specify whether valid entries survive an invalid entry and expose that choice consistently.

Unsupported operations need explicit, informative errors. Validate the input pair, requested properties, and options against the selected evaluator before entering numerical code or JIT when the request is static. Use distinct error types for an unsupported input pair, an unsupported property, and unavailable reference metadata. Messages should identify the fluid, evaluator, requested operation/property, reason, and relevant supported alternatives. For example, a perfect-gas `QT_INPUTS` request should explain that this evaluator has no phase-equilibrium calculation and list its supported pairs. An explicit request for `quality_mass`, saturation pressure, or surface tension must produce the same unsupported-property outcome through canonical and alias access.

Property availability must be part of the returned-state contract. The current `FluidState` declares public numeric fields with NaN defaults, so adding an error only to `get_state` cannot protect later requests such as `state.Q` or `state["Q"]`. Provide controlled property access with availability metadata, covering attribute, dictionary, and explicit property requests consistently. Numerical state storage and JAX tree operations can use fixed representations internally, while public requests enforce the declared capability. A supported property that is undefined at a particular state needs a state-specific unavailability result. Treat static unsupported requests consistently even when a caller chooses non-throwing numerical failure handling. [Current state fields and lookup](jaxprop/helpers_props.py#L11).

Observed behavior currently differs substantially. `Fluid(exceptions=False)` returns `None` for an entire batch when one element fails. `FluidJAX(exceptions=False)` preserves valid entries and fills failed entries with NaNs. Callback exceptions surface through JAX as `XlaRuntimeError`. Perfect-gas PT silently changes negative pressure and temperature to `0.1`. Bicubic can return a negative input pressure with NaN temperature, or finite extrapolated properties beyond its table domain. [Fluid exception wrapper](jaxprop/coolprop/fluid_properties.py#L50), [callback failure handling](jaxprop/coolprop/jax_wrapper.py#L20), [perfect-gas clipping](jaxprop/perfect_gas/perfect_gas.py#L191), [bicubic domain handling](jaxprop/bicubic/bicubic_interpolation.py#L1403).

Publish the table domain and an explicit out-of-domain policy. Check inverse-solver convergence, recovered inputs, and the returned domain. Bound the inverse problem where appropriate. Wire iteration limits to the solver's actual public argument. The current bicubic one-dimensional inverse methods place `max_steps` inside `options`, although the installed Optimistix `root_find` accepts it as a separate keyword. [Bicubic inversions](jaxprop/bicubic/bicubic_interpolation.py#L1442).

Acceptance requires consistent invalid-input and mixed-validity batch tests in eager and supported JAX modes, including invalid pressure, out-of-table enthalpy, unsuccessful roots, and option/pair errors. A successful result must recover the specified input properties within a documented tolerance.

## Changes for dependable use by other packages

### 7 Validate generated tables and cache identity

**Priority high for bicubic deployment.** Initialize table data with detectable invalid values, track failed nodes explicitly, and reject or explicitly mask incomplete interpolation cells. Validate the properties required by the advertised capabilities. Phase-specific fields can legitimately be unavailable and need a separate availability policy.

The ordinary bicubic generator allocates with `np.empty`, catches and skips failed grid evaluations, then calculates coefficients and saves the table. This can preserve unwritten data in a cache that appears usable. Validate ordered positive pressure bounds, ordered enthalpy bounds, grid sizes, and phase/composition settings before generation. [Allocation](jaxprop/bicubic/bicubic_interpolation.py#L234), [skipped nodes](jaxprop/bicubic/bicubic_interpolation.py#L317), [coefficient calculation and save](jaxprop/bicubic/bicubic_interpolation.py#L344).

Include gradient method, saturation-grid settings, cache format, and source/reference implementation information in cache identity and validation, in addition to the existing fluid/domain/grid/phase/composition data. Use atomic cache replacement so concurrent constructors cannot consume a partial file. Make generation and regeneration observable through controllable logging or progress options. [Current metadata](jaxprop/bicubic/bicubic_interpolation.py#L127), [cache loading](jaxprop/bicubic/bicubic_interpolation.py#L164), [cache writing](jaxprop/bicubic/bicubic_interpolation.py#L401).

Acceptance requires rejection of an incomplete table, deterministic invalidation when generation settings change, and successful concurrent construction using one cache location.

### 8 Define composition and repair the two-component entry point

**Priority high before advertising interchangeable mixture evaluation.** Define component order, composition basis, validation, and fixed versus per-call composition. A fixed composition at construction allows the usual three-argument property call. A dynamic composition can use an explicit documented keyword with broadcasting and differentiation rules.

`FluidMix` stores a ratio at construction and defaults it to `None`. `FluidTwoComponent.get_state` requires a fourth positional `R`. Both describe a mass ratio of component 1 to component 2. Replace name-string inference with explicit component/model configuration and apply the chosen composition contract uniformly. [FluidMix configuration](jaxprop/coolprop/fluid_properties.py#L766), [mixture ratio conversion](jaxprop/coolprop/core_calculations.py#L1649), [two-component call](jaxprop/two_component/mixture_calculations.py#L69), [bicubic mixture selection](jaxprop/bicubic/bicubic_interpolation.py#L186).

Repair `FluidTwoComponent` before exporting it as a usable evaluator. Direct import fails because the return annotation references an absent `MixtureState`. State construction also uses that missing class and old field names such as `mass_fraction_1` and `volume_fraction_1`. Align these with the common schema and preserve the object's identifier. [Import failure location](jaxprop/two_component/mixture_calculations.py#L209), [obsolete output schema](jaxprop/two_component/mixture_calculations.py#L283), [current mixture fields](jaxprop/helpers_props.py#L56).

Acceptance requires import/construction tests for every advertised class, validated composition, compatible mixture output fields, and successful declared scalar/array property operations.

### 9 Specify enthalpy and entropy reference conventions

**Priority medium.** Record reference conventions in model metadata and keep every forward and inverse relation consistent with the selected convention. This matters when a consuming package carries physical enthalpy design variables between evaluator configurations.

Perfect gas currently uses $h=c_pT$, while its entropy reference is taken from CoolProp. At the default CO2 reference conditions of 300 K and 101325 Pa, a probe returned perfect-gas enthalpy of approximately 250026.19 J/kg and CoolProp enthalpy of 507417.18 J/kg. The perfect-gas approximation remains an explicit physical choice. Its enthalpy zero can also be aligned at a chosen reference state, for example with $h=h_{\mathrm{ref}}+c_p(T-T_{\mathrm{ref}})$, if that is the intended contract. Such a change needs matching inverse formulas and migration guidance for saved enthalpy inputs. [Reference construction](jaxprop/perfect_gas/perfect_gas.py#L35), [current enthalpy evaluation](jaxprop/perfect_gas/perfect_gas.py#L191), [enthalpy inversion](jaxprop/perfect_gas/perfect_gas.py#L138).

Acceptance requires documented reference metadata, forward/inverse consistency, and reference-point tests for providers representing the same source model.

### 10 Document JAX behavior and application settings

**Priority medium.** Expose JIT, vectorization, and differentiation capabilities for each evaluator. State how derivatives are obtained and which transformations/orders are supported. Test advertised behavior with the common return type and error policy.

`FluidJAX` executes CoolProp through host callbacks and supplies a custom JVP from forward finite differences. Its step is $10^{-6}(|x|+1)$ for each input. Bicubic and perfect gas use native JAX operations. Scalar JIT and first-derivative probes succeeded for the JAX wrapper and native bicubic PH queries. Higher-order and all-input-pair support need their own checks. [Callback and JVP](jaxprop/coolprop/jax_wrapper.py#L47), [bicubic entry point](jaxprop/bicubic/bicubic_interpolation.py#L1365), [perfect-gas entry point](jaxprop/perfect_gas/perfect_gas.py#L290).

Allow the consuming application to choose device and precision configuration. Package import currently assigns `JAX_PLATFORM_NAME="cpu"` and enables x64 globally. The initializer also installs an IPython exception hook when available. Provide explicit setup choices for these application-wide settings. Review array-valued reference metadata marked static in Equinox, which currently produces a warning during bicubic construction. [Import settings](jaxprop/__init__.py#L1), [bicubic static metadata](jaxprop/bicubic/bicubic_interpolation.py#L73).

Acceptance requires a published transformation matrix, derivative checks for each claimed mode, and import tests that respect application configuration.

### 11 Define object reuse and serialization behavior

**Priority medium.** Make construction/setup reusable across repeated property evaluations, with a documented concurrency model for mutable CoolProp states. Keep plotting and reporting state from affecting numerical evaluations. State whether an evaluator can be serialized, reconstructed, or passed to worker processes.

The existing `Fluid` serialization method removes `_AS` but retains its unpickleable `abstract_state` alias. A serialization probe therefore fails. `FluidMix` can be serialized but restores its component states as `None` without rebuilding them, so subsequent evaluation fails. Reconstruct source states from stored configuration when restoring supported objects. [Fluid serialization](jaxprop/coolprop/fluid_properties.py#L156), [AbstractState alias](jaxprop/coolprop/fluid_properties.py#L180), [FluidMix serialization](jaxprop/coolprop/fluid_properties.py#L753).

Acceptance requires repeat evaluations without setup work leaking into the numerical contract and a working serialization/evaluation round trip for classes that advertise serialization.

### 12 Make plotting available through a common supported path

**Priority medium for full ThermOpt integration.** Offer phase plotting through a shared utility or a consistently exposed fluid method, with capability checks and an explicit reference-data source. ThermOpt currently calls `self.fluid.plot_phase_diagram(...)`. This method exists on `Fluid`, sits behind `.fluid` on `FluidJAX`, and is absent from the native bicubic and perfect-gas objects. [Fluid plotting](jaxprop/coolprop/fluid_properties.py#L486), [ThermOpt phase plotting](../thermopt/thermopt/optimize_cycle.py#L758).

Acceptance requires the usual ThermOpt cycle plotting path to work for a supported real-fluid evaluator, including preserved stream identifiers. Gas-only models need an applicable plotting outcome declared by their capabilities.

### 13 Fix shared state aliases and collection shapes

**Priority medium.** Remove the duplicated `isobaric_expansion_coefficient` entry in `PROPERTY_ALIASES`. Its later empty alias list overwrites `alpha_p`, which currently raises `KeyError` despite the canonical property being available. Validate duplicate source entries and alias collisions before building the lookup. [Alias definitions](jaxprop/helpers_props.py#L229), [overwriting entry](jaxprop/helpers_props.py#L251).

Define `FluidState.stack` as adding a leading state axis while preserving existing axes. Its current `.squeeze()` removes singleton axes, including the entire batch axis when stacking one scalar state. Preserve the already useful metadata checks, indexing, concatenation, reversal, and JAX differentiation behavior. [State collection operations](jaxprop/helpers_props.py#L123), [stack implementation](jaxprop/helpers_props.py#L168).

Acceptance requires canonical and alias lookup tests, one-state stacking, multidimensional singleton preservation, and the same identity/shape rules across all evaluators.

## Verification and refactoring

The [pre-refactor test audit](z_PRE_REFACTOR_TEST_AUDIT.md) maps current coverage and assertion weaknesses to the tests needed before changing the interface. Tests for valid physics should protect numerical results. Tests for construction, shapes, identity, options, capabilities, and failures should encode the agreed uniform behavior.

Add a shared conformance suite parametrized over advertised evaluator configurations. Use each provider's declared domain and capabilities. Numerical comparisons should use compatible physical models and reference conventions, with interpolation-appropriate tolerances.

The suite should cover the construction and identity contract, keyword compatibility and option semantics, core field availability, scalar/singleton/broadcast shapes, declared input pairs, unsupported requests, invalid batches, domain/convergence handling, reference metadata, applicable phase queries, JAX modes, and supported serialization. Include a ThermOpt-style isentropic compressor/expander and discretized heat exchanger using the same component entry points for each compatible evaluator.

Existing tests provide scalar perfect-gas round trips/gradients, bicubic interpolation checks, and shared-state collection checks. Add cross-provider contract cases to these. The bicubic consistency test currently samples `linspace(log(p_max), log(p_max), 25)`, so its pressure coverage needs correction. The example test module is entirely commented out. [Perfect-gas tests](tests/test_perfect_gas_eos.py#L40), [state tests](tests/test_fluid_state.py#L11), [bicubic pressure sampling](tests/test_bicubic_consistency.py#L87), [example test module](tests/test_example_cases.py#L1).

Two numerical corrections belong alongside the affected capability work. Pass bicubic inverse `max_steps` to the actual solver argument, and correct the saturation central derivative from `(fp_plus - fp_minus) / 2 * eps_p` to division by `(2 * eps_p)` before advertising that derivative path. [Inverse solver options](jaxprop/bicubic/bicubic_interpolation.py#L1456), [saturation derivative](jaxprop/bicubic/bicubic_interpolation.py#L1342).

For ThermOpt, adopt the JaxProp construction function, create/reuse the configured fluid objects outside repeated cycle evaluations, and obtain setup/reference and plotting data through the supported public paths. Query diagnostic capabilities where those outputs are needed. The current cycle constructs three `Fluid` objects during every evaluation and assumes the pure-real-fluid setup interface. These caller changes connect configuration to the proposed common contract. [ThermOpt cycle construction](../thermopt/thermopt/cycles/cycle_power_recuperated.py#L19), [ThermOpt setup construction](../thermopt/thermopt/optimize_cycle.py#L413).

Start implementation with construction/identity, query signatures, supported-input discovery, shapes, and failure handling. Then make reference/phase/plotting operations cover the intended ThermOpt cases and validate bicubic cache integrity. Complete broader mixture, reference-convention, JAX-configuration, and lifecycle support against the published capabilities. Choose consistent behavior when existing entry points conflict and document the consumer changes needed for the selected interface.
