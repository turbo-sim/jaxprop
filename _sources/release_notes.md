# JaxProp 0.6.0

**Released on 15 November 2026.** Hosted Linux and Windows HEOS tests passed,
and downstream Barotropy verification passed 271 tests against the published
package. ThermOpt integration remains a separately tracked follow-up; this
release does not claim its completion. This minor version contains
intentional interface and physical-model changes relative to 0.5.8; consult
the [migration guide](migration.md) before upgrading.

## Interface and numerical behavior

- Construct direct CoolProp, CoolProp/JAX callback, bicubic, perfect-gas and
  fixed-composition binary models through `Fluid(..., method=...)`. The
  initializer returns the actual evaluator class.
- Use consistent complete `FluidState` results with canonical property names,
  explicit statuses, property availability, identities and array semantics.
  Unsupported pairs, properties and phase operations report informative errors.
- Inspect capabilities and source-reference metadata at construction. Pure and
  pseudo-pure source limitations remain explicit; approximate models do not
  acquire saturation behavior simply because their source has phase data.
- Use explicit `errors="mask"` or `errors="clip"` for native JAX transformations.
  Direct-source and callback models support raise/mask, with first-order
  finite-difference callback derivatives. Supported operations differ by model.
- Perfect-gas equations consistently preserve calibrated reference enthalpy,
  entropy and heat capacity. The model exposes 22 properties. Results can
  intentionally differ from the older inconsistent reference equations.
- Bicubic tables validate generation provenance and numerical structure,
  support bounded clipping and declared inverses, and retain optional-property
  availability. Incompatible old tables regenerate. Table reuse does not imply
  whole-domain interpolation accuracy.
- Binary models share a fixed-composition parcel formulation with 27 fields.
  `FluidMix` translates the legacy constructor; `FluidTwoComponent` aliases
  `FluidBinaryMixture`. Composition is not vapor quality, and phase envelopes
  are unsupported for the composite model.
- Source phase/metastable solvers, blending, spinodal calculations, grids,
  plotting and worker serialization have repaired behavior and regression
  coverage. Application-owned JAX configuration is respected.

## Verification and performance

The numerical implementation preceding the version-only change passes
**3,029 HEOS/REFPROP tests** and **54,272 property comparisons**, with no
failures, skips or expected failures. The example/documentation phase separately
verified quick/full execution profiles and scientific limitations. These are
local CPU/x64 results, not a claim of GPU or downstream application validation.

Against the pinned, unchanged PyPI 0.5.8 package, the latest four-worker native
comparison records bicubic full HP batches of 1,024 states at **129.3 versus
575.7 microseconds**, and the 128-cell heat-exchanger consumer at **25.1 versus
50.8 microseconds**. Scalar and small-batch dot kernels remain in use; large
batches use fused polynomial reductions. Complete states, statuses, clipping
and derivatives are retained.

The native comparison passes 59 of 61 original timing budgets. PG scalar
compressor/expander consumers remain approximately **29–33% slower**, around
**3 microseconds** extra per call. Historical workload-context variability,
setup/cache-validation costs and additional first-call compilation remain
documented. All four latest cold-profile guards pass their separate allowances.
On 15 November the user accepted the recorded residual costs and closed
Phase 14. No timing threshold or historical measurement was changed. A final
119-case repeat on the last numerical source was not performed; application
measurements follow in ThermOpt and Barotropy.

See the repository's [performance report](https://github.com/turbo-sim/jaxprop/blob/main/development/performance/phase14/reduction_followup.md),
[benchmark instructions](https://github.com/turbo-sim/jaxprop/blob/main/benchmarks/README.md)
and [implementation checklist](https://github.com/turbo-sim/jaxprop/blob/main/z_IMPLEMENTATION_PLAN.md)
for measurement scope and optional future studies. This release does not
promise universal performance parity across models, shapes or devices.
