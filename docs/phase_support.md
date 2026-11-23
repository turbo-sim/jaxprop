# Opt-in bicubic phase diagnostics and saturation endpoints

JaxProp 0.6.1 introduces this opt-in functionality. Existing bicubic configurations retain their default
capabilities; enable the new payload explicitly in an ordinary single-source
equilibrium table:

```python
import jax
import jaxprop as jp

jax.config.update("jax_enable_x64", True)
fluid = jp.Fluid(
    "Water", method="bicubic", backend="HEOS",
    table_options={
        "h_min": 100e3, "h_max": 3.2e6,
        "p_min": 80e3, "p_max": 200e3,
        "N_h": 32, "N_p": 8, "N_p_sat": 64,
        "coarse_step": 1, "phase_support": True,
        "table_dir": "results/phase_support_tables",
    },
)
wet = fluid.get_state(jp.HmassP_INPUTS, 1.5e6, 130e3, supersaturation=True)
print(wet.subcooling, wet.superheating)
liquid = fluid.get_state(jp.PQ_INPUTS, 130e3, 0.0)
vapor = fluid.get_state_saturation(jp.PQ_INPUTS, 130e3, 1.0)
```

REFPROP is also tested where installed; select it explicitly with
`backend="REFPROP"`. Source queries occur during construction/table generation.
Runtime evaluation uses native array interpolation and supports the declared
JIT, vectorized and first-order differentiation modes without source callbacks.

## Diagnostics

`phase_support=True` generates the existing six source diagnostic fields and
their bicubic coefficients, including `subcooling` and `superheating`. It uses
the definitions in [the source calculations](../jaxprop/coolprop/core_calculations.py),
including their artificial wet-region and supercritical reference extensions.
The definitions are not replaced with a simple saturation-temperature
difference in regions where that would change their meaning.

As with the source provider, enable diagnostic access with
`supersaturation=True` on each state request. Otherwise these fields remain
disabled. Unsupported, disabled and locally unavailable properties preserve
their existing typed-error/availability distinctions. Individual optional
fields can remain unavailable; the option does not turn missing data into zero.

## Saturation endpoints

The phase-support payload contains separately validated liquid and vapor
endpoint values and pressure slopes, interpolated with cubic Hermite curves.
`N_p_sat` controls their pressure-grid resolution. Both public routes accept
`PQ_INPUTS` at **Q=0 or Q=1 only**; intermediate quality is not a new mixture
model. Scalar and broadcast batch calls use the same endpoint definitions.

The endpoint pressure interval is bounded below by the table/source triple
pressures and above by the smaller of the table maximum pressure and 98% of
critical pressure. It is recorded in the generation metadata. A wholly
supercritical table can expose the diagnostic fields without advertising
saturation support. Endpoint enthalpies need not be inside the separate HP
rectangle.

Use `errors="mask"` or `errors="clip"` under JAX transformations, as for other
native state requests. Masked invalid requests retain nonzero statuses and
NaN fields. Explicit clipping records the projection and selects the nearest
endpoint quality; the halfway tie selects liquid. Eager raising remains indexed.

## Accuracy, domains and caches

The published **0.6.1** package retains phase-boundary cell guards. Its HP cells
intersecting a phase boundary remain rejected, even when the requested state is
valid. The guard prevents a tensor polynomial from smoothing over the
thermodynamic derivative discontinuity or interpolating a discrete phase flag.

JaxProp **0.6.2** corrects
pure-fluid boundary evaluation with `phase_support=True`. The uniform
$h$–$\log(p)$ nodes, grid dimensions, original property arrays and coefficients
remain unchanged. One-sided Hermite bridges connect existing homogeneous cells
to the existing saturation endpoints. A smooth pressure partition preserves
continuity when the selected homogeneous cell changes. Pure-fluid wet states
use equilibrium endpoint lever rules and the PQ-map response Jacobians already
defined by the source provider. Runtime remains native JAX without EOS callbacks.

This correction is limited to the declared endpoint pressure interval and
pure fluids. Pseudo-pure temperature glide retains the existing cell guard;
it needs its actual quality-dependent map. Missing homogeneous anchors retain
the guard. Out-of-range/invalid inputs and locally unavailable optional fields
retain their existing policies. Singular pure-fluid wet responses remain
unavailable, exact endpoints expose single-phase responses, and saturation PT
remains ambiguous. A phase-aware PT seed avoids initializing single-phase
near-boundary flashes on the flat wet plateau.

Full wet-cycle/domain qualification is separate from property-level checks.
Release qualification is recorded separately from whole-cycle qualification.

Diagnostic accuracy and sensitivities depend on the two-dimensional table;
endpoint accuracy depends on its own pressure grid. Compare values and
sensitivities with the source EOS over the intended region before optimization
qualification. The ThermOpt development correction preserves the existing
uniform mesh and resolution; it does not use mesh refinement.

The opt-in setting, endpoint pressure grid, generator revision and relevant
source-code identities participate in cache provenance. The loader validates
both endpoint schemas and core numerical arrays; malformed payloads are rebuilt.
Use separate development cache folders when comparing with published results.
No published timing/reference artifacts are rebaselined by enabling this option.

Release qualification is indexed in
[the package validation note](https://github.com/turbo-sim/jaxprop/blob/main/development/performance/phase_support_release/README.md).

## Unreleased critical-region HP continuation

The isolated ThermOpt Phase 6 candidate extends pure-fluid HP evaluation
through the critical pressure strip. It retains the existing uniform mesh,
saturation grid and coefficients. Existing endpoint tangents continue beyond
the endpoint cap and fade smoothly to the original tensor polynomial at
critical pressure. The additional numerical payload uses critical anchors
already captured during setup; runtime introduces no source callbacks.

Phase/quality fields use a bounded approximate envelope anchored at the cap
and critical point, rather than fractional bicubic phase flags. The envelope
is quadratic in a cube-root reduced-pressure coordinate, matching cap values
and pressure slopes. One-phase quality follows the critical-entropy label.
Pure-fluid wet singular responses remain unavailable. The public PQ domain
is unchanged; near-critical HX snapping above its cap is not qualified by
this HP change. Pseudo-pure glide retains its existing guard.

This is an accuracy-qualified development approximation, not exact critical
thermodynamics. Independent EOS value checks and native AD/FD checks have
separate acceptance policies. Almost-critical reference flashes are checked
for input closure and source phase conventions before being used as accuracy
references. The published 0.6.2 installation does not contain this candidate.

## Unreleased deterministic tracing

Property iteration during phase-boundary and legacy saturation evaluation is
now canonical. This preserves the equations and arrays while removing Python
hash-seed differences from traced operation order. Identical native requests
in separate processes can therefore reuse the same persistent compilation
cache entry. Regression coverage compares StableHLO across two hash seeds;
the ThermOpt consumer also records an actual second-process cache hit.
