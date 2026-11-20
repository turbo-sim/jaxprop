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

Phase support does **not** remove the existing phase-boundary cell guards or
make an ambiguous two-phase PT inverse unique. HP cells intersecting a phase
boundary remain rejected. A valid PQ endpoint is not permission to interpolate
through a rejected HP cell. Full wet-cycle/domain qualification is separate
from exposing diagnostics and endpoint calls.

Diagnostic accuracy and sensitivities depend on the two-dimensional table;
endpoint accuracy depends on its own pressure grid. Refine and compare with
the source EOS over the intended region before optimization qualification.

The opt-in setting, endpoint pressure grid, generator revision and relevant
source-code identities participate in cache provenance. The loader validates
both endpoint schemas and core numerical arrays; malformed payloads are rebuilt.
Use separate development cache folders when comparing with published results.
No published timing/reference artifacts are rebaselined by enabling this option.

Release qualification is indexed in
[the package validation note](../development/performance/phase_support_release/README.md).
