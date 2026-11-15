# Migrating to the consistent fluid interface

This guide describes the 0.6.0 fluid-interface refactor relative to 0.5.8.
Version 0.6.0 was published on 15 November 2026. The common initializer returns an
actual evaluator object, and ordinary calculations use the same `get_state`
call across the supported models. Numerical providers are interchangeable over
their declared domains when their physical model and source/reference convention
match. A perfect-gas approximation, a parcel mixture and a real-fluid EOS describe
different physical models.

## Construct a model once

```python
import jaxprop as jp

source = jp.Fluid("CO2", method="coolprop", backend="HEOS")
callback = jp.Fluid("CO2", method="coolprop_jax", backend="HEOS")
perfect_gas = jp.Fluid(
    "Air",
    method="perfect_gas",
    model_options={"T_ref": 300.0, "p_ref": 101325.0},
)
table = jp.Fluid(
    "CO2",
    method="bicubic",
    backend="HEOS",
    table_options={
        "h_min": 600e3, "h_max": 850e3,
        "p_min": 5e6, "p_max": 12e6,
        "N_h": 8, "N_p": 8, "coarse_step": 1,
        "table_dir": "output/co2_tables",
    },
)

state = table.get_state(jp.HmassP_INPUTS, 720e3, 9e6)
print(state.T, state.rho, state.h)
```

The small table above demonstrates construction, not a resolution-independent
accuracy guarantee. Validate the table domain and resolution for its application.
Generation or validated cache loading happens during construction, outside
repeated state calls. There is no application-side backend adapter and no
automatic EOS fallback during interpolation.

`method` selects JaxProp's evaluator/model. `backend` selects its CoolProp source
and defaults to `HEOS` for a single-source model. An explicitly requested
`REFPROP` source must be usable locally; it does not silently fall back to HEOS.
`method="bicubic"` requests **JaxProp's FluidBicubic tables**, independently of
CoolProp's own tabular-backend names. Perfect-gas calibration/reference data also
use the selected source backend.

Unknown settings and options for the wrong model fail at construction. Inspect
`fluid.configuration`, `fluid.capabilities`, `fluid.references`, `fluid.provider`
and `fluid.model` at setup. A stream `identifier` is preserved separately from
source canonicalization. Public concrete classes remain available, and
`isinstance(fluid, jp.Fluid)` works for the selected evaluator.

## Current capability matrix

The object-level capability record is authoritative for its specific model,
table payload, enabled components and source backend.

| `method` | Model/evaluator | Input pairs | Execution/derivatives | Numerical policies | Phase operations |
| --- | --- | --- | --- | --- | --- |
| `coolprop` | Real fluid, direct source | 12 tested mass pairs | Eager NumPy-backed states; no JAX autodiff contract | `raise`, `mask` | Saturation; scalar equilibrium, homogeneous/metastable and blending studies |
| `coolprop_jax` | Same real-fluid source through host callbacks | Same source pairs | Eager, JIT, vmap, first-order JVP/grad through finite differences | `raise` eagerly; explicit `mask` in transformations | Saturation through supported source routes |
| `bicubic` | JaxProp real-fluid interpolation | 8 pairs within the declared table domain | Native JAX; first-order value/input/model AD in valid domains | `raise`, `mask`, `clip` | Stored liquid/vapor endpoint saturation where that payload exists; arbitrary Q and QT are unsupported |
| `perfect_gas` | Constant-heat-capacity model calibrated from the source | 6 analytic pairs | Native JAX; first-order input/model AD | `raise`, `mask`, `clip` | No quality pairs, saturation or own phase envelope |
| `binary_mixture` | Fixed-composition common-pressure/temperature parcels | PT, HP, PS | Native PG/bicubic combinations; all-active-source callbacks; direct/hybrid sources eager | Provider-dependent, as below | No own vapor quality or saturation envelope |

Ordinary source/callback/pure-table states enable 25 properties, perfect gas 22,
and binary states 27. Source/callback capability properties also include six
requested diagnostics. `capabilities.properties` describes the supported union;
`state.available_properties` is the set actually enabled for that request. A
locally undefined enabled property remains unavailable without invalidating an
otherwise successful core state.

The perfect-gas pairs are PT, HP, PS, HS, density/H and density/P. Bicubic adds
density/T and density/S. Use the exported mass-based input-pair constants rather
than guessing positional order: for `HmassP_INPUTS`, enthalpy is first and
pressure is second. All ordinary values use SI mass units.

One fluid object can use **different input pairs in the same simulation**. JAX
specializes on a static pair selector and numerical shape/dtype, while state
values and native model coefficients remain dynamic. Compilation does not lock
the object to one thermodynamic pair.

## Full states, broadcasting and property access

```python
import numpy as np

enthalpy = np.linspace(650e3, 750e3, 5)
pressure = np.linspace(6e6, 10e6, 4)
grid = table.get_state(
    jp.HmassP_INPUTS, enthalpy[:, None], pressure[None, :], errors="mask"
)
assert grid.shape == (5, 4)
temperature = grid.raw_property("temperature")
temperature_valid = grid.property_valid("temperature")
```

Inputs broadcast using ordinary NumPy/JAX rules. Two length-N vectors represent
paired points; use explicit singleton axes for a Cartesian grid. Ordinary calls
return the full enabled state of properties. Reading a returned numerical field
does not trigger a new EOS calculation. Canonical names and documented aliases
(`temperature`/`T`, `pressure`/`p`, `density`/`rho`, etc.) resolve consistently.
Raw exported dictionaries use canonical property names.

Guarded property access reports an enabled but undefined quantity through
`PropertyUnavailableError` under raising policy. `raw_property(name)` and
`property_valid(name)` are explicit numerical/availability access for compiled
consumers and plots. A property absent from the enabled schema is unsupported;
it is different from an enabled property that is locally unavailable.

Use `FluidState.stack` or the declared collection operations for compatible
states. Do not combine differing physical models, reference conventions or fixed
compositions as though they were one fluid. Plot numeric comparisons explicitly
when comparing different models. Derivative quantities use `PropertyDerivatives`
with a separate primal state and derivative units; they are not thermodynamic
states to concatenate with ordinary values.

## Explicit errors and clipping

| Request | Result |
| --- | --- |
| Invalid construction, unsupported pair/property/option or malformed arrays | Informative typed error; not suppressed by numerical policy |
| `errors="raise"` | Eager default; a failing element raises the relevant error with its index |
| `errors="mask"` | Failing elements get nonzero status and NaN numerical values; successful elements are preserved |
| `errors="clip"` where supported | A defined finite projection returns a consistent adjusted state and `state.clipped`; unrecoverable elements remain failed |

`state.status` uses the shared values: 0 success, 1 invalid input, 2 outside
domain, 3 nonconvergence, 4 provider failure. `state.valid` is exactly status zero.
An unavailable optional response at a successful point does not change that
status. Clipping is not extrapolation, an unsupported-operation selector, or a
guarantee that every phase-invalid cell/inverse can be recovered.

Replace ambiguous constructor/throwing flags such as `exceptions=False` with the
explicit per-call policy. Source evaluators do not advertise clipping; check
`fluid.capabilities.error_modes`. Known unsupported requests still fail
explicitly, such as quality input or saturation for a perfect gas.

## Compiled application functions

Choose the application's JAX device and precision before constructing models.
Importing JaxProp does not force CPU or install an exception hook. It honors
explicit environment/preloaded JAX settings; JaxProp-first imports retain a
double-precision default when no precision choice was supplied.

```python
import jax

@jax.jit
def evaluate(model, enthalpy, pressure):
    first = model.get_state(
        jp.HmassP_INPUTS, enthalpy, pressure, errors="mask"
    )
    recovered = model.get_state(
        jp.PSmass_INPUTS, pressure, first.raw_property("entropy"), errors="mask"
    )
    return first, recovered
```

For supported evaluators, compile complete consumers with reusable models and
dynamic numerical inputs. Return the complete state or retain status alongside
selected outputs. Eager raising lookups can synchronize a device; the explicit
mask/status contract avoids hidden host checks inside compiled numerical code.
Synchronize complete returns before timing and separate setup, compilation and
warmed evaluation. Source callbacks remain host work even under JIT.

Masked/compiled callback outputs honor active JAX floating precision, while
CoolProp computes in double precision on the host. Callback derivatives use
finite differences and are first-order only. Failed primal rows and unavailable
primal properties contribute zero in forward and reverse mode. A valid primal
whose required forward probe fails retains an unavailable NaN derivative; no
zero slope is invented at that boundary. Native model derivatives follow their
declared domain and availability.

## Fixed-composition binary models

```python
water = jp.Fluid("Water", method="coolprop")
nitrogen = jp.Fluid("Nitrogen", method="perfect_gas")
mixture = jp.Fluid(
    "water_nitrogen",
    method="binary_mixture",
    model_options={
        "components": (water, nitrogen),
        "mass_fractions": (0.2, 0.8),
    },
)
state = mixture.get_state(jp.PT_INPUTS, 1e6, 500.0)
```

The fractions are nonnegative scalars summing to one and remain fixed for that
object. They are not vapor quality and are not a fourth positional state input.
Construct explicit models for a composition study. Component providers and
backends belong to the component objects; a composite backend override is an
error. `FluidMix` and `FluidTwoComponent` compatibility constructors resolve to
the consolidated evaluator rather than duplicate thermodynamic kernels.

This model has additive specific volume and mass-weighted h/u/s at common p/T,
without molecular mixing entropy or mass exchange. It is not a chemical-mixture
EOS. The canonical sound speed is the pressure/temperature-equilibrium response;
`speed_of_sound_p` is the distinct thermally frozen Wood limit. Use the acoustic
model matching the study's relaxation assumptions.

Active PG/bicubic components support native JAX and applicable clipping.
All-active-source callback components support first-order compiled host
transactions with mask policy. Direct-source/native-callback hybrid
configurations remain eager and reject unsupported transformations. Zero-weight
components do not add numerical source calls. A binary parcel model has no single
inferred saturation curve or vapor-quality field.

## References, pseudo-pure sources and route limits

Every configured fluid exposes `critical_point`, `triple_point_liquid` and
`triple_point_vapor` records through `fluid.references`. Single-source models
evaluate these from the named CoolProp source during setup, even for a perfect-gas
approximation. Binary parcel models expose explicitly unavailable aggregate
points and retain their component source references. Inspect availability and
interpretation before using a point:

```python
point = source.references["critical_point"]
if point.available:
    pressure = point.properties["pressure"]
else:
    print(point.reason)
```

Reported coordinates, actual evaluation coordinates and offsets are retained.
Critical response singularities are explicitly unavailable. A pseudo-pure
source's lower saturation endpoints are not automatically a pure-fluid triple
point; records explain this interpretation. An unavailable reference does not
by itself prevent ordinary valid gas/liquid calculations. Explicit unavailable
state access such as `fluid.triple_point_vapor` raises
`UnavailableReferencePointError` instead of fabricating data.

Concrete locally validated route limits include:

- **HEOS Air at 100 K and HEOS R410A at 270 K:** QT accepts quality endpoints 0
  and 1. A fractional QT request at Q=0.3 raises `OutOfDomainError`; a masked
  Q=[0, 0.3, 1] batch has statuses [0, 2, 0].
- **HEOS Air/R410A at 1 MPa:** PQ at Q=0.3 is supported and preserves the actual
  flashed mass quality. Geometry and response calculations follow that source
  flash map, with actual phase densities.
- **REFPROP where installed:** its supported fractional QT routes are not
  subjected to the HEOS endpoint-only rule. The actual source still decides
  which phase/state requests exist.
- For pseudo-pure interiors, a PQ state's density/temperature may not reproduce
  the same pressure through HEOS DmassT; REFPROP can reject that interior route.
  Equal coordinate labels do not establish equivalent backend flash geometry.
  Validate the route used by the application and do not force inconsistent
  flashes to agree.

Pure-fluid interior equilibrium $c_p$ and other singular responses are also
unavailable under their definitions. Do not restore a finite mass-average
surrogate solely to mimic older results. The signed saturation-coordinate
derivative examples demonstrate the correct independent constraints and units.

## Tables and formal homogeneous studies

Cache reuse checks a complete generation manifest, source/reference/native-file
identity, numerical schema, grids and finite core arrays. A corrupt/incompatible
cache regenerates; failed generation/publication preserves the previous file.
Matching metadata is not an interpolation-accuracy certificate. Generation
checks stay outside repeated native state calls.

Physical metastable tables require positive sampled $c_v$ and $K_T$, including
derivative stencils. Explicit `metastable_domain="formal"` retains calorically
unstable EOS research states while retaining mechanical guards. Construction-time
polynomial bounds and optional masks prevent interpolation across identified
heat-capacity poles. They bound the interpolants, not the exact unsampled EOS.
The broad formal Nitrogen studies still show large errors near the excluded
pole; successful execution or a selected-point inverse is not whole-domain
response accuracy acceptance.

Advanced source inverses retain independent final-root checks. Their pressure
acceptance includes a narrowly scoped stiffness-scaled floating-roundoff floor
for low-net-pressure stiff liquids. Other residual limits are unchanged, and the
final check validates the exact returned properties without a second source
flash. This does not relax the source domain or add per-iteration checks.

## Workers, serialization and plotting

Use a configured evaluator pickle snapshot for process workers. Native
parameters/tables and shared binary component aliases are preserved; source
handles, helpers and locks are reconstructed under verified matching source
reference/configuration/native-definition/file identity. Restoration never
changes global CoolProp reference settings. A mismatching or unverifiable source
reports an explicit configuration error. Numerical curve caches survive, while
Matplotlib artists are transient. Pin the numerical environment/library revision
when comparing snapshots and results.

`fluid.configuration` is inspection metadata, not a universal factory/JSON
reconstruction recipe. PG `with_constants(...)` and `materialize()` provide
validated eager native model updates; low-level differentiated views retain
their original static identity until materialized.

`jp.plot_phase_diagram(fluid, ...)` provides common eager **source-EOS** plots
for direct, callback and real-fluid bicubic models. It may extend outside a
table's interpolation domain. PG/binary models explicitly reject their own phase
envelopes; plot appropriate component sources separately.

See [the lifecycle guide](fluid_lifecycle.md) for complete thread/worker,
precision, compiled-consumer and plotting recipes. Runnable examples are
indexed in the repository's
[example workflows](https://github.com/turbo-sim/jaxprop/tree/main/examples).
The [full interface contract](https://github.com/turbo-sim/jaxprop/blob/main/z_PUBLIC_FLUID_CONTRACT.md)
and [implementation checklist](https://github.com/turbo-sim/jaxprop/blob/main/z_IMPLEMENTATION_PLAN.md)
retain detailed model formulas, verification scope and deferred work.
