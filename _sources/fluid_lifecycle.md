# Reusing fluid models in applications

Construct a configured fluid once and reuse it for repeated state calls. The
common `Fluid(...)` initializer returns the actual evaluator class. Configuration,
source references, capability records and table validity are prepared at setup;
they are not rediscovered during native state evaluation.

## Application-owned JAX settings

Set the device and numerical precision before constructing models. For example:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jaxprop as jp

fluid = jp.Fluid("Air", method="perfect_gas")
```

JaxProp does not select a JAX platform or replace `sys.excepthook`. It honors
`JAX_ENABLE_X64` and a JAX configuration already loaded by the application. When
JaxProp imports JAX first and no precision setting is supplied, double precision
remains the default. CPU and accelerator selection use JAX's own environment and
configuration settings. Accelerator performance has not been inferred from the
CPU measurements retained in this repository.

Direct CoolProp calculations use the source's double precision. Masked/compiled
callback paths compute in the same host precision and return the active JAX
floating precision; their declared callback shapes/dtypes agree when double
precision is disabled. Eager raising binary source calls retain NumPy double
precision.
Choose solver tolerances appropriate to the model's numerical precision.

## Compiled consumers and failure status

Native perfect-gas, bicubic and supported native binary models can be arguments
of compiled application functions. Their numerical coefficients remain dynamic.
The input-pair selector and numerical policy are static choices; different pairs
can be used in the same simulation and are compiled as their respective calls.

```python
import jax
import jaxprop as jp

@jax.jit
def recover_state(fluid, enthalpy, pressure):
    return fluid.get_state(
        jp.HmassP_INPUTS, enthalpy, pressure, errors="mask"
    )

state = recover_state(fluid, enthalpy, pressure)
```

The complete returned state retains its properties, status and property
availability. A consumer returning selected numerical outputs should also return
`state.status`, and clipping flags when that policy is relevant. Required status
checks then belong at the application's chosen boundary. Eager raising property
access inspects concrete availability and can synchronize a device; use
`raw_property(...)` and the explicit status/availability contract inside compiled
numerical code. Source callbacks remain host operations even inside `jax.jit`.

Failed primal rows and unavailable primal optional properties contribute zero
callback derivatives in both forward and reverse mode. A valid state whose
required forward finite-difference probe fails retains an unavailable NaN
derivative; it does not acquire an invented zero slope at that boundary.

Synchronize numerical outputs before measuring JAX work. Separate construction,
table generation/loading, compilation and warmed full-state execution. The
repeatable checkpoint runner is
`development/performance/phase11_benchmark.py`; the published-release comparison
belongs to Phase 13.

## Independent states, threads and workers

Returned states are independent snapshots. Subsequent source flashes do not
change their values. Native numerical evaluators use immutable model data and
can be reused by compiled consumers. Direct-source objects and callbacks protect
their shared primary/scratch handles with an object lock for the complete source
transaction. This makes shared-source evaluation safe but serializes competing
host transactions. Source binaries use a stable ordering of component locks.

For process workers, reconstruct a configured evaluator from a pickle snapshot:

```python
import pickle

snapshot = pickle.dumps(fluid, protocol=pickle.HIGHEST_PROTOCOL)
restored = pickle.loads(snapshot)
state = restored.get_state(jp.HmassP_INPUTS, enthalpy, pressure)
```

Native snapshots retain the actual coefficients/tables without initializing a
source EOS. Nested binary snapshots retain their configured component models,
including shared aliases. Source snapshots rebuild all native handles, helper
objects and locks; they verify the original energy-reference convention, backend
configuration and native definitions/files before accepting the restore. A
mismatching environment raises an informative configuration error. Restoration
never changes process-global source reference settings. Backends whose native
identity cannot be verified remain usable for evaluation but explicitly reject
lossless source serialization.

`fluid.configuration` is an inspection record, not a general lossless factory
recipe. It can contain actual calibrated constants or nested component records
that differ from constructor arguments. Pickle is the complete reconstruction
recipe for configured evaluators. For a perfect-gas model, `with_constants(...)`
and `materialize()` provide validated eager model updates/reconstruction. An
arbitrary numerical view used for differentiation does not automatically refresh
its static physical identity or certify a changed model's physical domain.

## Phase/reference plotting

```python
figure, axes = jp.plot_phase_diagram(
    real_fluid, x_prop="enthalpy", y_prop="pressure", y_scale="log"
)
```

This common eager function plots the configured **source EOS** for direct-source,
callback and real-fluid bicubic objects. A bicubic source diagram can extend
beyond the table's numerical domain; it does not certify interpolation there.
The plotting-only source must match the table's reference and provenance.
Perfect-gas and binary parcel models explicitly reject their own phase diagram;
their available source-reference records can be inspected separately.

Repeated calls reuse/update the fluid's artists on the supplied axes. Numerical
curve caches include sampling and solver settings, and named streams keep
independent artists. Generated labels and limits refresh with the plotted
properties while caller overrides are preserved. Pickle retains numerical curves
and discards transient Matplotlib artists; a restored model creates fresh artists.

## Physical and formal metastable tables

The default `metastable_domain="physical"` requires positive constant-volume heat
capacity $c_v$ and isothermal bulk modulus $K_T$ at every homogeneous source sample,
including finite-difference stencil points. Failed construction does not publish
a partial table or overwrite an existing valid cache. A requested rectangle that
extends beyond these bounds must be deliberately redesigned.

For an explicitly formal homogeneous-EOS study, use
`metastable_domain="formal"` together with `metastable_phase="liquid"` or
`"vapor"` in `Fluid(..., method="bicubic", table_options=...)`. Formal mode retains
calorically unstable states; it still requires positive sampled $K_T$. The raw
homogeneous source solver remains available for scientific spinodal grids,
including mechanically unstable states; that is a different operation from
constructing a usable metastable interpolation table.

Construction converts bicubic power coefficients to Bernstein control values to
bound each interpolation polynomial over its entire cell. Physical-mode cells
must have strictly positive $c_v$ and $K_T$ lower bounds; formal-mode cells retain
the mechanical bound. Cells that may cross $c_v=0$ do not expose Grüneisen,
heat-capacity ratio, isentropic bulk modulus or speed of sound. A possible
$c_p=0$ crossing similarly disables Joule–Thomson and isentropic compressibility.
The original node values remain available for inspection. Disabled responses
use the existing optional-property masks and do not invalidate otherwise valid
formal core states.

These are conservative bounds on the interpolant, not a proof of the exact EOS
between samples or of every independently interpolated thermodynamic identity.
Positive sampled response values alone do not establish whole-domain accuracy.
The policy and domain are part of the generation manifest, cached guards are
validated, and runtime evaluation reuses existing masks without source queries
or new stability arithmetic.

## Verification profile

The scientific suite explicitly selects double precision in `tests/conftest.py`;
float32 and import-setting contracts have dedicated tests. A reproducible full
CPU run with both locally installed source backends is:

```powershell
$env:JAX_ENABLE_X64 = "true"
$env:JAX_PLATFORM_NAME = "cpu"
$env:JAXPROP_TEST_BACKENDS = "HEOS,REFPROP"
python -m pytest tests -c tests/pytest.ini -q
```

REFPROP cases are optional and reported as skipped when that backend is absent.
The retained Phase 11 verification record states the actual backends, precision,
source hash, numerical comparison counts and outcomes.
