# JAXprop

JAX-compatible Python package for real-fluid thermodynamic properties, phase equilibria and metastable states, with automatic differentiation.

🔗 **Docs**: [turbo-sim.github.io/jaxprop](https://turbo-sim.github.io/jaxprop/)  
📦 **PyPI**: [pypi.org/project/jaxprop](https://pypi.org/project/jaxprop/)

**Note**: This project is based on the [CoolProp](https://www.coolprop.org) library but is not affiliated with or endorsed by the CoolProp project.

## Key features

- Compute and plot phase envelopes and spinodal lines for pure fluids.
- Evaluate thermodynamic properties from Helmholtz energy–based equations of state, including metastable states inside the two-phase region.
- Perform flash calculations for any input pair with a custom solver and user-defined initial guesses.
- Work with structured property dictionaries and immutable `FluidState` objects.
- Evaluate properties over arrays of input conditions for efficient parametric studies and plotting.
- Use JAX-enabled evaluators with `jit`, `grad` and `vmap`, according to each model's declared capabilities.

## Installation

```bash
pip install jaxprop
```

## Uniform fluid constructor (0.6.0)

This checkout prepares version **0.6.0**. Publication is pending downstream
verification in ThermOpt and Barotropy; `pip install jaxprop` may still install
the earlier interface. See the [0.6.0 release notes](docs/release_notes.md) and
[migration guide](docs/migration.md) before upgrading.

```python
import jaxprop as jxp

fluid = jxp.Fluid("CO2", method="coolprop", backend="HEOS")
state = fluid.get_state(jxp.PT_INPUTS, 10e6, 400.0)
print(state.density)

callback = jxp.Fluid("CO2", method="coolprop_jax", backend="HEOS")
gas = jxp.Fluid("Air", method="perfect_gas")
table = jxp.Fluid(
    "CO2",
    method="bicubic",
    backend="HEOS",
    table_options={
        "h_min": 600e3, "h_max": 850e3,
        "p_min": 5e6, "p_max": 12e6,
        "N_h": 16, "N_p": 16,
    },
)
```

`method` selects the evaluator or approximate physical model. `backend` selects
the CoolProp source used for evaluation, table generation, calibration and
reference data; it defaults to HEOS. Inspect `fluid.capabilities` before using
phase operations, diagnostics or JAX transformations.

CoolProp and its callback accept `errors="raise"` (default) or `errors="mask"`.
Mask mode preserves successful batch elements and exposes failed elements as
NaNs with integer `state.status` codes. Transformed callback calls require it:

```python
import jax

temperature = jax.jit(
    lambda h: callback.get_state(jxp.HmassP_INPUTS, h, 8e6, errors="mask").T
)
```

The callback supports first-order finite-difference derivatives. Perfect gas
and bicubic tables support `raise`, `mask` and `clip`; their transformed calls
require explicit mask or clip mode. For example:

```python
adjusted = gas.get_state(jxp.PT_INPUTS, -1.0, 300.0, errors="clip")
print(adjusted.p, adjusted.clipped, adjusted.status)
```

Clipping returns a consistent adjusted state with a clipping flag. Nonfinite
inputs remain failed elements. Perfect gas now preserves calibrated source
enthalpy/entropy and heat capacity with consistent ideal-gas equations, enabling
22 properties. Gradient helpers return `PropertyDerivatives` with SI units and
a separate `.primal` state. Bicubic clipping projects onto the configured
enthalpy/pressure rectangle. Fixed-coordinate inverses can also project a target
when the table proves a monotone response; unsuccessful or ambiguous inverses
remain failed elements. Inverse budgets are configured once with the
`inverse_max_steps` and `inverse_tolerance` table options. Inspect capabilities
and the [public contract](z_PUBLIC_FLUID_CONTRACT.md) or
[implementation checklist](z_IMPLEMENTATION_PLAN.md) for current support.

### Binary mixtures

Configure fixed mass fractions once, then use the same three-argument state call:

```python
nitrogen = jxp.Fluid("Nitrogen", method="perfect_gas")
oxygen = jxp.Fluid("Oxygen", method="perfect_gas")
mixture = jxp.Fluid(
    "working_mix",
    method="binary_mixture",
    model_options={
        "components": (nitrogen, oxygen),
        "mass_fractions": (0.4, 0.6),
    },
)
state = mixture.get_state(jxp.PT_INPUTS, 1e6, 500.0)
```

The model supports `PT`, `HmassP` and `PSmass`, returning 27 fields. It represents
constituent parcels at common pressure and temperature without mass exchange or
molecular mixing entropy. Heat capacities and compressibilities follow that
model's thermodynamic identities. `speed_of_sound` and `speed_of_sound_pT` are
the pressure/temperature equilibrium speed; `speed_of_sound_p` is the distinct
thermally frozen Wood speed. Composition is not vapor quality; composite quality
and saturation requests are explicitly unsupported.

Active perfect-gas/bicubic components use native JAX with explicit mask/clip
inside transformations. All-active `FluidJAX` components use one whole-mixture
callback with mask mode and first-order state-input derivatives. Direct-source
and source/native/callback hybrids use eager raise/mask; compiled hybrid support
remains deferred. Host configurations are fixed component snapshots; native
component numerical parameters remain dynamic. Inspect `mixture.capabilities`.

`FluidMix` translates the legacy name/mass-ratio constructor once and returns
the same `FluidBinaryMixture` class; `FluidTwoComponent` is an alias of that class.
Optional inverse controls belong in `model_options`. See the [binary contract](z_PUBLIC_FLUID_CONTRACT.md#fixed-composition-binary-model)
and the [water/nitrogen example](examples/two_component/water_nitrogen_example.py).
See the [verified Phase 9 checklist](z_IMPLEMENTATION_PLAN.md#phase-9-consolidate-the-two-component-mixture-implementations)
and [full-state timing report](development/performance/README.md#phase-9-consolidated-binary-model).

For application-owned JAX settings, complete compiled consumers, worker
serialization, source-EOS plotting and physical/formal metastable tables, see
the [fluid lifecycle guide](docs/fluid_lifecycle.md). Configuration snapshots
are inspection records; verified evaluator snapshots preserve the full model.

For interface changes and reproducible example commands, see the
[migration guide](docs/migration.md) and [example workflows](examples/README.md).

<!-- research-software-metadata:start -->
## Development and citation

Roberto Agromayor is the main developer, with contributions from Srinivas Prakash Diwanji, Alberto Bergamin and Andrea Cioffi.

Citation metadata is provided in [CITATION.cff](CITATION.cff). For version-specific reproducibility, also record the release or commit used.
<!-- research-software-metadata:end -->
