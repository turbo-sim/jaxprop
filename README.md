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
- Full JAX compatibility: supports `jit`, `grad`, `vmap`, and parallel evaluation.

## Installation

```bash
pip install jaxprop
```

## Development constructor

The following API is available in this development checkout. The published
0.5.8 release uses the earlier construction interface.

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
supports `raise`, `mask` and `clip`; its transformed calls require explicit
mask or clip mode. For example:

```python
adjusted = gas.get_state(jxp.PT_INPUTS, -1.0, 300.0, errors="clip")
print(adjusted.p, adjusted.clipped, adjusted.status)
```

Clipping returns a consistent adjusted state with a clipping flag. Nonfinite
inputs remain failed elements. Perfect gas now preserves calibrated source
enthalpy/entropy and heat capacity with consistent ideal-gas equations, enabling
22 properties. Gradient helpers return `PropertyDerivatives` with SI units and
a separate `.primal` state. Bicubic numerical policies remain Phase 8 work;
inspect capabilities and the [public contract](z_PUBLIC_FLUID_CONTRACT.md) or
[implementation checklist](z_IMPLEMENTATION_PLAN.md) for current support.

<!-- research-software-metadata:start -->
## Development and citation

Roberto Agromayor is the main developer, with contributions from Srinivas Prakash Diwanji, Alberto Bergamin and Andrea Cioffi.

Citation metadata is provided in [CITATION.cff](CITATION.cff). For version-specific reproducibility, also record the release or commit used.
<!-- research-software-metadata:end -->
