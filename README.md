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

<!-- research-software-metadata:start -->
## Development and citation

Roberto Agromayor is the main developer, with contributions from Srinivas Prakash Diwanji, Alberto Bergamin and Andrea Cioffi.

Citation metadata is provided in [CITATION.cff](CITATION.cff). For version-specific reproducibility, also record the release or commit used.
<!-- research-software-metadata:end -->
