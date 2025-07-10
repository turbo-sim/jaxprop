# CoolPropX


``CoolPropX`` is a thin wrapper around the [CoolProp](https://www.coolprop.org) fluid property library that provides easy access to its low-level interface, along with several convenient utilities to:

- Compute and plot phase envelopes and spinodal lines for pure fluids.
- Evaluate thermodynamic properties from the Helmholtz energy equation of state, including metastable states inside the two-phase region.
- Perform flash calculations for any input pair with a custom solver with user-defined initial guesses 
- Accessing structured property dictionaries and immutable `FluidState` objects
- Evaluate properties over arrays of input conditions for efficient parametric studies and plotting.

**Note**: This project is developed independently and is not affiliated with or endorsed by the CoolProp project.



## Installation

```bash
pip install coolpropx
