Welcome to jaxprop's documentation!
=====================================

``jaxprop`` provides JAX-compatible thermodynamic property calculations with support for automatic differentiation, vectorization, and JIT compilation.  

.. note::

   This project is developed independently and is not affiliated with or endorsed by the CoolProp project.

Key features
------------

- Compute and plot phase envelopes and spinodal lines for pure fluids.
- Evaluate thermodynamic properties from the Helmholtz energy equation of state, including metastable states inside the two-phase region.
- Perform flash calculations for any input pair with a custom solver with user-defined initial guesses.
- Access structured property dictionaries and immutable ``FluidState`` objects.
- Evaluate properties over arrays of input conditions for efficient parametric studies and plotting.


Contents
------------


Use the panel to the left or the table of contents below to navigate the documentation.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   source/installation
   source/bibliography
   source/api/coolpropx
