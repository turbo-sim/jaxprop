Welcome to jaxprop's documentation!
=====================================

``jaxprop`` provides JAX-compatible thermodynamic property calculations with support for automatic differentiation, vectorization, and JIT compilation.  

.. note::

   This project is developed independently and is not affiliated with or endorsed by the CoolProp project.

Key features
------------

- Construct direct-source, callback, bicubic, perfect-gas and binary-parcel models through ``Fluid(..., method=...)``.
- Return complete immutable ``FluidState`` values with explicit status and property availability.
- Reuse native numerical models in supported JIT, vectorized and first-order differentiated consumers.
- Evaluate homogeneous/metastable and equilibrium source states under their declared physical domains.
- Inspect source reference records, validated table provenance and explicitly supported worker reconstruction.
- Draw source-EOS phase diagrams and run documented quick/full scientific example profiles.


Contents
------------


Use the panel to the left or the table of contents below to navigate the documentation.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   source/installation
   migration
   fluid_lifecycle
   source/bibliography
   source/api/jaxprop
