.. _installation:

Installation
============

JaxProp supports Python 3.11 through 3.13. Create a dedicated environment before
installing the package:

.. code-block:: bash

   python -m venv .venv

Activate it with ``.venv\Scripts\Activate.ps1`` on Windows PowerShell or
``source .venv/bin/activate`` on Linux/macOS, then install the published package:

.. code-block:: bash

   python -m pip install jaxprop
   python -c "import jaxprop; print(jaxprop.__version__)"

Version 0.6.0 is prepared locally and awaits ThermOpt and Barotropy verification
before publication. To test the prepared artifact before that release, install
its wheel explicitly from the JaxProp checkout:

.. code-block:: bash

   python -m pip install dist/jaxprop-0.6.0-py3-none-any.whl

Run import checks outside the source checkout and inspect ``jaxprop.__file__``
to confirm that the installed distribution is being tested. See the
:doc:`migration guide </migration>` for the 0.6.0 interface changes and
:doc:`release notes </release_notes>` for verification scope and known costs.

Backends and JAX configuration
------------------------------

CoolProp's HEOS backend is available through the standard dependencies.
REFPROP requires a separate licensed installation and its native library/data
configuration; it is not included in the JaxProp wheel. Select an available
backend when constructing a fluid, for example
``Fluid("CO2", method="coolprop", backend="HEOS")``.

Device and precision settings belong to the consuming application. Set
``JAX_ENABLE_X64=true`` and, for the verified CPU profile,
``JAX_PLATFORM_NAME=cpu`` before importing JAX. JaxProp respects existing JAX
configuration. See :doc:`the lifecycle guide </fluid_lifecycle>` for complete
compiled-consumer and worker examples.

Development installation
------------------------

Use Poetry 2.1 or later with this repository's lock file:

.. code-block:: bash

   git clone https://github.com/turbo-sim/jaxprop.git
   cd jaxprop
   poetry install
   poetry run python -m pytest tests -c tests/pytest.ini -q

The scientific test profile enables double precision. Select optional source
backends explicitly with ``JAXPROP_TEST_BACKENDS=HEOS,REFPROP`` only where
REFPROP is installed. For example execution requirements and bounded/full
profiles, see the repository's ``examples/README.md``.
