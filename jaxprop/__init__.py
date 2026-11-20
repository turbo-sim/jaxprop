"""Thermodynamic models with consumer-controlled JAX execution settings.

When JaxProp imports JAX first, double precision remains the default for
scientific calculations. An explicit ``JAX_ENABLE_X64`` environment setting,
or an application that imports/configures JAX before JaxProp, takes priority.
Choose devices through JAX's own configuration before constructing models;
importing this package does not select a platform or install an exception hook.
"""

import os as _os
import sys as _sys

_configure_default_precision = (
    "jax" not in _sys.modules and "JAX_ENABLE_X64" not in _os.environ
)
import jax

if _configure_default_precision:
    jax.config.update("jax_enable_x64", True)
del _configure_default_precision

from .graphics import *
from .utils import *
from .helpers_jax import *
from .helpers_props import *
from .errors import *
from .configuration import *
from .source_references import *
from .fluid_api import StatusCode
from .factory import create_fluid

# Import subpackages
from . import coolprop
from . import perfect_gas
from . import bicubic

# Import API classes
from .perfect_gas import FluidPerfectGas
from .coolprop import FluidCoolProp, FluidJAX
from .bicubic import FluidBicubic
from .binary_mixture import FluidBinaryMixture, FluidMix, FluidTwoComponent
from .fluid import Fluid
from .property_derivatives import PropertyDerivatives
from .plotting import plot_phase_diagram

# Virtual registration leaves concrete evaluator inheritance and numerical
# PyTrees unchanged, while exposing one runtime type for consuming packages.
for _fluid_type in (
    FluidCoolProp,
    FluidJAX,
    FluidBicubic,
    FluidPerfectGas,
    FluidBinaryMixture,
):
    Fluid.register(_fluid_type)
del _fluid_type


# Package info
__version__ = "0.6.2"
PACKAGE_NAME = "jaxprop"
URL_GITHUB = "https://github.com/turbo-sim/jaxprop"
URL_DOCS = "https://turbo-sim.github.io/jaxprop/"
URL_PYPI = "https://pypi.org/project/jaxprop/"
URL_DTU = "https://thermalpower.dtu.dk/"
BREAKLINE = 80 * "-"
