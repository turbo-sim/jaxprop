# Highlight exception messages
# https://stackoverflow.com/questions/25109105/how-to-colorize-the-output-of-python-errors-in-the-gnome-terminal/52797444#52797444
try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=False)


# print("I am fixing the error")

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)



from .graphics import *
from .utils import *
from .helpers_jax import *
from .helpers_props import *

# Import subpackages
from . import coolprop
from . import perfect_gas
from . import bicubic

# Import API classes
from .perfect_gas import FluidPerfectGas
from .coolprop import Fluid, FluidJAX
from .bicubic import FluidBicubic


from . import components


# Package info
__version__ = "0.4.2"
PACKAGE_NAME = "jaxprop"
URL_GITHUB = "https://github.com/turbo-sim/jaxprop"
URL_DOCS = "https://turbo-sim.github.io/jaxprop/"
URL_PYPI = "https://pypi.org/project/jaxprop/"
URL_DTU = "https://thermalpower.dtu.dk/"
BREAKLINE = 80 * "-"

