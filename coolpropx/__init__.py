
# Highlight exception messages
# https://stackoverflow.com/questions/25109105/how-to-colorize-the-output-of-python-errors-in-the-gnome-terminal/52797444#52797444
try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.FormattedTB(color_scheme='linux', call_pdb=False)


from .core_calculations import *
from .fluid_properties import *
from .graphics import *

# Package info
__version__ = "0.1.1"
PACKAGE_NAME = "thermopt"
URL_GITHUB = "https://github.com/turbo-sim/coolprop-x"
URL_DOCS = "https://turbo-sim.github.io/coolprop-x/"
URL_DTU = "https://thermalpower.dtu.dk/"
BREAKLINE = 80 * "-"

