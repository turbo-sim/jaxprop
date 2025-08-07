# Import JAX in a flexible way
try:
    import os
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    import numpy as np
    jax = None
    jnp = np
    JAX_AVAILABLE = False

import logging
logger = logging.getLogger("jax")
logger.setLevel(logging.WARNING)

__all__ = ["jax", "jnp", "JAX_AVAILABLE"]