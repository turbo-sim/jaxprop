"""Construction-only compatibility names for the single binary evaluator.

New applications should reuse ``Fluid(method='binary_mixture', ...)`` with
explicit components and mass fractions. No compatibility object wraps calls.
"""

from numbers import Real
import numpy as np

from ..configuration import build_configuration
from ..errors import InvalidConfigurationError
from .fluid import FluidBinaryMixture

FluidTwoComponent = FluidBinaryMixture


def fractions_from_ratio(ratio):
    """Preserve the smaller component when converting an extreme finite ratio."""
    ratio = float(ratio)
    if ratio <= 1:
        first = ratio / (1.0 + ratio)
        return first, 1.0 - first
    second = 1.0 / (1.0 + ratio)
    return 1.0 - second, second


def FluidMix(
    name, backend="HEOS", mixture_ratio=None, exceptions=True, identifier=None
):
    """Translate the former name/mass-ratio constructor once at setup.

    The returned object is the canonical FluidBinaryMixture. Per-call failure
    policies replace ``exceptions=False``; zero ratios are true pure limits.
    """
    context = dict(fluid_name=name, provider="native", model="binary_mixture")
    build_configuration(name, "native", "binary_mixture", backend, identifier)
    if (
        isinstance(mixture_ratio, (bool, np.bool_))
        or not isinstance(mixture_ratio, Real)
        or not np.isfinite(mixture_ratio)
        or mixture_ratio < 0
    ):
        raise InvalidConfigurationError(
            **context,
            request="mixture_ratio",
            reason="Provide a finite nonnegative scalar component mass ratio.",
        )
    if exceptions is not True:
        raise InvalidConfigurationError(
            **context,
            request="exceptions",
            reason="Use the explicit per-call errors='mask' policy instead of exceptions=False.",
        )
    if not isinstance(name, str) or not name.lower().endswith("_mixture"):
        raise InvalidConfigurationError(
            **context,
            request="name",
            reason="Use a legacy name such as water_nitrogen_mixture or explicit component objects.",
        )
    names = name[: -len("_mixture")].split("_")
    if len(names) != 2 or not all(names):
        raise InvalidConfigurationError(
            **context,
            request="name",
            reason="Expected exactly two component names; use explicit objects for compound names.",
        )
    from ..coolprop.fluid_properties import FluidCoolProp

    components = tuple(FluidCoolProp(value, backend=backend) for value in names)
    return FluidBinaryMixture(
        name, components, fractions_from_ratio(mixture_ratio), identifier
    )


def get_mixture_state(fluid_1, fluid_2, p, T, R):
    """Legacy convenience call; reuse a configured binary object in simulations."""
    if (
        isinstance(R, (bool, np.bool_))
        or not isinstance(R, Real)
        or not np.isfinite(R)
        or R < 0
    ):
        raise InvalidConfigurationError(
            request="mixture_ratio",
            reason="Provide a finite nonnegative scalar mass ratio.",
        )
    from ..helpers_props import PT_INPUTS

    fluid = FluidBinaryMixture(
        f"{fluid_1.name}_{fluid_2.name}", (fluid_1, fluid_2), fractions_from_ratio(R)
    )
    return fluid.get_state(PT_INPUTS, p, T)
