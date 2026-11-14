"""Compatibility imports for the consolidated binary fluid model.

The former two-dimensional solver and missing MixtureState schema are replaced
by the common fixed-composition evaluator. Set components and mass_fractions at
construction, then call get_state(input_type, prop_1, prop_2). Model equations,
acoustic assumptions and their derivations live in binary_mixture.physics.
"""

from ..binary_mixture import FluidBinaryMixture, FluidTwoComponent, get_mixture_state

__all__ = ["FluidBinaryMixture", "FluidTwoComponent", "get_mixture_state"]
