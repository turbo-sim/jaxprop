"""Public bicubic evaluator and coefficient-construction utility."""

from .bicubic_interpolation import FluidBicubic, compute_bicubic_coefficients

__all__ = ["FluidBicubic", "compute_bicubic_coefficients"]
