"""Shared request validation without heavyweight public method wrappers."""

from enum import IntEnum
from functools import lru_cache, partial, update_wrapper
from numbers import Integral, Real

import CoolProp.CoolProp as CP
import jax
import jax.numpy as jnp
import numpy as np

from .errors import (
    InvalidStateError,
    UnsupportedExecutionModeError,
    UnsupportedInputPairError,
    UnsupportedOptionError,
)
from .property_schema import PROPERTIES_CANONICAL

__all__ = ["StatusCode"]


class StatusCode(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    OUT_OF_DOMAIN = 2
    NONCONVERGENCE = 3
    PROVIDER_FAILURE = 4


PAIR_NAMES = {
    int(getattr(CP, name)): name
    for name in dir(CP)
    if name.endswith("_INPUTS") and isinstance(getattr(CP, name), Integral)
}
SOURCE_PAIRS = frozenset(
    getattr(CP, name)
    for name in (
        "PT_INPUTS",
        "HmassSmass_INPUTS",
        "HmassP_INPUTS",
        "PSmass_INPUTS",
        "DmassHmass_INPUTS",
        "DmassP_INPUTS",
        "DmassT_INPUTS",
        "DmassSmass_INPUTS",
        "PQ_INPUTS",
        "QT_INPUTS",
        "SmassT_INPUTS",
        "DmassUmass_INPUTS",
    )
)
DIAGNOSTIC_PROPERTIES = frozenset(
    {
        "subcooling",
        "superheating",
        "pressure_saturation",
        "temperature_saturation",
        "supersaturation_degree",
        "supersaturation_ratio",
    }
)
MIXTURE_PROPERTIES = frozenset(
    {
        "mixture_ratio",
        "mass_frac_1",
        "mass_frac_2",
        "vol_frac_1",
        "vol_frac_2",
    }
)
SOURCE_PROPERTIES = frozenset(PROPERTIES_CANONICAL) - (
    DIAGNOSTIC_PROPERTIES | MIXTURE_PROPERTIES | {"drho_dp", "drho_dh"}
)
LEGACY_MIXTURE_PROPERTIES = MIXTURE_PROPERTIES | frozenset(
    {
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "internal_energy",
        "entropy",
        "isochoric_heat_capacity",
        "isobaric_heat_capacity",
        "isothermal_compressibility",
        "isobaric_expansion_coefficient",
        "isothermal_bulk_modulus",
        "isentropic_compressibility",
        "isentropic_bulk_modulus",
        "speed_of_sound",
        "conductivity",
        "viscosity",
        "gruneisen",
    }
)


class ReadonlyConfigurationMixin:
    """Protect configuration identity while allowing source-state/plot updates."""

    _configuration_fields = frozenset(
        {
            "name",
            "backend",
            "provider",
            "model",
            "identifier",
            "reference_id",
            "_references",
        }
    )

    def __setattr__(self, name, value):
        if name in self._configuration_fields and (
            "_configuration" in self.__dict__ and name in self.__dict__
        ):
            raise AttributeError(f"Fluid configuration field {name!r} is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._configuration_fields and "_configuration" in self.__dict__:
            raise AttributeError(f"Fluid configuration field {name!r} is immutable")
        object.__delattr__(self, name)

    @property
    def configuration(self):
        return self._configuration

    @property
    def capabilities(self):
        return self._capabilities


def _context(fluid):
    return dict(
        fluid_name=fluid.name,
        provider=fluid.provider,
        model=fluid.model,
    )


def enabled_properties(fluid, *, supersaturation=False):
    properties = fluid.capabilities.properties
    return properties if supersaturation else properties - DIAGNOSTIC_PROPERTIES


def _pair_choices(capabilities):
    return tuple(PAIR_NAMES[pair] for pair in sorted(capabilities.input_pairs))


@lru_cache(maxsize=64)
def _is_real_dtype(dtype):
    # Dtype classification is immutable and independent of fluid/state values.
    # Reuse it for repeated native array calls; never cache shape/domain checks.
    return jnp.issubdtype(dtype, jnp.number) and not jnp.issubdtype(
        dtype, jnp.complexfloating
    )


def validate_request(
    fluid,
    input_type,
    prop_1,
    prop_2,
    *,
    generalize_quality=False,
    supersaturation=False,
    errors="raise",
    array_backend="numpy",
    options=None,
):
    """Validate static requests and numeric metadata before numerical work.

    Python numeric scalars remain dynamic inputs to native JIT, without eager
    per-call device conversion. Existing arrays/tracers are inspected through
    dtype/shape metadata only. Physical/domain failure policies belong to each
    evaluator; this layer never clips values or claims numerical mask support.
    """
    caps = fluid.capabilities
    if isinstance(input_type, (bool, np.bool_)) or not isinstance(input_type, Integral):
        raise UnsupportedInputPairError(
            request=str(input_type),
            reason="Use a supported static integer input-pair constant.",
            supported=_pair_choices(caps),
            **_context(fluid),
        )
    pair = int(input_type)
    pair_name = PAIR_NAMES.get(pair, str(pair))
    if pair not in caps.input_pairs:
        reason = (
            "Molar input pairs are unsupported; use mass-based SI inputs."
            if "molar" in pair_name.lower()
            else "This evaluator does not support the requested input pair."
        )
        raise UnsupportedInputPairError(
            request=pair_name,
            reason=reason,
            supported=_pair_choices(caps),
            **_context(fluid),
        )
    if options:
        name = next(iter(options))
        raise UnsupportedOptionError(
            request=name,
            reason="Unknown evaluation option.",
            supported=("generalize_quality", "supersaturation", "errors"),
            **_context(fluid),
        )
    for name, value in (
        ("generalize_quality", generalize_quality),
        ("supersaturation", supersaturation),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise UnsupportedOptionError(
                request=name,
                reason="Diagnostic options must be static booleans.",
                supported=(False, True) if name in caps.options else (False,),
                **_context(fluid),
            )
        if value and name not in caps.options:
            raise UnsupportedOptionError(
                request=name,
                reason="This diagnostic is unsupported by this evaluator.",
                supported=tuple(sorted(caps.options)),
                **_context(fluid),
            )
    if not isinstance(errors, str) or errors not in ("raise", "mask", "clip"):
        raise UnsupportedOptionError(
            request="errors",
            reason="Choose 'raise', 'mask' or 'clip'.",
            supported=("raise", "mask", "clip"),
            **_context(fluid),
        )

    def numeric(value):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("Boolean thermodynamic inputs are unsupported.")
        if isinstance(value, Real):
            return value, ()
        if isinstance(value, (jax.Array, jax.core.Tracer, np.ndarray, np.generic)):
            array = value
        else:
            leaves = jax.tree_util.tree_leaves(value)
            array = (
                jnp.asarray(value)
                if any(
                    isinstance(leaf, (jax.Array, jax.core.Tracer)) for leaf in leaves
                )
                else np.asarray(value)
            )
        if not _is_real_dtype(array.dtype):
            raise TypeError(
                "Thermodynamic inputs must contain real numbers, not bool/string/object/complex values."
            )
        return array, array.shape

    try:
        first, first_shape = numeric(prop_1)
        second, second_shape = numeric(prop_2)
        if first_shape != second_shape:
            np.broadcast_shapes(first_shape, second_shape)
    except (TypeError, ValueError) as error:
        raise InvalidStateError(
            request=pair_name,
            reason=str(error),
            **_context(fluid),
        ) from error
    if errors not in caps.error_modes:
        raise UnsupportedExecutionModeError(
            request=f"errors={errors}",
            reason="This evaluator does not implement the requested numerical error policy.",
            supported=tuple(sorted(caps.error_modes)),
            **_context(fluid),
        )
    if array_backend == "numpy":
        if any(isinstance(value, jax.core.Tracer) for value in (first, second)):
            raise UnsupportedExecutionModeError(
                request="JAX transformation",
                reason="This evaluator requires eager source evaluation.",
                supported=("eager",),
                **_context(fluid),
            )
        return np.broadcast_arrays(first, second)
    return first, second


class _CheckedJit:
    """A cheap descriptor: Equinox must not wrap this as an ordinary method."""

    def __init__(self, function):
        update_wrapper(self, function)
        self.compiled = jax.jit(
            function,
            static_argnames=(
                "input_type",
                "generalize_quality",
                "supersaturation",
                "errors",
            ),
        )

    def __get__(self, instance, owner=None):
        return self if instance is None else partial(self, instance)

    def __call__(
        self,
        fluid,
        input_type,
        prop_1,
        prop_2,
        *,
        generalize_quality=False,
        supersaturation=False,
        errors="raise",
        **options,
    ):
        first, second = validate_request(
            fluid,
            input_type,
            prop_1,
            prop_2,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
            errors=errors,
            array_backend="jax",
            options=options,
        )
        return self.compiled(
            fluid,
            input_type,
            first,
            second,
            generalize_quality=bool(generalize_quality),
            supersaturation=bool(supersaturation),
            errors=errors,
        )


def checked_jit(function):
    return _CheckedJit(function)
