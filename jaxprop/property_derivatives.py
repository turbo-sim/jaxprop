"""Property partial derivatives with separate primal-state provenance and units."""

from numbers import Real
from collections.abc import Mapping
import math

import CoolProp.CoolProp as CP
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .configuration import FrozenMapping
from .errors import (
    InvalidConfigurationError,
    InvalidStateError,
    UnsupportedExecutionModeError,
    UnsupportedOptionError,
    UnsupportedPropertyError,
)
from .fluid_api import validate_request
from .fluid_state import FluidState
from .property_schema import ALIAS_TO_CANONICAL, PROPERTY_ALIASES

__all__ = ["PropertyDerivatives", "get_props_gradient"]


_PAIR_INPUT_NAMES = {
    CP.PT_INPUTS: ("pressure", "temperature"),
    CP.HmassSmass_INPUTS: ("enthalpy", "entropy"),
    CP.HmassP_INPUTS: ("enthalpy", "pressure"),
    CP.PSmass_INPUTS: ("pressure", "entropy"),
    CP.DmassHmass_INPUTS: ("density", "enthalpy"),
    CP.DmassP_INPUTS: ("density", "pressure"),
    CP.DmassT_INPUTS: ("density", "temperature"),
    CP.DmassSmass_INPUTS: ("density", "entropy"),
    CP.PQ_INPUTS: ("pressure", "quality_mass"),
    CP.QT_INPUTS: ("quality_mass", "temperature"),
    CP.SmassT_INPUTS: ("entropy", "temperature"),
    CP.DmassUmass_INPUTS: ("density", "internal_energy"),
}
_PROPERTY_UNITS = {
    "pressure": "Pa",
    "temperature": "K",
    "density": "kg/m^3",
    "enthalpy": "J/kg",
    "entropy": "J/(kg K)",
    "internal_energy": "J/kg",
    "isobaric_heat_capacity": "J/(kg K)",
    "isochoric_heat_capacity": "J/(kg K)",
    "heat_capacity_ratio": "1",
    "compressibility_factor": "1",
    "isothermal_compressibility": "1/Pa",
    "isentropic_compressibility": "1/Pa",
    "isothermal_bulk_modulus": "Pa",
    "isentropic_bulk_modulus": "Pa",
    "speed_of_sound": "m/s",
    "viscosity": "Pa s",
    "conductivity": "W/(m K)",
    "gruneisen": "1",
    "isobaric_expansion_coefficient": "1/K",
    "isothermal_joule_thomson": "J/(kg Pa)",
    "joule_thomson": "K/Pa",
    "is_two_phase": "1",
    "quality_mass": "1",
    "quality_volume": "1",
    "surface_tension": "N/m",
    "pressure_saturation": "Pa",
    "temperature_saturation": "K",
    "supersaturation_degree": "K",
    "supersaturation_ratio": "1",
    "subcooling": "K",
    "superheating": "K",
    "mixture_ratio": "1",
    "mass_frac_1": "1",
    "mass_frac_2": "1",
    "vol_frac_1": "1",
    "vol_frac_2": "1",
    "drho_dp": "(kg/m^3)/Pa",
    "drho_dh": "(kg/m^3)/(J/kg)",
}


@jax.jit
def _mask_partial_values(partials, primal_properties):
    # A constant NaN property can have a formal zero AD tangent. It still has
    # no thermodynamic derivative when its primal value is unavailable.
    return {
        name: jnp.where(
            jnp.isfinite(primal_properties[name])[..., None] & jnp.isfinite(value),
            value,
            jnp.nan,
        )
        for name, value in partials.items()
    }


class PropertyDerivatives(eqx.Module):
    """Local two-input partials, distinct from ordinary thermodynamic values.

    ``shape`` is the primal broadcast shape; every derivative array has
    ``derivative_shape == shape + (2,)``. The final axis follows ``input_names``.
    ``primal`` retains the physical state, status, validity and clipping flags.
    """

    primal: FluidState
    _data: dict
    input_names: tuple = eqx.field(static=True)
    input_units: tuple = eqx.field(static=True)
    property_units: FrozenMapping = eqx.field(static=True)
    method: str = eqx.field(static=True)
    quantity_kind: str = eqx.field(static=True, default="property_derivatives")

    def __init__(self, primal, derivatives, input_names, *, method="autodiff"):
        if not isinstance(primal, FluidState):
            raise TypeError("PropertyDerivatives requires a primal FluidState")
        self.primal = primal
        context = dict(
            fluid_name=primal.fluid_name, provider=primal.provider, model=primal.model
        )
        if (
            not isinstance(input_names, (tuple, list))
            or len(input_names) != 2
            or any(not isinstance(name, str) for name in input_names)
        ):
            raise InvalidConfigurationError(
                request="input_names",
                reason="Provide two canonical thermodynamic input names.",
                **context,
            )
        names = tuple(ALIAS_TO_CANONICAL.get(name) for name in input_names)
        if any(name not in _PROPERTY_UNITS for name in names):
            raise InvalidConfigurationError(
                request="input_names",
                reason="Input names require known mass-based SI properties.",
                **context,
            )
        self.input_names = names
        self.input_units = tuple(_PROPERTY_UNITS[name] for name in names)
        if not isinstance(method, str) or method not in (
            "autodiff",
            "finite_difference",
        ):
            raise InvalidConfigurationError(
                request="method",
                reason="Derivative metadata must identify the calculation method.",
                supported=("autodiff", "finite_difference"),
                **context,
            )
        if not isinstance(derivatives, Mapping):
            raise InvalidConfigurationError(
                request="derivatives",
                reason="Provide a canonical-property derivative mapping.",
                **context,
            )
        self.method = method
        self.quantity_kind = "property_derivatives"
        arrays = {}
        for name, value in derivatives.items():
            canonical = ALIAS_TO_CANONICAL.get(name)
            if canonical not in primal.available_properties:
                raise UnsupportedPropertyError(
                    request=name,
                    reason="Derivative properties must be enabled in the primal state.",
                    supported=sorted(primal.available_properties),
                    **context,
                )
            if canonical in arrays:
                raise InvalidConfigurationError(
                    request=canonical,
                    reason="A derivative was supplied more than once through aliases.",
                    **context,
                )
            try:
                array = jnp.asarray(value)
            except (TypeError, ValueError) as error:
                raise InvalidStateError(
                    request=canonical,
                    reason="Property derivatives must contain real floating arrays.",
                    **context,
                ) from error
            if not jnp.issubdtype(array.dtype, jnp.floating):
                raise InvalidStateError(
                    request=canonical,
                    reason="Property derivatives must contain real floating arrays.",
                    **context,
                )
            if array.shape != primal.shape + (2,):
                raise InvalidStateError(
                    request="shape",
                    reason="Derivative arrays must have primal.shape + (2,) axes.",
                    **context,
                )
            arrays[canonical] = array
        self._data = _mask_partial_values(arrays, primal.to_dict())
        self.property_units = FrozenMapping(
            {name: _PROPERTY_UNITS[name] for name in arrays}
        )

    @property
    def shape(self):
        return self.primal.shape

    @property
    def primal_shape(self):
        return self.primal.shape

    @property
    def derivative_shape(self):
        return self.primal.shape + (2,)

    @property
    def fluid_name(self):
        return self.primal.fluid_name

    @property
    def identifier(self):
        return self.primal.identifier

    @property
    def provider(self):
        return self.primal.provider

    @property
    def providers(self):
        return self.primal.providers

    @property
    def model(self):
        return self.primal.model

    @property
    def reference_id(self):
        return self.primal.reference_id

    @property
    def status(self):
        return self.primal.status

    @property
    def valid(self):
        return self.primal.valid

    @property
    def clipped(self):
        return self.primal.clipped

    @property
    def error_policy(self):
        return self.primal.error_policy

    @property
    def available_properties(self):
        return frozenset(self._data)

    def _canonical(self, name):
        canonical = ALIAS_TO_CANONICAL.get(name)
        if canonical is None:
            raise KeyError(f"Unknown property alias: {name}")
        if canonical not in self._data:
            raise UnsupportedPropertyError(
                fluid_name=self.fluid_name,
                provider=self.provider,
                model=self.model,
                request=canonical,
                reason="This derivative property is unsupported or not enabled.",
                supported=sorted(self._data),
            )
        return canonical

    def supports_property(self, name):
        return ALIAS_TO_CANONICAL.get(name) in self._data

    def raw_property(self, name):
        return self._data[PropertyDerivatives._canonical(self, name)]

    def property_valid(self, name):
        return jnp.isfinite(self.raw_property(name))

    def derivative_units(self, name):
        """Return SI numerator/denominator units in final-axis order."""
        unit = self.property_units[PropertyDerivatives._canonical(self, name)]
        return tuple(f"({unit})/({denominator})" for denominator in self.input_units)

    def __getitem__(self, name):
        return PropertyDerivatives.raw_property(self, name)

    def __getattr__(self, name):
        if name in ALIAS_TO_CANONICAL:
            return PropertyDerivatives.raw_property(self, name)
        raise AttributeError(f"'PropertyDerivatives' object has no attribute '{name}'")

    def to_dict(self, include_aliases=False):
        result = dict(self._data)
        if include_aliases:
            for name, value in self._data.items():
                for alias in PROPERTY_ALIASES[name]:
                    result[alias] = value
        return result

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


def _option_error(fluid, name, reason, supported=()):
    return UnsupportedOptionError(
        fluid_name=fluid.name,
        provider=fluid.provider,
        model=fluid.model,
        request=name,
        reason=reason,
        supported=supported,
    )


def get_props_gradient(
    fluid,
    input_pair,
    x,
    y,
    method="auto",
    eps_rel=1e-6,
    eps_abs=1e-6,
    *,
    errors="mask",
):
    """Return local two-coordinate property partials and their primal state.

    Native AD linearizes one numerical property dictionary and applies its JVP
    to two unit directions. Array/broadcast inputs produce ``shape + (2,)``
    local partials, never a full cross-element Jacobian. The FD alternative
    uses independent centered perturbations. Differentiation follows the
    selected mask/clip policy, including zero sensitivities outside clip bounds.
    """
    choices = ("auto", "jax", "fd", "finite_difference")
    if not isinstance(method, str) or method not in choices:
        raise _option_error(
            fluid, "method", "Choose an advertised differentiation method.", choices
        )
    for name, value in (("eps_rel", eps_rel), ("eps_abs", eps_abs)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < 0
        ):
            raise _option_error(
                fluid, name, "Perturbation sizes must be finite nonnegative scalars."
            )
    if eps_rel == eps_abs == 0:
        raise _option_error(
            fluid, "eps_rel/eps_abs", "At least one perturbation size must be positive."
        )
    first, second = validate_request(
        fluid, input_pair, x, y, errors=errors, array_backend="jax"
    )
    if errors not in ("mask", "clip"):
        raise UnsupportedExecutionModeError(
            fluid_name=fluid.name,
            provider=fluid.provider,
            model=fluid.model,
            request=f"errors={errors}",
            reason="Property differentiation requires explicit mask or clip execution.",
            supported=("errors=mask", "errors=clip"),
        )
    first, second = jnp.broadcast_arrays(
        jnp.asarray(first, dtype=jnp.result_type(first, 0.0)),
        jnp.asarray(second, dtype=jnp.result_type(second, 0.0)),
    )

    def evaluate(a, b):
        primal = fluid.get_state(input_pair, a, b, errors=errors)
        # Only floating property fields enter the differentiated value. Integer
        # statuses and boolean clipping/validity remain undifferentiated aux.
        properties = {
            name: (
                value
                if jnp.issubdtype(value.dtype, jnp.floating)
                else jnp.asarray(value, dtype=jnp.float64)
            )
            for name, value in primal.to_dict().items()
        }
        return properties, primal

    if method in ("auto", "jax"):
        if "jvp" not in fluid.capabilities.jax_modes:
            raise UnsupportedExecutionModeError(
                fluid_name=fluid.name,
                provider=fluid.provider,
                model=fluid.model,
                request="property_derivatives",
                reason="The evaluator does not support JVP differentiation.",
                supported=tuple(sorted(fluid.capabilities.jax_modes)),
            )
        _, linearized, primal = jax.linearize(evaluate, first, second, has_aux=True)
        first_partials = linearized(jnp.ones_like(first), jnp.zeros_like(second))
        second_partials = linearized(jnp.zeros_like(first), jnp.ones_like(second))
        actual_method = "autodiff"
    else:
        _, primal = evaluate(first, second)
        dx = eps_rel * jnp.maximum(jnp.abs(first), 1.0) + eps_abs
        dy = eps_rel * jnp.maximum(jnp.abs(second), 1.0) + eps_abs
        plus_x, _ = evaluate(first + dx, second)
        minus_x, _ = evaluate(first - dx, second)
        plus_y, _ = evaluate(first, second + dy)
        minus_y, _ = evaluate(first, second - dy)
        first_partials = {
            name: (value - minus_x[name]) / (2 * dx) for name, value in plus_x.items()
        }
        second_partials = {
            name: (value - minus_y[name]) / (2 * dy) for name, value in plus_y.items()
        }
        actual_method = "finite_difference"
    partials = {
        name: jnp.stack((value, second_partials[name]), axis=-1)
        for name, value in first_partials.items()
    }
    return PropertyDerivatives(
        primal,
        partials,
        _PAIR_INPUT_NAMES[input_pair],
        method=actual_method,
    )
