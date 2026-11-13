"""CoolProp evaluation with eager typed errors and explicit JAX mask mode."""

from functools import partial, update_wrapper
from numbers import Integral

import numpy as np
import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero
import equinox as eqx

from .. import helpers_props as jxp
from ..configuration import (
    FluidCapabilities,
    FluidConfiguration,
    build_configuration,
)
from ..errors import (
    InvalidConfigurationError,
    JaxPropError,
    UnsupportedExecutionModeError,
    UnsupportedOperationError,
)
from ..fluid_api import (
    DIAGNOSTIC_PROPERTIES,
    SOURCE_PROPERTIES,
    enabled_properties,
    validate_request,
)
from ..source_references import SourceReferencesMixin

_EAGER_TRACE_STATE = jax.core.get_opaque_trace_state()
_BASE_PROPERTY_NAMES = tuple(
    name for name in jxp.PROPERTIES_CANONICAL if name in SOURCE_PROPERTIES
)
_EXTENDED_PROPERTY_NAMES = tuple(
    name
    for name in jxp.PROPERTIES_CANONICAL
    if name in SOURCE_PROPERTIES | DIAGNOSTIC_PROPERTIES
)


def _names(supersaturation):
    return _EXTENDED_PROPERTY_NAMES if supersaturation else _BASE_PROPERTY_NAMES


def _make_template(shape, supersaturation=False):
    return (
        {
            name: jax.ShapeDtypeStruct(shape, jnp.float64)
            for name in _names(supersaturation)
        },
        jax.ShapeDtypeStruct(shape, jnp.int32),
    )


def _get_props_python(
    input_type, x, y, fluid, generalize_quality=False, supersaturation=False
):
    """Return masked host data; static validation precedes this path."""
    x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    properties, status = fluid._evaluate_state_data(
        input_type,
        x,
        y,
        generalize_quality=generalize_quality,
        supersaturation=supersaturation,
        errors="mask",
    )
    return (
        {
            name: np.asarray(properties[name], dtype=np.float64)
            for name in _names(supersaturation)
        },
        np.asarray(status, dtype=np.int32),
    )


def _host_callback(input_pair, x, y, fluid, generalize_quality, supersaturation):
    template = _make_template(
        np.broadcast_shapes(jnp.shape(x), jnp.shape(y)), supersaturation
    )
    evaluate = partial(
        _get_props_python,
        input_pair,
        fluid=fluid,
        generalize_quality=generalize_quality,
        supersaturation=supersaturation,
    )
    return jax.pure_callback(evaluate, template, x, y, vmap_method="broadcast_all")


@partial(jax.custom_jvp, nondiff_argnums=(0, 3, 4, 5))
def _first_order_samples(input_pair, x, y, fluid, generalize_quality, supersaturation):
    return _host_callback(input_pair, x, y, fluid, generalize_quality, supersaturation)


@_first_order_samples.defjvp
def _reject_higher_derivatives(
    input_pair, fluid, generalize_quality, supersaturation, primals, tangents
):
    raise UnsupportedExecutionModeError(
        fluid_name=fluid.name,
        provider="coolprop_jax",
        model=fluid.model,
        request="derivative_order",
        reason="The callback supports first-order finite-difference derivatives only; higher-order differentiation is unsupported.",
        supported=("first-order",),
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 3, 4, 5))
def _callback_state(input_pair, x, y, fluid, generalize_quality, supersaturation):
    return _host_callback(input_pair, x, y, fluid, generalize_quality, supersaturation)


def _callback_state_jvp(
    input_pair, fluid, generalize_quality, supersaturation, primals, tangents
):
    """Forward differences with no work for symbolic-zero inputs."""
    x, y = primals
    x_dot, y_dot = tangents
    evaluate = partial(
        _first_order_samples,
        input_pair,
        fluid=fluid,
        generalize_quality=generalize_quality,
        supersaturation=supersaturation,
    )
    properties, status = evaluate(x, y)
    contributions = []
    for axis, (value, tangent) in enumerate(((x, x_dot), (y, y_dot))):
        if (
            isinstance(tangent, SymbolicZero)
            or getattr(tangent, "dtype", None) == jax.dtypes.float0
        ):
            continue
        step = 1e-6 * (jnp.abs(value) + 1.0)
        perturbed, _ = evaluate(x + step, y) if axis == 0 else evaluate(x, y + step)
        contributions.append(
            {
                name: (perturbed[name] - properties[name]) / step * tangent
                for name in properties
            }
        )
    numerical_tangent = {
        name: sum((term[name] for term in contributions), jnp.zeros_like(value))
        for name, value in properties.items()
    }
    status_tangent = np.zeros(status.shape, dtype=jax.dtypes.float0)
    return (properties, status), (numerical_tangent, status_tangent)


_callback_state.defjvp(_callback_state_jvp, symbolic_zeros=True)


def get_state(input_pair, x, y, fluid):
    """Compatibility numeric helper; use FluidJAX.get_state for policies."""
    properties, _ = _callback_state(input_pair, x, y, fluid, False, False)
    return properties


@partial(jax.jit, static_argnames=("metadata",))
def _assemble_state(properties, status, metadata):
    name, identifier, provider, model, reference_id, available, errors = metadata
    return jxp.FluidState(
        fluid_name=name,
        identifier=identifier,
        provider=provider,
        model=model,
        reference_id=reference_id,
        available_properties=available,
        error_policy=errors,
        status=status,
        **properties
    )


def _state_metadata(fluid, supersaturation, errors):
    return (
        fluid.name,
        fluid.identifier,
        fluid.provider,
        fluid.model,
        fluid.reference_id,
        enabled_properties(fluid, supersaturation=supersaturation),
        errors,
    )


def _eager_state(fluid, input_type, first, second, generalize_quality, supersaturation):
    first, second = np.broadcast_arrays(np.asarray(first), np.asarray(second))
    try:
        properties, status = fluid.fluid._evaluate_state_data(
            input_type,
            first,
            second,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
            errors="raise",
        )
    except JaxPropError as error:
        mapped = type(error)(
            fluid_name=fluid.name,
            provider=fluid.provider,
            model=fluid.model,
            request=error.request,
            reason=error.reason,
            supported=error.supported,
            index=error.index,
            indices=error.indices,
        )
        raise mapped from (error.__cause__ or error)
    return _assemble_state(
        properties, status, _state_metadata(fluid, supersaturation, "raise")
    )


class _CallbackStateCall:
    """Select eager errors or compiled masks without touching native JIT."""

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
        **options
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
        if errors == "raise":
            # Also cover a traced closure with only constant inputs.
            if (
                isinstance(first, jax.core.Tracer)
                or isinstance(second, jax.core.Tracer)
                or jax.core.get_opaque_trace_state() != _EAGER_TRACE_STATE
            ):
                raise UnsupportedExecutionModeError(
                    fluid_name=fluid.name,
                    provider=fluid.provider,
                    model=fluid.model,
                    request="errors=raise",
                    reason="JAX transformations require explicit errors='mask'; eager raising evaluation preserves typed source failures.",
                    supported=("errors=mask",),
                )
            return _eager_state(
                fluid,
                input_type,
                first,
                second,
                bool(generalize_quality),
                bool(supersaturation),
            )
        return self.compiled(
            fluid,
            input_type,
            first,
            second,
            generalize_quality=bool(generalize_quality),
            supersaturation=bool(supersaturation),
            errors="mask",
        )


class FluidJAX(SourceReferencesMixin, eqx.Module):
    name: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    identifier: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    model: str = eqx.field(static=True)
    reference_id: tuple = eqx.field(static=True)
    fluid: object = eqx.field(static=True)
    exceptions: bool = eqx.field(static=True)
    _configuration: FluidConfiguration = eqx.field(static=True)
    _capabilities: FluidCapabilities = eqx.field(static=True)
    _references: object = eqx.field(static=True)

    def __init__(
        self, name: str, backend: str = "HEOS", exceptions: bool = True, identifier=None
    ):
        from .fluid_properties import FluidCoolProp

        backend = "HEOS" if backend is None else backend
        self._configuration = build_configuration(
            name, "coolprop_jax", "real_fluid", backend, identifier
        )
        if not isinstance(exceptions, (bool, np.bool_)) or not bool(exceptions):
            raise InvalidConfigurationError(
                fluid_name=name,
                provider="coolprop_jax",
                model="real_fluid",
                request="exceptions",
                reason="Use per-call errors='mask' instead of legacy exceptions=False.",
                supported=(True,),
            )
        self.name = name
        self.backend = backend
        self.exceptions = True
        self.identifier = identifier
        self.provider = "coolprop_jax"
        self.model = "real_fluid"
        self.fluid = FluidCoolProp(name=name, backend=backend, identifier=identifier)
        source_capabilities = self.fluid.capabilities
        self._capabilities = FluidCapabilities(
            input_pairs=source_capabilities.input_pairs,
            properties=source_capabilities.properties,
            options=source_capabilities.options,
            phase_operations=source_capabilities.phase_operations,
            error_modes=frozenset({"raise", "mask"}),
            jax_modes=frozenset({"eager", "jit", "vmap", "jvp", "grad"}),
            derivative_order=1,
            derivative_method="finite_difference",
            domain=source_capabilities.domain,
        )
        self.reference_id = self.fluid.reference_id
        self._references = self.fluid.references

    @property
    def configuration(self):
        return self._configuration

    @property
    def capabilities(self):
        return self._capabilities

    @_CallbackStateCall
    def get_state(
        self,
        input_type,
        prop_1,
        prop_2,
        *,
        generalize_quality=False,
        supersaturation=False,
        errors="raise"
    ):
        """Evaluate eagerly with typed errors, or use explicit mask mode in JAX.

        Derivatives are first-order forward finite differences over successful
        source states, away from phase/domain boundaries.
        """
        if errors != "mask":
            raise UnsupportedExecutionModeError(
                fluid_name=self.name,
                provider=self.provider,
                model=self.model,
                request="errors=raise",
                reason="The compiled callback path requires errors='mask'.",
                supported=("errors=mask",),
            )
        first, second = jnp.broadcast_arrays(jnp.asarray(prop_1), jnp.asarray(prop_2))
        properties, status = _callback_state(
            input_type, first, second, self.fluid, generalize_quality, supersaturation
        )
        return _assemble_state(
            properties, status, _state_metadata(self, supersaturation, errors)
        )

    def get_state_saturation(self, input_type, prop_1, prop_2, *, errors="raise"):
        """Evaluate saturation through the common state path."""
        if (
            isinstance(input_type, (bool, np.bool_))
            or not isinstance(input_type, Integral)
            or input_type not in (jxp.PQ_INPUTS, jxp.QT_INPUTS)
        ):
            raise UnsupportedOperationError(
                fluid_name=self.name,
                provider=self.provider,
                model=self.model,
                request="saturation",
                reason="Saturation evaluation accepts only PQ_INPUTS or QT_INPUTS.",
                supported=("PQ_INPUTS", "QT_INPUTS"),
            )
        return self.get_state(input_type, prop_1, prop_2, errors=errors)
