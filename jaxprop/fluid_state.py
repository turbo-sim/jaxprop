"""Immutable, JAX-compatible fluid values with explicit property availability."""

import operator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .errors import (
    InvalidConfigurationError,
    InvalidStateError,
    PropertyUnavailableError,
    UnsupportedExecutionModeError,
    UnsupportedPropertyError,
)
from .property_schema import ALIAS_TO_CANONICAL, PROPERTY_ALIASES, PROPERTIES_CANONICAL
from .fluid_api import StatusCode


@jax.jit
def _jax_property_valid(value, status):
    """Materialize a requested mask in one compiled array operation."""
    return jnp.isfinite(value) & (status == StatusCode.SUCCESS)


@jax.jit
def _jax_property_validity(properties, status):
    """Materialize serialization masks together, avoiding per-field dispatch."""
    successful = status == StatusCode.SUCCESS
    return {
        name: jnp.isfinite(value) & successful for name, value in properties.items()
    }


class FluidState(eqx.Module):
    """A numeric pytree whose public properties enforce their availability.

    Raw leaves are private so Equinox/JAX transformations never invoke guarded
    public lookups. Metadata contains only immutable, non-array values.
    """

    fluid_name: str | None = eqx.field(static=True)
    identifier: str | None = eqx.field(static=True)
    provider: str | None = eqx.field(static=True)
    providers: frozenset = eqx.field(static=True)
    model: str | None = eqx.field(static=True)
    reference_id: object = eqx.field(static=True)
    available_properties: frozenset = eqx.field(static=True)
    error_policy: str = eqx.field(static=True)
    _data: dict
    status: jax.Array

    def __init__(
        self,
        fluid_name=None,
        identifier=None,
        *,
        provider=None,
        providers=None,
        model=None,
        reference_id=None,
        available_properties=None,
        error_policy="raise",
        status=None,
        property_validity=None,
        **properties,
    ):
        self.fluid_name = fluid_name
        self.identifier = identifier
        self.model = model
        self.reference_id = reference_id
        self.error_policy = error_policy
        context = dict(fluid_name=fluid_name, provider=provider, model=model)
        for name, value in (
            ("fluid_name", fluid_name),
            ("identifier", identifier),
            ("provider", provider),
            ("model", model),
        ):
            if value is not None and not isinstance(value, str):
                raise InvalidConfigurationError(
                    request=name,
                    reason="State identity metadata must be strings or None.",
                    **context,
                )
        if isinstance(providers, str):
            raise InvalidConfigurationError(
                request="providers",
                reason="Provider provenance must be a collection of names, not a string.",
                **context,
            )
        provenance = frozenset(
            providers
            if providers is not None
            else (() if provider is None else (provider,))
        )
        if any(
            not isinstance(value, str) or not value or value == "mixed"
            for value in provenance
        ):
            raise InvalidConfigurationError(
                request="providers",
                reason="Provider provenance must contain nonempty individual provider names.",
                **context,
            )
        derived_provider = (
            next(iter(provenance))
            if len(provenance) == 1
            else ("mixed" if provenance else None)
        )
        if provider is not None and provider != derived_provider:
            raise InvalidConfigurationError(
                request="providers",
                reason="Provider and provenance describe different evaluators.",
                **context,
            )
        self.provider = derived_provider
        self.providers = provenance
        context["provider"] = derived_provider
        if error_policy not in ("raise", "mask"):
            raise InvalidConfigurationError(
                request="error_policy",
                reason="Choose 'raise' or 'mask'.",
                supported=("raise", "mask"),
                **context,
            )
        try:
            hash(reference_id)
        except TypeError as error:
            raise InvalidConfigurationError(
                request="reference_id",
                reason="Reference identity must be immutable and hashable.",
                **context,
            ) from error
        # Keep ordinary CoolProp/table sampling on NumPy. Converting every
        # scalar/property to a JAX device array here adds thousands of tiny
        # dispatches during table generation. Native/traced inputs stay in JAX.
        leaves = jax.tree_util.tree_leaves((properties, status, property_validity))
        xp = (
            jnp
            if any(isinstance(value, (jax.Array, jax.core.Tracer)) for value in leaves)
            else np
        )
        values = {}
        for name, value in properties.items():
            canonical = ALIAS_TO_CANONICAL.get(name)
            if canonical is None:
                raise InvalidConfigurationError(
                    request=name,
                    reason="Unknown state property.",
                    supported=PROPERTIES_CANONICAL,
                    **context,
                )
            if canonical in values:
                raise InvalidConfigurationError(
                    request=canonical,
                    reason="A property was supplied more than once through canonical/alias names.",
                    **context,
                )
            values[canonical] = self._numeric_array(value, canonical, context, xp)
        enabled = (
            set(values) if available_properties is None else set(available_properties)
        )
        normalized = set()
        for name in enabled:
            if name not in ALIAS_TO_CANONICAL:
                raise InvalidConfigurationError(
                    request=name,
                    reason="Unknown enabled property.",
                    supported=PROPERTIES_CANONICAL,
                    **context,
                )
            normalized.add(ALIAS_TO_CANONICAL[name])
        self.available_properties = frozenset(normalized)
        masks = {}
        for name, value in (property_validity or {}).items():
            canonical = ALIAS_TO_CANONICAL.get(name)
            if canonical not in normalized:
                raise InvalidConfigurationError(
                    request=name,
                    reason="Validity masks require an enabled property.",
                    supported=sorted(normalized),
                    **context,
                )
            if canonical in masks:
                raise InvalidConfigurationError(
                    request=canonical,
                    reason="A validity mask was supplied more than once through canonical/alias names.",
                    **context,
                )
            mask = xp.asarray(value)
            if mask.dtype != np.bool_:
                raise InvalidStateError(
                    request=name,
                    reason="Property validity masks must contain booleans.",
                    **context,
                )
            masks[canonical] = mask
        status_array = xp.array(StatusCode.SUCCESS if status is None else status)
        if (
            not np.issubdtype(status_array.dtype, np.integer)
            or status_array.dtype == np.bool_
        ):
            raise InvalidStateError(
                request="status", reason="Status must contain integer codes.", **context
            )
        if not isinstance(status_array, jax.core.Tracer) and np.any(
            (np.asarray(status_array) < StatusCode.SUCCESS)
            | (np.asarray(status_array) > StatusCode.PROVIDER_FAILURE)
        ):
            raise InvalidStateError(
                request="status",
                reason="Status codes must be in the range 0 through 4.",
                **context,
            )
        try:
            shape = np.broadcast_shapes(
                *(
                    array.shape
                    for array in (*values.values(), *masks.values(), status_array)
                )
            )
        except ValueError as error:
            raise InvalidStateError(
                request="shape",
                reason="State properties, masks, and status must have compatible broadcast shapes.",
                **context,
            ) from error
        # Most evaluator outputs already have the final shape. Avoid dispatching
        # a broadcast for every value/mask, especially for ordinary NumPy scalars.
        self.status = (
            status_array
            if status_array.shape == shape
            else xp.broadcast_to(status_array, shape)
        )
        if xp is np:
            self.status.setflags(write=False)
        successful = self.status == StatusCode.SUCCESS
        self._data = {}
        for name in PROPERTIES_CANONICAL:
            if name not in normalized:
                continue
            # dict.get eagerly evaluates its default even when the key exists.
            # Allocate missing-value arrays only for genuinely missing fields.
            value = values[name] if name in values else xp.asarray(np.nan)
            if value.shape != shape:
                value = xp.broadcast_to(value, shape)
            valid = successful & xp.isfinite(value)
            if name in masks:
                mask = masks[name]
                if mask.shape != shape:
                    mask = xp.broadcast_to(mask, shape)
                valid = valid & mask
            self._data[name] = xp.where(valid, value, np.nan)
            if xp is np:
                self._data[name].setflags(write=False)
        # Unavailable values are already NaN. Recovering their validity on
        # demand avoids returning a redundant boolean array per property from
        # compiled evaluators. NumPy/JAX states share the same tree structure.

    @staticmethod
    def _numeric_array(value, name, context, xp):
        try:
            array = xp.asarray(value)
        except (TypeError, ValueError) as error:
            raise InvalidStateError(
                request=name,
                reason="State values must be real numeric arrays.",
                **context,
            ) from error
        if (
            not jnp.issubdtype(array.dtype, jnp.number)
            and not (name == "is_two_phase" and array.dtype == jnp.bool_)
        ) or jnp.issubdtype(array.dtype, jnp.complexfloating):
            raise InvalidStateError(
                request=name,
                reason="State values must be real numeric arrays.",
                **context,
            )
        return array

    @property
    def shape(self):
        return self.status.shape

    @property
    def valid(self):
        return self.status == StatusCode.SUCCESS

    def supports_property(self, name):
        return ALIAS_TO_CANONICAL.get(name) in self.available_properties

    def _canonical(self, name):
        canonical = ALIAS_TO_CANONICAL.get(name)
        if canonical is None:
            raise KeyError(f"Unknown property alias: {name}")
        if canonical not in self.available_properties:
            raise UnsupportedPropertyError(
                fluid_name=self.fluid_name,
                provider=self.provider,
                model=self.model,
                request=canonical,
                reason="This property is unsupported or not enabled for this state.",
                supported=sorted(self.available_properties),
            )
        return canonical

    def raw_property(self, name):
        return self._data[FluidState._canonical(self, name)]

    def property_valid(self, name):
        value = self._data[FluidState._canonical(self, name)]
        if isinstance(value, (jax.Array, jax.core.Tracer)) or isinstance(
            self.status, (jax.Array, jax.core.Tracer)
        ):
            return _jax_property_valid(value, self.status)
        valid = np.asarray(np.isfinite(value) & (self.status == StatusCode.SUCCESS))
        valid.setflags(write=False)
        return valid

    def _read_property(self, name):
        canonical = FluidState._canonical(self, name)
        value = self._data[canonical]
        # Evaluator wrappers enforce transformed error modes in later phases.
        # State operations never turn traced availability into a Python bool.
        if (
            self.error_policy == "raise"
            and not isinstance(value, jax.core.Tracer)
            and not isinstance(self.status, jax.core.Tracer)
        ):
            # Inspect concrete storage directly rather than launching a tiny
            # JAX validity kernel and immediately synchronizing it to the host.
            concrete_valid = np.isfinite(np.asarray(value)) & (
                np.asarray(self.status) == StatusCode.SUCCESS
            )
            all_valid = (
                bool(concrete_valid)
                if concrete_valid.ndim == 0
                else concrete_valid.all()
            )
            if not all_valid:
                unavailable = np.logical_not(concrete_valid)
                indices = (
                    [()]
                    if unavailable.ndim == 0
                    else list(zip(*np.nonzero(unavailable)))
                )
                raise PropertyUnavailableError(
                    fluid_name=self.fluid_name,
                    provider=self.provider,
                    model=self.model,
                    request=canonical,
                    reason="The enabled property is undefined at the requested state indices.",
                    supported=sorted(self.available_properties),
                    indices=indices,
                )
        return value

    def __getattr__(self, name):
        if name in ALIAS_TO_CANONICAL:
            return FluidState._read_property(self, name)
        raise AttributeError(f"'FluidState' object has no attribute '{name}'")

    def __getitem__(self, name):
        if name in (
            "fluid_name",
            "identifier",
            "provider",
            "providers",
            "model",
            "reference_id",
        ):
            return getattr(self, name)
        return FluidState._read_property(self, name)

    def to_dict(self, include_aliases=False):
        """Export enabled raw numeric fields, never metadata/status or guards."""
        result = dict(self._data)
        if include_aliases:
            for canonical in self._data:
                for alias in PROPERTY_ALIASES[canonical]:
                    result[alias] = self._data[canonical]
        return result

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def _metadata(self):
        return dict(
            fluid_name=self.fluid_name,
            identifier=self.identifier,
            provider=self.provider,
            providers=self.providers,
            model=self.model,
            reference_id=self.reference_id,
            available_properties=self.available_properties,
            error_policy=self.error_policy,
        )

    def to_data(self):
        """Export metadata, raw values, status and masks for a lossless round trip."""
        if isinstance(self.status, (jax.Array, jax.core.Tracer)) or any(
            isinstance(value, (jax.Array, jax.core.Tracer))
            for value in self._data.values()
        ):
            masks = _jax_property_validity(self._data, self.status)
        else:
            masks = {name: FluidState.property_valid(self, name) for name in self._data}
        return dict(
            metadata=self._metadata(),
            properties=self.to_dict(),
            status=self.status,
            property_validity=masks,
        )

    @classmethod
    def from_data(cls, data):
        return cls(
            **data["metadata"],
            **data["properties"],
            status=data["status"],
            property_validity=data["property_validity"],
        )

    def __reduce__(self):
        # Reconstruct through validation so NumPy's pickle restoration cannot
        # turn exported read-only numerical arrays into writable state storage.
        return type(self).from_data, (self.to_data(),)

    def at_index(self, index):
        if isinstance(index, jax.core.Tracer):
            raise UnsupportedExecutionModeError(
                fluid_name=self.fluid_name,
                provider=self.provider,
                model=self.model,
                request="at_index",
                reason="State indexing requires a static integer to enforce bounds.",
            )
        if isinstance(index, (bool, np.bool_)):
            raise TypeError("State index must be an integer")
        index = operator.index(index)
        size = self.shape[0] if self.shape else 1
        if not -size <= index < size:
            raise IndexError(
                f"State index {index} is outside the first-axis length {size}"
            )
        if not self.shape:
            return self
        return jax.tree_util.tree_map(lambda value: value[index], self)

    def flipped(self):
        if not self.shape:
            return self
        # Slicing preserves each leaf's storage backend and NumPy read-only
        # views; jnp.flip would transfer ordinary source states to JAX.
        return jax.tree_util.tree_map(lambda value: value[::-1], self)

    @classmethod
    def _from_validated(cls, metadata, properties, status):
        """Assemble collection results whose inputs/invariants are validated.

        This private path is only for shape-preserving combinations of existing
        states. Their values are already NaN-masked and their metadata/statuses
        have passed public construction checks. Collection
        compatibility is checked before this method is called. New NumPy arrays
        are owned by the result and frozen before exposing them.
        """
        for value in (*properties.values(), status):
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        result = object.__new__(cls)
        for name, value in metadata.items():
            object.__setattr__(result, name, value)
        object.__setattr__(result, "_data", properties)
        object.__setattr__(result, "status", status)
        return result

    @classmethod
    def _combine(cls, states, *, stack):
        first = states[0]
        for state in states[1:]:
            if (
                state.fluid_name != first.fluid_name
                or state.identifier != first.identifier
            ):
                raise ValueError(
                    "Cannot combine states with different fluid_name or identifier"
                )
            if state.model != first.model or state.reference_id != first.reference_id:
                raise ValueError(
                    "Cannot combine states with different physical model or reference_id"
                )
            if state.error_policy != first.error_policy:
                raise ValueError("Cannot combine states with different error_policy")
        shapes = [state.shape for state in states]
        if stack and any(shape != shapes[0] for shape in shapes):
            raise ValueError("Stack requires identical state shapes")
        if not stack and any(
            (shape or (1,))[1:] != (shapes[0] or (1,))[1:] for shape in shapes
        ):
            raise ValueError("Concatenation requires matching trailing state shapes")
        enabled = frozenset.intersection(
            *(state.available_properties for state in states)
        )
        providers = frozenset.union(*(state.providers for state in states))
        provider = (
            next(iter(providers))
            if len(providers) == 1
            else ("mixed" if providers else None)
        )
        # Keep ordinary source states on NumPy. Mixed storage and transformed
        # calls use JAX so no traced numerical leaves reach NumPy operations.
        xp = (
            jnp
            if any(
                isinstance(value, (jax.Array, jax.core.Tracer))
                for state in states
                for value in (
                    state.status,
                    *state._data.values(),
                )
            )
            else np
        )
        combine = (
            (lambda arrays: xp.stack(arrays, axis=0))
            if stack
            else (
                lambda arrays: xp.concatenate(
                    [xp.atleast_1d(value) for value in arrays], axis=0
                )
            )
        )
        properties = {
            name: combine([state._data[name] for state in states]) for name in enabled
        }
        metadata = FluidState._metadata(first) | dict(
            provider=provider, providers=providers, available_properties=enabled
        )
        status = combine([state.status for state in states])
        if not np.issubdtype(status.dtype, np.integer):
            # Signed/unsigned integer promotion can produce floats (for example
            # int64 + uint64). Existing states only contain validated codes 0–4,
            # so restoring an integer dtype is exact and keeps status semantic.
            status = status.astype(np.int32)
        return cls._from_validated(
            metadata,
            properties,
            status=status,
        )

    def __add__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._combine([self, other], stack=False)

    @classmethod
    def stack(cls, states):
        states = list(states)
        if not states:
            raise ValueError("No states provided to stack")
        if not all(isinstance(state, cls) for state in states):
            raise TypeError("Stack requires FluidState objects")
        return cls._combine(states, stack=True)

    def __repr__(self):
        return f"FluidState(fluid_name={self.fluid_name!r}, identifier={self.identifier!r}, provider={self.provider!r}, model={self.model!r}, shape={self.shape}, properties={sorted(self.available_properties)!r})"
