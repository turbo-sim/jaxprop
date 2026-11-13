"""Immutable configuration and capability records shared by fluid objects."""

from collections.abc import Mapping
from dataclasses import dataclass, field, is_dataclass
from numbers import Integral, Number, Real
import os
import math

from .errors import InvalidConfigurationError

__all__ = [
    "FrozenMapping",
    "FluidConfiguration",
    "FluidDomain",
    "FluidCapabilities",
    "build_configuration",
]


def _freeze(value):
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, (str, bytes, Number)):
        return value
    if is_dataclass(value) and value.__dataclass_params__.frozen:
        hash(value)
        return value
    raise TypeError(
        f"Configuration value {type(value).__name__} is not immutable metadata"
    )


class FrozenMapping(Mapping):
    """A hashable, pickleable mapping with recursively immutable values."""

    __slots__ = ("_items", "_hash")

    def __init__(self, values=()):
        copied = dict(values)
        items = tuple((_freeze(key), _freeze(value)) for key, value in copied.items())
        object.__setattr__(self, "_items", items)
        object.__setattr__(self, "_hash", hash(frozenset(items)))

    def __setattr__(self, name, value):
        raise AttributeError("FrozenMapping is immutable")

    def __delattr__(self, name):
        raise AttributeError("FrozenMapping is immutable")

    def __getitem__(self, key):
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self):
        return len(self._items)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"FrozenMapping({dict(self._items)!r})"

    def __reduce__(self):
        return type(self), (dict(self._items),)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _invalid(request, reason, *, name=None, provider=None, model=None):
    raise InvalidConfigurationError(
        fluid_name=name, provider=provider, model=model, request=request, reason=reason
    )


@dataclass(frozen=True)
class FluidConfiguration:
    name: str
    provider: str
    model: str
    backend: str | None
    identifier: str | None = None
    table_options: FrozenMapping = field(default_factory=FrozenMapping)
    model_options: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self):
        context = dict(name=self.name, provider=self.provider, model=self.model)
        for key in ("name", "provider", "model"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                _invalid(key, "Provide a nonempty string.", **context)
        if self.backend is not None and (
            not isinstance(self.backend, str) or not self.backend.strip()
        ):
            _invalid(
                "backend",
                "Provide a nonempty source backend string or None.",
                **context,
            )
        if self.identifier is not None and not isinstance(self.identifier, str):
            _invalid(
                "identifier", "A stream identifier must be a string or None.", **context
            )
        for key in ("table_options", "model_options"):
            value = getattr(self, key)
            if value is None:
                value = {}
            if not isinstance(value, Mapping):
                _invalid(key, "Options must be a mapping or None.", **context)
            try:
                object.__setattr__(self, key, FrozenMapping(value))
            except (TypeError, ValueError) as error:
                raise InvalidConfigurationError(
                    fluid_name=self.name,
                    provider=self.provider,
                    model=self.model,
                    request=key,
                    reason=str(error),
                ) from error


@dataclass(frozen=True)
class FluidDomain:
    description: str = ""
    bounds: FrozenMapping = field(default_factory=FrozenMapping)
    phases: frozenset = field(default_factory=frozenset)
    extrapolation: bool = False

    def __post_init__(self):
        if not isinstance(self.description, str):
            raise TypeError("Domain description must be a string")
        if not isinstance(self.extrapolation, bool):
            raise TypeError("Domain extrapolation must be a boolean")
        if not isinstance(self.bounds, Mapping):
            raise TypeError("Domain bounds must be a mapping")
        for name, limits in self.bounds.items():
            if not isinstance(name, str) or not name:
                raise TypeError("Domain bound names must be nonempty strings")
            if not isinstance(limits, (tuple, list)) or len(limits) != 2:
                raise TypeError("Domain bounds must be lower/upper pairs")
            for limit in limits:
                if limit is not None and (
                    isinstance(limit, bool)
                    or not isinstance(limit, Real)
                    or not math.isfinite(limit)
                ):
                    raise TypeError("Domain limits must be finite real scalars or None")
            lower, upper = limits
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("Domain lower bounds cannot exceed upper bounds")
        if isinstance(self.phases, str):
            raise TypeError("Domain phases must be a collection of nonempty strings")
        phases = frozenset(self.phases)
        if any(not isinstance(name, str) or not name for name in phases):
            raise TypeError("Domain phases must be a collection of nonempty strings")
        object.__setattr__(self, "bounds", FrozenMapping(self.bounds))
        object.__setattr__(self, "phases", phases)


@dataclass(frozen=True)
class FluidCapabilities:
    input_pairs: frozenset
    properties: frozenset
    options: frozenset = field(default_factory=frozenset)
    phase_operations: frozenset = field(default_factory=frozenset)
    jax_modes: frozenset = field(default_factory=lambda: frozenset({"eager"}))
    domain: FluidDomain = field(default_factory=FluidDomain)
    error_modes: frozenset = field(default_factory=lambda: frozenset({"raise"}))
    derivative_order: int = 0
    derivative_method: str | None = None

    def __post_init__(self):
        for key in (
            "input_pairs",
            "properties",
            "options",
            "phase_operations",
            "jax_modes",
            "error_modes",
        ):
            value = getattr(self, key)
            if isinstance(value, str):
                raise TypeError(f"{key} must be a collection, not a string")
            normalized = frozenset(value)
            if key == "input_pairs":
                if any(
                    isinstance(item, bool) or not isinstance(item, Integral)
                    for item in normalized
                ):
                    raise TypeError("Input pairs must be integer constants")
            elif any(not isinstance(item, str) or not item for item in normalized):
                raise TypeError(f"{key} must contain nonempty strings")
            object.__setattr__(self, key, normalized)
        if self.error_modes - {"raise", "mask", "clip"}:
            raise ValueError("Error modes are 'raise', 'mask' and 'clip'")
        if (
            isinstance(self.derivative_order, bool)
            or not isinstance(self.derivative_order, Integral)
            or self.derivative_order < 0
        ):
            raise TypeError("Derivative order must be a nonnegative integer")
        if self.derivative_method is not None and not isinstance(
            self.derivative_method, str
        ):
            raise TypeError("Derivative method must be a string or None")
        if isinstance(self.domain, Mapping):
            object.__setattr__(self, "domain", FluidDomain(**self.domain))
        elif not isinstance(self.domain, FluidDomain):
            raise TypeError("domain must be a FluidDomain or its field mapping")


def build_configuration(
    name,
    provider,
    model,
    backend,
    identifier=None,
    *,
    table_options=None,
    model_options=None,
):
    """Snapshot metadata without resolving provider/model routes or backends."""
    return FluidConfiguration(
        name=name,
        provider=provider,
        model=model,
        backend=backend,
        identifier=identifier,
        table_options=table_options,
        model_options=model_options,
    )
