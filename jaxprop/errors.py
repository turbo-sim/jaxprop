"""Typed public failures shared by fluid evaluators and returned states."""

__all__ = [
    "JaxPropError",
    "InvalidConfigurationError",
    "UnsupportedRequestError",
    "UnsupportedInputPairError",
    "UnsupportedPropertyError",
    "UnsupportedOptionError",
    "UnsupportedOperationError",
    "UnsupportedExecutionModeError",
    "UnavailableReferencePointError",
    "PropertyUnavailableError",
    "InvalidStateError",
    "OutOfDomainError",
    "CalculationError",
    "ConvergenceError",
]


def _short_text(value, limit=200):
    """Keep error messages readable even when a source supplies a long reason."""
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _restore_error(error_type, context):
    """Reconstruct keyword-only errors when unpickling them."""
    return error_type(**context)


class JaxPropError(Exception):
    """A fluid request failed, with context suitable for callers and logs.

    Construct errors with canonical request/property names. The message gives a
    bounded summary; the structured attributes retain the complete context.
    Provider failures should be raised with ``raise error from original``.
    """

    def __init__(
        self,
        *,
        request,
        reason,
        fluid_name=None,
        provider=None,
        model=None,
        supported=(),
        index=None,
        indices=None,
    ):
        self.fluid_name = fluid_name
        self.provider = provider
        self.model = model
        self.request = str(request)
        self.reason = str(reason)
        self.supported = (
            (supported,) if isinstance(supported, str) else tuple(supported)
        )
        self.index = None if index is None else tuple(int(value) for value in index)
        self.indices = (
            None
            if indices is None
            else tuple(tuple(int(value) for value in item) for item in indices)
        )

        identity = ", ".join(
            f"{name}={_short_text(value)}"
            for name, value in (
                ("fluid", fluid_name),
                ("provider", provider),
                ("model", model),
            )
            if value is not None
        )
        message = f"{type(self).__name__}"
        if identity:
            message += f" ({identity})"
        message += f": {_short_text(self.request)}: {_short_text(self.reason)}"
        if self.index is not None:
            message += f"; index={self.index}"
        if self.indices is not None:
            shown_indices = self.indices[:8]
            message += f"; indices={shown_indices}"
            if len(self.indices) > len(shown_indices):
                message += f" ({len(self.indices)} total)"
        if self.supported:
            choices = ", ".join(_short_text(item, 80) for item in self.supported[:8])
            message += f"; supported: {choices}"
            if len(self.supported) > 8:
                message += f" ({len(self.supported)} total)"
        super().__init__(message)

    def __reduce__(self):
        context = {
            name: getattr(self, name)
            for name in (
                "fluid_name",
                "provider",
                "model",
                "request",
                "reason",
                "supported",
                "index",
                "indices",
            )
        }
        return _restore_error, (type(self), context), self.__dict__


class InvalidConfigurationError(JaxPropError, ValueError):
    """Unknown, incompatible, or unavailable construction settings."""


class UnsupportedRequestError(JaxPropError, ValueError):
    """Base for requests outside the evaluator's declared capabilities."""


class UnsupportedInputPairError(UnsupportedRequestError):
    """The requested input pair is unsupported by this evaluator/model."""


class UnsupportedPropertyError(UnsupportedRequestError):
    """A known property is unsupported or was not enabled for this state."""


class UnsupportedOptionError(UnsupportedRequestError):
    """A requested option is unknown, invalid, or unsupported."""


class UnsupportedOperationError(UnsupportedRequestError):
    """The requested phase or other operation is unsupported."""


class UnsupportedExecutionModeError(UnsupportedRequestError):
    """The requested JAX transformation or failure policy is unsupported."""


class UnavailableReferencePointError(JaxPropError, LookupError):
    """The specifically requested source reference point is unavailable."""


class PropertyUnavailableError(JaxPropError, ValueError):
    """An enabled property is undefined at an otherwise valid state."""


class InvalidStateError(JaxPropError, ValueError):
    """State inputs have invalid values, types, or broadcast shapes."""


class OutOfDomainError(InvalidStateError):
    """The requested state lies outside the evaluator's declared domain."""


class CalculationError(JaxPropError, RuntimeError):
    """Evaluation failed after valid inputs; retain the underlying cause."""


class ConvergenceError(CalculationError):
    """A numerical inverse failed its convergence or residual criteria."""
