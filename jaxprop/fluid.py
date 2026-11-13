"""The common fluid constructor and public runtime fluid type."""

from abc import ABCMeta

from . import factory as _factory
from .configuration import FluidCapabilities, FluidConfiguration
from .errors import InvalidConfigurationError
from .fluid_state import FluidState
from .source_references import SourceReferences

__all__ = ["Fluid"]

_METHOD_ROUTES = {
    "coolprop": ("coolprop", "real_fluid"),
    "coolprop_jax": ("coolprop_jax", "real_fluid"),
    "bicubic": ("bicubic", "real_fluid"),
    "perfect_gas": ("native", "perfect_gas"),
    "binary_mixture": ("native", "binary_mixture"),
}


class Fluid(metaclass=ABCMeta):
    """Construct the selected evaluator without wrapping its numerical calls.

    ``method`` selects the numerical implementation and physical model.
    ``backend`` selects its CoolProp source, with ``None`` resolving to HEOS
    for single-source methods. Table and calibration settings use their
    respective option mappings. Binary construction remains reserved until
    the consolidated physical model is implemented.

    The result is the actual evaluator, such as ``FluidCoolProp`` or
    ``FluidPerfectGas``. Evaluator classes are registered virtual subclasses,
    so ``isinstance(result, Fluid)`` works without adding a wrapper or changing
    their JAX PyTrees. Virtual rather than physical inheritance also prevents
    Python from initializing the already constructed result a second time.
    """

    name: str
    provider: str
    model: str
    backend: str | None
    identifier: str | None
    reference_id: tuple
    configuration: FluidConfiguration
    capabilities: FluidCapabilities
    references: SourceReferences
    critical_point: FluidState
    triple_point_liquid: FluidState
    triple_point_vapor: FluidState

    def __new__(
        cls,
        name,
        method="coolprop",
        backend=None,
        identifier=None,
        *,
        table_options=None,
        model_options=None,
        **options,
    ):
        if cls is not Fluid:
            raise TypeError(
                "Construct Fluid directly, or subclass a concrete evaluator "
                "when extending a fluid model."
            )
        if not isinstance(method, str) or method not in _METHOD_ROUTES:
            raise InvalidConfigurationError(
                fluid_name=name,
                request="method",
                reason=f"Unknown fluid evaluation method {method!r}; choose a supported method.",
                supported=tuple(_METHOD_ROUTES),
            )
        provider, model = _METHOD_ROUTES[method]
        if options:
            option = next(iter(options))
            reason = (
                "Choose method instead of separate provider/model selectors."
                if option in {"provider", "model"}
                else (
                    "Set the numerical error policy on get_state using errors; "
                    "source-specific legacy options belong to FluidCoolProp."
                    if option == "exceptions"
                    else "Unknown fluid construction option."
                )
            )
            raise InvalidConfigurationError(
                fluid_name=name,
                provider=provider,
                model=model,
                request=option,
                reason=reason,
                supported=(
                    "name",
                    "method",
                    "backend",
                    "identifier",
                    "table_options",
                    "model_options",
                ),
            )
        return _factory.create_fluid(
            name=name,
            provider=provider,
            backend=backend,
            identifier=identifier,
            model=model,
            table_options=table_options,
            model_options=model_options,
        )

    def get_state(
        self,
        input_type,
        prop_1,
        prop_2,
        *,
        generalize_quality=False,
        supersaturation=False,
        errors="raise",
        **options,
    ) -> FluidState:
        """Evaluate through the selected concrete object's common interface.

        This declaration documents the runtime/type-checking contract; actual
        objects use their own implementation directly through virtual subclass
        registration, without inheriting this method.
        """
        raise NotImplementedError("Call get_state on the constructed fluid object.")
