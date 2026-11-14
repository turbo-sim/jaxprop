"""Eager phase/reference plotting through configured real-fluid sources.

These diagrams describe the source EOS, including states outside a bicubic
table's numerical domain. They do not turn an unsupported phase operation into
a state-evaluation capability, or add source calls to numerical evaluators.
"""

from .errors import InvalidConfigurationError, UnsupportedOperationError

__all__ = ["plot_phase_diagram"]


def plot_phase_diagram(fluid, x_prop="s", y_prop="T", axes=None, **options):
    """Plot the selected real fluid's source EOS and return ``(figure, axes)``.

    Direct-source and callback objects reuse their configured source. A bicubic
    real-fluid model prepares a plotting-only source under the table's exact
    reference/provenance convention. Repeated calls on the same axes reuse that
    source and update its artists, independently for each named stream.

    Perfect-gas and binary parcel models have no phase diagram of their own and
    raise ``UnsupportedOperationError``. Their source-reference records remain
    available explicitly through ``fluid.references``. This function is eager
    and must be called outside JAX transformations and numerical timing loops.
    """
    provider, model = getattr(fluid, "provider", None), getattr(fluid, "model", None)
    context = dict(
        fluid_name=getattr(fluid, "name", None), provider=provider, model=model
    )
    if model != "real_fluid" or provider not in {"coolprop", "coolprop_jax", "bicubic"}:
        raise UnsupportedOperationError(
            request="plot_phase_diagram",
            reason="A phase diagram requires a supported single real-fluid source EOS; this physical model has no own saturation envelope.",
            supported=("real-fluid source-EOS diagrams", "fluid.references"),
            **context,
        )
    if provider == "coolprop":
        source = fluid
    elif provider == "coolprop_jax":
        source = fluid.fluid
    else:
        if fluid.mixture_ratio is not None:
            raise UnsupportedOperationError(
                request="plot_phase_diagram",
                reason="A binary parcel table has no single source saturation envelope; plot its component source fluids explicitly.",
                **context,
            )
        key = (
            fluid.name,
            fluid.backend,
            fluid.identifier,
            fluid.references.reference_id,
        )
        sources = (
            getattr(axes, "_jaxprop_phase_sources", {}) if axes is not None else {}
        )
        source = sources.get(key)
        if source is None:
            from .bicubic.cache_metadata import source_signature
            from .coolprop.fluid_properties import FluidCoolProp

            expected = fluid.metadata["source_signature"]
            if (
                not expected.get("deterministic_cache_identity", False)
                or source_signature(fluid.name, fluid.backend) != expected
            ):
                raise InvalidConfigurationError(
                    request="plot_phase_diagram",
                    reason="The current source definition/reference convention does not match the bicubic table; a source-EOS diagram cannot be reconstructed safely.",
                    **context,
                )
            source = FluidCoolProp(
                fluid.name, fluid.backend, identifier=fluid.identifier
            )
            if source.references.reference_id != fluid.references.reference_id:
                raise InvalidConfigurationError(
                    request="plot_phase_diagram",
                    reason="The source reference convention changed while preparing the diagram.",
                    **context,
                )
        figure, axes = source.plot_phase_diagram(
            x_prop=x_prop, y_prop=y_prop, axes=axes, **options
        )
        sources = getattr(axes, "_jaxprop_phase_sources", {})
        sources[key] = source
        axes._jaxprop_phase_sources = sources
        return figure, axes
    return source.plot_phase_diagram(x_prop=x_prop, y_prop=y_prop, axes=axes, **options)
