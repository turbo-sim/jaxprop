"""Construct configured fluid evaluators without application-side adapters."""

from collections.abc import Mapping
from numbers import Integral, Real
import math
import os

from .configuration import build_configuration
from .errors import InvalidConfigurationError, JaxPropError

__all__ = ["create_fluid"]

_ROUTES = {
    "real_fluid": frozenset({"coolprop", "coolprop_jax", "bicubic"}),
    "perfect_gas": frozenset({"native"}),
    "binary_mixture": frozenset({"native"}),
}
_TABLE_REQUIRED = frozenset({"h_min", "h_max", "p_min", "p_max", "N_h", "N_p"})
_TABLE_ALLOWED = _TABLE_REQUIRED | {
    "N_p_sat",
    "coarse_step",
    "inverse_max_steps",
    "inverse_tolerance",
    "gradient_method",
    "table_dir",
    "table_name",
}


def create_fluid(
    name,
    provider="coolprop",
    backend=None,
    identifier=None,
    *,
    model="real_fluid",
    table_options=None,
    model_options=None,
):
    """Create the requested physical model and numerical provider.

    Provider selectors are exact lowercase strings. The backend is the source
    EOS and defaults to HEOS for single-source models. Binary component-object
    construction remains unavailable until its numerical model is consolidated.
    """
    # Primitive metadata validation must precede backend initialization. Options
    # are validated below before recursively snapshotting their accepted values.
    build_configuration(name, provider, model, backend, identifier)
    context = dict(fluid_name=name, provider=provider, model=model)

    def invalid(request, reason, supported=()):
        raise InvalidConfigurationError(
            request=request, reason=reason, supported=supported, **context
        )

    if model not in _ROUTES:
        invalid("model", "Unknown physical model.", _ROUTES)
    if provider not in set().union(*_ROUTES.values()):
        invalid(
            "provider", "Unknown numerical provider.", set().union(*_ROUTES.values())
        )
    if provider not in _ROUTES[model]:
        invalid(
            "provider/model",
            "The provider cannot evaluate this physical model.",
            _ROUTES[model],
        )

    def options(value, request):
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            invalid(request, "Options must be a mapping or None.")
        result = dict(value)
        if any(not isinstance(key, str) for key in result):
            invalid(request, "Option names must be strings.")
        return result

    table = options(table_options, "table_options")
    physical = options(model_options, "model_options")
    if provider != "bicubic" and table:
        invalid("table_options", "Table settings apply only to the bicubic provider.")

    if model == "binary_mixture":
        if backend is not None:
            invalid(
                "backend",
                "A composite uses its component source backends; an override is unsupported.",
            )
        expected = {"components", "mass_fractions"}
        if set(physical) != expected:
            invalid(
                "model_options",
                "A binary model requires only components and mass_fractions.",
                expected,
            )
        components = physical["components"]
        if (
            not isinstance(components, (tuple, list))
            or len(components) != 2
            or any(
                not callable(getattr(component, "get_state", None))
                for component in components
            )
        ):
            invalid(
                "components", "Provide exactly two configured component fluid objects."
            )
        fractions = physical["mass_fractions"]
        if (
            not isinstance(fractions, (tuple, list))
            or len(fractions) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
                for value in fractions
            )
        ):
            invalid(
                "mass_fractions",
                "Provide two finite, nonnegative scalar mass fractions.",
            )
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
            invalid(
                "mass_fractions",
                "Mass fractions must sum to one without normalization.",
            )
        invalid(
            "model",
            "Component-object binary construction awaits consolidation of the binary-mixture model.",
        )

    source_backend = "HEOS" if backend is None else backend
    if model == "perfect_gas":
        unknown = set(physical) - {"T_ref", "p_ref"}
        if unknown:
            invalid(
                "model_options",
                f"Unknown perfect-gas settings: {sorted(unknown)}.",
                ("T_ref", "p_ref"),
            )
        for key, value in physical.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                invalid(
                    key,
                    "Calibration reference coordinates must be finite positive real scalars.",
                )
    elif physical:
        invalid(
            "model_options", "Model settings do not apply to source-EOS real fluids."
        )

    if provider == "bicubic":
        unknown = set(table) - _TABLE_ALLOWED
        if unknown:
            invalid(
                "table_options",
                f"Unknown or unsupported table settings: {sorted(unknown)}.",
                _TABLE_ALLOWED,
            )
        missing = _TABLE_REQUIRED - set(table)
        if missing:
            invalid(
                "table_options",
                f"Missing required table settings: {sorted(missing)}.",
                _TABLE_REQUIRED,
            )
        for key in ("h_min", "h_max", "p_min", "p_max"):
            value = table[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                invalid(key, "Table bounds must be finite real scalars.")
        if table["h_min"] >= table["h_max"]:
            invalid("h_min/h_max", "Require h_min < h_max.")
        if not 0 < table["p_min"] < table["p_max"]:
            invalid("p_min/p_max", "Require 0 < p_min < p_max.")
        for key in ("N_h", "N_p", "N_p_sat", "coarse_step"):
            if key not in table:
                continue
            value = table[key]
            minimum = 1 if key == "coarse_step" else 2
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                invalid(key, f"Require an integer >= {minimum}.")
        gradient_method = table.get("gradient_method", "central")
        if not isinstance(gradient_method, str) or gradient_method not in {
            "forward",
            "central",
        }:
            invalid(
                "gradient_method",
                "Unsupported table gradient method.",
                ("forward", "central"),
            )
        if "table_dir" in table and not isinstance(
            table["table_dir"], (str, os.PathLike)
        ):
            invalid("table_dir", "Provide a directory path.")
        if table.get("table_name") is not None and (
            not isinstance(table["table_name"], str) or not table["table_name"]
        ):
            invalid("table_name", "Provide a nonempty table name or None.")

    build_configuration(
        name,
        provider,
        model,
        source_backend,
        identifier,
        table_options=table,
        model_options=physical,
    )
    try:
        if provider == "coolprop":
            from .coolprop.fluid_properties import FluidCoolProp

            return FluidCoolProp(
                name=name, backend=source_backend, identifier=identifier
            )
        if provider == "coolprop_jax":
            from .coolprop.jax_wrapper import FluidJAX

            return FluidJAX(name=name, backend=source_backend, identifier=identifier)
        if provider == "bicubic":
            from .bicubic.bicubic_interpolation import FluidBicubic

            return FluidBicubic(
                fluid_name=name, backend=source_backend, identifier=identifier, **table
            )
        from .perfect_gas.perfect_gas import FluidPerfectGas

        return FluidPerfectGas(
            name=name, backend=source_backend, identifier=identifier, **physical
        )
    except JaxPropError:
        raise
    except OSError as error:
        if provider == "bicubic":
            raise
        raise InvalidConfigurationError(
            request="construction",
            reason=f"Source initialization failed: {error}",
            **context,
        ) from error
    except (TypeError, ValueError, RuntimeError) as error:
        raise InvalidConfigurationError(
            request="construction",
            reason=f"Fluid construction failed: {error}",
            **context,
        ) from error
