"""Setup-only source scratch state for reference-consistent helper calculations."""

import CoolProp.CoolProp as CP

from ..errors import InvalidConfigurationError
from ..source_references import source_reference_metadata


class SourceEvaluationContext:
    """Keep helper calculations on the configured source's reference basis.

    CoolProp snapshots the active enthalpy/entropy convention when creating an
    AbstractState. Preparing this scratch state at construction avoids both
    repeated construction and accidentally adopting a later global convention.
    The owning source evaluator holds its batch lock while using this mutable
    scratch state, including evaluation from JAX host callbacks.
    """

    def __init__(self, primary_state, references, *, name, backend):
        self.references = references
        self.critical_entropy = references.critical.properties.get("entropy")
        # Bounds belong to the selected source, including separate bubble/dew
        # lower endpoints for pseudo-pure fluids. Reuse initialization records
        # rather than flashing these fixed endpoints on every diagnostic call.
        critical = references.critical.reported_coordinates
        liquid = references.triple_liquid.reported_coordinates
        vapor = references.triple_vapor.reported_coordinates
        self.critical_temperature = critical.get("temperature", float("nan"))
        self.critical_pressure = critical.get("pressure", float("nan"))
        self.critical_density = critical.get("density", float("nan"))
        self.triple_temperature = liquid.get(
            "temperature", vapor.get("temperature", float("nan"))
        )
        self.triple_pressure_liquid = liquid.get("pressure", float("nan"))
        self.triple_pressure_vapor = vapor.get("pressure", float("nan"))
        try:
            auxiliary = CP.AbstractState(backend, name)
            if len(primary_state.get_mole_fractions()) > 1:
                auxiliary.set_mole_fractions(primary_state.get_mole_fractions())
            identity = source_reference_metadata(name, backend, state=auxiliary)
        except (ValueError, RuntimeError, OSError, AttributeError) as error:
            raise InvalidConfigurationError(
                fluid_name=name,
                provider="coolprop",
                model="real_fluid",
                request="source_reference",
                reason=f"Cannot prepare a reference-compatible helper state: {error}",
            ) from error
        if identity["reference_id"] != references.reference_id:
            raise InvalidConfigurationError(
                fluid_name=name,
                provider="coolprop",
                model="real_fluid",
                request="source_reference",
                reason="The source convention changed while preparing its helper state.",
            )
        self.auxiliary_state = auxiliary


def helper_state(abstract_state, source_context=None):
    """Return setup scratch state, or a fresh legacy standalone helper state."""
    if source_context is not None:
        return source_context.auxiliary_state
    fluids = abstract_state.fluid_names()
    if len(fluids) != 1:
        raise ValueError(f"Expected one fluid, got {fluids}")
    return CP.AbstractState(abstract_state.backend_name(), fluids[0])
