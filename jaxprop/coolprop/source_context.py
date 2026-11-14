"""Setup-only source scratch state for reference-consistent helper calculations."""

import CoolProp.CoolProp as CP
import numpy as np

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
        self.entropy_scale = primary_state.gas_constant() / primary_state.molar_mass()
        self.is_pseudo_pure = references.critical.classification == "pseudo_pure"
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
    """Return configured scratch, or verify a legacy helper's reference basis.

    A new CoolProp object snapshots the *current* global convention, which can
    differ from an already configured object. The ordinary configured path is
    prepared once and does no repeated verification. Standalone legacy calls
    must reject a mismatching helper rather than mix entropy/energy gauges.
    The primary object is never flashed by this compatibility check.
    """
    if source_context is not None:
        return source_context.auxiliary_state
    # A homogeneous diagnostic snapshot may intentionally expose different
    # h/s than its underlying equilibrium source at the same density/T.
    # Check the convention against the raw source, never against that model.
    reference_state = getattr(abstract_state, "_reference_source_state", abstract_state)
    fluids = reference_state.fluid_names()
    if len(fluids) != 1:
        raise ValueError(f"Expected one fluid, got {fluids}")
    auxiliary = CP.AbstractState(reference_state.backend_name(), fluids[0])
    recipes = [(CP.DmassT_INPUTS, reference_state.rhomass(), reference_state.T())]
    quality = reference_state.Q()
    if reference_state.phase() == CP.iphase_twophase and 0 <= quality <= 1:
        # Pseudo-pure source maps need not use the same interpolation geometry
        # for DmassT and PQ. Accept either exact source reconstruction.
        recipes.append((CP.PQ_INPUTS, reference_state.p(), quality))
    compatible = False
    for recipe in recipes:
        try:
            auxiliary.update(*recipe)
            compatible = all(
                np.isclose(
                    getattr(auxiliary, field)(),
                    getattr(reference_state, field)(),
                    rtol=2e-8,
                    atol=1e-7,
                )
                for field in ("hmass", "smass")
            )
            if compatible:
                break
        except (ValueError, RuntimeError, NotImplementedError):
            pass
    if not compatible:
        raise InvalidConfigurationError(
            fluid_name=fluids[0],
            provider="coolprop",
            model="real_fluid",
            request="source_reference",
            reason=(
                "A newly created helper has a different enthalpy/entropy "
                "reference convention. Reuse a configured Fluid object "
                "and its prepared source context."
            ),
        )
    return auxiliary
