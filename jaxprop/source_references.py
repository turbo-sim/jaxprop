"""Immutable source-EOS reference data, independent of numerical evaluators.

Only scalar Python metadata is retained. Reference ``FluidState`` arrays are
materialized on explicit access, so these records are safe static JAX metadata.
Every initialization probes a fresh source anchor: CoolProp reference settings
are process-global and a cache keyed only by fluid name would become stale.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math

import CoolProp.CoolProp as CP

from .configuration import FrozenMapping
from .errors import InvalidConfigurationError, UnavailableReferencePointError
from .fluid_api import SOURCE_PROPERTIES
from .fluid_state import FluidState

__all__ = [
    "ReferencePoint",
    "SourceReferences",
    "SourceReferencesMixin",
    "initialize_source_references",
    "source_reference_metadata",
    "source_reference_identity",
    "reference_anchor",
    "composite_source_references",
]

_SOURCE_ERRORS = (
    ValueError,
    RuntimeError,
    NotImplementedError,
    AttributeError,
    OverflowError,
    OSError,
)
_POINT_NAMES = ("critical_point", "triple_point_liquid", "triple_point_vapor")
_CORE = ("pressure", "temperature", "density", "enthalpy", "entropy", "internal_energy")


def reference_anchor(state):
    """Probe h/s at deterministic coordinates independent of their offsets."""
    attempts = []
    try:
        temperature = float(state.T_critical()) * 1.2
        density = float(state.rhomass_critical()) * 0.1
        if not all(
            math.isfinite(value) and value > 0 for value in (temperature, density)
        ):
            raise ValueError("Nonfinite or nonpositive critical coordinates")
        candidates = [(CP.DmassT_INPUTS, density, temperature)]
    except _SOURCE_ERRORS as error:
        attempts.append(error)
        candidates = []
    candidates.extend(
        (CP.PT_INPUTS, pressure, temperature)
        for pressure, temperature in (
            (101325.0, 300.0),
            (101325.0, 500.0),
            (1e6, 500.0),
            (1e5, 1000.0),
        )
    )
    for input_pair, first, second in candidates:
        try:
            state.update(input_pair, first, second)
            anchor = {
                "input_pair": (
                    "DmassT_INPUTS" if input_pair == CP.DmassT_INPUTS else "PT_INPUTS"
                ),
                "inputs": [float(first), float(second)],
                "density": float(state.rhomass()),
                "temperature": float(state.T()),
                "pressure": float(state.p()),
                "enthalpy": float(state.hmass()),
                "entropy": float(state.smass()),
            }
            if not all(
                math.isfinite(anchor[key])
                for key in ("density", "temperature", "pressure", "enthalpy", "entropy")
            ):
                raise ValueError("Reference anchor contains nonfinite core properties")
            if any(anchor[key] <= 0 for key in ("density", "temperature", "pressure")):
                raise ValueError(
                    "Reference anchor has nonpositive physical coordinates"
                )
            return anchor
        except _SOURCE_ERRORS as error:
            attempts.append(error)
    raise ValueError(
        "No deterministic source-reference anchor could be evaluated"
    ) from attempts[-1]


def _version(key, fallback):
    try:
        return str(CP.get_global_param_string(key))
    except _SOURCE_ERRORS:
        return fallback


def _backend_name(state, backend):
    if backend is not None:
        return backend
    name = state.backend_name()
    return {"HelmholtzEOSBackend": "HEOS", "REFPROPBackend": "REFPROP"}.get(name, name)


def _identity_payload(state, anchor, backend):
    names = tuple(state.fluid_names())
    # REFPROP preserves input aliases in fluid_names; canonicalize catalogue
    # names while retaining native-only and predefined mixture selections.
    canonical_names = []
    for name in names:
        try:
            canonical = (
                name
                if name.upper().endswith(".MIX")
                else CP.get_fluid_param_string(name, "name")
            )
        except _SOURCE_ERRORS:
            canonical = name
        canonical_names.append(canonical)
    names = tuple(canonical_names)
    fractions = tuple(float(value) for value in state.get_mole_fractions())
    if (
        not names
        or not fractions
        or not all(math.isfinite(value) and value >= 0 for value in fractions)
    ):
        raise ValueError("The source did not provide a valid composition")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The source composition does not sum to one")
    version = _version("version", "unknown")
    backend_version = (
        _version("REFPROP_version", "unknown")
        if "REFPROP" in backend.upper()
        else version
    )
    payload = {
        "fluid_names": list(names),
        "mole_fractions": list(fractions),
        "backend": backend,
        "backend_version": backend_version,
        "coolprop_version": version,
        "coolprop_git_revision": _version("gitrevision", "unknown"),
        "reference_anchor": anchor,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    reference_id = ("real_fluid", "coolprop-source-v1", digest)
    payload["reference_id"] = reference_id
    return payload


def source_reference_identity(state, anchor=None, *, backend=None):
    """Return the canonical source/model identity used by states and caches."""
    return _identity_payload(
        state,
        reference_anchor(state) if anchor is None else anchor,
        _backend_name(state, backend),
    )["reference_id"]


def source_reference_metadata(name, backend="HEOS", *, state=None):
    """Return canonical composition/version/active-reference metadata.

    Supplying a newly constructed AbstractState avoids duplicate initialization;
    the function still evaluates the actual reference anchor on every call.
    """
    try:
        state = CP.AbstractState(backend, name) if state is None else state
        return _identity_payload(
            state, reference_anchor(state), _backend_name(state, backend)
        )
    except _SOURCE_ERRORS as error:
        raise InvalidConfigurationError(
            fluid_name=name,
            provider="coolprop",
            model="real_fluid",
            request="source_reference",
            reason=str(error),
        ) from error


@dataclass(frozen=True)
class ReferencePoint:
    name: str
    available: bool
    reason: str | None
    classification: str
    interpretation: str
    source: FrozenMapping
    reference_id: tuple
    reported_coordinates: FrozenMapping = field(default_factory=FrozenMapping)
    evaluation_coordinates: FrozenMapping = field(default_factory=FrozenMapping)
    evaluation_offsets: FrozenMapping = field(default_factory=FrozenMapping)
    properties: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self):
        for name in (
            "source",
            "reported_coordinates",
            "evaluation_coordinates",
            "evaluation_offsets",
            "properties",
        ):
            object.__setattr__(self, name, FrozenMapping(getattr(self, name)))

    @property
    def backend(self):
        return self.source["backend"]

    @property
    def source_fluid(self):
        return "&".join(self.source["fluid_names"])

    @property
    def reference_convention(self):
        return self.source["reference_convention"]

    @property
    def state(self):
        if not self.available:
            raise UnavailableReferencePointError(
                fluid_name=self.source_fluid,
                provider="native" if self.classification == "composite" else "coolprop",
                model=(
                    "binary_mixture"
                    if self.classification == "composite"
                    else "real_fluid"
                ),
                request=self.name,
                reason=self.reason or "The source reference is unavailable.",
            )
        return FluidState(
            fluid_name=self.source_fluid,
            provider="coolprop",
            model="real_fluid",
            reference_id=self.reference_id,
            available_properties=frozenset(self.properties),
            **{
                name: float("nan") if value is None else value
                for name, value in self.properties.items()
            },
        )


@dataclass(frozen=True)
class SourceReferences(Mapping):
    critical: ReferencePoint
    triple_liquid: ReferencePoint
    triple_vapor: ReferencePoint
    source: FrozenMapping
    anchor: FrozenMapping
    reference_id: tuple
    components: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "source", FrozenMapping(self.source))
        object.__setattr__(self, "anchor", FrozenMapping(self.anchor))
        object.__setattr__(self, "components", tuple(self.components))

    def __getitem__(self, key):
        try:
            return {
                "critical_point": self.critical,
                "triple_point_liquid": self.triple_liquid,
                "triple_point_vapor": self.triple_vapor,
            }[key]
        except KeyError:
            raise KeyError(key) from None

    def __iter__(self):
        return iter(_POINT_NAMES)

    def __len__(self):
        return 3


class SourceReferencesMixin:
    @property
    def references(self):
        return self._references

    def _reference_state(self, key):
        record = self.references[key]
        if not record.available:
            raise UnavailableReferencePointError(
                fluid_name=self.name,
                provider=self.provider,
                model=self.model,
                request=key,
                reason=record.reason or "The source reference is unavailable.",
                supported=tuple(
                    name for name, point in self.references.items() if point.available
                ),
            )
        return record.state

    @property
    def critical_point(self):
        return self._reference_state("critical_point")

    @property
    def triple_point_liquid(self):
        return self._reference_state("triple_point_liquid")

    @property
    def triple_point_vapor(self):
        return self._reference_state("triple_point_vapor")


def _optional(function):
    try:
        value = float(function())
        return value if math.isfinite(value) else None
    except _SOURCE_ERRORS + (ZeroDivisionError,):
        return None


def _reference_properties(state, quality=None, critical=False):
    methods = {
        "pressure": "p",
        "temperature": "T",
        "density": "rhomass",
        "enthalpy": "hmass",
        "entropy": "smass",
        "internal_energy": "umass",
        "isobaric_heat_capacity": "cpmass",
        "isochoric_heat_capacity": "cvmass",
        "compressibility_factor": "compressibility_factor",
        "speed_of_sound": "speed_sound",
        "viscosity": "viscosity",
        "conductivity": "conductivity",
        "isothermal_compressibility": "isothermal_compressibility",
        "isobaric_expansion_coefficient": "isobaric_expansion_coefficient",
        "surface_tension": "surface_tension",
    }
    props = {
        name: _optional(lambda method=method: getattr(state, method)())
        for name, method in methods.items()
    }

    def derive(name, function, *dependencies):
        props[name] = (
            _optional(lambda: function(*(props[key] for key in dependencies)))
            if all(props[key] is not None for key in dependencies)
            else None
        )

    derive(
        "heat_capacity_ratio",
        lambda cp, cv: cp / cv,
        "isobaric_heat_capacity",
        "isochoric_heat_capacity",
    )
    derive(
        "isothermal_bulk_modulus", lambda value: 1 / value, "isothermal_compressibility"
    )
    derive(
        "isentropic_bulk_modulus",
        lambda rho, a: rho * a * a,
        "density",
        "speed_of_sound",
    )
    derive(
        "isentropic_compressibility", lambda value: 1 / value, "isentropic_bulk_modulus"
    )
    derive(
        "gruneisen",
        lambda alpha, kappa, rho, cv: alpha / (kappa * rho * cv),
        "isobaric_expansion_coefficient",
        "isothermal_compressibility",
        "density",
        "isochoric_heat_capacity",
    )
    derive(
        "joule_thomson",
        lambda alpha, temperature, rho, cp: (alpha * temperature - 1) / (rho * cp),
        "isobaric_expansion_coefficient",
        "temperature",
        "density",
        "isobaric_heat_capacity",
    )
    derive(
        "isothermal_joule_thomson",
        lambda coefficient, cp: -coefficient * cp,
        "joule_thomson",
        "isobaric_heat_capacity",
    )
    props["is_two_phase"] = float(quality is not None)
    props["quality_mass"] = props["quality_volume"] = quality
    if critical:
        # Critical response functions are singular; near-critical fallback
        # numbers must not masquerade as finite critical susceptibilities.
        for name in (
            "isobaric_heat_capacity",
            "isochoric_heat_capacity",
            "heat_capacity_ratio",
            "isothermal_compressibility",
            "isobaric_expansion_coefficient",
            "isothermal_bulk_modulus",
            "isentropic_bulk_modulus",
            "isentropic_compressibility",
            "gruneisen",
            "joule_thomson",
            "isothermal_joule_thomson",
        ):
            props[name] = None
    return {name: props.get(name) for name in SOURCE_PROPERTIES}


def _coordinates(state):
    return {
        "temperature": float(state.T()),
        "pressure": float(state.p()),
        "density": float(state.rhomass()),
    }


def _evaluate_point(state, point_name, classification, source, reference_id):
    interpretation = (
        "critical_point"
        if point_name == "critical_point"
        else (
            "triple_point" if classification == "pure" else "lower_saturation_endpoint"
        )
    )
    reported = {}
    try:
        if classification in ("mixture", "unknown"):
            raise ValueError(
                "This source has no unambiguous pure or pseudo-pure reference point."
            )
        critical = point_name == "critical_point"
        quality = (
            None if critical else (0.0 if point_name == "triple_point_liquid" else 1.0)
        )
        if critical:
            reported = {
                "temperature": float(state.T_critical()),
                "pressure": float(state.p_critical()),
                "density": float(state.rhomass_critical()),
            }
            if not all(
                math.isfinite(value) and value > 0 for value in reported.values()
            ):
                raise ValueError(
                    "Source critical coordinates are nonfinite or nonpositive."
                )
            attempts = []
            for pair, first, second in (
                (CP.DmassT_INPUTS, reported["density"], reported["temperature"]),
                (CP.DmassT_INPUTS, reported["density"], reported["temperature"] - 1e-5),
                (CP.PT_INPUTS, reported["pressure"], reported["temperature"] - 1e-5),
            ):
                try:
                    state.update(pair, first, second)
                    props = _reference_properties(state, critical=True)
                    if not all(props[name] is not None for name in _CORE):
                        raise ValueError(
                            "Reference evaluation has unavailable core properties."
                        )
                    break
                except _SOURCE_ERRORS as error:
                    attempts.append(error)
            else:
                raise ValueError(
                    "Critical-point evaluation failed: " + str(attempts[-1])
                ) from attempts[-1]
        else:
            temperature = float(state.Ttriple())
            reported = {"temperature": temperature, "quality_mass": quality}
            state.update(CP.QT_INPUTS, quality, temperature)
            props = _reference_properties(state, quality=quality)
            if not all(props[name] is not None for name in _CORE):
                raise ValueError(
                    "Reference evaluation has unavailable core properties."
                )
            # Bubble/dew pressures are independently evaluated, including Air.
            reported.update(_coordinates(state))
        evaluation = _coordinates(state)
        if quality is not None:
            evaluation["quality_mass"] = quality
        if not all(
            math.isfinite(value) and value > 0
            for name, value in evaluation.items()
            if name != "quality_mass"
        ):
            raise ValueError(
                "Reference evaluation coordinates are nonfinite or nonpositive."
            )
        offsets = {
            name: evaluation[name] - value
            for name, value in reported.items()
            if name in evaluation
        }
        return ReferencePoint(
            point_name,
            True,
            None,
            classification,
            interpretation,
            source,
            reference_id,
            reported,
            evaluation,
            offsets,
            props,
        )
    except _SOURCE_ERRORS as error:
        return ReferencePoint(
            point_name,
            False,
            f"{type(error).__name__}: {error}",
            classification,
            interpretation,
            source,
            reference_id,
            reported,
        )


@lru_cache(maxsize=128)
def _intern_references(references):
    return references


def initialize_source_references(name, backend="HEOS", *, state=None):
    """Initialize independently available source critical/lower-endpoint records.

    An evaluator may supply its freshly constructed source state so ordinary
    evaluation and reference records capture the same convention, even if the
    process-global convention changes during construction. This is setup-only:
    callers must not pass a state serving an in-flight numerical evaluation.
    """
    try:
        state = CP.AbstractState(backend, name) if state is None else state
    except _SOURCE_ERRORS as error:
        raise InvalidConfigurationError(
            fluid_name=name,
            provider="coolprop",
            model="real_fluid",
            request="source",
            reason=str(error),
        ) from error
    metadata = source_reference_metadata(name, backend, state=state)
    reference_id = metadata.pop("reference_id")
    anchor = metadata.pop("reference_anchor")
    metadata["reference_convention"] = FrozenMapping(
        {
            "description": "Active CoolProp enthalpy/entropy convention; detected from actual source anchor values.",
            "source_reference_id": reference_id,
        }
    )
    source = FrozenMapping(metadata)
    if (
        len(source["fluid_names"]) > 1
        or len(source["mole_fractions"]) > 1
        or any(name.upper().endswith(".MIX") for name in source["fluid_names"])
    ):
        classification = "mixture"
    else:
        try:
            purity = state.fluid_param_string("pure")
        except _SOURCE_ERRORS:
            try:
                purity = CP.get_fluid_param_string(source["fluid_names"][0], "pure")
            except _SOURCE_ERRORS:
                try:
                    purity = CP.get_fluid_param_string(
                        state.fluid_param_string("CAS"), "pure"
                    )
                except _SOURCE_ERRORS:
                    purity = "unknown"
        classification = {"true": "pure", "false": "pseudo_pure"}.get(
            purity.lower(), "unknown"
        )
    points = [
        _evaluate_point(state, point, classification, source, reference_id)
        for point in _POINT_NAMES
    ]
    return _intern_references(
        SourceReferences(*points, source, FrozenMapping(anchor), reference_id)
    )


def _binary_model_reference_id(component_reference_ids, mass_fractions):
    """One physical-model identity shared by native and tabulated binaries."""
    return (
        "binary_mixture",
        tuple(component_reference_ids),
        ("mass_fractions", tuple(float(w) for w in mass_fractions)),
        "common_pT_no_mass_exchange",
        "pressure_temperature_equilibrium_acoustics",
    )


def composite_source_references(
    name,
    backend,
    component_references,
    mixture_ratio=None,
    *,
    mass_fractions=None,
    component_reference_ids=None,
):
    """Record that a component mixing model has no assumed unique phase points."""
    components = tuple(component_references)
    if mass_fractions is None:
        reference_id = (
            "binary_mixture",
            tuple(component.reference_id for component in components),
            ("mass_ratio", float(mixture_ratio)),
        )
    else:
        # Component model identities distinguish calibrated PG models from
        # source-equivalent evaluators without inventing an aggregate phase EOS.
        reference_id = _binary_model_reference_id(
            component_reference_ids or (c.reference_id for c in components),
            mass_fractions,
        )
    source = FrozenMapping(
        {
            "fluid_names": (name,),
            "mole_fractions": (),
            "backend": backend,
            "coolprop_version": _version("version", "unknown"),
            "classification": "composite",
            "reference_convention": FrozenMapping(
                {
                    "description": "Mass-weighted component enthalpy/entropy conventions.",
                    "source_reference_id": reference_id,
                    "component_reference_ids": tuple(
                        component.reference_id for component in components
                    ),
                }
            ),
        }
    )
    reason = "A component mixing model has no assumed unique critical or triple point; inspect its component source references."
    points = [
        ReferencePoint(
            point,
            False,
            reason,
            "composite",
            "not_defined_for_composite",
            source,
            reference_id,
        )
        for point in _POINT_NAMES
    ]
    return _intern_references(
        SourceReferences(*points, source, FrozenMapping(), reference_id, components)
    )
