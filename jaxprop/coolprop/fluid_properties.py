import difflib
import hashlib
import json
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import pysolver_view as psv
import math
from scipy.optimize import root_scalar
from threading import RLock
from numbers import Integral

from functools import lru_cache

from . import core_calculations as core
from ..errors import (
    InvalidConfigurationError,
    InvalidStateError,
    OutOfDomainError,
    CalculationError,
    ConvergenceError,
    UnsupportedOperationError,
    JaxPropError,
    UnsupportedExecutionModeError,
    UnsupportedOptionError,
    UnsupportedPropertyError,
)
from ..property_schema import ALIAS_TO_CANONICAL
from ..configuration import FluidCapabilities, FluidDomain, build_configuration
from ..source_references import (
    SourceReferencesMixin,
    initialize_source_references,
    composite_source_references,
)
from ..fluid_api import (
    SOURCE_PROPERTIES,
    DIAGNOSTIC_PROPERTIES,
    LEGACY_MIXTURE_PROPERTIES,
    SOURCE_PAIRS,
    ReadonlyConfigurationMixin,
    enabled_properties,
    validate_request,
    StatusCode,
)
from ..helpers_props import (
    FluidState,
    INPUT_TYPE_MAP,
    LABEL_MAPPING,
    PROPERTIES_CANONICAL,
)

_SOURCE_INPUT_FIELDS = {
    CP.PT_INPUTS: ("pressure", "temperature"),
    CP.HmassSmass_INPUTS: ("enthalpy", "entropy"),
    CP.HmassP_INPUTS: ("enthalpy", "pressure"),
    CP.PSmass_INPUTS: ("pressure", "entropy"),
    CP.DmassHmass_INPUTS: ("density", "enthalpy"),
    CP.DmassP_INPUTS: ("density", "pressure"),
    CP.DmassT_INPUTS: ("density", "temperature"),
    CP.DmassSmass_INPUTS: ("density", "entropy"),
    CP.PQ_INPUTS: ("pressure", "quality_mass"),
    CP.QT_INPUTS: ("quality_mass", "temperature"),
    CP.SmassT_INPUTS: ("entropy", "temperature"),
    CP.DmassUmass_INPUTS: ("density", "internal_energy"),
}
_SOURCE_CORE_FIELDS = (
    "pressure",
    "temperature",
    "density",
    "enthalpy",
    "entropy",
    "internal_energy",
)
# Library acceptance precision for arithmetic and inverse flashes at a declared
# source boundary. This is not an EOS accuracy claim or a clipping/extrapolation
# option. Reusing a recovered boundary coordinate must use the same policy.
_SOURCE_BOUNDARY_RTOL = 1e-9
_EAGER_TRACE_STATE = jax.core.get_opaque_trace_state()


@lru_cache(maxsize=128)
def _blended_reference_id(
    reference_id, phase_change, variable, onset, width, generalize_quality
):
    """Identify the constitutive interpolation independently of its source EOS.

    The bounded cache holds immutable settings only, never a source handle or a
    numerical state. A blended state must not be combined with unmodified EOS
    states under an identity that implies identical constitutive equations.
    """
    recipe = (
        "equilibrium-metastable-blend-v1",
        reference_id,
        core.PHASE_CHANGE_ALIASES[phase_change],
        variable,
        float(onset),
        float(width),
        bool(generalize_quality),
    )
    return hashlib.sha256(
        json.dumps(recipe, separators=(",", ":")).encode()
    ).hexdigest()


class FluidCoolProp(SourceReferencesMixin, ReadonlyConfigurationMixin):
    """
    Represents a fluid with various thermodynamic properties computed via CoolProp.

    This class provides a convenient interface to CoolProp for various fluid property calculations.

    Critical and triple point properties are computed upon initialization and stored internally for convenience.

    Attributes
    ----------
    name : str
        Name of the fluid.
    backend : str
        Backend used for CoolProp, default is 'HEOS'.
    exceptions : bool
        Legacy constructor option. Only True is accepted; select per-call
        ``errors="mask"`` for numerical failures without exceptions.
    converged_flag : bool
        Legacy last-call diagnostic. Use the returned state's status for the
        result of an individual call, especially with concurrent consumers.
    references : SourceReferences
        Immutable source reference records, independent of mutable evaluation
        handles. Numerical properties belong to the returned FluidState.
    critical_point : FluidState
        Properties at the fluid's critical point.
    triple_point_liquid : FluidState
        Properties at the fluid's triple point in the liquid state.
    triple_point_vapor : FluidState
        Properties at the fluid's triple point in the vapor state.

    Methods
    -------
    get_state(input_type, prop_1, prop_2):
        Set the thermodynamic state of the fluid using specified property inputs.

    Examples
    --------

    Calculating properties with Fluid.get_state()

    >>> fluid = bpy.Fluid(name="Water", backend="HEOS")
    >>> state = fluid.get_state(bpy.PT_INPUTS, 101325, 300)
    >>> print(f"Water density is {state.rho:0.2f} kg/m3 at p={state.p:0.2f} Pa and T={state.T:0.2f} K")
    Water density is 996.56 kg/m3 at p=101325.00 Pa and T=300.00 K

    >>> fluid = bpy.Fluid(name="Air", backend="HEOS")
    >>> state = fluid.get_state(bpy.PT_INPUTS, 101325, 300)
    >>> print(f"Air heat capacity ratio is {state.gamma:0.2f} at p={state.p:0.2f} Pa and T={state.T:0.2f} K")
    Air heat capacity ratio is 1.40 at p=101325.00 Pa and T=300.00 K


    Accessing critical point properties:

    >>> fluid.critical_point.p  # Retrieves critical pressure
    >>> fluid.critical_point['T']  # Retrieves critical temperature

    Accessing triple point properties:

    >>> fluid.triple_point_liquid.h  # Retrieves liquid enthalpy at the triple point
    >>> fluid.triple_point_vapor.s  # Retrieves vapor entropy at the triple point

    """

    def __getstate__(self):
        """Serialize configured physics, excluding mutable native/UI cursors.

        Both native state handles and the transaction lock are reconstructed
        after verifying the original source convention. Numerical curve caches
        remain reusable; Matplotlib artists belong to their original axes and
        are deliberately not part of a thermodynamic model snapshot.
        """
        with self._state_lock:
            identity = self._source_context.serialization_identity
            if identity is None:
                raise InvalidConfigurationError(
                    fluid_name=self.name,
                    provider=self.provider,
                    model=self.model,
                    request="serialization",
                    reason=(
                        "Lossless native-source serialization is unavailable: "
                        + self._source_context.serialization_reason
                    ),
                )
            state = self.__dict__.copy()
            for name in ("_AS", "abstract_state", "_state_lock", "_source_context"):
                state.pop(name, None)
            for name in tuple(state):
                if name.startswith("_graphic"):
                    state.pop(name)
            state["_serialized_source_identity"] = identity
            state["graphic_elements"] = {}
        return state

    def __setstate__(self, state):
        """Restore fresh native handles only under the same source convention.

        Never change process-global CoolProp references to make a pickle load.
        A mismatch reports the problem before an evaluator with stale identity
        can be used. The current native flash cursor is transient; all public
        evaluations provide their complete input pair explicitly.
        """
        from .source_context import SourceEvaluationContext

        state = state.copy()
        expected = state.pop("_serialized_source_identity", None)
        if expected is None:
            raise InvalidConfigurationError(
                fluid_name=state.get("name"),
                provider=state.get("provider", "coolprop"),
                model=state.get("model", "real_fluid"),
                request="serialization",
                reason="The serialized model has no verifiable native-source identity.",
            )
        primary = try_initialize_fluid(state["name"], state["backend"])
        context = SourceEvaluationContext(
            primary, state["_references"], name=state["name"], backend=state["backend"]
        )
        if context.serialization_identity != expected:
            raise InvalidConfigurationError(
                fluid_name=state["name"],
                provider=state["provider"],
                model=state["model"],
                request="serialization",
                reason=(
                    "The current source configuration, native definitions or files "
                    "differ from the serialized model. Restore the matching source "
                    "environment explicitly before loading it."
                ),
            )
        self.__dict__.update(state)
        self._AS = primary
        self.abstract_state = primary
        self._source_context = context
        self._state_lock = RLock()

    def __init__(
        self,
        name,
        backend="HEOS",
        exceptions=True,
        identifier=None,
        # initialize_critical=True,
        # initialize_triple=True,
    ):
        backend = "HEOS" if backend is None else backend
        if not isinstance(exceptions, (bool, np.bool_)) or not exceptions:
            raise InvalidConfigurationError(
                fluid_name=name,
                provider="coolprop",
                model="real_fluid",
                request="exceptions",
                reason="Constructor-level failure suppression is unsupported; use get_state(..., errors='mask') for per-element numerical failures.",
                supported=(True,),
            )
        configuration = build_configuration(
            name, "coolprop", "real_fluid", backend, identifier
        )
        self.name = name
        self.backend = backend
        self.identifier = identifier
        self.provider = "coolprop"
        self.model = "real_fluid"
        self._configuration = configuration
        try:
            self._AS = try_initialize_fluid(name, backend)
        except Exception as error:
            raise InvalidConfigurationError(
                fluid_name=name,
                provider=self.provider,
                model=self.model,
                request="source",
                reason=str(error),
            ) from error
        self.abstract_state = self._AS
        self._state_lock = RLock()
        self.exceptions = exceptions
        self.converged_flag = False
        self._references = initialize_source_references(name, backend, state=self._AS)
        self.reference_id = self._references.reference_id
        from .source_context import SourceEvaluationContext

        self._source_context = SourceEvaluationContext(
            self._AS, self._references, name=name, backend=backend
        )

        # Initialize variables
        self.sat_liq = None
        self.sat_vap = None
        self.spdl_liq = None
        self.spdl_vap = None
        self.pseudo_critical_line = None
        # self.quality_grid = None
        self.q_mesh = None
        self.graphic_elements = {}

        # Pressure and temperature limits
        self.p_min = 1
        self.p_max = self._AS.pmax()
        self.T_min = self._AS.Tmin()
        self.T_max = self._AS.Tmax()
        self._T_min_accepted = self.T_min * (1 - _SOURCE_BOUNDARY_RTOL)
        self._T_max_accepted = self.T_max * (1 + _SOURCE_BOUNDARY_RTOL)
        self._p_max_accepted = self.p_max * (1 + _SOURCE_BOUNDARY_RTOL)
        critical = self._references["critical_point"]
        self._saturation_T_max = (
            critical.reported_coordinates.get("temperature")
            if critical.available
            else None
        )
        self._saturation_p_max = (
            critical.reported_coordinates.get("pressure")
            if critical.available
            else None
        )
        self._qt_endpoints_only = (
            critical.classification == "pseudo_pure"
            and self._AS.backend_name() == "HelmholtzEOSBackend"
        )
        domain_description = "Positive pressure/density and the selected CoolProp source's temperature/pressure limits; declared bounds admit a relative numerical tolerance of 1e-9 without clipping. Phase flashes may have further restrictions."
        if self._qt_endpoints_only:
            domain_description += " HEOS pseudo-pure QT flashes accept only quality endpoints 0 and 1; PQ flashes also accept fractional quality."
        self._capabilities = FluidCapabilities(
            input_pairs=SOURCE_PAIRS,
            properties=SOURCE_PROPERTIES | DIAGNOSTIC_PROPERTIES,
            options={"generalize_quality", "supersaturation"},
            phase_operations={"saturation", "equilibrium", "metastable", "blending"},
            error_modes={"raise", "mask"},
            domain=FluidDomain(
                description=domain_description,
                bounds={
                    "temperature": (self.T_min, self.T_max),
                    "pressure": (0.0, self.p_max),
                },
            ),
        )

        # # Define reference state
        # self.reference_state = self.critical_point

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
    ):
        r"""
        Set the thermodynamic state of the fluid using the CoolProp low-level interface.

        This method updates the thermodynamic state of the fluid in the CoolProp ``AbstractState`` object
        using the specified input properties. It supports both scalar and array inputs for ``prop_1`` and
        ``prop_2``, automatically broadcasting them to a common shape and evaluating the fluid properties
        element-wise. The resulting thermodynamic properties are returned as arrays matching the broadcasted
        input dimensions.

        Depending on the specified inputs and fluid conditions, the calculation may involve either single-phase
        or two-phase states.

        Parameters
        ----------
        input_type : int
            The variable pair used to define the thermodynamic state. This should be one of the predefined
            CoolProp input pairs (e.g., ``PT_INPUTS`` for pressure and temperature).
        prop_1 : float or array_like
            The first property value corresponding to the input type.
        prop_2 : float or array_like
            The second property value corresponding to the input type.
        generalize_quality : bool, optional
            If True, requests the existing generalized reported-quality calculation.
        supersaturation : bool, optional
            Include available saturation-departure diagnostics. This does not
            select a metastable branch. Locally undefined fields remain NaN.
        errors : {"raise", "mask"}, optional
            Raise a typed error at the first failing row-major element, or
            preserve successful elements and encode failures in state.status.

        Returns
        -------
        FluidState
            Enabled canonical properties with the exact broadcast input shape,
            immutable identity metadata, status, and property availability.

        Raises
        ------
        JaxPropError
            Invalid or unsupported requests are rejected before source evaluation.
        CalculationError
            The source calculation fails after valid numerical inputs.
        """
        # Validate requests before any source calls, retaining typed failures.
        prop1_array, prop2_array = validate_request(
            self,
            input_type,
            prop_1,
            prop_2,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
            errors=errors,
            array_backend="numpy",
            options=options,
        )
        available = enabled_properties(self, supersaturation=supersaturation)
        results, status = self._evaluate_state_data(
            input_type,
            prop1_array,
            prop2_array,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
            errors=errors,
        )

        # Store properties in a FluidState object
        state = FluidState(
            fluid_name=self.name,
            identifier=self.identifier,
            provider=self.provider,
            model=self.model,
            reference_id=self.reference_id,
            available_properties=available,
            error_policy=errors,
            status=status,
            **results,
        )

        return state

    def _input_failure_status(self, input_type, first, second):
        """Classify numerical input arrays once, independently of CP flashes."""
        status = np.zeros(first.shape, dtype=np.int32)
        invalid = ~np.isfinite(first) | ~np.isfinite(second)
        outside = np.zeros(first.shape, dtype=bool)
        for name, values in zip(_SOURCE_INPUT_FIELDS[input_type], (first, second)):
            if name in ("pressure", "temperature", "density"):
                invalid |= values <= 0
            elif name == "quality_mass":
                invalid |= (values < 0) | (values > 1)
            if name == "temperature":
                outside |= (values < self._T_min_accepted) | (
                    values > self._T_max_accepted
                )
            elif name == "pressure":
                outside |= values > self._p_max_accepted
        if input_type == CP.QT_INPUTS and self._saturation_T_max is not None:
            outside |= second > self._saturation_T_max
        elif input_type == CP.PQ_INPUTS and self._saturation_p_max is not None:
            outside |= first > self._saturation_p_max
        if input_type == CP.QT_INPUTS and self._qt_endpoints_only:
            outside |= (first > 0) & (first < 1)
        status[invalid] = StatusCode.INVALID_INPUT
        status[outside & ~invalid] = StatusCode.OUT_OF_DOMAIN
        return status

    def _numerical_error(self, input_type, first, second, index, code, cause=None):
        request = INPUT_TYPE_MAP.get(input_type, str(input_type))
        if code == StatusCode.INVALID_INPUT:
            invalid = []
            for name, value in zip(_SOURCE_INPUT_FIELDS[input_type], (first, second)):
                if not math.isfinite(value):
                    invalid.append(f"{name} must be finite (received {value!r})")
                elif name in ("pressure", "temperature", "density") and value <= 0:
                    invalid.append(f"{name} must be positive (received {value!r})")
                elif name == "quality_mass" and not 0 <= value <= 1:
                    invalid.append(
                        f"quality_mass must lie in [0, 1] (received {value!r})"
                    )
            error_type = InvalidStateError
            reason = "; ".join(invalid)
        elif code == StatusCode.OUT_OF_DOMAIN:
            error_type = OutOfDomainError
            if cause is not None:
                reason = str(cause)
            elif (
                input_type == CP.QT_INPUTS and self._qt_endpoints_only and 0 < first < 1
            ):
                reason = "HEOS pseudo-pure QT flashes require quality_mass 0 or 1; use PQ_INPUTS for fractional quality."
            elif (
                input_type == CP.QT_INPUTS
                and self._saturation_T_max is not None
                and second > self._saturation_T_max
            ):
                reason = f"Saturation temperature cannot exceed the reported critical temperature {self._saturation_T_max} K."
            elif (
                input_type == CP.PQ_INPUTS
                and self._saturation_p_max is not None
                and first > self._saturation_p_max
            ):
                reason = f"Saturation pressure cannot exceed the reported critical pressure {self._saturation_p_max} Pa."
            else:
                reason = f"Requested temperature/pressure exceeds the source domain: temperature [{self.T_min}, {self.T_max}] K, pressure (0, {self.p_max}] Pa."
        else:
            error_type = (
                ConvergenceError
                if code == StatusCode.NONCONVERGENCE
                else CalculationError
            )
            reason = f"Source evaluation failed: {type(cause).__name__}: {cause}"
        return error_type(
            fluid_name=self.name,
            provider=self.provider,
            model=self.model,
            request=request,
            reason=reason,
            index=index,
        )

    def _evaluate_state_data(
        self,
        input_type,
        prop_1,
        prop_2,
        *,
        generalize_quality=False,
        supersaturation=False,
        errors="raise",
    ):
        """Evaluate static-validated inputs for eager and callback consumers.

        One lock guards the complete source/auxiliary-state transaction, because
        independent pure callbacks may run concurrently. Numerical checks occur
        once per request, not once per returned property. No source is rebuilt.
        """
        first = np.asarray(prop_1, dtype=np.float64)
        second = np.asarray(prop_2, dtype=np.float64)
        if first.shape != second.shape:
            first, second = np.broadcast_arrays(first, second)
        available = enabled_properties(self, supersaturation=supersaturation)
        status = self._input_failure_status(input_type, first, second)
        results = {
            name: (
                np.full(first.shape, np.nan, dtype=np.float64)
                if errors == "mask"
                else np.empty(first.shape, dtype=np.float64)
            )
            for name in available
        }
        with self._state_lock:
            self.converged_flag = True
            for index in np.ndindex(first.shape):
                left, right = float(first[index]), float(second[index])
                code = status[index]
                if code:
                    self.converged_flag = False
                    if errors == "raise":
                        raise self._numerical_error(
                            input_type, left, right, index, code
                        )
                    continue
                try:
                    properties = core.compute_properties_coolprop(
                        self._AS,
                        input_type,
                        left,
                        right,
                        generalize_quality=generalize_quality,
                        supersaturation=supersaturation,
                        source_context=self._source_context,
                    )
                    for name in _SOURCE_CORE_FIELDS:
                        value = properties.get(name, np.nan)
                        if not math.isfinite(value):
                            raise ValueError(
                                f"The source returned nonfinite core property {name!r}."
                            )
                        if (
                            name in ("pressure", "temperature", "density")
                            and value <= 0
                        ):
                            raise ValueError(
                                f"The source returned nonpositive core property {name!r}."
                            )
                    # Apply the same cached acceptance precision as explicit
                    # inputs, preserving recovered coordinates without clipping.
                    temperature, pressure = (
                        properties["temperature"],
                        properties["pressure"],
                    )
                    if (
                        temperature < self._T_min_accepted
                        or temperature > self._T_max_accepted
                        or pressure > self._p_max_accepted
                    ):
                        cause = ValueError(
                            f"The resolved state (temperature={temperature} K, pressure={pressure} Pa) exceeds the source temperature/pressure domain."
                        )
                        code = StatusCode.OUT_OF_DOMAIN
                    else:
                        for name in available:
                            value = properties.get(name, np.nan)
                            results[name][index] = np.nan if value is None else value
                        continue
                except ConvergenceError as error:
                    code, cause = StatusCode.NONCONVERGENCE, error
                except (
                    ValueError,
                    RuntimeError,
                    OverflowError,
                    ZeroDivisionError,
                    AttributeError,
                    TypeError,
                ) as error:
                    code, cause = StatusCode.PROVIDER_FAILURE, error
                self.converged_flag = False
                status[index] = code
                if errors == "raise":
                    raise self._numerical_error(
                        input_type, left, right, index, code, cause
                    ) from cause
                # A failed point may not retain partially written fields.
                for values in results.values():
                    values[index] = np.nan
        return results, status

    def get_state_saturation(self, input_type, prop_1, prop_2, *, errors="raise"):
        """Evaluate a source saturation request with the ordinary state policy."""
        if (
            isinstance(input_type, (bool, np.bool_))
            or not isinstance(input_type, Integral)
            or input_type not in (CP.PQ_INPUTS, CP.QT_INPUTS)
        ):
            raise UnsupportedOperationError(
                fluid_name=self.name,
                provider=self.provider,
                model=self.model,
                request="saturation",
                reason="Saturation evaluation requires PQ_INPUTS or QT_INPUTS.",
                supported=("PQ_INPUTS", "QT_INPUTS"),
            )
        return self.get_state(input_type, prop_1, prop_2, errors=errors)

    def _get_phase_state(
        self,
        calculation_type,
        prop_1,
        prop_1_value,
        prop_2,
        prop_2_value,
        *,
        errors="raise",
        **settings,
    ):
        """Run a scalar legacy phase calculation on the configured source basis.

        These scientific phase solvers are eager operations, separate from the
        ordinary vectorized/JAX state interface. One reentrant lock covers the
        complete solve, its scratch calculations and final state assembly.
        Numerical mask mode preserves the enabled schema; malformed requests
        and unsupported settings still raise their informative typed errors.
        """
        context = dict(fluid_name=self.name, provider=self.provider, model=self.model)
        request = calculation_type
        if not isinstance(errors, str) or errors not in ("raise", "mask", "clip"):
            raise UnsupportedOptionError(
                request="errors",
                reason="Choose 'raise' or 'mask' for source phase calculations.",
                supported=("raise", "mask"),
                **context,
            )
        if errors == "clip":
            raise UnsupportedExecutionModeError(
                request="errors=clip",
                reason="A unique physical projection is not defined for source phase solves.",
                supported=("raise", "mask"),
                **context,
            )
        if jax.core.get_opaque_trace_state() != _EAGER_TRACE_STATE:
            raise UnsupportedExecutionModeError(
                request=request,
                reason="Legacy phase solvers require eager scalar execution.",
                supported=("eager",),
                **context,
            )
        for name, value in (
            ("prop_1_value", prop_1_value),
            ("prop_2_value", prop_2_value),
        ):
            try:
                array = np.asarray(value)
            except (TypeError, ValueError) as cause:
                raise InvalidStateError(
                    request=name,
                    reason="Phase solver inputs must be real scalars.",
                    **context,
                ) from cause
            if array.ndim != 0 or array.dtype.kind not in "fiu":
                raise InvalidStateError(
                    request=name,
                    reason="Phase solvers accept real scalar inputs; use get_state for broadcast ordinary states.",
                    **context,
                )
        for option in ("supersaturation", "generalize_quality", "print_convergence"):
            if not isinstance(settings[option], (bool, np.bool_)):
                raise UnsupportedOptionError(
                    request=option,
                    reason="This option must be a static boolean.",
                    supported=(False, True),
                    **context,
                )
        available = enabled_properties(
            self, supersaturation=settings["supersaturation"]
        )
        for name in (prop_1, prop_2):
            canonical = ALIAS_TO_CANONICAL.get(name) if isinstance(name, str) else None
            if canonical not in available:
                raise UnsupportedPropertyError(
                    request=name,
                    reason="This source phase solver does not enable the requested property.",
                    supported=tuple(sorted(available)),
                    **context,
                )

        def state(data, *, blended=False, status=StatusCode.SUCCESS):
            model, identity = self.model, self.reference_id
            if blended:
                model = "blended_real_fluid"
                identity = _blended_reference_id(
                    self.reference_id,
                    settings["phase_change"],
                    ALIAS_TO_CANONICAL[settings["blending_variable"]],
                    float(settings["blending_onset"]),
                    float(settings["blending_width"]),
                    bool(settings["generalize_quality"]),
                )
            return FluidState(
                fluid_name=self.name,
                identifier=self.identifier,
                provider=self.provider,
                model=model,
                reference_id=identity,
                available_properties=available,
                error_policy=errors,
                status=np.asarray(status, dtype=np.int32),
                **{name: data.get(name, np.nan) for name in available},
            )

        def failed(code):
            if calculation_type == "blending":
                return (
                    state({}, blended=True, status=code),
                    state({}, status=code),
                    state({}, status=code),
                )
            return state({}, status=code)

        with self._state_lock:
            self.converged_flag = False
            try:
                result = core.compute_properties(
                    self._AS,
                    prop_1=prop_1,
                    prop_1_value=prop_1_value,
                    prop_2=prop_2,
                    prop_2_value=prop_2_value,
                    calculation_type=calculation_type,
                    source_context=self._source_context,
                    **settings,
                )
                rows = result if calculation_type == "blending" else (result,)
                if any(
                    not np.isfinite(data[name])
                    for data in rows
                    for name in _SOURCE_CORE_FIELDS
                ):
                    raise CalculationError(
                        request=request,
                        reason="The phase calculation returned nonfinite core properties.",
                        index=(),
                        **context,
                    )
                output = (
                    tuple(
                        state(data, blended=index == 0)
                        for index, data in enumerate(rows)
                    )
                    if calculation_type == "blending"
                    else state(result)
                )
                self.converged_flag = True
                return output
            except (
                OutOfDomainError,
                InvalidStateError,
                ConvergenceError,
                CalculationError,
            ) as error:
                if errors == "raise":
                    if error.fluid_name is None or error.index is None:
                        raise type(error)(
                            request=error.request,
                            reason=error.reason,
                            supported=error.supported,
                            index=() if error.index is None else error.index,
                            indices=error.indices,
                            **context,
                        ) from error
                    raise
                code = (
                    StatusCode.OUT_OF_DOMAIN
                    if isinstance(error, OutOfDomainError)
                    else (
                        StatusCode.INVALID_INPUT
                        if isinstance(error, InvalidStateError)
                        else (
                            StatusCode.NONCONVERGENCE
                            if isinstance(error, ConvergenceError)
                            else StatusCode.PROVIDER_FAILURE
                        )
                    )
                )
                return failed(code)
            except JaxPropError as error:
                if error.fluid_name is None:
                    raise type(error)(
                        request=error.request,
                        reason=error.reason,
                        supported=error.supported,
                        index=error.index,
                        indices=error.indices,
                        **context,
                    ) from error
                raise
            except (
                ValueError,
                RuntimeError,
                OverflowError,
                FloatingPointError,
            ) as cause:
                if errors == "mask":
                    return failed(StatusCode.PROVIDER_FAILURE)
                raise CalculationError(
                    request=request, reason=str(cause), index=(), **context
                ) from cause

    def get_state_equilibrium(
        self,
        prop_1,
        prop_1_value,
        prop_2,
        prop_2_value,
        rhoT_guess=None,
        supersaturation=True,
        generalize_quality=True,
        solver_algorithm="hybr",
        solver_tolerance=1e-6,
        solver_max_iterations=100,
        print_convergence=False,
        *,
        errors="raise",
    ):
        r"""
        Calculate fluid properties according to thermodynamic equilibrium.

        This is an eager scalar phase solver. Density/temperature pairs use a
        direct source call; other pairs need a density/temperature guess.
        ``errors="mask"`` returns a failed scalar state for numerical failures.
        Static settings remain validated and source clipping is unsupported.

        .. note::

            For a detailed description of input arguments and calculation methods, see the
            documentation of the function :ref:`compute_properties <compute_properties>`.

        """
        return self._get_phase_state(
            "equilibrium",
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            rhoT_guess_equilibrium=rhoT_guess,
            supersaturation=supersaturation,
            generalize_quality=generalize_quality,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
            errors=errors,
        )

    def get_state_metastable(
        self,
        prop_1,
        prop_1_value,
        prop_2,
        prop_2_value,
        rhoT_guess=None,
        supersaturation=True,
        generalize_quality=True,
        solver_algorithm="lm",  # hybr
        solver_tolerance=1e-6,
        solver_max_iterations=100,
        print_convergence=False,
        *,
        errors="raise",
    ):
        r"""
        Calculate fluid properties assuming phase metastability

        Evaluate the homogeneous Helmholtz branch, including formal unstable
        states needed by stability studies. Independent density and temperature
        must be positive and finite; optional singular responses are unavailable.
        Scalar ``raise``/``mask`` policies use the configured source reference
        context and protect the complete calculation with its reentrant lock.

        .. note::

            For a detailed description of input arguments and calculation methods, see the
            documentation of the function :ref:`compute_properties <compute_properties>`.

        """
        return self._get_phase_state(
            "metastable",
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            rhoT_guess_metastable=rhoT_guess,
            supersaturation=supersaturation,
            generalize_quality=generalize_quality,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
            errors=errors,
        )

    def get_state_blending(
        self,
        prop_1,
        prop_1_value,
        prop_2,
        prop_2_value,
        rhoT_guess_equilibrium,
        rhoT_guess_metastable,
        blending_onset,
        blending_width,
        phase_change,
        blending_variable="quality_mass",
        supersaturation=True,
        generalize_quality=True,
        solver_algorithm="hybr",
        solver_tolerance=1e-6,
        solver_max_iterations=100,
        print_convergence=False,
        *,
        errors="raise",
    ):
        r"""
        Calculate fluid properties by blending equilibrium and metastable properties

        Return blended, equilibrium and metastable FluidState objects. The blend
        is a constitutive interpolation, tagged ``blended_real_fluid`` with its
        own settings-dependent identity; it need not satisfy source-EOS response
        identities. Algorithm coordinates and weights are not thermodynamic
        fields (``sigma`` retains its surface-tension meaning). Execution is
        scalar/eager with explicit ``raise`` or ``mask`` numerical policies.

        .. note::

            For a detailed description of input arguments and calculation methods, see the
            documentation of the function :ref:`compute_properties <compute_properties>`.

        """
        return self._get_phase_state(
            "blending",
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            rhoT_guess_equilibrium=rhoT_guess_equilibrium,
            rhoT_guess_metastable=rhoT_guess_metastable,
            blending_variable=blending_variable,
            blending_onset=blending_onset,
            blending_width=blending_width,
            phase_change=phase_change,
            supersaturation=supersaturation,
            generalize_quality=generalize_quality,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
            errors=errors,
        )

    def plot_phase_diagram(
        self,
        x_prop="s",
        y_prop="T",
        axes=None,
        N=100,
        plot_saturation_line=True,
        plot_critical_point=True,
        plot_triple_point_liquid=False,
        plot_triple_point_vapor=False,
        plot_spinodal_line=False,
        spinodal_line_color=0.5 * np.array([1, 1, 1]),
        spinodal_line_width=1.25,
        spinodal_line_method="slsqp",  # Alternative is slsqp
        spinodal_line_use_previous=False,  # True is not as robust
        plot_quality_isolines=False,
        plot_two_phase_patch=False,
        plot_pseudocritical_line=False,
        quality_levels=np.linspace(0.1, 1.0, 10),
        quality_labels=False,
        show_in_legend=False,
        x_scale="linear",
        y_scale="linear",
        dT_crit=1.00,
    ):
        """Draw or update source-EOS phase curves on the selected axes.

        Numerical curve caches include their sampling/solver options. Artists
        belong to each fluid and axes pair: repeated calls update existing
        lines, and disabling an option hides/removes only that fluid's artists.
        Plotting is an eager Matplotlib operation, outside state evaluation.
        """
        for name, value in (("x_prop", x_prop), ("y_prop", y_prop)):
            canonical = (
                ALIAS_TO_CANONICAL.get(value) if isinstance(value, str) else None
            )
            if canonical not in self.capabilities.properties:
                raise UnsupportedPropertyError(
                    fluid_name=self.name,
                    provider=self.provider,
                    model=self.model,
                    request=str(value),
                    reason=f"Unsupported phase-diagram {name} property.",
                    supported=tuple(sorted(self.capabilities.properties)),
                )
        if not hasattr(self, "_phase_curve_settings"):
            self._phase_curve_settings = {}
        settings = self._phase_curve_settings
        if any(
            (
                plot_saturation_line,
                plot_spinodal_line,
                plot_quality_isolines,
                plot_two_phase_patch,
                plot_pseudocritical_line,
            )
        ):
            N = _grid_count(N, "N", minimum=2)

        # A two-phase patch needs the saturation boundary even if its outline
        # is hidden. Keep the numerical cache independent of artist visibility.
        if plot_saturation_line or plot_two_phase_patch:
            key = (N, float(dT_crit))
            if (
                self.sat_liq is None
                or self.sat_vap is None
                or settings.get("saturation_line") != key
            ):
                self.sat_liq, self.sat_vap = compute_saturation_line(
                    self, N, dT_crit=dT_crit
                )
                settings["saturation_line"] = key

        if axes is None:
            _, axes = plt.subplots(figsize=(6, 5))
        labels = getattr(axes, "_jaxprop_phase_axis_labels", {})
        for coordinate, prop in (("x", x_prop), ("y", y_prop)):
            current = getattr(axes, f"get_{coordinate}label")()
            if not current or current == labels.get(coordinate):
                label = LABEL_MAPPING.get(prop, prop)
                getattr(axes, f"set_{coordinate}label")(label)
                labels[coordinate] = label
        axes._jaxprop_phase_axis_labels = labels
        axes.set_xscale(x_scale)
        axes.set_yscale(y_scale)

        # Saturation line
        if plot_saturation_line:
            x = jnp.concatenate([jnp.flip(self.sat_liq[x_prop]), self.sat_vap[x_prop]])
            y = jnp.concatenate([jnp.flip(self.sat_liq[y_prop]), self.sat_vap[y_prop]])
            label = self._get_label("Saturation line", show_in_legend)
            params = {"label": label, "color": "black"}
            self._graphic_saturation_line = self._plot_or_update_line(
                axes,
                x,
                y,
                "saturation_line",
                **params,
            )
        else:
            self._set_visibility(axes, "saturation_line", False)

        # Spinodal line
        if plot_spinodal_line:
            key = (
                N,
                float(dT_crit),
                spinodal_line_method,
                bool(spinodal_line_use_previous),
            )
            if (
                self.spdl_liq is None
                or self.spdl_vap is None
                or settings.get("spinodal_line") != key
            ):
                self.spdl_liq, self.spdl_vap = compute_spinodal_line(
                    self,
                    N=N,
                    method=spinodal_line_method,
                    use_previous_as_initial_guess=spinodal_line_use_previous,
                    supersaturation=False,
                    dT_crit=dT_crit,
                )
                settings["spinodal_line"] = key
            x = jnp.concatenate(
                [jnp.flip(self.spdl_liq[x_prop]), self.spdl_vap[x_prop]]
            )
            y = jnp.concatenate(
                [jnp.flip(self.spdl_liq[y_prop]), self.spdl_vap[y_prop]]
            )
            label = self._get_label("Spinodal line", show_in_legend)
            params = {
                "label": label,
                "color": spinodal_line_color,
                "linewidth": spinodal_line_width,
            }
            self._graphic_spinodal_line = self._plot_or_update_line(
                axes,
                x,
                y,
                "spinodal_line",
                **params,
            )
        else:
            self._set_visibility(axes, "spinodal_line", False)

        # Plot pseudocritical line
        if plot_pseudocritical_line:
            if (
                self.pseudo_critical_line is None
                or settings.get("pseudocritical_line") != N
            ):
                self.pseudo_critical_line = compute_pseudocritical_line(
                    self, N_points=N
                )
                settings["pseudocritical_line"] = N
            x = self.pseudo_critical_line[x_prop]
            y = self.pseudo_critical_line[y_prop]
            label = self._get_label("Pseudocritical line", show_in_legend)
            params = {
                "label": label,
                "color": "black",
                "linestyle": "--",
                "linewidth": 0.75,
            }
            self._graphic_pseudocritical_line = self._plot_or_update_line(
                axes,
                x,
                y,
                "pseudocritical_line",
                **params,
            )
        else:
            self._set_visibility(axes, "pseudocritical_line", False)

        # Plot quality isolines
        if plot_quality_isolines:
            levels = np.asarray(quality_levels)
            if (
                levels.ndim != 1
                or not levels.size
                or levels.dtype.kind not in "fiu"
                or not np.all(np.isfinite(levels))
                or np.any((levels < 0) | (levels > 1))
                or np.any(np.diff(levels) <= 0)
            ):
                raise InvalidConfigurationError(
                    request="quality_levels",
                    reason="Expected distinct, increasing finite qualities in [0, 1].",
                )
            # Matplotlib needs a two-row mesh even for one requested isoline.
            mesh_levels = levels if levels.size > 1 else np.unique([0.0, *levels, 1.0])
            key = (N, float(dT_crit), tuple(mesh_levels))
            if self.q_mesh is None or settings.get("quality_isolines") != key:
                self.q_mesh = compute_quality_grid(
                    self, N, mesh_levels, dT_crit=dT_crit
                )
                settings["quality_isolines"] = key
            x = self.q_mesh[x_prop]
            y = self.q_mesh[y_prop]
            _, m = np.shape(x)
            z = np.tile(mesh_levels, (m, 1)).T
            params = {"colors": "black", "linestyles": ":", "linewidths": 0.75}
            self._graphics_q_lines = self._plot_or_update_contours(
                axes,
                x,
                y,
                z,
                levels,
                "quality_isolines",
                **params,
            )

            if quality_labels:
                axes.clabel(self._graphics_q_lines, fontsize=9, rightside_up=True)

        else:
            # Remove existing contour lines if they exist
            if "quality_isolines" in self.graphic_elements.get(axes, {}):
                self.graphic_elements[axes]["quality_isolines"].remove()
                del self.graphic_elements[axes]["quality_isolines"]

        # Plot critical point
        params = {
            "color": "black",
            "marker": "o",
            "markersize": 4.5,
            "markerfacecolor": "w",
        }
        if plot_critical_point:
            x = self.critical_point[x_prop]
            y = self.critical_point[y_prop]
            label = self._get_label("Critical point", show_in_legend)
            self._graphic_critical_point = self._plot_or_update_line(
                axes,
                x,
                y,
                "critical_point",
                label=label,
                **params,
            )
        else:
            self._set_visibility(axes, "critical_point", False)

        # Plot liquid triple point
        if plot_triple_point_liquid:
            x = self.triple_point_liquid[x_prop]
            y = self.triple_point_liquid[y_prop]
            description = (
                "Triple point liquid"
                if self.references.triple_liquid.interpretation == "triple_point"
                else "Lower saturation endpoint (liquid)"
            )
            label = self._get_label(description, show_in_legend)
            self._graphic_triple_point_liquid = self._plot_or_update_line(
                axes,
                x,
                y,
                "triple_point_liquid",
                label=label,
                **params,
            )
        else:
            self._set_visibility(axes, "triple_point_liquid", False)

        # Plot vapor triple point
        if plot_triple_point_vapor:
            x = self.triple_point_vapor[x_prop]
            y = self.triple_point_vapor[y_prop]
            description = (
                "Triple point vapor"
                if self.references.triple_vapor.interpretation == "triple_point"
                else "Lower saturation endpoint (vapor)"
            )
            label = self._get_label(description, show_in_legend)
            self._graphic_triple_point_vapor = self._plot_or_update_line(
                axes,
                x,
                y,
                "triple_point_vapor",
                label=label,
                **params,
            )
        else:
            self._set_visibility(axes, "triple_point_vapor", False)

        if plot_two_phase_patch:
            self._graphic_two_phase_patch = self._plot_or_update_patch(
                axes,
                np.concatenate([self.sat_liq[x_prop][::-1], self.sat_vap[x_prop]]),
                np.concatenate([self.sat_liq[y_prop][::-1], self.sat_vap[y_prop]]),
            )
        else:
            self._set_visibility(axes, "two_phase_patch", False)

        # set_data() does not refresh limits. Respect explicit user limits,
        # which Matplotlib records by disabling the corresponding autoscale.
        axes.relim(visible_only=True)
        axes.autoscale_view()
        return axes.figure, axes

    def _get_label(self, label, show_in_legend):
        """Returns the appropriate label value based on whether it should be shown in the legend."""
        return label if show_in_legend else "_no_legend_"

    def _plot_or_update_line(self, axes, x_data, y_data, line_name, **plot_params):
        # Ensure there is a dictionary for this axes
        if axes not in self.graphic_elements:
            self.graphic_elements[axes] = {}

        # Make sure elements are arrays (avoid error when plotting a single point)
        x_data = np.atleast_1d(x_data)
        y_data = np.atleast_1d(y_data)

        # Check if the line exists for this axes
        if line_name in self.graphic_elements[axes]:
            line = self.graphic_elements[axes][line_name]
            line.set_data(x_data, y_data)
            # Update line properties
            line.set(**plot_params)
            line.set_visible(True)
        else:
            # Create a new line with the provided plot parameters
            (line,) = axes.plot(x_data, y_data, **plot_params)
            self.graphic_elements[axes][line_name] = line
        return line

    def _plot_or_update_patch(self, axes, x_data, y_data):
        """Reuse the saturation-boundary polygon instead of stacking patches."""
        elements = self.graphic_elements.setdefault(axes, {})
        coordinates = np.column_stack((x_data, y_data))
        if "two_phase_patch" in elements:
            patch = elements["two_phase_patch"]
            patch.set_xy(coordinates)
            patch.set_visible(True)
        else:
            (patch,) = axes.fill(
                x_data, y_data, facecolor="white", edgecolor="none", zorder=1
            )
            elements["two_phase_patch"] = patch
        return patch

    def _plot_or_update_contours(
        self, axes, x_data, y_data, z_data, contour_levels, line_name, **contour_params
    ):

        # Ensure there is a dictionary for this axes
        if axes not in self.graphic_elements:
            self.graphic_elements[axes] = {}

        # Check if the contour exists for this axes
        if line_name in self.graphic_elements[axes]:
            # Remove the old contour
            self.graphic_elements[axes][line_name].remove()

        # Create a new contour
        contour = axes.contour(x_data, y_data, z_data, contour_levels, **contour_params)
        self.graphic_elements[axes][line_name] = contour
        return contour

    def _set_visibility(self, axes, line_name, visible):
        if axes in self.graphic_elements and line_name in self.graphic_elements[axes]:
            self.graphic_elements[axes][line_name].set_visible(visible)


# ------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------ #


# Source-only compatibility name. The package-level Fluid constructor is the
# generic method selector; internal source construction uses FluidCoolProp.
Fluid = FluidCoolProp


# The legacy constructor now returns the one consolidated binary evaluator.
# It translates names/mass ratio only at construction, without wrapping calls.
from ..binary_mixture.compat import FluidMix


# ------------------------------------------------------------------------------------ #
# Phase and reference curves
# ------------------------------------------------------------------------------------ #
def _grid_count(value, name, minimum=1):
    """Validate setup-only sampling controls before source evaluation."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise InvalidConfigurationError(
            request=name,
            reason=f"Expected an integer greater than or equal to {minimum}.",
        )
    return int(value)


def _saturation_temperature_levels(fluid, count, dT_crit):
    """Return distinct temperatures from the near-critical to lower endpoint."""
    count = _grid_count(count, "num_points", minimum=2)
    critical = float(fluid.critical_point.T)
    lower = max(float(fluid.triple_point_liquid.T), float(fluid.triple_point_vapor.T))
    delta = np.asarray(dT_crit)
    if (
        delta.shape != ()
        or delta.dtype.kind not in "fiu"
        or not np.isfinite(delta)
        or not 0 < delta < critical - lower
    ):
        raise InvalidConfigurationError(
            request="dT_crit",
            reason="Expected a positive critical offset smaller than the saturation temperature span.",
        )
    return critical - np.geomspace(float(delta), critical - lower, count)


def compute_saturation_line(fluid, N=100, dT_crit=0.5):
    """
    Compute the saturation line for a given fluid.

    Parameters
    ----------
    fluid : object
        The fluid object containing thermodynamic properties and methods.
    N : int, optional
        Number of distinct subcritical samples. The critical endpoint is
        prepended, so each returned branch has N+1 states. Default is 100.

    Returns
    -------
    saturation_liq : FluidState
        Batched FluidState for the liquid saturation line.
    saturation_vap : FluidState
        Batched FluidState for the vapor saturation line.
    """
    # Define temperature array with refinement close to the critical point
    T_sat = _saturation_temperature_levels(fluid, N, dT_crit)

    # Source references keep canonical source identity. A plotted curve belongs
    # to the caller's fluid/stream, matching its ordinary saturation states.
    reference = fluid.critical_point
    critical_state = FluidState(
        fluid_name=fluid.name,
        identifier=fluid.identifier,
        provider=reference.provider,
        model=reference.model,
        reference_id=reference.reference_id,
        status=reference.status,
        error_policy=reference.error_policy,
        available_properties=reference.available_properties,
        **reference.to_dict(),
    )
    liq_states = [critical_state]
    vap_states = [critical_state]

    for T in T_sat:
        liq_states.append(fluid.get_state_saturation(CP.QT_INPUTS, 0.00, T))
        vap_states.append(fluid.get_state_saturation(CP.QT_INPUTS, 1.00, T))

    # Reverse the liquid branch if needed
    # liq_states = list(reversed(liq_states))

    # Stack into batched FluidState
    saturation_liq = FluidState.stack(liq_states)
    saturation_vap = FluidState.stack(vap_states)

    return saturation_liq, saturation_vap


def compute_pseudocritical_line(fluid, N_points=100):
    """Compute pseudocritical line (approximate, defined here at critical density)."""
    # Define temperature array with refinement close to the critical point
    N_points = _grid_count(N_points, "N_points")
    tau = np.logspace(np.log10(1e-3), np.log10(1), N_points)
    T_range = (1 + tau) * fluid.critical_point.T

    maximum = fluid.capabilities.domain.bounds.get("temperature", (None, None))[1]
    if maximum is not None:
        T_range = T_range[T_range < maximum]
    if not T_range.size:
        raise OutOfDomainError(
            request="pseudocritical_line",
            reason="The source temperature domain contains no requested pseudocritical points.",
        )
    return fluid.get_state(CP.DmassT_INPUTS, fluid.critical_point.d, T_range)


def compute_spinodal_line(
    fluid,
    N=50,
    method="slsqp",
    use_previous_as_initial_guess=False,
    supersaturation=False,
    dT_crit=0.25,
    tolerance=1e-8,
):
    """
    Compute the spinodal line for a given fluid.

    Parameters
    ----------
    fluid : object
        The fluid object containing thermodynamic properties and methods.
    N : int, optional
        Number of points to compute along the spinodal line. Default is 50.
    method : str, optional
        The optimization method to solve the spinodal point problem ('bfgs' or 'slsqp'). Default is 'bfgs'.
    use_previous_as_initial_guess : bool, optional
        Whether to use the previous point as the initial guess for the next point. Default is False.
    supersaturation : bool, optional
        Whether to compute supersaturation properties. Default is False.

    Returns
    -------
    spinodal_liq : FluidState
        Batched FluidState for the liquid spinodal line.
    spinodal_vap : FluidState
        Batched FluidState for the vapor spinodal line.
    """

    N = _grid_count(N, "N", minimum=2)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise InvalidConfigurationError(
            request="tolerance", reason="Expected a finite positive tolerance."
        )

    # Temperature array with refinement close to the critical point
    alpha = 0.01
    T_max = fluid.critical_point.T - dT_crit
    T_min = alpha * T_max + (1 - alpha) * fluid.triple_point_liquid.T
    ratio = 1 - T_min / T_max
    t1 = np.logspace(np.log10(1 - 0.9999), np.log10(ratio / 10), int(np.ceil(N / 2)))
    t2 = np.logspace(np.log10(ratio / 10), np.log10(ratio), int(np.floor(N / 2)))
    T_spinodal = (1 - np.concatenate([t1, t2])) * T_max

    # Get limits of entropy to prevent points where EoS breaks down
    s_min, s_max = fluid.triple_point_liquid["s"], fluid.triple_point_vapor["s"]
    delta = (s_max - s_min) / 100
    s_min, s_max = s_min + delta, s_max - delta

    # --- liquid branch
    states_liq = []
    props_liq = compute_spinodal_point(
        T_spinodal[0],
        fluid,
        "liquid",
        method=method,
        supersaturation=supersaturation,
        tolerance=tolerance,
    )
    for T in T_spinodal:
        rho = props_liq["rho"] if use_previous_as_initial_guess else None
        props_liq = compute_spinodal_point(
            T,
            fluid,
            "liquid",
            rho_guess=rho,
            method=method,
            supersaturation=supersaturation,
            tolerance=tolerance,
        )
        if s_min < props_liq["s"] < s_max:
            states_liq.append(props_liq)
        else:
            break

    # --- vapor branch
    states_vap = []
    props_vap = compute_spinodal_point(
        T_spinodal[0],
        fluid,
        "vapor",
        method=method,
        supersaturation=supersaturation,
        tolerance=tolerance,
    )
    for T in T_spinodal:
        rho = props_vap["rho"] if use_previous_as_initial_guess else None
        props_vap = compute_spinodal_point(
            T,
            fluid,
            "vapor",
            rho_guess=rho,
            method=method,
            supersaturation=supersaturation,
            tolerance=tolerance,
        )
        if s_min < props_vap["s"] < s_max:
            states_vap.append(props_vap)
        else:
            break

    if not states_liq or not states_vap:
        raise ConvergenceError(
            request="spinodal_line",
            reason="No valid spinodal samples were obtained inside the source entropy bounds.",
        )
    # Convert to batched FluidState objects
    spinodal_liq = FluidState.stack(states_liq)
    spinodal_vap = FluidState.stack(states_vap)

    return spinodal_liq, spinodal_vap


# ------------------------------------------------------------------------------------ #
# Spinodal point calculations
# ------------------------------------------------------------------------------------ #

from scipy.optimize import root_scalar


def compute_spinodal_point_general(
    prop_type,
    prop_value,
    fluid,
    branch,
    rho_guess=None,
    N_trial=100,
    method="slsqp",
    tolerance=1e-8,
    print_convergence=False,
    supersaturation=False,
):
    """
    General function to compute the spinodal point for a given property name and value.

    This function uses the underlying `compute_spinodal_point` function and iterates
    on temperature until the specified property at the spinodal point matches the given value.

    Parameters
    ----------
    prop_type : str
        The type of property to match (e.g., 'rho', 'p').
    prop_value : float
        The value of the property to match at the spinodal point.
    fluid : object
        The fluid object containing thermodynamic properties and methods.
    branch : str
        The branch of the spinodal line. Options: 'liquid' or 'vapor'.
    rho_guess : float, optional
        Initial guess for the density. If provided, this value will be used directly.
        If not provided, the density initial guess will be generated based on a number of trial points.
    N_trial : int, optional
        Number of trial points to generate the density initial guess. Default is 100.
    method : str, optional
        The optimization method to solve the problem ('bfgs' or 'slsqp'). Default is 'bfgs'.
    tolerance : float, optional
        Tolerance for the solver termination. Defaults to 1e-6.
    print_convergence : bool, optional
        If True, displays the convergence progress. Defaults to False.

    Returns
    -------
    barotropy.State
        A State object containing the fluid properties

    Raises
    ------
    ValueError
        If the scalar root to determine the spinodal point fails to converge.

    Notes
    -----
    This function uses a root-finding algorithm to iterate on temperature until the
    property specified by `prop_type` at the spinodal point matches `prop_value`.
    The `compute_spinodal_point` function is called iteratively to evaluate the spinodal
    properties at different temperatures.

    Examples
    --------
    >>> state = compute_spinodal_point_general(
    ...     prop_type='density',
    ...     prop_value=500,
    ...     T_guess=300,
    ...     fluid=my_fluid,
    ...     branch='liquid',
    ...     rho_guess=10,
    ...     N_trial=150,
    ...     method='slsqp',
    ...     tolerance=1e-7,
    ...     print_convergence=True,
    ... )
    """

    canonical = (
        core.ALIAS_TO_CANONICAL.get(prop_type) if isinstance(prop_type, str) else None
    )
    if canonical is None:
        raise InvalidConfigurationError(
            request="prop_type", reason="Unknown spinodal target property or alias."
        )
    if np.ndim(prop_value) != 0 or not np.isfinite(prop_value):
        raise InvalidConfigurationError(
            request="prop_value", reason="Expected a finite scalar target."
        )
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise InvalidConfigurationError(
            request="tolerance", reason="Expected a finite positive tolerance."
        )
    scale = max(abs(float(prop_value)), 1.0)
    if canonical in {"enthalpy", "internal_energy", "entropy"}:
        specific_R = fluid._AS.gas_constant() / fluid._AS.molar_mass()
        scale = (
            specific_R
            if canonical == "entropy"
            else specific_R * float(fluid.critical_point.T)
        )

    # Define a dimensionless residual; h/s scales are reference-gauge invariant.
    def residual(T):
        props = compute_spinodal_point(
            T,
            fluid,
            branch,
            rho_guess=rho_guess,
            N_trial=N_trial,
            method=method,
            tolerance=tolerance,
            print_convergence=print_convergence,
            supersaturation=supersaturation,
        )
        return (props[canonical] - prop_value) / scale

    # Try to compute spinodal point
    try:
        bracket = [fluid.triple_point_vapor.T + 0.1, fluid.critical_point.T - 0.1]
        result = root_scalar(
            residual, method="toms748", bracket=bracket, xtol=tolerance
        )
    except ValueError as e:
        raise ValueError(
            f"Failed to find a spinodal point for {prop_type}={prop_value:.2e}.\n"
            f"This might be because there is no spinodal point for the given fluid property value.\n"
            f"Check the value of {prop_type} to ensure there is a matching spinodal point."
        ) from e
    if not result.converged:
        raise ValueError("Scalar root to determine spinodal point failed")

    # Manually check the residual at the solution
    final_residual = residual(result.root)
    if abs(final_residual) > 5 * tolerance:
        raise ValueError(
            f"The solution converged but the residual {final_residual:.2e} is not within the tolerance.\n"
            f"Check the value of {prop_type}={prop_value:.2e} and branch={branch} to ensure there is a matching spinodal point."
        )

    # Compute state for the computed solution
    state = compute_spinodal_point(
        result.root,
        fluid,
        branch,
        rho_guess=rho_guess,
        N_trial=N_trial,
        method=method,
        tolerance=tolerance,
        print_convergence=print_convergence,
        supersaturation=supersaturation,
    )

    return state


def compute_spinodal_point(
    temperature,
    fluid,
    branch,
    rho_guess=None,
    N_trial=100,
    method="slsqp",
    tolerance=1e-8,
    supersaturation=False,
    print_convergence=False,
):
    r"""
    Compute the vapor or liquid spinodal point of a fluid at a given temperature.

    Parameters
    ----------
    temperature : float
        Temperature of the fluid (K).
    fluid : barotropy.Fluid
        The fluid for which the spinodal point is to be calculated.
    branch : str
        Branch of the spinodal line used to determine the density initial guess.
        Options: 'liquid' or 'vapor'.
    rho_guess : float, optional
        Initial guess for the density. If provided, this value will be used directly.
        If not provided, the density initial guess will be generated based on a number of trial points.
    N_trial : int, optional
        Number of trial points to generate the density initial guess. Default is 50.
    method : str, optional
        The optimization method to solve the problem ('bfgs' or 'slsqp').
    tolerance : float, optional
        Tolerance for the solver termination. Defaults to 1e-6.
    print_convergence : bool, optional
        If True, displays the convergence progress. Defaults to False.

    Returns
    -------
    barotropy.State
        A State object containing the fluid properties

    Raises
    ------
    ValueError
        If the input temperature is higher than the critical temperature or lower than
        the triple temperature.

    Notes
    -----
    When a single-phase fluid undergoes a thermodynamic process and enters the two-phase region it
    can exist in a single-phase state that is different from the equilibrium two-phase state.
    Such states are know as metastable states and they are only possible in the thermodynamic
    region between the saturation lines and the spinodal lines. If the thermodynamic process
    continues and crosses the spinodal lines metastable states become unstable and the transition
    to two-distinct phase occurs rapidly. Therefore, the spinodal line represents the locus of points
    that separates the region where a mixture is thermodynamically unstable and prone to phase separation
    from the region where metastable states are physically possible.

    In mathematical terms, the spinodal line is defined as the loci of thermodynamic states in which the isothermal bulk modulus of the fluid is zero:

    .. math::

        K_T = \rho \left( \frac{\partial p}{\partial \rho} \right)_T = 0

    More precisely, a vapor spinodal point is the first local maximum of a isotherm line in a pressure-density diagram as the density increases.
    Conversely, a liquid spinodal point is the first local minimum of a isotherm line in a pressure-density diagram as the density decreases.
    The spinodal lines and phase envelope of carbon dioxide according to the HEOS developed by :cite:`span_new_1996` are illustrated in the figure below

    .. image:: /_static/spinodal_points_CO2.svg
        :alt: Pressure-density diagram and spinodal points for carbon dioxide.


    Some equations of state are not well-posed and do not satisfy the condition :math:`K_T=0` within the two-phase region.
    This is exemplified by the nitrogen HEOS developed by :cite:`span_reference_2000`.

    .. image:: /_static/spinodal_points_nitrogen.svg
        :alt: Pressure-density diagram and "pseudo" spinodal points for carbon dioxide.

    As seen in the figure, this HEOS is not well-posed because there are isotherms that do not have a local minimum/maximum corresponding to a state with zero isothermal bulk modulus.
    In such cases, this function returns the inflection point of the isotherms (corresponding to the point closest to zero bulk modulus) as the spinodal point.

    """

    # Validate scientific setup before allowing an optimizer to mutate a source.
    if branch not in ("liquid", "vapor"):
        raise InvalidConfigurationError(
            request="branch", reason="Expected 'liquid' or 'vapor'."
        )
    N_trial = _grid_count(N_trial, "N_trial", minimum=3)
    if np.ndim(temperature) != 0 or not np.isfinite(temperature) or temperature <= 0:
        raise InvalidConfigurationError(
            request="temperature",
            reason="Expected a finite positive scalar temperature.",
        )
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise InvalidConfigurationError(
            request="tolerance", reason="Expected a finite positive tolerance."
        )
    if rho_guess is not None and (
        np.ndim(rho_guess) != 0 or not np.isfinite(rho_guess) or rho_guess <= 0
    ):
        raise InvalidConfigurationError(
            request="rho_guess", reason="Expected a finite positive density guess."
        )

    # Instantiate new abstract state to compute saturation properties without changing the state of the class

    # Check that the inlet temperature is lower than the critical temperature
    T_critical = fluid.critical_point.T
    if temperature >= T_critical:
        msg = f"T={temperature:.3f}K must be less than T_critical={T_critical:.3f}K"
        raise ValueError(msg)

    # Check that the inlet temperature is greater than the triple temperature
    T_triple = fluid.triple_point_vapor.T
    if temperature < T_triple:
        msg = f"T={temperature:.3f}K must be greater than T_triple={T_triple:.3f}K"
        raise ValueError(msg)

    # Create spinodal point optimization problem
    problem = _SpinodalPointProblem(temperature, fluid, branch, supersaturation)

    # Check solver method (deprecated)
    if method == "bfgs":
        # problem.get_bounds = lambda: ([-np.inf], [np.inf])
        pass
    elif method == "slsqp":
        pass
    else:
        raise ValueError(
            f"Solver method {method} is not valid. Valid options are 'slsqp' and 'bfgs'"
        )

    # Create solver object
    solver = psv.OptimizationSolver(
        problem=problem,
        library="scipy",
        method=method,  # "l-bfgs-b", "slsqp" "slsqp"
        tolerance=tolerance,
        print_convergence=print_convergence,
        problem_scale=50,
    )

    # Generate initial guess if not provided
    if rho_guess is None:
        rho_guess = problem.generate_density_guess(N_trial)

    # Solve the problem using the provided or generated initial guess
    rho_opt = solver.solve(rho_guess).item()
    if not np.isfinite(rho_opt) or not problem.rho_vap <= rho_opt <= problem.rho_liq:
        raise ConvergenceError(
            request="spinodal",
            reason="Optimizer returned a density outside the saturation interval.",
        )
    if not solver.success:
        # Near-critical EOS curvature can make SLSQP report incompatible
        # inequalities at an almost correct density. Recover an actual zero of
        # the signed modulus on the requested branch instead of accepting the
        # optimizer's flag or discarding a physically validated stability limit.
        bracket = problem.first_bulk_modulus_bracket(N_trial)
        if bracket is not None:
            recovered = root_scalar(
                problem.signed_bulk_modulus,
                bracket=bracket,
                method="brentq",
                xtol=max(tolerance * problem.rho_liq, np.finfo(float).eps),
                rtol=max(tolerance, 4 * np.finfo(float).eps),
            )
            if not recovered.converged:
                raise ConvergenceError(
                    request="spinodal",
                    reason="The bounded bulk-modulus root did not converge.",
                )
            rho_opt = recovered.root
        elif float(problem.fitness(rho_opt)[0]) > max(100 * tolerance, 1e-6):
            raise ConvergenceError(
                request="spinodal",
                reason=f"Spinodal optimization failed: {solver.message}",
            )
    state = fluid.get_state_metastable(
        "rho",
        rho_opt,
        "T",
        temperature,
        supersaturation=supersaturation,
        generalize_quality=False,
    )

    # I tried different combinations of solvers and initial guesses
    # SLSQP is very fast, but also very aggresive and can lead to unpredictable results
    # BFGS is reliable when the initial guess is good
    # Achieving a good initial guess is possible by:
    #  1. Using previous point in the spinodal line and having high resolution on the spinodal line
    #  2. Generating an initial guess for each point with the initial guess strategy
    # Option 2. seems the most reliable, and even if the computational cost can be a bit hight
    # it seems to be the most effective way to get accurate spinodal lines.
    #
    # 11.07.2025
    # I improved the code with 2 modifications:
    #
    #  1. Use logspace instead of linspace to generate initial guess
    #  2. Apply problem scaling to SLSQP computation
    #
    # With this enhancements, SLSQP produces good spinodal line results
    #
    # Near criticality an SLSQP failure can still sit beside a true K_T zero.
    # The bounded signed-modulus recovery above follows the first crossing
    # from the requested saturation branch. It is distinct from the documented
    # closest-modulus pseudo-spinodal used when an EOS has no true crossing.

    return state


class _SpinodalPointProblem(psv.OptimizationProblem):
    """Auxiliary class for the determination of the liquid and vapor spinodal points"""

    def __init__(self, temperature, fluid, branch, supersaturation):

        # Declare class attributes
        self.T = temperature
        self.branch = branch
        self.fluid = fluid
        self.supersaturation = supersaturation

        # Compute saturation liquid density (used to determine initial guess)
        state_vap = self.fluid.get_state(CP.QT_INPUTS, 0.00, self.T)
        self.rho_liq = state_vap.rho

        # Calculate saturation vapor density (used to determine initial guess)
        state_liq = self.fluid.get_state(CP.QT_INPUTS, 1.00, self.T)
        self.rho_vap = state_liq.rho

    def generate_density_guess(self, N):
        """Generate a density initial guess that is close to the first local minima of the absolute value of the bulk modulus"""

        # Generate candidate densities between saturation and the critical value
        if self.branch == "liquid":
            # rho_array = np.linspace(self.rho_liq, self.rho_vap, N)
            rho_array = np.logspace(np.log10(self.rho_liq), np.log10(self.rho_vap), N)
        elif self.branch == "vapor":
            # rho_array = np.linspace(self.rho_vap, self.rho_liq, N)
            rho_array = np.logspace(np.log10(self.rho_vap), np.log10(self.rho_liq), N)
        else:
            msg = f"Invalid value for parameter branch={self.branch}. Options: 'liquid' or 'vapor'"
            raise ValueError(msg)

        # Evaluate residual vector at trial densities
        residual = np.abs([self.fitness(rho) for rho in rho_array])

        # Return the first local minima in the residual vector
        for i in range(1, N - 1):
            if residual[i - 1] > residual[i] < residual[i + 1]:
                self.rho_guess = rho_array[i + 1]
                return self.rho_guess

        # Some EOS have no true zero of the bulk modulus on this branch. Their
        # documented pseudo-spinodal is the closest sampled interior minimum.
        self.rho_guess = rho_array[1 + int(np.argmin(residual[1:-1]))]
        return self.rho_guess

    def fitness(self, rho):
        """
        Compute the objective function of the optimization problem: the absolute value of the isothermal bulk modulus.

        This function uses the absolute value of residual to solve an optimization problem
        rather than a non-linear equation having the bulk modulus as residual. This approach
        is adopted because some fluids (e.g., nitrogen) have ill-posed EoS that not have a well-defined
        spinodal points where the isothermal bulk modulus is zero.

        Not a good idea to scale the bulk modulus by pressure because it can take negative values or zero when evaluated with the HEOS.
        """
        return np.atleast_1d(np.abs(self.signed_bulk_modulus(rho))) / 1e6

    def signed_bulk_modulus(self, rho):
        """Return the actual homogeneous derivative, including unstable signs."""
        # Optimizers supply a one-variable vector; CoolProp requires scalar density.
        rho = np.asarray(rho).item()
        state = self.fluid.get_state_metastable(
            "rho",
            rho,
            "T",
            self.T,
            supersaturation=self.supersaturation,
            generalize_quality=False,
        )
        return float(state["isothermal_bulk_modulus"])

    def first_bulk_modulus_bracket(self, count):
        """Find the first true stability crossing from the selected endpoint."""
        densities = np.geomspace(self.rho_vap, self.rho_liq, count)
        if self.branch == "liquid":
            densities = densities[::-1]
        previous_density = float(densities[0])
        previous_value = self.signed_bulk_modulus(previous_density)
        for density in densities[1:]:
            density = float(density)
            value = self.signed_bulk_modulus(density)
            if (
                np.isfinite(previous_value)
                and np.isfinite(value)
                and (
                    previous_value == 0
                    or value == 0
                    or np.signbit(previous_value) != np.signbit(value)
                )
            ):
                return tuple(sorted((previous_density, density)))
            previous_density, previous_value = density, value
        return None

    def get_bounds(self):
        """Compute the bounds of the optimization problem."""
        return [(self.rho_vap,), (self.rho_liq,)]
        # rho_limit = self.rho_guess
        # # rho_limit = self.rho_crit
        # if self.branch == "liquid":
        #     return [(rho_limit,), (self.rho_liq,)]
        # elif self.branch == "vapor":
        #     return [(self.rho_vap,), (rho_limit,)]
        # else:
        #     raise ValueError(f"Invalid value for branch={self.branch}")

    def get_nec(self):
        return 0

    def get_nic(self):
        return 0


# ------------------------------------------------------------------------------------ #
# Other calculations
# ------------------------------------------------------------------------------------ #


# def states_to_dict(states):
#     """
#     Recursively convert a list of FluidState objects into a nested dictionary
#     of NumPy arrays. Handles missing keys by inserting NaNs or None.

#     Parameters
#     ----------
#     states : list of FluidState
#         A list of FluidState objects. Keys may differ between objects.

#     Returns
#     -------
#     dict
#         Nested dictionary where leaf nodes are 1D numpy arrays of field values.
#         Missing keys are filled with NaN or None.
#     """
#     # from .jax_import import jax, jnp, JAX_AVAILABLE
#     # np = jnp

#     from collections.abc import Mapping

#     def extract_value(state, key):
#         if hasattr(state, "get"):
#             return state.get(key, None)
#         return getattr(state, key, None)

#     def is_nested(value):
#         return isinstance(value, Mapping) or hasattr(value, "keys")

#     # Collect the union of all keys across states
#     all_keys = set()
#     for state in states:
#         all_keys.update(state.keys())

#     result = {}

#     for key in sorted(all_keys):
#         values = [extract_value(s, key) for s in states]

#         # Check if this field is nested (but skip None)
#         nested_values = [v for v in values if v is not None]
#         if nested_values and is_nested(nested_values[0]):
#             result[key] = states_to_dict(nested_values)
#         else:
#             # If values are mixed (e.g., some None), fill with None or np.nan
#             if all(isinstance(v, (int, float, np.number, type(None))) for v in values):
#                 result[key] = np.array([np.nan if v is None else v for v in values])
#             else:
#                 result[key] = np.array(values, dtype=object)

#     return result


# def states_to_dict_2d(states):
#     """
#     Convert a 2D list (grid) of state objects into a dictionary.

#     Each key is a field name of the state objects, and each value is a 2D Numpy array of all the values for that field.

#     Parameters
#     ----------
#     states_grid : list of lists of FluidState
#         A 2D grid where each element is a state object with the same keys.

#     Returns
#     -------
#     dict
#         A dictionary where keys are field names and values are 2D arrays of field values.
#     """
#     state_dict_2d = {}
#     m, n = len(states), len(states[0])
#     for i, row in enumerate(states):
#         for j, state in enumerate(row):
#             for field, value in state.items():
#                 if field not in state_dict_2d:
#                     dtype = type(value)  # Determine dtype from the first occurrence
#                     state_dict_2d[field] = np.empty((m, n), dtype=dtype)
#                 state_dict_2d[field][i, j] = value
#     return state_dict_2d


def compute_quality_grid(fluid, num_points, quality_levels, dT_crit=1.0):
    """Return a common state on quality/temperature axes, without source bypasses.

    HEOS pseudo-pure models implement QT only at bubble/dew endpoints. Interior
    quality is therefore rejected explicitly here; use ordinary PQ calls when
    a pressure/quality grid is the intended physical geometry.
    """
    # Define temperature levels
    temperature_levels = _saturation_temperature_levels(fluid, num_points, dT_crit)
    quality_levels = np.asarray(quality_levels)
    if (
        quality_levels.ndim != 1
        or not quality_levels.size
        or quality_levels.dtype.kind not in "fiu"
        or not np.all(np.isfinite(quality_levels))
        or np.any((quality_levels < 0) | (quality_levels > 1))
    ):
        raise InvalidConfigurationError(
            request="quality_levels",
            reason="Expected a nonempty vector of finite qualities in [0, 1].",
        )
    if (
        fluid.provider in {"coolprop", "coolprop_jax"}
        and fluid.references.critical.classification == "pseudo_pure"
        and fluid.references.critical.backend == "HEOS"
        and np.any((quality_levels > 0) & (quality_levels < 1))
    ):
        raise UnsupportedOperationError(
            fluid_name=fluid.name,
            provider=fluid.provider,
            model=fluid.model,
            request="quality_grid",
            reason="HEOS pseudo-pure QT grids support only qualities 0 and 1. Use PQ calls for interior pressure/quality states.",
            supported=(
                "quality_levels=(0, 1)",
                "get_state(PQ_INPUTS, pressure, quality)",
            ),
        )

    # Create meshgrid for (quality, temperature)
    Q, T = np.meshgrid(quality_levels, temperature_levels, indexing="ij")

    # Vectorized property evaluation
    return fluid.get_state_saturation(CP.QT_INPUTS, Q, T)


# def compute_property_grid(
#     fluid,
#     input_pair,
#     range_1,
#     range_2,
#     generalize_quality=False,
#     supersaturation=False,
# ):
#     """
#     Compute fluid properties over a specified range and store them in a dictionary.

#     This function creates a meshgrid of property values based on the specified ranges and input pair,
#     computes the properties of the fluid at each point on the grid, and stores the results in a
#     dictionary where each key corresponds to a fluid property.

#     Parameters
#     ----------
#     fluid : Fluid object
#         An instance of the Fluid class.
#     input_pair : tuple
#         The input pair specifying the property type (e.g., PT_INPUTS for pressure-temperature).
#     range1 : tuple
#         The range linspace(min, max, n) for the first property of the input pair.
#     range2 : tuple
#         The range linspace(min, max, n) for the second property of the input pair.

#     Returns
#     -------
#     properties_dict : dict
#         A dictionary where keys are property names and values are 2D numpy arrays of computed properties.
#     grid1, grid2 : numpy.ndarray
#         The meshgrid arrays for the first and second properties.
#     """

#     # Create the meshgrid
#     grid1, grid2 = np.meshgrid(range_1, range_2)

#     # Initialize dictionary to store properties and pre-allocate storage
#     properties_dict = {}
#     n, m = len(range_1), len(range_2)

#     # Compute properties at each point
#     for i in range(m):
#         for j in range(n):
#             # Set state of the fluid
#             state = fluid.get_state(
#                 input_pair,
#                 grid1[i, j],
#                 grid2[i, j],
#                 generalize_quality=generalize_quality,
#                 supersaturation=supersaturation,
#             )

#             # Store the properties (initialize as empty array if new key)
#             for key, value in state.items():
#                 if key not in properties_dict.keys():
#                     dtype = type(value)  # Determine dtype from the first occurrence
#                     properties_dict[key] = np.empty((m, n), dtype=dtype)
#                     # properties_dict[key] = np.zeros_like(grid1)
#                 properties_dict[key][i, j] = state[key]

#     return properties_dict


def compute_property_grid_rhoT(
    fluid,
    rho_array,
    T_array,
):
    """Evaluate homogeneous states on temperature rows and density columns.

    The returned FluidState retains identity, optional availability and the
    exact grid shape. Calls use the configured fluid's locked/reference-aware
    public homogeneous path rather than mutating its raw source handle.
    """
    axes = []
    for name, values in (("rho_array", rho_array), ("T_array", T_array)):
        array = np.asarray(values)
        if (
            array.ndim != 1
            or not array.size
            or array.dtype.kind not in "fiu"
            or not np.all(np.isfinite(array))
            or np.any(array <= 0)
        ):
            raise InvalidConfigurationError(
                request=name,
                reason="Expected a nonempty vector of finite positive coordinates.",
            )
        axes.append(array)
    rho_array, T_array = axes

    # Calculate property grid
    states_meta = []
    for T in T_array:
        row = []
        for rho in rho_array:
            row.append(
                fluid.get_state_metastable(
                    "density",
                    rho,
                    "temperature",
                    T,
                    supersaturation=False,
                    generalize_quality=False,
                )
            )
        states_meta.append(FluidState.stack(row))

    return FluidState.stack(states_meta)


# def get_state_Qs(fluid, Q, s):
#     # Define the residual equation
#     def get_residual(p):
#         state = fluid.get_state(PSmass_INPUTS, p, s, generalize_quality=True)
#         residual = Q - state.Q
#         return residual

#     # Solve the scalar equation
#     p_triple = 1.0 * fluid.triple_point_liquid.p
#     p_critical = 1.25 * fluid.critical_point.p
#     bounds = [p_triple, p_critical]
#     sol = scipy.optimize.root_scalar(get_residual, bracket=bounds, method="brentq")

#     # Check if the solver has converged
#     if not sol.converged:
#         raise ValueError("The root-finding algorithm did not converge!")

#     # Compute the outlet state
#     state = fluid.get_state(PSmass_INPUTS, sol.root, s, generalize_quality=True)

#     return state


# def get_isentropic_saturation_state(fluid, s_in):
#     # Calculate saturation sate
#     if s_in < fluid.critical_point.s:
#         state_sat = get_state_Qs(fluid, Q=0.00, s=s_in)
#     else:
#         state_sat = get_state_Qs(fluid, Q=1.00, s=s_in)

#     return state_sat


# # ------------------------------------------------------------------------------------ #
# # Sonic point calculations
# # ------------------------------------------------------------------------------------ #


# class _SonicStateProblem(psv.NonlinearSystemProblem):
#     """ """

#     def __init__(self, fluid, property_pair, prop_1, prop_2):
#         # Calculate the thermodynamic state
#         self.fluid = fluid
#         self.state = fluid.get_state(property_pair, prop_1, prop_2)

#         # Initial guess based in input sstate
#         self.initial_guess = [self.state.d, self.state.T]

#         # # Initial guess based on perfect gass relations
#         # gamma = self.state.gamma
#         # d_star = (2/(gamma + 1)) ** (1/(gamma-1)) * self.state.rho
#         # T_star =  (2/(gamma + 1)) * self.state.T
#         # self.initial_guess = [d_star, T_star]

#     def get_values(self, x):
#         # Ensure x can be indexed and contains exactly two elements
#         if not hasattr(x, "__getitem__") or len(x) != 2:
#             raise ValueError(
#                 "Input x must be a list, tuple or numpy array containing exactly two elements: density and temperature."
#             )

#         # Calculate state for the current density-temperature pair
#         crit_state = self.fluid.get_state(DmassT_INPUTS, x[0], x[1])

#         # Calculate the sonic state residual
#         residual = np.asarray(
#             [
#                 1.0 - (crit_state.h + 0.5 * crit_state.a**2) / self.state.h,
#                 1.0 - crit_state.s / self.state.s,
#             ]
#         )

#         return residual


# class _SonicStateProblem2(psv.OptimizationProblem):
#     """ """

#     def __init__(self, fluid, property_pair, prop_1, prop_2):
#         # Calculate the thermodynamic state
#         self.fluid = fluid
#         self.state = fluid.get_state(property_pair, prop_1, prop_2)

#         # Initial guess based in input sstate
#         self.initial_guess = [self.state.d, self.state.T * 0.9]

#         # # Initial guess based on perfect gass relations
#         # gamma = self.state.gamma
#         # d_star = (2/(gamma + 1)) ** (1/(gamma-1)) * self.state.rho
#         # T_star =  (2/(gamma + 1)) * self.state.T
#         # self.initial_guess = [d_star, T_star]

#     def get_values(self, x):
#         """
#         Compute the residuals for the given density and temperature.

#         Parameters
#         ----------
#         x : list
#             List containing the values for density and temperature.

#         Returns
#         -------
#         np.ndarray
#             Array containing residuals (difference) for the two properties.
#         """

#         # Ensure x can be indexed and contains exactly two elements
#         if not hasattr(x, "__getitem__") or len(x) != 2:
#             raise ValueError(
#                 "Input x must be a list, tuple or numpy array containing exactly two elements: density and temperature."
#             )

#         # Calculate state for the current density-temperature pair
#         crit_state = self.fluid.get_state(DmassT_INPUTS, x[0], x[1])

#         # Calculate the sonic state residual
#         residual = [
#             1.0 - crit_state.s / self.state.s,
#         ]

#         # Objective function
#         self.f = crit_state.d**2 * (self.state.h - crit_state.h)
#         self.f = -self.f / (self.state.d * self.state.a) ** 2

#         # Equality constraints
#         self.c_eq = residual

#         # No inequality constraints given for this problem
#         self.c_ineq = []

#         # Combine objective function and constraints
#         objective_and_constraints = self.merge_objective_and_constraints(
#             self.f, self.c_eq, self.c_ineq
#         )

#         return objective_and_constraints

#     def get_bounds(self):
#         bound_density = (
#             self.fluid.triple_point_vapor.d * 1.5,
#             self.fluid.critical_point.d * 3,
#         )
#         bound_temperature = (
#             self.fluid.triple_point_vapor.T * 1,
#             self.fluid.critical_point.T * 3,
#         )
#         # return [bound_density, bound_temperature]
#         return None

#     def get_n_eq(self):
#         return self.get_number_of_constraints(self.c_eq)

#     def get_n_ineq(self):
#         return self.get_number_of_constraints(self.c_ineq)


# ------------------------------------------------------------------------------------ #
# Utility functions
# ------------------------------------------------------------------------------------ #


def safe_alias_split(raw_string):
    """
    Split a CoolProp alias string safely, preserving multi-part names
    that include commas (e.g., chemical names like '1,2-dichloroethane').

    Parameters
    ----------
    raw_string : str
        Comma-separated alias string from CoolProp.

    Returns
    -------
    list of str
        Cleaned list of alias names, with spurious splits recombined.
    """
    tokens = raw_string.split(",")
    merged = []
    buffer = ""

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            if buffer:
                buffer += "," + token
            else:
                buffer = token
        else:
            if buffer:
                buffer += "," + token
                merged.append(buffer)
                buffer = ""
            else:
                merged.append(token)
    if buffer:
        merged.append(buffer)
    return merged


def print_fluid_names():
    """
    Print all canonical fluids and their aliases in a structured format.
    """
    canonical_fluids = CP.get_global_param_string("FluidsList").split(",")

    for fluid in canonical_fluids:
        raw_aliases = CP.get_fluid_param_string(fluid, "aliases")
        aliases = safe_alias_split(raw_aliases)
        print(f"{fluid}:")
        for alias in aliases:
            print(f"  - {alias}")
        print()  # empty line between fluids


def build_alias_map():
    """
    Build a mapping from alias names to canonical CoolProp fluid names.

    Returns
    -------
    dict
        Dictionary where keys are aliases and values are canonical fluid names.
    """
    alias_map = {}
    canonical_fluids = CP.get_global_param_string("FluidsList").split(",")

    for fluid in canonical_fluids:
        raw_aliases = CP.get_fluid_param_string(fluid, "aliases")
        aliases = safe_alias_split(raw_aliases)
        if aliases:
            for alias in aliases:
                alias_map[alias] = fluid
        # Always include canonical name as valid alias
        alias_map[fluid] = fluid

    return alias_map


def try_initialize_fluid(name: str, backend: str = "HEOS"):
    """
    Attempt to initialize a CoolProp AbstractState for the given fluid name.
    If initialization fails, raise a ValueError with close name suggestions.

    Parameters
    ----------
    name : str
        Fluid name or alias to initialize.
    backend : str, optional
        CoolProp backend to use (default is 'HEOS').

    Returns
    -------
    CP.AbstractState
        Initialized CoolProp AbstractState object.

    Raises
    ------
    ValueError
        If the fluid name is not recognized by CoolProp, with close suggestions.
    """
    try:
        return CP.AbstractState(backend, name)
    except ValueError as e:
        alias_map = build_alias_map()
        all_aliases = list(alias_map.keys())
        suggestions = difflib.get_close_matches(name, all_aliases, n=3)

        suggestion_msg = (
            f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        )

        raise ValueError(
            f"CoolProp could not initialize fluid '{name}'.{suggestion_msg}"
        ) from e


def is_pure_substance(AS) -> bool:
    backend = AS.backend_name()

    if backend == "HelmholtzEOSBackend":  # Alias HEOS
        return AS.fluid_param_string("pure").lower() == "true"

    elif backend in ["REFPROPBackend", "BicubicBackend"]:
        return len(AS.fluid_names()) == 1 and AS.name().lower() not in [
            "air",
            "air (dry)",
            "r404a",
            "r407c",
            "r410a",
            "r507a",
        ]

    else:
        raise NotImplementedError(
            f"Purity check not implemented for backend '{backend}'"
        )
