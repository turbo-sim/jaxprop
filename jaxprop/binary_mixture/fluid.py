"""A fixed-composition binary model using configured component evaluators.

The parcels share pressure and temperature, without mass exchange or molecular
mixing entropy. This differs from a mixture equation of state. Numerical
providers are selected on the components; no source is initialized by a call.
"""

from functools import partial
from numbers import Real

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero

from .. import helpers_props as pairs
from ..configuration import (
    FluidCapabilities,
    FluidConfiguration,
    FluidDomain,
    build_configuration,
)
from ..errors import (
    CalculationError,
    ConvergenceError,
    InvalidConfigurationError,
    InvalidStateError,
    OutOfDomainError,
    UnsupportedExecutionModeError,
    UnsupportedOperationError,
)
from ..fluid_api import PAIR_NAMES, StatusCode, validate_request
from ..fluid_state import FluidState
from ..source_references import (
    SourceReferences,
    SourceReferencesMixin,
    composite_source_references,
)
from .components import inverse_monotone, inverse_response, temperature_bounds
from .inverses import solve_temperature
from .physics import mix_properties

_EAGER_TRACE_STATE = jax.core.get_opaque_trace_state()
_CORE = ("pressure", "temperature", "density", "enthalpy", "entropy", "internal_energy")
_PROPERTIES = frozenset(_CORE) | frozenset(
    {
        "isobaric_heat_capacity",
        "isochoric_heat_capacity",
        "heat_capacity_ratio",
        "isobaric_expansion_coefficient",
        "isothermal_compressibility",
        "isentropic_compressibility",
        "isothermal_bulk_modulus",
        "isentropic_bulk_modulus",
        "speed_of_sound",
        "speed_of_sound_p",
        "speed_of_sound_pT",
        "gruneisen",
        "isothermal_joule_thomson",
        "joule_thomson",
        "viscosity",
        "conductivity",
        "mass_frac_1",
        "mass_frac_2",
        "vol_frac_1",
        "vol_frac_2",
        "mixture_ratio",
    }
)
_REQUIRED = frozenset(_CORE) | {
    "isobaric_heat_capacity",
    "isobaric_expansion_coefficient",
    "isothermal_compressibility",
}


class _OpaqueComponent:
    """Keep a host evaluator outside a numerical PyTree, including hybrid inputs."""

    def __init__(self, component):
        self.component = component


class _ComponentSlot(eqx.Module):
    numerical: object
    host: object = eqx.field(static=True)

    def __init__(self, component, dynamic):
        self.numerical = component if dynamic else None
        self.host = None if dynamic else _OpaqueComponent(component)

    @property
    def component(self):
        return self.numerical if self.host is None else self.host.component


def _metadata(fluid, errors):
    return dict(
        fluid_name=fluid.name,
        identifier=fluid.identifier,
        provider=fluid.provider,
        providers=frozenset({fluid.provider}),
        model=fluid.model,
        reference_id=fluid.reference_id,
        available_properties=fluid.capabilities.properties,
        error_policy=errors,
    )


def _safe_component_data(state):
    """Protect failed-row arithmetic before mixing; retain successful optional NaNs.

    A failed component already contains masked NaNs. Feeding those through
    reciprocal volumes could poison zero cotangents before the final mask.
    Finite placeholders depend only on its nondifferentiable status, and the
    original failure code still determines the returned mixture state.
    """
    positive = {
        "density",
        "isobaric_heat_capacity",
        "isochoric_heat_capacity",
        "isothermal_compressibility",
        "isentropic_compressibility",
        "speed_of_sound",
        "pressure",
        "temperature",
    }
    return {
        name: jnp.where(state.status == 0, value, 1.0 if name in positive else 0.0)
        for name, value in state.to_dict().items()
    }


def _raise_failure(fluid, input_type, status, causes=None):
    codes = np.asarray(status)
    positions = np.flatnonzero(codes)
    if not positions.size:
        return
    position = int(positions[0])
    index = (
        tuple(int(i) for i in np.unravel_index(position, codes.shape))
        if codes.ndim
        else ()
    )
    code = int(codes.flat[position])
    kind = {1: InvalidStateError, 2: OutOfDomainError, 3: ConvergenceError}.get(
        code, CalculationError
    )
    reason = {
        1: "Inputs must be finite and pressure/temperature must be positive.",
        2: "The request lies outside the active components' common domain.",
        3: "No unique, residual-validated common-temperature state was obtained inside the component domains.",
        4: "An active component or the mixture produced invalid core properties.",
    }[code]
    error = kind(
        fluid_name=fluid.name,
        provider=fluid.provider,
        model=fluid.model,
        request=PAIR_NAMES[input_type],
        reason=reason,
        index=index,
    )
    cause = None if causes is None else causes.get(index)
    if cause is not None:
        raise error from cause
    raise error


def _native_element(fluid, input_type, first, second, errors):
    """Evaluate one native point, or an analytic PG broadcast in one graph."""
    finite = jnp.isfinite(first) & jnp.isfinite(second)
    p = second if input_type == pairs.HmassP_INPUTS else first
    positive = p > 0
    if input_type == pairs.PT_INPUTS:
        positive = positive & (second > 0)
    safe_p = jnp.where(jnp.isfinite(p), p, fluid._p_fallback)
    outside_p = (
        (safe_p < fluid.p_min) | (safe_p > fluid.p_max)
        if fluid.p_max is not None
        else safe_p < fluid.p_min
    )
    p = jnp.maximum(safe_p, fluid.p_min)
    if fluid.p_max is not None:
        p = jnp.minimum(p, fluid.p_max)
    if fluid._endpoint_index is not None:
        component = fluid.components[fluid._endpoint_index]
        # A pure limit uses the component's own inverse and clipping semantics;
        # the inactive component is neither evaluated nor a domain restriction.
        active = component.get_state(
            input_type, first, second, errors="clip" if errors == "clip" else "mask"
        )
        data = _safe_component_data(active)
        properties = mix_properties(data, data, fluid.mass_fractions)
        status, clipped = active.status, active.clipped if errors == "clip" else None
    else:
        low, high = fluid._temperature_bounds(p)
        domain_ok = jnp.isfinite(low) & jnp.isfinite(high) & (low < high)
        if input_type == pairs.PT_INPUTS:
            original_T = jnp.where(jnp.isfinite(second), second, (low + high) / 2)
            outside_T = (original_T < low) | (original_T > high)
            T = jnp.clip(original_T, low, high)
            status = jnp.where(
                outside_T & (errors != "clip"), StatusCode.OUT_OF_DOMAIN, 0
            )
            projected = outside_T
        else:
            target = first if input_type == pairs.HmassP_INPUTS else second
            name = "enthalpy" if input_type == pairs.HmassP_INPUTS else "entropy"
            T, status, projected = solve_temperature(
                fluid, p, target, name, clip=errors == "clip"
            )
        safe_T = jnp.where(jnp.isfinite(T) & domain_ok, T, 300.0)
        states = [
            component.get_state(pairs.PT_INPUTS, p, safe_T, errors="mask")
            for component in fluid.components
        ]
        properties = mix_properties(
            _safe_component_data(states[0]),
            _safe_component_data(states[1]),
            fluid.mass_fractions,
        )
        for state in states:
            status = jnp.where(status == 0, state.status, status)
        status = jnp.where((status == 0) & ~domain_ok, StatusCode.OUT_OF_DOMAIN, status)
        if errors != "clip":
            status = jnp.where(outside_p, StatusCode.OUT_OF_DOMAIN, status)
        clipped = (outside_p | projected | ~positive) if errors == "clip" else None
    core_ok = jnp.ones_like(finite)
    for name in _CORE:
        core_ok = core_ok & jnp.isfinite(properties[name])
    core_ok = core_ok & (properties["temperature"] > 0) & (properties["density"] > 0)
    status = jnp.where((status == 0) & ~core_ok, StatusCode.PROVIDER_FAILURE, status)
    invalid = ~finite | (~positive if errors != "clip" else False)
    status = jnp.where(invalid, StatusCode.INVALID_INPUT, status).astype(jnp.int32)
    return FluidState._from_native(
        _metadata(fluid, errors),
        properties,
        status,
        clipped=clipped,
        mask_failed=errors != "raise",
    )


@partial(jax.jit, static_argnames=("input_type", "errors"))
def _native_state(fluid, input_type, first, second, errors):
    first, second = jnp.broadcast_arrays(first, second)
    # PG equations and pure limits already broadcast. Keeping whole batches
    # together retains the trusted state factory's all-success batch fast path.
    if first.ndim == 0 or fluid._all_perfect_gas or fluid._endpoint_index is not None:
        return _native_element(fluid, input_type, first, second, errors)
    state = jax.vmap(lambda a, b: _native_element(fluid, input_type, a, b, errors))(
        first.ravel(), second.ravel()
    )
    return jax.tree_util.tree_map(lambda value: value.reshape(first.shape), state)


def _callback_impl(evaluator, input_type, first, second):
    shape = jnp.broadcast_shapes(jnp.shape(first), jnp.shape(second))
    dtype = jax.dtypes.canonicalize_dtype(np.float64)
    template = (
        {name: jax.ShapeDtypeStruct(shape, dtype) for name in _PROPERTIES},
        jax.ShapeDtypeStruct(shape, jnp.int32),
    )
    return jax.pure_callback(
        partial(evaluator.callback, input_type, dtype=dtype),
        template,
        first,
        second,
        vmap_method="broadcast_all",
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _callback_values(evaluator, input_type, first, second):
    return _callback_impl(evaluator, input_type, first, second)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _first_order_samples(evaluator, input_type, first, second):
    return _callback_impl(evaluator, input_type, first, second)


@_first_order_samples.defjvp
def _reject_higher(evaluator, input_type, primals, tangents):
    raise UnsupportedExecutionModeError(
        fluid_name=getattr(evaluator, "name", "binary_mixture"),
        provider="native",
        model="binary_mixture",
        request="derivative_order",
        reason="Source-backed binary callbacks support first-order state-input derivatives only.",
        supported=("first-order",),
    )


def _callback_jvp(evaluator, input_type, primals, tangents):
    """Finite differences of a whole transaction, skipping inactive inputs.

    Failed primal rows and unavailable fields have zero derivative coefficients
    before multiplying tangents, avoiding reverse-mode ``0 * NaN``. A valid
    row with a failed forward sample instead has an unavailable NaN derivative.
    """
    first, second = primals
    da, db = tangents
    data, status = _first_order_samples(evaluator, input_type, first, second)
    output_dtype = jax.dtypes.canonicalize_dtype(np.float64)
    step_scale = max(1e-6, np.sqrt(np.finfo(output_dtype).eps))
    step_a, step_b = step_scale * (jnp.abs(first) + 1), step_scale * (
        jnp.abs(second) + 1
    )
    derivative = {name: jnp.zeros_like(value) for name, value in data.items()}
    primal_valid = status == 0
    if not isinstance(da, SymbolicZero):
        left, code = _first_order_samples(evaluator, input_type, first + step_a, second)
        derivative = {
            name: derivative[name]
            + jnp.where(
                primal_valid & jnp.isfinite(value),
                jnp.where(code == 0, (left[name] - value) / step_a, jnp.nan),
                0.0,
            )
            * da
            for name, value in data.items()
        }
    if not isinstance(db, SymbolicZero):
        right, code = _first_order_samples(
            evaluator, input_type, first, second + step_b
        )
        derivative = {
            name: derivative[name]
            + jnp.where(
                primal_valid & jnp.isfinite(value),
                jnp.where(code == 0, (right[name] - value) / step_b, jnp.nan),
                0.0,
            )
            * db
            for name, value in data.items()
        }
    return (data, status), (
        derivative,
        jnp.zeros(status.shape, dtype=jax.dtypes.float0),
    )


_callback_values.defjvp(_callback_jvp, symbolic_zeros=True)


class FluidBinaryMixture(SourceReferencesMixin, eqx.Module):
    """Fixed mass fractions of two component fluids in mechanical/thermal equilibrium.

    Parameters
    ----------
    name : str
        User label; component names and backends stay on the component objects.
    components : pair of configured fluid objects
        Components must provide PT core states and thermal response properties.
        Native PG/bicubic components remain dynamic JAX model data. Host/hybrid
        configurations are fixed snapshots; reconstruct to change their models.
    mass_fractions : pair of nonnegative real scalars
        Sum to one, without normalization. Both pure-component limits are valid.
        Composition is fixed, never an extra positional state input.
    identifier : str or None
        Stream label preserved on complete returned FluidState objects.
    inverse_max_steps, inverse_tolerance : int, float
        Setup-only iteration budget and scaled residual tolerance for HP/PS.

    The main sound speed assumes pressure and temperature relaxation. The named
    ``speed_of_sound_p`` field retains the distinct frozen-thermal Wood speed.
    Component mass fractions are not vapor qualities; saturation is unsupported.
    """

    name: str = eqx.field(static=True)
    identifier: object = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    model: str = eqx.field(static=True)
    backend: object = eqx.field(static=True)
    mass_fractions: tuple = eqx.field(static=True)
    inverse_max_steps: int = eqx.field(static=True)
    inverse_tolerance: float = eqx.field(static=True)
    p_min: float = eqx.field(static=True)
    p_max: object = eqx.field(static=True)
    _p_fallback: float = eqx.field(static=True)
    _execution: str = eqx.field(static=True)
    _all_perfect_gas: bool = eqx.field(static=True)
    _endpoint_index: object = eqx.field(static=True)
    _components: tuple
    _host_evaluator: object = eqx.field(static=True)
    _configuration: object = eqx.field(static=True)
    _capabilities: object = eqx.field(static=True)
    _references: object = eqx.field(static=True)
    reference_id: tuple = eqx.field(static=True)

    def __init__(
        self,
        name,
        components,
        mass_fractions,
        identifier=None,
        *,
        inverse_max_steps=64,
        inverse_tolerance=1e-10
    ):
        context = dict(fluid_name=name, provider="native", model="binary_mixture")

        def invalid(request, reason):
            raise InvalidConfigurationError(**context, request=request, reason=reason)

        build_configuration(name, "native", "binary_mixture", None, identifier)
        if not isinstance(components, (tuple, list)) or len(components) != 2:
            invalid(
                "components", "Provide exactly two configured component fluid objects."
            )
        for component in components:
            if (
                not callable(getattr(component, "get_state", None))
                or not isinstance(
                    getattr(component, "configuration", None), FluidConfiguration
                )
                or not isinstance(
                    getattr(component, "capabilities", None), FluidCapabilities
                )
                or not isinstance(
                    getattr(component, "references", None), SourceReferences
                )
            ):
                invalid(
                    "components",
                    "Components must expose the configured fluid interface and source references.",
                )
            if (component.provider, component.model) not in {
                ("coolprop", "real_fluid"),
                ("coolprop_jax", "real_fluid"),
                ("bicubic", "real_fluid"),
                ("native", "perfect_gas"),
            }:
                invalid(
                    "components",
                    "Use supported source, callback, bicubic or perfect-gas component evaluators.",
                )
            if (
                pairs.PT_INPUTS not in component.capabilities.input_pairs
                or not _REQUIRED <= component.capabilities.properties
                or component.model == "binary_mixture"
            ):
                invalid(
                    "components",
                    "Use single-fluid components with PT core states, cp, expansion and isothermal compressibility.",
                )
        if (
            not isinstance(mass_fractions, (tuple, list))
            or len(mass_fractions) != 2
            or any(
                isinstance(w, (bool, np.bool_))
                or not isinstance(w, Real)
                or not np.isfinite(w)
                or w < 0
                for w in mass_fractions
            )
            or not np.isclose(
                sum(float(w) for w in mass_fractions), 1.0, rtol=0, atol=1e-12
            )
        ):
            invalid(
                "mass_fractions",
                "Provide two finite nonnegative scalar mass fractions summing to one without normalization.",
            )
        if (
            isinstance(inverse_max_steps, (bool, np.bool_))
            or not isinstance(inverse_max_steps, (int, np.integer))
            or inverse_max_steps < 0
        ):
            invalid("inverse_max_steps", "Use a nonnegative integer iteration budget.")
        if (
            isinstance(inverse_tolerance, (bool, np.bool_))
            or not isinstance(inverse_tolerance, Real)
            or not np.isfinite(inverse_tolerance)
            or inverse_tolerance <= 0
        ):
            invalid(
                "inverse_tolerance", "Use a finite positive scaled residual tolerance."
            )
        self.name, self.identifier = name, identifier
        self.provider, self.model, self.backend = "native", "binary_mixture", None
        self.mass_fractions = tuple(float(w) for w in mass_fractions)
        self.inverse_max_steps, self.inverse_tolerance = int(inverse_max_steps), float(
            inverse_tolerance
        )
        active = [c for c, w in zip(components, self.mass_fractions) if w != 0]
        native = all(
            c.provider == "bicubic"
            or (c.provider == "native" and c.model == "perfect_gas")
            for c in active
        )
        callback = all(c.provider == "coolprop_jax" for c in active)
        self._execution = "native" if native else "callback" if callback else "host"
        self._all_perfect_gas = all(c.model == "perfect_gas" for c in components)
        self._endpoint_index = (
            0
            if self.mass_fractions[1] == 0
            else 1 if self.mass_fractions[0] == 0 else None
        )
        self._components = tuple(
            _ComponentSlot(
                c,
                native
                and (
                    c.provider == "bicubic"
                    or (c.provider == "native" and c.model == "perfect_gas")
                ),
            )
            for c in components
        )
        lower, upper = 0.0, None
        for component in active:
            low, high = component.capabilities.domain.bounds.get(
                "pressure", (0.0, None)
            )
            lower = max(lower, low or 0.0)
            if high is not None:
                upper = high if upper is None else min(upper, high)
        if upper is not None and not lower < upper:
            invalid("components", "The active pressure domains do not overlap.")
        self.p_min, self.p_max = lower, upper
        self._p_fallback = (
            max(lower, min(101325.0, upper))
            if upper is not None
            else max(lower, 101325.0)
        )
        self._host_evaluator = None
        if not native:
            from .host import HostEvaluator

            self._host_evaluator = HostEvaluator(
                tuple(components),
                self.mass_fractions,
                inverse_max_steps=self.inverse_max_steps,
                inverse_tolerance=self.inverse_tolerance,
            )
            self._host_evaluator.name = name
        self._configuration = build_configuration(
            name,
            "native",
            "binary_mixture",
            None,
            identifier,
            model_options={
                "components": tuple(c.configuration for c in components),
                "mass_fractions": self.mass_fractions,
                "inverse_max_steps": self.inverse_max_steps,
                "inverse_tolerance": self.inverse_tolerance,
                "acoustic_model": "pressure_temperature_equilibrium",
            },
        )
        modes = (
            {"eager", "jit", "vmap", "jvp", "grad"} if native or callback else {"eager"}
        )
        policies = (
            {"raise", "mask", "clip"}
            if native and all("clip" in c.capabilities.error_modes for c in active)
            else {"raise", "mask"}
        )
        self._capabilities = FluidCapabilities(
            input_pairs={pairs.PT_INPUTS, pairs.HmassP_INPUTS, pairs.PSmass_INPUTS},
            properties=_PROPERTIES,
            jax_modes=modes,
            error_modes=policies,
            derivative_order=1 if native or callback else 0,
            derivative_method=(
                "autodiff" if native else "finite_difference" if callback else None
            ),
            domain=FluidDomain(
                description="Fixed-composition common-p/T parcels without mass exchange. Active component domains and phase restrictions apply.",
                bounds={"pressure": (lower, upper)},
                phases={"common_pressure_temperature"},
            ),
        )
        self._references = composite_source_references(
            name,
            None,
            tuple(c.references for c in components),
            mass_fractions=self.mass_fractions,
            component_reference_ids=tuple(c.reference_id for c in components),
        )
        self.reference_id = self._references.reference_id

    @property
    def components(self):
        return tuple(slot.component for slot in self._components)

    @property
    def configuration(self):
        return self._configuration

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def mixture_ratio(self):
        return (
            self.mass_fractions[0] / self.mass_fractions[1]
            if self.mass_fractions[1]
            else float("inf")
        )

    def _temperature_bounds(self, pressure):
        return temperature_bounds(self, pressure)

    def _inverse_response(self, pressure, temperature, name):
        return inverse_response(self, pressure, temperature, name)

    def _inverse_monotone(self, pressure, name):
        return inverse_monotone(self, pressure, name)

    def get_state(
        self,
        input_type,
        prop_1,
        prop_2,
        *,
        generalize_quality=False,
        supersaturation=False,
        errors="raise",
        **options
    ):
        """Return complete broadcast states using a static pair and fixed composition.

        Numeric inputs and native component parameters remain dynamic. Eager
        raising reports the first failed row. Transformed supported providers
        require explicit mask/clip; source/callback clipping is not advertised.
        """
        transformed = jax.core.get_opaque_trace_state() != _EAGER_TRACE_STATE
        first, second = validate_request(
            self,
            input_type,
            prop_1,
            prop_2,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
            errors=errors,
            options=options,
            array_backend=(
                "numpy" if self._execution == "host" and not transformed else "jax"
            ),
        )
        if transformed and (errors == "raise" or self._execution == "host"):
            raise UnsupportedExecutionModeError(
                fluid_name=self.name,
                provider=self.provider,
                model=self.model,
                request="errors=raise" if errors == "raise" else "JAX execution",
                reason="Transformed calls require an implemented JAX component path and explicit mask/clip policy; direct-source/hybrid models are eager.",
                supported=("errors=mask",),
            )
        if self._execution == "native":
            state = _native_state(self, input_type, first, second, errors)
            if errors == "raise":
                _raise_failure(self, input_type, state.status)
            return state
        if self._execution == "callback" and (transformed or errors == "mask"):
            data, status = _callback_values(
                self._host_evaluator, input_type, first, second
            )
            return FluidState._from_native(_metadata(self, errors), data, status)
        data, status, causes = self._host_evaluator.evaluate(input_type, first, second)
        if errors == "raise":
            _raise_failure(self, input_type, status, causes)
        return FluidState._from_native(
            _metadata(self, errors), data, status, mask_failed=errors != "raise"
        )

    def get_state_saturation(self, *args, **kwargs):
        raise UnsupportedOperationError(
            fluid_name=self.name,
            provider=self.provider,
            model=self.model,
            request="saturation",
            reason="A fixed-composition parcel model has no defined mixture saturation curve; use the component fluids.",
        )
