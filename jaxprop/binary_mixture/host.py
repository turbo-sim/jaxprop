"""Efficient host evaluation of the fixed-composition binary model.

Source-backed inverse iterations update the already configured CoolProp states
and extract only enthalpy or entropy. Complete component properties are assembled
once after a root passes its independent residual check. A transaction holds all
active source locks in a common order, including the final property extraction.
No source or reference state is constructed during evaluation.

This helper owns a fixed component configuration. Native component constants
used in a host/source hybrid are snapshotted at construction; rebuild the owning
fluid to change those models. Host callbacks support numerical state-input
derivatives, not differentiation of mutable provider handles/model parameters.
"""

from contextlib import ExitStack
import math
from types import SimpleNamespace

import CoolProp.CoolProp as CP
import numpy as np
from scipy.optimize import root_scalar

from ..coolprop import core_calculations as source_core
from ..errors import ConvergenceError, InvalidStateError, OutOfDomainError
from ..fluid_api import StatusCode
from .physics import _COMPONENT_FIELDS, mix_properties

_NAMES = _COMPONENT_FIELDS + (
    "speed_of_sound_p",
    "speed_of_sound_pT",
    "mass_frac_1",
    "mass_frac_2",
    "vol_frac_1",
    "vol_frac_2",
    "mixture_ratio",
)
_CORE = ("pressure", "temperature", "density", "enthalpy", "internal_energy", "entropy")
# Borrowed legacy AbstractStates have no configured quality scratch state. The
# binary model does not use quality, so explicitly omit that diagnostic instead
# of creating an unnecessary helper AbstractState during property extraction.
_NO_QUALITY_CONTEXT = SimpleNamespace(critical_entropy=None)


class _Failure(Exception):
    def __init__(self, code, cause):
        self.code, self.cause = int(code), cause
        super().__init__(str(cause))


def _status_from_error(error):
    if isinstance(error, OutOfDomainError):
        return StatusCode.OUT_OF_DOMAIN
    if isinstance(error, InvalidStateError):
        return StatusCode.INVALID_INPUT
    if isinstance(error, ConvergenceError):
        return StatusCode.NONCONVERGENCE
    return StatusCode.PROVIDER_FAILURE


class _Component:
    """Setup-prepared provider access for one active component."""

    def __init__(self, component):
        self.original = component
        self.base = (
            component.fluid
            if getattr(component, "provider", None) == "coolprop_jax"
            else component
        )
        self.source = getattr(self.base, "_AS", None)
        self.lock = getattr(self.base, "_state_lock", None)
        self.context = getattr(self.base, "_source_context", _NO_QUALITY_CONTEXT)
        self.ideal = getattr(self.base, "model", None) == "perfect_gas"
        self.bicubic = getattr(self.base, "provider", None) == "bicubic"
        self.constants = None
        if self.ideal:
            self.constants = {
                name: float(np.asarray(getattr(self.base.constants, name)))
                for name in (
                    "R",
                    "gamma",
                    "T_ref",
                    "P_ref",
                    "h_ref",
                    "s_ref",
                    "mu_ref",
                    "S_mu",
                    "k_ref",
                    "S_k",
                )
            }
            self.cp = (
                self.constants["gamma"]
                * self.constants["R"]
                / (self.constants["gamma"] - 1)
            )
        domain = getattr(getattr(self.base, "capabilities", None), "domain", None)
        bounds = {} if domain is None else domain.bounds
        self.T_min, self.T_max = bounds.get("temperature", (None, None))
        self.p_min, self.p_max = bounds.get("pressure", (None, None))
        self.rho_min = bounds.get("density", (None, None))[0]
        if self.source is not None:
            self.T_min = self.source.Tmin() if self.T_min is None else self.T_min
            self.T_max = self.source.Tmax() if self.T_max is None else self.T_max
            self.p_max = self.source.pmax() if self.p_max is None else self.p_max
        elif self.bicubic:
            eos_domain = self.base.metadata.get("source_domain", {})
            self.T_min = eos_domain.get("T_min", self.T_min)
            self.T_max = eos_domain.get("T_max", self.T_max)
        self.boundary_precision = 1e-9 if self.source is not None else 1e-12

    def check_coordinates(self, pressure, temperature=None):
        """Use the component's declared domain and numerical boundary precision."""
        eps = self.boundary_precision
        outside = (self.p_min is not None and pressure < self.p_min * (1 - eps)) or (
            self.p_max is not None and pressure > self.p_max * (1 + eps)
        )
        if temperature is not None:
            outside = outside or (
                (self.T_min is not None and temperature < self.T_min * (1 - eps))
                or (self.T_max is not None and temperature > self.T_max * (1 + eps))
            )
        if outside:
            raise _Failure(
                StatusCode.OUT_OF_DOMAIN,
                ValueError(
                    "The request exceeds an active component's declared p/T domain."
                ),
            )

    def selected(self, pressure, temperature, name):
        if self.source is not None:
            self.source.update(CP.PT_INPUTS, pressure, temperature)
            return self.source.hmass() if name == "enthalpy" else self.source.smass()
        if self.ideal:
            c = self.constants
            if name == "enthalpy":
                return c["h_ref"] + self.cp * (temperature - c["T_ref"])
            return (
                c["s_ref"]
                + self.cp * (math.log(temperature) - math.log(c["T_ref"]))
                - c["R"] * (math.log(pressure) - math.log(c["P_ref"]))
            )
        state = self.original.get_state(
            CP.PT_INPUTS, pressure, temperature, errors="mask"
        )
        code = int(np.asarray(state.status))
        if code:
            raise _Failure(
                code,
                ValueError(
                    "An active component could not evaluate the inverse sample."
                ),
            )
        return float(np.asarray(state.raw_property(name)))

    def full(self, pressure, temperature, *, reuse_source=False):
        if self.source is not None:
            if not reuse_source:
                self.source.update(CP.PT_INPUTS, pressure, temperature)
            return source_core.compute_properties_1phase(
                self.source, source_context=self.context
            )
        if not self.ideal:
            state = self.original.get_state(
                CP.PT_INPUTS, pressure, temperature, errors="mask"
            )
            code = int(np.asarray(state.status))
            if code:
                raise _Failure(
                    code,
                    ValueError(
                        "An active component could not evaluate the recovered state."
                    ),
                )
            return state.to_dict()
        c = self.constants
        R, gamma, cp = c["R"], c["gamma"], self.cp
        density = pressure / (R * temperature)
        if self.rho_min is not None and density < self.rho_min * (1 - 1e-12):
            raise _Failure(
                StatusCode.OUT_OF_DOMAIN,
                ValueError("The ideal component density is below its model floor."),
            )
        h = c["h_ref"] + cp * (temperature - c["T_ref"])
        s = self.selected(pressure, temperature, "entropy")

        def transport(reference, coefficient):
            value, S = c[reference], c[coefficient]
            if (
                not math.isfinite(value)
                or not math.isfinite(S)
                or 1 + S / temperature == 0
            ):
                return np.nan
            with np.errstate(all="ignore"):
                fitted = (
                    value
                    * (np.sqrt(temperature) / np.sqrt(c["T_ref"]))
                    * (1 + S / c["T_ref"])
                    / (1 + S / temperature)
                )
            return fitted if np.isfinite(fitted) and fitted > 0 else np.nan

        return {
            "pressure": pressure,
            "temperature": temperature,
            "density": density,
            "enthalpy": h,
            "internal_energy": h - R * temperature,
            "entropy": s,
            "isobaric_heat_capacity": cp,
            "isochoric_heat_capacity": R / (gamma - 1),
            "heat_capacity_ratio": gamma,
            "isobaric_expansion_coefficient": 1 / temperature,
            "isothermal_compressibility": 1 / pressure,
            "isentropic_compressibility": (1 / pressure) / gamma,
            "isothermal_bulk_modulus": pressure,
            "isentropic_bulk_modulus": (
                gamma * pressure if math.isfinite(gamma * pressure) else np.nan
            ),
            "speed_of_sound": math.sqrt(gamma) * math.sqrt(R * temperature),
            "gruneisen": gamma - 1,
            "isothermal_joule_thomson": 0.0,
            "joule_thomson": 0.0,
            "viscosity": transport("mu_ref", "S_mu"),
            "conductivity": transport("k_ref", "S_k"),
        }

    def pure(self, pair, first, second):
        if self.source is not None and hasattr(self.base, "_evaluate_state_data"):
            data, _ = self.base._evaluate_state_data(
                pair, np.asarray(first), np.asarray(second), errors="raise"
            )
            return data
        if self.source is not None:
            self.source.update(pair, first, second)
            return source_core.compute_properties_1phase(
                self.source, source_context=self.context
            )
        state = self.original.get_state(pair, first, second, errors="mask")
        code = int(np.asarray(state.status))
        if code:
            raise _Failure(
                code, ValueError("The active pure component rejected the request.")
            )
        return state.to_dict()

    def temperature_bounds(self, pressure):
        lower = 1.0 if self.T_min is None else float(self.T_min)
        upper = math.inf if self.T_max is None else float(self.T_max)
        if self.ideal and self.rho_min is not None:
            # The PG density floor also limits temperature at fixed pressure.
            # Establish this bound before inverse samples, rather than solving
            # through states that the component would reject at final assembly.
            # Divide sequentially so R*rho_min need not be representable.
            upper = min(upper, (pressure / self.constants["R"]) / self.rho_min)
        if self.bicubic:
            # A rectangular h/p table has pressure-dependent temperature edges.
            # Probe those edges once; a global Tmin/Tmax alone is not a valid
            # PT bracket for the local interpolation domain.
            temperatures = []
            for h in (self.base.h_min, self.base.h_max):
                state = self.original.get_state(
                    CP.HmassP_INPUTS, h, pressure, errors="mask"
                )
                if int(np.asarray(state.status)):
                    raise _Failure(
                        StatusCode.NONCONVERGENCE,
                        ValueError(
                            "The table has no connected valid temperature bracket at this pressure."
                        ),
                    )
                temperatures.append(
                    float(np.asarray(state.raw_property("temperature")))
                )
            lower, upper = max(lower, min(temperatures)), min(upper, max(temperatures))
        return lower, upper


class HostEvaluator:
    """Setup-prepared scalar/broadcast PT, HP and PS mixture evaluation.

    ``evaluate`` returns numerical arrays, one status array and an indexed cause
    mapping. It never retains a mutable last-error transaction. ``callback``
    drops the Python causes and returns only fixed-shape numerical callback data.
    The owning public fluid validates static options and handles eager raising.
    """

    names = _NAMES

    def __init__(
        self,
        components,
        mass_fractions,
        *,
        inverse_max_steps=64,
        inverse_tolerance=1e-10
    ):
        self.components = tuple(components)
        self.mass_fractions = tuple(mass_fractions)
        self.total_fraction = sum(float(w) for w in mass_fractions)
        self.inverse_max_steps = inverse_max_steps
        self.inverse_tolerance = inverse_tolerance
        self.entries = tuple(
            _Component(component) if weight else None
            for component, weight in zip(components, mass_fractions)
        )
        self.active = tuple(entry for entry in self.entries if entry is not None)
        locks = {
            id(entry.lock): entry.lock
            for entry in self.active
            if entry.lock is not None
        }
        self.locks = tuple(locks[key] for key in sorted(locks))
        self.same_component = (
            self.entries[0] is not None
            and self.entries[1] is not None
            and self.entries[0].base is self.entries[1].base
        )

    def _selected(self, pressure, temperature, name):
        if self.same_component:
            return self.total_fraction * self.entries[0].selected(
                pressure, temperature, name
            )
        return sum(
            weight * entry.selected(pressure, temperature, name)
            for entry, weight in zip(self.entries, self.mass_fractions)
            if entry is not None
        )

    def _temperature(self, pair, pressure, target):
        if self.inverse_max_steps == 0:
            raise _Failure(
                StatusCode.NONCONVERGENCE,
                ValueError("The inverse iteration budget is zero."),
            )
        bounds = tuple(entry.temperature_bounds(pressure) for entry in self.active)
        lower, upper = max(value[0] for value in bounds), min(
            value[1] for value in bounds
        )
        if not lower < upper:
            raise _Failure(
                StatusCode.NONCONVERGENCE,
                ValueError("Active components have no common temperature interval."),
            )
        name = "enthalpy" if pair == CP.HmassP_INPUTS else "entropy"
        left = self._selected(pressure, lower, name)
        finite_upper = math.isfinite(upper)
        if not finite_upper:
            upper = max(2 * lower, 300.0)
        right = self._selected(pressure, upper, name)
        if not finite_upper:
            for _ in range(self.inverse_max_steps):
                if (left - target) * (right - target) <= 0:
                    break
                upper *= 2
                right = self._selected(pressure, upper, name)
        scale = max(
            abs(right - left), 1e-12 * max(abs(left), abs(right), abs(target)), 1.0
        )

        def residual(temperature):
            value = (self._selected(pressure, temperature, name) - target) / scale
            if not math.isfinite(value):
                raise ValueError("The inverse sample produced a nonfinite residual.")
            return value

        try:
            result = root_scalar(
                residual,
                bracket=(lower, upper),
                method="brentq",
                maxiter=self.inverse_max_steps,
                xtol=max(np.nextafter(0.0, 1.0), min(self.inverse_tolerance, 1e-10)),
                rtol=max(4 * np.finfo(float).eps, min(self.inverse_tolerance, 1e-12)),
            )
            temperature = float(result.root)
            acceptable = (
                result.converged
                and lower <= temperature <= upper
                and abs(residual(temperature))
                <= max(self.inverse_tolerance, 64 * np.finfo(float).eps)
            )
            if not acceptable:
                raise ValueError(
                    "No residual-validated mixture inverse was obtained; a phase gap or iteration limit may prevent a root."
                )
        except (
            ValueError,
            RuntimeError,
            OverflowError,
            ZeroDivisionError,
            _Failure,
        ) as error:
            cause = error.cause if isinstance(error, _Failure) else error
            raise _Failure(StatusCode.NONCONVERGENCE, cause) from error
        return temperature

    def _scalar(self, pair, first, second):
        if not math.isfinite(first) or not math.isfinite(second):
            raise _Failure(
                StatusCode.INVALID_INPUT, ValueError("Inputs must be finite.")
            )
        pressure = second if pair == CP.HmassP_INPUTS else first
        temperature = second if pair == CP.PT_INPUTS else None
        if pressure <= 0 or (temperature is not None and temperature <= 0):
            raise _Failure(
                StatusCode.INVALID_INPUT,
                ValueError("Pressure and temperature must be positive."),
            )
        for entry in self.active:
            entry.check_coordinates(pressure, temperature)
        if len(self.active) == 1:
            raw = self.active[0].pure(pair, first, second)
            return mix_properties(
                raw if self.entries[0] is not None else None,
                raw if self.entries[1] is not None else None,
                self.mass_fractions,
                xp=np,
            )
        if temperature is None:
            temperature = self._temperature(
                pair, pressure, first if pair == CP.HmassP_INPUTS else second
            )
        raw_first = self.entries[0].full(
            pressure, temperature, reuse_source=pair != CP.PT_INPUTS
        )
        raw_second = (
            raw_first
            if self.same_component
            else self.entries[1].full(
                pressure, temperature, reuse_source=pair != CP.PT_INPUTS
            )
        )
        result = mix_properties(raw_first, raw_second, self.mass_fractions, xp=np)
        # p and T are the common independent coordinates. A liquid EOS can
        # reconstruct p with small cancellation roundoff after its PT flash;
        # preserve the supplied pressure and the validated root temperature.
        result["pressure"], result["temperature"] = pressure, temperature
        return result

    def evaluate(self, pair, first, second):
        if pair not in (CP.PT_INPUTS, CP.HmassP_INPUTS, CP.PSmass_INPUTS):
            raise ValueError(
                "The host binary evaluator supports PT, HmassP and PSmass only."
            )
        first, second = np.broadcast_arrays(
            np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
        )
        values = {
            name: np.full(first.shape, np.nan, dtype=np.float64) for name in _NAMES
        }
        status = np.zeros(first.shape, dtype=np.int32)
        causes = {}
        with ExitStack() as transaction:
            for lock in self.locks:
                transaction.enter_context(lock)
            for index in np.ndindex(first.shape):
                try:
                    raw = self._scalar(pair, float(first[index]), float(second[index]))
                    if not all(math.isfinite(raw[name]) for name in _CORE) or any(
                        raw[name] <= 0
                        for name in ("pressure", "temperature", "density")
                    ):
                        raise _Failure(
                            StatusCode.PROVIDER_FAILURE,
                            ValueError(
                                "The mixture produced nonfinite or nonphysical core properties."
                            ),
                        )
                    for name in _NAMES:
                        values[name][index] = raw[name]
                except _Failure as error:
                    status[index], causes[index] = error.code, error.cause
                except (
                    ValueError,
                    RuntimeError,
                    OverflowError,
                    ZeroDivisionError,
                    AttributeError,
                    TypeError,
                    OSError,
                ) as error:
                    status[index], causes[index] = int(_status_from_error(error)), error
        return values, status, causes

    def callback(self, pair, first, second):
        values, status, _ = self.evaluate(pair, first, second)
        return values, status


def legacy_hp(h_mix, p_mix, AS1, AS2, R):
    """Serve the legacy raw-AS HP helper using the consolidated physical model.

    The caller owns the supplied handles and their synchronization/reference
    conventions. No handles or reference helpers are created. The old mass-ratio
    spelling is translated once; residual validation and final mixing are shared
    with ``HostEvaluator`` rather than maintaining a second set of equations.
    """
    if (
        isinstance(R, (bool, np.bool_))
        or not isinstance(R, (int, float, np.integer, np.floating))
        or not math.isfinite(R)
        or R < 0
    ):
        raise ValueError("Provide a finite nonnegative component mass ratio.")
    fractions = R / (1 + R), 1 / (1 + R)
    components = tuple(
        SimpleNamespace(_AS=state, _source_context=_NO_QUALITY_CONTEXT)
        for state in (AS1, AS2)
    )
    evaluator = HostEvaluator(components, fractions)
    values, status, causes = evaluator.evaluate(CP.HmassP_INPUTS, h_mix, p_mix)
    code = int(status)
    if code:
        cause = causes.get((), ValueError("Legacy mixture evaluation failed."))
        if code == StatusCode.NONCONVERGENCE:
            raise ConvergenceError(
                provider="native",
                model="binary_mixture",
                request="HmassP_INPUTS",
                reason=str(cause),
            ) from cause
        raise cause
    return {name: float(value) for name, value in values.items()}
