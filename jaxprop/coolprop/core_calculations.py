import numpy as np
import CoolProp.CoolProp as CP
import pysolver_view as psv
from scipy.optimize import root_scalar, root

from .. import math
from .. import utils

from ..helpers_props import PROPERTY_ALIASES, ALIAS_TO_CANONICAL
from ..fluid_api import SOURCE_PROPERTIES, DIAGNOSTIC_PROPERTIES
from collections.abc import Mapping
from ..errors import (
    ConvergenceError,
    InvalidConfigurationError,
    InvalidStateError,
    UnsupportedPropertyError,
)
from .source_context import helper_state

_OPTIONAL_PROPERTY_ERRORS = (
    ValueError,
    RuntimeError,
    NotImplementedError,
    AttributeError,
    OverflowError,
    ZeroDivisionError,
)


def _optional_value(function):
    """An undefined response/transport field does not invalidate core state."""
    try:
        value = float(function())
        return value if np.isfinite(value) else np.nan
    except _OPTIONAL_PROPERTY_ERRORS:
        return np.nan


def _optional_divide(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return _optional_value(lambda: numerator / denominator)


class _DiagnosticStateView:
    """Read fixed scalar inputs throughout one isolated diagnostic transaction."""

    def __init__(self, state, properties, phase, quality):
        self._state = state
        self._properties = properties
        self._phase = phase
        self._quality = quality

    def T(self):
        return self._properties["temperature"]

    def p(self):
        return self._properties["pressure"]

    def rhomass(self):
        return self._properties["density"]

    def hmass(self):
        return self._properties["enthalpy"]

    def smass(self):
        return self._properties["entropy"]

    def umass(self):
        return self._properties["internal_energy"]

    def phase(self):
        return self._phase

    def Q(self):
        return self._quality

    @property
    def _reference_source_state(self):
        """Use raw source thermodynamics solely to verify a helper's gauge.

        Homogeneous Helmholtz snapshots can lie inside the equilibrium dome;
        their h/s intentionally differ from the underlying equilibrium flash.
        They must still supply their own scalars to actual diagnostics.
        """
        return getattr(self._state, "_reference_source_state", self._state)

    def __getattr__(self, name):
        return getattr(self._state, name)


class _SaturationDiagnosticTransaction:
    """Share only requested endpoint samples within one diagnostic call.

    The owning evaluator holds its source lock. This short-lived cache stores
    immutable scalar samples, not a mutable source state shared across calls.
    Failure of a bubble, dew or pseudo-critical request is cached independently;
    unrelated optional diagnostics can still succeed.
    """

    def __init__(self, state, source_context):
        self.state = helper_state(state, source_context)
        self.context = source_context
        self.samples = {}
        self.critical_pressure = (
            state.p_critical()
            if source_context is None
            else source_context.critical_pressure
        )
        self.critical_temperature = (
            state.T_critical()
            if source_context is None
            else source_context.critical_temperature
        )
        self.critical_density = (
            state.rhomass_critical()
            if source_context is None
            else source_context.critical_density
        )
        self.triple_temperature = (
            state.Ttriple()
            if source_context is None
            else source_context.triple_temperature
        )

    def sample(self, pair, first, second):
        key = pair, float(first), float(second)
        if key not in self.samples:
            try:
                self.state.update(*key)
                self.samples[key] = {
                    "temperature": self.state.T(),
                    "pressure": self.state.p(),
                    "enthalpy": self.state.hmass(),
                    "cp": _optional_value(self.state.cpmass),
                }
            except _OPTIONAL_PROPERTY_ERRORS:
                self.samples[key] = None
        value = self.samples[key]
        if value is None:
            raise ValueError("This source saturation endpoint is unavailable.")
        return value

    def triple_pressure(self, quality):
        if self.context is not None:
            return (
                self.context.triple_pressure_liquid
                if quality == 0
                else self.context.triple_pressure_vapor
            )
        return self.sample(CP.QT_INPUTS, quality, self.triple_temperature)["pressure"]


def _add_optional_diagnostics(
    state, properties, source_context=None, *, phase=None, quality=None
):
    if phase is None:
        phase = state.phase()
    if quality is None:
        quality = _optional_value(state.Q)
    original = _DiagnosticStateView(state, properties, phase, quality)
    transaction = _SaturationDiagnosticTransaction(original, source_context)
    properties = calculate_supersaturation(
        original, properties, source_context=source_context, _transaction=transaction
    )
    properties["subcooling"] = _optional_value(
        lambda: calculate_subcooling(
            original, source_context=source_context, _transaction=transaction
        )
    )
    properties["superheating"] = _optional_value(
        lambda: calculate_superheating(
            original, source_context=source_context, _transaction=transaction
        )
    )
    return properties


# Define valid phase change types and their aliases
PHASE_CHANGE_ALIASES = {
    "condensation": "condensation",
    "evaporation": "evaporation",
    "flashing": "evaporation",
    "cavitation": "evaporation",
}


# Define NaN fallbacks for viscosity and thermal conductivity
def get_conductivity(AS):
    # AS.conductivity()
    try:
        return AS.conductivity()
    except _OPTIONAL_PROPERTY_ERRORS:
        return np.nan


def get_viscosity(AS):
    # AS.viscosity()
    try:
        return AS.viscosity()
    except _OPTIONAL_PROPERTY_ERRORS:
        return np.nan


# ------------------------------------------------------------------------------------ #
# Equilibrium property calculations in the single-phase region
# ------------------------------------------------------------------------------------ #


def compute_properties_1phase(
    abstract_state,
    generalize_quality=False,
    supersaturation=False,
    source_context=None,
):
    """Extract single-phase properties from CoolProp abstract state.
    Returns dict with canonical property names.
    """
    AS = abstract_state

    # --- basic thermodynamic properties
    T = AS.T()
    p = AS.p()
    rho = AS.rhomass()
    u = AS.umass()
    h = AS.hmass()
    s = AS.smass()
    cv = _optional_value(AS.cvmass)
    cp = _optional_value(AS.cpmass)
    gamma = _optional_divide(cp, cv)
    M = AS.molar_mass()
    Z = p / (rho * (AS.gas_constant() / M) * T)

    # --- thermodynamic properties involving derivatives
    a = _optional_value(AS.speed_sound)
    kappa_T = _optional_value(AS.isothermal_compressibility)
    alpha_p = _optional_value(AS.isobaric_expansion_coefficient)
    K_s = rho * a**2
    kappa_s = _optional_divide(1.0, K_s)
    K_T = _optional_divide(1.0, kappa_T)
    gruneisen = _optional_divide(alpha_p, kappa_T * rho * cv)
    mu_JT = _optional_divide(alpha_p * T - 1.0, rho * cp)
    mu_T = -mu_JT * cp

    # --- transport properties
    mu = get_viscosity(AS)
    k = get_conductivity(AS)

    # --- quality
    if generalize_quality:
        q_mass = _optional_value(
            lambda: calculate_generalized_quality(AS, source_context=source_context)
        )
        q_vol = np.nan
    else:
        try:
            q_mass, q_vol = calculate_trimmed_quality(AS, source_context=source_context)
        except (ValueError, RuntimeError, NotImplementedError, AttributeError):
            # Single-phase core properties remain usable when the source has
            # no evaluable critical/saturation reference for quality inference.
            q_mass, q_vol = np.nan, np.nan

    # --- canonical dict
    props = {
        "temperature": T,
        "pressure": p,
        "density": rho,
        "internal_energy": u,
        "enthalpy": h,
        "entropy": s,
        "isochoric_heat_capacity": cv,
        "isobaric_heat_capacity": cp,
        "heat_capacity_ratio": gamma,
        "compressibility_factor": Z,
        "speed_of_sound": a,
        "isothermal_compressibility": kappa_T,
        "isobaric_expansion_coefficient": alpha_p,
        "isentropic_bulk_modulus": K_s,
        "isentropic_compressibility": kappa_s,
        "isothermal_bulk_modulus": K_T,
        "gruneisen": gruneisen,
        "isothermal_joule_thomson": mu_T,
        "joule_thomson": mu_JT,
        # "gruneisen": compute_gruneisen(AS),
        # "isothermal_joule_thomson": compute_isothermal_joule_thomson(AS),
        # "joule_thomson": compute_joule_thomson(AS),
        "viscosity": mu,
        "conductivity": k,
        "quality_mass": q_mass,
        "is_two_phase": 0.0,
        "quality_volume": q_vol,
        "surface_tension": np.nan,
    }

    # Note: behavior of quality and void fraction in the single phase region should produce a behavior that is consistent with the two-phase values to ensure barotropy polynomials are well posed.

    # --- extra saturation departures
    if supersaturation:
        props = _add_optional_diagnostics(AS, props, source_context)

    return props


# ------------------------------------------------------------------------------------ #
# Equilibrium property calculations in the two-phase region
# ------------------------------------------------------------------------------------ #


def compute_properties_2phase(
    abstract_state, supersaturation=False, source_context=None
):
    """Return homogeneous-equilibrium responses of the actual source PQ map.

    Phase-change mass transfer is included in equilibrium response functions.
    In particular, cv is not a mass average of phase heat capacities. For a pure
    fluid in the dome, temperature is fixed by pressure: cp, expansion,
    isothermal compressibility and (dh/dp)_T are singular or not uniquely defined
    and remain unavailable. Exact endpoints expose one-phase responses instead.

    Pseudo-pure source models have temperature glide. Phase densities come from
    the actual flash, not a new QT flash at the bulk temperature. PQ Jacobians
    respect each backend's empirical interpolation geometry. Sound speed is
    sqrt((dp/drho)_s), equivalent to the usual pure-fluid HEM sound relation.
    Transport retains an empirical phase-volume average of the pressure's
    bubble/dew endpoint correlations. In particular, REFPROP does not expose
    actual quality-dependent phase transport through its saturation getters;
    these endpoint averages are a model approximation, not an exact mixture law.
    """
    AS = abstract_state
    T, p, rho = AS.T(), AS.p(), AS.rhomass()
    u, h, s, q = AS.umass(), AS.hmass(), AS.smass(), AS.Q()
    if q == 0.0 or q == 1.0:
        props = compute_properties_1phase(AS, source_context=source_context)
        props["is_two_phase"] = 1.0
        props["surface_tension"] = _optional_value(AS.surface_tension)
        if supersaturation:
            props = _add_optional_diagnostics(AS, props, source_context)
        return props

    # REFPROP pseudo-pure densities depend on the actual quality flash; even
    # pressure-endpoint densities can give an incorrect volume lever rule.
    rho_L = _optional_value(lambda: AS.saturated_liquid_keyed_output(CP.iDmass))
    rho_V = _optional_value(lambda: AS.saturated_vapor_keyed_output(CP.iDmass))
    vfrac_V = _optional_divide(q * rho, rho_V)
    vfrac_L = _optional_divide((1.0 - q) * rho, rho_L)

    scratch = helper_state(AS, source_context)
    scratch.update(CP.PQ_INPUTS, p, 0.0)
    k_L, mu_L = get_conductivity(scratch), get_viscosity(scratch)
    liquid = (scratch.rhomass(), scratch.hmass(), scratch.smass(), scratch.umass())
    scratch.update(CP.PQ_INPUTS, p, 1.0)
    k_V, mu_V = get_conductivity(scratch), get_viscosity(scratch)
    vapor = (scratch.rhomass(), scratch.hmass(), scratch.smass(), scratch.umass())
    pseudo = (
        source_context.is_pseudo_pure
        if source_context is not None
        else CP.get_fluid_param_string(AS.fluid_names()[0], "pure") == "false"
    )
    q_derivatives = (
        None
        if pseudo
        else np.asarray(
            (
                0.0,
                -(rho**2) * (1.0 / vapor[0] - 1.0 / liquid[0]),
                vapor[1] - liquid[1],
                vapor[2] - liquid[2],
                vapor[3] - liquid[3],
            )
        )
    )
    response = _two_phase_responses(
        scratch, p, q, rho, pseudo, q_derivatives=q_derivatives
    )
    props = {
        "temperature": T,
        "pressure": p,
        "density": rho,
        "internal_energy": u,
        "enthalpy": h,
        "entropy": s,
        "compressibility_factor": p / (rho * (AS.gas_constant() / AS.molar_mass()) * T),
        **response,
        "viscosity": vfrac_L * mu_L + vfrac_V * mu_V,
        "conductivity": vfrac_L * k_L + vfrac_V * k_V,
        "quality_mass": q,
        "quality_volume": vfrac_V,
        "surface_tension": _optional_value(AS.surface_tension),
        "is_two_phase": 1.0,
    }
    if supersaturation:
        props = _add_optional_diagnostics(AS, props, source_context)
    return props


def _two_phase_responses(
    scratch,
    pressure,
    quality,
    density,
    pseudo,
    *,
    relative_step=1e-5,
    q_derivatives=None,
):
    """Differentiate the PQ map without perturbing the primary source state.

    Central pressure/quality differences use at most four probes, sharing
    analytically known pure quality slopes from already sampled endpoints.
    For f(p,q),
    cv=(u_p*rho_q-u_q*rho_p)/(T_p*rho_q-T_q*rho_p). Corresponding Jacobian ratios
    give the sound speed, Gruneisen and Joule--Thomson coefficients. Steps depend
    on physical coordinates, never the arbitrary energy/entropy references.

    Pseudo-pure correlations can approximate thermodynamic identities
    differently: these fields describe derivatives of their actual flash map.
    """
    names = (
        "isochoric_heat_capacity",
        "isobaric_heat_capacity",
        "heat_capacity_ratio",
        "speed_of_sound",
        "isentropic_bulk_modulus",
        "isentropic_compressibility",
        "isothermal_bulk_modulus",
        "isothermal_compressibility",
        "isobaric_expansion_coefficient",
        "gruneisen",
        "isothermal_joule_thomson",
        "joule_thomson",
    )
    result = dict.fromkeys(names, np.nan)

    def sample(p, q):
        scratch.update(CP.PQ_INPUTS, p, q)
        return np.asarray(
            (
                scratch.T(),
                scratch.rhomass(),
                scratch.hmass(),
                scratch.smass(),
                scratch.umass(),
            )
        )

    try:
        dp = max(pressure * relative_step, 1e-3)
        dq = min(relative_step, quality / 4, (1.0 - quality) / 4)
        if not np.isfinite(dq) or dq <= 0:
            return result
        q_lo, q_hi = max(0.0, quality - dq), min(1.0, quality + dq)
        d_p = (sample(pressure + dp, quality) - sample(pressure - dp, quality)) / (
            2 * dp
        )
        d_q = (
            q_derivatives
            if q_derivatives is not None
            else (sample(pressure, q_hi) - sample(pressure, q_lo)) / (q_hi - q_lo)
        )
        tp, rp, hp, sp, up = d_p
        tq, rq, hq, sq, uq = d_q
        # Pure saturation temperature has no independent quality coordinate.
        if not pseudo:
            tq = 0.0
        determinant_u_rho = up * rq - uq * rp
        cv = _optional_divide(determinant_u_rho, tp * rq - tq * rp)
        cp = _optional_divide(hq, tq) if pseudo else np.nan
        a_squared = _optional_divide(1.0, rp - _optional_divide(rq * sp, sq))
        speed = np.sqrt(a_squared) if a_squared > 0 else np.nan
        kappa_t = (
            _optional_divide(rp - _optional_divide(rq * tp, tq), density)
            if pseudo
            else np.nan
        )
        result.update(
            {
                "isochoric_heat_capacity": cv,
                "isobaric_heat_capacity": cp,
                "heat_capacity_ratio": _optional_divide(cp, cv),
                "speed_of_sound": speed,
                "isentropic_bulk_modulus": density * a_squared,
                "isentropic_compressibility": _optional_divide(
                    1.0, density * a_squared
                ),
                "isothermal_compressibility": kappa_t,
                "isothermal_bulk_modulus": _optional_divide(1.0, kappa_t),
                "isobaric_expansion_coefficient": (
                    _optional_divide(-rq, density * tq) if pseudo else np.nan
                ),
                "gruneisen": _optional_divide(rq, density * determinant_u_rho),
                "joule_thomson": tp - _optional_divide(tq * hp, hq),
                "isothermal_joule_thomson": (
                    hp - _optional_divide(hq * tp, tq) if pseudo else np.nan
                ),
            }
        )
    except _OPTIONAL_PROPERTY_ERRORS:
        pass
    return result


def _source_restore_recipe(state):
    """Capture a reference-safe flash recipe without changing the source."""
    quality = state.Q()
    if state.phase() == CP.iphase_twophase and 0 <= quality <= 1:
        return CP.PQ_INPUTS, state.p(), quality
    # A one-phase PT flash also preserves REFPROP's phase/quality sentinel;
    # a density/temperature restore can return a different nonphysical Q label.
    return CP.PT_INPUTS, state.p(), state.T()


def _derivative_scratch(state, source_context):
    """Use prepared scratch, or restore a standalone object's original state."""
    if source_context is not None and source_context.auxiliary_state is not state:
        return source_context.auxiliary_state, None
    return state, _source_restore_recipe(state)


def _is_pseudo_source(state, source_context):
    if source_context is not None:
        return source_context.is_pseudo_pure
    return CP.get_fluid_param_string(state.fluid_names()[0], "pure") == "false"


def compute_dsdp_q(AS, pressure, quality, rel_dp=1e-4):
    """Return a saturation entropy derivative without retaining perturbations.

    Analytic endpoint derivatives are preferred. The finite-difference fallback
    is central in pressure, with its physical step independent of entropy zero.
    Even failed source flashes restore a standalone caller's original state.
    """
    recipe = _source_restore_recipe(AS)
    try:
        if not (
            AS.phase() == CP.iphase_twophase
            and AS.p() == pressure
            and AS.Q() == quality
        ):
            AS.update(CP.PQ_INPUTS, pressure, quality)
        try:
            return AS.first_saturation_deriv(CP.iSmass, CP.iP)
        except _OPTIONAL_PROPERTY_ERRORS:
            dp = rel_dp * pressure
            if not np.isfinite(dp) or dp <= 0:
                return np.nan
            AS.update(CP.PQ_INPUTS, pressure + dp, quality)
            above = AS.smass()
            AS.update(CP.PQ_INPUTS, pressure - dp, quality)
            below = AS.smass()
            return (above - below) / (2 * dp)
    except _OPTIONAL_PROPERTY_ERRORS:
        return np.nan
    finally:
        AS.update(*recipe)


def _equilibrium_derivative(AS, source_context, name, relative_step):
    """Share actual PQ derivatives for an interior two-phase helper call."""
    scratch, recipe = _derivative_scratch(AS, source_context)
    try:
        values = _two_phase_responses(
            scratch,
            AS.p(),
            AS.Q(),
            AS.rhomass(),
            _is_pseudo_source(AS, source_context),
            relative_step=relative_step,
        )
        return values[name]
    finally:
        if recipe is not None:
            scratch.update(*recipe)


def compute_gruneisen(AS, rel_du=1e-5, source_context=None):
    """Return Gamma=(1/rho)*(dp/du)_rho while preserving source state.

    Interior two-phase derivatives use the natural PQ map, avoiding unsupported
    pseudo-pure density/energy flashes. Homogeneous analytic EOS derivatives
    are preferred; a central energy fallback uses cv*T and p/rho as physical,
    reference-independent scales. The primary configured state is untouched.
    """
    if AS.phase() == CP.iphase_twophase and 0 < AS.Q() < 1:
        return _equilibrium_derivative(AS, source_context, "gruneisen", rel_du)
    try:
        return AS.first_partial_deriv(CP.iP, CP.iUmass, CP.iDmass) / AS.rhomass()
    except _OPTIONAL_PROPERTY_ERRORS:
        pass
    scratch, recipe = _derivative_scratch(AS, source_context)
    try:
        rho, u = AS.rhomass(), AS.umass()
        scale = max(abs(AS.cvmass() * AS.T()), abs(AS.p() / rho), 1.0)
        du = rel_du * scale
        if not np.isfinite(du) or du <= 0:
            return np.nan
        scratch.update(CP.DmassUmass_INPUTS, rho, u + du)
        above = scratch.p()
        scratch.update(CP.DmassUmass_INPUTS, rho, u - du)
        below = scratch.p()
        return (above - below) / (2 * du * rho)
    except _OPTIONAL_PROPERTY_ERRORS:
        return np.nan
    finally:
        if recipe is not None:
            scratch.update(*recipe)


def compute_joule_thomson(AS, rel_dp=1e-5, source_context=None):
    """Return mu_JT=(dT/dp)_h with isolated analytic/central source evaluation."""
    if AS.phase() == CP.iphase_twophase and 0 < AS.Q() < 1:
        return _equilibrium_derivative(AS, source_context, "joule_thomson", rel_dp)
    try:
        return AS.first_partial_deriv(CP.iT, CP.iP, CP.iHmass)
    except _OPTIONAL_PROPERTY_ERRORS:
        pass
    scratch, recipe = _derivative_scratch(AS, source_context)
    try:
        p, h = AS.p(), AS.hmass()
        dp = rel_dp * p
        if not np.isfinite(dp) or dp <= 0:
            return np.nan
        scratch.update(CP.HmassP_INPUTS, h, p + dp)
        above = scratch.T()
        scratch.update(CP.HmassP_INPUTS, h, p - dp)
        below = scratch.T()
        return (above - below) / (2 * dp)
    except _OPTIONAL_PROPERTY_ERRORS:
        return np.nan
    finally:
        if recipe is not None:
            scratch.update(*recipe)


def compute_isothermal_joule_thomson(AS, rel_dp=1e-5, source_context=None):
    """Return mu_T=(dh/dp)_T, unavailable on a pure saturation manifold.

    Pure interior states have dependent pressure and temperature, so this
    derivative is not defined by an equilibrium two-phase EOS. A PT perturbation
    would jump to a one-phase branch and produce a misleading finite number.
    Pseudo-pure glide is differentiated through its actual PQ map instead.
    """
    if AS.phase() == CP.iphase_twophase and 0 < AS.Q() < 1:
        if not _is_pseudo_source(AS, source_context):
            return np.nan
        return _equilibrium_derivative(
            AS, source_context, "isothermal_joule_thomson", rel_dp
        )
    try:
        return AS.first_partial_deriv(CP.iHmass, CP.iP, CP.iT)
    except _OPTIONAL_PROPERTY_ERRORS:
        pass
    scratch, recipe = _derivative_scratch(AS, source_context)
    try:
        p, temperature = AS.p(), AS.T()
        dp = rel_dp * p
        if not np.isfinite(dp) or dp <= 0:
            return np.nan
        scratch.update(CP.PT_INPUTS, p + dp, temperature)
        above = scratch.hmass()
        scratch.update(CP.PT_INPUTS, p - dp, temperature)
        below = scratch.hmass()
        return (above - below) / (2 * dp)
    except _OPTIONAL_PROPERTY_ERRORS:
        return np.nan
    finally:
        if recipe is not None:
            scratch.update(*recipe)


# ------------------------------------------------------------------------------------ #
# Property calculations using Helmholtz equations of state
# ------------------------------------------------------------------------------------ #
def _to_scalar(x):
    return np.asarray(x).squeeze()  # remove all length-1 dimensions


def compute_properties_metastable_rhoT(
    abstract_state,
    rho,
    T,
    supersaturation=False,
    generalize_quality=False,
    source_context=None,
):
    r"""
    Compute the thermodynamic properties of a fluid using temperature-density calls to the Helmholtz energy equation of state (HEOS).

    Parameters
    ----------
    abstract_state : CoolProp.AbstractState
        The abstract state of the fluid for which the properties are to be calculated.
    rho : float
        Density of the fluid (kg/m³).
    T : float
        Temperature of the fluid (K).
    supersaturation : bool, optional
        Whether to evaluate supersaturation properties. Default is False.
    generalize_quality : bool, optional
        Request the documented generalized entropy coordinate instead of the
        trimmed one-phase label. Neither is an equilibrium vapor fraction for
        a homogeneous metastable point.
    source_context : SourceEvaluationContext, optional
        Construction-prepared helper state and reference data matching the
        supplied source. Configured Fluid calls provide this context so no
        runtime source construction or later global reference convention is used.

    Returns
    -------
    dict
        Thermodynamic properties computed at the given density and temperature.

    Notes
    -----
    The Helmholtz energy equation of state (HEOS) expresses the Helmholtz energy as an explicit function
    of temperature and density:

    .. math::
        \Phi = \Phi(\rho, T)

    In dimensionless form, the Helmholtz energy is given by:

    .. math::
        \phi(\delta, \tau) = \frac{\Phi(\delta, \tau)}{R T}

    where:

    - :math:`\phi` is the dimensionless Helmholtz energy
    - :math:`R` is the gas constant of the fluid
    - :math:`\delta = \rho / \rho_r` is the reduced density
    - :math:`\tau = T_r / T` is the inverse of the reduced temperature

    The selected EOS supplies its gas constant and reducing parameters
    :math:`\rho_r` and :math:`T_r`; these need not equal the critical values.

    Thermodynamic properties can be derived from the Helmholtz energy and its partial derivatives.
    The following table summarizes the expressions for various properties:

    .. list-table:: Helmholtz energy thermodynamic relations
        :widths: 30 70
        :header-rows: 1

        * - Property name
          - Mathematical relation
        * - Pressure
          - .. math:: Z = \frac{p}{\rho R T} = \delta \cdot \phi_{\delta}
        * - Entropy
          - .. math:: \frac{s}{R} = \tau \cdot \phi_{\tau} - \phi
        * - Internal energy
          - .. math:: \frac{u}{R T} = \tau \cdot \phi_{\tau}
        * - Enthalpy
          - .. math:: \frac{h}{R T} = \tau \cdot \phi_{\tau} + \delta \cdot \phi_{\delta}
        * - Isochoric heat capacity
          - .. math:: \frac{c_v}{R} = -\tau^2 \cdot \phi_{\tau \tau}
        * - Isobaric heat capacity
          - .. math:: \frac{c_p}{R} = -\tau^2 \cdot \phi_{\tau \tau} + \frac{(\delta \cdot \phi_{\delta} - \tau \cdot \delta \cdot \phi_{\tau \delta})^2}{2 \cdot \delta \cdot \phi_{\delta} + \delta^2 \cdot \phi_{\delta \delta}}
        * - Isobaric expansivity
          - .. math:: \alpha \cdot T = \frac{\delta \cdot \phi_{\delta} - \tau \cdot \delta \cdot \phi_{\tau \delta}}{2 \cdot \delta \cdot \phi_{\delta} + \delta^2 \cdot \phi_{\delta \delta}}
        * - Isothermal compressibility
          - .. math:: \rho R T \beta = \left(2 \cdot \delta \cdot \phi_{\delta} + \delta^2 \cdot \phi_{\delta \delta} \right)^{-1}
        * - Isothermal bulk modulus
          - .. math:: \frac{K_T}{\rho R T} = 2 \cdot \delta \cdot \phi_{\delta} + \delta^2 \cdot \phi_{\delta \delta}
        * - Isentropic bulk modulus
          - .. math:: \frac{K_s}{\rho R T} = 2 \cdot \delta \cdot \phi_{\delta} + \delta^2 \ \cdot \phi_{\delta \delta} - \frac{(\delta \cdot \phi_{\delta} - \tau \cdot \delta \cdot \phi_{\tau \delta})^2}{\tau^2 \cdot \phi_{\tau \tau}}
        * - Joule-Thompson coefficient
          - .. math:: \rho R \mu_{\mathrm{JT}} = - \frac{\delta \cdot \phi_{\delta} + \tau \cdot \delta \cdot \phi_{\tau \delta} + \delta^2 \cdot \phi_{\delta \delta}}{(\delta \cdot \phi_{\delta} - \tau \cdot \delta \cdot \phi_{\tau \delta})^2 - \tau^2 \cdot \phi_{\tau \tau} \cdot (2 \cdot \delta \cdot \phi_{\delta} + \delta^2 \cdot \phi_{\delta \delta})}

    Where the following abbreviations are used:

    - :math:`\phi_{\delta} = \left(\frac{\partial \phi}{\partial \delta}\right)_{\tau}`
    - :math:`\phi_{\tau} = \left(\frac{\partial \phi}{\partial \tau}\right)_{\delta}`
    - :math:`\phi_{\delta \delta} = \left(\frac{\partial^2 \phi}{\partial \delta^2}\right)_{\tau, \tau}`
    - :math:`\phi_{\tau \tau} = \left(\frac{\partial^2 \phi}{\partial \tau^2}\right)_{\delta, \delta}`
    - :math:`\phi_{\tau \delta} = \left(\frac{\partial^2 \phi}{\partial \tau \delta}\right)_{\delta, \tau}`

    This function can be used to estimate metastable properties using the equation of state beyond the saturation lines.
    """

    # Homogeneous evaluation also serves spinodal searches, where pressure or
    # bulk modulus can be negative. Only its independent coordinates must be
    # finite and positive; do not silently project an unstable branch.
    if (
        np.ndim(rho) != 0
        or np.ndim(T) != 0
        or isinstance(rho, (bool, np.bool_))
        or isinstance(T, (bool, np.bool_))
    ):
        raise InvalidStateError(
            request="DmassT_INPUTS",
            reason="Homogeneous density/temperature inputs must be real scalars.",
        )
    rho, T = float(rho), float(T)
    if not np.isfinite(rho) or not np.isfinite(T) or rho <= 0 or T <= 0:
        raise InvalidStateError(
            request="DmassT_INPUTS",
            reason="Homogeneous density and temperature must be finite and positive.",
        )
    # Update CoolProp state
    AS = abstract_state
    # AS.update(CP.DmassT_INPUTS, _to_scalar(rho), _to_scalar(T))
    AS.update(CP.DmassT_INPUTS, rho, T)

    # Fluid constants
    R = AS.gas_constant()
    M = AS.molar_mass()
    # Helmholtz derivatives use the EOS reducing coordinates. They need not
    # coincide with numerically determined critical coordinates (e.g. R134a).
    tau = AS.tau()
    delta = AS.delta()

    # Helmholtz derivatives
    alpha = AS.alpha0() + AS.alphar()
    dalpha_dTau = AS.dalpha0_dTau() + AS.dalphar_dTau()
    dalpha_dDelta = AS.dalpha0_dDelta() + AS.dalphar_dDelta()
    d2alpha_dTau2 = AS.d2alpha0_dTau2() + AS.d2alphar_dTau2()
    d2alpha_dDelta2 = AS.d2alpha0_dDelta2() + AS.d2alphar_dDelta2()
    d2alpha_dDelta_dTau = AS.d2alpha0_dDelta_dTau() + AS.d2alphar_dDelta_dTau()

    # Thermodynamic properties
    u = (R / M) * T * (tau * dalpha_dTau)
    h = (R / M) * T * (tau * dalpha_dTau + delta * dalpha_dDelta)
    s = (R / M) * (tau * dalpha_dTau - alpha)
    cv = (R / M) * (-(tau**2) * d2alpha_dTau2)
    density_response = 2 * delta * dalpha_dDelta + delta**2 * d2alpha_dDelta2
    thermal_response = delta * dalpha_dDelta - delta * tau * d2alpha_dDelta_dTau
    cp = (R / M) * (
        -(tau**2) * d2alpha_dTau2
        + _optional_divide(thermal_response**2, density_response)
    )
    gamma = _optional_divide(cp, cv)

    p = (R / M) * T * rho * delta * dalpha_dDelta
    Z = delta * dalpha_dDelta

    # Speed of sound
    a_square = (R / M * T) * (
        density_response - _optional_divide(thermal_response**2, tau**2 * d2alpha_dTau2)
    )
    a = np.sqrt(a_square) if a_square > 0 else np.nan

    # Bulk/compressibility
    K_s = rho * a_square
    kappa_s = _optional_divide(1.0, K_s)
    K_T = (R / M) * T * rho * density_response
    kappa_T = _optional_divide(1.0, K_T)
    alpha_p = _optional_divide(thermal_response, T * density_response)

    # Transport
    mu = get_viscosity(AS)
    k = get_conductivity(AS)

    # Grüneisen
    # Cancel the isothermal denominator analytically. At a spinodal alpha and
    # cp diverge, but Gamma and mu_JT can retain finite well-defined limits.
    gruneisen = _optional_divide((R / M) * thermal_response, cv)

    # Joule-Thomson
    mu_JT = _optional_divide(
        thermal_response - density_response,
        rho
        * (R / M)
        * (thermal_response**2 - tau**2 * d2alpha_dTau2 * density_response),
    )
    mu_T = _optional_divide(density_response - thermal_response, rho * density_response)

    # Canonical dict
    props = {
        "temperature": T,
        "pressure": p,
        "density": rho,
        "internal_energy": u,
        "enthalpy": h,
        "entropy": s,
        "isochoric_heat_capacity": cv,
        "isobaric_heat_capacity": cp,
        "heat_capacity_ratio": gamma,
        "compressibility_factor": Z,
        "speed_of_sound": a,
        "isentropic_bulk_modulus": K_s,
        "isentropic_compressibility": kappa_s,
        "isothermal_bulk_modulus": K_T,
        "isothermal_compressibility": kappa_T,
        "isobaric_expansion_coefficient": alpha_p,
        "gruneisen": gruneisen,
        "isothermal_joule_thomson": mu_T,
        "joule_thomson": mu_JT,
        "viscosity": mu,
        "conductivity": k,
        "surface_tension": np.nan,
        "is_two_phase": 0.0,  # always for Helmholtz EoS
        "quality_mass": np.nan,
        "quality_volume": np.nan,
    }

    # Inside the envelope CoolProp's ordinary DmassT state can report an
    # equilibrium pressure/entropy although its Helmholtz derivatives still
    # describe the requested homogeneous point. Diagnostics must use the
    # homogeneous scalars above and never label it a two-phase mixture.
    diagnostic_state = _DiagnosticStateView(AS, props, CP.iphase_unknown, np.nan)
    if supersaturation:
        props = _add_optional_diagnostics(diagnostic_state, props, source_context)

    # Quality handling
    if generalize_quality:
        q_mass = calculate_generalized_quality(
            diagnostic_state, source_context=source_context
        )
        props["quality_mass"] = q_mass
        props["quality_volume"] = np.nan
        props["is_two_phase"] = False
    else:
        frac_mass, frac_volume = calculate_trimmed_quality(
            diagnostic_state, source_context=source_context
        )
        props["quality_mass"] = frac_mass
        props["quality_volume"] = frac_volume
        props["is_two_phase"] = False

    return props


# ------------------------------------------------------------------------------------ #
# Property calculations using CoolProp solver
# ------------------------------------------------------------------------------------ #


def compute_properties_coolprop(
    abstract_state,
    input_type,
    prop_1,
    prop_2,
    generalize_quality=False,
    supersaturation=False,
    source_context=None,
):
    r"""
    Set the thermodynamic state of the fluid based on input properties.

    This method updates the thermodynamic state of the fluid in the CoolProp ``abstractstate`` object
    using the given input properties. It then calculates either single-phase or two-phase
    properties based on the current phase of the fluid.

    If the calculation of properties fails, `converged_flag` is set to False, indicating an issue with
    the property calculation. Otherwise, it's set to True.

    Aliases of the properties are also added to the ``Fluid.properties`` dictionary for convenience.

    Parameters
    ----------
    input_type : int
        The variable pair used to define the thermodynamic state. This should be one of the
        predefined input pairs in CoolProp, such as ``PT_INPUTS`` for pressure and temperature.

    prop_1 : float
        The first property value corresponding to the input type.
    prop_2 : float
        The second property value corresponding to the input type.


    Returns
    -------
    dict
        A dictionary object containing the fluid properties

    """

    # Update Coolprop thermodynamic state
    abstract_state.update(input_type, prop_1, prop_2)

    # Retrieve single-phase properties
    is_two_phase = abstract_state.phase() == CP.iphase_twophase
    if not is_two_phase:
        properties = compute_properties_1phase(
            abstract_state,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
            source_context=source_context,
        )
    else:
        properties = compute_properties_2phase(
            abstract_state,
            supersaturation=supersaturation,
            source_context=source_context,
        )

    return properties


# ------------------------------------------------------------------------------------ #
# Property calculations using custom solver
# ------------------------------------------------------------------------------------ #


def compute_properties(
    abstract_state,
    prop_1,
    prop_1_value,
    prop_2,
    prop_2_value,
    calculation_type,
    rhoT_guess_metastable=None,
    rhoT_guess_equilibrium=None,
    phase_change=None,
    blending_variable=None,
    blending_onset=None,
    blending_width=None,
    supersaturation=False,
    generalize_quality=False,
    solver_algorithm="hybr",
    solver_tolerance=1e-6,
    solver_max_iterations=100,
    print_convergence=False,
    source_context=None,
):
    r"""
    .. _compute_properties:

    Determine the thermodynamic state for the given fluid property pair by iterating on the 
    density-temperature native inputs,

    This function uses a non-linear root finder to determine the solution of the nonlinear system given by:

    .. math::

        R_1(\rho,\, T) = y_1 - y_1(\rho,\, T) = 0 \\
        R_2(\rho,\, T) = y_2 - y_2(\rho,\, T) = 0

    where :math:`(y_1,\, y_2)` is the given thermodynamic property pair (e.g., enthalpy and pressure),
    while density and temperature :math:`(\rho,\, T)` are the independent variables that the solver 
    iterates until the residual of the problem is driven to zero.
     
      
    
    Equilibrium residuals use CoolProp's phase-equilibrium state. Metastable
    residuals evaluate the homogeneous Helmholtz potential at the supplied
    density and temperature, including continuation beyond saturation. The
    blending route solves both branches for the same two target properties
    before interpolating their numerical properties. That interpolation is a
    constitutive transition model, not a single differentiated EOS potential;
    independent thermodynamic identities need not hold inside its transition.

    Density/temperature pairs are evaluated directly without a root or initial
    guess. Other pairs require finite positive density/temperature guesses.
    Energy and entropy residuals use the reference-invariant scales
    :math:`R_sT_c` and :math:`R_s`, where :math:`R_s` belongs to the selected EOS.
    Zero quality/energy targets therefore remain well-defined. Solver success
    is checked against the actual final thermodynamic residual independently
    of its termination flag.


    Parameters
    ----------
    prop_1 : str
        The the type of the first thermodynamic property.
    prop_1_value : float
        The the numerical value of the first thermodynamic property.
    prop_2 : str
        The the type of the second thermodynamic property.
    prop_2_value : float
        The the numerical value of the second thermodynamic property.
    rho_guess : float
        Initial guess for density
    T_guess : float
        Initial guess for temperature
    calculation_type : str
        The type of calculation to perform. Valid options are 'equilibrium', 'metastable', and 'blending'.
    supersaturation : bool, optional
        If True, calculates supersaturation variables. Defaults to False.
    generalize_quality : bool, optional
        If True, generalizes quality outside two-phase region. Defaults to False.
    blending_variable : str, optional
        The variable used for blending properties. Required if `calculation_type` is 'blending'.
    blending_onset : float, optional
        The onset value for blending. Required if `calculation_type` is 'blending'.
    blending_width : float, optional
        The width value for blending. Required if `calculation_type` is 'blending'.
    solver_algorithm : str, optional
        Method to be used for solving the nonlinear system. Defaults to 'hybr'.
    solver_tolerance : float, optional
        Tolerance for the solver termination. Defaults to 1e-6.
    solver_max_iterations : integer, optional
        Maximum number of iterations of the solver. Defaults to 100.
    print_convergence : bool, optional
        If True, displays the convergence progress. Defaults to False.
    source_context : SourceEvaluationContext, optional
        Matching configured helper/reference context, reused throughout the
        complete solve. The owning Fluid holds its reentrant transaction lock.

    Returns
    -------
    dict       
        Thermodynamic properties corresponding to the given input pair.
    """

    # Construction/settings errors are independent of the numerical failure
    # policy. Validate them before a solver can mutate the source state.
    for name in (prop_1, prop_2):
        if not isinstance(name, str) or name not in ALIAS_TO_CANONICAL:
            raise UnsupportedPropertyError(
                request=str(name),
                reason="Unknown thermodynamic property name or alias.",
            )
    for option, value in (
        ("supersaturation", supersaturation),
        ("generalize_quality", generalize_quality),
        ("print_convergence", print_convergence),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise InvalidConfigurationError(
                request=option, reason="Expected a static boolean."
            )
    available = SOURCE_PROPERTIES | (
        DIAGNOSTIC_PROPERTIES if supersaturation else frozenset()
    )
    for name in (prop_1, prop_2):
        canonical = ALIAS_TO_CANONICAL[name]
        if canonical not in available:
            raise UnsupportedPropertyError(
                request=canonical,
                reason="This source calculation does not enable the requested input property.",
                supported=tuple(sorted(available)),
            )
    if ALIAS_TO_CANONICAL[prop_1] == ALIAS_TO_CANONICAL[prop_2]:
        raise InvalidConfigurationError(
            request="input_properties", reason="Two distinct properties are required."
        )
    if not isinstance(solver_algorithm, str) or solver_algorithm not in ("hybr", "lm"):
        raise InvalidConfigurationError(
            request="solver_algorithm",
            reason="Unsupported thermodynamic root algorithm.",
            supported=("hybr", "lm"),
        )
    tolerance_array = np.asarray(solver_tolerance)
    if (
        tolerance_array.shape != ()
        or tolerance_array.dtype.kind not in "fiu"
        or not np.isfinite(tolerance_array)
        or tolerance_array <= 0
    ):
        raise InvalidConfigurationError(
            request="solver_tolerance", reason="Expected a finite positive tolerance."
        )
    if (
        isinstance(solver_max_iterations, (bool, np.bool_))
        or not isinstance(solver_max_iterations, (int, np.integer))
        or solver_max_iterations < 1
    ):
        raise InvalidConfigurationError(
            request="solver_max_iterations", reason="Expected a positive integer."
        )

    if calculation_type == "blending":
        _validated_blending_settings(
            phase_change, blending_variable, blending_onset, blending_width
        )
    for name, value in ((prop_1, prop_1_value), (prop_2, prop_2_value)):
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in "fiu" or not np.isfinite(array):
            raise InvalidStateError(
                request=ALIAS_TO_CANONICAL[name],
                reason="Expected a finite real scalar thermodynamic input.",
            )
        if ALIAS_TO_CANONICAL[name] in {"density", "temperature"} and array <= 0:
            raise InvalidStateError(
                request=ALIAS_TO_CANONICAL[name],
                reason="Density and temperature inputs must be positive.",
            )

    coordinates = {
        ALIAS_TO_CANONICAL[prop_1]: prop_1_value,
        ALIAS_TO_CANONICAL[prop_2]: prop_2_value,
    }
    if coordinates.keys() == {"density", "temperature"}:
        if calculation_type == "metastable":
            return compute_properties_metastable_rhoT(
                abstract_state,
                coordinates["density"],
                coordinates["temperature"],
                supersaturation=supersaturation,
                generalize_quality=generalize_quality,
                source_context=source_context,
            )
        if calculation_type == "equilibrium":
            return compute_properties_coolprop(
                abstract_state,
                CP.DmassT_INPUTS,
                coordinates["density"],
                coordinates["temperature"],
                supersaturation=supersaturation,
                generalize_quality=generalize_quality,
                source_context=source_context,
            )

    # Perform calculations according to specified type
    if calculation_type == "equilibrium":
        rho_guess, T_guess = _validated_flash_guess(
            rhoT_guess_equilibrium, "rhoT_guess_equilibrium"
        )
        return _perform_flash_calculation(
            abstract_state=abstract_state,
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            rho_guess=rho_guess,
            T_guess=T_guess,
            calculation_type="equilibrium",
            supersaturation=supersaturation,
            generalize_quality=generalize_quality,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
            source_context=source_context,
        )

    elif calculation_type == "metastable":
        rho_guess, T_guess = _validated_flash_guess(
            rhoT_guess_metastable, "rhoT_guess_metastable"
        )
        return _perform_flash_calculation(
            abstract_state=abstract_state,
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            rho_guess=rho_guess,
            T_guess=T_guess,
            calculation_type="metastable",
            supersaturation=supersaturation,
            generalize_quality=generalize_quality,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
            source_context=source_context,
        )

    elif calculation_type == "blending":

        if (
            phase_change is None
            or blending_variable is None
            or blending_onset is None
            or blending_width is None
        ):
            msg = (
                f"The following variables must be specified when calculation_type='{calculation_type}':\n"
                f"   1. phase_change: {phase_change}\n"
                f"   2. blending_variable: {blending_variable}\n"
                f"   3. blending_onset: {blending_onset}\n"
                f"   4. blending_width: {blending_width}\n"
            )
            raise InvalidConfigurationError(request="blending", reason=msg)

        # Normalize/validate the settings before either potentially costly flash.
        phase_change, blending_variable = _validated_blending_settings(
            phase_change, blending_variable, blending_onset, blending_width
        )

        # Equilibrium state
        rho_guess, T_guess = _validated_flash_guess(
            rhoT_guess_equilibrium, "rhoT_guess_equilibrium"
        )
        meta_guess = _validated_flash_guess(
            rhoT_guess_metastable, "rhoT_guess_metastable"
        )
        props_eq = _perform_flash_calculation(
            abstract_state=abstract_state,
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            rho_guess=rho_guess,
            T_guess=T_guess,
            calculation_type="equilibrium",
            supersaturation=supersaturation,
            generalize_quality=generalize_quality,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
            source_context=source_context,
        )

        # Metastable properties
        rho_guess, T_guess = meta_guess
        props_meta = _perform_flash_calculation(
            abstract_state=abstract_state,
            prop_1=prop_1,
            prop_1_value=prop_1_value,
            prop_2=prop_2,
            prop_2_value=prop_2_value,
            rho_guess=rho_guess,
            T_guess=T_guess,
            calculation_type="metastable",
            supersaturation=supersaturation,
            generalize_quality=False,
            solver_algorithm=solver_algorithm,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            print_convergence=print_convergence,
            source_context=source_context,
        )
        props_meta["quality_mass"] = 1.00 if phase_change == "condensation" else 0.00

        # Blend properties
        props_blended = blend_properties(
            phase_change=phase_change,
            props_equilibrium=props_eq,
            props_metastable=props_meta,
            blending_variable=blending_variable,
            blending_onset=blending_onset,
            blending_width=blending_width,
        )

        return props_blended, props_eq, props_meta

    raise InvalidConfigurationError(
        request="calculation_type",
        reason=f"Unknown calculation type: {calculation_type}",
        supported=("equilibrium", "metastable", "blending"),
    )


# Helper functions
def _validated_flash_guess(value, name):
    """Validate one legacy scalar-solver guess before unpacking or source work."""
    try:
        array = np.asarray(value)
        valid = (
            array.shape == (2,)
            and array.dtype.kind in "fiu"
            and np.all(np.isfinite(array))
            and np.all(array > 0)
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise InvalidConfigurationError(
            request=name,
            reason="Expected two finite positive scalar guesses: density and temperature.",
        )
    return tuple(float(item) for item in array)


def _perform_flash_calculation(
    abstract_state,
    prop_1,
    prop_1_value,
    prop_2,
    prop_2_value,
    rho_guess,
    T_guess,
    calculation_type,
    supersaturation,
    generalize_quality,
    solver_algorithm,
    solver_tolerance,
    solver_max_iterations,
    print_convergence,
    source_context=None,
):

    # Normalize property names to canonical ---
    try:
        prop_1 = ALIAS_TO_CANONICAL[prop_1]
    except KeyError:
        raise UnsupportedPropertyError(
            request=prop_1, reason="Unknown property name or alias."
        )

    try:
        prop_2 = ALIAS_TO_CANONICAL[prop_2]
    except KeyError:
        raise UnsupportedPropertyError(
            request=prop_2, reason="Unknown property name or alias."
        )

    # Ensure prop_1_value and prop_2_value are scalar numbers
    if (
        not utils.is_float(prop_1_value)
        or not utils.is_float(prop_2_value)
        or not np.isfinite(prop_1_value)
        or not np.isfinite(prop_2_value)
    ):
        msg = f"Both prop_1_value and prop_2_value must be scalar numbers. Received: prop_1_value={prop_1_value}, prop_2_value={prop_2_value}"
        raise InvalidStateError(request="input_properties", reason=msg)

    # Validate initial guesses for rho and T
    if not utils.is_float(rho_guess) or not utils.is_float(T_guess):
        msg = f"A valid initial guess must be provided for density and temperature. Received: rho_guess={rho_guess}, T_guess={T_guess}."
        raise ValueError(msg)

    # Define problem (find root of temperature-density residual)
    if calculation_type == "equilibrium":
        function_handle = lambda rho, T: compute_properties_coolprop(
            abstract_state=abstract_state,
            input_type=CP.DmassT_INPUTS,
            prop_1=rho,
            prop_2=T,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
            source_context=source_context,
        )
    elif calculation_type == "metastable":
        # print("Are we here in the metastable section of _perform_flash_calculation?")
        function_handle = lambda rho, T: compute_properties_metastable_rhoT(
            abstract_state=abstract_state,
            rho=rho,
            T=T,
            generalize_quality=generalize_quality,
            supersaturation=supersaturation,
            source_context=source_context,
        )
    else:
        msg = f"Invalid calculation type '{calculation_type}'. Valid options are: 'equilibrium' and 'metastable'"
        raise InvalidConfigurationError(request="calculation_type", reason=msg)

    # Define the problem (find root of temperature-density residual)
    rho_crit = abstract_state.rhomass_critical()
    T_crit = abstract_state.T_critical()
    specific_R = abstract_state.gas_constant() / abstract_state.molar_mass()
    invariant_scales = {
        "enthalpy": specific_R * T_crit,
        "internal_energy": specific_R * T_crit,
        "entropy": specific_R,
        "quality_mass": 1.0,
        "quality_volume": 1.0,
    }
    # print(f"rho_crit: {rho_crit}, T_crit: {T_crit}")

    problem = _FlashCalculationResidual(
        prop_1=prop_1,
        prop_1_value=prop_1_value,
        prop_2=prop_2,
        prop_2_value=prop_2_value,
        function_handle=function_handle,
        rho_scale=rho_crit,
        T_scale=T_crit,
        residual_scales=tuple(
            invariant_scales.get(name, max(abs(float(value)), 1.0))
            for name, value in ((prop_1, prop_1_value), (prop_2, prop_2_value))
        ),
    )

    # Define root-finding solver object
    solver = psv.NonlinearSystemSolver(
        problem,
        method=solver_algorithm,
        tolerance=solver_tolerance,
        max_iterations=solver_max_iterations,
        print_convergence=print_convergence,
        update_on="function",
        # lm_factor=0.05
    )
    # print(f"rho_crit: {rho_crit}, T_crit: {T_crit}")

    # Define initial guess and solve the problem
    # x0 = np.asarray([rho_guess, T_guess])

    x0_reduced = np.asarray([rho_guess / rho_crit, T_guess / T_crit])
    try:
        xf_reduced = solver.solve(x0_reduced)
    except (InvalidConfigurationError, UnsupportedPropertyError):
        raise
    except (
        InvalidStateError,
        ValueError,
        RuntimeError,
        OverflowError,
        FloatingPointError,
    ) as cause:
        # The public targets/guess were validated before iteration. A solver
        # stepping outside the homogeneous/source domain is nonconvergence,
        # rather than invalid caller input or a partially successful state.
        raise ConvergenceError(
            request=calculation_type,
            reason=f"A thermodynamic root iterate could not be evaluated: {cause}",
        ) from cause
    xf_reduced = np.asarray(xf_reduced)
    if (
        xf_reduced.shape != (2,)
        or not np.all(np.isfinite(xf_reduced))
        or np.any(xf_reduced <= 0)
    ):
        raise ConvergenceError(
            request=calculation_type,
            reason="The root solver returned invalid density/temperature coordinates.",
        )
    rho, T = xf_reduced[0] * rho_crit, xf_reduced[1] * T_crit

    # Check if solver converged
    if not solver.success:
        msg = f"Property calculation did not converge for calculation_type={calculation_type}.\n{solver.message}"
        raise ConvergenceError(request=calculation_type, reason=msg)

    props = problem.compute_properties(rho, T)
    # Validate the exact properties we will return. Calling problem.residual
    # here would flash the same root again and certify that second evaluation
    # instead of the returned state, while adding unnecessary source work.
    requests = ((prop_1, prop_1_value), (prop_2, prop_2_value))
    residual = np.asarray(
        [
            (props[name] - target) / scale
            for (name, target), scale in zip(requests, problem.residual_scales)
        ]
    )
    residual_limits = np.full(2, max(5 * solver_tolerance, 1e-10))
    for index, ((name, _), scale) in enumerate(zip(requests, problem.residual_scales)):
        if name == "pressure":
            # A stiff liquid's small net pressure is obtained by cancellation
            # of much larger EOS terms. A relative-to-pressure test alone can
            # reject a valid root at the source's floating representability
            # floor. Use already returned response values, not extra EOS calls
            # or checks inside the nonlinear iterations. Other residuals keep
            # their original, reference-invariant acceptance limits.
            physical_scales = np.asarray(
                [
                    props.get("isothermal_bulk_modulus", np.nan),
                    props.get("isentropic_bulk_modulus", np.nan),
                    float(props["density"]) * specific_R * float(props["temperature"]),
                    props["pressure"],
                ],
                dtype=float,
            )
            finite_scales = np.abs(physical_scales[np.isfinite(physical_scales)])
            if finite_scales.size:
                pressure_roundoff = 128 * np.finfo(float).eps * np.max(finite_scales)
                residual_limits[index] = max(
                    residual_limits[index], pressure_roundoff / scale
                )
    # An LM least-squares termination flag can describe a stationary point with
    # a nonzero thermodynamic residual. Validate the returned root independently.
    if not np.all(np.isfinite(residual)) or np.any(np.abs(residual) > residual_limits):
        raise ConvergenceError(
            request=calculation_type,
            reason=f"Solver terminated without a validated thermodynamic root: residual={residual}, limits={residual_limits}.",
        )
    return props


class _FlashCalculationResidual(psv.NonlinearSystemProblem):
    """Class to compute the residual of property calculations"""

    def __init__(
        self,
        prop_1,
        prop_1_value,
        prop_2,
        prop_2_value,
        function_handle,
        rho_scale,
        T_scale,
        residual_scales=None,
    ):
        self.prop_1 = prop_1
        self.prop_2 = prop_2
        self.prop_1_value = prop_1_value
        self.prop_2_value = prop_2_value
        self.compute_properties = function_handle
        self.rho_scale = rho_scale
        self.T_scale = T_scale
        # A target may be exactly zero after an arbitrary h/s reference shift,
        # or at a quality endpoint. Never divide by that target itself.
        self.residual_scales = residual_scales or (
            max(abs(float(prop_1_value)), 1.0),
            max(abs(float(prop_2_value)), 1.0),
        )

    def residual(self, x):
        # Ensure x can be indexed and contains exactly two elements
        if not hasattr(x, "__getitem__") or len(x) != 2:
            msg = f"Input x={x} must be a list, tuple or numpy array containing exactly two elements: density and temperature."
            raise ValueError(msg)

        # Unscale rho-T and compute properties
        rho = x[0] * self.rho_scale
        T = x[1] * self.T_scale
        props = self.compute_properties(rho, T)

        # Compute residual
        def compute_residual(prop_name, target_value, scale):
            value = props[prop_name]
            return (value - target_value) / scale

        res_1 = compute_residual(
            self.prop_1, self.prop_1_value, self.residual_scales[0]
        )
        res_2 = compute_residual(
            self.prop_2, self.prop_2_value, self.residual_scales[1]
        )

        return np.asarray([res_1, res_2])


def blend_properties(
    phase_change,
    props_equilibrium,
    props_metastable,
    blending_variable,
    blending_onset,
    blending_width,
):
    """
    Calculate blending between equilibrium and metastable fluid properties.

    A cubic smoothstep has zero slope at each transition endpoint. At an exact
    endpoint the selected branch is recovered even if an optional field on the
    inactive branch is unavailable. Inside the transition both optional values
    are needed. ``is_two_phase`` is unavailable inside the constitutive blend,
    since a fractional phase flag would not have its usual physical meaning.
    The coordinate/factor are not returned as thermodynamic fields: ``sigma``
    already denotes surface tension in the public property schema.

    Parameters
    ----------
    phase_change : str
        The type of phase change (e.g., 'condensation', 'evaporation', 'flashing', 'cavitation'). Cavitation, flashing, and evaporation do the same calculations, they are aliases added for convenience.
    props_equilibrium : dict
        The equilibrium properties.
    props_metastable : dict
        The metastable properties.
    blending_variable : str
        The variable used for blending.
    blending_onset : float
        The onset value for blending.
    blending_width : float
        The width value for blending.

    Returns
    -------
    dict
        Blended thermodynamic properties.
    """

    normalized_phase_change, blending_variable = _validated_blending_settings(
        phase_change, blending_variable, blending_onset, blending_width
    )
    props_equilibrium = _blending_property_dict(props_equilibrium)
    props_metastable = _blending_property_dict(props_metastable)
    if blending_variable not in props_equilibrium:
        raise UnsupportedPropertyError(
            request=blending_variable,
            reason="The equilibrium state does not enable the selected blending property.",
        )
    coordinate = np.asarray(props_equilibrium[blending_variable])
    if not np.all(np.isfinite(coordinate)):
        raise InvalidStateError(
            request=blending_variable,
            reason="The blending coordinate must be finite at every requested state.",
        )

    # Calculate the value of "x" based on the normalized phase change
    if normalized_phase_change == "condensation":
        x = 1 + (coordinate - blending_onset) / blending_width
    elif normalized_phase_change == "evaporation":
        x = 1 - (coordinate - blending_onset) / blending_width

    # Calculate the blending factor sigma
    limited = np.clip(x, 0.0, 1.0)
    sigma = limited * limited * (3.0 - 2.0 * limited)

    # Blend properties
    props_blended = {}
    for key in props_equilibrium.keys() | props_metastable.keys():
        prop_equilibrium = props_equilibrium.get(key, np.nan)
        prop_metastable = props_metastable.get(key, np.nan)
        # Exact endpoint selection avoids NaN*0 from an inactive optional field.
        # The transition requires both endpoints. A phase flag is not a
        # continuously blendable physical quantity.
        interior = (
            np.nan
            if key == "is_two_phase"
            else prop_equilibrium * (1 - sigma) + prop_metastable * sigma
        )
        props_blended[key] = np.where(
            sigma == 0,
            prop_equilibrium,
            np.where(sigma == 1, prop_metastable, interior),
        )

    # The coordinate/factor are algorithm data, not extra thermodynamic fields.
    # In particular, sigma is already the public alias for surface tension.

    return props_blended


def _validated_blending_settings(phase_change, variable, onset, width):
    """Normalize static constitutive settings before source/solver evaluation."""
    normalized = (
        PHASE_CHANGE_ALIASES.get(phase_change)
        if isinstance(phase_change, str)
        else None
    )
    if normalized is None:
        raise InvalidConfigurationError(
            request="phase_change",
            reason="Unknown phase-change direction.",
            supported=tuple(PHASE_CHANGE_ALIASES),
        )
    if not isinstance(variable, str) or variable not in ALIAS_TO_CANONICAL:
        raise UnsupportedPropertyError(
            request=str(variable), reason="Unknown blending property name or alias."
        )
    for name, value in (("blending_onset", onset), ("blending_width", width)):
        array = np.asarray(value)
        if (
            array.shape != ()
            or array.dtype.kind not in "fiu"
            or not np.isfinite(array)
            or (name == "blending_width" and array <= 0)
        ):
            raise InvalidConfigurationError(
                request=name,
                reason="Expected a finite scalar; blending width must be positive.",
            )
    return normalized, ALIAS_TO_CANONICAL[variable]


def _blending_property_dict(properties):
    """Accept public states or mappings and retain canonical numeric data."""
    if hasattr(properties, "to_dict"):
        properties = properties.to_dict()
    if not isinstance(properties, Mapping):
        raise InvalidConfigurationError(
            request="blending_properties",
            reason="Expected a FluidState or property mapping.",
        )
    result = {}
    for name, value in properties.items():
        if not isinstance(name, str) or name not in ALIAS_TO_CANONICAL:
            raise UnsupportedPropertyError(
                request=str(name),
                reason="Unknown property in a blending input mapping.",
            )
        canonical = ALIAS_TO_CANONICAL[name]
        array = np.asarray(value)
        if canonical == "is_two_phase" and array.dtype.kind == "b":
            array = array.astype(float)
        if array.dtype.kind not in "fiu":
            raise InvalidStateError(
                request=canonical,
                reason="Blended properties must be real numeric arrays.",
            )
        if canonical in result and not np.array_equal(
            result[canonical], array, equal_nan=True
        ):
            raise InvalidConfigurationError(
                request=canonical,
                reason="Conflicting canonical and alias property values.",
            )
        result[canonical] = array
    return result


# ------------------------------------------------------------------------------------ #
# Additional property calculations
# ------------------------------------------------------------------------------------ #

# def calculate_generalized_quality(abstract_state, alpha=10):
#     r"""
#     Calculate the generalized quality of a fluid, extending the quality calculation beyond the two-phase region if necessary.

#     Below the critical temperature, the quality is calculated from the specific volume of the saturated liquid and vapor states.
#     Above the critical temperature, an artificial two-phase region is defined around the critical density line to allow for a finite-width quality computation.

#     The quality, :math:`Q`, is given by:

#     .. math::

#         Q = \frac{v - v(T, Q=0)}{v(T, Q=1) - v(T, Q=0)}

#     where :math:`v=1/\rho` is the specific volume and :math:`T` is the temperature.

#     Additionally, this function applies smoothing limiters to ensure the quality is bounded between -1 and 2.
#     These limiters prevent the quality value from taking arbitrarily large values, enhancing stability and robustness of downstream calculation.
#     The limiters use the `logsumexp` method for smooth transitions, with a specified alpha value controlling the smoothness.

#     Parameters
#     ----------
#     abstract_state : CoolProp.AbstractState
#         CoolProp abstract state of the fluid.

#     alpha : float
#         Smoothing factor of the quality-calculation limiters.

#     Returns
#     -------
#     float
#         The calculated quality of the fluid.
#     """
#     # Instantiate new abstract state to compute saturation properties without changing the state of the class
#     AS = abstract_state
#     fluids = AS.fluid_names()  # AS.name does not work well for REFPROP backend
#     if len(fluids) != 1:
#         raise ValueError(f"Expected one fluid, got {fluids}")
#     cloned_AS = CP.AbstractState(AS.backend_name(), fluids[0])

#     # Extend quality calculation beyond the two-phase region
#     # Checking if subcritical using temperature works better than with pressure
#     if abstract_state.T() < abstract_state.T_critical():
#         # Saturated liquid
#         cloned_AS.update(CP.QT_INPUTS, 0.00, abstract_state.T())
#         rho_liq = cloned_AS.rhomass()

#         # Saturated vapor
#         cloned_AS.update(CP.QT_INPUTS, 1.00, abstract_state.T())
#         rho_vap = cloned_AS.rhomass()

#     else:
#         # For states at or above the critical temperature, the concept of saturation states is not applicable
#         # Instead, an artificial two-phase region is created around the pseudo-critical density line (line of critical density)
#         # The width of the artificial two-phase region is assumed to increase linearly with temperature

#         # Rename properties
#         T = abstract_state.T()
#         T_crit = abstract_state.T_critical()
#         rho_crit = abstract_state.rhomass_critical()

#         # Define pseudocritical region
#         T_hat = 1.5 * T_crit
#         rho_hat_liq = 1.1 * rho_crit
#         rho_hat_vap = 0.9 * rho_crit
#         rho_liq = rho_crit + (rho_hat_liq - rho_crit) * (T - T_crit) / (T_hat - T_crit)
#         rho_vap = rho_crit + (rho_hat_vap - rho_crit) * (T - T_crit) / (T_hat - T_crit)

#     # Compute quality according to definition
#     rho = abstract_state.rhomass()
#     quality = (1 / rho - 1 / rho_liq) / (1 / rho_vap - 1 / rho_liq + 1e-6)

#     # Apply smoothing limiters so that the quality is bounded between [-1, 2]
#     # The margin is defined as delta_Q=1 to the left of Q=0 and to the right of Q=1
#     quality = math.smooth_minimum(quality, +2, method="logsumexp", alpha=alpha)
#     quality = math.smooth_maximum(quality, -1, method="logsumexp", alpha=alpha)

#     return quality.item()


def calculate_generalized_quality(abstract_state, alpha=10, source_context=None):
    r"""Return physical quality in the dome and an entropy extension outside.

    Subcritical one-phase states use saturated entropies at their temperature:
    Q=(s-s_L)/(s_V-s_L). For supercritical temperatures no physical vapor fraction
    exists. The conventional entropy coordinate is centered on the source's
    critical-density isochore, with positive half-width
    R_specific*max(T/Tc-1, sqrt(machine_epsilon)). This gas-constant scale replaces
    multiples of absolute critical entropy: reference changes cannot change the
    indicator or reverse its orientation. The tiny reduced-temperature floor
    defines the exact critical limit without division by zero.

    Outside the physical dome the log-sum-exp limiters smoothly bound the
    coordinate near [-1, 2]. It is an indicator, not an equilibrium quality or
    metastable branch selector. Missing source endpoints remain unavailable.
    """
    AS = abstract_state
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            "The quality smoothing parameter alpha must be finite and positive."
        )
    if AS.phase() == CP.iphase_twophase and 0.0 <= AS.Q() <= 1.0:
        return float(AS.Q())
    T = AS.T()
    T_crit = (
        AS.T_critical()
        if source_context is None
        else source_context.critical_temperature
    )
    scratch = helper_state(AS, source_context)
    if T < T_crit:
        scratch.update(CP.QT_INPUTS, 0.0, T)
        s_liq = scratch.smass()
        scratch.update(CP.QT_INPUTS, 1.0, T)
        s_vap = scratch.smass()
        quality = _optional_divide(AS.smass() - s_liq, s_vap - s_liq)
    else:
        rho_crit = (
            AS.rhomass_critical()
            if source_context is None
            else source_context.critical_density
        )
        entropy_scale = (
            AS.gas_constant() / AS.molar_mass()
            if source_context is None
            else source_context.entropy_scale
        )
        scratch.update(CP.DmassT_INPUTS, rho_crit, T)
        center = scratch.smass()
        reduced_width = max(T / T_crit - 1.0, np.sqrt(np.finfo(float).eps))
        quality = 0.5 + _optional_divide(
            AS.smass() - center, 2 * entropy_scale * reduced_width
        )
    # This is an eager host-source helper. NumPy avoids separate JAX scalar
    # dispatches solely to limit one scalar from an already eager EOS call.
    quality = np.clip(quality, -1.0 - 100 / alpha, 2.0 + 100 / alpha)
    quality = -np.logaddexp(-alpha * quality, -2.0 * alpha) / alpha
    quality = np.logaddexp(alpha * quality, -alpha) / alpha
    return float(quality)


def calculate_trimmed_quality(abstract_state, source_context=None):
    """Return exact flash quality/void fraction, or a one-phase entropy label.

    Volume quality uses actual flashed vapor density, including the quality-
    dependent density of gliding pseudo-pure REFPROP models. The one-phase 0/1
    extension is only a critical-entropy label, not a physical vapor fraction.
    """
    AS = abstract_state
    if AS.phase() == CP.iphase_twophase:
        quality = AS.Q()
        if quality == 0.0 or quality == 1.0:
            return float(quality), float(quality)
        vapor_density = _optional_value(
            lambda: AS.saturated_vapor_keyed_output(CP.iDmass)
        )
        return float(quality), _optional_divide(quality * AS.rhomass(), vapor_density)
    if source_context is None:
        scratch = helper_state(AS)
        scratch.update(CP.DmassT_INPUTS, AS.rhomass_critical(), AS.T_critical())
        critical_entropy = scratch.smass()
    else:
        critical_entropy = source_context.critical_entropy
        if critical_entropy is None:
            return np.nan, np.nan
    quality = float(AS.smass() > critical_entropy)
    return quality, quality


def calculate_superheating(abstract_state, source_context=None, *, _transaction=None):
    r"""
    Calculate the degree of superheating for a given CoolProp abstract state.

    This function computes the superheating of a fluid using the CoolProp library's
    abstract state class. It handles both subcritical and supercritical conditions
    to provide a measure of superheating for any thermodynamic state. This results in
    a continuous variation of superheating in the two-phase region, which is necessary
    to achieve in reliable convergence of systems of equations and optimization problems
    involving the degree of superheating.

    Calculation cases:
        - Physically meaningful superheating for subcritical states in the vapor region:

        .. math::

          \Delta T = T - T(p, Q=1) \quad \text{for} \quad h > h(p, Q=1)

        - Artificial superheating for subcritical states in the liquid and two-phase regions:

        .. math::

          \Delta T = \frac{h - h(p, Q=1)}{c_p(p, Q=1)}

        - Artificial superheating for supercritical states defined using the critical density line:

        .. math::

          \Delta T = T - T(p, \rho_{\text{crit}})

    Parameters
    ----------
    abstract_state : CoolProp.AbstractState
        The abstract state of the fluid for which the superheating is to be calculated.

    Returns
    -------
    float
        The degree of superheating in Kelvin.

    Examples
    --------
    >>> import CoolProp as CP
    >>> abstract_state = CP.AbstractState("HEOS", "water")
    >>> abstract_state.update(CP.PT_INPUTS, 101325, 120 + 273.15)
    >>> superheating = calculate_superheating(abstract_state)
    >>> print(f"Superheating is {superheating:+0.3f} K")
    Superheating is +20.026 K

    >>> abstract_state = CP.AbstractState("HEOS", "water")
    >>> abstract_state.update(CP.PQ_INPUTS, 101325, 0.95)
    >>> superheating = calculate_superheating(abstract_state)
    >>> print(f"Superheating is {superheating:+0.3f} K")
    Superheating is -54.244 K
    """
    AS = abstract_state
    transaction = _transaction or _SaturationDiagnosticTransaction(AS, source_context)
    pressure, enthalpy, temperature = AS.p(), AS.hmass(), AS.T()
    if pressure < transaction.critical_pressure:
        minimum = transaction.triple_pressure(1.0)
        endpoint = transaction.sample(CP.PQ_INPUTS, max(minimum, pressure), 1.0)
        if enthalpy < endpoint["enthalpy"]:
            return _optional_divide(enthalpy - endpoint["enthalpy"], endpoint["cp"])
        return temperature - endpoint["temperature"]
    endpoint = transaction.sample(
        CP.DmassP_INPUTS, transaction.critical_density, pressure
    )
    return temperature - endpoint["temperature"]


def calculate_subcooling(abstract_state, source_context=None, *, _transaction=None):
    r"""
    Calculate the degree of subcooling for a given CoolProp abstract state.

    This function computes the subcooling of a fluid using the CoolProp library's
    abstract state class. It handles both subcritical and supercritical conditions
    to provide a measure of subcooling for any thermodynamic state. This results in
    a continuous variation of subcooling in the two-phase region, which is necessary
    to achieve reliable convergence of systems of equations and optimization problems
    involving the degree of subcooling.

    Calculation cases:
        - Physically meaningful subcooling for subcritical states in the liquid region:

        .. math::

          \Delta T = T(p, Q=0) - T \quad \text{for} \quad h < h(p, Q=0)

        - Artificial subcooling for subcritical states in the vapor and two-phase regions:

        .. math::

          \Delta T = \frac{h(p, Q=0) - h}{c_p(p, Q=0)}

        - Artificial subcooling for supercritical states defined using the critical density line:

        .. math::

          \Delta T = T(p, \rho_{\text{crit}}) - T

    Parameters
    ----------
    abstract_state : CoolProp.AbstractState
        The abstract state of the fluid for which the subcooling is to be calculated.

    Returns
    -------
    float
        The degree of subcooling in Kelvin.

    Examples
    --------
    >>> import CoolProp as CP
    >>> abstract_state = CP.AbstractState("HEOS", "water")
    >>> abstract_state.update(CP.PT_INPUTS, 101325, 25+273.15)
    >>> subcooling = bpy.calculate_subcooling(abstract_state)
    >>> print(f"Subcooling is {subcooling:+0.3f} K")
    Subcooling is +74.974 K

    >>> abstract_state = CP.AbstractState("HEOS", "water")
    >>> abstract_state.update(CP.PQ_INPUTS, 101325, 0.05)
    >>> subcooling = bpy.calculate_subcooling(abstract_state)
    >>> print(f"Subcooling is {subcooling:+0.3f} K")
    Subcooling is -26.763 K
    """

    AS = abstract_state
    transaction = _transaction or _SaturationDiagnosticTransaction(AS, source_context)
    pressure, enthalpy, temperature = AS.p(), AS.hmass(), AS.T()
    if pressure < transaction.critical_pressure:
        minimum = transaction.triple_pressure(0.0)
        endpoint = transaction.sample(CP.PQ_INPUTS, max(minimum, pressure), 0.0)
        if enthalpy > endpoint["enthalpy"]:
            return _optional_divide(endpoint["enthalpy"] - enthalpy, endpoint["cp"])
        return endpoint["temperature"] - temperature
    endpoint = transaction.sample(
        CP.DmassP_INPUTS, transaction.critical_density, pressure
    )
    return endpoint["temperature"] - temperature


def calculate_supersaturation(
    abstract_state, props, source_context=None, *, _transaction=None
):
    r"""
    Evaluate degree of supersaturation and supersaturation ratio.

    The supersaturation degree is defined as the actual temperature minus the saturation temperature at the corresponding pressure:

    .. math::
        \Delta T = T - T_{\text{sat}}(p)

    The supersaturation ratio is defined as the actual pressure divided by the saturation pressure at the corresponding temperature:

    .. math::
        S = \frac{p}{p_{\text{sat}}(T)}

    The metastable liquid and metastable vapor regions are illustrated in the pressure-temperature diagram.
    In the metastable liquid region, :math:`\Delta T > 0` and :math:`S < 1`, indicating that the liquid temperature
    is higher than the equilibrium temperature and will tend to evaporate. Conversely, in the metastable vapor region,
    :math:`\Delta T < 0` and :math:`S > 1`, indicating that the vapor temperature is lower than the equilibrium temperature
    and will tend to condense.

    .. image:: /_static/metastable_regions_CO2.svg
        :alt: Pressure-density diagram and spinodal points for carbon dioxide.


    .. note::

        Supersaturation properties are only computed for subcritical pressures and temperatures.
        If the fluid is in the supercritical region, the function will return NaN for the supersaturation properties.


    Parameters
    ----------
    abstract_state : CoolProp.AbstractState
        The abstract state of the fluid for which the properties are to be calculated.
    props : dict
        Dictionary to store the computed properties.

    Returns
    -------
    dict
        Thermodynamic properties including supersaturation properties

    """
    AS = abstract_state
    keys = (
        "pressure_saturation",
        "supersaturation_ratio",
        "temperature_saturation",
        "supersaturation_degree",
    )
    props.update(dict.fromkeys(keys, np.nan))
    try:
        transaction = _transaction or _SaturationDiagnosticTransaction(
            AS, source_context
        )
        minimum_pressure = transaction.triple_pressure(0.0)
    except _OPTIONAL_PROPERTY_ERRORS:
        return props

    if (
        transaction.triple_temperature
        < props["temperature"]
        < transaction.critical_temperature
    ):
        try:
            endpoint = transaction.sample(CP.QT_INPUTS, 0.0, props["temperature"])
            props["pressure_saturation"] = endpoint["pressure"]
            props["supersaturation_ratio"] = props["pressure"] / endpoint["pressure"]
        except _OPTIONAL_PROPERTY_ERRORS:
            pass
    if minimum_pressure < props["pressure"] < transaction.critical_pressure:
        try:
            endpoint = transaction.sample(CP.PQ_INPUTS, props["pressure"], 0.0)
            props["temperature_saturation"] = endpoint["temperature"]
            props["supersaturation_degree"] = (
                props["temperature"] - endpoint["temperature"]
            )
        except _OPTIONAL_PROPERTY_ERRORS:
            pass
    return props


def calculate_mixture_properties_hp(h_mix, p_mix, AS1, AS2, R):
    """Evaluate the common-p/T parcel model from caller-owned source states.

    Parameters
    ----------
    h_mix, p_mix : float
        Mixture specific enthalpy [J/kg] and common pressure [Pa].
    AS1, AS2 : CoolProp.AbstractState
        Configured component states. Their existing reference conventions are
        retained; this compatibility call does not reconstruct source objects.
    R : float
        Nonnegative component mass ratio, mass_1 / mass_2.

    Returns
    -------
    dict
        Canonical full properties from the consolidated binary kernel. Density
        uses additive specific volumes; h/u/s use mass weights. Thermal response
        and acoustic properties follow the same equilibrium thermodynamics.

    Repeated applications should construct Fluid(method='binary_mixture') once
    and use its complete get_state API. This raw compatibility entry shares its
    inverse and mixing implementation rather than maintaining a second model.
    """
    from ..binary_mixture.host import legacy_hp

    return legacy_hp(h_mix, p_mix, AS1, AS2, R)
