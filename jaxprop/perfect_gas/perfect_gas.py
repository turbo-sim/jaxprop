import jax
import jax.numpy as jnp
import equinox as eqx
import math
import numpy as np

from ..coolprop import FluidCoolProp
from .. import helpers_props as jxp
from ..configuration import build_configuration, FluidCapabilities, FluidDomain
from ..errors import InvalidConfigurationError, JaxPropError, UnsupportedOperationError
from .request_policy import checked_perfect_gas_jit, evaluate_state
from ..source_references import SourceReferencesMixin

MIN_TEMPERATURE = 1.0
MIN_PRESSURE = 0.1
MIN_DENSITY = 1e-12


# ----------------------------------------------------------------------------- #
# Constant generation and Sutherland estimation
# ----------------------------------------------------------------------------- #
class PerfectGasConstants(eqx.Module):
    R: jnp.ndarray
    gamma: jnp.ndarray
    T_ref: jnp.ndarray
    P_ref: jnp.ndarray
    h_ref: jnp.ndarray
    s_ref: jnp.ndarray
    k_ref: jnp.ndarray
    mu_ref: jnp.ndarray
    S_k: jnp.ndarray
    S_mu: jnp.ndarray

    def __repr__(self) -> str:
        """Return a readable string representation of the constants,
        listing all field names and their scalar values."""
        lines = []
        for name, val in self.__dict__.items():
            try:
                val = jnp.array(val).item()  # scalar to Python number
            except Exception:
                pass
            lines.append(f"  {name}={val}")
        return "PerfectGasConstants(\n" + ",\n".join(lines) + "\n)"


def get_constants(fluid_name, T_ref, p_ref, dT=100.0, *, backend="HEOS"):
    """
    Compute perfect-gas constants from a real-fluid model at a reference state,
    and estimate Sutherland constants using offset temperatures.

    Parameters
    ----------
    fluid_name : str
        Name of the fluid (passed to CoolProp).
    T_ref : float
        Reference temperature in kelvin [K].
    P_ref : float
        Reference pressure in pascal [Pa].
    dT : float, optional
        Temperature offset used to evaluate `mu` and `k` at T_ref ± dT for
        Sutherland constant estimation [K]. Default is 100.0.
    backend : str, optional
        CoolProp source backend used for calibration. Default is "HEOS".

    Returns
    -------
    GasConstants
        An Equinox module containing the perfect-gas constants:
        - R : specific gas constant [J/(kg·K)]
        - gamma : specific heat ratio [-]
        - T_ref : reference temperature [K]
        - P_ref : reference pressure [Pa]
        - s_ref : reference entropy [J/(kg·K)]
        - mu_ref : dynamic viscosity at T_ref [Pa·s]
        - S_mu : Sutherland constant for viscosity [K]
        - k_ref : thermal conductivity at T_ref [W/(m·K)]
        - S_k : Sutherland constant for thermal conductivity [K]

    Raises
    ------
    ValueError
        If required properties are missing or invalid.
    ZeroDivisionError
        If the Sutherland estimation becomes numerically unstable.
    """
    return _get_constants_from_source(
        FluidCoolProp(name=fluid_name, backend=backend), T_ref, p_ref, dT
    )


def _get_constants_from_source(fluid, T_ref, p_ref, dT=100.0):
    """Calibrate one ideal-gas model, retaining the source h/s convention.

    The real-fluid heat capacities need not obey cp-cv=R. Preserve calibrated
    cp and derive cv/gamma from the ideal-gas identity instead of combining
    incompatible source values. Optional transport never determines whether a
    valid thermodynamic model can be constructed.
    """
    state = fluid.get_state(jxp.PT_INPUTS, p_ref, T_ref)
    R = float(jxp.GAS_CONSTANT / fluid.abstract_state.molar_mass())
    cp = float(state.raw_property("cp"))
    if not math.isfinite(cp) or not math.isfinite(R) or R <= 0 or cp <= R:
        raise InvalidConfigurationError(
            fluid_name=fluid.name,
            provider="native",
            model="perfect_gas",
            request="calibration",
            reason="The source calibration requires finite cp > R > 0 so ideal-gas cv is positive.",
        )
    cv = cp - R
    gamma = cp / cv
    h_ref = float(state.raw_property("h"))
    s_ref = float(state.raw_property("s"))
    mu_ref = float(state.raw_property("mu"))
    k_ref = float(state.raw_property("k"))

    # Use a distinct in-domain temperature. Near a source maximum, fit below
    # calibration rather than extrapolating T_ref + dT beyond the EOS limits.
    upper, lower = float(fluid.T_max), float(fluid.T_min)
    T2 = min(float(T_ref) + abs(float(dT)), upper)
    if T2 == T_ref:
        T2 = max(float(T_ref) - abs(float(dT)), lower)
    second = None
    if T2 != T_ref:
        try:
            second = fluid.get_state(jxp.PT_INPUTS, p_ref, T2)
        except JaxPropError:
            # Transport fitting is optional. Core calibration above succeeded.
            pass

    def fit(reference, name):
        if second is None:
            return float("nan"), float("nan")
        other = float(second.raw_property(name))
        if not all(math.isfinite(value) and value > 0 for value in (reference, other)):
            return float("nan"), float("nan")
        try:
            coefficient = float(
                estimate_sutherland_constant(T_ref, reference, T2, other)
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            return float("nan"), float("nan")
        if not math.isfinite(coefficient) or T_ref + coefficient == 0:
            return float("nan"), float("nan")
        return reference, coefficient

    mu_ref, S_mu = fit(mu_ref, "mu")
    k_ref, S_k = fit(k_ref, "k")
    consts = PerfectGasConstants(
        **{
            name: jnp.asarray(value, dtype=jnp.result_type(value))
            for name, value in dict(
                R=R,
                gamma=gamma,
                T_ref=T_ref,
                P_ref=p_ref,
                h_ref=h_ref,
                s_ref=s_ref,
                k_ref=k_ref,
                mu_ref=mu_ref,
                S_k=S_k,
                S_mu=S_mu,
            ).items()
        }
    )

    return consts


def estimate_sutherland_constant(T1, mu1, T2, mu2):
    """
    Estimate the Sutherland constant [K] from two values of a transport property
    (viscosity or thermal conductivity) at two temperatures, assuming a Sutherland-like law:

        mu(T) = mu_ref * (T / T_ref)^{3/2} * (T_ref + S) / (T + S)

    Parameters
    ----------
    T1, T2 : float
        Temperatures in kelvin [K].
    mu1, mu2 : float
        Transport property values at T1 and T2 (e.g. viscosity in Pa·s or conductivity in W/m·K).

    Returns
    -------
    float
        Estimated Sutherland constant in kelvin [K].
    """
    r = (mu2 / mu1) * (T2 / T1) ** (-1.5)
    return (r * T2 - T1) / (1 - r)


# ----------------------------------------------------------------------------- #
# Helper functions to calculate individual fluid properties
# ----------------------------------------------------------------------------- #


def specific_heat(constants):
    R = constants.R
    gamma = constants.gamma
    cp = (gamma * R) / (gamma - 1)
    cv = cp / gamma
    return cp, cv


def temperature_from_h(h, constants):
    cp, _ = specific_heat(constants)
    return constants.T_ref + (h - constants.h_ref) / cp


def temperature_from_Ps(p, s, constants):
    R = constants.R
    gamma = constants.gamma
    T_ref = constants.T_ref
    P_ref = constants.P_ref
    s_ref = constants.s_ref
    cp, _ = specific_heat(constants)
    exponent = jnp.log(T_ref) + ((s - s_ref) + R * (jnp.log(p) - jnp.log(P_ref))) / cp
    return jnp.exp(exponent)


def temperature_from_rhoP(rho, p, constants):
    R = constants.R
    return p / (rho * R)


def viscosity_from_T(T, constants):
    mu_ref = constants.mu_ref
    T_ref = constants.T_ref
    S_mu = constants.S_mu
    # Cancel one T/T_ref factor before evaluation. The equivalent T**1.5
    # expression overflows at high finite T even when transport is finite.
    value = (
        mu_ref * (jnp.sqrt(T) / jnp.sqrt(T_ref)) * ((1 + S_mu / T_ref) / (1 + S_mu / T))
    )
    return jnp.where(jnp.isfinite(value) & (value > 0), value, jnp.nan)


def conductivity_from_T(T, constants):
    k_ref = constants.k_ref
    T_ref = constants.T_ref
    S_k = constants.S_k
    value = (
        k_ref * (jnp.sqrt(T) / jnp.sqrt(T_ref)) * ((1 + S_k / T_ref) / (1 + S_k / T))
    )
    return jnp.where(jnp.isfinite(value) & (value > 0), value, jnp.nan)


def speed_of_sound_from_T(T, constants):
    R = constants.R
    gamma = constants.gamma
    # Avoid overflowing gamma*R*T when its square root is representable.
    value = jnp.sqrt(gamma) * jnp.sqrt(R * T)
    return jnp.where(jnp.isfinite(value), value, jnp.nan)


def entropy_from_PT(p, T, constants):
    R = constants.R
    T_ref = constants.T_ref
    P_ref = constants.P_ref
    s_ref = constants.s_ref
    cp, _ = specific_heat(constants)
    return (
        s_ref + cp * (jnp.log(T) - jnp.log(T_ref)) - R * (jnp.log(p) - jnp.log(P_ref))
    )


def entropy_from_rhoP(rho, p, constants):
    R = constants.R
    T = p / (rho * R)
    return entropy_from_PT(p, T, constants)


# ----------------------------------------------------------------------------- #
# Helper functions to calculate full fluid states
# ----------------------------------------------------------------------------- #


@jax.custom_jvp
def _density_from_PT(p, T, R):
    """Preserve the ideal-gas primal while using stable analytic derivatives.

    The request policy already supplies admissible coordinates. No additional
    validation is introduced here. Differentiating p/(R*T) mechanically can
    form p*d(R*T), overflowing even when the final derivative is finite.
    """
    return p / (R * T)


@_density_from_PT.defjvp
def _density_from_PT_jvp(primals, tangents):
    p, T, R = primals
    dp, dT, dR = tangents
    rho = _density_from_PT(p, T, R)
    # d(rho) = dp/(R*T) - rho*dT/T - rho*dR/R. Form the relative
    # directions first: tiny partials must not underflow before multiplication
    # by a large model direction, and large p need not appear in a product.
    tangent = dp / (R * T) - rho * (dT / T) - rho * (dR / R)
    return rho, tangent


def calculate_properties_PT(p, T, constants):
    R = constants.R
    rho = _density_from_PT(p, T, R)
    cp, _ = specific_heat(constants)
    h = constants.h_ref + cp * (T - constants.T_ref)
    s = entropy_from_PT(p, T, constants)
    return assemble_properties(T, p, rho, h, s, constants)


def calculate_properties_hs(h, s, constants):
    cp, _ = specific_heat(constants)
    T = temperature_from_h(h, constants)
    R = constants.R
    T_ref = constants.T_ref
    P_ref = constants.P_ref
    s_ref = constants.s_ref
    P = jnp.exp(jnp.log(P_ref) + (cp * (jnp.log(T) - jnp.log(T_ref)) - (s - s_ref)) / R)
    rho = _density_from_PT(P, T, R)
    return assemble_properties(T, P, rho, h, s, constants)


def calculate_properties_hP(h, p, constants):
    T = temperature_from_h(h, constants)
    R = constants.R
    rho = _density_from_PT(p, T, R)
    s = entropy_from_PT(p, T, constants)
    return assemble_properties(T, p, rho, h, s, constants)


def calculate_properties_Ps(p, s, constants):
    T = temperature_from_Ps(p, s, constants)
    R = constants.R
    rho = _density_from_PT(p, T, R)
    cp, _ = specific_heat(constants)
    h = constants.h_ref + cp * (T - constants.T_ref)
    return assemble_properties(T, p, rho, h, s, constants)


def calculate_properties_rhoh(rho, h, constants):
    T = temperature_from_h(h, constants)
    R = constants.R
    p = rho * R * T
    s = entropy_from_rhoP(rho, p, constants)
    return assemble_properties(T, p, rho, h, s, constants)


def calculate_properties_rhop(rho, p, constants):
    cp, _ = specific_heat(constants)
    T = temperature_from_rhoP(rho, p, constants)
    h = constants.h_ref + cp * (T - constants.T_ref)
    s = entropy_from_rhoP(rho, p, constants)
    return assemble_properties(T, p, rho, h, s, constants)


def assemble_properties(T, p, rho, h, s, constants):
    R = constants.R
    gamma = constants.gamma
    # Broadcast the CONSTANT-valued props (gamma, cp, cv, Z, Grüneisen) to the shape
    # of the input-dependent props so EVERY FluidState field shares one shape.  Real
    # backends (FluidBicubic) return all fields at the input shape; consumers rely on
    # it (e.g. st["G"][0]).  Without this, a scalar-input call leaves gamma 0-dim
    # while rho/T are (1,) → "too many indices" on st["G"][0].
    ones = jnp.ones_like(T)
    return {
        "temperature": T,
        "pressure": p,
        "density": rho,
        "enthalpy": h,
        "internal_energy": h - R * T,
        "entropy": s,
        "viscosity": viscosity_from_T(T, constants),
        "conductivity": conductivity_from_T(T, constants),
        "speed_of_sound": speed_of_sound_from_T(T, constants),
        "heat_capacity_ratio": gamma * ones,
        "isobaric_heat_capacity": (gamma * R / (gamma - 1)) * ones,
        "isochoric_heat_capacity": (R / (gamma - 1)) * ones,
        "compressibility_factor": ones,
        "gruneisen": (gamma - 1) * ones,
        "is_two_phase": jnp.zeros_like(T),
        "isobaric_expansion_coefficient": 1 / T,
        "isothermal_compressibility": 1 / p,
        "isentropic_compressibility": (1 / p) / gamma,
        "isothermal_bulk_modulus": p,
        "isentropic_bulk_modulus": jnp.where(
            jnp.isfinite(gamma * p), gamma * p, jnp.nan
        ),
        "joule_thomson": jnp.zeros_like(T),
        "isothermal_joule_thomson": jnp.zeros_like(T),
    }


# ----------------------------------------------------------------------------- #
# State evaluators (public API)
# ----------------------------------------------------------------------------- #

PROPERTY_CALCULATORS = {
    jxp.PT_INPUTS: calculate_properties_PT,
    jxp.HmassSmass_INPUTS: calculate_properties_hs,
    jxp.HmassP_INPUTS: calculate_properties_hP,
    jxp.PSmass_INPUTS: calculate_properties_Ps,
    jxp.DmassHmass_INPUTS: calculate_properties_rhoh,
    jxp.DmassP_INPUTS: calculate_properties_rhop,
}


def _validated_constant_snapshot(constants, *, name):
    """Validate an eagerly materialized model; never inspect traced calls."""
    values = {}
    arrays = {}
    for key in PerfectGasConstants.__dataclass_fields__:
        value = getattr(constants, key)
        try:
            array = np.asarray(value)
            if array.shape != () or array.dtype.kind not in "fiu":
                raise TypeError("Use a real numeric scalar.")
            values[key] = float(array)
            arrays[key] = array
        except (TypeError, ValueError, OverflowError) as error:
            raise InvalidConfigurationError(
                fluid_name=name,
                provider="native",
                model="perfect_gas",
                request=key,
                reason="Materialize model constants eagerly as real numeric scalars outside JAX transformations.",
            ) from error

    def reject(key, reason):
        raise InvalidConfigurationError(
            fluid_name=name,
            provider="native",
            model="perfect_gas",
            request=key,
            reason=reason,
        )

    for key in ("R", "T_ref", "P_ref"):
        if not math.isfinite(values[key]) or values[key] <= 0:
            reject(key, "The model constant must be finite and positive.")
        if (
            np.issubdtype(arrays[key].dtype, np.floating)
            and values[key] < np.finfo(arrays[key].dtype).tiny
        ):
            reject(
                key,
                "The model constant must be a positive normal value in its numerical dtype.",
            )
    if not math.isfinite(values["gamma"]) or values["gamma"] <= 1:
        reject(
            "gamma",
            "The ideal-gas heat capacity ratio must be finite and greater than one.",
        )
    # Respect actual coefficient dtypes. A float32 R/gamma combination can
    # overflow even though converting both values to Python float would hide it.
    cp_value, cv_value, reference_scale_value, intercept_value = (
        _coefficient_arithmetic(constants)
    )
    cp_array, cv_array = np.asarray(cp_value), np.asarray(cv_value)
    cp, cv = float(cp_array), float(cv_array)
    if not all(math.isfinite(value) and value > 0 for value in (cp, cv)):
        reject(
            "gamma",
            "The selected R/gamma must produce finite positive heat capacities.",
        )
    for key in ("h_ref", "s_ref"):
        if not math.isfinite(values[key]):
            reject(key, "Reference offsets must be finite real scalars.")
    output_dtype = jnp.result_type(
        constants.R,
        constants.gamma,
        constants.T_ref,
        constants.P_ref,
        constants.h_ref,
        constants.s_ref,
        0.0,
    )
    if not np.issubdtype(output_dtype, np.floating):
        output_dtype = np.dtype(float)
    info = np.finfo(output_dtype)
    reference_scale, energy_intercept = np.asarray(reference_scale_value), np.asarray(
        intercept_value
    )
    if not np.isfinite(reference_scale) or not np.isfinite(energy_intercept):
        reject(
            "h_ref",
            "Calibration energy scale and zero-temperature intercept must be representable in the model dtype.",
        )
    remaining = 1 - float(energy_intercept) / info.max
    with np.errstate(all="ignore"):
        upper_temperature = min(
            info.max,
            info.max / cp,
            info.max / values["R"],
            info.max * (remaining / cp),
            info.max * (remaining / cv),
        ) * (1 - 64 * info.eps)
    if not math.isfinite(upper_temperature) or upper_temperature < MIN_TEMPERATURE:
        reject(
            "h_ref",
            "The energy reference must leave a representable temperature domain above the declared numerical floor.",
        )
    # With stable logarithmic differences, every supported finite T/p has
    # bounded entropy increments. Establish that once, so the repeated kernel
    # need not calculate entropy merely to validate a temperature projection.
    # Both coordinates use this common model precision in the native resolver.
    temperature_dtype = pressure_dtype = output_dtype

    def entropy_steps(temperature, pressure):
        temperature_term = cp_value * (jnp.log(temperature) - jnp.log(constants.T_ref))
        first_sum = constants.s_ref + temperature_term
        pressure_term = constants.R * (jnp.log(pressure) - jnp.log(constants.P_ref))
        return temperature_term, first_sum, pressure_term, first_sum - pressure_term

    step_shapes = jax.eval_shape(
        entropy_steps,
        jax.ShapeDtypeStruct((), temperature_dtype),
        jax.ShapeDtypeStruct((), pressure_dtype),
    )
    step_limits = [float(np.finfo(shape.dtype).max) for shape in step_shapes]
    temperature_span = max(
        abs(math.log(MIN_TEMPERATURE) - math.log(values["T_ref"])),
        abs(math.log(np.finfo(temperature_dtype).max) - math.log(values["T_ref"])),
    )
    pressure_span = max(
        abs(math.log(MIN_PRESSURE) - math.log(values["P_ref"])),
        abs(math.log(np.finfo(pressure_dtype).max) - math.log(values["P_ref"])),
    )
    temperature_term = cp * temperature_span
    first_sum = abs(values["s_ref"]) + temperature_term
    pressure_term = values["R"] * pressure_span
    bounds = (temperature_term, first_sum, pressure_term, first_sum + pressure_term)
    if any(
        not math.isfinite(bound) or bound > limit
        for bound, limit in zip(bounds, step_limits)
    ):
        reject(
            "s_ref",
            "The entropy reference and heat capacities must admit finite entropy throughout the numerical model domain.",
        )
    for reference, coefficient in (("mu_ref", "S_mu"), ("k_ref", "S_k")):
        ref, fit = values[reference], values[coefficient]
        if math.isnan(ref) and math.isnan(fit):
            continue
        if not math.isfinite(ref) or ref <= 0 or not math.isfinite(fit):
            reject(
                reference,
                "Transport requires a finite positive reference and finite coefficient, or NaN for both unavailable constants.",
            )
        if values["T_ref"] + fit == 0:
            reject(
                coefficient,
                "A Sutherland pole cannot coincide with the calibration temperature.",
            )
    # NaN does not compare equal across independently constructed models.
    # Unavailable transport is an explicit, reflexive metadata value instead.
    return {
        key: value if math.isfinite(value) else None for key, value in values.items()
    }


@jax.jit
def _coefficient_arithmetic(constants):
    """Evaluate setup guarantees in the actual JAX coefficient dtypes."""
    cp, cv = specific_heat(constants)
    reference_scale = cp * constants.T_ref
    return cp, cv, reference_scale, constants.h_ref - reference_scale


def _materialized_constants(constants, *, name):
    """Make model precision explicit, retaining existing strong array objects."""
    values = {}
    changed = False
    for key in PerfectGasConstants.__dataclass_fields__:
        original = getattr(constants, key)
        try:
            value = jnp.asarray(original, dtype=jnp.result_type(original))
        except (TypeError, ValueError, OverflowError) as error:
            raise InvalidConfigurationError(
                fluid_name=name,
                provider="native",
                model="perfect_gas",
                request=key,
                reason="Materialize real numeric model constants outside JAX transformations.",
            ) from error
        values[key] = value
        changed = changed or value is not original
    if not changed:
        return constants
    result = object.__new__(PerfectGasConstants)
    for key, value in values.items():
        object.__setattr__(result, key, value)
    return result


def _model_reference_id(source_id, constants):
    return (
        "perfect_gas",
        source_id,
        ("calibration", constants["T_ref"], constants["P_ref"]),
        ("constants", tuple(constants.items())),
    )


class FluidPerfectGas(SourceReferencesMixin, eqx.Module):
    constants: PerfectGasConstants = eqx.field(static=False)
    fluid_name: str = eqx.field(static=True, default=None)
    identifier: str = eqx.field(static=True, default=None)
    provider: str = eqx.field(static=True, default="native")
    model: str = eqx.field(static=True, default="perfect_gas")
    reference_id: tuple = eqx.field(static=True, default=None)
    backend: str = eqx.field(static=True, default="HEOS")
    _configuration: object = eqx.field(static=True, default=None)
    _capabilities: object = eqx.field(static=True, default=None)
    _references: object = eqx.field(static=True, default=None)

    def __init__(
        self, name, T_ref=300.0, p_ref=101_325.0, identifier=None, *, backend=None
    ):
        backend = "HEOS" if backend is None else backend
        self._configuration = build_configuration(
            name,
            self.provider,
            self.model,
            backend,
            identifier,
            model_options={"T_ref": T_ref, "p_ref": p_ref},
        )
        from numbers import Real
        import math

        for key, value in (("T_ref", T_ref), ("p_ref", p_ref)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise InvalidConfigurationError(
                    fluid_name=name,
                    provider=self.provider,
                    model=self.model,
                    request=key,
                    reason="Calibration coordinates must be finite positive real scalars.",
                )
        self.backend = backend
        self._capabilities = FluidCapabilities(
            input_pairs=frozenset(PROPERTY_CALCULATORS),
            properties=frozenset(
                {
                    "temperature",
                    "pressure",
                    "density",
                    "enthalpy",
                    "entropy",
                    "viscosity",
                    "conductivity",
                    "speed_of_sound",
                    "heat_capacity_ratio",
                    "isobaric_heat_capacity",
                    "isochoric_heat_capacity",
                    "compressibility_factor",
                    "gruneisen",
                    "internal_energy",
                    "is_two_phase",
                    "isobaric_expansion_coefficient",
                    "isothermal_compressibility",
                    "isentropic_compressibility",
                    "isothermal_bulk_modulus",
                    "isentropic_bulk_modulus",
                    "joule_thomson",
                    "isothermal_joule_thomson",
                }
            ),
            jax_modes=frozenset({"eager", "jit", "vmap", "jvp", "grad"}),
            derivative_order=1,
            derivative_method="autodiff",
            error_modes={"raise", "mask", "clip"},
            domain=FluidDomain(
                description="Calibrated perfect-gas model with positive numerical temperature, pressure and density floors. Clip mode projects onto a consistent PT state; source-EOS bounds are calibration limits, not this model's runtime domain.",
                bounds={
                    "temperature": (MIN_TEMPERATURE, None),
                    "pressure": (MIN_PRESSURE, None),
                    "density": (MIN_DENSITY, None),
                },
                phases={"gas"},
            ),
        )
        try:
            source = FluidCoolProp(name=name, backend=backend)
        except InvalidConfigurationError as error:
            raise InvalidConfigurationError(
                fluid_name=name,
                provider=self.provider,
                model=self.model,
                request=error.request,
                reason=error.reason,
            ) from error
        try:
            self.constants = _get_constants_from_source(source, T_ref, p_ref)
        except JaxPropError as error:
            raise InvalidConfigurationError(
                fluid_name=name,
                provider=self.provider,
                model=self.model,
                request="calibration",
                reason=error.reason,
            ) from error
        self.fluid_name = name
        self.identifier = identifier
        self._references = source.references
        snapshot = _validated_constant_snapshot(self.constants, name=name)
        self.reference_id = _model_reference_id(source.reference_id, snapshot)
        self._configuration = build_configuration(
            name,
            self.provider,
            self.model,
            backend,
            identifier,
            model_options={
                "T_ref": float(T_ref),
                "p_ref": float(p_ref),
                "constants": snapshot,
            },
        )

    @property
    def name(self):
        return self.fluid_name

    @property
    def configuration(self):
        return self._configuration

    @property
    def capabilities(self):
        return self._capabilities

    def materialize(self):
        """Validate a dynamic model view and refresh its persistent identity.

        Low-level ``eqx.tree_at`` parameter edits remain efficient ephemeral
        differentiation views. Call this eager setup operation before storing
        or combining the resulting physical model's states. No source is
        initialized and the numerical parameter dtypes are retained.
        """
        constants = _materialized_constants(self.constants, name=self.name)
        snapshot = _validated_constant_snapshot(constants, name=self.name)
        options = dict(self.configuration.model_options)
        options.update(
            T_ref=snapshot["T_ref"], p_ref=snapshot["P_ref"], constants=snapshot
        )
        configuration = build_configuration(
            self.name,
            self.provider,
            self.model,
            self.backend,
            self.identifier,
            model_options=options,
        )
        result = object.__new__(type(self))
        for key, value in vars(self).items():
            object.__setattr__(result, key, value)
        object.__setattr__(result, "constants", constants)
        object.__setattr__(result, "_configuration", configuration)
        object.__setattr__(
            result,
            "reference_id",
            _model_reference_id(self.references.reference_id, snapshot),
        )
        return result

    def with_constants(self, **changes):
        """Return a validated model with updated constants and fresh identity.

        Constants are named as in :class:`PerfectGasConstants`; cp/cv remain
        derived from R/gamma, so they cannot become mutually inconsistent.
        Changes are materialized outside JAX transformations. Source reference
        records describe the calibration source, independently of this model.
        """
        unknown = changes.keys() - PerfectGasConstants.__dataclass_fields__.keys()
        if unknown:
            raise InvalidConfigurationError(
                fluid_name=self.name,
                provider=self.provider,
                model=self.model,
                request=sorted(unknown)[0],
                reason="Unknown perfect-gas model constant.",
                supported=tuple(PerfectGasConstants.__dataclass_fields__),
            )
        constants = self.constants
        for key, value in changes.items():
            try:
                value = jnp.asarray(value, dtype=jnp.result_type(value))
            except (TypeError, ValueError, OverflowError) as error:
                raise InvalidConfigurationError(
                    fluid_name=self.name,
                    provider=self.provider,
                    model=self.model,
                    request=key,
                    reason="Use a real numeric scalar model constant.",
                ) from error
            constants = eqx.tree_at(
                lambda data, key=key: getattr(data, key), constants, value
            )
        model = eqx.tree_at(lambda data: data.constants, self, constants)
        return model.materialize()

    def get_state_saturation(self, *args, **kwargs):
        raise UnsupportedOperationError(
            fluid_name=self.name,
            provider=self.provider,
            model=self.model,
            request="saturation",
            reason="A perfect gas has no phase-equilibrium states.",
            supported=tuple(sorted(self.capabilities.phase_operations)),
        )

    @checked_perfect_gas_jit
    def get_state(
        self,
        input_type,
        prop_1,
        prop_2,
        *,
        generalize_quality=False,
        supersaturation=False,
        errors="raise",
    ):
        """Evaluate with dynamic numerical inputs and a static pair selector.

        The model is already a numerical PyTree with static identity metadata,
        so native JIT avoids filtered-method binding and per-value compilation.
        """
        return evaluate_state(self, input_type, prop_1, prop_2, errors=errors)


# ----------------------------------------------------------------------------- #
# Gradient calculations (only used when JAX is installed)
# ----------------------------------------------------------------------------- #


def get_props_gradient(
    fluid, input_pair, x, y, method="auto", eps_rel=1e-6, eps_abs=1e-6, *, errors="mask"
):
    """Return typed property partials with units and a separate primal state."""
    from ..property_derivatives import get_props_gradient as evaluate_derivatives

    return evaluate_derivatives(
        fluid,
        input_pair,
        x,
        y,
        method=method,
        eps_rel=eps_rel,
        eps_abs=eps_abs,
        errors=errors,
    )
