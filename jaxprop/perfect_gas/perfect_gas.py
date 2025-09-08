import dataclasses
import jax
import jax.numpy as jnp
import equinox as eqx

from .. import helpers_coolprop as cpx

from ..coolpropx import Fluid

# Define property aliases
PROPERTY_ALIAS = {
    "P": "p",

    # density
    "rho": "rho",
    "density": "rho",
    "d": "rho",
    "rhomass": "rho",
    "dmass": "rho",

    # enthalpy
    "h": "h",
    "hmass": "h",

    # entropy
    "s": "s",
    "smass": "s",

    # internal energy (if added later)
    "u": "u",

    # heat capacities
    "cv": "cv",
    "cvmass": "cv",
    "cp": "cp",
    "cpmass": "cp",

    # speed of sound
    "a": "a",
    "speed_sound": "a",

    # compressibility factor
    "Z": "Z",
    "compressibility_factor": "Z",

    # transport properties
    "mu": "mu",
    "viscosity": "mu",
    "k": "k",
    "conductivity": "k",

    # quality placeholders
    "vapor_quality": "quality_mass",
    "void_fraction": "quality_volume",
}


# ----------------------------------------------------------------------------- #
# Equinox objects with fixed structure
# ----------------------------------------------------------------------------- #
class PerfectGasConstants(eqx.Module):
    R: jnp.ndarray
    gamma: jnp.ndarray
    T_ref: jnp.ndarray
    P_ref: jnp.ndarray
    s_ref: jnp.ndarray
    k_ref: jnp.ndarray
    mu_ref: jnp.ndarray
    S_k: jnp.ndarray
    S_mu: jnp.ndarray



class PerfectGasState(eqx.Module):
    T: jnp.ndarray
    p: jnp.ndarray
    rho: jnp.ndarray
    h: jnp.ndarray
    s: jnp.ndarray
    mu: jnp.ndarray
    k: jnp.ndarray
    a: jnp.ndarray
    gamma: jnp.ndarray
    cp: jnp.ndarray
    cv: jnp.ndarray
    Z: jnp.ndarray
    gruneisen: jnp.ndarray

    def __getitem__(self, key: str):
        """Allow dictionary-style access to state variables.
        Returns the attribute matching `key` or its alias in PROPERTY_ALIAS."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in PROPERTY_ALIAS:
            return getattr(self, PROPERTY_ALIAS[key])
        raise KeyError(f"Unknown property alias: {key}")

    def __repr__(self) -> str:
        """Return a readable string representation of the state,
        listing all field names and their scalar values."""
        lines = []
        for name, val in self.__dict__.items():
            try:
                val = jnp.array(val).item()  # scalar to Python number
            except Exception:
                pass
            lines.append(f"  {name}={val}")
        return "PerfectGasState(\n" + ",\n".join(lines) + "\n)"

    
# ----------------------------------------------------------------------------- #
# Constant generation and Sutherland estimation
# ----------------------------------------------------------------------------- #

def get_constants(fluid_name, T_ref, P_ref, dT=100.0):
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
    # Calculate fluid constants at the reference pressure and temperature
    fluid = Fluid(name=fluid_name, backend="HEOS")
    state = fluid.get_state(cpx.PT_INPUTS, P_ref, T_ref)
    R = jnp.asarray(cpx.GAS_CONSTANT / fluid.abstract_state.molar_mass())
    cp = jnp.asarray(state["cp"])
    cv = jnp.asarray(state["cv"])
    gamma = cp / cv
    s_ref = jnp.asarray(0.0)
    mu_ref = jnp.asarray(state["mu"])
    k_ref = jnp.asarray(state["k"])

    # Estimate Sutherland constants from values at T_ref ± dT
    T2 = T_ref + dT
    state2 = fluid.get_state(cpx.PT_INPUTS, P_ref, T2)
    S_mu = estimate_sutherland_constant(T_ref, mu_ref, T2, state2["mu"])
    S_k = estimate_sutherland_constant(T_ref, k_ref, T2, state2["k"])

    # Create object to store constants
    consts = PerfectGasConstants(
        R=R, gamma=gamma,
        T_ref=T_ref, P_ref=P_ref, s_ref=s_ref,
        k_ref=k_ref, mu_ref=mu_ref,
        S_k=S_k, S_mu=S_mu,
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
    return jnp.maximum(1.0, h / cp)

def temperature_from_Ps(P, s, constants):
    R = constants.R
    gamma = constants.gamma
    T_ref = constants.T_ref
    P_ref = constants.P_ref
    s_ref = constants.s_ref
    cp, _ = specific_heat(constants)
    exponent = ((s - s_ref) + (R * jnp.log(P / P_ref))) / cp
    return jnp.maximum(1.0, T_ref * jnp.exp(exponent))

def temperature_from_rhoP(rho, p, constants):
    R = constants.R
    return jnp.maximum(1.0, p / (rho * R))

def viscosity_from_T(T, constants):
    mu_ref = constants.mu_ref
    T_ref = constants.T_ref
    S_mu = constants.S_mu
    return mu_ref * ((T / T_ref) ** 1.5) * ((T_ref + S_mu) / (T + S_mu))

def conductivity_from_T(T, constants):
    k_ref = constants.k_ref
    T_ref = constants.T_ref
    S_k = constants.S_k
    return k_ref * ((T / T_ref) ** 1.5) * ((T_ref + S_k) / (T + S_k))

def speed_of_sound_from_T(T, constants):
    R = constants.R
    gamma = constants.gamma
    return jnp.sqrt(gamma * R * T)

def entropy_from_PT(P, T, constants):
    R = constants.R
    T_ref = constants.T_ref
    P_ref = constants.P_ref
    s_ref = constants.s_ref
    cp, _ = specific_heat(constants)
    return s_ref + (cp * jnp.log(T / T_ref)) - (R * jnp.log(P / P_ref))

def entropy_from_rhoP(rho, P, constants):
    R = constants.R
    T = P / (rho * R)
    return entropy_from_PT(P, T, constants)


# ----------------------------------------------------------------------------- #
# Helper functions to calculate full fluid states
# ----------------------------------------------------------------------------- #

def calculate_properties_PT(P, T, constants):
    T = jnp.maximum(T, 0.1)
    P = jnp.maximum(P, 0.1)
    R = constants.R
    rho = P / (R * T)
    cp, _ = specific_heat(constants)
    h = cp * T
    s = entropy_from_PT(P, T, constants)
    return assemble_properties(T, P, rho, h, s, constants)

def calculate_properties_hs(h, s, constants):
    cp, _ = specific_heat(constants)
    T = temperature_from_h(h, constants)
    R = constants.R
    T_ref = constants.T_ref
    P_ref = constants.P_ref
    s_ref = constants.s_ref
    P = P_ref * jnp.exp(((cp * jnp.log(T / T_ref)) - (s - s_ref)) / R)
    rho = P / (R * T)
    return assemble_properties(T, P, rho, h, s, constants)

def calculate_properties_hP(h, P, constants):
    T = temperature_from_h(h, constants)
    R = constants.R
    rho = P / (R * T)
    s = entropy_from_PT(P, T, constants)
    return assemble_properties(T, P, rho, h, s, constants)

def calculate_properties_Ps(P, s, constants):
    T = temperature_from_Ps(P, s, constants)
    R = constants.R
    rho = P / (R * T)
    cp, _ = specific_heat(constants)
    h = cp * T
    return assemble_properties(T, P, rho, h, s, constants)

def calculate_properties_rhoh(rho, h, constants):
    T = temperature_from_h(h, constants)
    R = constants.R
    P = rho * R * T
    s = entropy_from_rhoP(rho, P, constants)
    return assemble_properties(T, P, rho, h, s, constants)

def calculate_properties_rhop(rho, P, constants):
    cp, _ = specific_heat(constants)
    T = temperature_from_rhoP(rho, P, constants)
    h = cp * T
    s = entropy_from_rhoP(rho, P, constants)
    return assemble_properties(T, P, rho, h, s, constants)

def assemble_properties(T, P, rho, h, s, constants: PerfectGasConstants) -> PerfectGasState:
    R = constants.R
    gamma = constants.gamma

    return PerfectGasState(
        T=T,
        p=P,
        rho=rho,
        h=h,
        s=s,
        mu=viscosity_from_T(T, constants),
        k=conductivity_from_T(T, constants),
        a=speed_of_sound_from_T(T, constants),
        gamma=gamma,
        cp=gamma * R / (gamma - 1),
        cv=R / (gamma - 1),
        Z=jnp.asarray(1.0, dtype=jnp.float64),
        gruneisen=gamma - 1,
    )


# ----------------------------------------------------------------------------- #
# State evaluators (public API)
# ----------------------------------------------------------------------------- #

PROPERTY_CALCULATORS = {
    cpx.PT_INPUTS: calculate_properties_PT,
    cpx.HmassSmass_INPUTS: calculate_properties_hs,
    cpx.HmassP_INPUTS: calculate_properties_hP,
    cpx.PSmass_INPUTS: calculate_properties_Ps,
    cpx.DmassHmass_INPUTS: calculate_properties_rhoh,
    cpx.DmassP_INPUTS: calculate_properties_rhop,
}

@eqx.filter_jit
def get_props_perfect_gas(input_pair, prop1, prop2, constants):
    return PROPERTY_CALCULATORS[input_pair](prop1, prop2, constants)


class FluidPerfectGas(eqx.Module):
    constants: PerfectGasConstants

    def __init__(self, fluid_name: str, T_ref: float = 300.0, P_ref: float = 101325.0):
        consts = get_constants(fluid_name, T_ref, P_ref)
        object.__setattr__(self, "constants", consts)

    @eqx.filter_jit
    def get_props(self, input_pair: str, x: float, y: float):
        return get_props_perfect_gas(input_pair, x, y, self.constants)
    
    
# ----------------------------------------------------------------------------- #
# Gradient calculations (only used when JAX is installed)
# ----------------------------------------------------------------------------- #

def get_props_gradient(input_pair, constants, x, y, method="auto", eps_rel=1e-6, eps_abs=1e-6):
    """
    Return dict of gradients for perfect_gas_props at (x, y).
    Each entry: grads[prop] = jnp.array([∂prop/∂x, ∂prop/∂y])
    """
    def f(vec):
        return get_props_perfect_gas(input_pair, vec[0], vec[1], constants)

    # JAX autodiff path
    if method in ("auto", "jax"):
        return jax.jacfwd(f)(jnp.array([x, y]))

    # Finite difference path
    dx = eps_rel * jnp.maximum(jnp.abs(x), 1.0) + eps_abs
    dy = eps_rel * jnp.maximum(jnp.abs(y), 1.0) + eps_abs

    fx_p = f(jnp.array([x + dx, y]))
    fx_m = f(jnp.array([x - dx, y]))
    fy_p = f(jnp.array([x, y + dy]))
    fy_m = f(jnp.array([x, y - dy]))

    grads = {}
    for field in dataclasses.fields(fx_p):
        name = field.name
        dfdx = (getattr(fx_p, name) - getattr(fx_m, name)) / (2.0 * dx)
        dfdy = (getattr(fy_p, name) - getattr(fy_m, name)) / (2.0 * dy)
        grads[name] = jnp.array([dfdx, dfdy])


    return grads
