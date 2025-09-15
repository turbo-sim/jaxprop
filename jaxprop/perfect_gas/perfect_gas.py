import jax
import jax.numpy as jnp
import equinox as eqx

from ..coolprop import Fluid
from .. import helpers_props as jxp

# ----------------------------------------------------------------------------- #
# Constant generation and Sutherland estimation
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


def get_constants(fluid_name, T_ref, p_ref, dT=100.0):
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
    state = fluid.get_state(jxp.PT_INPUTS, p_ref, T_ref)
    R = jnp.asarray(jxp.GAS_CONSTANT / fluid.abstract_state.molar_mass())
    cp = jnp.asarray(state["cp"])
    cv = jnp.asarray(state["cv"])
    gamma = cp / cv
    s_ref = jnp.asarray(0.0)
    mu_ref = jnp.asarray(state["mu"])
    k_ref = jnp.asarray(state["k"])

    # Estimate Sutherland constants from values at T_ref ± dT
    T2 = T_ref + dT
    state2 = fluid.get_state(jxp.PT_INPUTS, p_ref, T2)
    S_mu = estimate_sutherland_constant(T_ref, mu_ref, T2, state2["mu"])
    S_k = estimate_sutherland_constant(T_ref, k_ref, T2, state2["k"])

    # Create object to store constants
    consts = PerfectGasConstants(
        R=R, gamma=gamma,
        T_ref=T_ref, P_ref=p_ref, s_ref=s_ref,
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

def temperature_from_Ps(p, s, constants):
    R = constants.R
    gamma = constants.gamma
    T_ref = constants.T_ref
    P_ref = constants.P_ref
    s_ref = constants.s_ref
    cp, _ = specific_heat(constants)
    exponent = ((s - s_ref) + (R * jnp.log(p / P_ref))) / cp
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

def entropy_from_PT(p, T, constants):
    R = constants.R
    T_ref = constants.T_ref
    P_ref = constants.P_ref
    s_ref = constants.s_ref
    cp, _ = specific_heat(constants)
    return s_ref + (cp * jnp.log(T / T_ref)) - (R * jnp.log(p / P_ref))

def entropy_from_rhoP(rho, p, constants):
    R = constants.R
    T = p / (rho * R)
    return entropy_from_PT(p, T, constants)


# ----------------------------------------------------------------------------- #
# Helper functions to calculate full fluid states
# ----------------------------------------------------------------------------- #
    
def calculate_properties_PT(p, T, constants):
    T = jnp.maximum(T, 0.1)
    p = jnp.maximum(p, 0.1)
    R = constants.R
    rho = p / (R * T)
    cp, _ = specific_heat(constants)
    h = cp * T
    s = entropy_from_PT(p, T, constants)
    return assemble_properties(T, p, rho, h, s, constants)

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

def calculate_properties_hP(h, p, constants):
    T = temperature_from_h(h, constants)
    R = constants.R
    rho = p / (R * T)
    s = entropy_from_PT(p, T, constants)
    return assemble_properties(T, p, rho, h, s, constants)

def calculate_properties_Ps(p, s, constants):
    T = temperature_from_Ps(p, s, constants)
    R = constants.R
    rho = p / (R * T)
    cp, _ = specific_heat(constants)
    h = cp * T
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
    h = cp * T
    s = entropy_from_rhoP(rho, p, constants)
    return assemble_properties(T, p, rho, h, s, constants)

def assemble_properties(T, p, rho, h, s, constants):
    R = constants.R
    gamma = constants.gamma
    return {
        "temperature": T,
        "pressure": p,
        "density": rho,
        "enthalpy": h,
        "entropy": s,
        "viscosity": viscosity_from_T(T, constants),
        "conductivity": conductivity_from_T(T, constants),
        "speed_of_sound": speed_of_sound_from_T(T, constants),
        "heat_capacity_ratio": gamma,
        "isobaric_heat_capacity": gamma * R / (gamma - 1),
        "isochoric_heat_capacity": R / (gamma - 1),
        "compressibility_factor": jnp.asarray(1.0, dtype=jnp.float64),
        "gruneisen": gamma - 1,
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


class FluidPerfectGas(eqx.Module):
    constants: PerfectGasConstants = eqx.field(static=False)
    fluid_name: str = eqx.field(static=True, default=None)
    identifier: str = eqx.field(static=True, default=None)

    def __init__(self, name, T_ref=300.0, p_ref=101_325.0, identifier=None):
        self.constants = get_constants(name, T_ref, p_ref)
        self.fluid_name = name
        self.identifier = identifier

    @eqx.filter_jit
    def get_props(self, input_pair: str, x: float, y: float):
        """Evaluate thermodynamic state for a perfect gas."""
        props = PROPERTY_CALCULATORS[input_pair](x, y, self.constants)
        return jxp.FluidState(
            fluid_name=self.fluid_name,
            identifier=self.identifier,
            **props,
        )
    
# ----------------------------------------------------------------------------- #
# Gradient calculations (only used when JAX is installed)
# ----------------------------------------------------------------------------- #

def get_props_gradient(fluid, input_pair, x, y, method="auto", eps_rel=1e-6, eps_abs=1e-6):
    """
    Return dict of gradients for perfect_gas_props at (x, y).
    Each entry: grads[prop] = jnp.array([∂prop/∂x, ∂prop/∂y])
    """
    def f(vec):
        return fluid.get_props(input_pair, vec[0], vec[1])

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
    for name in fx_p.to_dict().keys():
        if name in ("fluid_name", "identifier"):
            continue
        grads[name] = jnp.array([
            (fx_p[name] - fx_m[name]) / (2.0 * dx),
            (fy_p[name] - fy_m[name]) / (2.0 * dy),
        ])

    return jxp.FluidState(
        fluid_name=fluid.fluid_name,
        identifier=fluid.identifier,
        **grads,
    )

