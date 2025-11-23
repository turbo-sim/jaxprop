import jax.numpy as jnp
from .. import helpers_props as jxp


# ------------------------------------------------------------------------------------ #
# Compute the properties of a two-component mixture (e.g., water and nitrogen)
# ------------------------------------------------------------------------------------ #
def get_mixture_state(fluid_1, fluid_2, p, T, R) -> jxp.MixtureState:
    state_1 = fluid_1.get_state(jxp.PT_INPUTS, p, T)
    state_2 = fluid_2.get_state(jxp.PT_INPUTS, p, T)
    return _mix_from_states(state_1, state_2, R)


def _mix_from_states(
    state_1: jxp.FluidState, state_2: jxp.FluidState, R
) -> jxp.MixtureState:

    # --- mass-averaged thermodynamic properties
    y_1 = R / (1.0 + R)
    y_2 = 1.0 - y_1
    cv = y_1 * state_1.isochoric_heat_capacity + y_2 * state_2.isochoric_heat_capacity
    cp = y_1 * state_1.isobaric_heat_capacity + y_2 * state_2.isobaric_heat_capacity
    umass = y_1 * state_1.internal_energy + y_2 * state_2.internal_energy
    hmass = y_1 * state_1.enthalpy + y_2 * state_2.enthalpy
    smass = y_1 * state_1.entropy + y_2 * state_2.entropy

    # --- Joule-Thomson coefficients
    # k_P_1 = state_1.isobaric_expansion_coefficient
    # k_P_2 = state_2.isobaric_expansion_coefficient
    # dhdp_T_1 = (1 - state_1.temperature * k_P_1) / state_1.density
    # dhdp_T_2 = (1 - state_2.temperature * k_P_2) / state_2.density
    # dhdp_T = y_1 * dhdp_T_1 + y_2 * dhdp_T_2
    isothermal_joule_thomson = y_1 * state_1.isothermal_joule_thomson + y_2 * state_2.isothermal_joule_thomson
    joule_thomson = - isothermal_joule_thomson / cp

    # --- mixture density (harmonic mixture law)
    rho = 1.0 / (y_1 / state_1.density + y_2 / state_2.density)

    # --- volume fractions
    vol_1 = y_1 * rho / state_1.density
    vol_2 = y_2 * rho / state_2.density

    # --- simple quality identifiers
    # (use same criterion you had)
    quality_mass = y_1 if state_1.density < state_2.density else y_2
    quality_volume = vol_1 if state_1.density < state_2.density else vol_2

    # --- isothermal compressibility
    k_T_1 = state_1.isothermal_compressibility
    k_T_2 = state_2.isothermal_compressibility
    isothermal_compressibility = vol_1 * k_T_1 + vol_2 * k_T_2
    isothermal_bulk_modulus = 1.0 / isothermal_compressibility

    # --- compute the speed of sound and related properties
    rho_1 = state_1.density
    rho_2 = state_2.density
    a_1   = state_1.speed_of_sound
    a_2   = state_2.speed_of_sound
    Gamma_1 = state_1.gruneisen
    Gamma_2 = state_2.gruneisen
    cp_1 = state_1.isobaric_heat_capacity
    cp_2 = state_2.isobaric_heat_capacity
    C_1 = rho_1 * vol_1 * cp_1
    C_2 = rho_2 * vol_2 * cp_2
    inv_a2_p = rho * (
        vol_1 / (rho_1 * a_1**2) +
        vol_2 / (rho_2 * a_2**2)
    )
    Z_pT = (
        rho * state_1.temperature *
        (C_1 * C_2) / (C_1 + C_2) *
        (Gamma_1/(rho_1 * a_1**2) - Gamma_2/(rho_2 * a_2**2))**2
    )
    inv_a2_pT = inv_a2_p + Z_pT
    speed_sound_p = jnp.sqrt(1.0 / inv_a2_p)
    speed_sound_pT = jnp.sqrt(1.0 / inv_a2_pT)
    speed_sound = speed_sound_pT
    isentropic_bulk_modulus = rho * speed_sound**2
    isentropic_compressibility = 1.0 / isentropic_bulk_modulus

    # --- transport properties (volume-weighted)
    conductivity = vol_1 * state_1.conductivity + vol_2 * state_2.conductivity
    viscosity = vol_1 * state_1.viscosity + vol_2 * state_2.viscosity

    # --- build the mixture state
    mix = jxp.MixtureState(
        fluid_name=f"{state_1.fluid_name}_{state_2.fluid_name}",
        identifier=f"{state_1.fluid_name}_{state_2.fluid_name}",
        component_1=state_1,
        component_2=state_2,
        # composition
        mass_fraction_1=jnp.asarray(y_1),
        mass_fraction_2=jnp.asarray(y_2),
        volume_fraction_1=jnp.asarray(vol_1),
        volume_fraction_2=jnp.asarray(vol_2),
        # mixture thermodynamic state
        pressure=state_1.pressure,
        temperature=state_1.temperature,
        density=jnp.asarray(rho),
        enthalpy=jnp.asarray(hmass),
        internal_energy=jnp.asarray(umass),
        entropy=jnp.asarray(smass),
        isochoric_heat_capacity=jnp.asarray(cv),
        isobaric_heat_capacity=jnp.asarray(cp),
        compressibility_factor=jnp.nan,
        # compressibilities
        isothermal_compressibility=jnp.asarray(isothermal_compressibility),
        isothermal_bulk_modulus=jnp.asarray(isothermal_bulk_modulus),
        isentropic_compressibility=jnp.asarray(isentropic_compressibility),
        isentropic_bulk_modulus=jnp.asarray(isentropic_bulk_modulus),
        # wave propagation
        speed_of_sound=jnp.asarray(speed_sound),
        speed_sound_p=jnp.asarray(speed_sound_p),
        speed_sound_pT=jnp.asarray(speed_sound_pT),
        # transport
        conductivity=jnp.asarray(conductivity),
        viscosity=jnp.asarray(viscosity),
        # two-phase markers
        quality_mass=jnp.asarray(quality_mass),
        quality_volume=jnp.asarray(quality_volume),
        # auxiliary derivative
        joule_thomson=jnp.asarray(joule_thomson),
        isothermal_joule_thomson=jnp.asarray(isothermal_joule_thomson),
    )

    return mix
