import jax.numpy as jnp
from .. import helpers_props as jxp




import os
import time
import tqdm
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import jaxprop.coolprop as jxp


# ================================================================
# FluidTwoComponent class
# ================================================================
class FluidTwoComponent(eqx.Module):
    r"""
    Fluid model for a two-component mixture (i.e., water and nitrogen)

    
    Parameters
    ----------
    fluid_name_1, fluid_name_2 : str
        Fluid identifiers for CoolProp.
    backend_1, backend_2 : str
        CoolProp backend strings.
    identifier : str, optional
        Tag stored in the returned FluidState objects.

    """

    # Attributes
    fluid_1: jxp.FluidJAX = eqx.field(static=True)
    fluid_2: jxp.FluidJAX = eqx.field(static=True)
    fluid_name_1: str = eqx.field(static=True)
    fluid_name_2: str = eqx.field(static=True)
    backend_1: str = eqx.field(static=True)
    backend_2: str = eqx.field(static=True)
    identifier: str = eqx.field(static=True)

    def __init__(
        self,
        fluid_name_1: str,
        fluid_name_2: str,
        backend_1: str = "HEOS",
        backend_2: str = "HEOS",
        identifier: str = None,
    ):
        # Initialize variables
        self.fluid_name_1 = fluid_name_1
        self.fluid_name_2 = fluid_name_2
        self.backend_1 = backend_1
        self.backend_2 = backend_2
        self.identifier = identifier or fluid_name_1 + "_" + fluid_name_2

        # Initialize fluid components
        self.fluid_1 = jxp.FluidJAX(name=fluid_name_1, backend=backend_1)
        self.fluid_2 = jxp.FluidJAX(name=fluid_name_2, backend=backend_2)



    @eqx.filter_jit
    def get_state(self, input_type, val1, val2, R):
        """
        Compute thermodynamic states from any supported input pair on a vectorized way.

        This is the main user-facing entry point. It accepts either scalar or array
        inputs for `val1` and `val2`, automatically broadcasting them to a common shape
        and evaluating the corresponding fluid state at each point in a vectorized
        manner. Internally, the metho a Newton solver in normalized `(p, T)` space 
        to recover the unique thermodynamic state that matches both input properties simultaneously.

        All computations are compatible with JAX transformations such as `jit`, `vmap`,
        and automatic differentiation. Vectorization is handled internally using
        `vmap` over the scalar solver/interpolator. The output is a `MixtureState`
        object containing all thermodynamic and transport properties defined in the
        table, either for a single point (scalar inputs) or for each element of the
        broadcasted input arrays.

        Parameters
        ----------
        input_type : int
            Identifier of the input pair (e.g. `jxp.HmassP_INPUTS`, `jxp.PT_INPUTS`).
        val1 : float or array_like
            First input variable (e.g. enthalpy, pressure, density).
        val2 : float or array_like
            Second input variable (e.g. pressure, temperature, entropy).
        R : float or array_like
            Mixture ratio (mass of fluid 1 / mass of fluid 2).
        Returns
        -------
        MixtureState
            Interpolated fluid state(s) corresponding to the specified input pair.
            If inputs are arrays, each property is returned as an array with the
            broadcasted shape of `val1` and `val2`.
        """
        # Broadcast h and p to the same shape
        val1 = jnp.asarray(val1)
        val2 = jnp.asarray(val2)
        R = jnp.asarray(R)
        val1, val2, R = jnp.broadcast_arrays(val1, val2, R)

        # Define vectorized mapping explicitly using jax.vmap
        batched_fn = jax.vmap(lambda v1, v2, r: self._get_state_scalar(input_type, v1, v2, r))

        # Apply to flattened arrays
        props_batched = batched_fn(val1.ravel(), val2.ravel(), R.ravel())

        # Reshape each leaf in the pytree to the broadcasted shape
        props = jax.tree.map(lambda x: x.reshape(val1.shape), props_batched)

        return props


    @eqx.filter_jit
    def _get_state_scalar(self, input_pair, x, y, R, p_guess=101325., T_guess=300.0, tol=1e-10):
        r"""
        Solve for (p, T) corresponding to a given input pair (x, y) by inverting
        the the fluid property calculations

        This function finds the pressure and temperature that simultaneously match
        two target property values (e.g. density-temperature, pressure-entropy) using
        a Newton root-finding algorithm. The solver uses initial guesses for pressure
        and temperature provided by `p_guess` and `T_guess`.

        Parameters
        ----------
        input_pair : int
            Identifier for the input property pair (e.g. jxp.HmassSmass_INPUTS).
        x : float
            Target value of the first property.
        y : float
            Target value of the second property.
        R : float
            Mixture ratio (mass of fluid 1 / mass of fluid 2).
        p_guess : float
            Initial guess for the pressure.
        T_guess : float
            Initial guess for the temperature.  
        tol : float, optional
            Absolute and relative tolerance for the Newton solver (default: 1e-8).

        Returns
        -------
        MixtureState
            Interpolated fluid state corresponding to the recovered (p, T).

        """
        # Map input_pair to property names
        prop1, prop2 = jxp.INPUT_PAIR_MAP[input_pair]
        prop1 = jxp.ALIAS_TO_CANONICAL[prop1]
        prop2 = jxp.ALIAS_TO_CANONICAL[prop2]

        # Define initial guess
    
        state_guess = get_mixture_state(self.fluid_1, self.fluid_2, p_guess, T_guess, R)
        x_ref = state_guess[prop1]
        y_ref = state_guess[prop2]
        x0 = jnp.array([1.0, 1.0])

        # Define residual function
        def residual(xy, _):
            p_nd, T_nd = xy
            p = p_nd * p_guess
            T = T_nd * T_guess
            mix_state = get_mixture_state(self.fluid_1, self.fluid_2, p, T, R)
            res_x = (mix_state[prop1] - x) / x_ref
            res_y = (mix_state[prop2] - y) / y_ref
            # # debug print
            # jax.debug.print(
            #     """
            #     residual eval:
            #         p_nd = {p_nd}
            #         T_nd = {T_nd}
            #         p    = {p}
            #         T    = {T}
            #         mix_state[{prop1}] = {v1}
            #         mix_state[{prop2}] = {v2}
            #         res_x = {rx}
            #         res_y = {ry}
            #     """,
            #     p_nd=p_nd, T_nd=T_nd,
            #     p=p, T=T,
            #     prop1=prop1, prop2=prop2,
            #     v1=mix_state[prop1], v2=mix_state[prop2],
            #     rx=res_x, ry=res_y
            # )
            return jnp.array([res_x, res_y])

        # Solve root-finding problem
        solver = optx.Newton(rtol=tol, atol=tol)
        solution = optx.root_find(residual, solver, x0, throw=True)

        # convert back to dimensional variables
        p = solution.value[0] * p_guess
        T = solution.value[1] * T_guess

        return get_mixture_state(self.fluid_1, self.fluid_2, p, T, R)

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
    isothermal_joule_thomson = y_1 * state_1.isothermal_joule_thomson + y_2 * state_2.isothermal_joule_thomson
    joule_thomson = - isothermal_joule_thomson / cp

    # --- mixture density (harmonic mixture law)
    rho = 1.0 / (y_1 / state_1.density + y_2 / state_2.density)

    # --- volume fractions
    vol_1 = y_1 * rho / state_1.density
    vol_2 = y_2 * rho / state_2.density

    # --- simple quality identifiers
    # vapor_quality = y_1 if state_1.density < state_2.density else y_2
    # void_fraction = vol_1 if state_1.density < state_2.density else vol_2
    vapor_quality = jnp.where(state_1.density < state_2.density, y_1, y_2)
    void_fraction  = jnp.where(state_1.density < state_2.density, vol_1, vol_2)

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
    C_1 = vol_1 * rho_1 * cp_1
    C_2 = vol_2 * rho_2 * cp_2
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
    speed_of_sound_p = jnp.sqrt(1.0 / inv_a2_p)
    speed_of_sound_pT = jnp.sqrt(1.0 / inv_a2_pT)
    speed_of_sound = speed_of_sound_pT
    isentropic_bulk_modulus = rho * speed_of_sound**2
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
        speed_of_sound=jnp.asarray(speed_of_sound),
        speed_of_sound_p=jnp.asarray(speed_of_sound_p),
        speed_of_sound_pT=jnp.asarray(speed_of_sound_pT),
        # transport
        conductivity=jnp.asarray(conductivity),
        viscosity=jnp.asarray(viscosity),
        # two-phase markers
        vapor_quality=jnp.asarray(vapor_quality),
        void_fraction=jnp.asarray(void_fraction),
        # auxiliary derivative
        joule_thomson=jnp.asarray(joule_thomson),
        isothermal_joule_thomson=jnp.asarray(isothermal_joule_thomson),
    )

    return mix
