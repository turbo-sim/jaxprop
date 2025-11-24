
import jax
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt

jxp.set_plot_options()

if __name__ == "__main__":

    # define conditions
    p = 5e5      # Pa
    T = 350.0    # K
    R = jnp.linspace(50, 500, 51)

    # instantiate mixture
    fluidMix = jxp.FluidTwoComponent("water", "nitrogen")

    # compute mixture state for each R (vectorized)
    state = fluidMix.get_state(jxp.PT_INPUTS, p, T, R)

    # extract acoustic speeds
    speed_sound_pT = state.speed_of_sound_pT
    speed_sound_p  = state.speed_of_sound_p

    # Compute speed of sound via automatic differentiation
    def p_of_rho_s(rho, s, R, fluidMix):
        state = fluidMix.get_state(jxp.DmassSmass_INPUTS, rho, s, R)
        return state.p

    def speed_of_sound_AD(d0, s0, R, fluidMix):
        # differentiate pressure w.r.t. density only
        dp_drho = jax.jacfwd(lambda rho: p_of_rho_s(rho, s0, R, fluidMix))(d0)
        return jnp.sqrt(dp_drho)

    speed_sound_AD = jax.vmap(lambda rho, s, R: speed_of_sound_AD(rho, s, R, fluidMix))(
        state.density, state.entropy, R
    )

    # Plot the results
    plt.figure()
    plt.plot(R, speed_sound_pT, label="$pT$-equilibrium")
    plt.plot(R, speed_sound_p, label="$p$-equilibrium")
    plt.plot(R, speed_sound_AD, "ko", label="AD-based")
    plt.xlabel("Mixture ratio $R$")
    plt.ylabel("Speed of sound (m/s)")
    plt.legend()
    plt.tight_layout(pad=1.0)
    plt.show()

    # From the results of this plot, we can conclude that the pT-equilibrium
    # speed of sound matches the one computed via automatic differentiation, confirming
    # the correctness of the mathematical derivation and numerical implementation.

