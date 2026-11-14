import jaxprop as jxp

if __name__ == "__main__":

    # Define mixture properties
    p = 5e5  # Pa
    T = 350.0  # K
    R = 50.0  # mass ratio
    fluid_1 = jxp.Fluid("water", method="coolprop_jax")
    fluid_2 = jxp.Fluid("nitrogen", method="coolprop_jax")

    # Compute mixture properties
    fluid = jxp.Fluid(
        "water_nitrogen",
        method="binary_mixture",
        model_options={
            "components": (fluid_1, fluid_2),
            "mass_fractions": (R / (1 + R), 1 / (1 + R)),
        },
    )
    mix = fluid.get_state(jxp.PT_INPUTS, p, T)

    print(mix.speed_of_sound_p)
