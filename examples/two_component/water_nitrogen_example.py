
import jaxprop as jxp

if __name__ == "__main__":

    # Define mixture properties
    p = 5e5  # Pa
    T = 350.0  # K  
    R = 50.0  # mass ratio
    fluid_1 = jxp.FluidJAX("water")
    fluid_2 = jxp.FluidJAX("nitrogen")

    # Compute mixture properties
    mix = jxp.get_mixture_state(fluid_1, fluid_2, p=p, T=T, R=R)


    print(mix.speed_sound_p)
