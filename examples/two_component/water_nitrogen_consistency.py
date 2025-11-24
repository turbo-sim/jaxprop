import jax
import jax.numpy as jnp
import jaxprop as jxp


# Valid input pairs for consistency checks
# (PT is used as the reference and not repeated)
INPUT_PAIRS = [
    jxp.HmassP_INPUTS,
    jxp.DmassP_INPUTS,
    jxp.HmassSmass_INPUTS,
    jxp.DmassHmass_INPUTS,
    jxp.DmassT_INPUTS,
    jxp.DmassSmass_INPUTS,
    jxp.PSmass_INPUTS,
]

# Pretty printing util
def print_table(title, ref_state, test_states, labels, tol=1e-6):

    # Properties to include
    keys = [
        "mass_fraction_1", "mass_fraction_2",
        "volume_fraction_1", "volume_fraction_2",
        "pressure", "temperature", "density",
        "enthalpy", "internal_energy", "entropy",
        "isochoric_heat_capacity", "isobaric_heat_capacity",
        "compressibility_factor",
        "speed_of_sound", "speed_of_sound_p", "speed_of_sound_pT",
        "isothermal_compressibility", "isentropic_compressibility",
        "isothermal_bulk_modulus", "isentropic_bulk_modulus",
        "viscosity", "conductivity",
        "vapor_quality", "void_fraction",
        "joule_thomson", "isothermal_joule_thomson"
    ]

    print("\n" + "="*120)
    print(title)
    print("="*120)

    # Header row
    header = f"{'Property':28s}" + "".join([f"{lbl:>12s}" for lbl in labels])
    print(header)
    print("-"*120)

    # For each property, compute and print row of errors
    for key in keys:
        ref_val = getattr(ref_state, key)

        row = f"{key:28s}"
        for st in test_states:
            test_val = getattr(st, key)

            # Compute absolute error
            err = jnp.abs(ref_val - test_val)

            # NaN-safe: if both are NaN → treat as OK
            if jnp.isnan(ref_val) and jnp.isnan(test_val):
                entry = "OK"
            else:
                entry = "OK" if err < tol else f"{err:.2e}"

            row += f"{entry:>12s}"

        print(row)

    print("-"*120)
    print("\n")


# -------------------------------------------------------
# Main consistency test
# -------------------------------------------------------
if __name__ == "__main__":

    # Ambient condition (reference)
    p0 = 1.25e5      # Pa
    T0 = 310.0    # K

    # Mixture ratios to test
    R_vals = jnp.array([100., 200., 300.])

    # Mixture object
    fluidMix = jxp.FluidTwoComponent("water", "nitrogen")

    # Loop through mixture ratios
    for R in R_vals:

        # ---------------------------------------------------------------
        # 1) Reference state from PT_INPUTS
        # ---------------------------------------------------------------
        ref_state = fluidMix.get_state(jxp.PT_INPUTS, p0, T0, R)

        # Values needed to reconstruct via other input types
        p_ref = ref_state.p
        T_ref = ref_state.T
        d_ref = ref_state.rho
        h_ref = ref_state.h
        s_ref = ref_state.s

        # ---------------------------------------------------------------
        # 2) Compute state for each input type
        # ---------------------------------------------------------------
        test_states = []

        for ipair in INPUT_PAIRS:
            if ipair == jxp.HmassP_INPUTS:
                st = fluidMix.get_state(ipair, h_ref, p_ref, R)

            elif ipair == jxp.DmassP_INPUTS:
                st = fluidMix.get_state(ipair, d_ref, p_ref, R)

            elif ipair == jxp.HmassSmass_INPUTS:
                st = fluidMix.get_state(ipair, h_ref, s_ref, R)

            elif ipair == jxp.DmassHmass_INPUTS:
                st = fluidMix.get_state(ipair, d_ref, h_ref, R)

            elif ipair == jxp.DmassT_INPUTS:
                st = fluidMix.get_state(ipair, d_ref, T_ref, R)

            elif ipair == jxp.DmassSmass_INPUTS:
                st = fluidMix.get_state(ipair, d_ref, s_ref, R)

            elif ipair == jxp.PSmass_INPUTS:
                st = fluidMix.get_state(ipair, p_ref, s_ref, R)

            test_states.append(st)

        # ---------------------------------------------------------------
        # 3) Print consistency table
        # ---------------------------------------------------------------
        labels = [
            "H-P",
            "rho-P",
            "H-s",
            "rho-h",
            "rho-T",
            "rho-s",
            "P-s",
        ]
        print_table(f"Consistency check table at R = {R:.1f}",
                    ref_state, test_states, labels)
