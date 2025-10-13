import jax
import jax.numpy as jnp
import jaxprop as jxp


def rel_err(approx, ref):
    return (approx - ref) / ref if ref != 0 else jnp.nan


# Compute properties
test_h = 1000e3  # J/kg
test_P = 12e6  # Pa
fluid = jxp.FluidJAX(name="CO2", backend="HEOS")
st = fluid.get_state(jxp.HmassP_INPUTS, test_h, test_P)


# header
print(f"{'Property':<35} {'jax':>15} {'ref':>15} {'rel err':>15}")

# Isobaric_heat_capacity: c_p = (∂h/∂T)_p
dh_dT_p = jax.grad(lambda T: fluid.get_state(jxp.PT_INPUTS, st.p, T).h)(st.T)
f1 = dh_dT_p
f2 = st.isobaric_heat_capacity
rel_error = rel_err(f1, f2)
print(f"{'Isobaric_heat_capacity':<35} {f1:15.6e} {f2:15.6e} {rel_error:15.6e}")

# Isochoric_heat_capacity: c_v = (∂e/∂T)_rho
de_dT_rho = jax.grad(lambda T: fluid.get_state(jxp.DmassT_INPUTS, st.d, T).e)(st.T)
f1 = de_dT_rho
f2 = st.isochoric_heat_capacity
rel_error = rel_err(f1, f2)
print(f"{'Isochoric_heat_capacity':<35} {f1:15.6e} {f2:15.6e} {rel_error:15.6e}")

# Isothermal_compressibility: κ_T = (1/ρ)(∂ρ/∂p)_T
drho_dp_T = jax.grad(lambda p: fluid.get_state(jxp.PT_INPUTS, p, st.T).rho)(st.p)
f1 = 1 / st.rho * drho_dp_T
f2 = st.isothermal_compressibility
rel_error = rel_err(f1, f2)
print(f"{'Isothermal_compressibility':<35} {f1:15.6e} {f2:15.6e} {rel_error:15.6e}")

# Isobaric_expansion_coefficient: α_p = -(1/ρ)(∂ρ/∂T)_p
drho_dT_p = jax.grad(lambda T: fluid.get_state(jxp.PT_INPUTS, st.p, T).rho)(st.T)
f1 = -1 / st.rho * drho_dT_p
f2 = st.isobaric_expansion_coefficient
rel_error = rel_err(f1, f2)
print(f"{'Isobaric_expansion_coefficient':<35} {f1:15.6e} {f2:15.6e} {rel_error:15.6e}")

# Speed_of_sound: a² = (∂p/∂ρ)_s
dp_drho_s = jax.grad(lambda d: fluid.get_state(jxp.DmassSmass_INPUTS, d, st.s).p)(st.d)
f1 = jnp.sqrt(dp_drho_s)
f2 = st.speed_of_sound
rel_error = rel_err(f1, f2)
print(f"{'Speed_of_sound':<35} {f1:15.6e} {f2:15.6e} {rel_error:15.6e}")

# Gruneisen: Γ = (1/ρ)(∂p/∂e)_ρ
dp_de_rho = jax.grad(lambda e: fluid.get_state(jxp.DmassUmass_INPUTS, st.d, e).p)(st.u)
f1 = (1 / st.rho) * dp_de_rho
f2 = st.gruneisen
rel_error = rel_err(f1, f2)
print(f"{'Gruneisen':<35} {f1:15.6e} {f2:15.6e} {rel_error:15.6e}")

# Joule_thomson: μ_JT = (∂T/∂p)_h
dT_dp_h = jax.grad(lambda p: fluid.get_state(jxp.HmassP_INPUTS, st.h, p).T)(st.p)
f1 = dT_dp_h
f2 = st.joule_thomson
rel_error = rel_err(f1, f2)
print(f"{'Joule_thomson':<35} {f1:15.6e} {f2:15.6e} {rel_error:15.6e}")

# Isothermal_joule_thomson: μ_T = (∂h/∂p)_T
dh_dp_T = jax.grad(lambda p: fluid.get_state(jxp.PT_INPUTS, p, st.T).h)(st.p)
f1 = dh_dp_T
f2 = st.isothermal_joule_thomson
rel_error = rel_err(f1, f2)
print(f"{'Isothermal_joule_thomson':<35} {f1:15.6e} {f2:15.6e} {rel_error:15.6e}")



# Check that the metastable solver gives the same values
state_meta = fluid.fluid.get_state_metastable(
    prop_1="T",
    prop_2="rho",
    prop_1_value=st.T,
    prop_2_value=st.d,
    print_convergence=True,
    generalize_quality=False,
)

for key, val_st in st.items():
    val_meta = getattr(state_meta, key)
    if isinstance(val_st, (float, int, jnp.ndarray)):
        diff = (val_st - val_meta) / val_st
        print(f"{key:<30} st={val_st:15.6e} meta={val_meta:15.6e} diff={diff:15.6e}")
