import jax
import jax.numpy as jnp
import jaxprop as jxp

# Threshold for relative error
THRESHOLD = 1e-5
errors = []


# Helper
def rel_err(name, approx, ref):
    error = jnp.abs((approx - ref) / ref) if ref != 0 else jnp.nan
    errors.append((name, float(error)))
    return error


# Compute properties
test_h = 1000e3  # J/kg
test_P = 12e6  # Pa
fluid = jxp.Fluid(method="coolprop_jax", name="CO2", backend="HEOS")
st = fluid.get_state(jxp.HmassP_INPUTS, test_h, test_P)

# Header
print(f"{'Property':<35} {'jax':>15} {'ref':>15} {'rel err':>15}")

# Traced CoolProp callbacks use mask mode and expose failures through state.status.
# Isobaric_heat_capacity: c_p = (∂h/∂T)_p
name = "Isobaric_heat_capacity"
f1 = jax.grad(lambda T: fluid.get_state(jxp.PT_INPUTS, st.p, T, errors="mask").h)(st.T)
f2 = st.isobaric_heat_capacity
err = rel_err(name, f1, f2)
print(f"{name:<35} {f1:15.6e} {f2:15.6e} {err:15.6e}")

# Isochoric_heat_capacity: c_v = (∂e/∂T)_rho
name = "Isochoric_heat_capacity"
f1 = jax.grad(lambda T: fluid.get_state(jxp.DmassT_INPUTS, st.d, T, errors="mask").e)(st.T)
f2 = st.isochoric_heat_capacity
err = rel_err(name, f1, f2)
print(f"{name:<35} {f1:15.6e} {f2:15.6e} {err:15.6e}")

# Isothermal_compressibility: κ_T = (1/ρ)(∂ρ/∂p)_T
name = "Isothermal_compressibility"
f1 = 1 / st.rho * jax.grad(lambda p: fluid.get_state(jxp.PT_INPUTS, p, st.T, errors="mask").rho)(st.p)
f2 = st.isothermal_compressibility
err = rel_err(name, f1, f2)
print(f"{name:<35} {f1:15.6e} {f2:15.6e} {err:15.6e}")

# Isobaric_expansion_coefficient: α_p = -(1/ρ)(∂ρ/∂T)_p
name = "Isobaric_expansion_coefficient"
f1 = -1 / st.rho * jax.grad(lambda T: fluid.get_state(jxp.PT_INPUTS, st.p, T, errors="mask").rho)(st.T)
f2 = st.isobaric_expansion_coefficient
err = rel_err(name, f1, f2)
print(f"{name:<35} {f1:15.6e} {f2:15.6e} {err:15.6e}")

# Speed_of_sound: a² = (∂p/∂ρ)_s
name = "Speed_of_sound"
f1 = jnp.sqrt(
    jax.grad(lambda d: fluid.get_state(jxp.DmassSmass_INPUTS, d, st.s, errors="mask").p)(st.d)
)
f2 = st.speed_of_sound
err = rel_err(name, f1, f2)
print(f"{name:<35} {f1:15.6e} {f2:15.6e} {err:15.6e}")

# Gruneisen: Γ = (1/ρ)(∂p/∂e)_ρ
name = "Gruneisen"
f1 = (1 / st.rho) * jax.grad(
    lambda e: fluid.get_state(jxp.DmassUmass_INPUTS, st.d, e, errors="mask").p
)(st.u)
f2 = st.gruneisen
err = rel_err(name, f1, f2)
print(f"{name:<35} {f1:15.6e} {f2:15.6e} {err:15.6e}")

# Joule_thomson: μ_JT = (∂T/∂p)_h
name = "Joule_thomson"
f1 = jax.grad(lambda p: fluid.get_state(jxp.HmassP_INPUTS, st.h, p, errors="mask").T)(st.p)
f2 = st.joule_thomson
err = rel_err(name, f1, f2)
print(f"{name:<35} {f1:15.6e} {f2:15.6e} {err:15.6e}")

# Isothermal_joule_thomson: μ_T = (∂h/∂p)_T
name = "Isothermal_joule_thomson"
f1 = jax.grad(lambda p: fluid.get_state(jxp.PT_INPUTS, p, st.T, errors="mask").h)(st.p)
f2 = st.isothermal_joule_thomson
err = rel_err(name, f1, f2)
print(f"{name:<35} {f1:15.6e} {f2:15.6e} {err:15.6e}")

# Raise if any errors exceed threshold
violations = [(n, e) for n, e in errors if jnp.isnan(e) or abs(e) > THRESHOLD]
if violations:
    msg_lines = ["The following properties exceed the relative error threshold:"]
    for n, e in violations:
        msg_lines.append(f"  - {n}: rel err = {e:.3e}")
    raise ValueError("\n".join(msg_lines))
else:
    print(f"All property checks passed (relative errors < {THRESHOLD:.1e}).")
