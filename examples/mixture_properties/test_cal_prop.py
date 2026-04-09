# # import CoolProp.CoolProp as CP
# # import jaxprop.coolprop.core_calculations as CC
# # import jaxprop as jxp
# # import jax




# # if __name__  == "__main__":

# #     as1 = CP.AbstractState("HEOS", "water")
# #     as2 = CP.AbstractState("HEOS", "nitrogen")
# #     p_mix = 5e6
# #     h_mix = 120e3
# #     R = 70
# #     # props = CC.calculate_mixture_properties_hp(h_mix, p_mix, as1, as2, R)
# #     # T_mix = props["temperature"]

# #     # # print(f"T_mix:{T_mix}")

# #     # # print(f"enthalpy error={h_mix-props["enthalpy"]}")

# #     fluidmix  = jxp.FluidMix("water_nitrogen_mixture", mixture_ratio=R)

# #     props_mix = fluidmix.get_state(jxp.HmassP_INPUTS, h_mix, p_mix)

# #     # # print(props_mix)

# #     # outdir = "mixture_tables"
# #     # fluid_name = "water_nitrogen_mixture"
# #     # h_min = 100e3  # J/kg
# #     # h_max = 180e3  # J/kg
# #     # p_min = 2e6    # Pa
# #     # p_max = 20e6   # Pa
# #     # N_h = 100
# #     # N_p = 100
# #     # mixture_ratio = 70

# #     # fluid_bicubic = jxp.FluidBicubic(
# #     #     fluid_name=fluid_name,
# #     #     backend="HEOS",
# #     #     h_min=h_min, h_max=h_max,
# #     #     p_min=p_min, p_max=p_max,
# #     #     N_h=N_h, N_p=N_p,
# #     #     mixture_ratio=mixture_ratio,
# #     #     table_dir=outdir,
# #     #     table_name=f"{fluid_name}_table_{N_h}_{N_p}_{mixture_ratio}"
# #     # )

# #     # props_bicubic_mix_hp = fluid_bicubic.get_state(jxp.HmassP_INPUTS, h_mix, p_mix)
# #     # props_bicubic_mix_pT = fluid_bicubic.get_state(jxp.PT_INPUTS, p_mix, T_mix)
# #     # # print(props_bicubic_mix)

# #     # for k, v in props.items():
# #     #     diff = props_bicubic_mix_hp[k] - props_bicubic_mix_pT[k]
# #     #     print(f"{k}: {diff:0.20e}")

# #     # print(f"mix r:{props["mixture_ratio"]}")

# #     # print(CP.PT_INPUTS)

# #     def finite_diff_derivative(func, x):
# #         dx = 1e-8 * (abs(x) + 1)
# #         return (func(x + dx) - func(x - dx)) / (2 * dx)
    

# #     @staticmethod
# #     def gradients_forward(fluid, h, p, eps_h, eps_p):
# #         f0 = fluid.get_state(jxp.HmassP_INPUTS, h, p)
# #         fh = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p)
# #         fp = fluid.get_state(jxp.HmassP_INPUTS, h, p + eps_p)
# #         fhp = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p + eps_p)

# #         grads = {}
# #         for k in jxp.PROPERTIES_CANONICAL:
# #             val = f0[k]
# #             grad_h = (fh[k] - f0[k]) / eps_h
# #             grad_p = (fp[k] - f0[k]) / eps_p
# #             grad_hp = (fhp[k] - fh[k] - fp[k] + f0[k]) / (eps_h * eps_p)
# #             grads[k] = (val, grad_h, grad_p, grad_hp)
# #         return grads

# #     def get_rho_from_hp(h, p):
# #         props = CC.calculate_mixture_properties_hp(h, p, as1, as2, R)
# #         return props["density"]

# #     # Derivatives
# #     drho_dh_p = finite_diff_derivative(lambda h: get_rho_from_hp(h, p_mix), h_mix)
# #     drho_dp_h = finite_diff_derivative(lambda p: get_rho_from_hp(h_mix, p), p_mix)

# #     # @jax.custom_vjp
# #     # def get_rho_from_hp(h, p, AS1, AS2, R):
# #     #     props = CC.calculate_mixture_properties_hp(float(h), float(p), AS1, AS2, R)
# #     #     return props["density"]

# #     # # Forward and backward rules
# #     # def get_rho_from_hp_fwd(h, p, AS1, AS2, R):
# #     #     rho = get_rho_from_hp(h, p, AS1, AS2, R)
# #     #     props = CC.calculate_mixture_properties_hp(float(h), float(p), AS1, AS2, R)
# #     #     G = props["gruneisen_parameter"]
# #     #     c = props["speed_of_sound"]
# #     #     rho = props["density"]
# #     #     return rho, (rho, G, c)

# #     # def get_rho_from_hp_bwd(res, g):
# #     #     rho, G, c = res
# #     #     # Analytical partials
# #     #     drho_dh_p = -rho * G / c**2
# #     #     drho_dp_h = (1 + G) / c**2
# #     #     # Chain rule
# #     #     return (g * drho_dh_p, g * drho_dp_h, None, None, None)

# #     # get_rho_from_hp.defvjp(get_rho_from_hp_fwd, get_rho_from_hp_bwd)

# #     # drho_dh = jax.grad(lambda h: get_rho_from_hp(h, p_mix, as1, as2, R))(h_mix)
# #     # drho_dp = jax.grad(lambda p: get_rho_from_hp(h_mix, p, AS1, AS2, R))(p_mix)
# #     props = CC.calculate_mixture_properties_hp(h_mix, p_mix, as1, as2, R)
# #     rho = props["density"]
# #     c = props["speed_of_sound"]
# #     G = props["gruneisen"]

# #     drho_dh_p_model = -rho * G / c**2
# #     drho_dp_h_model = (1 + G) / c**2

# #     print("Comparison:")
# #     print(f"(∂ρ/∂h)_p  model = {drho_dh_p_model:.5e},  jax = {drho_dh_p:.5e}")
# #     print(f"(∂ρ/∂p)_h  model = {drho_dp_h_model:.5e},  jax = {drho_dp_h:.5e}")
# #     d1 = ((drho_dh_p_model - drho_dh_p)/ drho_dh_p_model)*100
# #     d2 = ((drho_dp_h_model - drho_dp_h) / drho_dp_h_model)*100
# #     print("Difference:")
# #     print(f"(∂ρ/∂h)_p   {d1:.8e}")
# #     print(f"(∂ρ/∂p)_h   {d2:.5e}")


# import CoolProp.CoolProp as CP
# import jaxprop.coolprop.core_calculations as CC
# import jaxprop as jxp


# # -------------------------------------------------
# # Forward finite-difference gradient for fluidmix
# # -------------------------------------------------
# def gradients_forward(fluid, h, p, eps_h, eps_p):
#     f0  = fluid.get_state(jxp.HmassP_INPUTS, h, p)
#     fh  = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p)
#     fp  = fluid.get_state(jxp.HmassP_INPUTS, h, p + eps_p)

#     grads = {}
#     for k in jxp.PROPERTIES_CANONICAL:
#         grad_h = (fh[k] - f0[k]) / eps_h
#         grad_p = (fp[k] - f0[k]) / eps_p
#         grads[k] = dict(value=f0[k], dh=grad_h, dp=grad_p)

#     return grads


# # -------------------------------------------------
# # Main
# # -------------------------------------------------
# if __name__ == "__main__":

#     # CoolProp pure fluid states
#     as1 = CP.AbstractState("HEOS", "water")
#     as2 = CP.AbstractState("HEOS", "nitrogen")

#     # Mixture state values
#     p_mix = 20e5    # Pa
#     h_mix = 100e3   # J/kg
#     R     = 70      # mixture ratio (water/nitrogen)

#     # JAXProp mixture fluid
#     fluidmix = jxp.FluidMix("water_nitrogen_mixture", mixture_ratio=R)

#     # Compute forward gradients 
#     eps_h = 1e-5 * max(abs(h_mix), 1.0)
#     eps_p = 1e-5 * max(abs(p_mix), 1.0)    # Pa
#     grads = gradients_forward(fluidmix, h_mix, p_mix, eps_h, eps_p)

#     # Extract forward gradients for density
#     drho_dh_fwd = grads["density"]["dh"]
#     drho_dp_fwd = grads["density"]["dp"]



#     # Analytical model gradients using Grüneisen parameter
#     props = CC.calculate_mixture_properties_hp(h_mix, p_mix, as1, as2, R)
#     print(props)
#     rho = props["density"]
#     c   = props["speed_of_sound"]
#     G   = props["gruneisen"]
#     kappaT = props["isobaric_expansion_coefficient"]
#     cp = props["isobaric_heat_capacity"]
#     kappaP = props["isothermal_compressibility"]
#     T_mix = props["temperature"]

#     print(f"G={G}")

#     drho_dh_model = - rho * kappaT / cp
#     drho_dp_model = rho * kappaP + (kappaT / cp) * (1 - kappaT * T_mix) 

#     drho_dh_fwd = -rho * G / c**2
#     drho_dp_fwd = (1 + G) / c**2

#     # -------------------------
#     # Comparison
#     # -------------------------
#     print("\n=== GRADIENT COMPARISON (density) ===")
#     print(f"Forward (gradients_forward):  dρ/dh = {drho_dh_fwd:.6e}")
#     # print(f"drho_dh definition:{drho_dh:.6e}")
#     print(f"Forward (gradients_forward):  dρ/dp = {drho_dp_fwd:.6e}")
#     # print(f"drho_dp definition:{drho_dp:.6e}\n")


#     print(f"Model (Grüneisen):            dρ/dh = {drho_dh_model:.6e}")
#     print(f"Model (Grüneisen):            dρ/dp = {drho_dp_model:.6e}\n")

#     print("Relative error:")
#     print(f"dρ/dh: {(drho_dh_fwd - drho_dh_model) / drho_dh_model * 100:.6e} %")
#     print(f"dρ/dp: {(drho_dp_fwd - drho_dp_model) / drho_dp_model * 100:.6e} %")

# -------------------------------------------------------------------------------------------------------------------

import CoolProp.CoolProp as CP
import jax
import jax.numpy as jnp
import jaxprop as jxp
import jaxprop.coolprop.core_calculations as CC


# -------------------------------------------------
# JAX-based gradient for density(h, p)
# -------------------------------------------------
def density_from_hp(fluid, h, p):
    """Wrap fluid.get_state so JAX can differentiate it."""
    state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
    return state["density"]


def gradients_jax(fluid, h, p):
    """Compute dρ/dh and dρ/dp using JAX autodiff."""
    # Gradient of density with respect to h and p
    grad_fn = jax.grad(lambda hp: density_from_hp(fluid, hp[0], hp[1]))

    d_rho_dh, d_rho_dp = grad_fn(jnp.array([h, p]))
    return float(d_rho_dh), float(d_rho_dp)


# ----------------------------------------------------
# Wrap internal energy as a function of (rho, p)
# ----------------------------------------------------
def u_from_rho_p(fluid, rho, p):
    state = fluid.get_state(jxp.DmassP_INPUTS, rho, p)
    return state["internal_energy"]

# ----------------------------------------------------
# Compute du/dp at constant rho
# ----------------------------------------------------
def du_dp_const_rho(fluid, rho0, p0):
    # gradient of u wrt (rho, p)
    grad_fn = jax.grad(lambda rp: u_from_rho_p(fluid, rp[0], rp[1]))
    
    du_drho, du_dp = grad_fn(jnp.array([rho0, p0]))
    
    # we return only du/dp at constant rho
    return float(du_dp)

def speed_of_sound_from_definition(fluid, rho0, s0):
    # Wrap pressure as a function of (rho, s)
    def p_from_rho_s(rho, s):
        state = fluid.get_state(jxp.DmassSmass_INPUTS, rho, s)
        return state["pressure"]

    # derivative dp/drho at constant s
    dp_drho = jax.grad(lambda rs: p_from_rho_s(rs[0], rs[1]))(jnp.array([rho0, s0]))[0]

    # speed of sound
    c = jnp.sqrt(dp_drho)
    return float(c)

def isentropic_compressibility_from_definition(fluid, rho0, p0, s0):
    """
    Compute isentropic compressibility beta_S = -1/V * (dV/dp)_S
    Inputs:
        fluid : object with get_state method
        V0    : reference volume
        p0    : reference pressure
        s0    : reference entropy
    """
    # Wrap volume as a function of (p, s)
    def V_from_p_s(p, s):
        state = fluid.get_state(jxp.PSmass_INPUTS, p, s)  # assuming rho = 1/V
        return 1 / state["density"]  # volume = 1/rho

    # derivative dV/dp at constant s
    dV_dp = jax.grad(lambda ps: V_from_p_s(ps[0], ps[1]))(jnp.array([p0, s0]))[0]

    # isentropic compressibility
    beta_S = -dV_dp * rho0
    return float(beta_S)


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":

    outdir = "mixture_tables"
    fluid_name = "water_nitrogen_mixture"
    h_min = 100e3  # J/kg
    h_max = 180e3  # J/kg
    p_min = 2e6    # Pa
    p_max = 20e6   # Pa
    N_h = 15
    N_p = 15
    mixture_ratio = 70

    fluid_bicubic = jxp.FluidBicubic(
        fluid_name=fluid_name,
        backend="HEOS",
        h_min=h_min, h_max=h_max,
        p_min=p_min, p_max=p_max,
        N_h=N_h, N_p=N_p,
        mixture_ratio=mixture_ratio,
        table_dir=outdir,
        table_name=f"{fluid_name}_table_{N_h}_{N_p}_R{mixture_ratio}"
    )

    # CoolProp pure fluid states
    

    # Mixture state values
    p_mix = 20e5    # Pa
    h_mix = 100e3   # J/kg
    R     = mixture_ratio 

    propers = fluid_bicubic.get_state(jxp.HmassP_INPUTS, h_mix, p_mix)
    print(f"Input CP:{CP.HmassP_INPUTS}")
    print(f"Input jxp:{jxp.HmassP_INPUTS}\n")



    # -------------------------
    # Analytical model gradients
    # -------------------------
    fluid = jxp.FluidMix(fluid_name, mixture_ratio=mixture_ratio)
    props = fluid.get_state(jxp.HmassP_INPUTS, h_mix, p_mix)
    rho = props["density"]
    c   = props["speed_of_sound"]
    s = props["entropy"]
    betaS = props["isentropic_compressibility"]
    # G   = props["gruneisen"]
    # kappaT = props["isobaric"]

    c_def = speed_of_sound_from_definition(fluid_bicubic, rho, s)

    sconst_compr = isentropic_compressibility_from_definition(fluid_bicubic, rho, p_mix, s)

    print(f"Def SoS      :{c_def}")
    print(f"Computed SoS :{c}")
    print(f"Delta        :{(c-c_def)/c_def*100}%\n")

    print(f"Def betaS       :{sconst_compr}")
    print(f"Computed betaS  :{betaS}")
    print(f"Delta betaS     :{(betaS-sconst_compr)/sconst_compr*100}")
    

    # drho_dh_model = props["drho_dh"]
    # drho_dp_model = props["drho_dp"]

    # # -------------------------
    # # Compute JAX gradients
    # # -------------------------
    # drho_dh_jax, drho_dp_jax = gradients_jax(fluid, h_mix, p_mix)

    # # -------------------------
    # # Comparison
    # # -------------------------
    # print("\n=== GRADIENT COMPARISON (density) ===")
    # print(f"JAX autodiff:                 dρ/dh = {drho_dh_jax:.6e}")
    # print(f"JAX autodiff:                 dρ/dp = {drho_dp_jax:.6e}\n")

    # print(f"Model (Grüneisen):            dρ/dh = {drho_dh_model:.6e}")
    # print(f"Model (Grüneisen):            dρ/dp = {drho_dp_model:.6e}\n")

    # # print(f"Drho_Dh:{rho * }")

    # print("Relative error:")
    # print(f"dρ/dh: {abs(drho_dh_jax - drho_dh_model) / abs(drho_dh_model) * 100:.10f} %")
    # print(f"dρ/dp: {abs(drho_dp_jax - drho_dp_model) / abs(drho_dp_model) * 100:.10f} %")
