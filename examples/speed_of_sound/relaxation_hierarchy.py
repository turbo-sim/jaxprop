
import os
import numpy as np
import jaxprop as jxp
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

jxp.set_plot_options()

os.makedirs("figures", exist_ok=True)

# -----------------------------------------------------------
# Main function
# -----------------------------------------------------------
def compute_mixture_sound_speed(fluid, input_pair, input1, input2, method="frozen"):
    """
    Compute mixture sound speed for:
        frozen, p-equilibrium, pT, pM, pTμ models.
    """
    # mixture state from (input1,input2)
    st = fluid.get_state(input_pair, input1, input2)

    # saturation states at same pressure
    st_l = fluid.get_state(jxp.PQ_INPUTS, st.pressure, 0.0)
    st_g = fluid.get_state(jxp.PQ_INPUTS, st.pressure, 1.0)

    # shortcut names
    p       = st.pressure
    rho_mix = st.density
    T_mix   = st.temperature

    # volume fractions (clamped)
    alpha_g = np.clip(st.quality_volume, 0.0, 1.0)
    alpha_l = 1.0 - alpha_g

    # mass fractions
    y_g = np.clip(st.quality_mass, 0.0, 1.0)
    y_l = 1.0 - y_g

    # outside two-phase dome → return EOS sound speed
    if st.enthalpy < st_l.enthalpy or st.enthalpy > st_g.enthalpy:
        return st.speed_of_sound

    # common phase properties (computed once)
    rho_l, rho_g = st_l.density,       st_g.density
    a_l,   a_g   = st_l.speed_of_sound, st_g.speed_of_sound
    cp_l,  cp_g  = st_l.cp,            st_g.cp
    Gamma_l, Gamma_g      = st_l.gruneisen, st_g.gruneisen
    s_l,  s_g    = st_l.entropy, st_g.entropy
    T_l,  T_g    = st_l.temperature,  st_g.temperature

    # frozen model (no relaxation)
    if method == "frozen":
        return np.sqrt(y_l * a_l**2 + y_g * a_g**2)

    # p-equilibrium (Wood formula)
    inv_a2_p =  rho_mix * (
        alpha_g / (rho_g * a_g**2) +
        alpha_l / (rho_l * a_l**2)
    )
    if method == "p_equilibrium":
        return np.sqrt(1.0 / inv_a2_p)

    # pT-equilibrium
    if method == "pT_equilibrium":

        Cp_l = rho_l * alpha_l * cp_l
        Cp_g = rho_g * alpha_g * cp_g

        Z_pT = (
            rho_mix * T_mix *
            (Cp_g * Cp_l) / (Cp_g + Cp_l) *
            (Gamma_l/(rho_l * a_l**2) - Gamma_g/(rho_g * a_g**2))**2
        )

        inv_a2_pT = inv_a2_p + Z_pT
        return np.sqrt(1.0 / inv_a2_pT)

    # pM-equilibrium (pressure + material)
    if method == "pM_equilibrium":

        Cp_l = rho_l * alpha_l * cp_l
        Cp_g = rho_g * alpha_g * cp_g

        denom = rho_g**2 * rho_l**2 * (Cp_l * s_l**2 * T_l + Cp_g * s_g**2 * T_g)

        delta = (
            rho_g - rho_l +
            rho_g * rho_l *
            (s_g*T_g*Gamma_g/(rho_g*a_g**2) -
             s_l*T_l*Gamma_l/(rho_l*a_l**2))
        )

        Z_pM = rho_mix * Cp_g * Cp_l / denom * delta**2
        inv_a2_pM = inv_a2_p + Z_pM

        return np.sqrt(1.0 / inv_a2_pM)


    # pTμ-equilibrium (full equilibrium / HEM)
    if method == "pTm_equilibrium":

        dsdp_g = ds_dp_sat(fluid, p, "g")
        dsdp_l = ds_dp_sat(fluid, p, "l")

        Z_pTm = (
            rho_mix * T_mix *
            (rho_g * alpha_g / cp_g * dsdp_g**2 +
             rho_l * alpha_l / cp_l * dsdp_l**2)
        )

        inv_a2_pTm = inv_a2_p + Z_pTm
        return np.sqrt(1.0 / inv_a2_pTm)


    # unknown model
    raise ValueError(f"Unknown method '{method}'")


# -----------------------------------------------------------
# Finite-difference entropy derivative along saturation curve
# -----------------------------------------------------------
def ds_dp_sat(fluid, p, phase, dp_rel=1e-3):
    if phase == "g":
        q = 1.0
    elif phase == "l":
        q = 0.0
    else:
        raise ValueError("phase must be 'g' or 'l'")

    dp = dp_rel * p
    p_plus = p + dp
    p_minus = p - dp

    st_plus  = fluid.get_state(jxp.PQ_INPUTS, p_plus,  q)
    st_minus = fluid.get_state(jxp.PQ_INPUTS, p_minus, q)

    s_plus  = st_plus.entropy
    s_minus = st_minus.entropy

    return (s_plus - s_minus) / (p_plus - p_minus)




# ==========================================================
# Demonstration script
# ==========================================================
if __name__ == "__main__":


    # models to evaluate
    sound_speed_models = {
        "frozen":          "No equilibrium",
        "p_equilibrium":   r"$p$-equilibrium",
        "pT_equilibrium":  r"$pT$-equilibrium",
        # "pM_equilibrium":  r"$p\mu$-equilibrium",
        "pTm_equilibrium": r"$pT\mu$-equilibrium",
    }

    # fluid and evaluation pressure
    fluid = jxp.Fluid("cyclopentane")
    p_crit = fluid.critical_point.pressure
    # p_eval = 0.1 * p_crit
    T_eval =200 + 273.15
    p_eval = fluid.get_state(jxp.QT_INPUTS, 0.0, T_eval).pressure
 

    # Crossing the liquid saturation line
    q_array = np.logspace(-5, 0, 120)
    q_percent = q_array * 100.0

    results_q = {key: np.zeros_like(q_array) for key in sound_speed_models}
    for key in sound_speed_models:
        for i, q in enumerate(q_array):
            results_q[key][i] = compute_mixture_sound_speed(
                fluid, jxp.PQ_INPUTS, p_eval, q, method=key
            )

    n_lines = len(sound_speed_models)
    cmap = plt.get_cmap("magma")
    colors = cmap(np.linspace(0.25, 0.8, n_lines))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for (key, label), c in zip(sound_speed_models.items(), colors):
        ax.plot(q_percent, results_q[key], lw=1.5, color=c, label=label)
    ax.plot(q_percent[0], results_q["frozen"][0], "ko", markersize=4.5, label="Saturated liquid")
    ax.plot(q_percent[-1], results_q["frozen"][-1], "kD", markersize=4., label="Saturated vapor")
    char_q_vals   = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]) * 100.0
    char_labels   = ["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"]

    ax.set_xscale("log")
    ax.set_xticks(char_q_vals)
    ax.set_xticklabels(char_labels)
    ax.set_xlabel("Vapor quality [%]")
    ax.set_ylabel("Mixture speed of sound [m/s]")
    ax.grid(False)
    ax.legend(fontsize=9)
    fig.tight_layout(pad=1)
    # jxp.savefig_in_formats(fig, "figures/relaxation_hierarchy_sound_speed_vs_quality")

    plt.show()
