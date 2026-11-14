"""Acoustic relaxation hierarchy for saturated Cyclopentane.

The full study retains 120 qualities at the original saturation temperature
473.15 K. Finite positive speeds, the relaxation inequalities and the full
chemical-equilibrium limit are checked against the configured source EOS.
Exact saturation endpoints use the library's homogeneous endpoint convention.
The separate pM expression remains research-only and is not part of acceptance.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options

import numpy as np
import jaxprop as jxp


def compute_mixture_sound_speed(fluid, input_pair, input1, input2, method="frozen"):
    """
    Compute the retained relaxation formulas.

    The CLI verifies frozen, p, pT and pT-material modes. The pM formula is
    retained as an explicitly unverified research expression; see notes.md.
    """
    # mixture state from (input1,input2)
    st = fluid.get_state(input_pair, input1, input2)

    # saturation states at same pressure
    st_l = fluid.get_state(jxp.PQ_INPUTS, st.pressure, 0.0)
    st_g = fluid.get_state(jxp.PQ_INPUTS, st.pressure, 1.0)

    # shortcut names
    p = st.pressure
    rho_mix = st.density
    T_mix = st.temperature

    # volume fractions (clamped)
    alpha_g = np.clip(st.quality_volume, 0.0, 1.0)
    alpha_l = 1.0 - alpha_g

    # mass fractions
    y_g = np.clip(st.quality_mass, 0.0, 1.0)
    y_l = 1.0 - y_g

    # outside two-phase dome → return EOS sound speed
    if st.enthalpy <= st_l.enthalpy or st.enthalpy >= st_g.enthalpy:
        return st.speed_of_sound

    # common phase properties (computed once)
    rho_l, rho_g = st_l.density, st_g.density
    a_l, a_g = st_l.speed_of_sound, st_g.speed_of_sound
    cp_l, cp_g = st_l.cp, st_g.cp
    Gamma_l, Gamma_g = st_l.gruneisen, st_g.gruneisen
    s_l, s_g = st_l.entropy, st_g.entropy
    T_l, T_g = st_l.temperature, st_g.temperature

    # frozen model (no relaxation)
    if method == "frozen":
        return np.sqrt(y_l * a_l**2 + y_g * a_g**2)

    # p-equilibrium (Wood formula)
    inv_a2_p = rho_mix * (alpha_g / (rho_g * a_g**2) + alpha_l / (rho_l * a_l**2))
    if method == "p_equilibrium":
        return np.sqrt(1.0 / inv_a2_p)

    # pT-equilibrium
    if method == "pT_equilibrium":

        Cp_l = rho_l * alpha_l * cp_l
        Cp_g = rho_g * alpha_g * cp_g

        Z_pT = (
            rho_mix
            * T_mix
            * (Cp_g * Cp_l)
            / (Cp_g + Cp_l)
            * (Gamma_l / (rho_l * a_l**2) - Gamma_g / (rho_g * a_g**2)) ** 2
        )

        inv_a2_pT = inv_a2_p + Z_pT
        return np.sqrt(1.0 / inv_a2_pT)

    # pM-equilibrium (pressure + material)
    if method == "pM_equilibrium":

        Cp_l = rho_l * alpha_l * cp_l
        Cp_g = rho_g * alpha_g * cp_g

        denom = rho_g**2 * rho_l**2 * (Cp_l * s_l**2 * T_l + Cp_g * s_g**2 * T_g)

        delta = (
            rho_g
            - rho_l
            + rho_g
            * rho_l
            * (
                s_g * T_g * Gamma_g / (rho_g * a_g**2)
                - s_l * T_l * Gamma_l / (rho_l * a_l**2)
            )
        )

        Z_pM = rho_mix * Cp_g * Cp_l / denom * delta**2
        inv_a2_pM = inv_a2_p + Z_pM

        return np.sqrt(1.0 / inv_a2_pM)

    # pTμ-equilibrium (full equilibrium / HEM)
    if method == "pTm_equilibrium":

        dsdp_g = ds_dp_sat(fluid, p, "g")
        dsdp_l = ds_dp_sat(fluid, p, "l")

        Z_pTm = (
            rho_mix
            * T_mix
            * (rho_g * alpha_g / cp_g * dsdp_g**2 + rho_l * alpha_l / cp_l * dsdp_l**2)
        )

        inv_a2_pTm = inv_a2_p + Z_pTm
        return np.sqrt(1.0 / inv_a2_pTm)

    # unknown model
    raise ValueError(f"Unknown method '{method}'")


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

    st_plus = fluid.get_state(jxp.PQ_INPUTS, p_plus, q)
    st_minus = fluid.get_state(jxp.PQ_INPUTS, p_minus, q)

    s_plus = st_plus.entropy
    s_minus = st_minus.entropy

    return (s_plus - s_minus) / (p_plus - p_minus)


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    import matplotlib.pyplot as plt

    jxp.set_plot_options()
    fluid = jxp.Fluid("cyclopentane")
    temperature = 473.15
    pressure = float(fluid.get_state(jxp.QT_INPUTS, 0.0, temperature).p)
    quality = np.logspace(-5, 0, 120 if options.profile == "full" else 21)
    models = {
        "frozen": "No equilibrium",
        "p_equilibrium": "$p$ equilibrium",
        "pT_equilibrium": "$pT$ equilibrium",
        "pTm_equilibrium": "$pT$ and material equilibrium",
    }
    results = {}
    for method in models:
        values = np.array(
            [
                compute_mixture_sound_speed(
                    fluid, jxp.PQ_INPUTS, pressure, float(q), method=method
                )
                for q in quality
            ]
        )
        assert np.all(np.isfinite(values)) and np.all(values > 0), method
        results[method] = values
    order = list(models)
    for high, low in zip(order[:-1], order[1:]):
        assert np.all(results[high] >= results[low] * (1 - 1e-9)), (high, low)
    reference = np.asarray(
        fluid.get_state(jxp.PQ_INPUTS, pressure, quality).speed_of_sound
    )
    np.testing.assert_allclose(results["pTm_equilibrium"], reference, rtol=5e-4)
    relative = np.abs(results["pTm_equilibrium"] / reference - 1)
    fig, axis = plt.subplots(figsize=(6, 4))
    for (method, label), colour in zip(
        models.items(), plt.cm.magma(np.linspace(0.25, 0.8, len(models)))
    ):
        axis.plot(quality * 100, results[method], color=colour, label=label)
    axis.plot(
        quality[0] * 100,
        results["frozen"][0],
        "ko",
        label="Near saturated liquid (Q=1e-5)",
    )
    axis.plot(quality[-1] * 100, results["frozen"][-1], "kD", label="Saturated vapor")
    axis.set_xscale("log")
    axis.set_xlabel("Vapor mass quality [%]")
    axis.set_ylabel("Sound speed [m/s]")
    axis.legend(fontsize=7)
    fig.tight_layout(pad=1)
    options.record(
        "settings",
        dict(
            fluid="Cyclopentane",
            saturation_temperature_K=temperature,
            pressure_Pa=pressure,
            quality_min=1e-5,
            quality_max=1.0,
            sample_points=int(quality.size),
            saturation_derivative_relative_step=1e-3,
        ),
    )
    options.record("maximum_HEM_relative_error", float(np.max(relative)))
    options.record("verified_relaxation_modes", list(models))
    options.record(
        "unverified_research_expression",
        "pM_equilibrium retained with its derivation in notes.md; excluded from the verified CLI study",
    )
    options.finish()


if __name__ == "__main__":
    main()
