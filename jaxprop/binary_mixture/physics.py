"""Thermodynamics of a fixed-composition, common-pressure/temperature mixture.

The constituents exchange heat and mechanical work, but not mass. Their
specific volume, energy, enthalpy and entropy are additive by mass. Equivalently,
the model Gibbs energy is ``g(p,T) = w1*g1(p,T) + w2*g2(p,T)``. It describes
separate constituent parcels, not a molecular mixture EOS: there is no entropy
of molecular mixing, chemical equilibrium calculation or inferred vapor quality.

For fixed mass fractions, differentiating the additive volume/entropy gives
volume-weighted expansion and isothermal compressibility, and mass-weighted
``cp``. The remaining equilibrium response functions follow from identities:

    cv = cp - T*alpha**2/(rho*kappa_T)
    kappa_s = kappa_T*cv/cp
    a_pT**2 = 1/(rho*kappa_s)
    Gamma = alpha/(rho*cv*kappa_T)

In particular, ``cv`` is generally NOT the mass average of the constituent
isochoric heat capacities: fixed total volume allows their volumes to change.
The default sound speed is the thermally and mechanically relaxed value
``a_pT``. ``speed_of_sound_p`` instead retains Wood's mechanically relaxed,
thermally frozen response, with ``kappa_s_p = sum(vol_i*kappa_s_i)``.

These equilibrium relations are derived, independently of a constituent EOS,
in S. Benjelloun, *Equilibrium Thermodynamic Properties of Two-Component and
Two-Phase Mixtures*, Proposition 3.1: https://arxiv.org/abs/2108.10148v2.
The transport laws remain the existing empirical volume averages.
"""

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

_COMPONENT_FIELDS = (
    "pressure",
    "temperature",
    "density",
    "enthalpy",
    "internal_energy",
    "entropy",
    "isobaric_heat_capacity",
    "isochoric_heat_capacity",
    "heat_capacity_ratio",
    "isobaric_expansion_coefficient",
    "isothermal_compressibility",
    "isentropic_compressibility",
    "isothermal_bulk_modulus",
    "isentropic_bulk_modulus",
    "speed_of_sound",
    "gruneisen",
    "isothermal_joule_thomson",
    "joule_thomson",
    "viscosity",
    "conductivity",
)


def _scaled_harmonic(rho1, rho2, w1, w2, xp):
    """Evaluate harmonic density/volume fractions with a dimensionless sum."""
    scale = xp.minimum(rho1, rho2)
    relative_v1, relative_v2 = w1 * (scale / rho1), w2 * (scale / rho2)
    total = relative_v1 + relative_v2
    return scale / total, relative_v1 / total, relative_v2 / total


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def _native_harmonic(rho1, rho2, w1, w2):
    return _scaled_harmonic(rho1, rho2, w1, w2, jnp)


@_native_harmonic.defjvp
def _native_harmonic_jvp(w1, w2, primals, tangents):
    """Differentiate the physical law, avoiding reciprocal-square AD factors.

    Even a scaled primal division can transpose through underflowed 1/rho_i**2.
    The exact partials w_i*(rho/rho_i)**2 stay finite for comparable large
    constituent densities. Multiplying sqrt(w_i) before the division also avoids
    a needlessly overflowing square when a constituent has a small mass weight.
    """
    rho1, rho2 = primals
    first_dot, second_dot = tangents
    density, vol1, vol2 = _native_harmonic(rho1, rho2, w1, w2)
    first_slope = (math.sqrt(w1) * density / rho1) ** 2
    second_slope = (math.sqrt(w2) * density / rho2) ** 2
    density_dot = first_slope * first_dot + second_slope * second_dot
    common = vol1 * vol2
    volume_dot = (common / rho2) * second_dot - (common / rho1) * first_dot
    return (density, vol1, vol2), (density_dot, volume_dot, -volume_dot)


def _composition(density, mass_fractions, volume_fractions, xp):
    """Return shaped composition fields, with the ratio undefined at w2=0."""
    w1, w2 = mass_fractions
    ratio = w1 / w2 if w2 else math.nan
    if not math.isfinite(ratio):
        ratio = math.nan
    if xp is np and np.ndim(density) == 0:
        return {
            "mass_frac_1": w1,
            "mass_frac_2": w2,
            "vol_frac_1": volume_fractions[0],
            "vol_frac_2": volume_fractions[1],
            "mixture_ratio": ratio,
        }
    return {
        "mass_frac_1": xp.full_like(density, w1),
        "mass_frac_2": xp.full_like(density, w2),
        "vol_frac_1": volume_fractions[0],
        "vol_frac_2": volume_fractions[1],
        "mixture_ratio": xp.full_like(density, ratio),
    }


def _pure_properties(active, mass_fractions, xp):
    """Recover the active constituent exactly without touching inactive data."""
    absent = xp.full_like(active["density"], xp.nan)
    result = {name: active.get(name, absent) for name in _COMPONENT_FIELDS}
    result["speed_of_sound_p"] = result["speed_of_sound"]
    result["speed_of_sound_pT"] = result["speed_of_sound"]
    result.update(
        _composition(
            result["density"],
            mass_fractions,
            tuple(xp.full_like(result["density"], w) for w in mass_fractions),
            xp,
        )
    )
    return result


def mix_properties(first, second, mass_fractions, *, xp=jnp):
    """Mix already evaluated, common-p/T component property mappings.

    Parameters
    ----------
    first, second : mapping
        Canonical numerical properties of the two constituents. Arrays already
        share the state shape; this kernel does not construct/validate states or
        evaluate providers. At a pure endpoint the inactive mapping may be None.
    mass_fractions : tuple of float
        Fixed fractions ``(w1, w2)`` validated once by the owning fluid object.
    xp : module, optional
        ``jax.numpy`` for a native compiled kernel, or ``numpy`` for host source
        evaluation. The equations and declared optional availability are shared.

    Returns
    -------
    dict
        Complete canonical mixture fields, including both named acoustic limits.
        Locally undefined response/transport fields are NaN. Their operands use
        finite fallback arithmetic so zero AD contributions remain safe.
    """
    w1, w2 = mass_fractions
    # These are construction-time decisions, not numerical branches per state.
    # Avoid both a wasted provider evaluation and 0*NaN at the pure endpoints.
    if w1 == 0:
        return _pure_properties(second, mass_fractions, xp)
    if w2 == 0:
        return _pure_properties(first, mass_fractions, xp)

    absent = xp.full_like(first["density"], xp.nan)
    scalar_host = xp is np and np.ndim(first["density"]) == 0

    def operand(state, name, *, positive=False):
        value = state.get(name, absent)
        # Host evaluation visits one point at a time. Scalar conditions retain
        # the same optional-field semantics without allocating np.where arrays
        # for every operand. Native/array evaluation uses fused array masks.
        if scalar_host:
            value = float(value)
            available = math.isfinite(value) and (not positive or value > 0)
            return (value if available else 1.0 if positive else 0.0), available
        defined = xp.isfinite(value)
        if positive:
            defined = defined & (value > 0)
        return xp.where(defined, value, 1.0 if positive else 0.0), defined

    def defined(value, available=True):
        if scalar_host:
            return value if available and math.isfinite(value) else math.nan
        return xp.where(available & xp.isfinite(value), value, xp.nan)

    rho1, valid_rho1 = operand(first, "density", positive=True)
    rho2, valid_rho2 = operand(second, "density", positive=True)
    # Algebraically rho=1/sum(w_i/rho_i), but differentiating that reciprocal
    # forms 1/volume**2. At large finite densities it overflows even when the
    # physical density derivative is small and representable. Scale by the
    # smaller density so the mixed denominator and volume fractions stay O(1).
    density, vol1, vol2 = (
        _scaled_harmonic(rho1, rho2, w1, w2, np)
        if xp is np
        else _native_harmonic(rho1, rho2, w1, w2)
    )
    valid_rho = valid_rho1 & valid_rho2 & xp.isfinite(density) & (density > 0)
    rho = xp.where(valid_rho, density, 1.0)

    temperature, valid_T = operand(first, "temperature", positive=True)
    cp1, valid_cp1 = operand(first, "isobaric_heat_capacity", positive=True)
    cp2, valid_cp2 = operand(second, "isobaric_heat_capacity", positive=True)
    cp = w1 * cp1 + w2 * cp2
    valid_cp = valid_cp1 & valid_cp2 & xp.isfinite(cp) & (cp > 0)
    cp = xp.where(valid_cp, cp, 1.0)

    alpha1, valid_alpha1 = operand(first, "isobaric_expansion_coefficient")
    alpha2, valid_alpha2 = operand(second, "isobaric_expansion_coefficient")
    alpha = vol1 * alpha1 + vol2 * alpha2
    valid_alpha = valid_rho & valid_alpha1 & valid_alpha2
    kT1, valid_kT1 = operand(first, "isothermal_compressibility", positive=True)
    kT2, valid_kT2 = operand(second, "isothermal_compressibility", positive=True)
    kT = vol1 * kT1 + vol2 * kT2
    valid_kT = valid_rho & valid_kT1 & valid_kT2 & xp.isfinite(kT) & (kT > 0)
    kT = xp.where(valid_kT, kT, 1.0)

    # All equilibrium response functions derive from the same cp/alpha/kT.
    # Form kappa_s through cv/cp to preserve their thermodynamic ratio identity.
    cv = cp - temperature * alpha**2 / (rho * kT)
    valid_cv = (
        valid_T
        & valid_rho
        & valid_cp
        & valid_alpha
        & valid_kT
        & xp.isfinite(cv)
        & (cv > 0)
    )
    cv = xp.where(valid_cv, cv, 1.0)
    kS = kT * (cv / cp)
    valid_kS = valid_cv & xp.isfinite(kS) & (kS > 0)
    kS = xp.where(valid_kS, kS, 1.0)
    speed = xp.sqrt(1.0 / (rho * kS))

    # This separate acoustic limit freezes inter-constituent heat transfer.
    # It is not interchangeable with the equilibrium derivative above.
    kS1, valid_kS1 = operand(first, "isentropic_compressibility", positive=True)
    kS2, valid_kS2 = operand(second, "isentropic_compressibility", positive=True)
    frozen_kS = vol1 * kS1 + vol2 * kS2
    valid_frozen = (
        valid_rho & valid_kS1 & valid_kS2 & xp.isfinite(frozen_kS) & (frozen_kS > 0)
    )
    frozen_kS = xp.where(valid_frozen, frozen_kS, 1.0)
    frozen_speed = xp.sqrt(1.0 / (rho * frozen_kS))

    muT = (1.0 - temperature * alpha) / rho
    valid_muT = valid_T & valid_rho & valid_alpha
    result = {
        "pressure": first["pressure"],
        "temperature": first["temperature"],
        "density": defined(density, valid_rho),
        "enthalpy": w1 * first["enthalpy"] + w2 * second["enthalpy"],
        "internal_energy": w1 * first["internal_energy"]
        + w2 * second["internal_energy"],
        "entropy": w1 * first["entropy"] + w2 * second["entropy"],
        "isobaric_heat_capacity": defined(cp, valid_cp),
        "isochoric_heat_capacity": defined(cv, valid_cv),
        "heat_capacity_ratio": defined(cp / cv, valid_cv),
        "isobaric_expansion_coefficient": defined(alpha, valid_alpha),
        "isothermal_compressibility": defined(kT, valid_kT),
        "isentropic_compressibility": defined(kS, valid_kS),
        "isothermal_bulk_modulus": defined(1.0 / kT, valid_kT),
        "isentropic_bulk_modulus": defined(1.0 / kS, valid_kS),
        "speed_of_sound": defined(speed, valid_kS),
        "speed_of_sound_pT": defined(speed, valid_kS),
        "speed_of_sound_p": defined(frozen_speed, valid_frozen),
        "gruneisen": defined(alpha / (rho * cv * kT), valid_cv),
        "isothermal_joule_thomson": defined(muT, valid_muT),
        "joule_thomson": defined(-muT / cp, valid_muT & valid_cp),
    }
    for name in ("viscosity", "conductivity"):
        first_value, valid_first = operand(first, name)
        second_value, valid_second = operand(second, name)
        result[name] = defined(
            vol1 * first_value + vol2 * second_value,
            valid_rho & valid_first & valid_second,
        )
    result.update(_composition(density, mass_fractions, (vol1, vol2), xp))
    return result
