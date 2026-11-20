"""Phase-aware evaluation on the unchanged uniform enthalpy/log-pressure grid.

The ordinary tensor polynomial is retained away from saturation. In the narrow
boundary band, one-sided Hermite bridges join an existing homogeneous grid
cell to the existing saturation endpoint. Overlapping pressure-column bridges
are blended with a C1 partition, avoiding jumps when the chosen cell changes.
Pure-fluid wet states use the source's equilibrium lever rules and PQ response
Jacobians. No source calls, extra grid points, or adaptive meshes occur at runtime.
"""

import jax.numpy as jnp
import numpy as np


def prepare(model, table):
    """Locate homogeneous anchors once; leave the persisted grid untouched."""
    if not model.phase_support or model.logP_sat_vals is None:
        return None
    import CoolProp.CoolProp as CP

    # Endpoint mixing is valid for pure fluids only. A pseudo-pure glide needs
    # its actual quality-dependent flash map and retains the existing guard.
    if CP.get_fluid_param_string(model.fluid_name, "pure") != "true":
        return None
    valid = np.asarray(model._cell_validity)
    if np.all(valid):
        # Entirely homogeneous tables (e.g. auxiliary streams) need no boundary
        # payload or extra runtime graph. Preserve their existing evaluator.
        return None
    hs, ps = np.asarray(model.h_vals), np.asarray(model.logP_vals)
    satp = np.asarray(model.logP_sat_vals)
    sat = table["saturation_props"]
    hl = np.interp(ps, satp, np.asarray(sat["liquid"]["enthalpy"]["value"]))
    hv = np.interp(ps, satp, np.asarray(sat["vapor"]["enthalpy"]["value"]))
    result = {}
    for side in ("liquid", "vapor"):
        indices = np.full(model.N_p, -1, dtype=np.int32)
        columns = np.minimum(np.arange(model.N_p), model.N_p - 2)
        for k in range(model.N_p):
            lo, hi = max(0, k - 1), min(model.N_p - 2, k)
            candidates = np.all(valid[:, lo : hi + 1], axis=1)
            if side == "liquid":
                candidates &= hs[1:] < np.min(hl[lo : hi + 2])
            else:
                candidates &= hs[:-1] > np.max(hv[lo : hi + 2])
            found = np.flatnonzero(candidates)
            if len(found):
                indices[k] = found[-1] if side == "liquid" else found[0]
        result[side] = {"index": jnp.asarray(indices), "column": jnp.asarray(columns)}
    return result


def _divide(a, b):
    ok = jnp.isfinite(a) & jnp.isfinite(b) & (b != 0)
    value = jnp.where(jnp.isfinite(a), a, 0.0) / jnp.where(ok, b, 1.0)
    return jnp.where(ok, value, jnp.nan)


def _endpoint(fluid, p, side, names):
    """Existing endpoint Hermite polynomial and its pressure derivative."""
    q = (jnp.log(p) - fluid.logP_sat_vals[0]) / fluid.delta_logP_sat
    k = jnp.clip(jnp.floor(q).astype(jnp.int32), 0, fluid.N_p_sat - 2)
    table, masks = fluid._saturation_table[side], fluid._saturation_validity[side]
    pa, pb = table["pressure"]["value"][k], table["pressure"]["value"][k + 1]
    width = pb - pa
    x = (p - pa) / width
    values, slopes = {}, {}
    for name in names:
        entry = table[name]
        a, b = entry["value"][k], entry["value"][k + 1]
        ga, gb = entry["grad_p"][k], entry["grad_p"][k + 1]
        c1 = ga * width
        c2 = 3 * (b - a) - width * (2 * ga + gb)
        c3 = 2 * (a - b) + width * (ga + gb)
        mask = masks[name]
        ok = True if mask is None else mask[k]
        values[name] = jnp.where(ok, a + x * (c1 + x * (c2 + x * c3)), jnp.nan)
        slopes[name] = jnp.where(ok, (c1 + x * (2 * c2 + 3 * x * c3)) / width, jnp.nan)
    values["pressure"], slopes["pressure"] = p, jnp.ones_like(p)
    return values, slopes


def _polynomial(fluid, h, p, i, j, name):
    x = (h - fluid.h_min) / fluid.delta_h - i
    y = (jnp.log(p) - jnp.log(fluid.p_min)) / fluid.delta_logP - j
    coefficients = fluid._table[name]["coeffs"][i, j].reshape((*i.shape, 4, 4))
    xp = jnp.stack((jnp.ones_like(x), x, x * x, x * x * x), axis=-1)
    dxp = (
        jnp.stack((jnp.zeros_like(x), jnp.ones_like(x), 2 * x, 3 * x * x), axis=-1)
        / fluid.delta_h
    )
    yp = jnp.stack((jnp.ones_like(y), y, y * y, y * y * y), axis=-1)
    value = jnp.sum(coefficients * yp[..., :, None] * xp[..., None, :], axis=(-2, -1))
    slope = jnp.sum(coefficients * yp[..., :, None] * dxp[..., None, :], axis=(-2, -1))
    mask = fluid._coefficient_validity[name]
    ok = True if mask is None else mask[i, j]
    return jnp.where(ok, value, jnp.nan), jnp.where(ok, slope, jnp.nan)


def _bridge(fluid, h, p, side, column, endpoint, raw, names):
    entry = fluid._phase_boundary[side]
    anchor_index = entry["index"][column]
    i, j = jnp.maximum(anchor_index, 0), entry["column"][column]
    ah = fluid.h_min + (i + (1 if side == "liquid" else 0)) * fluid.delta_h
    sh = endpoint["enthalpy"]
    width = ah - sh
    # Inactive/missing anchors must use finite arithmetic under reverse AD.
    width = jnp.where((anchor_index >= 0) & (width != 0), width, fluid.delta_h)
    x = (h - sh) / width
    bridge = (anchor_index >= 0) & ((h > ah) if side == "liquid" else (h < ah))
    hi = jnp.clip(
        jnp.floor((h - fluid.h_min) / fluid.delta_h).astype(jnp.int32), 0, fluid.N_h - 2
    )
    pj = jnp.clip(
        jnp.floor((jnp.log(p) - jnp.log(fluid.p_min)) / fluid.delta_logP).astype(
            jnp.int32
        ),
        0,
        fluid.N_p - 2,
    )
    valid = (anchor_index >= 0) | fluid._cell_validity[hi, pj]
    cp, rho, alpha = (
        endpoint["isobaric_heat_capacity"],
        endpoint["density"],
        endpoint["isobaric_expansion_coefficient"],
    )
    core_slopes = {
        "temperature": _divide(1.0, cp),
        "density": _divide(-rho * alpha, cp),
        "entropy": _divide(1.0, endpoint["temperature"]),
        "internal_energy": 1.0 - _divide(p * alpha, rho * cp),
    }
    out = {}
    for name in names:
        if name in (
            "pressure",
            "enthalpy",
            "is_two_phase",
            "quality_mass",
            "quality_volume",
        ):
            out[name] = raw[name]
            continue
        av, ag = _polynomial(fluid, ah, p, i, j, name)
        _, extrapolated_slope = _polynomial(fluid, sh, p, i, j, name)
        sg = core_slopes.get(name, extrapolated_slope)
        sv = endpoint[name]
        # Optional unavailable endpoint fields stay unavailable, without NaNs
        # entering arithmetic on the unselected homogeneous/raw path.
        ok = jnp.isfinite(av) & jnp.isfinite(ag) & jnp.isfinite(sv) & jnp.isfinite(sg)
        av, ag, sv, sg = (jnp.where(jnp.isfinite(v), v, 0.0) for v in (av, ag, sv, sg))
        if name == "density":
            # Specific volume is the equilibrium mixing coordinate and is
            # smoother than density near the liquid boundary. Transform both
            # endpoint slopes, then convert the interpolant back to density.
            ok &= (av > 0) & (sv > 0)
            av, sv = jnp.where(av > 0, av, 1.0), jnp.where(sv > 0, sv, 1.0)
            ag, sg = -ag / (av * av), -sg / (sv * sv)
            av, sv = 1.0 / av, 1.0 / sv
        value = sv + x * (
            sg * width
            + x
            * (
                3 * (av - sv)
                - width * (2 * sg + ag)
                + x * (2 * (sv - av) + width * (sg + ag))
            )
        )
        if name == "density":
            value = _divide(1.0, value)
        value = jnp.where(ok, value, jnp.nan)
        out[name] = jnp.where(bridge, value, raw[name])
    return out, valid


def _wet(h, p, liquid, vapor, lp, vp):
    hl, hv = liquid["enthalpy"], vapor["enthalpy"]
    q = jnp.clip((h - hl) / (hv - hl), 0.0, 1.0)
    mix = lambda a, b: (1 - q) * a + q * b
    rl, rv = liquid["density"], vapor["density"]
    rho = 1.0 / mix(1.0 / rl, 1.0 / rv)
    temperature = mix(liquid["temperature"], vapor["temperature"])
    entropy = mix(liquid["entropy"], vapor["entropy"])
    rq = -rho * rho * (1.0 / rv - 1.0 / rl)
    rp = rho * rho * mix(lp["density"] / (rl * rl), vp["density"] / (rv * rv))
    tp = mix(lp["temperature"], vp["temperature"])
    sq = vapor["entropy"] - liquid["entropy"]
    sp = mix(lp["entropy"], vp["entropy"])
    uq = vapor["internal_energy"] - liquid["internal_energy"]
    up = mix(lp["internal_energy"], vp["internal_energy"])
    determinant = up * rq - uq * rp
    sound2 = _divide(1.0, rp - _divide(rq * sp, sq))
    speed = jnp.where(sound2 > 0, jnp.sqrt(jnp.maximum(sound2, 0.0)), jnp.nan)
    z = _divide(rho * temperature, rl * liquid["temperature"])
    out = dict(
        pressure=p,
        enthalpy=h,
        temperature=temperature,
        density=rho,
        entropy=entropy,
        internal_energy=h - p / rho,
        quality_mass=q,
        quality_volume=q * rho / rv,
        is_two_phase=jnp.ones_like(h),
        compressibility_factor=_divide(liquid["compressibility_factor"], z),
        isochoric_heat_capacity=_divide(determinant, tp * rq),
        speed_of_sound=speed,
        isentropic_bulk_modulus=rho * sound2,
        isentropic_compressibility=_divide(1.0, rho * sound2),
        gruneisen=_divide(rq, rho * determinant),
        joule_thomson=tp,
        surface_tension=liquid["surface_tension"],
        temperature_saturation=liquid["temperature"],
        pressure_saturation=p,
        supersaturation_degree=temperature - liquid["temperature"],
        supersaturation_ratio=jnp.ones_like(h),
        superheating=_divide(h - hv, vapor["isobaric_heat_capacity"]),
        subcooling=_divide(hl - h, liquid["isobaric_heat_capacity"]),
    )
    for name in (
        "isobaric_heat_capacity",
        "heat_capacity_ratio",
        "isothermal_bulk_modulus",
        "isothermal_compressibility",
        "isobaric_expansion_coefficient",
        "isothermal_joule_thomson",
    ):
        out[name] = jnp.full_like(h, jnp.nan)
    for name in ("viscosity", "conductivity"):
        out[name] = (1 - q) * rho / rl * liquid[name] + q * rho / rv * vapor[name]
    return out


def evaluate(fluid, h, p, raw):
    """Return corrected fields plus point validity; supports scalar/batch AD."""
    if getattr(fluid, "_phase_boundary", None) is None:
        return raw, jnp.zeros_like(h, dtype=bool)
    dtype = jnp.result_type(h, p, fluid.h_vals)
    h, p = jnp.broadcast_arrays(
        jnp.asarray(h, dtype=dtype), jnp.asarray(p, dtype=dtype)
    )
    names = tuple(raw)
    required = set(names) | {
        "pressure",
        "enthalpy",
        "temperature",
        "density",
        "entropy",
        "internal_energy",
        "isobaric_heat_capacity",
        "isobaric_expansion_coefficient",
        "compressibility_factor",
        "surface_tension",
        "viscosity",
        "conductivity",
    }
    required &= fluid.capabilities.properties
    # Protect out-of-range branches before evaluating, and preserve the existing
    # explicit saturation pressure cap rather than extrapolating critical data.
    ep = jnp.clip(p, fluid._saturation_p_min, fluid._saturation_p_max)
    liquid, lp = _endpoint(fluid, ep, "liquid", required)
    vapor, vp = _endpoint(fluid, ep, "vapor", required)
    wet = _wet(h, p, liquid, vapor, lp, vp)
    coordinate = (jnp.log(p) - jnp.log(fluid.p_min)) / fluid.delta_logP
    j = jnp.clip(jnp.floor(coordinate).astype(jnp.int32), 0, fluid.N_p - 2)
    y = jnp.clip(coordinate - j, 0.0, 1.0)
    weight = y * y * (3.0 - 2.0 * y)
    hl, hv = liquid["enthalpy"], vapor["enthalpy"]
    is_liquid, is_vapor = h <= hl, h >= hv
    repaired = (
        (p >= fluid._saturation_p_min) & (p <= fluid._saturation_p_max) & (hv > hl)
    )
    results, validity = {}, {}
    for side, endpoint in (("liquid", liquid), ("vapor", vapor)):
        a, av = _bridge(fluid, h, p, side, j, endpoint, raw, names)
        b, bv = _bridge(fluid, h, p, side, j + 1, endpoint, raw, names)
        results[side] = {}
        for name in names:
            finite_a, finite_b = jnp.isfinite(a[name]), jnp.isfinite(b[name])
            value = (1 - weight) * jnp.where(
                finite_a, a[name], 0.0
            ) + weight * jnp.where(finite_b, b[name], 0.0)
            available = ((weight == 1) | finite_a) & ((weight == 0) | finite_b)
            results[side][name] = jnp.where(available, value, jnp.nan)
        validity[side] = ((weight == 1) | av) & ((weight == 0) | bv)
    repaired &= jnp.where(
        is_liquid, validity["liquid"], jnp.where(is_vapor, validity["vapor"], True)
    )
    output = {}
    for name in names:
        value = jnp.where(
            is_liquid,
            results["liquid"][name],
            jnp.where(
                is_vapor,
                results["vapor"][name],
                wet.get(name, jnp.full_like(h, jnp.nan)),
            ),
        )
        output[name] = jnp.where(repaired, value, raw[name])
    # Discrete fields must never be bicubic polynomials across a phase boundary.
    q = jnp.clip((h - hl) / (hv - hl), 0.0, 1.0)
    for name, value in (
        ("is_two_phase", ((h >= hl) & (h <= hv)).astype(h.dtype)),
        ("quality_mass", q),
        (
            "quality_volume",
            jnp.where(is_liquid, 0.0, jnp.where(is_vapor, 1.0, wet["quality_volume"])),
        ),
    ):
        if name in output:
            output[name] = jnp.where(repaired, value, raw[name])
    if "internal_energy" in output and "density" in output:
        output["internal_energy"] = jnp.where(
            repaired, h - p / output["density"], raw["internal_energy"]
        )
    if "temperature" in output:
        for name, value in (
            ("temperature_saturation", liquid["temperature"]),
            ("supersaturation_degree", output["temperature"] - liquid["temperature"]),
            (
                "superheating",
                jnp.where(
                    is_vapor,
                    output["temperature"] - vapor["temperature"],
                    wet["superheating"],
                ),
            ),
            (
                "subcooling",
                jnp.where(
                    is_liquid,
                    liquid["temperature"] - output["temperature"],
                    wet["subcooling"],
                ),
            ),
        ):
            if name in output:
                output[name] = jnp.where(repaired, value, raw[name])
    return output, repaired


def point_valid(fluid, h, p):
    """Cheap point guard, shared with request policies without field evaluation."""
    if getattr(fluid, "_phase_boundary", None) is None:
        return jnp.zeros_like(h, dtype=bool)
    ep = jnp.clip(p, fluid._saturation_p_min, fluid._saturation_p_max)
    liquid, _ = _endpoint(fluid, ep, "liquid", ("enthalpy",))
    vapor, _ = _endpoint(fluid, ep, "vapor", ("enthalpy",))
    hl, hv = liquid["enthalpy"], vapor["enthalpy"]
    coordinate = (jnp.log(p) - jnp.log(fluid.p_min)) / fluid.delta_logP
    j = jnp.clip(jnp.floor(coordinate).astype(jnp.int32), 0, fluid.N_p - 2)
    y = jnp.clip(coordinate - j, 0.0, 1.0)
    valid = jnp.ones_like(h, dtype=bool)
    hi = jnp.clip(
        jnp.floor((h - fluid.h_min) / fluid.delta_h).astype(jnp.int32), 0, fluid.N_h - 2
    )
    raw_valid = fluid._cell_validity[hi, j]
    for side in ("liquid", "vapor"):
        endpoint = hl if side == "liquid" else hv
        selected = h <= endpoint if side == "liquid" else h >= endpoint
        for k, active in ((j, y < 1), (j + 1, y > 0)):
            i = fluid._phase_boundary[side]["index"][k]
            valid &= ~(selected & active & (i < 0) & ~raw_valid)
    return (
        valid
        & (p >= fluid._saturation_p_min)
        & (p <= fluid._saturation_p_max)
        & (hv > hl)
    )


def temperature_seed(fluid, pressure, temperature, fallback):
    """Keep a near-saturation PT initial guess out of the flat wet plateau."""
    if getattr(fluid, "_phase_boundary", None) is None:
        return fallback
    p = jnp.clip(pressure, fluid._saturation_p_min, fluid._saturation_p_max)
    names = ("enthalpy", "temperature", "isobaric_heat_capacity")
    liquid, _ = _endpoint(fluid, p, "liquid", names)
    vapor, _ = _endpoint(fluid, p, "vapor", names)
    vapor_side = temperature > liquid["temperature"]
    sh = jnp.where(vapor_side, vapor["enthalpy"], liquid["enthalpy"])
    cp = jnp.where(
        vapor_side, vapor["isobaric_heat_capacity"], liquid["isobaric_heat_capacity"]
    )
    delta = cp * (temperature - liquid["temperature"])
    usable = (
        (pressure >= fluid._saturation_p_min)
        & (pressure <= fluid._saturation_p_max)
        & jnp.isfinite(delta)
        & (abs(delta) <= 2 * fluid.delta_h)
    )
    return jnp.where(usable, jnp.clip(sh + delta, fluid.h_min, fluid.h_max), fallback)
