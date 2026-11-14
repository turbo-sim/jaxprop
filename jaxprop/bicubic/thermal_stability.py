"""Construction-time stability and pole guards for homogeneous tables.

A mechanically admissible continuation can still have negative heat capacity.
The default physical table requires positive nodal/stencil ``cv`` and ``K_T``.
An explicit formal-EOS table retains such caloric states for research, while
responses with a ``cv`` denominator are unavailable in cells that may cross
zero. No stability calculations or EOS queries enter native state evaluation.

Bernstein control values conservatively bound each bicubic polynomial over
the whole unit cell, including overshoot invisible at its corners. These are
bounds on the interpolant, not a proof of the source EOS between sample sites.
"""

import numpy as np

METASTABLE_STABILITY_SETTINGS = {
    "policy": "positive_nodes_and_bernstein_cells_v1",
    "roundoff_ulps": 64,
    "cv_pole_properties": (
        "heat_capacity_ratio",
        "gruneisen",
        "isentropic_bulk_modulus",
        "speed_of_sound",
    ),
    "cp_pole_properties": ("joule_thomson", "isentropic_compressibility"),
}

# x**i = sum_{k=i}^3 binom(k,i)/binom(3,i) * B_k^3(x).
# The stored coefficient order is kron(y powers, x powers).
_POWER_TO_BERNSTEIN = np.array(
    [[1, 0, 0, 0], [1, 1 / 3, 0, 0], [1, 2 / 3, 1 / 3, 0], [1, 1, 1, 1]],
    dtype=float,
)


def polynomial_bounds(coefficients, *, roundoff_ulps=64):
    """Return conservative cell bounds, with a floating-arithmetic allowance."""
    coefficients = np.asarray(coefficients)
    power = coefficients.reshape(coefficients.shape[:-1] + (4, 4))
    bernstein = np.einsum(
        "ai,...ij,bj->...ab", _POWER_TO_BERNSTEIN, power, _POWER_TO_BERNSTEIN
    )
    allowance = (
        roundoff_ulps
        * np.finfo(coefficients.dtype).eps
        * np.maximum(np.sum(np.abs(power), axis=(-2, -1)), 1.0)
    )
    return (
        np.min(bernstein, axis=(-2, -1)) - allowance,
        np.max(bernstein, axis=(-2, -1)) + allowance,
    )


def metastable_cell_guards(table, domain, settings):
    """Prepare full-state eligibility and optional response-pole masks."""
    bounds = {
        name: polynomial_bounds(
            table[name]["coeffs"], roundoff_ulps=settings["roundoff_ulps"]
        )
        for name in (
            "isochoric_heat_capacity",
            "isobaric_heat_capacity",
            "isothermal_bulk_modulus",
        )
    }
    mechanical_lower, _ = bounds["isothermal_bulk_modulus"]
    eligible = np.isfinite(mechanical_lower) & (mechanical_lower > 0)
    cv_lower, _ = bounds["isochoric_heat_capacity"]
    if domain == "physical":
        eligible &= np.isfinite(cv_lower) & (cv_lower > 0)
    poles = {}
    for response, names in (
        ("isochoric_heat_capacity", settings["cv_pole_properties"]),
        ("isobaric_heat_capacity", settings["cp_pole_properties"]),
    ):
        lower, upper = bounds[response]
        crossing = (
            ~np.isfinite(lower) | ~np.isfinite(upper) | ((lower <= 0) & (upper >= 0))
        )
        for name in names:
            poles[name] = crossing
    return eligible, poles


def prepare_metastable_stability(table, domain, settings):
    """Mask possible poles once; retain all source nodes for inspection."""
    eligible, poles = metastable_cell_guards(table, domain, settings)
    for name, crossing in poles.items():
        if np.any(crossing):
            coefficients = np.asarray(table[name]["coeffs"]).copy()
            coefficients[crossing] = np.nan
            table[name]["coeffs"] = coefficients
    return eligible


def metastable_table_error(table, metadata):
    """Validate persisted stability guards without trusting their mask values."""
    domain = metadata["metastable_domain"]
    for name in ("isothermal_bulk_modulus", "isochoric_heat_capacity"):
        values = np.asarray(table[name]["value"])
        if name == "isochoric_heat_capacity" and domain == "formal":
            continue
        invalid = ~np.isfinite(values) | (values <= 0)
        if np.any(invalid):
            index = tuple(int(i) for i in np.argwhere(invalid)[0])
            return (
                f"{domain} metastable table requires positive {name}; "
                f"got {values[index]!r} at index {index}"
            )
    eligible, poles = metastable_cell_guards(
        table, domain, metadata["generation_settings"]["metastable_stability"]
    )
    if np.any(np.asarray(table["phase_cell_validity"]) & ~eligible):
        return "metastable cell validity permits an unverified stability boundary"
    for name, crossing in poles.items():
        if np.any(np.isfinite(np.asarray(table[name]["coeffs"])[crossing])):
            return f"{name}: coefficients cross an unmasked heat-capacity pole"
    return None
