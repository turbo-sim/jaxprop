"""Canonical floating property names and their validated public aliases."""


def _build_property_schema(definitions):
    """Validate entries before dict construction can hide duplicate names."""
    names = [name for name, _ in definitions]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate canonical property definition")
    canonical_names = set(names)
    aliases = {}
    lookup = {name: name for name in names}
    for canonical, alternatives in definitions:
        if not isinstance(canonical, str) or not canonical:
            raise ValueError("Canonical property names must be nonempty strings")
        aliases[canonical] = list(alternatives)
        seen = set()
        for alias in alternatives:
            if not isinstance(alias, str) or not alias or alias in seen:
                raise ValueError(
                    f"Invalid or duplicate alias for {canonical}: {alias!r}"
                )
            if alias in canonical_names and alias != canonical:
                raise ValueError(f"Alias {alias!r} shadows a canonical property")
            if alias in lookup and lookup[alias] != canonical:
                raise ValueError(f"Alias {alias!r} is defined for multiple properties")
            seen.add(alias)
            lookup[alias] = canonical
    return aliases, lookup


_PROPERTY_DEFINITIONS = (
    ("pressure", ("p", "P")),
    ("temperature", ("T",)),
    ("density", ("rho", "d", "rhomass", "dmass", "density")),
    ("enthalpy", ("h", "hmass", "enthalpy", "H")),
    ("entropy", ("s", "smass", "entropy")),
    ("internal_energy", ("u", "umass", "e", "energy", "internal_energy")),
    ("isobaric_heat_capacity", ("cp", "cpmass")),
    ("isochoric_heat_capacity", ("cv", "cvmass")),
    ("heat_capacity_ratio", ("gamma", "kappa")),
    ("compressibility_factor", ("Z", "compressibility_factor")),
    ("isothermal_compressibility", ("kappa_T",)),
    ("isentropic_compressibility", ("kappa_s",)),
    ("isothermal_bulk_modulus", ("K_T",)),
    ("isentropic_bulk_modulus", ("K_s",)),
    ("speed_of_sound", ("a", "speed_sound")),
    ("viscosity", ("mu",)),
    ("conductivity", ("k",)),
    ("gruneisen", ("gruneisen", "G")),
    ("isobaric_expansion_coefficient", ("alpha_p",)),
    ("isothermal_joule_thomson", ("mu_T",)),
    ("joule_thomson", ("mu_JT",)),
    ("is_two_phase", ()),
    ("quality_mass", ("vapor_quality", "Q", "q")),
    ("quality_volume", ("void_fraction", "alpha")),
    ("surface_tension", ("sigma",)),
    ("pressure_saturation", ()),
    ("temperature_saturation", ()),
    ("supersaturation_degree", ()),
    ("supersaturation_ratio", ()),
    ("subcooling", ()),
    ("superheating", ()),
    ("mixture_ratio", ()),
    ("mass_frac_1", ()),
    ("mass_frac_2", ()),
    ("vol_frac_1", ()),
    ("vol_frac_2", ()),
    ("drho_dp", ()),
    ("drho_dh", ()),
)

PROPERTY_ALIASES, ALIAS_TO_CANONICAL = _build_property_schema(_PROPERTY_DEFINITIONS)
PROPERTIES_CANONICAL = list(PROPERTY_ALIASES)
