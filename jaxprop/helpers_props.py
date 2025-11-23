import CoolProp.CoolProp as CP
import jax.numpy as jnp
import equinox as eqx
from dataclasses import fields

# Universal molar gas constant
GAS_CONSTANT = 8.3144598

# -------------------------------------------------------------------- #
# Add CoolProp constants to the module namespace
# -------------------------------------------------------------------- #

# Dynamically add INPUTS fields to the module
# for attr in dir(CP):
#     if attr.endswith('_INPUTS'):
#         globals()[attr] = getattr(CP, attr)

# Statically add phase indices to the module (IDE autocomplete)
iphase_critical_point = CP.iphase_critical_point
iphase_gas = CP.iphase_gas
iphase_liquid = CP.iphase_liquid
iphase_not_imposed = CP.iphase_not_imposed
iphase_supercritical = CP.iphase_supercritical
iphase_supercritical_gas = CP.iphase_supercritical_gas
iphase_supercritical_liquid = CP.iphase_supercritical_liquid
iphase_twophase = CP.iphase_twophase
iphase_unknown = CP.iphase_unknown

# Statically add INPUT fields to the module (IDE autocomplete)
QT_INPUTS = CP.QT_INPUTS
PQ_INPUTS = CP.PQ_INPUTS
QSmolar_INPUTS = CP.QSmolar_INPUTS
QSmass_INPUTS = CP.QSmass_INPUTS
HmolarQ_INPUTS = CP.HmolarQ_INPUTS
HmassQ_INPUTS = CP.HmassQ_INPUTS
DmolarQ_INPUTS = CP.DmolarQ_INPUTS
DmassQ_INPUTS = CP.DmassQ_INPUTS
PT_INPUTS = CP.PT_INPUTS
DmassT_INPUTS = CP.DmassT_INPUTS
DmolarT_INPUTS = CP.DmolarT_INPUTS
HmolarT_INPUTS = CP.HmolarT_INPUTS
HmassT_INPUTS = CP.HmassT_INPUTS
SmolarT_INPUTS = CP.SmolarT_INPUTS
SmassT_INPUTS = CP.SmassT_INPUTS
TUmolar_INPUTS = CP.TUmolar_INPUTS
TUmass_INPUTS = CP.TUmass_INPUTS
DmassP_INPUTS = CP.DmassP_INPUTS
DmolarP_INPUTS = CP.DmolarP_INPUTS
HmassP_INPUTS = CP.HmassP_INPUTS
HmolarP_INPUTS = CP.HmolarP_INPUTS
PSmass_INPUTS = CP.PSmass_INPUTS
PSmolar_INPUTS = CP.PSmolar_INPUTS
PUmass_INPUTS = CP.PUmass_INPUTS
PUmolar_INPUTS = CP.PUmolar_INPUTS
HmassSmass_INPUTS = CP.HmassSmass_INPUTS
HmolarSmolar_INPUTS = CP.HmolarSmolar_INPUTS
SmassUmass_INPUTS = CP.SmassUmass_INPUTS
SmolarUmolar_INPUTS = CP.SmolarUmolar_INPUTS
DmassHmass_INPUTS = CP.DmassHmass_INPUTS
DmolarHmolar_INPUTS = CP.DmolarHmolar_INPUTS
DmassSmass_INPUTS = CP.DmassSmass_INPUTS
DmolarSmolar_INPUTS = CP.DmolarSmolar_INPUTS
DmassUmass_INPUTS = CP.DmassUmass_INPUTS
DmolarUmolar_INPUTS = CP.DmolarUmolar_INPUTS

# Convert each input key to a tuple of FluidState variable names
# Capitalized names that should not be lowercased
preserve_case = {"T", "Q"}

def extract_vars(name):
    base = name.replace("_INPUTS", "")
    parts = []
    current = base[0]
    for c in base[1:]:
        if c.isupper():
            parts.append(current)
            current = c
        else:
            current += c
    parts.append(current)
    return tuple(p if p in preserve_case else p.lower() for p in parts)


# Define dictionary with dynamically generated fields
PHASE_INDEX = {attr: getattr(CP, attr) for attr in dir(CP) if attr.startswith("iphase")}
INPUT_PAIRS = {attr: getattr(CP, attr) for attr in dir(CP) if attr.endswith("_INPUTS")}
INPUT_TYPE_MAP = {v: k for k, v in sorted(INPUT_PAIRS.items(), key=lambda x: x[1])}
INPUT_PAIR_MAP = {k: extract_vars(v) for k, v in INPUT_TYPE_MAP.items()}


def _generate_coolprop_input_table():
    """Create table of input pairs as string to be copy-pasted in Sphinx documentation"""
    inputs_table = ".. list-table:: CoolProp input mappings\n"
    inputs_table += "   :widths: 50 30\n"
    inputs_table += "   :header-rows: 1\n\n"
    inputs_table += "   * - Input pair name\n"
    inputs_table += "     - Input pair mapping\n"
    for name, value in INPUT_PAIRS:
        inputs_table += f"   * - {name}\n"
        inputs_table += f"     - {value}\n"

    return inputs_table

# -------------------------------------------------------------------- #
# Define aliases for canonical property names
# -------------------------------------------------------------------- #

PROPERTY_ALIASES = {
    # --- basic thermodynamic properties
    "pressure": ["p", "P"],
    "temperature": ["T"],
    "density": ["rho", "d", "rhomass", "dmass"],  # add "D" when fixing nozzle overwrite
    "enthalpy": ["h", "hmass", "H"],
    "entropy": ["s", "smass"],
    "internal_energy": ["u", "umass", "e", "energy"],
    # --- heat capacities & ratios
    "isobaric_heat_capacity": ["cp", "cpmass"],
    "isochoric_heat_capacity": ["cv", "cvmass"],
    "heat_capacity_ratio": ["gamma", "kappa"],
    # --- compressibility & bulk moduli
    "compressibility_factor": ["Z", "compressibility_factor"],
    "isothermal_compressibility": ["kappa_T"],
    "isentropic_compressibility": ["kappa_s"],
    "isothermal_bulk_modulus": ["B_T"],
    "isentropic_bulk_modulus": ["B_s"],
    # --- transport & misc
    "speed_of_sound": ["a", "speed_sound", "soundspeed"],
    "viscosity": ["mu"],
    "conductivity": ["k"],
    "gruneisen": ["gruneisen_parameter", "G"],
    # --- expansion & JT effects
    "isobaric_expansion_coefficient": ["alpha_p"],
    "isothermal_joule_thomson": ["mu_T"],
    "joule_thomson": ["mu_JT"],
    # "dhdp_T": [],
    # --- two-phase
    "is_two_phase": [],
    "quality_mass": ["vapor_quality", "Q", "q"],  # add "x" when fixing nozzle overwrite
    "quality_volume": ["void_fraction", "alpha"],
    "surface_tension": ["sigma"],
    "pressure_saturation": [],
    "temperature_saturation": [],
    "supersaturation_degree": [],
    "supersaturation_ratio": [],
    "subcooling": [],
    "superheating": [],
}


# flat lookup alias -> canonical
ALIAS_TO_CANONICAL = {}
for canonical, aliases in PROPERTY_ALIASES.items():
    for alias in aliases:
        if alias in ALIAS_TO_CANONICAL:
            raise ValueError(f"Alias {alias} defined for multiple properties")
        ALIAS_TO_CANONICAL[alias] = canonical
    # also allow canonical name itself
    ALIAS_TO_CANONICAL[canonical] = canonical

PROPERTIES_CANONICAL = PROPERTY_ALIASES.keys()

LABEL_MAPPING = {
    "density": "Density (kg/m$^3$)",
    "viscosity": "Viscosity (Pa·s)",
    "speed_sound": "Speed of sound (m/s)",
    "void_fraction": "Void fraction",
    "vapor_quality": "Vapor quality",
    "p": "Pressure (Pa)",
    "s": "Entropy (J/kg/K)",
    "T": "Temperature (K)",
    "h": "Enthalpy (J/kg)",
    "pressure": "Pressure (Pa)",
    "entropy": "Entropy (J/kg/K)",
    "temperature": "Temperature (K)",
    "enthalpy": "Enthalpy (J/kg)",
    "rho": r"Density (kg/m$^3$)",
}


# -------------------------------------------------------------------- #
# Define equinox Module to represent fluid states
# -------------------------------------------------------------------- #

import jax.numpy as jnp
import equinox as eqx

class BaseState(eqx.Module):
    """
    Base class for state-like objects.

    Intentionally mirrors FluidState behavior with minimal generalization:
    - metadata fields are configurable via _meta_fields
    - alias lookup uses _alias_to_canonical (defaults to global ALIAS_TO_CANONICAL)
    """

    # --- configuration hooks (static so they don't become pytree leaves)
    _meta_fields: tuple = eqx.field(static=True, default=("fluid_name", "identifier", "_meta_fields"))

    # --- Access helpers
    def __getitem__(self, key: str):
        """Allow dictionary-style access via canonical or alias name"""
        # Metadata keys: passthrough
        if key in self._meta_fields:
            return getattr(self, key)

        # Canonical / alias keys
        if key in ALIAS_TO_CANONICAL:
            return getattr(self, ALIAS_TO_CANONICAL[key])

        raise KeyError(f"Unknown property alias: {key}")

    def __getattr__(self, key: str):
        """Allow attribute-style access via alias names"""
        if key in ALIAS_TO_CANONICAL:
            return getattr(self, ALIAS_TO_CANONICAL[key])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __repr__(self) -> str:
        """Readable string representation with scalars if possible"""
        lines = []
        for name, val in self.__dict__.items():
            if val is None:
                continue
            try:
                val = jnp.array(val).item()
            except Exception:
                pass
            lines.append(f"  {name}={val}")
        return f"{type(self).__name__}(\n" + ",\n".join(lines) + "\n)"
        # return "FluidState(\n" + ",\n".join(lines) + "\n)"


    def to_dict(self, include_aliases: bool = False):
        """Return dict of numeric properties, with optional aliases."""

        skip = set(self._meta_fields)
        out = {}

        for k, v in self.__dict__.items():
            if v is None or k in skip:
                continue

            # handle nested BaseState objects
            if isinstance(v, BaseState):
                out[k] = v.to_dict(include_aliases=include_aliases)
                continue

            # handle nested BaseState objects
            out[k] = jnp.asarray(v)

        # alias expansion
        if include_aliases:
            for canonical, aliases in PROPERTY_ALIASES.items():
                if canonical in out:
                    for alias in aliases:
                        if alias not in out:
                            out[alias] = out[canonical]

        return out

    def keys(self):
        """Dict-style iteration"""
        return self.to_dict().keys()

    def values(self):
        """Dict-style iteration"""
        return self.to_dict().values()

    def items(self):
        """Dict-style iteration"""
        return self.to_dict().items()

    def flipped(self):
        """
        Return a new state where all array-valued fields are reversed in order.

        Scalars and 0-D arrays are left unchanged, while 1-D or N-D arrays are flipped
        along their first axis. Metadata fields are copied unchanged.
        """
        flipped_data = {}

        for field, value in self.__dict__.items():
            # Keep metadata unchanged
            if field in self._meta_fields:
                flipped_data[field] = value
                continue

            # Convert to array
            arr = jnp.asarray(value)

            # Handle None, scalar, or NaN values
            if arr is None or arr.size <= 1:
                flipped_data[field] = arr
            else:
                # Flip along first axis
                flipped_data[field] = jnp.flip(arr, axis=0)

        return type(self)(**flipped_data)

    def at_index(self, idx: int):
        """
        Extract a single state at the given index from a batched state.

        This returns a new state with the same metadata,
        and each property reduced to the element at position `idx`
        along the first axis.
        """
        data = {}
        for field, value in self.__dict__.items():
            if field in self._meta_fields:
                data[field] = value
                continue

            arr = jnp.asarray(value)

            # Handle scalars or 0-D arrays
            if arr.size <= 1:
                data[field] = arr
            else:
                # Extract along first axis
                try:
                    data[field] = arr[idx]
                except Exception:
                    # In case array is shorter or malformed
                    data[field] = jnp.nan

        return type(self)(**data)

    @classmethod
    def stack(cls, states: list):
        """Combine a list of states into a batched state (values stacked into arrays)."""
        if not states:
            raise ValueError("No states provided to stack")

        meta_fields = states[0]._meta_fields

        # Check consistency of metadata
        meta_values = {m: getattr(states[0], m) for m in meta_fields}
        for s in states[1:]:
            for m in meta_fields:
                if getattr(s, m) != meta_values[m]:
                    raise ValueError(
                        f"All {cls.__name__} objects must have the same "
                        + " and ".join(meta_fields)
                    )

        # Collect stacked properties
        data = {}
        for field in states[0].__dict__.keys():
            if field in meta_fields:
                continue

            values = [getattr(s, field) for s in states]
            if all(v is None for v in values):
                data[field] = jnp.nan
            else:
                # Convert scalars to arrays before stacking
                arrs = [
                    jnp.atleast_1d(v) if v is not None else jnp.array([jnp.nan])
                    for v in values
                ]
                data[field] = jnp.stack(arrs).squeeze()

        return cls(**meta_values, **data)

    def __add__(self, other):
        """Concatenate two state objects along their first dimension."""
        if not isinstance(other, type(self)):
            return NotImplemented

        meta_fields = self._meta_fields
        for m in meta_fields:
            if getattr(self, m) != getattr(other, m):
                raise ValueError(
                    f"Cannot combine {type(self).__name__} objects with different "
                    + " and ".join(meta_fields) + "."
                )

        merged = {}

        for field, val_self in self.__dict__.items():

            # metadata unchanged
            if field in meta_fields:
                merged[field] = val_self
                continue

            val_other = getattr(other, field)

            #handle nested state objects (FluidState, MixtureState, etc.)
            if isinstance(val_self, BaseState):
                # Use the nested object's own __add__
                merged[field] = val_self + val_other
                continue

            # concatenation of numeric arrays
            arr_self = jnp.atleast_1d(val_self)
            arr_other = jnp.atleast_1d(val_other)
            merged[field] = jnp.concatenate([arr_self, arr_other])

        return type(self)(**merged)



class FluidState(BaseState):
    # --- metadata
    fluid_name: str = eqx.field(static=True, default=None)
    identifier: str = eqx.field(static=True, default=None)

    # --- basic thermodynamic properties
    pressure: jnp.ndarray = jnp.nan
    temperature: jnp.ndarray = jnp.nan
    density: jnp.ndarray = jnp.nan
    enthalpy: jnp.ndarray = jnp.nan
    entropy: jnp.ndarray = jnp.nan
    internal_energy: jnp.ndarray = jnp.nan
    compressibility_factor: jnp.ndarray = jnp.nan

    # --- thermodynamic properties involving derivatives
    isobaric_heat_capacity: jnp.ndarray = jnp.nan
    isochoric_heat_capacity: jnp.ndarray = jnp.nan
    heat_capacity_ratio: jnp.ndarray = jnp.nan
    speed_of_sound: jnp.ndarray = jnp.nan
    isothermal_compressibility: jnp.ndarray = jnp.nan
    isentropic_compressibility: jnp.ndarray = jnp.nan
    isothermal_bulk_modulus: jnp.ndarray = jnp.nan
    isentropic_bulk_modulus: jnp.ndarray = jnp.nan
    isobaric_expansion_coefficient: jnp.ndarray = jnp.nan
    isothermal_joule_thomson: jnp.ndarray = jnp.nan
    joule_thomson: jnp.ndarray = jnp.nan
    gruneisen: jnp.ndarray = jnp.nan

    # --- transport properties
    viscosity: jnp.ndarray = jnp.nan
    conductivity: jnp.ndarray = jnp.nan

    # --- two-phase properties
    is_two_phase: jnp.ndarray = jnp.nan
    quality_mass: jnp.ndarray = jnp.nan
    quality_volume: jnp.ndarray = jnp.nan
    surface_tension: jnp.ndarray = jnp.nan
    subcooling: jnp.ndarray = jnp.nan
    superheating: jnp.ndarray = jnp.nan
    pressure_saturation: jnp.ndarray = jnp.nan
    temperature_saturation: jnp.ndarray = jnp.nan
    supersaturation_degree: jnp.ndarray = jnp.nan
    supersaturation_ratio: jnp.ndarray = jnp.nan

class MixtureState(BaseState):
    """
    Two-component mixture state.

    This class follows the same structure and behavior style as FluidState,
    but holds two nested FluidState objects for the components.
    Only `stack` requires an override because BaseState.stack assumes that
    each field is numeric, whereas `component_1` and `component_2`
    are FluidState objects.
    """

    # --- metadata
    identifier: str = eqx.field(static=True, default=None)
    fluid_name: str = eqx.field(static=True, default=None)

    # --- component states
    component_1: FluidState = None
    component_2: FluidState = None

    # --- mixture composition
    mass_fraction_1: jnp.ndarray = jnp.nan
    mass_fraction_2: jnp.ndarray = jnp.nan
    volume_fraction_1: jnp.ndarray = jnp.nan
    volume_fraction_2: jnp.ndarray = jnp.nan

    # --- mixture properties (same naming pattern as FluidState)
    pressure: jnp.ndarray = jnp.nan
    temperature: jnp.ndarray = jnp.nan
    density: jnp.ndarray = jnp.nan
    enthalpy: jnp.ndarray = jnp.nan
    internal_energy: jnp.ndarray = jnp.nan
    entropy: jnp.ndarray = jnp.nan
    isochoric_heat_capacity: jnp.ndarray = jnp.nan
    isobaric_heat_capacity: jnp.ndarray = jnp.nan
    compressibility_factor: jnp.ndarray = jnp.nan
    speed_of_sound: jnp.ndarray = jnp.nan
    speed_sound_p: jnp.ndarray = jnp.nan
    speed_sound_pT: jnp.ndarray = jnp.nan
    isothermal_compressibility: jnp.ndarray = jnp.nan
    isentropic_compressibility: jnp.ndarray = jnp.nan
    isothermal_bulk_modulus: jnp.ndarray = jnp.nan
    isentropic_bulk_modulus: jnp.ndarray = jnp.nan
    viscosity: jnp.ndarray = jnp.nan
    conductivity: jnp.ndarray = jnp.nan
    quality_mass: jnp.ndarray = jnp.nan
    quality_volume: jnp.ndarray = jnp.nan
    joule_thomson: jnp.ndarray = jnp.nan
    isothermal_joule_thomson: jnp.ndarray = jnp.nan
    
