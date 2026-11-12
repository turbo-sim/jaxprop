"""Fixed public-contract cases, independent of evaluator dispatch/capabilities.

The legacy constructors let conformance tests exercise existing evaluators while
the common factory is implemented. They do not adapt signatures or state data.
"""

from dataclasses import dataclass

import jaxprop as jxp

CORE_PROPERTIES = (
    "pressure",
    "temperature",
    "density",
    "enthalpy",
    "entropy",
    "internal_energy",
)
SHAPE_PROPERTIES = CORE_PROPERTIES[:-1]
PERFECT_GAS_PAIRS = (
    "PT_INPUTS",
    "HmassSmass_INPUTS",
    "HmassP_INPUTS",
    "PSmass_INPUTS",
    "DmassHmass_INPUTS",
    "DmassP_INPUTS",
)
BICUBIC_PAIRS = PERFECT_GAS_PAIRS + ("DmassT_INPUTS", "DmassSmass_INPUTS")
SOURCE_PAIRS = BICUBIC_PAIRS + ("PQ_INPUTS", "QT_INPUTS")
BINARY_PAIRS = ("PT_INPUTS", "HmassP_INPUTS", "PSmass_INPUTS")

# Input order is specified independently of jaxprop.INPUT_PAIR_MAP.
PAIR_FIELDS = {
    "PT_INPUTS": ("pressure", "temperature"),
    "HmassSmass_INPUTS": ("enthalpy", "entropy"),
    "HmassP_INPUTS": ("enthalpy", "pressure"),
    "PSmass_INPUTS": ("pressure", "entropy"),
    "DmassHmass_INPUTS": ("density", "enthalpy"),
    "DmassP_INPUTS": ("density", "pressure"),
    "DmassT_INPUTS": ("density", "temperature"),
    "DmassSmass_INPUTS": ("density", "entropy"),
}


@dataclass(frozen=True)
class ContractCase:
    id: str
    provider: str
    model: str
    current_class: str
    name: str
    required_pairs: tuple[str, ...]


CONTRACT_CASES = (
    ContractCase("coolprop", "coolprop", "real_fluid", "Fluid", "CO2", SOURCE_PAIRS),
    ContractCase(
        "coolprop_jax", "coolprop_jax", "real_fluid", "FluidJAX", "CO2", SOURCE_PAIRS
    ),
    ContractCase(
        "bicubic", "bicubic", "real_fluid", "FluidBicubic", "CO2", BICUBIC_PAIRS
    ),
    ContractCase(
        "perfect_gas",
        "native",
        "perfect_gas",
        "FluidPerfectGas",
        "CO2",
        PERFECT_GAS_PAIRS,
    ),
    ContractCase(
        "binary_mixture",
        "native",
        "binary_mixture",
        "FluidMix",
        "water_nitrogen_mixture",
        BINARY_PAIRS,
    ),
)


def build_current_fluid(case, table_dir, identifier=None):
    """Construct the currently exported class without adapting its interface."""
    if case.id == "bicubic":
        return jxp.FluidBicubic(
            fluid_name=case.name,
            backend="HEOS",
            identifier=identifier,
            h_min=600e3,
            h_max=850e3,
            p_min=5e6,
            p_max=12e6,
            N_h=12,
            N_p=12,
            coarse_step=1,
            table_dir=str(table_dir),
        )
    if case.id == "perfect_gas":
        return jxp.FluidPerfectGas(
            name=case.name,
            identifier=identifier,
            T_ref=600.0,
            p_ref=1e5,
        )
    if case.id == "binary_mixture":
        return jxp.FluidMix(
            name=case.name,
            backend="HEOS",
            identifier=identifier,
            mixture_ratio=0.1,
        )
    constructor = getattr(jxp, case.current_class)
    # FluidJAX does not yet accept identifier, including an explicit None.
    if case.id == "coolprop_jax" and identifier is None:
        return constructor(name=case.name, backend="HEOS")
    return constructor(name=case.name, backend="HEOS", identifier=identifier)


def factory_arguments(case, table_dir):
    """Common-factory configuration for the selected physical model."""
    arguments = dict(
        name=case.name,
        provider=case.provider,
        model=case.model,
        backend=None if case.model == "binary_mixture" else "HEOS",
        identifier="working_fluid",
    )
    if case.id == "bicubic":
        arguments["table_options"] = dict(
            h_min=600e3,
            h_max=850e3,
            p_min=5e6,
            p_max=12e6,
            N_h=12,
            N_p=12,
            coarse_step=1,
            table_dir=str(table_dir),
        )
    if case.id == "perfect_gas":
        arguments["model_options"] = dict(T_ref=600.0, p_ref=1e5)
    if case.id == "binary_mixture":
        arguments["model_options"] = dict(
            components=(jxp.Fluid("Water"), jxp.Fluid("Nitrogen")),
            mass_fractions=(1 / 11, 10 / 11),
        )
    return arguments
