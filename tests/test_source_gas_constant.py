"""Source properties must use the selected EOS's own gas constant."""

import CoolProp.CoolProp as CP
import numpy as np
import pytest

from jaxprop.coolprop.core_calculations import (
    compute_properties_1phase,
    compute_properties_2phase,
    compute_properties_metastable_rhoT,
)

BACKENDS = ("HEOS", "REFPROP")
SINGLE_PHASE_CASES = (
    ("Water", 300.0, 1e6),
    ("Water", 500.0, 1e6),
    ("Water", 750.0, 35e6),
    ("CO2", 280.0, 8e6),
    ("CO2", 400.0, 8e6),
    ("CO2", 600.0, 1e5),
    ("Nitrogen", 100.0, 1e6),
    ("Nitrogen", 300.0, 1e5),
    ("Nitrogen", 170.0, 7e6),
    ("R134a", 300.0, 2e6),
    ("R134a", 350.0, 2e5),
    ("R134a", 410.0, 7e6),
)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,temperature,pressure", SINGLE_PHASE_CASES)
def test_ordinary_compressibility_matches_selected_source(
    backend, name, temperature, pressure
):
    state = CP.AbstractState(backend, name)
    state.update(CP.PT_INPUTS, pressure, temperature)
    state.update(CP.DmassT_INPUTS, state.rhomass(), temperature)
    expected = state.compressibility_factor()
    actual = compute_properties_1phase(state)["compressibility_factor"]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["Water", "CO2", "Nitrogen", "R134a"])
def test_two_phase_compressibility_uses_selected_source_gas_constant(backend, name):
    state = CP.AbstractState(backend, name)
    pressure = 0.5 * state.p_critical()
    state.update(CP.PQ_INPUTS, pressure, 0.4)
    expected = state.p() / (
        state.rhomass() * (state.gas_constant() / state.molar_mass()) * state.T()
    )
    actual = compute_properties_2phase(state)["compressibility_factor"]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,temperature,pressure", SINGLE_PHASE_CASES)
def test_homogeneous_helmholtz_properties_match_actual_source_EOS(
    backend, name, temperature, pressure
):
    state = CP.AbstractState(backend, name)
    state.update(CP.PT_INPUTS, pressure, temperature)
    density = state.rhomass()
    # Compare at the identical independent rho/T coordinates: reconstructing a
    # stiff liquid from rounded rho may otherwise move pressure by a few ULPs
    # of its bulk modulus rather than ULPs of its much smaller net pressure.
    state.update(CP.DmassT_INPUTS, density, temperature)
    expected = dict(
        temperature=state.T(),
        pressure=state.p(),
        density=state.rhomass(),
        enthalpy=state.hmass(),
        internal_energy=state.umass(),
        entropy=state.smass(),
        isobaric_heat_capacity=state.cpmass(),
        isochoric_heat_capacity=state.cvmass(),
        speed_of_sound=state.speed_sound(),
        compressibility_factor=state.compressibility_factor(),
        isothermal_bulk_modulus=1 / state.isothermal_compressibility(),
        isobaric_expansion_coefficient=state.isobaric_expansion_coefficient(),
    )
    actual = compute_properties_metastable_rhoT(state, density, temperature)
    for name, reference in expected.items():
        np.testing.assert_allclose(
            actual[name], reference, rtol=5e-10, atol=1e-10, err_msg=name
        )
