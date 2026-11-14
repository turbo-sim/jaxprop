"""Preserve homogeneous branches when equilibrium HP anchors are mixtures."""

import CoolProp.CoolProp as CP
import numpy as np
import pytest

import jaxprop as jp


def test_actual_nitrogen_liquid_seed_avoids_the_equilibrium_bulk_density():
    pressure, enthalpy = 108118.07510766084, -120000.0
    source = jp.FluidCoolProp("Nitrogen")
    ordinary = source.get_state(jp.HmassP_INPUTS, enthalpy, pressure)
    assert ordinary.raw_property("is_two_phase") == 1
    assert 0 < ordinary.Q < 1
    guess = jp.FluidBicubic._metastable_initial_guess(
        source, enthalpy, pressure, "liquid"
    )
    result = source.get_state_metastable(
        "p",
        pressure,
        "h",
        enthalpy,
        rhoT_guess=guess,
        solver_tolerance=1e-10,
        supersaturation=False,
        generalize_quality=False,
    )
    # Independent homogeneous-source evaluation, with the phase constraint
    # preventing CoolProp from replacing this point with an equilibrium bulk.
    raw = CP.AbstractState("HEOS", "Nitrogen")
    raw.specify_phase(CP.iphase_liquid)
    raw.update(CP.DmassT_INPUTS, float(result.rho), float(result.T))
    assert 1 / raw.isothermal_compressibility() > 0
    np.testing.assert_allclose([raw.p(), raw.hmass()], [pressure, enthalpy], rtol=1e-9)
    assert float(result.rho) > 1.5 * float(ordinary.rho)
    np.testing.assert_allclose(
        [result.rho, result.T], [801.5782558, 78.34016386], rtol=2e-9
    )


def test_actual_nitrogen_vapor_seed_starts_at_dew_endpoint_inside_the_dome():
    source = jp.FluidCoolProp("Nitrogen")
    pressure = 1e6
    dew = source.get_state(jp.PQ_INPUTS, pressure, 1.0)
    enthalpy = float(dew.h) - 500.0
    ordinary = source.get_state(jp.HmassP_INPUTS, enthalpy, pressure)
    assert ordinary.raw_property("is_two_phase") == 1 and 0 < ordinary.Q < 1
    guess = jp.FluidBicubic._metastable_initial_guess(
        source, enthalpy, pressure, "vapor"
    )
    np.testing.assert_allclose(guess, [dew.rho, dew.T], rtol=1e-12)
    result = source.get_state_metastable(
        "p",
        pressure,
        "h",
        enthalpy,
        rhoT_guess=guess,
        solver_tolerance=1e-10,
        supersaturation=False,
        generalize_quality=False,
    )
    raw = CP.AbstractState("HEOS", "Nitrogen")
    raw.specify_phase(CP.iphase_gas)
    raw.update(CP.DmassT_INPUTS, float(result.rho), float(result.T))
    assert 1 / raw.isothermal_compressibility() > 0
    np.testing.assert_allclose([raw.p(), raw.hmass()], [pressure, enthalpy], rtol=1e-9)
    assert float(result.T) < float(dew.T)


def test_liquid_columns_reuse_validated_homogeneous_anchor_rows(tmp_path, monkeypatch):
    calls = []
    original = jp.FluidCoolProp.get_state

    def observed(self, pair, first, second, **options):
        if pair == jp.HmassP_INPUTS:
            calls.append((float(first), float(second)))
        return original(self, pair, first, second, **options)

    monkeypatch.setattr(jp.FluidCoolProp, "get_state", observed)
    table = jp.FluidBicubic(
        "Nitrogen",
        "HEOS",
        -120e3,
        -119e3,
        8e4,
        2e5,
        3,
        5,
        N_p_sat=3,
        coarse_step=1,
        gradient_method="forward",
        metastable_phase="liquid",
        table_dir=str(tmp_path),
    )
    assert len(calls) == 1  # initial seed; subsequent columns use known branch data
    assert np.all(np.asarray(table.table["isothermal_bulk_modulus"]["value"]) > 0)
    raw = CP.AbstractState("HEOS", "Nitrogen")
    raw.specify_phase(CP.iphase_liquid)
    for index in np.ndindex(3, 5):
        rho = float(table.table["density"]["value"][index])
        temperature = float(table.table["temperature"]["value"][index])
        raw.update(CP.DmassT_INPUTS, rho, temperature)
        expected_h = float(table.table["enthalpy"]["value"][index])
        expected_p = float(table.table["pressure"]["value"][index])
        np.testing.assert_allclose(
            [raw.hmass(), raw.p()], [expected_h, expected_p], rtol=1e-9
        )


def test_vapor_columns_reuse_validated_homogeneous_anchor_rows(tmp_path, monkeypatch):
    raw = CP.AbstractState("HEOS", "Nitrogen")
    raw.update(CP.PQ_INPUTS, 1e6, 1.0)
    upper_h = raw.hmass() - 250.0
    calls = []
    original = jp.FluidCoolProp.get_state

    def observed(self, pair, first, second, **options):
        if pair == jp.HmassP_INPUTS:
            calls.append((float(first), float(second)))
        return original(self, pair, first, second, **options)

    monkeypatch.setattr(jp.FluidCoolProp, "get_state", observed)
    table = jp.FluidBicubic(
        "Nitrogen",
        "HEOS",
        upper_h - 500.0,
        upper_h,
        9e5,
        1e6,
        3,
        3,
        N_p_sat=3,
        coarse_step=1,
        gradient_method="forward",
        metastable_phase="vapor",
        table_dir=str(tmp_path),
    )
    assert len(calls) == 1
    assert np.all(np.asarray(table.table["isothermal_bulk_modulus"]["value"]) > 0)
