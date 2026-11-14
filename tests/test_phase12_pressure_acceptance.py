"""Source-root acceptance respects stiff-liquid pressure representability."""

import CoolProp.CoolProp as CP
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.coolprop import core_calculations as core

RHO = 976.9716092444803
TEMPERATURE = 344.52894981251956
PRESSURE = 100100.00000000001
ENTHALPY = 298900.0


@pytest.fixture
def source():
    return jp.Fluid("Water")


def fixed_success(monkeypatch):
    """A claimed root at the initial coordinates isolates the final check."""

    class FixedSuccess:
        success = True
        message = "synthetic optimizer termination"

        def __init__(self, problem, **options):
            self.problem = problem

        def solve(self, initial):
            return initial

    monkeypatch.setattr(core.psv, "NonlinearSystemSolver", FixedSuccess)


def evaluate(source, first, first_target, second, second_target):
    return core.compute_properties(
        source._AS,
        first,
        first_target,
        second,
        second_target,
        "metastable",
        rhoT_guess_metastable=[RHO, TEMPERATURE],
        solver_tolerance=1e-10,
        generalize_quality=False,
        supersaturation=False,
        source_context=source._source_context,
    )


def raw_point():
    raw = CP.AbstractState("HEOS", "Water")
    raw.specify_phase(CP.iphase_liquid)
    raw.update(CP.DmassT_INPUTS, RHO, TEMPERATURE)
    pressure_scale = max(
        abs(1 / raw.isothermal_compressibility()),
        abs(raw.rhomass() * raw.speed_sound() ** 2),
        raw.rhomass() * raw.gas_constant() / raw.molar_mass() * raw.T(),
        abs(raw.p()),
    )
    return raw, 128 * np.finfo(float).eps * pressure_scale


def test_actual_stiff_water_table_stencil_has_an_independently_valid_root(source):
    # This is the actual low-pressure mixed-derivative table stencil that
    # previously exceeded 5*tolerance*p by only approximately 6e-7 Pa.
    state = source.get_state_metastable(
        "h",
        ENTHALPY,
        "p",
        PRESSURE,
        rhoT_guess=[976.8504561035055, 344.7806244067203],
        solver_tolerance=1e-10,
        generalize_quality=False,
        supersaturation=False,
    )
    raw = CP.AbstractState("HEOS", "Water")
    raw.specify_phase(CP.iphase_liquid)
    raw.update(CP.DmassT_INPUTS, float(state.rho), float(state.T))
    assert state.valid and 1 / raw.isothermal_compressibility() > 0
    np.testing.assert_allclose(
        [state.p, state.h], [raw.p(), raw.hmass()], rtol=0, atol=1e-6
    )
    scale = max(
        abs(1 / raw.isothermal_compressibility()),
        abs(raw.rhomass() * raw.speed_sound() ** 2),
        raw.rhomass() * raw.gas_constant() / raw.molar_mass() * raw.T(),
        abs(raw.p()),
    )
    assert abs(float(state.p) - PRESSURE) <= max(
        5e-10 * PRESSURE, 128 * np.finfo(float).eps * scale
    )
    np.testing.assert_allclose(state.h, ENTHALPY, rtol=0, atol=1e-5)


@pytest.mark.parametrize("pressure_first", [True, False])
def test_representability_floor_applies_to_the_pressure_component_only(
    source, monkeypatch, pressure_first
):
    fixed_success(monkeypatch)
    raw, floor = raw_point()
    # The target differs by less than the physical source roundoff floor but
    # more than the old relative-to-net-pressure allowance.
    target_pressure = raw.p() + 0.9 * floor
    arguments = ("p", target_pressure, "h", raw.hmass())
    if not pressure_first:
        arguments = ("h", raw.hmass(), "p", target_pressure)
    actual = evaluate(source, *arguments)
    np.testing.assert_allclose(actual["pressure"], raw.p(), rtol=0, atol=1e-6)
    assert abs(actual["pressure"] - target_pressure) < floor


@pytest.mark.parametrize("bias_factor", [5, 1000])
def test_false_pressure_root_above_representability_budget_is_still_rejected(
    source, monkeypatch, bias_factor
):
    fixed_success(monkeypatch)
    raw, floor = raw_point()
    with pytest.raises(jp.ConvergenceError, match="validated thermodynamic root"):
        evaluate(source, "p", raw.p() + bias_factor * floor, "h", raw.hmass())


@pytest.mark.parametrize("other", ["temperature", "enthalpy"])
def test_pressure_floor_does_not_relax_the_other_requested_property(
    source, monkeypatch, other
):
    fixed_success(monkeypatch)
    raw, _ = raw_point()
    if other == "temperature":
        target = raw.T() * (1 + 6e-10)
    else:
        invariant_scale = raw.gas_constant() / raw.molar_mass() * raw.T_critical()
        target = raw.hmass() + 6e-10 * invariant_scale
    with pytest.raises(jp.ConvergenceError, match="validated thermodynamic root"):
        evaluate(source, "p", PRESSURE, other, target)


def test_nonfinite_optional_response_cannot_disable_root_validation(
    source, monkeypatch
):
    fixed_success(monkeypatch)
    raw, _ = raw_point()
    original = core.compute_properties_metastable_rhoT

    def unavailable_responses(*args, **kwargs):
        result = original(*args, **kwargs)
        result["isothermal_bulk_modulus"] = np.nan
        result["isentropic_bulk_modulus"] = np.inf
        return result

    monkeypatch.setattr(
        core, "compute_properties_metastable_rhoT", unavailable_responses
    )
    with pytest.raises(jp.ConvergenceError, match="validated thermodynamic root"):
        evaluate(source, "p", raw.p() + 1.0, "h", raw.hmass())


def test_final_validation_flashes_the_returned_root_once(source, monkeypatch):
    fixed_success(monkeypatch)
    raw, _ = raw_point()
    original = core.compute_properties_metastable_rhoT
    calls = []

    def counted(*args, **kwargs):
        calls.append((kwargs["rho"], kwargs["T"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "compute_properties_metastable_rhoT", counted)
    actual = evaluate(source, "p", PRESSURE, "h", raw.hmass())
    assert len(calls) == 1
    np.testing.assert_allclose(
        calls[0], [actual["density"], actual["temperature"]], rtol=0, atol=1e-12
    )
