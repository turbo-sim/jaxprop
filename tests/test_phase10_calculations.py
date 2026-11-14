"""Independent homogeneous, inverse and constitutive-blending checks."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import CoolProp.CoolProp as CP
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.coolprop import core_calculations as core
from utilities import get_available_backends

BACKENDS = get_available_backends()


@pytest.mark.parametrize(
    "direction", ("evaporation", "flashing", "cavitation", "condensation")
)
def test_blending_recovers_endpoints_optional_fields_and_canonical_schema(direction):
    coordinate = np.array([-0.2, 0.0, 0.5, 1.0, 1.2])
    quality = 2.0 - coordinate if direction != "condensation" else coordinate - 1.0
    equilibrium = {
        "Q": quality,
        "h": np.full(5, 10.0),
        "sigma": np.full(5, 0.07),
        "is_two_phase": np.ones(5),
    }
    metastable = {
        "quality_mass": np.full(5, 1.0),
        "enthalpy": np.full(5, 20.0),
        "surface_tension": np.full(5, np.nan),
        "is_two_phase": np.zeros(5),
        "conductivity": np.full(5, 0.1),
    }
    actual = core.blend_properties(
        direction,
        equilibrium,
        metastable,
        "vapor_quality",
        0.0 if direction == "condensation" else 1.0,
        1.0,
    )
    limited = np.clip(coordinate, 0, 1)
    weight = 3 * limited**2 - 2 * limited**3
    np.testing.assert_allclose(actual["enthalpy"], 10 + 10 * weight)
    np.testing.assert_allclose(actual["surface_tension"][:2], 0.07)
    assert np.all(np.isnan(actual["surface_tension"][2:]))
    assert np.isnan(actual["conductivity"][0])
    np.testing.assert_allclose(actual["conductivity"][-2:], 0.1)
    np.testing.assert_allclose(actual["is_two_phase"][:2], 1.0)
    assert np.isnan(actual["is_two_phase"][2])
    np.testing.assert_allclose(actual["is_two_phase"][-2:], 0.0)
    assert set(actual) == {
        "enthalpy",
        "quality_mass",
        "surface_tension",
        "conductivity",
        "is_two_phase",
    }


@pytest.mark.parametrize("endpoint", (0.0, 1.0))
def test_blending_transition_has_zero_endpoint_slope(endpoint):
    step = 1e-6
    values = []
    for coordinate in (endpoint - step, endpoint, endpoint + step):
        state = core.blend_properties(
            "evaporation", {"Q": 2 - coordinate, "h": 10.0}, {"h": 20.0}, "Q", 1.0, 1.0
        )
        values.append(float(state["enthalpy"]))
    assert abs((values[2] - values[0]) / (2 * step)) < 2e-5


@pytest.mark.parametrize(
    "settings",
    (
        {"phase_change": "melting"},
        {"blending_width": 0.0},
        {"blending_width": -1.0},
        {"blending_width": np.nan},
        {"blending_width": True},
        {"blending_width": [1.0]},
        {"blending_onset": np.inf},
        {"blending_variable": "unknown"},
    ),
)
def test_invalid_blending_settings_fail_explicitly(settings):
    options = dict(
        phase_change="evaporation",
        props_equilibrium={"Q": 0.3, "h": 10.0},
        props_metastable={"Q": 0.0, "h": 20.0},
        blending_variable="Q",
        blending_onset=1.0,
        blending_width=0.2,
    )
    options.update(settings)
    with pytest.raises((jp.InvalidConfigurationError, jp.UnsupportedPropertyError)):
        core.blend_properties(**options)


@pytest.mark.parametrize(
    "properties",
    ({"unknown": 1.0}, {"Q": 0.3, "h": 10.0, "enthalpy": 11.0}, {"Q": "bad"}),
)
def test_invalid_blending_property_mappings_fail_explicitly(properties):
    with pytest.raises(jp.JaxPropError):
        core.blend_properties("evaporation", properties, {"h": 10.0}, "Q", 1.0, 0.2)


@pytest.fixture(scope="module", params=BACKENDS)
def fluid(request):
    return jp.Fluid("CO2", backend=request.param)


@pytest.mark.parametrize("algorithm", ("hybr", "lm"))
@pytest.mark.parametrize("pair", (("h", "p"), ("p", "s")))
def test_homogeneous_inverse_recovers_independent_stable_source(fluid, algorithm, pair):
    reference = fluid.get_state(jp.PT_INPUTS, 8e6, 400.0)
    actual = core.compute_properties(
        fluid._AS,
        pair[0],
        float(reference[pair[0]]),
        pair[1],
        float(reference[pair[1]]),
        "metastable",
        rhoT_guess_metastable=[1.02 * float(reference.rho), 402.0],
        solver_algorithm=algorithm,
        solver_tolerance=1e-9,
        source_context=fluid._source_context,
    )
    for name in (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    ):
        np.testing.assert_allclose(
            actual[name], reference.raw_property(name), rtol=3e-9, atol=2e-7
        )


@pytest.mark.parametrize("branch,temperature_offset", (("liquid", 2.0), ("gas", -2.0)))
@pytest.mark.parametrize("algorithm", ("hybr", "lm"))
def test_actual_metastable_branches_match_imposed_phase_eos(
    branch, temperature_offset, algorithm
):
    fluid = jp.Fluid("Water", backend="HEOS")
    pressure = 1e5 if branch == "gas" else 1e6
    source = CP.AbstractState("HEOS", "Water")
    source.update(CP.PQ_INPUTS, pressure, float(branch == "gas"))
    temperature = source.T() + temperature_offset
    source.specify_phase(CP.iphase_gas if branch == "gas" else CP.iphase_liquid)
    source.update(CP.PT_INPUTS, pressure, temperature)
    expected = {
        "pressure": source.p(),
        "temperature": source.T(),
        "density": source.rhomass(),
        "enthalpy": source.hmass(),
        "entropy": source.smass(),
        "internal_energy": source.umass(),
    }
    direct = core.compute_properties_metastable_rhoT(
        fluid._AS, source.rhomass(), temperature, source_context=fluid._source_context
    )
    solved = core.compute_properties(
        fluid._AS,
        "h",
        expected["enthalpy"],
        "p",
        pressure,
        "metastable",
        rhoT_guess_metastable=[source.rhomass() * 1.0001, temperature + 0.01],
        solver_algorithm=algorithm,
        solver_tolerance=1e-9,
        source_context=fluid._source_context,
    )
    for actual in (direct, solved):
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name], value, rtol=2e-8, atol=5e-5)
        assert not actual["is_two_phase"]
        np.testing.assert_allclose(
            actual["isothermal_bulk_modulus"] * actual["isothermal_compressibility"],
            1.0,
            rtol=2e-14,
        )
        np.testing.assert_allclose(
            actual["isobaric_heat_capacity"] - actual["isochoric_heat_capacity"],
            temperature
            * actual["isobaric_expansion_coefficient"] ** 2
            / (actual["density"] * actual["isothermal_compressibility"]),
            rtol=2e-11,
        )


def test_homogeneous_direct_aliases_need_no_guess_and_no_runtime_source_construction(
    fluid, monkeypatch
):
    reference = fluid.get_state(jp.PT_INPUTS, 8e6, 400.0)
    fluid._AS.update(CP.DmassT_INPUTS, float(reference.rho), 400.0)
    expected_enthalpy = fluid._AS.hmass()

    def forbidden(*args, **kwargs):
        raise AssertionError("Source construction entered homogeneous evaluation")

    monkeypatch.setattr(CP, "AbstractState", forbidden)
    actual = core.compute_properties(
        fluid._AS,
        "dmass",
        float(reference.rho),
        "temperature",
        400.0,
        "metastable",
        source_context=fluid._source_context,
    )
    np.testing.assert_allclose(actual["enthalpy"], expected_enthalpy, rtol=2e-12)
    assert "subcooling" not in actual


@pytest.mark.parametrize(
    "settings",
    (
        {"solver_algorithm": "unknown"},
        {"solver_tolerance": 0.0},
        {"solver_tolerance": np.nan},
        {"solver_max_iterations": 0},
        {"solver_max_iterations": 1.5},
        {"rhoT_guess_metastable": None},
        {"rhoT_guess_metastable": [-1.0, 400.0]},
        {"rhoT_guess_metastable": [100.0]},
    ),
)
def test_invalid_inverse_settings_are_configuration_failures(fluid, settings):
    options = dict(rhoT_guess_metastable=[100.0, 400.0])
    options.update(settings)
    with pytest.raises(jp.InvalidConfigurationError):
        core.compute_properties(
            fluid._AS,
            "h",
            5e5,
            "p",
            8e6,
            "metastable",
            source_context=fluid._source_context,
            **options
        )


def test_stationary_solver_success_cannot_fabricate_a_thermodynamic_root(
    fluid, monkeypatch
):
    class FalseSuccess:
        success = True
        message = "stationary least squares"

        def __init__(self, *args, **kwargs):
            pass

        def solve(self, initial):
            return initial

    monkeypatch.setattr(core.psv, "NonlinearSystemSolver", FalseSuccess)
    with pytest.raises(jp.ConvergenceError, match="validated thermodynamic root"):
        core.compute_properties(
            fluid._AS,
            "T",
            700.0,
            "p",
            8e6,
            "metastable",
            rhoT_guess_metastable=[100.0, 400.0],
            source_context=fluid._source_context,
        )


def test_zero_property_targets_have_finite_residuals():
    problem = core._FlashCalculationResidual(
        "quality_mass",
        0.0,
        "enthalpy",
        0.0,
        lambda rho, T: {"quality_mass": rho, "enthalpy": T},
        1.0,
        1.0,
    )
    np.testing.assert_array_equal(problem.residual([0.0, 0.0]), [0.0, 0.0])
    np.testing.assert_array_equal(problem.residual([0.1, -0.2]), [0.1, -0.2])


def test_invalid_internal_iterate_is_nonconvergence_not_invalid_caller_input(
    fluid, monkeypatch
):
    class InvalidIterate:
        success = False
        message = "invalid internal point"

        def __init__(self, problem, **kwargs):
            self.problem = problem

        def solve(self, initial):
            return self.problem.residual([-1.0, 1.0])

    monkeypatch.setattr(core.psv, "NonlinearSystemSolver", InvalidIterate)
    with pytest.raises(jp.ConvergenceError) as captured:
        core.compute_properties(
            fluid._AS,
            "h",
            5e5,
            "p",
            8e6,
            "metastable",
            rhoT_guess_metastable=[100.0, 400.0],
            source_context=fluid._source_context,
        )
    assert isinstance(captured.value.__cause__, jp.InvalidStateError)


@pytest.mark.parametrize(
    "property_name", ("speed_of_sound_p", "mass_frac_1", "subcooling")
)
def test_known_unsupported_input_property_fails_before_a_solver_starts(
    fluid, property_name, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported input reached numerical solver construction")

    monkeypatch.setattr(core.psv, "NonlinearSystemSolver", forbidden)
    with pytest.raises(jp.UnsupportedPropertyError) as captured:
        core.compute_properties(
            fluid._AS,
            property_name,
            1.0,
            "T",
            400.0,
            "metastable",
            rhoT_guess_metastable=[100.0, 400.0],
            source_context=fluid._source_context,
        )
    assert captured.value.request == property_name


def test_homogeneous_spinodal_preserves_finite_response_limits():
    class SpinodalEOS:
        """At delta=tau=1 choose D=0, B=1, C=-1, and R/M=1."""

        def update(self, *args):
            pass

        def gas_constant(self):
            return 1.0

        def molar_mass(self):
            return 1.0

        def tau(self):
            return 1.0

        delta = tau

        def alpha0(self):
            return 0.0

        alphar = alpha0
        dalpha0_dTau = alpha0
        dalphar_dTau = alpha0
        dalpha0_dDelta = alpha0
        d2alpha0_dTau2 = alpha0
        d2alpha0_dDelta2 = alpha0
        d2alpha0_dDelta_dTau = alpha0
        d2alphar_dDelta_dTau = alpha0

        def dalphar_dDelta(self):
            return 1.0

        def d2alphar_dTau2(self):
            return -1.0

        def d2alphar_dDelta2(self):
            return -2.0

    state = core.compute_properties_metastable_rhoT(
        SpinodalEOS(), 1.0, 300.0, source_context=SimpleNamespace(critical_entropy=0.0)
    )
    assert state["isothermal_bulk_modulus"] == 0.0
    assert np.isnan(state["isobaric_heat_capacity"])
    assert np.isnan(state["isothermal_compressibility"])
    assert state["gruneisen"] == 1.0
    assert state["joule_thomson"] == 1.0
    np.testing.assert_allclose(state["speed_of_sound"] ** 2, 300.0, rtol=2e-15)


def test_actual_blending_returns_canonical_complete_properties():
    fluid = jp.Fluid("Water", backend="HEOS")
    source = CP.AbstractState("HEOS", "Water")
    source.update(CP.PQ_INPUTS, 1e6, 0.0)
    temperature = source.T() + 2.0
    source.specify_phase(CP.iphase_liquid)
    source.update(CP.PT_INPUTS, 1e6, temperature)
    equilibrium = fluid.get_state(jp.HmassP_INPUTS, source.hmass(), 1e6)
    blended, _, metastable = core.compute_properties(
        fluid._AS,
        "h",
        source.hmass(),
        "p",
        1e6,
        "blending",
        rhoT_guess_equilibrium=[float(equilibrium.rho), float(equilibrium.T)],
        rhoT_guess_metastable=[source.rhomass(), temperature],
        phase_change="flashing",
        blending_variable="Q",
        blending_onset=0.05,
        blending_width=0.05,
        solver_tolerance=1e-9,
        source_context=fluid._source_context,
    )
    assert "x" not in blended and "sigma" not in blended
    assert "surface_tension" in blended and len(blended) == 25
    assert metastable["quality_mass"] == 0.0
    np.testing.assert_allclose(blended["pressure"], 1e6, rtol=2e-9)
    np.testing.assert_allclose(blended["enthalpy"], source.hmass(), rtol=2e-9)


def test_reference_shift_does_not_loosen_homogeneous_flash_accuracy():
    script = r"""
import json
import CoolProp.CoolProp as CP
import numpy as np
import jaxprop as jp
from jaxprop.coolprop import core_calculations as core
name = "CO2"
source = CP.AbstractState("HEOS", name)
source.update(CP.PT_INPUTS, 8e6, 400.0)
rho, h, s, M = source.rhomass(), source.hmass(), source.smass(), source.molar_mass()
records = []
try:
    for shift in (0.0, 1e8, -h):
        CP.set_reference_state(name, 400.0, rho/M, (h+shift)*M, s*M)
        fluid = jp.Fluid(name, backend="HEOS")
        state = core.compute_properties(fluid._AS, "h", h+shift, "p", 8e6, "metastable", rhoT_guess_metastable=[rho*1.02, 402.0], solver_tolerance=1e-9, source_context=fluid._source_context)
        np.testing.assert_allclose([state["density"], state["temperature"], state["pressure"]], [rho, 400.0, 8e6], rtol=2e-9)
        assert abs(state["enthalpy"]-(h+shift)) < 1e-4
        records.append(state["temperature"])
finally:
    CP.set_reference_state(name, "DEF")
print(json.dumps(records))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(json.loads(result.stdout.strip().splitlines()[-1])) == 3
