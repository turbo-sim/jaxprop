"""Independent equilibrium responses, phase geometry and diagnostic isolation."""

import os
from pathlib import Path
import subprocess
import sys

import CoolProp.CoolProp as CP
import numpy as np
import pytest
from scipy.optimize import root, root_scalar

import jaxprop as jp
from jaxprop.coolprop import core_calculations as core
from utilities import get_available_backends

SOURCES = [
    (backend, name)
    for backend in get_available_backends()
    for name in ("Water", "CO2", "Air", "R410A")
]
CORE = ("temperature", "pressure", "density", "enthalpy", "entropy", "internal_energy")


@pytest.fixture(scope="module", params=SOURCES, ids=lambda item: "-".join(item))
def source(request):
    backend, name = request.param
    return jp.Fluid(name, backend=backend), CP.AbstractState(backend, name)


@pytest.mark.parametrize("quality", (0.05, 0.3, 0.8))
def test_actual_flash_quality_void_fraction_and_empirical_transport(source, quality):
    fluid, oracle = source
    oracle.update(CP.PQ_INPUTS, 1e6, quality)
    actual = fluid.get_state(jp.PQ_INPUTS, 1e6, quality)
    assert actual.status == 0
    np.testing.assert_allclose(actual.Q, oracle.Q(), rtol=0, atol=2e-15)
    vapor_density = oracle.saturated_vapor_keyed_output(CP.iDmass)
    liquid_density = oracle.saturated_liquid_keyed_output(CP.iDmass)
    vapor_volume = quality * oracle.rhomass() / vapor_density
    liquid_volume = (1 - quality) * oracle.rhomass() / liquid_density
    np.testing.assert_allclose(vapor_volume + liquid_volume, 1, rtol=2e-12)
    np.testing.assert_allclose(actual.quality_volume, vapor_volume, rtol=2e-12)
    np.testing.assert_allclose(
        core.calculate_trimmed_quality(fluid._AS, fluid._source_context),
        (quality, vapor_volume),
        rtol=2e-12,
    )
    assert (
        core.calculate_generalized_quality(
            fluid._AS, source_context=fluid._source_context
        )
        == quality
    )
    before = [
        getattr(fluid._AS, name)()
        for name in ("T", "p", "rhomass", "hmass", "smass", "umass", "Q")
    ]
    for component_quality in (0.0, 1.0):
        oracle.update(CP.PQ_INPUTS, 1e6, component_quality)
        if component_quality == 0:
            liquid_mu, liquid_k = core.get_viscosity(oracle), core.get_conductivity(
                oracle
            )
        else:
            vapor_mu, vapor_k = core.get_viscosity(oracle), core.get_conductivity(
                oracle
            )
    np.testing.assert_allclose(
        actual.raw_property("viscosity"),
        liquid_volume * liquid_mu + vapor_volume * vapor_mu,
        rtol=2e-10,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        actual.raw_property("conductivity"),
        liquid_volume * liquid_k + vapor_volume * vapor_k,
        rtol=2e-10,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        before,
        [
            getattr(fluid._AS, name)()
            for name in ("T", "p", "rhomass", "hmass", "smass", "umass", "Q")
        ],
    )


def _quality_at(oracle, pressure, field, target):
    """Independent scalar source solve, rather than a thermodynamic Jacobian."""

    def residual(quality):
        oracle.update(CP.PQ_INPUTS, pressure, quality)
        return getattr(oracle, field)() - target

    solution = root_scalar(residual, bracket=(0, 1), xtol=1e-13)
    assert solution.converged
    oracle.update(CP.PQ_INPUTS, pressure, solution.root)


def test_equilibrium_responses_match_independent_inverse_and_constrained_flashes(
    source,
):
    fluid, oracle = source
    pressure, quality = 1e6, 0.3
    state = fluid.get_state(jp.PQ_INPUTS, pressure, quality)
    temperature, density, h, s = map(float, (state.T, state.rho, state.h, state.s))
    dp = 50.0
    inverse = []
    for perturbed in (pressure - dp, pressure + dp):
        oracle.update(CP.PSmass_INPUTS, perturbed, s)
        rho_s = oracle.rhomass()
        oracle.update(CP.HmassP_INPUTS, h, perturbed)
        temperature_h = oracle.T()
        _quality_at(oracle, perturbed, "rhomass", density)
        inverse.append((rho_s, temperature_h, oracle.umass()))
    speed = np.sqrt(2 * dp / (inverse[1][0] - inverse[0][0]))
    jt = (inverse[1][1] - inverse[0][1]) / (2 * dp)
    gruneisen = 2 * dp / (density * (inverse[1][2] - inverse[0][2]))
    np.testing.assert_allclose(state.a, speed, rtol=3e-5)
    np.testing.assert_allclose(state.joule_thomson, jt, rtol=3e-5)
    np.testing.assert_allclose(state.gruneisen, gruneisen, rtol=3e-5)
    np.testing.assert_allclose(
        state.isentropic_bulk_modulus, density * speed**2, rtol=6e-5
    )
    np.testing.assert_allclose(
        state.isentropic_compressibility, 1 / (density * speed**2), rtol=6e-5
    )

    dt = temperature * 1e-5
    internal_energies = []
    for target in (temperature - dt, temperature + dt):
        if fluid._source_context.is_pseudo_pure:
            # Some pseudo-pure backends reject DmassT in the dome; HEOS's
            # pseudo-pure DmassT map can also differ from its PQ interpolation.
            # Solve the actual map's fixed-density thermal perturbation instead.
            def residual(coordinates):
                oracle.update(CP.PQ_INPUTS, pressure * coordinates[0], coordinates[1])
                return (oracle.T() / target - 1, oracle.rhomass() / density - 1)

            solution = root(residual, (1.0, quality), tol=1e-10)
            assert solution.success
            np.testing.assert_allclose(residual(solution.x), 0, atol=2e-10)
        else:
            oracle.update(CP.DmassT_INPUTS, density, target)
        internal_energies.append(oracle.umass())
    expected_cv = (internal_energies[1] - internal_energies[0]) / (2 * dt)
    np.testing.assert_allclose(state.cv, expected_cv, rtol=3e-5)

    if not fluid._source_context.is_pseudo_pure:
        for name in (
            "isobaric_heat_capacity",
            "heat_capacity_ratio",
            "isothermal_compressibility",
            "isothermal_bulk_modulus",
            "isobaric_expansion_coefficient",
            "isothermal_joule_thomson",
        ):
            assert not state.property_valid(name)
            with pytest.raises(jp.PropertyUnavailableError):
                state[name]
        return
    oracle.update(CP.PQ_INPUTS, pressure, 0.0)
    bubble_temperature = oracle.T()
    oracle.update(CP.PQ_INPUTS, pressure, 1.0)
    glide = oracle.T() - bubble_temperature
    # A generic T-relative step can traverse several percent of a narrow
    # glide's quality interval and measure a secant of harmonic density.
    thermal_step = min(dt, glide * 1e-4)
    values = []
    for target in (temperature - thermal_step, temperature + thermal_step):
        _quality_at(oracle, pressure, "T", target)
        values.append((oracle.hmass(), oracle.rhomass()))
    cp = (values[1][0] - values[0][0]) / (2 * thermal_step)
    alpha = -(values[1][1] - values[0][1]) / (2 * thermal_step * density)
    fixed_temperature = []
    constrained_dp = 0.5
    for perturbed in (pressure - constrained_dp, pressure + constrained_dp):
        _quality_at(oracle, perturbed, "T", temperature)
        fixed_temperature.append((oracle.rhomass(), oracle.hmass()))
    kappa_t = (fixed_temperature[1][0] - fixed_temperature[0][0]) / (
        2 * constrained_dp * density
    )
    mu_t = (fixed_temperature[1][1] - fixed_temperature[0][1]) / (2 * constrained_dp)
    np.testing.assert_allclose(state.cp, cp, rtol=3e-5)
    np.testing.assert_allclose(state.isobaric_expansion_coefficient, alpha, rtol=3e-5)
    np.testing.assert_allclose(state.isothermal_compressibility, kappa_t, rtol=3e-5)
    np.testing.assert_allclose(state.isothermal_joule_thomson, mu_t, rtol=3e-5)


@pytest.mark.parametrize("quality", (0.0, 1.0))
def test_exact_saturation_endpoints_recover_homogeneous_response_coefficients(
    source, quality
):
    fluid, oracle = source
    oracle.update(CP.PQ_INPUTS, 1e6, quality)
    state = fluid.get_state(jp.PQ_INPUTS, 1e6, quality)
    assert state.status == 0 and state.Q == quality and state.quality_volume == quality
    assert state.is_two_phase == 1
    expected = {
        "isobaric_heat_capacity": oracle.cpmass(),
        "isochoric_heat_capacity": oracle.cvmass(),
        "speed_of_sound": oracle.speed_sound(),
        "isothermal_compressibility": oracle.isothermal_compressibility(),
        "isobaric_expansion_coefficient": oracle.isobaric_expansion_coefficient(),
    }
    for name, value in expected.items():
        np.testing.assert_allclose(
            state.raw_property(name), value, rtol=2e-10, err_msg=name
        )


@pytest.mark.parametrize(
    "function",
    (
        core.compute_gruneisen,
        core.compute_joule_thomson,
        core.compute_isothermal_joule_thomson,
    ),
)
def test_public_derivative_helpers_preserve_configured_and_standalone_sources(
    source, function
):
    fluid, oracle = source
    for target, context in ((fluid._AS, fluid._source_context), (oracle, None)):
        target.update(CP.PQ_INPUTS, 1e6, 0.3)
        before = tuple(
            getattr(target, name)()
            for name in ("p", "T", "rhomass", "hmass", "smass", "umass", "Q")
        )
        result = function(target, source_context=context)
        if (
            function is core.compute_isothermal_joule_thomson
            and not fluid._source_context.is_pseudo_pure
        ):
            assert np.isnan(result)
        else:
            assert np.isfinite(result)
        np.testing.assert_allclose(
            [
                getattr(target, name)()
                for name in ("p", "T", "rhomass", "hmass", "smass", "umass", "Q")
            ],
            before,
            rtol=2e-10,
            atol=1e-8,
        )


def test_generalized_quality_is_gauge_invariant_and_oriented_by_entropy(tmp_path):
    script = r"""
import CoolProp.CoolProp as CP
import numpy as np
import jaxprop as jp
from jaxprop.coolprop import core_calculations as core
from jaxprop.coolprop.source_context import helper_state
CP.set_reference_state("CO2", "DEF")
old = jp.Fluid("CO2")
temperatures = (330.0, 400.0)
densities = (300.0, 467.6, 650.0)
def samples(fluid):
    values = []
    for T in temperatures:
        row=[]
        for rho in densities:
            row.append(float(fluid.get_state(jp.DmassT_INPUTS,rho,T,generalize_quality=True).Q))
        assert row[0] > row[1] > row[2], row
        values.extend(row)
    return values
before = samples(old)
def responses(fluid):
    state = fluid.get_state(jp.PQ_INPUTS,1e6,.3)
    return [float(state.raw_property(field)) for field in
            ('isochoric_heat_capacity','speed_of_sound','gruneisen',
             'joule_thomson','quality_volume')]
before_response = responses(old)
identity = old.reference_id
CP.set_reference_state("CO2",300.,500.,12345.,1e6)
fresh = jp.Fluid("CO2")
np.testing.assert_allclose(samples(old), before, rtol=0, atol=2e-9)
np.testing.assert_allclose(samples(fresh), before, rtol=0, atol=2e-9)
np.testing.assert_allclose(responses(old), before_response, rtol=0, atol=1e-10)
np.testing.assert_allclose(responses(fresh), before_response, rtol=3e-5, atol=1e-10)
assert fresh.reference_id != identity
old._AS.update(CP.PT_INPUTS,1e5,400.)
try:
    helper_state(old._AS)
except jp.InvalidConfigurationError as error:
    assert "reference convention" in str(error)
else:
    raise AssertionError("Standalone mismatching helper was silently accepted")
print("entropy-gauge and standalone-reference checks passed")
"""
    environment = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "entropy-gauge and standalone-reference checks passed" in result.stdout


@pytest.mark.parametrize("name", ("Water", "CO2"))
@pytest.mark.parametrize("quality", (1e-8, 1.0 - 1e-8))
def test_interior_sound_has_no_arbitrary_endpoint_quality_cutoff(name, quality):
    fluid = jp.Fluid(name)
    state = fluid.get_state(jp.PQ_INPUTS, 1e6, quality)
    oracle = CP.AbstractState("HEOS", name)
    compressibility = 0.0
    for endpoint in (0.0, 1.0):
        oracle.update(CP.PQ_INPUTS, 1e6, endpoint)
        density, cp, speed = oracle.rhomass(), oracle.cpmass(), oracle.speed_sound()
        volume = (quality if endpoint else 1 - quality) * float(state.rho) / density
        ds_dp = oracle.first_saturation_deriv(CP.iSmass, CP.iP)
        compressibility += (
            volume / (density * speed**2)
            + float(state.T) * volume * density * ds_dp**2 / cp
        )
    expected = np.sqrt(1 / (float(state.rho) * compressibility))
    np.testing.assert_allclose(state.a, expected, rtol=3e-7)
    assert state.Q == quality


@pytest.mark.parametrize("name,two_phase", (("Water", False), ("Air", True)))
@pytest.mark.parametrize(
    "function",
    (
        core.compute_gruneisen,
        core.compute_joule_thomson,
        core.compute_isothermal_joule_thomson,
    ),
)
def test_failed_standalone_finite_difference_restores_original_source(
    name, two_phase, function
):
    oracle = CP.AbstractState("HEOS", name)
    (
        oracle.update(CP.PQ_INPUTS, 1e6, 0.3)
        if two_phase
        else oracle.update(CP.PT_INPUTS, 1e5, 500.0)
    )
    fields = ("p", "T", "rhomass", "hmass", "smass", "umass", "Q")
    expected = [getattr(oracle, field)() for field in fields]

    class FailedPerturbation:
        count = 0

        def first_partial_deriv(self, *args):
            raise NotImplementedError("Exercise the finite-difference fallback")

        def update(self, *args):
            self.count += 1
            if self.count == 2:
                raise ValueError("The second perturbed state is unavailable")
            return oracle.update(*args)

        def __getattr__(self, field):
            return getattr(oracle, field)

    wrapped = FailedPerturbation()
    assert np.isnan(function(wrapped))
    assert wrapped.count == 3
    np.testing.assert_allclose(
        [getattr(oracle, field)() for field in fields], expected, rtol=2e-10, atol=1e-8
    )


@pytest.mark.parametrize("backend", get_available_backends())
def test_saturation_entropy_derivative_restores_standalone_state(backend):
    oracle = CP.AbstractState(backend, "Water")
    oracle.update(CP.PT_INPUTS, 1e5, 500.0)
    fields = ("p", "T", "rhomass", "hmass", "smass", "umass", "Q")
    before = [getattr(oracle, field)() for field in fields]
    assert np.isfinite(core.compute_dsdp_q(oracle, 1e6, 0.0))
    np.testing.assert_allclose(
        [getattr(oracle, field)() for field in fields], before, rtol=2e-10, atol=1e-8
    )


@pytest.mark.parametrize("backend", get_available_backends())
def test_standalone_homogeneous_snapshot_uses_raw_source_only_for_reference_verification(
    backend,
):
    fluid = jp.Fluid("Water", backend=backend)
    standalone = CP.AbstractState(backend, "Water")
    expected = core.compute_properties_metastable_rhoT(
        fluid._AS,
        6.0,
        450.0,
        generalize_quality=True,
        supersaturation=True,
        source_context=fluid._source_context,
    )
    actual = core.compute_properties_metastable_rhoT(
        standalone,
        6.0,
        450.0,
        generalize_quality=True,
        supersaturation=True,
    )
    assert standalone.phase() == CP.iphase_twophase
    assert actual["is_two_phase"] == 0
    assert actual.keys() == expected.keys()
    assert len(actual) == 31
    for field in (
        *CORE,
        "quality_mass",
        "subcooling",
        "superheating",
        "pressure_saturation",
        "temperature_saturation",
    ):
        np.testing.assert_allclose(
            actual[field],
            expected[field],
            rtol=2e-8,
            atol=1e-8,
            equal_nan=True,
            err_msg=field,
        )


def test_requested_diagnostics_share_samples_and_keep_default_calls_lean(monkeypatch):
    fluid = jp.Fluid("CO2")
    scratch = fluid._source_context.auxiliary_state
    updates = []

    class Counted:
        def update(self, *args):
            updates.append(args)
            return scratch.update(*args)

        def __getattr__(self, name):
            return getattr(scratch, name)

    monkeypatch.setattr(fluid._source_context, "auxiliary_state", Counted())
    fluid.get_state(jp.PT_INPUTS, 1e6, 270.0)
    assert not updates
    result = fluid.get_state(jp.PT_INPUTS, 1e6, 270.0, supersaturation=True)
    assert result.status == 0
    assert updates == [
        (CP.QT_INPUTS, 0.0, 270.0),
        (CP.PQ_INPUTS, float(result.p), 0.0),
        (CP.PQ_INPUTS, float(result.p), 1.0),
    ]
    updates.clear()
    supercritical = fluid.get_state(jp.PT_INPUTS, 1e7, 400.0, supersaturation=True)
    assert updates == [
        (
            CP.DmassP_INPUTS,
            fluid._source_context.critical_density,
            float(supercritical.p),
        )
    ]
