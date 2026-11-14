"""Independent numerical and source-domain guarantees for generated tables."""

from copy import deepcopy

import CoolProp.CoolProp as CP
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.bicubic.table_cache import table_validation_error

from test_bicubic_cache import cache_factory


class NumericResult:
    def __init__(self, values):
        self.values = values

    def to_dict(self):
        return self.values.copy()


class PolynomialSource:
    """Exact polynomials independent of the finite-difference implementation."""

    name = "PolynomialSource"
    T_min, T_max, p_max = 200.0, 1000.0, 1e9

    def __init__(self, *, quadratic=False, override=None, bias=None):
        self.quadratic = quadratic
        self.override = override or {}
        self.bias = bias or {}
        self.tolerances = []

    def properties(self, h, p):
        temperature = (
            300 + 2 * h**2 + 3 * p**2 + 4 * h * p
            if self.quadratic
            else 300 + 2 * h + 3 * p + 4 * h * p
        )
        values = dict(
            temperature=temperature,
            pressure=p,
            enthalpy=h,
            density=1000.0,
            entropy=2000.0,
            internal_energy=100.0,
            isobaric_heat_capacity=1000.0,
            isochoric_heat_capacity=800.0,
            isothermal_bulk_modulus=1e6,
            isentropic_bulk_modulus=1.2e6,
        )
        values.update(self.override)
        for name, change in self.bias.items():
            values[name] += change
        return NumericResult(values)

    def get_state(self, pair, first, second):
        if pair == jp.PQ_INPUTS:
            return NumericResult(
                dict(pressure=first, temperature=5 + 3 * first + 2 * first**2)
            )
        assert pair == jp.HmassP_INPUTS
        return self.properties(first, second)

    def get_state_metastable(self, **request):
        assert (request["prop_1"], request["prop_2"]) == ("p", "h")
        self.tolerances.append(request["solver_tolerance"])
        return self.properties(request["prop_2_value"], request["prop_1_value"])


@pytest.mark.parametrize("step", [0.03, 0.37])
def test_saturation_central_derivative_has_correct_step_units(step):
    pressure = 2.5
    values = jp.FluidBicubic._gradients_saturation_central(
        PolynomialSource(), pressure, 0.0, step
    )
    np.testing.assert_allclose(values["temperature"], [25.0, 13.0], rtol=2e-14)
    np.testing.assert_allclose(values["pressure"], [pressure, 1.0], rtol=2e-14)


@pytest.mark.parametrize("method", ["forward", "central"])
def test_equilibrium_gradients_match_independent_polynomial_partials(method):
    source = PolynomialSource(quadratic=method == "central")
    h, p, eh, ep = 4.0, 3.0, 0.2, 0.07
    function = getattr(jp.FluidBicubic, f"_gradients_{method}")
    actual = function(source, h, p, eh, ep)["temperature"]
    expected = (
        [300 + 2 * h**2 + 3 * p**2 + 4 * h * p, 4 * h + 4 * p, 6 * p + 4 * h, 4]
        if method == "central"
        else [300 + 2 * h + 3 * p + 4 * h * p, 2 + 4 * p, 3 + 4 * h, 4]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_metastable_gradients_match_polynomial_and_request_tight_solves():
    source = PolynomialSource()
    actual = jp.FluidBicubic._gradients_metastable_forward(
        source, 4.0, 3.0, 1 / 1024, 1 / 1024, 1000.0, 365.0
    )["temperature"]
    np.testing.assert_allclose(actual, [365, 14, 19, 4], rtol=1e-12, atol=1e-12)
    assert len(source.tolerances) == 7
    assert set(source.tolerances) == {1e-10}


@pytest.mark.parametrize(
    "override",
    [
        {"temperature": 100.0},
        {"temperature": 1200.0},
        {"temperature": np.nan},
        {"pressure": 2e9},
        {"pressure": 0.0},
        {"density": 0.0},
        {"density": np.nan},
    ],
)
def test_metastable_generation_rejects_extrapolated_or_invalid_samples(override):
    with pytest.raises(jp.CalculationError, match="source EOS domain"):
        jp.FluidBicubic._gradients_metastable_forward(
            PolynomialSource(override=override), 4.0, 3.0, 0.2, 0.07, 1000.0, 365.0
        )


@pytest.mark.parametrize("name", ["enthalpy", "pressure"])
def test_metastable_flash_residual_cannot_dominate_finite_difference_step(name):
    with pytest.raises(jp.CalculationError, match=f"{name} residual.*error budget"):
        jp.FluidBicubic._gradients_metastable_forward(
            PolynomialSource(bias={name: 1e-3}), 4.0, 3.0, 0.2, 0.07, 1000.0, 365.0
        )


def test_stiff_liquid_pressure_roundoff_budget_uses_bulk_modulus():
    source = PolynomialSource(
        override={"temperature": 365.0, "isothermal_bulk_modulus": 2e9},
        bias={"pressure": 1e-5},
    )
    result = jp.FluidBicubic._metastable_generation_sample(
        source, 4.0, 1e6, 0.2, 10.0, [1000.0, 365.0]
    )
    assert result["pressure"] == 1e6 + 1e-5


@pytest.mark.parametrize("temperature", [300.0, 500.0])
def test_real_metastable_temperature_partials_match_analytical_EOS(temperature):
    source = jp.FluidCoolProp("Water")
    initial = source.get_state(jp.PT_INPUTS, 1e6, temperature)
    h, rho = float(initial.h), float(initial.rho)
    gradients = jp.FluidBicubic._gradients_metastable_forward(
        source, h, 1e6, 1e-5 * abs(h), 10.0, rho, temperature
    )
    actual = gradients["temperature"]
    state = source.abstract_state
    state.update(CP.DmassT_INPUTS, gradients["density"][0], actual[0])
    # Source and homogeneous paths must agree at the matched rho/T point;
    # no gas-scale correction is permitted in the derivative comparison.
    np.testing.assert_allclose(gradients["pressure"][0], state.p(), rtol=5e-10)
    expected = [
        state.first_partial_deriv(CP.iT, CP.iHmass, CP.iP),
        state.first_partial_deriv(CP.iT, CP.iP, CP.iHmass),
        state.second_partial_deriv(CP.iT, CP.iHmass, CP.iP, CP.iP, CP.iHmass),
    ]
    np.testing.assert_allclose(actual[1:3], expected[:2], rtol=5e-5)
    np.testing.assert_allclose(actual[3], expected[2], rtol=5e-3)


@pytest.mark.parametrize("change", ["offset", "reverse"])
def test_saturation_pressure_nodes_must_match_declared_grid(cache_factory, change):
    create, _, _ = cache_factory
    model = create(metastable_phase="liquid", gradient_method="forward")
    table = deepcopy(model.table)
    pressure = np.asarray(table["saturation_props"]["pressure"]["value"]).copy()
    pressure = pressure[::-1] if change == "reverse" else pressure * 1.0001
    table["saturation_props"]["pressure"]["value"] = pressure
    assert "saturation pressure nodes" in table_validation_error(table, model.metadata)


@pytest.mark.parametrize("bound", ["minimum", "maximum"])
def test_cached_finite_temperatures_must_stay_inside_source_domain(
    cache_factory, bound
):
    create, _, _ = cache_factory
    model = create()
    table = deepcopy(model.table)
    domain = model.metadata["source_domain"]
    invalid = domain["T_min"] * 0.5 if bound == "minimum" else domain["T_max"] * 1.1
    values = np.asarray(table["temperature"]["value"]).copy()
    values[0, 0] = invalid
    table["temperature"]["value"] = values
    assert "source EOS domain" in table_validation_error(table, model.metadata)


def test_finite_source_extrapolation_is_not_published_as_a_table(tmp_path):
    # Raw HEOS supplies finite extrapolation beyond R134a's published Tmax.
    state = CP.AbstractState("HEOS", "R134a")
    state.update(CP.HmassP_INPUTS, 750e3, 2e5)
    assert np.isfinite(state.T()) and state.T() > state.Tmax()
    settings = dict(
        fluid_name="R134a",
        backend="HEOS",
        h_min=450e3,
        h_max=460e3,
        p_min=2e5,
        p_max=3e5,
        N_h=2,
        N_p=2,
        N_p_sat=2,
        coarse_step=1,
        table_name="source_domain",
        table_dir=str(tmp_path),
    )
    jp.FluidBicubic(**settings)
    cache = tmp_path / "source_domain.pkl"
    previous = cache.read_bytes()
    with pytest.warns(UserWarning, match="Regenerating"):
        with pytest.raises(jp.CalculationError) as rejected:
            jp.FluidBicubic(**(settings | {"h_min": 700e3, "h_max": 750e3}))
    assert isinstance(rejected.value.__cause__, jp.OutOfDomainError)
    assert rejected.value.request == "table_generation"
    assert cache.read_bytes() == previous
