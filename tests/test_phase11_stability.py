"""Independent stability, pole, provenance and construction-work guarantees."""

from copy import deepcopy

import CoolProp.CoolProp as CP
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.bicubic import bicubic_interpolation as interpolation
from jaxprop.bicubic.table_cache import table_validation_error
from jaxprop.bicubic.thermal_stability import (
    METASTABLE_STABILITY_SETTINGS,
    metastable_cell_guards,
    polynomial_bounds,
    prepare_metastable_stability,
)

from test_bicubic_cache import cache_factory
from test_phase8_table_integrity import PolynomialSource


def polynomial_table(cv_coefficients):
    table = {}
    for name in jp.PROPERTIES_CANONICAL:
        coefficients = np.zeros((1, 1, 16))
        coefficients[..., 0] = 1.0
        table[name] = {"coeffs": coefficients}
    table["isochoric_heat_capacity"]["coeffs"][0, 0] = cv_coefficients
    table["isothermal_bulk_modulus"]["coeffs"][..., 0] = 1e6
    table["isobaric_heat_capacity"]["coeffs"][..., 0] = 1000.0
    return table


@pytest.mark.parametrize("seed", [7, 31, 92])
def test_bernstein_bounds_enclose_independently_evaluated_bicubics(seed):
    coefficients = np.random.default_rng(seed).normal(size=(3, 2, 16))
    lower, upper = polynomial_bounds(coefficients)
    points = np.linspace(0, 1, 31)
    x, y = np.meshgrid(points, points)
    basis = np.stack([x**i * y**j for j in range(4) for i in range(4)])
    actual = np.einsum("...k,kij->...ij", coefficients, basis)
    assert np.all(actual >= lower[..., None, None])
    assert np.all(actual <= upper[..., None, None])


def test_positive_corners_do_not_certify_a_cell_with_negative_interior_cv():
    # cv(x,y) = 1-8x+8x**2: all four corners are 1; midpoint is -1.
    cv = np.zeros(16)
    cv[:3] = [1, -8, 8]
    table = polynomial_table(cv)
    physical, poles = metastable_cell_guards(
        table, "physical", METASTABLE_STABILITY_SETTINGS
    )
    formal, _ = metastable_cell_guards(table, "formal", METASTABLE_STABILITY_SETTINGS)
    assert not physical[0, 0] and formal[0, 0]
    assert poles["gruneisen"][0, 0]
    assert not poles["joule_thomson"][0, 0]
    before = deepcopy(table)
    prepare_metastable_stability(table, "formal", METASTABLE_STABILITY_SETTINGS)
    for name in METASTABLE_STABILITY_SETTINGS["cv_pole_properties"]:
        assert np.all(np.isnan(table[name]["coeffs"]))
    np.testing.assert_array_equal(
        table["enthalpy"]["coeffs"], before["enthalpy"]["coeffs"]
    )
    np.testing.assert_array_equal(
        table["isochoric_heat_capacity"]["coeffs"], cv[None, None]
    )


def test_formal_negative_cv_cells_remain_available_when_they_do_not_cross_zero():
    cv = np.zeros(16)
    cv[0] = -5
    table = polynomial_table(cv)
    eligible = prepare_metastable_stability(
        table, "formal", METASTABLE_STABILITY_SETTINGS
    )
    assert eligible[0, 0]
    assert np.all(np.isfinite(table["gruneisen"]["coeffs"]))


def test_formal_cp_pole_masks_only_its_affected_responses():
    cv = np.zeros(16)
    cv[0] = 100
    table = polynomial_table(cv)
    table["isobaric_heat_capacity"]["coeffs"][0, 0, :2] = [-1, 2]
    prepare_metastable_stability(table, "formal", METASTABLE_STABILITY_SETTINGS)
    for name in METASTABLE_STABILITY_SETTINGS["cp_pole_properties"]:
        assert np.all(np.isnan(table[name]["coeffs"]))
    assert np.all(np.isfinite(table["gruneisen"]["coeffs"]))


@pytest.mark.parametrize("cv", [0.0, -10.0, np.nan])
def test_physical_samples_reject_nonpositive_or_unknown_cv(cv):
    source = PolynomialSource(override={"isochoric_heat_capacity": cv})
    with pytest.raises(
        jp.CalculationError, match="positive cv.*metastable_domain='formal'"
    ):
        jp.FluidBicubic._metastable_generation_sample(
            source, 4, 3, 0.2, 0.07, [1000, 365]
        )


@pytest.mark.parametrize("domain", ["physical", "formal"])
@pytest.mark.parametrize("bulk", [0.0, -10.0, np.nan])
def test_both_table_modes_reject_mechanical_instability_at_source_samples(domain, bulk):
    source = PolynomialSource(override={"isothermal_bulk_modulus": bulk})
    with pytest.raises(jp.CalculationError, match="Mechanical instability"):
        jp.FluidBicubic._metastable_generation_sample(
            source, 4, 3, 0.2, 0.07, [1000, 365], metastable_domain=domain
        )


def test_stability_validation_includes_finite_difference_stencil_not_only_nodes():
    class StencilSource(PolynomialSource):
        def properties(self, h, p):
            result = super().properties(h, p)
            result.values["isochoric_heat_capacity"] = 100 if h <= 4 else -100
            return result

    with pytest.raises(jp.CalculationError, match="positive cv; got -100"):
        jp.FluidBicubic._gradients_metastable_forward(
            StencilSource(), 4, 3, 0.2, 0.07, 1000, 365
        )


def test_actual_formal_nitrogen_state_is_mechanically_but_not_calorically_stable():
    source = jp.FluidCoolProp("Nitrogen")
    p, h = 1e6, -10e3
    point = source.get_state_metastable(
        "p", p, "h", h, rhoT_guess=[600, 118], solver_tolerance=1e-10
    )
    raw = CP.AbstractState("HEOS", "Nitrogen")
    raw.specify_phase(CP.iphase_liquid)
    raw.update(CP.DmassT_INPUTS, float(point.rho), float(point.T))
    assert raw.cvmass() < 0 < 1 / raw.isothermal_compressibility()
    with pytest.raises(jp.CalculationError, match="require positive cv"):
        jp.FluidBicubic._metastable_generation_sample(
            source, h, p, 1.0, 10.0, [float(point.rho), float(point.T)]
        )
    formal = jp.FluidBicubic._metastable_generation_sample(
        source,
        h,
        p,
        1.0,
        10.0,
        [float(point.rho), float(point.T)],
        metastable_domain="formal",
    )
    np.testing.assert_allclose(
        formal["isochoric_heat_capacity"], raw.cvmass(), rtol=1e-10
    )


@pytest.fixture(scope="module")
def formal_nitrogen(tmp_path_factory):
    return jp.Fluid(
        "Nitrogen",
        method="bicubic",
        table_options=dict(
            h_min=-110e3,
            h_max=-10e3,
            p_min=8e5,
            p_max=1e6,
            N_h=8,
            N_p=5,
            N_p_sat=3,
            coarse_step=1,
            metastable_phase="liquid",
            metastable_domain="formal",
            gradient_method="forward",
            table_dir=str(tmp_path_factory.mktemp("formal_nitrogen")),
        ),
    )


def test_formal_table_pole_unavailability_preserves_core_status_and_jax_gradients(
    formal_nitrogen,
):
    model = formal_nitrogen
    table = model.table
    finite_gamma = np.all(np.isfinite(table["gruneisen"]["coeffs"]), axis=-1)
    crossing = ~finite_gamma & np.asarray(model._cell_validity)
    assert np.any(crossing) and np.any(finite_gamma)
    i, j = np.argwhere(crossing)[0]
    h = float(0.5 * (model.h_vals[i] + model.h_vals[i + 1]))
    p = float(np.exp(0.5 * (model.logP_vals[j] + model.logP_vals[j + 1])))
    state = model.get_state(jp.HmassP_INPUTS, h, p)
    assert state.valid and not state.property_valid("gruneisen")
    assert np.isnan(state.raw_property("gruneisen"))
    for name in (
        "density",
        "temperature",
        "enthalpy",
        "pressure",
        "entropy",
        "internal_energy",
    ):
        assert np.isfinite(state.raw_property(name))
    consumer = jax.jit(
        lambda h: model.get_state(jp.HmassP_INPUTS, h, p, errors="mask").raw_property(
            "temperature"
        )
    )
    assert np.isfinite(jax.grad(consumer)(jnp.asarray(h)))
    assert table_validation_error(table, model.metadata) is None
    damaged = deepcopy(table)
    coefficients = np.asarray(damaged["gruneisen"]["coeffs"]).copy()
    coefficients[i, j] = 0.0
    damaged["gruneisen"]["coeffs"] = coefficients
    assert "unmasked heat-capacity pole" in table_validation_error(
        damaged, model.metadata
    )


def test_metastable_domain_is_part_of_manifest_and_cache_identity(cache_factory):
    create, generated, _ = cache_factory
    first = create(metastable_phase="liquid", gradient_method="forward")
    with pytest.warns(UserWarning, match="metadata does not match"):
        formal = create(
            metastable_phase="liquid",
            gradient_method="forward",
            metastable_domain="formal",
        )
    assert generated == ["metastable_liquid", "metastable_liquid"]
    assert first.metadata["metastable_domain"] == "physical"
    assert formal.configuration.table_options["metastable_domain"] == "formal"
    damaged = deepcopy(first.table)
    cv = np.asarray(damaged["isochoric_heat_capacity"]["value"]).copy()
    cv[0, 0] = -1
    damaged["isochoric_heat_capacity"]["value"] = cv
    assert "positive isochoric_heat_capacity" in table_validation_error(
        damaged, first.metadata
    )


@pytest.mark.parametrize("phase", ["liquid", "vapor"])
def test_metastable_coefficients_are_assembled_once_and_match_independent_final_arrays(
    tmp_path, monkeypatch, phase
):
    original = interpolation.compute_bicubic_coefficients
    calls = []

    def counted(*args):
        calls.append(tuple(np.asarray(arg).copy() for arg in args))
        return original(*args)

    monkeypatch.setattr(interpolation, "compute_bicubic_coefficients", counted)
    if phase == "liquid":
        bounds = dict(h_min=-120e3, h_max=-119e3, p_min=8e4, p_max=2e5)
    else:
        raw = CP.AbstractState("HEOS", "Nitrogen")
        raw.update(CP.PQ_INPUTS, 1e6, 1)
        bounds = dict(
            h_min=raw.hmass() - 750, h_max=raw.hmass() - 250, p_min=9e5, p_max=1e6
        )
    model = jp.FluidBicubic(
        "Nitrogen",
        **bounds,
        N_h=3,
        N_p=4,
        N_p_sat=3,
        coarse_step=1,
        metastable_phase=phase,
        gradient_method="forward",
        table_dir=str(tmp_path),
    )
    assert len(calls) == len(jp.PROPERTIES_CANONICAL)
    for name, args in zip(jp.PROPERTIES_CANONICAL, calls):
        # Reconstruction is independent of the generator's call scheduling.
        entry = model.table[name]
        for key, actual in zip(("value", "grad_h", "grad_logP", "grad_hlogP"), args):
            np.testing.assert_array_equal(actual, entry[key])
        np.testing.assert_array_equal(original(*args), entry["coeffs"])
