"""Opt-in native diagnostics and both equilibrium saturation endpoints."""

from copy import deepcopy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jaxprop as jp
from utilities import get_available_backends


@pytest.fixture(scope="module", params=get_available_backends())
def backend(request):
    return request.param


@pytest.fixture(scope="module")
def phase_model(backend, tmp_path_factory):
    return jp.Fluid(
        "Water",
        method="bicubic",
        backend=backend,
        table_options=dict(
            h_min=100e3,
            h_max=3.2e6,
            p_min=80e3,
            p_max=200e3,
            N_h=32,
            N_p=8,
            N_p_sat=64,
            coarse_step=1,
            phase_support=True,
            table_dir=str(tmp_path_factory.mktemp("phase_support")),
        ),
    )


def test_capabilities_and_explicit_diagnostic_enabling(phase_model):
    model = phase_model
    assert {"subcooling", "superheating"} <= model.capabilities.properties
    assert "supersaturation" in model.capabilities.options
    assert jp.PQ_INPUTS in model.capabilities.input_pairs
    assert "saturation" in model.capabilities.phase_operations
    ordinary = model.get_state(jp.HmassP_INPUTS, 200e3, 130e3)
    assert not ordinary.supports_property("subcooling")
    with pytest.raises(jp.UnsupportedPropertyError):
        ordinary.subcooling
    diagnostic = model.get_state(jp.HmassP_INPUTS, 200e3, 130e3, supersaturation=True)
    assert float(diagnostic.subcooling) > 0
    assert np.isfinite(float(diagnostic.superheating))


@pytest.mark.parametrize("enthalpy", [200e3, 1500e3, 3000e3])
def test_liquid_wet_vapor_diagnostics_retain_source_definitions(phase_model, enthalpy):
    expected = jp.Fluid("Water", backend=phase_model.backend).get_state(
        jp.HmassP_INPUTS, enthalpy, 130e3, supersaturation=True
    )
    actual = phase_model.get_state(
        jp.HmassP_INPUTS, enthalpy, 130e3, supersaturation=True
    )
    assert int(actual.status) == 0
    for name in ("subcooling", "superheating"):
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-4, atol=0.05)
    assert (float(actual.is_two_phase) > 0.5) == (float(expected.is_two_phase) > 0.5)


def test_saturation_endpoints_scalar_batch_aliases_and_eos_accuracy(phase_model):
    source = jp.Fluid("Water", backend=phase_model.backend)
    p = jnp.array([80e3, 110e3, 150e3, 200e3])
    for quality in (0.0, 1.0):
        expected = source.get_state(jp.PQ_INPUTS, np.asarray(p), quality)
        actual = phase_model.get_state(jp.PQ_INPUTS, p, quality)
        alternate = phase_model.get_state_saturation(jp.PQ_INPUTS, p, quality)
        np.testing.assert_array_equal(actual.status, 0)
        np.testing.assert_array_equal(actual.Q, quality)
        for name in ("temperature", "enthalpy", "entropy", "density"):
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=2e-6, atol=1e-5
            )
            np.testing.assert_allclose(actual[name], alternate[name], rtol=0, atol=0)
    result = phase_model.get_state(jp.PQ_INPUTS, [[100e3], [150e3]], [0.0, 1.0])
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.Q, [[0.0, 1.0], [0.0, 1.0]])


def test_endpoint_and_diagnostic_jit_forward_reverse_derivatives(phase_model):
    for q in (0.0, 1.0):
        f = lambda p: phase_model.get_state(
            jp.PQ_INPUTS, p, q, errors="mask"
        ).temperature
        forward = jax.jit(jax.jacfwd(f))(130e3)
        reverse = jax.jit(jax.grad(f))(130e3)
        finite = (f(130e3 + 1) - f(130e3 - 1)) / 2
        np.testing.assert_allclose(forward, finite, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(forward, reverse, rtol=1e-12, atol=1e-12)
    f = lambda h: phase_model.get_state(
        jp.HmassP_INPUTS, h, 130e3, supersaturation=True, errors="mask"
    ).superheating
    ad = jax.jit(jax.grad(f))(1500e3)
    finite = (f(1500e3 + 1) - f(1500e3 - 1)) / 2
    np.testing.assert_allclose(ad, finite, rtol=1e-6, atol=1e-10)
    assert np.isfinite(float(ad))


def test_endpoint_policies_and_no_source_queries_after_construction(
    phase_model, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Source EOS entered native runtime")

    monkeypatch.setattr(jp.FluidCoolProp, "get_state", forbidden)
    f = jax.jit(lambda p, q: phase_model.get_state(jp.PQ_INPUTS, p, q, errors="mask"))
    result = f(jnp.array([130e3, 130e3, 1.0, np.nan]), jnp.array([0.0, 0.5, 1.0, 0.0]))
    assert int(result.status[0]) == 0
    assert np.all(np.asarray(result.status)[1:] != 0)
    assert np.all(np.isnan(np.asarray(result.raw_property("temperature"))[1:]))
    clip = phase_model.get_state(jp.PQ_INPUTS, 1.0, 0.6, errors="clip")
    assert int(clip.status) == 0 and bool(clip.clipped) and float(clip.Q) == 1.0
    with pytest.raises(jp.OutOfDomainError):
        phase_model.get_state(jp.PQ_INPUTS, 130e3, 0.5)
    assert np.isfinite(
        float(
            phase_model.get_state(
                jp.HmassP_INPUTS, 200e3, 130e3, supersaturation=True
            ).subcooling
        )
    )


def test_cache_roundtrip_inspection_and_corruption_detection(phase_model, monkeypatch):
    from jaxprop.bicubic.table_cache import table_validation_error

    original = phase_model.table
    assert table_validation_error(original, phase_model.metadata) is None
    malformed = deepcopy(original)
    malformed["saturation_props"].pop("vapor")
    assert "both saturation endpoints" in table_validation_error(
        malformed, phase_model.metadata
    )
    malformed = deepcopy(original)
    malformed["saturation_props"]["liquid"]["enthalpy"]["value"] = jnp.array([np.nan])
    assert table_validation_error(malformed, phase_model.metadata) is not None
    settings = dict(phase_model.configuration.table_options)
    assert (
        settings.pop("mixture_ratio") is None
    )  # Inspection metadata is not a factory recipe.

    def forbidden(self):
        raise AssertionError("Exact cache should load without generation")

    monkeypatch.setattr(jp.FluidBicubic, "_generate_property_table", forbidden)
    loaded = jp.Fluid(
        "Water", method="bicubic", backend=phase_model.backend, table_options=settings
    )
    assert loaded.table_name == phase_model.table_name
    np.testing.assert_allclose(
        loaded.get_state(jp.PQ_INPUTS, 130e3, 1.0).T,
        phase_model.get_state(jp.PQ_INPUTS, 130e3, 1.0).T,
        rtol=0,
        atol=0,
    )


def test_invalid_phase_support_configuration_is_rejected():
    common = dict(h_min=1e5, h_max=3e5, p_min=1e5, p_max=2e5, N_h=4, N_p=4)
    for value in (1, "yes", None):
        with pytest.raises(jp.InvalidConfigurationError):
            jp.Fluid(
                "Water",
                method="bicubic",
                table_options=dict(common, phase_support=value),
            )
    with pytest.raises(jp.InvalidConfigurationError):
        jp.Fluid(
            "Water",
            method="bicubic",
            table_options=dict(
                common,
                phase_support=True,
                metastable_phase="liquid",
                gradient_method="forward",
            ),
        )


def test_supercritical_diagnostics_without_a_saturation_domain(tmp_path):
    model = jp.Fluid(
        "CO2",
        method="bicubic",
        table_options=dict(
            h_min=600e3,
            h_max=900e3,
            p_min=8e6,
            p_max=12e6,
            N_h=8,
            N_p=8,
            N_p_sat=8,
            coarse_step=1,
            phase_support=True,
            table_dir=str(tmp_path),
        ),
    )
    expected = jp.Fluid("CO2").get_state(
        jp.HmassP_INPUTS, 700e3, 10e6, supersaturation=True
    )
    actual = model.get_state(jp.HmassP_INPUTS, 700e3, 10e6, supersaturation=True)
    for name in ("subcooling", "superheating"):
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-4, atol=0.05)
    assert jp.PQ_INPUTS not in model.capabilities.input_pairs
    assert "saturation" not in model.capabilities.phase_operations
    with pytest.raises(jp.UnsupportedOperationError):
        model.get_state_saturation(jp.PQ_INPUTS, 1e5, 0.0)


def test_phase_boundary_cells_remain_rejected_but_endpoint_route_is_valid(phase_model):
    endpoint = phase_model.get_state(jp.PQ_INPUTS, 130e3, 0.0)
    masked = phase_model.get_state(
        jp.HmassP_INPUTS, endpoint.h, endpoint.p, errors="mask"
    )
    assert int(endpoint.status) == 0 and int(masked.status) != 0
    assert np.isnan(float(masked.raw_property("temperature")))
    diagnostic = phase_model.get_state_saturation(
        jp.PQ_INPUTS, 130e3, 1.0, supersaturation=True
    )
    assert diagnostic.supports_property("superheating")
    assert abs(float(diagnostic.superheating)) < 1e-9
