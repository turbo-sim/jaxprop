"""Opt-in native diagnostics and both equilibrium saturation endpoints."""

from copy import deepcopy
from types import SimpleNamespace
import equinox as eqx
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


def test_phase_boundary_hp_and_endpoint_routes_agree(phase_model):
    endpoint = phase_model.get_state(jp.PQ_INPUTS, 130e3, 0.0)
    masked = phase_model.get_state(
        jp.HmassP_INPUTS, endpoint.h, endpoint.p, errors="mask"
    )
    assert int(endpoint.status) == 0 and int(masked.status) == 0
    for name in ("temperature", "density", "entropy", "internal_energy"):
        np.testing.assert_allclose(masked[name], endpoint[name], rtol=2e-10, atol=1e-6)
    diagnostic = phase_model.get_state_saturation(
        jp.PQ_INPUTS, 130e3, 1.0, supersaturation=True
    )
    assert diagnostic.supports_property("superheating")
    assert abs(float(diagnostic.superheating)) < 1e-9


def test_uniform_mesh_and_source_accuracy_across_both_boundaries(phase_model):
    model = phase_model
    np.testing.assert_allclose(np.diff(model.h_vals), model.delta_h, rtol=1e-13)
    np.testing.assert_allclose(np.diff(model.logP_vals), model.delta_logP, rtol=1e-13)
    assert (len(model.h_vals), len(model.logP_vals)) == (32, 8)
    source = jp.Fluid("Water", backend=model.backend)
    for quality in (0.0, 1.0):
        endpoint = source.get_state(jp.PQ_INPUTS, 130e3, quality)
        h = float(endpoint.h) + np.array([-100.0, -1.0, 1.0, 100.0])
        actual = model.get_state(jp.HmassP_INPUTS, h, 130e3, errors="mask")
        expected = source.get_state(jp.HmassP_INPUTS, h, 130e3)
        np.testing.assert_array_equal(actual.status, 0)
        np.testing.assert_array_equal(actual.is_two_phase, expected.is_two_phase)
        for name in ("temperature", "density", "entropy", "internal_energy"):
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=2e-5, atol=1e-4
            )
        for name in (
            "isochoric_heat_capacity",
            "speed_of_sound",
            "gruneisen",
            "joule_thomson",
        ):
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=1e-3, atol=1e-6, equal_nan=True
            )
        wet = np.asarray(expected.is_two_phase) == 1
        assert np.all(np.isnan(np.asarray(actual.cp)[wet]))


def test_near_boundary_jit_derivatives_and_inverse_flashes(phase_model, monkeypatch):
    model = phase_model
    p = 130e3
    endpoints = [model.get_state(jp.PQ_INPUTS, p, q) for q in (0.0, 1.0)]
    source = jp.Fluid("Water", backend=model.backend)
    inputs = [(float(s.h) + offset, p) for s in endpoints for offset in (-100.0, 100.0)]
    references = [source.get_state(jp.HmassP_INPUTS, *point) for point in inputs]
    source_jacobians = []
    for point in inputs:
        columns = []
        for i, step in enumerate((0.01, 0.1)):
            offset = np.eye(2)[i] * step
            a = source.get_state(jp.HmassP_INPUTS, *(np.asarray(point) + offset))
            b = source.get_state(jp.HmassP_INPUTS, *(np.asarray(point) - offset))
            columns.append(
                [
                    (float(a[n]) - float(b[n])) / (2 * step)
                    for n in ("temperature", "density", "entropy", "internal_energy")
                ]
            )
        source_jacobians.append(np.asarray(columns).T)

    def forbidden(*args, **kwargs):
        raise AssertionError("EOS runtime callback")

    monkeypatch.setattr(jp.FluidCoolProp, "get_state", forbidden)
    fields = ("temperature", "density", "entropy", "internal_energy")

    def vector(x):
        s = model.get_state(jp.HmassP_INPUTS, x[0], x[1], errors="mask")
        return jnp.stack(tuple(s[n] for n in fields))

    value = jax.jit(vector)
    fwd, rev = jax.jit(jax.jacfwd(vector)), jax.jit(jax.jacrev(vector))
    for point, reference, source_jacobian in zip(inputs, references, source_jacobians):
        x = jnp.asarray(point)
        a, b = np.asarray(fwd(x)), np.asarray(rev(x))
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)
        scale = np.array([max(abs(float(reference[n])), 1.0) for n in fields])[:, None]
        normalized_error = (
            abs(a - source_jacobian) * np.array([model.delta_h, p]) / scale
        )
        assert np.all(normalized_error < 1e-3)
        for i, step in enumerate((0.01, 0.1)):
            offset = jnp.eye(2)[i] * step
            fd = (np.asarray(value(x + offset)) - np.asarray(value(x - offset))) / (
                2 * step
            )
            np.testing.assert_allclose(a[:, i], fd, rtol=2e-5, atol=1e-8)
        state = model.get_state(jp.HmassP_INPUTS, *point, errors="mask")
        ps = model.get_state(jp.PSmass_INPUTS, point[1], state.s, errors="mask")
        assert int(ps.status) == 0
        np.testing.assert_allclose(ps.h, point[0], rtol=2e-9, atol=1e-4)
        if not float(reference.is_two_phase):
            pt = model.get_state(jp.PT_INPUTS, point[1], state.T, errors="mask")
            assert int(pt.status) == 0
            np.testing.assert_allclose(pt.h, point[0], rtol=2e-9, atol=1e-4)


def test_pressure_column_bridge_changes_are_continuous(phase_model):
    model = phase_model

    def vector(x):
        s = model.get_state(jp.HmassP_INPUTS, x[0], x[1], errors="mask")
        return jnp.stack((s.T, s.rho, s.s))

    value, derivative = jax.jit(vector), jax.jit(jax.jacfwd(vector))
    for logp in np.asarray(model.logP_vals)[1:-1]:
        p = float(np.exp(logp))
        for quality, offset in (
            (0.0, -0.01 * model.delta_h),
            (1.0, 0.01 * model.delta_h),
        ):
            h = float(model.get_state(jp.PQ_INPUTS, p, quality).h) + offset
            left, right = jnp.array([h, p * (1 - 1e-9)]), jnp.array([h, p * (1 + 1e-9)])
            np.testing.assert_allclose(value(left), value(right), rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(
                derivative(left), derivative(right), rtol=2e-5, atol=1e-8
            )


def test_missing_homogeneous_anchor_keeps_guard(phase_model):
    endpoint = phase_model.get_state(jp.PQ_INPUTS, 130e3, 0.0)
    broken = eqx.tree_at(
        lambda m: m._phase_boundary["liquid"]["index"],
        phase_model,
        jnp.full_like(phase_model._phase_boundary["liquid"]["index"], -1),
    )
    state = broken.get_state(
        jp.HmassP_INPUTS, endpoint.h - 1.0, endpoint.p, errors="mask"
    )
    assert int(state.status) == jp.StatusCode.OUT_OF_DOMAIN


def test_pseudo_pure_glide_does_not_use_pure_lever_rules():
    from jaxprop.bicubic.phase_boundary import prepare

    model = SimpleNamespace(
        phase_support=True, logP_sat_vals=np.array([1.0, 2.0]), fluid_name="R410A"
    )
    assert prepare(model, {}) is None
