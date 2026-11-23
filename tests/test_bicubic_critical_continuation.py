"""Critical continuation keeps uniform grids, bounded labels and native AD."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jaxprop as jp
from scipy.optimize import brentq
import CoolProp.CoolProp as CP


@pytest.fixture(scope="module")
def critical_model(tmp_path_factory):
    source = jp.Fluid("Water", backend="HEOS")
    pc, hc = float(source.critical_point.p), float(source.critical_point.h)
    model = jp.Fluid(
        "Water",
        backend="HEOS",
        method="bicubic",
        table_options=dict(
            h_min=hc - 3e5,
            h_max=hc + 3e5,
            p_min=0.94 * pc,
            p_max=1.04 * pc,
            N_h=24,
            N_p=12,
            N_p_sat=24,
            coarse_step=1,
            phase_support=True,
            table_dir=str(tmp_path_factory.mktemp("critical")),
        ),
    )
    return model, source


def core(model, x):
    state = model.get_state(jp.HmassP_INPUTS, x[0], x[1], errors="mask")
    return jnp.stack((state.T, state.rho, state.s, state.u, state.quality_mass))


def test_critical_hp_values_and_labels_preserve_grid_and_cache(critical_model):
    model, source = critical_model
    pc, hc = float(source.critical_point.p), float(source.critical_point.h)
    path = Path(model.table_dir) / (model.table_name + ".pkl")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    h_nodes, p_nodes = (
        np.asarray(model.h_vals).copy(),
        np.asarray(model.logP_vals).copy(),
    )
    for ratio in (0.984, 0.99, 1.0, 1.01):
        pressure = ratio * pc
        if ratio == 1.0:
            # CoolProp's HP flash can select an unusable gas bracket at its
            # numerically reported critical pressure. Obtain the same (h,p)
            # reference from explicitly liquid PT states instead. This is only
            # the independent test oracle: native point, mesh and comparison
            # tolerances remain unchanged, and runtime never invokes this root.
            tc = float(source.critical_point.T)
            target = hc - 2e4
            reference = CP.AbstractState("HEOS", "Water")
            # At h < hc and p = pc this is the stable liquid branch. Imposing
            # it also avoids CoolProp's automatic near-saturation PT guard.
            reference.specify_phase(CP.iphase_liquid)
            def residual(temperature):
                reference.update(CP.PT_INPUTS, pressure, temperature)
                return reference.hmass() - target
            temperature = brentq(residual, 0.95 * tc, (1 - 1e-9) * tc, xtol=1e-12)
            reference.update(CP.PT_INPUTS, pressure, temperature)
            expected = SimpleNamespace(h=reference.hmass(), T=reference.T(), rho=reference.rhomass())
            assert abs(float(expected.h) - target) < 0.02
            # The source PT solver's pressure closure itself has roundoff at
            # this nearly singular point; check it relatively, independently
            # of the unchanged native T/density comparison budgets below.
            np.testing.assert_allclose(reference.p(), pressure, rtol=1e-8, atol=0)
        else:
            expected = source.get_state(jp.HmassP_INPUTS, hc - 2e4, pressure)
        actual = model.get_state(jp.HmassP_INPUTS, hc - 2e4, pressure, errors="mask")
        assert int(actual.status) == 0
        assert float(actual.is_two_phase) in (0.0, 1.0)
        assert 0 <= float(actual.quality_mass) <= 1
        assert 0 <= float(actual.quality_volume) <= 1
        assert abs(float(actual.T - expected.T)) < 1.0
        np.testing.assert_allclose(actual.rho, expected.rho, rtol=0.05)
    np.testing.assert_array_equal(model.h_vals, h_nodes)
    np.testing.assert_array_equal(model.logP_vals, p_nodes)
    np.testing.assert_allclose(np.diff(h_nodes), model.delta_h)
    np.testing.assert_allclose(np.diff(p_nodes), model.delta_logP)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_critical_jit_ad_fd_has_no_runtime_source_calls(critical_model, monkeypatch):
    model, source = critical_model
    pc, hc = float(source.critical_point.p), float(source.critical_point.h)

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime EOS call")

    monkeypatch.setattr(jp.FluidCoolProp, "get_state", forbidden)
    values = jax.jit(lambda x: core(model, x))
    forward = jax.jit(jax.jacfwd(lambda x: core(model, x)))
    reverse = jax.jit(jax.jacrev(lambda x: core(model, x)))
    for point in ((hc - 2e4, 0.99 * pc), (hc + 2e4, 1.005 * pc)):
        x = jnp.asarray(point)
        a, b = np.asarray(forward(x)), np.asarray(reverse(x))
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, rtol=2e-8, atol=1e-8)
        columns = []
        for i, step in enumerate((0.1, 1.0)):
            delta = jnp.eye(2)[i] * step
            columns.append((values(x + delta) - values(x - delta)) / (2 * step))
        np.testing.assert_allclose(a, np.asarray(columns).T, rtol=3e-4, atol=2e-7)


def test_critical_cap_continuity_and_declared_domains(critical_model):
    model, source = critical_model
    hc = float(source.critical_point.h)
    p = model._saturation_p_max
    values = jax.jit(lambda x: core(model, x))
    derivatives = jax.jit(jax.jacfwd(lambda x: core(model, x)))
    left, right = jnp.asarray([hc, p * (1 - 1e-8)]), jnp.asarray([hc, p * (1 + 1e-8)])
    np.testing.assert_allclose(values(left), values(right), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        derivatives(left), derivatives(right), rtol=3e-3, atol=1e-8
    )
    # HP approximation does not silently widen the public PQ endpoint domain.
    pq = model.get_state(jp.PQ_INPUTS, 1.001 * p, 0.0, errors="mask")
    assert int(pq.status) == jp.StatusCode.OUT_OF_DOMAIN
    hp = model.get_state(jp.HmassP_INPUTS, hc, model.p_max * 1.1, errors="mask")
    assert int(hp.status) == jp.StatusCode.OUT_OF_DOMAIN


def test_critical_ps_roundtrip_and_wet_pt_ambiguity(critical_model):
    model, source = critical_model
    p, h = 0.99 * float(source.critical_point.p), float(source.critical_point.h)
    state = model.get_state(jp.HmassP_INPUTS, h, p, errors="mask")
    recovered = model.get_state(jp.PSmass_INPUTS, p, state.s, errors="mask")
    assert int(recovered.status) == 0
    np.testing.assert_allclose(recovered.h, h, rtol=1e-8)
    assert float(state.is_two_phase) == 1.0
    ambiguous = model.get_state(jp.PT_INPUTS, p, state.T, errors="mask")
    assert int(ambiguous.status) != 0
