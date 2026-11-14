"""Source-envelope geometry and reference-invariant table generation tests."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import CoolProp.CoolProp as CP
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.bicubic.phase_envelope import source_phase_cell_validity
from jaxprop.bicubic import phase_envelope
from jaxprop.bicubic.runtime import prepare_runtime
from jaxprop.bicubic.table_cache import table_validation_error

from test_bicubic_cache import cache_factory

BACKENDS = os.environ.get("JAXPROP_TEST_BACKENDS", "HEOS").split(",")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "name,pressures",
    [
        ("Water", (1e6, 8e6)),
        ("Air", (2e5, 3e6)),
        ("R410A", (5e5, 4e6)),
    ],
)
def test_actual_curved_dew_boundary_can_enter_a_cell_with_all_vapor_corners(
    backend, name, pressures
):
    source = CP.AbstractState(backend, name)
    sampled_p = np.geomspace(*pressures, 65)
    dew = []
    for pressure in sampled_p:
        source.update(CP.PQ_INPUTS, float(pressure), 1.0)
        dew.append(source.hmass())
    maximum = int(np.argmax(dew))
    endpoint_maximum = max(dew[0], dew[-1])
    assert dew[maximum] > endpoint_maximum + 100.0
    gap = dew[maximum] - endpoint_maximum
    enthalpies = endpoint_maximum + gap * np.array([0.3, 0.6])
    phases = np.empty((2, 2))
    for index in np.ndindex(phases.shape):
        source.update(CP.HmassP_INPUTS, enthalpies[index[0]], pressures[index[1]])
        phases[index] = float(source.phase() == CP.iphase_twophase)
    np.testing.assert_array_equal(phases, 0)
    source.update(CP.HmassP_INPUTS, float(np.mean(enthalpies)), sampled_p[maximum])
    assert source.phase() == CP.iphase_twophase and 0 < source.Q() < 1
    old_table = {
        "temperature": {"coeffs": np.zeros((1, 1, 16))},
        "is_two_phase": {"value": phases},
        "quality_mass": {"value": np.ones((2, 2))},
        "pressure": {"value": np.broadcast_to(pressures, (2, 2))},
    }
    _, _, nodal = prepare_runtime(
        old_table, frozenset(), "equilibrium", source.p_critical()
    )
    assert bool(nodal[0, 0])
    actual = source_phase_cell_validity(
        source,
        enthalpies,
        np.log(pressures),
        critical_pressure=source.p_critical(),
    )
    assert not actual[0, 0]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("quality", (0.0, 1.0))
def test_envelope_guard_rejects_both_source_saturation_boundaries(backend, quality):
    source = CP.AbstractState(backend, "Water")
    pressure = 2e6
    source.update(CP.PQ_INPUTS, pressure, quality)
    h = source.hmass()
    cells = source_phase_cell_validity(
        source,
        [h - 1000, h + 1000],
        np.log([0.99 * pressure, 1.01 * pressure]),
        critical_pressure=source.p_critical(),
    )
    assert not cells[0, 0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_cells_wholly_inside_two_phase_region_remain_eligible(backend):
    source = CP.AbstractState(backend, "Water")
    pressure = 2.1e6
    source.update(CP.PQ_INPUTS, pressure, 0)
    liquid = source.hmass()
    source.update(CP.PQ_INPUTS, pressure, 1)
    vapor = source.hmass()
    h = liquid + (vapor - liquid) * np.array([0.4, 0.6])
    cells = source_phase_cell_validity(
        source,
        h,
        np.log([2e6, 2.2e6]),
        critical_pressure=source.p_critical(),
    )
    assert cells[0, 0]


class SyntheticEnvelope:
    """A narrow curved boundary that is invisible at strip endpoints."""

    def update(self, pair, pressure, quality):
        assert pair == CP.PQ_INPUTS
        self.point, self.quality = np.log(pressure), quality

    def hmass(self):
        return (10.0 if self.quality == 0 else 30.0) + 8 * np.sin(np.pi * self.point)


def test_adaptive_source_guard_catches_interior_bend_and_preserves_separate_regions():
    cells = source_phase_cell_validity(
        SyntheticEnvelope(),
        [0, 5, 12, 15, 20, 25, 32, 35, 40, 45],
        [0, 1],
        critical_pressure=100,
    )
    np.testing.assert_array_equal(
        cells[:, 0], [True, False, False, False, True, False, False, False, True]
    )


def test_envelope_source_failures_invalidate_only_affected_subcritical_strip():
    class FailedSource:
        def update(self, *request):
            raise ValueError("No valid source endpoint")

    cells = source_phase_cell_validity(
        FailedSource(), [0, 1, 2], np.log([1, 2, 4]), critical_pressure=2
    )
    np.testing.assert_array_equal(cells, [[False, True], [False, True]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_critical_endpoint_is_included_without_masking_far_single_phase_cells(backend):
    source = CP.AbstractState(backend, "CO2")
    pc, tc = source.p_critical(), source.T_critical()
    source.update(CP.DmassT_INPUTS, source.rhomass_critical(), tc)
    hc = source.hmass()
    cells = source_phase_cell_validity(
        source,
        hc + np.array([-2e5, -1e5, -500, 500, 1e5, 2e5]),
        np.log([0.97 * pc, 1.03 * pc]),
        critical_pressure=pc,
        critical_enthalpy=hc,
    )
    assert cells[0, 0] and cells[-1, 0]
    assert not cells[2, 0]


def test_persisted_guard_is_validated_without_casting_malformed_cache_data(
    cache_factory,
):
    create, _, _ = cache_factory
    model = create()
    assert model.table["phase_cell_validity"].dtype == np.bool_
    for invalid in (np.ones((1, 1), dtype=bool), np.ones((3, 3), dtype=float)):
        table = deepcopy(model.table)
        table["phase_cell_validity"] = invalid
        assert "phase-cell validity" in table_validation_error(table, model.metadata)
    assert model.metadata["cache_format_version"] == 2
    assert model.metadata["generation_settings"]["phase_envelope"]["policy"]


def test_cached_guard_load_and_jit_evaluation_do_not_query_source_envelope(
    cache_factory, monkeypatch
):
    create, generated, _ = cache_factory
    create()

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached/native evaluation queried the phase envelope")

    monkeypatch.setattr(jp.FluidBicubic, "_source_phase_cell_validity", forbidden)
    restored = create()
    assert generated == ["equilibrium"]
    state = jax.jit(
        lambda h: restored.get_state(jp.HmassP_INPUTS, h, 8e6, errors="mask")
    )(jnp.asarray(700e3))
    assert state.status == 0


def test_generation_uses_the_envelope_policy_captured_in_its_manifest(
    cache_factory, monkeypatch
):
    create, _, _ = cache_factory
    original_generator = jp.FluidBicubic._generate_property_table

    def changed_default_during_generation(model):
        table = original_generator(model)
        # A deliberately unusable new default would reject every probed strip
        # if generation rediscovered it after capturing the table manifest.
        monkeypatch.setitem(
            phase_envelope.PHASE_ENVELOPE_SETTINGS, "minimum_segments", 0
        )
        return table

    monkeypatch.setattr(
        jp.FluidBicubic, "_generate_property_table", changed_default_during_generation
    )
    model = create()
    assert (
        model.metadata["generation_settings"]["phase_envelope"]["minimum_segments"] == 4
    )
    np.testing.assert_array_equal(model.table["phase_cell_validity"], True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_public_table_rejects_real_interior_phase_boundary(backend, tmp_path):
    model = jp.FluidBicubic(
        "Water",
        backend,
        2.79e6,
        2.795e6,
        1e6,
        8e6,
        2,
        2,
        N_p_sat=4,
        coarse_step=1,
        table_dir=str(tmp_path),
    )
    np.testing.assert_array_equal(model.table["is_two_phase"]["value"], 0)
    assert not bool(model._cell_validity[0, 0])
    for policy in ("mask", "clip"):
        state = model.get_state(jp.HmassP_INPUTS, 2.792e6, 3e6, errors=policy)
        assert state.status == 2 and not state.clipped and np.isnan(state.T)


def test_metastable_table_stencils_ignore_enthalpy_reference_offset(tmp_path):
    # Isolate CoolProp's process-global reference API from parallel tests.
    script = r"""
import json
import sys
from pathlib import Path
import CoolProp.CoolProp as CP
import numpy as np
import jaxprop as jp
root = Path(sys.argv[1])
source = CP.AbstractState("HEOS", "Water")
source.update(CP.PT_INPUTS, 1e6, 300.)
h, s, density, mass = source.hmass(), source.smass(), source.rhomass(), source.molar_mass()
settings = dict(fluid_name="Water", backend="HEOS", h_min=h-2000, h_max=h+2000,
                p_min=9e5, p_max=1.1e6, N_h=3, N_p=3, N_p_sat=3,
                coarse_step=1, gradient_method="forward", metastable_phase="liquid")
first = jp.FluidBicubic(**settings, table_dir=str(root / "original"))
offset = 1e8
try:
    CP.set_reference_state("Water", 300., density/mass, (h+offset)*mass, s*mass)
    shifted = jp.FluidBicubic(**(settings | {"h_min":h+offset-2000,"h_max":h+offset+2000}),
                              table_dir=str(root / "shifted"))
    for name in ("temperature", "density", "entropy"):
        for field in ("value", "grad_h", "grad_p", "grad_ph"):
            np.testing.assert_allclose(np.asarray(first.table[name][field]),
                                       np.asarray(shifted.table[name][field]),
                                       rtol=8e-3 if field=="grad_ph" else 1e-4,
                                       atol=2e-9 if field=="grad_ph" else 2e-8)
    assert first.reference_id != shifted.reference_id
    assert shifted.metadata["generation_settings"]["metastable_enthalpy_step_scale"]=="table_enthalpy_span"
finally:
    CP.set_reference_state("Water", "DEF")
print("REFERENCE_INVARIANT_STENCIL_OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "REFERENCE_INVARIANT_STENCIL_OK" in completed.stdout


@pytest.mark.parametrize("enthalpy", (1e16, -1e16))
def test_unresolvable_metastable_stencil_is_an_explicit_generation_error(enthalpy):
    class UnusedSource:
        name = "OffsetSource"

        def get_state_metastable(self, **request):
            raise AssertionError("An unresolved stencil must fail before source work")

    with pytest.raises(jp.CalculationError, match="cannot be represented accurately"):
        jp.FluidBicubic._gradients_metastable_forward(
            UnusedSource(), enthalpy, 1e6, 0.1, 10.0, 1000.0, 300.0
        )
