"""Real numerical generation/loading behind the bicubic cache manifest."""

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

import jaxprop as jxp
from jaxprop.bicubic.table_cache import table_validation_error


def _settings(directory, **overrides):
    settings = dict(
        fluid_name="CO2",
        backend="HEOS",
        h_min=600e3,
        h_max=850e3,
        p_min=5e6,
        p_max=12e6,
        N_h=4,
        N_p=4,
        N_p_sat=4,
        coarse_step=1,
        table_dir=str(directory),
    )
    return settings | overrides


def _cache_path(fluid):
    return Path(fluid.table_dir) / f"{fluid.table_name}.pkl"


@pytest.mark.parametrize("method", ["forward", "central"])
def test_real_equilibrium_table_is_complete_and_reload_preserves_values(
    tmp_path, monkeypatch, method
):
    settings = _settings(tmp_path, gradient_method=method)
    first = jxp.FluidBicubic(**settings)
    cache = _cache_path(first)
    assert cache.is_file()
    assert table_validation_error(first.table, first.metadata) is None
    assert np.all(np.isnan(np.asarray(first.table["surface_tension"]["value"])))

    # The guard proves the numerical generator is not revisited by this load.
    def unexpected_generation(fluid):
        raise AssertionError("matching numerical cache must be loaded")

    monkeypatch.setattr(
        jxp.FluidBicubic, "_generate_property_table", unexpected_generation
    )
    restored = jxp.FluidBicubic(**settings)
    assert restored.table_name == first.table_name
    assert restored.metadata == first.metadata
    with cache.open("rb") as stream:
        saved = pickle.load(stream)
    assert saved["metadata"] == first.metadata
    for name in jxp.PROPERTIES_CANONICAL:
        for field, values in first.table[name].items():
            np.testing.assert_array_equal(restored.table[name][field], values)
    for name in (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    ):
        np.testing.assert_array_equal(
            first.get_state(jxp.HmassP_INPUTS, 700e3, 8e6).raw_property(name),
            restored.get_state(jxp.HmassP_INPUTS, 700e3, 8e6).raw_property(name),
        )


def test_real_forward_and_central_tables_have_distinct_cache_identities(tmp_path):
    forward = jxp.FluidBicubic(**_settings(tmp_path, gradient_method="forward"))
    central = jxp.FluidBicubic(**_settings(tmp_path, gradient_method="central"))
    assert forward.metadata["gradient_method"] == "forward"
    assert central.metadata["gradient_method"] == "central"
    assert forward.table_name != central.table_name
    assert _cache_path(forward).is_file() and _cache_path(central).is_file()
    # This curved real-fluid property distinguishes the numerical recipes.
    assert not np.array_equal(
        forward.table["temperature"]["grad_h"],
        central.table["temperature"]["grad_h"],
    )


def test_real_binary_table_persists_composition_and_loads_complete_arrays(
    tmp_path, monkeypatch
):
    settings = _settings(
        tmp_path,
        fluid_name="water_nitrogen_mixture",
        mixture_ratio=0.1,
        h_min=500e3,
        h_max=510e3,
        p_min=8e6,
        p_max=9e6,
        N_h=3,
        N_p=3,
        gradient_method="central",
    )
    first = jxp.FluidBicubic(**settings)
    assert first.table["metadata"]["mixture_ratio"] == 0.1
    assert first.table["metadata"]["table_mode"] == "mixture"
    assert table_validation_error(first.table, first.metadata) is None

    def unexpected_generation(fluid):
        raise AssertionError("matching binary cache must be loaded")

    monkeypatch.setattr(
        jxp.FluidBicubic, "_generate_mixture_property_table", unexpected_generation
    )
    restored = jxp.FluidBicubic(**settings)
    for name in jxp.PROPERTIES_CANONICAL:
        for field, value in first.table[name].items():
            np.testing.assert_array_equal(restored.table[name][field], value)
    state = restored.get_state(jxp.HmassP_INPUTS, 505e3, 8.5e6)
    assert all(
        np.isfinite(state.raw_property(name)) for name in ("T", "rho", "h", "s", "u")
    )


@pytest.mark.parametrize(
    "phase,h_min,h_max,p_min,p_max",
    [
        ("liquid", 100e3, 110e3, 1e6, 2e6),
        ("vapor", 3e6, 3.1e6, 1e5, 2e5),
    ],
)
def test_real_metastable_tables_preserve_variation_saturation_and_cache_data(
    tmp_path, monkeypatch, phase, h_min, h_max, p_min, p_max
):
    settings = _settings(
        tmp_path,
        fluid_name="Water",
        h_min=h_min,
        h_max=h_max,
        p_min=p_min,
        p_max=p_max,
        N_h=3,
        N_p=3,
        N_p_sat=3,
        gradient_method="forward",
        metastable_phase=phase,
    )
    first = jxp.FluidBicubic(**settings)
    assert first.table["metadata"] == first.metadata
    assert first.metadata["table_mode"] == f"metastable_{phase}"
    assert first.metadata["metastable_phase"] == phase
    assert table_validation_error(first.table, first.metadata) is None
    for name in ("temperature", "density"):
        values = np.asarray(first.table[name]["value"])
        assert np.all(np.ptp(values, axis=0) > 0), name
    for name in (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    ):
        for values in first.table["saturation_props"][name].values():
            assert np.shape(values) == (3,)
            assert np.all(np.isfinite(values))

    def unexpected_generation(fluid):
        raise AssertionError("matching metastable cache must be loaded")

    monkeypatch.setattr(
        jxp.FluidBicubic,
        f"_generate_metastable_{phase}_property_table",
        unexpected_generation,
    )
    restored = jxp.FluidBicubic(**settings)
    assert restored.metadata == first.metadata
    for name in jxp.PROPERTIES_CANONICAL:
        for field, value in first.table[name].items():
            np.testing.assert_array_equal(restored.table[name][field], value)
        for field, value in first.table["saturation_props"][name].items():
            np.testing.assert_array_equal(
                restored.table["saturation_props"][name][field], value
            )


def test_failed_real_gradient_does_not_replace_the_last_valid_cache(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path, gradient_method="forward", table_name="shared")
    first = jxp.FluidBicubic(**settings)
    path = _cache_path(first)
    previous = path.read_bytes()

    def failed_sample(*args, **kwargs):
        raise RuntimeError("deliberately unavailable source sample")

    monkeypatch.setattr(
        jxp.FluidBicubic, "_gradients_central", staticmethod(failed_sample)
    )
    with pytest.warns(UserWarning, match="Regenerating.*metadata"):
        with pytest.raises(
            jxp.CalculationError, match="deliberately unavailable source sample"
        ):
            jxp.FluidBicubic(**(settings | {"gradient_method": "central"}))
    assert path.read_bytes() == previous
    assert sorted(item.name for item in tmp_path.iterdir()) == ["shared.pkl"]


def test_reference_change_regenerates_a_real_cache_in_an_isolated_process(tmp_path):
    """Changing CoolProp's global reference must never affect the parent app."""
    repo = Path(__file__).resolve().parents[1]
    script = r"""
import json
import sys
import numpy as np
import CoolProp.CoolProp as CP
import jaxprop as jxp

settings = dict(fluid_name="CO2",backend="HEOS",h_min=600e3,h_max=700e3,
                p_min=8e6,p_max=9e6,N_h=2,N_p=2,N_p_sat=2,coarse_step=1,
                table_dir=sys.argv[1],table_name="reference_change")
calls = []
original = jxp.FluidBicubic._generate_property_table
def counted(fluid):
    calls.append(fluid.metadata["source_signature"])
    return original(fluid)
jxp.FluidBicubic._generate_property_table = counted

first = jxp.FluidBicubic(**settings)
jxp.FluidBicubic(**settings)
anchor = CP.AbstractState("HEOS", "CO2")
anchor.update(CP.PT_INPUTS, 1e5, 300.0)
# Shift enthalpy by only 1 kJ/kg, retaining the same physical table domain.
CP.set_reference_state("CO2",300.0,anchor.rhomolar(),
                       anchor.hmolar()+1000*anchor.molar_mass(),anchor.smolar())
second = jxp.FluidBicubic(**settings)
jxp.FluidBicubic(**settings)
print(json.dumps({"generations":len(calls),
                  "source_changed":calls[0] != calls[1],
                  "metadata_changed":first.metadata != second.metadata,
                  "numerical_values_changed":not np.array_equal(first.table["temperature"]["value"],second.table["temperature"]["value"])}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    outcomes = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcomes == {
        "generations": 2,
        "source_changed": True,
        "metadata_changed": True,
        "numerical_values_changed": True,
    }
