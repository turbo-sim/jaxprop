"""Configured eager phase transactions, numerical policies and model identity."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

import CoolProp.CoolProp as CP
import jax
import numpy as np
import pytest

import jaxprop as jp
from jaxprop.coolprop import core_calculations as core
from jaxprop.fluid_api import DIAGNOSTIC_PROPERTIES, SOURCE_PROPERTIES
from utilities import get_available_backends

CORE = ("pressure", "temperature", "density", "enthalpy", "entropy", "internal_energy")


@pytest.fixture(scope="module", params=get_available_backends())
def fluid(request):
    return jp.Fluid("CO2", backend=request.param, identifier="phase-study")


def test_advanced_phase_capabilities_describe_only_the_implemented_provider(fluid):
    advanced = {"equilibrium", "metastable", "blending"}
    assert advanced <= fluid.capabilities.phase_operations
    callback = jp.Fluid("CO2", method="coolprop_jax", backend=fluid.backend)
    assert not advanced & callback.capabilities.phase_operations
    assert "saturation" in callback.capabilities.phase_operations


@pytest.mark.parametrize("method", ("get_state_equilibrium", "get_state_metastable"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("diagnostics", (False, True))
def test_actual_direct_phase_states_have_precise_schema_and_identity(
    fluid, method, reverse, diagnostics
):
    expected = fluid.get_state(jp.PT_INPUTS, 2e5, 400.0)
    arguments = ("density", expected.rho, "temperature", expected.T)
    if reverse:
        arguments = ("T", expected.T, "dmass", expected.rho)
    actual = getattr(fluid, method)(
        *arguments, supersaturation=diagnostics, generalize_quality=False
    )
    assert actual.shape == () and actual.valid and fluid.converged_flag
    assert actual.identifier == "phase-study"
    assert actual.reference_id == fluid.reference_id and actual.model == fluid.model
    assert actual.available_properties == SOURCE_PROPERTIES | (
        DIAGNOSTIC_PROPERTIES if diagnostics else frozenset()
    )
    for name in CORE:
        np.testing.assert_allclose(
            actual.raw_property(name), expected.raw_property(name), rtol=2e-9
        )
    with pytest.raises(jp.UnsupportedPropertyError):
        _ = actual.mass_frac_1


@pytest.mark.parametrize("method", ("get_state_equilibrium", "get_state_metastable"))
@pytest.mark.parametrize("value", (np.nan, np.inf))
def test_nonfinite_phase_inputs_raise_or_mask_without_losing_schema(
    fluid, method, value
):
    function = getattr(fluid, method)
    with pytest.raises(jp.InvalidStateError) as error:
        function("rho", 20.0, "T", value, supersaturation=False)
    assert error.value.fluid_name == fluid.name and error.value.index == ()
    state = function("rho", 20.0, "T", value, supersaturation=False, errors="mask")
    assert state.status == jp.StatusCode.INVALID_INPUT and not fluid.converged_flag
    assert state.available_properties == SOURCE_PROPERTIES
    assert all(
        np.isnan(state.raw_property(name)) for name in state.available_properties
    )


@pytest.mark.parametrize("value", (True, "400", [400.0], 400j, [[1.0], [1.0, 2.0]]))
def test_malformed_scalar_requests_still_raise_in_mask_mode(fluid, value):
    with pytest.raises(jp.InvalidStateError):
        fluid.get_state_metastable("rho", 20.0, "T", value, errors="mask")


@pytest.mark.parametrize(
    "property_name", ("mass_frac_1", "speed_of_sound_p", "subcooling")
)
def test_unsupported_or_disabled_input_fields_raise_before_numerical_masking(
    fluid, property_name
):
    with pytest.raises(jp.UnsupportedPropertyError) as error:
        fluid.get_state_equilibrium(
            property_name, 1.0, "T", 400.0, supersaturation=False, errors="mask"
        )
    assert error.value.request == property_name and error.value.fluid_name == fluid.name


@pytest.mark.parametrize("method", ("get_state_equilibrium", "get_state_metastable"))
def test_source_phase_clip_and_transformed_execution_are_explicitly_unsupported(
    fluid, method
):
    function = getattr(fluid, method)
    with pytest.raises(jp.UnsupportedExecutionModeError, match="projection"):
        function("rho", 20.0, "T", 400.0, errors="clip")
    with pytest.raises(jp.UnsupportedExecutionModeError, match="eager scalar"):
        jax.jit(
            lambda temperature: function("rho", 20.0, "T", temperature, errors="mask")
        )(400.0)


@pytest.mark.parametrize("failure, code", (("root", 3), ("provider", 4)))
def test_phase_failure_status_and_original_exception_are_retained(
    fluid, monkeypatch, failure, code
):
    cause = (
        jp.ConvergenceError(
            request="metastable", reason="Independent residual check failed."
        )
        if failure == "root"
        else RuntimeError("source flash failed")
    )

    def fail(*args, **kwargs):
        assert kwargs["source_context"] is fluid._source_context
        raise cause

    monkeypatch.setattr(core, "compute_properties", fail)
    state = fluid.get_state_metastable("rho", 20.0, "T", 400.0, errors="mask")
    assert state.status == code and not fluid.converged_flag
    with pytest.raises(
        jp.ConvergenceError if failure == "root" else jp.CalculationError
    ) as error:
        fluid.get_state_metastable("rho", 20.0, "T", 400.0)
    assert error.value.fluid_name == fluid.name and error.value.__cause__ is cause
    assert error.value.index == ()


def test_blending_is_a_separate_constitutive_model_with_canonical_settings(fluid):
    reference = fluid.get_state(jp.PT_INPUTS, 2e5, 400.0)
    settings = dict(
        rhoT_guess_equilibrium=(reference.rho, reference.T),
        rhoT_guess_metastable=(reference.rho, reference.T),
        blending_onset=0.5,
        blending_width=0.2,
        phase_change="flashing",
        supersaturation=False,
        generalize_quality=False,
    )
    args = ("rho", reference.rho, "T", reference.T)
    blended, equilibrium, metastable = fluid.get_state_blending(*args, **settings)
    assert blended.valid and equilibrium.valid and metastable.valid
    assert blended.model == "blended_real_fluid"
    assert equilibrium.model == metastable.model == fluid.model
    assert blended.reference_id != equilibrium.reference_id == metastable.reference_id
    assert all(
        state.available_properties == SOURCE_PROPERTIES
        for state in (blended, equilibrium, metastable)
    )
    for name in CORE:
        np.testing.assert_allclose(
            blended.raw_property(name), reference.raw_property(name), rtol=2e-8
        )
    same = fluid.get_state_blending(*args, blending_variable="Q", **settings)[0]
    assert same.reference_id == blended.reference_id
    alias = fluid.get_state_blending(
        *args, **(settings | {"phase_change": "evaporation"})
    )[0]
    assert alias.reference_id == blended.reference_id
    scalar_arrays = fluid.get_state_blending(
        *args,
        **(
            settings
            | {
                "blending_onset": np.asarray(0.5),
                "blending_width": np.asarray(0.2),
            }
        )
    )[0]
    assert scalar_arrays.reference_id == blended.reference_id
    changed = fluid.get_state_blending(*args, **(settings | {"blending_width": 0.3}))[0]
    assert changed.reference_id != blended.reference_id
    extended = fluid.get_state_blending(
        *args, **(settings | {"generalize_quality": True})
    )[0]
    assert extended.reference_id != blended.reference_id


def test_shared_source_phase_calls_are_isolated_across_threads(fluid):
    oracle = CP.AbstractState(fluid.backend, fluid.name)
    cases = []
    for temperature in np.linspace(330.0, 430.0, 12):
        oracle.update(CP.PT_INPUTS, 5e6, temperature)
        cases.append((oracle.rhomass(), temperature, oracle.hmass(), oracle.smass()))

    def evaluate(case):
        rho, temperature, enthalpy, entropy = case
        state = fluid.get_state_metastable(
            "rho", rho, "T", temperature, supersaturation=True, generalize_quality=True
        )
        return state, enthalpy, entropy

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(evaluate, cases))
    for state, enthalpy, entropy in results:
        assert state.valid
        np.testing.assert_allclose((state.h, state.s), (enthalpy, entropy), rtol=2e-9)


def test_configured_legacy_phase_helpers_keep_the_original_reference_basis():
    script = r"""
import CoolProp.CoolProp as CP
import numpy as np
import jaxprop as jp
fluid = jp.Fluid("CO2")
original = fluid.get_state(jp.PT_INPUTS, 2e5, 400.0)
identity = fluid.reference_id
CP.set_reference_state("CO2", "ASHRAE")
def forbidden(*args, **kwargs):
    raise AssertionError("A configured phase call reconstructed a source.")
CP.AbstractState = forbidden
for method in (fluid.get_state_equilibrium, fluid.get_state_metastable):
    actual = method("rho", original.rho, "T", original.T,
                    supersaturation=True, generalize_quality=True)
    np.testing.assert_allclose((actual.h, actual.s), (original.h, original.s), rtol=2e-9)
    assert actual.reference_id == identity
print("configured reference retained")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "configured reference retained" in result.stdout
