"""The public initializer selects native models without numerical wrappers."""

from functools import wraps
import inspect
import pickle
from types import SimpleNamespace

import CoolProp.CoolProp as CP
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp

METHODS = ("coolprop", "coolprop_jax", "bicubic", "perfect_gas")
CLASS_NAMES = {
    "coolprop": "FluidCoolProp",
    "coolprop_jax": "FluidJAX",
    "bicubic": "FluidBicubic",
    "perfect_gas": "FluidPerfectGas",
}
ROUTES = {
    "coolprop": ("coolprop", "real_fluid"),
    "coolprop_jax": ("coolprop_jax", "real_fluid"),
    "bicubic": ("bicubic", "real_fluid"),
    "perfect_gas": ("native", "perfect_gas"),
}


def arguments(method, table_directory, identifier="constructor-test"):
    result = dict(name="CO2", method=method, backend="HEOS", identifier=identifier)
    if method == "bicubic":
        result["table_options"] = dict(
            h_min=600e3,
            h_max=850e3,
            p_min=5e6,
            p_max=12e6,
            N_h=4,
            N_p=4,
            N_p_sat=4,
            coarse_step=1,
            table_dir=str(table_directory),
        )
    elif method == "perfect_gas":
        result["model_options"] = dict(T_ref=600.0, p_ref=1e5)
    return result


def concrete(method, settings):
    common = {key: settings[key] for key in ("name", "backend", "identifier")}
    if method == "bicubic":
        common["fluid_name"] = common.pop("name")
        common.update(settings["table_options"])
    elif method == "perfect_gas":
        common.update(settings["model_options"])
    return getattr(jxp, CLASS_NAMES[method])(**common)


@pytest.fixture(scope="module")
def configured_models(tmp_path_factory):
    directory = tmp_path_factory.mktemp("public_constructor")
    return {
        method: (
            jxp.Fluid(**(settings := arguments(method, directory))),
            concrete(method, settings),
        )
        for method in METHODS
    }


def assert_same_numerical_state(actual, expected):
    assert actual.available_properties == expected.available_properties
    assert actual.fluid_name == expected.fluid_name
    assert actual.identifier == expected.identifier
    assert actual.reference_id == expected.reference_id
    np.testing.assert_array_equal(actual.status, expected.status)
    for name in actual.available_properties:
        np.testing.assert_allclose(
            actual.raw_property(name),
            expected.raw_property(name),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            actual.property_valid(name), expected.property_valid(name)
        )


def test_fluid_is_a_real_public_runtime_type_and_preserves_source_aliases():
    assert inspect.isclass(jxp.Fluid)
    assert pickle.loads(pickle.dumps(jxp.Fluid)) is jxp.Fluid
    assert jxp.coolprop.Fluid is jxp.FluidCoolProp
    assert jxp.coolprop.Fluid is not jxp.Fluid
    assert not isinstance(jxp.FluidState(), jxp.Fluid)


@pytest.mark.parametrize("method", METHODS)
def test_constructor_returns_the_exact_native_evaluator(method, configured_models):
    fluid, direct = configured_models[method]
    native_class = getattr(jxp, CLASS_NAMES[method])
    assert type(fluid) is native_class
    assert isinstance(fluid, jxp.Fluid)
    assert issubclass(native_class, jxp.Fluid)
    assert fluid.configuration == direct.configuration
    assert fluid.capabilities == direct.capabilities
    assert fluid.references == direct.references
    assert (fluid.provider, fluid.model) == ROUTES[method]
    assert fluid.name == "CO2" and fluid.identifier == "constructor-test"
    assert fluid.backend == "HEOS"

    actual = fluid.get_state(jxp.HmassP_INPUTS, [[680e3], [720e3]], [7e6, 9e6])
    expected = direct.get_state(jxp.HmassP_INPUTS, [[680e3], [720e3]], [7e6, 9e6])
    assert actual.shape == (2, 2)
    assert_same_numerical_state(actual, expected)


@pytest.mark.parametrize("method", METHODS)
def test_selected_concrete_constructor_runs_exactly_once(method, tmp_path, monkeypatch):
    native_class = getattr(jxp, CLASS_NAMES[method])
    original = native_class.__init__
    calls = []

    @wraps(original)
    def counted(instance, *args, **kwargs):
        calls.append(instance)
        original(instance, *args, **kwargs)

    monkeypatch.setattr(native_class, "__init__", counted)
    fluid = jxp.Fluid(**arguments(method, tmp_path))
    assert type(fluid) is native_class
    assert len(calls) == 1 and calls[0] is fluid


@pytest.mark.parametrize(
    "method", [None, True, 1, [], {}, "HEOS", "CoolProp", "coolprop ", "unknown"]
)
def test_invalid_methods_fail_before_any_source_initialization(method, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("An invalid method must not initialize CoolProp")

    monkeypatch.setattr(CP, "AbstractState", forbidden)
    with pytest.raises(jxp.InvalidConfigurationError) as captured:
        jxp.Fluid("CO2", method=method)
    assert captured.value.request == "method"
    assert captured.value.fluid_name == "CO2"
    assert captured.value.reason
    assert set(METHODS) <= set(captured.value.supported)


@pytest.mark.parametrize(
    "options",
    [
        {"provider": "coolprop"},
        {"model": "perfect_gas"},
        {"exceptions": False},
        {"typo": 1},
    ],
)
def test_unknown_or_legacy_public_options_are_explicit(options, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected options must not initialize a source")

    monkeypatch.setattr(CP, "AbstractState", forbidden)
    with pytest.raises(jxp.InvalidConfigurationError) as captured:
        jxp.Fluid("CO2", **options)
    assert captured.value.request == next(iter(options))
    assert captured.value.provider == "coolprop"
    assert captured.value.model == "real_fluid"
    assert captured.value.reason
    assert "method" in captured.value.supported


def test_default_constructor_resolves_heos_and_preserves_empty_identifier():
    fluid = jxp.Fluid("CO2", identifier="")
    assert type(fluid) is jxp.FluidCoolProp
    assert fluid.identifier == fluid.configuration.identifier == ""
    assert fluid.backend == fluid.configuration.backend == "HEOS"
    assert jxp.Fluid("CO2").identifier is None


def test_binary_route_requires_the_configured_component_interface():
    component = SimpleNamespace(get_state=lambda *args: None)
    with pytest.raises(jxp.InvalidConfigurationError) as captured:
        jxp.Fluid(
            "binary",
            method="binary_mixture",
            model_options={
                "components": (component, component),
                "mass_fractions": (0.25, 0.75),
            },
        )
    assert captured.value.provider == "native"
    assert captured.value.model == "binary_mixture"
    assert captured.value.request == "components"
    assert "interface" in captured.value.reason


def test_subclassing_the_dispatch_constructor_has_an_explicit_boundary():
    class CustomFluid(jxp.Fluid):
        pass

    with pytest.raises(TypeError, match="concrete evaluator"):
        CustomFluid("CO2")


@pytest.mark.parametrize("method", METHODS[1:])
def test_constructor_keeps_native_jax_transforms_and_model_tree(
    method, configured_models
):
    fluid, direct = configured_models[method]
    h = jnp.asarray([680e3, 700e3, 720e3])
    p = jnp.asarray(8e6)
    policy = "mask" if method in {"coolprop_jax", "perfect_gas", "bicubic"} else "raise"
    evaluate = lambda enthalpy: fluid.get_state(
        jxp.HmassP_INPUTS, enthalpy, p, errors=policy
    )
    eager = evaluate(h)
    compiled = jax.jit(evaluate)(h)
    vectorized = jax.vmap(evaluate)(h)
    jax.block_until_ready((compiled, vectorized))
    assert_same_numerical_state(compiled, eager)
    assert_same_numerical_state(vectorized, eager)
    public_tree = jax.tree.structure(fluid)
    direct_tree = jax.tree.structure(direct)
    assert public_tree.num_nodes == direct_tree.num_nodes
    assert public_tree.num_leaves == direct_tree.num_leaves
    if method != "coolprop_jax":
        assert public_tree == direct_tree
    # Callback models intentionally carry distinct opaque source handles as
    # static metadata. Their full tree keys differ across separate instances.
    actual_derivative = jax.jit(jax.grad(lambda x: evaluate(x).T))(700e3)
    direct_derivative = jax.grad(
        lambda x: direct.get_state(jxp.HmassP_INPUTS, x, p, errors=policy).T
    )(700e3)
    assert np.isfinite(actual_derivative)
    np.testing.assert_allclose(
        actual_derivative, direct_derivative, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("method", ("perfect_gas", "bicubic"))
def test_publicly_constructed_model_parameters_remain_dynamic(
    method, configured_models
):
    fluid = configured_models[method][0]
    traces = []

    @jax.jit
    def temperature(model, enthalpy):
        traces.append(None)
        return model.get_state(jxp.HmassP_INPUTS, enthalpy, 8e6, errors="mask").T

    enthalpy = jnp.asarray(700e3)
    initial = temperature(fluid, enthalpy)
    if method == "perfect_gas":
        gamma = fluid.constants.gamma * 1.02
        updated = eqx.tree_at(lambda model: model.constants.gamma, fluid, gamma)
        expected = fluid.constants.T_ref + (enthalpy - fluid.constants.h_ref) / (
            gamma * fluid.constants.R / (gamma - 1)
        )
    else:
        coefficients = fluid._table["temperature"]["coeffs"] * 1.02
        updated = eqx.tree_at(
            lambda model: model._table["temperature"]["coeffs"],
            fluid,
            coefficients,
        )
        expected = initial * 1.02
    changed = temperature(updated, enthalpy)
    assert isinstance(updated, jxp.Fluid) and type(updated) is type(fluid)
    assert not np.isclose(initial, changed)
    np.testing.assert_allclose(changed, expected, rtol=1e-12, atol=1e-12)
    assert len(traces) == 1


@pytest.mark.parametrize("method", ("perfect_gas", "bicubic"))
def test_native_evaluator_round_trip_preserves_execution(method, configured_models):
    fluid = configured_models[method][0]
    restored = pickle.loads(pickle.dumps(fluid))
    assert type(restored) is type(fluid)
    assert isinstance(restored, jxp.Fluid)
    assert restored.configuration == fluid.configuration
    assert restored.references == fluid.references
    assert_same_numerical_state(
        restored.get_state(jxp.HmassP_INPUTS, 700e3, 8e6),
        fluid.get_state(jxp.HmassP_INPUTS, 700e3, 8e6),
    )


@pytest.mark.parametrize("method", ("coolprop", "coolprop_jax"))
def test_source_round_trip_requires_reconstructible_native_handles(
    method, configured_models
):
    fluid = configured_models[method][0]
    payload = pickle.dumps(fluid)
    restored = pickle.loads(payload)
    assert_same_numerical_state(
        restored.get_state(jxp.HmassP_INPUTS, 700e3, 8e6),
        fluid.get_state(jxp.HmassP_INPUTS, 700e3, 8e6),
    )
