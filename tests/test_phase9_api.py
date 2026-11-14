"""Public binary-fluid construction, complete states and provider integration."""

import os
import pickle

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jp

CORE = ("pressure", "temperature", "density", "enthalpy", "entropy", "internal_energy")


@pytest.fixture(scope="module")
def gases():
    return tuple(
        jp.Fluid(
            name, method="perfect_gas", model_options={"T_ref": 400.0, "p_ref": 1e5}
        )
        for name in ("Nitrogen", "Oxygen")
    )


def binary(components, fractions=(0.25, 0.75), **options):
    return jp.Fluid(
        "working-mixture",
        method="binary_mixture",
        identifier="mixture-stream",
        model_options={
            "components": components,
            "mass_fractions": fractions,
            **options,
        },
    )


@pytest.fixture(scope="module")
def mixture(gases):
    return binary(gases)


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    return jp.Fluid(
        "CO2",
        method="bicubic",
        table_options=dict(
            h_min=600e3,
            h_max=850e3,
            p_min=5e6,
            p_max=12e6,
            N_h=6,
            N_p=6,
            N_p_sat=4,
            coarse_step=1,
            table_dir=str(tmp_path_factory.mktemp("phase9_binary_table")),
        ),
    )


def assert_same_fields(actual, expected, names=None, *, tolerance=2e-10):
    for name in names or (actual.available_properties & expected.available_properties):
        np.testing.assert_allclose(
            actual.raw_property(name),
            expected.raw_property(name),
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
        )


def test_factory_returns_real_native_binary_model_and_preserves_identity(
    mixture, gases
):
    assert isinstance(mixture, jp.Fluid)
    assert isinstance(mixture, jp.FluidBinaryMixture)
    assert mixture.components == gases
    assert mixture.name == "working-mixture" and mixture.identifier == "mixture-stream"
    assert (mixture.provider, mixture.model, mixture.backend) == (
        "native",
        "binary_mixture",
        None,
    )
    assert mixture.mass_fractions == (0.25, 0.75)
    assert mixture.capabilities.input_pairs == {
        jp.PT_INPUTS,
        jp.HmassP_INPUTS,
        jp.PSmass_INPUTS,
    }
    assert mixture.capabilities.error_modes == {"raise", "mask", "clip"}
    assert {"jit", "vmap", "jvp", "grad"} <= mixture.capabilities.jax_modes
    assert "saturation" not in mixture.capabilities.phase_operations
    assert "quality_mass" not in mixture.capabilities.properties
    state = mixture.get_state(jp.PT_INPUTS, 2e5, 500.0)
    assert state.fluid_name == mixture.name and state.identifier == mixture.identifier
    assert (
        state.reference_id == mixture.reference_id and state.model == "binary_mixture"
    )


def test_constructor_snapshots_options_without_retaining_mutable_containers(gases):
    settings = {"components": list(gases), "mass_fractions": [0.25, 0.75]}
    fluid = jp.Fluid("snapshot", method="binary_mixture", model_options=settings)
    settings["components"][0] = gases[1]
    settings["mass_fractions"][0] = 1.0
    assert fluid.components[0] is gases[0] and fluid.mass_fractions == (0.25, 0.75)
    assert fluid.configuration.model_options["components"] == tuple(
        c.configuration for c in gases
    )
    with pytest.raises((AttributeError, TypeError)):
        fluid.mass_fractions = (1.0, 0.0)
    with pytest.raises(TypeError):
        fluid.configuration.model_options["mass_fractions"] = (1.0, 0.0)


def test_composite_references_are_explicit_and_components_remain_accessible(
    mixture, gases
):
    assert mixture.references.components == tuple(gas.references for gas in gases)
    for key in ("critical_point", "triple_point_liquid", "triple_point_vapor"):
        reference = mixture.references[key]
        assert not reference.available and reference.reason
        with pytest.raises(jp.UnavailableReferencePointError) as captured:
            _ = reference.state
        assert captured.value.provider == "native"
        assert captured.value.model == "binary_mixture"
    assert gases[0].references["critical_point"].available


@pytest.mark.parametrize("policy", ["raise", "mask", "clip"])
def test_complete_broadcast_states_have_one_common_shape_and_valid_composition(
    mixture, policy
):
    pressure, temperature = np.array([[2e5], [4e5]]), np.array([[300.0, 500.0, 700.0]])
    state = mixture.get_state(jp.PT_INPUTS, pressure, temperature, errors=policy)
    assert state.shape == (2, 3) and state.error_policy == policy
    np.testing.assert_array_equal(state.status, 0)
    np.testing.assert_array_equal(state.clipped, False)
    for name in state.available_properties:
        assert state.raw_property(name).shape == (2, 3)
    np.testing.assert_array_equal(state.mass_frac_1, 0.25)
    np.testing.assert_array_equal(state.mass_frac_2, 0.75)
    np.testing.assert_allclose(state.vol_frac_1 + state.vol_frac_2, 1, rtol=2e-15)
    np.testing.assert_allclose(state.u + state.p / state.rho, state.h, rtol=2e-14)
    np.testing.assert_allclose(state.a**2 * state.rho * state.kappa_s, 1.0, rtol=2e-14)


@pytest.mark.parametrize("pair", [jp.HmassP_INPUTS, jp.PSmass_INPUTS])
def test_full_batch_inverse_recovers_entire_pt_state(mixture, pair):
    pressure, temperature = jnp.linspace(2e5, 8e5, 9), jnp.linspace(250.0, 800.0, 9)
    expected = mixture.get_state(jp.PT_INPUTS, pressure, temperature)
    first, second = (
        (expected.h, pressure) if pair == jp.HmassP_INPUTS else (pressure, expected.s)
    )
    actual = jax.jit(lambda a, b: mixture.get_state(pair, a, b, errors="mask"))(
        first, second
    )
    np.testing.assert_array_equal(actual.status, 0)
    assert_same_fields(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("pair", [jp.PT_INPUTS, jp.HmassP_INPUTS, jp.PSmass_INPUTS])
def test_explicit_model_precision_is_preserved_in_all_complete_state_fields(
    gases, dtype, pair
):
    components = tuple(
        gas.with_constants(
            **{
                name: jnp.asarray(value, dtype=dtype)
                for name, value in vars(gas.constants).items()
            }
        )
        for gas in gases
    )
    fluid = binary(components)
    pressure = jnp.asarray([2e5, 3.5e5], dtype=dtype)
    temperature = jnp.asarray([425.0, 610.0], dtype=dtype)
    p, T = np.asarray(pressure, dtype=np.float64), np.asarray(
        temperature, dtype=np.float64
    )
    weights = fluid.mass_fractions

    # Independent ideal-mixture equations use the actual rounded model
    # coefficients, rather than a separately calibrated double-precision gas.
    gas_constant = 0.0
    heat_capacity = 0.0
    enthalpy, entropy = np.zeros_like(T), np.zeros_like(T)
    for component, weight in zip(components, weights):
        c = component.constants
        R, gamma = float(c.R), float(c.gamma)
        cp = gamma * R / (gamma - 1)
        gas_constant += weight * R
        heat_capacity += weight * cp
        enthalpy += weight * (float(c.h_ref) + cp * (T - float(c.T_ref)))
        entropy += weight * (
            float(c.s_ref)
            + cp * np.log(T / float(c.T_ref))
            - R * np.log(p / float(c.P_ref))
        )
    expected = {
        "pressure": p,
        "temperature": T,
        "density": p / (gas_constant * T),
        "enthalpy": enthalpy,
        "entropy": entropy,
        "internal_energy": enthalpy - gas_constant * T,
        "isobaric_heat_capacity": np.full_like(T, heat_capacity),
        "isochoric_heat_capacity": np.full_like(T, heat_capacity - gas_constant),
    }
    if pair == jp.PT_INPUTS:
        first, second = pressure, temperature
    elif pair == jp.HmassP_INPUTS:
        first, second = jnp.asarray(enthalpy, dtype=dtype), pressure
    else:
        first, second = pressure, jnp.asarray(entropy, dtype=dtype)
    eager = fluid.get_state(pair, first, second, errors="mask")
    compiled = jax.jit(lambda a, b: fluid.get_state(pair, a, b, errors="mask"))(
        first, second
    )
    tolerance = 4e-6 if dtype == np.float32 else 2e-12
    jt_fields = {"joule_thomson", "isothermal_joule_thomson"}
    for state in (eager, compiled):
        assert len(state.available_properties) == 27
        np.testing.assert_array_equal(state.status, 0)
        for name in state.available_properties:
            assert state.raw_property(name).dtype == np.dtype(dtype), name
            # The JT coefficients vanish for this model. Cancellation in their
            # volume identities needs an absolute precision allowance at zero.
            absolute = (
                (2e-7 if dtype == np.float32 else 2e-14) if name in jt_fields else 0.0
            )
            np.testing.assert_allclose(
                state.raw_property(name),
                eager.raw_property(name),
                rtol=tolerance,
                atol=absolute,
                equal_nan=True,
            )
        for name, value in expected.items():
            np.testing.assert_allclose(
                state.raw_property(name), value, rtol=tolerance, atol=0.0
            )
        np.testing.assert_allclose(
            state.u + state.p / state.rho, state.h, rtol=tolerance
        )
        np.testing.assert_allclose(
            state.a**2 * state.rho * state.kappa_s, 1.0, rtol=tolerance
        )


@pytest.mark.parametrize("fractions", [(1.0, 0.0), (0.0, 1.0)])
def test_pure_limits_recover_the_active_component_exactly(gases, fractions):
    fluid = binary(gases, fractions)
    active = gases[0] if fractions[0] else gases[1]
    expected = active.get_state(jp.PT_INPUTS, 2e5, 500.0)
    actual = fluid.get_state(jp.PT_INPUTS, 2e5, 500.0)
    assert_same_fields(actual, expected, tolerance=0.0)
    np.testing.assert_array_equal(actual.raw_property("speed_of_sound_p"), expected.a)
    np.testing.assert_array_equal(actual.raw_property("speed_of_sound_pT"), expected.a)
    assert actual.valid
    if fractions[1] == 0:
        assert np.isnan(actual.raw_property("mixture_ratio"))
        assert not actual.property_valid("mixture_ratio")


def test_inactive_source_is_never_evaluated_or_used_to_restrict_native_domain(
    gases, monkeypatch
):
    inactive = jp.Fluid("Water", method="coolprop")
    pressure = 2 * inactive.p_max

    def forbidden(*args, **kwargs):
        raise AssertionError("A zero-weight component must not be evaluated")

    monkeypatch.setattr(inactive, "get_state", forbidden)
    fluid = binary((gases[0], inactive), (1.0, 0.0))
    assert {"jit", "vmap", "grad"} <= fluid.capabilities.jax_modes
    expected = gases[0].get_state(jp.PT_INPUTS, pressure, 500.0)
    actual = jax.jit(lambda p: fluid.get_state(jp.PT_INPUTS, p, 500.0, errors="mask"))(
        pressure
    )
    assert actual.valid
    # Outer JIT can fuse the active evaluator differently from its standalone
    # executable; allow a few rounding ULPs, not a physical-model discrepancy.
    for name in actual.available_properties & expected.available_properties:
        np.testing.assert_allclose(
            actual.raw_property(name),
            expected.raw_property(name),
            rtol=3e-15,
            atol=0.0,
            equal_nan=True,
        )


def test_indexed_eager_errors_and_per_row_masks_preserve_successes(mixture):
    pressure, temperature = [2e5, 2e5, np.nan], [500.0, -1.0, 500.0]
    with pytest.raises(jp.InvalidStateError) as captured:
        mixture.get_state(jp.PT_INPUTS, pressure, temperature)
    assert captured.value.index == (1,) and captured.value.fluid_name == mixture.name
    state = mixture.get_state(jp.PT_INPUTS, pressure, temperature, errors="mask")
    np.testing.assert_array_equal(state.status, [0, 1, 1])
    assert state.raw_property("temperature")[0] == 500.0
    for name in state.available_properties:
        assert np.isnan(state.raw_property(name)[1:]).all()


def test_clipping_returns_a_consistent_adjusted_state_with_nonfinite_rows_failed(
    mixture,
):
    state = mixture.get_state(
        jp.PT_INPUTS, [2e5, -1.0, 2e5], [-1.0, 500.0, np.nan], errors="clip"
    )
    np.testing.assert_array_equal(state.status, [0, 0, 1])
    np.testing.assert_array_equal(state.clipped, [True, True, False])
    assert np.isfinite(state.raw_property("density")[:2]).all()
    np.testing.assert_allclose(
        state.raw_property("u")[:2]
        + state.raw_property("p")[:2] / state.raw_property("rho")[:2],
        state.raw_property("h")[:2],
        rtol=2e-13,
    )
    assert np.isnan(state.raw_property("temperature")[2])


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("pair", [jp.HmassP_INPUTS, jp.PSmass_INPUTS])
@pytest.mark.parametrize("policy", ["raise", "mask", "clip"])
def test_pg_inverse_lower_endpoint_policy_is_independent_of_its_distant_ceiling(
    gases, dtype, pair, policy
):
    components = tuple(
        gas.with_constants(
            **{
                name: jnp.asarray(value, dtype=dtype)
                for name, value in vars(gas.constants).items()
            }
        )
        for gas in gases
    )
    fluid = binary(components)
    pressure = jnp.asarray(2e5, dtype=dtype)
    boundary = fluid.get_state(jp.PT_INPUTS, pressure, jnp.asarray(1.0, dtype=dtype))
    name = "enthalpy" if pair == jp.HmassP_INPUTS else "entropy"
    endpoint = boundary.raw_property(name)
    relative = max(1e-12, 64 * np.finfo(dtype).eps)
    # One joule exposes the enormous HP representability ceiling. For PS use
    # a resolvable local perturbation rather than the much larger upper-S scale.
    decrement = (
        1.0
        if pair == jp.HmassP_INPUTS
        else 8 * relative * max(abs(float(endpoint)), 1.0)
    )
    targets = jnp.asarray([float(endpoint) - decrement, float(endpoint)], dtype=dtype)
    arguments = (targets, pressure) if pair == jp.HmassP_INPUTS else (pressure, targets)
    if policy == "raise":
        with pytest.raises(jp.OutOfDomainError) as captured:
            fluid.get_state(pair, *arguments, errors=policy)
        assert captured.value.index == (0,)
        arguments = (
            (endpoint, pressure) if pair == jp.HmassP_INPUTS else (pressure, endpoint)
        )
        state = fluid.get_state(pair, *arguments, errors=policy)
        assert state.valid and not state.clipped
        # At float32 the enthalpy/entropy reference subtraction limits the
        # recovered temperature's absolute precision at the one-kelvin floor.
        np.testing.assert_allclose(
            state.T, 1.0, rtol=0.0, atol=5e-4 if dtype == np.float32 else 2e-10
        )
    else:
        state = fluid.get_state(pair, *arguments, errors=policy)
        np.testing.assert_array_equal(
            state.status, [2, 0] if policy == "mask" else [0, 0]
        )
        np.testing.assert_array_equal(
            state.clipped, [False, False] if policy == "mask" else [True, False]
        )
        if policy == "mask":
            for field in state.available_properties:
                assert np.isnan(state.raw_property(field)[0])
        else:
            assert state.raw_property("temperature")[0] == 1.0


def test_native_failed_rows_have_zero_input_and_model_gradient_contributions(mixture):
    reference = mixture.get_state(jp.PT_INPUTS, 2e5, 500.0)
    enthalpy = jnp.array([reference.h, np.nan, 1e308])

    def temperatures(values):
        state = mixture.get_state(jp.HmassP_INPUTS, values, 2e5, errors="mask")
        return jnp.nan_to_num(state.raw_property("temperature")).sum()

    derivative = jax.jit(jax.grad(temperatures))(enthalpy)
    np.testing.assert_allclose(derivative[0], 1 / float(reference.cp), rtol=2e-12)
    np.testing.assert_array_equal(derivative[1:], [0.0, 0.0])

    def model_response(gas_constant, values):
        view = eqx.tree_at(
            lambda model: model.components[0].constants.R, mixture, gas_constant
        )
        state = view.get_state(jp.HmassP_INPUTS, values, 2e5, errors="mask")
        return jnp.nan_to_num(state.raw_property("temperature")).sum()

    mixed_gradient = jax.jit(jax.grad(model_response))(
        mixture.components[0].constants.R, enthalpy
    )
    valid_gradient = jax.jit(jax.grad(model_response))(
        mixture.components[0].constants.R, jnp.asarray(reference.h)
    )
    assert np.isfinite(mixed_gradient)
    np.testing.assert_allclose(mixed_gradient, valid_gradient, rtol=2e-12, atol=1e-12)


def test_native_component_coefficients_remain_dynamic_and_differentiable(mixture):
    component = mixture.components[0]
    pressure, temperature = 2e5, 500.0

    def enthalpy(gas_constant):
        view = eqx.tree_at(
            lambda model: model.components[0].constants.R, mixture, gas_constant
        )
        return view.get_state(jp.PT_INPUTS, pressure, temperature, errors="mask").h

    actual = jax.jit(jax.grad(enthalpy))(component.constants.R)
    expected = (
        mixture.mass_fractions[0]
        * component.constants.gamma
        / (component.constants.gamma - 1)
        * (temperature - component.constants.T_ref)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12)
    lowered = jax.jit(
        lambda model: model.get_state(
            jp.PT_INPUTS, pressure, temperature, errors="mask"
        )
    )
    assert "callback" not in lowered.lower(mixture).compile().as_text().lower()


def test_native_serialization_preserves_complete_states_and_configuration(mixture):
    restored = pickle.loads(pickle.dumps(mixture))
    assert restored.configuration == mixture.configuration
    assert restored.reference_id == mixture.reference_id
    expected = mixture.get_state(jp.PT_INPUTS, 2e5, 500.0)
    actual = restored.get_state(jp.PT_INPUTS, 2e5, 500.0)
    assert actual.available_properties == expected.available_properties
    assert_same_fields(actual, expected, tolerance=0.0)


@pytest.mark.parametrize(
    "settings,expected_request",
    [
        ({"mass_fractions": (0.2, 0.7)}, "mass_fractions"),
        ({"mass_fractions": (-0.1, 1.1)}, "mass_fractions"),
        ({"mass_fractions": (np.nan, 1.0)}, "mass_fractions"),
        ({"mass_fractions": (True, 0.0)}, "mass_fractions"),
        ({"mass_fractions": (1.0,)}, "mass_fractions"),
        # Validate the actual fractions after coercion, not a float32 sum that
        # rounds to one while hiding a roughly 2e-8 mass discrepancy.
        ({"mass_fractions": (np.float32(0.1), np.float32(0.9))}, "mass_fractions"),
        ({"inverse_max_steps": -1}, "inverse_max_steps"),
        ({"inverse_max_steps": True}, "inverse_max_steps"),
        ({"inverse_tolerance": 0.0}, "inverse_tolerance"),
        ({"inverse_tolerance": np.nan}, "inverse_tolerance"),
    ],
)
def test_constructor_rejects_invalid_composition_and_inverse_settings(
    gases, settings, expected_request
):
    kwargs = (
        dict(name="invalid", components=gases, mass_fractions=(0.25, 0.75)) | settings
    )
    with pytest.raises(jp.InvalidConfigurationError) as captured:
        jp.FluidBinaryMixture(**kwargs)
    assert captured.value.request == expected_request


def test_factory_rejects_backend_override_and_nested_binary_components(gases, mixture):
    with pytest.raises(jp.InvalidConfigurationError) as captured:
        jp.Fluid(
            "bad",
            method="binary_mixture",
            backend="HEOS",
            model_options={"components": gases, "mass_fractions": (0.5, 0.5)},
        )
    assert captured.value.request == "backend"
    with pytest.raises(jp.InvalidConfigurationError) as captured:
        binary((mixture, gases[0]))
    assert captured.value.request == "components"


@pytest.mark.parametrize("option", ["generalize_quality", "supersaturation"])
def test_inapplicable_diagnostics_are_informatively_unsupported(mixture, option):
    with pytest.raises(jp.UnsupportedOptionError):
        mixture.get_state(jp.PT_INPUTS, 2e5, 500.0, **{option: True})


def test_quality_saturation_and_transformed_raising_are_explicitly_unsupported(mixture):
    with pytest.raises(jp.UnsupportedInputPairError):
        mixture.get_state(jp.PQ_INPUTS, 2e5, 0.5)
    with pytest.raises(jp.UnsupportedOperationError):
        mixture.get_state_saturation(jp.PQ_INPUTS, 2e5, 0.5)
    with pytest.raises(jp.UnsupportedPropertyError):
        mixture.get_state(jp.PT_INPUTS, 2e5, 500.0).raw_property("quality_mass")
    with pytest.raises(jp.UnsupportedExecutionModeError):
        jax.jit(lambda p: mixture.get_state(jp.PT_INPUTS, p, 500.0).T)(2e5)


@pytest.mark.parametrize("pair", [jp.HmassP_INPUTS, jp.PSmass_INPUTS])
def test_real_bicubic_component_inverse_and_derivative_roundtrip(table, gases, pair):
    fluid = binary((table, gases[0]), (0.3, 0.7))
    temperature = float(table.get_state(jp.HmassP_INPUTS, 710e3, 8e6).T)
    reference = fluid.get_state(jp.PT_INPUTS, 8e6, temperature)
    first, second = (
        (reference.h, 8e6) if pair == jp.HmassP_INPUTS else (8e6, reference.s)
    )
    actual = fluid.get_state(pair, first, second)
    assert_same_fields(actual, reference, CORE, tolerance=2e-9)
    name = "enthalpy" if pair == jp.HmassP_INPUTS else "entropy"
    forward = jax.jit(
        jax.grad(
            lambda T: fluid.get_state(jp.PT_INPUTS, 8e6, T, errors="mask").raw_property(
                name
            )
        )
    )(temperature)
    inverse = jax.jit(
        jax.grad(
            lambda value: (
                fluid.get_state(pair, value, 8e6, errors="mask").T
                if pair == jp.HmassP_INPUTS
                else fluid.get_state(pair, 8e6, value, errors="mask").T
            )
        )
    )(reference.raw_property(name))
    np.testing.assert_allclose(forward * inverse, 1.0, rtol=2e-8, atol=1e-10)


def test_bicubic_inverse_iterations_do_not_construct_public_component_states(
    table, gases, monkeypatch
):
    temperature = float(table.get_state(jp.HmassP_INPUTS, 710e3, 8e6).T)
    reference = binary((table, gases[0]), (0.3, 0.7)).get_state(
        jp.PT_INPUTS, 8e6, temperature
    )
    calls = {"bicubic": 0, "perfect_gas": 0}
    for cls, label in (
        (jp.FluidBicubic, "bicubic"),
        (jp.FluidPerfectGas, "perfect_gas"),
    ):
        original = cls.get_state

        def observed(self, *args, _original=original, _label=label, **kwargs):
            calls[_label] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "get_state", observed)
    fluid = jp.FluidBinaryMixture(
        "raw-iteration-observer", (table, gases[0]), (0.3, 0.7)
    )
    state = fluid.get_state(jp.HmassP_INPUTS, reference.h, 8e6)
    assert state.valid
    assert calls == {"bicubic": 1, "perfect_gas": 1}


def test_public_inverse_budget_reaches_the_actual_native_solver(table, gases):
    fluid = binary((table, gases[0]), (0.3, 0.7), inverse_max_steps=0)
    temperature = float(table.get_state(jp.HmassP_INPUTS, 710e3, 8e6).T)
    reference = fluid.get_state(jp.PT_INPUTS, 8e6, temperature)
    with pytest.raises(jp.ConvergenceError):
        fluid.get_state(jp.HmassP_INPUTS, reference.h, 8e6)
    actual = fluid.get_state(jp.HmassP_INPUTS, reference.h, 8e6, errors="mask")
    assert actual.status == 3 and np.isnan(actual.raw_property("temperature"))


def test_active_source_bicubic_hybrid_has_bounded_eager_inverse_roundtrips(table):
    source = jp.Fluid("Nitrogen", method="coolprop")
    fluid = binary((table, source), (0.3, 0.7))
    temperature = float(table.get_state(jp.HmassP_INPUTS, 710e3, 8e6).T)
    reference = fluid.get_state(jp.PT_INPUTS, 8e6, temperature)
    assert fluid.capabilities.jax_modes == {"eager"}
    for pair, first, second in (
        (jp.HmassP_INPUTS, reference.h, 8e6),
        (jp.PSmass_INPUTS, 8e6, reference.s),
    ):
        actual = fluid.get_state(pair, first, second)
        assert_same_fields(actual, reference, CORE, tolerance=2e-9)
    with pytest.raises(jp.UnsupportedExecutionModeError):
        jax.jit(
            lambda p: fluid.get_state(jp.PT_INPUTS, p, temperature, errors="mask").T
        )(8e6)


@pytest.fixture(
    scope="module", params=os.environ.get("JAXPROP_TEST_BACKENDS", "HEOS").split(",")
)
def backend(request):
    return request.param


@pytest.fixture(scope="module", params=["coolprop", "coolprop_jax"])
def source_model(request, backend):
    components = tuple(
        jp.Fluid(name, method=request.param, backend=backend)
        for name in ("Water", "Nitrogen")
    )
    return binary(components, (0.25, 0.75))


def test_source_components_preserve_independent_weighted_core_values(source_model):
    pressure, temperature = 8e6, 350.0
    components = [
        component.get_state(jp.PT_INPUTS, pressure, temperature)
        for component in source_model.components
    ]
    state = source_model.get_state(jp.PT_INPUTS, pressure, temperature)
    assert state.valid
    for name in ("enthalpy", "internal_energy", "entropy", "isobaric_heat_capacity"):
        expected = sum(
            weight * float(component.raw_property(name))
            for weight, component in zip(source_model.mass_fractions, components)
        )
        np.testing.assert_allclose(state.raw_property(name), expected, rtol=2e-12)
    expected_density = 1 / sum(
        weight / float(component.rho)
        for weight, component in zip(source_model.mass_fractions, components)
    )
    np.testing.assert_allclose(state.rho, expected_density, rtol=2e-12)
    for pair, first, second in (
        (jp.HmassP_INPUTS, state.h, pressure),
        (jp.PSmass_INPUTS, pressure, state.s),
    ):
        recovered = source_model.get_state(pair, first, second)
        assert_same_fields(recovered, state, CORE, tolerance=2e-9)


def test_source_mode_capabilities_and_clipping_rejection_are_honest(source_model):
    assert source_model.capabilities.error_modes == {"raise", "mask"}
    native_callback = source_model.components[0].provider == "coolprop_jax"
    assert ("jit" in source_model.capabilities.jax_modes) == native_callback
    with pytest.raises(jp.UnsupportedExecutionModeError):
        source_model.get_state(jp.PT_INPUTS, 8e6, 350.0, errors="clip")
    if not native_callback:
        with pytest.raises(jp.UnsupportedExecutionModeError):
            jax.jit(
                lambda p: source_model.get_state(
                    jp.PT_INPUTS, p, 350.0, errors="mask"
                ).T
            )(8e6)
    else:
        state = jax.jit(
            lambda T: source_model.get_state(jp.PT_INPUTS, 8e6, T, errors="mask")
        )(350.0)
        assert state.valid and state.T == 350.0
        derivative = jax.jit(
            jax.grad(
                lambda T: source_model.get_state(jp.PT_INPUTS, 8e6, T, errors="mask").h
            )
        )(350.0)
        np.testing.assert_allclose(derivative, float(state.cp), rtol=2e-5)
        with pytest.raises(jp.UnsupportedExecutionModeError) as captured:
            jax.grad(
                jax.grad(
                    lambda T: source_model.get_state(
                        jp.PT_INPUTS, 8e6, T, errors="mask"
                    ).h
                )
            )(350.0)
        assert captured.value.request == "derivative_order"
        assert captured.value.fluid_name == source_model.name


@pytest.fixture(scope="module")
def nitrogen_table(table, tmp_path_factory):
    """A second real table whose PT domain overlaps the CO2 test region."""
    temperature = float(table.get_state(jp.HmassP_INPUTS, 700e3, 8e6).T)
    source = jp.Fluid("Nitrogen", method="coolprop")
    enthalpy = [
        float(source.get_state(jp.PT_INPUTS, pressure, T).h)
        for pressure in (5e6, 12e6)
        for T in (temperature - 40, temperature + 40)
    ]
    return jp.Fluid(
        "Nitrogen",
        method="bicubic",
        table_options=dict(
            h_min=min(enthalpy),
            h_max=max(enthalpy),
            p_min=5e6,
            p_max=12e6,
            N_h=5,
            N_p=5,
            N_p_sat=4,
            coarse_step=1,
            table_dir=str(tmp_path_factory.mktemp("phase9_second_binary_table")),
        ),
    )


@pytest.fixture(scope="module", params=("bicubic_pg", "bicubic_bicubic"))
def native_broadcast_model(request, table, gases):
    second = (
        gases[0]
        if request.param == "bicubic_pg"
        else request.getfixturevalue("nitrogen_table")
    )
    fluid = binary((table, second), (0.3, 0.7))
    temperature = float(table.get_state(jp.HmassP_INPUTS, 700e3, 8e6).T)
    return fluid, temperature


def _broadcast_arguments_from_scalar_pt(fluid, pair, layout, temperature):
    """Build inverse targets from valid scalar PT states, without broadcasting.

    The oracle deliberately uses the already validated scalar entry point. It
    must not accidentally execute the same vector broadcasting path under test.
    """
    if layout == "pressure_scalar":
        pressure = np.asarray(8e6)
        temperatures = temperature + np.array([-3.0, 0.0, 3.0])
    elif layout == "property_scalar":
        pressure = np.array([7e6, 8e6, 9e6])
        temperatures = np.asarray(temperature)
    else:
        pressure = np.array([[7.5e6], [8.5e6]])
        temperatures = temperature + np.array([[-3.0, 0.0, 3.0]])
    if pair == jp.PT_INPUTS:
        target = temperatures
    else:
        name = "enthalpy" if pair == jp.HmassP_INPUTS else "entropy"
        target = np.asarray(
            [
                float(fluid.get_state(jp.PT_INPUTS, 8e6, float(T)).raw_property(name))
                for T in temperatures.ravel()
            ]
        ).reshape(temperatures.shape)
    return (target, pressure) if pair == jp.HmassP_INPUTS else (pressure, target)


def _scalar_core_oracle(fluid, pair, first, second):
    left, right = np.broadcast_arrays(first, second)
    states = [
        fluid.get_state(pair, float(a), float(b))
        for a, b in zip(left.ravel(), right.ravel())
    ]
    for state in states:
        assert state.status == 0
    return {
        name: np.asarray([float(state.raw_property(name)) for state in states]).reshape(
            left.shape
        )
        for name in CORE
    }


def _assert_complete_broadcast_result(fluid, state, expected, tolerance):
    shape = expected["temperature"].shape
    assert state.shape == shape
    assert state.fluid_name == fluid.name and state.reference_id == fluid.reference_id
    assert state.available_properties == fluid.capabilities.properties
    assert len(state.available_properties) == 27
    np.testing.assert_array_equal(state.status, np.zeros(shape, dtype=np.int32))
    for name in state.available_properties:
        assert state.raw_property(name).shape == shape, name
    for name, values in expected.items():
        np.testing.assert_allclose(
            state.raw_property(name), values, rtol=tolerance, atol=0.0
        )


@pytest.mark.parametrize("pair", (jp.PT_INPUTS, jp.HmassP_INPUTS, jp.PSmass_INPUTS))
@pytest.mark.parametrize("layout", ("pressure_scalar", "property_scalar", "matrix"))
def test_real_native_provider_branches_broadcast_before_mapping(
    native_broadcast_model, pair, layout
):
    fluid, temperature = native_broadcast_model
    first, second = _broadcast_arguments_from_scalar_pt(
        fluid, pair, layout, temperature
    )
    expected = _scalar_core_oracle(fluid, pair, first, second)
    eager = fluid.get_state(pair, first, second, errors="mask")
    compiled = jax.jit(lambda a, b: fluid.get_state(pair, a, b, errors="mask"))(
        first, second
    )
    for state in (eager, compiled):
        _assert_complete_broadcast_result(fluid, state, expected, tolerance=2e-8)


@pytest.mark.parametrize("pair", (jp.PT_INPUTS, jp.HmassP_INPUTS, jp.PSmass_INPUTS))
@pytest.mark.parametrize("layout", ("pressure_scalar", "property_scalar", "matrix"))
def test_source_and_callback_provider_branches_preserve_complete_broadcast_states(
    source_model, pair, layout
):
    first, second = _broadcast_arguments_from_scalar_pt(
        source_model, pair, layout, 350.0
    )
    expected = _scalar_core_oracle(source_model, pair, first, second)
    state = source_model.get_state(pair, first, second, errors="mask")
    _assert_complete_broadcast_result(source_model, state, expected, tolerance=2e-9)
    if "jit" in source_model.capabilities.jax_modes:
        compiled = jax.jit(
            lambda a, b: source_model.get_state(pair, a, b, errors="mask")
        )(first, second)
        _assert_complete_broadcast_result(
            source_model, compiled, expected, tolerance=2e-9
        )


@pytest.mark.parametrize("pair", (jp.PT_INPUTS, jp.HmassP_INPUTS, jp.PSmass_INPUTS))
@pytest.mark.parametrize("layout", ("pressure_scalar", "property_scalar"))
def test_full_state_vmap_uses_only_advertised_source_execution_modes(
    source_model, pair, layout
):
    first, second = _broadcast_arguments_from_scalar_pt(
        source_model, pair, layout, 350.0
    )
    axes = (0 if np.ndim(first) else None, 0 if np.ndim(second) else None)
    mapped = jax.jit(
        jax.vmap(
            lambda a, b: source_model.get_state(pair, a, b, errors="mask"),
            in_axes=axes,
        )
    )
    if "vmap" not in source_model.capabilities.jax_modes:
        with pytest.raises(jp.UnsupportedExecutionModeError):
            mapped(first, second)
    else:
        expected = _scalar_core_oracle(source_model, pair, first, second)
        state = mapped(first, second)
        _assert_complete_broadcast_result(source_model, state, expected, tolerance=2e-9)
