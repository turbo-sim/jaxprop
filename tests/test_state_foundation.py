"""Independent acceptance checks for the shared state foundation."""

import pickle

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp

# This vocabulary is independent of the implementation's alias dictionaries.
EXPECTED_ALIASES = {
    "pressure": ("p", "P"),
    "temperature": ("T",),
    "density": ("rho", "d", "rhomass", "dmass"),
    "enthalpy": ("h", "hmass", "H"),
    "entropy": ("s", "smass"),
    "internal_energy": ("u", "umass", "e", "energy"),
    "compressibility_factor": ("Z",),
    "isobaric_heat_capacity": ("cp", "cpmass"),
    "isochoric_heat_capacity": ("cv", "cvmass"),
    "heat_capacity_ratio": ("gamma", "kappa"),
    "speed_of_sound": ("a", "speed_sound"),
    "speed_of_sound_p": (),
    "speed_of_sound_pT": (),
    "isothermal_compressibility": ("kappa_T",),
    "isentropic_compressibility": ("kappa_s",),
    "isothermal_bulk_modulus": ("K_T",),
    "isentropic_bulk_modulus": ("K_s",),
    "isobaric_expansion_coefficient": ("alpha_p",),
    "isothermal_joule_thomson": ("mu_T",),
    "joule_thomson": ("mu_JT",),
    "gruneisen": ("G",),
    "viscosity": ("mu",),
    "conductivity": ("k",),
    "is_two_phase": (),
    "quality_mass": ("vapor_quality", "Q", "q"),
    "quality_volume": ("void_fraction", "alpha"),
    "surface_tension": ("sigma",),
    "subcooling": (),
    "superheating": (),
    "pressure_saturation": (),
    "temperature_saturation": (),
    "supersaturation_degree": (),
    "supersaturation_ratio": (),
    "mixture_ratio": (),
    "mass_frac_1": ("mass_fraction_1",),
    "mass_frac_2": ("mass_fraction_2",),
    "vol_frac_1": ("volume_fraction_1",),
    "vol_frac_2": ("volume_fraction_2",),
    "drho_dp": (),
    "drho_dh": (),
}


def state(**kwargs):
    return jxp.FluidState(
        **(
            dict(
                fluid_name="CO2",
                identifier="working_fluid",
                provider="coolprop",
                model="real_fluid",
                reference_id="same-source-reference",
            )
            | kwargs
        )
    )


def assert_same_state(actual, expected):
    for name in (
        "fluid_name",
        "identifier",
        "provider",
        "providers",
        "model",
        "reference_id",
        "error_policy",
    ):
        assert getattr(actual, name) == getattr(expected, name)
    assert actual.available_properties == expected.available_properties
    np.testing.assert_array_equal(actual.status, expected.status)
    np.testing.assert_array_equal(actual.valid, expected.valid)
    for name in expected.available_properties:
        np.testing.assert_array_equal(
            actual.raw_property(name), expected.raw_property(name)
        )
        np.testing.assert_array_equal(
            actual.property_valid(name), expected.property_valid(name)
        )


def test_canonical_numeric_schema_excludes_status_and_metadata():
    assert set(jxp.PROPERTIES_CANONICAL) == set(EXPECTED_ALIASES)
    assert len(jxp.PROPERTIES_CANONICAL) == len(EXPECTED_ALIASES)
    assert not {
        "status",
        "valid",
        "provider",
        "reference_id",
        "available_properties",
    } & set(jxp.PROPERTIES_CANONICAL)


@pytest.mark.parametrize("canonical,aliases", EXPECTED_ALIASES.items())
def test_every_canonical_property_and_declared_alias_has_controlled_access(
    canonical, aliases
):
    values = jnp.array([[1.0], [2.0]])
    current = state(**{canonical: values})
    for name in (canonical, *aliases):
        assert current.supports_property(name)
        np.testing.assert_array_equal(current[name], values)
        np.testing.assert_array_equal(getattr(current, name), values)
        np.testing.assert_array_equal(current.raw_property(name), values)
        np.testing.assert_array_equal(current.property_valid(name), [[True], [True]])


@pytest.mark.parametrize("name", ["quality_mass", "Q", "q"])
def test_disabled_known_properties_raise_consistent_errors_for_all_access_paths(name):
    current = state(model="perfect_gas", provider="native", temperature=500.0)
    assert not current.supports_property(name)
    for read in (
        lambda: current[name],
        lambda: getattr(current, name),
        lambda: current.raw_property(name),
        lambda: current.property_valid(name),
    ):
        with pytest.raises(jxp.UnsupportedPropertyError) as captured:
            read()
        assert captured.value.fluid_name == "CO2"
        assert captured.value.provider == "native"
        assert captured.value.model == "perfect_gas"
        assert captured.value.request == "quality_mass"
        assert captured.value.reason
        assert "temperature" in captured.value.supported


def test_unknown_property_names_keep_python_attribute_and_mapping_errors():
    current = state(pressure=1e5)
    with pytest.raises(AttributeError):
        current.no_such_property
    with pytest.raises(KeyError):
        current["no_such_property"]


def test_constructor_broadcasts_enabled_values_status_and_explicit_validity_masks():
    current = state(
        pressure=jnp.array([[1e5], [2e5]]),
        temperature=jnp.array([[400.0, 500.0, 600.0]]),
        enthalpy=7e5,
        status=jnp.array([[0], [1]]),
        property_validity={"temperature": jnp.array([[True, False, True]])},
        error_policy="mask",
    )
    assert current.available_properties == frozenset(
        {"pressure", "temperature", "enthalpy"}
    )
    assert isinstance(current.available_properties, frozenset)
    assert current.providers == frozenset({"coolprop"})
    for value in current.to_dict().values():
        assert value.shape == (2, 3)
    assert current.status.shape == current.valid.shape == (2, 3)
    np.testing.assert_array_equal(
        current.property_valid("temperature"),
        [[True, False, True], [False, False, False]],
    )
    np.testing.assert_array_equal(
        current.valid, [[True, True, True], [False, False, False]]
    )
    assert np.all(np.isnan(np.asarray(current.raw_property("pressure"))[1]))
    assert np.isnan(np.asarray(current.raw_property("temperature"))[0, 1])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_supported_but_nonfinite_property_reports_local_unavailability(bad_value):
    current = state(pressure=[1e5, bad_value])
    assert current.supports_property("p")
    np.testing.assert_array_equal(current.property_valid("p"), [True, False])
    assert np.isnan(current.raw_property("p")[1])
    with pytest.raises(jxp.PropertyUnavailableError) as captured:
        current.p
    assert captured.value.request == "pressure"
    assert (1,) in captured.value.indices
    np.testing.assert_array_equal(current.valid, [True, True])


def test_explicit_unavailability_mask_controls_eager_access_and_raw_dictionary():
    current = state(pressure=[1e5, 2e5], property_validity={"pressure": [True, False]})
    np.testing.assert_array_equal(current.property_valid("p"), [True, False])
    assert np.isnan(current.raw_property("p")[1])
    assert np.isnan(current.to_dict()["pressure"][1])
    with pytest.raises(jxp.PropertyUnavailableError):
        current.pressure


def test_mask_policy_returns_enabled_raw_arrays_and_keeps_unsupported_access_explicit():
    current = state(pressure=[1e5, np.nan], error_policy="mask")
    np.testing.assert_array_equal(current.p, current.raw_property("pressure"))
    with pytest.raises(jxp.UnsupportedPropertyError):
        current.Q


def test_property_status_can_determine_batch_shape_without_any_numeric_values():
    current = state(status=np.array([[0], [4]]), error_policy="mask")
    assert current.available_properties == frozenset()
    assert current.to_dict() == {}
    np.testing.assert_array_equal(current.valid, [[True], [False]])
    assert current.status.shape == (2, 1)


def test_empty_enabled_arrays_and_masks_keep_their_exact_broadcast_shape():
    current = state(pressure=np.empty((0, 1)), temperature=np.ones((1, 3)))
    assert current.p.shape == current.T.shape == current.status.shape == (0, 3)
    assert current.property_valid("p").shape == current.valid.shape == (0, 3)


@pytest.mark.parametrize("operation", ["stack", "concatenate"])
@pytest.mark.parametrize(
    "metadata", [{"model": "perfect_gas"}, {"reference_id": "other-reference"}]
)
def test_collections_reject_incompatible_physical_meaning_even_with_matching_names(
    operation, metadata
):
    first = state(pressure=[1e5])
    other = state(pressure=[2e5], **metadata)
    with pytest.raises(ValueError, match="model|reference"):
        jxp.FluidState.stack([first, other]) if operation == "stack" else first + other


def test_collections_intersect_enabled_properties_and_preserve_provider_provenance():
    first = state(pressure=[1e5, 2e5], temperature=500.0)
    second = state(provider="bicubic", pressure=[3e5], enthalpy=7e5)
    combined = first + second
    assert combined.available_properties == frozenset({"pressure"})
    assert combined.providers == frozenset({"coolprop", "bicubic"})
    assert combined.provider == "mixed"
    assert combined.model == "real_fluid"
    assert combined.reference_id == "same-source-reference"
    np.testing.assert_array_equal(combined.p, [1e5, 2e5, 3e5])
    assert set(combined.to_dict()) == {"pressure"}
    with pytest.raises(jxp.UnsupportedPropertyError):
        combined.T
    np.testing.assert_array_equal(first.p, [1e5, 2e5])
    np.testing.assert_array_equal(second.p, [3e5])


def test_stack_preserves_shape_status_validity_masks_and_provenance():
    first = state(
        pressure=[[1e5], [2e5]],
        property_validity={"pressure": [[True], [False]]},
        error_policy="mask",
    )
    second = state(
        provider="bicubic",
        pressure=[[3e5], [4e5]],
        status=[[0], [4]],
        error_policy="mask",
    )
    stacked = jxp.FluidState.stack([first, second])
    assert stacked.raw_property("p").shape == (2, 2, 1)
    assert stacked.status.shape == stacked.valid.shape == (2, 2, 1)
    np.testing.assert_array_equal(
        stacked.property_valid("p"), [[[True], [False]], [[True], [False]]]
    )
    np.testing.assert_array_equal(stacked.status, [[[0], [0]], [[0], [4]]])
    assert stacked.providers == frozenset({"coolprop", "bicubic"})
    reversed_state = stacked.flipped()
    np.testing.assert_array_equal(
        reversed_state.status, np.asarray(stacked.status)[::-1]
    )
    np.testing.assert_array_equal(reversed_state.at_index(0).status, second.status)
    assert reversed_state.at_index(0).providers == stacked.providers


def test_stack_rejects_different_state_shapes_instead_of_broadcasting_new_state_axis():
    with pytest.raises(ValueError, match="shape|dimension"):
        jxp.FluidState.stack([state(pressure=[1e5]), state(pressure=[[2e5]])])


def test_concatenation_rejects_different_trailing_shapes():
    with pytest.raises(ValueError, match="shape|dimension"):
        state(pressure=np.ones((2, 1))) + state(pressure=np.ones((1, 2)))


@pytest.mark.parametrize("index", [2, -3])
def test_eager_indexing_rejects_out_of_bounds_instead_of_jax_clamping(index):
    current = state(pressure=[1e5, 2e5])
    with pytest.raises(IndexError):
        current.at_index(index)


@pytest.mark.parametrize("index", [0, -1])
def test_scalar_indexing_accepts_only_the_single_state_positions(index):
    current = state(pressure=1e5)
    assert_same_state(current.at_index(index), current)


@pytest.mark.parametrize("index", [1, -2])
def test_scalar_out_of_bounds_indexing_is_explicit(index):
    with pytest.raises(IndexError):
        state(pressure=1e5).at_index(index)


def test_dynamic_indexing_is_rejected_before_silent_traced_clamping():
    current = state(pressure=jnp.array([1e5, 2e5]))
    with pytest.raises(jxp.UnsupportedExecutionModeError):
        jax.jit(lambda index: current.at_index(index).p)(jnp.asarray(0))


def test_state_roundtrips_preserve_all_metadata_status_availability_and_raw_values():
    first = state(
        pressure=[[1e5], [2e5]],
        temperature=500.0,
        property_validity={"pressure": [[True], [False]]},
        error_policy="mask",
    )
    second = state(
        provider="bicubic",
        pressure=[[3e5], [4e5]],
        temperature=600.0,
        status=[[0], [4]],
        error_policy="mask",
    )
    original = jxp.FluidState.stack([first, second])
    assert_same_state(pickle.loads(pickle.dumps(original)), original)
    assert_same_state(jxp.FluidState.from_data(original.to_data()), original)
    canonical = original.to_dict()
    aliased = original.to_dict(include_aliases=True)
    assert set(canonical) == {"pressure", "temperature"}
    np.testing.assert_array_equal(aliased["p"], canonical["pressure"])
    assert not {"status", "provider", "model", "reference_id"} & canonical.keys()


def test_jit_and_filter_jit_return_states_with_dynamic_status_and_static_metadata():
    def evaluate(values):
        return state(
            pressure=values, status=jnp.where(values > 0, 0, 1), error_policy="mask"
        )

    for transform in (jax.jit, eqx.filter_jit):
        current = transform(evaluate)(jnp.array([1e5, -1.0]))
        assert current.fluid_name == "CO2"
        assert current.reference_id == "same-source-reference"
        assert current.providers == frozenset({"coolprop"})
        assert np.issubdtype(np.asarray(current.status).dtype, np.integer)
        assert np.asarray(current.valid).dtype == bool
        np.testing.assert_array_equal(current.status, [0, 1])
        np.testing.assert_array_equal(current.valid, [True, False])
        assert np.isnan(current.raw_property("p")[1])


def test_jax_projection_gradient_survives_collection_and_dictionary_operations():
    def objective(values):
        first = state(pressure=values.reshape((2, 1)), error_policy="mask")
        other = state(
            provider="bicubic", pressure=3 * values.reshape((2, 1)), error_policy="mask"
        )
        return (
            jxp.FluidState.stack([first, other])
            .flipped()
            .at_index(0)
            .to_dict()["pressure"]
            .sum()
        )

    values = jnp.array([1.0, 2.0])
    assert jax.jit(objective)(values) == 9.0
    np.testing.assert_allclose(jax.grad(objective)(values), [3.0, 3.0])


@pytest.mark.parametrize(
    "configuration",
    [
        {"pressure": [1.0, 2.0], "temperature": [1.0, 2.0, 3.0]},
        {"pressure": 1e5, "error_policy": "ignore"},
        {"pressure": 1e5, "available_properties": {"not_a_property"}},
        {"pressure": 1e5, "property_validity": {"temperature": False}},
    ],
)
def test_invalid_state_schema_and_shapes_fail_explicitly(configuration):
    with pytest.raises(ValueError):
        state(**configuration)


def test_phase_flag_accepts_boolean_arrays_while_thermodynamic_values_do_not():
    current = state(pressure=[1e5, 2e5], is_two_phase=False)
    np.testing.assert_array_equal(current.is_two_phase, [False, False])
    np.testing.assert_array_equal(current.property_valid("is_two_phase"), [True, True])
    with pytest.raises(jxp.InvalidStateError):
        state(pressure=True)
    with pytest.raises(jxp.InvalidStateError):
        state(status=True)


def test_duplicate_canonical_and_alias_validity_masks_cannot_silently_overwrite():
    with pytest.raises(
        jxp.InvalidConfigurationError, match="more than once|duplicate|Duplicate"
    ):
        state(pressure=1e5, property_validity={"pressure": True, "p": False})


def test_invalid_batch_element_does_not_contaminate_successful_projection_gradient():
    def objective(values):
        current = state(pressure=values, status=jnp.array([0, 1]), error_policy="mask")
        return current.raw_property("pressure")[0]

    np.testing.assert_array_equal(
        jax.grad(objective)(jnp.array([1.0, jnp.nan])), [1.0, 0.0]
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"fluid_name": 123},
        {"identifier": []},
        {"model": []},
        {"provider": 123},
        {"providers": "coolprop"},
        {"providers": {"mixed"}},
        {"providers": {"bicubic"}},
        {"reference_id": {}},
    ],
)
def test_identity_and_provider_provenance_reject_mutable_or_inconsistent_configuration(
    metadata,
):
    with pytest.raises(jxp.InvalidConfigurationError):
        state(pressure=1e5, **metadata)


def test_provider_is_derived_consistently_from_explicit_provenance():
    current = state(provider=None, providers={"bicubic"}, pressure=1e5)
    assert current.provider == "bicubic"
    assert current.providers == frozenset({"bicubic"})


def test_numpy_state_is_isolated_from_mutating_constructor_arrays_and_export_dictionaries():
    pressure = np.array([1e5, 2e5])
    status = np.array([0, 4])
    validity = np.array([True, True])
    current = state(
        pressure=pressure,
        status=status,
        property_validity={"pressure": validity},
        error_policy="mask",
    )
    pressure[:] = 9e5
    status[:] = 0
    validity[:] = False
    np.testing.assert_array_equal(current.status, [0, 4])
    np.testing.assert_array_equal(current.valid, [True, False])
    np.testing.assert_array_equal(current.raw_property("pressure"), [1e5, np.nan])
    np.testing.assert_array_equal(current.property_valid("pressure"), [True, False])

    exported = current.to_data()
    exported["metadata"]["identifier"] = "other"
    exported["properties"].clear()
    exported["property_validity"].clear()
    assert current.identifier == "working_fluid"
    assert current.available_properties == frozenset({"pressure"})
    for value in (
        current.raw_property("pressure"),
        current.status,
        current.property_valid("pressure"),
    ):
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
