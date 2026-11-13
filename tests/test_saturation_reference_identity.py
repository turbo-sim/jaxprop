"""Saturation curves preserve caller identity and canonical source references."""

import numpy as np
import pytest

import jaxprop as jxp


@pytest.mark.parametrize("name", ["CO2", "water"])
@pytest.mark.parametrize("identifier", [None, "working_fluid"])
def test_saturation_curve_relabels_only_the_reference_endpoint(name, identifier):
    fluid = jxp.Fluid(name, identifier=identifier)
    reference = fluid.critical_point
    canonical_name = reference.fluid_name
    original_reference = reference.to_data()

    liquid, vapor = jxp.coolprop.compute_saturation_line(fluid, N=6)

    assert canonical_name != name
    for branch in (liquid, vapor):
        assert branch.fluid_name == name
        assert branch.identifier == identifier
        assert branch.reference_id == fluid.reference_id
        assert branch.shape == (7,)
        np.testing.assert_array_equal(branch.status, np.zeros(7, dtype=int))
        for key in (
            "pressure",
            "temperature",
            "density",
            "enthalpy",
            "entropy",
            "internal_energy",
        ):
            assert np.all(np.isfinite(branch.raw_property(key))), key
            np.testing.assert_array_equal(
                branch.raw_property(key)[0], reference.raw_property(key)
            )
        assert np.isnan(branch.raw_property("isobaric_heat_capacity")[0])
        assert not branch.property_valid("isobaric_heat_capacity")[0]

    np.testing.assert_array_equal(liquid.raw_property("quality_mass")[1:], 0.0)
    np.testing.assert_array_equal(vapor.raw_property("quality_mass")[1:], 1.0)
    assert fluid.critical_point.fluid_name == canonical_name
    assert fluid.critical_point.identifier is None
    for key, value in original_reference["properties"].items():
        np.testing.assert_array_equal(fluid.critical_point.raw_property(key), value)
