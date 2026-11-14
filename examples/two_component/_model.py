"""Shared setup and checks for fixed-composition parcel examples.

The ratio R=m1/m2 configures a model once. It is neither a state input nor a
vapor quality. Parcels share pressure and temperature and exchange no mass.
"""

import numpy as np
import jaxprop as jp


def mass_fractions(ratio):
    """Convert the finite mass ratio to the public fraction convention."""
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError("The component mass ratio must be finite and nonnegative.")
    return ratio / (1 + ratio), 1 / (1 + ratio)


def mixture(components, ratio, *, name="water_nitrogen"):
    """Construct the public binary evaluator from prepared component objects."""
    return jp.Fluid(
        name,
        method="binary_mixture",
        model_options=dict(components=components, mass_fractions=mass_fractions(ratio)),
    )


def check_state(state, *, identity_rtol=1e-12):
    """Check finite core data, parcel conservation and response identities."""
    assert np.all(np.asarray(state.valid)), np.asarray(state.status)
    for name in (
        "pressure",
        "temperature",
        "density",
        "enthalpy",
        "entropy",
        "internal_energy",
    ):
        assert np.all(np.isfinite(np.asarray(state.raw_property(name)))), name
    for first, second in (("mass_frac_1", "mass_frac_2"), ("vol_frac_1", "vol_frac_2")):
        np.testing.assert_allclose(
            state.raw_property(first) + state.raw_property(second),
            1.0,
            rtol=0,
            atol=4 * np.finfo(float).eps,
        )
    np.testing.assert_allclose(
        state.u, state.h - state.p / state.rho, rtol=identity_rtol, atol=1e-7
    )
    np.testing.assert_allclose(
        state.speed_of_sound**2,
        1 / (state.rho * state.isentropic_compressibility),
        rtol=1e-12,
    )


def compare_states(actual, expected, *, rtol=2e-8, atol=1e-7):
    """Compare signed enabled fields and their optional availability."""
    check_state(actual)
    check_state(expected)
    assert actual.available_properties == expected.available_properties
    for name in sorted(expected.available_properties):
        np.testing.assert_array_equal(
            actual.property_valid(name), expected.property_valid(name)
        )
        np.testing.assert_allclose(
            actual.raw_property(name),
            expected.raw_property(name),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=name,
        )
