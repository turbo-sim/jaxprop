"""Signed derivative checks for a native tabulated water/nitrogen parcel model.

The original study used one experimental aggregate mixture table and called
unsupported density-based inverses. This version uses the public fixed-ratio
binary evaluator with two ordinary bicubic component objects, plus an independent
configured source/callback reference. The physical point is unchanged:
R=39.1, h=100 kJ/kg, p=200 kPa. Full resolution is 100-by-100 per component.

Both components cover the same 20 kPa to 3 MPa pressure range. Their enthalpy
rectangles are selected inside 278.15-318.15 K: Water is wholly liquid below
its 20 kPa saturation temperature, and Nitrogen is above its critical temperature.
This is an explicit interpolation-representation/domain change, rather than a
claim that old aggregate coefficients are preserved.

HP derivatives also supply du/dp at fixed rho by a Jacobian coordinate change;
PS derivatives supply acoustic speed and isentropic compressibility. No extra
input-pair capability or application-side fluid adapter is introduced.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options


def component_table(name, table_dir, points):
    """Prepare an ordinary, phase-safe table covering the declared point.

    Use the maximum lower-bound and minimum upper-bound source enthalpy across
    pressure, so every rectangular enthalpy column stays inside the temperature
    band. Generation independently checks source domain and phase-envelope cells.
    """
    import numpy as np
    import jaxprop as jp

    source = jp.Fluid(name)
    pressures = np.geomspace(2e4, 3e6, max(2 * points - 1, 65))
    lower = source.get_state(jp.PT_INPUTS, pressures, 278.15)
    upper = source.get_state(jp.PT_INPUTS, pressures, 318.15)
    h_min, h_max = float(np.max(lower.h)), float(np.min(upper.h))
    if name.lower() == "water":
        bubble = source.get_state(jp.PQ_INPUTS, 2e4, 0.0)
        assert 318.15 < bubble.T
    else:
        assert 278.15 > source.references["critical_point"].properties["temperature"]
    model = jp.Fluid(
        name,
        method="bicubic",
        table_options=dict(
            h_min=h_min,
            h_max=h_max,
            p_min=2e4,
            p_max=3e6,
            N_h=points,
            N_p=points,
            N_p_sat=4,
            coarse_step=1,
            table_dir=str(table_dir),
        ),
    )
    temperatures = np.asarray(model.table["temperature"]["value"])
    assert np.min(temperatures) >= 278.15 - 1e-6
    assert np.max(temperatures) <= 318.15 + 1e-6
    return model


def hp_response_jacobian(fluid, enthalpy, pressure):
    """Return signed partials of [rho,u] with respect to [h,p]."""
    import jax
    import jax.numpy as jnp
    import jaxprop as jp

    def response(hp):
        state = fluid.get_state(jp.HmassP_INPUTS, hp[0], hp[1], errors="mask")
        return jnp.stack(
            (state.raw_property("density"), state.raw_property("internal_energy"))
        )

    return jax.jacfwd(response)(jnp.asarray([enthalpy, pressure]))


def response_checks(fluid, enthalpy, pressure, *, rtol):
    """Compare independent response identities with actual numerical derivatives."""
    import numpy as np
    import jaxprop as jp
    from examples.two_component._model import check_state
    from examples.two_component.water_nitrogen_speed_sound import isentropic_sound_speed

    state = fluid.get_state(jp.HmassP_INPUTS, enthalpy, pressure)
    # Component u/rho fields are independently interpolated, so their caloric
    # identity is checked at the declared resolution budget rather than EOS
    # roundoff. Direct-source examples retain the strict default check.
    check_state(state, identity_rtol=rtol)
    jacobian = np.asarray(hp_response_jacobian(fluid, enthalpy, pressure))
    assert np.all(np.isfinite(jacobian))
    expected_density = np.array(
        [
            -float(state.rho * state.isobaric_expansion_coefficient / state.cp),
            float(
                state.rho * state.isothermal_compressibility
                - state.rho * state.isobaric_expansion_coefficient * state.joule_thomson
            ),
        ]
    )
    np.testing.assert_allclose(jacobian[0], expected_density, rtol=rtol, atol=1e-12)
    # d rho=0 => dh/dp=-rho_p/rho_h, with u=u(h,p).
    du_dp = jacobian[1, 1] - jacobian[1, 0] * jacobian[0, 1] / jacobian[0, 0]
    np.testing.assert_allclose(du_dp, 1 / float(state.rho * state.gruneisen), rtol=rtol)
    acoustic = float(isentropic_sound_speed(fluid, pressure, state.s))
    np.testing.assert_allclose(acoustic, state.speed_of_sound, rtol=rtol)
    kappa_s = 1 / (float(state.rho) * acoustic**2)
    np.testing.assert_allclose(kappa_s, state.isentropic_compressibility, rtol=2 * rtol)
    return dict(
        density_hp_partials=jacobian[0].tolist(),
        density_hp_identity_partials=expected_density.tolist(),
        du_dp_at_fixed_density=float(du_dp),
        acoustic_speed_from_PS=float(acoustic),
        relative_acoustic_error=float(abs(acoustic / float(state.speed_of_sound) - 1)),
    )


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    import numpy as np
    import jaxprop as jp
    from examples.two_component._model import mixture

    points = 100 if options.profile == "full" else 12
    components = tuple(
        component_table(name, options.table_dir, points)
        for name in ("Water", "Nitrogen")
    )
    native = mixture(components, 39.1)
    callbacks = tuple(
        jp.Fluid(name, method="coolprop_jax") for name in ("Water", "Nitrogen")
    )
    reference = mixture(callbacks, 39.1)
    h, p = 100e3, 2e5
    tolerance = 2e-5 if options.profile == "full" else 2e-3
    native_checks = response_checks(native, h, p, rtol=tolerance)
    source_checks = response_checks(reference, h, p, rtol=2e-5)
    actual, expected = native.get_state(jp.HmassP_INPUTS, h, p), reference.get_state(
        jp.HmassP_INPUTS, h, p
    )
    comparisons = 0
    for name in sorted(actual.available_properties & expected.available_properties):
        a, b = actual.raw_property(name), expected.raw_property(name)
        np.testing.assert_allclose(
            a, b, rtol=tolerance, atol=1e-6, equal_nan=True, err_msg=name
        )
        comparisons += 1
    print("Native HP density partials:", native_checks["density_hp_partials"])
    print("Source HP density partials:", source_checks["density_hp_partials"])
    print("Native PS acoustic speed:", native_checks["acoustic_speed_from_PS"])
    options.record(
        "settings",
        dict(
            h=h,
            p=p,
            mass_ratio=39.1,
            component_names=["Water", "Nitrogen"],
            component_grid=[points, points],
            pressure_bounds=[2e4, 3e6],
            temperature_band=[278.15, 318.15],
            relative_comparison_budget=tolerance,
        ),
    )
    options.record(
        "representation_change",
        "Two ordinary component tables and the public fixed-composition binary model replace the experimental aggregate binary table",
    )
    options.record("native_checks", native_checks)
    options.record("source_checks", source_checks)
    options.record("source_state_field_comparisons", comparisons)
    options.finish()


if __name__ == "__main__":
    main()
