"""Independent values and derivatives in a narrow near-critical CO2 region."""

import os

import CoolProp.CoolProp as CP
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jp


@pytest.fixture(
    scope="module", params=os.environ.get("JAXPROP_TEST_BACKENDS", "HEOS").split(",")
)
def backend(request):
    return request.param


@pytest.fixture(scope="module")
def nearcritical(backend, tmp_path_factory):
    source = CP.AbstractState(backend, "CO2")
    pc, tc = source.p_critical(), source.T_critical()
    source.update(CP.PT_INPUTS, 1.04 * pc, 1.025 * tc)
    h = source.hmass()
    fluid = jp.FluidBicubic(
        "CO2",
        backend,
        h - 2000.0,
        h + 2000.0,
        1.02 * pc,
        1.06 * pc,
        12,
        12,
        N_p_sat=4,
        coarse_step=1,
        table_dir=str(tmp_path_factory.mktemp("phase8_nearcritical")),
    )
    return fluid, source, h, pc


def test_nearcritical_forward_values_remain_source_compatible(nearcritical):
    fluid, source, centre_h, pc = nearcritical
    hs = np.linspace(centre_h - 1000.0, centre_h + 1000.0, 3)
    ps = np.linspace(1.03 * pc, 1.05 * pc, 3)
    h, p = np.meshgrid(hs, ps, indexing="ij")
    state = fluid.get_state(jp.HmassP_INPUTS, h, p)
    assert np.all(state.valid)
    expected = {name: np.empty_like(h) for name in ("T", "rho", "s", "u")}
    for index in np.ndindex(h.shape):
        source.update(CP.HmassP_INPUTS, h[index], p[index])
        for name, value in zip(
            expected, (source.T(), source.rhomass(), source.smass(), source.umass())
        ):
            expected[name][index] = value
    for name in expected:
        # Budgets are declared for this resolution and include near-critical
        # interpolation error, rather than claiming exact EOS representation.
        tolerance = 2e-4 if name == "rho" else 1e-5
        np.testing.assert_allclose(
            state.raw_property(name), expected[name], rtol=tolerance
        )


def test_nearcritical_forward_derivatives_match_native_partial_derivatives(
    nearcritical,
):
    fluid, source, h, pc = nearcritical
    pressure = 1.04 * pc
    source.update(CP.HmassP_INPUTS, h, pressure)
    expected = np.array(
        [
            source.first_partial_deriv(CP.iT, CP.iHmass, CP.iP),
            source.first_partial_deriv(CP.iT, CP.iP, CP.iHmass),
        ]
    )

    def temperature(point):
        return fluid.get_state(jp.HmassP_INPUTS, point[0], point[1], errors="mask").T

    actual = jax.jit(jax.grad(temperature))(jnp.asarray([h, pressure]))
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-12)


def test_nearcritical_inverse_root_and_implicit_derivative_are_consistent(nearcritical):
    fluid, source, h, pc = nearcritical
    pressure = 1.04 * pc
    source.update(CP.HmassP_INPUTS, h, pressure)
    reference = fluid.get_state(jp.HmassP_INPUTS, h, pressure)
    recovered = fluid.get_state(jp.PT_INPUTS, pressure, reference.T)
    np.testing.assert_allclose(recovered.h, h, rtol=1e-10)
    derivative = jax.jit(
        jax.grad(
            lambda temperature: fluid.get_state(
                jp.PT_INPUTS, pressure, temperature, errors="mask"
            ).h
        )
    )(reference.T)
    np.testing.assert_allclose(derivative, source.cpmass(), rtol=2e-3)
