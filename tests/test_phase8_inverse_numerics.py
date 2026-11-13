"""Independent bounded inverse equations, failure checks and derivatives."""

from math import comb

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp
from jaxprop.bicubic.numerics import interpolate_properties, solve_inverse
from jaxprop.configuration import FluidCapabilities


class PolynomialFluid(eqx.Module):
    h_min: float = eqx.field(static=True, default=100e3)
    h_max: float = eqx.field(static=True, default=400e3)
    p_min: float = eqx.field(static=True, default=1e5)
    p_max: float = eqx.field(static=True, default=1e7)
    N_h: int = eqx.field(static=True, default=4)
    N_p: int = eqx.field(static=True, default=4)
    coarse_step: int = eqx.field(static=True, default=1)
    delta_h: float = eqx.field(static=True, default=100e3)
    delta_logP: float = eqx.field(static=True, default=float(np.log(100) / 3))
    capabilities: object = eqx.field(static=True)
    h_vals: jax.Array
    logP_vals: jax.Array
    _table: dict

    def __init__(self, *, flat_temperature=False, nonmonotone_temperature=False):
        self.capabilities = FluidCapabilities(
            input_pairs=frozenset(),
            properties=frozenset(
                {"temperature", "density", "entropy", "pressure", "enthalpy"}
            ),
        )
        self.h_vals = jnp.linspace(self.h_min, self.h_max, self.N_h)
        self.logP_vals = jnp.linspace(
            jnp.log(self.p_min), jnp.log(self.p_max), self.N_p
        )
        global_coefficients = {
            "temperature": {
                (0, 0): 280.0,
                (1, 0): 90,
                (0, 1): 12,
                (2, 0): 8,
                (1, 1): 2,
                (0, 2): 1.5,
            },
            "density": {(0, 0): 8, (1, 0): -2, (0, 1): 5, (1, 1): 0.3},
            "entropy": {(0, 0): 1100, (1, 0): 30, (0, 1): -15, (2, 0): 2, (0, 2): 0.5},
        }
        if flat_temperature:
            global_coefficients["temperature"] = {(0, 0): 300.0}
        if nonmonotone_temperature:
            # Every grid node is finite, but the derivative changes sign.
            global_coefficients["temperature"] = {
                (0, 0): 300,
                (1, 0): 100,
                (2, 0): -100,
            }
        u = np.linspace(0.0, 1.0, self.N_h)
        v = np.linspace(0.0, 1.0, self.N_p)
        self._table = {}
        for name, coefficients in global_coefficients.items():
            local = np.zeros((self.N_h - 1, self.N_p - 1, 16))
            nodal = np.zeros((self.N_h, self.N_p))
            for (a, b), value in coefficients.items():
                nodal += value * u[:, None] ** a * v[None, :] ** b
                for i in range(self.N_h - 1):
                    for j in range(self.N_p - 1):
                        for r in range(a + 1):
                            for s in range(b + 1):
                                local[i, j, 4 * s + r] += (
                                    value
                                    * comb(a, r)
                                    * u[i] ** (a - r)
                                    * (1 / (self.N_h - 1)) ** r
                                    * comb(b, s)
                                    * v[j] ** (b - s)
                                    * (1 / (self.N_p - 1)) ** s
                                )
            self._table[name] = {
                "value": jnp.asarray(nodal),
                "coeffs": jnp.asarray(local),
            }


@pytest.fixture(scope="module")
def fluid():
    return PolynomialFluid()


PAIRS = (
    (jxp.PT_INPUTS, ("pressure", "temperature")),
    (jxp.DmassP_INPUTS, ("density", "pressure")),
    (jxp.PSmass_INPUTS, ("pressure", "entropy")),
    (jxp.HmassSmass_INPUTS, ("enthalpy", "entropy")),
    (jxp.DmassHmass_INPUTS, ("density", "enthalpy")),
    (jxp.DmassT_INPUTS, ("density", "temperature")),
    (jxp.DmassSmass_INPUTS, ("density", "entropy")),
)


def physical_values(h, p):
    u = (h - 100e3) / 300e3
    v = jnp.log(p / 1e5) / jnp.log(100)
    return {
        "enthalpy": h,
        "pressure": p,
        "temperature": 280 + 90 * u + 12 * v + 8 * u**2 + 2 * u * v + 1.5 * v**2,
        "density": 8 - 2 * u + 5 * v + 0.3 * u * v,
        "entropy": 1100 + 30 * u - 15 * v + 2 * u**2 + 0.5 * v**2,
    }


@pytest.mark.parametrize("pair,names", PAIRS)
@pytest.mark.parametrize("h,p", [(100e3, 1e5), (247e3, 1.7e6), (400e3, 1e7)])
def test_all_inverse_equations_recover_independent_polynomial_coordinates(
    fluid, pair, names, h, p
):
    expected = physical_values(h, p)
    evaluate = jax.jit(
        lambda a, b: solve_inverse(fluid, pair, a, b, return_status=True)
    )
    actual_h, actual_p, converged, projected, status = evaluate(
        *(expected[name] for name in names)
    )
    assert converged
    assert not projected
    assert status == jxp.StatusCode.SUCCESS
    np.testing.assert_allclose(actual_h, h, rtol=2e-11, atol=1e-6)
    np.testing.assert_allclose(actual_p, p, rtol=2e-11, atol=1e-6)


@pytest.mark.parametrize("pair,names", PAIRS)
def test_inverse_input_derivatives_match_independent_forward_jacobian(
    fluid, pair, names
):
    h, p = 247e3, 1.7e6
    expected = physical_values(h, p)
    arguments = jnp.stack(tuple(expected[name] for name in names))
    actual = jax.jit(
        jax.jacfwd(lambda args: jnp.stack(solve_inverse(fluid, pair, *args)[:2]))
    )(arguments)
    forward = jax.jacfwd(
        lambda coords: jnp.stack(
            tuple(physical_values(*coords)[name] for name in names)
        )
    )(jnp.asarray([h, p]))
    np.testing.assert_allclose(actual, np.linalg.inv(forward), rtol=1e-9, atol=1e-7)


def test_solver_iteration_limit_is_wired_and_failed_ad_is_zero(fluid):
    target = physical_values(247e3, 1.7e6)["temperature"]
    evaluate = lambda value: solve_inverse(
        fluid, jxp.PT_INPUTS, 1.7e6, value, max_steps=0, return_status=True
    )
    result = jax.jit(evaluate)(target)
    assert not result[2]
    assert result[4] == jxp.StatusCode.NONCONVERGENCE
    loss = lambda value: jnp.where(evaluate(value)[2], evaluate(value)[0], 0.0)
    assert jax.jit(jax.grad(loss))(target) == 0


@pytest.mark.parametrize("clip", [False, True])
def test_degenerate_inverse_cannot_claim_success_or_poison_gradients(clip):
    model = PolynomialFluid(flat_temperature=True)
    evaluate = lambda value: solve_inverse(
        model, jxp.PT_INPUTS, 1e6, value, clip=clip, return_status=True
    )
    result = jax.jit(evaluate)(300.0)
    assert not result[2]
    assert result[4] == jxp.StatusCode.NONCONVERGENCE
    derivative = jax.jit(
        jax.grad(lambda value: jnp.where(evaluate(value)[2], evaluate(value)[0], 0.0))
    )(300.0)
    assert np.isfinite(derivative)
    assert derivative == 0


@pytest.mark.parametrize("pair,names", PAIRS[:5])
def test_monotone_inverse_clipping_projects_to_a_real_boundary_state(
    fluid, pair, names
):
    expected = physical_values(247e3, 1.7e6)
    arguments = [expected[name] for name in names]
    target_index = 0 if pair in (jxp.DmassP_INPUTS, jxp.DmassHmass_INPUTS) else 1
    arguments[target_index] = 1e100
    masked = jax.jit(lambda a, b: solve_inverse(fluid, pair, a, b, return_status=True))(
        *arguments
    )
    assert not masked[2]
    assert masked[4] == jxp.StatusCode.OUT_OF_DOMAIN
    clipped = jax.jit(
        lambda a, b: solve_inverse(fluid, pair, a, b, clip=True, return_status=True)
    )(*arguments)
    assert clipped[2]
    assert clipped[3]
    assert clipped[4] == jxp.StatusCode.SUCCESS
    assert fluid.h_min <= clipped[0] <= fluid.h_max
    assert fluid.p_min <= clipped[1] <= fluid.p_max
    coordinate = (
        clipped[0]
        if pair in (jxp.PT_INPUTS, jxp.DmassP_INPUTS, jxp.PSmass_INPUTS)
        else clipped[1]
    )
    endpoints = (
        (fluid.h_min, fluid.h_max)
        if pair in (jxp.PT_INPUTS, jxp.DmassP_INPUTS, jxp.PSmass_INPUTS)
        else (fluid.p_min, fluid.p_max)
    )
    assert np.isclose(coordinate, endpoints[0]) or np.isclose(coordinate, endpoints[1])


def test_nonmonotone_slice_is_not_misrepresented_as_a_clippable_endpoint():
    model = PolynomialFluid(nonmonotone_temperature=True)
    result = jax.jit(
        lambda temperature: solve_inverse(
            model, jxp.PT_INPUTS, 1e6, temperature, clip=True, return_status=True
        )
    )(1000.0)
    assert not result[2]
    assert not result[3]
    assert result[4] == jxp.StatusCode.NONCONVERGENCE


@pytest.mark.parametrize("pair", [jxp.PT_INPUTS, jxp.DmassT_INPUTS])
@pytest.mark.parametrize("parameter", ["target", "coefficient"])
def test_failed_full_model_input_and_coefficient_gradients_remain_finite_zero(
    fluid, pair, parameter
):
    def evaluate(value):
        model = fluid
        target = jnp.asarray(1e308)
        if parameter == "coefficient":
            model = eqx.tree_at(
                lambda current: current._table["temperature"]["coeffs"],
                model,
                model._table["temperature"]["coeffs"] * value,
            )
        else:
            target = value
        first = 1e6 if pair == jxp.PT_INPUTS else 10.0
        h, p, converged, _ = solve_inverse(model, pair, first, target)
        values = interpolate_properties(model, h, p)
        return sum(jnp.where(converged, field, 0.0) for field in values.values())

    value = jnp.asarray(1.0 if parameter == "coefficient" else 1e308)
    gradient = jax.jit(jax.grad(evaluate))(value)
    assert np.isfinite(gradient)
    assert gradient == 0


def test_forward_and_derivative_programs_contain_no_error_callback(fluid):
    evaluate = jax.jit(
        lambda temperature: solve_inverse(
            fluid, jxp.PT_INPUTS, 1.7e6, temperature, return_status=True
        )
    )
    gradient = jax.jit(jax.grad(lambda temperature: evaluate(temperature)[0]))
    for value in (320.0, 1e308):
        for function in (evaluate, gradient):
            program = function.lower(jnp.asarray(value)).compile().as_text().lower()
            assert "callback" not in program


def test_successful_inverse_coefficient_derivative_matches_implicit_equation(fluid):
    h, p = 247e3, 1.7e6
    target = physical_values(h, p)["temperature"]

    def evaluate(multiplier):
        model = eqx.tree_at(
            lambda current: current._table["temperature"]["coeffs"],
            fluid,
            fluid._table["temperature"]["coeffs"] * multiplier,
        )
        return solve_inverse(model, jxp.PT_INPUTS, p, target)[0]

    derivative = jax.jit(jax.grad(evaluate))(jnp.asarray(1.0))
    u, v = (h - fluid.h_min) / (fluid.h_max - fluid.h_min), np.log(
        p / fluid.p_min
    ) / np.log(fluid.p_max / fluid.p_min)
    slope = (90 + 16 * u + 2 * v) / (fluid.h_max - fluid.h_min)
    np.testing.assert_allclose(derivative, -target / slope, rtol=1e-10)


def test_mixed_vmap_failures_do_not_contaminate_model_parameter_gradient(fluid):
    h, p = 247e3, 1.7e6
    target = physical_values(h, p)["temperature"]
    targets = jnp.asarray([target, 1e308, jnp.nan])

    def evaluate(multiplier):
        model = eqx.tree_at(
            lambda current: current._table["temperature"]["coeffs"],
            fluid,
            fluid._table["temperature"]["coeffs"] * multiplier,
        )
        roots = jax.vmap(
            lambda temperature: solve_inverse(model, jxp.PT_INPUTS, p, temperature)
        )(targets)
        return jnp.where(roots[2], roots[0], 0.0).sum()

    derivative = jax.jit(jax.grad(evaluate))(jnp.asarray(1.0))
    u, v = (h - fluid.h_min) / (fluid.h_max - fluid.h_min), np.log(
        p / fluid.p_min
    ) / np.log(fluid.p_max / fluid.p_min)
    slope = (90 + 16 * u + 2 * v) / (fluid.h_max - fluid.h_min)
    np.testing.assert_allclose(derivative, -target / slope, rtol=1e-10)


def test_clipped_inverse_target_direction_has_zero_derivative(fluid):
    evaluate = jax.jit(
        lambda temperature: solve_inverse(
            fluid, jxp.PT_INPUTS, 1e6, temperature, clip=True
        )
    )
    assert evaluate(1e100)[2]
    assert evaluate(1e100)[3]
    assert jax.jit(jax.grad(lambda temperature: evaluate(temperature)[0]))(1e100) == 0
