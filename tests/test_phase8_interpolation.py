"""Independent polynomial and coordinate checks for bicubic interpolation."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprop as jxp
from jaxprop.bicubic.bicubic_interpolation import compute_bicubic_coefficients


def polynomial(coefficients, u, v, derivative_u=0, derivative_v=0):
    """Evaluate a physical polynomial independently of Hermite matrix ordering."""
    result = np.zeros(np.broadcast_shapes(np.shape(u), np.shape(v)))
    for a in range(derivative_u, 4):
        for b in range(derivative_v, 4):
            multiplier = 1
            for k in range(derivative_u):
                multiplier *= a - k
            for k in range(derivative_v):
                multiplier *= b - k
            result = result + (
                coefficients[a, b]
                * multiplier
                * u ** (a - derivative_u)
                * v ** (b - derivative_v)
            )
    return result


@pytest.mark.parametrize("h_degree", range(4))
@pytest.mark.parametrize("logp_degree", range(4))
def test_all_sixteen_hermite_basis_coefficients_have_independent_order(
    h_degree, logp_degree
):
    coefficients = np.zeros((4, 4))
    coefficients[h_degree, logp_degree] = 1.0
    u, v = np.meshgrid([0.0, 1.0], [0.0, 1.0], indexing="ij")
    dh, dlogp = 2.0, 0.5
    generated = compute_bicubic_coefficients(
        polynomial(coefficients, u, v),
        polynomial(coefficients, u, v, 1, 0) / dh,
        polynomial(coefficients, u, v, 0, 1) / dlogp,
        polynomial(coefficients, u, v, 1, 1) / (dh * dlogp),
        dh,
        dlogp,
    )
    expected = np.zeros(16)
    # The evaluator multiplies kron([1,y,y^2,y^3], [1,x,x^2,x^3]).
    expected[4 * logp_degree + h_degree] = 1.0
    assert generated.shape == (1, 1, 16)
    np.testing.assert_array_equal(generated[0, 0], expected)


@pytest.fixture(scope="module")
def polynomial_table():
    h = np.linspace(100.0, 250.0, 4)
    logp = np.linspace(np.log(2e5), np.log(2e6), 5)
    h_span, logp_span = h[-1] - h[0], logp[-1] - logp[0]
    u, v = np.meshgrid((h - h[0]) / h_span, (logp - logp[0]) / logp_span, indexing="ij")
    polynomial_coefficients = np.array(
        [
            [500.0, 3.0, -0.25, 0.125],
            [2.0, 0.5, -0.1, 0.08],
            [0.9, -0.2, 0.15, -0.03],
            [0.5, 0.04, -0.06, 0.02],
        ]
    )
    table = {}
    for name in jxp.PROPERTIES_CANONICAL:
        value = np.ones_like(u)
        grad_h = np.zeros_like(u)
        grad_logp = np.zeros_like(u)
        mixed = np.zeros_like(u)
        if name == "temperature":
            value = polynomial(polynomial_coefficients, u, v)
            grad_h = polynomial(polynomial_coefficients, u, v, 1, 0) / h_span
            grad_logp = polynomial(polynomial_coefficients, u, v, 0, 1) / logp_span
            mixed = polynomial(polynomial_coefficients, u, v, 1, 1) / (
                h_span * logp_span
            )
        elif name == "density":
            value = np.full_like(u, 100.0)
        elif name == "viscosity":
            value = grad_h = grad_logp = mixed = np.full_like(u, np.nan)
        table[name] = {
            "value": jnp.asarray(value),
            "coeffs": jnp.asarray(
                compute_bicubic_coefficients(
                    value, grad_h, grad_logp, mixed, h[1] - h[0], logp[1] - logp[0]
                )
            ),
        }
    model = SimpleNamespace(
        h_min=float(h[0]),
        h_max=float(h[-1]),
        p_min=float(np.exp(logp[0])),
        p_max=float(np.exp(logp[-1])),
        N_h=len(h),
        N_p=len(logp),
        h_vals=jnp.asarray(h),
        logP_vals=jnp.asarray(logp),
        delta_h=float(h[1] - h[0]),
        delta_logP=float(logp[1] - logp[0]),
        coarse_step=1,
        _table=table,
        capabilities=SimpleNamespace(
            properties=frozenset(
                ("temperature", "pressure", "enthalpy", "density", "viscosity")
            )
        ),
    )
    return model, polynomial_coefficients, h, logp


def locations(h, logp, location):
    if location == "nodes":
        return np.meshgrid(h, np.exp(logp), indexing="ij")
    if location == "cell_centers":
        return np.meshgrid(
            (h[:-1] + h[1:]) / 2, np.exp((logp[:-1] + logp[1:]) / 2), indexing="ij"
        )
    # Include every domain edge, internal cell edge, and both sides of a knot.
    hs = np.concatenate((h, (h[:-1] + h[1:]) / 2, h[1:-1] - 1e-8, h[1:-1] + 1e-8))
    ps = np.exp(np.concatenate((logp, (logp[:-1] + logp[1:]) / 2)))
    return np.meshgrid(hs, ps, indexing="ij")


@pytest.mark.parametrize("location", ("nodes", "cell_centers", "edges"))
def test_global_bicubic_polynomial_is_exact_across_cells_and_boundaries(
    polynomial_table, location
):
    from jaxprop.bicubic.numerics import interpolate_properties

    model, coefficients, h, logp = polynomial_table
    enthalpy, pressure = locations(h, logp, location)
    actual = jax.jit(
        jax.vmap(
            lambda x, p: interpolate_properties(model, x, p, names=("temperature",))[
                "temperature"
            ]
        )
    )(jnp.asarray(enthalpy.ravel()), jnp.asarray(pressure.ravel()))
    expected = polynomial(
        coefficients,
        (enthalpy.ravel() - h[0]) / (h[-1] - h[0]),
        (np.log(pressure.ravel()) - logp[0]) / (logp[-1] - logp[0]),
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-11)


@pytest.mark.parametrize("location", ("nodes", "cell_centers", "edges"))
def test_logpressure_chain_rule_and_mixed_derivative_match_analytic_polynomial(
    polynomial_table, location
):
    from jaxprop.bicubic.numerics import interpolate_properties

    model, coefficients, h, logp = polynomial_table
    enthalpy, pressure = locations(h, logp, location)
    enthalpy, pressure = enthalpy.ravel(), pressure.ravel()
    dh, dlogp = h[-1] - h[0], logp[-1] - logp[0]
    u, v = (enthalpy - h[0]) / dh, (np.log(pressure) - logp[0]) / dlogp

    def temperature(x, p):
        return interpolate_properties(model, x, p, names=("temperature",))[
            "temperature"
        ]

    first = jax.jit(jax.vmap(jax.grad(temperature, argnums=(0, 1))))(
        jnp.asarray(enthalpy), jnp.asarray(pressure)
    )
    mixed = jax.jit(jax.vmap(jax.grad(jax.grad(temperature, argnums=0), argnums=1)))(
        jnp.asarray(enthalpy), jnp.asarray(pressure)
    )
    np.testing.assert_allclose(
        first[0], polynomial(coefficients, u, v, 1, 0) / dh, rtol=2e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        first[1],
        polynomial(coefficients, u, v, 0, 1) / (dlogp * pressure),
        rtol=2e-11,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        mixed,
        polynomial(coefficients, u, v, 1, 1) / (dh * dlogp * pressure),
        rtol=2e-9,
        atol=1e-16,
    )


def test_requested_coordinates_are_exact_and_optional_nan_has_zero_ad_contribution(
    polynomial_table,
):
    from jaxprop.bicubic.numerics import interpolate_properties

    model, _, _, _ = polynomial_table
    enthalpy, pressure = jnp.asarray(176.25), jnp.asarray(761234.5)

    def evaluate(x, p):
        return interpolate_properties(
            model, x, p, names=("enthalpy", "pressure", "temperature", "viscosity")
        )

    actual = jax.jit(evaluate)(enthalpy, pressure)
    np.testing.assert_array_equal(actual["enthalpy"], enthalpy)
    np.testing.assert_array_equal(actual["pressure"], pressure)
    assert np.isnan(actual["viscosity"])
    derivative = jax.jit(
        jax.grad(lambda x: jnp.nan_to_num(evaluate(x, pressure)["viscosity"]))
    )(enthalpy)
    assert derivative == 0.0


def test_prepared_table_preserves_local_optional_values_and_zero_failed_gradients(
    polynomial_table,
):
    from jaxprop.bicubic.numerics import interpolate_properties
    from jaxprop.bicubic.runtime import prepare_runtime

    original, coefficients, h, logp = polynomial_table
    table = {name: dict(entry) for name, entry in original._table.items()}
    # One available cell surrounded by genuinely unavailable optional data.
    optional = jnp.full_like(table["viscosity"]["coeffs"], jnp.nan)
    table["viscosity"]["coeffs"] = optional.at[0, 0].set(0.0).at[0, 0, 0].set(1e-5)
    numerical, masks, _ = prepare_runtime(
        table, original.capabilities.properties, "mixture"
    )
    model = SimpleNamespace(
        **(vars(original) | {"_table": numerical, "_coefficient_validity": masks})
    )
    enthalpy = jnp.asarray([(h[0] + h[1]) / 2, (h[1] + h[2]) / 2])
    pressure = jnp.asarray(np.exp((logp[0] + logp[1]) / 2))

    def evaluate(x):
        return interpolate_properties(
            model, x, pressure, names=("temperature", "viscosity")
        )

    actual = jax.jit(jax.vmap(evaluate))(enthalpy)
    u = (np.asarray(enthalpy) - h[0]) / (h[-1] - h[0])
    v = (np.log(pressure) - logp[0]) / (logp[-1] - logp[0])
    np.testing.assert_allclose(
        actual["temperature"], polynomial(coefficients, u, v), rtol=3e-14
    )
    assert actual["viscosity"][0] == 1e-5
    assert np.isnan(actual["viscosity"][1])
    derivative = jax.jit(
        jax.vmap(jax.grad(lambda x: jnp.nan_to_num(evaluate(x)["viscosity"])))
    )(enthalpy)
    np.testing.assert_array_equal(derivative, [0.0, 0.0])
