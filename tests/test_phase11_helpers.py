"""Numerical smoothing and explicit solver-selection utility contracts."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxprop import math as smoothing
from jaxprop.helpers_jax import make_diffrax_adjoint, make_diffrax_solver


@pytest.mark.parametrize("method", ["logsumexp", "boltzmann", "p-norm"])
def test_smooth_extrema_broadcast_and_approach_the_requested_extremum(method):
    left = np.array([[1.0], [2.0]])
    right = np.array([1.25, 2.25, 3.25])
    maximum = smoothing.smooth_maximum(left, right, method=method, alpha=100)
    minimum = smoothing.smooth_minimum(left, right, method=method, alpha=100)
    assert maximum.shape == minimum.shape == (2, 3)
    np.testing.assert_allclose(maximum, np.maximum(left, right), rtol=0.01, atol=0.01)
    # The p-norm reciprocal formulation is restricted to positive inputs.
    np.testing.assert_allclose(minimum, np.minimum(left, right), rtol=0.01, atol=0.01)


@pytest.mark.parametrize(
    "method", ["quadratic", "hyperbolic", "logarithmic", "logcosh"]
)
def test_smooth_absolute_value_is_even_finite_and_differentiable(method):
    values = jnp.array([-10.0, -0.1, 0.0, 0.1, 10.0])
    function = lambda value: smoothing.smooth_abs(value, method=method, epsilon=1e-3)
    result = jax.jit(function)(values)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, result[::-1], rtol=1e-13)
    assert abs(float(jax.grad(function)(0.0))) < 1e-12
    assert np.all(np.isfinite(jax.vmap(jax.grad(function))(values)))
    np.testing.assert_allclose(
        np.asarray(result)[[0, 1, 3, 4]],
        np.abs(np.asarray(values)[[0, 1, 3, 4]]),
        atol=0.01,
    )


def test_logcosh_smoothing_derivative_matches_the_analytical_tanh():
    epsilon = 0.2
    values = jnp.linspace(-2, 2, 19)
    actual = jax.vmap(
        jax.grad(lambda x: smoothing.smooth_abs(x, "logarithmic", epsilon))
    )(values)
    np.testing.assert_allclose(
        actual, np.tanh(values / epsilon), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("name", ["sigmoid_smoothstep", "sigmoid_smootherstep"])
def test_smooth_steps_clip_and_have_stationary_endpoint_derivatives(name):
    function = getattr(smoothing, name)
    np.testing.assert_allclose(
        function(jnp.array([-1, 0, 0.5, 1, 2])), [0, 0, 0.5, 1, 1]
    )
    np.testing.assert_allclose(
        jax.vmap(jax.grad(function))(jnp.array([0.0, 1.0])), 0.0, atol=1e-12
    )
    if name == "sigmoid_smootherstep":
        np.testing.assert_allclose(
            jax.vmap(jax.grad(jax.grad(function)))(jnp.array([0.0, 1.0])),
            0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "factory,name,expected",
    [
        (make_diffrax_solver, " Tsit5 ", dfx.Tsit5),
        (make_diffrax_solver, "Dopri5", dfx.Dopri5),
        (
            make_diffrax_adjoint,
            "RecursiveCheckpointAdjoint",
            dfx.RecursiveCheckpointAdjoint,
        ),
        (make_diffrax_adjoint, "DirectAdjoint", dfx.DirectAdjoint),
    ],
)
def test_solver_and_adjoint_factories_select_actual_supported_classes(
    factory, name, expected
):
    assert isinstance(factory(name), expected)


@pytest.mark.parametrize(
    "factory,name",
    [
        (make_diffrax_solver, "Tsit6"),
        (make_diffrax_adjoint, "RecursiveCheckpointAdjont"),
    ],
)
def test_invalid_solver_and_adjoint_names_include_suggestions_and_supported_options(
    factory, name
):
    with pytest.raises(ValueError, match="Did you mean.*") as captured:
        factory(name)
    assert "List of valid" in str(captured.value)


@pytest.mark.parametrize(
    "function",
    [smoothing.smooth_maximum, smoothing.smooth_minimum, smoothing.smooth_abs],
)
def test_smoothing_rejects_unknown_methods_explicitly(function):
    args = (1.0,) if function is smoothing.smooth_abs else (1.0, 2.0)
    with pytest.raises(ValueError, match="Unsupported method"):
        function(*args, method="unknown")
