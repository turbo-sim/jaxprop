import jax.numpy as jnp


def smooth_maximum(x1, x2, method="boltzmann", alpha=10):
    r"""Element-wise smooth maximum of two broadcast-compatible inputs.

    This is a smooth counterpart of ``numpy.maximum(x1, x2)``, not an axis
    reduction. The two inputs broadcast to a common shape and are compared at
    each element. Let :math:`a=|\alpha|>0` and :math:`m=\max(x_1,x_2)`.

    The positive-input power-sum approximation is given by
    :cite:p:`weisstein_vector_2023`:

    .. math::

       P_a(x_1,x_2) = \left(x_1^a+x_2^a\right)^{1/a}.

    The Boltzmann approximation is an exponentially weighted mean:

    .. math::

       B_a(x_1,x_2) =
       \frac{\sum_{i=1}^{2}x_i\exp\left(a(x_i-m)\right)}
            {\sum_{i=1}^{2}\exp\left(a(x_i-m)\right)}.

    It favors the larger input and remains between the inputs up to numerical
    roundoff. The implementation adds machine epsilon to the weight denominator.
    This weighted mean differs from the log-sum-exp approximation described by
    :cite:p:`blanchard_accurately_2021`:

    .. math::

       L_a(x_1,x_2) = m + \frac{1}{a}
       \log\left(\sum_{i=1}^{2}\exp\left(a(x_i-m)\right)\right).

    Log-sum-exp approaches the maximum from above, with an excess no greater
    than :math:`\log(2)/a`. Shifting the exponential arguments by :math:`m`
    avoids overflow in those weights. The direct power-sum implementation does
    not use this exponential stabilization.

    Parameters
    ----------
    x1, x2 : array_like
        Scalar or array inputs compatible under JAX broadcasting.
    method : {"boltzmann", "logsumexp", "p-norm"}, optional
        Approximation method. The default is the Boltzmann weighted mean.
    alpha : float, optional
        Finite nonzero sharpness parameter; default 10. Its magnitude is used,
        so its sign does not select a minimum. Increasing its magnitude tightens
        the approximation in the method's valid numerical regime.

    Returns
    -------
    max_approx : jax.Array
        Element-wise approximation with the common broadcast shape. Scalar
        inputs produce a zero-dimensional array.

    Raises
    ------
    ValueError
        If an unsupported method is specified.

    Notes
    -----
    Use strictly positive inputs for ``p-norm``. It evaluates powers of the raw
    inputs, not their absolute values, and is not a signed-vector norm. Its
    positive-input domain and the nonzero-sharpness requirement are caller
    preconditions, not runtime validation. Nonfinite inputs, zero sharpness or
    overflowing powers can therefore produce nonfinite numerical results.

    See Also
    --------
    smooth_minimum : Element-wise approximation favoring the smaller input.
    """
    # Ensure x1 and x2 have the same shape by broadcasting
    x1, x2 = jnp.broadcast_arrays(x1, x2)

    # Stack the input arrays along a new axis to treat them as a single array
    x = jnp.stack([x1, x2], axis=0)

    # Compute smooth maximum approximation according to specified method
    if method == "logsumexp":
        return _smooth_max_logsumexp(x, jnp.abs(alpha), axis=0)
    elif method == "boltzmann":
        return _smooth_max_boltzmann(x, jnp.abs(alpha), axis=0)
    elif method == "p-norm":
        return _smooth_max_pnorm(x, jnp.abs(alpha), axis=0)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods are:\n"
            "- 'logsumexp'\n"
            "- 'boltzmann'\n"
            "- 'p-norm'"
        )


def smooth_minimum(x1, x2, method="boltzmann", alpha=10):
    r"""Element-wise smooth minimum of two broadcast-compatible inputs.

    This is a smooth counterpart of ``numpy.minimum(x1, x2)``, not an axis
    reduction. It uses the same methods as :func:`smooth_maximum` with the
    internally signed exponent :math:`-a`, where :math:`a=|\alpha|>0`.

    For log-sum-exp, let :math:`m=\min(x_1,x_2)`:

    .. math::

       L_{-a}(x_1,x_2) = m - \frac{1}{a}
       \log\left(\sum_{i=1}^{2}\exp\left(-a(x_i-m)\right)\right).

    It approaches the minimum from below, with a deficit no greater than
    :math:`\log(2)/a`. Boltzmann instead uses the weighted mean

    .. math::

       B_{-a}(x_1,x_2) =
       \frac{\sum_{i=1}^{2}x_i\exp\left(-a(x_i-m)\right)}
            {\sum_{i=1}^{2}\exp\left(-a(x_i-m)\right)},

    favoring the smaller input and remaining between the inputs up to numerical
    roundoff. The implementation adds machine epsilon to the weight denominator.
    The power-sum method uses

    .. math::

       P_{-a}(x_1,x_2) = \left(x_1^{-a}+x_2^{-a}\right)^{-1/a}.

    Its negative exponent requires strictly positive inputs; zero inputs are
    singular and negative inputs do not define the intended real-valued method.

    Parameters
    ----------
    x1, x2 : array_like
        Scalar or array inputs compatible under JAX broadcasting.
    method : {"boltzmann", "logsumexp", "p-norm"}, optional
        Approximation method. The default is the Boltzmann weighted mean.
    alpha : float, optional
        Finite nonzero sharpness parameter; default 10. Only its magnitude is
        used. Increasing its magnitude tightens the approximation in the
        method's valid numerical regime.

    Returns
    -------
    min_approx : jax.Array
        Element-wise approximation with the common broadcast shape. Scalar
        inputs produce a zero-dimensional array.

    Raises
    ------
    ValueError
        If an unsupported method is specified.

    Notes
    -----
    The ``p-norm`` implementation powers the raw inputs without taking absolute
    values or stabilizing the power sum. Positive inputs and nonzero sharpness
    are caller preconditions, not runtime validation. Extreme powers or invalid
    inputs can produce nonfinite numerical results.

    See Also
    --------
    smooth_maximum : Element-wise approximation favoring the larger input.
    """
    # Ensure x1 and x2 have the same shape by broadcasting
    x1, x2 = jnp.broadcast_arrays(x1, x2)

    # Stack the input arrays along a new axis to treat them as a single array
    x = jnp.stack([x1, x2], axis=0)

    # Compute smooth maximum approximation according to specified method
    if method == "logsumexp":
        return _smooth_max_logsumexp(x, -jnp.abs(alpha), axis=0)
    elif method == "boltzmann":
        return _smooth_max_boltzmann(x, -jnp.abs(alpha), axis=0)
    elif method == "p-norm":
        return _smooth_max_pnorm(x, -jnp.abs(alpha), axis=0)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods are:\n"
            "- 'logsumexp'\n"
            "- 'boltzmann'\n"
            "- 'p-norm'"
        )


def _smooth_max_logsumexp(x, alpha, axis=None, keepdims=False):
    r"""Reduce an array using shifted log-sum-exp and a signed exponent.

    Positive ``alpha`` approximates a maximum; negative ``alpha`` approximates
    a minimum. The shift follows the corresponding extremum and stabilizes
    the exponential weights. ``axis`` and ``keepdims`` apply to this internal
    reduction, not to the public two-input element-wise interface.
    """

    # Determine the shift for numerical stability
    shift_value = jnp.sign(alpha) * jnp.max(
        jnp.sign(alpha) * x, axis=axis, keepdims=True
    )

    # Compute log-sum-exp with the shift and scale by alpha
    log_sum = jnp.log(
        jnp.sum(jnp.exp(alpha * (x - shift_value)), axis=axis, keepdims=True)
    )

    # Normalize the result by alpha and correct for the shift
    smooth_max = (log_sum + alpha * shift_value) / alpha

    # Remove the dimensions of size one if keepdims is False
    if not keepdims:
        smooth_max = jnp.squeeze(smooth_max, axis=axis)

    return smooth_max


def _smooth_max_boltzmann(x, alpha, axis=None, keepdims=False):
    r"""Reduce an array using an exponentially weighted arithmetic mean.

    Positive ``alpha`` favors larger values; negative ``alpha`` favors smaller
    values. Shifted weights avoid exponential overflow. Machine epsilon is
    added to the denominator. This weighted mean is distinct from log-sum-exp.
    """

    # Compute the shift for numerical stability
    shift = jnp.sign(alpha) * jnp.max(jnp.sign(alpha) * x, axis=axis, keepdims=True)

    # Compute the weighted sum (numerator) of the elements of x
    weighted_sum = jnp.sum(x * jnp.exp(alpha * (x - shift)), axis=axis, keepdims=True)

    # Compute the sum of the smooth_max weights (denominator)
    weight_sum = jnp.sum(jnp.exp(alpha * (x - shift)), axis=axis, keepdims=True)

    # Compute the Boltzmann-weighted average avoiding division by zero
    smooth_max = weighted_sum / (weight_sum + jnp.finfo(float).eps)

    # Remove the dimensions of size one if keepdims is False
    if not keepdims:
        smooth_max = jnp.squeeze(smooth_max, axis=axis)

    return smooth_max


def _smooth_max_pnorm(x, alpha, axis=None, keepdims=False):
    r"""Reduce strictly positive inputs with a signed power-sum approximation.

    Positive ``alpha`` approaches the maximum and negative ``alpha`` approaches
    the minimum. Powers use the raw inputs without absolute values, domain
    validation or overflow stabilization. In particular, the negative-exponent
    minimum is singular at zero. ``alpha`` must be finite and nonzero.
    """

    # Compute the p-norm approximation
    smooth_max = jnp.sum(jnp.power(x, alpha), axis=axis, keepdims=True) ** (1 / alpha)

    # Remove the dimensions of size one if keepdims is False
    if not keepdims:
        smooth_max = jnp.squeeze(smooth_max, axis=axis)

    return smooth_max


def smooth_abs(x, method="quadratic", epsilon=1e-5):
    r"""Compute an element-wise smooth approximation of the absolute value.

    The quadratic approximation is given by :cite:p:`ramirez_x_2013`:

    .. math::

       f_{\epsilon}(x) = \sqrt{x^2 + \epsilon}.

    The hyperbolic-tangent approximation is given by
    :cite:p:`bagul_smooth_2017`:

    .. math::

       f_{\epsilon}(x) = x\tanh\left(\frac{x}{\epsilon}\right).

    The log-cosh approximation is given by :cite:p:`saleh_statistical_2022`:

    .. math::

       f_{\epsilon}(x) = \epsilon
       \log\left(\cosh\left(\frac{x}{\epsilon}\right)\right).

    All three are even, smooth functions approaching :math:`|x|` as positive
    :math:`\epsilon` tends to zero. Their smoothing scales differ: the quadratic
    formula adds epsilon to :math:`x^2`, whereas the other two divide x by
    epsilon. Equal numerical epsilon values do not imply equal smoothing error
    or physical units. The log-cosh implementation uses ``logaddexp`` to avoid
    overflowing an intermediate ``cosh`` at small smoothing widths.

    Parameters
    ----------
    x : float or jax.Array
        Input value or array of values.
    method : {"quadratic", "hyperbolic", "logcosh", "logarithmic"}, optional
        Approximation method. ``quadratic`` is the default. ``logarithmic`` is
        an alias of ``logcosh``.
    epsilon : float, optional
        Finite positive smoothing parameter; default 1e-5. Its numerical
        interpretation depends on the selected formula as described above.

    Returns
    -------
    jax.Array
        Element-wise smooth approximation with the input shape.

    Raises
    ------
    ValueError
        If an unsupported approximation method is provided.

    Notes
    -----
    Positivity and finiteness of epsilon are caller preconditions, not runtime
    validation. The formulas retain their ordinary floating-point limitations;
    for example, the direct quadratic expression can overflow when squaring an
    extremely large input.
    """

    if method == "quadratic":
        return _smooth_abs_quadratic(x, epsilon)
    elif method == "hyperbolic":
        return _smooth_abs_tanh(x, epsilon)
    elif method in ("logarithmic", "logcosh"):
        return _smooth_abs_logcosh(x, epsilon)

    else:
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods are:\n"
            "- 'quadratic'\n"
            "- 'hyperbolic'\n"
            "- 'logcosh'"
        )


def _smooth_abs_quadratic(x, epsilon):
    """Quadratic approximation of the absolute value function."""
    return jnp.sqrt(x**2 + epsilon)


def _smooth_abs_tanh(x, epsilon):
    """Hyperbolic tangent approximation of the absolute value function."""
    return x * jnp.tanh(x / epsilon)


def _smooth_abs_logcosh(x, epsilon):
    """Log-cosh approximation without overflowing the intermediate cosh.

    log(cosh(z)) = logaddexp(z, -z) - log(2). The direct expression overflowed
    for ordinary inputs when the smoothing width was small.
    """
    scaled = x / epsilon
    return epsilon * (jnp.logaddexp(scaled, -scaled) - jnp.log(2.0))


def sind(x):
    """Compute the sine of an angle given in degrees."""
    return jnp.sin(x * jnp.pi / 180)


def cosd(x):
    """Compute the cosine of an angle given in degrees."""

    return jnp.cos(x * jnp.pi / 180)


def tand(x):
    """Compute the tangent of an angle given in degrees."""
    return jnp.tan(x * jnp.pi / 180)


def arcsind(x):
    """Compute the arcsine of a value and return the result in degrees."""
    return jnp.arcsin(x) * 180 / jnp.pi


def arccosd(x):
    """Compute the arccosine of a value and return the result in degrees."""
    return jnp.arccos(x) * 180 / jnp.pi


def arctand(x):
    """Compute the arctangent of a value and return the result in degrees."""
    return jnp.arctan(x) * 180 / jnp.pi


def is_odd(number):
    """Check if a number is odd. Returns True if the provided number is odd, and False otherwise."""
    return number % 2 != 0


def is_even(number):
    """Check if a number is even. Returns True if the provided number is even, and False otherwise."""
    return number % 2 == 0


def all_numeric(array):
    """Check if all items in Numpy array are numeric (floats or ints)"""
    return jnp.issubdtype(array.dtype, jnp.number)


def all_non_negative(array):
    "Check if all items in Numpy array are non-negative"
    return jnp.all(array >= 0)


def sigmoid_hyperbolic(x, x0=0.5, alpha=1):
    r"""Compute a sigmoid using the hyperbolic tangent.

    The formula is

    .. math::

        \sigma(x) = \frac{1 + \tanh\left(\frac{x - x_0}{\alpha}\right)}{2}.

    Parameters
    ----------
    x : array_like
        Input data.
    x0 : float, optional
        Center of the sigmoid; default 0.5.
    alpha : float, optional
        Nonzero scale parameter; default 1. A positive value gives an increasing
        transition, while a negative value reverses its direction. Its magnitude
        controls the transition width. The scale is not validated at runtime.

    Returns
    -------
    jax.Array
        Sigmoid values with the input shape.
    """
    x = jnp.array(x)  # Ensure x is a NumPy array for vectorized operations
    sigma = (1 + jnp.tanh((x - x0) / alpha)) / 2
    return sigma


def sigmoid_rational(x, n, m):
    r"""
    Compute the sigmoid rational function.

    This function calculates a sigmoid function using an algebraic approach based on a rational function.
    The formula is given by:

    .. math::
        \sigma(x) = \frac{x^n}{x^n + (1-x)^m}

    Parameters
    ----------
    x : array_like
        Input data.
    n : int
        Power of the numerator.
    m : int
        Power of the denominator.

    Returns
    -------
    array_like
        Output of the sigmoid algebraic function.

    """
    x = jnp.array(x)  # Ensure x is a NumPy array for vectorized operations
    x = jnp.where(x < 0, 0, x)  # Set x to 0 where x < 0
    x = jnp.where(x > 1, 1, x)  # Set x to 1 where x > 1
    sigma = x**n / (x**n + (1 - x) ** m)
    return sigma


def sigmoid_smoothstep(x):
    """
    Compute the smooth step function.

    This function calculates a smooth step function with zero first-order
    derivatives at the endpoints. More information available at:
    https://resources.wolframcloud.com/FunctionRepository/resources/SmoothStep/

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    array_like
        Output of the sigmoid smoothstep function.

    """
    x = jnp.array(x)  # Ensure x is a NumPy array for vectorized operations
    x = jnp.where(x < 0, 0, x)  # Set x to 0 where x < 0
    x = jnp.where(x > 1, 1, x)  # Set x to 1 where x > 1
    return x**2 * (3 - 2 * x)


def sigmoid_smootherstep(x):
    """
    Compute the smoother step function.

    This function calculates a smoother step function with zero second-order
    derivatives at the endpoints. It is a modification of the smoothstep
    function to provide smoother transitions.

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    array_like
        Output of the sigmoid smootherstep function.

    """
    x = jnp.array(x)  # Ensure x is a NumPy array for vectorized operations
    x = jnp.where(x < 0, 0, x)  # Set x to 0 where x < 0
    x = jnp.where(x > 1, 1, x)  # Set x to 1 where x > 1
    return 6 * x**5 - 15 * x**4 + 10 * x**3
