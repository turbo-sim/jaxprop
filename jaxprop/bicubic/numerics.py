"""Raw bicubic equations and bounded, callback-free inverse calculations.

This module does not construct public states or choose the caller's failure
policy. The request layer owns those operations. Iterative residuals evaluate
only their requested fields, and failed implicit derivatives are zero.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

from .. import helpers_props as pairs
from ..fluid_api import StatusCode
from ..property_schema import ALIAS_TO_CANONICAL, PROPERTIES_CANONICAL

_CORE = frozenset(
    {"pressure", "temperature", "density", "enthalpy", "entropy", "internal_energy"}
)
_BOUNDARY_RTOL = 1e-12


def _cell_basis(fluid, h, p):
    ii = (h - fluid.h_min) / fluid.delta_h
    jj = (jnp.log(p) - jnp.log(fluid.p_min)) / fluid.delta_logP
    i = jnp.clip(jnp.floor(ii).astype(jnp.int32), 0, fluid.N_h - 2)
    j = jnp.clip(jnp.floor(jj).astype(jnp.int32), 0, fluid.N_p - 2)
    x, y = ii - i, jj - j
    # The trailing axis is the four polynomial powers, leaving all leading
    # thermodynamic batch axes intact. Scalar residual callers retain (4,).
    xm = jnp.stack((jnp.ones_like(x), x, x * x, x * x * x), axis=-1)
    ym = jnp.stack((jnp.ones_like(y), y, y * y, y * y * y), axis=-1)
    return i, j, xm, ym


def interpolate_properties(fluid, h, p, names=None, *, cell_basis=None):
    """Evaluate canonical properties without state assembly/checks.

    Bounds and cell validity are checked by the caller. A prepared native table
    stores finite coefficient arrays and separate cell availability masks, so
    unavailable optional data cannot poison otherwise zero cotangents.

    Natural-coordinate batches use leading batch axes and a trailing 16-term
    polynomial axis. The request layer may share its already protected cell
    coordinates/basis, so validation and interpolation locate each cell once.
    The caller supplies h and p already broadcast to the same batch shape;
    ``cell_basis``, when supplied, belongs to those same coordinates.
    """
    if names is None:
        enabled = fluid.capabilities.properties
        names = tuple(name for name in PROPERTIES_CANONICAL if name in enabled)
    else:
        names = tuple(ALIAS_TO_CANONICAL[name] for name in names)
    i, j, xm, ym = _cell_basis(fluid, h, p) if cell_basis is None else cell_basis
    if xm.ndim == 1:
        basis = jnp.kron(ym, xm)
    else:
        # kron is a tensor product of complete arrays, not a pointwise batch
        # operation. Keep its scalar coefficient ordering within each row.
        basis = (ym[..., :, None] * xm[..., None, :]).reshape((*i.shape, 16))
    masks = getattr(fluid, "_coefficient_validity", {})
    values = {}
    for name in names:
        if name == "enthalpy":
            values[name] = h
        elif name == "pressure":
            values[name] = p
        else:
            coefficients = fluid._table[name]["coeffs"][i, j]
            if name in masks:
                available = True if masks[name] is None else masks[name][i, j]
            elif name not in _CORE:
                # Compatibility for standalone raw-table callers. Configured
                # native evaluators prepare this mask once during setup.
                available = jnp.all(jnp.isfinite(coefficients), axis=-1)
                coefficients = jnp.where(jnp.isfinite(coefficients), coefficients, 0)
            else:
                available = True
            # Small dot products are efficient for scalar/small requests.
            # On larger batches, reductions fuse gathers and polynomial
            # products and avoid a separate batched-dot dispatch per field.
            # The selector uses static shapes, with no numerical runtime test.
            if basis.ndim == 1:
                value = jnp.dot(coefficients, basis)
            elif i.size < 64:
                value = jnp.einsum("...k,...k->...", coefficients, basis)
            else:
                value = jnp.sum(coefficients * basis, axis=-1)
            values[name] = jnp.where(available & jnp.isfinite(value), value, jnp.nan)
    return values


def interpolate_property(fluid, h, p, name):
    """Evaluate one residual field without calculating unrelated properties."""
    canonical = ALIAS_TO_CANONICAL[name]
    return interpolate_properties(fluid, h, p, (canonical,))[canonical]


def _coordinates(fluid, point, fixed, mode):
    if mode == "fixed_pressure":
        h = fluid.h_min + point[0] * (fluid.h_max - fluid.h_min)
        return h, fixed
    if mode == "fixed_enthalpy":
        p = jnp.exp(
            jnp.log(fluid.p_min)
            + point[0] * (jnp.log(fluid.p_max) - jnp.log(fluid.p_min))
        )
        return fixed, p
    h = fluid.h_min + point[0] * (fluid.h_max - fluid.h_min)
    p = jnp.exp(
        jnp.log(fluid.p_min) + point[1] * (jnp.log(fluid.p_max) - jnp.log(fluid.p_min))
    )
    return h, p


def _root_residual(point, context):
    fluid, names, targets, scales, fixed, mode = context
    h, p = _coordinates(fluid, point, fixed, mode)
    values = interpolate_properties(fluid, h, p, names)
    actual = jnp.stack(tuple(values[name] for name in names))
    return _scaled_difference(actual, targets, scales)


def _scaled_difference(actual, target, scale):
    # Saturation preserves every zero while keeping unreachable huge requests
    # finite inside linear solves and squared initial-guess distances.
    limit = jnp.sqrt(jnp.finfo(jnp.result_type(actual, target, scale)).max) / 16
    return jnp.clip((actual - target) / scale, -limit, limit)


@eqx.filter_custom_jvp
def _bounded_root(initial, context, *, max_steps, tol):
    """Solve in normalized h/log-p coordinates and independently verify the root.

    A solver success flag alone is insufficient: the final scaled residual and
    local Jacobian must also admit a finite, identifiable inverse before the
    custom implicit derivative is allowed to use it.
    """
    solver = optx.Newton(rtol=tol, atol=tol)
    solution = optx.root_find(
        _root_residual,
        solver,
        initial,
        args=context,
        options={"lower": jnp.zeros_like(initial), "upper": jnp.ones_like(initial)},
        max_steps=max_steps,
        throw=False,
    )
    finite = jnp.all(jnp.isfinite(solution.value))
    point = jnp.where(jnp.isfinite(solution.value), solution.value, initial)
    point = jnp.clip(point, 0.0, 1.0)
    residual = _root_residual(point, context)
    precision = jnp.finfo(point.dtype)
    residual_limit = jnp.maximum(tol, 64 * precision.eps)
    jacobian = jax.jacfwd(_root_residual)(point, context)
    if point.size == 1:
        identifiable = jnp.isfinite(jacobian[0, 0]) & (
            jnp.abs(jacobian[0, 0]) > precision.tiny
        )
    else:
        magnitude = jnp.max(jnp.abs(jacobian))
        normalized = jacobian / jnp.where(magnitude > 0, magnitude, 1.0)
        determinant = (
            normalized[0, 0] * normalized[1, 1] - normalized[0, 1] * normalized[1, 0]
        )
        identifiable = jnp.isfinite(determinant) & (
            jnp.abs(determinant) > 64 * precision.eps
        )
    successful = (
        (solution.result == optx.RESULTS.successful)
        & finite
        & jnp.all(jnp.isfinite(residual))
        & (jnp.max(jnp.abs(residual)) <= residual_limit)
        & identifiable
    )
    return point, successful


@_bounded_root.def_jvp
def _bounded_root_jvp(primals, tangents, *, max_steps, tol):
    initial, context = primals
    _, context_tangent = tangents
    point, successful = _bounded_root(initial, context, max_steps=max_steps, tol=tol)
    jacobian = jax.jacfwd(_root_residual)(point, context)
    _, rhs = eqx.filter_jvp(
        lambda args: _root_residual(point, args), (context,), (context_tangent,)
    )
    # For a root R(z, theta)=0, J_z dz = -dR/dtheta. Differentiate this
    # implicit equation rather than the sequence of Newton iterations.
    # Optimistix's default implicit adjoint uses a throwing linear solve. Small
    # analytical systems let failed inverses use safe zero tangents without a
    # conditional callback or differentiation through unsuccessful iterations.
    identity = jnp.eye(point.size, dtype=point.dtype)
    active = successful & jnp.all(jnp.isfinite(jacobian))
    matrix = jnp.where(active, jacobian, identity)
    rhs = jnp.where(active, rhs, jnp.zeros_like(rhs))
    if point.size == 1:
        denominator = matrix[0, 0]
        nonsingular = denominator != 0
        denominator = jnp.where(nonsingular, denominator, 1.0)
        tangent = -rhs / denominator
    else:
        a, b = matrix[0]
        c, d = matrix[1]
        determinant = a * d - b * c
        nonsingular = jnp.isfinite(determinant) & (determinant != 0)
        determinant = jnp.where(nonsingular, determinant, 1.0)
        tangent = jnp.stack((b * rhs[1] - d * rhs[0], c * rhs[0] - a * rhs[1]))
        tangent = tangent / determinant
    tangent = jnp.where(active & nonsingular, tangent, jnp.zeros_like(tangent))
    return (point, successful), (tangent, None)


def _response_scale(values):
    values = jnp.asarray(values)
    largest = jnp.max(jnp.abs(values))
    return jnp.maximum(jnp.ptp(values), jnp.maximum(1e-12 * largest, 1.0))


def _quadratic_extrema(cubic):
    """Exact derivative extrema of each cell's cubic on the unit interval."""
    c1, c2, c3 = cubic[..., 1], cubic[..., 2], cubic[..., 3]
    denominator = jnp.where(c3 != 0, 3 * c3, 1.0)
    vertex = -c2 / denominator
    vertex = jnp.clip(vertex, 0.0, 1.0)
    values = jnp.stack(
        (c1, c1 + 2 * c2 + 3 * c3, c1 + 2 * c2 * vertex + 3 * c3 * vertex**2),
        axis=-1,
    )
    return jnp.min(values), jnp.max(values)


def _monotone_target_bounds(fluid, fixed, mode, name):
    """Prove monotonicity over all cubic cells of a fixed-coordinate slice."""
    if mode == "fixed_pressure":
        _, j, _, ym = _cell_basis(fluid, fluid.h_min, fixed)
        matrices = fluid._table[name]["coeffs"][:, j].reshape((-1, 4, 4))
        cubic = jnp.sum(matrices * ym[None, :, None], axis=1)
        low = interpolate_property(fluid, fluid.h_min, fixed, name)
        high = interpolate_property(fluid, fluid.h_max, fixed, name)
        slice_index = (slice(None), j)
    else:
        i, _, xm, _ = _cell_basis(fluid, fixed, fluid.p_min)
        matrices = fluid._table[name]["coeffs"][i].reshape((-1, 4, 4))
        cubic = jnp.sum(matrices * xm[None, None, :], axis=2)
        low = interpolate_property(fluid, fixed, fluid.p_min, name)
        high = interpolate_property(fluid, fixed, fluid.p_max, name)
        slice_index = (i, slice(None))
    minimum, maximum = _quadratic_extrema(cubic)
    scale = jnp.maximum(jnp.abs(minimum), jnp.abs(maximum))
    roundoff = 64 * jnp.finfo(cubic.dtype).eps * scale
    monotone = jnp.all(jnp.isfinite(cubic)) & (
        (minimum >= -roundoff) | (maximum <= roundoff)
    )
    if hasattr(fluid, "_cell_validity"):
        monotone = monotone & jnp.all(fluid._cell_validity[slice_index])
    masks = getattr(fluid, "_coefficient_validity", {})
    if name in masks and masks[name] is not None:
        monotone = monotone & jnp.all(masks[name][slice_index])
    return jnp.minimum(low, high), jnp.maximum(low, high), monotone


def _boundary_tolerance(low, high):
    dtype = jnp.result_type(low, high, 0.0)
    relative = jnp.maximum(_BOUNDARY_RTOL, 64 * jnp.finfo(dtype).eps)
    return relative * jnp.maximum(jnp.maximum(jnp.abs(low), jnp.abs(high)), 1.0)


def solve_inverse(
    fluid,
    input_type,
    first,
    second,
    *,
    clip=False,
    max_steps=64,
    tol=1e-10,
    return_status=False,
):
    """Return ``(h, p, converged, projected)`` for one bounded inverse.

    With ``return_status=True``, append an integer numerical status. Fixed-h/p
    requests have a provable domain error only when their slice is monotone.
    An unreachable or ambiguous two-property target remains a solver failure;
    clipping never fabricates a two-dimensional root.
    """
    dtype = jnp.result_type(first, second, fluid.h_vals, fluid.logP_vals, 0.0)
    first, second = jnp.asarray(first, dtype=dtype), jnp.asarray(second, dtype=dtype)
    finite = jnp.isfinite(first) & jnp.isfinite(second)
    h_mid = 0.5 * (fluid.h_min + fluid.h_max)
    p_mid = jnp.sqrt(fluid.p_min) * jnp.sqrt(fluid.p_max)
    projected = jnp.asarray(False)
    outside = jnp.asarray(False)

    if input_type in (pairs.PT_INPUTS, pairs.DmassP_INPUTS, pairs.PSmass_INPUTS):
        mode = "fixed_pressure"
        if input_type == pairs.DmassP_INPUTS:
            target, fixed = first, second
            name = "density"
        else:
            fixed, target = first, second
            name = "temperature" if input_type == pairs.PT_INPUTS else "entropy"
        fixed = jnp.where(jnp.isfinite(fixed), fixed, p_mid)
        margin = _boundary_tolerance(fluid.p_min, fluid.p_max)
        fixed_outside = (fixed < fluid.p_min - margin) | (fixed > fluid.p_max + margin)
        fixed = jnp.clip(fixed, fluid.p_min, fluid.p_max)
        j = jnp.argmin(jnp.abs(fluid.logP_vals - jnp.log(fixed)))
        values = fluid._table[name]["value"][:, j]
        scale = _response_scale(fluid._table[name]["value"])
        target = jnp.where(jnp.isfinite(target), target, values[fluid.N_h // 2])
        low, high, monotone = _monotone_target_bounds(fluid, fixed, mode, name)
        margin = _boundary_tolerance(low, high)
        target_outside = monotone & ((target < low - margin) | (target > high + margin))
        target = jnp.where(monotone, jnp.clip(target, low, high), target)
        if clip:
            projected = fixed_outside | target_outside
        else:
            outside = fixed_outside | target_outside
        i = jnp.argmin(jnp.abs(values - target))
        initial = jnp.reshape(
            (fluid.h_vals[i] - fluid.h_min) / (fluid.h_max - fluid.h_min), (1,)
        )
        names, targets, scales = (
            (name,),
            jnp.reshape(target, (1,)),
            jnp.reshape(scale, (1,)),
        )
    elif input_type in (pairs.HmassSmass_INPUTS, pairs.DmassHmass_INPUTS):
        mode = "fixed_enthalpy"
        if input_type == pairs.HmassSmass_INPUTS:
            fixed, target, name = first, second, "entropy"
        else:
            target, fixed, name = first, second, "density"
        fixed = jnp.where(jnp.isfinite(fixed), fixed, h_mid)
        margin = _boundary_tolerance(fluid.h_min, fluid.h_max)
        fixed_outside = (fixed < fluid.h_min - margin) | (fixed > fluid.h_max + margin)
        fixed = jnp.clip(fixed, fluid.h_min, fluid.h_max)
        i = jnp.argmin(jnp.abs(fluid.h_vals - fixed))
        values = fluid._table[name]["value"][i]
        scale = _response_scale(fluid._table[name]["value"])
        target = jnp.where(jnp.isfinite(target), target, values[fluid.N_p // 2])
        low, high, monotone = _monotone_target_bounds(fluid, fixed, mode, name)
        margin = _boundary_tolerance(low, high)
        target_outside = monotone & ((target < low - margin) | (target > high + margin))
        target = jnp.where(monotone, jnp.clip(target, low, high), target)
        if clip:
            projected = fixed_outside | target_outside
        else:
            outside = fixed_outside | target_outside
        j = jnp.argmin(jnp.abs(values - target))
        initial = jnp.reshape(
            (fluid.logP_vals[j] - jnp.log(fluid.p_min))
            / (jnp.log(fluid.p_max) - jnp.log(fluid.p_min)),
            (1,),
        )
        names, targets, scales = (
            (name,),
            jnp.reshape(target, (1,)),
            jnp.reshape(scale, (1,)),
        )
    elif input_type in (pairs.DmassT_INPUTS, pairs.DmassSmass_INPUTS):
        mode, fixed = "two_dimensional", jnp.asarray(0.0, dtype=dtype)
        names = (
            "density",
            "temperature" if input_type == pairs.DmassT_INPUTS else "entropy",
        )
        values = tuple(fluid._table[name]["value"] for name in names)
        targets = jnp.stack((first, second))
        fallback = jnp.stack(
            tuple(value[fluid.N_h // 2, fluid.N_p // 2] for value in values)
        )
        targets = jnp.where(jnp.isfinite(targets), targets, fallback)
        scales = jnp.stack(tuple(_response_scale(value) for value in values))
        stride = max(1, fluid.coarse_step)
        distances = jnp.stack(
            tuple(
                _scaled_difference(value[::stride, ::stride], target, scale)
                for value, target, scale in zip(values, targets, scales)
            )
        )
        distance_scale = jnp.maximum(jnp.max(jnp.abs(distances)), 1.0)
        error = jnp.sum((distances / distance_scale) ** 2, axis=0)
        index = jnp.argmin(error)
        i, j = jnp.unravel_index(index, error.shape)
        initial = jnp.stack(
            (
                (fluid.h_vals[::stride][i] - fluid.h_min) / (fluid.h_max - fluid.h_min),
                (fluid.logP_vals[::stride][j] - jnp.log(fluid.p_min))
                / (jnp.log(fluid.p_max) - jnp.log(fluid.p_min)),
            )
        )
    else:
        raise ValueError(f"Unsupported internal bicubic inverse pair: {input_type}")

    context = fluid, names, targets, scales, fixed, mode
    point, converged = _bounded_root(initial, context, max_steps=max_steps, tol=tol)
    h, p = _coordinates(fluid, point, fixed, mode)
    # A log/exp round trip may lie one ULP beyond a pressure endpoint. Keep
    # smooth polynomial derivatives inside the solve, then return exact bounds.
    h = jnp.clip(h, fluid.h_min, fluid.h_max)
    p = jnp.clip(p, fluid.p_min, fluid.p_max)
    converged = converged & finite & ~outside
    status = jnp.where(converged, StatusCode.SUCCESS, StatusCode.NONCONVERGENCE)
    status = jnp.where(outside, StatusCode.OUT_OF_DOMAIN, status)
    status = jnp.where(finite, status, StatusCode.INVALID_INPUT).astype(jnp.int32)
    result = h, p, converged, projected
    return (*result, status) if return_status else result
