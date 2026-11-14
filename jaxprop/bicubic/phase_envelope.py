"""Construction-time rejection of cells crossed by source phase boundaries.

Corner phase flags do not determine the interior of an enthalpy/log-pressure
cell. In particular, the dew enthalpy can have an extremum between pressure
nodes. Probe the configured source's bubble and dew curves adaptively and
reject every cell intersecting either curve's padded enthalpy range over its
pressure strip. Cells wholly within the two-phase region remain eligible.

This is a conservative, finite-resolution source-envelope policy, not a proof
of the exact EOS phase boundary at all real pressures. Unresolved curvature
widens the excluded boundary range by a full enthalpy cell; failed source
flashes invalidate the affected strip. The mask is persisted in
the validated table and combined with the existing nodal guard at setup; no
source calls or extra numerical tests enter native state evaluation.
"""

import numpy as np
import CoolProp.CoolProp as CP

PHASE_ENVELOPE_SETTINGS = {
    "policy": "adaptive_pq_boundary_ranges_v1",
    "minimum_segments": 4,
    "maximum_depth": 8,
    "curvature_cell_fraction": 1e-3,
    "padding_factor": 4.0,
    "unresolved_padding_cells": 1.0,
    "critical_gap_fraction": 1e-6,
    "roundoff_ulps": 64,
}


def source_phase_cell_validity(
    source,
    enthalpy_nodes,
    log_pressure_nodes,
    *,
    critical_pressure,
    critical_enthalpy=None,
    minimum_saturation_pressure=None,
    settings=None,
):
    """Return source-envelope cell eligibility using only setup-time flashes.

    ``source`` must already capture the table's source/reference convention.
    Pressure limits refer to the source's actual bubble/dew endpoints, including
    pseudo-pure fluids with temperature glide. Only their endpoint enthalpies
    are required; no common saturation-temperature assumption is made.
    """
    h = np.asarray(enthalpy_nodes, dtype=float)
    logp = np.asarray(log_pressure_nodes, dtype=float)
    allowed = np.ones((h.size - 1, logp.size - 1), dtype=bool)
    if critical_pressure is None or not np.isfinite(critical_pressure):
        # Some sources have no evaluable critical coordinate. Their existing
        # nodal guard remains applicable; do not invent a saturation envelope.
        return allowed
    critical_logp = np.log(critical_pressure)
    lower_logp = (
        -np.inf
        if minimum_saturation_pressure is None
        else np.log(minimum_saturation_pressure)
    )
    # The table constructor supplies the policy captured in its manifest.
    # Standalone callers may use a fresh default snapshot. Never rediscover a
    # mutable module policy halfway through a generation transaction.
    settings = dict(PHASE_ENVELOPE_SETTINGS if settings is None else settings)
    cell_span = float(np.min(np.diff(h)))
    tolerance = settings["curvature_cell_fraction"] * cell_span
    cache = {}

    def endpoints(point):
        point = float(point)
        if point not in cache:
            pressure = float(np.exp(point))
            values = []
            for quality in (0.0, 1.0):
                source.update(CP.PQ_INPUTS, pressure, quality)
                values.append(float(source.hmass()))
            result = np.asarray(values)
            if not np.all(np.isfinite(result)) or result[0] > result[1]:
                raise ValueError("Source saturation endpoints are not ordered/finite")
            cache[point] = result
        return cache[point]

    def segment(a, b, depth):
        left, right = endpoints(a), endpoints(b)
        middle = 0.5 * (a + b)
        center = endpoints(middle)
        bend = np.abs(center - (0.5 * left + 0.5 * right))
        unresolved = np.max(bend) > tolerance
        if unresolved and depth < settings["maximum_depth"]:
            first = segment(a, middle, depth + 1)
            second = segment(middle, b, depth + 1)
            return (
                np.minimum(first[0], second[0]),
                np.maximum(first[1], second[1]),
            )
        values = np.stack((left, center, right))
        # Curvature padding is in enthalpy units, and depends on the cell span,
        # not the arbitrary absolute enthalpy reference. A small ULP allowance
        # additionally accounts for representing offset source coordinates.
        roundoff = (
            settings["roundoff_ulps"]
            * np.finfo(float).eps
            * np.maximum(np.max(np.abs(values), axis=0), 1.0)
        )
        padding = settings["padding_factor"] * np.maximum(bend, tolerance) + roundoff
        if unresolved:
            padding = np.maximum(
                padding, settings["unresolved_padding_cells"] * cell_span
            )
        return np.min(values, axis=0) - padding, np.max(values, axis=0) + padding

    safe_critical = np.log(critical_pressure * (1 - settings["critical_gap_fraction"]))
    for column, (left, right) in enumerate(zip(logp[:-1], logp[1:])):
        if left >= critical_logp or right < lower_logp:
            continue
        a, b = max(left, lower_logp), min(right, safe_critical)
        try:
            if a < b:
                subdivisions = np.linspace(a, b, settings["minimum_segments"] + 1)
                ranges = [
                    segment(x, y, 0)
                    for x, y in zip(subdivisions[:-1], subdivisions[1:])
                ]
                low = np.min(np.stack([pair[0] for pair in ranges]), axis=0)
                high = np.max(np.stack([pair[1] for pair in ranges]), axis=0)
            elif left >= safe_critical:
                # A very narrow strip can start inside the source's excluded
                # critical gap. Enclose it using the outer sampled endpoints
                # and the critical coordinate instead of rejecting far-away
                # single-phase enthalpies throughout that pressure strip.
                outer = endpoints(safe_critical)
                padding = settings["padding_factor"] * tolerance
                low, high = outer - padding, outer + padding
            else:
                raise ValueError("No resolvable saturation interval")
            if right >= safe_critical:
                if critical_enthalpy is None or not np.isfinite(critical_enthalpy):
                    raise ValueError("Critical endpoint enthalpy is unavailable")
                low = np.minimum(low, critical_enthalpy)
                high = np.maximum(high, critical_enthalpy)
            crossing = np.any(
                (h[:-1, None] <= high[None, :]) & (h[1:, None] >= low[None, :]),
                axis=1,
            )
            allowed[:, column] = ~crossing
        except (ValueError, RuntimeError, NotImplementedError, OverflowError):
            # An unverified strip must not become trusted by a failed flash.
            allowed[:, column] = False
    return allowed
