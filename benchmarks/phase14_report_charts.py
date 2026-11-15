"""Render the saved JaxProp/PyPI timing comparison as standalone SVG figures.

The report generator owns document layout and interactive tables. This module
uses only the recorded measurements and never evaluates a fluid or benchmark.
"""

from __future__ import annotations

from collections import Counter
from io import StringIO
import math
import re
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

PUBLISHED = "#64748b"
CURRENT = "#0d847c"
FLAG = "#b94e35"
WITHIN = "#a0acba"
TEXT = "#203441"
MUTED = "#657887"
GRID = "#e6edf1"
LIMIT = "#354957"

FAMILIES = (
    "Perfect gas",
    "Bicubic",
    "CoolProp source",
    "JAX callbacks",
    "Metastable",
    "Binary mixtures",
    "State utilities",
)

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def case_family(name: str, case: dict[str, Any] | None = None) -> str:
    """Return a stable, human-readable family for a registered workload."""
    prefixes = (
        ("pg_", "Perfect gas"),
        ("bicubic_", "Bicubic"),
        ("source/", "CoolProp source"),
        ("callback/", "JAX callbacks"),
        ("metastable/", "Metastable"),
        ("binary_", "Binary mixtures"),
        ("state_", "State utilities"),
    )
    for prefix, family in prefixes:
        if name.startswith(prefix):
            return family
    return "Other"


def _state_label(name: str) -> str:
    labels = {
        "state_alias_access": "Alias property access",
        "state_canonical_access": "Canonical property access",
        "state_actual_dictionary_export": "Complete dictionary export",
        "state_common13_dictionary": "13-property dictionary export",
        "state_index_batch16": "Index a 16-state batch",
        "state_stack_two_scalars": "Stack two scalar states",
        "state_concatenate_two_batches": "Concatenate two batches",
        "state_public_construction_scalar": "Public scalar construction",
        "state_public_construction_batch16": "Public 16-state construction",
        "state_lossless_status_validity_export": "Status and validity export",
    }
    return labels.get(name, name.replace("state_", "").replace("_", " "))


def _format_tail(tail: str) -> str:
    """Expand recurring workload tokens without obscuring the input pair."""
    replacements = (
        ("full_state_and_JVP", "full state + JVP"),
        ("fullstate_and_dTdh", "full state + dT/dh"),
        ("common13_projection", "13-property projection"),
        ("heat_exchanger_128cells", "HX, 128 cells"),
        ("scalar_pressure", "scalar pressure"),
        ("broadcast_independent_axes8x16", "8 × 16 independent axes"),
        ("broadcast8x16", "8 × 16 broadcast"),
        ("broadcast16x8", "16 × 8 broadcast"),
        ("broadcast4x4", "4 × 4 broadcast"),
        ("batch1024", "1,024 states"),
        ("batch16", "16 states"),
        ("batch1", "1 state"),
        ("flat128", "128 states, flat"),
        ("application_loop", "application loop"),
        ("near_critical", "near critical"),
        ("supercritical", "supercritical"),
        ("dynamic_model", "dynamic model"),
        ("compiled", "JIT"),
        ("jit", "JIT"),
        ("rhoT", "ρT"),
        ("co2", "CO₂"),
        ("water_nitrogen", "Water/Nitrogen"),
        ("water", "Water"),
        ("air", "Air"),
    )
    for old, new in replacements:
        tail = tail.replace(old, new)
    return tail.replace("_", " · ")


def case_label(name: str) -> str:
    """Make every recorded case distinguishable in figures and HTML tables."""
    if name.startswith("state_"):
        return _state_label(name)
    if name.startswith("pg_phase14_"):
        return _format_tail(name.removeprefix("pg_phase14_")) + " · shape control"
    if name.startswith("bicubic_phase14_"):
        return _format_tail(name.removeprefix("bicubic_phase14_")) + " · shape control"
    if name.startswith("pg_matched_"):
        return _format_tail(name.removeprefix("pg_matched_")) + " · matched"
    if name.startswith("pg_default_"):
        return _format_tail(name.removeprefix("pg_default_")) + " · default"
    if name.startswith("bicubic_"):
        tail = name.removeprefix("bicubic_")
        tail = tail.replace("CO2_", "CO₂ · ").replace("Water_", "Water · ")
        return _format_tail(tail)
    if "/" in name:
        parts = name.split("/")[1:]
        return " · ".join(_format_tail(part) for part in parts)
    return _format_tail(name.removeprefix("binary_native_"))


def _setup_label(name: str) -> str:
    if name.startswith("setup/"):
        return (
            " · ".join(name.split("/")[1:])
            .replace("CO2", "CO₂")
            .replace("water_nitrogen", "Water/Nitrogen")
        )
    if name.startswith("bicubic_"):
        return (
            name.removeprefix("bicubic_")
            .replace("_cached_loading", "")
            .replace("_fresh_generation", "")
            .replace("CO2", "CO₂")
            .replace("_", " · ")
        )
    return {
        "pg_default_construction": "Perfect gas · default",
        "pg_matched_construction": "Perfect gas · matched",
    }.get(name, name.replace("_", " "))


def _number(value: float) -> str:
    """Use compact absolute-time labels across the report's three units."""
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}" if value < 0.1 else f"{value:.2f}"


def _ratio_number(value: float) -> str:
    return f"{value:.2f}×" if value >= 0.1 else f"{value:.3f}×"


def _style_axis(ax: Any) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(axis="both", length=0, pad=8, labelcolor=MUTED)
    ax.tick_params(axis="y", labelcolor=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color(MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def _legend(*, limits: bool = False, outcomes: bool = False) -> list[Any]:
    if outcomes:
        return [
            Patch(color=CURRENT, label="Lower median"),
            Patch(color=WITHIN, label="Higher median, within budget"),
            Patch(color=FLAG, label="Above budget"),
        ]
    handles: list[Any] = [
        Patch(color=PUBLISHED, label="PyPI 0.5.8"),
        Patch(color=CURRENT, label="Current"),
    ]
    if limits:
        handles.append(
            Line2D(
                [],
                [],
                color=LIMIT,
                linestyle="None",
                marker="|",
                markersize=12,
                markeredgewidth=1.7,
                label="Declared budget",
            )
        )
    return handles


def _svg(
    fig: Any,
    name: str,
    title: str,
    description: str,
    label_pairs: list[tuple[Any, Any]] | None = None,
) -> str:
    """Render accessible text and namespace every SVG identifier/reference."""
    if label_pairs:
        # Measure after the caller has set its final subplot layout. A short
        # value label can otherwise cross a nearby budget marker on wide axes.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for label, marker in label_pairs:
            if label.get_window_extent(renderer).overlaps(
                marker.get_window_extent(renderer)
            ):
                label.set_position((5, -14))
    stream = StringIO()
    fig.savefig(
        stream,
        format="svg",
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={"Date": None, "Creator": "JaxProp timing report"},
    )
    plt.close(fig)
    root = ET.fromstring(stream.getvalue())
    prefix = "jaxprop-" + re.sub(r"[^a-zA-Z0-9_-]", "-", name)
    identifiers = {
        element.attrib["id"]: f"{prefix}-{element.attrib['id']}"
        for element in root.iter()
        if "id" in element.attrib
    }
    reference = re.compile(r"#([A-Za-z_][\w:.-]*)")

    def rewrite(value: str) -> str:
        return reference.sub(
            lambda match: "#" + identifiers.get(match.group(1), match.group(1)),
            value,
        )

    for element in root.iter():
        for key, value in list(element.attrib.items()):
            element.attrib[key] = identifiers[value] if key == "id" else rewrite(value)
        if element.tag == f"{{{SVG_NS}}}text" and "style" in element.attrib:
            element.attrib["style"] = element.attrib["style"].replace(
                "font-family: 'DejaVu Sans'",
                "font-family: 'DejaVu Sans', Arial, sans-serif",
            )
        if element.tag == f"{{{SVG_NS}}}style" and element.text:
            element.text = rewrite(element.text)
    title_id, desc_id = prefix + "-title", prefix + "-desc"
    title_element = ET.Element(f"{{{SVG_NS}}}title", {"id": title_id})
    title_element.text = title
    desc_element = ET.Element(f"{{{SVG_NS}}}desc", {"id": desc_id})
    desc_element.text = description
    root.insert(0, title_element)
    root.insert(1, desc_element)
    root.set("role", "img")
    root.set("aria-labelledby", f"{title_id} {desc_id}")
    root.set("style", "max-width:100%;height:auto;display:block")
    return ET.tostring(root, encoding="unicode")


def _paired_bars(
    ax: Any,
    items: list[tuple[str, dict[str, Any]]],
    *,
    divisor: float = 1,
    unit: str = "µs",
    labels: list[str] | None = None,
    limits: bool = False,
    flag_color: bool = False,
    annotation: bool = False,
) -> list[tuple[Any, Any]]:
    """Plot actual medians and the range of four per-process medians."""
    height, offset = 0.27, 0.17
    extent = 0.0
    label_pairs = []
    for index, (_, case) in enumerate(items):
        current_label = None
        for version, shift, color in (
            ("published", -offset, PUBLISHED),
            ("current", offset, FLAG if flag_color else CURRENT),
        ):
            median = case["median_us"][version] / divisor
            workers = case["worker_medians_us"][version]
            lower, upper = min(workers) / divisor, max(workers) / divisor
            ax.barh(index + shift, median, height=height, color=color, zorder=2)
            ax.errorbar(
                median,
                index + shift,
                xerr=[[max(0, median - lower)], [max(0, upper - median)]],
                fmt="none",
                ecolor=color,
                elinewidth=1.05,
                capsize=2.2,
                zorder=3,
            )
            label = ax.annotate(
                _number(median),
                (upper, index + shift),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=8.5,
                color=color,
            )
            if version == "current":
                current_label = label
            extent = max(extent, upper)
        if limits:
            budget = case["limit_us"] / divisor
            marker = ax.plot(
                budget,
                index + offset,
                marker="|",
                color=LIMIT,
                markersize=13,
                markeredgewidth=1.5,
                zorder=4,
            )[0]
            label_pairs.append((current_label, marker))
            extent = max(extent, budget)
        if annotation:
            ratio = case["median_us"]["current"] / case["median_us"]["published"]
            ax.text(
                1.02,
                index,
                f"{100 * (ratio - 1):+.1f}%",
                transform=ax.get_yaxis_transform(),
                va="center",
                color=FLAG,
                fontsize=9,
                fontweight="medium",
            )
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels or [case_label(name) for name, _ in items], fontsize=9)
    ax.set_ylim(len(items) - 0.55, -0.55)
    ax.set_xlim(0, extent * 1.19 if extent else 1)
    ax.set_xlabel(f"Elapsed time per complete call ({unit})", fontsize=9)
    _style_axis(ax)
    return label_pairs


def _overview(cases: dict[str, Any]) -> str:
    counts = {family: Counter() for family in FAMILIES}
    for name, case in cases.items():
        family = case_family(name, case)
        counts.setdefault(family, Counter())
        status = (
            "flag"
            if case["above_original_budget"]
            else "lower" if case["candidate_to_published_ratio"] < 1 else "higher"
        )
        counts[family][status] += 1
    families = [family for family, count in counts.items() if sum(count.values())]
    fig, ax = plt.subplots(figsize=(11.8, 4.2))
    left = [0] * len(families)
    for status, color in (("lower", CURRENT), ("higher", WITHIN), ("flag", FLAG)):
        values = [counts[family][status] for family in families]
        ax.barh(range(len(families)), values, left=left, height=0.58, color=color)
        for index, value in enumerate(values):
            if value:
                ax.text(
                    left[index] + value / 2,
                    index,
                    str(value),
                    ha="center",
                    va="center",
                    color=TEXT if status == "higher" else "white",
                    fontweight="bold",
                    fontsize=9,
                )
        left = [start + value for start, value in zip(left, values)]
    for index, total in enumerate(left):
        ax.text(total + 0.35, index, str(total), va="center", color=MUTED, fontsize=9)
    ax.set_yticks(range(len(families)), families, fontsize=10)
    ax.set_ylim(len(families) - 0.6, -0.6)
    ax.set_xlim(0, max(left) * 1.13)
    ax.set_xlabel("Comparable warmed workloads", fontsize=10)
    _style_axis(ax)
    fig.legend(
        handles=_legend(outcomes=True),
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.55, 1.02),
    )
    fig.subplots_adjust(left=0.18, right=0.97, top=0.87, bottom=0.14)
    return _svg(
        fig,
        "overview",
        "Warmed timing outcomes by workload family",
        "119 comparable workloads. 99 current medians are lower, 15 are higher "
        "within their declared budget and five exceed their budget.",
    )


def _native_panels(data: dict[str, Any], family: str) -> str:
    prefix = "pg_matched_" if family == "pg" else "bicubic_CO2_"
    scalar = [
        f"{mode}_{pair}_scalar"
        for pair in ("PT", "HP", "PS")
        for mode in ("eager", "compiled")
    ]
    batch = [
        "compiled_HP_batch16",
        "compiled_HP_batch1024",
        "compiled_PT_batch1024",
        "compiled_PS_batch1024",
        "compiled_HP_broadcast8x16",
    ]
    if family == "pg":
        specifications = [
            ("Scalar states", scalar, 1, "µs"),
            ("Complete batches", batch + ["dynamic_model_HP_batch1024"], 1, "µs"),
        ]
    else:
        specifications = [
            (
                "Eager scalar states",
                [f"eager_{p}_scalar" for p in ("PT", "HP", "PS")],
                1000,
                "ms",
            ),
            (
                "JIT scalar states",
                [f"compiled_{p}_scalar" for p in ("PT", "HP", "PS")],
                1,
                "µs",
            ),
            ("Complete JIT batches", batch, 1000, "ms"),
        ]
    panel_rows = [len(tails) for _, tails, _, _ in specifications]
    height = 0.45 * sum(panel_rows) + 0.8 * len(specifications) + 0.8
    fig, axes = plt.subplots(
        len(specifications),
        1,
        figsize=(11.8, height),
        gridspec_kw={"height_ratios": panel_rows},
    )
    for ax, (title, tails, divisor, unit) in zip(axes, specifications):
        names = [prefix + tail for tail in tails]
        labels = [
            _format_tail(tail).replace("compiled", "JIT").replace(" · scalar", "")
            for tail in tails
        ]
        _paired_bars(
            ax,
            [(name, data["cases"][name]) for name in names],
            labels=labels,
            divisor=divisor,
            unit=unit,
        )
        ax.tick_params(axis="y", labelsize=9.5)
        ax.xaxis.label.set_fontsize(9.5)
        for label in ax.texts:
            label.set_fontsize(9.5)
        ax.set_title(title, loc="left", color=TEXT, fontsize=11, pad=15)
    fig.legend(
        handles=_legend(),
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.61, 1.01),
    )
    fig.text(
        0.61,
        0.01,
        "Whiskers show the range of four worker medians.",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.32, right=0.95, hspace=0.54, top=0.92, bottom=0.10)
    return _svg(
        fig,
        family,
        f"{'Perfect-gas' if family == 'pg' else 'Bicubic'} full-state timings",
        "Paired PyPI and current median elapsed times for representative complete "
        "scalar and batch state evaluations. Each panel uses a linear axis in the "
        "displayed units. Whiskers span four independent worker medians.",
    )


def _flags(data: dict[str, Any]) -> str:
    names = data["timing_flags"]
    items = [(name, data["cases"][name]) for name in names]
    labels = [
        ("PG" if case_family(name) == "Perfect gas" else "Bicubic")
        + " · "
        + case_label(name).replace(" · matched", "").replace("CO₂ · ", "")
        for name in names
    ]
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    label_pairs = _paired_bars(
        ax, items, labels=labels, limits=True, flag_color=True, annotation=True
    )
    handles = _legend(limits=True)
    handles[1] = Patch(color=FLAG, label="Current, above budget")
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.6, 1.02),
    )
    fig.text(
        0.6,
        0.01,
        "Whiskers show the range of four worker medians.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.39, right=0.89, top=0.87, bottom=0.17)
    return _svg(
        fig,
        "flags",
        "Five warmed workloads exceed their original budgets",
        "Four status-preserving consumers and one added bicubic shape control "
        "remain above budget. Bars show absolute microseconds, whiskers show "
        "worker-median ranges and vertical markers show each declared limit.",
        label_pairs,
    )


def _cold(data: dict[str, Any]) -> str:
    items = []
    for name, profile in data["cold"]["profiles"].items():
        times = profile["public"]["first_synchronized_public_call_us"]
        items.append(
            (
                name,
                {
                    "median_us": times["median_us"],
                    "worker_medians_us": times["raw_us"],
                    "limit_us": profile["original_public_gate"]["limit_us"],
                },
            )
        )
    labels = [
        ("PG" if case_family(name) == "Perfect gas" else "Bicubic")
        + " · "
        + case_label(name).replace(" · matched", "").replace("CO₂ · ", "")
        for name, _ in items
    ]
    fig, ax = plt.subplots(figsize=(12.5, 4.1))
    label_pairs = _paired_bars(
        ax, items, divisor=1000, unit="ms", labels=labels, limits=True
    )
    fig.legend(
        handles=_legend(limits=True),
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.6, 1.02),
    )
    fig.text(
        0.6,
        0.01,
        "Whiskers show the range of four fresh-process calls.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.39, right=0.96, top=0.86, bottom=0.18)
    return _svg(
        fig,
        "cold",
        "First synchronized public-call timings",
        "Four independent unrestricted processes per version and profile. "
        "Elapsed milliseconds include tracing, compilation and first execution. "
        "Each current median meets its original budget marker.",
        label_pairs,
    )


def _setup(data: dict[str, Any], category: str) -> str:
    items = [
        (name, case)
        for name, case in data["setup_cases"].items()
        if case["category"] == category
    ]
    divisor, unit = (1_000_000, "s") if category == "fresh_generation" else (1000, "ms")
    fig, ax = plt.subplots(figsize=(12.2, max(3.2, 0.48 * len(items) + 1.2)))
    label_pairs = _paired_bars(
        ax,
        items,
        divisor=divisor,
        unit=unit,
        labels=[_setup_label(name) for name, _ in items],
        limits=True,
    )
    ax.set_xlabel(f"Elapsed setup time ({unit})", fontsize=9)
    fig.legend(
        handles=_legend(limits=True),
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.6, 1.01),
    )
    fig.text(
        0.6,
        0.01,
        "Whiskers show the range of four worker medians.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.33, right=0.96, top=0.89, bottom=0.14)
    return _svg(
        fig,
        "setup-" + category,
        f"Setup timings for {category.replace('_', ' ')}",
        f"{len(items)} setup comparisons. Bars show PyPI and current elapsed "
        f"times in {unit}; whiskers span four worker medians. Budget markers "
        "include the original category-specific absolute allowance.",
        label_pairs,
    )


def _ratios(family: str, items: list[tuple[str, dict[str, Any]]]) -> str:
    items = sorted(
        items, key=lambda item: item[1]["candidate_to_published_ratio"], reverse=True
    )
    ratios = [case["candidate_to_published_ratio"] for _, case in items]
    xmin = min(min(ratios) * 0.72, 0.72)
    xmax = max(max(ratios) * 1.65, 1.65)
    fig, ax = plt.subplots(figsize=(13.0, max(3.2, 0.35 * len(items) + 1.35)))
    for index, (_, case) in enumerate(items):
        ratio = case["candidate_to_published_ratio"]
        color = (
            FLAG if case["above_original_budget"] else CURRENT if ratio < 1 else WITHIN
        )
        ax.barh(index, ratio - 1, left=1, color=color, height=0.61, zorder=2)
        ax.plot(ratio, index, "o", markersize=3.1, color=color, zorder=3)
        limit = case["limit_us"] / case["median_us"]["published"]
        if xmin <= limit <= xmax:
            ax.plot(
                limit,
                index,
                marker="|",
                markersize=10,
                markeredgewidth=1.2,
                color=LIMIT,
                zorder=4,
            )
        ax.text(
            1.015,
            index,
            _ratio_number(ratio),
            transform=ax.get_yaxis_transform(),
            color=color,
            va="center",
            fontsize=9,
            fontweight="medium",
        )
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(len(items) - 0.6, -0.6)
    ax.set_yticks(
        range(len(items)), [case_label(name) for name, _ in items], fontsize=8.5
    )
    ax.set_xlabel("Current elapsed time / PyPI elapsed time (log scale)", fontsize=9)
    _style_axis(ax)
    ticks = [
        value
        for value in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)
        if xmin <= value <= xmax
    ]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axvline(1, color=LIMIT, linewidth=1.0, zorder=1)
    ax.text(
        1,
        1.015,
        "Equal time",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )
    handles = _legend(outcomes=True)
    handles.append(
        Line2D(
            [],
            [],
            color=LIMIT,
            linestyle="None",
            marker="|",
            markersize=10,
            label="Budget, when visible",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.65, 1.01),
    )
    fig.subplots_adjust(left=0.40, right=0.91, top=0.91, bottom=0.10)
    return _svg(
        fig,
        "ratios-" + family,
        f"All {family.lower()} timing ratios",
        f"{len(items)} comparable warmed workloads, sorted by current-to-PyPI "
        "elapsed-time ratio. Ratios below one have lower current medians. "
        "Bars begin at parity and the logarithmic axis displays both large "
        "improvements and higher medians. Markers show visible declared budgets.",
    )


def build_charts(data: dict[str, Any]) -> dict[str, Any]:
    """Build SVG chart strings from the validated portable timing artifact."""
    comparable = {
        name: case
        for name, case in data["cases"].items()
        if case["manifest"]["comparable"] and "published" in case["median_us"]
    }
    expected = data["workload_counts"]["comparable"]
    if len(comparable) != expected:
        raise ValueError(
            f"Expected {expected} comparable cases, found {len(comparable)}"
        )
    if any(
        not math.isfinite(case["candidate_to_published_ratio"])
        or case["candidate_to_published_ratio"] <= 0
        for case in comparable.values()
    ):
        raise ValueError("Timing ratio charts require finite, positive ratios")
    settings = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": "jaxprop-phase14-report",
        "savefig.transparent": False,
    }
    with matplotlib.rc_context(settings):
        charts = {
            "overview": _overview(comparable),
            "pg": _native_panels(data, "pg"),
            "bicubic": _native_panels(data, "bicubic"),
            "flags": _flags(data),
            "cold": _cold(data),
            "setup_construction": _setup(data, "construction"),
            "setup_cache": _setup(data, "cached_loading"),
            "setup_generation": _setup(data, "fresh_generation"),
        }
        families = [*FAMILIES, "Other"]
        charts["ratios"] = {
            family: _ratios(family, items)
            for family in families
            if (
                items := [
                    (name, case)
                    for name, case in comparable.items()
                    if case_family(name, case) == family
                ]
            )
        }
    return charts
