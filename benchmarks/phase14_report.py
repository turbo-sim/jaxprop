"""Build the standalone visual report from recorded Phase 14 measurements.

Run from the checkout with ``python benchmarks/phase14_report.py``.
The report embeds its figures, case data, styles and interactions in one file.
This script reads saved measurements and performs no thermodynamic evaluations.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re

from phase14_report_charts import build_charts, case_family, case_label

SUPPORT = Path(__file__).resolve().parents[1] / "development/performance/phase14"

STYLE = """
:root {
  color-scheme: light;
  --ink: #172c35;
  --muted: #52666c;
  --line: #dce5e5;
  --page: #f3f6f5;
  --panel: #ffffff;
  --teal: #0d847c;
  --teal-soft: #e8f4f1;
  --slate: #64748b;
  --rust: #b94e35;
  --rust-soft: #fff0e9;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--page);
  font-family: Inter, "Segoe UI", Arial, sans-serif;
  font-size: 16px;
  line-height: 1.55;
}
a { color: #0c716f; text-underline-offset: 3px; }
a:hover { color: #13474f; }
button, input, select { font: inherit; }
button, select { cursor: pointer; }
button, input, select, summary { accent-color: var(--teal); }
:focus-visible { outline: 3px solid #156cba; outline-offset: 3px; }
header, main, footer { max-width: 1320px; margin: 0 auto; }
header { padding: 46px 48px 24px; }
main { padding: 0 48px 52px; }
footer { padding: 0 48px 40px; color: var(--muted); font-size: 13px; }
.heading-row { display: flex; justify-content: space-between; gap: 24px; align-items: start; }
.eyebrow { margin: 0 0 10px; font-size: 12px; letter-spacing: .13em; text-transform: uppercase; color: var(--teal); font-weight: 700; }
h1 { margin: 0; font-size: clamp(30px, 4vw, 48px); line-height: 1.12; letter-spacing: -.04em; font-weight: 650; }
.subtitle { margin: 14px 0 8px; font-size: 19px; color: var(--muted); }
.meta { margin: 0; font-size: 13px; color: var(--muted); }
.gate-status { flex: 0 0 auto; margin-top: 4px; background: var(--rust-soft); color: #963f2e; padding: 9px 14px; border-radius: 7px; font-size: 13px; font-weight: 650; }
nav { display: flex; flex-wrap: wrap; gap: 8px 22px; padding: 23px 0 0; font-size: 14px; }
nav a { text-decoration: none; padding: 5px 0; }
.metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 0 0 26px; }
.metric { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 21px 22px; }
.metric-label { margin: 0 0 7px; font-size: 13px; color: var(--muted); }
.metric-value { margin: 0; font-size: 34px; line-height: 1.15; letter-spacing: -.035em; font-weight: 650; font-variant-numeric: tabular-nums; }
.metric-note { margin: 9px 0 0; font-size: 12px; color: var(--muted); }
section { margin: 28px 0 0; scroll-margin-top: 24px; }
.section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 20px; margin-bottom: 13px; }
h2 { margin: 0; font-size: 25px; line-height: 1.25; letter-spacing: -.025em; font-weight: 650; }
h3 { margin: 0; font-size: 18px; font-weight: 650; }
.section-note { margin: 8px 0 16px; color: var(--muted); font-size: 14px; max-width: 970px; }
.section-count { color: var(--muted); font-size: 13px; white-space: nowrap; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 22px 24px; }
.chart { margin: 0; }
.chart-scroll { overflow-x: auto; }
.chart svg { display: block; width: 100%; height: auto; }
.chart text { font-family: Inter, "Segoe UI", Arial, sans-serif !important; }
figcaption { margin-top: 12px; font-size: 13px; color: var(--muted); }
.figure-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; }
.figure-grid > * { min-width: 0; }
.atlas { margin-top: 14px; }
details { border-top: 1px solid var(--line); padding: 15px 0; }
details:last-child { padding-bottom: 0; }
summary { cursor: pointer; font-weight: 600; font-size: 15px; }
summary .summary-note { font-size: 12px; color: var(--muted); font-weight: 400; margin-left: 10px; }
details .chart { margin-top: 16px; }
.table-controls { display: grid; grid-template-columns: 1.8fr 1.1fr 1fr 1.2fr; gap: 12px; margin: 18px 0 14px; }
label { display: block; font-size: 12px; color: var(--muted); }
input, select { width: 100%; min-height: 44px; margin-top: 6px; background: #fff; border: 1px solid #b9c9cc; border-radius: 7px; padding: 9px 10px; font-size: 14px; color: var(--ink); }
.table-summary { display: flex; justify-content: space-between; align-items: center; gap: 20px; margin: 8px 0 14px; font-size: 13px; color: var(--muted); }
.plain-button { color: #0c716f; background: transparent; border: 0; padding: 7px 2px; text-decoration: underline; text-underline-offset: 3px; font-size: 13px; }
.table-scroll { overflow: auto; max-height: 780px; border-top: 1px solid var(--line); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { background: #f6f9f9; text-align: left; color: #3a535c; font-weight: 600; position: sticky; top: 0; z-index: 1; }
th, td { padding: 12px 11px; border-bottom: 1px solid #e6ecec; vertical-align: top; }
th.numeric, td.numeric { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.workload-cell { min-width: 260px; }
.workload-name { display: block; font-weight: 600; }
.case-id { display: block; margin-top: 5px; color: var(--muted); font-family: "IBM Plex Mono", Consolas, monospace; font-size: 11px; line-height: 1.4; overflow-wrap: anywhere; }
.case-kind { color: var(--muted); font-size: 11px; margin-top: 4px; display: block; }
.good-value { color: #076b62; }
.higher-value { color: #335a79; }
.bad-value { color: var(--rust); }
.status { display: inline-block; white-space: nowrap; font-size: 11px; padding: 3px 7px; border-radius: 4px; }
.status-pass { background: var(--teal-soft); color: #076b62; }
.status-flag { background: var(--rust-soft); color: #963f2e; }
.status-new { background: #edf1f5; color: #485c70; }
.empty { margin: 24px 0; color: var(--muted); }
.method-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px 32px; }
.method-grid h3 { margin-bottom: 7px; }
.method-grid p { margin: 0; font-size: 14px; color: var(--muted); }
.provenance { margin-top: 25px; padding-top: 20px; border-top: 1px solid var(--line); }
.provenance dl { display: grid; grid-template-columns: 175px 1fr; gap: 8px 20px; font-size: 12px; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
code { font-family: "IBM Plex Mono", Consolas, monospace; font-size: .92em; }
.callout { margin-top: 16px; padding-left: 14px; border-left: 3px solid var(--rust); color: var(--muted); font-size: 14px; }
.current-only { margin-top: 22px; }
.current-only table { margin-top: 10px; }
.current-only th { position: static; }
[hidden] { display: none !important; }
@media (max-width: 1050px) {
  header { padding: 32px 28px 22px; }
  main { padding: 0 28px 36px; }
  footer { padding: 0 28px 30px; }
  .heading-row { display: block; }
  .gate-status { display: inline-block; margin-top: 18px; }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .figure-grid { grid-template-columns: 1fr; }
  .table-controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 560px) {
  header { padding: 28px 18px 20px; }
  main { padding: 0 18px 30px; }
  footer { padding: 0 18px 24px; }
  .subtitle { font-size: 17px; }
  .metrics { gap: 10px; }
  .metric { padding: 16px 13px; }
  .metric-value { font-size: 28px; }
  .metric-note { font-size: 11px; }
  .panel { padding: 18px 15px; }
  .section-head { display: block; }
  .section-count { display: block; margin-top: 5px; }
  .method-grid { grid-template-columns: 1fr; }
  .table-controls { grid-template-columns: 1fr; }
  input, select { font-size: 16px; }
  .provenance dl { grid-template-columns: 1fr; gap: 4px; }
  dd { margin-bottom: 10px; }
  summary .summary-note { display: block; margin-left: 0; margin-top: 4px; }
}
@media print {
  body { background: #fff; font-size: 11pt; }
  header, main, footer { padding: 15px 0; max-width: none; }
  nav, .table-controls, .table-summary, .plain-button { display: none; }
  .panel, .metric { break-inside: avoid; box-shadow: none; }
  .metrics { grid-template-columns: repeat(4, 1fr); }
  .figure-grid { grid-template-columns: 1fr; }
  .table-scroll, .chart-scroll { max-height: none; overflow: visible; }
  .chart-canvas { min-width: 0 !important; }
  .table-scroll { overflow-wrap: anywhere; }
  .case-id { font-size: 7pt; }
  th { position: static; }
  .workload-cell { min-width: 0; }
  .gate-status { border: 1px solid #b94e35; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
}
"""

SCRIPT = """
(() => {
  "use strict";
  const dataset = JSON.parse(document.getElementById("report-data").textContent);
  const byId = new Map(dataset.cases.map(item => [item.id, item]));
  const body = document.getElementById("case-body");
  const rows = Array.from(body.rows);
  const search = document.getElementById("case-search");
  const family = document.getElementById("case-family");
  const status = document.getElementById("case-status");
  const sort = document.getElementById("case-sort");
  const count = document.getElementById("case-count");
  const empty = document.getElementById("no-results");

  function compare(left, right) {
    const a = byId.get(left.dataset.caseId);
    const b = byId.get(right.dataset.caseId);
    if (sort.value === "time") return b.current - a.current;
    if (sort.value === "name") return a.label.localeCompare(b.label);
    if (sort.value === "fastest") {
      return (a.ratio ?? Infinity) - (b.ratio ?? Infinity);
    }
    if (sort.value === "slowest") {
      return (b.ratio ?? -Infinity) - (a.ratio ?? -Infinity);
    }
    return Number(b.flag) - Number(a.flag)
      || (b.ratio ?? -Infinity) - (a.ratio ?? -Infinity)
      || a.label.localeCompare(b.label);
  }

  function update() {
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    rows.sort(compare);
    for (const row of rows) {
      const item = byId.get(row.dataset.caseId);
      const matchesText = !query || item.search.includes(query);
      const matchesFamily = family.value === "all" || family.value === item.family;
      const matchesStatus = status.value === "all"
        || (status.value === "flag" && item.flag)
        || (status.value === "pass" && item.ratio !== null && !item.flag)
        || (status.value === "new" && item.ratio === null);
      row.hidden = !(matchesText && matchesFamily && matchesStatus);
      if (!row.hidden) visible += 1;
      body.appendChild(row);
    }
    count.textContent = `${visible} of ${rows.length} workloads shown`;
    empty.hidden = visible !== 0;
  }

  search.addEventListener("input", update);
  for (const control of [family, status, sort]) {
    control.addEventListener("change", update);
  }
  document.getElementById("reset-filters").addEventListener("click", () => {
    search.value = "";
    family.value = "all";
    status.value = "all";
    sort.value = "budget";
    update();
  });

  let printState = [];
  window.addEventListener("beforeprint", () => {
    printState = Array.from(document.querySelectorAll("details"))
      .map(element => [element, element.open]);
    for (const [element] of printState) element.open = true;
  });
  window.addEventListener("afterprint", () => {
    for (const [element, open] of printState) element.open = open;
  });
  update();
})();
"""


def time_text(microseconds):
    """Format durations without mixing units inside one numeric label."""
    if microseconds is None:
        return "—"
    if microseconds >= 1e6:
        return f"{microseconds / 1e6:,.3f} s"
    if microseconds >= 1e3:
        return f"{microseconds / 1e3:,.3f} ms"
    return f"{microseconds:,.3f} µs"


def ratio_text(ratio):
    return "—" if ratio is None else f"{ratio:.3f}×"


def percent_text(ratio):
    if ratio is None:
        return "—"
    change = (ratio - 1) * 100
    return f"{change:+.1f}%".replace("-", "−")


def figure(svg, caption):
    """Preserve the plot's readable physical size on narrow screens."""
    width = re.search(r'<svg\b[^>]*\bwidth="([\d.]+)pt"', svg)
    if width is None:
        raise ValueError("A chart must declare its Matplotlib point width")
    minimum = math.ceil(float(width.group(1)) * 96 / 72)
    return (
        '<figure class="chart"><div class="chart-scroll">'
        f'<div class="chart-canvas" style="min-width:{minimum}px">{svg}</div></div>'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )


def case_rows(data):
    records = []
    for name, case in data["cases"].items():
        old = case["median_us"].get("published")
        current = case["median_us"]["current"]
        ratio = current / old if old is not None else None
        family = case_family(name, case)
        label = case_label(name)
        scope = case["comparison_scope"]
        note = case["manifest"].get("note", "")
        records.append(
            dict(
                id=name,
                label=label,
                family=family,
                old=old,
                current=current,
                ratio=ratio,
                limit=case.get("limit_us"),
                flag=bool(case.get("above_original_budget", False)),
                scope=scope,
                workers=case["worker_medians_us"],
                search=" ".join((name, label, family, note)).lower(),
            )
        )
    return records


def render_case_table(records):
    parts = []
    for item in records:
        ratio = item["ratio"]
        value_class = (
            "bad-value"
            if item["flag"]
            else ("good-value" if ratio is not None and ratio < 1 else "higher-value")
        )
        if ratio is None:
            status = '<span class="status status-new">Current only</span>'
        elif item["flag"]:
            status = '<span class="status status-flag">Above budget</span>'
        else:
            status = '<span class="status status-pass">Within budget</span>'
        workers = item["workers"]["current"]
        worker_range = f"{time_text(min(workers))}–{time_text(max(workers))}"
        scope = (
            "Added shape control" if item["scope"] == "added_shape" else item["family"]
        )
        parts.append(
            f'<tr data-case-id="{escape(item["id"], quote=True)}">'
            f'<td class="workload-cell"><span class="workload-name">{escape(item["label"])}</span>'
            f'<code class="case-id">{escape(item["id"])}</code>'
            f'<span class="case-kind">{escape(scope)}</span></td>'
            f'<td class="numeric">{time_text(item["old"])}</td>'
            f'<td class="numeric">{time_text(item["current"])}</td>'
            f'<td class="numeric {value_class}">{ratio_text(ratio)}'
            f"<br><small>{percent_text(ratio)}</small></td>"
            f'<td class="numeric">{time_text(item["limit"])}</td>'
            f'<td class="numeric">{escape(worker_range)}</td>'
            f"<td>{status}</td></tr>"
        )
    return "\n".join(parts)


def render_setup_table(data):
    rows = []
    for name, case in data["setup_cases"].items():
        old = case["median_us"]["published"]
        current = case["median_us"]["current"]
        rows.append(
            "<tr>"
            f'<td>{escape(case_label(name))}<code class="case-id">{escape(name)}</code></td>'
            f'<td class="numeric">{time_text(old)}</td>'
            f'<td class="numeric">{time_text(current)}</td>'
            f'<td class="numeric">{ratio_text(current / old)}</td>'
            f'<td class="numeric">{time_text(case["limit_us"])}</td>'
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th>Setup operation</th>'
        '<th class="numeric">PyPI</th><th class="numeric">Current</th>'
        '<th class="numeric">Current / PyPI</th><th class="numeric">Allowance</th>'
        "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></div>"
    )


def render_report(data, verification, data_hash):
    if verification["source_sha256"] != data["source_hashes"]["current"]:
        raise ValueError("Timing and correctness records identify different sources")
    if verification["performance"]["evidence_sha256"] != data_hash:
        raise ValueError(
            "The timing data differs from the verified comparison artifact"
        )
    charts = build_charts(data)
    records = case_rows(data)
    comparable = [item for item in records if item["ratio"] is not None]
    faster = sum(item["ratio"] < 1 for item in comparable)
    slower = sum(item["ratio"] > 1 for item in comparable)
    flags = sum(item["flag"] for item in comparable)
    counts = data["workload_counts"]
    assert len(comparable) == counts["comparable"] == 119
    assert flags == counts["original_budget_flags"]
    assert data["correctness"]["passed"]
    tests = verification["full_suite"]
    assert tests["verification_passed"]
    family_counts = Counter(item["family"] for item in comparable)
    family_flags = Counter(item["family"] for item in comparable if item["flag"])
    atlas = []
    for family, svg in charts["ratios"].items():
        total, failed = family_counts[family], family_flags[family]
        description = f"{total} cases · {total - failed} within budget"
        if failed:
            description += f" · {failed} above budget"
        atlas.append(
            "<details><summary>"
            + escape(family)
            + '<span class="summary-note">'
            + escape(description)
            + "</span></summary>"
            + figure(
                svg,
                "Current / PyPI time on a logarithmic scale. Lower values mean less elapsed time. "
                "Budget markers use each case's recorded allowance where visible.",
            )
            + "</details>"
        )
    family_options = "\n".join(
        f'<option value="{escape(name, quote=True)}">{escape(name)}</option>'
        for name in sorted({item["family"] for item in records})
    )
    current_only = [item for item in records if item["ratio"] is None]
    current_only_rows = "\n".join(
        "<tr>"
        f'<td>{escape(item["label"])}<code class="case-id">{escape(item["id"])}</code></td>'
        f'<td class="numeric">{time_text(item["current"])}</td>'
        "<td>Unavailable</td></tr>"
        for item in current_only
    )
    date = datetime.fromisoformat(data["date_utc"]).strftime("%d %B %Y").lstrip("0")
    full_passed = tests["counts"]["passed"]
    comparisons = tests["property_comparisons"]
    source = data["source_hashes"]["current"]
    release = data["baseline"]["latest_non_yanked_stable"]
    pg_cold = data["cold"]["profiles"]["pg_matched_compiled_HP_batch1024"]
    pg_cold_times = pg_cold["public"]["first_synchronized_public_call_us"]["median_us"]
    pg_cold_ratio = pg_cold_times["current"] / pg_cold_times["published"]
    pg_cold_limit = pg_cold["original_public_gate"]["limit_us"]
    runtime_caption = "Each batch duration covers all labelled states in that batch."
    payload = json.dumps(dict(cases=records), separators=(",", ":"), ensure_ascii=False)
    payload = (
        payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; base-uri 'none'">
<title>JaxProp performance — current vs PyPI {escape(release)}</title>
<style>{STYLE}</style>
</head>
<body>
<header>
  <div class="heading-row">
    <div>
      <p class="eyebrow">JaxProp · benchmark report</p>
      <h1>Performance review</h1>
      <p class="subtitle">Current development build vs PyPI {escape(release)}</p>
      <p class="meta">{date} · CPU / x64 · Warm measurements use 4 workers per version × 12 samples</p>
    </div>
    <div class="gate-status">{flags} warm workloads above budget</div>
  </div>
  <nav aria-label="Report sections">
    <a href="#overview">Overview</a><a href="#native">Perfect gas &amp; bicubic</a>
    <a href="#flags">Remaining flags</a><a href="#startup">First call &amp; setup</a>
    <a href="#all-workloads">All workloads</a><a href="#method">Method &amp; provenance</a>
  </nav>
</header>
<main>
  <div class="metrics" aria-label="Benchmark summary">
    <div class="metric"><p class="metric-label">Warm cases within budget</p><p class="metric-value">{len(comparable) - flags}<span style="color:var(--muted);font-weight:400"> / {len(comparable)}</span></p><p class="metric-note">{faster} lower medians · {slower} higher medians</p></div>
    <div class="metric"><p class="metric-label">Setup cases within budget</p><p class="metric-value">{counts['setup_comparable'] - counts['setup_flags']}<span style="color:var(--muted);font-weight:400"> / {counts['setup_comparable']}</span></p><p class="metric-note">Construction, table loading and generation</p></div>
    <div class="metric"><p class="metric-label">First calls within budget</p><p class="metric-value">{counts['cold_profiles'] - counts['cold_flags']}<span style="color:var(--muted);font-weight:400"> / {counts['cold_profiles']}</span></p><p class="metric-note">Fresh-process public-call measurements</p></div>
    <div class="metric"><p class="metric-label">Tests passed</p><p class="metric-value">{full_passed:,}</p><p class="metric-note">{comparisons:,} numerical comparisons passed</p></div>
  </div>

  <section id="overview">
    <div class="section-head"><h2>Performance across the library</h2><span class="section-count">111 original cases + 8 shape controls</span></div>
    <p class="section-note">Runtime direction and the declared allowance are shown separately. A higher median can fall within a case's budget.</p>
    <div class="panel">{figure(charts['overview'], 'Each case contributes one count.')}</div>
  </section>

  <section id="native">
    <div class="section-head"><h2>Perfect gas and bicubic</h2><span class="section-count">Actual full-state returns</span></div>
    <p class="section-note">Representative scalar and complete-batch calls from the matched-placement warm run. The full case atlas includes additional pairs, shapes, fluids and model parameters.</p>
    <div class="figure-grid">
      <div class="panel"><h3>Perfect gas</h3>{figure(charts['pg'], runtime_caption)}</div>
      <div class="panel"><h3>Bicubic</h3>{figure(charts['bicubic'], runtime_caption)}</div>
    </div>
  </section>

  <section id="flags">
    <div class="section-head"><h2>Five workloads above allowance</h2><span class="section-count">4 original cases + 1 shape control</span></div>
    <p class="section-note">Four compiled consumers and the added bicubic 8 × 16 shape control exceed their original 10% limits. Consumer measurements include the current status outputs.</p>
    <div class="panel">{figure(charts['flags'], 'Durations cover the complete consumer or batch. The added shape control covers 128 states.')}</div>
  </section>

  <section id="startup">
    <div class="section-head"><h2>First-call and setup costs</h2><span class="section-count">4 cold profiles · 22 setup operations</span></div>
    <p class="section-note">Cold measurements use four fresh unrestricted workers per profile. Warm and setup measurements use matched performance-core placement.</p>
    <div class="panel"><h3>First synchronized public call</h3>{figure(charts['cold'], 'Durations include tracing, lowering, compilation and first synchronized execution. The original allowance is 1.5 × PyPI + 100 ms.')}
      <p class="callout">PG's first-call median is {pg_cold_times['current'] / 1000:.2f} ms versus {pg_cold_times['published'] / 1000:.2f} ms for PyPI, about {pg_cold_ratio:.2f}× the elapsed time. Its recorded allowance is {pg_cold_limit / 1000:.2f} ms.</p>
    </div>
    <div class="panel" style="margin-top:18px"><h3>Construction and source initialization</h3>{figure(charts['setup_construction'], 'The original allowance is 1.25 × PyPI + 10 ms.')}</div>
    <div class="figure-grid" style="margin-top:18px">
      <div class="panel"><h3>Cached table loading</h3>{figure(charts['setup_cache'], 'The original allowance is 1.25 × PyPI + 100 ms.')}</div>
      <div class="panel"><h3>Fresh table generation</h3>{figure(charts['setup_generation'], 'The original allowance is 1.25 × PyPI + 0.1 s.')}</div>
    </div>
    <details><summary>Exact setup medians and allowances</summary>{render_setup_table(data)}</details>
  </section>

  <section id="atlas">
    <div class="section-head"><h2>All 119 timing ratios</h2><span class="section-count">Current time ÷ PyPI time</span></div>
    <p class="section-note">Open a family to inspect every comparable case. A ratio of 0.5× is half the elapsed time, 1× is parity and 2× is twice the elapsed time. The logarithmic axis spans the full measured range.</p>
    <div class="panel atlas">{''.join(atlas)}</div>
  </section>

  <section id="all-workloads">
    <div class="section-head"><h2>Exact measurements</h2><span class="section-count">119 comparable + {len(current_only)} current-only cases</span></div>
    <p class="section-note">Durations apply to one complete call, consumer or batch as labelled. The current worker range is the minimum–maximum of four worker medians.</p>
    <div class="panel">
      <div class="table-controls">
        <label for="case-search">Search workloads<input id="case-search" type="search" placeholder="Fluid, input pair, backend or case ID" autocomplete="off"></label>
        <label for="case-family">Family<select id="case-family"><option value="all">All families</option>{family_options}</select></label>
        <label for="case-status">Budget status<select id="case-status"><option value="all">All cases</option><option value="flag">Above budget</option><option value="pass">Within budget</option><option value="new">Current only</option></select></label>
        <label for="case-sort">Order<select id="case-sort"><option value="budget">Budget flags first</option><option value="slowest">Largest slowdown ratio</option><option value="fastest">Largest speedup ratio</option><option value="time">Longest current duration</option><option value="name">Workload name</option></select></label>
      </div>
      <div class="table-summary"><span id="case-count" aria-live="polite">{len(records)} workloads shown</span><button id="reset-filters" class="plain-button" type="button">Reset filters</button></div>
      <div class="table-scroll"><table id="case-table"><thead><tr><th>Workload</th><th class="numeric">PyPI {escape(release)}</th><th class="numeric">Current</th><th class="numeric">Current / PyPI</th><th class="numeric">Allowance</th><th class="numeric">Current worker range</th><th>Budget status</th></tr></thead><tbody id="case-body">{render_case_table(records)}</tbody></table></div>
      <p id="no-results" class="empty" hidden>No workloads match these filters.</p>
    </div>
    <div class="current-only panel">
      <h3>Current-only capabilities</h3>
      <p class="section-note">The recorded comparison provides a current timing for these four capabilities. A released-version timing is unavailable.</p>
      <div class="table-scroll"><table><thead><tr><th>Workload</th><th class="numeric">Current median</th><th>PyPI timing</th></tr></thead><tbody>{current_only_rows}</tbody></table></div>
    </div>
  </section>

  <section id="method">
    <div class="section-head"><h2>Method and provenance</h2></div>
    <div class="panel">
      <div class="method-grid">
        <div><h3>Warm measurement</h3><p>Four independent workers per version, 12 samples per case, synchronized actual returns and balanced case order. Windows worker affinity is 0xffff, covering performance-class logical processors 0–15. Scalar, batch and host workloads use their recorded fixed repetition counts.</p></div>
        <div><h3>Aggregation and spread</h3><p>Each headline duration is the median of four worker medians. Warm whiskers show the range of those four medians. Cold whiskers show the range of four fresh-process calls. These ranges describe observed spread.</p></div>
        <div><h3>Allowances</h3><p>Each case uses its saved limit. Most warm cases allow 10% overhead. The declared closed-model PG 1,024-state cohort allows 40%, and state inspection/construction include additive allowances. Setup and cold calls have their own recorded formulas. Eleven warm cases exceed a strict 10% increase, and five exceed their declared budgets.</p></div>
        <div><h3>Returned work and correctness</h3><p>Current PG full states materialize 22 properties, compared with 13 in the release. Current compiled consumers return their real status arrays. Benchmark comparisons passed, and the full HEOS/REFPROP suite passed {full_passed:,} tests with {comparisons:,} audited property comparisons.</p></div>
      </div>
      <div class="provenance">
        <h3>Recorded identities</h3>
        <dl>
          <dt>Current source SHA256</dt><dd><code>{escape(source)}</code></dd>
          <dt>PyPI source SHA256</dt><dd><code>{escape(data['source_hashes']['published'])}</code></dd>
          <dt>Wheel</dt><dd><code>{escape(data['baseline']['wheel_filename'])}</code></dd>
          <dt>Wheel SHA256</dt><dd><code>{escape(data['baseline']['wheel_sha256'])}</code></dd>
          <dt>Input data SHA256</dt><dd><code>{escape(data_hash)}</code></dd>
          <dt>Measurement records</dt><dd><a href="followup_validation.json">Timing data</a> · <a href="followup_verification.json">Correctness data</a> · <a href="followup_protocol.json">Protocol</a></dd>
        </dl>
        <p class="section-note">The earlier unrestricted C3 run recorded 17 timing flags. This report uses the later declared matched-placement warm run and its separate unrestricted cold cohort. Core-context and provider-history observations are documented in the attribution record. <a href="mixed_attribution_20261115.md">Attribution record</a></p>
      </div>
    </div>
  </section>
</main>
<footer>Saved measurements from 15 November 2026 · Full-state and consumer durations · Phase 14 performance acceptance remains open for five warm cases</footer>
<script id="report-data" type="application/json">{payload}</script>
<script>{SCRIPT}</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=SUPPORT / "followup_validation.json"
    )
    parser.add_argument(
        "--verification", type=Path, default=SUPPORT / "followup_verification.json"
    )
    parser.add_argument("--output", type=Path, default=SUPPORT / "timing_report.html")
    args = parser.parse_args()
    raw = args.data.read_bytes()
    data = json.loads(raw)
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    report = render_report(data, verification, hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            dict(
                report=str(args.output.resolve()),
                bytes=args.output.stat().st_size,
                comparable_cases=data["workload_counts"]["comparable"],
                budget_flags=len(data["timing_flags"]),
            )
        )
    )


if __name__ == "__main__":
    main()
