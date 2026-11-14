"""Matched public-API performance comparison against a pinned published wheel.

Run ``--validate-only`` first, then the default interleaved four-worker guard.
Every worker imports exactly one implementation with identical dependencies,
uses independent table/compiler caches, and synchronizes complete return values.
The retained acceptance manifest is frozen before acceptance timing begins.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import zipfile
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = Path(__file__).resolve().parent / "phase13"
CORE = ("pressure", "temperature", "density", "enthalpy", "entropy", "internal_energy")
PG_FIELDS = (
    "pressure",
    "temperature",
    "density",
    "enthalpy",
    "entropy",
    "viscosity",
    "conductivity",
    "speed_of_sound",
    "heat_capacity_ratio",
    "isobaric_heat_capacity",
    "isochoric_heat_capacity",
    "compressibility_factor",
    "gruneisen",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted((Path(root) / "jaxprop").rglob("*.py")):
        digest.update(
            path.relative_to(root).as_posix().encode()
            + b"\0"
            + path.read_bytes()
            + b"\0"
        )
    return digest.hexdigest()


def harness_hash():
    digest = hashlib.sha256()
    for path in [
        Path(__file__),
        *sorted(SUPPORT.glob("*.py")),
        SUPPORT / "acceptance.json",
    ]:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


class Context:
    """Register full-return workloads and keep validation outside timed regions."""

    def __init__(self, args, jp, CP, jax, jnp, np, eqx, api):
        self.args, self.jp, self.CP = args, jp, CP
        self.jax, self.jnp, self.np, self.eqx, self.api = jax, jnp, np, eqx, api
        self.directory = args.record.resolve().parent
        self.directory.mkdir(parents=True, exist_ok=True)
        self.models, self.cases, self.checks, self.skipped = {}, [], {}, {}
        self.validators = []
        self.setups, self.first_calls, self.snapshots, self.graphs = {}, {}, {}, {}
        self.payloads, self.timings, self.unique_timings = {}, {}, {}
        self.acceptance = read(SUPPORT / "acceptance.json")

    def synchronized(self, value):
        return self.jax.block_until_ready(value)

    def setup(self, name, factory, repeats=3):
        samples, first = [], None
        for index in range(repeats):
            start = time.perf_counter_ns()
            value = factory()
            self.synchronized(self.jax.tree_util.tree_leaves(value))
            samples.append((time.perf_counter_ns() - start) / 1e3)
            if index == 0:
                first = value
        self.setups[name] = dict(
            first_microseconds=samples[0],
            subsequent_microseconds=samples[1:],
            samples_microseconds=samples,
            median_microseconds=statistics.median(samples),
        )
        return first

    def check(self, name, value):
        self.checks[name] = clean(value, self.np)

    def skip(self, name, reason):
        self.skipped[name] = str(reason)

    def validate(self, name, callback):
        """Defer model-consuming independent checks until first-call capture."""
        self.validators.append((name, callback))

    def add(
        self,
        name,
        fn,
        variants,
        calls=32,
        fields=None,
        compiled=None,
        category="full_state",
        budget=1.10,
        note="",
        rtol=1e-7,
        atol=1e-7,
        comparable=True,
        **extra,
    ):
        if any(case["name"] == name for case in self.cases):
            raise ValueError(f"Duplicate workload {name}")
        self.cases.append(
            dict(
                name=name,
                fn=fn,
                variants=list(variants),
                calls=max(1, round(calls * self.args.calls_scale)),
                fields=tuple(fields) if fields is not None else CORE,
                compiled=compiled,
                category=category,
                budget=budget,
                note=note,
                rtol=rtol,
                atol=atol,
                comparable=comparable,
                **extra,
            )
        )

    def array_snapshot(self, value):
        array = self.np.asarray(value)
        if array.dtype.kind == "V":
            return {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "float0": True,
            }
        if array.dtype.kind not in "biufc":
            raise TypeError(f"Unsupported numeric result dtype {array.dtype}")
        assert not self.np.any(self.np.isinf(array)), "Infinite numerical result"
        flat = array.ravel()
        indices = self.np.unique(
            self.np.linspace(0, max(0, flat.size - 1), min(16, flat.size), dtype=int)
        )
        return dict(
            shape=list(array.shape),
            dtype=str(array.dtype),
            size=int(array.size),
            sample_indices=indices.tolist(),
            sample=[
                None if self.np.isnan(item) else float(item) for item in flat[indices]
            ],
            finite=int(self.np.count_nonzero(self.np.isfinite(array))),
            unavailable=int(self.np.count_nonzero(self.np.isnan(array))),
            values=[None if self.np.isnan(item) else float(item) for item in flat],
            unavailable_mask_sha256=hashlib.sha256(
                self.np.isnan(array).tobytes()
            ).hexdigest(),
        )

    def snapshot(self, value, fields, *, name="result"):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (set, frozenset)):
            return [
                self.snapshot(item, fields, name=name)
                for item in sorted(value, key=str)
            ]
        if isinstance(value, self.jp.FluidState):
            values = {
                field: self.array_snapshot(self.api.raw(value, field))
                for field in fields
            }
            status = self.api.status(value)
            if status is not None:
                actual = self.np.asarray(status)
                if actual.dtype.kind != "V":
                    assert self.np.all(actual == 0), f"{name}: unsuccessful state"
            raw = self.api.dictionary(value)
            if name not in self.payloads:
                self.payloads[name] = dict(
                    property_shapes={
                        key: list(self.np.shape(item)) for key, item in raw.items()
                    },
                    numeric_properties=len(raw),
                    all_finite_properties=[
                        key
                        for key, item in raw.items()
                        if self.np.asarray(item).dtype.kind != "V"
                        and self.np.all(self.np.isfinite(self.np.asarray(item)))
                    ],
                    numerical_leaf_count=len(self.jax.tree_util.tree_leaves(value)),
                    status_present=status is not None,
                )
            return dict(
                state=values,
                status=self.array_snapshot(status) if status is not None else None,
            )
        if isinstance(value, dict):
            selected = (
                fields if any(field in value for field in fields) else tuple(value)
            )
            result = {}
            for key in selected:
                if key not in value:
                    continue
                item = value[key]
                if str(key).startswith("status"):
                    if item is not None:
                        assert self.np.all(self.np.asarray(item) == 0), name
                    self.payloads[name + "." + str(key)] = dict(
                        status_present=item is not None,
                        shape=list(self.np.shape(item)) if item is not None else None,
                        all_success=True if item is not None else None,
                    )
                    result[key] = (
                        None  # Presence is recorded by state payloads; no fabricated legacy status.
                    )
                else:
                    result[key] = self.snapshot(
                        item, fields, name=name + "." + str(key)
                    )
            return result
        if isinstance(value, (tuple, list)):
            return [
                self.snapshot(item, fields, name=name + "." + str(index))
                for index, item in enumerate(value)
            ]
        return self.array_snapshot(value)

    def execute(self):
        def prepare(case):
            name, fn, variants = case["name"], case["fn"], case["variants"]
            print("Validating " + name, flush=True)
            start = time.perf_counter_ns()
            initial = self.synchronized(fn(*variants[0]))
            self.first_calls[name] = (time.perf_counter_ns() - start) / 1e3
            self.snapshots[name] = [self.snapshot(initial, case["fields"], name=name)]
            for inputs in variants[1:]:
                self.snapshots[name].append(
                    self.snapshot(
                        self.synchronized(fn(*inputs)), case["fields"], name=name
                    )
                )
            if case["compiled"] is not None:
                graph = case["compiled"].lower(*variants[0]).compile().as_text()
                normalized = re.sub(r"metadata=\{[^{}]*\}", "", graph)
                normalized = "\n".join(
                    line.strip() for line in normalized.splitlines()[1:] if line.strip()
                )
                self.graphs[name] = dict(
                    normalized_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                    instructions=len(re.findall(r"^\s*%[^\n]*=", graph, re.MULTILINE)),
                    contains_callback="callback" in graph.lower(),
                )
                graph_path = self.directory / (name + ".hlo.txt")
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                graph_path.write_text(graph, encoding="utf-8")
            warm_start = time.perf_counter_ns()
            for index in range(self.acceptance["warmup_calls"]):
                self.synchronized(fn(*variants[index % len(variants)]))
            warm_microseconds = (
                (time.perf_counter_ns() - warm_start)
                / self.acceptance["warmup_calls"]
                / 1e3
            )
            case["requested_calls"] = case["calls"]
            case["calls"] = min(
                case["calls"],
                max(
                    1,
                    math.ceil(
                        self.acceptance["maximum_target_sample_microseconds"]
                        / max(warm_microseconds, 1e-6)
                    ),
                ),
            )

        for case in self.cases:
            prepare(case)
        # State-helper inputs require property evaluations. Construct them only
        # after the thermodynamic first calls have been captured.
        if not self.args.case or any("state" in pattern for pattern in self.args.case):
            previous = len(self.cases)
            state_cases(self)
            for case in self.cases[previous:]:
                if not self.args.case or any(
                    re.search(pattern, case["name"]) for pattern in self.args.case
                ):
                    prepare(case)
            if self.args.case:
                self.cases = [
                    case
                    for case in self.cases
                    if any(
                        re.search(pattern, case["name"]) for pattern in self.args.case
                    )
                ]
        for name, callback in self.validators:
            result = callback()
            if result is not None:
                self.check(name, result)
        if self.args.validate_only:
            return
        samples = {case["name"]: [] for case in self.cases}
        # Rotate starting cases to avoid giving one model a fixed warm/late slot.
        for sample in range(self.args.samples):
            offset = sample % len(self.cases)
            for case in self.cases[offset:] + self.cases[:offset]:
                fn, variants = case["fn"], case["variants"]
                start = time.perf_counter_ns()
                for call in range(case["calls"]):
                    self.synchronized(fn(*variants[(sample + call) % len(variants)]))
                elapsed = (time.perf_counter_ns() - start) / case["calls"] / 1e3
                samples[case["name"]].append(elapsed)
        for case in self.cases:
            values = samples[case["name"]]
            self.timings[case["name"]] = dict(
                microseconds_per_call=values,
                median=statistics.median(values),
                calls_per_sample=case["calls"],
            )

    def unique_python_numbers(self):
        """Measure genuine numeric specialization without prewarming timed values."""
        for label in ("pg", "bicubic"):
            if label not in self.models:
                continue
            model = self.models[label]

            def evaluate(h, p):
                return self.api.evaluate(model, self.jp.HmassP_INPUTS, h, p)

            # These values are outside the timed sequence for every worker.
            for h in (689731.125, 689733.125):
                self.synchronized(evaluate(h, 7999723.125))
            events = []
            capture = [False]

            class CompilerEvents(logging.Handler):
                def emit(self, record):
                    message = record.getMessage()
                    if capture[0] and message.startswith("Compiling "):
                        events.append(message)

            handler = CompilerEvents()
            logger = logging.getLogger("jax")
            logger.addHandler(handler)
            values, inputs, snapshots, per_call_events = [], [], [], []
            try:
                with self.jax.log_compiles(True):
                    for index in range(12):
                        # No value in this sequence has appeared in any other case.
                        h = float(692413.375 + index * 13.125)
                        p = float(8000711.875 + index * 17.375)
                        event_start = len(events)
                        capture[0] = True
                        start = time.perf_counter_ns()
                        state = self.synchronized(evaluate(h, p))
                        values.append((time.perf_counter_ns() - start) / 1e3)
                        capture[0] = False
                        per_call_events.append(len(events) - event_start)
                        inputs.append(dict(enthalpy=h, pressure=p))
                        snapshots.append(
                            self.snapshot(
                                state,
                                PG_FIELDS if label == "pg" else CORE,
                                name=label + ".unique_python",
                            )
                        )
            finally:
                logger.removeHandler(handler)
            self.unique_timings[label] = dict(
                microseconds_per_call=values,
                median=statistics.median(values),
                numbers_previously_seen=False,
                calls=len(values),
                compiler_events=len(events),
                compiler_messages=events,
                per_call_compiler_events=per_call_events,
                inputs=inputs,
                snapshots=snapshots,
                compiler_events_scope="Only evaluate+complete-return synchronization; logging enabled during these diagnostic calls, snapshot/export events excluded.",
                includes_compilation=True,
                not_a_warmed_arithmetic_speedup=True,
            )


def clean(value, np):
    if isinstance(value, dict):
        return {str(key): clean(item, np) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item, np) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if hasattr(value, "tolist"):
        return clean(value.tolist(), np)
    raise TypeError(type(value))


def state_cases(ctx):
    """Measure actual public state helpers, separately from thermodynamic calls."""
    jp, jnp, api = ctx.jp, ctx.jnp, ctx.api
    model = ctx.models["pg"]
    first = api.evaluate(model, jp.PT_INPUTS, jnp.asarray(1e5), jnp.asarray(500.0))
    second = api.evaluate(model, jp.PT_INPUTS, jnp.asarray(1.1e5), jnp.asarray(510.0))
    batch = api.evaluate(
        model, jp.PT_INPUTS, jnp.linspace(1e5, 1.2e5, 16), jnp.linspace(500, 520, 16)
    )
    ctx.add(
        "state_alias_access",
        lambda state: state.rho,
        [(first,), (second,)],
        calls=256,
        category="state_inspection",
    )
    ctx.add(
        "state_canonical_access",
        lambda state: state["density"],
        [(first,), (second,)],
        calls=256,
        category="state_inspection",
    )
    ctx.add(
        "state_actual_dictionary_export",
        api.dictionary,
        [(first,), (second,)],
        calls=64,
        fields=PG_FIELDS,
        category="state_inspection",
        note="Actual public export schemas differ; all actual exported leaves are returned/synchronized, common13 values checked.",
    )
    ctx.add(
        "state_common13_dictionary",
        lambda state: {name: api.raw(state, name) for name in PG_FIELDS},
        [(first,), (second,)],
        calls=64,
        fields=PG_FIELDS,
        category="state_inspection",
        note="Secondary common-property diagnostic.",
    )
    ctx.add(
        "state_index_batch16",
        lambda state: state.at_index(7),
        [(batch,)],
        calls=32,
        fields=PG_FIELDS,
        category="state_inspection",
    )
    ctx.add(
        "state_stack_two_scalars",
        lambda a, b: jp.FluidState.stack([a, b]),
        [(first, second)],
        calls=16,
        fields=PG_FIELDS,
        category="state_inspection",
    )
    ctx.add(
        "state_concatenate_two_batches",
        lambda a, b: a + b,
        [(batch, batch)],
        calls=16,
        fields=PG_FIELDS,
        category="state_inspection",
    )
    data = {name: api.raw(first, name) for name in PG_FIELDS}
    batch_data = {name: api.raw(batch, name) for name in PG_FIELDS}
    constructor = lambda properties: jp.FluidState(
        fluid_name="benchmark", identifier="benchmark", **properties
    )
    ctx.add(
        "state_public_construction_scalar",
        constructor,
        [(data,)],
        calls=16,
        fields=PG_FIELDS,
        category="state_construction",
    )
    ctx.add(
        "state_public_construction_batch16",
        constructor,
        [(batch_data,)],
        calls=8,
        fields=PG_FIELDS,
        category="state_construction",
    )
    if not api.legacy:
        ctx.add(
            "state_lossless_status_validity_export",
            lambda state: state.to_data(),
            [(batch,)],
            calls=16,
            fields=PG_FIELDS,
            category="state_inspection",
            comparable=False,
            note="New-only metadata/status/validity serialization; no invented legacy equivalent.",
        )
    else:
        ctx.skip(
            "state_lossless_status_validity_export",
            "Published package has no equivalent lossless metadata/status/validity export.",
        )


def worker(args):
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ.update(
        JAX_ENABLE_X64="true", JAX_PLATFORM_NAME="cpu", MPLBACKEND="Agg", PYTHONUTF8="1"
    )
    cache_directory = args.record.resolve().parent / ("compiler_cache_" + uuid4().hex)
    assert not cache_directory.exists()
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_directory)
    root = args.source.resolve()
    sys.path.insert(0, str(root))
    import CoolProp.CoolProp as CP
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jaxprop as jp
    import numpy as np
    from phase13.api import API
    from phase13 import native_cases, source_cases

    assert Path(jp.__file__).resolve().parent == root / "jaxprop"
    assert jax.default_backend() == "cpu" and jax.config.x64_enabled
    before, before_harness = source_hash(root), harness_hash()
    jax.block_until_ready(
        jnp.asarray(0.0)
    )  # Exclude device startup from setup observations.
    ctx = Context(
        args,
        jp,
        CP,
        jax,
        jnp,
        np,
        eqx,
        API(jp, args.version == "baseline", CP, jax, jnp, np, eqx),
    )
    native_cases.register(ctx)
    source_cases.register(ctx)
    if args.case:
        ctx.cases = [
            case
            for case in ctx.cases
            if any(re.search(pattern, case["name"]) for pattern in args.case)
        ]
    assert ctx.cases or (args.case and any("state" in pattern for pattern in args.case))
    ctx.execute()
    if not args.no_unique and not args.case:
        ctx.unique_python_numbers()
    after = source_hash(root)
    assert after == before and harness_hash() == before_harness
    manifest = [
        {
            key: value
            for key, value in case.items()
            if key not in {"fn", "variants", "compiled"}
        }
        for case in ctx.cases
    ]
    environment = {
        name: importlib.metadata.version(name)
        for name in (
            "jax",
            "jaxlib",
            "numpy",
            "scipy",
            "CoolProp",
            "equinox",
            "optimistix",
            "diffrax",
            "pysolver_view",
        )
    }
    environment.update(
        python=sys.version,
        executable=sys.executable,
        device=str(jax.devices()[0]),
        x64=jax.config.x64_enabled,
        threads={
            key: os.environ[key]
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    )
    record = dict(
        date_utc=datetime.now(timezone.utc).isoformat(),
        version=args.version,
        imported_package=str(Path(jp.__file__).resolve()),
        source_sha256=before,
        source_unchanged=True,
        harness_sha256=before_harness,
        environment=environment,
        manifest=manifest,
        setup=ctx.setups,
        first_calls_microseconds=ctx.first_calls,
        snapshots=ctx.snapshots,
        payloads=ctx.payloads,
        independent_checks=ctx.checks,
        skipped=ctx.skipped,
        timings=ctx.timings,
        unique_python_numbers=ctx.unique_timings,
        graphs=ctx.graphs,
        validate_only=args.validate_only,
        compilation_cache_directory=str(cache_directory),
        compilation_cache_was_fresh=True,
    )
    save(args.record, record)
    print(
        json.dumps(
            dict(
                record=str(args.record),
                cases=len(ctx.cases),
                new_only=sum(not case["comparable"] for case in ctx.cases),
                skipped=len(ctx.skipped),
                validate_only=args.validate_only,
            )
        ),
        flush=True,
    )


def compare_snapshots(old, new, case):
    """Compare sampled physical values, retaining declared unavailable outcomes."""
    import numpy as np

    failures, counts = [], {
        "finite": 0,
        "paired_unavailable": 0,
        "broadcast_constant_fields": 0,
    }
    field_atol = read(SUPPORT / "acceptance.json")["correctness"]["field_atol"]

    def visit(a, b, path):
        if isinstance(a, dict) and "sample" in a:
            if not isinstance(b, dict) or "sample" not in b:
                failures.append(
                    dict(path=path, reason="Different numerical result structure")
                )
                return
            if a["dtype"] != b["dtype"]:
                failures.append(
                    dict(
                        path=path,
                        reason="Different numerical dtypes",
                        baseline=a["dtype"],
                        current=b["dtype"],
                    )
                )
                return
            x = np.asarray([np.nan if v is None else v for v in a["values"]]).reshape(
                a["shape"]
            )
            y = np.asarray([np.nan if v is None else v for v in b["values"]]).reshape(
                b["shape"]
            )
            if x.shape != y.shape:
                # Published PG leaves retain independent input-coordinate axes.
                # Only its declared physical broadcast cases permit expansion;
                # arbitrary unequal state lengths can never pass this gate.
                allowed = "pg_" in case["name"] and (
                    "broadcast" in case["name"] or "scalar_pressure" in case["name"]
                )
                if allowed:
                    try:
                        x, y = np.broadcast_arrays(x, y)
                    except ValueError:
                        pass
                    else:
                        counts["broadcast_constant_fields"] += 1
            if x.shape != y.shape:
                failures.append(
                    dict(
                        path=path,
                        reason="Different physical array shapes",
                        baseline=a["shape"],
                        current=b["shape"],
                    )
                )
                return
            same_missing = np.isnan(x) == np.isnan(y)
            field = path.rsplit(".", 1)[-1]
            tolerance = (
                case["atol"]
                if "deriv" in path or "tangent" in path
                else field_atol.get(field, case["atol"])
            )
            finite = np.isfinite(x) & np.isfinite(y)
            expected_unavailable = case.get("expected_unavailable", ())
            if np.any(np.isnan(x) | np.isnan(y)) and field not in expected_unavailable:
                failures.append(
                    dict(
                        path=path,
                        reason="Undeclared unavailable result in selected physical/derivative fields",
                    )
                )
            counts["finite"] += int(np.count_nonzero(finite))
            counts["paired_unavailable"] += int(
                np.count_nonzero(np.isnan(x) & np.isnan(y))
            )
            if not np.all(same_missing) or not np.allclose(
                x[finite], y[finite], rtol=case["rtol"], atol=tolerance
            ):
                failures.append(
                    dict(
                        path=path,
                        baseline=a["sample"],
                        current=b["sample"],
                        rtol=case["rtol"],
                        atol=tolerance,
                    )
                )
            return
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                failures.append(
                    dict(
                        path=path,
                        reason="Different result keys",
                        baseline=list(a),
                        current=list(b) if isinstance(b, dict) else str(type(b)),
                    )
                )
                return
            for key in a:
                if key == "status":
                    continue
                visit(a[key], b[key], path + "." + key)
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                failures.append(
                    dict(path=path, reason="Different result sequence lengths")
                )
                return
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + "." + str(index))
        elif a != b:
            failures.append(dict(path=path, baseline=a, current=b))

    visit(old, new, case["name"])
    return dict(**counts, mismatches=failures)


def category_limit(case, old, acceptance):
    category = case["category"]
    if category in ("state_inspection", "state_construction"):
        budget = acceptance["budgets"][category]
        extra = (
            budget.get(
                "batch_additional_microseconds", budget["additional_microseconds"]
            )
            if "batch" in case["name"]
            else budget["additional_microseconds"]
        )
        return budget["ratio"] * old + extra
    return case["budget"] * old


def controller(args):
    baseline_manifest = read(args.baseline_manifest)
    wheel = args.baseline_manifest.parent / baseline_manifest["wheel_filename"]
    assert (
        hashlib.sha256(wheel.read_bytes()).hexdigest()
        == baseline_manifest["wheel_sha256"]
    )
    # Verify installed package bytes against the immutable official artifact,
    # not just a source hash that could describe an altered extraction.
    with zipfile.ZipFile(wheel) as archive:
        names = {
            name
            for name in archive.namelist()
            if name.startswith("jaxprop/") and name.endswith(".py")
        }
        actual = {
            path.relative_to(args.baseline_source).as_posix()
            for path in (args.baseline_source / "jaxprop").rglob("*.py")
        }
        assert (
            actual == names
        ), "Installed published package file set differs from wheel"
        for name in names:
            assert (args.baseline_source / name).read_bytes() == archive.read(
                name
            ), name
    area = args.output_directory.resolve()
    area.mkdir(parents=True, exist_ok=True)
    sources = dict(
        baseline=args.baseline_source.resolve(), current=args.current_source.resolve()
    )
    frozen = {name: source_hash(path) for name, path in sources.items()}
    acceptance = read(SUPPORT / "acceptance.json")
    frozen_harness = harness_hash()
    save(
        area / "frozen_run_manifest.json",
        dict(
            date_utc=datetime.now(timezone.utc).isoformat(),
            baseline=baseline_manifest,
            source_hashes=frozen,
            harness_sha256=frozen_harness,
            acceptance=acceptance,
            validate_only=args.validate_only,
        ),
    )
    order = (
        ("baseline", "current") if args.validate_only else tuple(acceptance["order"])
    )
    if not args.validate_only and args.workers != 4:
        order = tuple(
            version for _ in range(args.workers) for version in ("baseline", "current")
        )
    counts, records = dict(baseline=0, current=0), {}
    for version in order:
        counts[version] += 1
        name = version + str(counts[version])
        folder = area / "workers" / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "record.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--version",
            version,
            "--source",
            str(sources[version]),
            "--record",
            str(path),
            "--samples",
            str(args.samples),
            "--calls-scale",
            str(args.calls_scale),
        ]
        if args.validate_only:
            command.append("--validate-only")
        if args.no_unique:
            command.append("--no-unique")
        for pattern in args.case or ():
            command += ["--case", pattern]
        print("Starting " + name, flush=True)
        started = time.perf_counter()
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=folder,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {folder / 'worker.log'}")
        record = read(path)
        assert (
            record["source_sha256"] == frozen[version]
            and record["harness_sha256"] == frozen_harness
        )
        records[name] = record
        print(f"{name} completed in {time.perf_counter() - started:.1f}s", flush=True)
    reference = records["baseline1"]
    for name, record in records.items():
        assert (
            record["environment"] == reference["environment"]
        ), f"Unmatched numerical/interpreter/device environment in {name}"
    reference_manifest = {case["name"]: case for case in reference["manifest"]}
    correctness = {}
    for worker_name, record in records.items():
        manifest = {case["name"]: case for case in record["manifest"]}
        current_results = {}
        for name, case in reference_manifest.items():
            if not case["comparable"]:
                continue
            if name not in manifest:
                current_results[name] = dict(
                    mismatches=[dict(reason="Comparable case missing")]
                )
                continue
            current_results[name] = compare_snapshots(
                reference["snapshots"][name], record["snapshots"][name], case
            )
        for label, original in reference["unique_python_numbers"].items():
            actual = record["unique_python_numbers"][label]
            assert original["inputs"] == actual["inputs"]
            case = dict(name=label + "_unseen_python_inputs", rtol=1e-7, atol=1e-7)
            current_results[case["name"]] = compare_snapshots(
                original["snapshots"], actual["snapshots"], case
            )
        correctness[worker_name] = current_results
    failures = {
        worker_name: {
            case: result["mismatches"]
            for case, result in values.items()
            if result["mismatches"]
        }
        for worker_name, values in correctness.items()
    }
    failures = {key: value for key, value in failures.items() if value}
    save(area / "correctness.json", dict(results=correctness, failures=failures))
    if failures:
        raise RuntimeError(
            f"Cross-version numerical differences require classification; inspect {area / 'correctness.json'}"
        )
    cold_compilation = {}
    if not args.validate_only and not args.case and args.cold_compilation:
        cold_cases = (
            "pg_matched_compiled_HP_batch1024",
            "bicubic_CO2_compiled_HP_batch1024",
            "pg_matched_compiled_heat_exchanger_128cells",
            "bicubic_CO2_compiled_compressor",
        )
        for case_number, case_name in enumerate(cold_cases):
            observations = dict(baseline=[], current=[])
            for sequence, version in enumerate(
                ("baseline", "current", "current", "baseline")
            ):
                # Atomic table temporary names include the full cache basename;
                # short controller paths avoid the host Windows path ceiling.
                folder = area / "cold" / f"{case_number}_{sequence}_{version[0]}"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / "record.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--version",
                    version,
                    "--source",
                    str(sources[version]),
                    "--record",
                    str(path),
                    "--validate-only",
                    "--no-unique",
                    "--case",
                    "^" + re.escape(case_name) + "$",
                ]
                print(f"Cold public compilation {case_name}: {version}", flush=True)
                with (folder / "worker.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command,
                        cwd=folder,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                if result.returncode:
                    raise RuntimeError(
                        f"Cold worker failed; inspect {folder / 'worker.log'}"
                    )
                record = read(path)
                assert (
                    record["source_sha256"] == frozen[version]
                    and record["harness_sha256"] == frozen_harness
                )
                assert record["environment"] == reference["environment"]
                comparison = compare_snapshots(
                    reference["snapshots"][case_name],
                    record["snapshots"][case_name],
                    reference_manifest[case_name],
                )
                assert not comparison["mismatches"], comparison
                observations[version].append(
                    record["first_calls_microseconds"][case_name]
                )
            old, new = (statistics.median(observations[version]) for version in sources)
            budget = acceptance["budgets"]["first_public_call"]
            limit = old * budget["ratio"] + budget["additional_microseconds"]
            cold_compilation[case_name] = dict(
                observations_microseconds=observations,
                baseline_median_us=old,
                current_median_us=new,
                ratio=new / old,
                limit_microseconds=limit,
                above_budget=new > limit,
                condition="Separate fresh process and cache for one selected public workload; device startup excluded, no other thermodynamic case evaluated before capture.",
            )
    cases = {}
    if not args.validate_only:
        for name, case in reference_manifest.items():
            if not case["comparable"]:
                continue
            groups = {
                version: [
                    record["timings"][name]["median"]
                    for worker_name, record in records.items()
                    if worker_name.startswith(version)
                ]
                for version in sources
            }
            old, new = statistics.median(groups["baseline"]), statistics.median(
                groups["current"]
            )
            limit = category_limit(case, old, acceptance)
            cases[name] = dict(
                **case,
                baseline_worker_medians_us=groups["baseline"],
                current_worker_medians_us=groups["current"],
                baseline_median_us=old,
                current_median_us=new,
                ratio=new / old,
                limit_microseconds=limit,
                above_budget=new > limit,
                above_strict_10_percent=new > 1.1 * old,
                paired_worker_ratios=[
                    b / a for a, b in zip(groups["baseline"], groups["current"])
                ],
                raw_samples_us={
                    worker_name: record["timings"][name]["microseconds_per_call"]
                    for worker_name, record in records.items()
                },
                calls_per_sample_by_worker={
                    worker_name: record["timings"][name]["calls_per_sample"]
                    for worker_name, record in records.items()
                },
            )
    setup_summary = {}
    for name in reference["setup"]:
        if not all(name in record["setup"] for record in records.values()):
            continue
        groups = {
            version: [
                record["setup"][name]["median_microseconds"]
                for worker_name, record in records.items()
                if worker_name.startswith(version)
            ]
            for version in sources
        }
        old, new = (statistics.median(groups[version]) for version in sources)
        category = (
            "fresh_generation"
            if "fresh" in name or "generation" in name
            else (
                "cached_loading"
                if "cached" in name or "loading" in name
                else "construction"
            )
        )
        budget = acceptance["budgets"][category]
        limit = old * budget["ratio"] + budget["additional_microseconds"]
        setup_summary[name] = dict(
            category=category,
            baseline_worker_medians_us=groups["baseline"],
            current_worker_medians_us=groups["current"],
            baseline_median_us=old,
            current_median_us=new,
            ratio=new / old,
            limit_microseconds=limit,
            above_budget=new > limit,
            samples={
                worker_name: record["setup"][name]
                for worker_name, record in records.items()
            },
        )
    current_manifest = {case["name"]: case for case in records["current1"]["manifest"]}
    capability_differences = {
        name: dict(
            current_case=case,
            baseline_reason=reference["skipped"].get(
                name, "No equivalent published public evaluator/request."
            ),
        )
        for name, case in current_manifest.items()
        if name not in reference_manifest or not case["comparable"]
    }
    summary = dict(
        date_utc=datetime.now(timezone.utc).isoformat(),
        baseline=baseline_manifest,
        source_hashes=frozen,
        harness_sha256=frozen_harness,
        acceptance=acceptance,
        validate_only=args.validate_only,
        environment={name: record["environment"] for name, record in records.items()},
        comparison_counts=dict(
            finite=sum(
                result.get("finite", 0)
                for name, values in correctness.items()
                if name.startswith("current")
                for result in values.values()
            ),
            paired_unavailable=sum(
                result.get("paired_unavailable", 0)
                for name, values in correctness.items()
                if name.startswith("current")
                for result in values.values()
            ),
            mismatches=0,
        ),
        cases=cases,
        setup_summary=setup_summary,
        capability_differences=capability_differences,
        cold_public_compilation=cold_compilation,
        first_public_call_conditions="First encounter of each wrapper in a shared worker; related inner kernels may already be warm. Cold compilation acceptance is measured separately in isolated case workers.",
        baseline_package_bytes_verified_against_wheel=True,
        all_worker_environments_identical=True,
        setup={name: record["setup"] for name, record in records.items()},
        first_public_calls_microseconds={
            name: record["first_calls_microseconds"] for name, record in records.items()
        },
        unique_python_numbers={
            name: record["unique_python_numbers"] for name, record in records.items()
        },
        new_only={
            name: [case for case in record["manifest"] if not case["comparable"]]
            for name, record in records.items()
        },
        skipped={name: record["skipped"] for name, record in records.items()},
        payloads={name: record["payloads"] for name, record in records.items()},
        independent_checks={
            name: record["independent_checks"] for name, record in records.items()
        },
        graphs={name: record["graphs"] for name, record in records.items()},
        worker_records={
            name: str(area / "workers" / name / "record.json") for name in records
        },
    )
    save(area / "summary.json", summary)
    print(
        json.dumps(
            dict(
                summary=str(area / "summary.json"),
                comparable_cases=len(reference_manifest),
                numerically_valid=True,
                regressions=[
                    name for name, case in cases.items() if case["above_budget"]
                ],
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=SUPPORT / "baseline" / "baseline_manifest.json",
    )
    parser.add_argument(
        "--baseline-source",
        type=Path,
        default=ROOT / "development/performance/results/phase13/published",
    )
    parser.add_argument("--current-source", type=Path, default=ROOT)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--calls-scale", type=float, default=1)
    parser.add_argument(
        "--case", action="append", help="Regex filter for targeted follow-up workloads."
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-unique", action="store_true")
    parser.add_argument(
        "--cold-compilation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure selected true-cold public compilations in separate fresh processes after the warmed guard.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--version", choices=("baseline", "current"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.workers < 1 or args.samples < 1 or args.calls_scale <= 0:
        parser.error("workers, samples and calls-scale must be positive")
    if args.worker:
        worker(args)
    elif args.output_directory is None:
        parser.error("--output-directory is required")
    else:
        controller(args)


if __name__ == "__main__":
    main()
