"""Three-source Phase 14 experiments with fixed matched public-call repetitions.

This runner reuses Phase 13's public API translations, workload definitions,
independent oracles and legacy comparison rules. It adds complete candidate to
frozen-state comparisons, fixed repetitions, matched shape controls and explicit
consumer-status diagnostics. It never modifies or times a patched PyPI package.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import uuid
import zipfile

import phase13_benchmark as phase13

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = Path(__file__).resolve().parent / "phase14"
PROTOCOL = SUPPORT / "protocol.json"
FROZEN_HASH = "9fc27b79041931d98cbcb8349cdab7d81fe31000a49cd3fe14dc7cd85ebe3651"
VERSIONS = ("published", "frozen", "current")


def harness_hash():
    digest = hashlib.sha256(phase13.harness_hash().encode())
    for path in [Path(__file__), *sorted(SUPPORT.glob("*.py")), PROTOCOL]:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def worker_cpu_affinity(requested=None):
    """Read, or explicitly set, only this worker's Windows processor mask.

    Apply a requested mask before importing numerical libraries so their worker
    threads inherit the same allowed processors. The default only observes the
    inherited mask; it does not change scheduling or machine power settings.
    """
    if os.name != "nt":
        if requested is not None:
            raise RuntimeError("--cpu-affinity currently requires Windows")
        return dict(requested_mask=None, actual_mask=None, system_mask=None)
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    pointer = ctypes.POINTER(ctypes.c_size_t)
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, pointer, pointer]
    kernel.GetProcessAffinityMask.restype = wintypes.BOOL
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel.SetProcessAffinityMask.restype = wintypes.BOOL
    handle = kernel.GetCurrentProcess()
    actual, system = ctypes.c_size_t(), ctypes.c_size_t()

    def read():
        if not kernel.GetProcessAffinityMask(
            handle, ctypes.byref(actual), ctypes.byref(system)
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    read()
    if requested is not None:
        if requested <= 0 or requested & ~actual.value:
            raise ValueError(
                f"CPU affinity {requested:#x} must be a nonzero subset of "
                f"the inherited process mask {actual.value:#x}"
            )
        if not kernel.SetProcessAffinityMask(handle, requested):
            raise ctypes.WinError(ctypes.get_last_error())
        read()
        if actual.value != requested:
            raise RuntimeError(
                f"Requested CPU affinity {requested:#x}, read back {actual.value:#x}"
            )
    return dict(
        requested_mask=None if requested is None else hex(requested),
        actual_mask=hex(actual.value),
        system_mask=hex(system.value),
    )


def host_observation():
    """Read available host context without managing unrelated processes."""
    record = dict(wall_time_utc=datetime.now(timezone.utc).isoformat())
    if hasattr(os, "getloadavg"):
        record["load_average"] = list(os.getloadavg())
    try:
        import psutil
    except ImportError:
        record["psutil"] = "not installed"
    else:
        process = psutil.Process()
        record.update(
            logical_cpus=psutil.cpu_count(),
            system_cpu_percent=psutil.cpu_percent(interval=None),
            process_cpu_times=dict(process.cpu_times()._asdict()),
            process_context_switches=dict(process.num_ctx_switches()._asdict()),
        )
        frequency = psutil.cpu_freq()
        if frequency is not None:
            record["cpu_frequency_mhz"] = dict(frequency._asdict())
    return record


class Context(phase13.Context):
    """Phase 13 workloads with fixed repetitions and full current-state checks."""

    def __init__(self, *args):
        super().__init__(*args)
        self.protocol = phase13.read(PROTOCOL)
        self.complete_snapshots = {}
        self.reference_identities = {}
        self.host_samples = []

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)
        case = self.cases[-1]
        leaves = self.jax.tree_util.tree_leaves(case["variants"][0])
        batched = "heat_exchanger" in case["name"] or any(
            hasattr(value, "shape") and self.np.prod(value.shape) > 1
            for value in leaves
        )
        case["phase13_requested_calls"] = case["calls"]
        host = case["category"] in {
            "source_full_state",
            "callback_full_state",
            "callback_full_state_JVP",
            "metastable_full_state",
            "legacy_common_binary_full_state",
        } or case["name"].startswith(
            ("source/", "callback/", "metastable/", "binary_source/")
        )
        case["calls"] = (
            self.args.host_calls
            if host
            else (self.args.batch_calls if batched else self.args.scalar_calls)
        )
        case["fixed_repetition_class"] = (
            "host" if host else ("batch" if batched else "scalar")
        )

    def complete_snapshot(self, value, *, path="result"):
        """Keep every actual candidate/frozen output, including real statuses."""
        if isinstance(value, self.jp.FluidState):
            raw = self.api.dictionary(value)
            self.reference_identities[path] = repr(getattr(value, "reference_id", None))
            metadata = {
                name: getattr(value, name, None)
                for name in (
                    "fluid_name",
                    "identifier",
                    "provider",
                    "model",
                    "error_policy",
                )
            }
            metadata["providers"] = sorted(getattr(value, "providers", ()))
            metadata["available_properties"] = sorted(
                getattr(value, "available_properties", raw)
            )
            return dict(
                kind="FluidState",
                metadata=metadata,
                properties={
                    key: self.array_snapshot(item) for key, item in raw.items()
                },
                status=self.complete_snapshot(
                    self.api.status(value), path=path + ".status"
                ),
                clipped=self.complete_snapshot(
                    getattr(value, "_clipped", None), path=path + ".clipped"
                ),
            )
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                child_path = path + "." + str(key)
                if key == "reference_id":
                    self.reference_identities[child_path] = repr(item)
                    result[str(key)] = "source provenance retained separately"
                else:
                    result[str(key)] = self.complete_snapshot(item, path=child_path)
            return result
        if isinstance(value, (set, frozenset)):
            return [
                self.complete_snapshot(item, path=path)
                for item in sorted(value, key=str)
            ]
        if isinstance(value, (tuple, list)):
            return [
                self.complete_snapshot(item, path=path + "." + str(index))
                for index, item in enumerate(value)
            ]
        return self.array_snapshot(value)

    def execute(self):
        """Measure first calls, validate, then time fixed balanced case blocks."""

        def prepare(case):
            name, fn, variants = case["name"], case["fn"], case["variants"]
            print("Preparing " + name, flush=True)
            # Exclude pending fixture placement from a first-public-call timer.
            self.synchronized(variants[0])
            start = time.perf_counter_ns()
            initial = self.synchronized(fn(*variants[0]))
            self.first_calls[name] = (time.perf_counter_ns() - start) / 1e3
            values = [initial]
            values.extend(
                self.synchronized(fn(*arguments)) for arguments in variants[1:]
            )
            self.snapshots[name] = [
                self.snapshot(value, case["fields"], name=name) for value in values
            ]
            if not self.api.legacy:
                self.complete_snapshots[name] = [
                    self.complete_snapshot(value, path=name + "." + str(index))
                    for index, value in enumerate(values)
                ]
            if case["compiled"] is not None:
                executable = case["compiled"].lower(*variants[0]).compile().as_text()
                normalized = re.sub(r"metadata=\{[^{}]*\}", "", executable)
                self.graphs[name] = dict(
                    normalized_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                    instructions=len(
                        re.findall(r"^\s*%[^\n]*=", executable, re.MULTILINE)
                    ),
                    contains_callback="callback" in executable.lower(),
                    entry_layout=executable.splitlines()[0],
                )
                (self.directory / (name.replace("/", "_") + ".hlo.txt")).write_text(
                    executable, encoding="utf-8"
                )
            for index in range(self.protocol["warmup_calls"]):
                self.synchronized(fn(*variants[index % len(variants)]))

        for case in self.cases:
            prepare(case)
        if self.args.suite == "full":
            # Creating helper inputs evaluates public states. Keep that work
            # after every selected thermodynamic first-call observation.
            previous = len(self.cases)
            phase13.state_cases(self)
            helpers = self.cases[previous:]
            if self.args.case:
                helpers = [
                    case
                    for case in helpers
                    if any(
                        re.search(pattern, case["name"]) for pattern in self.args.case
                    )
                ]
            self.cases = self.cases[:previous] + helpers
            for case in helpers:
                prepare(case)
        assert self.cases, "No workload matches the requested cases"
        for name, callback in self.validators:
            value = callback()
            if value is not None:
                self.check(name, value)
        if self.args.validate_only:
            return
        samples = {case["name"]: [] for case in self.cases}
        for sample in range(self.args.samples):
            # Forward/reverse blocks keep opposite positions paired. All
            # versions use the same counts, sequence and numerical variants.
            ordered = self.cases if sample % 2 == 0 else list(reversed(self.cases))
            self.host_samples.append(dict(sample=sample, before=host_observation()))
            for case in ordered:
                start = time.perf_counter_ns()
                for call in range(case["calls"]):
                    self.synchronized(
                        case["fn"](
                            *case["variants"][(sample + call) % len(case["variants"])]
                        )
                    )
                samples[case["name"]].append(
                    (time.perf_counter_ns() - start) / case["calls"] / 1e3
                )
            self.host_samples[-1]["after"] = host_observation()
        for case in self.cases:
            values = samples[case["name"]]
            self.timings[case["name"]] = dict(
                microseconds_per_call=values,
                median=statistics.median(values),
                calls_per_sample=case["calls"],
            )


def register_controls(ctx):
    """Add identical-point shape controls and explicitly diagnostic ablations."""
    jp, jnp, jax, api = ctx.jp, ctx.jnp, ctx.jax, ctx.api
    for label, model in (("pg", ctx.models["pg"]), ("bicubic", ctx.models["bicubic"])):
        h_axis = jnp.linspace(680e3, 700e3, 8)[:, None]
        p_axis = jnp.linspace(6e6, 7e6, 16)[None, :]
        full_h, full_p = jnp.broadcast_arrays(h_axis, p_axis)
        shapes = {
            "flat128": (full_h.ravel(), full_p.ravel()),
            "broadcast8x16": (full_h, full_p),
            "broadcast16x8": (full_h.reshape(16, 8), full_p.reshape(16, 8)),
            "broadcast_independent_axes8x16": (h_axis, p_axis),
        }
        for shape, (enthalpy, pressure) in shapes.items():
            evaluate = jax.jit(
                lambda h, p, model=model: api.evaluate(model, jp.HmassP_INPUTS, h, p)
            )
            ctx.add(
                label + "_phase14_HP_" + shape,
                evaluate,
                [(enthalpy, pressure), (enthalpy + 1e3, pressure + 2e4)],
                fields=phase13.PG_FIELDS if label == "pg" else phase13.CORE,
                compiled=evaluate,
                category="shape_control_full_state",
                note="The same 128 physical HP points in each declared shape; actual public return and status retained.",
            )
    if api.legacy or not ctx.args.diagnostics:
        return
    originals = [case for case in ctx.cases if case["category"] == "consumer"]
    for original in originals:
        for projection in ("numerical_only", "status_only"):

            def selected(*arguments, original=original, projection=projection):
                result = original["fn"](*arguments)
                return {
                    key: value
                    for key, value in result.items()
                    if key.startswith("status") == (projection == "status_only")
                }

            evaluate = jax.jit(selected)
            ctx.add(
                original["name"] + "_diagnostic_" + projection,
                evaluate,
                original["variants"],
                compiled=evaluate,
                category="diagnostic_ablation",
                comparable=False,
                note="Current/frozen attribution only; never a replacement consumer gate and no fabricated published status.",
            )


def worker(args):
    affinity = worker_cpu_affinity(args.cpu_affinity)
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
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(
        args.record.parent / ("cache_" + uuid.uuid4().hex)
    )
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
    before, harness = phase13.source_hash(root), harness_hash()
    jax.block_until_ready(jnp.asarray(0.0))
    ctx = Context(
        args,
        jp,
        CP,
        jax,
        jnp,
        np,
        eqx,
        API(jp, args.version == "published", CP, jax, jnp, np, eqx),
    )
    native_cases.register(ctx)
    if args.suite == "full":
        source_cases.register(ctx)
    register_controls(ctx)
    if args.case:
        ctx.cases = [
            case
            for case in ctx.cases
            if any(re.search(pattern, case["name"]) for pattern in args.case)
        ]
    ctx.execute()
    assert phase13.source_hash(root) == before and harness_hash() == harness
    final_affinity = worker_cpu_affinity()
    if args.cpu_affinity is not None:
        assert (
            final_affinity["actual_mask"] == affinity["actual_mask"]
        ), "Worker CPU affinity changed during the experiment"
    affinity["final_mask"] = final_affinity["actual_mask"]
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
        cpu_affinity=affinity,
        python=sys.version,
        executable=sys.executable,
        device=str(jax.devices()[0]),
        x64=jax.config.x64_enabled,
        threads={
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    )
    manifest = [
        {
            key: value
            for key, value in case.items()
            if key not in {"fn", "variants", "compiled"}
        }
        for case in ctx.cases
    ]
    phase13.save(
        args.record,
        dict(
            version=args.version,
            source_sha256=before,
            harness_sha256=harness,
            imported_package=str(jp.__file__),
            environment=environment,
            manifest=manifest,
            snapshots=ctx.snapshots,
            complete_snapshots=ctx.complete_snapshots,
            reference_identities=ctx.reference_identities,
            timings=ctx.timings,
            setup=ctx.setups,
            first_calls_microseconds=ctx.first_calls,
            first_call_condition="First encounter of each wrapper in a shared worker; fixture placement synchronized and related kernels may be warm.",
            payloads=ctx.payloads,
            independent_checks=ctx.checks,
            skipped=ctx.skipped,
            graphs=ctx.graphs,
            host_observations=ctx.host_samples,
            validate_only=args.validate_only,
        ),
    )
    print(json.dumps(dict(record=str(args.record), cases=len(ctx.cases))), flush=True)


def compare_complete(old, new, protocol):
    """Compare all candidate/frozen leaves without dropping status dictionaries."""
    import numpy as np

    failures, counts = [], dict(finite=0, paired_unavailable=0)
    correctness = protocol["candidate_correctness"]

    def visit(a, b, path):
        if isinstance(a, dict) and "values" in a and "dtype" in a:
            if (
                not isinstance(b, dict)
                or a["shape"] != b.get("shape")
                or a["dtype"] != b.get("dtype")
            ):
                failures.append(dict(path=path, reason="Numerical shape/dtype changed"))
                return
            x, y = (
                np.asarray(
                    [np.nan if value is None else value for value in item["values"]]
                ).reshape(item["shape"])
                for item in (a, b)
            )
            if not np.array_equal(np.isfinite(x), np.isfinite(y)) or np.any(
                np.isinf(x) | np.isinf(y)
            ):
                failures.append(
                    dict(
                        path=path,
                        reason="Finite/nonfinite pattern changed or infinity returned",
                    )
                )
                return
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                failures.append(
                    dict(path=path, reason="Availability/NaN pattern changed")
                )
                return
            finite = np.isfinite(x) & np.isfinite(y)
            counts["finite"] += int(np.count_nonzero(finite))
            counts["paired_unavailable"] += int(np.count_nonzero(np.isnan(x)))
            field = path.rsplit(".", 1)[-1]
            if (
                a["dtype"].startswith(("int", "uint", "bool"))
                or field in ("status", "clipped")
                or field.startswith("status_")
            ):
                equal = np.array_equal(x, y, equal_nan=True)
            else:
                equal = np.allclose(
                    x[finite],
                    y[finite],
                    rtol=correctness["floating_rtol"],
                    atol=correctness["field_atol"].get(
                        field, correctness["default_atol"]
                    ),
                )
            if not equal:
                failures.append(
                    dict(
                        path=path,
                        reason="Numerical values changed",
                        baseline_sample=a["sample"],
                        candidate_sample=b["sample"],
                    )
                )
        elif isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                failures.append(
                    dict(path=path, reason="Returned fields/structure changed")
                )
                return
            for key in a:
                visit(a[key], b[key], path + "." + key)
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                failures.append(dict(path=path, reason="Returned sequence changed"))
                return
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + "." + str(index))
        elif a != b:
            failures.append(
                dict(path=path, reason="Metadata/value changed", frozen=a, current=b)
            )

    visit(old, new, "result")
    return dict(**counts, mismatches=failures)


def verify_wheel(manifest_path, source):
    manifest = phase13.read(manifest_path)
    wheel = manifest_path.parent / manifest["wheel_filename"]
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == manifest["wheel_sha256"]
    with zipfile.ZipFile(wheel) as archive:
        expected = {
            name
            for name in archive.namelist()
            if name.startswith("jaxprop/") and name.endswith(".py")
        }
        actual = {
            path.relative_to(source).as_posix()
            for path in (source / "jaxprop").rglob("*.py")
        }
        assert expected == actual
        for name in expected:
            assert (source / name).read_bytes() == archive.read(name), name
    return manifest


def aggregate_case(case, name, records, versions, acceptance):
    groups = {
        version: [
            record["timings"][name]["median"]
            for key, record in records.items()
            if key.startswith(version)
        ]
        for version in versions
    }
    medians = {version: statistics.median(values) for version, values in groups.items()}
    result = dict(
        worker_medians_us=groups,
        median_us=medians,
        samples_us={
            key: record["timings"][name]
            for key, record in records.items()
            if name in record["timings"]
        },
    )
    if "published" in medians and case["comparable"]:
        old, new = medians["published"], medians["current"]
        limit = phase13.category_limit(case, old, acceptance)
        result.update(
            candidate_to_published_ratio=new / old,
            limit_us=limit,
            above_original_budget=new > limit,
            above_strict_10_percent=new > old * 1.1,
        )
    result["candidate_to_frozen_ratio"] = medians["current"] / medians["frozen"]
    result["above_frozen_10_percent"] = medians["current"] > medians["frozen"] * 1.1
    return result


def controller(args):
    all_sources = dict(
        published=args.published_source.resolve(),
        frozen=args.frozen_source.resolve(),
        current=args.current_source.resolve(),
    )
    sources = {version: all_sources[version] for version in args.versions}
    baseline = (
        verify_wheel(args.baseline_manifest.resolve(), sources["published"])
        if "published" in sources
        else phase13.read(args.baseline_manifest)
    )
    frozen = {version: phase13.source_hash(path) for version, path in sources.items()}
    assert (
        frozen["frozen"] == FROZEN_HASH
    ), "Frozen comparison must be the verified Phase 13 package"
    area = args.output_directory.resolve()
    area.mkdir(parents=True, exist_ok=True)
    assert not (
        area / "experiment_manifest.json"
    ).exists(), (
        "Choose a new experiment output directory; prior evidence is never overwritten"
    )
    protocol, acceptance, harness = (
        phase13.read(PROTOCOL),
        phase13.read(phase13.SUPPORT / "acceptance.json"),
        harness_hash(),
    )
    blocks = protocol["version_blocks"]
    order = []
    for index in range(args.workers):
        block_index = index + args.order_offset
        if len(sources) == 3:
            block = blocks[block_index % len(blocks)]
        else:
            block = list(sources) if block_index % 2 == 0 else list(reversed(sources))
        order.extend(block)
    experiment = dict(
        declared_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol=protocol,
        baseline=baseline,
        sources={key: str(value) for key, value in sources.items()},
        source_hashes=frozen,
        harness_sha256=harness,
        suite=args.suite,
        case_filters=args.case,
        samples=args.samples,
        workers_per_version=args.workers,
        version_order=order,
        fixed_scalar_calls=args.scalar_calls,
        fixed_batch_calls=args.batch_calls,
        fixed_host_calls=args.host_calls,
        diagnostics=args.diagnostics,
        order_offset=args.order_offset,
        original_acceptance=acceptance,
        validate_only=args.validate_only,
        cpu_affinity_requested=(
            None if args.cpu_affinity is None else hex(args.cpu_affinity)
        ),
        cpu_affinity_scope="Own worker before numerical imports; inherited when omitted; actual mask verified in worker environment",
    )
    # This declaration is persisted before the first worker/measurement starts.
    phase13.save(area / "experiment_manifest.json", experiment)
    records, counts = {}, {version: 0 for version in VERSIONS}
    for sequence, version in enumerate(order):
        counts[version] += 1
        key = version + str(counts[version])
        directory = area / "workers" / key
        directory.mkdir(parents=True, exist_ok=True)
        record_path = directory / "record.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--version",
            version,
            "--source",
            str(sources[version]),
            "--record",
            str(record_path),
            "--suite",
            args.suite,
            "--samples",
            str(args.samples),
            "--scalar-calls",
            str(args.scalar_calls),
            "--batch-calls",
            str(args.batch_calls),
            "--host-calls",
            str(args.host_calls),
        ]
        if args.diagnostics:
            command.append("--diagnostics")
        if args.cpu_affinity is not None:
            command.extend(("--cpu-affinity", hex(args.cpu_affinity)))
        if args.validate_only:
            command.append("--validate-only")
        for pattern in args.case or ():
            command += ["--case", pattern]
        print(f"Starting {sequence + 1}/{len(order)}: {key}", flush=True)
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"Worker failed; inspect {directory / 'worker.log'}")
        record = phase13.read(record_path)
        assert (
            record["source_sha256"] == frozen[version]
            and record["harness_sha256"] == harness
        )
        records[key] = record
    reference = records["frozen1"]
    for key, record in records.items():
        assert record["environment"] == reference["environment"], key
    published = records.get("published1")
    old_manifest = (
        {case["name"]: case for case in published["manifest"]} if published else {}
    )
    current_manifest = {case["name"]: case for case in reference["manifest"]}
    correctness = {}
    for key, record in records.items():
        results = {}
        if not key.startswith("published"):
            assert set(record["complete_snapshots"]) == set(
                reference["complete_snapshots"]
            ), key
            for name in reference["complete_snapshots"]:
                results[name + "/frozen_complete"] = compare_complete(
                    reference["complete_snapshots"][name],
                    record["complete_snapshots"][name],
                    protocol,
                )
        for name, case in old_manifest.items():
            if case["comparable"]:
                assert name in record["snapshots"], (key, name)
                results[name + "/published_common"] = phase13.compare_snapshots(
                    published["snapshots"][name], record["snapshots"][name], case
                )
        correctness[key] = results
    failures = {
        key: {
            name: result["mismatches"]
            for name, result in values.items()
            if result["mismatches"]
        }
        for key, values in correctness.items()
    }
    failures = {key: value for key, value in failures.items() if value}
    phase13.save(
        area / "correctness.json", dict(results=correctness, failures=failures)
    )
    if failures:
        raise RuntimeError(
            f"Complete-array validation failed; inspect {area / 'correctness.json'}"
        )
    cases = {}
    if not args.validate_only:
        for name, case in current_manifest.items():
            versions = (
                tuple(sources)
                if name in old_manifest and case["comparable"]
                else ("frozen", "current")
            )
            cases[name] = dict(
                manifest=case,
                **aggregate_case(case, name, records, versions, acceptance),
            )
    summary = dict(
        experiment=experiment,
        correctness_passed=True,
        cases=cases,
        correctness_record=str(area / "correctness.json"),
        worker_records={
            key: str(area / "workers" / key / "record.json") for key in records
        },
        first_public_calls={
            key: record["first_calls_microseconds"] for key, record in records.items()
        },
        setup={key: record["setup"] for key, record in records.items()},
        graphs={key: record["graphs"] for key, record in records.items()},
        host_observations={
            key: record["host_observations"] for key, record in records.items()
        },
        diagnostic_cases=[
            name
            for name, case in current_manifest.items()
            if case["category"] == "diagnostic_ablation"
        ],
        limitations=[
            "Targeted experiments do not replace the complete original Phase 13 gate.",
            "Shared-worker first encounters are not isolated cold compilation measurements.",
            "Diagnostic status projections are attribution tools, never substituted acceptance results.",
        ],
    )
    phase13.save(area / "summary.json", summary)
    print(
        json.dumps(
            dict(
                summary=str(area / "summary.json"),
                correctness_passed=True,
                cases=len(cases),
                above_original_budget=[
                    name
                    for name, value in cases.items()
                    if value.get("above_original_budget")
                ],
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--published-source",
        type=Path,
        default=ROOT / "benchmarks/results/published",
    )
    parser.add_argument(
        "--frozen-source",
        type=Path,
        default=ROOT / "benchmarks/results/frozen",
    )
    parser.add_argument("--current-source", type=Path, default=ROOT)
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=phase13.SUPPORT / "baseline/baseline_manifest.json",
    )
    parser.add_argument("--suite", choices=("native", "full"), default="native")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=VERSIONS,
        default=list(VERSIONS),
        help="Use frozen current for balanced candidate-isolation experiments; include published for three-way comparisons",
    )
    parser.add_argument(
        "--order-offset",
        type=int,
        default=0,
        help="Start at another predeclared balanced order block; use 1 to reverse the initial two-version pair",
    )
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument(
        "--cpu-affinity",
        type=lambda value: int(value, 0),
        help="Windows worker-process allowed CPU mask, decimal or 0x hexadecimal; omitted preserves inherited scheduling",
    )
    parser.add_argument("--scalar-calls", type=int, default=48)
    parser.add_argument("--batch-calls", type=int, default=16)
    parser.add_argument(
        "--host-calls",
        type=int,
        default=1,
        help="Fixed public calls per source/callback/metastable/parcel workload sample",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Add separate consumer numerical-only/status-only attribution cases",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Regex filter; repeat to select several workload groups",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--version", choices=VERSIONS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.calls_scale = 1
    if args.cpu_affinity is not None and args.cpu_affinity <= 0:
        parser.error("--cpu-affinity must be a nonzero positive mask")
    if min(args.samples, args.scalar_calls, args.batch_calls, args.host_calls) <= 0:
        parser.error("Sample and repetition counts must be positive")
    if len(set(args.versions)) != len(args.versions) or not {
        "frozen",
        "current",
    }.issubset(args.versions):
        parser.error(
            "Select frozen and current once each, optionally including published"
        )
    if args.worker:
        worker(args)
    elif args.output_directory is None:
        parser.error("--output-directory is required")
    else:
        controller(args)


if __name__ == "__main__":
    main()
