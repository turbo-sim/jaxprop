"""Instrumented workload attribution; never a replacement acceptance gate.

The optional full preparation retains the original compilation cache and full
state snapshots, then times a small declared selection. Windows metrics use
only the standard library. Provider order, numerical variants and GC activity
are retained so process effects can be distinguished from numerical changes.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import gc
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

import phase14_benchmark as benchmark

ROOT = benchmark.ROOT
PROTOCOL = ROOT / "benchmarks/phase14/attribution_protocol.json"
DEFAULT_CASES = (
    "pg_matched_compiled_compressor",
    "pg_matched_compiled_expander",
    "pg_matched_compiled_HP_batch1024",
    "bicubic_CO2_compiled_HP_broadcast8x16",
    "source/REFPROP/water_liquid_PT_scalar",
    "source/REFPROP/co2_supercritical_HP_scalar",
)


def harness_hash():
    digest = hashlib.sha256(benchmark.harness_hash().encode())
    for path in (Path(__file__), PROTOCOL):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


class WindowsMetrics:
    """Read process/system counters without an optional Python dependency."""

    class Memory(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                    "PrivateUsage",
                )
            ],
        ]

    def __init__(self):
        self.reason = None
        if os.name != "nt":
            self.reason = "Windows counters are unavailable on this platform"
            return
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel.GetCurrentProcess.argtypes = []
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.handle = self.kernel.GetCurrentProcess()
        self.kernel.GetSystemTimes.argtypes = [ctypes.POINTER(wintypes.FILETIME)] * 3
        self.kernel.GetSystemTimes.restype = wintypes.BOOL
        self.kernel.GetCurrentProcessorNumber.argtypes = []
        self.kernel.GetCurrentProcessorNumber.restype = wintypes.DWORD
        self.kernel.GetProcessAffinityMask.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.kernel.GetProcessAffinityMask.restype = wintypes.BOOL
        self.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(self.Memory),
            wintypes.DWORD,
        ]
        self.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    @staticmethod
    def ticks(value):
        return (value.dwHighDateTime << 32) | value.dwLowDateTime

    def snapshot(self):
        result = dict(
            wall_utc=datetime.now(timezone.utc).isoformat(),
            monotonic_ns=time.perf_counter_ns(),
            process_cpu_ns=time.process_time_ns(),
            thread_cpu_ns=time.thread_time_ns(),
            gc_enabled=gc.isenabled(),
            gc_count=list(gc.get_count()),
            gc_stats=gc.get_stats(),
        )
        if self.reason:
            return result | dict(windows_metrics_unavailable=self.reason)
        idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
        if self.kernel.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            result["system_cpu_ticks_100ns"] = dict(
                idle=self.ticks(idle), kernel=self.ticks(kernel), user=self.ticks(user)
            )
        else:
            result["system_cpu_error"] = ctypes.get_last_error()
        memory = self.Memory()
        memory.cb = ctypes.sizeof(memory)
        if self.psapi.GetProcessMemoryInfo(
            self.handle, ctypes.byref(memory), memory.cb
        ):
            result["memory"] = {
                name: int(getattr(memory, name))
                for name, _ in memory._fields_
                if name != "cb"
            }
        else:
            result["memory_error"] = ctypes.get_last_error()
        result["main_logical_processor"] = int(self.kernel.GetCurrentProcessorNumber())
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if self.kernel.GetProcessAffinityMask(
            self.handle, ctypes.byref(process_mask), ctypes.byref(system_mask)
        ):
            result["affinity"] = dict(
                process=int(process_mask.value), system=int(system_mask.value)
            )
        return result


class GCEvents:
    """Retain actual collection intervals, including their timed-case overlap."""

    def __init__(self):
        self.events, self.started = [], {}

    def callback(self, phase, info):
        generation = info["generation"]
        if phase == "start":
            self.started[generation] = time.perf_counter_ns()
        else:
            self.events.append(
                dict(
                    generation=generation,
                    start_ns=self.started.pop(generation, None),
                    end_ns=time.perf_counter_ns(),
                    collected=info["collected"],
                    uncollectable=info["uncollectable"],
                )
            )


def metric_delta(before, after):
    result = dict(
        process_cpu_ns=after["process_cpu_ns"] - before["process_cpu_ns"],
        thread_cpu_ns=after["thread_cpu_ns"] - before["thread_cpu_ns"],
    )
    if "memory" in before and "memory" in after:
        result["memory"] = {
            key: after["memory"][key] - value for key, value in before["memory"].items()
        }
    if "system_cpu_ticks_100ns" in before and "system_cpu_ticks_100ns" in after:
        ticks = {
            key: after["system_cpu_ticks_100ns"][key] - value
            for key, value in before["system_cpu_ticks_100ns"].items()
        }
        total = ticks["kernel"] + ticks["user"]
        result["system_busy_fraction"] = (
            (total - ticks["idle"]) / total if total else None
        )
        result["system_cpu_ticks_100ns"] = ticks
    return result


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
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(
        args.record.parent / ("cache_" + uuid.uuid4().hex)
    )
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import CoolProp.CoolProp as CP
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jaxprop as jp
    import numpy as np
    from phase13 import native_cases, source_cases
    from phase13.api import API

    assert Path(jp.__file__).resolve().parent == source / "jaxprop"
    assert jax.default_backend() == "cpu" and jax.config.x64_enabled
    before_source, before_harness = (
        benchmark.phase13.source_hash(source),
        harness_hash(),
    )
    jax.block_until_ready(jnp.asarray(0.0))
    metrics, collections = WindowsMetrics(), GCEvents()
    gc.callbacks.append(collections.callback)
    original_gc = gc.isenabled()
    args.calls_scale, args.diagnostics = 1, False
    requested_patterns = args.case
    args.case, args.suite, args.validate_only = None, "full", True
    context = benchmark.Context(
        args,
        jp,
        CP,
        jax,
        jnp,
        np,
        eqx,
        API(jp, args.version == "published", CP, jax, jnp, np, eqx),
    )
    observations = dict(before_registration=metrics.snapshot())
    native_cases.register(context)
    source_cases.register(context)
    benchmark.register_controls(context)
    all_cases = list(context.cases)
    matches = lambda case: (
        any(re.search(pattern, case["name"]) for pattern in requested_patterns)
        if requested_patterns
        else case["name"] in DEFAULT_CASES
    )
    requested = [case for case in all_cases if matches(case)]
    assert requested, "No selected timing case"
    pg_control = next(
        case for case in all_cases if case["name"] == "pg_matched_compiled_compressor"
    )
    neighbors = {}
    for case in requested:
        index = all_cases.index(case)
        neighbors[case["name"]] = {
            "forward": all_cases[index - 1] if index else None,
            "reverse": all_cases[index + 1] if index + 1 < len(all_cases) else None,
        }
    all_validator_names = [name for name, _ in context.validators]
    if not args.prepare_full:
        context.cases = requested + ([] if pg_control in requested else [pg_control])
        if args.prime_predecessor:
            for adjacent in neighbors.values():
                for case in adjacent.values():
                    if case is not None and case not in context.cases:
                        context.cases.append(case)
        selected_names = [case["name"] for case in context.cases]
        # The original filtered harness runs all registered validators; several
        # invoke/compile unselected public functions. Minimal preparation must
        # filter these too, otherwise the compilation-cache contrast is false.
        context.validators = [
            (name, callback)
            for name, callback in context.validators
            if any(
                name == selected
                or name.startswith(selected + "/")
                or name.startswith(selected + "_")
                for selected in selected_names
            )
        ]
        args.suite = "native"  # Do not allocate unrelated state-helper fixtures.
    observations["before_preparation"] = metrics.snapshot()
    context.execute()
    observations["after_preparation"] = metrics.snapshot()
    prepared_cases = [case["name"] for case in context.cases]
    timed_cases = [case for case in context.cases if matches(case)]
    # Write the same sidecar in both arms so serialization/I/O do not differ.
    # Disk mode releases only verification artefacts, retaining the numerical
    # models, input buffers, functions and compiled executables in the context.
    snapshot_path = args.record.parent / "preparation_snapshots.json"
    observations["before_snapshot_storage"] = metrics.snapshot()
    benchmark.phase13.save(
        snapshot_path,
        dict(
            snapshots=context.snapshots, complete_snapshots=context.complete_snapshots
        ),
    )
    storage = dict(
        mode=args.snapshot_storage,
        sidecar=str(snapshot_path),
        bytes=snapshot_path.stat().st_size,
        sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    )
    if args.snapshot_storage == "disk":
        context.snapshots.clear()
        context.complete_snapshots.clear()
    observations["after_snapshot_storage"] = metrics.snapshot()
    runtime_control = jax.jit(lambda x: x * 1.0001 + 0.125)
    control_input = jnp.linspace(1.0, 2.0, 1024)
    for _ in range(8):
        jax.block_until_ready(runtime_control(control_input))
    interventions = dict(mode=args.gc_mode, initial_gc_enabled=original_gc)
    if args.gc_mode in ("collected", "disabled"):
        start = time.perf_counter_ns()
        interventions.update(
            collected_objects=gc.collect(), collection_ns=time.perf_counter_ns() - start
        )
    if args.gc_mode == "disabled":
        gc.disable()
    observations["before_timing"] = metrics.snapshot()
    rows, controls, history = (
        [],
        [],
        dict(last_case="preparation", last_refprop_case=None),
    )

    def run(case, sample, role):
        count = case["calls"]
        prior = len(collections.events)
        before = metrics.snapshot()
        indices = [(sample + call) % len(case["variants"]) for call in range(count)]
        start = time.perf_counter_ns()
        for index in indices:
            jax.block_until_ready(case["fn"](*case["variants"][index]))
        end = time.perf_counter_ns()
        after = metrics.snapshot()
        events = [
            event
            for event in collections.events[prior:]
            if event["start_ns"] is not None
            and event["start_ns"] < end
            and event["end_ns"] > start
        ]
        result = dict(
            case=case["name"],
            sample=sample,
            role=role,
            microseconds_per_call=(end - start) / count / 1e3,
            calls=count,
            variant_indices=indices,
            predecessor_case=history["last_case"],
            predecessor_refprop_case=history["last_refprop_case"],
            before=before,
            after=after,
            metrics_delta=metric_delta(before, after),
            gc_events=events,
            overlapping_gc_ns=sum(
                max(0, min(end, event["end_ns"]) - max(start, event["start_ns"]))
                for event in events
            ),
        )
        history["last_case"] = case["name"]
        if "REFPROP" in case["name"]:
            history["last_refprop_case"] = case["name"]
        return result

    sentinel = dict(
        name="runtime_control_array1024",
        calls=args.control_calls,
        fn=runtime_control,
        variants=[(control_input,)],
    )
    try:
        for sample in range(args.samples):
            controls.append(run(sentinel, sample, "before_block"))
            controls.append(run(pg_control, sample, "pg_before_block"))
            ordered = timed_cases if sample % 2 == 0 else list(reversed(timed_cases))
            for case in ordered:
                if args.prime_predecessor:
                    adjacent = neighbors[case["name"]][
                        "forward" if sample % 2 == 0 else "reverse"
                    ]
                    if adjacent is not None:
                        jax.block_until_ready(
                            adjacent["fn"](
                                *adjacent["variants"][
                                    sample % len(adjacent["variants"])
                                ]
                            )
                        )
                        history["last_case"] = adjacent["name"]
                        if "REFPROP" in adjacent["name"]:
                            history["last_refprop_case"] = adjacent["name"]
                rows.append(run(case, sample, "timed"))
            controls.append(run(pg_control, sample, "pg_after_block"))
            controls.append(run(sentinel, sample, "after_block"))
    finally:
        if original_gc:
            gc.enable()
        else:
            gc.disable()
        gc.callbacks.remove(collections.callback)
    observations["after_timing"] = metrics.snapshot()
    assert (
        benchmark.phase13.source_hash(source) == before_source
        and harness_hash() == before_harness
    )
    if args.snapshot_storage == "disk":
        assert (
            hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == storage["sha256"]
        )
        restored = benchmark.phase13.read(snapshot_path)
        context.snapshots = restored["snapshots"]
        context.complete_snapshots = restored["complete_snapshots"]
        del restored
    observations["after_snapshot_restore"] = metrics.snapshot()
    timings = {}
    for case in timed_cases:
        values = [
            row["microseconds_per_call"] for row in rows if row["case"] == case["name"]
        ]
        timings[case["name"]] = dict(
            median=statistics.median(values),
            microseconds_per_call=values,
            calls_per_sample=case["calls"],
        )
    environment = {
        name: importlib.metadata.version(name)
        for name in ("jax", "jaxlib", "numpy", "CoolProp", "equinox")
    }
    environment.update(
        python=sys.version,
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
        python_hash_seed=os.environ.get("PYTHONHASHSEED", "random"),
    )
    benchmark.phase13.save(
        args.record,
        dict(
            version=args.version,
            source_sha256=before_source,
            harness_sha256=before_harness,
            imported_package=str(jp.__file__),
            environment=environment,
            prepare_full=args.prepare_full,
            snapshot_storage=storage,
            preparation_cases=prepared_cases,
            gc_intervention=interventions,
            observations=observations,
            rows=rows,
            controls=controls,
            timings=timings,
            gc_events=collections.events,
            validators_registered=all_validator_names,
            validators_executed=[name for name, _ in context.validators],
            prime_predecessor=args.prime_predecessor,
            snapshots={name: context.snapshots[name] for name in timings},
            complete_snapshots={
                name: context.complete_snapshots[name]
                for name in timings
                if name in context.complete_snapshots
            },
            manifest=[
                {
                    k: v
                    for k, v in case.items()
                    if k not in ("fn", "variants", "compiled")
                }
                for case in timed_cases
            ],
            graphs={
                name: context.graphs[name] for name in timings if name in context.graphs
            },
            diagnostic_only=True,
        ),
    )
    print(
        json.dumps(dict(record=str(args.record), timed_cases=len(timings))), flush=True
    )


def controller(args):
    sources = {
        key: path.resolve()
        for key, path in dict(
            published=args.published_source,
            frozen=args.frozen_source,
            current=args.current_source,
        ).items()
        if key in args.versions
    }
    hashes = {key: benchmark.phase13.source_hash(path) for key, path in sources.items()}
    assert hashes["frozen"] == benchmark.FROZEN_HASH
    if "published" in sources:
        benchmark.verify_wheel(args.baseline_manifest, sources["published"])
    area = args.output_directory.resolve()
    area.mkdir(parents=True, exist_ok=True)
    assert not (
        area / "experiment_manifest.json"
    ).exists(), "Use a new output directory"
    order = []
    for index in range(args.workers):
        versions = args.versions if index % 2 == 0 else list(reversed(args.versions))
        order.extend(versions)
    experiment = dict(
        declared_at_utc=datetime.now(timezone.utc).isoformat(),
        diagnostic_only=True,
        source_hashes=hashes,
        sources={key: str(path) for key, path in sources.items()},
        harness_sha256=harness_hash(),
        prepare_full=args.prepare_full,
        snapshot_storage=args.snapshot_storage,
        gc_mode=args.gc_mode,
        samples=args.samples,
        prime_predecessor=args.prime_predecessor,
        fixed_calls=dict(
            scalar=args.scalar_calls,
            batch=args.batch_calls,
            host=args.host_calls,
            control=args.control_calls,
        ),
        order=order,
        cases=args.case or list(DEFAULT_CASES),
        protocol=benchmark.phase13.read(PROTOCOL),
    )
    benchmark.phase13.save(area / "experiment_manifest.json", experiment)
    counts, records = {version: 0 for version in sources}, {}
    for sequence, version in enumerate(order):
        counts[version] += 1
        key = version + str(counts[version])
        directory = area / "w" / key
        directory.mkdir(parents=True)
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
            "--gc-mode",
            args.gc_mode,
            "--snapshot-storage",
            args.snapshot_storage,
            "--samples",
            str(args.samples),
            "--scalar-calls",
            str(args.scalar_calls),
            "--batch-calls",
            str(args.batch_calls),
            "--host-calls",
            str(args.host_calls),
            "--control-calls",
            str(args.control_calls),
        ]
        if args.prepare_full:
            command.append("--prepare-full")
        if args.prime_predecessor:
            command.append("--prime-predecessor")
        for pattern in args.case or ():
            command.extend(("--case", pattern))
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
            raise RuntimeError(f"Worker failed: {directory / 'worker.log'}")
        record = benchmark.phase13.read(record_path)
        assert (
            record["source_sha256"] == hashes[version]
            and record["harness_sha256"] == experiment["harness_sha256"]
        )
        records[key] = record
    reference = records["frozen1"]
    correctness = {}
    published = records.get("published1")
    for key, record in records.items():
        assert record["environment"] == reference["environment"]
        correctness[key] = {}
        if not key.startswith("published"):
            correctness[key].update(
                {
                    name
                    + "/frozen_complete": benchmark.compare_complete(
                        reference["complete_snapshots"][name],
                        snapshot,
                        benchmark.phase13.read(benchmark.PROTOCOL),
                    )
                    for name, snapshot in record["complete_snapshots"].items()
                }
            )
        if published is not None:
            for case in published["manifest"]:
                if case["comparable"]:
                    correctness[key][case["name"] + "/published_common"] = (
                        benchmark.phase13.compare_snapshots(
                            published["snapshots"][case["name"]],
                            record["snapshots"][case["name"]],
                            case,
                        )
                    )
    failures = {
        key: {
            name: value["mismatches"]
            for name, value in result.items()
            if value["mismatches"]
        }
        for key, result in correctness.items()
    }
    failures = {key: value for key, value in failures.items() if value}
    benchmark.phase13.save(
        area / "correctness.json", dict(results=correctness, failures=failures)
    )
    assert not failures, "Numerical/metadata parity failed"
    cases = {}
    for name in reference["timings"]:
        groups = {
            version: [
                record["timings"][name]["median"]
                for key, record in records.items()
                if key.startswith(version)
            ]
            for version in sources
        }
        medians = {
            version: statistics.median(values) for version, values in groups.items()
        }
        cases[name] = dict(
            worker_medians_us=groups,
            median_us=medians,
            current_to_frozen_ratio=medians["current"] / medians["frozen"],
        )
    benchmark.phase13.save(
        area / "summary.json",
        dict(
            experiment=experiment,
            diagnostic_only=True,
            correctness_passed=True,
            cases=cases,
            worker_records={
                key: str(area / "w" / key / "record.json") for key in records
            },
            interpretation="Instrumented attribution only. No acceptance budgets are changed, no samples discarded and no ratios normalized by controls.",
        ),
    )
    print(
        json.dumps(
            dict(
                summary=str(area / "summary.json"),
                correctness_passed=True,
                diagnostic_only=True,
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
        default=benchmark.phase13.SUPPORT / "baseline/baseline_manifest.json",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=benchmark.VERSIONS,
        default=["frozen", "current"],
    )
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--scalar-calls", type=int, default=48)
    parser.add_argument("--batch-calls", type=int, default=16)
    parser.add_argument("--host-calls", type=int, default=1)
    parser.add_argument("--control-calls", type=int, default=48)
    parser.add_argument("--prepare-full", action="store_true")
    parser.add_argument(
        "--snapshot-storage",
        choices=("memory", "disk"),
        default="memory",
        help="Retain validation vectors in memory or release them during timing and restore from the verified sidecar afterward",
    )
    parser.add_argument(
        "--prime-predecessor",
        action="store_true",
        help="Untimed replay of each selected case's immediate original full-mix predecessor, respecting forward/reverse order",
    )
    parser.add_argument(
        "--gc-mode", choices=("default", "collected", "disabled"), default="default"
    )
    parser.add_argument("--case", action="append")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--version", choices=benchmark.VERSIONS, help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 12 or args.samples % 6:
        parser.error(
            "Use at least12 samples, divisible by6, to balance order and three numerical variants"
        )
    if (
        min(args.scalar_calls, args.batch_calls, args.host_calls, args.control_calls)
        < 1
    ):
        parser.error("Repetition counts must be positive")
    if not {"frozen", "current"}.issubset(args.versions) or len(
        set(args.versions)
    ) != len(args.versions):
        parser.error("Select frozen/current once each, optionally with published")
    if args.worker:
        worker(args)
    elif args.output_directory is None:
        parser.error("--output-directory is required")
    else:
        controller(args)


if __name__ == "__main__":
    main()
