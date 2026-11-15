"""Fresh-process public-call and separate lowering/compilation diagnostics.

Each version/profile/mode uses a new process and UUID compilation cache. The
public mode preserves the complete first synchronized public call. Staged mode
times a deliberately different path and must not be summed and relabeled as the
public observation. Phase 13's pinned wheel and Phase 14's exact-state checks are
reused; this diagnostic never changes the production evaluators.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

import phase14_benchmark as benchmark

ROOT = benchmark.ROOT
DEFAULT_PROFILES = (
    "pg_matched_compiled_HP_batch1024",
    "bicubic_CO2_compiled_HP_batch1024",
    "pg_matched_compiled_heat_exchanger_128cells",
    "bicubic_CO2_compiled_compressor",
)


def harness_hash():
    return hashlib.sha256(
        benchmark.harness_hash().encode() + b"\0" + Path(__file__).read_bytes()
    ).hexdigest()


def timed(function, jax):
    start = time.perf_counter_ns()
    value = function()
    jax.block_until_ready(value)
    return value, (time.perf_counter_ns() - start) / 1e3


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
    root = args.source.resolve()
    sys.path.insert(0, str(root))
    import CoolProp.CoolProp as CP
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jaxprop as jp
    import numpy as np
    from phase13 import native_cases
    from phase13.api import API

    assert Path(jp.__file__).resolve().parent == root / "jaxprop"
    assert jax.default_backend() == "cpu" and jax.config.x64_enabled
    before, harness = benchmark.phase13.source_hash(root), harness_hash()
    jax.block_until_ready(jnp.asarray(0.0))
    # Only the numerical Context's construction/registration facilities are
    # used. Calling execute() here would warm other public workloads.
    args.calls_scale = 1
    args.scalar_calls, args.batch_calls, args.host_calls = 48, 16, 1
    args.suite, args.diagnostics = "native", False
    ctx = benchmark.Context(
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
    benchmark.register_controls(ctx)
    matching = [case for case in ctx.cases if case["name"] == args.profile]
    assert len(matching) == 1, f"Expected one registered profile: {args.profile}"
    case = matching[0]
    assert (
        case["compiled"] is not None
    ), "Staged profiles require a compiled public function"
    fn, inputs = case["fn"], case["variants"][0]
    jax.block_until_ready(inputs)
    observations, events = {}, []

    class CompilerEvents(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if message.startswith("Compiling "):
                events.append(message)

    handler = CompilerEvents()
    logger = logging.getLogger("jax")
    if args.log_compiles:
        logger.addHandler(handler)
    host_before = benchmark.host_observation()
    try:
        with jax.log_compiles(args.log_compiles):
            if args.mode == "public":
                result, observations["first_synchronized_public_call_us"] = timed(
                    lambda: fn(*inputs), jax
                )
                executable = None
            else:
                start = time.perf_counter_ns()
                lowered = case["compiled"].lower(*inputs)
                observations["trace_and_lower_us"] = (
                    time.perf_counter_ns() - start
                ) / 1e3
                start = time.perf_counter_ns()
                executable = lowered.compile()
                observations["compile_lowered_us"] = (
                    time.perf_counter_ns() - start
                ) / 1e3
                result, observations["first_executable_dispatch_us"] = timed(
                    lambda: executable(*inputs), jax
                )
    finally:
        if args.log_compiles:
            logger.removeHandler(handler)
    host_after = benchmark.host_observation()
    # Both public and compiled-callable outputs retain the actual complete
    # state/status. Snapshot work and warmed replay are outside cold timers.
    snapshot = ctx.snapshot(result, case["fields"], name=args.profile)
    complete = (
        None if ctx.api.legacy else ctx.complete_snapshot(result, path=args.profile)
    )
    replay = fn if executable is None else executable
    for _ in range(8):
        jax.block_until_ready(replay(*inputs))
    samples = []
    for _ in range(args.replay_samples):
        start = time.perf_counter_ns()
        for _ in range(args.replay_calls):
            jax.block_until_ready(replay(*inputs))
        samples.append((time.perf_counter_ns() - start) / args.replay_calls / 1e3)
    observations["warmed_replay_us"] = samples
    observations["warmed_replay_median_us"] = statistics.median(samples)
    if executable is not None:
        graph = executable.as_text()
        (args.record.parent / "optimized.hlo.txt").write_text(graph, encoding="utf-8")
        graph_summary = dict(
            entry_layout=graph.splitlines()[0],
            text_sha256=hashlib.sha256(graph.encode()).hexdigest(),
        )
    else:
        graph_summary = None
    assert benchmark.phase13.source_hash(root) == before and harness_hash() == harness
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
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    )
    benchmark.phase13.save(
        args.record,
        dict(
            version=args.version,
            profile=args.profile,
            mode=args.mode,
            source_sha256=before,
            harness_sha256=harness,
            imported_package=str(jp.__file__),
            environment=environment,
            observations=observations,
            snapshot=snapshot,
            complete_snapshot=complete,
            reference_identities=ctx.reference_identities,
            case_manifest={
                key: value
                for key, value in case.items()
                if key not in {"fn", "variants", "compiled"}
            },
            compiler_logging_enabled=args.log_compiles,
            compiler_events=events if args.log_compiles else None,
            graph=graph_summary,
            host_before=host_before,
            host_after=host_after,
            condition="Fresh process/cache; device/model/fixture construction and input placement excluded. No public workload evaluated before the selected profile.",
            independent_oracles="Cross-version and frozen complete-array validation are performed by the controller; the already verified Phase13 oracles are not rerun inside these cold timers.",
        ),
    )


def controller(args):
    available = dict(
        published=args.published_source.resolve(),
        frozen=args.frozen_source.resolve(),
        current=args.current_source.resolve(),
    )
    sources = {key: available[key] for key in args.versions}
    if "published" in sources:
        baseline = benchmark.verify_wheel(args.baseline_manifest, sources["published"])
    else:
        baseline = benchmark.phase13.read(args.baseline_manifest)
    hashes = {key: benchmark.phase13.source_hash(path) for key, path in sources.items()}
    assert hashes["frozen"] == args.frozen_sha256
    area = args.output_directory.resolve()
    area.mkdir(parents=True, exist_ok=True)
    assert not (
        area / "experiment_manifest.json"
    ).exists(), "Use a new cold experiment directory"
    protocol = benchmark.phase13.read(benchmark.PROTOCOL)
    acceptance = benchmark.phase13.read(benchmark.phase13.SUPPORT / "acceptance.json")
    profiles, harness = args.case or list(DEFAULT_PROFILES), harness_hash()
    manifest = dict(
        declared_at_utc=datetime.now(timezone.utc).isoformat(),
        baseline=baseline,
        source_hashes=hashes,
        frozen_reference_sha256=args.frozen_sha256,
        harness_sha256=harness,
        profiles=profiles,
        modes=args.modes,
        workers_per_version=args.workers,
        replay_samples=args.replay_samples,
        replay_calls=args.replay_calls,
        compiler_logging_enabled=args.log_compiles,
        first_call_budget=acceptance["budgets"]["first_public_call"],
        interpretation="Public and staged modes use separate fresh workers. Do not sum diagnostic stage timings and relabel them as public-call observations. Stage replay uses the compiled executable; public replay uses the public function.",
    )
    benchmark.phase13.save(area / "experiment_manifest.json", manifest)
    records = {}
    for profile_index, profile in enumerate(profiles):
        for mode_index, mode in enumerate(args.modes):
            for block in range(args.workers):
                order = list(sources) if block % 2 == 0 else list(reversed(sources))
                for version in order:
                    key = f"{profile_index}_{mode_index}_{block}_{version}"
                    folder = area / "w" / key
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
                        "--profile",
                        profile,
                        "--mode",
                        mode,
                        "--replay-samples",
                        str(args.replay_samples),
                        "--replay-calls",
                        str(args.replay_calls),
                    ]
                    if args.log_compiles:
                        command.append("--log-compiles")
                    print("Starting cold " + key + " " + profile, flush=True)
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
                    record = benchmark.phase13.read(path)
                    assert (
                        record["source_sha256"] == hashes[version]
                        and record["harness_sha256"] == harness
                    )
                    records[key] = record
    environment = next(iter(records.values()))["environment"]
    assert all(record["environment"] == environment for record in records.values())
    comparisons, summaries = {}, {}
    for profile in profiles:
        observations = [
            record for record in records.values() if record["profile"] == profile
        ]
        frozen = next(
            record for record in observations if record["version"] == "frozen"
        )
        old = next(
            (record for record in observations if record["version"] == "published"),
            None,
        )
        for key, record in records.items():
            if record["profile"] != profile:
                continue
            results = {}
            if record["version"] != "published":
                results["frozen_complete"] = benchmark.compare_complete(
                    frozen["complete_snapshot"], record["complete_snapshot"], protocol
                )
            if old is not None:
                results["published_common"] = benchmark.phase13.compare_snapshots(
                    old["snapshot"], record["snapshot"], old["case_manifest"]
                )
            comparisons[key] = results
        groups = {}
        for mode in args.modes:
            metrics = {}
            keys = (
                ("first_synchronized_public_call_us",)
                if mode == "public"
                else (
                    "trace_and_lower_us",
                    "compile_lowered_us",
                    "first_executable_dispatch_us",
                )
            )
            for metric in (*keys, "warmed_replay_median_us"):
                values = {
                    version: [
                        record["observations"][metric]
                        for record in observations
                        if record["mode"] == mode and record["version"] == version
                    ]
                    for version in sources
                }
                metrics[metric] = dict(
                    raw_us=values,
                    median_us={
                        version: statistics.median(items)
                        for version, items in values.items()
                    },
                )
            groups[mode] = metrics
        if "public" in groups and "published" in sources:
            medians = groups["public"]["first_synchronized_public_call_us"]["median_us"]
            budget = acceptance["budgets"]["first_public_call"]
            limit = (
                medians["published"] * budget["ratio"]
                + budget["additional_microseconds"]
            )
            groups["original_public_gate"] = dict(
                limit_us=limit,
                current_us=medians["current"],
                above_budget=medians["current"] > limit,
            )
        summaries[profile] = groups
    failures = {
        key: {
            name: result["mismatches"]
            for name, result in values.items()
            if result["mismatches"]
        }
        for key, values in comparisons.items()
    }
    failures = {key: value for key, value in failures.items() if value}
    benchmark.phase13.save(
        area / "correctness.json", dict(results=comparisons, failures=failures)
    )
    if failures:
        raise RuntimeError(
            f"Cold output validation failed; inspect {area / 'correctness.json'}"
        )
    benchmark.phase13.save(
        area / "summary.json",
        dict(
            experiment=manifest,
            profiles=summaries,
            correctness_passed=True,
            records=records,
        ),
    )
    print(
        json.dumps(dict(summary=str(area / "summary.json"), correctness_passed=True)),
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
    parser.add_argument(
        "--frozen-sha256",
        default=benchmark.FROZEN_HASH,
        help="Expected exact source hash; explicitly select a later checkpoint for an isolated follow-up",
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
        default=list(benchmark.VERSIONS),
    )
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument(
        "--case",
        action="append",
        help="Exact registered compiled profile name; repeat to select several",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("public", "stages"), default=["public", "stages"]
    )
    parser.add_argument("--replay-samples", type=int, default=9)
    parser.add_argument("--replay-calls", type=int, default=16)
    parser.add_argument(
        "--log-compiles",
        action="store_true",
        help="Enable compiler-event logging; instrumentation overhead is reported explicitly",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--version", choices=benchmark.VERSIONS, help=argparse.SUPPRESS)
    parser.add_argument("--profile", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("public", "stages"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.replay_samples, args.replay_calls) <= 0:
        parser.error("Replay sample/call counts must be positive")
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
