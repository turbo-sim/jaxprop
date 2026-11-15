"""Native same-process attribution, separate from the independent-worker gate.

Current JaxProp is imported canonically first. Frozen and published packages use
distinct module namespaces while sharing numerical dependencies and one JAX
runtime. PG uses each package's own model. Optional bicubic cases compare only
current and frozen native evaluators, with identical source-physics files
verified for the canonical CoolProp setup dependency. Published bicubic is
excluded because its absolute import would change its timed state class.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

import phase14_benchmark as benchmark


def alias_package(name, source):
    package = source / "jaxprop"
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve().parent == package.resolve()
    return module


def build_cases(package, api, model, jax, jnp, eqx):
    """Retain Phase13 PG return contracts, including real new statuses."""

    def hp(h, p):
        return api.evaluate(model, package.HmassP_INPUTS, h, p)

    cp = float(model.constants.gamma * model.constants.R / (model.constants.gamma - 1))
    h = cp * jnp.linspace(600.0, 610.0, 1024)
    p = jnp.linspace(6e6, 6.2e6, 1024)
    cases = {
        "full_HP_batch1024": dict(
            fn=jax.jit(hp), variants=[(h, p), (h + cp * 10.0, p + 2e5)], batched=True
        )
    }
    numerical, static = eqx.partition(model, eqx.is_array)
    changed = eqx.tree_at(
        lambda value: value.constants.mu_ref, model, model.constants.mu_ref * 1.001
    )
    changed_numerical, _ = eqx.partition(changed, eqx.is_array)
    traces = []

    def dynamic(buffers, h, p):
        traces.append(1)
        return api.evaluate(eqx.combine(buffers, static), package.HmassP_INPUTS, h, p)

    cases["dynamic_HP_batch1024"] = dict(
        fn=jax.jit(dynamic),
        variants=[(buffers, h, p) for buffers in (numerical, changed_numerical)],
        batched=True,
        traces=traces,
    )

    def machine(h, p, ratio, efficiency, expansion):
        inlet = hp(h, p)
        ideal = api.evaluate(
            model, package.PSmass_INPUTS, p * ratio, api.raw(inlet, "entropy")
        )
        increment = api.raw(ideal, "enthalpy") - h
        actual_h = h + (efficiency * increment if expansion else increment / efficiency)
        outlet = hp(actual_h, p * ratio)
        return dict(
            work=(h - actual_h) if expansion else (actual_h - h),
            outlet_temperature=api.raw(outlet, "temperature"),
            outlet_pressure=api.raw(outlet, "pressure"),
            status_inlet=api.status(inlet),
            status_ideal=api.status(ideal),
            status_outlet=api.status(outlet),
        )

    for expansion, label in ((False, "compressor"), (True, "expander")):
        fn = jax.jit(
            lambda h, p, ratio, efficiency, expansion=expansion: machine(
                h, p, ratio, efficiency, expansion
            )
        )
        variants = [
            (
                jnp.asarray(cp * (600.0 + offset)),
                jnp.asarray(6e6 + 2e4 * offset),
                jnp.asarray(0.95 if expansion else 1.1),
                jnp.asarray(0.85),
            )
            for offset in (0.0, 10.0)
        ]
        cases[label + "_with_actual_status"] = dict(
            fn=fn, variants=variants, batched=False
        )

    def heat_exchanger(h_cold, h_hot, pressure, duty):
        fraction = jnp.linspace(0.0, 1.0, 128)
        cold_h, hot_h = h_cold + fraction * duty, h_hot - (1.0 - fraction) * duty
        cold, hot = hp(cold_h, pressure), hp(hot_h, pressure * 1.05)
        delta = api.raw(hot, "temperature") - api.raw(cold, "temperature")
        return dict(
            temperature_difference=delta,
            minimum_delta_T=jnp.min(delta),
            heat_absorbed=api.raw(cold, "enthalpy")[-1] - api.raw(cold, "enthalpy")[0],
            heat_released=api.raw(hot, "enthalpy")[-1] - api.raw(hot, "enthalpy")[0],
            status_cold=api.status(cold),
            status_hot=api.status(hot),
        )

    hx_inputs = [
        (
            jnp.asarray(cp * 600.0 + offset),
            jnp.asarray(cp * 660.0 + offset),
            jnp.asarray(8e6),
            jnp.asarray(20e3),
        )
        for offset in (0.0, 1e3)
    ]
    cases["heat_exchanger128_with_actual_status"] = dict(
        fn=jax.jit(heat_exchanger), variants=hx_inputs, batched=True
    )
    for name, original in list(cases.items()):
        if not name.endswith("with_actual_status"):
            continue
        for projection in ("numerical_only", "status_only"):
            if api.legacy and projection == "status_only":
                continue  # Legacy absence must not become a constant fake test.

            def selected(*arguments, original=original, projection=projection):
                result = original["fn"](*arguments)
                return {
                    key: value
                    for key, value in result.items()
                    if key.startswith("status") == (projection == "status_only")
                }

            cases[name + "_diagnostic_" + projection] = dict(
                fn=jax.jit(selected),
                variants=original["variants"],
                batched=original["batched"],
                diagnostic=True,
            )
    return cases


def build_bicubic_cases(package, api, model, jax, jnp):
    """Current/frozen native BC only; published absolute imports are excluded."""

    def hp(h, p):
        return api.evaluate(model, package.HmassP_INPUTS, h, p)

    cases = {
        "bicubic_full_HP_batch1024": dict(
            fn=jax.jit(hp),
            batched=True,
            variants=[
                (
                    jnp.linspace(680e3 + offset, 710e3 + offset, 1024),
                    jnp.linspace(6e6, 7e6, 1024),
                )
                for offset in (0.0, 1e3)
            ],
        )
    }
    h_axis, p_axis = (
        jnp.linspace(680e3, 700e3, 8)[:, None],
        jnp.linspace(6e6, 7e6, 16)[None, :],
    )
    full_h, full_p = jnp.broadcast_arrays(h_axis, p_axis)
    for name, (h, p) in {
        "flat128": (full_h.ravel(), full_p.ravel()),
        "broadcast8x16": (full_h, full_p),
        "broadcast16x8": (full_h.reshape(16, 8), full_p.reshape(16, 8)),
        "independent_broadcast8x16": (h_axis, p_axis),
    }.items():
        cases["bicubic_HP_" + name] = dict(
            fn=jax.jit(hp), variants=[(h, p), (h + 1e3, p + 2e4)], batched=True
        )

    def heat_exchanger(h_cold, h_hot, pressure, duty):
        fraction = jnp.linspace(0.0, 1.0, 128)
        cold = hp(h_cold + fraction * duty, pressure)
        hot = hp(h_hot - (1.0 - fraction) * duty, pressure * 1.05)
        delta = api.raw(hot, "temperature") - api.raw(cold, "temperature")
        return dict(
            temperature_difference=delta,
            minimum_delta_T=jnp.min(delta),
            heat_absorbed=api.raw(cold, "enthalpy")[-1] - api.raw(cold, "enthalpy")[0],
            heat_released=api.raw(hot, "enthalpy")[-1] - api.raw(hot, "enthalpy")[0],
            status_cold=api.status(cold),
            status_hot=api.status(hot),
        )

    original = dict(
        fn=jax.jit(heat_exchanger),
        variants=[
            tuple(
                jnp.asarray(value)
                for value in (670e3 + offset, 730e3 + offset, 8e6, 20e3)
            )
            for offset in (0.0, 1e3)
        ],
        batched=True,
    )
    cases["bicubic_heat_exchanger128_with_actual_status"] = original
    for projection in ("numerical_only", "status_only"):

        def selected(*arguments, projection=projection):
            result = original["fn"](*arguments)
            return {
                key: value
                for key, value in result.items()
                if key.startswith("status") == (projection == "status_only")
            }

        cases["bicubic_heat_exchanger128_diagnostic_" + projection] = dict(
            fn=jax.jit(selected),
            variants=original["variants"],
            batched=True,
            diagnostic=True,
        )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--published-source",
        type=Path,
        default=benchmark.ROOT / "benchmarks/results/published",
    )
    parser.add_argument(
        "--frozen-source",
        type=Path,
        default=benchmark.ROOT / "benchmarks/results/frozen",
    )
    parser.add_argument(
        "--frozen-sha256",
        default=benchmark.FROZEN_HASH,
        help="Expected exact source hash; explicitly select a later checkpoint for an isolated follow-up",
    )
    parser.add_argument("--current-source", type=Path, default=benchmark.ROOT)
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
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--packed-statuses",
        action="store_true",
        help="Add current/frozen consumer diagnostics that stack all real statuses into one output array",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("pg", "bicubic"),
        default=["pg"],
        help="Bicubic is current/frozen only because the published alias changes its absolute-import state class",
    )
    parser.add_argument("--scalar-calls", type=int, default=48)
    parser.add_argument("--batch-calls", type=int, default=16)
    args = parser.parse_args()
    if min(args.samples, args.scalar_calls, args.batch_calls) <= 0:
        parser.error("Sample/call counts must be positive")
    if len(set(args.versions)) != len(args.versions) or not {
        "frozen",
        "current",
    }.issubset(args.versions):
        parser.error(
            "Select frozen and current once each, optionally including published"
        )
    sources = {
        key: path.resolve()
        for key, path in dict(
            published=args.published_source,
            frozen=args.frozen_source,
            current=args.current_source,
        ).items()
        if key in args.versions
    }
    hashes = {
        key: benchmark.phase13.source_hash(source) for key, source in sources.items()
    }
    assert hashes["frozen"] == args.frozen_sha256
    baseline = (
        benchmark.verify_wheel(args.baseline_manifest, sources["published"])
        if "published" in sources
        else benchmark.phase13.read(args.baseline_manifest)
    )
    area = args.output_directory.resolve()
    area.mkdir(parents=True, exist_ok=True)
    assert not (
        area / "experiment_manifest.json"
    ).exists(), "Use a new paired diagnostic directory"
    harness = hashlib.sha256(
        benchmark.harness_hash().encode() + Path(__file__).read_bytes()
    ).hexdigest()
    benchmark.phase13.save(
        area / "experiment_manifest.json",
        dict(
            declared_at_utc=datetime.now(timezone.utc).isoformat(),
            source_hashes=hashes,
            frozen_reference_sha256=args.frozen_sha256,
            baseline=baseline,
            harness_sha256=harness,
            versions=args.versions,
            samples=args.samples,
            models=args.models,
            packed_statuses=args.packed_statuses,
            scalar_calls=args.scalar_calls,
            batch_calls=args.batch_calls,
            interpretation="Diagnostic only: co-loaded aliases share one JAX process and are not the independent-worker acceptance gate. Canonical current is imported first. Published bicubic is excluded because its absolute imports would replace the timed legacy state class. Current/frozen BC uses canonical CoolProp only during setup, with identical source-physics files verified.",
            ordering="Each sample rotates the initial version; successive groups reverse it. Function blocks use fixed matched repetitions and full synchronization.",
        ),
    )
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
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(area / ("cache_" + uuid.uuid4().hex))
    sys.path.insert(0, str(sources["current"]))
    import CoolProp.CoolProp as CP
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jaxprop as current
    import numpy as np
    from phase13.api import API

    assert Path(current.__file__).resolve().parent == sources["current"] / "jaxprop"
    assert jax.default_backend() == "cpu" and jax.config.x64_enabled
    jax.block_until_ready(jnp.asarray(0.0))
    packages = {"current": current}
    for version in ("frozen", "published"):
        if version in sources:
            packages[version] = alias_package("_phase14_" + version, sources[version])
    contexts, cases, snapshots, complete, graphs = {}, {}, {}, {}, {}
    for version in args.versions:
        package = packages[version]
        api = API(package, version == "published", CP, jax, jnp, np, eqx)
        local = argparse.Namespace(
            record=area / version / "record.json",
            calls_scale=1,
            scalar_calls=args.scalar_calls,
            batch_calls=args.batch_calls,
            host_calls=1,
            diagnostics=False,
            suite="native",
            case=None,
            validate_only=False,
            samples=args.samples,
        )
        ctx = benchmark.Context(local, package, CP, jax, jnp, np, eqx, api)
        contexts[version], cases[version] = ctx, {}
        if "pg" in args.models:
            model = api.make_pg("CO2", 600.0, 1e5, matched=True)
            assert type(model).__module__.startswith(package.__name__ + ".perfect_gas")
            cases[version].update(build_cases(package, api, model, jax, jnp, eqx))
        if "bicubic" in args.models and version != "published":
            for source_file in (sources["frozen"] / "jaxprop/coolprop").glob("*.py"):
                assert (
                    source_file.read_bytes()
                    == (
                        sources["current"] / "jaxprop/coolprop" / source_file.name
                    ).read_bytes()
                ), "Alias BC setup requires identical frozen/current source physics"
            options = dict(
                h_min=600e3,
                h_max=850e3,
                p_min=5e6,
                p_max=12e6,
                N_h=16,
                N_p=16,
                N_p_sat=4,
                coarse_step=1,
                gradient_method="central",
                table_dir=str(area / version / "tables"),
            )
            model = api.make_table("CO2", "HEOS", options)
            assert type(model).__module__.startswith(package.__name__ + ".bicubic")
            cases[version].update(build_bicubic_cases(package, api, model, jax, jnp))
        if args.packed_statuses and version != "published":
            for name, original in list(cases[version].items()):
                if not name.endswith("_with_actual_status"):
                    continue
                status_order = (
                    ["status_cold", "status_hot"]
                    if "heat_exchanger" in name
                    else ["status_inlet", "status_ideal", "status_outlet"]
                )

                def packed(*arguments, original=original, status_order=status_order):
                    result = original["fn"](*arguments)
                    # Keep every stage/row code; only output-buffer packaging
                    # changes. All original numerical guards remain required.
                    statuses = [result[key] for key in status_order]
                    numerical = {
                        key: value
                        for key, value in result.items()
                        if not key.startswith("status")
                    }
                    numerical["status_packed"] = jnp.stack(statuses, axis=0)
                    return numerical

                cases[version][name + "_diagnostic_packed_status"] = dict(
                    fn=jax.jit(packed),
                    variants=original["variants"],
                    batched=original["batched"],
                    diagnostic=True,
                    packed_status_order=status_order,
                )
    # Reuse the identical numerical input buffers, not merely equal values.
    # Dynamic models keep their own genuine coefficient PyTrees; only their
    # physical HP arguments are shared across implementations.
    for version in args.versions:
        ctx = contexts[version]
        for name, case in cases[version].items():
            shared = cases["frozen"][name]["variants"]
            case["variants"] = (
                [(own[0], *common[1:]) for own, common in zip(case["variants"], shared)]
                if name == "dynamic_HP_batch1024"
                else list(shared)
            )
        snapshots[version], complete[version], graphs[version] = {}, {}, {}
        for name, case in cases[version].items():
            outputs = [
                jax.block_until_ready(case["fn"](*inputs))
                for inputs in case["variants"]
            ]
            fields = (
                benchmark.phase13.PG_FIELDS
                if "HP_batch1024" in name
                else benchmark.phase13.CORE
            )
            snapshots[version][name] = [
                ctx.snapshot(output, fields, name=name) for output in outputs
            ]
            if version != "published":
                complete[version][name] = [
                    ctx.complete_snapshot(output, path=name + "." + str(index))
                    for index, output in enumerate(outputs)
                ]
            executable = case["fn"].lower(*case["variants"][0]).compile().as_text()
            (area / version / (name + ".hlo.txt")).write_text(
                executable, encoding="utf-8"
            )
            graphs[version][name] = dict(
                entry_layout=executable.splitlines()[0],
                text_sha256=hashlib.sha256(executable.encode()).hexdigest(),
            )
            for index in range(8):
                jax.block_until_ready(
                    case["fn"](*case["variants"][index % len(case["variants"])])
                )
            if "traces" in case:
                assert len(case["traces"]) == 1
    protocol = benchmark.phase13.read(benchmark.PROTOCOL)
    correctness = {}
    for version in args.versions:
        checks = {}
        for name in cases[version]:
            if version != "published":
                checks[name + "/frozen_complete"] = benchmark.compare_complete(
                    complete["frozen"][name], complete[version][name], protocol
                )
            if "published" in cases and name in cases["published"]:
                fields = (
                    benchmark.phase13.PG_FIELDS
                    if "HP_batch1024" in name
                    else benchmark.phase13.CORE
                )
                checks[name + "/published_common"] = (
                    benchmark.phase13.compare_snapshots(
                        snapshots["published"][name],
                        snapshots[version][name],
                        dict(name=name, rtol=1e-7, atol=1e-7, fields=fields),
                    )
                )
        correctness[version] = checks
    failures = {
        version: {
            name: result["mismatches"]
            for name, result in checks.items()
            if result["mismatches"]
        }
        for version, checks in correctness.items()
    }
    failures = {key: value for key, value in failures.items() if value}
    benchmark.phase13.save(
        area / "correctness.json", dict(results=correctness, failures=failures)
    )
    if failures:
        raise RuntimeError(
            f"Paired diagnostic values differ; inspect {area / 'correctness.json'}"
        )
    names = list(cases["frozen"])
    samples = {
        version: {name: [] for name in cases[version]} for version in args.versions
    }
    host = []
    for sample in range(args.samples):
        order = (
            args.versions[sample % len(args.versions) :]
            + args.versions[: sample % len(args.versions)]
        )
        if sample // len(args.versions) % 2:
            order = list(reversed(order))
        host.append(dict(sample=sample, before=benchmark.host_observation()))
        for name in (names if sample % 2 == 0 else list(reversed(names))):
            for version in order:
                if name not in cases[version]:
                    continue
                case = cases[version][name]
                count = args.batch_calls if case["batched"] else args.scalar_calls
                start = time.perf_counter_ns()
                for call in range(count):
                    jax.block_until_ready(
                        case["fn"](
                            *case["variants"][(sample + call) % len(case["variants"])]
                        )
                    )
                samples[version][name].append(
                    (time.perf_counter_ns() - start) / count / 1e3
                )
        host[-1]["after"] = benchmark.host_observation()
    assert {
        key: benchmark.phase13.source_hash(source) for key, source in sources.items()
    } == hashes
    assert (
        hashlib.sha256(
            benchmark.harness_hash().encode() + Path(__file__).read_bytes()
        ).hexdigest()
        == harness
    )
    results = {}
    for name in names:
        medians = {
            version: statistics.median(values[name])
            for version, values in samples.items()
            if name in values
        }
        results[name] = dict(
            median_us=medians,
            raw_samples_us={
                version: values[name]
                for version, values in samples.items()
                if name in values
            },
            candidate_to_frozen_ratio=medians["current"] / medians["frozen"],
        )
        if "published" in medians:
            results[name]["candidate_to_published_ratio"] = (
                medians["current"] / medians["published"]
            )
    environment = {
        name: importlib.metadata.version(name)
        for name in ("jax", "jaxlib", "equinox", "numpy", "CoolProp")
    }
    environment.update(
        python=sys.version, device=str(jax.devices()[0]), x64=jax.config.x64_enabled
    )
    benchmark.phase13.save(
        area / "summary.json",
        dict(
            diagnostic_only=True,
            source_hashes=hashes,
            harness_sha256=harness,
            environment=environment,
            cases=results,
            correctness_passed=True,
            graphs=graphs,
            host_observations=host,
            imported_packages={
                key: str(package.__file__) for key, package in packages.items()
            },
            semantic_differences="Actual legacy consumer outputs have no status; legacy status-only cases are omitted. Full PG returns keep their actual13-vs22 materialization.",
            packed_status_layouts={
                version: {
                    name: case["packed_status_order"]
                    for name, case in values.items()
                    if "packed_status_order" in case
                }
                for version, values in cases.items()
            },
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


if __name__ == "__main__":
    main()
