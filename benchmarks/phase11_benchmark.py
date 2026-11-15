"""Compare complete-state calls against a frozen package source snapshot.

Run from the checkout with its Python environment, for example::

    python benchmarks/phase11_benchmark.py \
        --baseline-source development/performance/phase11_baseline.zip \
        --output-directory development/performance/results/phase11/guard

The controller starts independent serial workers in ABBAABBA order. Each worker
imports only its selected package source, builds independent tables, checks the
numerical results and synchronizes every timed complete-state return. Setup and
first-call costs are recorded separately. ``--validate-only`` exercises exactly
the same workloads without collecting warmed timings. This is a development
checkpoint against a recorded source snapshot, not the PyPI release benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import zipfile


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted((root / "jaxprop").rglob("*.py")):
        digest.update(
            path.relative_to(root).as_posix().encode()
            + b"\0"
            + path.read_bytes()
            + b"\0"
        )
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def worker(arguments):
    # Configure both implementations identically before any numerical imports.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["MPLBACKEND"] = "Agg"
    root = arguments.source.resolve()
    sys.path.insert(0, str(root))

    import CoolProp.CoolProp as CP
    import jax
    import jax.numpy as jnp
    import jaxprop as jp
    import numpy as np

    assert Path(jp.__file__).resolve().parent == root / "jaxprop"
    assert jax.config.x64_enabled and jax.default_backend() == "cpu"
    before = source_hash(root)
    directory = arguments.record.resolve().parent
    directory.mkdir(parents=True, exist_ok=True)
    setup = {}
    first_calls = {}
    graphs = {}
    checks = {}
    table_payloads = {}
    cases = []

    def synchronized(value):
        return jax.block_until_ready(value)

    def construct(name, factory):
        started = time.perf_counter_ns()
        value = factory()
        synchronized(jax.tree_util.tree_leaves(value))
        setup[name] = (time.perf_counter_ns() - started) / 1e3
        return value

    def snapshot(value):
        if hasattr(value, "available_properties"):
            result = {}
            for name in sorted(value.available_properties):
                array = np.asarray(value.raw_property(name)).ravel()
                indices = np.unique(
                    np.linspace(0, array.size - 1, min(16, array.size), dtype=int)
                )
                result[name] = [
                    None if np.isnan(x) else float(x) for x in array[indices]
                ]
                assert not np.any(np.isinf(array)), name
            result["status"] = np.asarray(value.status).ravel().tolist()
            result["clipped"] = np.asarray(value.clipped).ravel().tolist()
            assert np.all(np.asarray(value.status) == 0)
            for name in (
                "pressure",
                "temperature",
                "density",
                "enthalpy",
                "entropy",
                "internal_energy",
            ):
                assert np.all(np.isfinite(np.asarray(value.raw_property(name)))), name
            return result
        if isinstance(value, (tuple, list)):
            return [snapshot(item) for item in value]
        array = np.asarray(value)
        assert np.all(np.isfinite(array))
        return array.tolist()

    def fingerprint_table(model):
        """Check exact numerical construction, treating optional NaNs as paired.

        Generation manifests intentionally differ between package revisions.
        Numerical property arrays must retain their exact finite bit patterns;
        masks of unavailable entries are hashed separately from their values.
        The physical-domain cell policy is recorded but is not conflated with
        the coefficients it determines whether callers may interpolate.
        """
        arrays = {}

        def record(name, value):
            array = np.asarray(value)
            missing = np.isnan(array)
            canonical = np.where(missing, np.zeros((), dtype=array.dtype), array)
            digest = hashlib.sha256()
            digest.update(str(array.dtype).encode() + b"\0")
            digest.update(str(array.shape).encode() + b"\0")
            digest.update(missing.tobytes(order="C"))
            digest.update(canonical.tobytes(order="C"))
            arrays[name] = digest.hexdigest()

        payload = model.table
        for name in jp.PROPERTIES_CANONICAL:
            for component, value in payload[name].items():
                record(name + "/" + component, value)
        for name, entry in payload["saturation_props"].items():
            for component, value in entry.items():
                record("saturation/" + name + "/" + component, value)
        validity = np.asarray(payload["phase_cell_validity"])
        return dict(
            arrays=arrays,
            array_count=len(arrays),
            phase_cells=int(validity.size),
            phase_valid_cells=int(np.count_nonzero(validity)),
        )

    def add(name, evaluate, variants, calls=32, *, compiled=None):
        started = time.perf_counter_ns()
        initial = synchronized(evaluate(*variants[0]))
        first_calls[name] = (time.perf_counter_ns() - started) / 1e3
        checks[name] = [snapshot(initial)]
        checks[name].extend(
            snapshot(synchronized(evaluate(*inputs))) for inputs in variants[1:]
        )
        for index in range(8):
            synchronized(evaluate(*variants[index % len(variants)]))
        if compiled is not None:
            executable = compiled.lower(*variants[0]).compile().as_text()
            normalized = re.sub(r"metadata=\{[^{}]*\}", "", executable)
            normalized = "\n".join(
                line.strip() for line in normalized.splitlines()[1:] if line.strip()
            )
            graphs[name] = {
                "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                "instructions": len(
                    re.findall(r"^\s*%[^\n]*=", executable, re.MULTILINE)
                ),
                "contains_callback": "callback" in executable.lower(),
            }
            (directory / (name + ".hlo.txt")).write_text(executable, encoding="utf-8")
        cases.append(
            (name, evaluate, variants, max(1, round(calls * arguments.calls_scale)))
        )

    pg = construct(
        "perfect_gas_construction",
        lambda: jp.Fluid(
            "CO2", method="perfect_gas", model_options=dict(T_ref=600.0, p_ref=1e5)
        ),
    )
    table_options = dict(
        h_min=600e3,
        h_max=850e3,
        p_min=5e6,
        p_max=12e6,
        N_h=16,
        N_p=16,
        N_p_sat=4,
        coarse_step=1,
        gradient_method="central",
        table_dir=str(directory / "equilibrium_tables"),
    )
    bicubic = construct(
        "bicubic_fresh_generation",
        lambda: jp.Fluid("CO2", method="bicubic", table_options=table_options),
    )
    loaded = construct(
        "bicubic_initial_cached_loading",
        lambda: jp.Fluid("CO2", method="bicubic", table_options=table_options),
    )
    table_payloads["equilibrium_CO2_16x16"] = fingerprint_table(bicubic)
    for name in bicubic.capabilities.properties:
        np.testing.assert_array_equal(
            bicubic.get_state(jp.HmassP_INPUTS, 700e3, 8e6).raw_property(name),
            loaded.get_state(jp.HmassP_INPUTS, 700e3, 8e6).raw_property(name),
        )
    source = construct(
        "source_construction", lambda: jp.Fluid("CO2", method="coolprop")
    )
    callback = construct(
        "callback_construction", lambda: jp.Fluid("CO2", method="coolprop_jax")
    )
    water = construct(
        "water_source_construction", lambda: jp.Fluid("Water", method="coolprop")
    )

    def hp(model, enthalpy, pressure):
        return model.get_state(jp.HmassP_INPUTS, enthalpy, pressure, errors="mask")

    compiled_hp = jax.jit(hp)

    def hp_with_derivative(model, enthalpy, pressure):
        state = hp(model, enthalpy, pressure)
        derivative = jax.jvp(
            lambda h: hp(model, h, pressure).raw_property("temperature"),
            (enthalpy,),
            (jnp.ones_like(enthalpy),),
        )[1]
        return state, derivative

    compiled_derivative = jax.jit(hp_with_derivative)

    def compressor(model, enthalpy, pressure, pressure_ratio):
        first = hp(model, enthalpy, pressure)
        ideal = model.get_state(
            jp.PSmass_INPUTS,
            pressure * pressure_ratio,
            first.raw_property("entropy"),
            errors="mask",
        )
        actual = hp(
            model,
            enthalpy + (ideal.raw_property("enthalpy") - enthalpy) / 0.85,
            pressure * pressure_ratio,
        )
        return (
            actual.raw_property("enthalpy") - enthalpy,
            actual.raw_property("temperature"),
            first.status,
            ideal.status,
            actual.status,
        )

    compiled_compressor = jax.jit(compressor)
    for label, model in (("pg", pg), ("bicubic", bicubic)):
        scalar = [
            (model, jnp.asarray(h), jnp.asarray(p))
            for h, p in ((680e3, 6e6), (720e3, 8e6))
        ]
        batch = [
            (model, jnp.linspace(h, h + 30e3, 1024), jnp.linspace(p, p + 1e6, 1024))
            for h, p in ((680e3, 6e6), (720e3, 8e6))
        ]
        add(label + "_eager_hp_scalar", hp, scalar)
        add(label + "_eager_hp_batch1024", hp, batch, calls=12)
        add(label + "_compiled_hp_scalar", compiled_hp, scalar, compiled=compiled_hp)
        add(
            label + "_compiled_hp_batch1024",
            compiled_hp,
            batch,
            calls=12,
            compiled=compiled_hp,
        )
        add(
            label + "_changing_python_hp_scalar",
            hp,
            [(model, h, p) for h, p in ((680e3, 6e6), (720e3, 8e6))],
        )
        add(
            label + "_compiled_full_state_and_dTdh",
            compiled_derivative,
            scalar,
            compiled=compiled_derivative,
        )
        consumer_inputs = [
            (model, jnp.asarray(h), jnp.asarray(p), jnp.asarray(1.1))
            for h, p in ((680e3, 6e6), (720e3, 8e6))
        ]
        add(
            label + "_compiled_compressor_with_status",
            compiled_compressor,
            consumer_inputs,
            compiled=compiled_compressor,
        )
        for _, enthalpy, pressure in scalar:
            eager = synchronized(hp(model, enthalpy, pressure))
            complete = synchronized(compiled_hp(model, enthalpy, pressure))
            for name in eager.available_properties:
                np.testing.assert_allclose(
                    eager.raw_property(name),
                    complete.raw_property(name),
                    rtol=1e-11,
                    atol=1e-8,
                    equal_nan=True,
                )
        if label == "pg":
            cp = float(
                model.constants.gamma * model.constants.R / (model.constants.gamma - 1)
            )
            np.testing.assert_allclose(
                checks[label + "_compiled_full_state_and_dTdh"][0][1],
                1 / cp,
                rtol=1e-13,
            )

    pt_inputs = [
        (jnp.asarray(p), jnp.asarray(t)) for p, t in ((8e6, 500.0), (9e6, 520.0))
    ]
    pt_batches = [
        (jnp.linspace(p, p + 1e6, 16), jnp.linspace(t, t + 20, 16))
        for p, t in ((8e6, 500.0), (9e6, 520.0))
    ]
    source_pt = lambda pressure, temperature: source.get_state(
        jp.PT_INPUTS, pressure, temperature
    )
    callback_pt = jax.jit(
        lambda pressure, temperature: callback.get_state(
            jp.PT_INPUTS, pressure, temperature, errors="mask"
        )
    )
    add("source_pt_scalar", source_pt, pt_inputs, calls=16)
    add("source_pt_batch16", source_pt, pt_batches, calls=8)
    add(
        "callback_compiled_pt_scalar",
        callback_pt,
        pt_inputs,
        calls=16,
        compiled=callback_pt,
    )
    add(
        "callback_compiled_pt_batch16",
        callback_pt,
        pt_batches,
        calls=8,
        compiled=callback_pt,
    )

    def complete_callback_and_dhdT(model, pressure, temperature):
        state = model.get_state(jp.PT_INPUTS, pressure, temperature, errors="mask")
        derivative = jax.jvp(
            lambda value: model.get_state(
                jp.PT_INPUTS, pressure, value, errors="mask"
            ).raw_property("enthalpy"),
            (temperature,),
            (jnp.ones_like(temperature),),
        )[1]
        return state, derivative

    callback_derivative = jax.jit(
        lambda pressure, temperature: complete_callback_and_dhdT(
            callback, pressure, temperature
        )
    )
    add(
        "callback_compiled_full_state_and_dhdT_scalar",
        callback_derivative,
        pt_inputs,
        calls=8,
        compiled=callback_derivative,
    )
    add(
        "callback_compiled_full_state_and_dhdT_batch16",
        callback_derivative,
        pt_batches,
        calls=4,
        compiled=callback_derivative,
    )
    for inputs in pt_inputs:
        state, derivative = synchronized(callback_derivative(*inputs))
        np.testing.assert_allclose(
            derivative, state.raw_property("isobaric_heat_capacity"), rtol=2e-5
        )

    callback_components = construct(
        "binary_callback_component_construction",
        lambda: tuple(
            jp.Fluid(name, method="coolprop_jax") for name in ("Water", "Nitrogen")
        ),
    )
    binary_callback = construct(
        "binary_callback_construction",
        lambda: jp.Fluid(
            "Water/Nitrogen",
            method="binary_mixture",
            model_options=dict(
                components=callback_components, mass_fractions=(0.3, 0.7)
            ),
        ),
    )
    binary_inputs = [
        (jnp.asarray(p), jnp.asarray(t)) for p, t in ((1e6, 350.0), (2e6, 360.0))
    ]
    binary_batches = [
        (jnp.linspace(p, p + 5e5, 16), jnp.linspace(t, t + 5, 16))
        for p, t in ((1e6, 350.0), (2e6, 360.0))
    ]
    binary_pt = jax.jit(
        lambda pressure, temperature: binary_callback.get_state(
            jp.PT_INPUTS, pressure, temperature, errors="mask"
        )
    )
    binary_derivative = jax.jit(
        lambda pressure, temperature: complete_callback_and_dhdT(
            binary_callback, pressure, temperature
        )
    )
    for label, inputs in (("scalar", binary_inputs), ("batch16", binary_batches)):
        add(
            "binary_callback_compiled_pt_" + label,
            binary_pt,
            inputs,
            calls=8 if label == "scalar" else 4,
            compiled=binary_pt,
        )
        add(
            "binary_callback_compiled_full_state_and_dhdT_" + label,
            binary_derivative,
            inputs,
            calls=4,
            compiled=binary_derivative,
        )
    for inputs in binary_inputs:
        state, derivative = synchronized(binary_derivative(*inputs))
        np.testing.assert_allclose(
            derivative, state.raw_property("isobaric_heat_capacity"), rtol=2e-5
        )
    raw = CP.AbstractState("HEOS", "CO2")
    raw.update(CP.PT_INPUTS, 8e6, 500.0)
    direct = source_pt(*pt_inputs[0])
    np.testing.assert_allclose(
        [direct.rho, direct.h, direct.s],
        [raw.rhomass(), raw.hmass(), raw.smass()],
        rtol=1e-12,
    )

    metastable = CP.AbstractState("HEOS", "Water")
    metastable.specify_phase(CP.iphase_liquid)
    metastable.update(CP.PT_INPUTS, 0.7e6, 450.0)
    rho, temperature, enthalpy = (
        metastable.rhomass(),
        metastable.T(),
        metastable.hmass(),
    )
    meta_rhot = lambda density, temp: water.get_state_metastable(
        "rho", density, "T", temp, supersaturation=False, generalize_quality=True
    )
    meta_ph = lambda pressure, h: water.get_state_metastable(
        "p",
        pressure,
        "h",
        h,
        rhoT_guess=(rho * 0.99, temperature * 1.001),
        solver_tolerance=1e-9,
        supersaturation=False,
        generalize_quality=True,
    )
    add("metastable_water_rhoT", meta_rhot, [(rho, temperature)], calls=16)
    add("metastable_water_PH_inverse", meta_ph, [(0.7e6, enthalpy)], calls=6)
    state = meta_ph(0.7e6, enthalpy)
    np.testing.assert_allclose(
        [state.rho, state.T, state.p, state.h],
        [rho, temperature, 0.7e6, enthalpy],
        rtol=1e-8,
    )

    # The original Nitrogen branch is stable throughout this small rectangle.
    # Fresh generation is kept separate from all warmed interpolation timings.
    metastable_options = dict(
        h_min=-120e3,
        h_max=-110e3,
        p_min=8e4,
        p_max=2e5,
        N_h=16,
        N_p=16,
        N_p_sat=4,
        coarse_step=1,
        gradient_method="forward",
        metastable_phase="liquid",
        table_dir=str(directory / "metastable_tables"),
    )
    metastable_table = construct(
        "metastable_liquid_fresh_generation",
        lambda: jp.FluidBicubic("Nitrogen", "HEOS", **metastable_options),
    )
    table_payloads["metastable_Nitrogen_liquid_16x16"] = fingerprint_table(
        metastable_table
    )

    samples = {}
    if not arguments.validate_only:
        for name, evaluate, variants, calls in cases:
            values = []
            for sample in range(arguments.samples):
                started = time.perf_counter_ns()
                for index in range(calls):
                    synchronized(evaluate(*variants[(index + sample) % len(variants)]))
                values.append((time.perf_counter_ns() - started) / (calls * 1e3))
            samples[name] = {
                "microseconds_per_call": values,
                "calls_per_sample": calls,
                "median": statistics.median(values),
            }
    assert source_hash(root) == before, "Package source changed during the worker"
    environment = {
        name: importlib.metadata.version(name)
        for name in ("jax", "jaxlib", "CoolProp", "equinox", "numpy", "scipy")
    }
    environment.update(
        python=sys.version,
        platform=platform.platform(),
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
    save_json(
        arguments.record,
        dict(
            source_sha256=before,
            import_path=str(Path(jp.__file__).resolve()),
            source_unchanged_during_run=True,
            validate_only=arguments.validate_only,
            environment=environment,
            setup_microseconds=setup,
            first_call_microseconds=first_calls,
            numerical_snapshots=checks,
            graphs=graphs,
            table_payloads=table_payloads,
            timings=samples,
            own_checks_passed=True,
        ),
    )


def compare_snapshots(left, right):
    finite = unavailable = 0
    mismatches = []

    def visit(a, b, path):
        nonlocal finite, unavailable
        if isinstance(a, dict):
            if set(a) != set(b):
                mismatches.append(path + ": keys differ")
                return
            for key in a:
                visit(a[key], b[key], path + "/" + key)
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                mismatches.append(path + ": shape differs")
                return
            for index, (first, second) in enumerate(zip(a, b)):
                visit(first, second, path + "/" + str(index))
        elif a is None or b is None:
            if a is None and b is None:
                unavailable += 1
            else:
                mismatches.append(path + ": availability differs")
        elif abs(a - b) <= 1e-7 + 1e-8 * abs(a):
            finite += 1
        else:
            mismatches.append(path + ": numerical values differ")

    visit(left, right, "")
    return dict(finite=finite, paired_unavailable=unavailable, mismatches=mismatches)


def controller(arguments):
    current = arguments.current_source.resolve()
    area = arguments.output_directory.resolve()
    area.mkdir(parents=True, exist_ok=True)
    baseline = area / "baseline_package"
    baseline.mkdir(exist_ok=True)
    with zipfile.ZipFile(arguments.baseline_source) as archive:
        for item in archive.infolist():
            destination = (baseline / item.filename).resolve()
            if (
                not destination.is_relative_to(baseline.resolve())
                or not item.filename.startswith("jaxprop/")
                or not item.filename.endswith(".py")
            ):
                raise ValueError(
                    "Baseline archive must contain package Python sources only"
                )
        archive.extractall(baseline)
    versions = dict(baseline=baseline, current=current)
    order = (
        "baseline",
        "current",
        "current",
        "baseline",
        "baseline",
        "current",
        "current",
        "baseline",
    )
    if arguments.workers != 4:
        order = tuple(
            version
            for _ in range(arguments.workers)
            for version in ("baseline", "current")
        )
    if arguments.validate_only:
        order = ("baseline", "current")
    counters = dict(baseline=0, current=0)
    records = {}
    frozen_hashes = {name: source_hash(root) for name, root in versions.items()}
    for version in order:
        counters[version] += 1
        name = version + str(counters[version])
        worker_directory = area / "workers" / name
        worker_directory.mkdir(parents=True, exist_ok=True)
        record = worker_directory / "record.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--source",
            str(versions[version]),
            "--record",
            str(record),
            "--samples",
            str(arguments.samples),
            "--calls-scale",
            str(arguments.calls_scale),
        ]
        if arguments.validate_only:
            command.append("--validate-only")
        started = time.perf_counter()
        print("Starting " + name, flush=True)
        with (worker_directory / "worker.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=worker_directory, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(
                "Worker failed; inspect " + str(worker_directory / "worker.log")
            )
        records[name] = json.loads(record.read_text(encoding="utf-8"))
        assert records[name]["source_sha256"] == frozen_hashes[version]
        print(
            name
            + " complete in "
            + str(round(time.perf_counter() - started, 2))
            + " s",
            flush=True,
        )
    correctness = []
    baseline_reference = records["baseline1"]["numerical_snapshots"]
    for name, record in records.items():
        correctness.append(
            dict(
                worker=name,
                **compare_snapshots(baseline_reference, record["numerical_snapshots"]),
            )
        )
    assert not any(item["mismatches"] for item in correctness), correctness
    payload_comparison = {}
    for table, baseline_payload in records["baseline1"]["table_payloads"].items():
        expected = baseline_payload["arrays"]
        mismatches = {
            name: [
                key
                for key in set(expected)
                | set(record["table_payloads"][table]["arrays"])
                if expected.get(key)
                != record["table_payloads"][table]["arrays"].get(key)
            ]
            for name, record in records.items()
        }
        assert not any(mismatches.values()), (table, mismatches)
        payload_comparison[table] = dict(
            arrays_compared_per_worker=len(expected),
            bitwise_equal_finite_values_and_paired_NaNs=True,
            mismatches=mismatches,
            baseline_array_sha256=expected,
        )
    cases = {}
    for case in records["baseline1"]["timings"]:
        groups = {
            version: [
                record["timings"][case]["median"]
                for name, record in records.items()
                if name.startswith(version)
            ]
            for version in versions
        }
        old, new = (statistics.median(groups[version]) for version in versions)
        cases[case] = dict(
            baseline_worker_medians_us=groups["baseline"],
            current_worker_medians_us=groups["current"],
            baseline_median_us=old,
            current_median_us=new,
            ratio=new / old,
            above_10_percent_budget=new / old > 1.10,
            paired_worker_ratios=[
                b / a for a, b in zip(groups["baseline"], groups["current"])
            ],
            calls_per_sample=records["baseline1"]["timings"][case]["calls_per_sample"],
            raw_samples_us={
                name: record["timings"][case]["microseconds_per_call"]
                for name, record in records.items()
            },
        )
    summary = dict(
        date_utc=datetime.now(timezone.utc).isoformat(),
        baseline_label=arguments.baseline_label,
        baseline_archive_sha256=hashlib.sha256(
            arguments.baseline_source.read_bytes()
        ).hexdigest(),
        benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes=frozen_hashes,
        method=dict(
            order=list(records),
            samples_per_worker=arguments.samples,
            warmup_calls=8,
            calls_scale=arguments.calls_scale,
            synchronization="jax.block_until_ready after every complete return",
            dynamic_values="Alternate two precreated input sets, except the fixed physical metastable root",
            comparison=dict(rtol=1e-8, atol=1e-7, paired_unavailable=True),
            regression_budget=1.10,
            validate_only=arguments.validate_only,
            setup="One initial construction/generation/cached-loading observation per worker, separate from warmed state timing",
        ),
        environment=records["baseline1"]["environment"],
        correctness=correctness,
        table_payload_comparison=payload_comparison,
        cases=cases,
        setup_microseconds={
            name: record["setup_microseconds"] for name, record in records.items()
        },
        first_call_microseconds={
            name: record["first_call_microseconds"] for name, record in records.items()
        },
        graphs={name: record["graphs"] for name, record in records.items()},
        record_locations={
            name: (
                str((area / "workers" / name / "record.json").relative_to(current))
                if (area / "workers" / name / "record.json").is_relative_to(current)
                else str(area / "workers" / name / "record.json")
            )
            for name in records
        },
    )
    save_json(area / "summary.json", summary)
    print(
        json.dumps(
            {
                "summary": str(area / "summary.json"),
                "cases": len(cases),
                "regressions": [
                    name
                    for name, value in cases.items()
                    if value["above_10_percent_budget"]
                ],
                "numerical_mismatches": 0,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument(
        "--baseline-label", default="632b7fd / accepted Phase 10 example follow-up"
    )
    parser.add_argument(
        "--current-source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--calls-scale", type=float, default=1.0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.workers < 1 or arguments.samples < 1 or arguments.calls_scale <= 0:
        parser.error("workers, samples and calls-scale must be positive")
    if arguments.worker:
        worker(arguments)
    elif arguments.baseline_source is None or arguments.output_directory is None:
        parser.error("--baseline-source and --output-directory are required")
    else:
        controller(arguments)


if __name__ == "__main__":
    main()
