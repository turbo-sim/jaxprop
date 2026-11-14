"""Run catalogued examples in isolated, headless processes.

Quick completion is reported as quick completion. Missing inputs, inactive
reference files, timeouts and exceptions are never counted as passing programs.
Elapsed times describe execution resources; this runner is not a benchmark.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXAMPLES_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLES_DIR.parent


def load_catalogue():
    """Read the explicit catalogue without importing scientific dependencies."""
    data = json.loads((EXAMPLES_DIR / "catalogue.json").read_text(encoding="utf-8"))
    for entry in data["examples"]:
        source = (EXAMPLES_DIR / entry["path"]).resolve()
        if not source.is_relative_to(EXAMPLES_DIR) or not source.is_file():
            raise ValueError(f"Invalid catalogue path: {entry['path']!r}")
    return data["examples"]


def package_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "jaxprop").rglob("*.py")):
        digest.update(
            path.relative_to(ROOT).as_posix().encode()
            + b"\0"
            + path.read_bytes()
            + b"\0"
        )
    return digest.hexdigest()


def framework_hash():
    digest = hashlib.sha256()
    for name in ("_common.py", "run_examples.py", "catalogue.json"):
        digest.update(
            name.encode() + b"\0" + (EXAMPLES_DIR / name).read_bytes() + b"\0"
        )
    return digest.hexdigest()


def run_example(
    entry, *, profile, output_dir, table_dir, input_path=None, timeout=None
):
    """Execute one real program and retain its logs, profile and actual outcome."""
    source = EXAMPLES_DIR / entry["path"]
    result = dict(
        example=entry["path"],
        kind=entry["kind"],
        profile=profile,
        example_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        package_sha256=package_hash(),
        framework_sha256=framework_hash(),
    )
    if entry["kind"] == "reference":
        return result | dict(
            outcome="reference_only",
            reason=entry["classification_reason"],
            executed=False,
        )
    if entry["kind"] == "input-tool" and input_path is None:
        return result | dict(
            outcome="input_required",
            reason="Supply --input-for PATH=FILE for this input-dependent tool.",
            executed=False,
        )
    folder = output_dir / source.parent.name / source.stem
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "example_result.json"
    # A rerun must not accept a previous process's successful result.
    manifest.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(source),
        "--profile",
        profile,
        "--headless",
        "--output-dir",
        str(folder),
        "--table-dir",
        str(table_dir),
    ]
    if input_path is not None:
        command.extend(["--input", str(input_path)])
    environment = os.environ.copy()
    environment.update(
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        JAX_ENABLE_X64="true",
        JAX_PLATFORM_NAME="cpu",
        MPLBACKEND="Agg",
        DISABLE_PLOTS="1",
    )
    limit = timeout if timeout is not None else entry["timeouts_seconds"][profile]
    result.update(
        command=command,
        timeout_seconds=limit,
        executed=True,
        cwd=str(folder),
        timings_are_benchmarks=False,
    )
    started = time.perf_counter()
    try:
        with (folder / "stdout.log").open("w", encoding="utf-8") as stdout, (
            folder / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            process = subprocess.run(
                command,
                cwd=folder,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=limit,
            )
        result["exit_code"] = process.returncode
        if process.returncode != 0:
            result["outcome"] = "failed"
        elif not manifest.is_file():
            result.update(
                outcome="failed",
                reason="Program exited without its execution/figure manifest.",
            )
        else:
            execution = json.loads(manifest.read_text(encoding="utf-8"))
            if not execution.get("completed") or execution.get("profile") != profile:
                result.update(
                    outcome="failed",
                    reason="Execution manifest does not certify this requested profile.",
                )
            elif not execution.get("checks"):
                result.update(
                    outcome="failed",
                    reason="No declared numerical or workflow checks were recorded.",
                )
            else:
                result.update(outcome="completed", execution=execution)
    except subprocess.TimeoutExpired:
        result.update(
            outcome="timeout",
            reason="Declared execution budget exhausted; no successful run is claimed.",
        )
    except (OSError, ValueError, TypeError) as error:
        result.update(outcome="failed", reason=f"{type(error).__name__}: {error}")
    result["seconds"] = time.perf_counter() - started
    result["package_unchanged"] = package_hash() == result["package_sha256"]
    result["example_unchanged"] = (
        hashlib.sha256(source.read_bytes()).hexdigest() == result["example_sha256"]
    )
    result["framework_unchanged"] = framework_hash() == result["framework_sha256"]
    if not all(
        result[name]
        for name in ("package_unchanged", "example_unchanged", "framework_unchanged")
    ):
        result.update(
            outcome="failed",
            reason="Source changed during execution; rerun the final version.",
        )
    (folder / "run.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="List every classified original example."
    )
    parser.add_argument(
        "--suite", choices=("smoke", "runnable", "all"), default="smoke"
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        help="Explicit catalogue paths or glob patterns; overrides --suite.",
    )
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "development/example_audit/results"
        / datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        help="Shared validated table cache, separate from case outputs.",
    )
    parser.add_argument(
        "--input-for",
        action="append",
        default=[],
        metavar="PATH=FILE",
        help="Explicit input for one catalogue tool; repeat as needed.",
    )
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=1)
    parser.add_argument(
        "--timeout",
        type=float,
        help="Explicit per-program execution budget override in seconds.",
    )
    args = parser.parse_args(argv)
    entries = load_catalogue()
    if args.list:
        for entry in entries:
            print(f"{entry['kind']:16s} {entry['path']:72s} {entry['purpose']}")
        return 0
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    inputs = {}
    for item in args.input_for:
        name, separator, filename = item.partition("=")
        if not separator or not filename:
            parser.error("--input-for requires catalogue/path.py=FILE")
        inputs[name] = Path(filename).resolve()
    if args.examples:
        selected = [
            entry
            for entry in entries
            if any(
                fnmatch.fnmatch(entry["path"], pattern)
                or fnmatch.fnmatch(Path(entry["path"]).stem, pattern)
                for pattern in args.examples
            )
        ]
        if not selected:
            parser.error("No catalogue entries match --examples")
    elif args.suite == "all":
        selected = entries
    elif args.suite == "runnable":
        selected = [
            entry
            for entry in entries
            if entry["kind"] not in ("reference", "input-tool")
        ]
    else:
        selected = [entry for entry in entries if entry["smoke"]]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table_dir = (
        args.table_dir.resolve() if args.table_dir is not None else output / "tables"
    )
    before = package_hash()
    summary = dict(
        date_utc=datetime.now(timezone.utc).isoformat(),
        profile=args.profile,
        suite=args.suite,
        package_sha256=before,
        concurrent_jobs=args.jobs,
        timings_are_benchmarks=False,
        outcomes=[],
    )
    print(
        f"Running {len(selected)} catalogued entries, profile={args.profile}, jobs={args.jobs}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {
            pool.submit(
                run_example,
                entry,
                profile=args.profile,
                output_dir=output,
                table_dir=table_dir,
                input_path=inputs.get(entry["path"]),
                timeout=args.timeout,
            ): entry
            for entry in selected
        }
        for future in as_completed(pending):
            row = future.result()
            summary["outcomes"].append(row)
            print(
                f"{row['outcome']:16s} {row['example']} ({row.get('seconds', 0):.1f} s)",
                flush=True,
            )
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
    summary["outcomes"].sort(key=lambda row: row["example"])
    summary["counts"] = {
        name: sum(row["outcome"] == name for row in summary["outcomes"])
        for name in (
            "completed",
            "failed",
            "timeout",
            "input_required",
            "reference_only",
        )
    }
    summary["package_unchanged"] = package_hash() == before
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["counts"]), flush=True)
    return (
        0
        if summary["package_unchanged"]
        and not any(
            summary["counts"][name] for name in ("failed", "timeout", "input_required")
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
