"""Restore isolated benchmark references from the retained wheel and source archive.

Use the same Python environment for this command and the timing runners.
Existing references are verified on every invocation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import phase14_benchmark as benchmark

ROOT = Path(__file__).resolve().parents[1]


def restore(archive_path, destination, verify):
    """Verify a staged extraction before making it available to workers."""
    if destination.exists():
        verify(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=destination.parent) as temp:
        stage = Path(temp).resolve()
        assert stage.parent == destination.parent.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (stage / member.filename).resolve()
                if not target.is_relative_to(stage):
                    raise ValueError(
                        f"Archive member escapes its directory: {member.filename}"
                    )
            archive.extractall(stage)
        verify(stage)
        stage.rename(destination)
        # TemporaryDirectory cleanup sees the now-absent staging directory.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "phase13/baseline",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    manifest_path = args.baseline_directory.resolve() / "baseline_manifest.json"
    manifest = benchmark.phase13.read(manifest_path)
    wheel = manifest_path.parent / manifest["wheel_filename"]
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != manifest["wheel_sha256"]:
        raise ValueError("Published wheel SHA256 differs from the pinned manifest")
    frozen = ROOT / "development/performance/phase14/frozen_source.zip"

    def verify_frozen(source):
        if benchmark.phase13.source_hash(source) != benchmark.FROZEN_HASH:
            raise ValueError(
                "Frozen source differs from the retained Phase 13 checkpoint"
            )

    restore(
        wheel,
        output / "published",
        lambda source: benchmark.verify_wheel(manifest_path, source),
    )
    restore(frozen, output / "frozen", verify_frozen)
    print(
        json.dumps(
            dict(
                published=str(output / "published"),
                frozen=str(output / "frozen"),
                baseline_manifest=str(manifest_path),
                version=manifest["latest_non_yanked_stable"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
