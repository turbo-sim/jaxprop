"""Verify an installed wheel, then run tests without importing the source package."""

import argparse
from email.parser import BytesParser
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    wheel = Path(args.wheel).resolve()
    if wheel.is_dir():
        candidates = list(wheel.glob("jaxprop-*.whl"))
        assert len(candidates) == 1, candidates
        wheel = candidates[0]
    # Invoking this file puts only its directory, not the repository root, on
    # sys.path. Import before pytest loads any source-oriented test utilities.
    import jaxprop as jp

    installed = Path(jp.__file__).resolve().parent
    expected = Path(sys.prefix).resolve() / "Lib/site-packages/jaxprop"
    if sys.platform != "win32":
        import sysconfig

        expected = Path(sysconfig.get_path("purelib")) / "jaxprop"
    assert installed == expected.resolve(), (installed, expected)
    hashes = {}
    with zipfile.ZipFile(wheel) as archive:
        metadata_file, = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        artifact_version = BytesParser().parsebytes(archive.read(metadata_file))["Version"]
        for name in archive.namelist():
            if name.startswith("jaxprop/") and not name.endswith("/"):
                payload = archive.read(name)
                target = installed.parent / name
                assert target.read_bytes() == payload, name
                hashes[name] = hashlib.sha256(payload).hexdigest()
    assert jp.__version__ == importlib.metadata.version("jaxprop") == artifact_version
    receipt = dict(
        version=jp.__version__,
        imported_package=str(installed),
        wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        installed_files=hashes,
    )
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Verified installed JaxProp {jp.__version__}: {installed}", flush=True)
    if args.pytest_args:
        import pytest

        class InstalledPackage:
            def pytest_collection_finish(self, session):
                assert Path(jp.__file__).resolve().parent == installed
                print(f"Collected tests retain installed import: {jp.__file__}", flush=True)

        argv = args.pytest_args
        if argv[0] == "--":
            argv = argv[1:]
        raise SystemExit(pytest.main(argv, plugins=[InstalledPackage()]))


if __name__ == "__main__":
    main()
