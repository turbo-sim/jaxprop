"""Pin a live official PyPI wheel without changing an existing phase baseline."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

from packaging.version import Version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    manifest_path = destination / "baseline_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = destination / manifest["wheel_filename"]
        if (
            hashlib.sha256(artifact.read_bytes()).hexdigest()
            != manifest["wheel_sha256"]
        ):
            raise ValueError("The pinned wheel no longer matches its recorded SHA256.")
        print(
            f"Retaining pinned JaxProp {manifest['latest_non_yanked_stable']} baseline from {manifest['resolved_at_utc']}."
        )
        return
    endpoint = "https://pypi.org/pypi/jaxprop/json"
    request = urllib.request.Request(
        endpoint, headers={"User-Agent": "JaxProp-performance-verification/1"}
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read()
    metadata = json.loads(raw)
    versions = [
        Version(version)
        for version, files in metadata["releases"].items()
        if not Version(version).is_prerelease
        and not Version(version).is_devrelease
        and any(not item.get("yanked", False) for item in files)
    ]
    version = str(max(versions))
    wheels = [
        item
        for item in metadata["releases"][version]
        if item["packagetype"] == "bdist_wheel"
        and not item.get("yanked", False)
        and item["filename"].endswith("py3-none-any.whl")
    ]
    if len(wheels) != 1:
        raise ValueError(
            "Expected exactly one portable Python wheel for the latest stable release."
        )
    wheel = wheels[0]
    with urllib.request.urlopen(wheel["url"], timeout=45) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != wheel["digests"]["sha256"]:
        raise ValueError("Downloaded content differs from the official PyPI SHA256.")
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / wheel["filename"]
    artifact.write_bytes(content)
    with zipfile.ZipFile(artifact) as archive:
        package_metadata = archive.read(
            f"jaxprop-{version}.dist-info/METADATA"
        ).decode()
    manifest = dict(
        resolved_at_utc=datetime.now(timezone.utc).isoformat(),
        source=endpoint,
        pypi_project_url="https://pypi.org/project/jaxprop/",
        api_current_version=metadata["info"]["version"],
        latest_non_yanked_stable=version,
        wheel_filename=artifact.name,
        wheel_url=wheel["url"],
        wheel_sha256=digest,
        wheel_bytes=len(content),
        uploaded_utc=wheel["upload_time_iso_8601"],
        requires_python=wheel["requires_python"],
        requires_dist=[
            line[len("Requires-Dist: ") :]
            for line in package_metadata.splitlines()
            if line.startswith("Requires-Dist: ")
        ],
        pypi_metadata_sha256=hashlib.sha256(raw).hexdigest(),
        installed=False,
    )
    (destination / "pypi_metadata.json").write_bytes(raw)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
