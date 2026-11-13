"""Deterministic identities for the physical sources used to build tables.

Do not memoize these identities: CoolProp permits the application to change
global configuration, fluid definitions, and enthalpy/entropy references.
"""

import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re

import CoolProp.CoolProp as CP

from ..errors import InvalidConfigurationError
from ..source_references import (
    reference_anchor as _reference_anchor,
    source_reference_metadata,
)


def metadata_fingerprint(metadata):
    """Hash JSON-compatible metadata independently of dictionary insertion order."""
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path):
    """Hash file contents, not timestamps that can survive a replacement."""
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _loaded_refprop_library():
    """Locate the library actually loaded in this process where observable."""
    if os.name == "nt":
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel.GetModuleHandleW.restype = ctypes.c_void_p
        kernel.GetModuleFileNameW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_uint,
        ]
        kernel.GetModuleFileNameW.restype = ctypes.c_uint
        for name in ("REFPRP64.DLL", "REFPRP32.DLL", "REFPROP.DLL", "librefprop.dll"):
            handle = kernel.GetModuleHandleW(name)
            if handle:
                buffer = ctypes.create_unicode_buffer(32768)
                length = kernel.GetModuleFileNameW(handle, buffer, len(buffer))
                if 0 < length < len(buffer):
                    return Path(buffer.value)
    elif Path("/proc/self/maps").is_file():
        # A bounded process map read is not a filesystem search. Resolve only
        # the native library already loaded by CoolProp, not similarly named
        # Python extension modules.
        with Path("/proc/self/maps").open(encoding="utf-8") as stream:
            matches = {
                Path(line.rstrip().split(maxsplit=5)[-1])
                for line in stream
                if re.search(
                    r"/librefprop[^/]*\.(?:so|dylib)(?:\.\d+)*$",
                    line.rstrip(),
                    re.IGNORECASE,
                )
            }
        if len(matches) == 1:
            return next(iter(matches))
    return None


def _refprop_data_roots(configuration, library):
    """Resolve only configured/conventional roots; never scan the filesystem."""
    # CoolProp 7.2's loader gives COOLPROP_REFPROP_ROOT priority over the
    # alternative path configuration. Both are explicit source selections.
    root_override = os.environ.get("COOLPROP_REFPROP_ROOT")
    alternative = configuration.get("ALTERNATIVE_REFPROP_PATH")
    if root_override:
        return [Path(root_override)]
    if alternative:
        return [Path(alternative)]
    roots = []
    if os.environ.get("RPPREFIX"):
        roots.append(Path(os.environ["RPPREFIX"]))
    if library is not None:
        roots.append(Path(library).parent)
    if os.name == "nt":
        for variable in ("PROGRAMFILES(X86)", "PROGRAMFILES"):
            if os.environ.get(variable):
                roots.append(Path(os.environ[variable]) / "REFPROP")
        roots.extend(
            (Path(r"C:\Program Files (x86)\REFPROP"), Path(r"C:\Program Files\REFPROP"))
        )
    else:
        roots.insert(0, Path("/opt/refprop"))
    unique = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in unique and resolved.is_dir():
            unique.append(resolved)
    return unique


def _existing_file(directories, filenames):
    """Honor known filename/suffix precedence inside a bounded directory set."""
    for filename in filenames:
        for directory in directories:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Cannot resolve native file {filenames[0]!r}")


def _native_fluid_file(root, name):
    if Path(name).is_absolute():
        return Path(name)
    directories = [root / "fluids", root / "FLUIDS", root]
    suffix = Path(name).suffix.upper()
    if suffix in (".FLD", ".PPF"):
        return _existing_file(directories, (name.upper(), name, name.lower()))
    # This matches CoolProp's REFPROP SETUP suffix order, including the
    # pseudo-pure PPF fallback. Air must not be confused with AIR.MIX.
    return _existing_file(
        directories,
        tuple(name.upper() + ending for ending in ("", ".FLD", ".fld", ".PPF", ".ppf")),
    )


def _native_mixture_dependencies(root, name):
    path = (
        Path(name)
        if Path(name).is_absolute()
        else _existing_file(
            (root / "mixtures", root / "MIXTURES", root),
            (name, name.upper(), name.lower()),
        )
    )
    lines = [
        line.strip()
        for line in path.read_text(encoding="latin-1").splitlines()
        if line.strip()
    ]
    for index, line in enumerate(lines):
        if re.fullmatch(r"\d+", line) and 0 < int(line) <= 20:
            count = int(line)
            names = lines[index + 1 : index + 1 + count]
            if len(names) == count and all(
                Path(item).suffix.upper() in (".FLD", ".PPF") for item in names
            ):
                return path, [_native_fluid_file(root, item) for item in names]
    raise ValueError(f"Cannot establish component-file dependencies of {path.name!r}")


def _refprop_native_files(fluid_names, configuration):
    """Return complete native provenance or an explicit non-cacheable result."""
    environment = {
        name: os.environ.get(name) for name in ("COOLPROP_REFPROP_ROOT", "RPPREFIX")
    }
    try:
        library = _loaded_refprop_library()
        roots = _refprop_data_roots(configuration, library)
        if library is None:
            configured_library = configuration.get("ALTERNATIVE_REFPROP_LIBRARY_PATH")
            library_names = (
                ("REFPRP64.DLL",)
                if os.name == "nt"
                else ("librefprop.so", "librefprop.dylib")
            )
            explicit_root = os.environ.get("COOLPROP_REFPROP_ROOT")
            root_libraries = {
                path.resolve()
                for name in library_names
                if explicit_root and (path := Path(explicit_root) / name).is_file()
            }
            if len(root_libraries) == 1:
                library = root_libraries.pop()
            elif configured_library:
                library = Path(configured_library)
            else:
                candidates = {
                    path.resolve()
                    for root in roots
                    for name in library_names
                    if (path := root / name).is_file()
                }
                if len(candidates) != 1:
                    raise ValueError(
                        "Cannot uniquely identify the loaded REFPROP native library"
                    )
                library = candidates.pop()
        matches = []
        failures = []
        for root in roots:
            try:
                component_files = []
                for name in fluid_names:
                    if str(name).upper().endswith(".MIX"):
                        mixture, dependencies = _native_mixture_dependencies(root, name)
                        component_files.extend((mixture, *dependencies))
                    else:
                        component_files.append(_native_fluid_file(root, name))
                interaction = configuration.get("ALTERNATIVE_REFPROP_HMX_BNC_PATH")
                interaction = (
                    Path(interaction)
                    if interaction
                    else _existing_file(
                        (root / "fluids", root / "FLUIDS", root), ("HMX.BNC", "hmx.bnc")
                    )
                )
                if not interaction.is_file():
                    raise FileNotFoundError(
                        "The configured HMX.BNC file is unavailable"
                    )
                files = sorted({path.resolve() for path in component_files}, key=str)
                candidate = (root, interaction.resolve(), files)
                if not any(existing[1:] == candidate[1:] for existing in matches):
                    matches.append(candidate)
            except (OSError, ValueError) as error:
                failures.append(f"{root}: {error}")
                continue
        if not matches:
            detail = (
                "; ".join(failures[:3]) or "no configured/conventional data roots exist"
            )
            raise ValueError(
                f"Cannot resolve REFPROP component files and HMX.BNC: {detail}"
            )
        if len(matches) > 1:
            raise ValueError(
                "Cannot uniquely resolve REFPROP component files and HMX.BNC from multiple data roots"
            )
        root, interaction, component_files = matches[0]
        return {
            "available": True,
            "data_root": str(root),
            "environment": environment,
            "library": _file_identity(library),
            "interaction": _file_identity(interaction),
            "components": [_file_identity(path) for path in component_files],
        }
    except (OSError, ValueError, AttributeError, RuntimeError) as error:
        return {
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
            "environment": environment,
        }


def _configuration_error(fluid_name, request, reason, *, mixture=False):
    return InvalidConfigurationError(
        fluid_name=fluid_name,
        provider="bicubic",
        model="binary_mixture" if mixture else "real_fluid",
        request=request,
        reason=reason,
    )


def _component_signature(fluid_name, backend, configuration):
    try:
        state = CP.AbstractState(backend, fluid_name)
        canonical_names = list(state.fluid_names())
        mole_fractions = [float(value) for value in state.get_mole_fractions()]
        predefined_mixture = len(canonical_names) == 1 and canonical_names[
            0
        ].upper().endswith(".MIX")
        if (
            len(canonical_names) != len(mole_fractions) and not predefined_mixture
        ) or not mole_fractions:
            raise ValueError("The source did not provide a defined composition")
        if not all(math.isfinite(value) and value >= 0 for value in mole_fractions):
            raise ValueError("The source provided an invalid composition")
        if not math.isclose(sum(mole_fractions), 1.0, abs_tol=1e-12, rel_tol=0.0):
            raise ValueError("Source mole fractions do not sum to one")
        reference = source_reference_metadata(fluid_name, backend, state=state)
        signature = {
            "backend": state.backend_name(),
            "fluid_names": canonical_names,
            "composition": {"basis": "mole", "fractions": mole_fractions},
            "reference_anchor": reference["reference_anchor"],
            "source_reference_id": reference["reference_id"],
        }
        if predefined_mixture:
            signature["composition"]["predefined_mixture"] = canonical_names[0]
    except (ValueError, RuntimeError, NotImplementedError, OSError) as error:
        raise _configuration_error(
            fluid_name,
            "source_reference",
            f"Cannot establish a verifiable cache source identity: {error}",
        ) from error

    # This is the selected backend's own definition, not another backend's
    # catalogue entry. REFPROP does not expose JSON through this API.
    try:
        definition_json = state.fluid_param_string("JSON")
    except (ValueError, RuntimeError, NotImplementedError) as error:
        signature["native_definition"] = {
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
        }
    else:
        try:
            signature["native_definition"] = {
                "available": True,
                "sha256": metadata_fingerprint(json.loads(definition_json)),
            }
        except (TypeError, ValueError) as error:
            raise _configuration_error(
                fluid_name,
                "source_definition",
                "The backend returned an invalid native fluid definition",
            ) from error
    if "REFPROP" in backend.upper():
        signature["native_files"] = _refprop_native_files(
            canonical_names, configuration
        )
    return signature


def source_signature(fluid_name, backend, mixture_ratio=None):
    """Describe the exact requested source and its current reference convention.

    Legacy binary models carry their two source identities independently and
    record their unrounded mass-flow ratio. This helper never changes CoolProp
    configuration or reference states and evaluates only dedicated source
    objects, leaving already configured application fluid objects untouched.
    """
    if not isinstance(fluid_name, str) or not fluid_name:
        raise _configuration_error(
            fluid_name, "fluid_name", "A nonempty source fluid name is required"
        )
    if not isinstance(backend, str) or not backend:
        raise _configuration_error(
            fluid_name, "backend", "A nonempty CoolProp source backend is required"
        )

    legacy_mixture = fluid_name.lower().endswith("_mixture")
    if legacy_mixture or mixture_ratio is not None:
        components = fluid_name[: -len("_mixture")].split("_") if legacy_mixture else []
        if len(components) != 2 or not all(components):
            raise _configuration_error(
                fluid_name,
                "mixture_ratio",
                "A binary source requires the '<component>_<component>_mixture' name",
                mixture=True,
            )
        if isinstance(mixture_ratio, bool) or not isinstance(mixture_ratio, Real):
            raise _configuration_error(
                fluid_name,
                "mixture_ratio",
                "A finite nonnegative component-1/component-2 mass ratio is required",
                mixture=True,
            )
        try:
            ratio = float(mixture_ratio)
        except (ValueError, OverflowError) as error:
            raise _configuration_error(
                fluid_name,
                "mixture_ratio",
                "A finite nonnegative component-1/component-2 mass ratio is required",
                mixture=True,
            ) from error
        if not math.isfinite(ratio) or ratio < 0:
            raise _configuration_error(
                fluid_name,
                "mixture_ratio",
                "A finite nonnegative component-1/component-2 mass ratio is required",
                mixture=True,
            )
        composition = {
            "basis": "mass_ratio",
            "component_1_to_2": ratio,
            "mass_fractions": [ratio / (1.0 + ratio), 1.0 / (1.0 + ratio)],
        }
    else:
        components = [fluid_name]
        composition = None

    try:
        coolprop_version = CP.get_global_param_string("version")
        configuration = json.loads(CP.get_config_as_json_string())
        signature = {
            "coolprop_version": coolprop_version,
            "coolprop_git_revision": CP.get_global_param_string("gitrevision"),
            "coolprop_configuration": configuration,
            "backend_version": (
                CP.get_global_param_string("REFPROP_version")
                if "REFPROP" in backend.upper()
                else coolprop_version
            ),
            "components": [
                _component_signature(name, backend, configuration)
                for name in components
            ],
        }
    except InvalidConfigurationError:
        raise
    except (ValueError, RuntimeError, NotImplementedError, OSError) as error:
        raise _configuration_error(
            fluid_name,
            "source_metadata",
            f"Cannot establish CoolProp/backend provenance: {error}",
            mixture=legacy_mixture,
        ) from error
    if composition is not None:
        signature["composition"] = composition
        signature["source_reference_id"] = (
            "binary_mixture",
            tuple(item["source_reference_id"] for item in signature["components"]),
            ("mass_ratio", ratio),
        )
    else:
        signature["source_reference_id"] = signature["components"][0][
            "source_reference_id"
        ]
    signature["deterministic_cache_identity"] = all(
        item["native_definition"]["available"]
        or item.get("native_files", {}).get("available", False)
        for item in signature["components"]
    )
    # Validate that no backend supplied an unserializable or nonfinite value.
    try:
        metadata_fingerprint(signature)
    except (TypeError, ValueError) as error:
        raise _configuration_error(
            fluid_name,
            "source_metadata",
            "The source identity is not finite JSON-compatible metadata",
            mixture=legacy_mixture,
        ) from error
    return signature


__all__ = ["metadata_fingerprint", "source_signature"]
