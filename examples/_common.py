"""Small command-line/output helpers shared by the example applications.

No thermodynamic models, plots, directories or JAX arrays are constructed on
import. Numerical checks remain visible in each scientific example.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys

# These are application defaults, not library import policy. Explicit caller
# settings take priority. Set them before the examples import scientific code.
os.environ.setdefault("JAX_ENABLE_X64", "true")
if "--headless" in sys.argv:
    os.environ.setdefault("MPLBACKEND", "Agg")


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class ExampleOptions:
    """Explicit execution paths/profile and a small result/figure manifest."""

    source: Path
    args: argparse.Namespace
    records: dict = field(default_factory=dict)
    sample_sizes: dict = field(default_factory=dict)

    @property
    def profile(self):
        return self.args.profile

    @property
    def output_dir(self):
        return self.args.output_dir

    @property
    def table_dir(self):
        return self.args.table_dir

    @property
    def headless(self):
        return self.args.headless

    @property
    def input_path(self):
        return self.args.input

    def record(self, name, value):
        """Record scientific settings/check results without hiding assertions."""
        self.records[name] = _json_value(value)

    def points(self, full, quick=21):
        """Select and record an explicit sampling reduction for quick execution."""
        actual = min(int(full), int(quick)) if self.profile == "quick" else int(full)
        self.sample_sizes[str(full)] = actual
        return actual

    def check_states(self, namespace, *, allow_failures=False):
        """Check all named common states; optional-property gaps remain explicit.

        Broad domain plots may intentionally include invalid requests. Their
        successful core values must still be finite and their failure counts
        are recorded, rather than silently promoting a masked grid to all-valid.
        This is example verification, outside timed numerical sections.
        """
        import numpy as np
        from jaxprop import FluidState

        checks = {}
        for name, state in namespace.items():
            if not isinstance(state, FluidState):
                continue
            valid = np.asarray(state.status) == 0
            if valid.size == 0 or not np.any(valid):
                raise AssertionError(
                    f"{name}: example has no successful numerical states"
                )
            if not allow_failures and not np.all(valid):
                raise AssertionError(
                    f"{name}: example contains failed numerical states"
                )
            for field_name in (
                "pressure",
                "temperature",
                "density",
                "enthalpy",
                "entropy",
                "internal_energy",
            ):
                values = np.asarray(state.raw_property(field_name))
                if not np.all(np.isfinite(values[valid])):
                    raise AssertionError(
                        f"{name}.{field_name}: nonfinite successful core values"
                    )
                if field_name in ("temperature", "density") and np.any(
                    values[valid] <= 0
                ):
                    raise AssertionError(
                        f"{name}.{field_name}: nonpositive successful core values"
                    )
            checks[name] = dict(
                points=int(valid.size),
                successful=int(np.count_nonzero(valid)),
                failed=int(np.count_nonzero(~valid)),
                core_finite=True,
            )
        self.records.setdefault("state_checks", {}).update(checks)
        return checks

    def finish(self):
        """Render/save figures, publish execution metadata, then optionally show."""
        figures = []
        pyplot = sys.modules.get("matplotlib.pyplot")
        if pyplot is not None:
            for number in pyplot.get_fignums():
                figure = pyplot.figure(number)
                figure.canvas.draw()
                finite_limits = all(
                    all(
                        math.isfinite(float(value))
                        for value in axis.get_xlim() + axis.get_ylim()
                    )
                    for axis in figure.axes
                )
                if not finite_limits:
                    raise AssertionError(
                        "A rendered example figure has nonfinite limits"
                    )
                path = self.output_dir / f"figure_{number:02d}.png"
                figure.savefig(path, dpi=120)
                figures.append(
                    dict(
                        file=path.name,
                        axes=len(figure.axes),
                        drawn=True,
                        finite_limits=True,
                    )
                )
        record = dict(
            example=self.source.relative_to(Path(__file__).parent).as_posix(),
            profile=self.profile,
            headless=self.headless,
            output_dir=str(self.output_dir),
            table_dir=str(self.table_dir),
            input_path=str(self.input_path) if self.input_path is not None else None,
            completed=True,
            checks=self.records,
            sample_sizes=self.sample_sizes,
            figures=figures,
        )
        (self.output_dir / "example_result.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if pyplot is not None:
            if self.headless:
                pyplot.close("all")
            else:
                pyplot.show()
        return record


def example_options(source, argv=None, *, description=None, configure_parser=None):
    """Parse uniform paths/profiles; allocate output directories only in main.

    Extra example-specific arguments may be registered by ``configure_parser``
    and are available through ``options.args``. Full profiles preserve each
    declared scientific study; quick profiles have separately documented sizes.
    """
    source = Path(source).resolve()
    # Preserve scientific labels in redirected Windows logs as well as terminals.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument(
        "--headless", action="store_true", help="Save figures without opening windows."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output") / source.stem)
    parser.add_argument(
        "--table-dir",
        type=Path,
        help="Property-table cache directory (default: output/tables).",
    )
    parser.add_argument(
        "--input", type=Path, help="Explicit source data for input-dependent tools."
    )
    if configure_parser is not None:
        configure_parser(parser)
    arguments = parser.parse_args(argv)
    arguments.output_dir = arguments.output_dir.resolve()
    arguments.table_dir = (
        arguments.table_dir.resolve()
        if arguments.table_dir is not None
        else arguments.output_dir / "tables"
    )
    if arguments.input is not None:
        arguments.input = arguments.input.resolve()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    # Tables are allocated by a model only when requested. A plotting example
    # should not create an unused cache directory merely to parse its options.
    if arguments.headless:
        import matplotlib

        matplotlib.use("Agg", force=True)
    return ExampleOptions(source, arguments)
