"""Demo table generation.

Quick/full profiles have distinct declared grid sizes. Use explicit output and
table-cache paths; full retains the scientific resolution.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import example_options

import os
import time
import shutil
import jaxprop as jxp
from examples.bicubic_interpolation.export_pkl_to_c_binary import assert_roundtrip


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    # ---------------------------
    # Define table parameters
    # ---------------------------
    # fluid_name = "CO2"
    # h_min = 200e3  # J/kg
    # h_max = 600e3  # J/kg
    # p_min = 2e6  # Pa
    # p_max = 20e6  # Pa
    # N_p = 80  # Number of pressure points
    # N_h = 80  # Number of enthalpy points
    # fluid_name = "CO2"
    # h_min = 150e3  # J/kg
    # h_max = 600e3  # J/kg
    # p_min = 0.5e6  # Pa
    # p_max = 20e6  # Pa
    # N_p = 300
    # N_h = 300

    fluid_name = "nitrogen"
    h_min = 90e3  # J/kg
    h_max = 350e3  # J/kg
    p_min = 1e4  # Pa
    p_max = 3e6  # Pa
    N_p = 145 if options.profile == "full" else 12
    N_h = 145 if options.profile == "full" else 12

    # fluid_name = "water"
    # h_min = 1e3  # J/kg
    # h_max = 140e3  # J/kg
    # p_min = 1e4  # Pa
    # p_max = 3e6  # Pa
    # N_p = 145
    # N_h = 145

    # ---------------------------
    # Delete existing tables
    # ---------------------------
    outdir = str(options.output_dir)
    # if os.path.exists(outdir):
    #     shutil.rmtree(outdir, ignore_errors=True)

    # ---------------------------
    # First call: generate a missing table or reuse the explicit cache.
    # ---------------------------
    previous_files = (
        set(options.table_dir.glob("*.pkl")) if options.table_dir.exists() else set()
    )
    start = time.perf_counter()
    fluid = jxp.Fluid(
        name=fluid_name,
        method="bicubic",
        backend="HEOS",
        table_options={
            "h_min": h_min,
            "h_max": h_max,
            "p_min": p_min,
            "p_max": p_max,
            "N_h": N_h,
            "N_p": N_p,
            "table_dir": str(options.table_dir),
        },
    )
    end = time.perf_counter()
    first_seconds = end - start
    first_table = fluid.table
    cache_was_present = (
        options.table_dir / (fluid.table_name + ".pkl") in previous_files
    )
    print(f"FluidBicubic init took {end - start:.3f} s")
    print()

    # ---------------------------
    # Second call: load table
    # ---------------------------
    start = time.perf_counter()
    fluid = jxp.Fluid(
        name=fluid_name,
        method="bicubic",
        backend="HEOS",
        table_options={
            "h_min": h_min,
            "h_max": h_max,
            "p_min": p_min,
            "p_max": p_max,
            "N_h": N_h,
            "N_p": N_p,
            "table_dir": str(options.table_dir),
        },
    )
    end = time.perf_counter()
    cached_seconds = end - start
    arrays_checked = assert_roundtrip(first_table, fluid.table)
    print(f"FluidBicubic init took {end - start:.3f} s")

    options.record(
        "settings",
        {
            "fluid_name": fluid_name,
            "h_min": h_min,
            "h_max": h_max,
            "p_min": p_min,
            "p_max": p_max,
            "N_h": N_h,
            "N_p": N_p,
        },
    )
    options.record(
        "cache_roundtrip",
        dict(
            arrays_checked=arrays_checked,
            exact_finite_and_paired_nan_match=True,
            first_cache_was_present=cache_was_present,
            first_initialization_s=first_seconds,
            cached_initialization_s=cached_seconds,
            timing_scope="Single initialization observations, not warmed property-call benchmarks",
        ),
    )
    return options.finish()


if __name__ == "__main__":
    main()
