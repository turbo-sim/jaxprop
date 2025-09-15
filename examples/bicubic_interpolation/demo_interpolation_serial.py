import os
import time
import shutil
import jaxprop as jxp
import numpy as np
from scipy.stats import qmc

outdir = "fluid_tables"
# if os.path.exists(outdir):
#     shutil.rmtree(outdir, ignore_errors=True)

# ---------------------------
# Configuration
# ---------------------------
fluid_name = "CO2"
h_min = 200e3  # J/kg
h_max = 600e3  # J/kg
p_min = 2e6    # Pa
p_max = 20e6   # Pa
N_p = 30       # Grid size for table
N_h = 30
N_samples = 50  # reduced for demo, increase if needed

# ---------------------------
# Build fluids
# ---------------------------
fluid_bicubic = jxp.FluidBicubic(
    fluid_name=fluid_name,
    backend="HEOS",
    h_min=h_min,
    h_max=h_max,
    p_min=p_min,
    p_max=p_max,
    N_h=N_h,
    N_p=N_p,
    table_dir=outdir,
)
fluid_coolprop = jxp.FluidJAX(fluid_name, backend="HEOS")
fluid_perfect = jxp.FluidPerfectGas(fluid_name, p_ref=p_max, T_ref=600)

# ---------------------------
# Latin Hypercube Sampling
# ---------------------------
sampler = qmc.LatinHypercube(d=2, seed=42)
lhs_points = sampler.random(N_samples)
l_bounds = [h_min, p_min]
u_bounds = [h_max, p_max]
scaled_samples = qmc.scale(lhs_points, l_bounds, u_bounds)
h_samples = scaled_samples[:, 0]
p_samples = scaled_samples[:, 1]

# ---------------------------
# Serial evaluation
# ---------------------------
print("\nEvaluating samples serially...\n")
print(f"{'Sample':>6} | {'h [J/kg]':>12} | {'p [Pa]':>12} | {'PerfectGas [ms]':>15} | {'Bicubic [ms]':>15} | {'CoolProp [ms]':>15}")

# store timings
timings_pg = []
timings_bi = []
timings_cp = []

for i, (h, p) in enumerate(zip(h_samples, p_samples), 1):
    # Perfect gas
    t0 = time.perf_counter()
    _ = fluid_perfect.get_props(jxp.HmassP_INPUTS, h, p)
    t1 = time.perf_counter()
    dt_pg = (t1 - t0) * 1e3  # ms

    # Bicubic
    t0 = time.perf_counter()
    _ = fluid_bicubic.get_props(jxp.HmassP_INPUTS, h, p)
    t1 = time.perf_counter()
    dt_bi = (t1 - t0) * 1e3  # ms

    # CoolProp
    t0 = time.perf_counter()
    _ = fluid_coolprop.get_props(jxp.HmassP_INPUTS, h, p)
    t1 = time.perf_counter()
    dt_cp = (t1 - t0) * 1e3  # ms

    # collect
    timings_pg.append(dt_pg)
    timings_bi.append(dt_bi)
    timings_cp.append(dt_cp)

    # Print row
    print(f"{i:6d} | {h:12.2f} | {p:12.2f} | {dt_pg:15.2f} | {dt_bi:15.2f} | {dt_cp:15.2f}")

# ---------------------------
# Averages (skip first entry)
# ---------------------------
if len(timings_pg) > 1:
    avg_pg = np.mean(timings_pg[1:])
    avg_bi = np.mean(timings_bi[1:])
    avg_cp = np.mean(timings_cp[1:])

    print("\nAverage timings (skipping first sample):")
    print(f"  PerfectGas = {avg_pg:.2f} ms")
    print(f"  Bicubic    = {avg_bi:.2f} ms")
    print(f"  CoolProp   = {avg_cp:.2f} ms")