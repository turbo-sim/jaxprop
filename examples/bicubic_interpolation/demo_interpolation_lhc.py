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
N_samples = 1000  # number of LHS samples

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
fluid_cp = jxp.FluidJAX(fluid_name, backend="HEOS")
fluid_pg = jxp.FluidPerfectGas(fluid_name, p_ref=p_max, T_ref=600)

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
# Warmup (to trigger JIT compilation)
# ---------------------------
print("\nWarming up JIT for all solvers...\n")
_ = fluid_bicubic.get_props(jxp.HmassP_INPUTS, h_samples, p_samples)
_ = fluid_cp.get_props(jxp.HmassP_INPUTS, h_samples, p_samples)
_ = fluid_pg.get_props(jxp.HmassP_INPUTS, h_samples, p_samples)


# ---------------------------
# Batched evaluation
# ---------------------------
print("\nEvaluating samples in batch...\n")

t0 = time.perf_counter()
interp_props = fluid_bicubic.get_props(jxp.HmassP_INPUTS, h_samples, p_samples)
t1 = time.perf_counter()

coolprop_props = fluid_cp.get_props(jxp.HmassP_INPUTS, h_samples, p_samples)
t2 = time.perf_counter()

pg_props = fluid_pg.get_props(jxp.HmassP_INPUTS, h_samples, p_samples)
t3 = time.perf_counter()

# ---------------------------
# Timing summary
# ---------------------------
print("\nTiming summary (JIT warmup excluded):")
print(f"  Bicubic   = {t1 - t0:8.6f} s")
print(f"  CoolProp  = {t2 - t1:8.6f} s")
print(f"  PerfectGas= {t3 - t2:8.6f} s")


