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
N_samples = 100  # number of LHS samples

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
# Warmup (to trigger JIT compilation)
# ---------------------------
print("Warming up JIT for all solvers...")
_ = fluid_perfect.get_props(jxp.HmassP_INPUTS, h_samples, p_samples)
_ = fluid_bicubic.get_props(jxp.HmassP_INPUTS, h_samples, p_samples)
_ = fluid_coolprop.get_props(jxp.HmassP_INPUTS, h_samples, p_samples)


# ---------------------------
# Batched evaluation
# ---------------------------
print("Evaluating samples in batch...")

t0 = time.perf_counter()
interp_props = fluid_bicubic.get_props(jxp.HmassP_INPUTS, h_samples, p_samples)
t1 = time.perf_counter()

coolprop_props = fluid_coolprop.get_props(jxp.HmassP_INPUTS, h_samples, p_samples)
t2 = time.perf_counter()

pg_props = fluid_perfect.get_props(jxp.HmassP_INPUTS, h_samples, p_samples)
t3 = time.perf_counter()

# ---------------------------
# Timing summary
# ---------------------------
print("Timing summary (JIT warmup excluded):")
print(f"  PerfectGas = {1000*(t3 - t2):8.3f} ms")
print(f"  Bicubic    = {1000*(t1 - t0):8.3f} ms")
print(f"  CoolProp   = {1000*(t2 - t1):8.3f} ms")



