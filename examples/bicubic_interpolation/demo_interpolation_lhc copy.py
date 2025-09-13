import os
import time
import shutil
import jaxprop as jxp
import numpy as np
from scipy.stats import qmc



outdir = "fluid_tables"
if os.path.exists(outdir):
    shutil.rmtree(outdir, ignore_errors=True)



# ---------------------------
# Configuration
# ---------------------------
outdir = "fluid_tables"
fluid_name = "CO2"
h_min = 200e3  # J/kg
h_max = 600e3  # J/kg
p_min = 2e6    # Pa
p_max = 20e6   # Pa
N_p = 30       # Grid size for table
N_h = 30
N_samples = 10  # number of LHS samples

# ---------------------------
# Build bicubic table
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

# Reference CoolProp fluid
fluid_cp = jxp.FluidJAX(fluid_name, backend="HEOS")

fluid_pg = jxp.FluidPerfectGas(fluid_name, p_ref=p_max, T_ref=600)

# ---------------------------
# Latin Hypercube Sampling (SciPy)
# ---------------------------
sampler = qmc.LatinHypercube(d=2, seed=42)
lhs_points = sampler.random(N_samples)
l_bounds = [h_min, p_min]
u_bounds = [h_max, p_max]
scaled_samples = qmc.scale(lhs_points, l_bounds, u_bounds)
h_samples = scaled_samples[:, 0]
p_samples = scaled_samples[:, 1]



import time
import numpy as np

# ---------------------------
# Error evaluation per sample with timings
# ---------------------------
for idx, (h, p) in enumerate(zip(h_samples, p_samples)):
    # try:
    t0 = time.perf_counter()
    interp_props = fluid_bicubic.get_props(jxp.HmassP_INPUTS, h, p)
    t1 = time.perf_counter()

    coolprop_props = fluid_cp.get_props(jxp.HmassP_INPUTS, h, p)
    t2 = time.perf_counter()

    pg_props = fluid_pg.get_props(jxp.HmassP_INPUTS, h, p)
    # except Exception:
    #     continue  # skip invalid state points
    t3 = time.perf_counter()

    errors = []
    for prop in jxp.PROPERTIES_CANONICAL:
        val_interp = interp_props[prop]
        val_cp = coolprop_props[prop]
        if (
            val_interp is not None
            and val_cp is not None
            and np.isfinite(val_interp)
            and np.isfinite(val_cp)
            and val_cp != 0.0
        ):
            rel_err = (val_interp - val_cp) / val_cp * 100.0
            errors.append(abs(rel_err))

    if not errors:
        continue  # skip if all properties failed


    errors = np.array(errors)
    max_err = errors.max()
    min_err = errors.min()
    avg_err = errors.mean()

    print(
        f"Sample {idx:4d} | "
        f"Max abs error: {max_err:9.3f} % | "
        f"Min abs error: {min_err:9.3f} % | "
        f"Avg abs error: {avg_err:9.3f} % | "
        f"t_bicubic={t1 - t0:8.6f} s | "
        f"t_coolprop={t2 - t1:8.6f} s | "
        f"t_pg={t3 - t2:8.6f} s | ",
        )

