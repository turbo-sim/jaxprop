import os
import time
import shutil
import jaxprop as jxp


# ---------------------------
# Define table parameters
# ---------------------------
fluid_name = "CO2"
h_min = 200e3  # J/kg
h_max = 600e3  # J/kg
p_min = 2e6  # Pa
p_max = 20e6  # Pa
N_p = 80  # Number of pressure points
N_h = 80  # Number of enthalpy points


# ---------------------------
# Delete existing tables
# ---------------------------
outdir = "demo_table_generation"
if os.path.exists(outdir):
    shutil.rmtree(outdir, ignore_errors=True)


# ---------------------------
# First call: generate table
# ---------------------------
start = time.time()
fluid = jxp.FluidBicubic(
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
end = time.time()
print(f"FluidBicubic init took {end - start:.3f} s")
print()


# ---------------------------
# Second call: load table
# ---------------------------
start = time.time()
fluid = jxp.FluidBicubic(
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
end = time.time()
print(f"FluidBicubic init took {end - start:.3f} s")
