import jaxprop as jxp



fluid_name = "nitrogen"
h_min = -110e3  # J/kg
h_max = -10e3   # J/kg
p_min = 3e4     # Pa
p_max = 1e6     # Pa
N_p = 165       # Number of pressure points
N_h = 165       # Number of enthalpy points
metastable_phase = "liquid"

outdir = "demo_metastable_table_generation"

# ---------------------------
# Build bicubic table
# ---------------------------
fluid_bicubic = jxp.FluidBicubic(
    fluid_name=fluid_name,
    backend="HEOS",
    h_min=h_min, h_max=h_max,
    p_min=p_min, p_max=p_max,
    N_h=N_h, N_p=N_p,
    table_dir=outdir,
    metastable_phase=metastable_phase,
    gradient_method="forward",
)


hl = -85528.8
p = 5.0719e+05

T_test = fluid_bicubic.get_state(jxp.HmassP_INPUTS, hl, p)["T"]
print(T_test)

h_test = fluid_bicubic.get_state(jxp.PT_INPUTS, p, T_test)["h"]
print(h_test)


sat = fluid_bicubic.get_state_saturation(p)
print(sat)
