import jaxprop as jxp
import numpy as np

fluid_name = "nitrogen"
h_min = -110e3  # J/kg
h_max = -10e3  # J/kg
p_min = 3e4  # Pa
p_max = 1e6  # Pa
N_p = 165  # Number of pressure points
N_h = 165  # Number of enthalpy points
metastable_phase = "liquid"

outdir = "demo_metastable_table_generation"

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
    metastable_phase=metastable_phase,
    gradient_method="forward",
)

# This local HP/PT/saturation check does not validate the whole rectangular
# domain. Preserve that domain and report its caloric instability explicitly.
cv_nodes = np.asarray(fluid_bicubic.table["isochoric_heat_capacity"]["value"])
calorically_unstable_nodes = np.count_nonzero(cv_nodes <= 0)
if calorically_unstable_nodes:
    print(
        "Caloric-domain notice: "
        f"{calorically_unstable_nodes}/{cv_nodes.size} nodes have cv <= 0. "
        "Gamma has a pole at cv=0; review the physical domain/model. "
        "Agreement at the selected point does not establish whole-table accuracy."
    )


hl = -85528.8
p = 5.0719e05

T_test = fluid_bicubic.get_state(jxp.HmassP_INPUTS, hl, p)["T"]
print(T_test)

h_test = fluid_bicubic.get_state(jxp.PT_INPUTS, p, T_test)["h"]
print(h_test)


quality = 0.0 if metastable_phase == "liquid" else 1.0
sat = fluid_bicubic.get_state_saturation(jxp.PQ_INPUTS, p, quality)
print(sat)
