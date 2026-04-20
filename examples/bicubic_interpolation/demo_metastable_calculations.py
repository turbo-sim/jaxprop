import os
import shutil
import jaxprop as jxp
import pickle


# fluid_name = "nitrogen"
# h_min = -145e3  # J/kg
# h_max = -1e3  # J/kg
# p_min = 3e4  # Pa
# p_max = 2e6  # Pa
# N_p = 150 # Number of pressure points
# N_h = 150 # Number of enthalpy points
# metastable_phase = "liquid"

fluid_name = "water"
h_min = 200e3  # J/kg
h_max = 1800e3  # J/kg
p_min = 1e5    # Pa
p_max = 13e6   # Pa
N_p = 20 # Number of pressure points
N_h = 20 # Number of enthalpy points
metastable_phase = "liquid"

# fluid_name = "nitrogen"
# h_min = 1e3  # J/kg
# h_max = 160e3  # J/kg
# p_min = 3e4  # Pa
# p_max = 2e6  # Pa
# N_p = 150 # Number of pressure points
# N_h = 150 # Number of enthalpy points
# metastable_phase = "vapor"

# fluid_name = "cyclopentane"
# h_min = 250e3  # J/kg
# h_max = 700e3  # J/kg
# p_min = 1e5  # Pa
# p_max = 1e6  # Pa
# N_p = 30 # Number of pressure points
# N_h = 30 # Number of enthalpy points
# metastable_phase = "vapor"

# fluid_name = "CO2"
# h_min = 100e3  # J/kg
# h_max = 250e3  # J/kg
# p_min = 1e6    # Pa
# p_max = 10e6   # Pa
# N_p = 50 # Number of pressure points
# N_h = 50 # Number of enthalpy points
# metastable_phase = "liquid"


# fluid_name = "CO2"
# h_min = 360e3  # J/kg
# h_max = 460e3  # J/kg
# p_min = 1e6    # Pa
# p_max = 4e6   # Pa
# N_p = 30 # Number of pressure points
# N_h = 30 # Number of enthalpy points
# metastable_phase = "vapor"

# fluid_name = "water"
# h_min = 300e3  # J/kg
# h_max = 1800e3  # J/kg
# p_min = 1e5    # Pa
# p_max = 13e6   # Pa
# N_p = 10 # Number of pressure points
# N_h = 10 # Number of enthalpy points
# metastable_phase = "liquid"

# fluid_name = "water"
# h_min = 2200e3  # J/kg
# h_max = 5200e3  # J/kg  3400e3
# p_min = 1e5    # Pa
# p_max = 13e6   # Pa
# N_p = 140 # Number of pressure points
# N_h = 140 # Number of enthalpy points
# metastable_phase = "vapor"

# fl = jxp.Fluid(fluid_name)
# print(fl.critical_point["pressure"])


# ---------------------------
# Delete existing tables3
# ---------------------------
outdir = "demo_metastable_table_generation"
# if os.path.exists(outdir):
#     shutil.rmtree(outdir, ignore_errors=True)


# # ---------------------------
# # First call: generate table
# # ---------------------------
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
    gradient_method = "forward",
    N_p_sat=150,
)

# state_critical = fluid_bicubic.get_state(jxp.HmassP_INPUTS, 3912677.9, 8822847.22)
# rho = state_critical["surface_tension"]

# print(f"Density:{rho}")
# print(fluid_bicubic.p_min)


# import pandas as pd

# # 1. Read the pickle file
# # Replace 'input_file.pkl' with your file path
# df = pd.read_pickle('demo_metastable_table_generation/water_41x41.pkl')

# # 2. Save to Excel
# # 'index=False' prevents pandas from writing the row numbers (0, 1, 2...) as the first column
# # df.to_excel('CO2_40x40.csv', index=False)

# dataframe = pd.DataFrame(df)

# # 2. Save to Excel
# # Note: You used '.csv' in your filename but called 'to_excel'.
# # If you want a real Excel file, change the extension to .xlsx
# dataframe.to_excel('water_41x41.xlsx', index=False)

# print("Conversion complete!")
 
# # ---------------------------
# # 1. Load Data
# # ---------------------------
# outdir = "demo_metastable_table_generation"
# filename = "CO2_51x51.pkl"
# file_path = os.path.join(outdir, filename)

# with open(file_path, "rb") as f:
#     data = pickle.load(f)

# # %%

# def get_values(data_dict, key):
#     obj = data_dict.get(key)
#     if isinstance(obj, dict):
#         return obj.get('values', list(obj.values())[0])
#     return obj


# sat_props = get_values(data, "saturation_props")

# print(sat_props)
