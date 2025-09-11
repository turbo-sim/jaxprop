import os
import pickle
import jaxprop as jxp


outdir = "fluid_tables"


# ---------------------------
# Test parameters
# ---------------------------s
fluid = "CO2"
h_min = 200e3  # J/kg
h_max = 600e3  # J/kg
p_min = 2e6  # Pa
p_max = 20e6  # Pa
N_p = 20  # Grid size for quick test
N_h = 16

# ---------------------------
# Run table generation
# ---------------------------
table = jxp.bicubic.generate_property_table(
    fluid_name=fluid,
    backend="HEOS",
    h_min=h_min,
    h_max=h_max,
    p_min=p_min,
    p_max=p_max,
    N_h=N_h,
    N_p=N_p,
    outdir=outdir,
)


print(table)

# # ---------------------------
# # Load and preview output
# # ---------------------------

# pkl_file = os.path.join(outdir, f"{fluid}_{Nh}_x_{Np}.pkl")

# print("\n Verifying saved table structure...\n")

# # Load pickled dict
# with open(pkl_file, 'rb') as f:
#     table = pickle.load(f)

# # Check and print summary
# print(" Keys in table:", list(table.keys()))

# print("\n h values:", table['h'][:5])
# print(" P values:", table['P'][:5])

# print("\n Preview for property 'T':")
# if 'T' in table:
#     print(" - Value shape:", table['T']['value'].shape)
#     print(" - Sample values (top-left corner):")
#     print(table['T']['value'][:3, :3])
# else:
#     print(" - Property 'T' not found.")
