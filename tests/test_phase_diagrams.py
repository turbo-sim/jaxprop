# import pytest
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import jaxprop.coolprop as cpx
# import matplotlib

# from utilities import get_available_backends

# matplotlib.use("Agg")

# cpx.set_plot_options(grid=False)

# # Define list of calculation backends
# BACKENDS = get_available_backends()

# # Define all working fluids
# FLUID_NAMES = [
#     "Water",
#     "CO2",
#     # "Ammonia",
#     # "Nitrogen",
#     # "Pentane",
#     "R134a",
#     # "R245fa",
#     # "R1233ZDE",
# ]

# # Define diagram types (x/y axis pairs)
# DIAGRAMS = [
#     ("T", "p"),    # p-T
#     ("s", "T"),    # T-s
#     ("h", "p"),    # p-h
#     ("s", "h"),    # h-s
# ]

# # Output directory for saving plots
# OUTPUT_DIR = "output"
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# @pytest.mark.parametrize("fluid_name", FLUID_NAMES)
# def test_phase_diagrams(fluid_name):
#     fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6), constrained_layout=True)
#     fig.suptitle(f"Phase diagrams for {fluid_name}", fontsize=16)

#     for row, backend in enumerate(BACKENDS):
#         fluid = cpx.Fluid(fluid_name, backend)
#         for col, (x_prop, y_prop) in enumerate(DIAGRAMS):

#             # Generate plot for given property pair
#             ax = axs[row, col]
#             fluid.plot_phase_diagram(
#                 axes=ax,
#                 x_prop=x_prop,
#                 y_prop=y_prop,
#                 plot_saturation_line=True,
#                 plot_critical_point=True,
#                 plot_triple_point_liquid=True,
#                 plot_triple_point_vapor=True,
#                 plot_pseudocritical_line=True,
#                 plot_spinodal_line=True,
#                 plot_quality_isolines=True,
#                 N=25,
#                 dT_crit=2,
#             )

#             # Ensure axes start at zero, but preserve automatic upper limit
#             xmin, xmax = ax.get_xlim()
#             ymin, ymax = ax.get_ylim()
#             fluid.triple_point_liquid[x_prop]
#             ax.set_xlim(left=max(fluid.triple_point_liquid[x_prop], xmin), right=xmax)
#             ax.set_ylim(bottom=max(fluid.triple_point_liquid[y_prop], ymin), top=ymax)
#             ax.set_title(f"{backend}: {y_prop}-{x_prop}")
#             ax.set_xlabel(x_prop)
#             ax.set_ylabel(y_prop)
#             ax.set_xscale("linear" if x_prop != "p" else "log")
#             ax.set_yscale("linear" if y_prop != "p" else "log")

#     # # Save figure
#     # fname = os.path.join(OUTPUT_DIR, f"phase_diagrams_{fluid_name.lower()}.png")
#     # fig.savefig(fname, dpi=250)
#     # plt.close(fig)
#     # print(f"Phase diagrams created for '{fluid_name}'")


# if __name__ == "__main__":
#     for fluid_name in FLUID_NAMES:
#         test_phase_diagrams(fluid_name)

#     # plt.show()

