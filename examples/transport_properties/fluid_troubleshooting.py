# import matplotlib.pyplot as plt
# import coolpropx as cpx

# import CoolProp.CoolProp as CP
# # from CoolProp import AbstractState


# import CoolProp


# # cpx.print_fluid_names()
# # cpx.set_plot_options(grid=False)


# # # Create fluid
# # fluid_name = "R134a"
# # fluid = cpx.Fluid(name=fluid_name, backend="HEOS")
# # print(fluid.critical_point)

# # from CoolProp.CoolProp import AbstractState
# # state = AbstractState("REFPROP", "R1233ZDE")
# # state.update(CP.PQ_INPUTS, 500000, 0)  # 500 kPa, saturated liquid
# # print(state.T())
# # 
# # fluid_name = "R1233ZDE"
# # fluid = cpx.Fluid(name=fluid_name, backend="HEOS")
# # # # print(fluid.get_state(CP.PQ_INPUTS, 500000, 0))
# # print(fluid.critical_point)


# # fluid_name = "R1233ZDE"
# # fluid = cpx.Fluid(name=fluid_name, backend="REFPROP")
# # print(fluid.triple_point_vapor)

# # fluid.plot_phase_diagram(plot_quality_isolines=False)
# # plt.show()


# # fluid_name = "air"
# # fluid_name = "R134a"
# fluid_name = "R1233ZDE"
# fluid = cpx.Fluid(name=fluid_name, backend="BICUBIC&REFPROP")
# fluid = cpx.Fluid(name=fluid_name, backend="BICUBIC&HEOS")
# # print(fluid.critical_point)
# # print(fluid.get_state(cpx.PQ_INPUTS, 3.785311e+06*0.9, 0.5))

# print(fluid.triple_point_liquid)

# print()

# print(fluid.triple_point_vapor)
# # ampersand

# fluid.plot_phase_diagram(plot_quality_isolines=True)

# plt.show()

