import coolpropx as cpx

fluid = cpx.Fluid(name="CO2", backend="REFPROP")


state = fluid.get_state(cpx.PQ_INPUTS, 0.8*fluid.critical_point.p, 0.5)
print(state.surface_tension)


state = fluid.get_state(cpx.PT_INPUTS, 1.8*fluid.critical_point.p, 500)
print(state.surface_tension)

