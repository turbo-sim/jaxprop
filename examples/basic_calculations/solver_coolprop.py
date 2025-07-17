"""
This demo illustrates how to do basic thermodynamic state calculations using
the barotropy.fluid.get_state() method using different types of input pairs.
"""

import coolpropx as cpx

# State calculation
fluid = cpx.Fluid(name="Water", backend="HEOS")
state = fluid.get_state(cpx.PT_INPUTS, 101325, 300)
print(f"Water density is {state.rho:0.2f} kg/m3 at p={state.p:0.2f} Pa and T={state.T:0.2f} K")

# State calculation
fluid = cpx.Fluid(name="Air", backend="HEOS")
state = fluid.get_state(cpx.PT_INPUTS, 101325, 300)
print(f"Air heat capacity ratio is {state.gamma:0.2f} at p={state.p:0.2f} Pa and T={state.T:0.2f} K")

# Properties of liquid water
fluid = cpx.Fluid("Water", backend="HEOS")
props_stable = fluid.get_state(cpx.PT_INPUTS, 101325, 300)
print("\nProperties of liquid water")
print(props_stable)

# Properties of water/steam mixture
fluid = cpx.Fluid("Water", backend="HEOS")
props = fluid.get_state(cpx.QT_INPUTS, 0.5, 300)
print("\nProperties of water/steam mixture")
print(props)

# Get subset of properties for meanline code
props = fluid.compute_properties_meanline(cpx.QT_INPUTS, 0.5, 300)
print()
print("Properties for the meanline code")
print(f"{'Property':15} {'value':6}")
for key, value in props.items():
    print(f"{key:15} {value:.6e}")
