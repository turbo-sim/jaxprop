"""
This demo illustrates how to do calculate the generalized degrees of superheating and subcooling
"""

import CoolProp as cp
import coolpropx as cpx

# Create high-level Fluid object
fluid = cpx.Fluid(name="water", backend="HEOS")

# --- Superheating: low-level (only once) ---
as_superheat = cp.AbstractState("HEOS", "water")
as_superheat.update(cp.PT_INPUTS, 101325, 120 + 273.15)
superheat_low = cpx.calculate_superheating(as_superheat)
print(f"[low-level] Superheating: {superheat_low:+0.3f} K")

# --- Superheating: high-level ---
state = fluid.get_state(cp.PT_INPUTS, 101325, 120 + 273.15, supersaturation=True)
print(f"[high-level] Superheating (PQ): {state.superheating:+0.3f} K")

# --- Superheating: high-level ---
state = fluid.get_state(cp.PQ_INPUTS, 101325, 0.95, supersaturation=True)
print(f"[high-level] Superheating (PQ): {state.superheating:+0.3f} K")

# --- Subcooling: high-level ---
state_2 = fluid.get_state(cp.PT_INPUTS, 101325, 25 + 273.15, supersaturation=True)
print(f"[high-level] Subcooling (PT): {state_2.subcooling:+0.3f} K")

# --- Subcooling: high-level ---
state_3 = fluid.get_state(cp.PQ_INPUTS, 101325, 0.05, supersaturation=True)
print(f"[high-level] Subcooling (PQ): {state_3.subcooling:+0.3f} K")

