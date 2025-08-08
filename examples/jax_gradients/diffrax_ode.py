"""
Demonstration script: JAX-compatible thermodynamic ODE + gradient verification

This script shows how to:
  1. Wrap CoolProp property calls with a JAX-safe bridge (`jax.pure_callback`) so they
     can be used inside JAX transformations (`jacfwd`, `jacrev`, `jit`, etc.).
  2. Define a custom JVP rule for a property function (`rho_of`) so forward-mode AD
     works, even though the property evaluation is done via an external (non-JAX) library.
  3. Implement a polytropic expansion process model using `diffrax.diffeqsolve` to
     integrate the ODE dh/dp = eta / rho(h, p).
     - The ODE right-hand side calls the JAX-safe `rho_of` bridge.
     - We use `dfx.DirectAdjoint()` so the ODE solution supports *both* forward-mode
       and reverse-mode autodiff through the integration.
  4. Postprocess the solution by reconstructing the full thermodynamic states along
     the integration path (outside the JAX trace, so these are concrete Python objects).
  5. Define a scalar output of interest — the exit temperature — as a function of four
     scalar inputs: inlet enthalpy h_in, inlet pressure p_in, outlet pressure p_out,
     and efficiency η.
  6. Compute the gradient of the scalar output with respect to the four inputs using:
       - Forward-mode autodiff (`jax.jacfwd`)
       - Reverse-mode autodiff (`jax.jacrev`)
       - Finite-difference approximation (`scipy.optimize._numdiff.approx_derivative`)
  7. Compare the three results to validate correctness of the JAX gradients.

Key points:
- `rho_of` uses `jax.pure_callback` because CoolProp cannot be traced by JAX.
- The custom JVP for `rho_of` uses simple relative-step finite differences to
  propagate tangents in forward mode.
- `dfx.DirectAdjoint` is computationally less efficient than
  `RecursiveCheckpointAdjoint` or `ForwardMode` but allows both fwd/rev autodiff.
- The script serves as a working example for coupling thermodynamic property
  libraries with JAX and diffrax for differentiable process modeling.
"""

import jax
import numpy as np
import diffrax as dfx
import jax.numpy as jnp
import coolpropx as cpx
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from coolpropx.jaxprop import get_props  # your JAX bridge
from functools import partial

cpx.set_plot_options()


# ---- JAX-safe density bridge (only what RHS needs) ----
@partial(jax.custom_jvp, nondiff_argnums=(0, 3))   # <-- input_state and fluid are static
def rho_of(input_state, h, p, fluid):
    """Density via CoolProp using pure_callback; scalar float64."""
    out_aval = jax.ShapeDtypeStruct((), jnp.float64)

    def _cb(hp):
        h_val, p_val = hp
        d = fluid.get_state(input_state, float(h_val), float(p_val)).to_dict()["d"]
        return np.array(d, dtype=np.float64)

    return jax.pure_callback(_cb, out_aval, (h, p))

@rho_of.defjvp
def _rho_of_jvp(input_state, fluid, primals, tangents):
    h, p = primals
    h_dot, p_dot = tangents

    rho0 = rho_of(input_state, h, p, fluid)

    # relative FD steps
    dh = 1e-6 * (jnp.abs(h))
    dp = 1e-6 * (jnp.abs(p))

    rho_h = rho_of(input_state, h + dh, p, fluid)
    rho_p = rho_of(input_state, h, p + dp, fluid)

    drho_dh = (rho_h - rho0) / dh
    drho_dp = (rho_p - rho0) / dp

    jvp = drho_dh * h_dot + drho_dp * p_dot
    return rho0, jvp


def expansion_process(
    fluid,
    h_in,
    p_in,
    p_out,
    efficiency=None,
    efficiency_type="isentropic",
    mass_flow=None,
    data_in={},
    num_steps=50,
):
    """
    Calculate properties along a compression process defined by a isentropic or polytropic efficiency

    Parameters
    ----------
    fluid : Fluid
        The fluid object used to evaluate thermodynamic properties
    h_in : float
        Enthalpy at the start of the compression process.
    p_in : float
        Pressure at the start of the compression process.
    p_out : float
        Pressure at the end of the compression process.
    efficiency : float
        The efficiency of the compression process.
    efficiency_type : str, optional
        The type of efficiency to be used in the process ('isentropic' or 'polytropic'). Default is 'isentropic'.
    num_steps : int, optional
        The number of steps for the polytropic process calculation. Default is 50.

    Returns
    -------
    tuple
        Tuple containing (state_out, states) where states is a list with all intermediate states

    Raises
    ------
    ValueError
        If an invalid 'efficiency_type' is provided.

    """
    # Compute inlet state
    state_in = get_props(cpx.HmassP_INPUTS, h_in, p_in, fluid)    
    state_out_is = get_props( 
        cpx.PSmass_INPUTS,
        p_out,
        state_in["s"], fluid
    )
    if efficiency_type == "isentropic":
        # Compute outlet state according to the definition of isentropic efficiency
        h_out = state_in["h"] - efficiency * (state_in["h"] - state_out_is["h"])
        state_out = get_props(
            cpx.HmassP_INPUTS,
            h_out,
            p_out,
            fluid,
        )
        states = [state_in, state_out]

    elif efficiency_type == "polytropic":
        # RHS: t := p, y := h
        def rhs(p, h, eff):                 # <-- args is just efficiency (JAX type)
            rho = rho_of(cpx.HmassP_INPUTS, h, p, fluid)   # fluid is closed over; static via nondiff_argnums
            # rho = get_props(cpx.HmassP_INPUTS, h, p, fluid)["rho"]
            return eff / rho

        term = dfx.ODETerm(rhs)
        solver = dfx.Dopri5()
        controller = dfx.PIDController(rtol=1e-6, atol=1e-9)
        saveat = dfx.SaveAt(ts=jnp.linspace(state_in["p"], p_out, num_steps))

        sol = dfx.diffeqsolve(
            terms=term,
            solver=solver,
            t0=state_in["p"],
            t1=p_out,
            dt0=None,
            y0=state_in["h"],
            args=efficiency,
            saveat=saveat,
            stepsize_controller=controller,
            adjoint=dfx.DirectAdjoint(),   # <-- inneficient but works for fwd and rev
            # adjoint=dfx.ForwardMode()  # When using fwd pass
            # adjoint not defined for rev pass
        )

        # rebuild states (outside the trace; all concrete)
        states = [
            get_props(cpx.HmassP_INPUTS, h_i, p_i, fluid)
            for p_i, h_i in zip(sol.ts, sol.ys)            # <-- use ts and ys
        ]
        state_out = states[-1]


    else:
        raise ValueError("Invalid efficiency_type. Use 'isentropic' or 'polytropic'.")

    # Compute work
    isentropic_work = state_in["h"] - state_out_is["h"]
    specific_work = state_in["h"] - state_out["h"]

    # Create result dictionary
    result = {
        "type": "expander",
        "fluid_name": fluid.name,
        "states": states,
        "state_in": state_in,
        "state_out": state_out,
        "efficiency": efficiency,
        "efficiency_type": efficiency_type,
        "specific_work": specific_work,
        "isentropic_work": isentropic_work,
        "pressure_ratio": state_in["p"] / state_out["p"],
        "color": "black",
    }

    return result


def exit_temp(h_in, p_in, p_out, efficiency, fluid, efficiency_type):
    res = expansion_process(fluid, h_in, p_in, p_out, efficiency, efficiency_type)
    return res["state_out"]["T"]


# --- test values ---
fluid_name = "CO2"
p_in = 300e5
p_out = 100e5
T_in = 500
efficiency = 0.8
efficiency_type = "polytropic"

# Compute inlet state
fluid = cpx.Fluid(name=fluid_name, backend="HEOS")
state_in = fluid.get_state(cpx.PT_INPUTS, p_in, T_in)
h_in = state_in["h"]

# Solve the polytropic expansion
solution = expansion_process(
    fluid,
    h_in=h_in,
    p_in=p_in,
    p_out=p_out,
    efficiency=efficiency,
    efficiency_type=efficiency_type,
)

# scalar we care about: T_exit(x)
def exit_T_vec(x):
    h, pin, pout, eta = x
    return exit_temp(h, pin, pout, eta, fluid=fluid, efficiency_type=efficiency_type)

# base point
x0 = jnp.array([h_in, p_in, p_out, efficiency], dtype=jnp.float64)

# JAX grads (vector in R^4)
grad_fwd = jax.jacfwd(exit_T_vec)(x0)   # forward-mode
grad_rev = jax.jacrev(exit_T_vec)(x0)   # reverse-mode (same as jax.grad for scalar output)

print("grad (fwd):", grad_fwd)
print("grad (rev):", grad_rev)


# approx_derivative expects R^n->R^m; wrap to return shape (1,)
from scipy.optimize._numdiff import approx_derivative
fd = approx_derivative(lambda x: np.atleast_1d(exit_T_vec(x)),
                       x0=np.asarray(x0), method="2-point", rel_step=1e-6)
print("grad (FD) :", fd)


# Plot expansion in T-s diagram
prop_x = "s"
prop_y = "T"
fig, ax = fluid.plot_phase_diagram(x_prop=prop_x, y_prop=prop_y)
states = cpx.states_to_dict(solution["states"])
ax.plot(states[prop_x], states[prop_y])
plt.show()