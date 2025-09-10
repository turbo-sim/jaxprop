import time
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import optimistix as optx
import jaxprop as jxp


jxp.set_plot_options(grid=False)

# ---------- Compute static state from stagnation and Mach number ----------
@eqx.filter_jit
def compute_static_state(p0, d0, Ma, fluid):
    """solve h0 - h(p,s0) - 0.5 a(p,s0)^2 Ma^2 = 0 for p"""

    st0 = fluid.get_props(jxp.DmassP_INPUTS, d0, p0)
    s0, h0 = st0["s"], st0["h"]

    def residual(p_vec, _):
        p = p_vec[0]
        st = fluid.get_props(jxp.PSmass_INPUTS, p, s0)
        a, h = st["a"], st["h"]
        v = a * Ma
        return jnp.array([h0 - h - 0.5 * v * v])

    p_init = jnp.array([0.9 * p0])
    solver = optx.Newton(rtol=1e-10, atol=1e-10)
    sol = optx.root_find(residual, solver, y0=p_init, args=())

    state = fluid.get_props(jxp.PSmass_INPUTS, sol.value[0], s0)
    return state


# -------------------------------------------------------------------------
# Mach number sweep
# -------------------------------------------------------------------------
if __name__ == "__main__":

    # Fluid object
    fluid = jxp.FluidPerfectGas("air", T_ref=300.0, P_ref=101325.0)
    # fluid = jxp.FluidJAX("air")

    # Fixed inlet total conditions
    p0 = jnp.asarray(1e5)   # Pa
    d0 = jnp.asarray(1.2)     # kg/m3

    # Mach number sweep
    print("\n" + "-" * 60)
    print("Running Mach number sweep (static state calculation)")
    print("-" * 60)

    Ma_array = jnp.linspace(0.0, 1.0, 21)
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(Ma_array)))

    results = []
    for Ma, color in zip(Ma_array, colors):
        t0 = time.perf_counter()
        state = compute_static_state(p0, d0, Ma, fluid)
        dt_ms = (time.perf_counter() - t0) * 1e3

        print(f"Ma = {Ma:0.3f} | p = {state['p']*1e-5:0.3f} bar | time = {dt_ms:7.2f} ms")

        results.append({"Ma": Ma, "color": color, "state": state})

    # ---------------------------------------------------------------------
    # Plot pressure vs Mach number
    # ---------------------------------------------------------------------
    plt.figure(figsize=(5, 4))
    Ma_vals = [r["Ma"] for r in results]
    p_vals = [r["state"]["p"] * 1e-5 for r in results]  # convert Pa -> bar
    plt.plot(Ma_vals, p_vals, "o-", color="tab:blue")
    plt.xlabel("Mach number (-)")
    plt.ylabel("Static pressure (bar)")
    plt.tight_layout(pad=1)
    plt.show()
