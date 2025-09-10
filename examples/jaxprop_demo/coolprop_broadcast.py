import time
import jax
import jax.numpy as jnp
import jaxprop as cpx

# ---------------------------------------------------------------------
# reference state and fluids
# ---------------------------------------------------------------------
p0 = 101325.0   # Pa
T0 = 300.0      # K

fluid_perfect_gas = cpx.FluidPerfectGas(name="air", T_ref=T0, p_ref=p0)
fluid_coolprop    = cpx.FluidJAX(name="air")  # requires broadcast-aware get_props

# sizes to test (compilation time included; no warmup)
sizes = [10, 10, 10, 100, 100, 100, 1000, 1000, 1000]

# ---------------------------------------------------------------------
# case A: P and T are arrays
# ---------------------------------------------------------------------
print("\n--- case A: P and T are arrays ---")
for n in sizes:
    key = jax.random.key(0)
    kP, kT = jax.random.split(key)
    pressures = p0 * (1.0 + 0.1 * jax.random.normal(kP, (n,)))
    temperatures = T0 * (1.0 + 0.1 * jax.random.normal(kT, (n,)))

    t0 = time.perf_counter()
    state_pg = fluid_perfect_gas.get_props(cpx.PT_INPUTS, pressures, temperatures)
    _ = state_pg.rho.block_until_ready()
    t1 = time.perf_counter()
    time_pg = (t1 - t0) * 1000.0

    t0 = time.perf_counter()
    state_cp = fluid_coolprop.get_props(cpx.PT_INPUTS, pressures, temperatures)
    _ = state_cp.rho.block_until_ready()
    t1 = time.perf_counter()
    time_cp = (t1 - t0) * 1000.0

    print(f"n={n:<4d}  PerfectGas: {time_pg:7.2f} ms | FluidJAX: {time_cp:7.2f} ms")

# ---------------------------------------------------------------------
# case B: P array, T scalar (broadcast)
# ---------------------------------------------------------------------
print("\n--- case B: P array, T scalar ---")
for n in sizes:
    key = jax.random.key(0)
    pressures = p0 * (1.0 + 0.1 * jax.random.normal(key, (n,)))
    T_scalar = jnp.asarray(T0)

    t0 = time.perf_counter()
    state_pg = fluid_perfect_gas.get_props(cpx.PT_INPUTS, pressures, T_scalar)
    _ = state_pg.rho.block_until_ready()
    t1 = time.perf_counter()
    time_pg = (t1 - t0) * 1000.0

    t0 = time.perf_counter()
    state_cp = fluid_coolprop.get_props(cpx.PT_INPUTS, pressures, T_scalar)
    _ = state_cp.rho.block_until_ready()
    t1 = time.perf_counter()
    time_cp = (t1 - t0) * 1000.0

    print(f"n={n:<4d}  PerfectGas: {time_pg:7.2f} ms | FluidJAX: {time_cp:7.2f} ms")
