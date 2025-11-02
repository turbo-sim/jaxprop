import jax
import jax.numpy as jnp
import jaxprop as jxp
# import jaxprop.perfect_gas as pg

from scipy.optimize._numdiff import approx_derivative

# ------------------------------------------------------------------
# Polytropic temperature calculation:
# Compressor (common): T2 = T1 * (p2/p1)^((gamma-1)/(gamma*eta_p))
# Turbine variant would be: ((gamma-1)*eta_p/gamma)
# ------------------------------------------------------------------
@jax.jit
def rho_out_polytropic(x, fluid):
    T_in, p_in, p_out, eta_p = x
    gamma = fluid.constants.gamma
    pr = p_out / p_in
    eta_p = eta_p / 100
    exponent = (gamma - 1.0) / (gamma * eta_p)  # compressor form
    T_out = T_in * jnp.power(pr, exponent)
    d_out = fluid.get_state(jxp.PT_INPUTS, p_out, T_out)["d"]
    return d_out

# Inputs
fluid = "air"
p_in = 101325.0
p_out = 5.0 * p_in
T_in = 300.0
eta_p = 90

# Pull gamma from your perfect-gas constants at the inlet state
fluid = jxp.FluidPerfectGas("air", T_ref=298.15, p_ref=101325.0)

# Base point (match function order!)
x0 = jnp.array([T_in, p_in, p_out, eta_p])

# Value
rho_out = rho_out_polytropic(x0, fluid)
print("Fluid properties:")
print(f"  eta_p    : {float(eta_p):+0.6f} -")
print(f"  T_in     : {float(T_in):+0.6f} K")
print(f"  p_in     : {float(p_in):+0.6f} Pa")
print(f"  p_out    : {float(p_out):+0.6f} Pa")
print(f"  rho_out    : {float(rho_out):+0.6f} K")

# Forward JAX gradients
grad_fwd = jax.jacfwd(rho_out_polytropic)(x0, fluid)
print("\nForward JAX gradients:")
print(f"  dT/dT_in     : {float(grad_fwd[0]):+0.6f} -")
print(f"  dT/dp_in     : {float(grad_fwd[1]):+0.6f} K/Pa")
print(f"  dT/dp_out    : {float(grad_fwd[2]):+0.6f} K/Pa")
print(f"  dT/deta_p    : {float(grad_fwd[3]):+0.6f} K")

# Reverse JAX gradients
grad_rev = jax.jacrev(rho_out_polytropic)(x0, fluid)
print("\nReverse JAX gradients:")
print(f"  dT/dT_in     : {float(grad_rev[0]):+0.6f} -")
print(f"  dT/dp_in     : {float(grad_rev[1]):+0.6f} K/Pa")
print(f"  dT/dp_out    : {float(grad_rev[2]):+0.6f} K/Pa")
print(f"  dT/deta_p    : {float(grad_rev[3]):+0.6f} K")

# SciPy finite-difference gradient with gamma passed as an extra argument
grad_fd = approx_derivative(
    fun=rho_out_polytropic,
    x0=x0,
    method="2-point",
    rel_step=1e-7,
    args=(fluid,),
)

print("\ngradients (scipy approx_derivative):")
print(f"  dT/dT_in     : {float(grad_fd[0]):+0.6f} -")
print(f"  dT/dp_in     : {float(grad_fd[1]):+0.6f} K/Pa")
print(f"  dT/dp_out    : {float(grad_fd[2]):+0.6f} K/Pa")
print(f"  dT/deta_p    : {float(grad_fd[3]):+0.6f} K")

# Relative errors (FD vs JAX forward)
def rel(a, b):
    a = float(a); b = float(b)
    return abs(a - b) / max(1.0, abs(b))

print("\nrelative error (fd vs jax forward):")
print(f"  dT/dT_in     : {rel(grad_fd[0], grad_fwd[0]):+0.10e}")
print(f"  dT/dp_in     : {rel(grad_fd[1], grad_fwd[1]):+0.10e}")
print(f"  dT/dp_out    : {rel(grad_fd[2], grad_fwd[2]):+0.10e}")
print(f"  dT/deta_p    : {rel(grad_fd[3], grad_fwd[3]):+0.10e}")


# -------------------------- hessian via jax.hessian (scalar) --------------------------
H_scalar = jax.hessian(lambda x: rho_out_polytropic(x, fluid), argnums=0)(x0)

names = ["T_in", "p_in", "p_out", "eta_p"]
print("\nHessian for rho_out using jax.hessian")
header = "           " + "".join([f"{n:>16s}" for n in names])
print(header)
for i, ri in enumerate(names):
    row = "".join(f"{H_scalar[i, j]:+16.6e}" for j in range(len(names)))
    print(f"{ri:>10s} {row}")


# ----------- hessians via forward-over-reverse for multiple outputs -------------
fun = lambda x: jnp.atleast_1d(rho_out_polytropic(x, fluid))
H_multi = jax.jacfwd(jax.jacrev(fun), argnums=0)(x0)
out_names = ["rho_out"]
print("\nHessian for rho_out using jax.jacrev + jax.jacfwd")
for m_idx, m_name in enumerate(out_names):
    header = "           " + "".join([f"{n:>16s}" for n in names])
    print(header)
    for i, ri in enumerate(names):
        row = "".join(f"{H_multi[m_idx, i, j]:+16.6e}" for j in range(len(names)))
        print(f"{ri:>10s} {row}")



# --------------------------- final verification --------------------------- #
TOL = 1e-6
violations = []

names = ["T_in", "p_in", "p_out", "eta_p"]
for i, name in enumerate(names):
    fwd_val = float(grad_fwd[i])
    rev_val = float(grad_rev[i])
    fd_val  = float(grad_fd[i])

    err_fwd_rev = abs(fwd_val - rev_val) / max(1e-14, abs(rev_val))
    err_fwd_fd  = abs(fwd_val - fd_val) / max(1e-14, abs(fd_val))

    if err_fwd_rev > TOL:
        violations.append((name, "fwd vs rev", err_fwd_rev))
    if err_fwd_fd > TOL:
        violations.append((name, "fwd vs fd", err_fwd_fd))

if violations:
    msg_lines = ["The following gradient checks failed:"]
    for name, kind, err in violations:
        msg_lines.append(f"  - {name}: {kind} relative error = {err:.3e}")
    raise ValueError("\n".join(msg_lines))
else:
    print(f"\nAll gradient checks passed (relative errors < {TOL:.1e}).")
