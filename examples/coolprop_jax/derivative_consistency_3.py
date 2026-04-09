import jax
import jaxprop as jxp
import numpy as np
from scipy.optimize._numdiff import approx_derivative as fd

# baseline (PT)
T_ref = 300.0
p_ref = 101325.0
fluid = jxp.FluidJAX(name="nitrogen", backend="HEOS")
ref = fluid.get_state(jxp.PT_INPUTS, p_ref, T_ref)

# input pair HmassP
input_pair = jxp.HmassP_INPUTS
h0, p0 = ref["h"], ref["p"]

# primals
st = fluid.get_state(input_pair, h0, p0)
print("state (HmassP_INPUTS):")
for k in ("T", "p", "d", "h", "s", "a", "mu", "k", "gamma"):
    print(f"  {k:>5s}: {float(np.asarray(st[k])):+0.6f}")

# scalar extractors
def T_of(h, p):
    return fluid.get_state(input_pair, h, p)["T"]

def rho_of(h, p):
    return fluid.get_state(input_pair, h, p)["d"]

# forward-mode
dT_dh = jax.jacfwd(T_of, argnums=0)(h0, p0)
dT_dp = jax.jacfwd(T_of, argnums=1)(h0, p0)
dd_dh = jax.jacfwd(rho_of, argnums=0)(h0, p0)
dd_dp = jax.jacfwd(rho_of, argnums=1)(h0, p0)
print("\npartials (forward-mode):")
print(f"  dT/dh|p   : {float(dT_dh):+0.6e} K*kg/J")
print(f"  dT/dp|h   : {float(dT_dp):+0.6e} K/Pa")
print(f"  dd/dh|p   : {float(dd_dh):+0.6e} kg*m^-3*kg/J")
print(f"  dd/dp|h   : {float(dd_dp):+0.6e} kg*m^-3/Pa")

# reverse-mode
dT_dh_rev = jax.jacrev(T_of, argnums=0)(h0, p0)
dT_dp_rev = jax.jacrev(T_of, argnums=1)(h0, p0)
dd_dh_rev = jax.jacrev(rho_of, argnums=0)(h0, p0)
dd_dp_rev = jax.jacrev(rho_of, argnums=1)(h0, p0)
print("\npartials (reverse-mode):")
print(f"  dT/dh|p   : {float(dT_dh_rev):+0.6e} K*kg/J")
print(f"  dT/dp|h   : {float(dT_dp_rev):+0.6e} K/Pa")
print(f"  dd/dh|p   : {float(dd_dh_rev):+0.6e} kg*m^-3*kg/J")
print(f"  dd/dp|h   : {float(dd_dp_rev):+0.6e} kg*m^-3/Pa")


# finite difference with relative step 1e-6
rel_step = 1e-6

# T_of derivatives
dT_dh_fd = fd(
    lambda h: T_of(h, p0), x0=np.atleast_1d(h0), rel_step=rel_step, method="2-point"
)[0]

dT_dp_fd = fd(
    lambda p: T_of(h0, p), x0=np.atleast_1d(p0), rel_step=rel_step, method="2-point"
)[0]

# rho_of derivatives
dd_dh_fd = fd(
    lambda h: rho_of(h, p0), x0=np.atleast_1d(h0), rel_step=rel_step, method="2-point"
)[0]

dd_dp_fd = fd(
    lambda p: rho_of(h0, p), x0=np.atleast_1d(p0), rel_step=rel_step, method="2-point"
)[0]

print("\nfinite-difference (SciPy, rel_step=1e-6):")
print(f"  dT/dh|p   : {dT_dh_fd:+0.6e} K*kg/J")
print(f"  dT/dp|h   : {dT_dp_fd:+0.6e} K/Pa")
print(f"  dd/dh|p   : {dd_dh_fd:+0.6e} kg*m^-3*kg/J")
print(f"  dd/dp|h   : {dd_dp_fd:+0.6e} kg*m^-3/Pa")


# --------------------- derivative consistency check summary --------------------- #
def rel_err(val, ref):
    denom = max(1e-12, abs(ref))
    return abs(val - ref) / denom

TOL = 1e-6
checks = [
    ("dT/dh|p", dT_dh, dT_dh_rev, dT_dh_fd),
    ("dT/dp|h", dT_dp, dT_dp_rev, dT_dp_fd),
    ("dd/dh|p", dd_dh, dd_dh_rev,   dd_dh_fd),
    ("dd/dp|h", dd_dp, dd_dp_rev,   dd_dp_fd),
]

violations = []

print("\nconsistency checks (forward vs reverse vs finite diff):")
print(f"{'Derivative':<15} {'fwd':>15} {'rev':>15} {'fd':>15} {'fwd-rev err':>15} {'fwd-fd err':>15}")

for name, fwd, rev, fd in checks:
    fwd_val = float(np.asarray(fwd))
    rev_val = float(np.asarray(rev))
    fd_val  = float(np.asarray(fd))

    err_fwd_rev = rel_err(fwd_val, rev_val)
    err_fwd_fd  = rel_err(fwd_val, fd_val)

    print(f"{name:<15} {fwd_val:15.6e} {rev_val:15.6e} {fd_val:15.6e} {err_fwd_rev:15.3e} {err_fwd_fd:15.3e}")

    if err_fwd_rev > TOL:
        violations.append((name, "fwd-rev", err_fwd_rev))
    if err_fwd_fd > TOL:
        violations.append((name, "fwd-fd", err_fwd_fd))

# summary
if violations:
    msg_lines = ["\nThe following derivative checks exceed the tolerance:"]
    for name, kind, err in violations:
        msg_lines.append(f"  - {name} ({kind}): rel err = {err:.3e}")
    raise ValueError("\n".join(msg_lines))
else:
    print(f"\nAll derivatives consistent (relative errors ≤ {TOL:.1e}).")
