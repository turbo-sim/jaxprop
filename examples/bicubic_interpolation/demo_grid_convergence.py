import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxprop as jxp


jxp.set_plot_options(grid=True)


# ---------------------------
# Config
# ---------------------------
grad_method = "central"
outdir = f"results_grid_convergence_{grad_method}"
fluid_name = "CO2"
backend = "HEOS"


h_min, h_max = 600e3, 1200e3  # J/kg
p_min, p_max = 2e6, 20e6  # Pa

# Grid sizes to test (square grids here; adjust as you like)
resolutions = [8, 16, 32, 64, 128]
# resolutions = [8, 16, 32, 64, 128, 256]

# Properties to check (keys must match FluidState)
props_to_check = ["temperature", "density", "entropy", "speed_sound"]

# ---------------------------
# Validation grid
# ---------------------------
N_test = 30
h_nodes_val = jnp.linspace(h_min, h_max, N_test + 1)
logp_nodes_val = jnp.linspace(jnp.log(p_min), jnp.log(p_max), N_test + 1)
h_val = 0.5 * (h_nodes_val[:-1] + h_nodes_val[1:])
p_val = jnp.exp(0.5 * (logp_nodes_val[:-1] + logp_nodes_val[1:]))
H_val, P_val = jnp.meshgrid(h_val, p_val, indexing="ij")
fluid_ref = jxp.FluidJAX(fluid_name, backend=backend)
state_ref = fluid_ref.get_state(jxp.HmassP_INPUTS, H_val, P_val)

# ---------------------------
# Convergence sweep
# ---------------------------
results = {prop: {"abs_error": [], "rel_error": []} for prop in props_to_check}

for N in resolutions:

    # Define bicubic interpolation fluid
    print(f"\n=== Building bicubic model N_h=N_p={N} ===")
    fluid_bicubic = jxp.FluidBicubic(
        fluid_name=fluid_name,
        backend=backend,
        h_min=h_min,
        h_max=h_max,
        p_min=p_min,
        p_max=p_max,
        N_h=N,
        N_p=N,
        table_dir=outdir,
        gradient_method=grad_method,
    )

    # Interpolate on validation points
    state_interp = fluid_bicubic.get_state(jxp.HmassP_INPUTS, state_ref.h, state_ref.p)
    # state_interp = fluid_bicubic.get_state(jxp.PT_INPUTS, state_ref.p, state_ref.T)
    # state_interp = fluid_bicubic.get_state(jxp.PT_INPUTS, state_ref.p, state_ref.T)

    # Compute errors
    for prop in props_to_check:
        eps = 1e-16
        interp = np.ravel(np.array(state_interp[prop]))
        ref = np.ravel(np.array(state_ref[prop]))
        diff = interp - ref
        N_test = diff.size
        abs_error = np.linalg.norm(diff, ord=2) / np.sqrt(N_test)
        rel = diff / np.maximum(np.abs(ref), eps)
        rel_error = np.linalg.norm(rel, ord=2) / np.sqrt(N_test)
        results[prop]["abs_error"].append(abs_error)
        results[prop]["rel_error"].append(rel_error)


# ---------------------------
# Print summary table
# ---------------------------
header = f"{'Property':<20}{'N':>8}{'Abs error':>15}{'Rel error':>15}"
print("\n" + "-" * len(header))
print(header)
print("-" * len(header))
for prop in props_to_check:
    for N, abs_error, rel_error in zip(
        resolutions, results[prop]["abs_error"], results[prop]["rel_error"]
    ):
        print(f"{prop:<20}{N:8d}{abs_error:15.3e}{rel_error:15.3e}")
print("-" * len(header))


# ---------------------------------------------------
# Visualize interpolation region + convergence plot
# ---------------------------------------------------

# Plot thermodynamic region
fig, ax = fluid_ref.fluid.plot_phase_diagram(
    x_prop="enthalpy", y_prop="pressure", x_scale="linear", y_scale="log"
)
ax.scatter(H_val, P_val, s=10, c="tab:orange")
h_box = [h_min, h_max, h_max, h_min, h_min]
p_box = [p_min, p_min, p_max, p_max, p_min]
ax.plot(h_box, p_box, "r-", linewidth=1.25, label="interpolation domain")
ax.set_xlabel("Enthalpy [J/kg]")
ax.set_ylabel("Pressure [Pa]")
ax.legend(loc="best", frameon=True)
fig.tight_layout(pad=1)

# Plot grid convergence trend
N0 = min(resolutions)
fig, ax = plt.subplots(figsize=(6, 4))
colors = plt.cm.magma(jnp.linspace(0.4, 0.8, len(props_to_check)))

# Reference N^-4 line
ax.loglog(
    [N0, max(resolutions)],
    [1.0, (max(resolutions) / N0) ** (-4)],
    "k--",
    lw=1.75,
    label=r"$\mathcal{O}(N^{-4})$",
)

for prop, color in zip(props_to_check, colors):
    N_vals = np.array(resolutions)
    error = np.array(results[prop]["rel_error"])
    order = np.argsort(N_vals)
    N_vals = N_vals[order]
    error = error[order]
    err0 = error[N_vals.tolist().index(N0)]  # error at coarsest grid
    err_norm = error / err0

    ax.loglog(
        N_vals,
        err_norm,
        marker="o",
        color=color,
        linewidth=1.75,
        label=prop.replace("_", " ").capitalize(),
    )

ax.set_xlabel("Number of grid points per direction (N)")
ax.set_ylabel("Normalized relative error")
ax.legend(ncol=1, fontsize=10.5, loc="upper right", frameon=True)
ax.set_xscale("log", base=2)
ax.set_xticks(resolutions)
ax.set_xticklabels([str(n) for n in resolutions])
plt.tight_layout(pad=1)
jxp.savefig_in_formats(fig, os.path.join(outdir, f"grid_convergence_{grad_method}"))


# Show figures
plt.show()
