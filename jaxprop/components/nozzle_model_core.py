import jax
import jax.numpy as jnp
import jaxprop as jxp




def nozzle_single_phase_core(t, y, args):
    """Wrapper that adapts from (t, y) to autonomous form."""
    Y = jnp.concatenate([jnp.atleast_1d(t), y])
    return nozzle_single_phase_autonomous(0.0, Y, args)


def nozzle_single_phase_autonomous(tau, Y, args):
    """
    Autonomous formulation of the nozzle equations:
        dx/dt   = det(A)
        dy_i/dt = det(A with column i replaced by b)

    State vector: Y = [x, v, rho, p]
    """
    x, v, d, p = Y

    # v = jnp.where(jnp.abs(v) < 1e-6, jnp.sign(v)*1e-6, v)

    # --- Geometry / model parameters ---
    fluid = args.fluid
    L = args.length
    eps_wall = args.roughness
    T_ext = args.T_wall
    wall_friction = args.wall_friction
    heat_transfer = args.heat_transfer

    # --- Thermodynamic state ---
    state = fluid.get_props(jxp.DmassP_INPUTS, d, p)
    T = state["T"]
    h = state["h"]
    s = state["s"]
    a = state["a"]
    cp = state["cp"]
    mu = state["mu"]
    G = state["gruneisen"]

    # Stagnation state
    h0 = state["h"] + 0.5 * v**2
    state0 = fluid.get_props(jxp.HmassSmass_INPUTS, h0, state["s"])
    p0 = state0["p"]
    T0 = state0["T"]
    d0 = state0["d"]

    # --- Geometry ---
    A, dAdx, perimeter, diameter = args.geometry(x, L)

    # --- Wall heat transfer and friction ---
    Re = v * d * diameter / jnp.maximum(mu, 1e-12)
    f_D = get_friction_factor_haaland(Re, eps_wall, diameter)
    tau_w = get_wall_viscous_stress(f_D, d, v)
    htc = 10000*get_heat_transfer_coefficient(v, d, cp, f_D)
    htc = jnp.clip(htc, 0.0, 1e6)   # pick bound based on your scaling
    q_w = htc * (T_ext - T)

    # Mask with booleans (convert to 0.0 if disabled)
    tau_w = wall_friction * tau_w
    f_D = wall_friction * f_D
    q_w = heat_transfer * q_w
    htc = heat_transfer * htc

    # jax.debug.print(
    #     "raw htc={:.4e}, raw q_w={:.4e}, T_ext={:.2f}, T={:.2f}", 
    #     htc, q_w, T_ext, T
    # )

    # --- Build A matrix and b vector ---
    A_mat = jnp.array([[d, v, 0.0], [d * v, 0.0, 1.0], [0.0, -(a**2), 1.0]])

    b_vec = jnp.array(
        [
            -d * v / A * dAdx,
            -(perimeter / A) * tau_w,
            (perimeter / A) * (G / v) * (tau_w * v + q_w),
        ]
    )

    # --- Determinants ---
    D = jnp.linalg.det(A_mat)

    # Replace columns one by one to compute N_i
    N = []
    for i in range(3):
        A_mod = A_mat.at[:, i].set(b_vec)
        N.append(jnp.linalg.det(A_mod))
    N = jnp.array(N)

    # --- Autonomous system: dx/dτ = D, dy/dτ = N_i ---
    dx_dtau = D
    dv_dtau = N[0]
    dd_dtau = N[1]
    dp_dtau = N[2]
    rhs = jnp.array([dv_dtau / dx_dtau, dd_dtau / dx_dtau, dp_dtau / dx_dtau])
    rhs_autonomous = jnp.array([dx_dtau, dv_dtau, dd_dtau, dp_dtau])

    # Export data
    out = {
        "x": x,
        "v": v,
        "d": d,
        "p": p,
        "rhs": rhs,
        "rhs_autonomous": rhs_autonomous,
        "A": A,
        "dAdx": dAdx,
        "diameter": diameter,
        "perimeter": perimeter,
        "h0": h0,
        "p0": p0,
        "T0": T0,
        "d0": d0,
        "Ma": v / state["a"],
        "Re": Re,
        "f_D": f_D,
        "tau_w": tau_w,
        "q_w": q_w,
        "htc": htc,
        "m_dot": d * v * A,
        "A_mat": A_mat,
        "b_vec": b_vec,
        "D": D,
        "N": N,
    }



    # A = {**out, **state.to_dict(include_aliases=True)}

    # print("Contents of A:")
    # for k, v in A.items():

    #     print(f"{k}: {type(v)} = {v}")


    return {**out, **state.to_dict(include_aliases=True)}
    # return {**out, **state}


# -----------------------------------------------------------------------------
# Functions to calculate heat transfer and friction
# -----------------------------------------------------------------------------


def get_friction_factor_haaland(Reynolds, roughness, diameter):
    """
    Computes the Darcy-Weisbach friction factor using the Haaland equation.

    The Haaland equation provides an explicit formulation for the friction factor
    that is simpler to use than the Colebrook equation, with an acceptable level
    of accuracy for most engineering applications.
    This function implements the Haaland equation as it is presented in many fluid
    mechanics textbooks, such as "Fluid Mechanics Fundamentals and Applications"
    by Cengel and Cimbala (equation 12-93).

    Parameters
    ----------
    reynolds : float
        The Reynolds number, dimensionless, indicating the flow regime.
    roughness : float
        The absolute roughness of the pipe's internal surface (m).
    diameter : float
        The inner diameter of the pipe (m).

    Returns
    -------
    f : float
        The computed friction factor, dimensionless.
    """
    Re_safe = jnp.maximum(Reynolds, 1.0)
    term = 6.9 / Re_safe + (roughness / diameter / 3.7) ** 1.11
    f = (-1.8 * jnp.log10(term)) ** -2
    return f


def get_wall_viscous_stress(darcy_friction_factor, density, velocity):
    """Wall shear stress from Darcy-Weisbach friction factor.

    Parameters
    ----------
    darcy_friction_factor : float
        Darcy-Weisbach friction factor (dimensionless).
    density : float
        Fluid density (kg/m^3).
    velocity : float
        Fluid velocity (m/s).

    Returns
    -------
    float
        Wall shear stress (Pa).
    """
    return 0.125 * darcy_friction_factor * density * velocity**2


def get_heat_transfer_coefficient(
    velocity, density, heat_capacity, darcy_friction_factor
):
    """
    Estimates the heat transfer using the Reynolds analogy.

    This function is an adaptation of the Reynolds analogy which relates the heat transfer
    coefficient to the product of the Fanning friction factor, velocity, density, and heat
    capacity of the fluid. The Fanning friction factor one fourth of the Darcy friction factor.

    Parameters
    ----------
    velocity : float
        Velocity of the fluid (m/s).
    density : float
        Density of the fluid (kg/m^3).
    heat_capacity : float
        Specific heat capacity of the fluid at constant pressure (J/kg·K).
    darcy_friction_factor : float
        Darcy friction factor, dimensionless.

    Returns
    -------
    float
        Estimated heat transfer coefficient (W/m^2·K).

    """
    fanning_friction_factor = darcy_friction_factor / 4
    return 0.5 * fanning_friction_factor * velocity * density * heat_capacity


# ------------------------------------------------------------------
# Describe the geometry of the converging diverging nozzle
# ------------------------------------------------------------------
def symmetric_nozzle_geometry(x, L, A_inlet=0.30, A_throat=0.15):
    """
    Return A (m^2), dA/dx (m), perimeter (m), diameter (m) for a symmetric parabolic CD nozzle.
    x: position in m (scalar or array)
    L: total length in m (scalar)
    """

    def area_fn(x_):
        xi = x_ / L
        return A_inlet - 4.0 * (A_inlet - A_throat) * xi * (1.0 - xi)

    # make it work for both scalar and array x
    A = area_fn(x)

    # jacfwd works for vector outputs directly
    dAdx = jax.jacfwd(area_fn)(x)

    radius = jnp.sqrt(A / jnp.pi)          # m
    diameter = 2.0 * radius                # m
    perimeter = jnp.pi * diameter          # m

    return A, dAdx, perimeter, diameter

