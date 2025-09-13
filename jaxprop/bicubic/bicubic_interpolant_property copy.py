import jax
import jax.numpy as jnp
import numpy as np

# from .jax_bicubic_HEOS_interpolation_1 import (
#     compute_bicubic_coefficients_of_ij,
#     bicubic_interpolant,
# )

# ======================== Config ========================
# NCORES = psutil.cpu_count(logical=False)
# os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NCORES}"
# jax.config.update("jax_enable_x64", True)

from .. import helpers_props as jxp


def bicubic_interpolant_property(h, P, table):
    """
    Interpolates all properties at a given enthalpy (h) and pressure (P) using bicubic interpolation.

    Args:
        h (float): Enthalpy in J/kg
        P (float): Pressure in Pa
        table_path (str): Path to saved property table (dict-based .pkl)

    Returns:
        dict: Dictionary of interpolated property values at (h, P)
    """

    h = jnp.asarray(h)
    P = jnp.asarray(P)
    h_vals = jnp.array(table["h_vals"])
    P_vals = jnp.array(table["p_vals"])

    Nh, Np = len(h_vals), len(P_vals)
    hmin, hmax = float(h_vals[0]), float(h_vals[-1])
    Lmin, Lmax = float(jnp.log(P_vals[0])), float(jnp.log(P_vals[-1]))

    logP = jnp.log(P)

    # Identify cell (i, j)
    i = int((h - hmin) / (hmax - hmin) * (Nh - 1))
    j = int((logP - Lmin) / (Lmax - Lmin) * (Np - 1))

    i = np.clip(i, 0, Nh - 2)
    j = np.clip(j, 0, Np - 2)

    deltah = float(h_vals[1] - h_vals[0])
    deltaL = float(jnp.log(P_vals[1]) - jnp.log(P_vals[0]))

    interpolated_props = {}

    for prop_name in jxp.PROPERTIES_CANONICAL:
        # if prop_name in ["enthalpy", "pressure"]:
        #     continue  # skip grid axes

        prop_data = table[prop_name]
        f_grid = prop_data["value"]
        fx_grid = prop_data["grad_h"]
        fy_grid = prop_data["grad_p"]
        fxy_grid = prop_data["grad_ph"]

        coeffs_local = compute_bicubic_coefficients_of_ij(
            i, j, f_grid, fx_grid * deltah, fy_grid * deltaL, fxy_grid * deltah * deltaL
        )

        coeffs = jnp.zeros((Nh, Np, 16), dtype=jnp.float64)
        coeffs = coeffs.at[i, j, :].set(coeffs_local)

        val = bicubic_interpolant(
            h, P, h_vals, jnp.log(P_vals), coeffs, Nh, Np, hmin, hmax, Lmin, Lmax
        )

        interpolated_props[prop_name] = float(val)

    return interpolated_props


# Global precision
float64 = jnp.dtype("float64")
complex128 = jnp.dtype("complex128")

# fmt: off
A = [
    [+1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [-3.0, +3.0, +0.0, +0.0, -2.0, -1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+2.0, -2.0, +0.0, +0.0, +1.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, -3.0, +3.0, +0.0, +0.0, -2.0, -1.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +2.0, -2.0, +0.0, +0.0, +1.0, +1.0, +0.0, +0.0],
    [-3.0, +0.0, +3.0, +0.0, +0.0, +0.0, +0.0, +0.0, -2.0, +0.0, -1.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, -3.0, +0.0, +3.0, +0.0, +0.0, +0.0, +0.0, +0.0, -2.0, +0.0, -1.0, +0.0],
    [+9.0, -9.0, -9.0, +9.0, +6.0, +3.0, -6.0, -3.0, +6.0, -6.0, +3.0, -3.0, +4.0, +2.0, +2.0, +1.0],
    [-6.0, +6.0, +6.0, -6.0, -3.0, -3.0, +3.0, +3.0, -4.0, +4.0, -2.0, +2.0, -2.0, -2.0, -1.0, -1.0],
    [+2.0, +0.0, -2.0, +0.0, +0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0],
    [+0.0, +0.0, +0.0, +0.0, +2.0, +0.0, -2.0, +0.0, +0.0, +0.0, +0.0, +0.0, +1.0, +0.0, +1.0, +0.0],
    [-6.0, +6.0, +6.0, -6.0, -4.0, -2.0, +4.0, +2.0, -3.0, +3.0, -3.0, +3.0, -2.0, -1.0, -2.0, -1.0],
    [+4.0, -4.0, -4.0, +4.0, +2.0, +2.0, -2.0, -2.0, +2.0, -2.0, +2.0, -2.0, +1.0, +1.0, +1.0, +1.0],
]
# fmt: on


# =================== Functions to Export ===================
# @jax.jit
def compute_bicubic_coefficients_of_ij(i, j, f, fx, fy, fxy):
    # xx=f(0,0)&f(1,0)&f(0,1)&f(1,1)&f_x(0,0)&f_x(1,0)&f_x(0,1)&f_x(1,1)&f_y(0,0)&f_y(1,0)&f_y(0,1)&f_y(1,1)&f_{xy}(0,0)&f_{xy}(1,0)&f_{xy}(0,1)&f_{xy}(1,1)
    xx = [
        f[i, j],
        f[i + 1, j],
        f[i, j + 1],
        f[i + 1, j + 1],
        fx[i, j],
        fx[i + 1, j],
        fx[i, j + 1],
        fx[i + 1, j + 1],
        fy[i, j],
        fy[i + 1, j],
        fy[i, j + 1],
        fy[i + 1, j + 1],
        fxy[i, j],
        fxy[i + 1, j],
        fxy[i, j + 1],
        fxy[i + 1, j + 1],
    ]

    return jnp.matmul(jnp.array(A, dtype=f.dtype), jnp.array(xx, dtype=f.dtype))


# @partial(jit, static_argnums=(5, 6))
def bicubic_interpolant(h, P, h_vals, P_vals, coeffs, Nh, Np, hmin, hmax, Lmin, Lmax):
    """
    Evaluate the bicubic interpolant at (h, P) using precomputed coefficients.
    """
    # Log-transform P
    L = jnp.log(P)

    # Normalize positions to [0, 1] cell coordinates
    ii = (h - hmin) / (hmax - hmin) * (Nh - 1)
    i = ii.astype(int)
    x = ii - i

    jj = (L - Lmin) / (Lmax - Lmin) * (Np - 1)
    j = jj.astype(int)
    y = jj - j

    # Evaluate bicubic polynomial
    result = jnp.zeros_like(h)  # use h shape
    x_pow = jnp.ones_like(h)  # x^0

    for m in range(4):  # m = x power
        y_pow = jnp.ones_like(h)  # y^0 initially
        for n in range(4):  # n = y power
            c = coeffs[i, j, 4 * n + m]
            result += c * x_pow * y_pow
            y_pow = y_pow * y
        x_pow = x_pow * x

    return result
