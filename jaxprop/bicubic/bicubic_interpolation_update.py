import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
# import CoolProp.CoolProp as cp
# import turboflow as tf
import jaxprop.coolprop as jxp  # needed for input_tests


PROPERTY_ALIASES = {
    "T": ["temperature", "Temperature"],
    "d": ["density", "Density", "rho", "D"],
    "s": ["S", "entropy", "Entropy"],
    "h": ["H", "enthalpy", "Enthalpy"],
    "P": ["p", "pressure", "Pressure"],
    "mu": ["viscosity", "Viscosity"],
    "k": ["thermal_conductivity", "conductivity"],
}

ALIAS_TO_CANONICAL = {}
for canon, aliases in PROPERTY_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias] = canon
    ALIAS_TO_CANONICAL[canon] = canon  # also allow canonical itself

# ================================================================
# Bicubic coefficient computation (Code 2 style)
# ================================================================
@jax.jit
def compute_bicubic_coefficients_of_cell(i, j, f, fx, fy, fxy):
    xx = [
        f[i, j], f[i+1, j], f[i, j+1], f[i+1, j+1],
        fx[i, j], fx[i+1, j], fx[i, j+1], fx[i+1, j+1],
        fy[i, j], fy[i+1, j], fy[i, j+1], fy[i+1, j+1],
        fxy[i, j], fxy[i+1, j], fxy[i, j+1], fxy[i+1, j+1],
    ]

    A = np.array([
        [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        [0,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0],
        [-3,3,0,0, -2,-1,0,0, 0,0,0,0, 0,0,0,0],
        [2,-2,0,0, 1,1,0,0, 0,0,0,0, 0,0,0,0],

        [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0],
        [0,0,0,0, 0,0,0,0, -3,3,0,0, -2,-1,0,0],
        [0,0,0,0, 0,0,0,0, 2,-2,0,0, 1,1,0,0],

        [-3,0,3,0, 0,0,0,0, -2,0,-1,0, 0,0,0,0],
        [0,0,0,0, -3,0,3,0, 0,0,0,0, -2,0,-1,0],
        [9,-9,-9,9, 6,3,-6,-3, 6,-6,3,-3, 4,2,2,1],
        [-6,6,6,-6, -3,-3,3,3, -4,4,-2,2, -2,-2,-1,-1],

        [2,0,-2,0, 0,0,0,0, 1,0,1,0, 0,0,0,0],
        [0,0,0,0, 2,0,-2,0, 0,0,0,0, 1,0,1,0],
        [-6,6,6,-6, -4,-2,4,2, -3,3,-3,3, -2,-1,-2,-1],
        [4,-4,-4,4, 2,2,-2,-2, 2,-2,2,-2, 1,1,1,1],
    ], dtype=np.float64)

    return jnp.matmul(A, jnp.array(xx, dtype=f.dtype))


def compute_coefficients(f, fx, fy, fxy, deltah, deltaL):
    Nh, Np = f.shape
    coeffs = np.zeros((Nh-1, Np-1, 16))
    for i in range(Nh-1):
        for j in range(Np-1):
            coeffs[i,j,:] = compute_bicubic_coefficients_of_cell(
                i, j,
                f,
                fx * deltah,
                fy * deltaL,
                fxy * deltah * deltaL
            )
    return coeffs


# ================================================================
# Bicubic interpolant evaluation
# ================================================================
@jax.jit
def bicubic_interpolant(h, P, h_vals, logP_vals, coeffs, Nh, Np, hmin, hmax, Lmin, Lmax):
    L = jnp.log(P)

    ii = ((h - hmin) / (hmax - hmin) * (Nh - 1))
    i = jnp.clip(ii.astype(int), 0, Nh-2)
    x = ii - i

    jj = ((L - Lmin) / (Lmax - Lmin) * (Np - 1))
    j = jnp.clip(jj.astype(int), 0, Np-2)
    y = jj - j

    result = 0.0
    x_pow = 1.0
    for m in range(4):
        y_pow = 1.0
        for n in range(4):
            c = coeffs[i,j,4*n+m]
            result += c * x_pow * y_pow
            y_pow *= y
        x_pow *= x
    return result


# ================================================================
# FluidBicubic class (Code 2 style names preserved)
# ================================================================
class FluidBicubic(eqx.Module):
    fluid_name: str = eqx.field(static=True)
    h_min: float = eqx.field(static=True)
    h_max: float = eqx.field(static=True)
    p_min: float = eqx.field(static=True)
    p_max: float = eqx.field(static=True)
    N: int = eqx.field(static=True)
    table_name: str = eqx.field(static=True)
    table_dir: str = eqx.field(static=True)
    table: dict = eqx.field(static=False)

    def __init__(self, fluid_name, h_min, h_max, p_min, p_max, N, table_dir="fluid_tables"):
        self.fluid_name = fluid_name
        self.h_min, self.h_max = h_min, h_max
        self.p_min, self.p_max = p_min, p_max
        self.N = N
        self.table_name = f"{fluid_name}_{N}x{N}"
        self.table_dir = table_dir
        self.table = self._load_or_generate_table()

    # ------------------ Table handling ------------------
    def _load_or_generate_table(self):
        pkl_path = os.path.join(self.table_dir, f"{self.table_name}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        else:
            return self._generate_property_table(pkl_path)

    def _generate_property_table(self, pkl_path):
        # fluid = tf.Fluid(self.fluid_name)
        fluid = jxp.FluidJAX(self.fluid_name)
        h_vals = np.linspace(self.h_min, self.h_max, self.N)
        logP_vals = np.linspace(np.log(self.p_min), np.log(self.p_max), self.N)

        deltah = float(h_vals[1] - h_vals[0])
        deltaL = float(logP_vals[1] - logP_vals[0])

        table = {"h": h_vals, "P": np.exp(logP_vals)}

        props = {"T":"T","d":"D","s":"S","mu":"V","k":"L"}
        for key in props:
            table[key] = {
                "value": np.zeros((self.N,self.N)),
                "grad_h": np.zeros((self.N,self.N)),
                "grad_p": np.zeros((self.N,self.N)),
                "grad_logp": np.zeros((self.N,self.N)),
                "grad_ph": np.zeros((self.N,self.N)),
                "grad_logph": np.zeros((self.N,self.N)),
                "coeffs": np.zeros((self.N-1,self.N-1,16)),
            }

        for i, h in enumerate(h_vals):
            for j, L in enumerate(logP_vals):
                P = np.exp(L)
                eps_h = max(1e-6*abs(h), 1e-3*deltah)
                eps_P = max(1e-6*abs(P), 1e-3*(np.exp(deltaL)-1.0)*P)
                try:
                    # f0  = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h, P)
                    # fh  = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h+eps_h, P)
                    # fp  = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h, P+eps_P)
                    # fhp = tf.get_props_custom_jvp(fluid, cp.HmassP_INPUTS, h+eps_h, P+eps_P)

                    f0 = fluid.get_state(jxp.HmassP_INPUTS, h, P)
                    fh = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, P)
                    fp = fluid.get_state(jxp.HmassP_INPUTS, h, P + eps_P)
                    fhp = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, P + eps_P)

                except Exception:
                    continue

                for key in props:
                    val = f0[key]
                    dval_dh = (fh[key] - f0[key]) / eps_h
                    dval_dP = (fp[key] - f0[key]) / eps_P
                    d2val_dhdP = (fhp[key] - fh[key] - fp[key] + f0[key]) / (eps_h*eps_P)

                    table[key]["value"][i,j] = val
                    table[key]["grad_h"][i,j] = dval_dh
                    table[key]["grad_p"][i,j] = dval_dP
                    table[key]["grad_logp"][i,j] = dval_dP * P
                    table[key]["grad_ph"][i,j] = d2val_dhdP
                    table[key]["grad_logph"][i,j] = d2val_dhdP * P

        for key in props:
            table[key]["coeffs"] = compute_coefficients(
                table[key]["value"],
                table[key]["grad_h"],
                table[key]["grad_logp"],
                table[key]["grad_logph"],
                deltah, deltaL
            )

        os.makedirs(self.table_dir, exist_ok=True)
        with open(pkl_path,"wb") as f:
            pickle.dump(table,f)
        return table
    
    def _resolve_key(self, key: str) -> str:
        """Map property alias to canonical key (e.g. 'temperature' -> 'T')."""
        if key in ALIAS_TO_CANONICAL:
            return ALIAS_TO_CANONICAL[key]
        raise KeyError(f"Unknown property alias: {key}")

    # ------------------ Main bicubic property interpolation ------------------
    def bicubic_interpolant_property(self, h, P):
        h_vals = self.table["h"]
        p_vals = self.table["P"]
        logP_vals = np.log(p_vals)

        Nh, Np = len(h_vals), len(p_vals)
        hmin, hmax = h_vals[0], h_vals[-1]
        Lmin, Lmax = logP_vals[0], logP_vals[-1]

        props_out = {}
        for key in ["T","d","s","mu","k"]:
            coeffs = self.table[key]["coeffs"]
            val = bicubic_interpolant(h, P, h_vals, logP_vals, coeffs,
                                      Nh, Np, hmin, hmax, Lmin, Lmax)
            props_out[key] = float(val)

        props_out["h"] = float(h)
        props_out["P"] = float(P)

        # Add aliases for user-friendly access
        for canon, aliases in PROPERTY_ALIASES.items():
            if canon in props_out:
                for alias in aliases:
                    props_out[alias] = props_out[canon]

        return props_out

    # ------------------ Inverse solvers ------------------
    def inverse_interpolant_hx(self, h, x_target, x_prop, tol=1e-8, max_iter=1000):
        x_prop = self._resolve_key(x_prop)  # normalize alias
        P_vals = self.table["P"]
        P_lo, P_hi = float(P_vals[0]), float(P_vals[-1])

        def residual(P):
            props = self.bicubic_interpolant_property(h, P)
            return props[x_prop] - x_target

        for it in range(max_iter):
            P_mid = 0.5*(P_lo+P_hi)
            f_mid = residual(P_mid)
            if abs(f_mid) < tol:
                return self.bicubic_interpolant_property(h, P_mid), it
            f_lo = residual(P_lo)
            if f_lo * f_mid < 0:
                P_hi = P_mid
            else:
                P_lo = P_mid
        raise RuntimeError(f"[hx] Bisection failed for h={h}, {x_prop}={x_target}")

    def inverse_interpolant_xP(self, x_target, P, x_prop, tol=1e-8, max_iter=1000):
        x_prop = self._resolve_key(x_prop)  # normalize alias
        h_vals = self.table["h"]
        h_lo, h_hi = float(h_vals[0]), float(h_vals[-1])

        def residual(h):
            props = self.bicubic_interpolant_property(h, P)
            return props[x_prop] - x_target

        for it in range(max_iter):
            h_mid = 0.5*(h_lo+h_hi)
            f_mid = residual(h_mid)
            if abs(f_mid) < tol:
                return self.bicubic_interpolant_property(h_mid, P), it
            f_lo = residual(h_lo)
            if f_lo * f_mid < 0:
                h_hi = h_mid
            else:
                h_lo = h_mid
        raise RuntimeError(f"[xP] Bisection failed for P={P}, {x_prop}={x_target}")

    # ------------------ Dispatcher for different input pairs ------------------
    def get_props(self, input_type, val1, val2):
        if input_type == jxp.PT_INPUTS:
            return self.inverse_interpolant_xP(val2, val1, "T")[0]
        elif input_type == jxp.HmassSmass_INPUTS:
            return self.inverse_interpolant_hx(val1, val2, "s")[0]
        elif input_type == jxp.PSmass_INPUTS:
            return self.inverse_interpolant_xP(val2, val1, "s")[0]
        elif input_type == jxp.DmassHmass_INPUTS:
            return self.inverse_interpolant_hx(val2, val1, "d")[0]
        elif input_type == jxp.DmassP_INPUTS:
            return self.inverse_interpolant_xP(val1, val2, "d")[0]
        elif input_type == jxp.HmassP_INPUTS:
            return self.bicubic_interpolant_property(val1, val2)
        else:
            raise NotImplementedError(f"Input type {input_type} not supported.")
