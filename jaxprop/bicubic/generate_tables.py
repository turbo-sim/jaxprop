import os
import pickle

import numpy as np

from ..coolprop import Fluid
from .. import helpers_coolprop as jxp


def generate_property_table(
    fluid_name,
    backend,
    h_min,
    h_max,
    p_min,
    p_max,
    N_h,
    N_p,
    outdir="fluid_tables",
):

    # Create fluid
    fluid = Fluid(fluid_name, backend)
    h_vals = np.linspace(h_min, h_max, N_h)
    logP_min = np.log(p_min)  # Log of P
    logP_max = np.log(p_max)  # Log of P
    logP_vals = np.linspace(logP_min, logP_max, N_p)

    delta_h = h_vals[1] - h_vals[0]
    delta_logP = logP_vals[1] - logP_vals[0]

    properties = {
        "T": "T",  # Temperature [K]
        "d": "D",  # Density [kg/m³]
        "s": "S",  # Entropy [J/kg/K]
        "mu": "V",  # Viscosity [Pa·s]
        "k": "L",  # Thermal conductivity [W/m/K]
    }

    # Initialize property grids
    table = {"h": np.array(h_vals), "p": np.array(np.exp(logP_vals))}

    for k in properties:
        table[k] = {
            "val": np.zeros((N_h, N_p)),
            "dval_dh": np.zeros((N_h, N_p)),
            "dval_dp": np.zeros((N_h, N_p)),
            "d2val_dhdp": np.zeros((N_h, N_p)),
        }

    # Loop over grid and populate values
    for i, h in enumerate(h_vals):
        for j, logP in enumerate(logP_vals):
            p = np.exp(logP)

            eps_h = max(1e-6 * abs(h), 1e-3 * delta_h)
            eps_p = max(1e-6 * abs(p), 1e-3 * (np.exp(delta_logP) - 1.0) * p)

            # try:
            f_0 = fluid.get_state(jxp.HmassP_INPUTS, h, p)
            f_h = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p)
            f_p = fluid.get_state(jxp.HmassP_INPUTS, h, p + eps_p)
            f_hp = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p + eps_p)

            # except Exception:
            #     continue  # Skip invalid points

            for k in properties:
                val = f_0[k]
                dval_dh = (f_h[k] - f_0[k]) / eps_h
                dval_dP = (f_p[k] - f_0[k]) / eps_p
                d2val_dhdP = (f_hp[k] - f_h[k] - f_p[k] + f_0[k]) / (eps_h * eps_p)

                table[k]["val"][i, j] = val
                table[k]["dval_dh"][i, j] = dval_dh
                table[k]["dval_dp"][i, j] = dval_dP
                table[k]["d2val_dhdp"][i, j] = d2val_dhdP

    # Save as pickle only (most useful for JAX processing)
    os.makedirs(outdir, exist_ok=True)
    pkl_path = os.path.join(outdir, f"{fluid_name}_{N_h}_x_{N_p}.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(table, f)

    print(f" Saved the table to:\n -> Pickle: {pkl_path}")

    return table
