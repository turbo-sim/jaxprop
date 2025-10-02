import os
import jaxprop as jxp
# ---------------------------
# Configuration
# ---------------------------
fluid_name = "CO2"
hmin = 200e3     # J/kg
hmax = 600e3     # J/kg
Pmin = 2e6       # Pa
Pmax = 20e6      # Pa
N = 120           # Grid size

# ---------------------------
# Create fluid interpolator
# ---------------------------
fluid = jxp.FluidBicubic(fluid_name, hmin, hmax, Pmin, Pmax, N)

# ---------------------------
# Reference point
# ---------------------------
test_h = 265e3    # J/kg
test_P = 7.75e6      # Pa
props_ref = fluid.bicubic_interpolant_property(test_h, test_P)
h_ref = test_h
P_ref = test_P

print("=" * 60)
print(f"Reference state from (h={h_ref:.1f}, P={P_ref:.1f}):")
print(f"  T = {props_ref['T']:.2f}, s = {props_ref['s']:.2f}, d = {props_ref['d']:.2f}")
print("=" * 60)

# ---------------------------
# Define input test cases
# ---------------------------
input_tests = {
    "PT_INPUTS":          ("pressure", P_ref,       "T", props_ref["T"]),
    "HmassSmass_INPUTS":  ("enthalpy", h_ref,       "s", props_ref["s"]),
    "PSmass_INPUTS":      ("pressure", P_ref,       "s", props_ref["s"]),
    "DmassHmass_INPUTS":  ("d", props_ref["d"],     "enthalpy", h_ref),
    "DmassP_INPUTS":      ("d", props_ref["d"],     "pressure", P_ref),
}

# ---------------------------
# Run test loop
# ---------------------------
results = []

for label, (x_name, x_val, y_name, y_val) in input_tests.items():
    try:
        if x_name == "enthalpy":
            props_back, num_iter = fluid.inverse_interpolant_hx(x_val, y_val, y_name)
        elif y_name == "enthalpy":
            props_back, num_iter = fluid.inverse_interpolant_hx(y_val, x_val, x_name)
        elif x_name == "pressure":
            props_back, num_iter = fluid.inverse_interpolant_xP(y_val, x_val, y_name)
        elif y_name == "pressure":
            props_back, num_iter = fluid.inverse_interpolant_xP(x_val, y_val, x_name)
        else:
            raise NotImplementedError(f"Unsupported input pair: {x_name}, {y_name}")

        # Extract reconstructed values
        h_rec = props_back["h"]
        P_rec = props_back["P"]
        dh = h_rec - h_ref
        dp = P_rec - P_ref
        rel_dh = dh / h_ref
        rel_dp = dp / P_ref

        results.append({
            "label": label,
            "success": True,
            "h_rec": h_rec,
            "P_rec": P_rec,
            "dh": dh,
            "dp": dp,
            "rel_dh": rel_dh,
            "rel_dp": rel_dp,
            "iterations": num_iter,
        })

    except Exception as e:
        results.append({
            "label": label,
            "success": False,
            "error": str(e),
        })

# ---------------------------
# Final report with true and reconstructed h, P
# ---------------------------
print("\n" + "=" * 160)
print(" Inverse Interpolant Reconstruction Report (Detailed)")
print("=" * 160)
header = (
    f"{'Label':22s} | {'h_true':>10s} | {'h_rec':>10s} | {'dh (abs)':>10s} | {'dh (rel)':>9s} | "
    f"{'P_true':>10s} | {'P_rec':>10s} | {'dP (abs)':>10s} | {'dP (rel)':>9s} | {'Iter':^6s} | {'Output':>10s}"
)
print(header)
print("-" * len(header))

for res in results:
    label = res["label"]
    if res["success"]:
        # Evaluate at reconstructed (h, P)
        props_eval = fluid.bicubic_interpolant_property(res["h_rec"], res["P_rec"])

        if label.startswith("PT"):
            output_val = props_eval["T"]
        elif label.startswith("HmassSmass") or label.startswith("PSmass"):
            output_val = props_eval["s"]
        elif label.startswith("Dmass"):
            output_val = props_eval["d"]
        else:
            output_val = float("nan")

        print(
            f"{label:22s} | "
            f"{h_ref:10.2f} | {res['h_rec']:10.2f} | {res['dh']:10.2e} | {res['rel_dh']:9.2e} | "
            f"{P_ref:10.2f} | {res['P_rec']:10.2f} | {res['dp']:10.2e} | {res['rel_dp']:9.2e} | "
            f"{res['iterations']:6d} | {output_val:10.4f}"
        )
    else:
        print(
            f"{label:22s} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*9} | "
            f"{'-'*10} | {'-'*10} | {'-'*10} | {'-'*9} | {'-'*6} | {res['error']}"
        )

print("=" * 160)
