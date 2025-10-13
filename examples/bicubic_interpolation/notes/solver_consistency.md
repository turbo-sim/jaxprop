
# Consistency check of property solver

## Overview

This script evaluates the **internal consistency** of the bicubic property interpolation solver by computing all thermodynamic properties at a set of off-node mesh points using different input variable pairs. For each input pair, the recovered state is compared against a reference state computed from the bicubic interpolant using `(h, p)` inputs. The goal is to verify that all property calculations are consistent and mutually invertible across the entire thermodynamic domain of the tabulated fluid.

The consistency check involves the following steps:

1. **Property table loading**
   A bicubic interpolation table is loaded from disk for the specified fluid (CO₂ in this case) and grid resolution.

2. **Reference state generation**
   A structured mesh is generated at **midpoints between tabulation nodes** in both enthalpy and log-pressure directions. For each mesh point, a reference state is computed using `(h, p)` as inputs.

3. **Property evaluation for alternative input pairs**
   For each of several input pairs (e.g. `(p, T)`, `(h, s)`, `(ρ, T)`), the solver is used to invert the table and recover the corresponding `(h, p)` values and thermodynamic properties.

4. **Error calculation**

   * **Absolute error**: root-mean-square (L²) norm of the property difference relative to the reference, normalized by the number of points.
   * **Relative error**: L² norm of the difference normalized by the L² norm of the reference field, again independent of the number of points.

5. **Consistency check**
   The elementwise relative error between interpolated and reference properties is checked against a fixed tolerance (1×10⁻¹²). Any property-input pair combination exceeding this tolerance is flagged as a violation.

6. **Performance measurement**
   Evaluation times are measured for each input pair after a warm-up call to exclude JIT compilation time.

7. **Visualization**
   The thermodynamic region used for the check is shown on the phase diagram of the fluid.

---

## Results summary


### Raw results



```bash
Loaded property table from: fluid_tables\CO2_64x64.pkl

Evaluation times per input type:
  HmassP              :    3.272 ms
  PT                  :    8.978 ms
  HmassSmass          :    9.112 ms
  PSmass              :    9.419 ms
  DmassHmass          :    8.513 ms
  DmassP              :    9.030 ms
  DmassT              :   49.075 ms
  DmassSmass          :   44.080 ms

------------------------------------------------------------------------------------------------------------------------------------------------
 Absolute two-norm error across mesh
------------------------------------------------------------------------------------------------------------------------------------------------
property                       |      HmassP |          PT |  HmassSmass |      PSmass |  DmassHmass |      DmassP |      DmassT |  DmassSmass
------------------------------------------------------------------------------------------------------------------------------------------------
pressure                       |  +0.000e+00 |  +0.000e+00 |  +1.134e-08 |  +0.000e+00 |  +8.663e-09 |  +0.000e+00 |  +9.026e-09 |  +9.248e-09
temperature                    |  +0.000e+00 |  +2.769e-14 |  +2.563e-14 |  +1.323e-13 |  +6.024e-15 |  +7.426e-14 |  +6.082e-14 |  +1.989e-13
density                        |  +0.000e+00 |  +8.229e-15 |  +8.998e-14 |  +1.624e-14 |  +2.306e-14 |  +3.892e-15 |  +4.598e-14 |  +6.877e-14
enthalpy                       |  +0.000e+00 |  +6.421e-11 |  +0.000e+00 |  +1.533e-10 |  +0.000e+00 |  +8.357e-11 |  +7.362e-11 |  +2.139e-10
entropy                        |  +0.000e+00 |  +1.075e-13 |  +1.716e-13 |  +1.077e-13 |  +6.778e-14 |  +1.289e-13 |  +2.040e-13 |  +2.603e-13
internal_energy                |  +0.000e+00 |  +5.856e-11 |  +1.525e-11 |  +1.357e-10 |  +4.456e-12 |  +7.657e-11 |  +9.183e-11 |  +2.002e-10
compressibility_factor         |  +0.000e+00 |  +2.030e-17 |  +4.719e-17 |  +4.399e-17 |  +1.214e-17 |  +2.038e-17 |  +3.700e-17 |  +7.095e-17
isobaric_heat_capacity         |  +0.000e+00 |  +4.720e-14 |  +1.245e-13 |  +8.940e-14 |  +3.082e-14 |  +4.337e-14 |  +8.693e-14 |  +1.483e-13
isochoric_heat_capacity        |  +0.000e+00 |  +3.107e-14 |  +3.730e-14 |  +6.611e-14 |  +9.058e-15 |  +3.772e-14 |  +5.197e-14 |  +9.516e-14
heat_capacity_ratio            |  +0.000e+00 |  +4.882e-17 |  +9.989e-17 |  +9.127e-17 |  +2.526e-17 |  +3.572e-17 |  +7.664e-17 |  +1.463e-16
speed_of_sound                 |  +0.000e+00 |  +1.796e-14 |  +2.434e-14 |  +4.003e-14 |  +6.405e-15 |  +2.056e-14 |  +3.189e-14 |  +6.407e-14
isothermal_compressibility     |  +0.000e+00 |  +3.732e-24 |  +2.665e-22 |  +1.080e-23 |  +8.075e-23 |  +4.542e-24 |  +1.027e-22 |  +1.476e-22
isentropic_compressibility     |  +0.000e+00 |  +1.850e-24 |  +2.211e-22 |  +5.977e-24 |  +6.811e-23 |  +3.685e-24 |  +8.471e-23 |  +1.166e-22
isothermal_bulk_modulus        |  +0.000e+00 |  +2.292e-10 |  +1.256e-08 |  +4.185e-10 |  +2.951e-09 |  +1.362e-10 |  +5.819e-09 |  +8.572e-09
isentropic_bulk_modulus        |  +0.000e+00 |  +3.414e-10 |  +1.648e-08 |  +5.517e-10 |  +3.978e-09 |  +2.902e-10 |  +8.077e-09 |  +1.198e-08
isobaric_expansion_coefficient |  +0.000e+00 |  +3.601e-19 |  +3.803e-19 |  +6.783e-19 |  +7.917e-20 |  +2.379e-19 |  +4.072e-19 |  +1.178e-18
isothermal_joule_thomson       |  +0.000e+00 |  +4.526e-19 |  +5.719e-19 |  +1.192e-18 |  +1.610e-19 |  +4.156e-19 |  +7.784e-19 |  +2.415e-18
joule_thomson                  |  +0.000e+00 |  +2.825e-22 |  +7.986e-22 |  +9.591e-22 |  +2.054e-22 |  +3.542e-22 |  +6.807e-22 |  +2.068e-21
gruneisen                      |  +0.000e+00 |  +8.684e-18 |  +2.500e-17 |  +2.063e-17 |  +6.802e-18 |  +9.230e-18 |  +1.937e-17 |  +3.209e-17
viscosity                      |  +0.000e+00 |  +1.825e-21 |  +2.977e-21 |  +4.364e-21 |  +9.015e-22 |  +2.350e-21 |  +3.562e-21 |  +7.165e-21
conductivity                   |  +0.000e+00 |  +3.744e-18 |  +5.694e-18 |  +8.989e-18 |  +1.596e-18 |  +4.827e-18 |  +7.061e-18 |  +1.489e-17
is_two_phase                   |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00
quality_mass                   |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00
quality_volume                 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00
surface_tension                |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
subcooling                     |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
superheating                   |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
pressure_saturation            |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
temperature_saturation         |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
supersaturation_degree         |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
supersaturation_ratio          |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------
 Relative two-norm error across mesh
------------------------------------------------------------------------------------------------------------------------------------------------
property                       |      HmassP |          PT |  HmassSmass |      PSmass |  DmassHmass |      DmassP |      DmassT |  DmassSmass
------------------------------------------------------------------------------------------------------------------------------------------------
pressure                       |  +0.000e+00 |  +0.000e+00 |  +1.223e-15 |  +0.000e+00 |  +9.343e-16 |  +0.000e+00 |  +9.734e-16 |  +9.973e-16
temperature                    |  +0.000e+00 |  +3.394e-17 |  +3.142e-17 |  +1.621e-16 |  +7.384e-18 |  +9.102e-17 |  +7.455e-17 |  +2.438e-16
density                        |  +0.000e+00 |  +1.043e-16 |  +1.141e-15 |  +2.058e-16 |  +2.924e-16 |  +4.934e-17 |  +5.829e-16 |  +8.719e-16
enthalpy                       |  +0.000e+00 |  +6.170e-17 |  +0.000e+00 |  +1.473e-16 |  +0.000e+00 |  +8.029e-17 |  +7.073e-17 |  +2.056e-16
entropy                        |  +0.000e+00 |  +3.727e-17 |  +5.949e-17 |  +3.733e-17 |  +2.350e-17 |  +4.469e-17 |  +7.071e-17 |  +9.022e-17
internal_energy                |  +0.000e+00 |  +6.611e-17 |  +1.721e-17 |  +1.532e-16 |  +5.030e-18 |  +8.643e-17 |  +1.037e-16 |  +2.259e-16
compressibility_factor         |  +0.000e+00 |  +2.055e-17 |  +4.775e-17 |  +4.451e-17 |  +1.229e-17 |  +2.062e-17 |  +3.744e-17 |  +7.180e-17
isobaric_heat_capacity         |  +0.000e+00 |  +3.912e-17 |  +1.032e-16 |  +7.409e-17 |  +2.554e-17 |  +3.594e-17 |  +7.205e-17 |  +1.229e-16
isochoric_heat_capacity        |  +0.000e+00 |  +3.215e-17 |  +3.860e-17 |  +6.840e-17 |  +9.372e-18 |  +3.903e-17 |  +5.378e-17 |  +9.847e-17
heat_capacity_ratio            |  +0.000e+00 |  +3.863e-17 |  +7.904e-17 |  +7.221e-17 |  +1.998e-17 |  +2.826e-17 |  +6.064e-17 |  +1.158e-16
speed_of_sound                 |  +0.000e+00 |  +4.221e-17 |  +5.720e-17 |  +9.407e-17 |  +1.505e-17 |  +4.832e-17 |  +7.494e-17 |  +1.506e-16
isothermal_compressibility     |  +0.000e+00 |  +1.592e-17 |  +1.137e-15 |  +4.607e-17 |  +3.444e-16 |  +1.938e-17 |  +4.380e-16 |  +6.296e-16
isentropic_compressibility     |  +0.000e+00 |  +9.707e-18 |  +1.160e-15 |  +3.137e-17 |  +3.575e-16 |  +1.934e-17 |  +4.445e-16 |  +6.120e-16
isothermal_bulk_modulus        |  +0.000e+00 |  +2.472e-17 |  +1.355e-15 |  +4.514e-17 |  +3.184e-16 |  +1.469e-17 |  +6.276e-16 |  +9.246e-16
isentropic_bulk_modulus        |  +0.000e+00 |  +2.866e-17 |  +1.383e-15 |  +4.631e-17 |  +3.340e-16 |  +2.436e-17 |  +6.780e-16 |  +1.006e-15
isobaric_expansion_coefficient |  +0.000e+00 |  +1.755e-16 |  +1.854e-16 |  +3.306e-16 |  +3.859e-17 |  +1.160e-16 |  +1.985e-16 |  +5.744e-16
isothermal_joule_thomson       |  +0.000e+00 |  +1.793e-16 |  +2.266e-16 |  +4.724e-16 |  +6.377e-17 |  +1.646e-16 |  +3.084e-16 |  +9.567e-16
joule_thomson                  |  +0.000e+00 |  +1.286e-16 |  +3.637e-16 |  +4.368e-16 |  +9.352e-17 |  +1.613e-16 |  +3.100e-16 |  +9.418e-16
gruneisen                      |  +0.000e+00 |  +3.906e-17 |  +1.125e-16 |  +9.279e-17 |  +3.059e-17 |  +4.151e-17 |  +8.712e-17 |  +1.443e-16
viscosity                      |  +0.000e+00 |  +5.152e-17 |  +8.403e-17 |  +1.232e-16 |  +2.544e-17 |  +6.634e-17 |  +1.005e-16 |  +2.022e-16
conductivity                   |  +0.000e+00 |  +6.334e-17 |  +9.631e-17 |  +1.521e-16 |  +2.700e-17 |  +8.165e-17 |  +1.194e-16 |  +2.518e-16
is_two_phase                   |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00
quality_mass                   |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00
quality_volume                 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00 |  +0.000e+00
surface_tension                |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
subcooling                     |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
superheating                   |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
pressure_saturation            |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
temperature_saturation         |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
supersaturation_degree         |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
supersaturation_ratio          |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan |        +nan
------------------------------------------------------------------------------------------------------------------------------------------------

Consistency check passed: all 753664 relative error values are within tolerance (tol = 1e-12).
```

### Timing

For a 64×64 property table of CO₂, the evaluation times per input pair are typically between 3 and 10 ms, with the exception of density–temperature and density–entropy inputs, which take approximately 40–50 ms. These slower cases involve solving a 2D nonlinear system, whereas the other input types require solving a scalar nonlinear equation or direct evaluation.

### Error analysis

The absolute and relative two-norm errors are on the order of 10⁻¹⁴ to 10⁻⁸, depending on the property. Errors are smallest for direct evaluations `(h, p)` and remain extremely low for other input types. No violations of the 1×10⁻¹² relative error tolerance were detected across approximately 7.5×10⁵ evaluated points.

The results confirm that:

* The bicubic table and its gradient information are internally consistent.
* Inversions for various input pairs are accurate to machine precision across the mesh.
* The error norms remain stable with respect to the number of mesh points due to proper normalization.




## Conclusion

The consistency check demonstrates that the bicubic property solver yields mutually consistent results for all tested input pairs across a large set of thermodynamic states. Both absolute and relative errors are at or below machine precision, and no tolerance violations are detected. The timings give a clear indication of the computational cost of different inversion paths, which can guide performance optimization if needed.



