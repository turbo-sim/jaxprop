# Examples execution audit

Audited on 13 November 2026 against Phase 5 commit `b647ffb` plus the small repairs below, committed as `32e18c9`. This audit covers all **65 Python files** under `examples/`; all parsed successfully at that checkpoint. The common `Fluid(..., method=...)` constructor was still planned then. Later constructor migrations are verified separately; this report and its hashes preserve the original audit rather than claiming its outcomes were rerun for every subsequent edit.

The subsequent constructor revision migrates 40 files, verifies unchanged expressions for all 74 constructor argument sets and reruns 19 representative programs successfully. [Constructor migration record](development/example_audit/constructor_migration.json). Advanced table modes and legacy failure studies remain explicit until their owning phases are implemented.

Phase 6 separately reruns the three callback derivative programs and the three previously blocked undersaturation programs successfully, with unchanged numerical defaults/tolerances. Callback transformations now select explicit mask mode; ordinary requested diagnostics now work. All 65 programs still parse. The three bicubic midpoint programs and Diffrax receive minimal API compatibility edits with syntax verification only; their larger execution/physics work remains open. [Phase 6 record](development/example_audit/phase6_compatibility.json). The original table below remains the historical audit, not a claim about the later tree.

Phase 7 reruns all four perfect-gas programs and the callback/broadcast program successfully at unchanged scientific defaults/tolerances, including the existing strict derivative and Hessian checks. Native transformed PG calls now select mask mode, and the derivative example describes its distinct derivative quantity. The Optimistix ideal-gas example remains a recorded failure: its solver reaches the correct pressure but the dimensional energy residual stalls near $1.16\times10^{-10}$ J/kg against $10^{-10}$ absolute tolerance. Residual scaling/acceptance belongs to Phase 14; no tolerance, iteration limit or case was relaxed. All 65 programs parse and the Sutherland figure renders. [Phase 7 record](development/example_audit/phase7_compatibility.json).

Phase 8 makes minimal bicubic API compatibility edits in four programs. The derivative demo now selects explicit mask mode inside JAX transformations. Three metastable-table demos use the uniform `get_state_saturation(PQ_INPUTS, p, Q)` call at their configured liquid/vapor endpoint; the saturation-midpoint demo explicitly selects clipping for its existing extension beyond the saturation table. The saturation comparison iterates over common enabled properties and reads raw values so unavailable optional fields remain visible as NaNs. All **65 programs parse**, but **no example numerical executions or figure checks were performed in this phase**. Bounds, resolutions, equations and tolerances are unchanged; the historical execution outcomes below are not promoted to passes. Shared AD helpers in `mixture_properties/test_cal_prop.py` remain Phases 9/14 work because they serve both bicubic and the unconsolidated binary model. Expensive metastable example execution and scientific-domain review remain Phase 14 work. [Phase 8 compatibility record](development/example_audit/phase8_compatibility.json).

| Outcome | Files |
| --- | ---: |
| Completed default active programs | 34 |
| Completed binary-export CLI with explicit generated fixture | 1 |
| Failed; larger repairs deferred | 17 |
| Time limit exceeded; completion unverified | 8 |
| Require external table input | 2 |
| Entirely commented files; no runnable example | 3 |

## Execution scope

Actual unchanged-resolution programs in isolated headless subprocesses. One converter uses an explicit generated fixture; other input-dependent and commented-only files are separate. Default timeout 75-90 seconds; final N100 spinodal retry allowed 180 seconds. Groups ran concurrently, so elapsed times are not benchmarks. Existing numerical checks retained; script completion does not certify unchecked formulas or dormant helper functions.

Execution used the repository virtual environment, Matplotlib Agg, closed stdin, and isolated working directories under `development/example_audit/results/`. Updated plotting programs were explicitly drawn; representative generated output was inspected. Text-only and inactive files are distinguished from figure checks. No existing user tables or source-folder exports were modified.

The [portable per-file record](development/example_audit/summary.json) retains audited source hashes, final outcomes, and log paths. Group records retain exact commands and earlier attempts: [bicubic](development/example_audit/results/bicubic/final_summary.json), [source/phase](development/example_audit/results/source_phase/final_summary.json), [JAX/models](development/example_audit/results/jax_models/summary.json), and [additional derivative examples](development/example_audit/results/jax_models/additional_summary.json). Generated tables, figures, and logs remain ignored by Git.

## Small repairs completed

- Resolved the pre-existing merge conflict in `demo_table_interpolation.py` using its existing coherent 30-by-30 branch and in-domain demonstration state.
- Replaced stale `get_props`/`get_states`/`states_to_dict` usage with current calls, explicit Cartesian broadcasting, and `FluidState.stack`. Corrected helper import namespaces and canonical property names.
- Updated availability-aware comparison/plotting: compare all enabled properties, preserve paired optional NaNs, and leave undefined diagnostics as graphical gaps while keeping core properties checked.
- The perfect-gas inverse example now uses eight float64 rounding units instead of an unattainable `1e-16` threshold. Observed failures before the fix were only `2.872e-16` to `3.774e-16`; finite/availability checks were strengthened.
- The bicubic eight-pair consistency example retains its original `1e-12` tolerance and passes 188,416 relative numerical values. The derivative demo retains its existing `1e-2` percent limits.
- Repaired a small library integration regression: saturation curves now assign their critical endpoint the caller fluid name/identifier, while source-reference records remain canonical. Four new alias/identifier cases plus the reference suite pass: **48 passed, 1 existing strict expected failure**. [Verification](development/example_audit/results/reference_identity.xml).

**16 example files** received small repairs. Some corrected files still encounter a later deferred library failure; they are not counted as completed.

## Deferred findings and ownership

- **Phases 6/10:** ordinary `supersaturation=True` is explicitly unsupported in three undersaturation programs; retain that scientific feature until implemented. The nitrogen PS contour grid also reaches a source flash failure requiring deliberate domain/failure handling.
- **Phase 8:** three metastable programs encounter source nonconvergence or incomplete-grid rejection. Seven other bicubic programs exceed the 75-second limit at their original full resolutions. No table nodes, model bounds, equations, or solver tolerances were altered to manufacture completion.
- **Phase 9:** four two-component scripts depend on absent legacy `FluidTwoComponent`/`get_mixture_state` functionality; migrate them to the validated consolidated model.
- **Phase 10:** three blending paths call `.to_dict()` on a dictionary, and a spinodal plotting path calls removed `states_to_dict_2d` inside the library. The active saturation-derivative script needs its legacy nested schema and copied model repaired.
- **Phase 14:** Diffrax polytropic expansion needs coordinated parameter flow, saved-state/postprocessing and gradient repair. The enthalpy demo active body runs, but its unused copied two-phase kernel remains explicitly unvalidated.
- **Phase 14:** the CSV-named converter expects a missing hardcoded table and writes XLSX; the legacy metastable plot also requires an external old-schema table. Add explicit inputs/outputs and fixtures. Three transport files contain only comments.
- **Scientific checks:** the Gibbs example completes but prints a 0.001063728% discrepancy without an accuracy assertion. Review its numerical budget. Some derivative examples compare magnitudes instead of signed differences; several print checks without enforcing them.
- **Presentation/performance:** review derivative units, optimizer result labels, the unrelated heat-capacity legend in the pressure-isoline plot, and asynchronous JAX timing examples. They need the explicit checks and synchronized methodology listed in Phase 14.

The [implementation checklist](z_IMPLEMENTATION_PLAN.md) now contains Phase 14 with the full example migration, resource profiles, CI smoke/full runs, numerical checks, converter fixtures and release acceptance. Existing phase numbers are preserved; finalize examples after the required APIs stabilize and before publishing the refactor.

## Final per-file outcomes

| Example | Outcome | Detail |
| --- | --- | --- |
| [basic_calculations/solver_coolprop.py](examples/basic_calculations/solver_coolprop.py) | Completed | Default active program |
| [basic_calculations/solver_equilibrium.py](examples/basic_calculations/solver_equilibrium.py) | Completed | Default active program |
| [basic_calculations/solver_metastable.py](examples/basic_calculations/solver_metastable.py) | Completed | Default active program |
| [bicubic_interpolation/convert_pkl_to_csv.py](examples/bicubic_interpolation/convert_pkl_to_csv.py) | Input required | Hardcoded script-relative fluid_tables/CO2_200x200.pkl; writesxlsx despitecsvfilename. No user source-folder table/export touched. Needs explicitinput/output CLI andpandas/openpyxl documentation. |
| [bicubic_interpolation/demo_derivatives_jax.py](examples/bicubic_interpolation/demo_derivatives_jax.py) | Completed | Default active program |
| [bicubic_interpolation/demo_grid_convergence.py](examples/bicubic_interpolation/demo_grid_convergence.py) | Timeout | Unchanged full workload; not a pass |
| [bicubic_interpolation/demo_interpolation_midpoints.py](examples/bicubic_interpolation/demo_interpolation_midpoints.py) | Timeout | Unchanged full workload; not a pass |
| [bicubic_interpolation/demo_interpolation_midpoints_metastable.py](examples/bicubic_interpolation/demo_interpolation_midpoints_metastable.py) | Deferred failure | Nitrogen metastable source evaluation does not converge at p=3 MPa, h approximately0 J/kg. Defer solver/domain work to Phase8. |
| [bicubic_interpolation/demo_interpolation_serial.py](examples/bicubic_interpolation/demo_interpolation_serial.py) | Completed | Default active program |
| [bicubic_interpolation/demo_interpolation_vectorized.py](examples/bicubic_interpolation/demo_interpolation_vectorized.py) | Completed | Default active program |
| [bicubic_interpolation/demo_metastable_calculations.py](examples/bicubic_interpolation/demo_metastable_calculations.py) | Deferred failure | Water120x120 metastable continuation stops inside requested grid at index(112,119); generator correctly refuses copied/extrapolated cells. Defer domain/continuation work to Phase8. |
| [bicubic_interpolation/demo_metastable_single_point_comparison.py](examples/bicubic_interpolation/demo_metastable_single_point_comparison.py) | Timeout | Unchanged full workload; not a pass |
| [bicubic_interpolation/demo_midpoints_metastable_comparison.py](examples/bicubic_interpolation/demo_midpoints_metastable_comparison.py) | Timeout | Unchanged full workload; not a pass |
| [bicubic_interpolation/demo_saturation_midpoints.py](examples/bicubic_interpolation/demo_saturation_midpoints.py) | Deferred failure | Nitrogen metastable source evaluation does not converge at p=200 kPa, h=-100 kJ/kg; current strict table generation rejects the incomplete physical grid. Defer solver/domain work to Phase8. |
| [bicubic_interpolation/demo_solver_consistency.py](examples/bicubic_interpolation/demo_solver_consistency.py) | Completed | Default active program |
| [bicubic_interpolation/demo_table_generation.py](examples/bicubic_interpolation/demo_table_generation.py) | Timeout | Unchanged full workload; not a pass |
| [bicubic_interpolation/demo_table_interpolation.py](examples/bicubic_interpolation/demo_table_interpolation.py) | Completed | Default active program |
| [bicubic_interpolation/demo_table_interpolation_with_saturation.py](examples/bicubic_interpolation/demo_table_interpolation_with_saturation.py) | Timeout | Unchanged full workload; not a pass |
| [bicubic_interpolation/demo_two_phase_speed_sound.py](examples/bicubic_interpolation/demo_two_phase_speed_sound.py) | Timeout | Unchanged full workload; not a pass |
| [bicubic_interpolation/export_pkl_to_c_binary.py](examples/bicubic_interpolation/export_pkl_to_c_binary.py) | Fixture completed | Explicit fresh 30-by-30 table; binary header checked |
| [bicubic_interpolation/plot_metastable_point.py](examples/bicubic_interpolation/plot_metastable_point.py) | Input required | Requires legacy relative demo_metastable_table_generation/water_meta_liquid_20x20.pkl; no input artifact exists in isolated fresh cwd. It also uses a legacy table-schema reader needing migration. |
| [blending_calculations/blending_condensing.py](examples/blending_calculations/blending_condensing.py) | Deferred failure | Library blending path calls .to_dict() on a dict; repair the common blending state/schema in Phase 10. |
| [blending_calculations/blending_flashing.py](examples/blending_calculations/blending_flashing.py) | Deferred failure | Library blending path calls .to_dict() on a dict; repair the common blending state/schema in Phase 10. |
| [contour_plots/compressibility_factor.py](examples/contour_plots/compressibility_factor.py) | Completed | Default active program |
| [contour_plots/demo_joule_thomson.py](examples/contour_plots/demo_joule_thomson.py) | Completed | Default active program |
| [contour_plots/isentropic_cooling.py](examples/contour_plots/isentropic_cooling.py) | Deferred failure | The nitrogen pressure/entropy grid reaches a CoolProp flash failure. Keep this source/domain failure explicit; do not change the grid or suppress invalid core results. |
| [contour_plots/pressure_isolines.py](examples/contour_plots/pressure_isolines.py) | Completed | Default active program |
| [contour_plots/pressure_isolines_timing.py](examples/contour_plots/pressure_isolines_timing.py) | Completed | Default active program |
| [coolprop_jax/coolprop_broadcast.py](examples/coolprop_jax/coolprop_broadcast.py) | Completed | Default active program |
| [coolprop_jax/derivative_consistency_1.py](examples/coolprop_jax/derivative_consistency_1.py) | Completed | Default active program |
| [coolprop_jax/derivative_consistency_2.py](examples/coolprop_jax/derivative_consistency_2.py) | Completed | Default active program |
| [coolprop_jax/derivative_consistency_3.py](examples/coolprop_jax/derivative_consistency_3.py) | Completed | Default active program |
| [mixture_properties/test_cal_prop.py](examples/mixture_properties/test_cal_prop.py) | Timeout | Unchanged full workload; not a pass |
| [optimistix/demo_ideal_gas.py](examples/optimistix/demo_ideal_gas.py) | Completed | Default active program |
| [optimistix/demo_minimize.py](examples/optimistix/demo_minimize.py) | Completed | Default active program |
| [optimistix/demo_minimize_io.py](examples/optimistix/demo_minimize_io.py) | Completed | Default active program |
| [optimistix/demo_root_find.py](examples/optimistix/demo_root_find.py) | Completed | Default active program |
| [optimistix/diffrax_polytropic_expansion.py](examples/optimistix/diffrax_polytropic_expansion.py) | Deferred failure | Diffrax ODE uses obsolete get_props, global state in parameter-dependent code, and full-state postprocessing; requires coordinated ODE/gradient repair. |
| [optimistix/diffrax_rosenbrock.py](examples/optimistix/diffrax_rosenbrock.py) | Completed | Default active program |
| [perfect_gas/demo_perfect_gas_eos.py](examples/perfect_gas/demo_perfect_gas_eos.py) | Completed | Default active program |
| [perfect_gas/demo_perfect_gas_gradients.py](examples/perfect_gas/demo_perfect_gas_gradients.py) | Completed | Default active program |
| [perfect_gas/demo_sutherland.py](examples/perfect_gas/demo_sutherland.py) | Completed | Default active program |
| [perfect_gas/polytropic_compression.py](examples/perfect_gas/polytropic_compression.py) | Completed | Default active program |
| [props_supersaturation/supersaturation_along_isentrope.py](examples/props_supersaturation/supersaturation_along_isentrope.py) | Deferred failure | Library blending path calls .to_dict() on a dict; repair the common blending state/schema in Phase 10. |
| [props_supersaturation/supersaturation_along_isobar.py](examples/props_supersaturation/supersaturation_along_isobar.py) | Completed | Default active program |
| [props_supersaturation/supersaturation_along_spinodal.py](examples/props_supersaturation/supersaturation_along_spinodal.py) | Completed | Default active program |
| [props_undersaturation/demo_calculations.py](examples/props_undersaturation/demo_calculations.py) | Deferred failure | Ordinary Fluid.get_state explicitly rejects supersaturation=True pending diagnostic implementation; retain the feature rather than dropping it. |
| [props_undersaturation/quality_along_isolines.py](examples/props_undersaturation/quality_along_isolines.py) | Completed | Default active program |
| [props_undersaturation/subcooling_along_isobar.py](examples/props_undersaturation/subcooling_along_isobar.py) | Deferred failure | Ordinary Fluid.get_state explicitly rejects supersaturation=True pending diagnostic implementation; retain the feature rather than dropping it. |
| [props_undersaturation/superheating_along_isobar.py](examples/props_undersaturation/superheating_along_isobar.py) | Deferred failure | Ordinary Fluid.get_state explicitly rejects supersaturation=True pending diagnostic implementation; retain the feature rather than dropping it. |
| [speed_of_sound/relaxation_hierarchy.py](examples/speed_of_sound/relaxation_hierarchy.py) | Completed | Default active program |
| [spinodal_lines/spinodal_lines.py](examples/spinodal_lines/spinodal_lines.py) | Deferred failure | Library compute_property_grid_rhoT refers to removed states_to_dict_2d (NameError). |
| [spinodal_lines/spinodal_lines_PT.py](examples/spinodal_lines/spinodal_lines_PT.py) | Completed | Default active program |
| [transport_properties/fluid_troubleshooting.py](examples/transport_properties/fluid_troubleshooting.py) | Inactive | Only comments; not an execution pass |
| [transport_properties/plot_surface_tension.py](examples/transport_properties/plot_surface_tension.py) | Inactive | Only comments; not an execution pass |
| [transport_properties/trying_refprop.py](examples/transport_properties/trying_refprop.py) | Inactive | Only comments; not an execution pass |
| [two_component/water_nitrogen_calculations.py](examples/two_component/water_nitrogen_calculations.py) | Deferred failure | Unexported/unimplemented FluidTwoComponent or get_mixture_state; common binary model is Phase 9. |
| [two_component/water_nitrogen_consistency.py](examples/two_component/water_nitrogen_consistency.py) | Deferred failure | Unexported/unimplemented FluidTwoComponent or get_mixture_state; common binary model is Phase 9. |
| [two_component/water_nitrogen_example.py](examples/two_component/water_nitrogen_example.py) | Deferred failure | Unexported/unimplemented FluidTwoComponent or get_mixture_state; common binary model is Phase 9. |
| [two_component/water_nitrogen_speed_sound.py](examples/two_component/water_nitrogen_speed_sound.py) | Deferred failure | Unexported/unimplemented FluidTwoComponent or get_mixture_state; common binary model is Phase 9. |
| [two_phase_derivatives/check_gruneisen.py](examples/two_phase_derivatives/check_gruneisen.py) | Completed | Default active program |
| [two_phase_derivatives/check_speed_sound.py](examples/two_phase_derivatives/check_speed_sound.py) | Completed | Default active program |
| [two_phase_derivatives/enthalpy_derivatives.py](examples/two_phase_derivatives/enthalpy_derivatives.py) | Completed | Default active program |
| [two_phase_derivatives/gibbs_saturation.py](examples/two_phase_derivatives/gibbs_saturation.py) | Completed | Default active program |
| [two_phase_derivatives/saturation_derivatives.py](examples/two_phase_derivatives/saturation_derivatives.py) | Deferred failure | Active copied two-phase model uses obsolete PROPERTY_ALIAS, nested raw-state schemas and the removed states_to_dict helper; broader repair deferred. |
