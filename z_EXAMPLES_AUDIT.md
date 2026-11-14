# Examples execution audit

Audited on 13 November 2026 against Phase 5 commit `b647ffb` plus the small repairs below, committed as `32e18c9`. This audit covers all **65 Python files** under `examples/`; all parsed successfully at that checkpoint. The common `Fluid(..., method=...)` constructor was still planned then. Later constructor migrations are verified separately; this report and its hashes preserve the original audit rather than claiming its outcomes were rerun for every subsequent edit.

The subsequent constructor revision migrates 40 files, verifies unchanged expressions for all 74 constructor argument sets and reruns 19 representative programs successfully. [Constructor migration record](development/example_audit/constructor_migration.json). Advanced table modes and legacy failure studies remain explicit until their owning phases are implemented.

Phase 6 separately reruns the three callback derivative programs and the three previously blocked undersaturation programs successfully, with unchanged numerical defaults/tolerances. Callback transformations now select explicit mask mode; ordinary requested diagnostics now work. All 65 programs still parse. The three bicubic midpoint programs and Diffrax receive minimal API compatibility edits with syntax verification only; their larger execution/physics work remains open. [Phase 6 record](development/example_audit/phase6_compatibility.json). The original table below remains the historical audit, not a claim about the later tree.

Phase 7 reruns all four perfect-gas programs and the callback/broadcast program successfully at unchanged scientific defaults/tolerances, including the existing strict derivative and Hessian checks. Native transformed PG calls now select mask mode, and the derivative example describes its distinct derivative quantity. The Optimistix ideal-gas example remains a recorded failure: its solver reaches the correct pressure but the dimensional energy residual stalls near $1.16\times10^{-10}$ J/kg against $10^{-10}$ absolute tolerance. Residual scaling/acceptance belongs to Phase 12; no tolerance, iteration limit or case was relaxed. All 65 programs parse and the Sutherland figure renders. [Phase 7 record](development/example_audit/phase7_compatibility.json).

Phase 8 makes minimal bicubic API compatibility edits in four programs. The derivative demo now selects explicit mask mode inside JAX transformations. Three metastable-table demos use the uniform `get_state_saturation(PQ_INPUTS, p, Q)` call at their configured liquid/vapor endpoint; the saturation-midpoint demo explicitly selects clipping for its existing extension beyond the saturation table. The saturation comparison iterates over common enabled properties and reads raw values so unavailable optional fields remain visible as NaNs. All **65 programs parse**, but **no example numerical executions or figure checks were performed in this phase**. Bounds, resolutions, equations and tolerances are unchanged; the historical execution outcomes below are not promoted to passes. Shared AD helpers in `mixture_properties/test_cal_prop.py` remain Phases 9/14 work because they serve both bicubic and the unconsolidated binary model. Expensive metastable example execution and scientific-domain review remain Phase 12 work. [Phase 8 compatibility record](development/example_audit/phase8_compatibility.json).

Phase 9 migrates **one** small program, `two_component/water_nitrogen_example.py`, to configured component fluids and fixed mass fractions through `Fluid(method="binary_mixture")`. Its actual default run completes at unchanged $p=5\times10^5$ Pa, $T=350$ K and mass ratio $R=50$, printing the finite frozen acoustic speed **66.73909731321142 m/s**. No figures are produced or verified. This is a successful compatibility run, not a benchmark or blanket scientific acceptance of the other studies. The remaining three programs need deliberate fixed-composition workflows, supported input/property selections and physical derivative/ODE checks; they remain Phase 12 work. The historical per-file outcomes below retain their original source/version meaning. Final all-example AST verification is tracked separately with the Phase 9 verification evidence. [Phase 9 compatibility record](development/example_audit/phase9_compatibility.json).

Phase 10 repairs the library paths behind blending, homogeneous calculations, derivative helpers and scientific grids. A **read-only compatibility review** identifies the consequences below; no example program, full-resolution sweep or figure was executed for this record, and no example source was edited. The original execution counts and per-file outcomes remain historical. The final Phase 10 verification records any separate all-example syntax check. [Phase 10 compatibility review](development/example_audit/phase10_compatibility.json).

- `compute_property_grid_rhoT` now returns a common `FluidState` with shape `(len(T_array), len(rho_array))`. Existing density/temperature indexing in `spinodal_lines/spinodal_lines.py` remains applicable; optional unavailable fields need deliberate raw-value plotting rather than substitution with invented finite values.
- Saturation curves have **N+1** states: one critical reference followed by N distinct subcritical samples. The critical row can have unavailable optional properties. A pseudocritical curve has at most `N_points` ordinary states after source-domain filtering, without a prepended critical row. Quality grids have axes `(quality, temperature)`; HEOS pseudo-pure Air/R410A support QT endpoint grids only, with explicit PQ guidance for interior qualities.
- Pure-fluid two-phase interior $c_p$ and the associated heat-capacity ratio are unavailable; the repaired physical $c_v$ follows the equilibrium derivative. Studies must verify the intended thermodynamic constraint and preserve unavailable values. The copied two-phase kernels in the derivative examples must not be treated as current library formulas or repaired by restoring a mass-averaged response solely to reproduce an old result.
- Blending returns canonical property data without algorithmic `x` or smoothing-factor `sigma` fields. `sigma` remains the alias for surface tension. The blended state's model/reference identity differs from its equilibrium and homogeneous EOS branches; collect each model separately. Its interior phase flag is unavailable, and endpoint optional values recover the selected branch exactly.
- The phase solvers validate settings, aliases, finite targets and positive density/temperature guesses. Direct density/temperature calls need no initial guess. Other inverse studies must supply valid guesses and handle independently checked nonconvergence, rather than treating an optimizer termination flag as sufficient evidence of a thermodynamic root.

## Metastable example execution follow-up — 14 November 2026

The requested follow-up actually runs the default metastable/equilibrium/blending callers, spinodal/grid programs and all six metastable-table builders. **17 programs were reviewed: 14 default programs complete, two retain physical-domain failures, and one requires an absent external table.** Completion is distinct from full scientific accuracy: the broad Nitrogen tables include caloric instability, and their Gruneisen interpolation cannot be trusted across the resulting pole. [Consolidated per-file record](development/example_audit/metastable_solver_followup.json), [bicubic details and accuracy measurements](development/example_audit/metastable_bicubic_followup.json).

| Default program or group | Actual result |
| --- | --- |
| `basic_calculations/solver_equilibrium.py`, `solver_metastable.py` | Both complete; three full-core inverse recoveries each, with defined heat-capacity/acoustic checks. |
| `blending_calculations/blending_flashing.py`, `blending_condensing.py` | Both complete their 100-point sweeps; pressure/entropy targets, finite states, model identity and convexity checked. |
| The three `props_supersaturation/` programs | All complete; 800 returned-state constraint checks and 202 independent spinodal checks. All six figures rendered and inspected. |
| `spinodal_lines/spinodal_lines.py`, `spinodal_lines_PT.py` | Both complete. The full 6-by-500 homogeneous grid has finite core values/status success; branch ordering and stability residuals checked. |
| `two_phase_derivatives/gibbs_saturation.py` | Migrated to one configured fluid, retaining an independent CoolProp saturation reference; relative pressure error $1.13\times10^{-11}$. |
| `demo_table_interpolation_with_saturation.py` | Full 60-by-60 default completes after the continuation repair below; the previously rejected low-pressure liquid point is valid. |
| `demo_saturation_midpoints.py` | Full 21-by-21 default completes. The 59 in-range reference points are finite; maximum temperature/enthalpy errors are $2.94\times10^{-5}$ K and 86.79 J/kg. Relative enthalpy error alone is gauge-sensitive near zero. |
| `demo_midpoints_metastable_comparison.py` | Full 100-by-100 program completes, but the domain crosses $c_v=0$. Gruneisen has severe interpolation errors there, explicitly reported and annotated without masking points. |
| `demo_metastable_single_point_comparison.py` | Full 165-by-165 generation and the selected HP/PT check complete; local enthalpy round-trip error $1.46\times10^{-11}$ J/kg. Its broad table still contains nonpositive $c_v$ and is not globally certified by that selected point. |
| `demo_interpolation_midpoints_metastable.py` | Original 40-by-40 Nitrogen rectangle fails at $h=10{,}000$ J/kg, $p=3$ MPa, beyond the connected liquid spinodal $h_{sp}=7{,}250.624$ J/kg. |
| `demo_metastable_calculations.py` | Original 120-by-120 Water rectangle fails at $h=1{,}711{,}764.706$ J/kg, $p=13$ MPa, beyond $h_{sp}=1{,}710{,}150.809$ J/kg. |
| `plot_metastable_point.py` | Required legacy Water table is absent at both repository-relative and script-relative paths. Its input/schema/plotting study remains a separate migration; no substitute fixture is counted as a default pass. |

The example work also exposed and repaired a **table-construction continuation bug**. After each pressure column, the generator discarded its validated homogeneous state and used an ordinary HP flash to seed the next column. Inside coexistence that flash returns bulk two-phase density, which can send the homogeneous solve onto the wrong branch. Initial guesses now select the requested source phase or its saturation endpoint, and subsequent columns reuse their validated liquid/vapor anchor row. Four actual Nitrogen regressions and the full HEOS/REFPROP suite pass: **2,737 passed, two existing serialization expected failures**, all **54,272 numerical comparisons** retained with zero failures. The fix is limited to construction and its cache metadata; runtime evaluators are unchanged. [Current full run](development/test_audit/results/metastable_followup/full_20261114_103619/summary.json).

Small example changes use enabled/raw optional properties, correct homogeneous reference calls and canonical field names, add actual inverse/curve checks, clarify generalized-quality labels and fix clipped negative supersaturation plot limits. The basic/blending adaptations preserve all 3,636 compared original core values bitwise. All 65 example files parse. **Physical defaults, resolutions, solver tolerances and iteration budgets are unchanged.** Initial 180-second audit timeouts are retained and superseded by complete heavy runs with longer execution budgets; concurrent wall times are not benchmarks. Eighteen figures were drawn and representative outputs inspected. No user/source-folder cache or export was overwritten.

The two mechanically invalid rectangles and the Nitrogen caloric-pole domains require deliberate scientific-domain work, recorded in Phases 11/14. Increasing resolution or relaxing root tolerances does not remove a source EOS pole. Historical results below retain their original source/version meaning; this scoped follow-up does not claim that every example or inactive low-level branch was rerun.

## Phase 11 lifecycle and formal-table follow-up

The four direct solver/blending programs complete again on frozen Phase 11
source `cb75ac4…`, with original numerical settings and both 100-point blending
sweeps. All **3,636 core scalar values are bitwise identical** to the previous
verified executions. Three figures were drawn and inspected. The common
source-EOS plotting path additionally has headless lifecycle tests across direct,
callback and bicubic sources, including named streams and pickle restoration
after quality contours/patches. [Direct example record](development/example_audit/phase11_direct_examples.json).

The 100-by-100 and 165-by-165 Nitrogen studies now explicitly select
`metastable_domain='formal'` to retain their original research rectangles. Default
physical tables reject their nonpositive-$c_v$ source samples. Formal tables keep
their core data, while responses that may cross a caloric pole are unavailable
through the existing optional-property masks; plots and statistics show the
finite comparison coverage. Neither physical bounds, resolution nor solver
tolerances were changed.

Both complete their full studies. The 100-by-100 comparison retains 2,368 of
2,401 finite Gruneisen comparisons, with 33 unavailable near pole-crossing cells.
Its remaining maximum relative error is about **750%**, so the rectangle is
**not certified for response-property accuracy**. Temperature, density and
entropy retain accurate core comparisons. The 165-by-165 selected HP/PT check
retains a $1.46\times10^{-11}$ J/kg enthalpy round-trip error; that local success
does not certify its whole formal table.

These two programs executed the exact earlier frozen Phase 11 source `f99fee2…`;
the second completed fresh generation and then a verified matching-cache rerun.
The final package differs only in callback AD and source-serialization bodies
plus docstrings, outside their direct-source/table calculation paths. Retain the
exact source archive and complete AST-equivalence proof rather than relabeling
those executions with the final source hash. [Large-example record](development/example_audit/phase11_metastable_examples.json),
[code-equivalence proof](development/example_audit/phase11_example_code_equivalence.json).
Concurrent execution times are not benchmark measurements. All 65 files still
parse. The two previously proven mechanical-domain failures, absent external
table and wider Phase 12 scientific/example work remain explicitly outstanding.

## Original audit outcome counts

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

- **Phase 12, after the Phase 6/10 library repairs:** the original ordinary `supersaturation=True` blocker was resolved and the three undersaturation programs were actually rerun in Phase 6, as recorded above. The nitrogen PS contour study still needs deliberate domain/failure handling and a fresh scientific execution; its earlier source-flash failure is not silently promoted to a pass.
- **Phase 12, after Phases 8/10:** three metastable programs originally encountered source nonconvergence or incomplete-grid rejection, and seven other bicubic programs exceeded the 75-second limit at their full resolutions. Library repairs do not establish a new execution outcome for those studies. Retain their resource/domain checks and original bounds, resolutions, equations and tolerances until the full cases are deliberately rerun.
- **Phase 12, after the Phase 9 API migration:** the original audit's four two-component failures are now superseded only for the small `water_nitrogen_example.py` default run. The consistency and acoustic studies still use a fourth positional varying composition, obsolete field names or unsupported density-based inverses; migrate their scientific checks without suppressing unavailable properties. The larger calculations/ODE study also needs configured-object reuse and a check of its temperature equation's Joule–Thomson sign. These broader studies have not been rerun or promoted to passing outcomes.
- **Phase 12, after the Phase 10 library repairs:** blending no longer calls `.to_dict()` on an already raw mapping, and density/temperature grid generation no longer calls the removed `states_to_dict_2d`. The two default high-level blending sweeps and the larger supersaturation/spinodal studies still need actual full-resolution execution and numerical/figure acceptance. Their inactive low-level branches read aliases such as `"rho"`/`"T"` from canonical raw mappings and access a raw source handle; migrate those branches deliberately to the supported configured-state path or documented canonical data. The active saturation-derivative script still needs its nested schema, removed collection helper and copied physical model repaired.
- **Phase 12:** Diffrax polytropic expansion needs coordinated parameter flow, saved-state/postprocessing and gradient repair. The enthalpy demo active body runs, but its unused copied two-phase kernel remains explicitly unvalidated.
- **Phase 12:** the CSV-named converter expects a missing hardcoded table and writes XLSX; the legacy metastable plot also requires an external old-schema table. Add explicit inputs/outputs and fixtures. Three transport files contain only comments.
- **Scientific checks:** the Gibbs example completes but prints a 0.001063728% discrepancy without an accuracy assertion. Review its numerical budget. Some derivative examples compare magnitudes instead of signed differences; several print checks without enforcing them.
- **Presentation/performance:** review derivative units, optimizer result labels, the unrelated heat-capacity legend in the pressure-isoline plot, and asynchronous JAX timing examples. They need the explicit checks and synchronized methodology listed in Phase 12.

The [implementation checklist](z_IMPLEMENTATION_PLAN.md) now contains Phase 12 with the full example migration, resource profiles, CI smoke/full runs, numerical checks, converter fixtures and release acceptance. The unfinished example work moved from Phase 14 to Phase 12 on 14 November 2026 so it precedes standalone benchmarking and final ThermOpt verification. Archived execution records retain their original source/version meaning.

## Original per-file outcomes

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
