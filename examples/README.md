# Example workflows

The [catalogue](catalogue.json) classifies all **65 original Python files**:
**59 standalone programs**, **3 input-dependent tools**, and **3 commented
reference files**. Standalone includes tutorials, scientific checks and explicitly
identified research studies. Importing an example does not construct fluids,
compute grids, create figures or write its output files; execute its `main()` or
command-line entry point to run the study.

## Run a bounded smoke profile

Use the repository root and the Python environment containing this checkout:

```sh
python -m pip install -e .
python examples/run_examples.py --list
python examples/run_examples.py --suite smoke --profile quick --output-dir output/examples_smoke
```

Every run is headless, with isolated per-program output directories, double
precision, CPU execution and explicit source/table paths. The smoke selection is
recorded by `smoke: true` in the catalogue. It is a bounded workflow check, not the
full scientific verification suite.

## Choose a program or execution profile

```sh
python examples/run_examples.py --examples 'two_phase_derivatives/*.py' --profile full --output-dir output/derivatives
python examples/run_examples.py --suite runnable --profile quick --jobs 2 --output-dir output/all_quick
python examples/run_examples.py --suite runnable --profile full --jobs 2 --output-dir output/all_full
```

`runnable` selects all 59 standalone programs. `--examples` accepts catalogue
paths, filename stems or glob patterns and overrides the suite selection.
`--jobs` accepts 1 through 4; a serial run is the most conservative resource
choice. Per-program timeout budgets are listed below; `--timeout SECONDS`
explicitly overrides them. A timeout is a failed execution outcome.

Each standalone script also accepts the common options directly:

```sh
python examples/perfect_gas/demo_perfect_gas_eos.py --profile quick --headless --output-dir output/perfect_gas
```

The runner defaults to **quick**, while a directly executed script defaults to
**full**. Commands below select quick explicitly. Quick profiles use their
separately declared sampling and retain numerical checks; their completion is
not evidence that a full-resolution study passed. Exact physical settings and
full resolutions remain visible in each source file. `example_result.json`
records the selected profile, declared checks and sample sizes.

`--output-dir` controls figures, exported data and manifests. `--table-dir`
selects a separate cache directory; the runner otherwise uses its output's
`tables` directory. Cache reuse retains complete source/reference/schema
validation. Fresh table generation, especially full metastable studies, can take
minutes and use substantial memory/disk space. Resource descriptions and timeout
budgets are execution planning aids, not portable speed measurements.

## Supply explicit input data

These three tools need a current-schema JaxProp property-table pickle:

- [CSV/XLSX converter](bicubic_interpolation/convert_pkl_to_csv.py): CSV is the
  default (`nodes.csv`, `cells.csv`, `saturation.csv`, `metadata.json`), with a
  verified numeric round trip. `--format xlsx` is optional and requires `pandas`
  and `openpyxl`; spreadsheet values use a declared relative tolerance.
- [C-binary exporter](bicubic_interpolation/export_pkl_to_c_binary.py): writes a
  versioned `.cbin` file and `_layout.h` header and verifies its Python reader
  round trip. The separate C reader source is an optional format-consumer study. Its regression
  test detects an available C compiler; `JAXPROP_TEST_CC` selects one explicitly.
  Absence of a compiler is reported separately from Python format validation.
- [Table plot](bicubic_interpolation/plot_metastable_point.py): inspects stored
  nodes and excluded cells. `--source-curves` optionally overlays a verified
  matching source-EOS diagram; it does not certify the table's interpolation.

A small fixture can be generated with the normal table-construction example:

```sh
python examples/bicubic_interpolation/demo_table_generation.py --profile quick --headless --output-dir output/fixture --table-dir output/fixture_tables
```

Use the actual generated `.pkl` path reported by that program. Filenames can
contain generation-manifest digests, so a historical guessed filename is not a
substitute for the fixture. Pass that explicit path to a tool:

```sh
python examples/bicubic_interpolation/convert_pkl_to_csv.py --input output/fixture_tables/ACTUAL_TABLE.pkl --format csv --headless --output-dir output/table_csv
python examples/run_examples.py --examples bicubic_interpolation/export_pkl_to_c_binary.py --profile quick --input-for bicubic_interpolation/export_pkl_to_c_binary.py=output/fixture_tables/ACTUAL_TABLE.pkl --output-dir output/table_binary
```

Repeat `--input-for CATALOGUE_PATH=FILE` for each selected input tool. `--suite all`
includes all 65 classifications; missing tool inputs are reported as
`input_required`, not counted as passes, and give the runner a nonzero exit code.
Reference files are recorded as `reference_only` and are not executed or counted
as passing programs. Supplying all three inputs permits 62 actual executions
plus three reference classifications.

## Validation and output contract

The index uses these compact output/check labels:

| Label | Expected outputs and acceptance |
| --- | --- |
| T | Tutorial: execution JSON, numerical/workflow assertions and finite successful core states; PNG figures when plotted, and a declared table cache when built. |
| S | Scientific study: the same outputs plus its independent identities, signed derivative/constraint/conservation checks and declared full sampling. |
| R | Research study: explicit coverage/errors/limitations and figures/data; successful execution does not establish whole-domain physical or interpolation accuracy. |
| I | Input tool: explicit validated fixture, export/inspection outputs and schema/round-trip/coverage checks appropriate to that format. |
| Ref | Commented historical reference: no runnable profile, numerical result or passing-execution claim. |

For runner executions, `summary.json` aggregates per-program
`example_result.json`, `stdout.log`, `stderr.log` and applicable PNG/data files in
each group/program directory. Direct script calls write their result/figures and
print to the caller's streams. Package, framework and example hashes, commands,
profiles, exit statuses and execution limits are retained. Exceptions, absent
inputs, stale/missing manifests and timeouts remain explicit outcomes. Locally
unavailable properties remain gaps with reported coverage; successful core
values must remain finite.

The two broad formal Nitrogen rectangles deliberately include nonpositive
$c_v$ and steep responses around a caloric pole. Physical tables reject such
samples. Formal tables report pole-crossing responses as unavailable, and large
errors can remain in nearby finite cells. Neither successful generation nor a
successful selected-point inverse certifies the whole formal table. See the
[scientific execution audit](../z_EXAMPLES_AUDIT.md) for retained failures,
intentional physical-domain corrections and later results.

Elapsed runner times describe resources, not comparative performance. Dedicated
benchmarks must warm up, synchronize, repeat and separate construction/loading,
compilation and warmed evaluation. The formal published-release comparison is a
separate implementation-plan phase.

## CI and local verification

[The workflow](../.github/workflows/verify_examples.yaml) is configured to run the
HEOS library suite and bounded quick smoke profiles on pushes and pull requests.
Scheduled or manually requested full runs select the 59 standalone full profiles.
This describes the checked-in configuration; hosted CI outcomes are established
only by actual workflow runs. Local scientific checks and example executions are
recorded separately in the [implementation plan](../z_IMPLEMENTATION_PLAN.md).

```sh
python -m pytest tests -c tests/pytest.ini -q
```

The scientific test profile explicitly chooses double precision. Optional
REFPROP cases require a locally usable REFPROP installation; request them with
`JAXPROP_TEST_BACKENDS=HEOS,REFPROP` and retain any reported skips. The ordinary
catalogued example defaults use HEOS where a source is needed. Import policy and
complete worker/compiled-consumer recipes are in the
[lifecycle guide](../docs/fluid_lifecycle.md); the development API changes are in
the [migration guide](../docs/migration.md).

## Programme index

Generated from `catalogue.json`. Each command is the catalogue's explicit quick
command; replace `--profile quick` with `--profile full` for the declared full
study. Timeout budgets are upper execution limits in seconds. Optional tool
requirements are listed below each relevant group.

### basic_calculations

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [solver_coolprop.py](basic_calculations/solver_coolprop.py)<br>`python examples/basic_calculations/solver_coolprop.py --profile quick --headless --output-dir output/solver_coolprop` | Evaluate liquid, gas and equilibrium saturation states.<br>CoolProp HEOS<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [solver_equilibrium.py](basic_calculations/solver_equilibrium.py)<br>`python examples/basic_calculations/solver_equilibrium.py --profile quick --headless --output-dir output/solver_equilibrium` | Recover the same equilibrium state through several input pairs.<br>CoolProp HEOS<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [solver_metastable.py](basic_calculations/solver_metastable.py)<br>`python examples/basic_calculations/solver_metastable.py --profile quick --headless --output-dir output/solver_metastable` | Recover a homogeneous state through direct and inverse pairs.<br>CoolProp HEOS<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |

### bicubic_interpolation

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [convert_pkl_to_csv.py](bicubic_interpolation/convert_pkl_to_csv.py)<br>`python examples/bicubic_interpolation/convert_pkl_to_csv.py --profile quick --headless --output-dir output/convert_pkl_to_csv --input PATH_TO_VALID_TABLE.pkl` | Export a validated table to CSV, or an optional Excel workbook, and verify its round trip.<br>JaxProp bicubic / configured source<br>Source backend: not requested by catalogue | input-tool | 180 / 600 s | small/medium; normally seconds | I |
| [demo_derivatives_jax.py](bicubic_interpolation/demo_derivatives_jax.py)<br>`python examples/bicubic_interpolation/demo_derivatives_jax.py --profile quick --headless --output-dir output/demo_derivatives_jax` | Compare signed bicubic AD derivatives with source responses.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_grid_convergence.py](bicubic_interpolation/demo_grid_convergence.py)<br>`python examples/bicubic_interpolation/demo_grid_convergence.py --profile quick --headless --output-dir output/demo_grid_convergence` | Study finite-difference/interpolation convergence at declared resolutions.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | research-study | 240 / 1800 s | large tables/ODE; seconds to several minutes | R |
| [demo_interpolation_midpoints.py](bicubic_interpolation/demo_interpolation_midpoints.py)<br>`python examples/bicubic_interpolation/demo_interpolation_midpoints.py --profile quick --headless --output-dir output/demo_interpolation_midpoints` | Compare ordinary bicubic midpoints with the source EOS.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_interpolation_midpoints_metastable.py](bicubic_interpolation/demo_interpolation_midpoints_metastable.py)<br>`python examples/bicubic_interpolation/demo_interpolation_midpoints_metastable.py --profile quick --headless --output-dir output/demo_interpolation_midpoints_metastable` | Check physical Nitrogen liquid midpoint interpolation.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | tutorial | 240 / 1800 s | large tables/ODE; seconds to several minutes | T |
| [demo_interpolation_serial.py](bicubic_interpolation/demo_interpolation_serial.py)<br>`python examples/bicubic_interpolation/demo_interpolation_serial.py --profile quick --headless --output-dir output/demo_interpolation_serial` | Demonstrate synchronized scalar full-state interpolation.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_interpolation_vectorized.py](bicubic_interpolation/demo_interpolation_vectorized.py)<br>`python examples/bicubic_interpolation/demo_interpolation_vectorized.py --profile quick --headless --output-dir output/demo_interpolation_vectorized` | Demonstrate synchronized batch full-state interpolation.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_metastable_calculations.py](bicubic_interpolation/demo_metastable_calculations.py)<br>`python examples/bicubic_interpolation/demo_metastable_calculations.py --profile quick --headless --output-dir output/demo_metastable_calculations` | Build a physically admissible Water liquid table and recover a state.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | tutorial | 240 / 1800 s | large tables/ODE; seconds to several minutes | T |
| [demo_metastable_single_point_comparison.py](bicubic_interpolation/demo_metastable_single_point_comparison.py)<br>`python examples/bicubic_interpolation/demo_metastable_single_point_comparison.py --profile quick --headless --output-dir output/demo_metastable_single_point_comparison` | Check a selected point in an explicitly formal research table.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | research-study | 240 / 1800 s | large tables/ODE; seconds to several minutes | R |
| [demo_midpoints_metastable_comparison.py](bicubic_interpolation/demo_midpoints_metastable_comparison.py)<br>`python examples/bicubic_interpolation/demo_midpoints_metastable_comparison.py --profile quick --headless --output-dir output/demo_midpoints_metastable_comparison` | Diagnose formal-EOS interpolation near a caloric pole.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | research-study | 240 / 1800 s | large tables/ODE; seconds to several minutes | R |
| [demo_saturation_midpoints.py](bicubic_interpolation/demo_saturation_midpoints.py)<br>`python examples/bicubic_interpolation/demo_saturation_midpoints.py --profile quick --headless --output-dir output/demo_saturation_midpoints` | Verify the stored saturated endpoint interpolation.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_solver_consistency.py](bicubic_interpolation/demo_solver_consistency.py)<br>`python examples/bicubic_interpolation/demo_solver_consistency.py --profile quick --headless --output-dir output/demo_solver_consistency` | Verify supported bicubic input-pair inverses.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_table_generation.py](bicubic_interpolation/demo_table_generation.py)<br>`python examples/bicubic_interpolation/demo_table_generation.py --profile quick --headless --output-dir output/demo_table_generation` | Generate and inspect a source-compatible ordinary bicubic table.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_table_interpolation.py](bicubic_interpolation/demo_table_interpolation.py)<br>`python examples/bicubic_interpolation/demo_table_interpolation.py --profile quick --headless --output-dir output/demo_table_interpolation` | Evaluate a simple ordinary bicubic table.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_table_interpolation_with_saturation.py](bicubic_interpolation/demo_table_interpolation_with_saturation.py)<br>`python examples/bicubic_interpolation/demo_table_interpolation_with_saturation.py --profile quick --headless --output-dir output/demo_table_interpolation_with_saturation` | Compare a physical homogeneous table and stored saturation endpoints.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_two_phase_speed_sound.py](bicubic_interpolation/demo_two_phase_speed_sound.py)<br>`python examples/bicubic_interpolation/demo_two_phase_speed_sound.py --profile quick --headless --output-dir output/demo_two_phase_speed_sound` | Map supported two-phase acoustic responses and unavailable cells.<br>JaxProp bicubic / configured source<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [export_pkl_to_c_binary.py](bicubic_interpolation/export_pkl_to_c_binary.py)<br>`python examples/bicubic_interpolation/export_pkl_to_c_binary.py --profile quick --headless --output-dir output/export_pkl_to_c_binary --input PATH_TO_VALID_TABLE.pkl` | Export an explicit validated table and round-trip its binary schema.<br>JaxProp bicubic / configured source<br>Source backend: not requested by catalogue | input-tool | 180 / 600 s | small/medium; normally seconds | I |
| [plot_metastable_point.py](bicubic_interpolation/plot_metastable_point.py)<br>`python examples/bicubic_interpolation/plot_metastable_point.py --profile quick --headless --output-dir output/plot_metastable_point --input PATH_TO_VALID_TABLE.pkl` | Inspect an explicitly supplied table and state.<br>JaxProp bicubic / configured source<br>Source backend: not requested by catalogue | input-tool | 180 / 600 s | small/medium; normally seconds | I |

Optional requirements used by selected routes: `openpyxl`, `pandas`. CSV itself does not require spreadsheet packages.

### blending_calculations

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [blending_condensing.py](blending_calculations/blending_condensing.py)<br>`python examples/blending_calculations/blending_condensing.py --profile quick --headless --output-dir output/blending_condensing` | Follow a condensing branch with explicit constitutive blending.<br>CoolProp homogeneous/equilibrium/blending<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [blending_flashing.py](blending_calculations/blending_flashing.py)<br>`python examples/blending_calculations/blending_flashing.py --profile quick --headless --output-dir output/blending_flashing` | Follow a flashing branch with explicit constitutive blending.<br>CoolProp homogeneous/equilibrium/blending<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |

### contour_plots

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [compressibility_factor.py](contour_plots/compressibility_factor.py)<br>`python examples/contour_plots/compressibility_factor.py --profile quick --headless --output-dir output/compressibility_factor` | Map compressibility with declared source/domain coverage.<br>CoolProp / callback<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_joule_thomson.py](contour_plots/demo_joule_thomson.py)<br>`python examples/contour_plots/demo_joule_thomson.py --profile quick --headless --output-dir output/demo_joule_thomson` | Plot available Joule-Thomson responses and explicit property gaps.<br>CoolProp / callback<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [isentropic_cooling.py](contour_plots/isentropic_cooling.py)<br>`python examples/contour_plots/isentropic_cooling.py --profile quick --headless --output-dir output/isentropic_cooling` | Plot signed constant-entropy temperature sensitivity with explicit EOS gaps.<br>CoolProp / callback<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [pressure_isolines.py](contour_plots/pressure_isolines.py)<br>`python examples/contour_plots/pressure_isolines.py --profile quick --headless --output-dir output/pressure_isolines` | Plot source pressure isolines with correct SI/display coordinates.<br>CoolProp / callback<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [pressure_isolines_timing.py](contour_plots/pressure_isolines_timing.py)<br>`python examples/contour_plots/pressure_isolines_timing.py --profile quick --headless --output-dir output/pressure_isolines_timing` | Separate fresh and cached phase-diagram drawing costs.<br>CoolProp / callback<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |

### coolprop_jax

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [coolprop_broadcast.py](coolprop_jax/coolprop_broadcast.py)<br>`python examples/coolprop_jax/coolprop_broadcast.py --profile quick --headless --output-dir output/coolprop_broadcast` | Compare synchronized complete-state scalar/array broadcasting costs.<br>CoolProp callbacks / native controls<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [derivative_consistency_1.py](coolprop_jax/derivative_consistency_1.py)<br>`python examples/coolprop_jax/derivative_consistency_1.py --profile quick --headless --output-dir output/derivative_consistency_1` | Verify source callback thermodynamic and Maxwell identities.<br>CoolProp callbacks / native controls<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [derivative_consistency_2.py](coolprop_jax/derivative_consistency_2.py)<br>`python examples/coolprop_jax/derivative_consistency_2.py --profile quick --headless --output-dir output/derivative_consistency_2` | Check signed source callback first derivatives.<br>CoolProp callbacks / native controls<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [derivative_consistency_3.py](coolprop_jax/derivative_consistency_3.py)<br>`python examples/coolprop_jax/derivative_consistency_3.py --profile quick --headless --output-dir output/derivative_consistency_3` | Check callback derivatives across supported input pairs.<br>CoolProp callbacks / native controls<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |

### mixture_properties

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [test_cal_prop.py](mixture_properties/test_cal_prop.py)<br>`python examples/mixture_properties/test_cal_prop.py --profile quick --headless --output-dir output/test_cal_prop` | Verify native binary HP Jacobians and PS acoustic definitions.<br>native binary with tabulated components<br>Source backend: HEOS | scientific-study | 240 / 1800 s | large tables/ODE; seconds to several minutes | S |

### optimistix

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [demo_ideal_gas.py](optimistix/demo_ideal_gas.py)<br>`python examples/optimistix/demo_ideal_gas.py --profile quick --headless --output-dir output/demo_ideal_gas` | Solve dimensionless nozzle energy residuals and verify the analytic solution.<br>native perfect gas<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_minimize.py](optimistix/demo_minimize.py)<br>`python examples/optimistix/demo_minimize.py --profile quick --headless --output-dir output/demo_minimize` | Verify a smooth optimization objective and its result.<br>analytic optimization objective<br>Source backend: not requested by catalogue | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_minimize_io.py](optimistix/demo_minimize_io.py)<br>`python examples/optimistix/demo_minimize_io.py --profile quick --headless --output-dir output/demo_minimize_io` | Verify ordered objective/gradient instrumentation during optimization.<br>analytic optimization objective<br>Source backend: not requested by catalogue | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_root_find.py](optimistix/demo_root_find.py)<br>`python examples/optimistix/demo_root_find.py --profile quick --headless --output-dir output/demo_root_find` | Check a root solver against an independent known root.<br>analytic optimization objective<br>Source backend: not requested by catalogue | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [diffrax_polytropic_expansion.py](optimistix/diffrax_polytropic_expansion.py)<br>`python examples/optimistix/diffrax_polytropic_expansion.py --profile quick --headless --output-dir output/diffrax_polytropic_expansion` | Integrate a differentiable expansion and check the exit objective gradients.<br>native perfect gas / configured callback source<br>Source backend: HEOS | tutorial | 240 / 1800 s | large tables/ODE; seconds to several minutes | T |
| [diffrax_rosenbrock.py](optimistix/diffrax_rosenbrock.py)<br>`python examples/optimistix/diffrax_rosenbrock.py --profile quick --headless --output-dir output/diffrax_rosenbrock` | Demonstrate constrained optimization of the Rosenbrock objective.<br>analytic optimization objective<br>Source backend: not requested by catalogue | tutorial | 180 / 600 s | small/medium; normally seconds | T |

### perfect_gas

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [demo_perfect_gas_eos.py](perfect_gas/demo_perfect_gas_eos.py)<br>`python examples/perfect_gas/demo_perfect_gas_eos.py --profile quick --headless --output-dir output/demo_perfect_gas_eos` | Verify full-state perfect-gas input-pair round trips.<br>native perfect gas<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_perfect_gas_gradients.py](perfect_gas/demo_perfect_gas_gradients.py)<br>`python examples/perfect_gas/demo_perfect_gas_gradients.py --profile quick --headless --output-dir output/demo_perfect_gas_gradients` | Compare perfect-gas derivatives with finite differences and identities.<br>native perfect gas<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [demo_sutherland.py](perfect_gas/demo_sutherland.py)<br>`python examples/perfect_gas/demo_sutherland.py --profile quick --headless --output-dir output/demo_sutherland` | Plot and verify the perfect-gas viscosity law.<br>native perfect gas<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [polytropic_compression.py](perfect_gas/polytropic_compression.py)<br>`python examples/perfect_gas/polytropic_compression.py --profile quick --headless --output-dir output/polytropic_compression` | Check polytropic density gradients and Hessians.<br>native perfect gas<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |

### props_supersaturation

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [supersaturation_along_isentrope.py](props_supersaturation/supersaturation_along_isentrope.py)<br>`python examples/props_supersaturation/supersaturation_along_isentrope.py --profile quick --headless --output-dir output/supersaturation_along_isentrope` | Follow isentropic equilibrium, homogeneous and blended branches.<br>CoolProp homogeneous/equilibrium/blending<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |
| [supersaturation_along_isobar.py](props_supersaturation/supersaturation_along_isobar.py)<br>`python examples/props_supersaturation/supersaturation_along_isobar.py --profile quick --headless --output-dir output/supersaturation_along_isobar` | Compare equilibrium and homogeneous branches at fixed pressure.<br>CoolProp homogeneous/equilibrium/blending<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |
| [supersaturation_along_spinodal.py](props_supersaturation/supersaturation_along_spinodal.py)<br>`python examples/props_supersaturation/supersaturation_along_spinodal.py --profile quick --headless --output-dir output/supersaturation_along_spinodal` | Check supersaturation diagnostics along both spinodal branches.<br>CoolProp homogeneous/equilibrium/blending<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |

### props_undersaturation

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [demo_calculations.py](props_undersaturation/demo_calculations.py)<br>`python examples/props_undersaturation/demo_calculations.py --profile quick --headless --output-dir output/demo_calculations` | Compare configured superheating/subcooling with independent raw source calls.<br>CoolProp requested diagnostics<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [quality_along_isolines.py](props_undersaturation/quality_along_isolines.py)<br>`python examples/props_undersaturation/quality_along_isolines.py --profile quick --headless --output-dir output/quality_along_isolines` | Compare generalized quality along pressure and temperature isolines.<br>CoolProp requested diagnostics<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [subcooling_along_isobar.py](props_undersaturation/subcooling_along_isobar.py)<br>`python examples/props_undersaturation/subcooling_along_isobar.py --profile quick --headless --output-dir output/subcooling_along_isobar` | Plot generalized subcooling with explicit source diagnostics.<br>CoolProp requested diagnostics<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |
| [superheating_along_isobar.py](props_undersaturation/superheating_along_isobar.py)<br>`python examples/props_undersaturation/superheating_along_isobar.py --profile quick --headless --output-dir output/superheating_along_isobar` | Plot generalized superheating with explicit source diagnostics.<br>CoolProp requested diagnostics<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |

### speed_of_sound

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [relaxation_hierarchy.py](speed_of_sound/relaxation_hierarchy.py)<br>`python examples/speed_of_sound/relaxation_hierarchy.py --profile quick --headless --output-dir output/relaxation_hierarchy` | Compare stated acoustic relaxation limits.<br>explicit relaxation models<br>Source backend: HEOS | tutorial | 180 / 600 s | small/medium; normally seconds | T |

### spinodal_lines

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [spinodal_lines.py](spinodal_lines/spinodal_lines.py)<br>`python examples/spinodal_lines/spinodal_lines.py --profile quick --headless --output-dir output/spinodal_lines` | Check homogeneous response grids and liquid/vapor spinodal roots.<br>CoolProp homogeneous<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |
| [spinodal_lines_PT.py](spinodal_lines/spinodal_lines_PT.py)<br>`python examples/spinodal_lines/spinodal_lines_PT.py --profile quick --headless --output-dir output/spinodal_lines_PT` | Plot independently ordered saturation and spinodal pressure curves.<br>CoolProp homogeneous<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |

### transport_properties

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [fluid_troubleshooting.py](transport_properties/fluid_troubleshooting.py)<br>No execution (commented reference) | Historical commented transport/source notes.<br>inactive reference<br>Source backend: not requested by catalogue | reference | n/a | Reference only | Ref |
| [plot_surface_tension.py](transport_properties/plot_surface_tension.py)<br>No execution (commented reference) | Historical commented transport/source notes.<br>inactive reference<br>Source backend: not requested by catalogue | reference | n/a | Reference only | Ref |
| [trying_refprop.py](transport_properties/trying_refprop.py)<br>No execution (commented reference) | Historical commented transport/source notes.<br>inactive reference<br>Source backend: not requested by catalogue | reference | n/a | Reference only | Ref |

### two_component

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [water_nitrogen_calculations.py](two_component/water_nitrogen_calculations.py)<br>`python examples/two_component/water_nitrogen_calculations.py --profile quick --headless --output-dir output/water_nitrogen_calculations` | Integrate a binary expansion and locate a genuine sonic crossing.<br>fixed-composition binary<br>Source backend: HEOS | scientific-study | 240 / 1800 s | large tables/ODE; seconds to several minutes | S |
| [water_nitrogen_consistency.py](two_component/water_nitrogen_consistency.py)<br>`python examples/two_component/water_nitrogen_consistency.py --profile quick --headless --output-dir output/water_nitrogen_consistency` | Verify binary PT/HP/PS recovery and response identities.<br>fixed-composition binary<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |
| [water_nitrogen_example.py](two_component/water_nitrogen_example.py)<br>`python examples/two_component/water_nitrogen_example.py --profile quick --headless --output-dir output/water_nitrogen_example` | Evaluate a configured fixed-composition binary model.<br>fixed-composition binary<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |
| [water_nitrogen_speed_sound.py](two_component/water_nitrogen_speed_sound.py)<br>`python examples/two_component/water_nitrogen_speed_sound.py --profile quick --headless --output-dir output/water_nitrogen_speed_sound` | Compare binary equilibrium and frozen acoustic limits.<br>fixed-composition binary<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |

### two_phase_derivatives

| Program and command | Purpose / model-provider | Kind | Budget quick/full | Resources | Output/check |
| --- | --- | --- | ---: | --- | --- |
| [check_gruneisen.py](two_phase_derivatives/check_gruneisen.py)<br>`python examples/two_phase_derivatives/check_gruneisen.py --profile quick --headless --output-dir output/check_gruneisen` | Check equilibrium Gruneisen from its energy-density definition.<br>CoolProp equilibrium / independent source Jacobians<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |
| [check_speed_sound.py](two_phase_derivatives/check_speed_sound.py)<br>`python examples/two_phase_derivatives/check_speed_sound.py --profile quick --headless --output-dir output/check_speed_sound` | Check equilibrium sound speed from the isentropic density response.<br>CoolProp equilibrium / independent source Jacobians<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |
| [enthalpy_derivatives.py](two_phase_derivatives/enthalpy_derivatives.py)<br>`python examples/two_phase_derivatives/enthalpy_derivatives.py --profile quick --headless --output-dir output/enthalpy_derivatives` | Verify both independent partial derivatives of enthalpy.<br>CoolProp equilibrium / independent source Jacobians<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |
| [gibbs_saturation.py](two_phase_derivatives/gibbs_saturation.py)<br>`python examples/two_phase_derivatives/gibbs_saturation.py --profile quick --headless --output-dir output/gibbs_saturation` | Recover saturation pressure from equal homogeneous Gibbs energies.<br>CoolProp equilibrium / independent source Jacobians<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |
| [saturation_derivatives.py](two_phase_derivatives/saturation_derivatives.py)<br>`python examples/two_phase_derivatives/saturation_derivatives.py --profile quick --headless --output-dir output/saturation_derivatives` | Compare signed saturation-coordinate response Jacobians.<br>CoolProp equilibrium / independent source Jacobians<br>Source backend: HEOS | scientific-study | 180 / 600 s | small/medium; normally seconds | S |
