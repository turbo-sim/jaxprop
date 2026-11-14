# Bicubic example domains, profiles and data interchange

Every Python program in this directory has an import-safe `main(argv=None)`.
The default `--profile full` retains the declared full resolution; `quick`
uses a separately reported smaller grid or validation mesh. Use `--headless`
to render figures without a GUI, `--output-dir` for results and `--table-dir`
for a reusable table cache. Each successful run writes `example_result.json`
with the actual settings, numerical checks, coverage and figure paths.

## Deliberately revised physical liquid studies

The following changes are physical-domain redesigns. They do not reproduce the
old rejected rectangles and do not relax solver tolerances. The original
failure evidence remains in `z_EXAMPLES_AUDIT.md` and its retained JSON records.

| Study | Original enthalpy bounds | Revised physical bounds | Full resolution |
| --- | --- | --- | --- |
| Nitrogen midpoint comparison | $[-120,10]$ kJ/kg | $[-120,-40]$ kJ/kg | $40\times40$ |
| Water table and PT/HP recovery | $[0.3,1.8]$ MJ/kg | $[0.3,1.4]$ MJ/kg | $120\times120$, 150 saturation points |
| Nitrogen state/saturation demonstration | $[-120,0.001]$ kJ/kg | $[-120,-40]$ kJ/kg | $60\times60$ |
| Nitrogen saturation midpoint comparison | $[-100,-10]$ kJ/kg | $[-100,-40]$ kJ/kg | $21\times21$ |

All four retain their original pressure ranges. At 3 MPa, the original Nitrogen
midpoint rectangle extended beyond the first connected-liquid spinodal,
$h_{sp}=7.250624$ kJ/kg. At lower pressure its broad range also enters a
calorically unstable formal EOS region. An independent homogeneous HEOS probe
at 0.1 MPa, $h=-20$ kJ/kg gives $c_v\approx-1722$ J/(kg K) despite positive
$K_T$. The revised $h_{max}=-40$ kJ/kg leaves positive caloric and mechanical
margins, checked through independent source probes, all source stencils and
the table's interpolation-cell guard.

The original Water study reached $h=1.711764706$ MJ/kg at 13 MPa, beyond
$h_{sp}=1.710150809$ MJ/kg. The new bounded liquid study also leaves room below
the lower-pressure stability limits and retains the original 12 MPa, 578.95 K
demonstration point. Its script checks positive $c_v$/$K_T$ throughout the
sampled rectangle, every interpolation-cell guard and nine independent
homogeneous source points.

The full table-generation solver tolerance remains $10^{-10}$. A stiff-liquid
pressure rounding edge discovered at 0.1001 MPa is treated by the library's
EOS-representability acceptance floor, using the already evaluated bulk modulus;
it is not addressed by relaxing this example's requested solver tolerance.

## Formal EOS studies remain research examples

The 100-by-100 Nitrogen midpoint study and 165-by-165 selected-point study retain
their original broad rectangles under explicit `metastable_domain="formal"`.
Quick mode uses 24-by-24 tables. The midpoint study retains a dense independent
reference sweep so quick interpolation resolution does not turn the source
continuation into a different poorly seeded problem.

Nonpositive heat capacity and possible pole-crossing response cells are
reported. Locally unavailable responses remain gaps in figures/statistics;
finite comparison coverage is recorded. Large errors can remain near the pole
even on the finite side. Successful execution or a selected HP/PT round trip
does **not** establish whole-domain response accuracy. Adaptive near-pole meshes
are a separate research task.

## Numerical checks and exploratory timing

`demo_derivatives_jax.py` retains its 32-by-32 table and 0.01% error limits.
It compares signed derivatives using the maximum **absolute** relative error,
requires finite positive acoustic derivatives and evaluates interior validation
midpoints. The full validation mesh still has $50\times50$ points. A hard
domain projection's derivative at a clipping face is not the one-sided source
EOS response, so those faces are not appropriate thermodynamic identity checks.

The ordinary midpoint and sound-speed studies report phase-cell/source-domain
coverage and retain the excluded gaps. The midpoint study now compares within
its configured table rectangle; an explicit 700 kJ/kg request demonstrates the
out-of-domain status instead of aborting before the actual study.

The serial, vectorized and inverse examples synchronize complete returned states
and separate initial calls from repeated samples. Perfect gas is a distinct
physical model. These exploratory measurements do not replace the isolated
PyPI performance gate, and concurrent execution times are not performance
acceptance evidence.

## Explicit-input tools

The converter, binary exporter and table plot require `--input PATH`. They do
not select an arbitrary cache, prompt interactively, infer a missing historical
file or silently adapt a legacy schema. For example:

```powershell
python examples/bicubic_interpolation/export_pkl_to_c_binary.py --input path/to/table.pkl --output-dir output/cbin --headless
python examples/bicubic_interpolation/convert_pkl_to_csv.py --input path/to/table.pkl --output-dir output/csv --headless
python examples/bicubic_interpolation/plot_metastable_point.py --input path/to/table.pkl --output-dir output/inspection --headless
```

The CSV converter writes actual CSV node/cell/saturation files and a JSON
metadata manifest, then verifies all arrays and cell guards through a round
trip. `--format xlsx` is optional and imports `pandas`/`openpyxl` only when
invoked. The inspection workbook checks its sheet/metadata round trip at a
declared $10^{-12}$ floating-point tolerance; CSV and binary numerical arrays
retain exact finite values and paired NaNs.

The little-endian C binary format is now **version 2** (`JXTBIN2`). It retains
the phase-cell mask, complete metadata and saturation payload in addition to
all node, derivative and coefficient arrays. Missing arrays are rejected,
rather than filled with manufactured NaNs. The matching C reader checks version,
layout, offsets and phase flags. Python tests independently verify the C ABI
layout and numerical/provenance round trips. The native interoperability test
also compiles and runs the reader against both ordinary and metastable fixture
exports and compares the C values and mask directly with Python. It detects
`cc`/`gcc`/`clang`, or an explicitly selected `JAXPROP_TEST_CC`; if no compiler is
available, that optional check is reported as skipped. For example, on Windows:

```powershell
$env:JAXPROP_TEST_CC = "C:/msys64/mingw64/bin/gcc.exe"
python -m pytest tests/test_phase12_bicubic_examples.py -q
```

The compiler-directory PATH change is restricted to the native subprocesses,
so it cannot alter Python's CoolProp/REFPROP DLL resolution.

The binary format is data interchange, not an independent fluid backend. A C
consumer must honor excluded cells, the stated domain and unavailable optional
properties. The offline plot shows every node, with gray unavailable response
values and explicit excluded-cell markers. `--source-curves` additionally
requires locally matching source/reference provenance.
