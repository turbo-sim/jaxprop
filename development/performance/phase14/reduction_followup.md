# Native interpolation and consumer follow-up

The retained source is
`d0fa73ec6953546cabc5402c62bd0a3462ef0c3596d8ec798fa36c8fbe0e8676`.
The immediate reference is commit `d875252`'s numerical package
(`6f93cb82…`), alongside the pinned PyPI 0.5.8 wheel.

## Implementation

Bicubic batches of at least 64 states evaluate the 16-term polynomial with
`sum(coefficients * basis, axis=-1)`. XLA can fuse coefficient gathers and
polynomial products. Optimized large-state graphs replace 21 dot operations
with reductions, and the HX consumer replaces eight. Output layouts and copy
counts match the preceding implementation. Core checks, optional availability,
clipping, masks and input/model derivatives participate in evaluation.

Scalar and small batches retain their dot calculations. The rejected all-size
reduction passed focused correctness tests but slowed scalar/inverse workloads
by 10–31% against the committed reference. The retained selector uses static
shapes. Existing 63/64/65-state tests cover its boundary. A preliminary mixed
scalar-dot/batch-reduction version failed exact three-row vectorization parity
and was also rejected.

PG removes redundant Boolean checks. Direct-pressure pairs already have finite,
floor-protected pressure. Generated-pressure HS/DH keep their finiteness and
lower-bound checks. Resolved-invalid coordinates also fail the positive domain
floor, so the success predicate uses that existing classification. Status-code
precedence and representability bounds are preserved.

## Independent timings

Four fresh workers per version run 30 balanced samples on the same allowed
processors (`0xffff`). All numerical comparisons pass across 61 comparable
native workloads and two current-only capabilities. Every comparable workload
is within 10% of the committed reference. Original PyPI allowances pass for
59/61 workloads.

Times are microseconds per complete call, batch or consumer. Values are medians
of four worker medians.

| Workload | PyPI | Committed | Candidate | Candidate / PyPI |
| --- | ---: | ---: | ---: | ---: |
| Bicubic full HP, 1,024 states | 575.703 | 631.033 | 129.331 | 0.225× |
| Bicubic HX, 128 cells and statuses | 50.764 | 64.217 | 25.123 | 0.495× |
| Added bicubic HP shape, 8 by 16 | 84.459 | 92.616 | 76.775 | 0.909× |
| Bicubic compiled compressor | 177.765 | 22.489 | 21.194 | 0.119× |
| Bicubic compiled expander | 178.792 | 23.520 | 21.425 | 0.120× |
| PG dynamic HP, 1,024 states | 112.075 | 104.731 | 91.072 | 0.813× |
| PG compressor and statuses | 9.184 | 13.610 | 11.847 | 1.290× |
| PG expander and statuses | 9.083 | 13.536 | 12.072 | 1.329× |
| PG HX, 128 cells and statuses | 17.858 | 21.283 | 18.780 | 1.052× |

Both bicubic flags pass. PG compressor/expander retain flags. PG HX and dynamic
batches pass in this native mix, while earlier targeted contexts recorded higher
relative costs. Those observations remain in the measurement record. The earlier
119-case matrix describes its own source and full workload mix. Final full-mix
acceptance remains part of Phase 14.

[Portable measurements and comparison results](reduction_validation.json).

## Status-layout diagnosis

Collecting the same three actual scalar statuses into one array reduced the PG
compressor from 10.591 to 8.968 µs, and expander from 10.638 to 8.941 µs in a
separate diagnostic. PyPI measured 7.973 and 7.819 µs. Numerical-only PG scalar
consumers were already close to PyPI in the paired experiment. Classification
and status delivery account for remaining work.

The diagnostic records all status values and component mapping. Its changed
output layout remains an application experiment, while the original consumer
returns determine the original gates. The planned ThermOpt phase can assess
status collection at the compiled application boundary. Performance acceptance
remains open.

## Verification

The retained candidate passes 205 focused checks covering complete properties,
size boundaries, clipping, errors, input/model AD and vectorization. The complete
HEOS/REFPROP run passes **3,029 tests**, with zero failures/skips/expected failures.
All **54,272 comparisons** remain, comprising 47,867 finite and 6,405 paired
unavailable values, with zero failed comparisons. The before/after source hashes
match the retained candidate. All 272 preceding source docstrings are preserved.

[Verification record](reduction_verification.json).

The four original first-call profiles also pass a two-fresh-worker-per-version
guard. Current PG HP-1,024 measures 130.64 ms against a 169.27 ms allowance;
bicubic HP-1,024 is 191.80 ms against 399.17 ms; PG HX is 113.73 ms against
174.92 ms; and bicubic compressor is 518.63 ms against 1,259.62 ms. Each current
median is below the committed-reference median in that cohort. These fresh
unrestricted processes are recorded separately from the warm native workers.

The main benchmark now accepts an explicit frozen-source SHA256 for comparisons
against another checkpoint. The paired diagnostic also accepts worker-only CPU
affinity. The following invocation reproduces the independent native recipe
after restoring the committed reference under the indicated path.

```powershell
python benchmarks/prepare.py
python -m zipfile -e development/performance/phase14/helper_source.zip benchmarks/results/phase14-next/base
python benchmarks/phase14_benchmark.py --suite native --workers 4 --samples 30 --cpu-affinity 0xffff --frozen-source benchmarks/results/phase14-next/base --frozen-sha256 6f93cb822c058f40fe1e1503f7a92dd0ee185de0eb4b84014c24e919251cec56 --output-directory benchmarks/results/my-native-repeat
```
