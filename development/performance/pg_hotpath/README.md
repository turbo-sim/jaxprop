# Perfect-gas performance record

The user accepted the measured performance tradeoff on **13 November 2026**.
Phase 7 is accepted and Phase 8 may proceed. Compiled full-state batches remain
31–38% slower than PyPI 0.5.8; this was accepted explicitly, not eliminated.
The broader release/device/application comparison remains Phase 13.

`get_state()` returns a complete, materialized 22-property `FluidState`, with
immediate failed-row masks. The public interface has no new property-specific
entry points or deferred getter work.

The retained [results](results.json) use normal CPU affinity, four independent
workers per version, ABBAABBA pair ordering and twelve synchronized samples per
worker. Timings below are medians in microseconds per call, excluding compilation.

| Workload | PyPI 0.5.8 | Before repair | Accepted implementation |
| --- | ---: | ---: | ---: |
| Eager PT full state, scalar | 123.06 | 46.83 | 46.17 |
| Eager HP full state, scalar | 122.56 | 46.87 | 45.82 |
| Eager PT full state, 1,024 states | 190.45 | 123.19 | 103.54 |
| Eager HP full state, 1,024 states | 172.83 | 109.68 | 108.84 |
| Compiled HP full state, 1,024 states | 58.19 | 69.10 | 76.41 |
| Compiled HP full state, array pressure | 56.34 | 75.78 | 77.98 |
| Compiled complete property export | 68.01 | 106.64 | 70.17 |
| Compiled common 13 properties plus status | 53.91 | 61.20 | 63.25 |
| Compiled heat exchanger, 128 cells | 11.47 | 16.35 | 16.68 |

Final HEOS/REFPROP verification: **1,983 passed, 15 existing expected failures,
zero unexpected failures/skips**. All **54,272** numerical comparisons pass
(50,189 finite and 4,083 paired unavailable values). The repair adds 97 passing
cases: 37 native-assembly, 54 numerical/boundary/model-gradient and six
compiler-pruning tests. [Verification record](verification.json).

All **147,854** same-physics benchmark comparisons against the pre-repair Phase 7
source match. Every worker also passes independent ideal-gas and application
gradient checks. Fixtures match physical temperature/pressure while respecting
each model's heat capacity and enthalpy convention. PyPI enables 13 finite PG
properties plus 25 unavailable schema placeholders; the accepted model enables
22 finite properties, uniform broadcast shapes and numerical status. The matching
common-13 consumers also remain slower, so payload size explains only part of
the difference. Returned leaves/shapes and variability remain in the results.

Environment: CPU/x64, Python 3.13.2, JAX/jaxlib 0.6.2, NumPy 2.4.4, CoolProp 7.2.0
and Equinox 0.12.2. All versions used identical dependencies in separate serial
processes. The official PyPI 0.5.8 wheel was unmodified and SHA256-verified:
`3b6db9bfa5bd9ee37a4c3441f92f86b772e0c88ce8609bf76d32c322fbc795a3`.
The measured package SHA256 is
`38e0c041191302b5021595727f24dfa6aa325f4fd1c7353a682026cdfcdcc644`.

The implementation resolves each pair once, validates coefficient arithmetic at
setup, uses equivalent pair-specific numerical guards and trusted native state
assembly, and shares the successful batch branch. Stable Sutherland algebra
fixes a high-temperature overflow that could contaminate masked gradients.
Inspected native programs contain no callbacks. Model construction measures
4.15 ms versus 6.98 ms for PyPI and 3.13 ms before the repair; the extra setup
range/precision validation is retained.

Profiling attributed only a few microseconds to Python state packaging. Lazy
masking, integer-status-derived masking and independently recomputed PT/HP
success coordinates were tested and rejected. A single-CPU diagnostic changed
JAX batch timings substantially and was not substituted for normal-affinity
results. These findings inform optional future resolver/output-fusion work in
[Phases 11 and 13](../../../z_IMPLEMENTATION_PLAN.md).

Temporary timing drivers, worker logs, extracted packages/wheels, source copies,
profiles and exploratory runs have been removed. This compact record, case
summaries and final correctness evidence are retained. Stored historical paths
inside the JSON identify the measured run; they are not active rerun commands.
