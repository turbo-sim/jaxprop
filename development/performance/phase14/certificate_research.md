# Optional bicubic core-certificate research

**UNIMPLEMENTED.** This is a feasibility proposal, not an accepted optimization,
a numerical proof or completion of the performance gate. The current evaluator
continues to validate core properties at runtime. No numerical experiment or
production change was made for this note.

## Purpose and conservative scope

The compiled bicubic heat-exchanger consumer retains eight interpolation dots:
temperature, density, entropy and internal energy for each stream. The published
consumer retains two temperature dots. Required current status checks keep the
other core fields alive even when the consumer does not return them. A sound
construction-time certificate could let the compiler prune unused core fields
while preserving full-state returns and all status meanings.

The first research target should be a static, whole-model guarantee for **every
currently accepted interpolation cell**, initially under the measured CPU/x64
numerical kernel. If any accepted cell cannot be proved, use the existing guarded
path for the whole model. Partial per-cell certificates cannot justify globally
removing core checks; a runtime choice between certified and generic paths is a
separate, more complex study and may retain the expensive computations anyway.
Input/domain, accepted-cell, phase and inverse-convergence checks remain active.

## What must be proved

- Bound the complete bicubic polynomial throughout each cell, including the
  actual protected-coordinate range, cell-edge rounding and tolerated boundary
  slack. Positive grid nodes do not prove a positive interior.
- Consider conservative conversion to Bernstein form: lower Bernstein bounds
  can certify strictly positive temperature and density. Use outward-rounded
  bounds for the conversion and evaluation errors; an inconclusive bound means
  fallback, not a relaxed tolerance.
- Prove finite temperature, density, entropy and internal energy under the
  actual floating-point evaluation. Bound monomial terms and absolute partial
  sums to exclude intermediate overflow despite cancellation in the final
  polynomial. Include basis/coordinate rounding, underflow and strictly positive
  margins that survive rounding. Finite coefficients alone are insufficient.
- Specify coefficient/coordinate dtype, precision mode, kernel/proof version and
  supported backend assumptions. Other modes fall back unless their arithmetic
  is covered by the proof. Preserve optional-property availability handling.

## Dispatch without per-state fingerprints

A dedicated immutable certificate object should hold strong references to the
authoritative runtime arrays and the immutable proof dependencies, outside the
numerical PyTree. The public Python request wrapper may compare array objects
with `is` **before** entering its inner JIT and pass a static certified/generic
selector. Integer object IDs alone permit reuse after garbage collection.

Coefficient identity alone is unsafe. A valid decision must cover the four
interpolated core arrays, their availability masks, accepted-cell geometry/mask,
grid bounds/scales/dimensions, effective coordinate/coefficient dtype and the
actual evaluator/proof version. Changing a core availability mask can produce a
core NaN without changing coefficients. Admitting a formerly rejected cell can
expose a polynomial that was never proved. Both must trigger fallback unless
the original proof explicitly covers the changed dependency.

Closed configured models captured by an outer JIT can use the proof; it is
checked during tracing. Dynamic-model/core-coefficient tracers and changed
functional model views use the generic evaluator. Input JVP/VJP and coordinate
`vmap` can remain certified when model dependencies are unchanged. Optional-only
coefficient updates need not invalidate a core proof, but their existing runtime
availability/finite handling remains. Keep model arrays dynamic in the numerical
PyTree and do not introduce host callbacks, device-to-host comparisons, source
queries or coefficient fingerprints in repeated property evaluation.

## Lifecycle and acceptance requirements

Do not reuse `InspectionPayload` unchanged as proof identity. Its equality/hash
compare generation metadata, which does not identify arbitrary dynamic numerical
payloads, and normalized runtime arrays may differ from original inspection
arrays. Use a certificate-specific cache identity/version, or a unique immutable
instance token; assess the effect on cross-model compilation reuse.

Deepcopy, device transfer, serialization and reconstruction must preserve sound
invalidation. Reprove or disable certification when restoring a pickle; do not
silently trust restored aliases under an old proof/kernel version. Changed
array objects safely fall back. The proposal assumes functional immutable model
updates: current private runtime dictionaries are mutable, and unsupported
in-place/concurrent private mutation cannot be made safe by an outer-JIT check.

Required experiments include interior-negative polynomials with positive nodes,
finite-coefficient overflow/cancellation, near-zero/underflow margins, dtype and
boundary changes, core-mask and accepted-cell updates, dynamic coefficient AD,
copy/pickle/device lifecycles and distinct payloads sharing generation metadata.
Compare complete fields, exact statuses/clipping/NaN patterns and input/model
JVP/VJP against the generic path, including mixed failures and zero failed-row
derivatives. No status or property output may be omitted to obtain a speedup.

Measure eager wrapper identity-check overhead, setup/proof cost, cold tracing and
compilation, complete full-state batches and actual-status compiled consumers.
Retain the original PyPI and frozen-source gates. Smaller IR or a mathematically
plausible certificate is insufficient evidence of a useful optimization.
