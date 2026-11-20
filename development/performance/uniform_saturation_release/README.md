# JaxProp 0.6.2 patch qualification

Prepared on 21 November 2026 from committed uniform-grid saturation fix
`6cc77bec92977f09de98c8ef1e5d6f6a8e390ce6`. The user authorized publication
and the subsequent ThermOpt dependency update. The separate dirty main-checkout
performance/boundary experiments are excluded and preserved.

The patch preserves the uniform $h$–$\log(p)$ mesh and uses existing endpoint
data for phase-aware evaluation. Its scope and remaining limits are described
in [the API guide](../../../docs/phase_support.md). ThermOpt's reviewed local
evidence qualifies 20 rejected points, eight initial cycles and 20 native
snapping tests; it does not establish full optimization coverage or timings.

Build and install the wheel in this worktree's separate `.venv`. Scientific
dependencies are shared read-only from ThermOpt's unchanged environment through
an explicit `.pth` entry. The installed package must win import resolution and
match every wheel file. The generic
[wheel verifier](../phase_support_release/wheel_checks.py) checks both before
and after pytest collection. Full library checks include separately identified
source/subprocess tests. Select HEOS and REFPROP explicitly for local checks.

Generated artifacts, JUnit results and logs belong under `results/` here.
The checked verification record will identify the tested and published
artifacts separately. Publication waits for installed-wheel checks and hosted
Linux/Windows checks. No warm speedup is inferred from validation wall times.
