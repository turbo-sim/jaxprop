# JaxProp 0.6.1 release qualification

Prepared on 20 November 2026 in an isolated Git worktree. The release includes
the committed immutable-metadata equality correction and the new opt-in native
bicubic phase diagnostics/endpoints. The original checkout's unfinished
nozzle/shared-core optimization and boundary-derivative changes are excluded.
Original source snapshots, tests and timing results remain unchanged.

The [API guide](../../../docs/phase_support.md) defines the feature and retained
HP phase-boundary guards. Property/component evidence from the earlier mixed
working tree motivates this release; it is not substituted for testing this
exact artifact. Full ThermOpt Phase 6 remains a separate qualification.

## Artifact checks

Build the sdist and wheel with `python -m build --no-isolation`. Install the wheel
with `python -m pip install --no-deps --force-reinstall <wheel>` in a separate
release environment. This local environment shares the unchanged scientific
dependencies through an explicit `.pth` file; it does not install over the
original ThermOpt environment. Check imports outside repository search paths.

`wheel_checks.py` compares every installed package file with the wheel, verifies
its metadata and runs requested pytest arguments after importing the installed
package. Its pytest plugin checks the import location again after collection.
The full suite includes deliberate source/subprocess tests, which remain
identified as source checks rather than installed-wheel coverage.

Generated build/test logs, JUnit receipts, wheels and JSON evidence go under
the ignored `results/` directory. The checked `verification.json` records final
results and artifact hashes when qualification completes. No unmeasured speedup
is claimed for this release.

## Local outcome

The exact installed release wheel passes **3,064 HEOS/REFPROP library tests**,
with two optional C interoperability skips because no compiler is configured.
The seven import-policy tests separately pass after improving failure output.
The full suite intentionally retains source/subprocess checks alongside the
installed package tests, as described above.

ThermOpt's selected component/state/configuration, pilot and Phase 6A suite
passes **236 tests**, with one unchanged expected co-current accounting failure.
The installed-wheel property runner separately passes water, butane, R134a and
CO2 diagnostics, native endpoints and eight component/derivative comparisons.
Full wet-cycle domain and warm optimization gates remain open.

[verification.json](verification.json) records counts, package-file and artifact
hashes. Receipts are retained under `results/`; ThermOpt receipts are under its
`development/jax_development/results/phase_06/` tree. The first hosted attempts
exposed a missing IPython test dependency and Poetry argument forwarding in the
new wheel check. Both workflow fixes retain the original tests and thresholds.
Hosted qualification and PyPI publication are pending their corrected checks.
