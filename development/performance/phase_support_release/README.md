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
