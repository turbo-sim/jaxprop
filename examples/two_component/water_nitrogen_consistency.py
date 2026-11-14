"""Recover water/nitrogen parcel states through the supported PT/HP/PS pairs.

The binary model has no advertised density-based or h/s inverse. Its fixed
component fractions are not vapor qualities. All enabled fields and optional
availability are checked rather than simply printed.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import example_options


def main(argv=None):
    options = example_options(__file__, argv, description=__doc__)
    import jaxprop as jp
    from examples.two_component._model import mixture, compare_states, check_state

    p0, T0 = 1.25e5, 310.0
    ratios = (100.0, 200.0, 300.0) if options.profile == "full" else (200.0,)
    components = tuple(jp.Fluid(name) for name in ("water", "nitrogen"))
    comparisons = 0
    for ratio in ratios:
        fluid = mixture(components, ratio)
        reference = fluid.get_state(jp.PT_INPUTS, p0, T0)
        check_state(reference)
        for label, pair, first, second in (
            ("H-P", jp.HmassP_INPUTS, reference.h, reference.p),
            ("P-s", jp.PSmass_INPUTS, reference.p, reference.s),
        ):
            actual = fluid.get_state(pair, first, second)
            compare_states(actual, reference)
            comparisons += len(actual.available_properties)
            print(
                f"R={ratio:.0f}, {label}: all {len(actual.available_properties)} fields and availability agree"
            )
    options.record("settings", dict(p=p0, T=T0, mass_ratios=list(ratios)))
    options.record("field_comparisons", comparisons)
    options.finish()


if __name__ == "__main__":
    main()
