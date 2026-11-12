#!/usr/bin/env python3
"""Run the complete tests directory from any working directory.

Additional pytest arguments are forwarded, for example ``--collect-only`` or
``-k fluid_state``. Set ``JAXPROP_TEST_BACKENDS=HEOS,REFPROP`` to opt into
REFPROP cases, which are reported as skipped if REFPROP is unavailable.
"""

from pathlib import Path
import sys

import pytest


TESTS_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = TESTS_DIR.parent


def main(argv=None):
    """Discover all test modules and preserve pytest's process status."""
    arguments = sys.argv[1:] if argv is None else list(argv)
    # Test the checkout even when this script uses another virtual environment.
    sys.path.insert(0, str(REPOSITORY_DIR))
    return int(
        pytest.main(
            [str(TESTS_DIR), "-c", str(TESTS_DIR / "pytest.ini"), *arguments]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())


