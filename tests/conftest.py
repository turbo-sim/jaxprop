"""Availability checks for explicitly requested external test backends."""

from functools import cache

import CoolProp.CoolProp as CP
import pytest


@cache
def _refprop_unavailability_reason():
    """Check once whether REFPROP can initialize a standard component."""
    try:
        CP.AbstractState("REFPROP", "Water")
    except (ValueError, OSError) as error:
        detail = str(error).split("\n", 1)[0] or type(error).__name__
        return f"Requested REFPROP backend is unavailable. {detail}"
    return None


def pytest_collection_modifyitems(config, items):
    """Report unavailable REFPROP cases without skipping HEOS failures."""
    refprop_items = [
        item
        for item in items
        if getattr(item, "callspec", None) is not None
        and item.callspec.params.get("backend") == "REFPROP"
    ]
    if not refprop_items:
        return
    reason = _refprop_unavailability_reason()
    if reason is not None:
        marker = pytest.mark.skip(reason=reason)
        for item in refprop_items:
            item.add_marker(marker)
