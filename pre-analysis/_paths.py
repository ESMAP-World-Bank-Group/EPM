"""
Resolve paths to sibling repositories checked out next to this one.

Several pre-analysis scripts export data straight into the explorer front-end,
which lives beside the EPM clone under EPM_Models/. The clone depth is not
fixed — EPM_Models/EPM and EPM_Models/<study>/EPM are both in use — so
hardcoded parents[] indexing silently resolves to a different directory
depending on where the repo was cloned. Walk up and look for the sibling
instead.
"""
from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve()

# The sibling repo we anchor on. Its presence is what identifies EPM_Models/.
_EXPLORER_NAME = "regional-power-explorer"


def models_root() -> Path:
    """Return the EPM_Models/ directory holding the sibling repositories."""
    for p in _HERE.parents:
        if (p / _EXPLORER_NAME).is_dir():
            return p
    raise SystemExit(
        f"Could not locate the EPM_Models root: no '{_EXPLORER_NAME}/' found in "
        f"any parent of {_HERE.parent}. Clone it next to this repository."
    )


def explorer_dir() -> Path:
    """Return the regional-power-explorer checkout."""
    return models_root() / _EXPLORER_NAME
