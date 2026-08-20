"""Convenient entry points for EPM post-processing utilities."""

import sys
from pathlib import Path

# epm.py runs with epm/ as the working directory and imports this as a
# top-level `postprocessing` package, so `epm.geodata` -- which maps.py and
# utils.py read the zone geometry from -- would not resolve. Put the repository
# root on the path here, once, rather than in each module that needs it.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from .data_inception_report import run_data_inception_report

__all__ = ["run_data_inception_report"]
