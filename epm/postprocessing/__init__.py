"""Convenient entry points for EPM post-processing utilities."""

import importlib.util
import sys
from pathlib import Path

# epm.py runs with epm/ as the working directory and imports this as a
# top-level `postprocessing` package, so `epm.geodata` -- which maps.py and
# utils.py read the zone geometry from -- would not resolve. Put the repository
# root on the path here, once, rather than in each module that needs it.
_EPM_DIR = Path(__file__).resolve().parents[1]
if _EPM_DIR.name == "epm":
    _REPO_ROOT = str(_EPM_DIR.parent)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
elif not hasattr(sys.modules.get("epm"), "__path__"):
    # The code was copied into a folder with another name (e.g. a training
    # bundle), so `import epm` would find epm.py rather than the package and
    # import itself in a loop. Register this folder as the `epm` package.
    _spec = importlib.util.spec_from_file_location(
        "epm", _EPM_DIR / "__init__.py", submodule_search_locations=[str(_EPM_DIR)]
    )
    _epm = importlib.util.module_from_spec(_spec)
    sys.modules["epm"] = _epm
    _spec.loader.exec_module(_epm)

from .data_inception_report import run_data_inception_report

__all__ = ["run_data_inception_report"]
