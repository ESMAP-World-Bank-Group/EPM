"""
Fix pyproj PROJ database path on Windows conda environments.
Must be imported before any pyproj/geopandas CRS operations.

On Windows, pyproj's bundled CRS context sometimes can't find the database
because the conda Library/share/proj path isn't set. Calling set_data_dir
with the conda Library path resolves it.
"""
from __future__ import annotations
import os
from pathlib import Path


def _fix_proj_data_dir() -> None:
    """Set PROJ data dir to conda Library/share/proj if pyproj can't find it."""
    try:
        import pyproj.datadir as _dd
        # Prefer conda Library path (has the DLL-paired database)
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if not conda_prefix:
            # Infer from pyproj location
            import pyproj as _pp
            conda_prefix = str(Path(_pp.__file__).resolve().parents[4])

        candidates = [
            Path(conda_prefix) / "Library" / "share" / "proj",
            Path(conda_prefix) / "share" / "proj",
            # Fallback: pyproj's own bundled path
            Path(_dd.get_data_dir()),
        ]
        for p in candidates:
            if (p / "proj.db").exists():
                _dd.set_data_dir(str(p))
                return
    except Exception:
        pass


_fix_proj_data_dir()
