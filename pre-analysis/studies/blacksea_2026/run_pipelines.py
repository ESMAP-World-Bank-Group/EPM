"""Run all 6 Black Sea country zone pipelines."""

# ── Moved from pre-analysis/ on 2026-08-14. The anchors below restore the
# ── directories this script used to resolve, so its paths still hold.
import sys as _sys
from pathlib import Path as _Path

_PRE_ANALYSIS = _Path(__file__).resolve().parents[2]   # pre-analysis/
_REPO_ROOT = _PRE_ANALYSIS.parent                      # repository root
_sys.path.insert(0, str(_PRE_ANALYSIS))
import sys, os
sys.path.insert(0, str(_PRE_ANALYSIS))
from pipelines.zone_pipeline import run_zone_pipeline

ref = os.path.join(str(_PRE_ANALYSIS), "data", "reference_lines.csv")

configs = [
    (["TUR"], 3),
    (["ROU"], 2),
    (["ARM"], 1),
    (["AZE"], 2),
    (["BGR"], 1),
    (["GEO"], 1),
]

for countries, n_zones in configs:
    tag = f"{countries[0]} {n_zones}z"
    print(f"\n=== {tag} ===\n")
    try:
        run_zone_pipeline(countries=countries, n_zones=n_zones, reference_lines_path=ref)
    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
