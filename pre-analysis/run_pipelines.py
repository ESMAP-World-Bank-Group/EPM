"""Run all 6 Black Sea country zone pipelines."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pipelines.zone_pipeline import run_zone_pipeline

ref = os.path.join(os.path.dirname(__file__), "data", "reference_lines.csv")

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
