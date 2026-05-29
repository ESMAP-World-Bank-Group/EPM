import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "c:/Users/wb590892/Documents/EPM_Models/black_sea_2026/EPM/pre-analysis")
sys.path.insert(0, "c:/Users/wb590892/Documents/EPM_Models/black_sea_2026/EPM/pre-analysis/resolution_advisor")

from pipelines.zone_pipeline import run_zone_pipeline
from pathlib import Path

out = Path("c:/Users/wb590892/Documents/EPM_Models/black_sea_2026/EPM/pre-analysis/output_workflow/zoning_study/TUR_3z_admin")
paths = run_zone_pipeline(["TUR"], n_zones=3, output_root=out, boundary_mode="admin")
print("Done:", list(paths.keys()))
