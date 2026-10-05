"""Extract power plants in the map window from the GEM Global Integrated Power Tracker.

Units are grouped by GEM location ID into one plant per site, keeping the
dominant type, the summed capacity by status, and the coordinates. The output is
small enough to version beside the brief, so the brief builds without the
workbook.

Usage
    python extract_gem.py <path to the GIPT workbook>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import openpyxl

from mapkit import BBOX

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "gem_plants.json"

# Statuses kept. Operating and construction are drawn on the ground map. The
# pre construction set feeds counts only.
KEEP = {"operating", "construction", "pre-construction", "announced"}
PAD = 1.0  # degrees around the window, so plants on the frame edge survive


def main(workbook: Path) -> None:
    wb = openpyxl.load_workbook(workbook, read_only=True)
    ws = wb["Power facilities"]
    rows = ws.iter_rows(values_only=True)
    head = next(rows)
    col = {name: i for i, name in enumerate(head)}
    x0, y0, x1, y1 = BBOX
    sites = {}
    for r in rows:
        status = (r[col["Status"]] or "").strip().lower()
        if status not in KEEP:
            continue
        lat, lon = r[col["Latitude"]], r[col["Longitude"]]
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        if not (x0 - PAD <= lon <= x1 + PAD and y0 - PAD <= lat <= y1 + PAD):
            continue
        mw = r[col["Capacity (MW)"]]
        if not isinstance(mw, (int, float)) or mw <= 0:
            continue
        key = r[col["GEM location ID"]] or f'{r[col["Plant / Project name"]]}|{lat}|{lon}'
        site = sites.setdefault(key, {
            "name": r[col["Plant / Project name"]],
            "country": r[col["Country/area"]],
            "lat": round(lat, 4), "lon": round(lon, 4),
            "mw": defaultdict(float), "type_mw": defaultdict(float),
            "start": None,
        })
        site["mw"][status] += mw
        site["type_mw"][r[col["Type"]]] += mw
        year = r[col["Start year"]]
        if status == "operating" and isinstance(year, (int, float)):
            site["start"] = int(year) if site["start"] is None else min(site["start"], int(year))
    out = []
    for site in sites.values():
        kind = max(site["type_mw"], key=site["type_mw"].get)
        out.append({
            "name": site["name"], "country": site["country"],
            "lat": site["lat"], "lon": site["lon"], "type": kind,
            "start": site["start"],
            **{k.replace("-", "_"): round(v, 1) for k, v in site["mw"].items()},
        })
    out.sort(key=lambda p: -(p.get("operating", 0) + p.get("construction", 0)))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf8")
    print(f"Wrote {OUT}: {len(out)} sites")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
