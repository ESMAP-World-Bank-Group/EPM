# -*- coding: utf-8 -*-
"""External transfer limits for LC_Baseline_RoBg, where Romania and Bulgaria are internal.

Writes, next to the reference file and without touching it:

    epm/input/data_blacksea/trade/pExtTransferLimit_robg.csv
    epm/input/data_blacksea/trade/zext_robg.csv

and the border statistics to pre-analysis/output_prices/robg/border_flows_2025.csv.

Method
------
* Start from trade/pExtTransferLimit_baseline.csv, the file LC_Baseline reads.
* Drop the rows whose external zone is Romania or Bulgaria (Georgia-Romania,
  Trakia-Bulgaria). Both become internal zones; their links move to
  pTransferLimit in the RoBg family.
* Add the Romanian and Bulgarian borders with the rest of Europe. Capacity,
  symmetric, flat over all years and seasons, is the 99th percentile of the
  ENTSO-E 2025 physical flow (Transparency Platform, A11 cross-border physical
  flows) in the dominant direction, rounded to 50 MW.

  Why physical flows and not the published NTC: the year-ahead NTC is only
  the firm slice auctioned once a year, and Romania-Hungary is allocated
  flow-based in the Core region, so the published NTC understates what the
  border actually carries. The 2025 year-ahead values (RO-HU 350, RO-RS 300,
  BG-RS 150, BG-GR 400) all sit below the observed p99.

* Ukraine and Moldova are left closed (option A, 2026-10-04). Caution: in
  2025 Romania exported about 3.3 TWh to Moldova (p99 1,161 MW), about
  5 percent of Romanian demand, which the RoBg perimeter does not carry.
* Trakia-Greece is kept at its LC_Baseline values.

Flows are cached in Data/cache_entso_e/flows/ (outside git). A missing file is
fetched with entsoe-py; the token is read from config/api_tokens.ini.

Usage (from EPM/):  python pre-analysis/pipelines/build_ext_limits_robg.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
_PRE_ANALYSIS = _HERE.parents[1]
_EPM = _PRE_ANALYSIS.parent
sys.path.insert(0, str(_HERE.parent))

TRADE_DIR = _EPM / "epm" / "input" / "data_blacksea" / "trade"
REF_LIMIT = TRADE_DIR / "pExtTransferLimit_baseline.csv"
REF_ZEXT = TRADE_DIR / "zext.csv"
OUT_LIMIT = TRADE_DIR / "pExtTransferLimit_robg.csv"
OUT_ZEXT = TRADE_DIR / "zext_robg.csv"
OUT_DIR = _PRE_ANALYSIS / "output_prices" / "robg"
FLOW_DIR = _PRE_ANALYSIS.parents[1] / "Data" / "cache_entso_e" / "flows"

YEAR = 2025
QUANTILE = 0.99
STEP_MW = 50
DROPPED = ("Romania", "Bulgaria")
SEASONS = ("Q1", "Q2", "Q3", "Q4")

# (internal zone, external zone, ISO code inside, ISO code outside)
BORDERS = [
    ("Romania", "Hungary", "RO", "HU"),
    ("Romania", "Serbia", "RO", "RS"),
    ("Bulgaria", "Serbia", "BG", "RS"),
    ("Bulgaria", "NorthMacedonia", "BG", "MK"),
    ("Bulgaria", "Greece", "BG", "GR"),
]
# Closed in RoBg, reported for the record only.
CLOSED = [("Romania", "Ukraine", "RO", "UA"), ("Romania", "Moldova", "RO", "MD")]
# Internal in RoBg, reported for step 2.
INTERNAL = [("Romania", "Bulgaria", "RO", "BG")]


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_ext_limits_robg] FAIL: {msg}")


def flow(a: str, b: str) -> pd.Series:
    """Physical flow a -> b over YEAR in MW, cached."""
    path = FLOW_DIR / f"flow_{a}_{b}_{YEAR}.csv"
    if not path.exists():
        from entsoe import EntsoePandasClient
        from entsoe_pipeline import load_api_token
        client = EntsoePandasClient(api_key=load_api_token("entsoe"))
        s = client.query_crossborder_flows(
            a, b, start=pd.Timestamp(f"{YEAR}0101", tz="Europe/Brussels"),
            end=pd.Timestamp(f"{YEAR + 1}0101", tz="Europe/Brussels"))
        FLOW_DIR.mkdir(parents=True, exist_ok=True)
        s.rename("mw").to_csv(path)
    s = pd.read_csv(path, index_col=0)["mw"]
    if s.empty:
        _fail(f"{path.name} is empty")
    return s


def stats(z: str, x: str, a: str, b: str) -> dict:
    out_, in_ = flow(a, b), flow(b, a)
    p_out, p_in = out_.quantile(QUANTILE), in_.quantile(QUANTILE)
    dominant = max(p_out, p_in)
    return {
        "zone": z, "zext": x, "border": f"{a}-{b}",
        "p99_out_mw": round(p_out, 1), "p99_in_mw": round(p_in, 1),
        "export_twh": round(out_.sum() * _step_hours(out_) / 1e6, 2),
        "import_twh": round(in_.sum() * _step_hours(in_) / 1e6, 2),
        "capacity_mw": int(STEP_MW * math.floor(dominant / STEP_MW + 0.5)),
    }


def _step_hours(s: pd.Series) -> float:
    """Hours per record: ENTSO-E serves 15 min or 60 min depending on the border."""
    return 8760.0 / len(s) if len(s) > 9000 else 1.0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, items in (("modelled", BORDERS), ("closed", CLOSED), ("internal", INTERNAL)):
        for it in items:
            r = stats(*it)
            r["group"] = group
            rows.append(r)
    # out = from the first ISO code (inside) to the second; in = the reverse.
    table = pd.DataFrame(rows)[["group", "zone", "zext", "border", "p99_out_mw", "p99_in_mw",
                                "export_twh", "import_twh", "capacity_mw"]]
    table.to_csv(OUT_DIR / "border_flows_2025.csv", index=False)

    # pExtTransferLimit_robg: reference rows minus the Romania and Bulgaria zext, plus the borders.
    raw = REF_LIMIT.read_text(encoding="utf-8").splitlines()
    header = raw[0].split(",")
    years = header[4:]
    kept = [ln for ln in raw[1:] if ln and ln.split(",")[1] not in DROPPED]
    new = []
    for r in table[table["group"] == "modelled"].itertuples():
        for q in SEASONS:
            for d in ("Import", "Export"):
                new.append(",".join([r.zone, r.zext, q, d] + [str(r.capacity_mw)] * len(years)))
    # Same LF convention as the reference file (write_bytes: no newline translation on Windows).
    OUT_LIMIT.write_bytes(("\n".join([raw[0]] + kept + new) + "\n").encode("utf-8"))

    # zext_robg: reference list minus Romania and Bulgaria, plus the new neighbours.
    zext = [z for z in REF_ZEXT.read_text(encoding="utf-8").splitlines()[1:] if z and z not in DROPPED]
    for x in table[table["group"] == "modelled"]["zext"]:
        if x not in zext:
            zext.append(x)
    OUT_ZEXT.write_bytes(("\r\n".join(["zext"] + zext) + "\r\n").encode("utf-8"))

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.fillna("").to_string(index=False))
    print(f"\n  wrote {OUT_LIMIT.name}: {len(kept) + len(new)} rows; {OUT_ZEXT.name}: {zext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
