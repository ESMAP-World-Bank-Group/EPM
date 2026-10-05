# -*- coding: utf-8 -*-
"""Internal lines for LC_Baseline_RoBg, where Romania and Bulgaria are internal.

Writes, next to the reference files and without touching them:

    epm/input/data_blacksea/trade/pTransferLimit_robg.csv
    epm/input/data_blacksea/trade/pLossFactorInternal_robg.csv

Method
------
* Start from trade/pTransferLimit.csv and trade/pLossFactorInternal.csv, the
  files LC_Baseline reads, and keep every row byte for byte.
* Romania-Bulgaria becomes an internal link. Capacity, symmetric, flat over all
  years and seasons, is the 99th percentile of the ENTSO-E 2025 physical flow
  in the dominant direction, rounded to 50 MW: the same rule as the external
  borders of the RoBg family (build_ext_limits_robg.py, which writes the
  statistic to output_prices/robg/border_flows_2025.csv). 2025: p99 RO to BG
  1,347 MW, BG to RO 1,394 MW, so 1,400 MW. The year-ahead NTC (600 MW) is the
  firm slice only; the lines workbook gives about 1,500 MW combined over the
  three 400 kV lines.
* Trakia-Bulgaria moves from pExtTransferLimit to pTransferLimit with the
  LC_Baseline values unchanged (Bulgaria to Trakia 334 MW, Trakia to Bulgaria
  200 MW), so the RoBg family differs from LC_Baseline by its perimeter only.
* Loss factor 0.025 on both links, both directions: the value of the other AC
  cross-border links in pLossFactorInternal and the lambda of the EU netback.
* Georgia-Romania (BSSC) and the EWTC upgrade are projects, absent from the
  baseline (0 MW in pExtTransferLimit_baseline.csv), so no row is added here.

Usage (from EPM/):  python pre-analysis/pipelines/build_internal_lines_robg.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
_PRE_ANALYSIS = _HERE.parents[1]
_EPM = _PRE_ANALYSIS.parent

TRADE_DIR = _EPM / "epm" / "input" / "data_blacksea" / "trade"
REF_LIMIT = TRADE_DIR / "pTransferLimit.csv"
REF_LOSS = TRADE_DIR / "pLossFactorInternal.csv"
REF_EXT = TRADE_DIR / "pExtTransferLimit_baseline.csv"
OUT_LIMIT = TRADE_DIR / "pTransferLimit_robg.csv"
OUT_LOSS = TRADE_DIR / "pLossFactorInternal_robg.csv"
FLOWS = _PRE_ANALYSIS / "output_prices" / "robg" / "border_flows_2025.csv"

SEASONS = ("Q1", "Q2", "Q3", "Q4")
LOSS = 0.025
NEW_ZONES = ("Romania", "Bulgaria")


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_internal_lines_robg] FAIL: {msg}")


def trakia_bulgaria() -> tuple[int, int]:
    """(Bulgaria to Trakia, Trakia to Bulgaria) from the LC_Baseline external file."""
    ext = pd.read_csv(REF_EXT)
    rows = ext[(ext["z"] == "Trakia") & (ext["zext"] == "Bulgaria")]
    years = [c for c in ext.columns if c.isdigit()]
    vals = {}
    for d in ("Import", "Export"):
        v = rows[rows.iloc[:, 3] == d][years].to_numpy()
        if v.size == 0 or (v != v.flat[0]).any():
            _fail(f"Trakia-Bulgaria {d} is missing or not flat")
        vals[d] = int(v.flat[0])
    return vals["Import"], vals["Export"]


def main() -> int:
    flows = pd.read_csv(FLOWS)
    robg = flows[(flows["group"] == "internal") & (flows["border"] == "RO-BG")]
    if len(robg) != 1:
        _fail(f"RO-BG not found in {FLOWS.name}")
    ro_bg = int(robg["capacity_mw"].iloc[0])
    bg_tr, tr_bg = trakia_bulgaria()
    links = [("Romania", "Bulgaria", ro_bg), ("Bulgaria", "Romania", ro_bg),
             ("Bulgaria", "Trakia", bg_tr), ("Trakia", "Bulgaria", tr_bg)]

    raw = REF_LIMIT.read_bytes().decode("utf-8")
    lines = raw.splitlines()
    years = lines[0].split(",")[3:]
    if any(z in ln.split(",")[:2] for ln in lines[1:] for z in NEW_ZONES):
        _fail(f"{REF_LIMIT.name} already holds Romania or Bulgaria rows")
    new = [",".join([a, b, q] + [f"{float(mw):.1f}"] * len(years))
           for a, b, mw in links for q in SEASONS]
    OUT_LIMIT.write_bytes(("\n".join(lines + new) + "\n").encode("utf-8"))

    raw = REF_LOSS.read_bytes().decode("utf-8-sig")
    lines = raw.splitlines()
    years = lines[0].split(",")[2:]
    new = [",".join([a, b] + [str(LOSS)] * len(years)) for a, b, _ in links]
    OUT_LOSS.write_bytes(("\n".join(lines + new) + "\n").encode("utf-8-sig"))

    for a, b, mw in links:
        print(f"  {a:>8} -> {b:<8} {mw:>5} MW  loss {LOSS}")
    print(f"  wrote {OUT_LIMIT.name} and {OUT_LOSS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
