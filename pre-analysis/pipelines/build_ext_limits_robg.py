# -*- coding: utf-8 -*-
"""External transfer limits for LC_Baseline_RoBg, where Romania and Bulgaria are internal.

Writes, next to the reference file and without touching it:

    epm/input/data_blacksea/trade/pExtTransferLimit_robg.csv
    epm/input/data_blacksea/trade/zext_robg.csv

and the border statistics to pre-analysis/output_prices/robg/border_flows_2025.csv
and market_limits_md_ua.csv.

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

* Moldova and Ukraine are open since 2026-10-19 (they were closed from
  2026-10-04, option A). In 2025 Romania exported about 3.3 TWh to Moldova,
  about 5 percent of Romanian demand, which a closed border left out of the
  Romanian balance.
  Market rule since 2026-10-20: on these two borders the neighbouring market,
  not the line, limits the trade. An external zone in EPM buys and sells any
  volume at its price, so a line limit (1,150 MW to Moldova) let the model
  export about 9 TWh a year to a country that consumes about 4 to 5 TWh, and
  import 3.5 TWh from Moldova in 2025 against 0.09 TWh measured. The limit per
  direction is therefore the mean flow of a reference year (annual volume /
  hours), so the model cannot carry more than the reference volume:
    - 2025: ENTSO-E 2025 physical flows (same cache as above);
    - 2030 and 2040: TYNDP 2024 National Trends flows (NT2030, NT2040), the
      market simulation that models Moldova and Ukraine with their own demand;
    - linear between 2025, 2030 and 2040, flat after 2040, same for all seasons.
  The TYNDP-limits family copies these rows (build_ext_limits_robg_tyndp.py).
  Caveat: the flow becomes a flat band; the volume is reproduced, the hourly
  shape is not.
* Trakia-Greece is kept at its LC_Baseline values.

Flows are cached in Data/cache_entso_e/flows/ (outside git). A missing file is
fetched with entsoe-py; the token is read from config/api_tokens.ini.

Usage (from EPM/):  python pre-analysis/pipelines/build_ext_limits_robg.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
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
    ("Romania", "Moldova", "RO", "MD"),
    ("Romania", "Ukraine", "RO", "UA"),
]
# Internal in RoBg, reported for step 2.
INTERNAL = [("Romania", "Bulgaria", "RO", "BG")]
# Market rule (see Method): external zone -> (TYNDP node inside, TYNDP node outside).
MARKET = {"Moldova": ("RO00", "MD00"), "Ukraine": ("RO00", "UA00")}
MARKET_YEARS = (2030, 2040)


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


def market_limits(years: list[int]) -> tuple[dict, pd.DataFrame]:
    """Market rule: {(zone, zext, direction): MW per model year} and the trace table."""
    from build_ext_limits_robg_tyndp import tyndp_exchanges

    items = [it for it in BORDERS if it[1] in MARKET]
    ref = {}
    for z, x, a, b in items:
        ref[(z, x, "Export", YEAR)] = float(flow(a, b).mean())
        ref[(z, x, "Import", YEAR)] = float(flow(b, a).mean())
    for y in MARKET_YEARS:
        for (z, x), s in tyndp_exchanges(y, [(z, x) + MARKET[x] for z, x, _, _ in items]).items():
            ref[(z, x, "Export", y)] = float(s.clip(lower=0).mean())
            ref[(z, x, "Import", y)] = float((-s).clip(lower=0).mean())

    anchors = (YEAR,) + MARKET_YEARS
    path, trace = {}, []
    for z, x, _, _ in items:
        for d in ("Export", "Import"):
            pts = [ref[(z, x, d, y)] for y in anchors]
            path[(z, x, d)] = np.round(np.interp(years, anchors, pts)).astype(int)
            row = {"zone": z, "zext": x, "direction": d}
            row.update({f"mw_{y}": round(v, 1) for y, v in zip(anchors, pts)})
            row.update({f"twh_{y}": round(v * 8.76 / 1e3, 2) for y, v in zip(anchors, pts)})
            trace.append(row)
    return path, pd.DataFrame(trace)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, items in (("modelled", BORDERS), ("internal", INTERNAL)):
        for it in items:
            r = stats(*it)
            r["group"] = group
            rows.append(r)
    # out = from the first ISO code (inside) to the second; in = the reverse.
    table = pd.DataFrame(rows)[["group", "zone", "zext", "border", "p99_out_mw", "p99_in_mw",
                                "export_twh", "import_twh", "capacity_mw"]]

    # pExtTransferLimit_robg: reference rows minus the Romania and Bulgaria zext, plus the borders.
    raw = REF_LIMIT.read_text(encoding="utf-8").splitlines()
    header = raw[0].split(",")
    years = header[4:]
    kept = [ln for ln in raw[1:] if ln and ln.split(",")[1] not in DROPPED]
    market, mtable = market_limits([int(y) for y in years])
    new = []
    for r in table[table["group"] == "modelled"].itertuples():
        for q in SEASONS:
            for d in ("Import", "Export"):
                if r.zext in MARKET:
                    vals = market[(r.zone, r.zext, d)]
                    if vals.min() < 0 or vals.max() > r.capacity_mw:
                        _fail(f"{r.zext} {d}: market limit {vals.min()}..{vals.max()} MW outside "
                              f"0..{r.capacity_mw} (it may only tighten the line rule)")
                else:
                    vals = [r.capacity_mw] * len(years)
                new.append(",".join([r.zone, r.zext, q, d] + [str(v) for v in vals]))

    # capacity_mw is the line rule; the market borders read market_limits_md_ua.csv instead.
    table["rule"] = np.where((table["group"] == "modelled") & table["zext"].isin(list(MARKET)), "market", "line")
    table["capacity_mw"] = table["capacity_mw"].astype("Int64").mask(table["rule"] == "market")
    table.to_csv(OUT_DIR / "border_flows_2025.csv", index=False)
    mtable.to_csv(OUT_DIR / "market_limits_md_ua.csv", index=False)
    # Same LF convention as the reference file (write_bytes: no newline translation on Windows).
    OUT_LIMIT.write_bytes(("\n".join([raw[0]] + kept + new) + "\n").encode("utf-8"))

    # zext_robg: reference list minus Romania and Bulgaria, plus the new neighbours.
    zext = [z for z in REF_ZEXT.read_text(encoding="utf-8").splitlines()[1:] if z and z not in DROPPED]
    for x in table[table["group"] == "modelled"]["zext"]:
        if x not in zext:
            zext.append(x)
    OUT_ZEXT.write_bytes(("\r\n".join(["zext"] + zext) + "\r\n").encode("utf-8"))

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.astype(object).where(table.notna(), "").to_string(index=False))
        print("\n  market rule (mean flow of the reference year, MW; TWh = MW x 8760 / 1e6):")
        print(mtable.to_string(index=False))
    print(f"\n  wrote {OUT_LIMIT.name}: {len(kept) + len(new)} rows; {OUT_ZEXT.name}: {zext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
