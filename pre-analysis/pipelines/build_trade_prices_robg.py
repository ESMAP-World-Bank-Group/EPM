# -*- coding: utf-8 -*-
"""Trade prices for LC_Baseline_RoBg, where Romania and Bulgaria are internal.

Writes, next to the reference files and without touching them:

    epm/input/data_blacksea/trade/pTradePrice_robg.csv
    epm/input/data_blacksea/trade/pTradePriceExport_robg.csv

and the intermediate artefacts to pre-analysis/output_prices/robg/.

What changes against pTradePrice_eu_central.csv
-----------------------------------------------
* Romania and Bulgaria rows are dropped: both become internal zones.
* Hungary is priced with the same chain as the existing EU zones, P = L x S,
  run on the Hungarian bidding zone only:
    S  eu_price.py, 2023 day-ahead (ENTSO-E), sampled on the model's
       representative days and normalised to a pHours-weighted mean of 1;
    L  eu_price_level.py, anchored on observed 2024 and joined to TYNDP 2024
       National Trends (HU00), CENTRAL trajectory.
* Serbia and North Macedonia (zone NorthMacedonia) take the Hungarian price. Their TYNDP 2024 NT 2030
  marginal costs (RS00 81.8, MK00 83.1 EUR2022/MWh) sit within 1.5 % of HU00
  (82.0), and the ENTSO-E cache holds no day-ahead series for either zone.
* Netback on the three new borders: the seller bears an AC line loss of 2.5 %
  (bareme of 2026-08-19, `lambda` in config/eu_price_netback.csv). W = 0: the
  ITC perimeter fee falls on non-ITC parties, and Romania, Bulgaria, Hungary,
  Serbia and North Macedonia all sit inside the ITC mechanism. C = 0: CBAM
  applies to imports into the EU from third countries, which an export from
  Romania or Bulgaria is not.
* Every other zone (Greece, Iran, Iraq, Syria, Russia, Kazakhstan) is copied
  byte for byte from the reference files, so the Trakia corridor is priced
  exactly as in LC_Baseline.
* Ukraine and Moldova are not added: the RoBg frontier keeps them closed.

Usage (from EPM/):  python pre-analysis/pipelines/build_trade_prices_robg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_PRE_ANALYSIS = _HERE.parents[1]
_EPM = _PRE_ANALYSIS.parent
sys.path.insert(0, str(_HERE.parent))

import eu_price  # noqa: E402
import eu_price_level  # noqa: E402

HOURS = [f"t{h:02d}" for h in range(1, 25)]
KEY = ["zext", "q", "d", "year"]

TRADE_DIR = _EPM / "epm" / "input" / "data_blacksea" / "trade"
REF_IMPORT = TRADE_DIR / "pTradePrice_eu_central.csv"
REF_EXPORT = TRADE_DIR / "pTradePriceExport_eu_central.csv"
OUT_IMPORT = TRADE_DIR / "pTradePrice_robg.csv"
OUT_EXPORT = TRADE_DIR / "pTradePriceExport_robg.csv"
OUT_DIR = _PRE_ANALYSIS / "output_prices" / "robg"

CACHE_DIR = _PRE_ANALYSIS.parents[1] / "Data" / "cache_entso_e"
TYNDP_DIR = _PRE_ANALYSIS.parents[1] / "Data" / "TYNDP" / "2024"
REPDAYS_DIR = _PRE_ANALYSIS / "representative_days" / "output" / "blacksea"
DEFLATORS = _PRE_ANALYSIS / "config" / "price_deflators.csv"
NETBACK_CFG = _PRE_ANALYSIS / "config" / "eu_price_netback.csv"

DROPPED = ("Romania", "Bulgaria")
PRICED = "Hungary"
PROXIES = ("Serbia", "NorthMacedonia")
LEVEL = "CENTRAL"
FLOOR_USD = 0.01          # same strictly positive floor as eu_price_netback.py
DECIMALS = 3


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_trade_prices_robg] FAIL: {msg}")


def build_hungary() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Shape and level for Hungary, by the reference pipelines, into OUT_DIR."""
    eu_price.ZONES = {"HU": PRICED}
    eu_price_level.ZONES = {"HU": PRICED}
    eu_price_level.TYNDP_CODES = {PRICED: "HU00"}

    qc_s = eu_price.run(CACHE_DIR, REPDAYS_DIR, OUT_DIR, 2023, 15.0, "Georgia")
    if qc_s["G2bis_failed_zones"]:
        _fail(f"shape gate G2bis failed: {qc_s['G2bis_failed_zones']}")
    qc_l = eu_price_level.run(CACHE_DIR, TYNDP_DIR, DEFLATORS, OUT_DIR,
                              2022, [2024], [2023, 2024])
    # G3 (2023 back-cast) and G3ter (ladder order) are diagnostics in the
    # reference chain too: the promoted eu_central files carry G3 misses for
    # Bulgaria and Greece and a G3ter inversion for Romania. Reported, not fatal.
    if qc_l["failed"]:
        print(f"  !! level diagnostics not met (same status as the reference "
              f"zones, recorded in qc_level_L.json): {qc_l['failed']}")

    shape = pd.read_csv(OUT_DIR / "shape_S.csv")
    level = pd.read_csv(OUT_DIR / "level_L.csv")
    level = level[(level["zone"] == PRICED) & (level["scenario"] == LEVEL)]
    if level.empty:
        _fail(f"no {LEVEL} level for {PRICED}")
    return shape, level, {"shape": qc_s, "level": qc_l}


def netback(shape: pd.DataFrame, level: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = pd.read_csv(NETBACK_CFG, comment="#")
    lam = float(cfg[(cfg["series"] == "lambda") & (cfg["year"] == 0)]["value"].iloc[0])
    dfl = pd.read_csv(DEFLATORS, comment="#")
    rate = float(dfl[(dfl["series"] == "eur_usd") & (dfl["year"] == 2024)]["value"].iloc[0])

    sh = shape[shape["zone"] == PRICED][["q", "d"] + HOURS]
    grid = level[["year", "L_eur2024"]].merge(sh, how="cross")
    hub = grid[HOURS].to_numpy() * grid[["L_eur2024"]].to_numpy() * rate

    imp = grid[["q", "d", "year"]].copy()
    imp[HOURS] = hub
    exp = grid[["q", "d", "year"]].copy()
    exp[HOURS] = np.maximum(hub * (1.0 - lam), FLOOR_USD)
    return imp, exp, {"lambda": lam, "eur_usd_2024": rate}


def assemble(ref_path: Path, priced: pd.DataFrame) -> pd.DataFrame:
    """Reference rows minus Romania and Bulgaria, plus the three new zones.

    The new rows follow the (q, d, year) order of the Greece block, so the file
    reads like the reference one, and every key must find a price.
    """
    ref = pd.read_csv(ref_path, dtype=str)
    if list(ref.columns) != KEY + HOURS:
        _fail(f"{ref_path.name}: unexpected header")
    kept = ref[~ref["zext"].isin(DROPPED)]
    order = ref[ref["zext"] == "Greece"][["q", "d", "year"]].copy()
    order["year"] = order["year"].astype(int)

    blocks = [kept]
    for z in (PRICED,) + PROXIES:
        b = order.merge(priced, on=["q", "d", "year"], how="left")
        if b[HOURS].isna().any().any():
            _fail(f"{z}: {int(b[HOURS].isna().any(axis=1).sum())} keys without a price")
        b.insert(0, "zext", z)
        b[HOURS] = b[HOURS].map(lambda v: f"{v:.{DECIMALS}f}")
        b["year"] = b["year"].astype(str)
        blocks.append(b[KEY + HOURS])
    return pd.concat(blocks, ignore_index=True)


def write_like(df: pd.DataFrame, ref_path: Path, out_path: Path) -> None:
    """Same line ending and BOM convention as the reference file."""
    raw = ref_path.read_bytes()
    eol = "\r\n" if b"\r\n" in raw[:4096] else "\n"
    enc = "utf-8-sig" if raw[:3] == b"\xef\xbb\xbf" else "utf-8"
    df.to_csv(out_path, index=False, lineterminator=eol, encoding=enc)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shape, level, _ = build_hungary()
    imp, exp, meta = netback(shape, level)
    imp.to_csv(OUT_DIR / "hungary_import_usd.csv", index=False, float_format="%.4f")
    exp.to_csv(OUT_DIR / "hungary_export_usd.csv", index=False, float_format="%.4f")

    a = assemble(REF_IMPORT, imp)
    b = assemble(REF_EXPORT, exp)
    write_like(a, REF_IMPORT, OUT_IMPORT)
    write_like(b, REF_EXPORT, OUT_EXPORT)

    w = eu_price.hour_weights(pd.read_csv(REPDAYS_DIR / "pHours.csv"))
    print(f"\n  lambda {meta['lambda']:.3f} | EUR->USD {meta['eur_usd_2024']:.4f}")
    for name, df in (("import", imp), ("export", exp)):
        long = df.melt(id_vars=["q", "d", "year"], value_vars=HOURS,
                       var_name="t", value_name="p").merge(w, on=["q", "d", "t"])
        avg = long.groupby("year").apply(
            lambda g: (g["p"] * g["w"]).sum() / g["w"].sum(), include_groups=False)
        print(f"  Hungary {name} weighted mean USD/MWh: "
              + "  ".join(f"{y} {avg[y]:.1f}" for y in (2024, 2030, 2040, 2050) if y in avg))
    for p, df in ((OUT_IMPORT, a), (OUT_EXPORT, b)):
        print(f"  wrote {p.name}: {len(df)} rows, zones {sorted(df['zext'].unique())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
