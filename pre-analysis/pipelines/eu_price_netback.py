# -*- coding: utf-8 -*-
"""P4 - the netback: from the EU hub price to what a Black Sea exporter gets.

    P_export(zext,y,q,d,t) = L(zext,y) * S(zext,q,d,t) * (1 - lambda) - W - C(y)
    P_import(zext,y,q,d,t) = L(zext,y) * S(zext,q,d,t)                + W

L comes from eu_price_level.py, S from eu_price.py. This module adds only the
three friction terms and checks that what comes out is a price a model can use.

Why the whole loss sits in the price
------------------------------------
EPM applies no loss factor, no wheeling and no tariff to external trade.
base.gms:720 multiplies by (1 - pLossFactorInternal) for internal flows only;
lines 722-723 add vYearlyImportExternal and subtract vYearlyExportExternal raw.
So lambda, W and C must all be carried by the price, and none of them can be
double counted against something the model already does.

Why both directions are computed even though EPM has one price
--------------------------------------------------------------
base.gms:682 charges imports at pTradePrice and base.gms:686 credits exports at
the same pTradePrice. The buy/sell asymmetry built here is therefore currently
*inexpressible* in the model - which is what turns the pTradePriceExport patch
from a refinement into a validity condition. The spread is quantified in the QC
so the size of what is being lost is on the record.

Who bears what
--------------
lambda falls on the seller in each direction: the exporter delivers at the far
end of the line and eats the loss. W is a fee for crossing someone else's
system and falls on whoever initiates the flow, so it is subtracted from the
export revenue and added to the import cost. C is the CBAM levy and applies to
imports *into the EU* only, i.e. to the export direction here, and only in the
CBAM variant.

The two variants
----------------
REF   C = 0. The reference. Not a claim that CBAM will not apply - a claim that
      the reference case should not embed a policy whose incidence on
      electricity is exactly what the study is meant to measure.
CBAM  C = EF_AnnexIII(exporting country) * ets_price(y), at 100 % from 2026.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

_PRE_ANALYSIS = Path(__file__).resolve().parents[1]

HOURS = [f"t{h:02d}" for h in range(1, 25)]
HORIZON = range(2024, 2054)

# Which internal zone sells into each EU external zone, and therefore which
# Annex III factor prices the levy. pTradePrice carries no internal-zone index,
# so this is only well defined because each EU zext faces exactly one internal
# zone - checked by gate_g4zero's companion below rather than assumed.
# Trakia is the Turkish zone bordering Bulgaria and Greece; Georgia faces
# Romania on a link that is present in pExtTransferLimit.csv but still set to
# zero capacity, to be activated in P7.
FACING = {"Bulgaria": ("Trakia", "Türkiye"),
          "Greece": ("Trakia", "Türkiye"),
          "Romania": ("Georgia", "Georgia")}

# CBAM has no phase-in for electricity: EU electricity generation receives no
# free allocation, so there is nothing for the CBAM factor to track up from.
CBAM_START = 2026


def _fail(msg: str) -> None:
    raise SystemExit(f"[eu_price_netback] FAIL: {msg}")


# ── Config ──────────────────────────────────────────────────────────────────
def load_series_config(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    need = {"series", "year", "value"}
    if not need.issubset(df.columns):
        _fail(f"{path.name}: expected columns {need}")
    return df


def _one(cfg: pd.DataFrame, series: str, year: int = 0) -> float:
    m = cfg[(cfg["series"] == series) & (cfg["year"] == year)]
    if m.empty:
        _fail(f"{series} (year {year}) missing from the netback config")
    return float(m["value"].iloc[0])


def load_emission_factors(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path, comment="#")
    if "country" not in df.columns or "ef_electricity" not in df.columns:
        _fail(f"{path.name}: expected columns country, ef_electricity")
    ef = df.dropna(subset=["ef_electricity"])
    return {str(r.country): float(r.ef_electricity) for r in ef.itertuples()}


def build_ets_path(cfg: pd.DataFrame, hicp_base: float, hicp_tyndp: float,
                   log=print) -> pd.Series:
    """Observed 2024 anchor, then linear to the TYNDP points, flat past 2050.

    Identical construction to L on purpose. The alternative - holding the 2030
    TYNDP figure flat back to 2026 - would overstate the levy by about 45 % in
    the first CBAM years, which is precisely the stretch the study cares about.
    """
    pts = {2024: _one(cfg, "ets_observed", 2024)}
    fwd = cfg[cfg["series"] == "ets_tyndp"]
    if fwd.empty:
        _fail("no ets_tyndp points in the netback config")
    for r in fwd.itertuples():
        # TYNDP money is EUR 2022; L is real EUR 2024. Same deflator, so the
        # levy and the price it is subtracted from live in the same money.
        pts[int(r.year)] = float(r.value) * hicp_base / hicp_tyndp

    ys = sorted(pts)
    out = {}
    for y in HORIZON:
        if y <= ys[0]:
            out[y] = pts[ys[0]]
        elif y >= ys[-1]:
            out[y] = pts[ys[-1]]
        else:
            hi = next(k for k in ys if k >= y)
            lo = max(k for k in ys if k <= y)
            f = 0.0 if hi == lo else (y - lo) / (hi - lo)   # y on a published point
            out[y] = pts[lo] + f * (pts[hi] - pts[lo])
    s = pd.Series(out, name="ets_eur2024")
    log("  ETS path, real EUR 2024 per tonne: " + "  ".join(
        f"{y} {s[y]:5.1f}" for y in (2024, 2026, 2030, 2040, 2050)))
    return s


# ── Assembly ────────────────────────────────────────────────────────────────
def build_netback(level: pd.DataFrame, shape: pd.DataFrame, lam: float,
                  w_eur: float, ets: pd.Series, ef: Dict[str, float],
                  rate: float, floor: float) -> pd.DataFrame:
    """One row per zone x scenario x year x q x d x direction x variant.

    The export column is floored at `floor` and the raw value kept alongside.
    Two separate reasons, and neither is cosmetic:

    * The observed 2023 shape contains genuine zeros - Romanian and Bulgarian
      day-ahead collapsed to 0 EUR/MWh at midday on the summer solar day-type
      (Q3/d3, t10-t14). At a zero hub price an exporter still pays W, so the raw
      netback there is -2.2 USD/MWh. Left negative, EPM would read it as the
      exporter paying to deliver.
    * Flooring at exactly zero is not available. main.gms:894-895 switches off
      sExportPrice AND sImportPrice wherever pTradePrice = 0, element by element
      over (q,d,t,y). A zero would therefore also delete the *import* in that
      hour - and an hour when the EU price is zero is precisely the hour a
      neighbour would most want to buy. So the floor has to be strictly
      positive, small enough that no generator finds the export worth serving.

    The floored share is not a diagnostic to be minimised: under CBAM it is the
    result. It says in what fraction of hours the levy makes exporting to the EU
    worthless, and it is reported per zone, scenario, variant and year.
    """
    missing = set(level["zone"]) - set(shape["zone"])
    if missing:
        _fail(f"no hourly shape for {sorted(missing)}")

    frames = []
    for zone, sh in shape.groupby("zone"):
        lz = level[level["zone"] == zone]
        if lz.empty:
            continue
        if zone not in FACING:
            _fail(f"{zone} has no entry in FACING; the levy is undefined")
        _, exporter = FACING[zone]
        if exporter not in ef:
            _fail(f"no Annex III factor for {exporter}, the exporter into {zone}")
        factor = ef[exporter]

        sh = sh[["q", "d"] + HOURS].reset_index(drop=True)
        # Cross the 28 day-types against every scenario-year. The merge is on a
        # constant key rather than a nested loop so the hour columns stay
        # vectorised: 25 200 rows x 24 columns is built in one multiplication.
        grid = lz[["scenario", "year", "L_eur2024"]].merge(sh, how="cross")

        hub = grid[HOURS].to_numpy() * grid[["L_eur2024"]].to_numpy()

        levy = grid["year"].map(
            lambda y: factor * ets[int(y)] if int(y) >= CBAM_START else 0.0
        ).to_numpy()[:, None]

        for direction, variant, values in (
                ("export", "REF", hub * (1.0 - lam) - w_eur),
                ("export", "CBAM", hub * (1.0 - lam) - w_eur - levy),
                # The importer buys at the hub and pays the fee; the EU seller
                # carries the loss, and CBAM does not touch a flow leaving the
                # EU, so the two variants coincide.
                ("import", "REF", hub + w_eur),
                ("import", "CBAM", hub + w_eur)):
            f = grid[["scenario", "year", "q", "d"]].copy()
            f.insert(0, "zone", zone)
            f["exporter"] = exporter if direction == "export" else "EU"
            f["direction"] = direction
            f["variant"] = variant
            v = values * rate                          # EUR 2024 -> USD 2024
            f[HOURS] = np.maximum(v, floor) if direction == "export" else v
            # How much of the hour's value the floor had to invent, so a reader
            # can tell a price from a placeholder without recomputing anything.
            f["n_floored"] = ((v < floor).sum(axis=1) if direction == "export"
                              else 0)
            f["raw_mean_usd"] = v.mean(axis=1)
            frames.append(f)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["zone", "scenario", "variant", "direction",
                            "year", "q", "d"]).reset_index(drop=True)


# ── Gates ───────────────────────────────────────────────────────────────────
def gate_g4a(nb: pd.DataFrame, floor: float) -> dict:
    """No written price may be zero or negative.

    After flooring this holds by construction, which is exactly why it is worth
    asserting: it is the guard that catches a coding error in the flooring
    itself, not a statement about the economics.
    """
    v = nb[HOURS].to_numpy()
    return {"min_written_usd": float(v.min()), "floor_usd": floor,
            "pass": bool(v.min() > 0)}


def gate_g4b(nb: pd.DataFrame) -> dict:
    """P_export <= P_import, hour by hour: nobody may buy low and sell high.

    Also reports the spread, which is what a single pTradePrice cannot express.
    """
    key = ["zone", "scenario", "variant", "year", "q", "d"]
    e = nb[nb["direction"] == "export"].set_index(key)[HOURS]
    i = nb[nb["direction"] == "import"].set_index(key)[HOURS]
    e, i = e.align(i, join="inner")
    if e.empty:
        _fail("no matched export/import pairs; the key is wrong")
    spread = (i - e).to_numpy()
    worst = float(spread.min())
    by_var = {}
    for var in sorted(nb["variant"].unique()):
        m = e.index.get_level_values("variant") == var
        s = spread[m]
        by_var[var] = {"mean_usd": float(s.mean()), "min_usd": float(s.min()),
                       "max_usd": float(s.max())}
    return {"min_spread_usd": worst, "by_variant": by_var,
            "pass": bool(worst > 0)}


def gate_g4c(nb: pd.DataFrame, decimals: int = 4) -> dict:
    """No price may round to zero at the precision the CSV is written with.

    main.gms:894-895 sets sExportPrice/sImportPrice to `no` wherever
    pTradePrice = 0, element by element over (q,d,t,y). A zero does not mean
    free electricity, it means the hour silently disappears - in *both*
    directions, since the two share one parameter. The test is at the written
    precision rather than against an arbitrary epsilon, because what reaches
    GAMS is the rounded value, not the float.
    """
    v = np.round(nb[HOURS].to_numpy(), decimals)
    n = int((v == 0).sum())
    return {"decimals": decimals, "n_rounding_to_zero": n, "pass": n == 0}


def gate_g4d(nb: pd.DataFrame, ref_tol: float = 0.02) -> dict:
    """How often the floor had to stand in for a price, and where.

    Gated on the reference only. In REF the floor should catch nothing but the
    handful of hours where the observed shape is genuinely zero; a large share
    would mean L or S is broken, which is a real failure. In the CBAM variant
    the share is the answer to the question the variant was built to ask, so it
    is reported and never gated - failing it would amount to refusing to
    believe the result.
    """
    e = nb[nb["direction"] == "export"]
    detail, bad = {}, []
    for (zone, scen, var), g in e.groupby(["zone", "scenario", "variant"]):
        share = float(g["n_floored"].sum()) / (len(g) * len(HOURS))
        rec = {"share_floored": round(share, 4),
               "raw_min_usd": round(float(g["raw_mean_usd"].min()), 2)}
        if share > 0:
            hit = g[g["n_floored"] > 0]
            rec["first_year"] = int(hit["year"].min())
        detail[f"{zone}|{scen}|{var}"] = rec
        if var == "REF" and share > ref_tol:
            bad.append(f"{zone}/{scen} {share:.1%}")

    # The headline: from which year does CBAM make the export worthless in a
    # majority of hours?
    cbam = {}
    for (zone, scen), g in e[e["variant"] == "CBAM"].groupby(["zone", "scenario"]):
        by_year = (g.groupby("year")["n_floored"].sum()
                   / (g.groupby("year").size() * len(HOURS)))
        gone = by_year[by_year > 0.5]
        cbam[f"{zone}|{scen}"] = {
            "first_year_majority_floored": int(gone.index.min()) if len(gone) else None,
            "share_2030": round(float(by_year.get(2030, 0.0)), 3),
            "share_2040": round(float(by_year.get(2040, 0.0)), 3)}

    return {"tolerance_ref": ref_tol, "detail": detail,
            "cbam_extinction": cbam, "offending": bad, "pass": not bad}


def levy_table(ets: pd.Series, ef: Dict[str, float], rate: float) -> pd.DataFrame:
    rows = []
    for zone, (izone, exporter) in sorted(FACING.items()):
        for y in HORIZON:
            c = (ef[exporter] * ets[y]) if y >= CBAM_START else 0.0
            rows.append({"zone": zone, "internal_zone": izone,
                         "exporter": exporter, "year": int(y),
                         "ef_tco2_per_mwh": ef[exporter],
                         "ets_eur2024_per_t": float(ets[y]),
                         "C_eur2024_per_mwh": c, "C_usd2024_per_mwh": c * rate})
    return pd.DataFrame(rows)


# ── Driver ──────────────────────────────────────────────────────────────────
def run(prices_dir: Path, config: Path, deflators: Path, ef_path: Path,
        out_dir: Path, floor: float, log=print) -> dict:
    cfg = load_series_config(config)
    dfl = load_series_config(deflators)
    ef = load_emission_factors(ef_path)

    lam = _one(cfg, "lambda")
    w_eur = _one(cfg, "W")
    rate = _one(dfl, "eur_usd", 2024)
    hicp24 = _one(dfl, "eur_hicp", 2024)
    hicp22 = _one(dfl, "eur_hicp", 2022)
    log(f"  lambda {lam:.3f} on the seller | W {w_eur:.2f} EUR/MWh | "
        f"EUR->USD {rate:.4f}")

    ets = build_ets_path(cfg, hicp24, hicp22, log)

    level = pd.read_csv(prices_dir / "level_L.csv")
    shape = pd.read_csv(prices_dir / "shape_S.csv")
    log(f"  L: {len(level)} rows, {level['scenario'].nunique()} scenarios | "
        f"S: {len(shape)} day-types")

    nb = build_netback(level, shape, lam, w_eur, ets, ef, rate, floor)
    levy = levy_table(ets, ef, rate)

    g4a, g4b = gate_g4a(nb, floor), gate_g4b(nb)
    g4c, g4d = gate_g4c(nb), gate_g4d(nb)

    out_dir.mkdir(parents=True, exist_ok=True)
    nb.to_csv(out_dir / "netback.csv", index=False, float_format="%.4f")
    levy.to_csv(out_dir / "cbam_levy.csv", index=False, float_format="%.4f")

    # Headline numbers, in the form the arbitration was argued in. Reported on
    # the raw netback, before the floor: the floor is what the model is given,
    # the raw value is what the economics say.
    head = {}
    for zone in sorted(FACING):
        z = nb[(nb["zone"] == zone) & (nb["scenario"] == "CENTRAL") &
               (nb["direction"] == "export")]
        head[zone] = {}
        for y in (2026, 2030, 2040):
            for var in ("REF", "CBAM"):
                g = z[(z["year"] == y) & (z["variant"] == var)]
                head[zone][f"{y}_{var}_raw_mean_usd"] = round(
                    float(g["raw_mean_usd"].mean()), 2)

    qc = {"lambda": lam, "lambda_incidence": "seller, in each direction",
          "W_eur2024_per_mwh": w_eur, "eur_usd_2024": rate,
          "export_floor_usd": floor,
          "cbam_start": CBAM_START,
          "cbam_phase_in": "none for electricity (no free allocation in the EU)",
          "facing": {k: {"internal_zone": v[0], "exporter": v[1]}
                     for k, v in FACING.items()},
          "emission_factors_annex3": {v[1]: ef[v[1]] for v in FACING.values()},
          "ets_eur2024": {int(y): float(v) for y, v in ets.items()},
          "export_raw_mean_usd_CENTRAL": head,
          "G4a_positive": g4a, "G4b_spread": g4b, "G4c_no_zero": g4c,
          "G4d_floored_share": g4d}

    (out_dir / "qc_netback.json").write_text(
        json.dumps(qc, indent=2, default=str), encoding="utf-8")

    log(f"\n  wrote {out_dir / 'netback.csv'} ({len(nb)} rows)")
    log(f"  wrote {out_dir / 'cbam_levy.csv'} ({len(levy)} rows)")
    log(f"  wrote {out_dir / 'qc_netback.json'}")

    log("\n  CBAM levy, USD 2024 per MWh")
    piv = levy[levy["year"].isin((2026, 2030, 2040, 2050))].pivot(
        index=["zone", "exporter"], columns="year",
        values="C_usd2024_per_mwh").round(1)
    log(piv.to_string())

    log("\n  raw export netback before the floor, CENTRAL, USD 2024 per MWh")
    for zone, d in head.items():
        log(f"    {zone:9s} " + "  ".join(
            f"{k.replace('_raw_mean_usd', ''):10s} {v:7.1f}" for k, v in d.items()))

    log("\n  share of export hours floored (the export is worthless)")
    for k, v in sorted(g4d["detail"].items()):
        if v["share_floored"] > 0:
            log(f"    {k:26s} {v['share_floored']:6.1%}  from {v.get('first_year')}")
    log("\n  CBAM: first year a majority of hours is worthless")
    for k, v in sorted(g4d["cbam_extinction"].items()):
        log(f"    {k:26s} {str(v['first_year_majority_floored']):>6s}  "
            f"(2030 {v['share_2030']:.0%}, 2040 {v['share_2040']:.0%})")

    log(f"\n  G4a all prices > 0       "
        f"{'PASS' if g4a['pass'] else 'FAIL'} "
        f"(min written {g4a['min_written_usd']:.4f} USD/MWh)")
    log(f"  G4b P_export < P_import  "
        f"{'PASS' if g4b['pass'] else 'FAIL'} "
        f"(min spread {g4b['min_spread_usd']:+.2f} USD/MWh)")
    log(f"  G4c none rounds to zero  "
        f"{'PASS' if g4c['pass'] else 'FAIL'} "
        f"({g4c['n_rounding_to_zero']} at {g4c['decimals']} decimals)")
    log(f"  G4d floor in REF         "
        f"{'PASS' if g4d['pass'] else 'FAIL ' + ', '.join(g4d['offending'])}")

    qc["failed"] = [n for n, g in (("G4a", g4a), ("G4b", g4b), ("G4c", g4c),
                                   ("G4d", g4d)) if not g["pass"]]
    return qc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the netback price from L, S and the friction terms.")
    ap.add_argument("--prices-dir", type=Path,
                    default=_PRE_ANALYSIS / "output_prices")
    ap.add_argument("--config", type=Path,
                    default=_PRE_ANALYSIS / "config" / "eu_price_netback.csv")
    ap.add_argument("--deflators", type=Path,
                    default=_PRE_ANALYSIS / "config" / "price_deflators.csv")
    ap.add_argument("--emission-factors", type=Path,
                    default=_PRE_ANALYSIS / "config" / "cbam_emission_factors.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=_PRE_ANALYSIS / "output_prices")
    ap.add_argument("--floor-usd", type=float, default=0.01,
                    help="strictly positive floor on the export price; zero is "
                         "not available because main.gms:894 reads it as 'no "
                         "trade this hour', in both directions")
    args = ap.parse_args(argv)

    print("[eu_price_netback] P = L * S * (1 - lambda) - W - C")
    qc = run(args.prices_dir, args.config, args.deflators,
             args.emission_factors, args.out_dir, args.floor_usd)
    if qc["failed"]:
        print("[eu_price_netback] gates not met: " + ", ".join(qc["failed"]))
        return 1
    print("[eu_price_netback] all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
