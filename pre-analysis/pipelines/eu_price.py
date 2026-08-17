"""eu_price.py - Extract the *shape* of EU day-ahead prices onto EPM slices.

Problem this solves
-------------------
External trade in EPM is priced by `pTradePrice(zext,q,d,y,t)`. A flat value
(today: $70 everywhere) makes the model export at the mean price during the
very hours when the EU price collapses - i.e. sunny midday hours, which are
exactly the hours a Caucasus solar build would export into. That
systematically overvalues exports.

Method
------
Same shape/level decomposition as `vre_cf_anchoring.py`, applied to prices:

    P(zext,q,d,y,t) = L(zext,y) x S(q,d,t) x (1 - lambda) - W - C

`L` is the annual *level* (a scenario variable, built in a later phase). `S`
is the dimensionless *shape*, measured here from historical ENTSO-E day-ahead
prices and normalised so its pHours-weighted mean is exactly 1.

The shape must be sampled **by real calendar date**: each representative slice
(q,d) maps to one actual 2023 day, and its 24 hourly prices are taken from
that day. That is what preserves the correlation between price and VRE output.

Clock convention
----------------
The representative-day inputs are in **UTC** (verified empirically: PV peaks at
h=8 for Georgia/Armenia UTC+4, h=9 for Turkiye UTC+3, h=10 for Romania/Bulgaria
UTC+2/+3 - i.e. local solar noon in every zone), and `t1` is hour 0. The
ENTSO-E parquet cache is stored in *market local time* (Europe/Bucharest,
Europe/Sofia, Europe/Athens), so it MUST be converted before slicing. A silent
1-2 h offset would destroy precisely the correlation this pipeline exists to
preserve, so the conversion is asserted, not assumed.

Converting to UTC also removes three calendar traps that exist only in local
time: 2023-01-01 has 22 h (the fetch window starts at 00:00 UTC), 2023-03-26
has 23 h and 2023-10-29 has 25 h (DST). In UTC, 2023 is exactly 8760 clean
hours.

Gates
-----
G2    pHours-weighted mean of S == 1 (tolerance 1e-6), per zone.
G2bis price-duration curve of the reconstructed 8760 vs the true 8760:
      |dP10| and |dP90| < 15 % of the annual mean.
G2ter every slice has 24 finite values; the 28 (q,d) pairs resolve to distinct
      real calendar dates.

Outputs (pre-analysis/output_prices/)
-------------------------------------
shape_S.csv                 zone,q,d,t01..t24 - the deliverable
sampled_prices.csv          same grid, raw EUR/MWh, before normalisation
observed_price_stats.csv    annual mean/percentiles/negative-hour counts (feeds L)
qc_shape_S.json             every gate result and diagnostic
qc_shape_S.png              duration curves, true vs reconstructed (if matplotlib)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_PRE_ANALYSIS = _HERE.parents[1]

# ENTSO-E bidding-zone code -> EPM zext name (see input/data_blacksea/trade/zext.csv)
ZONES: Dict[str, str] = {"RO": "Romania", "BG": "Bulgaria", "GR": "Greece"}

# Must mirror SEASONS_MAP in representative_days/run_blacksea_repdays.py
SEASONS_MAP = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3,
               10: 4, 11: 4, 12: 4}

TCOLS = [f"t{h + 1:02d}" for h in range(24)]


def _fail(msg: str) -> None:
    raise SystemExit(f"[eu_price] FAIL: {msg}")


# ── Load ────────────────────────────────────────────────────────────────────
def load_prices(cache_dir: Path, code: str, year: int) -> pd.Series:
    """Return the `year` hourly day-ahead series for `code`, indexed in UTC.

    Blocking checks: the cache must be timezone-aware (a naive index cannot be
    aligned to the UTC repday clock), the year must be complete, hourly,
    unique and finite.
    """
    hits = sorted(cache_dir.glob(f"da_{code}_*.parquet"))
    if not hits:
        _fail(f"no day-ahead parquet for {code} in {cache_dir}")
    # Prefer a file whose name does not carry the '_lt' long-history marker.
    path = next((p for p in hits if "_lt_" not in p.name), hits[0])

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        _fail(f"{path.name}: index is {type(df.index).__name__}, expected DatetimeIndex")
    if df.index.tz is None:
        _fail(f"{path.name}: timezone-naive index - cannot align to the UTC repday "
              f"clock. Refetch the cache with a tz-aware index.")

    native_tz = str(df.index.tz)
    s = df.iloc[:, 0].tz_convert("UTC")
    s = s[s.index.year == year].sort_index()

    expected = 8784 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 8760
    if len(s) != expected:
        _fail(f"{code} {year}: {len(s)} hours in UTC, expected {expected}")
    if s.index.has_duplicates:
        _fail(f"{code} {year}: duplicate timestamps after UTC conversion")
    gaps = s.index.to_series().diff().dropna().unique()
    if len(gaps) != 1 or gaps[0] != pd.Timedelta(hours=1):
        _fail(f"{code} {year}: index is not strictly hourly ({gaps})")
    if not np.isfinite(s.to_numpy()).all():
        _fail(f"{code} {year}: {int(s.isna().sum())} missing values")

    s.attrs["native_tz"] = native_tz
    s.attrs["source"] = path.name
    return s


# ── Slice ───────────────────────────────────────────────────────────────────
def to_slices(s: pd.Series) -> pd.DataFrame:
    """Attach (season, day-in-season, hour) to a UTC hourly series.

    Reproduces `representative_days/utils.month_to_season`: Feb 29 dropped,
    then days renumbered sequentially from 1 inside each season.
    """
    df = pd.DataFrame({"value": s.to_numpy()}, index=s.index)
    df["month"] = df.index.month
    df["hour"] = df.index.hour
    df["season"] = df["month"].map(SEASONS_MAP)
    df["date"] = df.index.date
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    df = df.sort_index()
    df["day"] = df.groupby("season").cumcount() // 24 + 1
    return df


def sample_representative_days(df: pd.DataFrame, repr_days: pd.DataFrame) -> pd.DataFrame:
    """Pull the 24 hourly values of each representative day. Wide, t01..t24."""
    rd = repr_days.copy()
    rd["season_i"] = rd["season"].str[1:].astype(int)

    keep = df.merge(rd[["season_i", "day", "season", "daytype"]],
                    left_on=["season", "day"], right_on=["season_i", "day"],
                    how="inner", suffixes=("", "_q"))
    wide = keep.pivot_table(index=["season_q", "daytype"], columns="hour",
                            values="value", aggfunc="first")
    if wide.shape != (len(rd), 24):
        _fail(f"sampled grid is {wide.shape}, expected ({len(rd)}, 24) - "
              f"a representative day is missing hours")
    wide.columns = TCOLS
    wide.index.names = ["q", "d"]

    dates = (keep.groupby(["season_q", "daytype"])["date"].nunique())
    if (dates != 1).any():
        _fail("a representative slice spans more than one calendar date")
    return wide.reset_index()


# ── Weights and normalisation ───────────────────────────────────────────────
def hour_weights(hours: pd.DataFrame) -> pd.DataFrame:
    """pHours.csv -> long (q, d, t, w). Same convention as vre_cf_anchoring."""
    qcol = "q" if "q" in hours.columns else "season"
    dcol = "d" if "d" in hours.columns else "daytype"
    tcols = [c for c in hours.columns if c.startswith("t") and c[1:].isdigit()]
    m = hours.melt(id_vars=[qcol, dcol], value_vars=tcols, var_name="t", value_name="w")
    m["t"] = m["t"].str.replace(r"^t(\d+)$", lambda g: f"t{int(g.group(1)):02d}", regex=True)
    return m.rename(columns={qcol: "q", dcol: "d"})


def normalise(wide: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Divide by the pHours-weighted mean so the weighted mean of S is 1."""
    long = wide.melt(id_vars=["q", "d"], value_vars=TCOLS, var_name="t", value_name="p")
    long = long.merge(w, on=["q", "d", "t"], how="left")
    if long["w"].isna().any():
        _fail("pHours does not cover every (q,d,t) of the sampled grid")
    mean = float((long["p"] * long["w"]).sum() / long["w"].sum())
    if abs(mean) < 1e-9:
        _fail("weighted mean price is zero - cannot normalise")
    shape = wide.copy()
    shape[TCOLS] = shape[TCOLS] / mean
    return shape, mean


# ── Gates ───────────────────────────────────────────────────────────────────
def gate_g2(shape: pd.DataFrame, w: pd.DataFrame, tol: float = 1e-6) -> float:
    long = shape.melt(id_vars=["q", "d"], value_vars=TCOLS, var_name="t", value_name="s")
    long = long.merge(w, on=["q", "d", "t"], how="left")
    m = float((long["s"] * long["w"]).sum() / long["w"].sum())
    if abs(m - 1.0) > tol:
        _fail(f"G2: weighted mean of S is {m:.9f}, expected 1 +/- {tol}")
    return m


def reconstruct(wide: pd.DataFrame, w: pd.DataFrame) -> np.ndarray:
    """The 8760 h the model actually sees: each sampled day repeated by its weight."""
    long = wide.melt(id_vars=["q", "d"], value_vars=TCOLS, var_name="t", value_name="p")
    long = long.merge(w, on=["q", "d", "t"], how="left")
    return np.repeat(long["p"].to_numpy(), long["w"].to_numpy().astype(int))


def gate_g2bis(wide: pd.DataFrame, w: pd.DataFrame, truth: pd.Series,
               tol_pct: float = 15.0) -> dict:
    """Compare the reconstructed price-duration curve to the true 8760.

    The comparison is made on *normalised* curves - S against truth/mean(truth) -
    because that is the object the model receives: the annual level is carried by
    L(zext,y) and re-anchored on the observed mean, so a level bias in the sample
    is corrected by construction and must not be charged to the shape. Deviations
    are expressed as a share of the annual mean rather than of the percentile
    itself: EU prices pass through zero, so a relative error on P10 would be
    meaningless when P10 is near zero.

    Returns a verdict rather than aborting: a miss here is a property of the
    representative-day selection, not a bug in this pipeline, and the evidence
    must reach disk before anyone decides what to do about it.
    """
    reps = reconstruct(wide, w)
    real = truth.to_numpy()
    mean, mrep = float(real.mean()), float(reps.mean())

    out = {"n_reconstructed": int(reps.size), "n_true": int(real.size),
           "annual_mean_true": mean, "annual_mean_reconstructed_raw": mrep,
           "level_bias_pct": (mrep - mean) / abs(mean) * 100.0,
           "tolerance_pct_of_mean": tol_pct, "basis": "normalised (S vs truth/mean)"}

    worst, worst_q = 0.0, None
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        a_raw, b = float(np.percentile(reps, q)), float(np.percentile(real, q))
        a_norm = float(np.percentile(reps / mrep, q))
        dev = abs(a_norm - b / mean) * 100.0
        out[f"P{q}"] = {"true": b, "reconstructed_raw": a_raw,
                        "true_norm": b / mean, "reconstructed_norm": a_norm,
                        "dev_pct_of_mean": dev,
                        "dev_pct_of_mean_raw": abs(a_raw - b) / abs(mean) * 100.0}
        if q in (10, 90) and dev > worst:
            worst, worst_q = dev, q

    out["worst_p10_p90_dev_pct"] = worst
    out["worst_percentile"] = f"P{worst_q}"
    out["pass"] = bool(worst <= tol_pct)
    return out


def trade_value_bias(wide: pd.DataFrame, w: pd.DataFrame, truth: pd.Series) -> dict:
    """What the duration-curve error costs in money.

    A percentile miss only matters if it lands on hours we trade in. For a set of
    merit-order strategies - export in the N % most expensive hours, import in the
    N % cheapest - compare the price realised on the true 8760 with the price the
    model would realise on L x S (L anchored on the true annual mean).
    """
    reps = reconstruct(wide, w)
    real = np.sort(truth.to_numpy())
    mean = float(truth.mean())
    model = np.sort(reps / reps.mean() * mean)          # what the model sees
    n = real.size
    out = {}
    for pct in (10, 25, 50):
        k = int(round(n * pct / 100.0))
        exp_t, exp_m = float(real[-k:].mean()), float(model[-k:].mean())
        imp_t, imp_m = float(real[:k].mean()), float(model[:k].mean())
        out[f"top{pct}pct_export"] = {
            "true": exp_t, "model": exp_m, "err_pct": (exp_m - exp_t) / abs(exp_t) * 100.0}
        out[f"bottom{pct}pct_import"] = {
            "true": imp_t, "model": imp_m,
            "err_eur_mwh": imp_m - imp_t,
            "err_pct_of_annual_mean": (imp_m - imp_t) / abs(mean) * 100.0}
    return out


def gate_g2ter(wide: pd.DataFrame, df: pd.DataFrame, repr_days: pd.DataFrame) -> dict:
    vals = wide[TCOLS].to_numpy()
    if not np.isfinite(vals).all():
        _fail("G2ter: non-finite value in the sampled grid")

    rd = repr_days.copy()
    rd["season_i"] = rd["season"].str[1:].astype(int)
    first = df.groupby(["season", "day"])["date"].first()
    cal = {}
    for _, r in rd.iterrows():
        key = (int(r["season_i"]), int(r["day"]))
        if key not in first.index:
            _fail(f"G2ter: {r['season']} day {r['day']} has no calendar date")
        cal[f"{r['season']}/{r['daytype']}"] = str(first.loc[key])
    if len(set(cal.values())) != len(cal):
        _fail("G2ter: two representative slices map to the same calendar date")
    return cal


# ── Diagnostics ─────────────────────────────────────────────────────────────
def price_stats(s: pd.Series) -> dict:
    a = s.to_numpy()
    return {"mean": float(a.mean()), "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max()),
            "n_negative": int((a < 0).sum()), "n_zero": int((a == 0).sum())}


def vre_correlation(df: pd.DataFrame, wide: pd.DataFrame, w: pd.DataFrame,
                    repr_days: pd.DataFrame, pv_csv: Path, pv_zone: str) -> dict:
    """corr(price, PV) on the true 8760 vs on the weighted sampled slices.

    This is the whole point of sampling by calendar date. If the two numbers
    are close, the 28 days preserve the price/VRE correlation; if the sampled
    one drifts towards zero, the shape has lost the very structure it exists
    to carry, and exports would be valued at the mean price during surplus
    hours.
    """
    if not pv_csv.exists():
        return {"skipped": f"{pv_csv.name} not found"}
    pv = pd.read_csv(pv_csv)
    pv = pv[pv["zone"] == pv_zone].copy()
    if pv.empty:
        return {"skipped": f"zone {pv_zone} absent from {pv_csv.name}"}

    # Same (season, day-in-season) construction as the repday pipeline. The VRE
    # inputs are padded to 366 days, so Feb 29 must be dropped *before* the
    # renumbering or every day after February lands one slot off.
    pv = pv[~((pv["month"] == 2) & (pv["day"] == 29))]
    pv["season"] = pv["month"].map(SEASONS_MAP)
    pv = pv.sort_values(["month", "day", "hour"])
    pv["day"] = pv.groupby("season").cumcount() // 24 + 1
    pv = pv[["season", "day", "hour", "value"]].rename(columns={"value": "pv"})
    if pv["pv"].isna().any():
        _fail(f"{pv_csv.name}: {pv_zone} still holds NaN after dropping Feb 29")

    full = df.merge(pv, on=["season", "day", "hour"], how="inner")
    if len(full) != len(df):
        _fail(f"PV coverage mismatch: {len(full)} of {len(df)} hours matched")
    corr_full = float(np.corrcoef(full["value"], full["pv"])[0, 1])

    rd = repr_days.copy()
    rd["season"] = rd["season"].astype(str)
    long_p = wide.melt(id_vars=["q", "d"], value_vars=TCOLS,
                       var_name="t", value_name="p")
    long_p["hour"] = long_p["t"].str[1:].astype(int) - 1
    long_p = long_p.merge(rd[["season", "daytype", "day"]],
                          left_on=["q", "d"], right_on=["season", "daytype"])
    long_p["season_i"] = long_p["q"].str[1:].astype(int)
    long_p = long_p.merge(pv, left_on=["season_i", "day", "hour"],
                          right_on=["season", "day", "hour"], how="left",
                          suffixes=("", "_pv"))
    long_p = long_p.merge(w, on=["q", "d", "t"], how="left")
    if long_p[["p", "pv", "w"]].isna().any().any():
        _fail("sampled correlation: missing PV or weight on the sampled grid")

    wt = long_p["w"].to_numpy()
    x, y = long_p["p"].to_numpy(), long_p["pv"].to_numpy()
    mx = (wt * x).sum() / wt.sum()
    my = (wt * y).sum() / wt.sum()
    cov = (wt * (x - mx) * (y - my)).sum() / wt.sum()
    sx = np.sqrt((wt * (x - mx) ** 2).sum() / wt.sum())
    sy = np.sqrt((wt * (y - my) ** 2).sum() / wt.sum())
    corr_slices = float(cov / (sx * sy))

    return {"pv_zone": pv_zone, "corr_true_8760": corr_full,
            "corr_representative_days": corr_slices,
            "drift": corr_slices - corr_full}


# ── Driver ──────────────────────────────────────────────────────────────────
def run(cache_dir: Path, repdays_dir: Path, out_dir: Path, year: int,
        tol_dur: float, pv_zone: str = "Georgia", log=print) -> dict:
    repr_days = pd.read_csv(repdays_dir / "repr_days.csv")
    hours = pd.read_csv(repdays_dir / "pHours.csv")
    w = hour_weights(hours)

    wsum = w.groupby(["q", "d"])["w"].first().reset_index()
    chk = repr_days.merge(wsum, left_on=["season", "daytype"], right_on=["q", "d"])
    if not np.allclose(chk["weight"], chk["w"]):
        _fail("pHours weights disagree with repr_days weights")
    total = float(w["w"].sum())
    if abs(total - 8760.0) > 1e-6:
        _fail(f"pHours sums to {total} hours, expected 8760")

    pv_csv = repdays_dir.parents[1] / "input" / f"PV_{year}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes, raws, stats, failed = [], [], [], []
    qc = {"year": year, "gate_G2bis_tolerance_pct": tol_dur, "zones": {}}

    for code, zname in ZONES.items():
        s = load_prices(cache_dir, code, year)
        df = to_slices(s)
        wide = sample_representative_days(df, repr_days)
        shape, mean = normalise(wide, w)

        g2 = gate_g2(shape, w)
        g2bis = gate_g2bis(wide, w, s, tol_pct=tol_dur)
        cal = gate_g2ter(wide, df, repr_days)
        value = trade_value_bias(wide, w, s)
        if not g2bis["pass"]:
            failed.append(f"{zname} {g2bis['worst_percentile']} "
                          f"{g2bis['worst_p10_p90_dev_pct']:.1f} %")

        by_hour = (shape.melt(id_vars=["q", "d"], value_vars=TCOLS,
                              var_name="t", value_name="s")
                   .merge(w, on=["q", "d", "t"])
                   .assign(ws=lambda x: x["s"] * x["w"])
                   .groupby("t").apply(lambda g: g["ws"].sum() / g["w"].sum(),
                                       include_groups=False))

        # Against our own export-side PV (does the shape still say "cheap when
        # Georgia is generating"?) and, where available, against the price
        # zone's own PV, which is what physically sets that price.
        corr = vre_correlation(df, wide, w, repr_days, pv_csv, pv_zone)
        corr_local = vre_correlation(df, wide, w, repr_days, pv_csv, zname)

        qc["zones"][zname] = {
            "source": s.attrs["source"], "native_tz": s.attrs["native_tz"],
            "weighted_mean_price_eur_mwh": mean,
            "G2_weighted_mean_of_S": g2,
            "G2bis_duration_curve": g2bis,
            "G2ter_calendar_dates": cal,
            "trade_value_bias": value,
            "vre_correlation": corr,
            "vre_correlation_local": corr_local,
            "shape_by_hour_utc": {k: float(v) for k, v in by_hour.items()},
            "stats_sampled": price_stats(
                pd.Series(wide[TCOLS].to_numpy().ravel())),
        }
        def _cc(c: dict, tag: str) -> str:
            if "corr_true_8760" not in c:
                return f"{tag} n/a"
            return (f"{tag} {c['corr_true_8760']:+.3f}->"
                    f"{c['corr_representative_days']:+.3f}")

        cc = f"{_cc(corr, 'PV.' + pv_zone[:3])} {_cc(corr_local, 'PV.loc')}"
        log(f"  {zname:9s} mean {mean:7.2f} EUR/MWh | G2 {g2:.9f} | "
            f"G2bis {'PASS' if g2bis['pass'] else 'FAIL'} "
            f"{g2bis['worst_percentile']} {g2bis['worst_p10_p90_dev_pct']:5.1f} % of mean | "
            f"S {shape[TCOLS].to_numpy().min():.2f}-{shape[TCOLS].to_numpy().max():.2f} | {cc}")
        log(f"            export value top10% {value['top10pct_export']['err_pct']:+5.1f} % | "
            f"import cost bottom10% {value['bottom10pct_import']['err_eur_mwh']:+6.2f} EUR/MWh")

        shape.insert(0, "zone", zname)
        wide = wide.copy()
        wide.insert(0, "zone", zname)
        shapes.append(shape)
        raws.append(wide)

        for y in (year, year + 1):
            try:
                sy = load_prices(cache_dir, code, y)
            except SystemExit:
                continue
            stats.append({"zone": zname, "year": y, **price_stats(sy)})

    shape_all = pd.concat(shapes, ignore_index=True)
    raw_all = pd.concat(raws, ignore_index=True)
    stats_df = pd.DataFrame(stats)

    shape_all.to_csv(out_dir / "shape_S.csv", index=False, float_format="%.6f")
    raw_all.to_csv(out_dir / "sampled_prices.csv", index=False, float_format="%.4f")
    stats_df.to_csv(out_dir / "observed_price_stats.csv", index=False, float_format="%.3f")
    (out_dir / "qc_shape_S.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")

    log(f"\n  wrote {out_dir / 'shape_S.csv'} ({len(shape_all)} rows)")
    log(f"  wrote {out_dir / 'sampled_prices.csv'}")
    log(f"  wrote {out_dir / 'observed_price_stats.csv'}")
    log(f"  wrote {out_dir / 'qc_shape_S.json'}")

    qc["G2bis_failed_zones"] = failed
    if failed:
        log(f"\n  G2bis FAILED (tolerance {tol_dur} % of the annual mean): "
            + "; ".join(failed))
        log("  Artefacts written; the threshold was agreed in advance, so it is "
            "not for this script to relax it.")
    return qc


def plot_qc(cache_dir: Path, repdays_dir: Path, out_dir: Path, year: int,
            log=print) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  (matplotlib absent - skipping the QC figure)")
        return

    # Same objects as gate G2bis: each sampled day repeated by its weight, and
    # both curves normalised by their own mean, because the level is carried by
    # L and re-anchored on the observed mean.
    raw = pd.read_csv(out_dir / "sampled_prices.csv")
    w = hour_weights(pd.read_csv(repdays_dir / "pHours.csv"))
    fig, axes = plt.subplots(1, len(ZONES), figsize=(4.2 * len(ZONES), 3.4), sharey=True)
    for ax, (code, zname) in zip(np.atleast_1d(axes), ZONES.items()):
        t = load_prices(cache_dir, code, year).to_numpy()
        r = reconstruct(raw[raw["zone"] == zname].drop(columns="zone"), w)
        true = np.sort(t / t.mean())[::-1]
        rec = np.sort(r / r.mean())[::-1]
        ax.plot(np.linspace(0, 100, true.size), true, lw=1.1, label=f"true {year} (8760 h)")
        ax.plot(np.linspace(0, 100, rec.size), rec, lw=1.4, ls="--",
                label="representative days (weighted to 8760 h)")
        ax.axvline(90, color="0.6", lw=0.8, ls=":")
        ax.set_title(zname)
        ax.set_xlabel("% of hours exceeded")
        ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("price / annual mean  (= S)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "qc_shape_S.png", dpi=140)
    log(f"  wrote {out_dir / 'qc_shape_S.png'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", type=Path,
                    default=_PRE_ANALYSIS.parents[1] / "Data" / "cache_entso_e",
                    help="folder holding da_<code>_<from>_<to>.parquet")
    ap.add_argument("--repdays-dir", type=Path,
                    default=_PRE_ANALYSIS / "representative_days" / "output" / "blacksea",
                    help="folder holding repr_days.csv and pHours.csv")
    ap.add_argument("--out-dir", type=Path, default=_PRE_ANALYSIS / "output_prices")
    ap.add_argument("--year", type=int, default=2023,
                    help="shape year; must be the year the repdays were built on")
    ap.add_argument("--tol-duration", type=float, default=15.0,
                    help="G2bis tolerance, in %% of the annual mean")
    ap.add_argument("--pv-zone", default="Georgia",
                    help="zone whose PV profile the price/VRE correlation is measured against")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    if not args.cache_dir.is_dir():
        _fail(f"price cache not found: {args.cache_dir} (pass --cache-dir)")
    if not args.repdays_dir.is_dir():
        _fail(f"repdays output not found: {args.repdays_dir} (pass --repdays-dir)")

    print(f"[eu_price] shape year {args.year}")
    print(f"  cache   {args.cache_dir}")
    print(f"  repdays {args.repdays_dir}")
    qc = run(args.cache_dir, args.repdays_dir, args.out_dir, args.year,
             args.tol_duration, args.pv_zone)
    if not args.no_plot:
        plot_qc(args.cache_dir, args.repdays_dir, args.out_dir, args.year)
    if qc["G2bis_failed_zones"]:
        print("[eu_price] G2bis NOT met - see qc_shape_S.json")
        return 1
    print("[eu_price] all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
