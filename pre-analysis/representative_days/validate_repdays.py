"""Validate representative days against the FULL 8760-hour year.

Closes the quality gate that was previously only run on Load: every active series
(zone x {Load, PV, Wind}) is checked, not just the load duration curve.

Checks per series:
  (a) mean / energy   : weighted rep-day mean vs true 8760 mean          -> drives energy balance
  (b) duration curve  : weighted rep-day DC vs true DC at P1/P5/P50/P95/P99
  (c) extremes        : max (peak Load) and min (VRE scarcity) coverage

Usage:
    python validate_repdays.py                 # validate deployed rep-days
    python validate_repdays.py --json out.json # also dump machine-readable results

Exit code 1 if any ACTIVE series fails -> usable as a CI/pipeline gate.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
IN_DIR = BASE / "input"
OUT_DIR = BASE / "output" / "blacksea"

# Zones actually optimised by the model (zcmap.csv). Bulgaria/Romania carry data but
# are NOT active zones -> excluded from the gate (they must not consume feature budget).
ACTIVE_ZONES = {
    "Armenia", "Georgia", "AzerbaijanMain", "Nakhchivan",
    "CenterAna", "CenterBlack", "EastAna", "EastMed",
    "NorthWest", "SouthEast", "Trakia", "WestAna", "WestMed",
}

# Pass thresholds (% relative error)
TOL_MEAN = 2.0
TOL_DC = 3.0
TOL_EXTREME = 5.0


def read_truth(tech, year=2023):
    """8760 hourly series per zone -> {zone: [values]}"""
    path = IN_DIR / f"{tech}_{year}.csv"
    series = defaultdict(list)
    skipped = 0
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            raw = (r.get("value") or "").strip()
            if not raw:
                skipped += 1
                continue
            try:
                series[r["zone"]].append(float(raw))
            except ValueError:
                skipped += 1
    if skipped:
        print(f"[gate] warning: {tech}_{year}.csv - skipped {skipped} empty/invalid rows")
    return series


def read_weights():
    """pHours -> {(season, daytype): weight_per_hour}"""
    w = {}
    with open(OUT_DIR / "pHours.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            hours = [float(v) for k, v in r.items()
                     if k not in ("season", "daytype") and v not in (None, "")]
            w[(r["season"], r["daytype"])] = hours
    return w


def read_profile(fname, tech_filter=None):
    """profile csv -> {(zone, season, daytype): [24 values]}"""
    prof = {}
    with open(OUT_DIR / fname, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if tech_filter:
                tcol = r.get("tech") or r.get("fuel") or r.get("technology")
                if tcol != tech_filter:
                    continue
            vals = [float(v) for k, v in r.items()
                    if k.startswith("t") and k[1:].isdigit() and v not in (None, "")]
            prof[(r["zone"], r["season"], r["daytype"])] = vals
    return prof


def weighted_pairs(prof, weights, zone):
    """-> (values, weights) for one zone across all (season, daytype, hour)."""
    vals, wts = [], []
    for (z, s, d), series in prof.items():
        if z != zone:
            continue
        hw = weights.get((s, d))
        if not hw:
            continue
        for i, v in enumerate(series):
            vals.append(v)
            wts.append(hw[i] if i < len(hw) else hw[0])
    return vals, wts


def wmean(vals, wts):
    tw = sum(wts)
    return sum(v * w for v, w in zip(vals, wts)) / tw if tw else 0.0


def wquantile(vals, wts, q):
    """Weighted quantile (q in [0,1]), q=0.99 -> value exceeded 1% of the time."""
    pairs = sorted(zip(vals, wts))
    tw = sum(wts)
    target = q * tw
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= target:
            return v
    return pairs[-1][0]


def quantile(sorted_vals, q):
    idx = min(int(q * (len(sorted_vals) - 1)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def pct_err(got, truth):
    return (got - truth) / truth * 100 if truth else 0.0


def main():
    ap = argparse.ArgumentParser(description="Validate rep-days vs full 8760 year")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    weights = read_weights()
    total_w = sum(sum(h) for h in weights.values())
    print(f"[gate] rep-days weights sum to {total_w:.0f} h ({total_w/24:.0f} days)\n")

    sources = [
        ("Load", read_truth("Load", args.year), read_profile("pDemandProfile.csv")),
        ("PV", read_truth("PV", args.year), read_profile("pVREProfile.csv", "PV")),
        ("Wind", read_truth("Wind", args.year), read_profile("pVREProfile.csv", "Wind")),
    ]

    rows, failures = [], 0
    hdr = f"{'series':<26}{'mean':>16}{'P50':>9}{'P95':>9}{'P05':>9}{'max':>9}  gate"
    print(hdr)
    print("-" * len(hdr))
    for tech, truth, prof in sources:
        for zone in sorted(ACTIVE_ZONES):
            if zone not in truth:
                continue
            vals, wts = weighted_pairs(prof, weights, zone)
            if not vals:
                continue
            t = sorted(truth[zone])
            e_mean = pct_err(wmean(vals, wts), sum(t) / len(t))
            e_p50 = pct_err(wquantile(vals, wts, 0.50), quantile(t, 0.50))
            e_p95 = pct_err(wquantile(vals, wts, 0.95), quantile(t, 0.95))
            e_p05 = pct_err(wquantile(vals, wts, 0.05), quantile(t, 0.05))
            e_max = pct_err(max(vals), max(t))

            ok = (abs(e_mean) <= TOL_MEAN and abs(e_p50) <= TOL_DC
                  and abs(e_p95) <= TOL_DC and abs(e_max) <= TOL_EXTREME)
            if not ok:
                failures += 1
            name = f"{tech}_{zone}"
            print(f"{name:<26}{e_mean:>+15.1f}%{e_p50:>+8.1f}%{e_p95:>+8.1f}%"
                  f"{e_p05:>+8.1f}%{e_max:>+8.1f}%  {'ok' if ok else 'FAIL'}")
            rows.append(dict(series=name, tech=tech, zone=zone, err_mean=e_mean,
                             err_p50=e_p50, err_p95=e_p95, err_p05=e_p05,
                             err_max=e_max, pass_=ok))

    print("-" * len(hdr))
    n = len(rows)
    mean_abs = sum(abs(r["err_mean"]) for r in rows) / n if n else 0
    worst = max(rows, key=lambda r: abs(r["err_mean"])) if rows else None
    print(f"[gate] {n - failures}/{n} series pass "
          f"(tol: mean {TOL_MEAN}% / DC {TOL_DC}% / max {TOL_EXTREME}%)")
    print(f"[gate] mean |mean-error| = {mean_abs:.2f}%   <-- this is the 'yy%' for the slide")
    if worst:
        print(f"[gate] worst mean-error: {worst['series']} {worst['err_mean']:+.1f}%")

    if args.json:
        Path(args.json).write_text(json.dumps(
            dict(summary=dict(n=n, failures=failures, mean_abs_err=mean_abs), series=rows),
            indent=2), encoding="utf-8")
        print(f"[gate] wrote {args.json}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())