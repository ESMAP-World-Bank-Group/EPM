"""Check a representative-days run against the full 8760-hour year it came from.

Every active series -- zone x {Load, PV, Wind} -- is compared to the metered year, not
just the load. Per series:

  (a) mean      the weighted rep-day mean against the true 8760 mean, which is what the
                energy balance of the model rests on
  (b) shape     the duration curve at P05, P50 and P95
  (c) extremes  the maximum, which is the peak the reserve is sized against, and for VRE
                the minimum, which is the scarcity hour storage is sized against

WHY THE PEAK MATTERS MORE HERE THAN THE MEAN. The existing 360-block reduction already
holds the mean and the trough of the CASA zones to within a couple of points; where it
fails is the peak PLATEAU -- northern Kazakhstan spends 104 hours of its metered year
within five per cent of peak and the reduction gives it 28. A gate that only watched the
mean would have passed that reduction without a word. TOL_EXTREME is therefore the tight
one, and the peak-hour count is printed beside the peak value so a sharpened peak cannot
hide behind a correct maximum.

The zone list is read from the deployed zcmap.csv rather than written down here: a zone
that leaves the perimeter must leave the gate with it, and West Kazakhstan has already
done exactly that once.

Exit code 1 if any active series fails, so this can be a pipeline gate.

ENVIRONMENT. This runs in gams_env, not epm_env: the pipeline needs scikit-learn, scipy
and seaborn for the clustering and gams.transfer for the weight optimisation, and epm_env
carries none of them.

    conda run -n gams_env python validate_casa_repdays.py

Usage
    python validate_casa_repdays.py
    python validate_casa_repdays.py --json gate.json
"""
from pathlib import Path
import argparse
import csv
import json
import sys
from collections import defaultdict

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
IN_DIR = BASE / "input"
OUT_DIR = BASE / "output" / "casa"

TOL_MEAN = 2.0      # per cent, on the annual mean
TOL_DC = 3.0        # per cent, on the duration curve quantiles
TOL_EXTREME = 5.0   # per cent, on the maximum


def active_zones():
    path = REPO / "epm" / "input" / "data_casa" / "zcmap.csv"
    with open(path, encoding="utf-8-sig") as fh:
        return {row["z"].strip() for row in csv.DictReader(fh) if row.get("z", "").strip()}


def read_truth(name):
    """The hourly year per zone, from what the driver staged."""
    path = IN_DIR / name
    if not path.exists():
        return {}
    series = defaultdict(list)
    skipped = 0
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("value") or "").strip()
            try:
                series[row["zone"]].append(float(raw))
            except ValueError:
                skipped += 1
    if skipped:
        print("[gate] warning: {0} skipped {1} unreadable rows".format(name, skipped))
    return series


def read_weights():
    """pHours -> {(season, daytype): [hours per hour-column]}"""
    weights = {}
    with open(OUT_DIR / "pHours.csv", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            hours = [float(v) for k, v in row.items()
                     if k not in ("season", "daytype", "q", "d") and v not in (None, "")]
            key = (row.get("season") or row.get("q"), row.get("daytype") or row.get("d"))
            weights[key] = hours
    return weights


def read_profile(name, tech=None):
    path = OUT_DIR / name
    if not path.exists():
        return {}
    profile = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if tech:
                column = row.get("tech") or row.get("fuel") or row.get("technology")
                if column != tech:
                    continue
            values = [float(v) for k, v in row.items()
                      if k.startswith("t") and k[1:].lstrip("0").isdigit()
                      and v not in (None, "")]
            key = (row["zone"], row.get("season") or row.get("q"),
                   row.get("daytype") or row.get("d"))
            profile[key] = values
    return profile


def weighted_pairs(profile, weights, zone):
    values, hours = [], []
    for (z, season, day), series in profile.items():
        if z != zone:
            continue
        weight = weights.get((season, day))
        if not weight:
            continue
        for i, value in enumerate(series):
            values.append(value)
            hours.append(weight[i] if i < len(weight) else weight[0])
    return values, hours


def wmean(values, hours):
    total = sum(hours)
    return sum(v * w for v, w in zip(values, hours)) / total if total else 0.0


def wquantile(values, hours, q):
    pairs = sorted(zip(values, hours))
    target = q * sum(hours)
    seen = 0.0
    for value, weight in pairs:
        seen += weight
        if seen >= target:
            return value
    return pairs[-1][0]


def quantile(ordered, q):
    return ordered[min(int(q * (len(ordered) - 1)), len(ordered) - 1)]


def pct(got, truth):
    return (got - truth) / truth * 100 if truth else 0.0


def hours_near_peak(values, hours, peak):
    return sum(w for v, w in zip(values, hours) if v >= 0.95 * peak)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    if not (OUT_DIR / "pHours.csv").exists():
        raise SystemExit("no run at {0}: run run_casa_repdays.py first".format(OUT_DIR))

    zones = active_zones()
    weights = read_weights()
    total = sum(sum(h) for h in weights.values())
    print("[gate] weights sum to {0:.0f} h ({1:.0f} days)".format(total, total / 24))
    if round(total) != 8760:
        print("[gate] WARNING: that is not 8760 h; input_verification.py will refuse it")

    sources = [
        ("Load", read_truth("Load_casa.csv"), read_profile("pDemandProfile.csv")),
        ("PV", read_truth("PV_casa.csv"), read_profile("pVREProfile.csv", "PV")),
        ("Wind", read_truth("Wind_casa.csv"), read_profile("pVREProfile.csv", "Wind")),
    ]
    if not sources[1][1] and not sources[2][1]:
        print("[gate] no renewable series staged: this run clustered on Load alone and\n"
              "       the gate can say nothing about PV or wind.")

    rows, failures = [], 0
    header = ("{0:<24}{1:>10}{2:>9}{3:>9}{4:>9}{5:>9}{6:>13}  gate"
              .format("series", "mean", "P05", "P50", "P95", "max", "h>=95% pk"))
    print("\n" + header)
    print("-" * len(header))

    for tech, truth, profile in sources:
        for zone in sorted(zones):
            if zone not in truth:
                continue
            values, hours = weighted_pairs(profile, weights, zone)
            if not values:
                continue
            ordered = sorted(truth[zone])

            e_mean = pct(wmean(values, hours), sum(ordered) / len(ordered))
            e_p05 = pct(wquantile(values, hours, 0.05), quantile(ordered, 0.05))
            e_p50 = pct(wquantile(values, hours, 0.50), quantile(ordered, 0.50))
            e_p95 = pct(wquantile(values, hours, 0.95), quantile(ordered, 0.95))
            e_max = pct(max(values), max(ordered))

            true_peak = max(ordered)
            true_plateau = sum(1 for v in ordered if v >= 0.95 * true_peak)
            model_plateau = hours_near_peak(values, hours, max(values))

            ok = (abs(e_mean) <= TOL_MEAN and abs(e_p50) <= TOL_DC
                  and abs(e_p95) <= TOL_DC and abs(e_max) <= TOL_EXTREME)
            failures += 0 if ok else 1

            name = "{0}_{1}".format(tech, zone)
            print("{0:<24}{1:>+9.1f}%{2:>+8.1f}%{3:>+8.1f}%{4:>+8.1f}%{5:>+8.1f}%"
                  "{6:>7.0f}/{7:<5.0f}  {8}".format(
                      name, e_mean, e_p05, e_p50, e_p95, e_max,
                      model_plateau, true_plateau, "ok" if ok else "FAIL"))
            rows.append(dict(series=name, tech=tech, zone=zone, err_mean=e_mean,
                             err_p05=e_p05, err_p50=e_p50, err_p95=e_p95,
                             err_max=e_max, plateau_model=model_plateau,
                             plateau_truth=true_plateau, passed=ok))

    print("-" * len(header))
    n = len(rows)
    if not n:
        raise SystemExit("[gate] no series could be compared")

    mean_abs = sum(abs(r["err_mean"]) for r in rows) / n
    worst = max(rows, key=lambda r: abs(r["err_mean"]))
    lost = [r for r in rows
            if r["tech"] == "Load" and r["plateau_truth"] > 2 * r["plateau_model"]]

    print("[gate] {0}/{1} series pass (mean {2}% / curve {3}% / max {4}%)".format(
        n - failures, n, TOL_MEAN, TOL_DC, TOL_EXTREME))
    print("[gate] mean |mean error| {0:.2f}%, worst {1} {2:+.1f}%".format(
        mean_abs, worst["series"], worst["err_mean"]))
    if lost:
        print("[gate] peak plateau more than halved in {0} zone(s): {1}".format(
            len(lost), ", ".join(r["zone"] for r in lost)))
        print("       the reduction is sharpening the peak, which understates how long "
              "the system is under stress and undervalues peaking capacity.")

    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            summary=dict(n=n, failures=failures, mean_abs_err=mean_abs,
                         plateau_halved=[r["zone"] for r in lost]),
            series=rows), indent=2), encoding="utf-8")
        print("[gate] wrote {0}".format(args.json))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
