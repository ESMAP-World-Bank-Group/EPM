# -*- coding: utf-8 -*-
"""Cross-check the cache against summary.csv before anything is drawn from it.

Three independent checks, none of which can pass by accident:
  1. energy balance  : generation + imports - exports - charging ~ demand
  2. dispatch closure: hourly dispatch reweighted by pHours ~ annual energy
  3. trade symmetry  : what a scope exports to a partner is what the partner
                       records as an import from it
"""
import json
import sys
from pathlib import Path

CACHE = Path(__file__).parent / "cache"
TOL = 0.005          # 0.5 % — dispatch rounding to 2 decimals costs ~0.1 %


def pct(a, b):
    return abs(a - b) / max(abs(b), 1e-6)


def main():
    files = sorted(CACHE.glob("*.json"))
    if not files:
        sys.exit("aucun cache dans %s" % CACHE)
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    years, hours = d["years"], d["hours"]
    fails = []

    print("cache : %s" % files[-1].name)

    # ---- 1. energy balance -------------------------------------------------
    # gen + imports + unmet - exports - surplus should equal demand; the
    # few percent left over are network and storage round-trip losses.
    print("\n1. bilan energetique  (gen + imp + non-servi - exp - surplus)")
    for scope in d["annual"]:
        for scen in d["scenarios"]:
            a = d["annual"][scope][scen]
            worst, wy = 0.0, None
            for i, y in enumerate(years):
                gen = sum(v[i] for v in a["gen"].values())
                imp = sum(t["imp"][i] for t in a["trade"].values())
                exp = sum(t["exp"][i] for t in a["trade"].values())
                supply = gen + imp + a["unmet"][i] - exp - a["surplus"][i]
                gap = pct(supply, a["demand"][i])
                if gap > worst:
                    worst, wy = gap, y
            flag = "OK " if worst < 0.03 else "ECART"
            print("   %-5s %-10s %-12s max %5.1f %% (%s)"
                  % (flag, scope, scen, worst * 100, wy))
            if worst >= 0.03:
                fails.append("bilan %s/%s %.1f%%" % (scope, scen, worst * 100))

    # ---- 2. dispatch closure ----------------------------------------------
    print("\n2. bouclage dispatch  (horaire repondere vs energie annuelle)")
    for scope in d["dispatch"]:
        for scen in d["scenarios"]:
            disp = d["dispatch"][scope][scen]
            axis = disp["axis"]
            w = [hours["%s|%s|%s" % (s["q"], s["d"], s["t"])] for s in axis]
            tot_h = sum(w)
            if abs(tot_h - 8760) > 1:
                fails.append("pHours somme %.0f h != 8760" % tot_h)
                print("   ECART somme des heures = %.0f" % tot_h)
            a = d["annual"][scope][scen]
            for y in d["dispatch_years"]:
                i = years.index(y)
                for fuel, series in disp["years"][y].items():
                    if fuel not in a["gen"]:
                        continue
                    twh = sum(v * ww for v, ww in zip(series, w)) / 1e6
                    ref = a["gen"][fuel][i]
                    if ref < 0.05:
                        continue
                    g = pct(twh, ref)
                    tag = "OK " if g < TOL else "ECART"
                    if g >= TOL:
                        fails.append("%s/%s/%s %s %.2f vs %.2f"
                                     % (scope, scen, y, fuel, twh, ref))
                        print("   %s %-8s %-12s %s %-14s %8.3f vs %8.3f  (%.1f %%)"
                              % (tag, scope, scen, y, fuel, twh, ref, g * 100))
            # demand closes the same way
            for y in d["dispatch_years"]:
                i = years.index(y)
                dem = disp["years"][y].get("Demand")
                if not dem:
                    continue
                twh = sum(v * ww for v, ww in zip(dem, w)) / 1e6
                g = pct(twh, a["demand"][i])
                if g >= TOL:
                    fails.append("%s/%s/%s demande %.2f vs %.2f"
                                 % (scope, scen, y, twh, a["demand"][i]))
                    print("   ECART %-8s %-12s %s demande %8.3f vs %8.3f (%.1f %%)"
                          % (scope, scen, y, twh, a["demand"][i], g * 100))
            print("   ...  %-8s %-12s %d combustibles x %d annees verifies"
                  % (scope, scen, len(a["gen"]), len(d["dispatch_years"])))

    # ---- 3. trade symmetry -------------------------------------------------
    print("\n3. symetrie des echanges  (couloirs vs agregats annuels)")
    for scen in d["scenarios"]:
        cor = d["corridors"][scen]
        a = d["annual"]["Georgia"][scen]
        worst, wk = 0.0, None
        for key, c in cor.items():
            if "Georgia" not in (c["a"], c["b"]):
                continue
            other = c["b"] if c["a"] == "Georgia" else c["a"]
            if other not in a["trade"]:
                continue
            out_ = c["fwd"] if c["a"] == "Georgia" else c["rev"]
            in_ = c["rev"] if c["a"] == "Georgia" else c["fwd"]
            for i in range(len(years)):
                for got, ref in ((out_[i], a["trade"][other]["exp"][i]),
                                 (in_[i], a["trade"][other]["imp"][i])):
                    if max(abs(got), abs(ref)) < 0.01:
                        continue
                    g = pct(got, ref)
                    if g > worst:
                        worst, wk = g, "%s %s" % (key, years[i])
        flag = "OK " if worst < TOL else "ECART"
        print("   %-5s %-12s max %5.2f %% (%s)" % (flag, scen, worst * 100, wk))
        if worst >= TOL:
            fails.append("symetrie %s %.2f%%" % (scen, worst * 100))

    print("\n%s" % ("TOUT PASSE" if not fails
                    else "%d ECART(S):\n  - %s" % (len(fails), "\n  - ".join(fails[:20]))))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
