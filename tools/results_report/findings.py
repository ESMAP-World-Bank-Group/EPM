# -*- coding: utf-8 -*-
"""Key findings, computed from the run rather than written by hand.

Every bullet is a sentence template filled from the cache, so re-running the
generator on a new set of scenarios produces findings about that run.  Nothing
here hard-codes a conclusion; a bullet that has no numbers behind it is simply
not emitted.
"""

RENEW = ("Reservoir", "ROR", "PSH", "PV", "Onshore Wind", "Offshore Wind",
         "Biomass", "Geothermal")
FIRM = ("Nuclear", "Coal", "Gas", "Diesel")


def n(v, dp=1):
    """Number formatted the way the prose reads it, thin-space free."""
    if v is None:
        return "n/a"
    s = ("%%.%df" % dp) % v
    return s.rstrip("0").rstrip(".") if "." in s else s


def b(v, dp=1, unit=""):
    return "<b>%s%s</b>" % (n(v, dp), (" " + unit) if unit else "")


def cagr(a, z, yrs):
    if a <= 0 or z <= 0 or yrs <= 0:
        return None
    return ((z / a) ** (1.0 / yrs) - 1.0) * 100.0


def tot(block, i):
    return sum(v[i] for v in block.values())


def _label(scen):
    return {"LC_Baseline": "Baseline", "LC_Iso": "Isolated"}.get(
        scen, scen.replace("LC_", ""))


def _crossings(base, iso, years, key):
    """First year the two scenarios diverge by more than 2 %."""
    for i, y in enumerate(years):
        a, z = base[key][i], iso[key][i]
        if max(abs(a), abs(z)) > 0.05 and abs(a - z) / max(abs(a), 1e-6) > 0.02:
            return y
    return None


def country_findings(d, scope, ref=None, alt=None):
    """Bullets for one country (or the region) comparing two scenarios."""
    years = d["years"]
    scens = d["scenarios"]
    ref = ref or scens[0]
    alt = alt if alt in scens else (scens[1] if len(scens) > 1 else None)
    a = d["annual"][scope][ref]
    out = []
    last = len(years) - 1
    span = int(years[last]) - int(years[0])

    # --- demand ------------------------------------------------------------
    g = cagr(a["demand"][0], a["demand"][last], span)
    out.append(
        "Demand grows from %s to %s between %s and %s%s."
        % (b(a["demand"][0], 1, "TWh"), b(a["demand"][last], 1, "TWh"),
           years[0], years[last],
           (", i.e. %s a year" % b(g, 1, "%/yr")) if g else ""))

    # --- capacity mix ------------------------------------------------------
    cap = a["cap"]
    moves = sorted(((v[last] - v[0], k) for k, v in cap.items()), reverse=True)
    if moves:
        up = [m for m in moves if m[0] > 0.05][:2]
        dn = [m for m in moves if m[0] < -0.05][-2:]
        s = "Installed capacity goes from %s to %s" % (
            b(tot(cap, 0), 1, "GW"), b(tot(cap, last), 1, "GW"))
        if up:
            s += "; the build is led by " + ", ".join(
                "%s (%s)" % (k, b(v, 1, "GW")) for v, k in up)
        if dn:
            s += ", while " + ", ".join(
                "%s sheds %s" % (k, b(-v, 1, "GW")) for v, k in dn)
        out.append(s + ".")

    # --- renewable share ---------------------------------------------------
    gen = a["gen"]
    for i, tag in ((0, years[0]), (last, years[last])):
        pass
    r0 = sum(gen[k][0] for k in RENEW if k in gen)
    r1 = sum(gen[k][last] for k in RENEW if k in gen)
    t0, t1 = tot(gen, 0), tot(gen, last)
    if t0 > 0 and t1 > 0:
        out.append(
            "Renewables cover %s of generation in %s and %s in %s."
            % (b(100 * r0 / t0, 0, "%"), years[0],
               b(100 * r1 / t1, 0, "%"), years[last]))

    # --- trade -------------------------------------------------------------
    imp = [sum(t["imp"][i] for t in a["trade"].values()) for i in range(len(years))]
    exp = [sum(t["exp"][i] for t in a["trade"].values()) for i in range(len(years))]
    if max(imp + exp) > 0.05:
        net = [imp[i] - exp[i] for i in range(len(years))]
        j = max(range(len(years)), key=lambda i: abs(net[i]))
        side = "net importer" if net[j] > 0 else "net exporter"
        share = 100 * abs(net[j]) / max(a["demand"][j], 1e-6)
        out.append(
            "%s is a %s throughout, peaking at %s in %s — %s of its demand."
            % (scope, side, b(abs(net[j]), 1, "TWh"), years[j], b(share, 0, "%")))
        big = sorted(a["trade"].items(),
                     key=lambda kv: -(kv[1]["imp"][j] + kv[1]["exp"][j]))[:2]
        out.append(
            "Its largest counterparties in %s are %s."
            % (years[j], " and ".join(
                "%s (%s in, %s out)" % (k, b(v["imp"][j], 1, "TWh"),
                                        b(v["exp"][j], 1, "TWh"))
                for k, v in big)))

    # --- congestion --------------------------------------------------------
    cor = d["corridors"][ref]
    hot = []
    for key, c in cor.items():
        if scope != "Region" and d["zcmap"].get(c["a"]) != scope \
                and d["zcmap"].get(c["b"]) != scope:
            continue
        peak = max(range(len(years)), key=lambda i: c["util"][i])
        if c["util"][peak] >= 0.85:
            hot.append((c["util"][peak], "%s–%s (%s in %s)"
                        % (c["a"], c["b"], b(100 * c["util"][peak], 0, "%"),
                           years[peak])))
    if hot:
        hot.sort(reverse=True)
        out.append(
            "%d corridor%s run%s at or above 85 %% utilisation: %s."
            % (len(hot), "s" if len(hot) > 1 else "",
               "" if len(hot) > 1 else "s",
               "; ".join(h[1] for h in hot[:4])))

    # --- unserved energy ---------------------------------------------------
    if max(a["unmet"]) > 0.01:
        j = max(range(len(years)), key=lambda i: a["unmet"][i])
        out.append(
            "Even connected, %s leaves %s unserved in %s (%s of demand)."
            % (scope, b(a["unmet"][j], 2, "TWh"), years[j],
               b(100 * a["unmet"][j] / max(a["demand"][j], 1e-6), 1, "%")))

    # --- what isolation costs ---------------------------------------------
    if alt:
        z = d["annual"][scope][alt]
        du = sum(z["unmet"]) - sum(a["unmet"])
        if abs(du) > 0.01:
            out.append(
                "Cutting the interconnectors (%s) adds %s of unserved energy "
                "over the horizon, against %s in %s."
                % (_label(alt), b(du, 1, "TWh"), b(sum(a["unmet"]), 1, "TWh"),
                   _label(ref)))
        if z.get("price") and a.get("price"):
            dp = z["price"][last] - a["price"][last]
            if abs(dp) > 0.5:
                out.append(
                    "The %s marginal cost in %s is %s %s than under %s "
                    "(%s vs %s)."
                    % (_label(alt), years[last], b(abs(dp), 1, "$/MWh"),
                       "higher" if dp > 0 else "lower", _label(ref),
                       b(z["price"][last], 1, "$/MWh"),
                       b(a["price"][last], 1, "$/MWh")))
        dg = tot(z["gen"], last) - tot(gen, last)
        if abs(dg) > 0.2:
            out.append(
                "Domestic generation in %s has to %s by %s in %s to stand alone."
                % (years[last], "rise" if dg > 0 else "fall", b(abs(dg), 1, "TWh"),
                   _label(alt)))
        y = _crossings(a, z, years, "demand")
        del y

    return out


def regional_findings(d, ref=None, alt=None):
    """Bullets that only make sense at regional level."""
    years = d["years"]
    scens = d["scenarios"]
    ref = ref or scens[0]
    alt = alt if alt in scens else (scens[1] if len(scens) > 1 else None)
    last = len(years) - 1
    out = []

    # Cross-border volume, counted once per corridor.
    cor = d["corridors"][ref]
    vol = [sum(c["fwd"][i] + c["rev"][i] for c in cor.values())
           for i in range(len(years))]
    if vol[last] > 0.05:
        out.append(
            "Cross-border and external trade in the region moves %s in %s, "
            "up from %s in %s."
            % (b(vol[last], 1, "TWh"), years[last], b(vol[0], 1, "TWh"), years[0]))

    # Who trades with whom.
    pairs = sorted(cor.items(),
                   key=lambda kv: -(kv[1]["fwd"][last] + kv[1]["rev"][last]))[:3]
    if pairs and (pairs[0][1]["fwd"][last] + pairs[0][1]["rev"][last]) > 0.05:
        out.append(
            "The busiest links in %s are %s."
            % (years[last], "; ".join(
                "%s–%s (%s)" % (c["a"], c["b"],
                                b(c["fwd"][last] + c["rev"][last], 1, "TWh"))
                for _, c in pairs)))

    # Idle investment: capacity that never carries anything.
    idle = [k for k, c in cor.items()
            if c["ntc"][last] > 0
            and (c["fwd"][last] + c["rev"][last]) < 0.01]
    if idle:
        out.append(
            "%d link%s with capacity in %s carr%s no energy: %s."
            % (len(idle), "s" if len(idle) > 1 else "", years[last],
               "y" if len(idle) > 1 else "ies",
               ", ".join("%s–%s" % (cor[k]["a"], cor[k]["b"]) for k in idle[:4])))

    # Country balances.
    bal = []
    for c in sorted(d["annual"]):
        if c == "Region":
            continue
        a = d["annual"][c][ref]
        i = sum(t["imp"][last] for t in a["trade"].values())
        e = sum(t["exp"][last] for t in a["trade"].values())
        bal.append((e - i, c))
    bal.sort(reverse=True)
    if bal and abs(bal[0][0]) > 0.05:
        out.append(
            "In %s the region's net exporter is %s (%s) and its net importer "
            "is %s (%s)."
            % (years[last], bal[0][1], b(bal[0][0], 1, "TWh net out"),
               bal[-1][1], b(-bal[-1][0], 1, "TWh net in")))

    if alt:
        du = sum(d["annual"]["Region"][alt]["unmet"]) - \
             sum(d["annual"]["Region"][ref]["unmet"])
        if abs(du) > 0.01:
            out.append(
                "Region-wide, isolation costs %s of additional unserved energy "
                "across %s–%s."
                % (b(du, 1, "TWh"), years[0], years[last]))
        e0 = d["annual"]["Region"][ref]["emissions"][last]
        e1 = d["annual"]["Region"][alt]["emissions"][last]
        if max(e0, e1) > 0.1 and abs(e1 - e0) > 0.05:
            out.append(
                "%s emissions in %s are %s %s than %s (%s vs %s)."
                % (_label(alt), years[last], b(abs(e1 - e0), 1, "MtCO2"),
                   "higher" if e1 > e0 else "lower", _label(ref),
                   b(e1, 1, "Mt"), b(e0, 1, "Mt")))
    return out
