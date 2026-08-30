"""Capex accounting for the *external* interconnectors of the Black Sea study.

EPM prices internal transmission expansion (eAnnualizedTransmissionCapex,
base.gms:701) but external corridors are exogenous: pExtTransferLimit carries no
investment variable, so a scenario that widens a border gets the capacity for
free and every expansion scenario wins trivially.

This script closes that gap outside GAMS. It reads the capacity steps straight
out of the pExtTransferLimit_*.csv files, prices them against a catalogue
(trade/pExtTransmissionCost.csv, which GAMS never sees) and annualises them with
exactly the convention of base.gms:705-707, so the figures line up with the
model's own transmission cost line.

Nothing here is read by GAMS, and nothing here depends on the optimisation
result except the optional switching value.
"""

import argparse
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Conventions copied from the model. Do not "improve" these: the whole point is
# that external capex stays comparable, line for line, with the transmission
# cost the model already reports.
#
#   base.gms:705-707  annuity = Capex * WACC / (1 - (1+WACC)^-Life)
#                     The /2 there corrects the double counting of internal
#                     symmetric lines (charged from both zones); an external
#                     corridor has a single modelled zone, so no /2. Life only
#                     feeds the CRF - vNewTransmissionLine is a cumulative
#                     stock, so the model keeps paying the annuity every year to
#                     the end of the horizon, past end of life.
#   main.gms:677-683  pWeightYear / pRR, the discounting generate_report.gms
#                     :755-759 uses to build the NPV.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "epm" / "input" / "data_blacksea"
CATALOGUE_NAME = "pExtTransmissionCost.csv"
BASELINE_SCENARIO = "LC_Baseline"
COST_LABEL = "External interconnection capex: $m"


def fail(msg):
    raise SystemExit("ext_transmission_cost: " + msg)


def crf(wacc, life):
    return wacc / (1.0 - (1.0 / ((1.0 + wacc) ** life)))


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.reader(f) if r and any(c.strip() for c in r)]


# ---------------------------------------------------------------------------
# Model settings: horizon, year weights, discount factors
# ---------------------------------------------------------------------------

def read_settings(data):
    """WACC and discount rate out of pSettings.csv (label, name, value)."""
    out = {}
    for row in read_rows(data / "pSettings.csv"):
        if len(row) >= 3:
            out[row[1].strip()] = row[2].strip()
    try:
        return float(out["WACC"]), float(out["DR"])
    except (KeyError, ValueError):
        fail("could not read WACC / DR from pSettings.csv")


def read_years(data):
    rows = read_rows(data / "y.csv")
    years = sorted(int(r[0]) for r in rows[1:])
    if not years:
        fail("y.csv holds no year")
    return years


def year_weights_and_discount(years, dr):
    """Reproduce main.gms:677-683 exactly."""
    weight = {years[0]: 1.0}
    for prev, y in zip(years, years[1:]):
        weight[y] = float(y - prev)
    rr = {years[0]: 1.0}
    for i, y in enumerate(years):
        if i == 0:
            continue
        expo = sum(weight[y2] for y2 in years[:i]) - 1.0 + weight[y] / 2.0
        rr[y] = 1.0 / ((1.0 + dr) ** expo)
    return weight, rr


# ---------------------------------------------------------------------------
# Scenario -> pExtTransferLimit file resolution
# ---------------------------------------------------------------------------

def resolve_ext_files(data):
    """Map every scenario to the pExtTransferLimit file it actually reads.

    A blank cell in scenarios.csv falls back on config.csv, the rule epm.py
    applies.
    """
    default = None
    for row in read_rows(data / "config.csv"):
        if len(row) >= 3 and row[1].strip() == "pExtTransferLimit":
            default = row[2].strip()
    if default is None:
        fail("config.csv has no pExtTransferLimit entry")

    rows = read_rows(data / "scenarios.csv")
    header = [c.strip() for c in rows[0]]
    scenarios = header[1:]
    line = next((r for r in rows[1:] if r[0].strip() == "pExtTransferLimit"), None)
    if line is None:
        return {s: default for s in scenarios}
    cells = (line + [""] * len(header))[1:len(header)]
    return {s: (c.strip() or default) for s, c in zip(scenarios, cells)}


# ---------------------------------------------------------------------------
# Capacity trajectories
# ---------------------------------------------------------------------------

def read_ext_transfer(path):
    """(z, zext, quarter, direction) -> {year: MW}."""
    rows = read_rows(path)
    years = [int(c) for c in rows[0][4:]]
    traj = {}
    for row in rows[1:]:
        key = tuple(c.strip() for c in row[:4])
        vals = [float(c) if c.strip() else 0.0 for c in row[4:]]
        traj[key] = dict(zip(years, vals))
    return traj


def increments(base_traj, scen_traj, years):
    """Capacity commissioned each year by the scenario, over the horizon.

    A corridor is one physical asset, but its NTC is directional: Trakia-
    Bulgaria carries 334 MW of import and 200 MW of export on the same
    conductors. Reinforcing it to a common target therefore adds a different
    number of MW to each direction, and the asset has to be sized on the
    binding one - the largest addition across quarters and directions. Taking
    the max also keeps the pricing conservative: no direction is charged twice
    and none is left free.

    Corridors that never change are left alone, whatever their asymmetry.
    """
    per_corridor = {}
    for key in set(base_traj) | set(scen_traj):
        b, s = base_traj.get(key, {}), scen_traj.get(key, {})
        delta = tuple(round(s.get(y, 0.0) - b.get(y, 0.0), 6) for y in years)
        if all(abs(v) < 1e-6 for v in delta):
            continue
        if any(v < -1e-6 for v in delta):
            fail("%s-%s (%s %s): scenario capacity falls below baseline; this "
                 "script only prices additions" % key)
        per_corridor.setdefault(key[:2], []).append(delta)

    out = {}
    for corridor, variants in per_corridor.items():
        delta = tuple(max(v[i] for v in variants) for i in range(len(years)))
        steps, peak = [], 0.0
        for y, d in zip(years, delta):
            if d - peak > 1e-6:
                steps.append((y, d - peak))
            peak = max(peak, d)
        if steps:
            out[corridor] = steps
    return out


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

def read_catalogue(path):
    """(z, zext) -> {family: [phase, ... in file order]}."""
    cat = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["z"].strip(), r["zext"].strip())
            capex = r["CapexMUSD"].strip()
            cat.setdefault(key, {}).setdefault(r["family"].strip(), []).append({
                "phase": r["phase"].strip(),
                "mw": float(r["CapacityMW"]),
                "capex": float(capex) if capex else None,
                "life": float(r["Life"]),
            })
    return cat


def pick_family(cat, key, total_mw):
    """Select the catalogue phases whose capacity matches what the data adds.

    A scenario may stop part-way through a family - LC_BSSC builds the first
    phase of Georgia-Romania and LC_GECO both - so any leading run of phases is
    a candidate, and the phases have to be listed in build order.

    Matching on capacity rather than on a hard-coded scenario list keeps the
    catalogue from drifting away from the pExtTransferLimit files: rescale a
    corridor and this raises instead of mispricing it.
    """
    families = cat.get(key)
    if not families:
        fail("%s-%s gains %.0f MW but has no row in %s; add it, or the scenario "
             "is priced at zero" % (key[0], key[1], total_mw, CATALOGUE_NAME))
    hits, offers = [], {}
    for family, phases in families.items():
        running = 0.0
        offers[family] = []
        for i, p in enumerate(phases):
            running += p["mw"]
            offers[family].append(running)
            if abs(running - total_mw) < 1.0:
                hits.append((family, phases[:i + 1]))
    if not hits:
        fail("%s-%s: data adds %.0f MW, catalogue offers %s; no match"
             % (key[0], key[1], total_mw, offers))
    if len(hits) > 1:
        fail("%s-%s: %.0f MW matches several catalogue entries %s; capacities "
             "must be distinct" % (key[0], key[1], total_mw,
                                   [h[0] for h in hits]))
    return hits[0]


def allocate(steps, phases):
    """Spread a family's phases over the capacity steps read from the data.

    Phases are consumed in catalogue order. A step may consume several phases
    (LC_GECO commissions BSSC and GECO at once) and a phase may span several
    steps (the 60% scenario ramps one reinforcement from 2027 to 2040), in which
    case its capex follows the capacity pro rata.
    """
    full = dict((p["phase"], p["mw"]) for p in phases)
    queue = [dict(p) for p in phases]
    out = []
    for year, mw in steps:
        left = mw
        while left > 1e-6:
            if not queue:
                fail("capacity steps exceed the catalogue capacity by %.0f MW "
                     "in %d" % (left, year))
            head = queue[0]
            take = min(left, head["mw"])
            out.append({
                "year": year,
                "phase": head["phase"],
                "mw": take,
                "capex": (None if head["capex"] is None
                          else head["capex"] * take / full[head["phase"]]),
                "life": head["life"],
            })
            head["mw"] -= take
            left -= take
            if head["mw"] <= 1e-6:
                queue.pop(0)
    unused = sum(p["mw"] for p in queue)
    if unused > 1e-6:
        fail("catalogue holds %.0f MW the data never commissions" % unused)
    return out


# ---------------------------------------------------------------------------
# Costing
# ---------------------------------------------------------------------------

def cost_scenario(steps_by_corridor, cat, years, wacc):
    """Annuity by year, the per-phase detail, and the phases left unpriced."""
    by_year = dict.fromkeys(years, 0.0)
    detail, unpriced = [], []
    for key, steps in sorted(steps_by_corridor.items()):
        family, phases = pick_family(cat, key, sum(mw for _, mw in steps))
        for a in allocate(steps, phases):
            rec = {"z": key[0], "zext": key[1], "family": family,
                   "phase": a["phase"], "commissioning": a["year"],
                   "mw": a["mw"], "capex_musd": a["capex"], "life": a["life"]}
            if a["capex"] is None:
                rec["annuity_musd_per_yr"] = None
                unpriced.append(rec)
            else:
                ann = a["capex"] * crf(wacc, a["life"])
                rec["annuity_musd_per_yr"] = ann
                for y in years:
                    if y >= a["year"]:
                        by_year[y] += ann
            detail.append(rec)
    return by_year, detail, unpriced


def npv(by_year, years, weight, rr):
    return sum(by_year[y] * weight[y] * rr[y] for y in years)


def npv_by_zone(r):
    """Per scenario and internal zone, the discounted cost of the external corridors.

    The model objective never sees this capex - pExtTransferLimit carries no
    investment variable - so no output of a run can be made to yield it. Written
    beside summary.csv as npv_external.csv, it is the one component the results
    dashboard cannot rebuild on its own, and the zone column is what lets it land
    on the right country.

    Same discounting as npv(): the annuity is paid from commissioning to the end
    of the horizon, on the model's own year weights.
    """
    rows = []
    for scenario in r["ext_files"]:
        by_zone = {}
        for d in r["rows_detail"]:
            if d["scenario"] != scenario or d["annuity_musd_per_yr"] is None:
                continue
            factor = sum(r["weight"][y] * r["rr"][y]
                         for y in r["years"] if y >= d["commissioning"])
            by_zone[d["z"]] = (by_zone.get(d["z"], 0.0)
                               + d["annuity_musd_per_yr"] * factor)
        for z, v in sorted(by_zone.items()):
            if abs(v) > 1e-9:
                rows.append({"scenario": scenario, "zone": z,
                             "value": round(v, 4)})
    return rows


def annuity_discount_factor(years, weight, rr, commissioning, wacc, life):
    """Discounted cost of one MUSD of capex commissioned in a given year.

    Divide a benefit by this and you get the capex at which the corridor exactly
    breaks even - the one figure that survives an unknown cost estimate.
    """
    c = crf(wacc, life)
    return sum(c * weight[y] * rr[y] for y in years if y >= commissioning)


# ---------------------------------------------------------------------------
# Optional: the model's own NPV, for the switching value
# ---------------------------------------------------------------------------

def load_system_npv(results):
    """Scenario -> NPV of system cost ($m) from a postprocessed run, or None."""
    results = Path(results)
    for path in sorted(results.glob("**/pNetPresentCostSystem.csv")):
        rows = read_rows(path)
        head = [c.strip().lower() for c in rows[0]]
        try:
            i_s, i_a, i_v = (head.index("scenario"), head.index("attribute"),
                             head.index("value"))
        except ValueError:
            continue
        out = {}
        for r in rows[1:]:
            if r[i_a].strip().startswith("NPV of system cost"):
                out[r[i_s].strip()] = float(r[i_v])
        if out:
            return out
    sys.path.insert(0, str(ROOT / "epm"))
    try:
        from postprocessing.utils import extract_epm_folder_by_scenario
        res = extract_epm_folder_by_scenario(str(results), file="epmresults.gdx")
        df = res["pNetPresentCostSystem"]
        col = [c for c in df.columns if c not in ("scenario", "value")][0]
        df = df[df[col].astype(str).str.startswith("NPV of system cost")]
        return dict(zip(df["scenario"], df["value"]))
    except Exception as exc:
        print("  (no system NPV available: %s)" % exc)
        return None


# ---------------------------------------------------------------------------

def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("  wrote %s" % path)


def build(data, baseline):
    """Everything that does not need a simulation result."""
    wacc, dr = read_settings(data)
    years = read_years(data)
    weight, rr = year_weights_and_discount(years, dr)
    cat = read_catalogue(data / "trade" / CATALOGUE_NAME)
    ext_files = resolve_ext_files(data)
    if baseline not in ext_files:
        fail("baseline scenario %s is not in scenarios.csv" % baseline)
    base_traj = read_ext_transfer(data / ext_files[baseline])

    rows_year, rows_detail, rows_npv, unpriced_all = [], [], [], []
    for scenario, rel in ext_files.items():
        steps = increments(base_traj, read_ext_transfer(data / rel), years)
        by_year, detail, unpriced = cost_scenario(steps, cat, years, wacc)
        unpriced_all += [(scenario, u) for u in unpriced]
        for y in years:
            rows_year.append({"scenario": scenario, "attribute": COST_LABEL,
                              "year": y, "value": round(by_year[y], 4)})
        for d in detail:
            rows_detail.append(dict(
                {"scenario": scenario},
                **dict((k, round(v, 4) if isinstance(v, float) else v)
                       for k, v in d.items())))
        rows_npv.append({
            "scenario": scenario,
            "external_capex_musd": round(sum(
                d["capex_musd"] for d in detail if d["capex_musd"] is not None), 2),
            "external_capex_annuity_musd_per_yr": round(
                max(by_year.values()) if by_year else 0.0, 2),
            "external_capex_npv_musd": round(npv(by_year, years, weight, rr), 2),
            "unpriced_phases": ";".join(sorted(set(
                "%s-%s:%s" % (d["z"], d["zext"], d["phase"])
                for d in detail if d["capex_musd"] is None))),
        })
    return dict(wacc=wacc, dr=dr, years=years, weight=weight, rr=rr,
                rows_year=rows_year, rows_detail=rows_detail,
                rows_npv=rows_npv, unpriced=unpriced_all, ext_files=ext_files)


def selftest(data, baseline):
    """Checks the costing has to pass before anyone quotes a number from it."""
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))

    c = crf(0.06, 30)
    check("CRF(6%, 30y) = 0.072649", abs(c - 0.072649) < 1e-6, "%.6f" % c)
    check("14000 MUSD annualises to 1017.1 MUSD/yr",
          abs(14000 * c - 1017.1) < 0.1, "%.1f" % (14000 * c))

    # A phase spread over a ramp must cost exactly the same as the same phase
    # commissioned in one step: only the timing moves.
    ph = [{"phase": "P", "mw": 1000.0, "capex": 470.0, "life": 30.0}]
    step = allocate([(2030, 1000.0)], ph)
    ramp = allocate([(2030, 400.0), (2035, 600.0)], ph)
    check("ramp and step cost the same capex",
          abs(sum(a["capex"] for a in step) - sum(a["capex"] for a in ramp)) < 1e-9)
    check("ramp allocates capex pro rata to capacity",
          abs(ramp[0]["capex"] - 188.0) < 1e-9, "%.3f" % ramp[0]["capex"])

    r = build(data, baseline)
    npv_by = dict((x["scenario"], x) for x in r["rows_npv"])
    det = {}
    for d in r["rows_detail"]:
        det.setdefault(d["scenario"], []).append(d)

    for s in (baseline, "LC_Iso"):
        if s in npv_by:
            check("%s carries no external capex" % s,
                  npv_by[s]["external_capex_musd"] == 0
                  and npv_by[s]["external_capex_npv_musd"] == 0)

    # Scenarios that only touch internal transmission must not pick up an
    # external corridor by accident.
    for s in ("LC_BSTN", "LC_Zangezur", "LC_CTN", "LC_FreeExp"):
        if s in npv_by:
            check("%s touches no external corridor" % s, not det.get(s))

    # Phasing: building both phases of a family must cost the sum of the two.
    if "LC_BSSC" in det and "LC_GECO" in det:
        gr = lambda s: [d for d in det[s]
                        if (d["z"], d["zext"]) == ("Georgia", "Romania")]
        bssc = sum(d["capex_musd"] for d in gr("LC_BSSC"))
        geco = sum(d["capex_musd"] for d in gr("LC_GECO"))
        check("LC_GECO = LC_BSSC + the GECO phase",
              abs(geco - 14000.0) < 1e-6 and abs(bssc - 3500.0) < 1e-6,
              "BSSC %.0f, GECO %.0f" % (bssc, geco))
        check("LC_BSSC is a strict prefix of LC_GECO",
              [d["phase"] for d in gr("LC_BSSC")] ==
              [d["phase"] for d in gr("LC_GECO")][:len(gr("LC_BSSC"))])

    # The per-zone split published for the dashboard must be the same money as the
    # scenario total: a zone dropped there is a benefit invented downstream.
    zone_npv = {}
    for row in npv_by_zone(r):
        zone_npv[row["scenario"]] = zone_npv.get(row["scenario"], 0.0) + row["value"]
    for s, x in npv_by.items():
        if x["unpriced_phases"]:
            continue
        check("%s npv_external splits its whole NPV" % s,
              abs(zone_npv.get(s, 0.0) - x["external_capex_npv_musd"]) < 0.01,
              "total %.1f, zones %.1f"
              % (x["external_capex_npv_musd"], zone_npv.get(s, 0.0)))

    # Capacity conservation: what the catalogue prices must equal what the
    # pExtTransferLimit files actually commission.
    years = r["years"]
    ext = resolve_ext_files(data)
    base_traj = read_ext_transfer(data / ext[baseline])
    for s, rel in ext.items():
        want = sum(mw for steps in increments(
            base_traj, read_ext_transfer(data / rel), years).values()
            for _, mw in steps)
        got = sum(d["mw"] for d in det.get(s, []))
        check("%s prices every MW it commissions" % s, abs(want - got) < 1e-6,
              "data %.0f, priced %.0f" % (want, got))

    # The annuity must be flat from commissioning to the end of the horizon:
    # base.gms keeps paying past end of life, and so do we.
    for s, rows in det.items():
        priced = [d for d in rows if d["annuity_musd_per_yr"] is not None]
        if not priced:
            continue
        by_year = dict((x["year"], x["value"]) for x in r["rows_year"]
                       if x["scenario"] == s)
        check("%s annuity never falls back" % s,
              all(by_year[a] <= by_year[b] + 1e-6
                  for a, b in zip(years, years[1:])),
              "%d = %.1f" % (years[-1], by_year[years[-1]]))

    width = max(len(n) for n, _, _ in checks)
    bad = 0
    for name, ok, detail in checks:
        bad += not ok
        print("  %-4s %-*s %s" % ("PASS" if ok else "FAIL", width, name, detail))
    print("\n%d checks, %d failed" % (len(checks), bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA), help="input folder (data_blacksea)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: --results, else ./ext_transmission_cost)")
    ap.add_argument("--results", default=None,
                    help="a simulation output folder, to add the switching value")
    ap.add_argument("--baseline", default=BASELINE_SCENARIO,
                    help="counterfactual scenario")
    ap.add_argument("--selftest", action="store_true",
                    help="run the consistency checks and exit")
    args = ap.parse_args()

    data = Path(args.data)
    if args.selftest:
        raise SystemExit(selftest(data, args.baseline))
    out = Path(args.out) if args.out else (
        Path(args.results) if args.results else Path("ext_transmission_cost"))

    r = build(data, args.baseline)
    print("WACC=%s  DR=%s  horizon %d-%d  CRF(30y)=%.6f"
          % (r["wacc"], r["dr"], r["years"][0], r["years"][-1], crf(r["wacc"], 30)))

    write_csv(out / "ext_transmission_capex_by_year.csv",
              ["scenario", "attribute", "year", "value"], r["rows_year"])
    write_csv(out / "npv_external.csv",
              ["scenario", "zone", "value"], npv_by_zone(r))
    write_csv(out / "ext_transmission_capex_detail.csv",
              ["scenario", "z", "zext", "family", "phase", "commissioning",
               "mw", "capex_musd", "life", "annuity_musd_per_yr"], r["rows_detail"])

    npv_sys = load_system_npv(args.results) if args.results else None
    for row in r["rows_npv"]:
        row.update({"system_npv_musd": "", "benefit_vs_baseline_musd": "",
                    "switching_value_capex_musd": "", "capex_ratio": ""})
        if not (npv_sys and row["scenario"] in npv_sys and args.baseline in npv_sys):
            continue
        benefit = npv_sys[args.baseline] - npv_sys[row["scenario"]]
        det = [d for d in r["rows_detail"] if d["scenario"] == row["scenario"]]
        tot_mw = sum(d["mw"] for d in det)
        if tot_mw <= 0:
            continue
        # Capacity-weighted discount factor: one MUSD spent on this scenario's
        # commissioning profile costs this much in NPV terms.
        fac = sum(annuity_discount_factor(r["years"], r["weight"], r["rr"],
                                          d["commissioning"], r["wacc"], d["life"])
                  * d["mw"] for d in det) / tot_mw
        row["system_npv_musd"] = round(npv_sys[row["scenario"]], 2)
        row["benefit_vs_baseline_musd"] = round(benefit, 2)
        if fac > 0:
            sv = benefit / fac
            row["switching_value_capex_musd"] = round(sv, 2)
            if row["external_capex_musd"]:
                row["capex_ratio"] = round(sv / row["external_capex_musd"], 3)

    write_csv(out / "ext_transmission_switching_value.csv",
              ["scenario", "external_capex_musd",
               "external_capex_annuity_musd_per_yr", "external_capex_npv_musd",
               "system_npv_musd", "benefit_vs_baseline_musd",
               "switching_value_capex_musd", "capex_ratio", "unpriced_phases"],
              r["rows_npv"])

    if r["unpriced"]:
        print("\nUnpriced phases (capex blank in the catalogue):")
        for scenario, u in r["unpriced"]:
            print("  %-18s %s-%s %s %.0f MW from %d"
                  % (scenario, u["z"], u["zext"], u["phase"], u["mw"],
                     u["commissioning"]))

    print("\nSummary (MUSD):")
    print("  %-18s %10s %12s %10s" % ("scenario", "capex", "annuity/yr", "NPV"))
    for row in r["rows_npv"]:
        print("  %-18s %10.1f %12.1f %10.1f"
              % (row["scenario"], row["external_capex_musd"],
                 row["external_capex_annuity_musd_per_yr"],
                 row["external_capex_npv_musd"]))


if __name__ == "__main__":
    main()
