"""Power figures for the Ad hoc digital and power map, from one EPM run.

Reads the run's output_csv and the rep day weights it ran with, writes
data/model_power.json. The run is named in the output so the page can say which
results it shows. Stdlib only.

    python extract_model.py [run_folder] [scenario] [year]
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPM = HERE.parents[1] / "epm"
RUN = sys.argv[1] if len(sys.argv) > 1 else "simulations_run_20260913"
SCENARIO = sys.argv[2] if len(sys.argv) > 2 else "LC_Baseline"
YEAR = sys.argv[3] if len(sys.argv) > 3 else "2035"
OUT_CSV = EPM / "output_view" / RUN / SCENARIO / "output_csv"
HOURS = EPM / "input" / "data_blacksea" / "pHours.csv"
# EU side: the border price the run imports at, not a modelled price.
EU_PRICE = EPM / "input" / "data_blacksea" / "trade" / "pTradePrice_eu_central.csv"
EU_ZONES = ("Romania", "Bulgaria")

COUNTRIES = ("Georgia", "Armenia", "Azerbaijan", "Turkiye")
LOW_CARBON = {"Nuclear", "ROR", "Reservoir", "PV", "Solar", "Onshore Wind", "Offshore Wind",
              "Geothermal", "Biomass", "CSP", "Solar Thermal", "PV+Storage"}
NOT_GENERATION = {"Imports", "Battery", "PSH"}
CHEAP = 40.0  # $/MWh: hours at or below count as cheap


def rows(name):
    with open(OUT_CSV / name, encoding="utf8", newline="") as f:
        yield from csv.DictReader(f)


def weights():
    out = {}
    with open(HOURS, encoding="utf8", newline="") as f:
        for r in csv.DictReader(f):
            for t, v in r.items():
                if t not in ("q", "d"):
                    out[(r["q"], r["d"], t)] = float(v)
    return out


def main():
    w = weights()
    demand = {}
    for r in rows("pYearlyZoneMerged.csv"):
        if r["attribute"] == "DemandEnergyZone" and r["y"] == YEAR:
            demand[r["z"]] = float(r["value"])
    zone_country = {}
    num = defaultdict(float)
    den = defaultdict(float)
    q_num = defaultdict(float)
    q_den = defaultdict(float)
    cheap = defaultdict(float)
    for r in rows("pHourlyPrice.csv"):
        if r["y"] != YEAR or r["c"] not in COUNTRIES:
            continue
        zone_country[r["z"]] = r["c"]
        h = w[(r["q"], r["d"], r["t"])]
        dw = h * demand.get(r["z"], 0.0)
        p = float(r["value"])
        num[r["c"]] += p * dw
        den[r["c"]] += dw
        q_num[(r["c"], r["q"])] += p * dw
        q_den[(r["c"], r["q"])] += dw
        if p <= CHEAP:
            cheap[r["c"]] += dw
    gen = defaultdict(float)
    low = defaultdict(float)
    for r in rows("pTechFuelMerged.csv"):
        if (r["attribute"] != "EnergyTechFuelComplete" or r["y"] != YEAR
                or r["c"] not in COUNTRIES or r["techfuel"] in NOT_GENERATION):
            continue
        v = float(r["value"])
        gen[r["c"]] += v
        if r["techfuel"] in LOW_CARBON:
            low[r["c"]] += v
    bal = defaultdict(float)
    for r in rows("pEnergyBalance.csv"):
        if r["y"] == YEAR and r["c"] in COUNTRIES:
            bal[(r["c"], r["uni"])] += float(r["value"])
    out = {"run": RUN, "scenario": SCENARIO, "year": int(YEAR), "cheap_threshold": CHEAP,
           "countries": {}}
    for c in COUNTRIES:
        out["countries"][c] = {
            "price": num[c] / den[c],
            "price_by_quarter": {q: q_num[(c, q)] / q_den[(c, q)]
                                 for q in ("Q1", "Q2", "Q3", "Q4") if q_den[(c, q)]},
            "cheap_share": cheap[c] / den[c],
            "low_carbon_share": low[c] / gen[c] if gen[c] else None,
            "generation_gwh": gen[c],
            "demand_gwh": bal[(c, "Demand: GWh")],
            "exports_gwh": bal[(c, "Exports exchange: GWh")],
            "imports_gwh": bal[(c, "Imports exchange: GWh")],
        }
    num_eu, den_eu = defaultdict(float), defaultdict(float)
    with open(EU_PRICE, encoding="utf8", newline="") as f:
        for r in csv.DictReader(f):
            if r["zext"] not in EU_ZONES or r["year"] != YEAR:
                continue
            for t, v in r.items():
                if t.startswith("t"):
                    h = w[(r["q"], r["d"], t)]
                    num_eu[r["zext"]] += float(v) * h
                    den_eu[r["zext"]] += h
    out["eu_border_price"] = {z: num_eu[z] / den_eu[z] for z in EU_ZONES if den_eu[z]}
    print("EU border", {z: round(v, 1) for z, v in out["eu_border_price"].items()})
    (HERE / "data" / "model_power.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    for c, v in out["countries"].items():
        print(c, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()
                  if k != "price_by_quarter"},
              {q: round(x, 1) for q, x in v["price_by_quarter"].items()})


if __name__ == "__main__":
    main()
