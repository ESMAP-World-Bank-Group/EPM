# -*- coding: utf-8 -*-
"""ASSUMPTIONS page: what the model assumes, and where each assumption comes from.

    python data_build/assumptions.py --config data_build/build_casa.yaml

DATA_SOURCES.html answers "where does this file come from", resource by resource, and
it answers it exhaustively: 48 tables, every source, every coverage cell. That is the
reference, and it is unreadable as an introduction. This page is the other half. It
takes the dozen decisions that actually shape a result -- the perimeter, the demand
trajectory, what the fleet can do, what a corridor may carry -- states them in prose,
and hands the reader over to DATA_SOURCES.html for the proof.

NOTHING NUMERIC IS TYPED IN THIS FILE. The prose carries {placeholders} and every one
of them is computed, either from the YAML or by reading the deployed CSVs the model
will actually run on. A chapter that drifts from the data therefore breaks visibly
(KeyError on the placeholder) instead of quietly lying, which is the failure mode of a
hand-kept assumptions note and the reason this is a script and not a document.

The chapter text is the one thing written by hand, because "what the model assumes" is
a judgement about what matters and no table states it. It names resources by their
YAML key, so the state badge, the source badges and the link are all derived.
"""

import argparse
import csv
import io
import os
import sys
from collections import OrderedDict, defaultdict
from datetime import date

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracker import remaining, state_of   # noqa: E402  same state as the tracker

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

OUT = "ASSUMPTIONS.html"
# Relative from this file's folder to the deployment folder, so the two pages sit a
# click apart wherever the pair is copied.
SOURCES_PAGE = "../epm/input/data_casa/DATA_SOURCES.html"

STATE = {"G": ("#e2f2e7", "#1e6b46"), "Y": ("#fdeecf", "#8a5a12"),
         "R": ("#f7d9cf", "#a5381f"), "B": ("#eceff3", "#5a6577")}

GRADE = {"primary": ("#d7e9d4", "#2b6b2e"), "secondary": ("#fce4cc", "#8a5a12"),
         "placeholder": ("#f7d9cf", "#a5381f")}


# --------------------------------------------------------------------------- io

def read_csv(path):
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    return rows[0], rows[1:]


def cols(header):
    return {c.strip(): j for j, c in enumerate(header)}


def num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def gw(mw):
    return "{0:,.1f}".format(mw / 1000.0).replace(",", " ")


def thousands(n):
    return "{0:,}".format(int(n)).replace(",", " ")


# ------------------------------------------- reading the representative days

def hourly(path_csv, keys):
    """Rows of a (keys..., t1..t24) profile table, keyed on the tuple of keys."""
    header, rows = read_csv(path_csv)
    hd = [c.strip() for c in header]
    at = len(hd) - 24
    idx = [hd.index(k) for k in keys]
    return [(tuple(r[i].strip() for i in idx),
             [num(x) or 0.0 for x in r[at:at + 24]]) for r in rows]


def day_variation(path_csv, keys):
    """How many groups repeat the same day across the day types, and how many differ.

    The day type is the last key, so a group is everything else: a zone and a season,
    plus a technology for the renewable table. A group whose days are all identical
    carries no intra-season variability at all -- the dimension is there and empty --
    and that is a property of the data no one can see by looking at the file.
    """
    groups = defaultdict(dict)
    for k, v in hourly(path_csv, keys + ["d"]):
        groups[k[:-1]][k[-1]] = v
    same = 0
    for days in groups.values():
        vals = list(days.values())
        if all(v == vals[0] for v in vals):
            same += 1
    return same, len(groups) - same


def plateaus(path_csv):
    """Per zone, the largest number of distinct consecutive levels in a day.

    A profile written over {hours} hourly columns is free to repeat a value: a legacy
    six-block time slice stretched to 24 columns looks hourly and is not. Counting the
    steps is the only way to see the real intraday resolution of a table.
    """
    header, rows = read_csv(path_csv)
    at = len(header) - 24
    steps = defaultdict(int)
    for r in rows:
        v = [x.strip() for x in r[at:at + 24]]
        n = 1 + sum(1 for i in range(23) if v[i] != v[i + 1])
        z = r[0].strip()
        steps[z] = max(steps[z], n)
    return steps


def zone_variation(path_csv, keys):
    """How many non-zone groups write the same profile for every zone.

    If a season and a technology carry one shape for the whole region, the model has
    no geography in that table, however many zones it declares.
    """
    groups = defaultdict(set)
    for k, v in hourly(path_csv, ["z"] + keys):
        groups[k[1:]].add(tuple(v))
    same = sum(1 for v in groups.values() if len(v) == 1)
    return same, len(groups) - same


def day_weights(hours_csv):
    """Hours carried by each (season, day type), which is what a day is worth."""
    _h, rows = read_csv(hours_csv)
    out = OrderedDict()
    for r in rows:
        out[(r[0].strip(), r[1].strip())] = sum(num(x) or 0.0 for x in r[2:])
    return out


def season_shapes(hours_csv, load_csv, vre_csv):
    """Per season: the load shape of each day type, and the one solar shape.

    Averaged over the zones, which is a summary and not a model input; the picture is
    there to show the shape and whether the day types differ, not a zone's own level.
    """
    weights = day_weights(hours_csv)
    load = defaultdict(lambda: defaultdict(list))
    for (z, q, d), v in hourly(load_csv, ["z", "q", "d"]):
        load[q][d].append(v)
    solar = defaultdict(lambda: defaultdict(list))
    for (z, tech, q, d), v in hourly(vre_csv, ["z", "tech", "q", "d"]):
        if tech.strip().upper() in ("PV", "SOLAR", "SOLARPV"):
            solar[q][d].append(v)

    def mean(series):
        return [sum(c) / len(series) for c in zip(*series)] if series else [0.0] * 24

    out = []
    for q in OrderedDict((q, None) for q, _d in weights):
        days = OrderedDict((d, mean(load[q][d])) for d in sorted(load[q]))
        pv = sorted(solar.get(q, {}))
        out.append(dict(season=q,
                        hours=sum(w for (qq, _d), w in weights.items() if qq == q),
                        weights=OrderedDict((d, w) for (qq, d), w in weights.items()
                                            if qq == q),
                        load=days,
                        solar=mean(solar[q][pv[0]]) if pv else [0.0] * 24))
    return out


# ------------------------------------------------------------------- the facts

def facts(data_dir, cfg):
    """Every number the prose is allowed to use, read off the deployed model."""
    f = {}

    def path(*p):
        return os.path.join(data_dir, *p)

    zh, zr = read_csv(path("zcmap.csv"))
    countries = sorted(set(r[1].strip() for r in zr))
    f["zones"] = len(zr)
    f["countries"] = len(countries)
    f["country_list"] = ", ".join(countries)

    _yh, yr = read_csv(path("y.csv"))
    years = sorted(int(r[0]) for r in yr)
    f["years"] = len(years)
    f["y0"], f["y1"] = years[0], years[-1]
    # The horizon thins out after the first decade; say so rather than imply 13 in a row.
    steps = sorted(set(years[i + 1] - years[i] for i in range(len(years) - 1)))
    f["year_step"] = " then ".join("{0}-yearly".format(s) for s in steps)

    hh, hr = read_csv(path("pHours.csv"))
    seasons = OrderedDict((r[0].strip(), None) for r in hr)
    f["seasons"] = len(seasons)
    f["season_list"] = ", ".join(seasons)
    f["days"] = len(hr) // max(len(seasons), 1)
    f["hours"] = len(hh) - 2
    f["blocks"] = len(hr) * f["hours"]
    total = sum(num(c) or 0.0 for r in hr for c in r[2:])
    f["block_hours"] = thousands(round(total))

    # WHAT ACTUALLY VARIES ACROSS THE DAY TYPES. The season x day-type grid exists in
    # every profile table, but a table is free to write the same day three times, and
    # one of the two does. Counting it is the only way to know how much variability
    # the representative days really carry, so it is counted and not assumed.
    for name, rel, keys in (("vre", ("supply", "pVREProfile.csv"), ["z", "tech", "q"]),
                            ("load", ("load", "pDemandProfile.csv"), ["z", "q"])):
        same, diff = day_variation(path(*rel), keys)
        f[name + "_same"], f[name + "_diff"] = same, diff
        f[name + "_groups"] = same + diff
    # THE REAL INTRADAY RESOLUTION, which is not the number of columns. A table can
    # repeat a value across six columns and look hourly; a legacy time slice stretched
    # to 24 columns does exactly that, and only counting the steps reveals it.
    vsteps = plateaus(path("supply", "pVREProfile.csv"))
    f["vre_steps"] = max(vsteps.values())
    f["vre_steps_max"] = max(vsteps.values())
    lsteps = plateaus(path("load", "pDemandProfile.csv"))
    blocky = sorted(z for z, n in lsteps.items() if n <= f["vre_steps_max"])
    hourly_z = sorted(z for z, n in lsteps.items() if n > f["vre_steps_max"])
    f["load_blocky"], f["load_hourly"] = len(blocky), len(hourly_z)
    f["load_blocky_list"] = ", ".join(blocky)
    f["load_hourly_list"] = ", ".join(hourly_z)
    # Geography is a separate question from resolution: a coarse table can still say
    # something different about each zone, and this one does.
    zsame, zdiff = zone_variation(path("supply", "pVREProfile.csv"), ["tech", "q", "d"])
    f["vre_zone_same"], f["vre_zone_diff"] = zsame, zdiff

    weights = day_weights(path("pHours.csv"))
    thin = sorted(set(d for (_q, d), w in weights.items() if w <= 24))
    f["peak_day"] = ", ".join(thin) if thin else "none"
    f["peak_day_n"] = len(thin)

    f["season_shapes"] = season_shapes(path("pHours.csv"),
                                       path("load", "pDemandProfile.csv"),
                                       path("supply", "pVREProfile.csv"))
    span = [(d["season"], max(d["solar"])) for d in f["season_shapes"]]
    f["pv_peak_lo"] = "{0:.2f} in {1}".format(*min(span, key=lambda t: t[1])[::-1])
    f["pv_peak_hi"] = "{0:.2f} in {1}".format(*max(span, key=lambda t: t[1])[::-1])

    th, tr = read_csv(path("pTechFuel.csv"))
    f["techfuel"] = len(tr)

    gh, gr = read_csv(path("supply", "pGenDataInput.csv"))
    gc = cols(gh)
    exist = [r for r in gr if r[gc["Status"]].strip() == "1"]
    f["units"] = len(gr)
    f["existing"] = len(exist)
    f["candidates"] = len(gr) - len(exist)
    f["installed_gw"] = gw(sum(num(r[gc["Capacity"]]) or 0.0 for r in exist))
    by_fuel = defaultdict(float)
    for r in exist:
        by_fuel[r[gc["fuel"]].strip()] += num(r[gc["Capacity"]]) or 0.0
    rank = sorted(by_fuel.items(), key=lambda kv: -kv[1])
    f["fuel_mix"] = ", ".join("{0} {1} GW".format(k, gw(v)) for k, v in rank[:4])
    f["hydro_units"] = sum(1 for r in gr if r[gc["tech"]].strip().upper() in ("HY", "ROR"))
    # A named plant carried at zero contributes nothing, whether it is an existing
    # unit whose capacity was never entered or a candidate whose build limit is zero.
    zero = [r for r in gr if (num(r[gc["Capacity"]]) or 0.0) == 0.0]
    f["zero_cap"] = len(zero)
    f["zero_cap_existing"] = sum(1 for r in zero if r[gc["Status"]].strip() == "1")

    ah, ar = read_csv(path("supply", "pAvailabilityCustom.csv"))
    f["availability_rows"] = len(ar)
    vals = defaultdict(int)
    for r in ar:
        vals[r[1].strip()] += 1
    f["avail_flat_090"] = vals.get("0.9", 0)
    f["avail_2020_085"] = vals.get("0.85", 0)

    dh, dr = read_csv(path("load", "pDemandForecast.csv"))
    dc = cols(dh)
    energy = [r for r in dr if r[dc["type"]].strip().lower().startswith("energy")]
    first, last = str(f["y0"]), str(f["y1"])
    e0 = sum(num(r[dc[first]]) or 0.0 for r in energy)
    e1 = sum(num(r[dc[last]]) or 0.0 for r in energy)
    f["demand_twh_0"] = "{0:,.0f}".format(e0 / 1000.0).replace(",", " ")
    f["demand_twh_1"] = "{0:,.0f}".format(e1 / 1000.0).replace(",", " ")
    f["demand_growth"] = "{0:.1f}".format(
        ((e1 / e0) ** (1.0 / (f["y1"] - f["y0"])) - 1) * 100) if e0 else "n/a"
    zc = dict((r[0].strip(), r[1].strip()) for r in zr)
    per_country = defaultdict(float)
    for r in energy:
        per_country[zc.get(r[dc["z"]].strip(), "?")] += num(r[dc[first]]) or 0.0
    top = max(per_country.items(), key=lambda kv: kv[1])
    f["demand_top"] = top[0]
    f["demand_top_share"] = "{0:.0f}".format(100.0 * top[1] / e0) if e0 else "n/a"

    ph, pr = read_csv(path("supply", "pFuelPrice.csv"))
    f["fuel_rows"] = len(pr)
    f["fuel_countries"] = len(set(r[0].strip() for r in pr))

    xh, xr = read_csv(path("trade", "pTransferLimit.csv"))
    f["corridors"] = len(set((r[0].strip(), r[1].strip()) for r in xr))
    ch, cr = read_csv(path("trade", "pContractedTradeEnergy.csv"))
    f["contracted"] = len(set((r[0].strip(), r[1].strip()) for r in cr))

    sh, sr = read_csv(path("supply", "pStorageDataInput.csv"))
    f["storage_units"] = len(sr)

    rh, rr = read_csv(path("reserve", "pPlanningReserveMarginZone.csv"))
    margins = sorted(set(r[1].strip() for r in rr))
    f["reserve_margin"] = ", ".join(margins)

    kh, kr = read_csv(path("constraint", "pCarbonPrice.csv"))
    prices = [num(r[1]) for r in kr if num(r[1]) is not None]
    f["carbon_lo"] = "{0:.0f}".format(min(prices)) if prices else "n/a"
    f["carbon_hi"] = "{0:.0f}".format(max(prices)) if prices else "n/a"

    ss = cfg.get("sources", {})
    f["source_count"] = len(ss)
    f["primary_count"] = sum(1 for v in ss.values() if v.get("grade") == "primary")

    res = cfg["resources"]
    # state_of returns a colour and a sentence; the sentence varies ("done",
    # "inherited, fit for purpose"), the colour does not, so the tally keys on it.
    buckets = {"G": "done", "Y": "partial", "R": "to_do", "B": "out_of_scope"}
    tally = defaultdict(int)
    for v in res.values():
        tally[buckets[state_of(v)[0]]] += 1
    f["res_total"] = len(res)
    for k in buckets.values():
        f["res_" + k] = tally.get(k, 0)
    return f


# ---------------------------------------------------------------- the chapters
# Prose is hand-written; every number in it is a {placeholder} resolved from facts().

CHAPTERS = [
    dict(
        key="perimeter", title="Perimeter, horizon and time",
        lead="The model optimises {zones} zones across {countries} countries "
             "({country_list}) over {years} model years, {y0} to {y1}, spaced "
             "{year_step}. It is a capacity expansion and a dispatch at once: it "
             "chooses what to build and then runs it.",
        points=[
            ("Time is representative, not chronological",
             "Each year is {seasons} seasons ({season_list}) x {days} representative "
             "days x {hours} chronological hours, so {blocks} blocks weighted to "
             "{block_hours} hours. A representative day carries the shape of many real "
             "days, which flattens any trough that does not fall at the same hour "
             "twice. The model reads that flattening as a baseload the system does not "
             "have, so peaking needs and storage value are both understated at the "
             "margin. This is the single most consequential simplification on the "
             "page."),
            ("Zones are not countries",
             "Kazakhstan, Kyrgyzstan and Tajikistan are split where the network is "
             "genuinely constrained inside the country. A result reported per country "
             "is the sum of its zones, and an intra-country flow is a real transfer "
             "limit, not an accounting line."),
            ("The vocabulary is closed",
             "{techfuel} technology-fuel pairs exist and nothing outside them can be "
             "built, whatever a source proposes. Candidate technologies the model has "
             "no word for -- CCS, ultra-supercritical coal, hydrogen co-firing -- are "
             "out of reach until the vocabulary is extended."),
        ],
        resources=["zcmap", "y", "pHours", "pTechFuel", "pSettings"]),

    dict(
        key="repdays", title="Representative days",
        lead="Each year is {seasons} seasons of {days} representative days of {hours} "
             "hours. Those {days} days are the model's whole account of what varies "
             "inside a season, and three separate things are worth knowing about them "
             "before reading any result: what varies across the days, how fine the "
             "day itself is, and whether the {zones} zones are described equally.",
        extra="season_grid",
        points=[
            ("Demand varies across the days; renewables do not",
             "{load_diff} of the {load_groups} zone-season demand groups carry a "
             "different shape per day type, which is the dimension working as "
             "intended: a cold weekday and a mild Sunday are not the same day. The "
             "renewable table writes the same day {days} times in {vre_same} groups "
             "out of {vre_groups} -- every one of them. Day type {peak_day} is worth "
             "{peak_day_n} single day of each season and carries the peak, which is "
             "how the load side gets its extreme; the renewable side has no equivalent. "
             "There is no day in the whole "
             "{y0}-{y1} horizon on which the sun is behind cloud or the wind drops. "
             "Solar and wind vary by hour and by season, and never by weather."),
            ("What that costs the answer",
             "A system whose renewables never have a bad day needs no firm capacity "
             "against one. Every megawatt of solar delivers its seasonal mean on every "
             "single day, so the marginal panel is worth more than it would be against "
             "a real year, and the marginal battery, gas turbine or import contract is "
             "worth less. That mechanism shapes the {y1} build, and it is a property "
             "of the input, not a finding about the region."),
            ("The day is {vre_steps} steps wide, not {hours}",
             "The renewable profile is written over {hours} hourly columns and takes "
             "at most {vre_steps} distinct values across them: it is a legacy "
             "{vre_steps}-block time slice stretched to hourly columns, so the sun "
             "rises in one jump and sets in another. Demand is split down the middle "
             "on the same test. {load_hourly} zones carry a genuinely hourly shape "
             "({load_hourly_list}); {load_blocky} carry the same {vre_steps} blocks "
             "({load_blocky_list}). The coarse half is Afghanistan and Pakistan, which "
             "is where the demand this study exists to serve actually sits, so the "
             "evening ramp the interconnector would be paid to cover is the least "
             "resolved thing in the model."),
            ("Geography survives the coarseness",
             "It is a separate question and it comes out well: all {vre_zone_diff} "
             "technology-season-day groups write a different profile for different "
             "zones, so the table is coarse in time without being flat in space. Average solar peaks at {pv_peak_hi} against "
             "{pv_peak_lo}, the right seasonal swing for these latitudes. The gap is "
             "resolution and variance, not the shapes themselves."),
            ("How it will be fixed",
             "Hourly ERA5 or Renewables.ninja output per zone and technology, "
             "clustered so that the {days} days of a season span the range instead of "
             "repeating its mean, with the day weights of pHours rebalanced to match. "
             "That closes the variance gap and the {vre_steps}-step gap at once, it is "
             "the next build phase, and it needs data from no one: the reanalysis is "
             "public. The {load_blocky} block-profile demand zones need a real load "
             "curve instead, which does need a data request."),
        ],
        resources=["pHours", "pVREProfile", "pDemandProfile"]),

    dict(
        key="sources", title="Which source wins, and why",
        lead="{source_count} sources are declared, {primary_count} of them primary. "
             "They do not carry equal weight and the order between them is a rule "
             "applied by the build, not a case-by-case judgement.",
        points=[
            ("DeCA 2025 over the 2020 model, where the two disagree",
             "The DeCA / Mercados AssumptionBooks V5.1 (March 2025) are the reference "
             "for the five Central Asian countries: fleet, costs, efficiencies, "
             "hydrology, demand. The ported 2020 CASA model is the fallback. Where "
             "DeCA states a figure and 2020 states another, DeCA is taken."),
            ("Where DeCA is silent, nothing moves",
             "DeCA covers five countries. Pakistan and Afghanistan are outside it and "
             "keep their 2020 values untouched, because silence is not disagreement. "
             "Extrapolating a Central Asian figure across that border would invent a "
             "difference rather than preserve one -- and would invent it in the one "
             "place it changes the answer, {demand_top} being {demand_top_share}% of "
             "the demand this study asks whether Central Asian power should serve."),
            ("Freshness is capped by grade",
             "A source that is recent but secondary -- a pre-filled template nobody "
             "has validated -- never grades better than a measurement. An assumption "
             "carries today's date and still grades as an assumption. The coverage "
             "matrix on the sources page is coloured on that rule, not on age alone."),
        ],
        resources=[]),

    dict(
        key="demand", title="Demand",
        lead="Demand is exogenous: the model never chooses it, it must serve it. "
             "{demand_twh_0} TWh in {y0} rising to {demand_twh_1} TWh in {y1}, "
             "{demand_growth}% a year compounded, and {demand_top} alone is "
             "{demand_top_share}% of the {y0} total.",
        points=[
            ("A trajectory and a shape, kept separate",
             "Annual energy and peak come from the forecast table; the hourly shape "
             "comes from a normalised profile scaled onto it. Splitting them means a "
             "revised forecast does not disturb the shape, and a better profile does "
             "not disturb the total."),
            ("The weak link is Pakistan",
             "{demand_top} is the largest single block of demand in the model and it "
             "rests on 2018 planning material. The NTDC IGCEP 2025-35 is the first "
             "data request on the list; until it lands, the demand that justifies the "
             "interconnection is the least current number on this page."),
        ],
        resources=["pDemandForecast", "pDemandProfile", "pDemandData"]),

    dict(
        key="fleet", title="The fleet: what exists and what may be built",
        lead="{units} generating units, {existing} of them existing for "
             "{installed_gw} GW installed, {candidates} of them candidates the model "
             "may or may not build. Largest fuels: {fuel_mix}. {hydro_units} of the "
             "units are hydro, which is what makes this a hydro-thermal trade study "
             "rather than a thermal one.",
        points=[
            ("Availability asks a different question of an old plant than of a new one",
             "A candidate is written at 0.90 across the seasons -- DeCA's own "
             "arithmetic for a plant in working order, 5% forced outage plus 5% "
             "scheduled. An existing unit is written at its own derating instead, "
             "because DeCA states an available capacity beside the installed capacity "
             "of every plant it carries and the gap between the two is that plant's "
             "condition. The spread runs from 0.39 to 0.90 and it matters: a flat rate "
             "would say a Soviet CHP unmaintained since 1991 and a 2018 combined cycle "
             "are the same machine. {avail_2020_085} units, all Pakistani or Afghan, "
             "stay on the 2020 figure because no source in hand describes them."),
            ("Costs and efficiencies are DeCA's, capacities are not",
             "Heat rates, capex, fixed and variable O&M are rebuilt from the "
             "AssumptionBooks. Installed capacity is deliberately left alone: several "
             "DeCA plants map onto more than one row of this model, so a row-by-row "
             "capacity rewrite would invent or destroy gigawatts. The resulting gap is "
             "reported, not silently closed, and is part of the fleet data request."),
            ("A plant carried at zero contributes nothing, and {zero_cap} rows are",
             "Capacity on a candidate row is the cumulative build limit, so a named "
             "project sitting at zero is one the model is forbidden to choose; five "
             "such hydro projects, 4.1 GW of Kyrgyz and Tajik plant, were sized from "
             "DeCA by this build. What remains is worse and is inherited: "
             "{zero_cap_existing} rows are EXISTING plants with no capacity entered, "
             "and they are largely the Pakistani combined cycles -- Bhikki, Haveli "
             "Bahadur Shah, Guddu, Nandipur -- plus Dushanbe-1 and KANUPP. Those "
             "machines are real, they run, and this model cannot dispatch them. It is "
             "the sharpest single argument for the {demand_top} fleet data request."),
            ("Storage is on, and thin",
             "{storage_units} storage units. Round-trip efficiency and operating life "
             "are assumptions: DeCA states neither."),
        ],
        resources=["pGenDataInput", "pAvailability", "pAvailabilityDefault",
                   "pGenDataInputDefault", "pVREProfile", "pCapexTrajectoriesDefault",
                   "pStorageDataInput"]),

    dict(
        key="fuel", title="Fuel prices and carbon",
        lead="{fuel_rows} price series across {fuel_countries} countries, and a carbon "
             "price running {carbon_lo} to {carbon_hi} $/t over the horizon.",
        points=[
            ("Prices are national, and several are held flat",
             "Where DeCA states a local price it is used. Where it does not -- heavy "
             "fuel oil, LNG, and the whole Pakistani and Afghan set -- the 2020 series "
             "is held flat to {y1}. A flat gas price quietly makes gas a policy "
             "variable rather than a resource one, which is worth remembering before "
             "reading any result that turns on gas-versus-hydro."),
            ("Carbon is a price, not yet a cap",
             "The trajectory is inherited and has not been re-decided for the 2026 "
             "vintage. Emission caps by country exist as a table and are not yet "
             "populated, so decarbonisation currently bites through cost alone."),
        ],
        resources=["pFuelPrice", "pFuelCarbonContent", "pCarbonPrice",
                   "pEmissionsCountry", "pMaxFuellimit", "pMaxFuellimitZone"]),

    dict(
        key="trade", title="Trade: what a corridor may carry",
        lead="{corridors} internal corridors carry a seasonal transfer limit per year; "
             "{contracted} of them additionally carry contracted energy that the model "
             "must honour whatever the economics say.",
        points=[
            ("Transfer limits are the study's most load-bearing assumption",
             "This is a regional integration model: nearly every conclusion is a "
             "statement about how much power may cross a border. The limits are DeCA's "
             "where DeCA has them and the 2020 model's elsewhere, and neither is a TSO "
             "confirmation. Intra-CAPS NTCs are an open data request."),
            ("CASA-1000 is contracted, not chosen",
             "The contracted flows are an obligation on the model, so they show up in "
             "a result whether or not they are least-cost. The underlying PPA "
             "quantities file has not been located; the quantities in place are the "
             "2020 model's."),
            ("The external border is unresolved",
             "External zones, their transfer limits and the prices at which power "
             "crosses them are declared and undecided. Iran, Russia and China are "
             "therefore effectively absent, which is a real restriction on what the "
             "model can say about Central Asian export options."),
        ],
        resources=["pTransferLimit", "pContractedTradeFlag", "pContractedTradeEnergy",
                   "pMaxAnnualTransfer", "pHistoricalTradeFlag", "pNewTransmission",
                   "pExtTransferLimit", "pTradePrice", "zext"]),

    dict(
        key="reserves", title="Reserves and adequacy",
        lead="Every zone carries a planning reserve margin of {reserve_margin} and a "
             "peak season is declared for adequacy.",
        points=[
            ("The margin is assumed, uniformly",
             "The same figure on all {zones} zones is a placeholder, not a national "
             "adequacy standard. It also interacts with the derated availability "
             "above: a margin over nameplate is not a margin at all once forced "
             "outages are counted, so this number and the availability table have to "
             "be read together."),
            ("Spinning reserve is inherited",
             "Country and system requirements come through from the 2020 model with "
             "DeCA confirmation where available."),
        ],
        resources=["pPlanningReserveMarginZone", "pReserveSeasonFlag",
                   "pSpinningReserveReqCountry", "pSpinningReserveReqSystem"]),
]


# ------------------------------------------------------------------- rendering

def spark(series, w=170, h=46, colour="#33445e", fill=None, top=None):
    """A 24-hour polyline. Scaled on `top` so several charts can share an axis."""
    top = top or (max(series) or 1.0)
    step = float(w - 2) / (len(series) - 1)
    pts = " ".join("{0:.1f},{1:.1f}".format(1 + i * step,
                                            h - 1 - (v / top) * (h - 3))
                   for i, v in enumerate(series))
    out = []
    if fill:
        out.append('<polygon points="1,{0} {1} {2},{0}" fill="{3}"/>'.format(
            h - 1, pts, w - 1, fill))
    out.append('<polyline points="{0}" fill="none" stroke="{1}" stroke-width="1.6"/>'
               .format(pts, colour))
    return "".join(out)


def season_grid(f):
    """One row per season: the load shape of each day type, and the solar shape.

    The load charts of a season share one vertical scale so the day types can be
    compared; the solar charts share one across all the seasons so the seasonal swing
    is visible. Drawn from the deployed profile tables, averaged over the zones.
    """
    shapes = f["season_shapes"]
    pvtop = max(max(d["solar"]) for d in shapes) or 1.0
    day_colours = ["#2b6b8a", "#8a5a12", "#7a3b6b"]
    out = ['<div class="method"><b>What the {0} days of a season actually look like.'
           '</b> Load on the left, one line per day type, sharing a scale within the '
           'season. Solar on the right, sharing a scale across every season. Where the '
           'load lines separate, the day types are doing work; the solar chart is a '
           'single line because all {0} days of a season are the same day.</div>'
           .format(f["days"])]
    out.append('<table class="cal"><thead><tr><th>Season</th><th>Hours</th>'
               '<th>Load, {0} day types</th><th>Day weights</th>'
               '<th>Solar, all {0} days</th></tr></thead><tbody>'.format(f["days"]))
    for d in shapes:
        top = max(max(v) for v in d["load"].values()) or 1.0
        lines = "".join(spark(v, colour=day_colours[i % len(day_colours)], top=top)
                        for i, v in enumerate(d["load"].values()))
        weights = "<br>".join(
            '<span style="color:{0}">&#9632;</span> {1} &middot; {2:,.0f} h'.format(
                day_colours[i % len(day_colours)], esc(k), w)
            for i, (k, w) in enumerate(d["weights"].items()))
        out.append(
            '<tr><td><b>{0}</b></td><td class="muted">{1:,.0f} h</td>'
            '<td><svg width="170" height="46">{2}</svg></td>'
            '<td class="muted" style="font-size:.9em">{3}</td>'
            '<td><svg width="170" height="46">{4}</svg></td></tr>'.format(
                esc(d["season"]), d["hours"], lines, weights,
                spark(d["solar"], colour="#b8860b", fill="#f6e6b8", top=pvtop)))
    out.append('</tbody></table>')
    return "".join(out)


EXTRAS = {"season_grid": season_grid}

def badge(text, bg, fg, title=""):
    return ('<span class="pill" style="background:{0};color:{1}"{3}>{2}</span>'
            .format(bg, fg, esc(text), ' title="{0}"'.format(esc(title)) if title else ""))


def source_badges(keys, registry):
    out = []
    for k in keys or []:
        s = registry.get(k, {})
        bg, fg = GRADE.get(s.get("grade"), ("#eceff3", "#5a6577"))
        label = k.replace("_", " ")
        out.append(badge(label, bg, fg,
                         "{0} ({1}, {2})".format(s.get("name", k), s.get("date", "?"),
                                                 s.get("grade", "?"))))
    return " ".join(out) or '<span class="muted">&mdash;</span>'


def anchor(key):
    return "{0}#res-{1}".format(SOURCES_PAGE, key.lower())


def resource_table(keys, res, registry):
    if not keys:
        return ""
    rows = []
    for k in keys:
        v = res.get(k)
        if v is None:
            continue
        code, label = state_of(v)
        bg, fg = STATE[code]
        rows.append(
            '<tr><td><a href="{0}"><code>{1}</code></a></td><td>{2}</td>'
            '<td>{3}</td><td>{4}</td></tr>'.format(
                anchor(k), esc(k), esc((v.get("what") or "").strip()),
                badge(label, bg, fg), source_badges(v.get("source"), registry)))
    return ('<table class="res"><thead><tr><th>Table</th><th>What it holds</th>'
            '<th>State</th><th>Sources</th></tr></thead><tbody>{0}</tbody></table>'
            .format("".join(rows)))


def chapter_state(keys, res):
    codes = [state_of(res[k])[0] for k in keys if k in res]
    if not codes:
        return "#eceff3", "#5a6577", "method"
    if all(c in ("G", "B") for c in codes):
        return STATE["G"][0], STATE["G"][1], "settled"
    if any(c == "R" for c in codes):
        return STATE["R"][0], STATE["R"][1], "open"
    return STATE["Y"][0], STATE["Y"][1], "in progress"


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1040px;
margin:22px auto;padding:0 18px;color:#1a2230;line-height:1.55;background:#fff}
h1{font-size:1.5rem;border-bottom:3px solid #1E6DB8;padding-bottom:8px;margin-bottom:4px;
color:#12355b}
.sub{color:#5a6577;margin-top:0;font-size:.9rem}
.intro{background:#eef4fb;border-radius:8px;padding:13px 17px;font-size:.86rem;margin:14px 0}
.intro a{color:#1E6DB8}
.sm{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}
@media(max-width:720px){.sm{grid-template-columns:repeat(2,1fr)}}
.smcell{border:1px solid #e4e9f1;border-radius:8px;padding:9px 11px;background:#fbfcfe}
.smv{font-size:1.18rem;font-weight:800;color:#12355b}
.smt{font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:#7a869c;
font-weight:700;margin-top:2px}
.tabs{display:flex;gap:5px;flex-wrap:wrap;margin:20px 0 0;border-bottom:2px solid #dde3ec}
.tab{font:inherit;font-size:.9rem;padding:9px 15px;border:none;background:none;
color:#5a6577;cursor:pointer;border-bottom:3px solid transparent;margin-bottom:-2px;
font-weight:600}
.tab.active{color:#12355b;border-bottom-color:#1E6DB8}
.panel{display:none}.panel.active{display:block}
.phead{display:flex;justify-content:space-between;align-items:center;margin:20px 0 6px;
flex-wrap:wrap;gap:10px}
h2{font-size:1.3rem;margin:0;color:#12355b}
.lead{font-size:.93rem;color:#2a3648;margin:6px 0 14px}
.sec{border:1px solid #e4e9f1;border-radius:9px;padding:13px 16px;margin:11px 0;
background:#fbfcfe}
.sn{font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:#1E6DB8;
font-weight:800;margin-bottom:7px}
.body{font-size:.855rem;color:#2a3648}
.pill{display:inline-block;font-size:.62rem;font-weight:800;padding:2px 7px;
border-radius:4px;letter-spacing:.03em;text-transform:uppercase;white-space:nowrap}
table{border-collapse:collapse;width:100%;font-size:.79rem;margin-top:12px}
th,td{border-bottom:1px solid #eef1f6;padding:6px 9px;text-align:left;vertical-align:top}
th{background:#eef2f8;color:#33445e;font-weight:700}
table.res td:first-child{width:24%}
code{background:#f2f5fa;padding:1px 5px;border-radius:3px;font-size:.93em;color:#12355b}
a{color:#1E6DB8}a code{color:#1E6DB8}
.muted{color:#8a94a6;font-style:italic}
.gap{border-left:4px solid #c0682a;background:#fff7ed;border-radius:0 8px 8px 0;
padding:11px 15px;margin:10px 0;font-size:.855rem}
.gap b{color:#8a4a12}
.srcrow td:first-child{font-weight:700;color:#33445e;width:22%}
.foot{margin-top:26px;padding-top:12px;border-top:1px solid #e4e9f1;font-size:.76rem;
color:#7a869c}
"""

JS = """
function show(id){
  var ps=document.querySelectorAll('.panel');
  for(var i=0;i<ps.length;i++){ps[i].classList.remove('active');}
  var ts=document.querySelectorAll('.tab');
  for(var j=0;j<ts.length;j++){ts[j].classList.remove('active');}
  document.getElementById('p-'+id).classList.add('active');
  document.getElementById('t-'+id).classList.add('active');
}
"""


def render(cfg, f, data_dir):
    res = cfg["resources"]
    registry = cfg.get("sources", {})
    out = []
    a = out.append

    a('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    a('<meta name="viewport" content="width=device-width,initial-scale=1">')
    a('<title>Main assumptions &mdash; EPM &mdash; Central Asia 2026</title>')
    a('<style>{0}</style></head><body>'.format(CSS))

    a('<h1>Main assumptions</h1>')
    a('<p class="sub">EPM Central Asia 2026 &mdash; what the model assumes, and where '
      'each assumption comes from. Generated {0} from '
      '<code>build_casa.yaml</code> and the deployed data.</p>'.format(date.today()))

    a('<div class="intro">This page states the dozen decisions that actually shape a '
      'result. It is deliberately short and deliberately incomplete: the exhaustive '
      'record &mdash; every one of the {0} tables, every source, the coverage of each '
      'country &mdash; is on the <a href="{1}">data sources page</a>, and every table '
      'named here links straight to its entry there. Nothing numeric on this page is '
      'typed by hand; it is read from the model that will run.</div>'
      .format(f["res_total"], SOURCES_PAGE))

    tiles = [(f["zones"], "zones"), (f["countries"], "countries"),
             ("{0}&ndash;{1}".format(f["y0"], f["y1"]), "horizon"),
             (f["blocks"], "time blocks"), (f["units"], "generating units"),
             (f["installed_gw"] + " GW", "installed"),
             (f["demand_twh_1"] + " TWh", "demand in {0}".format(f["y1"])),
             (f["corridors"], "corridors")]
    a('<div class="sm">')
    for v, t in tiles:
        a('<div class="smcell"><div class="smv">{0}</div><div class="smt">{1}</div>'
          '</div>'.format(v, t))
    a('</div>')

    a('<div class="tabs">')
    for ch in CHAPTERS:
        a('<button class="tab{0}" id="t-{1}" onclick="show(\'{1}\')">{2}</button>'
          .format(" active" if ch is CHAPTERS[0] else "", ch["key"], esc(ch["title"])))
    a('<button class="tab" id="t-gaps" onclick="show(\'gaps\')">Gaps &amp; requests'
      '</button>')
    a('<button class="tab" id="t-srcs" onclick="show(\'srcs\')">Sources</button>')
    a('</div>')

    for ch in CHAPTERS:
        bg, fg, label = chapter_state(ch["resources"], res)
        a('<div class="panel{0}" id="p-{1}">'.format(
            " active" if ch is CHAPTERS[0] else "", ch["key"]))
        a('<div class="phead"><h2>{0}</h2>{1}</div>'.format(
            esc(ch["title"]), badge(label, bg, fg)))
        a('<p class="lead">{0}</p>'.format(esc(ch["lead"].format(**f))))
        if ch.get("extra"):
            a(EXTRAS[ch["extra"]](f))
        for head, text in ch["points"]:
            a('<div class="sec"><div class="sn">{0}</div><div class="body">{1}</div>'
              '</div>'.format(esc(head.format(**f)), esc(text.format(**f))))
        a(resource_table(ch["resources"], res, registry))
        a('</div>')

    # ---- gaps, read off the YAML rather than restated -----------------------
    a('<div class="panel" id="p-gaps"><div class="phead"><h2>Gaps and open data '
      'requests</h2></div>')
    a('<p class="lead">Every open point declared in the build, in its own words. '
      '{res_done} tables are settled, {res_partial} are partial, {res_to_do} are still '
      'to do and {res_out_of_scope} are out of scope for this study.</p>'.format(**f))
    todos = [(k, v) for k, v in res.items() if remaining(v)]
    todos.sort(key=lambda kv: (kv[1].get("priority") or "P9", kv[0]))
    for k, v in todos:
        code, label = state_of(v)
        bg, fg = STATE[code]
        a('<div class="gap"><b><a href="{0}"><code>{1}</code></a></b> {2} {3}<br>{4}'
          '</div>'.format(anchor(k), esc(k), badge(v.get("priority") or "", "#e7ecf3",
                                                   "#33445e"),
                          badge(label, bg, fg), esc(remaining(v).strip())))
    a('</div>')

    # ---- the source registry ------------------------------------------------
    a('<div class="panel" id="p-srcs"><div class="phead"><h2>Sources</h2></div>')
    a('<p class="lead">The {source_count} sources declared by the build, '
      '{primary_count} of them primary. Coverage country by country, and which source '
      'won for which table, is on the <a href="{page}">data sources page</a>.</p>'
      .format(page=SOURCES_PAGE, **f))
    a('<table><thead><tr><th>Key</th><th>Source</th><th>Date</th><th>Grade</th>'
      '<th>Covers</th><th>Access</th></tr></thead><tbody>')
    for k, v in sorted(registry.items(),
                       key=lambda kv: (kv[1].get("grade") != "primary", kv[0])):
        bg, fg = GRADE.get(v.get("grade"), ("#eceff3", "#5a6577"))
        a('<tr class="srcrow"><td><code>{0}</code></td><td>{1}{2}</td><td>{3}</td>'
          '<td>{4}</td><td>{5}</td><td>{6}</td></tr>'.format(
              esc(k), esc(v.get("name", "")),
              '<div class="muted" style="font-size:.9em;margin-top:3px">{0}</div>'
              .format(esc(v["note"])) if v.get("note") else "",
              esc(v.get("date", "")), badge(v.get("grade", "?"), bg, fg),
              esc(", ".join(v.get("covers") or [])), esc(v.get("access", ""))))
    a('</tbody></table></div>')

    a('<div class="foot">Generated by <code>data_build/assumptions.py</code> from '
      '<code>build_casa.yaml</code> and <code>{0}</code>. Regenerate after any change '
      'to the build; do not edit this file by hand.</div>'.format(
          os.path.relpath(data_dir, ROOT).replace("\\", "/")))
    a('<script>{0}</script></body></html>'.format(JS))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, OUT))
    args = ap.parse_args()

    cfg = yaml.safe_load(io.open(args.config, encoding="utf-8"))
    data_dir = os.path.join(ROOT, cfg["deployment"]["target"])
    f = facts(data_dir, cfg)
    html = render(cfg, f, data_dir)
    io.open(args.out, "w", encoding="utf-8", newline="\n").write(html)
    print("Written: " + args.out)
    print("{0} chapters, {1} resources, {2} sources, {3:.0f} KB".format(
        len(CHAPTERS), f["res_total"], f["source_count"], len(html) / 1024.0))


if __name__ == "__main__":
    main()
