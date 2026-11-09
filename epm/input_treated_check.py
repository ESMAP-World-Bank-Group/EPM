"""
**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
Description:
    Consistency checks on ``input_treated.gdx``: the data exactly as GAMS
    uses it, after defaults are filled and storage merged by
    ``input_treatment.py``. Each finding states what the problem does in
    the model (e.g. "free build", "generates 0", "zero CO2") so data
    issues can be fixed before a full run.

    Complements ``input_verification.py`` (raw CSV checks run inside GAMS)
    and ``input_diagnostic.py`` (file presence / empty tables).

Usage:
    python input_treated_check.py path/to/input_treated.gdx [--csv out.csv]
    (or through ``check_inputs.py``, which first builds input_treated.gdx)

Severity:
    HIGH  changes results materially (unit free / never built / no output / no emissions)
    MED   likely wrong, check it
    LOW   informative or harmless
**********************************************************************
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEVERITY_ORDER = {"HIGH": 0, "MED": 1, "LOW": 2}

# Typical heat-rate ranges (MMBtu/MWh) by technology
HEAT_RATE_RANGE = {"CCGT": (5.5, 8.5), "OCGT": (8.0, 13.5), "ST": (8.0, 13.0), "ICE": (7.5, 11.5)}
# Plausible hours-weighted capacity factors for VRE profiles
CF_RANGE = {"PV": (0.12, 0.32), "OnshoreWind": (0.15, 0.55), "OffshoreWind": (0.20, 0.60), "CSPPlant": (0.15, 0.70)}
NON_FUEL = {"Water", "Solar", "Wind", "CSP", "Ocean", "Battery", "Uranium", "Geothermal", "Import", "Export"}
BIO_FUELS = {"Biomass", "Bagasse", "Woodchip", "Biogas"}
NON_THERMAL_TECH = {"Storage", "ReservoirHydro", "ROR", "CSPPlant", "PVwSTO", "STOPV"}


class _Findings:
    def __init__(self):
        self.rows = []

    def add(self, severity, check, item, detail, effect):
        self.rows.append(dict(severity=severity, check=check, item=str(item), detail=detail, effect=effect))

    def frame(self):
        df = pd.DataFrame(self.rows, columns=["severity", "check", "item", "detail", "effect"])
        if df.empty:
            return df
        df["_o"] = df.severity.map(SEVERITY_ORDER)
        return df.sort_values(["_o", "check", "item"]).drop(columns="_o").reset_index(drop=True)


def _records(container, name, cols):
    """Records of a GDX symbol as a plain DataFrame (string keys, no categoricals)."""
    if name not in container.data or container[name].number_records == 0:
        return pd.DataFrame(columns=cols)
    d = container[name].records.copy()
    d.columns = cols
    for c in cols[:-1]:
        d[c] = d[c].astype(str)
    return d


def _pivot(df, index, header="h", value="v"):
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index=index, columns=header, values=value, aggfunc="first", observed=True)


def _get(row, key):
    v = row.get(key, np.nan)
    return np.nan if v is None else v


def _weighted_cf(profile, hours):
    """Hours-weighted mean of a profile over all representative hours (missing hours = 0)."""
    m = hours.merge(profile[["q", "d", "t", "v"]], on=["q", "d", "t"], how="left")
    w = m.w.astype(float)
    return float((m.v.fillna(0).astype(float) * w).sum() / w.sum()) if w.sum() else np.nan


def run_checks(gdx_path):
    """Run all checks on an input_treated.gdx and return a findings DataFrame."""
    import gams.transfer as gt

    C = gt.Container(str(gdx_path))
    F = _Findings()

    years = sorted(int(float(y)) for y in _records(C, "y", ["y", "x"]).y)
    y0, yn = years[0], years[-1]
    zones = set(_records(C, "z", ["z", "x"]).z)
    zc = _records(C, "zcmap", ["z", "c", "x"]).set_index("z").c

    # ------------------------------------------------------------------ generators
    gen = _records(C, "pGenDataInput", ["g", "z", "tech", "f", "h", "v"])
    G = _pivot(gen, ["g", "z", "tech", "f"]).reset_index()
    for g in G.g[G.g.duplicated()].unique():
        F.add("HIGH", "gen: duplicate name", g, "same name on several zone/tech/fuel rows", "GAMS sums the rows: capacity, capex... doubled")
    G = G.drop_duplicates("g").set_index("g")
    G["c"] = G.z.map(zc)
    excluded = G[G.Status.isna()]
    if len(excluded):
        F.add("LOW", "gen: blank Status (excluded)", f"{len(excluded)} units",
              ", ".join(excluded.index[:8]) + ("..." if len(excluded) > 8 else ""), "ignored by the model")
    G = G[G.Status.notna()]

    tf = _pivot(_records(C, "pTechFuel", ["tech", "f", "h", "v"]), ["tech", "f"])
    vre_tech = {t for (t, _), r in tf.iterrows() if r.get("HourlyVariation", 0) == 1} if not tf.empty else set()
    carbon = _records(C, "pFuelCarbonContent", ["f", "v"]).set_index("f").v
    fuel_price = _records(C, "pFuelPrice", ["c", "f", "y", "v"])
    fuel_price["y"] = fuel_price.y.astype(float).astype(int)

    is_thermal = (~G.tech.isin(vre_tech | NON_THERMAL_TECH) & ~G.f.isin(NON_FUEL)
                  & ~G.f.str.startswith("Import") & ~G.f.str.startswith("DSM") & (G.tech != "Nuclear"))
    thermal = G[is_thermal]

    for g, r in G.iterrows():
        cap, sy, ry, status = _get(r, "Capacity"), _get(r, "StYr"), _get(r, "RetrYr"), r.Status
        label = f"{r.z} {r.tech}-{r.f} (status {status:.0f})"
        if status not in (1, 2, 3):
            F.add("HIGH", "gen: invalid Status", g, f"Status={status}", "unit ignored")
        if pd.isna(cap) or cap <= 0:
            sev = "HIGH" if status in (2, 3) else "MED"
            F.add(sev, "gen: no Capacity", g, label, "vCap.up = 0: unit can never exist")
            continue
        if pd.isna(sy):
            F.add("MED", "gen: no StYr", g, label, "StYr = 0: available from the first year")
        if not pd.isna(sy) and not pd.isna(ry) and ry <= sy:
            F.add("LOW", "gen: RetrYr <= StYr", g, f"StYr {sy:.0f}, RetrYr {ry:.0f}", "unit never online (parked entry?)")
        if status in (2, 3) and not pd.isna(sy) and sy > yn:
            F.add("LOW", "gen: StYr after horizon", g, f"StYr {sy:.0f} > {yn}", "can never be built")
        if status == 1 and not pd.isna(ry) and ry < y0:
            F.add("LOW", "gen: retired before horizon", g, f"RetrYr {ry:.0f}", "existing unit never online")
        if status == 1 and not pd.isna(sy) and sy > y0:
            F.add("MED", "gen: existing with future StYr", g, f"Status 1, StYr {sy:.0f} > {y0}",
                  "treated as a build up to Capacity; should probably be status 2")
        if status in (2, 3) and r.tech != "Storage":
            if pd.isna(_get(r, "Capex")) or _get(r, "Capex") <= 0:
                F.add("HIGH" if status == 3 else "MED", "gen: no Capex", g, label,
                      "built for free" if status == 3 else "investment cost understated")
            if pd.isna(_get(r, "Life")) or _get(r, "Life") <= 0:
                F.add("HIGH" if status == 3 else "MED", "gen: no Life", g, label, "capital recovery factor = 0: capex annuity = 0")
        if status == 3:
            bl = _get(r, "BuildLimitperYear")
            if pd.isna(bl) or bl <= 0:
                F.add("HIGH", "gen: candidate without BuildLimitperYear", g, f"{label}, {cap:.0f} MW",
                      "vBuild.up = 0: candidate can never be built")
            elif not pd.isna(sy):
                n = yn - max(sy, y0) + 1
                if 0 < n and bl * n < 0.99 * cap:
                    F.add("LOW", "gen: potential unreachable at build rate", g,
                          f"{r.tech}-{r.f} {r.z}: {bl:.0f} MW/yr x {n:.0f} yrs = {bl * n:.0f} < {cap:.0f} MW",
                          "BuildLimitperYear, not Capacity, is the effective limit")
        mg = _get(r, "MinGenShareAllHours")
        if not pd.isna(mg) and mg > 0.8:
            F.add("LOW", "gen: high MinGenShareAllHours", g, f"{mg}", "near must-run every hour (if fApplyMinGenShareAllHours=1)")

    for g, r in thermal.iterrows():
        cap = _get(r, "Capacity")
        if pd.isna(cap) or cap <= 0:
            continue
        hr = _get(r, "HeatRate")
        label = f"{r.z} {r.tech}-{r.f} (status {r.Status:.0f}), {cap:.0f} MW"
        if pd.isna(hr) or hr <= 0:
            F.add("HIGH" if cap >= 5 else "LOW", "gen: thermal without HeatRate", g, label,
                  "no fuel use: zero fuel cost and zero CO2")
        else:
            lo, hi = HEAT_RATE_RANGE.get(r.tech, (5, 15))
            if not lo <= hr <= hi:
                F.add("MED", "gen: HeatRate out of typical range", g, f"{r.tech}-{r.f} HeatRate {hr} (typical {lo}-{hi})",
                      "fuel cost and emissions scaled accordingly")
        if r.f not in carbon.index and r.f not in BIO_FUELS:
            F.add("HIGH", "gen: fossil fuel without carbon content", g, f"fuel {r.f}", "unit emits 0 tCO2")
        fp = fuel_price[(fuel_price.c == r.c) & (fuel_price.f == r.f)]
        if fp.empty or (fp.v <= 0).all():
            F.add("MED" if r.f in BIO_FUELS else "HIGH", "gen: no fuel price", g, f"{r.c} / {r.f}",
                  "fuel cost = 0: dispatched on VOM only")
        else:
            missing = sorted(set(years) - set(fp.y))
            if missing:
                F.add("MED", "gen: fuel price missing some years", g, f"{r.c}/{r.f}: {missing[:6]}{'...' if len(missing) > 6 else ''}",
                      "fuel cost = 0 in those years")

    capex_ok = G[G.Status.isin([2, 3]) & (G.tech != "Storage") & (G.Capex > 0)]
    for (t, f), grp in capex_ok.groupby(["tech", "f"]):
        med = grp.Capex.median()
        for g, r in grp.iterrows():
            if r.Capex > 20:
                F.add("HIGH", "gen: Capex implausible", g, f"{t}-{f} Capex {r.Capex} $m/MW", "likely $/kW entered instead of $m/MW")
            elif len(grp) >= 3 and (r.Capex > 3 * med or r.Capex < med / 3):
                F.add("MED", "gen: Capex outlier", g, f"{t}-{f} Capex {r.Capex} vs median {med:.2f} $m/MW", "possible unit error")

    # ------------------------------------------------------------------ availability & VRE profiles
    av = _records(C, "pAvailability", ["g", "y", "q", "v"])
    av_stats = av.groupby("g").v.agg(["max", "mean"]) if not av.empty else pd.DataFrame(columns=["max", "mean"])
    built = G[G.Capacity > 0]
    for g, r in built[~built.tech.isin({"Storage"})].iterrows():
        if g not in av_stats.index or av_stats.at[g, "max"] <= 0:
            F.add("HIGH", "avail: no availability", g, f"{r.z} {r.tech}-{r.f} (status {r.Status:.0f})", "eMaxCF forces output to 0")
        elif av_stats.at[g, "max"] > 1.0001:
            F.add("HIGH", "avail: above 1", g, f"max {av_stats.at[g, 'max']:.2f}", "output above installed capacity")
        elif r.tech in ("CCGT", "OCGT", "ST", "ICE", "Nuclear") and av_stats.at[g, "mean"] < 0.3:
            F.add("LOW", "avail: very low for thermal", g, f"{r.tech} mean {av_stats.at[g, 'mean']:.2f}", "unit mostly unavailable (intended?)")

    hours = _records(C, "pHours", ["q", "d", "t", "w"])
    plant_prof = _records(C, "pVREgenProfile", ["g", "q", "d", "t", "v"])
    zone_prof = _records(C, "pVREProfile", ["z", "tech", "q", "d", "t", "v"])
    own = set(plant_prof.g)
    zone_cf = {k: _weighted_cf(grp, hours) for k, grp in zone_prof.groupby(["z", "tech"])} if not hours.empty else {}
    for g, r in built[built.tech.isin(vre_tech)].iterrows():
        if g in own:
            cf = _weighted_cf(plant_prof[plant_prof.g == g], hours)
            src = "plant profile"
        elif (r.z, r.tech) in zone_cf:
            cf = zone_cf[(r.z, r.tech)]
            src = f"zone profile {r.z}/{r.tech}"
        else:
            F.add("HIGH", "vre: no profile", g, f"{r.z} {r.tech}, {r.Capacity:.0f} MW", "no plant or zone profile: generates 0")
            continue
        lo, hi = CF_RANGE.get(r.tech, (0.05, 0.95))
        if cf < 0.10 and r.tech in CF_RANGE:
            F.add("HIGH", "vre: capacity factor < 10%", g, f"{r.tech} {r.z}, {r.Capacity:.0f} MW, CF {cf:.3f} ({src})",
                  "unit almost useless: never built / understated output")
        elif not lo <= cf <= hi:
            F.add("MED", "vre: capacity factor unusual", g, f"{r.tech} {r.z}, CF {cf:.3f} (typical {lo}-{hi}, {src})", "check profile data")
    for nm, prof, keys in (("plant", plant_prof, ["g"]), ("zone", zone_prof, ["z", "tech"])):
        if not prof.empty:
            mx = prof.groupby(keys).v.max()
            for k, v in mx[mx > 1.0001].items():
                F.add("HIGH", f"vre: {nm} profile above 1", k, f"max {v:.2f}", "output above installed capacity")

    # ------------------------------------------------------------------ storage
    sto = _pivot(_records(C, "pStorageDataInput", ["g", "z", "tech", "f", "h", "v"]), ["g", "z", "tech", "f"])
    if not sto.empty:
        sto = sto.reset_index().set_index("g")
        sto = sto[sto.Status.notna()]
        for g, r in sto.iterrows():
            cap, status = _get(r, "Capacity"), r.Status
            if pd.isna(cap) or cap <= 0:
                F.add("HIGH" if status in (2, 3) else "MED", "sto: no Capacity", g, f"{r.z} (status {status:.0f})", "storage can never exist")
                continue
            if status == 3 and (pd.isna(_get(r, "BuildLimitperYear")) or _get(r, "BuildLimitperYear") <= 0):
                F.add("HIGH", "sto: candidate without BuildLimitperYear", g, r.z, "never built")
            if status in (2, 3):
                if pd.isna(_get(r, "CapexMWh")) or _get(r, "CapexMWh") <= 0:
                    F.add("HIGH" if status == 3 else "MED", "sto: no CapexMWh", g, f"{r.z} (status {status:.0f})", "energy capacity built for free")
                if pd.isna(_get(r, "Capex")) or _get(r, "Capex") <= 0:
                    F.add("MED", "sto: no Capex (power)", g, f"{r.z} (status {status:.0f})", "power capacity built for free")
            eff = _get(r, "Efficiency")
            if pd.isna(eff) or not 0.5 <= eff <= 1:
                F.add("MED", "sto: Efficiency missing or odd", g, f"{eff}", "round-trip losses wrong")
            mwh = _get(r, "CapacityMWh")
            if not pd.isna(mwh) and mwh > 0 and not 0.5 <= mwh / cap <= 24:
                F.add("MED", "sto: duration odd", g, f"{mwh / cap:.1f} h", "check MW vs MWh")

    # ------------------------------------------------------------------ fuels
    for (c, f), grp in fuel_price.groupby(["c", "f"]):
        s = grp.set_index("y").v.sort_index()
        if (s < 0).any():
            F.add("HIGH", "fuel: negative price", f"{c}/{f}", f"min {s.min()}", "units paid to burn fuel")
        jump = s.pct_change().abs()
        if (jump > 0.5).any():
            yy = jump.idxmax()
            F.add("MED", "fuel: price jump > 50%", f"{c}/{f}", f"{yy}: {s.shift().get(yy):.2f} -> {s.get(yy):.2f}", "check for typo")
        if (s > 50).any():
            F.add("MED", "fuel: price > 50 $/MMBtu", f"{c}/{f}", f"max {s.max():.1f}", "possible unit error")

    # ------------------------------------------------------------------ demand
    dem = _records(C, "pDemandForecast", ["z", "pe", "y", "v"])
    dem["y"] = dem.y.astype(float).astype(int)
    D = dem.pivot_table(index=["z", "y"], columns="pe", values="v", observed=True)
    for z in sorted(zones - set(dem.z)):
        F.add("MED", "demand: zone without demand", z, "", "no load in this zone (fine for supply-only zones)")
    for z, grp in D.groupby(level="z"):
        grp = grp.droplevel("z")
        missing = sorted(set(years) - set(grp.index))
        if missing:
            F.add("HIGH", "demand: missing years", z, f"{missing[:6]}", "demand = 0 in those years")
        if {"energy", "peak"} <= set(grp.columns):
            active = grp[grp.energy > 1]   # ~0 values = zone switched off (e.g. plant closure scenario)
            if len(active) < len(grp):
                F.add("LOW", "demand: ~zero demand some years", z, f"{len(grp) - len(active)} yrs (first {grp.index[grp.energy <= 1].min()})",
                      "zone load switched off (intended?)")
            lf = active.energy * 1e3 / (active.peak * 8760)
            bad = lf[(lf < 0.3) | (lf > 0.98)]
            if len(bad):
                F.add("MED", "demand: load factor unusual", z, f"{len(bad)} yrs, {lf.min():.2f}-{lf.max():.2f} (energy GWh / peak MW)", "check units or peak")
            growth = active.energy.pct_change().abs()
            if (growth > 0.25).any():
                yy = growth.idxmax()
                F.add("MED", "demand: energy step > 25%/yr", z, f"{yy}: {active.energy.shift().get(yy):.0f} -> {active.energy.get(yy):.0f} GWh",
                      "check step change")
    prof = _records(C, "pDemandProfile", ["z", "q", "d", "t", "v"])
    if not prof.empty:
        for z, m in prof.groupby("z").v.max().items():
            if abs(m - 1) > 0.02:
                F.add("MED", "demand: profile max != 1", z, f"max {m:.3f}", "model peak differs from forecast peak")
        for z in sorted(set(dem.z) - set(prof.z)):
            F.add("HIGH", "demand: no hourly profile", z, "", "load not distributed over hours")

    # ------------------------------------------------------------------ transmission & trade
    tl = _records(C, "pTransferLimit", ["z", "z2", "q", "y", "v"])
    limit = tl.groupby(["z", "z2"]).v.max() if not tl.empty else pd.Series(dtype=float)
    for (a, b), v in limit.items():
        if (b, a) not in limit.index:
            F.add("LOW", "trans: one-directional limit", f"{a}->{b}", f"{v:.0f} MW, reverse missing", "no flow in reverse direction")
    nt = _pivot(_records(C, "pNewTransmission", ["z", "z2", "ts", "h", "v"]), ["z", "z2", "ts"])
    if not nt.empty:
        nt = nt.reset_index()
        for _, r in nt.iterrows():
            name = f"{r.z}-{r.z2} ({r.ts})"
            if r.z not in zones or r.z2 not in zones:
                F.add("HIGH", "trans: unknown zone", name, "", "line ignored")
            if pd.isna(_get(r, "Status")):
                continue
            for col, effect in (("CapacityPerLine", "building the line adds 0 MW"), ("CostPerLine", "line built for free"),
                                ("MaximumNumOfLines", "line can never be built"), ("Life", "annuity = 0: line built for free")):
                if col in nt.columns and (pd.isna(r[col]) or r[col] <= 0):
                    F.add("HIGH", f"trans: new line without {col}", name, f"status {r.Status:.0f}", effect)
    mi = _records(C, "pMinImport", ["z2", "z", "y", "v"])
    for (src, dst), grp in mi.groupby(["z2", "z"]):
        # eMinImportRequirement: vFlow(z2 -> z) >= pMinImport(z2, z, y)
        exist = limit.get((src, dst), 0)
        committed = 0
        if not nt.empty:
            same = nt[((nt.z == src) & (nt.z2 == dst)) | ((nt.z == dst) & (nt.z2 == src))]
            same = same[same.Status == 2]
            committed = float((same.CapacityPerLine.fillna(0) * same.MaximumNumOfLines.fillna(0)).sum())
        need = grp.v.max()
        if need > exist + committed + 1e-6:
            F.add("HIGH", "trade: min import above transfer capacity", f"{src}->{dst}",
                  f"min {need:.0f} MW > limit {exist:.0f} + committed lines {committed:.0f}", "infeasible or slack")
        elif need >= 0.95 * (exist + committed) and need > 0:
            F.add("LOW", "trade: min import uses ~all capacity", f"{src}->{dst}",
                  f"min {need:.0f} MW vs limit {exist:.0f} (+{committed:.0f} committed)", "line forced full every hour")

    # ------------------------------------------------------------------ policy & settings
    mg = _records(C, "pMinGenByFuel", ["c", "tech", "f", "y", "v"])
    for (c, t, f), grp in mg.groupby(["c", "tech", "f"]):
        capc = G[(G.c == c) & (G.tech == t) & (G.f == f)].Capacity.sum()
        avg_mw = grp.v.max() / 8760
        F.add("MED" if capc and avg_mw > 0.5 * capc else "LOW", "policy: minimum generation by fuel", f"{c} {t}-{f}",
              f"{grp.v.max() / 1e6:.2f} TWh/yr = {avg_mw:.0f} MW average on {capc:.0f} MW installed",
              "forced generation (and emissions) every year")
    settings = _records(C, "pSettings", ["k", "v"]).set_index("k").v
    if settings.get("fApplySystemCo2Constraint", 0) and C["pEmissionsTotal"].number_records == 0:
        F.add("HIGH", "policy: system CO2 cap on without pEmissionsTotal", "pSettings", "", "constraint has no target")
    if settings.get("fApplyCountryCo2Constraint", 0) and C["pEmissionsCountry"].number_records == 0:
        F.add("HIGH", "policy: country CO2 cap on without pEmissionsCountry", "pSettings", "", "constraint has no target")
    if settings.get("fEnableCarbonPrice", 0) and C["pCarbonPrice"].number_records == 0:
        F.add("HIGH", "policy: carbon price on without pCarbonPrice", "pSettings", "", "carbon price = 0")

    # ------------------------------------------------------------------ year coverage
    _check_year_coverage(C, F, years, settings)

    return F.frame()


# Year-indexed inputs that input_treatment.py does NOT extend to all model years
# (only YEARLY_OUTPUT there is interpolated and held flat). A year with no value is
# read as 0 by GAMS. For each: (setting that activates it or None, effect of the gap).
YEAR_COVERAGE_EFFECTS = {
    "pCarbonPrice": ("fEnableCarbonPrice", "carbon price = 0 in those years"),
    "pSpinningReserveReqCountry": ("fApplyCountrySpinReserveConstraint", "no country spinning reserve requirement in those years"),
    "pSpinningReserveReqSystem": ("fApplySystemSpinReserveConstraint", "no system spinning reserve requirement in those years"),
    "pEmissionsCountry": ("fApplyCountryCo2Constraint", "country CO2 cap = 0 in those years (backstop used)"),
    "pEmissionsTotal": ("fApplySystemCo2Constraint", "system CO2 cap = 0 in those years (backstop used)"),
    "pMaxFuellimit": ("fApplyFuelConstraint", "no fuel limit in those years"),
    "pMaxAnnualExternalTradeShare": ("fEnableExternalExchange", "no external imports allowed in those years"),
    "pEnergyEfficiencyFactor": ("fEnableEnergyEfficiency", "factor reset to 1 in those years"),
    "pLossFactorInternal": (None, "no transmission losses in those years"),
}
# Derived or sparse by design, plus the parameters input_treatment.py already extends to all
# model years (YEARLY_OUTPUT there): a gap left in those is a deliberate zero, not missing data.
YEAR_COVERAGE_SKIP = {"pAvailability", "pDemandData", "pEvolutionAvailability", "pTransferOutage",
                      "pCapexTrajectoriesDefault", "pCapexTrajectoriesGeneric",
                      "pTradePrice", "pDemandForecast", "pCapexTrajectories", "pExtTransferLimit",
                      "pTransferLimit", "pFuelPrice", "pMaxGenerationByFuel"}


def _check_year_coverage(C, F, years, settings):
    """Flag year-indexed inputs whose data stops before the last model year."""
    import gams.transfer as gt

    last = years[-1]
    for name, sym in C.data.items():
        if (not isinstance(sym, gt.Parameter) or "y" not in sym.domain_names
                or name in YEAR_COVERAGE_SKIP or sym.number_records == 0):
            continue
        d = sym.records.copy()
        y_col = d.columns[sym.domain_names.index("y")]
        d["_y"] = pd.to_numeric(d[y_col].astype(str), errors="coerce")
        d = d.dropna(subset=["_y"])
        if d.empty:
            continue
        keys = [c for c, dn in zip(d.columns, sym.domain_names) if dn not in ("y", "q", "d", "t")]
        spans = d.groupby(keys, observed=True)["_y"].agg(["min", "max"]) if keys else \
            pd.DataFrame({"min": [d._y.min()], "max": [d._y.max()]}, index=["all"])
        short_end = spans[spans["max"] < last]
        if short_end.empty:
            continue
        flag, effect = YEAR_COVERAGE_EFFECTS.get(name, (None, "value read as 0 in those years"))
        active = flag is None or bool(settings.get(flag, 0))
        if name in YEAR_COVERAGE_EFFECTS:
            severity = ("HIGH" if flag else "MED") if active else "LOW"
        else:
            severity = "MED" if active else "LOW"
        if not active:
            effect = f"no effect while {flag} = 0 ({effect} if switched on)"
        stop = int(short_end["max"].max()) if short_end["max"].nunique() == 1 else \
            f"{int(short_end['max'].min())}-{int(short_end['max'].max())}"
        examples = ", ".join("/".join(map(str, k)) if isinstance(k, tuple) else str(k) for k in short_end.index[:4])
        more = f" (+{len(short_end) - 4} more)" if len(short_end) > 4 else ""
        F.add(severity, "years: data stops before last model year", name,
              f"{len(short_end)} of {len(spans)} entries end in {stop}, model runs to {last}: {examples}{more}",
              effect + " (or 0 on purpose: check)")


def format_report(df, title="", max_rows=25):
    """Human-readable report of a findings frame (HIGH and MED in detail, LOW counted)."""
    lines = []
    if title:
        lines += ["=" * 100, title, "=" * 100]
    if df.empty:
        return "\n".join(lines + ["No findings."])
    counts = df.severity.value_counts()
    lines.append("Findings: " + ", ".join(f"{s} {counts.get(s, 0)}" for s in SEVERITY_ORDER))
    for sev in ("HIGH", "MED"):
        for check, grp in df[df.severity == sev].groupby("check", sort=False):
            same_effect = grp.effect.nunique() == 1
            lines.append(f"\n[{sev}] {check} ({len(grp)})" + (f" -> {grp.effect.iloc[0]}" if same_effect else ""))
            for _, r in grp.head(max_rows).iterrows():
                lines.append(f"    {r['item']:<45} {r.detail}")
                if not same_effect:
                    lines.append(f"    {'':<45} -> {r.effect}")
            if len(grp) > max_rows:
                lines.append(f"    ... {len(grp) - max_rows} more (see CSV)")
    low = df[df.severity == "LOW"].groupby("check").size()
    if len(low):
        lines.append("\n[LOW] " + "; ".join(f"{k} ({v})" for k, v in low.items()))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Consistency checks on an EPM input_treated.gdx")
    parser.add_argument("gdx", help="Path to input_treated.gdx")
    parser.add_argument("--csv", default=None, help="Write findings to this CSV (default: next to the GDX)")
    args = parser.parse_args(argv)
    gdx = Path(args.gdx)
    df = run_checks(gdx)
    out = Path(args.csv) if args.csv else gdx.with_name("input_check.csv")
    df.to_csv(out, index=False)
    print(format_report(df, title=f"Input check: {gdx}"))
    print(f"\nFull list: {out}")
    return 1 if (df.severity == "HIGH").any() else 0


if __name__ == "__main__":
    sys.exit(main())
