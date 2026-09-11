"""Build the CESI_Full input files of a deployment from cesi/cesi_register.yaml.

Usage:
    python pre-analysis/catalog/build_cesi_full.py --deployment data_blacksea
    python pre-analysis/catalog/build_cesi_full.py --deployment data_blacksea --dry-run

The fully aligned scenarios (CESI_Full with the corridor, CESI_NoCorr without) read
copies of our base files in which Azerbaijan, Georgia and the two GEC zones carry the
CESI GEC feasibility study assumptions. This script writes those copies under
<deployment>/cesi/<param>_cesi.csv. It holds no CESI figure: every value comes from
cesi/cesi_register.yaml (DVC only, client confidential) at run time, and this file may
be tracked by git. Register ids are cited in the comments so the log can be read against
the register.

What is built, and how a CESI value becomes an EPM input:

  pDemandForecast   Annual energy and peak of AzerbaijanMain and Georgia. CESI anchors
                    (a01, a02, g01, g02) are interpolated year by year from our own value of
                    the first model year, and extended beyond the last anchor with our own
                    growth. Electrolysers (a04, a05, g04, g05) are added as a flat load: the
                    hydrogen energy at 50 kWh per kg, and its average power on the peak.
                    Azerbaijan anchors cover the whole country, so AzerbaijanMain gets the
                    anchor minus our Nakhchivan forecast, which is left unchanged.
  pGenDataInput     Fleet of AzerbaijanMain and Georgia imposed: for PV, onshore wind,
                    offshore wind, gas and hydro, a committed tranche (Status 2) fills the gap
                    between our existing and committed units and the CESI capacity at each
                    anchor year (a06, a09, a10, g06, g09, g11); a CESI level below the fleet
                    retires the latest tranche, never an existing unit. Every candidate
                    (Status 3) of the two zones and of the GEC_AZ hub is dropped. The hubs are
                    committed tranches too: GEC_AZ at the CESI steps (a07), GEC_GE in equal
                    thirds at the three link commissioning years (g07, the study gives the
                    2040 total only).
  pStorageDataInput Candidates of the two zones dropped, one committed BESS per zone at the
                    CESI capacity from the first anchor year (a17, g14), costs of our candidate.
  pGenDataInputDefault
                    VRE capex, FOM as a share of capex, and life, for AzerbaijanMain, Georgia
                    and the two GEC zones (a11, a12, c02). GEC_GE rows are copies of Georgia.
  pCapexTrajectoriesDefault
                    VRE multipliers set to 1 in the same four zones, so the CESI capex applies
                    flat. GEC_GE rows are copies of Georgia.
  pSettings         WACC and discount rate (c01).
  pTradePriceExport Romania rows rebuilt with the CESI cable loss (k05): the seller side
                    deduction is refitted from our import and export files, so W and the
                    price level stay ours and only lambda changes.
  pVREProfile       PV and wind profiles of AzerbaijanMain and Georgia rescaled to the CESI
                    mean capacity factors (a13 to a15, g12, g13), clipped at 1, weighted by
                    pHours. GEC_AZ and GEC_GE carry the rescaled national profiles.

The import price file is not copied: the loss sits on the seller, so the buy side is
unchanged. pExtTransferLimit is not copied either, the GECO link files already carry
the corridor (k01, k02). Scenario columns CESI_Full and CESI_NoCorr in scenarios.csv
point at these files.
"""
from __future__ import annotations

import argparse
import codecs
import csv
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = REPO_ROOT / "epm" / "input"

# Our own conversions, none of them a CESI figure.
EUR_USD = 1.08            # as in the register conversions
H2_KWH_PER_KG = 50.0      # electrolyser consumption, ours
BESS_HOURS = 4            # duration of the committed BESS, as our candidates
RETIRE_YEAR = 2060        # tranches never retire inside the horizon, as our existing units
HUB_GE_STEPS = 3          # GEC_GE hub built in equal steps, one per GEC link

# Fleet groups imposed by the study, with the tech and fuel of the tranche that fills a gap.
GROUPS = {
    "PV": (lambda t, f: t == "PV", "PV", "Solar"),
    "OnshoreWind": (lambda t, f: t == "OnshoreWind", "OnshoreWind", "Wind"),
    "OffshoreWind": (lambda t, f: t == "OffshoreWind", "OffshoreWind", "Wind"),
    "Gas": (lambda t, f: f == "Gas", "CCGT", "Gas"),
    "Hydro": (lambda t, f: f == "Water" and t != "Storage", "ReservoirHydro", "Water"),
}
VRE = ("PV", "OnshoreWind", "OffshoreWind")
VRE_FUEL = {"PV": "Solar", "OnshoreWind": "Wind", "OffshoreWind": "Wind"}
ZONES = {"AzerbaijanMain": "a", "Georgia": "g"}      # register id prefix per zone
HUB = {"AzerbaijanMain": "GEC_AZ", "Georgia": "GEC_GE"}
EXCLAVE = "Nakhchivan"                                # inside the Azerbaijan anchors, kept as ours
EXT_ZONE = "Romania"                                  # end of the subsea link
LINK_YEARS_ID = "k01_gec_capacity"                    # commissioning years of the three links


# ---------------------------------------------------------------------------
# Small CSV helpers. Files are written back with the header, BOM and column
# order and line endings of the base file, so a diff against the base is readable.
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> tuple[list[str], list[dict], tuple[bool, bool]]:
    """Header, rows, and the (bom, crlf) style of the file so the copy can match it."""
    raw = path.read_bytes()
    style = (raw.startswith(codecs.BOM_UTF8), b"\r\n" in raw)
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    header = list(rows[0].keys()) if rows else []
    return header, rows, style


def write_csv(path: Path, header: list[str], rows: list[dict], style: tuple[bool, bool], dry: bool) -> None:
    if dry:
        print(f"  [dry-run] would write {path.name}: {len(rows)} rows")
        return
    bom, crlf = style
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator="\r\n" if crlf else "\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in header})
    print(f"  wrote {path.relative_to(INPUT_ROOT).as_posix()}: {len(rows)} rows")


def hour_cols(header: list[str]) -> list[str]:
    return [c for c in header if len(c) == 3 and c[0] == "t" and c[1:].isdigit()]


def num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def fmt(v: float, nd: int = 3) -> str:
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def interp(anchors: dict[int, float], year: int) -> float:
    """Piecewise linear between anchor years, flat outside."""
    ys = sorted(anchors)
    if year <= ys[0]:
        return anchors[ys[0]]
    if year >= ys[-1]:
        return anchors[ys[-1]]
    for a, b in zip(ys, ys[1:]):
        if a <= year <= b:
            return anchors[a] + (anchors[b] - anchors[a]) * (year - a) / (b - a)
    raise AssertionError


# ---------------------------------------------------------------------------
# Register and deployment
# ---------------------------------------------------------------------------

def load_register(dep: Path) -> dict:
    path = dep / "cesi" / "cesi_register.yaml"
    if not path.exists():
        sys.exit(f"{path} is absent (DVC only, run dvc pull -r r2)")
    entries = yaml.safe_load(path.read_text(encoding="utf-8-sig")).get("entries") or []
    return {e["id"]: e for e in entries}


def load_config(dep: Path) -> dict[str, Path]:
    with open(dep / "config.csv", encoding="utf-8-sig", newline="") as fh:
        return {r["paramNames"].strip(): dep / r["file"].strip()
                for r in csv.DictReader(fh) if (r.get("paramNames") or "").strip() and (r.get("file") or "").strip()}


def reg_val(reg: dict, rid: str):
    if rid not in reg:
        sys.exit(f"register entry {rid} is missing")
    return reg[rid]["cesi_value"]


def year_dict(d: dict) -> dict[int, float]:
    return {int(k): float(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# pDemandForecast
# ---------------------------------------------------------------------------

def build_demand(dep: Path, reg: dict, cfg: dict, out: Path, dry: bool) -> None:
    header, rows, bom = read_csv(cfg["pDemandForecast"])
    years = [int(c) for c in header[2:]]
    y0 = years[0]
    by = {(r["z"], r["type"]): r for r in rows}

    def series(row: dict) -> dict[int, float]:
        return {y: num(row[str(y)], 0.0) for y in years}

    def align(base: dict[int, float], anchors: dict[int, float]) -> dict[int, float]:
        """Our first year, linear to the first anchor, the anchors, then our growth beyond the last."""
        first, last = min(anchors), max(anchors)
        # The first model year of the run is the second column of the file: the base year
        # of the forecast stays as it is, the path bends from the year after it.
        start = years[1] if len(years) > 1 else y0
        outp = {}
        for y in years:
            if y <= start:
                outp[y] = base[y]
            elif y < first:
                outp[y] = base[start] + (anchors[first] - base[start]) * (y - start) / (first - start)
            elif y <= last:
                outp[y] = interp(anchors, y)
            else:
                outp[y] = anchors[last] * base[y] / base[last] if base[last] else anchors[last]
        return outp

    def h2_load(p: str) -> tuple[dict[int, float], dict[int, float]]:
        """Electrolyser load as a flat block: GWh per year, and its average MW on the peak."""
        kt = year_dict(reg_val(reg, f"{p}04_h2_production"))
        gwh_anchor = {y: v * H2_KWH_PER_KG for y, v in kt.items()}   # 1 kt at 50 kWh/kg = 50 GWh
        first = min(gwh_anchor)
        ramp = dict(gwh_anchor)
        ramp.setdefault(years[1], 0.0)          # nothing before the first model year, linear ramp to the first anchor
        gwh = {y: (interp(ramp, y) if y >= years[1] else 0.0) for y in years}
        mw = {y: gwh[y] / 8.76 for y in years}
        return gwh, mw

    log = []
    for zone, p in ZONES.items():
        e_anchor = {y: v * 1000.0 for y, v in year_dict(reg_val(reg, f"{p}01_dem_energy")).items()}   # TWh to GWh
        p_anchor = year_dict(reg_val(reg, f"{p}02_dem_peak"))
        e_row, p_row = by[(zone, "Energy")], by[(zone, "Peak")]
        e_base, p_base = series(e_row), series(p_row)
        e_excl = p_excl = None
        if zone == "AzerbaijanMain":
            # the study counts the whole country: our exclave forecast is taken out of the anchor
            e_excl, p_excl = series(by[(EXCLAVE, "Energy")]), series(by[(EXCLAVE, "Peak")])
            e_base = {y: e_base[y] + e_excl[y] for y in years}
            p_base = {y: p_base[y] + p_excl[y] for y in years}
        e_new, p_new = align(e_base, e_anchor), align(p_base, p_anchor)
        h2_gwh, h2_mw = h2_load(p)
        for y in years:
            e = e_new[y] + h2_gwh[y] - (e_excl[y] if e_excl else 0.0)
            pk = p_new[y] + h2_mw[y] - (p_excl[y] if p_excl else 0.0)
            e_row[str(y)], p_row[str(y)] = fmt(e, 2), fmt(pk, 2)
        for y in sorted(set(e_anchor) | set(p_anchor)):
            log.append(f"    {zone} {y}: energy {e_row[str(y)]} GWh (h2 {h2_gwh[y]:.0f}), "
                       f"peak {p_row[str(y)]} MW (h2 {h2_mw[y]:.0f})")
    print("  demand anchors as written (exclave removed where it applies):")
    print("\n".join(log))
    write_csv(out / "pDemandForecast_cesi.csv", header, rows, bom, dry)


# ---------------------------------------------------------------------------
# pGenDataInput and pStorageDataInput
# ---------------------------------------------------------------------------

def link_years(reg: dict) -> list[int]:
    v = reg_val(reg, LINK_YEARS_ID)
    if isinstance(v, dict):
        ys = sorted(int(k) for k in v)
    else:
        ys = sorted(int(k) for k in (reg[LINK_YEARS_ID].get("detail") or {}).get("years", {}))
    if not ys:
        sys.exit(f"{LINK_YEARS_ID}: no commissioning years found")
    return ys


def in_service(r: dict, y: int) -> bool:
    st, rt = num(r.get("StYr")), num(r.get("RetrYr"))
    return (st is None or st <= y) and (rt is None or rt > y)


def tranche(name: str, zone: str, tech: str, fuel: str, y: int, mw: float, header: list[str]) -> dict:
    r = {k: "" for k in header}
    r.update({"g": name, "z": zone, "tech": tech, "f": fuel, "Status": "2", "StYr": str(y),
              "RetrYr": str(RETIRE_YEAR), "Capacity": fmt(mw, 1), "BuildLimitperYear": fmt(mw, 1)})
    return r


def impose(rows: list[dict], zone: str, group: str, targets: dict[int, float], header: list[str], log: list) -> list[dict]:
    """Committed tranches so that the (zone, group) fleet meets the CESI level at each anchor year."""
    match, tech, fuel = GROUPS[group]
    added: list[dict] = []
    for y in sorted(targets):
        target = targets[y] * 1000.0
        fleet = [r for r in rows + added if r["z"] == zone and match(r["tech"], r["f"])
                 and r["Status"] in ("1", "2") and in_service(r, y)]
        have = sum(num(r["Capacity"], 0.0) for r in fleet)
        gap = target - have
        if gap > 0.5:
            added.append(tranche(f"{zone}_CESI_{tech}_{y}", zone, tech, fuel, y, gap, header))
            log.append(f"    {zone} {group} {y}: fleet {have:.0f} -> +{gap:.0f} MW tranche")
        elif gap < -0.5:
            # retire the latest tranches first, splitting one if needed; existing units are never touched
            excess = -gap
            for t in sorted((t for t in added if in_service(t, y)), key=lambda t: -int(t["StYr"])):
                cap = num(t["Capacity"])
                if cap <= excess + 1e-9:
                    t["RetrYr"] = str(y)
                    excess -= cap
                else:
                    part = dict(t, g=t["g"] + f"_to{y}", Capacity=fmt(excess, 1), BuildLimitperYear=fmt(excess, 1), RetrYr=str(y))
                    t["Capacity"] = t["BuildLimitperYear"] = fmt(cap - excess, 1)
                    added.append(part)
                    excess = 0.0
                if excess <= 1e-9:
                    break
            log.append(f"    {zone} {group} {y}: fleet {have:.0f} -> -{-gap:.0f} MW"
                       + (f", {excess:.0f} MW above the study kept (existing units)" if excess > 0.5 else " retired from tranches"))
        else:
            log.append(f"    {zone} {group} {y}: fleet {have:.0f} MW already at the study level")
    return added


def build_gendata(dep: Path, reg: dict, cfg: dict, out: Path, dry: bool) -> None:
    header, rows, bom = read_csv(cfg["pGenDataInput"])
    hubs = set(HUB.values())
    closed = [r for r in rows if r["z"] in set(ZONES) | hubs and r["Status"] == "3"]
    kept = [r for r in rows if r not in closed]
    print(f"  candidates dropped in {', '.join(sorted(set(ZONES) | hubs))}: {len(closed)}")
    log: list[str] = []
    added: list[dict] = []
    for zone, p in ZONES.items():
        vre = reg_val(reg, f"{p}06_vre_national")
        years = link_years(reg)
        for tech, levels in vre.items():
            added += impose(kept + added, zone, tech, dict(zip(years, levels)), header, log)
        added += impose(kept + added, zone, "Gas", year_dict(reg_val(reg, f"{p}11_gas_fleet" if p == "g" else f"{p}09_gas_fleet")), header, log)
        added += impose(kept + added, zone, "Hydro", year_dict(reg_val(reg, f"{p}09_hydro_fleet" if p == "g" else f"{p}10_hydro_fleet")), header, log)
    # hubs: cumulative GW per tech and year, committed step by step
    az = reg["a07_vre_hub_2040"]
    steps = {int(k): {t: float(v) for t, v in d.items()} for k, d in (az.get("detail") or {}).get("by_year", {}).items()}
    steps[max(link_years(reg))] = {t: float(v) for t, v in az["cesi_value"].items()}
    added += hub_tranches("GEC_AZ", steps, header, log)
    ge = reg_val(reg, "g07_vre_hub_2040")
    ys = link_years(reg)
    steps_ge = {y: {t: float(v) * (i + 1) / len(ys) for t, v in ge.items()} for i, y in enumerate(ys)}
    added += hub_tranches("GEC_GE", steps_ge, header, log)
    print("  fleet alignment:")
    print("\n".join(log))
    write_csv(out / "pGenDataInput_cesi.csv", header, kept + added, bom, dry)


def hub_tranches(zone: str, steps: dict[int, dict[str, float]], header: list[str], log: list) -> list[dict]:
    added = []
    prev: dict[str, float] = {}
    for y in sorted(steps):
        for tech, gw in steps[y].items():
            gap = gw * 1000.0 - prev.get(tech, 0.0)
            if gap > 0.5:
                added.append(tranche(f"{zone}_CESI_{tech}_{y}", zone, tech, VRE_FUEL[tech], y, gap, header))
                log.append(f"    {zone} {tech} {y}: +{gap:.0f} MW")
            prev[tech] = gw * 1000.0
    return added


def build_storage(dep: Path, reg: dict, cfg: dict, out: Path, dry: bool) -> None:
    header, rows, bom = read_csv(cfg["pStorageDataInput"])
    first = min(link_years(reg))
    kept, added = [], []
    for zone, p in ZONES.items():
        cands = [r for r in rows if r["z"] == zone and r["Status"] == "3"]
        model = next((r for r in cands if r["f"] == "Battery"), None)
        if model is None:
            sys.exit(f"no battery candidate to copy the costs from in {zone}")
        mw = float(reg_val(reg, "a17_bess" if p == "a" else "g14_storage")) * 1000.0
        r = dict(model)
        life = num(r.get("Life"), 15)
        r.update({"g": f"{zone}_CESI_BESS", "Status": "2", "StYr": str(first), "RetrYr": str(int(first + life)),
                  "Capacity": fmt(mw, 1), "BuildLimitperYear": fmt(mw, 1), "CapacityMWh": fmt(mw * BESS_HOURS, 1)})
        added.append(r)
        print(f"    {zone}: {len(cands)} storage candidate(s) dropped, committed BESS {mw:.0f} MW from {first}")
    kept = [r for r in rows if not (r["z"] in ZONES and r["Status"] == "3")]
    write_csv(out / "pStorageDataInput_cesi.csv", header, kept + added, bom, dry)


# ---------------------------------------------------------------------------
# Defaults, settings
# ---------------------------------------------------------------------------

def with_hub_copies(rows: list[dict], zone_col: str) -> list[dict]:
    """Add GEC_GE rows copied from Georgia where the base file has none (GEC_AZ already has its own)."""
    present = {r[zone_col] for r in rows}
    outp = list(rows)
    for nat, hub in HUB.items():
        if hub not in present:
            outp += [dict(r, **{zone_col: hub}) for r in rows if r[zone_col] == nat]
    return outp


def build_gendefault(dep: Path, reg: dict, cfg: dict, out: Path, dry: bool) -> None:
    header, rows, bom = read_csv(cfg["pGenDataInputDefault"])
    rows = with_hub_copies(rows, "z")
    capex = {t: float(v) * EUR_USD for t, v in reg_val(reg, "a11_vre_capex").items()}   # EUR/W to MUSD/MW
    fom = {t: float(v) for t, v in reg_val(reg, "a12_vre_fom").items()}                 # % of capex per year
    life = int(reg_val(reg, "c02_asset_life"))
    zones = set(ZONES) | set(HUB.values())
    n = 0
    for r in rows:
        if r["z"] in zones and r["tech"] in capex:
            r["Capex"] = fmt(capex[r["tech"]], 4)
            r["FOMperMW"] = fmt(capex[r["tech"]] * 1e6 * fom[r["tech"]] / 100.0, 1)
            r["Life"] = str(life)
            n += 1
    print(f"  VRE capex, FOM and life set on {n} default rows in {', '.join(sorted(zones))}")
    write_csv(out / "pGenDataInputDefault_cesi.csv", header, rows, bom, dry)


def build_capex_traj(dep: Path, reg: dict, cfg: dict, out: Path, dry: bool) -> None:
    header, rows, bom = read_csv(cfg["pCapexTrajectoriesDefault"])
    rows = with_hub_copies(rows, "zone")
    zones = set(ZONES) | set(HUB.values())
    n = 0
    for r in rows:
        if r["zone"] in zones and r["tech"] in VRE:
            for c in header[3:]:
                r[c] = "1"
            n += 1
    print(f"  VRE capex trajectories flattened to 1 on {n} rows")
    write_csv(out / "pCapexTrajectoriesDefault_cesi.csv", header, rows, bom, dry)


def build_settings(dep: Path, reg: dict, cfg: dict, out: Path, dry: bool) -> None:
    header, rows, bom = read_csv(cfg["pSettings"])
    rate = float(reg_val(reg, "c01_discount_rate"))
    hit = 0
    for r in rows:
        if r["Abbreviation"] in ("WACC", "DR"):
            r["Value"] = fmt(rate, 4)
            hit += 1
    if hit != 2:
        sys.exit("WACC and DR rows not both found in pSettings")
    print("  WACC and DR set to the study discount rate")
    write_csv(out / "pSettings_cesi.csv", header, rows, bom, dry)


# ---------------------------------------------------------------------------
# pTradePriceExport
# ---------------------------------------------------------------------------

def build_export_price(dep: Path, reg: dict, cfg: dict, out: Path, dry: bool) -> None:
    """Romania export price with the study cable loss.

    Our files hold P_import = L*S + W and P_export = L*S*(1 - lambda) - W - C, the loss on
    the seller. On the reference (no CBAM) file the export price is an affine function of
    the import price, P_export = a*P_import + b with a = 1 - lambda and b = -W*(2 - lambda).
    Both are refitted here by least squares on the hours above the export floor, then the
    study lambda replaces ours with the same W.
    """
    h_imp, imp, _ = read_csv(cfg["pTradePrice"])
    header, rows, bom = read_csv(cfg["pTradePriceExport"])
    hours = hour_cols(header)
    key = lambda r: (r["zext"], r["q"], r["d"], r["year"])
    imp_by = {key(r): r for r in imp if r["zext"] == EXT_ZONE}
    xs, ys = [], []
    for r in rows:
        if r["zext"] != EXT_ZONE:
            continue
        i = imp_by.get(key(r))
        if i is None:
            sys.exit(f"no import price for {key(r)}")
        for c in hours:
            x, y = float(i[c]), float(r[c])
            if y > 0.011:
                xs.append(x)
                ys.append(y)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    b = my - a * mx
    lam_ours = 1.0 - a
    w = -b / (2.0 - lam_ours)
    resid = max(abs(y - (a * x + b)) for x, y in zip(xs, ys))
    if resid > 0.01:
        sys.exit(f"export price is not affine in the import price (max residual {resid:.3f}); is CBAM in the reference file?")
    lam = float(reg_val(reg, "k05_loss_submarine")) / 100.0
    floor = min(float(r[c]) for r in rows if r["zext"] == EXT_ZONE for c in hours)
    n_rows = 0
    for r in rows:
        if r["zext"] != EXT_ZONE:
            continue
        i = imp_by[key(r)]
        for c in hours:
            r[c] = fmt(max((float(i[c]) - w) * (1.0 - lam) - w, floor), 3)
        n_rows += 1
    print(f"  {EXT_ZONE}: refit of our files gives lambda {lam_ours:.4f} and W {w:.3f} USD/MWh "
          f"(max residual {resid:.4f}); {n_rows} rows rebuilt with the study loss, floor {floor}")
    write_csv(out / "pTradePriceExport_cesi.csv", header, rows, bom, dry)


# ---------------------------------------------------------------------------
# pVREProfile
# ---------------------------------------------------------------------------

def build_vre(dep: Path, reg: dict, cfg: dict, out: Path, dry: bool) -> None:
    header, rows, bom = read_csv(cfg["pVREProfile"])
    _, hrs, _ = read_csv(dep / "pHours.csv")
    days = {(r["q"], r["d"]): float(r["t01"]) for r in hrs}
    hours = hour_cols(header)
    total_h = sum(days.values()) * len(hours)
    targets = {}
    for zone, p in ZONES.items():
        for tech, rid in (("PV", f"{p}12_cf_pv" if p == "g" else "a13_cf_pv"),
                          ("OnshoreWind", f"{p}13_cf_onshore" if p == "g" else "a14_cf_onshore"),
                          ("OffshoreWind", "a15_cf_offshore" if p == "a" else None)):
            if rid and rid in reg:
                targets[(zone, tech)] = float(reg_val(reg, rid)) / 100.0

    def mean_cf(block: list[dict]) -> float:
        return sum(days[(r["season"], r["daytype"])] * sum(float(r[c]) for c in hours) for r in block) / total_h

    nat = {z: [r for r in rows if r["zone"] == z] for z in ZONES}
    for (zone, tech), target in targets.items():
        block = [r for r in nat[zone] if r["tech"] == tech]
        before = mean_cf(block)
        vals = {id(r): [float(r[c]) for c in hours] for r in block}
        cur = before
        for _ in range(60):
            k = target / cur if cur else 1.0
            for r in block:
                vals[id(r)] = [min(1.0, v * k) for v in vals[id(r)]]
            for r in block:
                for c, v in zip(hours, vals[id(r)]):
                    r[c] = fmt(v, 3)
            cur = mean_cf(block)
            if abs(cur - target) < 5e-5:
                break
        print(f"    {zone} {tech}: mean CF {before:.4f} -> {cur:.4f} (study {target:.4f})")
    # hubs carry the rescaled national profiles
    others = [r for r in rows if r["zone"] not in set(ZONES) | set(HUB.values())]
    hubs = [dict(r, zone=HUB[z]) for z in ZONES for r in nat[z]]
    write_csv(out / "pVREProfile_cesi.csv", header, others + [r for z in ZONES for r in nat[z]] + hubs, bom, dry)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deployment", default="data_blacksea")
    ap.add_argument("--dry-run", action="store_true", help="log everything, write nothing")
    args = ap.parse_args()
    dep = INPUT_ROOT / args.deployment
    reg = load_register(dep)
    cfg = load_config(dep)
    out = dep / "cesi"
    print(f"build_cesi_full: {dep.name}, {len(reg)} register entries")
    for name, fn in (("pDemandForecast", build_demand), ("pGenDataInput", build_gendata),
                     ("pStorageDataInput", build_storage), ("pGenDataInputDefault", build_gendefault),
                     ("pCapexTrajectoriesDefault", build_capex_traj), ("pSettings", build_settings),
                     ("pTradePriceExport", build_export_price), ("pVREProfile", build_vre)):
        print(f"\n== {name}")
        fn(dep, reg, cfg, out, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
