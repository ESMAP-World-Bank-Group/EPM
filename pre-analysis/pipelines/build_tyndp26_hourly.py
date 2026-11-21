# -*- coding: utf-8 -*-
"""Hourly marginal cost and native demand of TYNDP 2026 NT+ for the price nodes of the model.

Reads the hourly dashboards of the TYNDP 2026 draft (National Trends+) without extracting the
package and writes to pre-analysis/output_tyndp/hourly/:

    tyndp26_ntplus_hourly.parquet   node, year, ws, kind, hour, value (8,760 hours per series)
    ws_weights.csv                  weight of each weather scenario in the "Weighted WS" columns
    annual_means.csv                time mean and demand weighted mean per node, year, weather scenario
    kpi_check.csv                   KPI price rebuilt from the hours against the KPI dashboard
    kpi_components.csv              load and price times load of every load column behind that check
    hourly_manifest.csv             packages read with their sha256, rows and sha256 of the tables

Method
------
* Package: 2026/outputs/NT+_TimeSeriesDashboard.zip, one workbook per country, target year
  (2030, 2035, 2040, 2050) and weather scenario (three per year). Sheet E-Market.
* A node is the market zone <country>00. price = the column "Marginal Cost [EUR/MWh_e]" of the
  node (exactly one), native_demand = the sum of its "Native Demand [MW_e]" columns.
  Prices are EUR 2024 per MWh.
* hour = position in the workbook, 0 to 8759. The labels run from "01. Jan. 00:00" to
  "31. Dec. 23:00" in UTC+1 without daylight saving. No clock shift is applied here.
* Weather scenario weights are not published as such. They are solved from the KPI dashboard,
  where every row gives the three weather scenarios and their weighted value, then checked
  against the expected two decimal weights below. A drift stops the run.
* The price kept for the model is the wholesale node (sheet E-Market). The KPI dashboard publishes
  another quantity, the "Load-Weighted Average" of a country: its electricity nodes pooled
  (E-Market and Prosumer sheets, the prosumer node has its own, higher marginal cost), weighted by
  every column flagged Load except the electrolysers. A node whose net load over the year is not
  positive is left out of the pool (the Austrian prosumer node). That KPI is rebuilt here from the
  hours, per weather scenario and weighted, as the proof that the hourly columns and the weights
  are the right ones. A gap above 0.05 EUR/MWh stops the run. The KPI is not the wholesale price:
  over the twelve nodes and four years its weighted value sits between 5.8 EUR/MWh below (Austria
  2040) and 28.2 above (Slovakia 2050) the time mean of the wholesale node.
* Both packages are checked against Data/EU/TYNDP/manifest.csv. A different sha256 stops the run.
* One cache file per workbook (hourly/cache/), so an interrupted run resumes where it stopped.
* Nothing here writes to epm/input.

Usage (from EPM/):  python pre-analysis/pipelines/build_tyndp26_hourly.py [--workers 4]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import yaml

_PRE_ANALYSIS = Path(__file__).resolve().parent.parent
REPO = _PRE_ANALYSIS.parent
CONFIG = _PRE_ANALYSIS / "config" / "tyndp_scope.yaml"
OUT_DIR = _PRE_ANALYSIS / "output_tyndp" / "hourly"
CACHE_DIR = OUT_DIR / "cache"

FOLDER = "2026/outputs"
ZIP_NAME = "NT+_TimeSeriesDashboard.zip"
KPI_NAME = "NT+_KPI_Dashboard.xlsx"
SHEET = "E-Market"
YEARS = (2030, 2035, 2040, 2050)
HOURS = 8760
FIRST_ROW = 10                                   # first hourly row of the sheet, 0 based
HEAD_ZONE, HEAD_CAT, HEAD_FLAG = 5, 6, 8         # header rows, 0 based: zone, category, Load / Generation / Other
KPI_SHEETS = ("E-Market", "Prosumer")            # electricity nodes pooled in the KPI price of a country
KPI_EXCLUDED = "electrolyz"                      # load categories left out of the KPI weights (links to the H2 zones)
CACHE_VERSION = 2
MEMBER_RE = re.compile(r"/(\d{4})/Weather scenario (\d{3})/([A-Z]{2})_TimeSeriesDashboard_TY\d{4}_WS\d{3}_SoS\.xlsx$")

# Price nodes of the v6 model: the borders of Romania, Bulgaria and Hungary, plus the three
# countries themselves (Romania and Bulgaria are price zones in the Ext scope).
NODES = ["RO00", "BG00", "HU00", "RS00", "GR00", "MD00", "UA00", "MK00", "AT00", "SK00", "HR00", "SI00"]

# Weights read off the "Weighted WS" columns of the KPI dashboard. The solver must land on them.
EXPECTED_WEIGHTS = {
    2030: {"003": 0.63, "021": 0.07, "029": 0.30},
    2035: {"032": 0.24, "037": 0.43, "059": 0.33},
    2040: {"065": 0.20, "071": 0.40, "077": 0.40},
    2050: {"091": 0.23, "092": 0.57, "106": 0.20},
}
WEIGHT_TOL = 0.002
KPI_TOL = 0.05                                   # EUR/MWh, KPI price rebuilt from the hours against the KPI row


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def checked(root: Path, name: str) -> tuple[Path, str]:
    """Path and sha256 of a package, after comparison with the manifest."""
    manifest = pd.read_csv(root / "manifest.csv", dtype=str)
    row = manifest[(manifest["folder"] == FOLDER) & (manifest["file"] == name)]
    if len(row) != 1:
        raise SystemExit(f"{FOLDER}/{name}: {len(row)} rows in manifest.csv, expected 1")
    path = root / FOLDER / name
    digest = sha256(path)
    if digest != row["sha256"].iloc[0]:
        raise SystemExit(f"{FOLDER}/{name}: sha256 differs from manifest.csv")
    return path, digest


def cache_path(code: str, year: int, ws: str) -> Path:
    return CACHE_DIR / f"{code}_{year}_WS{ws}_v{CACHE_VERSION}.npz"


def read_sheet(wb, name: str, member: str) -> tuple[list, list]:
    """Header rows and the 8,760 hourly rows of one sheet, after a check that it ends there."""
    rows = []
    for i, r in enumerate(wb[name].iter_rows(values_only=True)):
        rows.append(r)
        if i >= FIRST_ROW + HOURS:               # one row past the last hour, to check the sheet ends there
            break
    body = rows[FIRST_ROW:FIRST_ROW + HOURS]
    if len(body) != HOURS:
        raise SystemExit(f"{member} [{name}]: {len(body)} hourly rows, expected {HOURS}")
    extra = rows[FIRST_ROW + HOURS] if len(rows) > FIRST_ROW + HOURS else None
    if extra is not None and len(extra) > 1 and extra[1] not in (None, ""):
        raise SystemExit(f"{member} [{name}]: a row follows hour {HOURS}: {extra[:3]}")
    return rows[:FIRST_ROW], body


def column(body: list, j: int, where: str, strict: bool) -> tuple[np.ndarray, int]:
    """One hourly column as floats. Blank cells stop the run when strict, count as zero otherwise."""
    col = [r[j] if j < len(r) else None for r in body]
    blanks = sum(1 for x in col if x is None or x == "")
    if blanks and strict:
        raise SystemExit(f"{where}: {blanks} blank cells in column {j}")
    if not all(isinstance(x, (int, float)) or x is None or x == "" for x in col):
        raise SystemExit(f"{where}: non numeric cell in column {j}")
    v = np.array([0.0 if x is None or x == "" else float(x) for x in col])
    if not np.isfinite(v).all():
        raise SystemExit(f"{where}: non finite value in column {j}")
    return v, blanks


def read_member(args: tuple[str, str]) -> tuple[str, str]:
    """Parse one workbook into its cache file. Returns (member, message)."""
    zip_path, member = args
    m = MEMBER_RE.search(member)
    year, ws, code = int(m.group(1)), m.group(2), m.group(3)
    node = code + "00"
    with zipfile.ZipFile(zip_path) as z:
        data = z.read(member)
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    sheets = {name: read_sheet(wb, name, member) for name in KPI_SHEETS if name in wb.sheetnames}
    wb.close()
    if SHEET not in sheets:
        raise SystemExit(f"{member}: no sheet {SHEET}")

    out, labels, blanks = {}, None, 0
    comp = {"sheet": [], "zone": [], "category": [], "load_mwh": [], "price_load": []}
    for name, (head, body) in sheets.items():
        lab = [str(r[1]) for r in body]
        if labels is None:
            labels = lab
        elif lab != labels:
            raise SystemExit(f"{member} [{name}]: hour labels differ from {SHEET}")
        zones, cats, flags = head[HEAD_ZONE], head[HEAD_CAT], head[HEAD_FLAG]
        width = len(cats)
        zone_of = [str(zones[j] or "").strip() if j < len(zones) else "" for j in range(width)]
        cat_of = [str(cats[j] or "").strip() for j in range(width)]
        flag_of = [str(flags[j] or "").strip() if j < len(flags) else "" for j in range(width)]
        for zone in sorted({z for z in zone_of[2:] if z}):
            js = [j for j in range(2, width) if zone_of[j] == zone]
            price_js = [j for j in js if cat_of[j].lower().startswith("marginal cost")]
            if not price_js:
                continue                         # not a priced electricity zone
            if len(price_js) != 1:
                raise SystemExit(f"{member} [{name}]: {len(price_js)} marginal cost columns for {zone}")
            where = f"{member} [{name}] {zone}"
            price, _ = column(body, price_js[0], where, strict=True)
            native = np.zeros(HOURS)
            for j in js:
                if flag_of[j] != "Load":
                    continue
                is_native = cat_of[j].lower().startswith("native demand")
                v, nb = column(body, j, where, strict=is_native)
                blanks += nb
                if is_native:
                    native += v
                comp["sheet"].append(name)
                comp["zone"].append(zone)
                comp["category"].append(cat_of[j])
                comp["load_mwh"].append(float(v.sum()))
                comp["price_load"].append(float((price * v).sum()))
            if name == SHEET and zone == node:
                if not native.any():
                    raise SystemExit(f"{where}: no native demand")
                out = {"price": price, "native_demand": native}
    if not out:
        raise SystemExit(f"{member}: no marginal cost column for {node} in {SHEET}")
    np.savez_compressed(cache_path(code, year, ws), labels=np.array(labels), blanks=blanks,
                        **out, **{"comp_" + k: np.array(v) for k, v in comp.items()})
    return member, "%s %d WS%s price mean %.2f" % (node, year, ws, out["price"].mean())


def kpi_blocks(rows: list[list]) -> list[tuple[int, list[int], int, list[str]]]:
    """(year, three weather scenario columns, weighted column, labels) from the two header rows."""
    r3, r4 = rows[2], rows[3]
    blocks = []
    for j, v in enumerate(r4):
        if isinstance(v, str) and v.strip().lower().startswith("weighted"):
            ws_cols = [j - 3, j - 2, j - 1]
            blocks.append((int(r3[j - 3]), ws_cols, j, [str(r4[k]).replace("WS", "").strip() for k in ws_cols]))
    return blocks


def read_kpi(path: Path) -> tuple[dict, pd.DataFrame]:
    """Equations for the weights, and the electricity load weighted price per country and year."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    eq, labels_by_year, price = {}, {}, []
    for name in wb.sheetnames:
        rows = [list(r) for r in wb[name].iter_rows(values_only=True)]
        if len(rows) < 6:
            continue
        try:
            blocks = kpi_blocks(rows)
        except (TypeError, ValueError):
            continue
        if not blocks:
            continue
        block, sub = None, None
        for r in rows[4:]:
            if r[0] and not r[1]:
                block, sub = str(r[0]), None
            elif r[0]:
                sub = str(r[0]).strip()
            item = str(r[1]).replace("\xa0", " ").strip() if r[1] is not None else None
            for year, wcols, jw, labels in blocks:
                labels_by_year.setdefault(year, labels)
                if labels_by_year[year] != labels:
                    raise SystemExit(f"KPI sheet {name}: weather scenarios {labels} for {year}")
                vals = [r[k] if k < len(r) else None for k in wcols + [jw]]
                if not all(isinstance(v, (int, float)) for v in vals):
                    continue
                if (block or "").startswith("Prices") and sub == "Electricity" and item == "Load-Weighted Average":
                    price.append({"node": name + "00", "year": year, "kpi_weighted": float(vals[3]),
                                  **{"kpi_ws" + lab: float(v) for lab, v in zip(labels, vals[:3])}})
                a, b = np.array(vals[:3], float), float(vals[3])
                scale = max(abs(a).max(), 1e-9)
                if (a.max() - a.min()) / scale < 1e-3:   # the three scenarios agree: no information on the weights
                    continue
                eq.setdefault(year, []).append((a / scale, b / scale))
    wb.close()
    return {"eq": eq, "labels": labels_by_year}, pd.DataFrame(price)


def solve_weights(kpi: dict) -> pd.DataFrame:
    """Least squares over every KPI row, outliers trimmed, checked against the expected weights."""
    recs = []
    for year in YEARS:
        rows = kpi["eq"][year]
        a = np.array([x[0] for x in rows])
        b = np.array([x[1] for x in rows])
        keep = np.ones(len(b), bool)
        for _ in range(3):
            w, *_ = np.linalg.lstsq(a[keep], b[keep], rcond=None)
            resid = np.abs(a @ w - b)
            keep = resid < max(1e-4, np.percentile(resid, 60) * 5)
        w, *_ = np.linalg.lstsq(a[keep], b[keep], rcond=None)
        labels = kpi["labels"][year]
        if set(labels) != set(EXPECTED_WEIGHTS[year]):
            raise SystemExit(f"{year}: weather scenarios {labels}, expected {sorted(EXPECTED_WEIGHTS[year])}")
        for lab, wi in zip(labels, w):
            exp = EXPECTED_WEIGHTS[year][lab]
            if abs(wi - exp) > WEIGHT_TOL:
                raise SystemExit(f"{year} WS{lab}: solved weight {wi:.5f}, expected {exp:.2f}")
            recs.append({"year": year, "ws": lab, "weight": exp, "weight_solved": round(float(wi), 6),
                         "kpi_rows": len(b), "kpi_rows_kept": int(keep.sum())})
        if abs(sum(EXPECTED_WEIGHTS[year].values()) - 1) > 1e-9:
            raise SystemExit(f"{year}: expected weights do not sum to 1")
    return pd.DataFrame(recs)


def assemble(members: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int]:
    """Hourly long table and KPI load components from the cache files. One label list for the package."""
    frames, comps, labels, blanks = [], [], None, 0
    for member in members:
        m = MEMBER_RE.search(member)
        year, ws, code = int(m.group(1)), m.group(2), m.group(3)
        with np.load(cache_path(code, year, ws)) as z:
            lab = [str(x) for x in z["labels"]]
            if labels is None:
                labels = lab
            elif lab != labels:
                raise SystemExit(f"{member}: hour labels differ from the first workbook")
            blanks += int(z["blanks"])
            for kind in ("price", "native_demand"):
                frames.append(pd.DataFrame({"node": code + "00", "year": year, "ws": ws, "kind": kind,
                                            "hour": np.arange(HOURS, dtype=np.int32), "value": z[kind]}))
            comps.append(pd.DataFrame({"node": code + "00", "year": year, "ws": ws,
                                       "sheet": z["comp_sheet"], "zone": z["comp_zone"],
                                       "category": z["comp_category"], "load_mwh": z["comp_load_mwh"],
                                       "price_load": z["comp_price_load"]}))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["node", "year", "ws", "kind", "hour"], kind="stable").reset_index(drop=True)
    comp = pd.concat(comps, ignore_index=True)
    comp = comp.groupby(["node", "year", "ws", "sheet", "zone", "category"], as_index=False)[
        ["load_mwh", "price_load"]].sum()
    comp["in_kpi"] = ~comp["category"].str.lower().str.startswith(KPI_EXCLUDED)
    return df, comp, labels, blanks


def kpi_reproduction(comp: pd.DataFrame, means: pd.DataFrame, weights: pd.DataFrame,
                     kpi_price: pd.DataFrame) -> pd.DataFrame:
    """KPI "Load-Weighted Average" rebuilt from the hourly package, per weather scenario and weighted.

    The KPI price of a country pools its electricity nodes (E-Market and Prosumer sheets): the sum of
    price times load over the sum of load, every column flagged Load except the electrolysers. A node
    whose net load over the year is not positive is left out of the pool. This is the case of the
    Austrian prosumer node in 11 of its 12 workbooks, where the negative Fixed Demand column outweighs
    the other loads.
    """
    sheets = comp[comp["in_kpi"]].groupby(["node", "year", "ws", "sheet"], as_index=False)[
        ["load_mwh", "price_load"]].sum()
    used = sheets[sheets["load_mwh"] > 0].groupby(["node", "year", "ws"], as_index=False)[
        ["load_mwh", "price_load"]].sum()
    used["kpi_reproduced"] = used["price_load"] / used["load_mwh"]
    pub = kpi_price.melt(id_vars=["node", "year"], var_name="ws", value_name="kpi_published").dropna()
    pub["ws"] = pub["ws"].str.replace("kpi_ws", "").str.replace("kpi_weighted", "weighted")
    used = used.merge(weights[["year", "ws", "weight"]], on=["year", "ws"], validate="many_to_one")
    tot = used.assign(kpi_reproduced=used["kpi_reproduced"] * used["weight"]).groupby(
        ["node", "year"], as_index=False)["kpi_reproduced"].sum()
    tot["ws"] = "weighted"
    out = pd.concat([used[["node", "year", "ws", "kpi_reproduced"]], tot], ignore_index=True)
    out = out.merge(means[["node", "year", "ws", "price_time_mean"]], on=["node", "year", "ws"], how="left")
    out = out.merge(pub, on=["node", "year", "ws"], how="left")
    out["gap"] = out["kpi_reproduced"] - out["kpi_published"]
    out = out[["node", "year", "ws", "price_time_mean", "kpi_reproduced", "kpi_published", "gap"]]
    return out.sort_values(["node", "year", "ws"]).reset_index(drop=True)


def annual_means(df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Time mean, demand weighted mean and demand per node, year, weather scenario, then weighted."""
    wide = df.pivot_table(index=["node", "year", "ws", "hour"], columns="kind", values="value").reset_index()
    wide["pd"] = wide["price"] * wide["native_demand"]
    g = wide.groupby(["node", "year", "ws"], as_index=False).agg(
        price_time_mean=("price", "mean"), pd=("pd", "sum"), demand=("native_demand", "sum"),
        price_min=("price", "min"), price_max=("price", "max"))
    g["price_demand_weighted"] = g["pd"] / g["demand"]
    g["native_demand_twh"] = g["demand"] / 1e6
    g = g.merge(weights[["year", "ws", "weight"]], on=["year", "ws"], how="left", validate="many_to_one")
    if g["weight"].isna().any():
        raise SystemExit("a weather scenario of the hourly package has no weight")
    cols = ["price_time_mean", "price_demand_weighted", "native_demand_twh"]
    tot = g.assign(**{c: g[c] * g["weight"] for c in cols}).groupby(["node", "year"], as_index=False).agg(
        **{c: (c, "sum") for c in cols}, weight=("weight", "sum"),
        price_min=("price_min", "min"), price_max=("price_max", "max"))
    tot["ws"] = "weighted"
    keep = ["node", "year", "ws", "weight", *cols, "price_min", "price_max"]
    return pd.concat([g[keep], tot[keep]], ignore_index=True).sort_values(["node", "year", "ws"]).reset_index(drop=True)


def write_csv(df: pd.DataFrame, name: str, digits: int = 6) -> dict:
    text = df.round(digits).to_csv(index=False, lineterminator="\n")
    (OUT_DIR / name).write_text(text, encoding="utf-8", newline="")
    return {"table": name, "rows": len(df), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, help="Folder with manifest.csv, default from the config")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    scope = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    root = args.root or (REPO / scope["tyndp_root"]).resolve()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    zip_path, zip_sha = checked(root, ZIP_NAME)
    kpi_path, kpi_sha = checked(root, KPI_NAME)
    print("packages match manifest.csv", flush=True)

    kpi, kpi_price = read_kpi(kpi_path)
    weights = solve_weights(kpi)
    print(weights.to_string(index=False), flush=True)

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if "__MACOSX" not in n and MEMBER_RE.search(n)]
    members = sorted(n for n in names if MEMBER_RE.search(n).group(3) + "00" in NODES
                     and int(MEMBER_RE.search(n).group(1)) in YEARS)
    found = {(MEMBER_RE.search(n).group(3) + "00", int(MEMBER_RE.search(n).group(1)), MEMBER_RE.search(n).group(2))
             for n in members}
    wanted = {(node, y, ws) for node in NODES for y in YEARS for ws in EXPECTED_WEIGHTS[y]}
    if found != wanted:
        raise SystemExit(f"workbooks: missing {sorted(wanted - found)[:5]}, unexpected {sorted(found - wanted)[:5]}")
    todo = [n for n in members
            if not cache_path(MEMBER_RE.search(n).group(3), int(MEMBER_RE.search(n).group(1)),
                              MEMBER_RE.search(n).group(2)).exists()]
    print(f"workbooks {len(members)}, to read {len(todo)}", flush=True)
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, (_, msg) in enumerate(pool.map(read_member, [(str(zip_path), n) for n in todo]), 1):
                print(f"  {i:3d}/{len(todo)} {msg} ({time.time() - t0:.0f} s)", flush=True)

    df, comp, labels, blanks = assemble(members)
    if len(df) != len(NODES) * len(YEARS) * 3 * 2 * HOURS:
        raise SystemExit(f"hourly table has {len(df)} rows")
    if labels[0] != "01. Jan. 00:00" or labels[-1] != "31. Dec. 23:00":
        raise SystemExit(f"hour labels run from {labels[0]} to {labels[-1]}")
    print(f"blank cells read as zero in the load columns: {blanks}", flush=True)
    df.to_parquet(OUT_DIR / "tyndp26_ntplus_hourly.parquet", compression="zstd", index=False)

    means = annual_means(df, weights)
    check = kpi_reproduction(comp, means, weights, kpi_price)
    print(check[check["ws"] == "weighted"].round(3).to_string(index=False), flush=True)
    if check["kpi_published"].isna().any():
        raise SystemExit("a node has no price row in the KPI dashboard")
    bad = check[check["gap"].abs() > KPI_TOL]
    manifest = [write_csv(weights, "ws_weights.csv"), write_csv(means, "annual_means.csv"),
                write_csv(check, "kpi_check.csv"), write_csv(comp, "kpi_components.csv", digits=3)]
    manifest.append({"table": "tyndp26_ntplus_hourly.parquet", "rows": len(df),
                     "sha256": hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()})
    manifest += [{"table": f"source:{FOLDER}/{ZIP_NAME}", "rows": len(members), "sha256": zip_sha},
                 {"table": f"source:{FOLDER}/{KPI_NAME}", "rows": len(kpi_price), "sha256": kpi_sha}]
    pd.DataFrame(manifest).to_csv(OUT_DIR / "hourly_manifest.csv", index=False, lineterminator="\n")
    print(f"written to {OUT_DIR}", flush=True)
    if len(bad):
        raise SystemExit(f"{len(bad)} of {len(check)} KPI prices more than {KPI_TOL} EUR/MWh away from the "
                         "KPI dashboard:\n" + bad.round(3).to_string(index=False))
    print(f"KPI check passed on {len(check)} prices: largest gap {check['gap'].abs().max():.4f} EUR/MWh", flush=True)


if __name__ == "__main__":
    main()
