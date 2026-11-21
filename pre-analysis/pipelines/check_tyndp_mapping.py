# -*- coding: utf-8 -*-
"""Check the TYNDP to EPM correspondence and write the table for review.

Reads pre-analysis/config/tyndp_epm_mapping.yaml, the tidy tables of
pre-analysis/output_tyndp/ and the EPM files named in the mapping, and writes to
the same folder:

    mapping_nodes.csv         TYNDP nodes against EPM zones
    mapping_borders.csv       one row per direction, TYNDP values next to the EPM limit
    mapping_candidates.csv    GRID.xlsx candidates with their action
    mapping_technologies.csv  tidy technologies against EPM technology and fuel
    mapping_checks.csv        one row per check, with the count of failures
    mapping_manifest.csv      rows and sha256 of each table and of each input

Method
------
* A tidy row is either mapped or explicitly excluded. Anything else fails.
* Every EPM target must exist: zones in zcmap or zext, borders in the limit files,
  technology and fuel pairs in pTechFuel. The plant list is only reported, a pair
  absent from it is allowed.
* Every cross-country EPM link must appear in the mapping, with or without a
  TYNDP source.
* Several nodes on one side of a border are summed. Editions are never mixed.
* Nothing here writes to epm/input.

The output is deterministic: a rerun on the same inputs gives the same hashes.

Usage (from EPM/):  python pre-analysis/pipelines/check_tyndp_mapping.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml

from build_tyndp_tidy import CONFIG, OUT_DIR, REPO, write

MAPPING = CONFIG.parent / "tyndp_epm_mapping.yaml"
EPM_INPUT = REPO / "epm" / "input" / "data_blacksea"
SPLIT_TOL = 1.0  # MW
EPM_YEARS = ["2025", "2030", "2036", "2040"]
TYNDP_COLUMNS = [("2024", 2030), ("2026", 2025), ("2026", 2030), ("2026", 2035), ("2026", 2040)]


def read_epm(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def resolve_technology(name: str, mapping: dict) -> dict | None:
    if name in mapping["technologies"]:
        return mapping["technologies"][name]
    for pattern in mapping["technology_patterns"]:
        if name.startswith(pattern["starts_with"]):
            return pattern
    return None


def epm_limits(ext: pd.DataFrame, internal: pd.DataFrame) -> dict:
    """{(from, to): {year: MW}} for both limit files, largest season per year."""
    limits: dict = {}
    direction = ext.columns[3]
    for (z, x, way), group in ext.groupby([ext.columns[0], ext.columns[1], direction]):
        key = (z, x) if way == "Export" else (x, z)
        limits[key] = {y: float(group[y].max()) for y in EPM_YEARS}
    for (z, z2), group in internal.groupby([internal.columns[0], internal.columns[1]]):
        limits[(z, z2)] = {y: float(group[y].max()) for y in EPM_YEARS}
    return limits


def tyndp_limit(ntc: pd.DataFrame, nodes_from: list, nodes_to: list, edition: str, year: int) -> float | None:
    metric = "ntc_2025" if year == 2025 else "ntc_reference"
    rows = ntc[(ntc.edition == edition) & (ntc.year == year) & (ntc.metric == metric)
               & ntc.node_from.isin(nodes_from) & ntc.node_to.isin(nodes_to)]
    return None if rows.empty else float(rows.value.sum())


def border_rows(ntc: pd.DataFrame, mapping: dict, limits: dict) -> list[dict]:
    rows = []
    for border in mapping["borders"]:
        for start, end, nodes_from, nodes_to in ((border["a"], border["b"], border["nodes_a"], border["nodes_b"]),
                                                 (border["b"], border["a"], border["nodes_b"], border["nodes_a"])):
            row = dict(epm_from=start, epm_to=end, tyndp_from="+".join(nodes_from), tyndp_to="+".join(nodes_to),
                       epm_file=mapping["epm_reference"]["ext_limits" if border["file"] == "ext" else "int_limits"],
                       current_rule=border["current_rule"], b2=border["b2"])
            for edition, year in TYNDP_COLUMNS:
                row[f"tyndp{edition}_{year}"] = tyndp_limit(ntc, nodes_from, nodes_to, edition, year)
            current = limits.get((start, end), {})
            for year in EPM_YEARS:
                row[f"epm_{year}"] = current.get(year)
            rows.append(row)
    return rows


def candidate_rows(candidates: pd.DataFrame, mapping: dict) -> list[dict]:
    rows = []
    real = mapping["candidates"]["real"]
    for item in candidates.itertuples():
        entry = real.get(item.border)
        concept = "Concept" in item.border
        action = entry["action"] if entry else mapping["candidates"]["rule_concept"] if concept else None
        rows.append(dict(border=item.border, year=item.year, capacity_forward_mw=item.capacity_forward_mw,
                         capacity_backward_mw=item.capacity_backward_mw, capex_meur=item.capex_meur, action=action,
                         same_as=entry.get("same_as") if entry else None, epm_file=entry.get("file") if entry else None,
                         note=entry.get("note") if entry else None))
    return rows


def technology_rows(capacity: pd.DataFrame, mapping: dict, tech_fuel: set, plants: set) -> list[dict]:
    rows = []
    for name, group in capacity.groupby("technology"):
        target = resolve_technology(name, mapping)
        tech, fuel = (target["tech"], target["fuel"]) if target else (None, None)
        rows.append(dict(technology=name, family=group.family.iloc[0], level="+".join(sorted(group.level.unique())),
                         rows_2024=int((group.edition == "2024").sum()), rows_2026=int((group.edition == "2026").sum()),
                         tech=tech, fuel=fuel, match=target["match"] if target else "unmapped",
                         in_tech_fuel=(tech, fuel) in tech_fuel if tech else None,
                         in_plants=(tech, fuel) in plants if tech else None,
                         note=target.get("note") if target else None))
    return rows


def split_gaps(capacity: pd.DataFrame) -> list[dict]:
    """2026 rows where the split parts do not add up to the total of their technology."""
    parts = {"Photovoltaic": ["Photovoltaic dedicated", "Photovoltaic e-market", "Rooftop"],
             "Onshore wind": ["Onshore wind dedicated", "Onshore wind e-market"],
             "Offshore wind": ["Offshore wind dedicated - radial", "Offshore wind e-market - radial",
                               "Offshore wind e-market - hub ready"],
             "Battery": ["Battery Utility Scale", "Battery Residential"]}
    data = capacity[(capacity.edition == "2026") & (capacity.metric == "capacity")]
    gaps = []
    for total, names in parts.items():
        whole = data[data.technology == total].set_index(["node", "year"]).value
        summed = data[data.technology.isin(names)].groupby(["node", "year"]).value.sum()
        for key, value in summed.items():
            if key in whole.index and abs(whole[key] - value) > SPLIT_TOL:
                gaps.append(dict(technology=total, node=key[0], year=key[1], total=whole[key], parts=value))
    return gaps


def main() -> int:
    mapping = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
    reference = mapping["epm_reference"]
    ntc = pd.read_csv(OUT_DIR / "ntc.csv", dtype={"edition": str}, low_memory=False)
    capacity = pd.read_csv(OUT_DIR / "capacity.csv", dtype={"edition": str}, low_memory=False)
    exchanges = pd.read_csv(OUT_DIR / "exchanges.csv", dtype={"edition": str}, low_memory=False)
    candidates = pd.read_csv(OUT_DIR / "candidates.csv", low_memory=False)
    zcmap = read_epm(EPM_INPUT / reference["zcmap"])
    zext = read_epm(EPM_INPUT / reference["zext"]).zext.tolist()
    ext = read_epm(EPM_INPUT / reference["ext_limits"])
    internal = read_epm(EPM_INPUT / reference["int_limits"])
    tech_fuel_table = read_epm(EPM_INPUT / reference["tech_fuel"])
    plant_table = read_epm(EPM_INPUT / reference["plants"])
    tech_fuel = set(zip(tech_fuel_table.tech, tech_fuel_table.fuel))
    plants = set(zip(plant_table.tech, plant_table.f))

    zones = set(zcmap.z) | set(zext)
    country = dict(zip(zcmap.z, zcmap.c))
    limits = epm_limits(ext, internal)
    checks = []

    def check(name: str, failures: list) -> None:
        checks.append(dict(check=name, failures=len(failures), detail="; ".join(str(f) for f in failures[:12])))

    # nodes
    seen = set(capacity.node) | set(ntc.node_from) | set(ntc.node_to) | (set(exchanges.node) - {"ALL"})
    check("tidy nodes are in the mapping", sorted(seen - set(mapping["nodes"])))
    targets = {z for node in mapping["nodes"].values() for z in node["zones"]}
    check("node targets are EPM zones", sorted(targets - zones))
    check("every EPM zone has a node or is listed", sorted(zones - targets - set(mapping["zones_without_node"])))

    # borders
    mapped = set()
    for border in mapping["borders"]:
        for x in border["nodes_a"]:
            for y in border["nodes_b"]:
                mapped |= {(x, y), (y, x)}
    excluded = {tuple(item["nodes"]) for item in mapping["excluded_borders"]}
    excluded |= {(y, x) for x, y in excluded}
    pairs = set(zip(ntc.node_from, ntc.node_to))
    check("tidy borders are mapped or excluded", sorted(pairs - mapped - excluded))
    check("mapping borders exist in the tidy table", sorted((mapped | excluded) - pairs))
    borders = {tuple(sorted((b["a"], b["b"]))) for b in mapping["borders"]}
    without = {tuple(sorted((b["a"], b["b"]))) for b in mapping["borders_without_tyndp"]}
    links = {tuple(sorted(key)) for key in limits}
    cross = {pair for pair in links if not any(z.startswith("GEC_") for z in pair)
             and country.get(pair[0], pair[0]) != country.get(pair[1], pair[1])}
    check("cross-country EPM links are in the mapping", sorted(cross - borders - without))
    check("mapping links exist in the EPM limit files", sorted((borders | without) - links))
    check("no link listed twice", sorted(borders & without))
    table_borders = pd.DataFrame(border_rows(ntc, mapping, limits))

    spot = []
    for item in mapping["spot_checks"]:
        for start, end, expected in ((item["a"], item["b"], item["a_to_b"]), (item["b"], item["a"], item["b_to_a"])):
            row = table_borders[(table_borders.epm_from == start) & (table_borders.epm_to == end)].iloc[0]
            value = row[f"tyndp{item['edition']}_{item['year']}"]
            if value != expected:
                spot.append((start, end, item["edition"], item["year"], value, expected))
    check("spot values read in the source", spot)
    active = table_borders[table_borders.b2 == "tyndp_limit"]
    check("default B2 borders have a 2026 value every year",
          active.loc[active[["tyndp2026_2030", "tyndp2026_2035", "tyndp2026_2040"]].isna().any(axis=1), "epm_from"].tolist())

    # candidates
    table_candidates = pd.DataFrame(candidate_rows(candidates, mapping))
    check("every candidate has an action", table_candidates.loc[table_candidates.action.isna(), "border"].tolist())
    projects = set(ntc.project.dropna())
    check("candidate twins are 2026 projects",
          sorted(set(table_candidates.same_as.dropna()) - projects))

    # technologies
    table_tech = pd.DataFrame(technology_rows(capacity, mapping, tech_fuel, plants))
    check("tidy technologies are mapped", table_tech.loc[table_tech.match == "unmapped", "technology"].tolist())
    check("technology targets are in pTechFuel",
          table_tech.loc[table_tech.tech.notna() & (table_tech.in_tech_fuel != True), "technology"].tolist())  # noqa: E712
    gaps = split_gaps(capacity)
    check("2026 split rows add up to their total", gaps)
    names = {item for item in exchanges.technology.unique()}
    check("exchanges rows are mapped or excluded", sorted(names - set(mapping["exchanges"])))

    # nodes table
    table_nodes = pd.DataFrame([dict(node=node, zones="+".join(item["zones"]), role=item["role"],
                                     split=item["split"], note=item.get("note")) for node, item in mapping["nodes"].items()])

    table_checks = pd.DataFrame(checks)
    manifest: list[dict] = []
    for name, frame in (("mapping_nodes", table_nodes), ("mapping_borders", table_borders),
                        ("mapping_candidates", table_candidates), ("mapping_technologies", table_tech),
                        ("mapping_checks", table_checks)):
        write(frame, name, manifest)
    inputs = [MAPPING] + [OUT_DIR / f"{n}.csv" for n in ("ntc", "capacity", "exchanges", "candidates")]
    inputs += [EPM_INPUT / reference[k] for k in ("zcmap", "zext", "ext_limits", "int_limits", "tech_fuel", "plants")]
    for path in inputs:
        manifest.append(dict(table=f"input {path.name}", rows=0, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(manifest).to_csv(OUT_DIR / "mapping_manifest.csv", index=False, lineterminator="\n")
    print(table_checks[["check", "failures"]].to_string(index=False))
    for item in table_checks[table_checks.failures > 0].itertuples():
        print(f"  {item.check}: {item.detail[:400]}")
    return 1 if table_checks.failures.sum() else 0


if __name__ == "__main__":
    sys.exit(main())
