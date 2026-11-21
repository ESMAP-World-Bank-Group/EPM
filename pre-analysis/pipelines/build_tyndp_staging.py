# -*- coding: utf-8 -*-
"""Staged TYNDP inputs for EPM: border limits and candidate lines.

Reads pre-analysis/config/tyndp_epm_mapping.yaml, the tidy tables of
pre-analysis/output_tyndp/ and the EPM limit files of the reference scenario, and
writes to pre-analysis/output_tyndp/staging/:

    trade/pExtTransferLimit_robg_tyndp26.csv       external limits
    trade/pTransferLimit_robg_hvdc_tyndp26.csv     limits between modelled zones
    trade/pNewTransmission_robg_tyndp26.csv        candidate between modelled zones
    trade/pNewTransmissionExt_robg_tyndp26.csv     candidate to an external zone
    staging_borders.csv     one row per direction, current against staged
    staging_candidates.csv  candidates next to their source and the freeexp line
    staging_checks.csv      one row per check, with the count of failures
    staging_manifest.csv    rows and sha256 of each output and each input
    DIFF.md                 what would change, for review

Method
------
* Limits start from the current file. A row of a border marked tyndp_limit in the
  mapping takes max(current, NT+ 2026 reference value), per season and year.
* The reference value in force is the latest of 2030, 2035, 2040 at or before the
  year, so a line enters at its reference year. Before 2030 the current value stays.
* Every other row is copied byte for byte, the held borders RO-UA and RO-MD included.
* Candidates are the rows marked candidate in the mapping. CapacityPerLine is the
  weaker direction and the cost is the 2024 capex converted to USD.
* Nothing here writes to epm/input. Promotion is a separate step.

The output is deterministic: a rerun on the same inputs gives the same hashes.

Usage (from EPM/):  python pre-analysis/pipelines/build_tyndp_staging.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml

from build_tyndp_tidy import OUT_DIR
from check_tyndp_mapping import EPM_INPUT, MAPPING, tyndp_limit

EDITION = "2026"
REPORT_YEARS = [2025, 2030, 2032, 2036, 2040]
CANDIDATE_HEADER = ["From", "To", "EarliestEntry", "MaximumNumOfLines", "CapacityPerLine", "CostPerLine", "Life", "Status"]
HELD_NOTE = "Held for the Europe point, current value kept"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stem_name(source: str, suffix: str) -> str:
    path = Path(source)
    return f"{path.stem}_{suffix}{path.suffix}"


def reference_in_force(year: int, references: list) -> int | None:
    passed = [r for r in references if r <= year]
    return max(passed) if passed else None


def targets_from(mapping: dict, ntc: pd.DataFrame) -> tuple[dict, dict]:
    """({(from, to): {reference year: MW}}, {(from, to): action}) for every mapped direction."""
    references = sorted(set(mapping["limit_rule"]["years"].values()))
    targets, actions = {}, {}
    for border in mapping["borders"]:
        for start, end, nodes_from, nodes_to in ((border["a"], border["b"], border["nodes_a"], border["nodes_b"]),
                                                 (border["b"], border["a"], border["nodes_b"], border["nodes_a"])):
            actions[(start, end)] = border["b2"]
            if border["b2"] == "tyndp_limit":
                targets[(start, end)] = {r: tyndp_limit(ntc, nodes_from, nodes_to, EDITION, r) for r in references}
    return targets, actions


def stage_limits(text: str, first_year: int, key_of, targets: dict) -> str:
    """Raise the target rows of a limit file. Other rows keep their exact text."""
    lines = text.split("\n")
    years = [int(y) for y in lines[0].split(",")[first_year:]]
    out = [lines[0]]
    for line in lines[1:]:
        if not line:
            out.append(line)
            continue
        parts = line.split(",")
        target = targets.get(key_of(parts))
        if target is None:
            out.append(line)
            continue
        references = sorted(target)
        cells = parts[:first_year]
        for year, token in zip(years, parts[first_year:]):
            reference = reference_in_force(year, references)
            value = float(token)
            if reference is None or target[reference] <= value:
                cells.append(token)
            else:
                cells.append(f"{target[reference]:.1f}" if "." in token else f"{target[reference]:.0f}")
        out.append(",".join(cells))
    return "\n".join(out)


def ext_key(parts: list) -> tuple:
    return (parts[0], parts[1]) if parts[3] == "Export" else (parts[1], parts[0])


def int_key(parts: list) -> tuple:
    return (parts[0], parts[1])


def limit_frame(text: str, first_year: int, key_of) -> pd.DataFrame:
    """Long table (from, to, q, year, mw) of a limit file."""
    lines = [line for line in text.split("\n") if line]
    years = [int(y) for y in lines[0].split(",")[first_year:]]
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        start, end = key_of(parts)
        for year, token in zip(years, parts[first_year:]):
            rows.append((start, end, parts[2], year, float(token)))
    return pd.DataFrame(rows, columns=["start", "end", "q", "year", "mw"])


def candidate_tables(mapping: dict, table: pd.DataFrame, freeexp: dict) -> tuple[dict, list]:
    """({file: csv text}, rows of staging_candidates.csv)."""
    rule = mapping["candidates"]["rule_real"]
    files: dict = {}
    rows = []
    for border, entry in mapping["candidates"]["real"].items():
        source = table[table.border == border].iloc[0]
        row = dict(border=border, action=entry["action"], same_as=entry["same_as"], year=int(source.year),
                   capacity_forward_mw=source.capacity_forward_mw, capacity_backward_mw=source.capacity_backward_mw,
                   capex_meur=source.capex_meur)
        if entry["action"] == "candidate":
            start, end = entry["epm"]
            capacity = min(source.capacity_forward_mw, source.capacity_backward_mw)
            cost = round(source.capex_meur * rule["eur_usd"], 1)
            cells = [start, end, int(source.year), rule["max_lines"], capacity, cost, rule["life"], rule["status"]]
            files.setdefault(entry["file"], []).append(",".join(str(c) for c in cells))
            twin = freeexp.get(frozenset((start, end)))
            row.update(epm_from=start, epm_to=end, capacity_per_line_mw=capacity, cost_musd=cost,
                       musd_per_mw=round(cost / capacity, 3), freeexp_mw=twin[0] if twin else None,
                       freeexp_musd=twin[1] if twin else None,
                       freeexp_musd_per_mw=round(twin[1] / twin[0], 3) if twin else None)
        rows.append(row)
    texts = {name: "\n".join([",".join(CANDIDATE_HEADER)] + lines) + "\n" for name, lines in files.items()}
    return texts, rows


def diff_markdown(borders: pd.DataFrame, candidates: pd.DataFrame, names: dict) -> str:
    moved = borders[borders.status == "raised"]
    same = borders[borders.status == "unchanged"]
    lines = ["# TYNDP staging: what would change", "",
             "Staged for the reference scenario Int_Base_5y. Nothing is in epm/input yet.", "",
             "## Files", ""]
    lines += [f"- `{name}` from `{source}`" for name, source in names.items()]
    lines += ["", "## Border limits, MW", "",
              f"{len(moved)} directions rise, {len(same)} stay, {int((borders.status == 'held').sum())} are held.", "",
              "| From | To | Current 2040 | 2030 | 2032 | 2036 | 2040 | Change 2040 | Status |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in borders.itertuples():
        lines.append(f"| {row.epm_from} | {row.epm_to} | {row.current_2040:,.0f} | {row.staged_2030:,.0f} | {row.staged_2032:,.0f} "
                     f"| {row.staged_2036:,.0f} | {row.staged_2040:,.0f} | {row.change_2040:+,.0f} | {row.status} |")
    lines += ["", "2025 is unchanged on every border. Seasons stay flat.", "", "## Candidate lines", "",
              "| Line | Project | Action | Earliest entry | MW | Cost, MUSD | MUSD per MW | Freeexp MW | Freeexp MUSD | Freeexp MUSD per MW |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in candidates.itertuples():
        kept = row.action == "candidate"
        cells = [row.border, row.same_as, row.action]
        cells += [row.year, f"{row.capacity_per_line_mw:,.0f}", f"{row.cost_musd:,.1f}", f"{row.musd_per_mw:.3f}",
                  f"{row.freeexp_mw:,.0f}", f"{row.freeexp_musd:,.0f}", f"{row.freeexp_musd_per_mw:.3f}"] if kept else ["n/a"] * 7
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines += ["", "## Read before promotion", "",
              "- External zones are price takers, so higher limits give an upper bound on export volume and value.",
              "- The gross external trade cap of 0.2 for Romania and Bulgaria still applies, so part of a higher external limit may stay unused.",
              "- `fCountIntercoForReserves` is 1: the higher Romania-Bulgaria limit and a built Bulgaria-Trakia candidate also raise the planning reserve credit of those countries.",
              "- Candidate capex comes from the 2024 edition, the one exception to no mixing. GRID.xlsx states no price year; converted at 1.08 USD per EUR.",
              "- The TYNDP capex sits well below the cost of the freeexp line on the same border. The freeexp candidates are not merged here.",
              "- `CapacityPerLine` holds one value: Bulgaria-Trakia uses 700 MW, the weaker direction, for the full cost of the 1,100/700 MW project.",
              "- 2z variants and the corridor limit files (BSSC, GEC, onland) are not covered: these files replace the base limits only.", ""]
    return "\n".join(lines)


def main() -> int:
    mapping = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
    reference = mapping["epm_reference"]
    staging = mapping["staging"]
    folder = OUT_DIR / staging["folder"]
    (folder / "trade").mkdir(parents=True, exist_ok=True)

    ntc = pd.read_csv(OUT_DIR / "ntc.csv", dtype={"edition": str}, low_memory=False)
    source_candidates = pd.read_csv(OUT_DIR / "candidates.csv", low_memory=False)
    zones = set(pd.read_csv(EPM_INPUT / reference["zcmap"], encoding="utf-8-sig").z)
    externals = set(pd.read_csv(EPM_INPUT / reference["zext"], encoding="utf-8-sig").zext)
    inputs = [EPM_INPUT / reference[k] for k in ("zcmap", "zext", "ext_limits", "int_limits")]
    inputs += [EPM_INPUT / staging[k] for k in ("freeexp_int", "freeexp_ext", "losses")]
    before = {path: sha(path.read_bytes()) for path in inputs}

    ext_path, int_path = EPM_INPUT / reference["ext_limits"], EPM_INPUT / reference["int_limits"]
    ext_text, int_text = ext_path.read_bytes().decode("utf-8"), int_path.read_bytes().decode("utf-8")
    targets, actions = targets_from(mapping, ntc)
    new_ext = stage_limits(ext_text, 4, ext_key, targets)
    new_int = stage_limits(int_text, 3, int_key, targets)

    freeexp = {}
    for key in ("freeexp_int", "freeexp_ext"):
        table = pd.read_csv(EPM_INPUT / staging[key], encoding="utf-8-sig")
        for row in table.itertuples():
            freeexp[frozenset((row.From, row.To))] = (row.CapacityPerLine, row.CostPerLine)
    texts, candidate_rows = candidate_tables(mapping, source_candidates, freeexp)

    written = {stem_name(reference["ext_limits"], staging["suffix"]): (new_ext, reference["ext_limits"]),
               stem_name(reference["int_limits"], staging["suffix"]): (new_int, reference["int_limits"])}
    for name, text in texts.items():
        written[f"{name}_robg_{staging['suffix']}.csv"] = (text, "none, new candidate list")
    for name, (text, _) in written.items():
        (folder / "trade" / name).write_bytes(text.encode("utf-8"))

    old_long = pd.concat([limit_frame(ext_text, 4, ext_key), limit_frame(int_text, 3, int_key)])
    new_long = pd.concat([limit_frame(new_ext, 4, ext_key), limit_frame(new_int, 3, int_key)])
    merged = old_long.merge(new_long, on=["start", "end", "q", "year"], how="outer", suffixes=("_old", "_new"))

    rows = []
    for (start, end), action in actions.items():
        part = merged[(merged.start == start) & (merged.end == end)]
        row = dict(epm_from=start, epm_to=end, b2=action)
        for year in REPORT_YEARS:
            at_year = part[part.year == year]
            row[f"current_{year}"] = at_year.mw_old.max()
            row[f"staged_{year}"] = at_year.mw_new.max()
        for reference_year in (2030, 2035, 2040):
            row[f"tyndp_{reference_year}"] = targets.get((start, end), {}).get(reference_year)
        row["change_2040"] = row["staged_2040"] - row["current_2040"]
        row["status"] = "held" if action != "tyndp_limit" else "raised" if row["change_2040"] > 0 else "unchanged"
        row["note"] = HELD_NOTE if action != "tyndp_limit" else None
        rows.append(row)
    table_borders = pd.DataFrame(rows)
    table_candidates = pd.DataFrame(candidate_rows)

    checks = []

    def check(name: str, failures: list) -> None:
        checks.append(dict(check=name, failures=len(failures), detail="; ".join(str(f) for f in failures[:12])))

    def changed_rows(old: str, new: str, key_of) -> list:
        return [line for line, other in zip(old.split("\n"), new.split("\n"))
                if line != other and (not line or key_of(line.split(",")) not in targets)]

    check("same line count as the source",
          [n for n, old, new in (("ext", ext_text, new_ext), ("int", int_text, new_int)) if old.count("\n") != new.count("\n")])
    check("same header as the source",
          [n for n, old, new in (("ext", ext_text, new_ext), ("int", int_text, new_int)) if old.split("\n")[0] != new.split("\n")[0]])
    check("same row keys in the same order",
          [n for n, old, new, size in (("ext", ext_text, new_ext, 4), ("int", int_text, new_int, 3))
           if [line.split(",")[:size] for line in old.split("\n")] != [line.split(",")[:size] for line in new.split("\n")]])
    check("rows outside the target borders are identical",
          changed_rows(ext_text, new_ext, ext_key) + changed_rows(int_text, new_int, int_key))
    check("held borders are identical", table_borders[(table_borders.status == "held")
                                                      & (table_borders.change_2040 != 0)].epm_from.tolist())
    check("no value below the current limit", merged[merged.mw_new < merged.mw_old].head(12).to_dict("records"))
    check("years before 2030 are identical", merged[(merged.year < 2030) & (merged.mw_new != merged.mw_old)].head(12).to_dict("records"))
    steps = new_long.sort_values("year").groupby(["start", "end", "q"]).mw.apply(lambda s: s.is_monotonic_increasing)
    check("limits never fall over the years", [k for k, ok in steps.items() if not ok and k[:2] in targets])
    spread = merged[merged.apply(lambda r: (r.start, r.end) in targets, axis=1)].groupby(["start", "end", "year"]).mw_new.nunique()
    check("target limits are flat across seasons", [k for k, n in spread.items() if n != 1])

    expected = []
    years = mapping["limit_rule"]["years"]
    for (start, end), values in targets.items():
        for year, reference_year in years.items():
            row = table_borders[(table_borders.epm_from == start) & (table_borders.epm_to == end)].iloc[0]
            if row[f"staged_{year}"] != max(row[f"current_{year}"], values[reference_year]):
                expected.append((start, end, year))
    check("staged value equals max(current, TYNDP) at model years", expected)
    check("values stay whole MW", merged[(merged.mw_new % 1 != 0) & (merged.mw_new != merged.mw_old)].head(12).to_dict("records"))

    spots = []
    for item in mapping["staging_spot_checks"]:
        row = table_borders[(table_borders.epm_from == item["from"]) & (table_borders.epm_to == item["to"])].iloc[0]
        if row[f"staged_{item['year']}"] != item["mw"]:
            spots.append((item["from"], item["to"], item["year"], row[f"staged_{item['year']}"], item["mw"]))
    check("spot values worked out by hand", spots)

    kept = table_candidates[table_candidates.action == "candidate"]
    bad = []
    for row in kept.itertuples():
        entry = mapping["candidates"]["real"][row.border]
        pool = externals if entry["file"] == "pNewTransmissionExt" else zones
        if row.epm_from not in zones or row.epm_to not in pool:
            bad.append(row.border)
    check("candidate ends exist in the right zone set", bad)
    check("every line of the mapping has a source row", [b for b in mapping["candidates"]["real"]
                                                         if b not in set(source_candidates.border)])
    dropped = set(table_candidates[table_candidates.action != "candidate"].border)
    check("dropped lines are not staged", [b for b in dropped if any(str(b) in t for t in texts.values())])
    losses = pd.read_csv(EPM_INPUT / staging["losses"], encoding="utf-8-sig")
    pairs = set(zip(losses.z, losses.z2))
    check("internal candidates have a loss factor",
          [row.border for row in kept.itertuples() if mapping["candidates"]["real"][row.border]["file"] == "pNewTransmission"
           and (row.epm_from, row.epm_to) not in pairs])
    check("staged files use LF, no BOM, final newline",
          [n for n, (text, _) in written.items() if "\r" in text or text.startswith("﻿") or not text.endswith("\n")])
    check("EPM inputs unchanged during the run", [p.name for p, h in before.items() if sha(p.read_bytes()) != h])

    table_checks = pd.DataFrame(checks)
    manifest: list[dict] = []
    for name, frame in (("staging_borders", table_borders), ("staging_candidates", table_candidates), ("staging_checks", table_checks)):
        text = frame.to_csv(index=False, lineterminator="\n")
        (folder / f"{name}.csv").write_bytes(text.encode("utf-8"))
        manifest.append(dict(item=name, rows=len(frame), sha256=sha(text.encode("utf-8"))))
    for name, (text, _) in written.items():
        manifest.append(dict(item=f"trade/{name}", rows=text.count("\n") - 1, sha256=sha(text.encode("utf-8"))))
    report = diff_markdown(table_borders, table_candidates, {n: s for n, (_, s) in written.items()})
    (folder / "DIFF.md").write_bytes(report.encode("utf-8"))
    manifest.append(dict(item="DIFF.md", rows=0, sha256=sha(report.encode("utf-8"))))
    for path in [MAPPING, OUT_DIR / "ntc.csv", OUT_DIR / "candidates.csv"] + inputs:
        manifest.append(dict(item=f"input {path.name}", rows=0, sha256=sha(path.read_bytes())))
    pd.DataFrame(manifest).to_csv(folder / "staging_manifest.csv", index=False, lineterminator="\n")

    print(table_checks[["check", "failures"]].to_string(index=False))
    for item in table_checks[table_checks.failures > 0].itertuples():
        print(f"  {item.check}: {item.detail[:400]}")
    return 1 if table_checks.failures.sum() else 0


if __name__ == "__main__":
    sys.exit(main())
