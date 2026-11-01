"""
Build epm/input/data_blacksea/scenarios.csv from internal_analysis/scenarios/scenario_grid.yaml.

The yaml holds the scenario grid of the Black Sea RoBg model: one reference column (the latest
input version), the grids (five-year and two-year), the two scopes (Int: Romania and Bulgaria
modelled, Ext: the two as price zones), the cases and the sensitivities. Every column of
scenarios.csv is the reference overlaid by its grid, its scope, its case and its sensitivity,
in that order, and is named Scope_Case[_Sensitivity]_Grid.

    python tools/scenarios_build.py              # write scenarios.csv and the report
    python tools/scenarios_build.py --check      # exit 1 if scenarios.csv drifted from the yaml
    python tools/scenarios_build.py --show Int_GECO_GasHi_2y

What it guarantees.

The row list and the row order are those of scenarios_archive.csv (the pre-redesign file), so
the new file has exactly the parameters epm.py already knows. A parameter named in the yaml
that the archive does not have is an error.

The columns listed under identical_to_archive must equal the archive column row by row. The
build stops before writing anything if one of them does not.

Every file a column points to is checked on disk. A missing file is reported, not refused:
the Ext files are created by a later step and the build must be able to name them first.

The file is written atomically (temporary file, then replace) in the format of the archive:
UTF-8 with a byte order mark, LF line endings, no quoting.

The report (internal_analysis/scenarios/scenario_grid_report.md) lists every column with its
overrides against the reference, the archive column it continues and what changed since, and
the files still to create.
"""
import argparse
import csv
import datetime as dt
import hashlib
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GRID_YAML = ROOT / "internal_analysis" / "scenarios" / "scenario_grid.yaml"
REPORT_MD = ROOT / "internal_analysis" / "scenarios" / "scenario_grid_report.md"
PARAM_COL = "paramNames"
GRID_TOKEN = "{g}"

# Archive columns that the v3 EU price sensitivities were named after.
SENS_ARCHIVE_ALIAS = {"EUVeryLow": "VeryLow", "EUCBAM": "CBAM"}


# ---------------------------------------------------------------------------- reading
def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def read_table(path):
    """Return (header, rows) of a csv file, byte order mark tolerated, cells kept as text."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def read_scenarios(path):
    """Return (param order, {column: {param: value}}) of a scenarios.csv style file."""
    header, rows = read_table(path)
    if header[0] != PARAM_COL:
        raise SystemExit(f"{path}: first column must be {PARAM_COL}, found {header[0]!r}")
    params = [r[0] for r in rows]
    columns = {}
    for j, name in enumerate(header[1:], start=1):
        columns[name] = {r[0]: (r[j] if j < len(r) else "") for r in rows}
    return params, columns


def load_spec(path):
    with open(path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    for key in ("data_folder", "archive", "reference", "grids", "scopes", "cases", "sensitivities"):
        if key not in spec:
            raise SystemExit(f"{path}: missing block {key!r}")
    return spec


# ---------------------------------------------------------------------------- building
def overlay(values, layer, name, provenance):
    """Apply one layer of overrides; None means the config default (blank cell)."""
    for param, value in (layer or {}).items():
        values[param] = "" if value is None else str(value)
        provenance[param] = name


def build_columns(spec, params):
    """Return an ordered dict name -> column, and a dict name -> metadata for the report."""
    known = set(params)
    sens_cases = spec["sensitivities"]["cases"]
    columns, meta = {}, {}

    def add(name, grid, scope, case, sens):
        if name in columns:
            raise SystemExit(f"duplicate scenario name {name}")
        g = spec["grids"][grid]
        values, prov = {}, {}
        overlay(values, spec["reference"], "reference", prov)
        overlay(values, {"y": g["y"], "cplexfile": g.get("cplexfile")}, f"grid {grid}", prov)
        overlay(values, spec["scopes"][scope].get("overrides"), f"scope {scope}", prov)
        case_spec = spec["cases"][case]
        if scope not in case_spec:
            raise SystemExit(f"case {case} has no block for scope {scope}")
        overlay(values, case_spec[scope], f"case {case}", prov)
        if sens:
            sens_spec = {k: v for k, v in spec["sensitivities"][scope][sens].items() if k != "label"}
            overlay(values, sens_spec, f"sensitivity {sens}", prov)
        unknown = sorted(set(values) - known)
        if unknown:
            raise SystemExit(f"{name}: parameters not in the archive row list: {unknown}")
        values = {p: v.replace(GRID_TOKEN, g["token"]) for p, v in values.items()}
        columns[name] = {p: values.get(p, "") for p in params}
        meta[name] = dict(grid=grid, scope=scope, case=case, sens=sens or "", provenance=prov)

    for grid in spec["grids"]:
        for scope, sc in spec["scopes"].items():
            for case in sc["cases"]:
                add(f"{scope}_{case}_{grid}", grid, scope, case, None)
        for scope, sc in spec["scopes"].items():
            for sens in spec["sensitivities"].get(scope, {}):
                for case in sens_cases:
                    if case not in sc["cases"]:
                        raise SystemExit(f"sensitivity case {case} is not a case of scope {scope}")
                    add(f"{scope}_{case}_{sens}_{grid}", grid, scope, case, sens)
    return columns, meta


def reference_column(spec, params, grid):
    """The reference overlaid by one grid, as the report's baseline for the overrides."""
    g = spec["grids"][grid]
    values = {}
    overlay(values, spec["reference"], "reference", {})
    overlay(values, {"y": g["y"], "cplexfile": g.get("cplexfile")}, "grid", {})
    return {p: values.get(p, "").replace(GRID_TOKEN, g["token"]) for p in params}


def archive_twin(name, m, archive):
    """The archive column a scenario continues, by the naming rules of the v3 and v4 runs."""
    scope, case, sens, grid = m["scope"], m["case"], m["sens"], m["grid"]
    candidates = []
    if grid == "5y":
        if scope == "Int" and case == "Base" and not sens:
            candidates.append("S5_Base_v4")
        suffix = f"_{SENS_ARCHIVE_ALIAS.get(sens, sens)}" if sens else ""
        prefix = "S5_" if scope == "Int" else "S5_Ext_"
        candidates.append(f"{prefix}{case}{suffix}_v3")
    elif grid == "2y" and scope == "Int" and not sens:
        candidates.append(f"LC_{case}_RoBg_2y_v3")
    for c in candidates:
        if c in archive:
            return c
    return None


def column_diff(a, b):
    """Parameters whose value differs between two columns, as (param, a value, b value)."""
    return [(p, a[p], b[p]) for p in a if a[p] != b.get(p, "")]


# ---------------------------------------------------------------------------- writing
def write_scenarios(path, params, columns):
    names = list(columns)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        w.writerow([PARAM_COL] + names)
        for p in params:
            w.writerow([p] + [columns[n][p] for n in names])
    os.replace(tmp, path)


def missing_files(columns, data_folder):
    """{relative file: [scenario names]} for every file a column names that is not on disk."""
    missing = {}
    for name, col in columns.items():
        for value in col.values():
            if value and (".csv" in value or ".opt" in value) and not (data_folder / value).is_file():
                missing.setdefault(value, []).append(name)
    return missing


def fmt(value):
    return value if value else "(default)"


def write_report(path, spec, params, columns, meta, archive, data_folder, archive_path, out_path):
    missing = missing_files(columns, data_folder)
    files = sorted({v for c in columns.values() for v in c.values() if v and (".csv" in v or ".opt" in v)})
    lines = []
    add = lines.append
    add("# Scenario grid report")
    add("")
    add(f"Generated on {dt.date.today().isoformat()} by tools/scenarios_build.py from "
        f"internal_analysis/scenarios/scenario_grid.yaml.")
    add("")
    add(f"- scenarios.csv: {len(columns)} columns, {len(params)} rows, md5 {md5(out_path)}")
    add(f"- archive: {archive_path.name}, {len(archive)} columns, md5 {md5(archive_path)}")
    by_grid = {g: sum(1 for m in meta.values() if m['grid'] == g) for g in spec["grids"]}
    by_scope = {s: sum(1 for m in meta.values() if m['scope'] == s) for s in spec["scopes"]}
    add(f"- columns per grid: {', '.join(f'{g} {n}' for g, n in by_grid.items())}; "
        f"per scope: {', '.join(f'{s} {n}' for s, n in by_scope.items())}")
    add(f"- files referenced: {len(files)}, of which {len(missing)} not on disk yet (listed at the end)")
    add("")
    add("## Reference")
    add("")
    add("Int_Base on each grid. Every other column is this column plus the overrides listed below.")
    add("")
    add("| Parameter | " + " | ".join(spec["grids"]) + " |")
    add("|---|" + "---|" * len(spec["grids"]))
    refs = {g: reference_column(spec, params, g) for g in spec["grids"]}
    for p in params:
        vals = [refs[g][p] for g in spec["grids"]]
        if any(vals):
            add(f"| {p} | " + " | ".join(fmt(v) for v in vals) + " |")
    add("")
    add("## Columns")
    add("")
    add("Overrides are given against the reference of the same grid. The archive column is the "
        "one the scenario continues; the last column says what changed since that run.")
    add("")
    add("| Scenario | Scope | Case | Sensitivity | Grid | Overrides vs reference | Archive column | Changed since |")
    add("|---|---|---|---|---|---|---|---|")
    for name, col in columns.items():
        m = meta[name]
        over = column_diff(col, refs[m["grid"]])
        over_txt = "; ".join(f"{p}: {fmt(v)}" for p, v, _ in over) or "none"
        twin = archive_twin(name, m, archive)
        if twin:
            changed = column_diff(col, archive[twin])
            changed_txt = "identical" if not changed else "; ".join(f"{p}: {fmt(a)} to {fmt(v)}" for p, v, a in changed)
        else:
            twin, changed_txt = "n/a", "new scenario"
        add(f"| {name} | {m['scope']} | {m['case']} | {m['sens'] or 'n/a'} | {m['grid']} | {over_txt} | {twin} | {changed_txt} |")
    add("")
    add("## Files to create")
    add("")
    if not missing:
        add("Every file referenced exists in the data folder.")
    else:
        add("| File | Used by |")
        add("|---|---|")
        for f, names in sorted(missing.items()):
            add(f"| {f} | {len(names)} columns ({names[0]}, ...) |" if len(names) > 1 else f"| {f} | {names[0]} |")
    add("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return missing


# ---------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid", default=str(GRID_YAML), help="scenario grid yaml")
    ap.add_argument("--check", action="store_true", help="compare the yaml with scenarios.csv on disk, write nothing")
    ap.add_argument("--show", metavar="NAME", help="print one resolved column and exit")
    args = ap.parse_args(argv)

    spec = load_spec(args.grid)
    data_folder = ROOT / spec["data_folder"]
    archive_path = data_folder / spec["archive"]
    out_path = data_folder / "scenarios.csv"
    params, archive = read_scenarios(archive_path)
    columns, meta = build_columns(spec, params)

    if args.show:
        if args.show not in columns:
            raise SystemExit(f"{args.show}: not a column of the grid")
        m = meta[args.show]
        for p in params:
            if columns[args.show][p]:
                print(f"{p:32s} {columns[args.show][p]:60s} {m['provenance'].get(p, '')}")
        return 0

    for name, twin in (spec.get("identical_to_archive") or {}).items():
        if twin not in archive:
            raise SystemExit(f"identical_to_archive: {twin} is not an archive column")
        diff = column_diff(columns[name], archive[twin])
        if diff:
            for p, v, a in diff:
                print(f"  {p}: built {fmt(v)}, archive {fmt(a)}")
            raise SystemExit(f"{name} differs from archive column {twin} on {len(diff)} rows, nothing written")
        print(f"{name} equals archive column {twin} on all {len(params)} rows")

    if args.check:
        if not out_path.is_file():
            raise SystemExit(f"{out_path} does not exist")
        disk_params, disk = read_scenarios(out_path)
        drift = []
        if disk_params != params:
            drift.append("row list differs from the archive row list")
        if list(disk) != list(columns):
            drift.append(f"column list differs: {len(disk)} on disk, {len(columns)} in the yaml")
        for name in columns:
            if name in disk:
                drift += [f"{name} {p}: disk {fmt(d)}, yaml {fmt(v)}" for p, v, d in column_diff(columns[name], disk[name])]
        for line in drift:
            print("  " + line)
        print(f"{out_path.relative_to(ROOT)}: {'drifted, ' + str(len(drift)) + ' differences' if drift else 'matches the yaml'}")
        missing = missing_files(columns, data_folder)
        if missing:
            print(f"{len(missing)} referenced files are not on disk yet")
        return 1 if drift else 0

    write_scenarios(out_path, params, columns)
    missing = write_report(REPORT_MD, spec, params, columns, meta, archive, data_folder, archive_path, out_path)
    print(f"wrote {out_path.relative_to(ROOT)}: {len(columns)} columns, {len(params)} rows, md5 {md5(out_path)}")
    print(f"wrote {REPORT_MD.relative_to(ROOT)}")
    if missing:
        print(f"{len(missing)} referenced files are not on disk yet (see the report):")
        for f in sorted(missing):
            print("  " + f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
