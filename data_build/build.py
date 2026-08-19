# -*- coding: utf-8 -*-
"""Build an EPM input folder from a reference folder and a YAML config.

The engine is generic: it holds no country, zone, season or plant name. Everything
specific to a deployment lives in the build YAML and in the CSV lookup tables under
mappings/.

    python data_build/build.py --config data_build/build_casa.yaml --check
    python data_build/build.py --config data_build/build_casa.yaml --apply

--check builds into a temporary folder and compares it with the existing target,
without writing anything. That is the test that matters: the target must be
reproducible. --apply replaces the target with the build result.

Available verbs (the "action" field of a resource):

    keep          do nothing, the file is carried over as is        (default)
    empty         keep the header row only
    rows          replace the whole content with the given rows
    drop_where    drop the rows where a column holds a given value
    drop_column   drop a column
    table         replace the whole file with a table built by an extractor

A file absent from the YAML is simply copied over.
"""

import argparse
import csv
import filecmp
import json
import os
import shutil
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


# ── CSV reading / writing ─────────────────────────────────────────────────────

def read_csv(path):
    """Return (header, rows). utf-8-sig absorbs a possible BOM."""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def write_csv(path, header, rows):
    """Write UTF-8 with LF line endings and no trailing blank line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def col_index(header, name, path):
    """Index of a column, by name, case and whitespace insensitive."""
    want = str(name).strip().lower()
    for i, h in enumerate(header):
        if str(h).strip().lower() == want:
            return i
    raise KeyError("column '{}' missing from {} (header: {})".format(name, path, header))


# ── The verbs ─────────────────────────────────────────────────────────────────

def act_keep(header, rows, spec, path):
    return header, rows


def act_empty(header, rows, spec, path):
    return header, []


def act_rows(header, rows, spec, path):
    """Replace the content. A scalar counts as a single-cell row."""
    new = []
    for item in spec.get("rows", []):
        new.append([item] if not isinstance(item, (list, tuple)) else list(item))
    return header, new


def act_drop_where(header, rows, spec, path):
    i = col_index(header, spec["column"], path)
    values = spec["value"]
    if not isinstance(values, (list, tuple)):
        values = [values]
    values = {str(v).strip() for v in values}
    return header, [r for r in rows if len(r) <= i or str(r[i]).strip() not in values]


def act_drop_column(header, rows, spec, path):
    i = col_index(header, spec["column"], path)
    cut = lambda r: [c for j, c in enumerate(r) if j != i]
    return cut(header), [cut(r) for r in rows]


def act_table(header, rows, spec, path):
    """Replace the whole file with a table produced outside the engine.

    The "from" field points at a CSV, relative to data_build/, written by one of the
    extract_*.py scripts. The engine does not know what is inside: it only checks
    that the table is not empty and that every row has the same width.
    """
    src = os.path.join(HERE, str(spec["from"]).replace("/", os.sep))
    if not os.path.isfile(src):
        raise IOError("resource '{}': table missing ({}). Run the matching "
                      "extractor.".format(path, spec["from"]))
    new_header, new_rows = read_csv(src)
    if not new_rows:
        raise IOError("resource '{}': table {} is empty.".format(path, spec["from"]))
    widths = {len(r) for r in new_rows} | {len(new_header)}
    if len(widths) > 1:
        raise IOError("resource '{}': table {} has rows of differing widths ({})."
                      .format(path, spec["from"], sorted(widths)))
    return new_header, new_rows


ACTIONS = {
    "keep": act_keep,
    "empty": act_empty,
    "rows": act_rows,
    "drop_where": act_drop_where,
    "drop_column": act_drop_column,
    "table": act_table,
}


# ── Resolving resource paths ──────────────────────────────────────────────────

def resource_paths(datapackage, source_dir):
    """resource name -> relative path.

    Two authorities, in this order. datapackage.json holds the vocabulary shared by
    every deployment, but it is incomplete: as of 2026-08, 12 parameters read by the
    Central Asia config.csv are missing from it (pReserveSeasonFlag,
    pPlanningReserveMarginZone, pContractedTrade*, and so on). The deployment
    config.csv, on the other hand, is exact by construction since it is what GAMS
    actually reads.
    """
    known = {}
    if os.path.isfile(datapackage):
        with open(datapackage, "r", encoding="utf-8") as fh:
            pkg = json.load(fh)
        known.update({r["name"]: r["path"] for r in pkg.get("resources", []) if "name" in r})

    cfg_path = os.path.join(source_dir, "config.csv")
    if os.path.isfile(cfg_path):
        _, rows = read_csv(cfg_path)
        for r in rows:                      # metadata, paramNames, file
            if len(r) >= 3 and r[1].strip() and r[2].strip().endswith(".csv"):
                known.setdefault(r[1].strip(), r[2].strip())
    return known


def resolve(name, spec, known, source_dir):
    """Relative path of a resource: explicit 'path', else the datapackage."""
    rel = spec.get("path") or known.get(name)
    if rel is None:
        raise KeyError("unknown resource '{}': missing from datapackage.json and "
                       "no 'path' field".format(name))
    rel = rel.replace("\\", "/")
    if rel.startswith(".."):
        return None, rel   # shared resource, outside the deployment
    if not os.path.isfile(os.path.join(source_dir, rel.replace("/", os.sep))):
        raise IOError("resource '{}': file not found in the reference ({})"
                      .format(name, rel))
    return rel, rel


# ── Build ─────────────────────────────────────────────────────────────────────

def build(cfg, out_dir):
    """Copy the reference, then apply the transformations. Return the report."""
    source_dir = os.path.join(REPO, cfg["deployment"]["source"].replace("/", os.sep))
    datapackage = os.path.join(REPO, cfg["deployment"]["datapackage"].replace("/", os.sep))

    if not os.path.isdir(source_dir):
        raise IOError("reference folder not found: " + source_dir)

    # 1. Copy everything, so that nothing is lost (geojson, cplex options, etc.).
    # copyfile rather than copy2: only the content is copied. Copying the metadata
    # fails on Windows in this environment (WinError 127), and the reference file
    # timestamps are worthless to the build anyway.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(source_dir, out_dir, copy_function=shutil.copyfile)

    known = resource_paths(datapackage, source_dir)
    report = []

    # 2. Apply the declared transformations.
    for name, spec in (cfg.get("resources") or {}).items():
        spec = spec or {}
        action = spec.get("action", "keep")
        if action not in ACTIONS:
            raise KeyError("resource '{}': unknown action '{}' (known: {})"
                           .format(name, action, ", ".join(sorted(ACTIONS))))

        rel, shown = resolve(name, spec, known, source_dir)
        # Every descriptive field of the YAML is carried over as is into the report:
        # the engine has no business knowing the tracking vocabulary.
        entry = dict(spec)
        entry.pop("rows", None)
        entry.update({"resource": name, "path": shown, "action": action})

        if rel is None:                       # shared resource, left untouched
            entry["rows_in"] = entry["rows_out"] = "-"
            entry["action"] = "shared"
            report.append(entry)
            continue

        path = os.path.join(out_dir, rel.replace("/", os.sep))
        header, rows = read_csv(path)
        entry["rows_in"] = len(rows)

        if action != "keep":
            header, rows = ACTIONS[action](header, rows, spec, rel)
            write_csv(path, header, rows)

        entry["rows_out"] = len(rows)
        report.append(entry)

    return report


# Files that live in the target but are not produced by the build: docs.py writes
# the page inside the deployment folder, next to the data it describes. They are
# neither compared nor destroyed.
GENERATED = ("DATA_SOURCES.html", "DATA_SOURCES.md")


def diff(a, b):
    """Files that differ between two trees, ignoring the generated documentation."""
    out = []
    for root, _, files in os.walk(a):
        for f in files:
            pa = os.path.join(root, f)
            rel = os.path.relpath(pa, a)
            pb = os.path.join(b, rel)
            if not os.path.exists(pb):
                out.append((rel, "missing from target"))
            elif not filecmp.cmp(pa, pb, shallow=False):
                out.append((rel, "content differs"))
    for root, _, files in os.walk(b):
        for f in files:
            if f in GENERATED:
                continue
            rel = os.path.relpath(os.path.join(root, f), b)
            if not os.path.exists(os.path.join(a, rel)):
                out.append((rel, "extra in target"))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--check", action="store_true",
                    help="build aside and compare, without writing anything")
    ap.add_argument("--apply", action="store_true",
                    help="replace the target with the build result")
    args = ap.parse_args()

    if not (args.check or args.apply):
        ap.error("choose --check or --apply")

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    target = os.path.join(REPO, cfg["deployment"]["target"].replace("/", os.sep))
    staging = os.path.join(tempfile.mkdtemp(prefix="epm_build_"), "out")

    report = build(cfg, staging)

    if args.check:
        if os.path.isdir(target):
            d = diff(staging, target)
            if d:
                print("MISMATCH between the build and the existing target:")
                for rel, why in d:
                    print("   {:<45s} {}".format(rel, why))
            else:
                print("The build reproduces the target byte for byte.")
        else:
            print("No target yet, nothing to compare.")
        print("(build discarded: {})".format(staging))
    else:
        # The target is replaced wholesale, so the generated page has to be carried
        # across, otherwise --apply would silently delete a file the build never made.
        kept = {}
        for f in GENERATED:
            p = os.path.join(target, f)
            if os.path.isfile(p):
                with open(p, "rb") as fh:
                    kept[f] = fh.read()
        if os.path.isdir(target):
            shutil.rmtree(target)
        shutil.move(staging, target)
        for f, blob in kept.items():
            with open(os.path.join(target, f), "wb") as fh:
                fh.write(blob)
        print("Written: " + target)
        if kept:
            print("Kept: {} (re-run docs.py to refresh)".format(", ".join(sorted(kept))))

    out = os.path.join(HERE, "build_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"config": os.path.basename(args.config),
                   "sources": cfg.get("sources", {}),
                   "resources": report}, fh, indent=2, ensure_ascii=False,
                  default=str)   # YAML dates are date objects
    print("Report: " + out)

    acted = sum(1 for r in report if r["action"] not in ("keep", "shared"))
    print("{} resources declared, {} transformed.".format(len(report), acted))


if __name__ == "__main__":
    main()
