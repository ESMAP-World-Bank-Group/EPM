"""
Merge several EPM run zips into one run folder that EPM View and the report tools read.

A server run comes back as a zip that already has the layout of an output_view run:
one folder per scenario (epmresults.gdx plus output_csv/) and a few run level tables
(summary.csv, summary_generators.csv, input_scenarios.csv, solver_metrics.csv) plus the
map layers. Merging is therefore a file operation. Nothing goes through GAMS or the
postprocessing, which does not fit in laptop memory beyond a handful of scenarios.

    python tools/merge_runs.py --target epm/output_view/robg_s5_v3_20261024 a.zip b.zip ...

What it guarantees.

The zips stay the source of truth. Each one is recorded in merge_manifest.csv with its
md5, and a copy of its run level tables is kept (gzipped) under _merge/<zip stem>/, so
the merged tables can always be rebuilt from what the folder holds.

Rerunning is safe. A scenario already merged from the same zip is skipped. The same
scenario name coming from a different zip is refused unless --replace is given, so two
runs can never be mixed silently.

Nothing half written. Every scenario is extracted into a temporary folder and renamed
into place only once complete, and every run level file is written next to its target
and then swapped in. The manifest is written last.

Each scenario is checked before it is accepted: it must be a column of the zip's
input_scenarios.csv, it must hold epmresults.gdx and output_csv/, and its name must not
leave the target folder. The zip CRC is checked on extraction.

The tables are read and written as text, so the numbers come out exactly as EPM wrote
them. Missing cells (a parameter one batch has and another has not) stay empty.

Ignored from each zip: img/ and SummaryHeatmap.pdf, which describe one batch only. The
map layers of the first zip become the run layers (EPM View falls back on them only
when a scenario has no layers of its own); the others are kept under _merge/.
"""
import argparse
import csv
import gzip
import hashlib
import io
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

MANIFEST = "merge_manifest.csv"
MANIFEST_COLS = ["scenario", "source_zip", "zip_md5", "zip_bytes", "merged_at"]
WIDE_TABLES = ["summary.csv", "summary_generators.csv"]   # key columns + one column per scenario
LAYERS = ["zones.geojson", "linestring_countries.geojson"]
SKIP_ROOT = {"img", "SummaryHeatmap.pdf"}
REQUIRED_IN_SCENARIO = ["epmresults.gdx", "output_csv/"]


def md5_of(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_text_csv(data, index_col=None):
    """Read a CSV as text: no float reformatting, empty stays empty."""
    return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False,
                       index_col=index_col)


def write_atomic(path, write):
    """Write through a sibling temporary file, then swap it in."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def to_csv_atomic(df, path, index=False):
    write_atomic(path, lambda p: df.to_csv(p, index=index, quoting=csv.QUOTE_MINIMAL))


def load_manifest(target):
    p = target / MANIFEST
    if not p.exists():
        return pd.DataFrame(columns=MANIFEST_COLS)
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def inspect_zip(zpath):
    """Return (scenarios, members by scenario) after checking the zip layout."""
    z = zipfile.ZipFile(zpath)
    names = z.namelist()
    for n in names:
        parts = Path(n).parts
        if n.startswith(("/", "\\")) or ".." in parts or (parts and ":" in parts[0]):
            raise SystemExit(f"  {zpath.name}: unsafe path in zip: {n}")
    if "input_scenarios.csv" not in names:
        raise SystemExit(f"  {zpath.name}: no input_scenarios.csv at the zip root")
    scenarios = list(read_text_csv(z.read("input_scenarios.csv"), index_col=0).columns)
    members = {s: [n for n in names if n.startswith(s + "/")] for s in scenarios}
    for s, mem in members.items():
        missing = [r for r in REQUIRED_IN_SCENARIO if f"{s}/{r}" not in mem
                   and not any(m.startswith(f"{s}/{r}") for m in mem)]
        if missing:
            raise SystemExit(f"  {zpath.name}: scenario {s} lacks {', '.join(missing)} "
                             f"(failed solve?)")
    tops = {n.split("/")[0] for n in names}
    stray = sorted(t for t in tops - set(scenarios) - SKIP_ROOT
                   if t not in WIDE_TABLES + LAYERS + ["input_scenarios.csv",
                                                       "solver_metrics.csv", "epm.log"])
    if stray:
        print(f"  {zpath.name}: ignored at the root: {', '.join(stray)}")
    return z, scenarios, members


def extract_scenario(z, scen, members, target):
    final = target / scen
    tmp = target / f".tmp_{scen}"
    if tmp.exists():
        shutil.rmtree(tmp)
    for n in members:
        rel = n[len(scen) + 1:]
        if not rel:
            continue
        out = tmp / rel
        if n.endswith("/"):
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with z.open(n) as src, open(out, "wb") as dst:    # CRC is checked on read
            shutil.copyfileobj(src, dst, 1 << 22)
    if final.exists():
        shutil.rmtree(final)
    replace_with_retry(tmp, final)


def replace_with_retry(src, dst, tries=10):
    """Rename a freshly written folder. On Windows the antivirus or the indexer keeps a
    handle on large new files for a few seconds, and the rename is refused meanwhile."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(min(2 ** i, 15))


def keep_sources(z, stem, target):
    """Copy the run level tables of one zip under _merge/<stem>/ (gzipped, so that the
    R2 uploader, which sends *.csv only, leaves them out)."""
    d = target / "_merge" / stem
    d.mkdir(parents=True, exist_ok=True)
    for name in WIDE_TABLES + ["input_scenarios.csv", "solver_metrics.csv"]:
        if name in z.namelist():
            write_atomic(d / f"{name}.gz", lambda p, n=name: p.write_bytes(gzip.compress(z.read(n))))
    for name in LAYERS + ["epm.log"]:
        if name in z.namelist():
            write_atomic(d / name, lambda p, n=name: p.write_bytes(z.read(n)))


def source_table(target, stem, name, index_col=None):
    p = target / "_merge" / stem / f"{name}.gz"
    if not p.exists():
        return None
    return read_text_csv(gzip.decompress(p.read_bytes()), index_col=index_col)


def rebuild_tables(target, manifest):
    """Rebuild the run level tables from the kept sources, each scenario taken from the
    zip that owns it in the manifest."""
    order = list(dict.fromkeys(manifest["source_zip"]))
    owned = {stem: list(manifest.loc[manifest["source_zip"] == stem, "scenario"])
             for stem in order}
    scen_order = list(manifest["scenario"])

    for name in WIDE_TABLES:
        frames, keys = [], None
        for stem in order:
            df = source_table(target, stem, name)
            if df is None:
                continue
            k = [c for c in df.columns if c not in owned[stem]
                 and c not in scen_order]
            if keys is None:
                keys = k
            elif k != keys:
                raise SystemExit(f"  {name}: key columns differ in {stem}: {k} against {keys}")
            if df.duplicated(keys).any():
                raise SystemExit(f"  {name}: duplicate keys in {stem}")
            cols = [s for s in owned[stem] if s in df.columns]
            frames.append(df.set_index(keys)[cols])
        if not frames:
            continue
        merged = pd.concat(frames, axis=1, join="outer", sort=False)
        merged = merged[[s for s in scen_order if s in merged.columns]].reset_index()
        to_csv_atomic(merged.fillna(""), target / name)
        print(f"  {name}: {len(merged):,} rows, {merged.shape[1] - len(keys)} scenarios")

    frames = []
    for stem in order:
        df = source_table(target, stem, "input_scenarios.csv", index_col=0)
        if df is not None:
            frames.append(df[[s for s in owned[stem] if s in df.columns]])
    inp = pd.concat(frames, axis=1, join="outer", sort=False)[scen_order]
    to_csv_atomic(inp.fillna(""), target / "input_scenarios.csv", index=True)
    print(f"  input_scenarios.csv: {inp.shape[0]} parameters, {inp.shape[1]} scenarios")

    frames = []
    for stem in order:
        df = source_table(target, stem, "solver_metrics.csv")
        if df is not None:
            frames.append(df[df["scenario"].isin(owned[stem])])
    if frames:
        sm = pd.concat(frames, ignore_index=True)
        sm = sm.set_index("scenario").loc[[s for s in scen_order if s in set(sm["scenario"])]]
        to_csv_atomic(sm.reset_index(), target / "solver_metrics.csv")
        lacking = [s for s in scen_order if s not in sm.index]
        print(f"  solver_metrics.csv: {len(sm)} scenarios"
              + (f" (no metrics for {', '.join(lacking)})" if lacking else ""))

    first = target / "_merge" / order[0]
    for name in LAYERS:
        if (first / name).exists():
            shutil.copyfile(first / name, target / name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("zips", nargs="+", type=Path, help="run zips, in the order to merge")
    ap.add_argument("--target", required=True, type=Path, help="merged run folder")
    ap.add_argument("--replace", action="store_true",
                    help="allow a scenario already merged from another zip to be replaced")
    ap.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
    args = ap.parse_args()

    target = args.target
    manifest = load_manifest(target) if target.exists() else pd.DataFrame(columns=MANIFEST_COLS)

    plan = []      # (zip path, stem, md5, zipfile, scenarios to extract, members)
    seen = {}
    for zpath in args.zips:
        if not zpath.is_file():
            raise SystemExit(f"  not found: {zpath}")
        stem = zpath.stem
        digest = md5_of(zpath)
        z, scenarios, members = inspect_zip(zpath)
        todo = []
        for s in scenarios:
            if s in seen:
                raise SystemExit(f"  {s} is in both {seen[s]} and {zpath.name}")
            seen[s] = zpath.name
            row = manifest.loc[manifest["scenario"] == s]
            if len(row) and row["zip_md5"].iloc[0] == digest:
                continue                                           # already merged
            if len(row) and not args.replace:
                raise SystemExit(f"  {s} is already merged from {row['source_zip'].iloc[0]} "
                                 f"(md5 {row['zip_md5'].iloc[0][:8]}); {zpath.name} "
                                 f"(md5 {digest[:8]}) differs. Use --replace to take it.")
            todo.append(s)
        print(f"  {zpath.name}: md5 {digest[:8]}, {len(scenarios)} scenarios, "
              f"{len(todo)} to extract")
        plan.append((zpath, stem, digest, z, todo, members))

    if args.dry_run:
        print("  dry run: nothing written")
        return

    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    for zpath, stem, digest, z, todo, members in plan:
        keep_sources(z, stem, target)
        for s in todo:
            extract_scenario(z, s, members[s], target)
            manifest = manifest[manifest["scenario"] != s]
            manifest = pd.concat([manifest, pd.DataFrame([{
                "scenario": s, "source_zip": stem, "zip_md5": digest,
                "zip_bytes": str(zpath.stat().st_size), "merged_at": stamp}])],
                ignore_index=True)
            print(f"    + {s}")
        z.close()

    rebuild_tables(target, manifest)
    to_csv_atomic(manifest[MANIFEST_COLS], target / MANIFEST)
    print(f"  {MANIFEST}: {len(manifest)} scenarios from "
          f"{manifest['source_zip'].nunique()} zip(s) -> {target}")


if __name__ == "__main__":
    sys.exit(main())
