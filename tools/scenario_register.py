"""
Maintain internal_analysis/scenarios/scenario_register.csv: the run status of every scenario
of the grid, one row per column of scenarios.csv.

    python tools/scenario_register.py           # rebuild from the grid, the local runs and the previous register
    python tools/scenario_register.py --r2      # also ask the store which scenarios EPM View serves
    python tools/scenario_register.py --note Int_BSSC_5y "launched on the server on 2 Nov"

Where each column comes from.

The grid columns (name, scope, case, sens, grid, input_version) come from scenario_grid.yaml
through tools/scenarios_build.py, so the register always lists exactly the scenarios of
scenarios.csv.

The result columns are read from the explorer runs under epm/output_view: a scenario found in
a run's merge_manifest.csv (or, for the runs merged before the merge tool, in its
input_scenarios.csv) is done; its zip, run date and objective come from the manifest and
solver_metrics.csv. The run date is the date in the zip name, else the merge date, else the
date in the run folder name. When a scenario sits in several runs the most recent wins.

The published columns say whether the store holds the scenario folder of that run, which is
what EPM View serves. They are refreshed only with --r2 (keys from tools/.env, never
printed); otherwise they are carried over from the previous register and said so.

The twin columns do the same for the archive column the scenario continues (the v3 or v4 run
of the same case), so the register also says which results are already available under the
old names.

status: to run, done (a local run holds the result), published (that run is on the store).
twin_status adds n/a (no archive twin) and no local result (twin in the archive only).

note is free text kept across rebuilds and set with --note.
"""
import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios_build as sb  # noqa: E402

REGISTER = sb.ROOT / "internal_analysis" / "scenarios" / "scenario_register.csv"
OUTPUT_VIEW = sb.ROOT / "epm" / "output_view"
ENV_FILE = sb.ROOT / "tools" / ".env"
COLUMNS = ["name", "scope", "case", "sens", "grid", "input_version",
           "status", "result_run", "source_zip", "run_date", "objective_busd", "published_run",
           "twin", "twin_status", "twin_run", "twin_zip", "twin_date", "twin_objective_busd", "twin_published_run",
           "note"]
DATE_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")


# ---------------------------------------------------------------------------- local runs
def date_in(text):
    m = DATE_RE.search(text or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def read_csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def scan_runs(folder):
    """{scenario: {run, zip, date, objective}} over the explorer runs, most recent run first."""
    results = {}
    runs = sorted((p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("_")),
                  key=lambda p: date_in(p.name), reverse=True)
    for run in runs:
        objective = {}
        metrics = run / "solver_metrics.csv"
        if metrics.is_file():
            for r in read_csv_rows(metrics):
                objective[r.get("scenario", "")] = r.get("Objective (Billion USD)", "")
        manifest = run / "merge_manifest.csv"
        if manifest.is_file():
            entries = [(r["scenario"], r.get("source_zip", ""), date_in(r.get("source_zip", "")) or (r.get("merged_at", "")[:10]))
                       for r in read_csv_rows(manifest)]
        else:
            scen = run / "input_scenarios.csv"
            if not scen.is_file():
                continue
            with open(scen, encoding="utf-8-sig", newline="") as fh:
                header = next(csv.reader(fh))
            entries = [(name, "", date_in(run.name)) for name in header[1:]]
        for name, zip_, date in entries:
            if name and name not in results:
                results[name] = dict(run=run.name, zip=zip_, date=date or date_in(run.name), objective=objective.get(name, ""))
    return results, [p.name for p in runs]


# ---------------------------------------------------------------------------- store
def load_env(path):
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def remote_scenarios(run_names):
    """{run: set of scenario folders the store holds}. Keys and bucket names stay out of the output."""
    import s3fs  # imported here so that the register builds without the store libraries
    load_env(ENV_FILE)
    branch = os.environ.get("EPM_BRANCH") or subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, cwd=sb.ROOT).stdout.strip()
    endpoint, bucket = os.environ["STORE_ENDPOINT"], os.environ["STORE_BUCKET"]
    fs = s3fs.S3FileSystem(anon=False, client_kwargs={"endpoint_url": endpoint})
    prefix = f"{bucket}/{branch}/epm/output"
    held = {}
    for run in run_names:
        try:
            held[run] = {Path(p).name for p in fs.ls(f"{prefix}/{run}", detail=False)}
        except FileNotFoundError:
            held[run] = set()
    print(f"store: branch {branch}, {sum(len(v) for v in held.values())} scenario folders over {len(run_names)} runs")
    return held


def unpublished(results, held):
    """Local results whose run is on the store without them: the explorer run is behind the local one."""
    return sorted(n for n, r in results.items() if held.get(r["run"]) and n not in held[r["run"]])


# ---------------------------------------------------------------------------- register
def previous_register(path):
    if not path.is_file():
        return {}
    return {r["name"]: r for r in read_csv_rows(path)}


def status_of(result, published):
    if not result:
        return "to run"
    return "published" if published else "done"


def build_register(spec, meta, archive, results, held, previous, notes):
    rows = []
    version = spec.get("input_version", "")
    for name, m in meta.items():
        prev = previous.get(name, {})
        res = results.get(name)
        twin = sb.archive_twin(name, m, archive)
        tres = results.get(twin) if twin else None

        def published_run(r, key):
            if not r:
                return ""
            if held is None:                      # offline: carry the previous answer over
                return prev.get(key, "") if prev.get(key, "") == r["run"] else ""
            return r["run"] if name_in_store(r) else ""

        def name_in_store(r, scen=None):
            return (scen or r["scen"]) in held.get(r["run"], set())

        if res:
            res = dict(res, scen=name)
        if tres:
            tres = dict(tres, scen=twin)
        pub = published_run(res, "published_run")
        tpub = published_run(tres, "twin_published_run")
        if twin and not tres:
            twin_status = "no local result"
        elif not twin:
            twin_status = "n/a"
        else:
            twin_status = status_of(tres, tpub)
        rows.append(dict(
            name=name, scope=m["scope"], case=m["case"], sens=m["sens"] or "", grid=m["grid"], input_version=version,
            status=status_of(res, pub), result_run=res["run"] if res else "", source_zip=res["zip"] if res else "",
            run_date=res["date"] if res else "", objective_busd=res["objective"] if res else "", published_run=pub,
            twin=twin or "", twin_status=twin_status, twin_run=tres["run"] if tres else "", twin_zip=tres["zip"] if tres else "",
            twin_date=tres["date"] if tres else "", twin_objective_busd=tres["objective"] if tres else "", twin_published_run=tpub,
            note=notes.get(name, prev.get("note", "")),
        ))
    return rows


def write_register(path, rows):
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def summary(rows):
    out = []
    for grid in sorted({r["grid"] for r in rows}, reverse=True):
        for scope in ("Int", "Ext"):
            sel = [r for r in rows if r["grid"] == grid and r["scope"] == scope]
            if not sel:
                continue
            counts = {s: sum(1 for r in sel if r["status"] == s) for s in ("published", "done", "to run")}
            twins = sum(1 for r in sel if r["twin_status"] in ("published", "done"))
            out.append(f"  {grid} {scope}: {len(sel)} scenarios, " + ", ".join(f"{n} {s}" for s, n in counts.items() if n)
                       + f"; {twins} with a result under the old name")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid", default=str(sb.GRID_YAML), help="scenario grid yaml")
    ap.add_argument("--r2", action="store_true", help="refresh the published columns from the store")
    ap.add_argument("--note", nargs=2, metavar=("NAME", "TEXT"), action="append", default=[], help="set the note of a scenario")
    args = ap.parse_args(argv)

    spec = sb.load_spec(args.grid)
    archive_path = sb.ROOT / spec["data_folder"] / spec["archive"]
    params, archive = sb.read_scenarios(archive_path)
    _, meta = sb.build_columns(spec, params)
    results, run_names = scan_runs(OUTPUT_VIEW) if OUTPUT_VIEW.is_dir() else ({}, [])
    print(f"local runs: {', '.join(run_names) or 'none'} ({len(results)} scenario results)")
    held = remote_scenarios(run_names) if args.r2 else None
    if held is None:
        print("published columns carried over from the previous register (use --r2 to refresh them)")
    else:
        behind = unpublished(results, held)
        if behind:
            print(f"local results not on the store ({len(behind)}): {', '.join(behind)}")
    previous = previous_register(REGISTER)
    notes = dict(args.note)
    unknown = sorted(set(notes) - set(meta))
    if unknown:
        raise SystemExit(f"--note: not scenarios of the grid: {unknown}")
    rows = build_register(spec, meta, archive, results, held, previous, notes)
    write_register(REGISTER, rows)
    print(f"wrote {REGISTER.relative_to(sb.ROOT)}: {len(rows)} rows, generated {dt.date.today().isoformat()}")
    print(summary(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
