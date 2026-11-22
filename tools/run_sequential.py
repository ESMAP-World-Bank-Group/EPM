"""
Run a list of scenarios one after the other on a laptop, one epm.py call per scenario.

    python tools/run_sequential.py --scenarios scenarios_v6.csv --prefix v6c5 --date 20261121 -S Int_Base_5y Ext_Base_5y
    python tools/run_sequential.py ... --list        # print the plan and what is already there

Why one call per scenario. The postprocessing of epm.py loads every scenario of the run folder
at once, which does not fit in laptop memory beyond a handful of scenarios. With one call per
scenario it never holds more than one, a failed solve does not stop the others, and each
scenario comes back as its own zip, epm/output/<prefix>_<scenario>_<date>.zip, ready for
tools/merge_runs.py.

Resumable. A scenario whose zip exists and holds its epmresults.gdx is skipped, so the same
command can be relaunched after an interruption or a failure.

One run at a time. The script refuses to start while a gams.exe process is alive, and it asks
Windows not to sleep until it ends. That request only holds on mains power with the lid open: a
laptop with modern standby suspends the solve on battery or when the lid closes, and the run
resumes when the machine wakes.

Progress. While a scenario runs, one line every five minutes (--progress) gives the minutes elapsed
and the last line of its GAMS log, epm/output/<label>/<scenario>/<scenario>_main.log, which can
also be followed with tail -f.

Tracking. Next to the logs (--work, default epm/output/_sequential/<prefix>_<date>) it writes
status.csv, one row per attempt: start, end, minutes, exit code, zip size and the solver summary
line of epm.py.
"""
import argparse
import csv
import ctypes
import datetime as dt
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPM = ROOT / "epm"
FIELDS = ["scenario", "label", "start", "end", "minutes", "exit", "zip_mb", "summary"]
SUMMARY_RE = re.compile(r"solver summary .*?: (objective .*)$")


def gams_alive():
    if sys.platform != "win32":
        return False
    out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    return "gams.exe" in out


def keep_awake(on):
    """Ask Windows not to sleep while the runs are alive (released when the process ends)."""
    if sys.platform != "win32":
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def zip_ok(path, name):
    """The zip exists and holds the result gdx of the scenario."""
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return any(n.replace("\\", "/").endswith(f"{name}/epmresults.gdx") for n in z.namelist())
    except zipfile.BadZipFile:
        return False


def append_status(path, row):
    new = not path.is_file()
    with open(path, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def last_line(path):
    """Last non-empty line of a text file, read from its tail."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 4096))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return lines[-1][:110] if lines else ""


def wait_with_progress(proc, log_path, start, every):
    """Wait for the run; every `every` seconds print the minutes elapsed and the last line of the GAMS log."""
    due = time.time() + every
    while True:
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        if every and time.time() >= due:
            due += every
            minutes = (dt.datetime.now() - start).total_seconds() / 60
            print(f"{dt.datetime.now():%H:%M:%S}   running {minutes:.0f} min | {last_line(log_path)}", flush=True)


def run_one(name, label, args, work):
    out_path = work / "logs" / (label + ".out")
    gams_log = EPM / "output" / label / name / f"{name}_main.log"
    cmd = [sys.executable, "epm.py", "--folder_input", args.folder_input, "--scenarios", args.scenarios, "-S", name,
           "--cpu", "1", "--simulation_label", label, "--output_zip", "--no_plot_dispatch"]
    start = dt.datetime.now()
    print(f"{start:%H:%M:%S} start {name}", flush=True)
    with open(out_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=EPM, stdout=fh, stderr=subprocess.STDOUT)
        try:
            code = wait_with_progress(proc, gams_log, start, args.progress * 60)
        except BaseException:
            proc.kill()
            raise
    end = dt.datetime.now()
    summary = ""
    for line in out_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SUMMARY_RE.search(line)
        if m:
            summary = m.group(1)
    z = EPM / "output" / (label + ".zip")
    good = code == 0 and zip_ok(z, name)
    append_status(work / "status.csv", dict(
        scenario=name, label=label, start=f"{start:%Y-%m-%d %H:%M:%S}", end=f"{end:%Y-%m-%d %H:%M:%S}",
        minutes=f"{(end - start).total_seconds() / 60:.1f}", exit=code,
        zip_mb=f"{z.stat().st_size / 1e6:.0f}" if z.is_file() else "", summary=summary))
    print(f"{end:%H:%M:%S} {'done' if good else 'FAILED'} {name} exit {code} {summary}", flush=True)
    return good


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--folder_input", default="data_blacksea", help="input folder under epm/input")
    ap.add_argument("--scenarios", required=True, help="scenario file of the input folder, for instance scenarios_v6.csv")
    ap.add_argument("-S", dest="names", nargs="+", required=True, help="scenarios, in the order to run")
    ap.add_argument("--prefix", required=True, help="label prefix, for instance v6c5")
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"), help="date of the labels, YYYYMMDD")
    ap.add_argument("--work", type=Path, help="folder of the logs and of status.csv")
    ap.add_argument("--progress", type=float, default=5, help="minutes between two progress lines, 0 for none")
    ap.add_argument("--list", action="store_true", help="print the plan and stop")
    args = ap.parse_args(argv)

    labels = {n: f"{args.prefix}_{n}_{args.date}" for n in args.names}
    todo = [n for n in args.names if not zip_ok(EPM / "output" / (labels[n] + ".zip"), n)]
    print(f"plan: {len(args.names)} scenarios, {len(args.names) - len(todo)} already there, {len(todo)} to run", flush=True)
    if args.list:
        for n in args.names:
            print(f"  {'to run' if n in todo else 'there '} {labels[n]}")
        return 0
    if gams_alive():
        raise SystemExit("a gams.exe process is alive: not launching")
    work = args.work or EPM / "output" / "_sequential" / f"{args.prefix}_{args.date}"
    (work / "logs").mkdir(parents=True, exist_ok=True)
    keep_awake(True)
    failed = []
    try:
        for n in todo:
            if not run_one(n, labels[n], args, work):
                failed.append(n)
            time.sleep(5)
    finally:
        keep_awake(False)
    print(f"finished: {len(todo) - len(failed)} of {len(todo)} done" + (f", failed: {' '.join(failed)}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
