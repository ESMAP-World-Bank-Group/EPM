"""
Run a list of scenarios several at a time on the server, one epm.py call per scenario.

    python tools/run_parallel.py --scenarios scenarios_v62.csv --prefix v62c5 --date 20261127 --jobs 4 -S Int_Base_v62_5y Int_BSSC_v62_5y
    python tools/run_parallel.py ... --list        # print the plan and what is already there

Same runs as tools/run_sequential.py: one epm.py call per scenario with --cpu 1, the same labels
and one zip per scenario, epm/output/<prefix>_<scenario>_<date>.zip, ready for tools/merge_runs.py.
The only difference is that up to --jobs calls are alive at the same time. A scenario starts as
soon as a slot is free, so the runs do not wait for a whole group to end.

For the server only. Several solves side by side do not fit in laptop memory, so on Windows the
script refuses more than one job: use tools/run_sequential.py there.

Sizing. Each solve uses the threads of its solver option file (8 in cplex_s5.opt) and several GB
of memory, so --jobs times these must fit in the machine. Two guards help: while other scenarios
run, a new one starts only when at least --min_free_gb of memory are available (read from
/proc/meminfo), and two starts are at least --stagger seconds apart. The guards do not see the
memory a run takes later in its solve, so the first use of a new --jobs value should be watched:
the progress lines give the free memory and the last line gives the lowest value seen.

Resumable. A scenario whose zip exists and holds its epmresults.gdx is skipped, so the same
command can be relaunched after an interruption or a failure.

Stopping. Ctrl+C stops every scenario that is running, GAMS included. A closed terminal or a kill
of this script does the same.

Not next to another batch. The script refuses to start while a GAMS process is alive, whoever
owns it (--allow_running lifts the check).

Tracking. Next to the logs (--work, default epm/output/_parallel/<prefix>_<date>) it writes
status.csv, one row per attempt that ended, with the columns of run_sequential.py.
logs/<label>.out holds the console output of each epm.py call. The epm.log copied into each zip is
shared by the scenarios that ran at the same time.
"""
import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sequential as rs  # noqa: E402  labels, zip check, status row and log tail are shared

EPM = rs.EPM
POLL = 5  # seconds between two looks at the runs


def say(text):
    print(f"{dt.datetime.now():%H:%M:%S} {text}", flush=True)


def parse_meminfo(text):
    """Available memory in GB from the text of /proc/meminfo, None when the line is missing."""
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    return None


def free_gb():
    try:
        return parse_meminfo(Path("/proc/meminfo").read_text())
    except OSError:
        return None


def gams_names(ps_text):
    """Names of the GAMS processes in the output of `ps -eo comm=`."""
    return sorted({n.strip() for n in ps_text.splitlines() if n.strip().lower().startswith("gams")})


def gams_alive():
    """Names of the GAMS processes that are alive, an empty list when there is none."""
    if sys.platform == "win32":
        return ["gams.exe"] if rs.gams_alive() else []
    try:
        out = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True).stdout
    except OSError:
        return []
    return gams_names(out)


def raise_exit(signum, frame):
    raise SystemExit(f"signal {signum} received: stopping the runs")


def check_platform(jobs):
    if sys.platform == "win32" and jobs > 1:
        raise SystemExit("several runs side by side are for the server: on a laptop use tools/run_sequential.py")


def build_cmd(name, label, args):
    return [sys.executable, "epm.py", "--folder_input", args.folder_input, "--scenarios", args.scenarios, "-S", name,
            "--cpu", "1", "--simulation_label", label, "--output_zip", "--no_plot_dispatch"]


def start_run(name, label, args, work):
    out_path = work / "logs" / (label + ".out")
    cmd = build_cmd(name, label, args)
    fh = open(out_path, "w", encoding="utf-8", errors="replace")
    fh.write(" ".join(cmd) + "\n")
    fh.flush()
    # EPM_LOG_INITIALIZED keeps a starting epm.py from deleting the epm.log the other runs write to.
    env = dict(os.environ, EPM_LOG_INITIALIZED="1")
    # Own process group, so that stop_all reaches the GAMS process of the run as well.
    own_group = {"start_new_session": True} if os.name == "posix" else {}
    proc = subprocess.Popen(cmd, cwd=EPM, stdout=fh, stderr=subprocess.STDOUT, env=env, **own_group)
    say(f"start {name}")
    return dict(name=name, label=label, proc=proc, fh=fh, out=out_path, start=dt.datetime.now(),
                gams_log=EPM / "output" / label / name / f"{name}_main.log")


def close_run(run, work):
    """Write the status row of a run that ended and say whether its zip is good."""
    run["fh"].close()
    code, end = run["proc"].returncode, dt.datetime.now()
    summary = ""
    for line in run["out"].read_text(encoding="utf-8", errors="replace").splitlines():
        m = rs.SUMMARY_RE.search(line)
        if m:
            summary = m.group(1)
    z = EPM / "output" / (run["label"] + ".zip")
    good = code == 0 and rs.zip_ok(z, run["name"])
    minutes = (end - run["start"]).total_seconds() / 60
    rs.append_status(work / "status.csv", dict(
        scenario=run["name"], label=run["label"], start=f"{run['start']:%Y-%m-%d %H:%M:%S}",
        end=f"{end:%Y-%m-%d %H:%M:%S}", minutes=f"{minutes:.1f}", exit=code,
        zip_mb=f"{z.stat().st_size / 1e6:.0f}" if z.is_file() else "", summary=summary))
    say(f"{'done' if good else 'FAILED'} {run['name']} exit {code} in {minutes:.1f} min {summary}")
    return good


def signal_run(run, sig):
    proc = run["proc"]
    if os.name == "posix":
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
    elif proc.poll() is None:
        # Windows: the whole process tree, so that GAMS goes with its epm.py.
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)


def stop_all(running, grace=10):
    """Stop every run that is still alive, with its GAMS process."""
    alive = [run for run in running if run["proc"].poll() is None]
    for run in alive:
        signal_run(run, signal.SIGTERM)
    deadline = time.time() + grace
    for run in alive:
        try:
            run["proc"].wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            pass
    for run in alive:
        if os.name == "posix":
            signal_run(run, signal.SIGKILL)  # whatever is left of the group, GAMS included
        run["fh"].close()
    if alive:
        say("stopped: " + " ".join(run["name"] for run in alive))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--folder_input", default="data_blacksea", help="input folder under epm/input")
    ap.add_argument("--scenarios", required=True, help="scenario file of the input folder, for instance scenarios_v62.csv")
    ap.add_argument("-S", dest="names", nargs="+", required=True, help="scenarios, in the order to start")
    ap.add_argument("--prefix", required=True, help="label prefix, for instance v62c5")
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"), help="date of the labels, YYYYMMDD")
    ap.add_argument("--jobs", type=int, default=4, help="scenarios alive at the same time")
    ap.add_argument("--min_free_gb", type=float, default=20,
                    help="memory that must be available to start a scenario next to others, 0 for no check")
    ap.add_argument("--stagger", type=float, default=30, help="seconds between two starts")
    ap.add_argument("--allow_running", action="store_true", help="start even if a GAMS process is alive")
    ap.add_argument("--work", type=Path, help="folder of the logs and of status.csv")
    ap.add_argument("--progress", type=float, default=5, help="minutes between two progress reports, 0 for none")
    ap.add_argument("--list", action="store_true", help="print the plan and stop")
    args = ap.parse_args(argv)

    if len(set(args.names)) != len(args.names):
        raise SystemExit("a scenario is listed twice")
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    labels = {n: f"{args.prefix}_{n}_{args.date}" for n in args.names}
    todo = [n for n in args.names if not rs.zip_ok(EPM / "output" / (labels[n] + ".zip"), n)]
    print(f"plan: {len(args.names)} scenarios, {len(args.names) - len(todo)} already there, "
          f"{len(todo)} to run, {args.jobs} at a time", flush=True)
    if args.list:
        for n in args.names:
            print(f"  {'to run' if n in todo else 'there '} {labels[n]}")
        return 0
    check_platform(args.jobs)
    if not args.allow_running:
        alive = gams_alive()
        if alive:
            raise SystemExit(f"a GAMS process is alive ({', '.join(alive)}): not launching "
                             "(--allow_running lifts this check)")
        try:
            (EPM / "epm.log").unlink()  # as a lone epm.py does when it starts
        except OSError:
            pass
    work = args.work or EPM / "output" / "_parallel" / f"{args.prefix}_{args.date}"
    (work / "logs").mkdir(parents=True, exist_ok=True)

    # A closed terminal or a kill of this script stops the runs as Ctrl+C does.
    for signal_name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), raise_exit)
    rs.keep_awake(True)
    queue, running, failed = list(todo), [], []
    t0 = time.time()
    last_start, due, lowest, waiting = 0.0, t0 + args.progress * 60, None, False
    try:
        while queue or running:
            for run in [r for r in running if r["proc"].poll() is not None]:
                running.remove(run)
                if not close_run(run, work):
                    failed.append(run["name"])
            free = free_gb()
            if free is not None:
                lowest = free if lowest is None else min(lowest, free)
            if queue and len(running) < args.jobs and time.time() - last_start >= args.stagger:
                if running and free is not None and free < args.min_free_gb:
                    if not waiting:
                        say(f"waiting to start {queue[0]}: {free:.0f} GB free, {args.min_free_gb:.0f} needed")
                        waiting = True
                else:
                    name = queue.pop(0)
                    running.append(start_run(name, labels[name], args, work))
                    last_start, waiting = time.time(), False
            if args.progress and time.time() >= due:
                due += args.progress * 60
                for run in running:
                    minutes = (dt.datetime.now() - run["start"]).total_seconds() / 60
                    say(f"  {run['name']} {minutes:.0f} min | {rs.last_line(run['gams_log'])}")
                memory = "" if free is None else f", {free:.0f} GB free"
                say(f"  {len(running)} running, {len(queue)} waiting{memory}")
            if queue or running:
                time.sleep(POLL)
    except BaseException:
        stop_all(running)
        raise
    finally:
        rs.keep_awake(False)
    text = f"finished: {len(todo) - len(failed)} of {len(todo)} done in {(time.time() - t0) / 60:.0f} min"
    if lowest is not None:
        text += f", lowest free memory {lowest:.0f} GB"
    if failed:
        text += ", failed: " + " ".join(failed)
    print(text, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
