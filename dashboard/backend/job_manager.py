"""
EPM Dashboard — Job Manager
Launches epm.py as a subprocess, streams logs, tracks job state.
"""

import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# In-memory job registry  {job_id: JobState}
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}


def _job_id() -> str:
    return str(uuid.uuid4())[:8]


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def get_all_jobs() -> list[dict]:
    return sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)


def get_latest_job() -> dict | None:
    jobs = get_all_jobs()
    return jobs[0] if jobs else None


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch_run(
    folder_input: str,
    cpu: int = 1,
    run_name: str = "",
    scenarios: list | None = None,
    modeltype: str = "MIP",
    analysis: str = "standard",
    output_opts: list | None = None,
    label: str = "",
    advanced: list | None = None,
    repo_root: Path | None = None,
) -> str:
    """
    Launch epm.py as a background subprocess.

    Parameters
    ----------
    folder_input : str
        Name of the input data folder (e.g. 'data_eapp').
    cpu : int
        Number of CPU cores to pass via --cpu.
    run_name : str
        Optional label for display in the UI.
    scenarios : list, optional
        Scenario names to run (passed as --selected_scenarios).
    modeltype : str
        'MIP' or 'RMIP'.
    analysis : str
        'standard', 'sensitivity', or 'montecarlo'.
    output_opts : list, optional
        Flags like ['reduced_output', 'output_zip', 'reduce_definition_csv'].
    label : str
        --simulation_label value.
    advanced : list, optional
        Flags like ['simple', 'debug', 'trace'].
    repo_root : Path, optional
        Root of the EPM repo. Defaults to two levels up from this file.

    Returns
    -------
    str
        job_id that can be polled via get_job().
    """
    if repo_root is None:
        repo_root = Path(__file__).parent.parent.parent  # dashboard/backend/ → EPM/

    epm_dir  = repo_root / "epm"          # epm.py lives here
    job_id   = _job_id()
    log_file = repo_root / "dashboard" / "assets" / f"job_{job_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    conda_env = "gams_env"
    scenarios   = scenarios   or []
    output_opts = output_opts or []
    advanced    = advanced    or []

    # Build the python command (runs inside the activated env)
    parts = [f"python -u epm.py --folder_input {folder_input}"]

    if modeltype and modeltype != "MIP":
        parts.append(f"--modeltype {modeltype}")

    if cpu and cpu > 1:
        parts.append(f"--cpu {cpu}")

    # Always pass scenarios file; add selected subset if specified
    parts.append("--scenarios scenarios.csv")
    if scenarios:
        parts.append("--selected_scenarios " + " ".join(scenarios))

    # Analysis mode
    if analysis == "sensitivity":
        parts.append("--sensitivity")
    elif analysis == "montecarlo":
        parts.append("--montecarlo")

    # Output options
    for opt in output_opts:
        parts.append(f"--{opt}")

    # Simulation label
    if label:
        parts.append(f"--simulation_label {label}")

    # Advanced flags
    if "simple" in advanced:
        parts.append("--simple")
    if "debug" in advanced:
        parts.append("--debug")
    if "trace" in advanced:
        parts.append("--trace")

    py_args = " ".join(parts)

    # Full shell command: activate env then run
    shell_cmd = f"conda activate {conda_env} && {py_args}"

    job = {
        "job_id":     job_id,
        "run_name":   run_name or f"Run {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "folder":     folder_input,
        "cwd":        str(epm_dir),
        "conda_env":  conda_env,
        "status":     "running",   # running | completed | failed | stopped
        "started_at": datetime.now().isoformat(),
        "ended_at":   None,
        "log_file":   str(log_file),
        "log_lines":  [],          # last N lines kept in memory for quick display
        "command":    shell_cmd,   # full command string for display
        "returncode": None,
        "process":    None,        # subprocess.Popen — not JSON-serialisable
    }
    _jobs[job_id] = job

    # Paths for the temp batch file and the completion sentinel
    bat_file  = log_file.with_suffix(".bat")
    done_file = log_file.with_suffix(".done")
    job["done_file"] = str(done_file)

    def _run():
        # Write metadata to the log file (no model output captured here —
        # the CMD window streams it directly to the user).
        with open(log_file, "w", encoding="utf-8") as lf:
            for line in [
                "=" * 60,
                f"  Job ID     : {job_id}",
                f"  Run name   : {job['run_name']}",
                f"  Conda env  : {conda_env}",
                f"  Working dir: {epm_dir}",
                f"  Command    : {shell_cmd}",
                f"  Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 60,
            ]:
                lf.write(line + "\n")

        # Write batch file: runs the model, then drops a sentinel file
        # with the exit code so the dashboard knows when it's done.
        bat_lines = [
            "@echo off",
            f"echo ============================================================",
            f"echo   EPM Run  : {job['run_name']}",
            f"echo   Folder   : {folder_input}",
            f"echo   Started  : %DATE% %TIME%",
            f"echo ============================================================",
            f"echo.",
            f"call conda activate {conda_env}",
            f"cd /d \"{epm_dir}\"",
            py_args,
            "set _RC=%ERRORLEVEL%",
            f"echo.",
            f"echo ============================================================",
            f"echo   Finished with exit code: %_RC%",
            f"echo ============================================================",
            f"(echo %_RC%) > \"{done_file}\"",
            "exit /b %_RC%",
        ]
        bat_file.write_text("\r\n".join(bat_lines), encoding="utf-8")

        # Open a visible CMD window (/k keeps it open after run for review)
        subprocess.Popen(
            f'start "EPM Run — {folder_input}" cmd /k "{bat_file}"',
            shell=True,
        )

        # Poll the sentinel file until the model finishes or the job is stopped
        while not done_file.exists():
            if job["status"] != "running":
                break   # externally stopped
            time.sleep(2)

        if done_file.exists():
            try:
                rc = int(done_file.read_text().strip())
            except Exception:
                rc = -1
            job["returncode"] = rc
            if job["status"] == "running":          # don't overwrite "stopped"
                job["status"] = "completed" if rc == 0 else "failed"
            try:
                done_file.unlink()
            except Exception:
                pass
        elif job["status"] == "stopped":
            job["returncode"] = -1

        job["ended_at"] = datetime.now().isoformat()

        # Append completion info to the metadata log
        with open(log_file, "a", encoding="utf-8") as lf:
            for line in [
                "",
                f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"  Return code: {job['returncode']}",
                f"  Status     : {job['status'].upper()}",
                "=" * 60,
            ]:
                lf.write(line + "\n")

        # Clean up batch file
        try:
            bat_file.unlink()
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return job_id


def stop_job(job_id: str) -> bool:
    """Mark a running job as stopped; the polling loop will exit on next tick."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "running":
        return False
    job["status"]   = "stopped"
    job["ended_at"] = datetime.now().isoformat()
    return True


def read_log(job_id: str, last_n: int = 200) -> list[str]:
    """Return the last N log lines for a job, reading live from disk."""
    job = _jobs.get(job_id)
    if not job:
        return []
    path = Path(job["log_file"])
    if path.exists():
        try:
            text  = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            return lines[-last_n:] if lines else []
        except Exception:
            pass
    return job["log_lines"][-last_n:]


def read_log_file(job_id: str) -> str:
    """Read full log from disk (for download)."""
    job = _jobs.get(job_id)
    if not job:
        return ""
    path = Path(job["log_file"])
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""
