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
    scenarios_csv: str | None = None,
    cpu: int = 2,
    run_name: str = "",
    repo_root: Path | None = None,
) -> str:
    """
    Launch epm.py as a background subprocess.

    Parameters
    ----------
    folder_input : str
        Name of the input data folder (e.g. 'data_eapp').
    scenarios_csv : str, optional
        Path to a custom scenarios.csv. If None, uses the one inside folder_input.
    cpu : int
        Number of CPU cores to pass via --cpu.
    run_name : str
        Optional label for display in the UI.
    repo_root : Path, optional
        Root of the EPM repo. Defaults to two levels up from this file.

    Returns
    -------
    str
        job_id that can be polled via get_job().
    """
    if repo_root is None:
        repo_root = Path(__file__).parent.parent.parent  # dashboard/backend/ → EPM/

    job_id    = _job_id()
    log_file  = repo_root / "dashboard" / "assets" / f"job_{job_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda", "run", "--no-capture-output", "-n", "esmap_env",
        "python", "epm.py",
        "--folder_input", folder_input,
        "--cpu", str(cpu),
    ]
    if scenarios_csv:
        cmd += ["--scenarios", scenarios_csv]

    job = {
        "job_id":     job_id,
        "run_name":   run_name or f"Run {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "folder":     folder_input,
        "status":     "running",   # running | completed | failed | stopped
        "started_at": datetime.now().isoformat(),
        "ended_at":   None,
        "log_file":   str(log_file),
        "log_lines":  [],          # last N lines kept in memory for quick display
        "returncode": None,
        "process":    None,        # subprocess.Popen — not JSON-serialisable
    }
    _jobs[job_id] = job

    def _run():
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            job["process"] = proc
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                job["log_lines"].append(line.rstrip())
                if len(job["log_lines"]) > 500:
                    job["log_lines"] = job["log_lines"][-500:]
            proc.wait()
            job["returncode"] = proc.returncode
            job["status"]     = "completed" if proc.returncode == 0 else "failed"
            job["ended_at"]   = datetime.now().isoformat()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return job_id


def stop_job(job_id: str) -> bool:
    """Send SIGTERM to a running job. Returns True if signal was sent."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "running":
        return False
    proc = job.get("process")
    if proc:
        proc.terminate()
        job["status"]   = "stopped"
        job["ended_at"] = datetime.now().isoformat()
        return True
    return False


def read_log(job_id: str, last_n: int = 200) -> list[str]:
    """Return the last N log lines for a job (fast, from memory)."""
    job = _jobs.get(job_id)
    if not job:
        return []
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
