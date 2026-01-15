"""
EPM Runner Service

Executes EPM model runs and manages job status.
Uses asyncio.create_subprocess_exec for safe subprocess execution.
"""

import asyncio
import csv
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

# Path to EPM root
EPM_ROOT = Path(__file__).parent.parent.parent.parent / "epm"
RUNS_FOLDER = Path(__file__).parent.parent / "runs"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# In-memory job store (for MVP - would use database in production)
_jobs: dict[str, dict] = {}


def create_job(scenario_id: str, input_folder: Path) -> str:
    """
    Create a new job record.

    Args:
        scenario_id: ID of the scenario to run
        input_folder: Path to the input folder

    Returns:
        Job ID
    """
    job_id = f"job_{scenario_id}"

    _jobs[job_id] = {
        "id": job_id,
        "scenario_id": scenario_id,
        "input_folder": str(input_folder),
        "status": JobStatus.PENDING,
        "created_at": datetime.now(),
        "started_at": None,
        "completed_at": None,
        "error_message": None,
        "progress_pct": 0,
        "logs": [],
        "output_folder": None,
    }

    return job_id


def get_job(job_id: str) -> Optional[dict]:
    """Get job status by ID."""
    return _jobs.get(job_id)


def get_all_jobs() -> list[dict]:
    """Get all jobs."""
    return list(_jobs.values())


def update_job(job_id: str, **kwargs):
    """Update job properties."""
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)


def add_log(job_id: str, message: str):
    """Add a log message to a job."""
    if job_id in _jobs:
        timestamp = datetime.now().strftime("%H:%M:%S")
        _jobs[job_id]["logs"].append(f"[{timestamp}] {message}")


async def run_epm_async(job_id: str):
    """
    Run EPM model asynchronously using safe subprocess execution.

    Args:
        job_id: ID of the job to run
    """
    job = get_job(job_id)
    if not job:
        return

    update_job(job_id, status=JobStatus.RUNNING, started_at=datetime.now())
    add_log(job_id, "Starting EPM model run...")

    input_folder = Path(job["input_folder"])
    folder_name = input_folder.name

    try:
        # Use create_subprocess_exec for safe execution (no shell injection)
        # Arguments are passed as a list, not interpolated into a shell command
        add_log(job_id, f"Input folder: {folder_name}")
        update_job(job_id, progress_pct=10)

        # Run EPM using subprocess_exec (safe, no shell)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m", "epm.epm",
            "--folder_input", folder_name,
            "--cpu", "1",
            "--modeltype", "RMIP",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(EPM_ROOT.parent)
        )

        update_job(job_id, progress_pct=20)
        add_log(job_id, "EPM process started, waiting for completion...")

        # Wait for completion with progress updates
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            update_job(job_id, progress_pct=90)
            add_log(job_id, "EPM run completed successfully")

            # Find output folder
            output_base = EPM_ROOT / "output"
            output_folders = sorted(output_base.glob(f"*_{folder_name}_*"), reverse=True)

            if output_folders:
                output_folder = output_folders[0]
                update_job(job_id, output_folder=str(output_folder))
                add_log(job_id, f"Output saved to: {output_folder.name}")

            update_job(
                job_id,
                status=JobStatus.COMPLETED,
                completed_at=datetime.now(),
                progress_pct=100
            )
        else:
            error_msg = stderr.decode() if stderr else "Unknown error"
            add_log(job_id, f"EPM run failed: {error_msg}")
            update_job(
                job_id,
                status=JobStatus.FAILED,
                completed_at=datetime.now(),
                error_message=error_msg[:500]
            )

    except Exception as e:
        add_log(job_id, f"Error: {str(e)}")
        update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.now(),
            error_message=str(e)[:500]
        )


def start_job(job_id: str):
    """
    Start a job in the background.

    Args:
        job_id: ID of the job to start
    """
    asyncio.create_task(run_epm_async(job_id))


def parse_results(job_id: str) -> Optional[dict]:
    """
    Parse EPM results from output folder.

    Args:
        job_id: ID of the completed job

    Returns:
        Dictionary with parsed results or None if not available
    """
    job = get_job(job_id)
    if not job or not job.get("output_folder"):
        return None

    output_folder = Path(job["output_folder"])
    results = {
        "job_id": job_id,
        "scenario_name": job["scenario_id"],
        "total_cost_musd": 0,
        "capacity_by_year": {},
        "generation_by_year": {},
        "emissions_by_year": {},
        "cost_breakdown": {}
    }

    # Try to read solver metrics
    metrics_file = output_folder / "solver_metrics.csv"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "ObjectiveCost" in row:
                    try:
                        results["total_cost_musd"] = float(row["ObjectiveCost"]) / 1e6
                    except (ValueError, TypeError):
                        pass

    # Try to read capacity results
    capacity_file = output_folder / "oCapacity.csv"
    if capacity_file.exists():
        with open(capacity_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                year = row.get("y", row.get("year", ""))
                tech = row.get("tech", row.get("technology", ""))
                capacity = row.get("value", row.get("Capacity", 0))

                if year and tech:
                    if year not in results["capacity_by_year"]:
                        results["capacity_by_year"][year] = {}
                    try:
                        results["capacity_by_year"][year][tech] = float(capacity)
                    except (ValueError, TypeError):
                        pass

    # Try to read generation results
    gen_file = output_folder / "oGeneration.csv"
    if gen_file.exists():
        with open(gen_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                year = row.get("y", row.get("year", ""))
                tech = row.get("tech", row.get("technology", ""))
                generation = row.get("value", row.get("Generation", 0))

                if year and tech:
                    if year not in results["generation_by_year"]:
                        results["generation_by_year"][year] = {}
                    try:
                        results["generation_by_year"][year][tech] = float(generation)
                    except (ValueError, TypeError):
                        pass

    # Try to read emissions
    emissions_file = output_folder / "oEmissions.csv"
    if emissions_file.exists():
        with open(emissions_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                year = row.get("y", row.get("year", ""))
                emissions = row.get("value", row.get("Emissions", 0))

                if year:
                    try:
                        results["emissions_by_year"][year] = float(emissions) / 1e6  # Convert to Mt
                    except (ValueError, TypeError):
                        pass

    return results


def cleanup_old_runs(max_age_hours: int = 24):
    """
    Clean up old run folders.

    Args:
        max_age_hours: Maximum age in hours before cleanup
    """
    import shutil
    import time

    cutoff = time.time() - (max_age_hours * 3600)

    for folder in RUNS_FOLDER.glob("data_*"):
        if folder.is_dir() and folder.stat().st_mtime < cutoff:
            shutil.rmtree(folder, ignore_errors=True)
