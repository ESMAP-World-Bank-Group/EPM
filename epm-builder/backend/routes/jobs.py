"""
Jobs API routes.

Handles running EPM jobs and checking status.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks

from models.schemas import JobCreate, JobResponse, JobStatus
from services.epm_runner import create_job, get_job, get_all_jobs, start_job, run_epm_async
from routes.scenarios import get_scenario_folder

router = APIRouter()


@router.post("/", response_model=JobResponse)
async def create_and_start_job(job_request: JobCreate, background_tasks: BackgroundTasks):
    """
    Create and start a new EPM job.

    The job runs asynchronously in the background.
    """
    try:
        # Get scenario folder
        folder_path = get_scenario_folder(job_request.scenario_id)

        # Create job record
        job_id = create_job(job_request.scenario_id, Path(folder_path))

        # Start job in background
        background_tasks.add_task(run_epm_async, job_id)

        job = get_job(job_id)
        return JobResponse(
            id=job["id"],
            scenario_id=job["scenario_id"],
            status=JobStatus(job["status"].value),
            created_at=job["created_at"],
            started_at=job["started_at"],
            completed_at=job["completed_at"],
            error_message=job["error_message"],
            progress_pct=job["progress_pct"],
            logs=job["logs"]
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get job status by ID."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        id=job["id"],
        scenario_id=job["scenario_id"],
        status=JobStatus(job["status"].value),
        created_at=job["created_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        error_message=job["error_message"],
        progress_pct=job["progress_pct"],
        logs=job["logs"]
    )


@router.get("/")
async def list_jobs():
    """List all jobs."""
    jobs = get_all_jobs()
    return [
        JobResponse(
            id=job["id"],
            scenario_id=job["scenario_id"],
            status=JobStatus(job["status"].value),
            created_at=job["created_at"],
            started_at=job["started_at"],
            completed_at=job["completed_at"],
            error_message=job["error_message"],
            progress_pct=job["progress_pct"],
            logs=job["logs"]
        )
        for job in jobs
    ]
