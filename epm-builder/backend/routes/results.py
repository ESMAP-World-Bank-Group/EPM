"""
Results API routes.

Handles fetching EPM run results.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from models.schemas import ResultsSummary
from services.epm_runner import get_job, parse_results

router = APIRouter()


@router.get("/{job_id}", response_model=ResultsSummary)
async def get_results(job_id: str):
    """
    Get parsed results for a completed job.

    Returns summary statistics and data for visualization.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"].value != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job['status'].value}"
        )

    results = parse_results(job_id)
    if not results:
        raise HTTPException(status_code=404, detail="Results not found")

    return ResultsSummary(**results)


@router.get("/{job_id}/download/{filename}")
async def download_result_file(job_id: str, filename: str):
    """
    Download a specific result file.

    Args:
        job_id: Job ID
        filename: Name of file to download (e.g., 'oCapacity.csv')
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.get("output_folder"):
        raise HTTPException(status_code=404, detail="Output folder not found")

    # Sanitize filename to prevent path traversal
    safe_filename = Path(filename).name
    file_path = Path(job["output_folder"]) / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {safe_filename} not found")

    return FileResponse(
        path=str(file_path),
        filename=safe_filename,
        media_type="text/csv"
    )


@router.get("/{job_id}/files")
async def list_result_files(job_id: str):
    """List available result files for a job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.get("output_folder"):
        raise HTTPException(status_code=404, detail="Output folder not found")

    output_folder = Path(job["output_folder"])
    files = []

    for f in output_folder.glob("*.csv"):
        files.append({
            "name": f.name,
            "size_kb": round(f.stat().st_size / 1024, 2)
        })

    return {"files": files}
