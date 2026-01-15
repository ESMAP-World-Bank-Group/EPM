"""
Templates API routes.

Provides default values and reference data from the EPM template.
"""

import csv
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
import shutil

from models.schemas import TemplateResponse, TemplateZone, TemplateTechnology
from services.data_builder import get_template_data, RUNS_FOLDER, TEMPLATE_FOLDER

router = APIRouter()


@router.get("/", response_model=TemplateResponse)
async def get_templates():
    """
    Get template data including zones, technologies, fuels, and default settings.

    This data can be used to populate form dropdowns and provide defaults.
    """
    try:
        data = get_template_data()

        return TemplateResponse(
            zones=[TemplateZone(**z) for z in data["zones"]],
            technologies=[TemplateTechnology(**t) for t in data["technologies"]],
            fuels=data["fuels"],
            default_years=data["default_years"],
            default_settings=data["default_settings"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filepath:path}")
async def download_template(filepath: str):
    """
    Download a template CSV file from data_test.

    Args:
        filepath: Path to the file relative to data_test (e.g., "load/pDemandForecast.csv")

    Returns:
        The CSV file as a download
    """
    # Security: prevent path traversal
    if ".." in filepath:
        raise HTTPException(status_code=400, detail="Invalid filepath")

    file_path = TEMPLATE_FOLDER / filepath

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Template file not found: {filepath}")

    if not file_path.suffix == ".csv":
        raise HTTPException(status_code=400, detail="Only CSV files can be downloaded")

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="text/csv"
    )


@router.get("/preview/{filepath:path}")
async def preview_template(filepath: str, rows: int = 5):
    """
    Get a preview (first N rows) of a template CSV file.

    Args:
        filepath: Path to the file relative to data_test
        rows: Number of rows to return (default 5, max 50)

    Returns:
        JSON with headers and data rows
    """
    # Security: prevent path traversal
    if ".." in filepath:
        raise HTTPException(status_code=400, detail="Invalid filepath")

    rows = min(rows, 50)  # Cap at 50 rows

    file_path = TEMPLATE_FOLDER / filepath

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Template file not found: {filepath}")

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            data = []
            for i, row in enumerate(reader):
                if i >= rows:
                    break
                data.append(row)

        return {
            "filepath": filepath,
            "filename": file_path.name,
            "headers": headers,
            "data": data,
            "total_preview_rows": len(data),
            "has_more": len(data) == rows
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@router.post("/upload/{scenario_id}/{file_type}")
async def upload_csv(scenario_id: str, file_type: str, file: UploadFile = File(...)):
    """
    Upload a CSV file to override template data for a scenario.

    Args:
        scenario_id: ID of the scenario to update
        file_type: Type of file (e.g., 'generators', 'demand', 'settings')
        file: The CSV file to upload
    """
    # Validate file type
    allowed_types = {
        "generators": "supply/pGenDataInput.csv",
        "demand": "load/pDemandForecast.csv",
        "settings": "pSettings.csv",
        "zones": "zcmap.csv",
        "fuel_prices": "supply/pFuelPrice.csv"
    }

    if file_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {list(allowed_types.keys())}"
        )

    # Check file extension
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    # Find scenario folder
    scenario_folder = RUNS_FOLDER / f"data_{scenario_id}"
    if not scenario_folder.exists():
        raise HTTPException(status_code=404, detail="Scenario not found")

    # Save file
    target_path = scenario_folder / allowed_types[file_type]
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "message": f"Successfully uploaded {file_type}",
            "path": str(target_path.relative_to(scenario_folder))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
