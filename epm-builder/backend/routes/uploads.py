"""
File Upload API routes.

Handles CSV file uploads for scenario inputs.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse

router = APIRouter()

# Temporary storage for uploaded files before scenario creation
UPLOAD_TEMP_FOLDER = Path(__file__).parent.parent / "uploads_temp"

# Define valid CSV files that can be uploaded, organized by category
# This comprehensive schema covers all EPM input files
VALID_CSV_FILES = {
    "general": {
        "pSettings.csv": {
            "description": "Model settings and parameters (WACC, discount rate, VoLL, etc.)",
            "folder": "",
            "required_columns": ["Category", "Abbr", "Value"],
        },
        "y.csv": {
            "description": "Planning years",
            "folder": "",
            "required_columns": ["y"],
        },
        "zcmap.csv": {
            "description": "Zone to country mapping",
            "folder": "",
            "required_columns": ["z", "c"],
        },
        "pHours.csv": {
            "description": "Representative hours/days configuration",
            "folder": "",
            "required_columns": ["d", "h"],
        },
    },
    "demand": {
        "pDemandForecast.csv": {
            "description": "Annual demand forecast by zone (Energy and Peak)",
            "folder": "load",
            "required_columns": ["z", "type"],
        },
        "pDemandProfile.csv": {
            "description": "Hourly demand profile shapes",
            "folder": "load",
            "required_columns": ["z", "d", "h"],
        },
        "pDemandData.csv": {
            "description": "Demand data configuration",
            "folder": "load",
            "required_columns": ["z"],
        },
        "pEnergyEfficiencyFactor.csv": {
            "description": "Energy efficiency improvement factors by year",
            "folder": "load",
            "required_columns": ["z"],
        },
    },
    "supply_generation": {
        "pGenDataInput.csv": {
            "description": "Generator fleet data (capacity, costs, characteristics)",
            "folder": "supply",
            "required_columns": ["g", "z", "tech", "fuel", "Status", "Capacity"],
        },
        "pGenDataInputDefault.csv": {
            "description": "Default generator parameters by technology",
            "folder": "supply",
            "required_columns": ["tech"],
        },
        "pAvailabilityCustom.csv": {
            "description": "Custom availability factors by technology",
            "folder": "supply",
            "required_columns": ["tech"],
        },
        "pAvailabilityDefault.csv": {
            "description": "Default availability factors by technology",
            "folder": "supply",
            "required_columns": ["tech"],
        },
        "pEvolutionAvailability.csv": {
            "description": "Availability factor evolution over time",
            "folder": "supply",
            "required_columns": ["tech"],
        },
    },
    "supply_storage": {
        "pStorageDataInput.csv": {
            "description": "Storage units data (batteries, pumped hydro)",
            "folder": "supply",
            "required_columns": ["s", "z", "tech"],
        },
        "pStorageDataInputDefault.csv": {
            "description": "Default storage parameters by technology",
            "folder": "supply",
            "required_columns": ["tech"],
        },
    },
    "supply_costs": {
        "pFuelPrice.csv": {
            "description": "Fuel prices by fuel type and year",
            "folder": "supply",
            "required_columns": ["fuel"],
        },
        "pCapexTrajectoriesCustom.csv": {
            "description": "Custom CAPEX cost trajectories by technology",
            "folder": "supply",
            "required_columns": ["tech"],
        },
        "pCapexTrajectoriesDefault.csv": {
            "description": "Default CAPEX cost trajectories by technology",
            "folder": "supply",
            "required_columns": ["tech"],
        },
    },
    "supply_renewables": {
        "pVREProfile.csv": {
            "description": "Variable renewable energy hourly capacity factors",
            "folder": "supply",
            "required_columns": ["z", "tech", "d", "h"],
        },
        "pVREgenProfile.csv": {
            "description": "VRE generation profiles for existing plants",
            "folder": "supply",
            "required_columns": ["g", "d", "h"],
        },
        "pCSPData.csv": {
            "description": "Concentrated Solar Power plant data",
            "folder": "supply",
            "required_columns": ["g"],
        },
    },
    "transmission": {
        "pTransferLimit.csv": {
            "description": "Transfer capacity limits between zones",
            "folder": "trade",
            "required_columns": ["zfrom", "zto"],
        },
        "pExtTransferLimit.csv": {
            "description": "External transfer limits (imports/exports)",
            "folder": "trade",
            "required_columns": ["z", "zext"],
        },
        "pNewTransmission.csv": {
            "description": "New transmission line candidates for expansion",
            "folder": "trade",
            "required_columns": ["zfrom", "zto"],
        },
        "pLossFactorInternal.csv": {
            "description": "Transmission loss factors between zones",
            "folder": "trade",
            "required_columns": ["zfrom", "zto"],
        },
        "pTradePrice.csv": {
            "description": "External trade prices (import/export)",
            "folder": "trade",
            "required_columns": ["zext"],
        },
        "zext.csv": {
            "description": "External zone definitions",
            "folder": "trade",
            "required_columns": ["zext"],
        },
        "pMinImport.csv": {
            "description": "Minimum import requirements",
            "folder": "trade",
            "required_columns": ["z"],
        },
        "pMaxAnnualExternalTradeShare.csv": {
            "description": "Maximum share of external trade",
            "folder": "trade",
            "required_columns": ["z"],
        },
    },
    "emissions": {
        "pCarbonPrice.csv": {
            "description": "Carbon price trajectory by year ($/tCO2)",
            "folder": "constraint",
            "required_columns": ["y", "CarbonPrice"],
        },
        "pEmissionsTotal.csv": {
            "description": "System-wide CO2 emission caps by year",
            "folder": "constraint",
            "required_columns": ["y"],
        },
        "pEmissionsCountry.csv": {
            "description": "Country-level CO2 emission caps by year",
            "folder": "constraint",
            "required_columns": ["c", "y"],
        },
        "pMaxFuellimit.csv": {
            "description": "Maximum fuel consumption limits",
            "folder": "constraint",
            "required_columns": ["z", "fuel"],
        },
    },
    "policy": {
        "pRenewableTarget.csv": {
            "description": "Renewable energy share targets by year",
            "folder": "constraint",
            "required_columns": ["y"],
        },
    },
    "reserves": {
        "pPlanningReserveMargin.csv": {
            "description": "Planning reserve margin requirements",
            "folder": "reserve",
            "required_columns": ["z"],
        },
        "pSpinningReserveReqCountry.csv": {
            "description": "Spinning reserve requirements by country",
            "folder": "reserve",
            "required_columns": ["c"],
        },
        "pSpinningReserveReqSystem.csv": {
            "description": "System-wide spinning reserve requirements",
            "folder": "reserve",
            "required_columns": [],
        },
    },
    "hydrogen": {
        "pAvailabilityH2.csv": {
            "description": "Hydrogen production availability factors",
            "folder": "h2",
            "required_columns": ["tech"],
        },
        "pCapexTrajectoryH2.csv": {
            "description": "Hydrogen technology CAPEX trajectories",
            "folder": "h2",
            "required_columns": ["tech"],
        },
        "pFuelDataH2.csv": {
            "description": "Hydrogen fuel data and pricing",
            "folder": "h2",
            "required_columns": [],
        },
        "pExternalH2.csv": {
            "description": "External hydrogen supply/demand",
            "folder": "h2",
            "required_columns": [],
        },
        "pH2DataExcel.csv": {
            "description": "Hydrogen system configuration",
            "folder": "h2",
            "required_columns": [],
        },
    },
}

# Category display metadata
CATEGORY_METADATA = {
    "general": {"label": "General Settings", "color": "gray", "icon": "cog"},
    "demand": {"label": "Demand", "color": "green", "icon": "chart-line"},
    "supply_generation": {"label": "Supply - Generation", "color": "blue", "icon": "bolt"},
    "supply_storage": {"label": "Supply - Storage", "color": "blue", "icon": "battery-full"},
    "supply_costs": {"label": "Supply - Costs", "color": "blue", "icon": "dollar-sign"},
    "supply_renewables": {"label": "Supply - Renewables", "color": "green", "icon": "sun"},
    "transmission": {"label": "Transmission & Trade", "color": "purple", "icon": "arrows-alt-h"},
    "emissions": {"label": "Emissions & Carbon", "color": "orange", "icon": "cloud"},
    "policy": {"label": "Policy & Targets", "color": "indigo", "icon": "flag"},
    "reserves": {"label": "Reserves", "color": "yellow", "icon": "shield-alt"},
    "hydrogen": {"label": "Hydrogen", "color": "teal", "icon": "atom"},
}


def ensure_upload_folder():
    """Create uploads temp folder if it doesn't exist."""
    UPLOAD_TEMP_FOLDER.mkdir(parents=True, exist_ok=True)


def get_session_folder(session_id: str) -> Path:
    """Get or create folder for a session's uploads."""
    folder = UPLOAD_TEMP_FOLDER / session_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def validate_csv_content(content: bytes, filename: str) -> tuple[bool, str]:
    """
    Basic validation of CSV content.

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Decode content
        text = content.decode("utf-8-sig")
        lines = text.strip().split("\n")

        if len(lines) < 2:
            return False, "CSV file must have at least a header row and one data row"

        # Check for valid CSV structure
        header = lines[0].split(",")
        if len(header) < 2:
            return False, "CSV file must have at least 2 columns"

        # Find the file spec
        file_spec = None
        for category, files in VALID_CSV_FILES.items():
            if filename in files:
                file_spec = files[filename]
                break

        if file_spec:
            # Check required columns
            header_clean = [h.strip().strip('"') for h in header]
            missing_cols = [col for col in file_spec["required_columns"] if col not in header_clean]
            if missing_cols:
                return False, f"Missing required columns: {', '.join(missing_cols)}"

        return True, f"Valid CSV with {len(lines)-1} data rows"

    except UnicodeDecodeError:
        return False, "File must be UTF-8 encoded"
    except Exception as e:
        return False, f"Error parsing CSV: {str(e)}"


@router.get("/schema")
async def get_upload_schema():
    """
    Get the schema of valid CSV files that can be uploaded.

    Returns list of files organized by category with descriptions and metadata.
    """
    return {
        "files": VALID_CSV_FILES,
        "categories": CATEGORY_METADATA
    }


@router.post("/")
async def upload_csv(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
):
    """
    Upload a CSV file for a scenario.

    Args:
        file: The CSV file to upload
        session_id: Optional session ID to group uploads (will be created if not provided)
        category: Optional category (supply, load, trade, constraint, settings)

    Returns:
        Upload confirmation with session_id and validation results
    """
    ensure_upload_folder()

    # Validate filename
    filename = file.filename
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    # Check if this is a known file type
    file_spec = None
    file_category = None
    for cat, files in VALID_CSV_FILES.items():
        if filename in files:
            file_spec = files[filename]
            file_category = cat
            break

    if not file_spec:
        # Allow custom filenames but warn
        file_category = category or "custom"

    # Create or use session
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    session_folder = get_session_folder(session_id)

    # Determine target subfolder
    if file_spec and file_spec["folder"]:
        target_folder = session_folder / file_spec["folder"]
        target_folder.mkdir(parents=True, exist_ok=True)
    else:
        target_folder = session_folder

    # Read and validate content
    content = await file.read()
    is_valid, message = validate_csv_content(content, filename)

    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    # Save file
    target_path = target_folder / filename
    with open(target_path, "wb") as f:
        f.write(content)

    return {
        "success": True,
        "session_id": session_id,
        "filename": filename,
        "category": file_category,
        "validation": message,
        "path": str(target_path.relative_to(UPLOAD_TEMP_FOLDER)),
    }


@router.get("/{session_id}")
async def list_session_uploads(session_id: str):
    """
    List all uploaded files for a session.
    """
    session_folder = UPLOAD_TEMP_FOLDER / session_id

    if not session_folder.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    files = []
    for path in session_folder.rglob("*.csv"):
        rel_path = path.relative_to(session_folder)

        # Find category
        category = "custom"
        for cat, file_specs in VALID_CSV_FILES.items():
            if path.name in file_specs:
                category = cat
                break

        files.append({
            "filename": path.name,
            "path": str(rel_path),
            "category": category,
            "size_bytes": path.stat().st_size,
        })

    return {
        "session_id": session_id,
        "files": files,
    }


@router.get("/{session_id}/preview/{filepath:path}")
async def preview_upload(session_id: str, filepath: str, rows: int = 5):
    """
    Get a preview (first N rows) of an uploaded CSV file.

    Args:
        session_id: Upload session ID
        filepath: Path to the file relative to session folder
        rows: Number of rows to return (default 5, max 50)

    Returns:
        JSON with headers and data rows
    """
    import csv

    # Security: prevent path traversal
    if ".." in filepath:
        raise HTTPException(status_code=400, detail="Invalid filepath")

    rows = min(rows, 50)  # Cap at 50 rows

    session_folder = UPLOAD_TEMP_FOLDER / session_id
    if not session_folder.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    file_path = session_folder / filepath

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")

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


@router.delete("/{session_id}/{filename}")
async def delete_upload(session_id: str, filename: str):
    """
    Delete an uploaded file from a session.
    """
    session_folder = UPLOAD_TEMP_FOLDER / session_id

    if not session_folder.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # Find the file (could be in a subfolder)
    target_file = None
    for path in session_folder.rglob(filename):
        target_file = path
        break

    if not target_file or not target_file.exists():
        raise HTTPException(status_code=404, detail="File not found")

    os.remove(target_file)

    return {"success": True, "deleted": filename}


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """
    Delete all uploads for a session.
    """
    session_folder = UPLOAD_TEMP_FOLDER / session_id

    if not session_folder.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    shutil.rmtree(session_folder)

    return {"success": True, "deleted_session": session_id}


def get_session_files(session_id: str) -> dict[str, Path]:
    """
    Get all uploaded files for a session as a dict mapping filename to path.
    Used by data_builder when creating scenario folder.
    """
    session_folder = UPLOAD_TEMP_FOLDER / session_id

    if not session_folder.exists():
        return {}

    files = {}
    for path in session_folder.rglob("*.csv"):
        files[path.name] = path

    return files


def copy_session_files_to_scenario(session_id: str, scenario_folder: Path):
    """
    Copy all uploaded files from a session to a scenario folder.
    Preserves subfolder structure.
    """
    session_folder = UPLOAD_TEMP_FOLDER / session_id

    if not session_folder.exists():
        return

    for src_path in session_folder.rglob("*.csv"):
        rel_path = src_path.relative_to(session_folder)
        dest_path = scenario_folder / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
