"""
Scenarios API routes.

Handles scenario creation and retrieval.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException

from models.schemas import ScenarioCreate, ScenarioResponse
from services.data_builder import build_scenario_folder

router = APIRouter()

# In-memory scenario store (for MVP)
_scenarios: dict[str, dict] = {}


@router.post("/", response_model=ScenarioResponse)
async def create_scenario(scenario: ScenarioCreate):
    """
    Create a new scenario configuration.

    This creates the input folder structure needed to run EPM.
    """
    try:
        # Build the scenario folder
        scenario_id, folder_path = build_scenario_folder(scenario.model_dump())

        # Store scenario metadata
        _scenarios[scenario_id] = {
            "id": scenario_id,
            "name": scenario.name,
            "description": scenario.description,
            "created_at": datetime.now(),
            "status": "ready",
            "folder_path": str(folder_path),
            "config": scenario.model_dump()
        }

        return ScenarioResponse(
            id=scenario_id,
            name=scenario.name,
            description=scenario.description,
            created_at=_scenarios[scenario_id]["created_at"],
            status="ready"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{scenario_id}", response_model=ScenarioResponse)
async def get_scenario(scenario_id: str):
    """Get scenario details by ID."""
    if scenario_id not in _scenarios:
        raise HTTPException(status_code=404, detail="Scenario not found")

    s = _scenarios[scenario_id]
    return ScenarioResponse(
        id=s["id"],
        name=s["name"],
        description=s["description"],
        created_at=s["created_at"],
        status=s["status"]
    )


@router.get("/")
async def list_scenarios():
    """List all scenarios."""
    return [
        ScenarioResponse(
            id=s["id"],
            name=s["name"],
            description=s["description"],
            created_at=s["created_at"],
            status=s["status"]
        )
        for s in _scenarios.values()
    ]


def get_scenario_folder(scenario_id: str) -> str:
    """Get the folder path for a scenario (used by jobs route)."""
    if scenario_id not in _scenarios:
        raise ValueError(f"Scenario {scenario_id} not found")
    return _scenarios[scenario_id]["folder_path"]
