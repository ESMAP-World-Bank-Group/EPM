"""
Pydantic models for EPM User Interface API.

These schemas define the structure of requests and responses for the API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of an EPM job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GeneratorInput(BaseModel):
    """A single generator/power plant definition."""
    name: str = Field(..., description="Generator name/ID")
    zone: str = Field(..., description="Zone where generator is located")
    technology: str = Field(..., description="Technology type (e.g., CCGT, PV, Wind)")
    fuel: str = Field(..., description="Fuel type (e.g., Gas, Solar, Wind)")
    capacity_mw: float = Field(..., ge=0, description="Installed capacity in MW")
    status: int = Field(1, ge=1, le=3, description="1=existing, 2=candidate, 3=retired")
    start_year: Optional[int] = Field(None, description="Year plant starts operating")
    retirement_year: Optional[int] = Field(None, description="Year plant retires")
    capex_per_mw: Optional[float] = Field(None, ge=0, description="Capital cost $/MW")
    fixed_om_per_mw: Optional[float] = Field(None, ge=0, description="Fixed O&M $/MW/year")
    variable_om: Optional[float] = Field(None, ge=0, description="Variable O&M $/MWh")
    heat_rate: Optional[float] = Field(None, ge=0, description="Heat rate MMBtu/MWh")


class DemandInput(BaseModel):
    """Demand/load configuration."""
    zone: str = Field(..., description="Zone name")
    base_year_energy_gwh: float = Field(..., ge=0, description="Base year energy demand in GWh")
    base_year_peak_mw: float = Field(..., ge=0, description="Base year peak demand in MW")
    annual_growth_rate: float = Field(0.03, description="Annual demand growth rate (e.g., 0.03 = 3%)")


class EconomicsInput(BaseModel):
    """Economic parameters."""
    wacc: float = Field(0.08, ge=0, le=1, description="Weighted average cost of capital")
    discount_rate: float = Field(0.06, ge=0, le=1, description="Discount rate")
    voll: float = Field(1000, ge=0, description="Value of lost load $/MWh")


class FeaturesInput(BaseModel):
    """Feature toggles for the model."""
    enable_capacity_expansion: bool = Field(True, description="Allow new capacity investments")
    enable_transmission_expansion: bool = Field(False, description="Allow new transmission investments")
    enable_storage: bool = Field(True, description="Include battery storage")
    enable_hydrogen: bool = Field(False, description="Include hydrogen production")
    apply_carbon_price: bool = Field(False, description="Apply carbon pricing")
    apply_co2_constraint: bool = Field(False, description="Apply CO2 emission constraint")
    enable_economic_retirement: bool = Field(False, description="Allow economic retirement of plants")


class EmissionsInput(BaseModel):
    """Emissions and carbon configuration."""
    carbon_price_per_ton: Optional[float] = Field(None, ge=0, description="Carbon price $/tCO2")
    annual_co2_limit_mt: Optional[float] = Field(None, ge=0, description="Annual CO2 limit in Mt")
    min_renewable_share: float = Field(0, ge=0, le=1, description="Minimum renewable share (0-1)")


class ScenarioCreate(BaseModel):
    """Request body for creating a new scenario."""
    name: str = Field(..., min_length=1, max_length=100, description="Scenario name")
    description: Optional[str] = Field(None, description="Scenario description")

    # Planning horizon
    start_year: int = Field(2025, ge=2020, le=2050, description="First year of planning horizon")
    end_year: int = Field(2040, ge=2025, le=2060, description="Last year of planning horizon")

    # Core inputs
    zones: list[str] = Field(default_factory=list, description="Zones to include")
    demand: list[DemandInput] = Field(default_factory=list, description="Demand configuration per zone")
    generators: list[GeneratorInput] = Field(default_factory=list, description="Generator fleet")

    # Parameters
    economics: EconomicsInput = Field(default_factory=EconomicsInput)
    features: FeaturesInput = Field(default_factory=FeaturesInput)
    emissions: EmissionsInput = Field(default_factory=EmissionsInput)

    # Model settings
    model_type: str = Field("RMIP", description="MIP (integer) or RMIP (relaxed)")

    # Upload session (optional - for custom CSV files)
    upload_session_id: Optional[str] = Field(None, description="Session ID for uploaded CSV files")


class ScenarioResponse(BaseModel):
    """Response body for a scenario."""
    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    status: str

    class Config:
        from_attributes = True


class JobCreate(BaseModel):
    """Request body for creating a new job."""
    scenario_id: str = Field(..., description="ID of scenario to run")


class JobResponse(BaseModel):
    """Response body for a job."""
    id: str
    scenario_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_pct: float = Field(0, ge=0, le=100)
    logs: list[str] = Field(default_factory=list)

    class Config:
        from_attributes = True


class ResultsSummary(BaseModel):
    """Summary of EPM results."""
    job_id: str
    scenario_name: str
    total_cost_musd: float

    # Capacity results by year
    capacity_by_year: dict[str, dict[str, float]]  # {year: {tech: MW}}

    # Generation results by year
    generation_by_year: dict[str, dict[str, float]]  # {year: {tech: GWh}}

    # Emissions by year
    emissions_by_year: dict[str, float]  # {year: MtCO2}

    # Cost breakdown
    cost_breakdown: dict[str, float]  # {category: $M}


class TemplateZone(BaseModel):
    """Zone information from template."""
    code: str
    name: str
    country: str


class TemplateTechnology(BaseModel):
    """Technology information from template."""
    code: str
    name: str
    fuel: str
    is_renewable: bool
    typical_capex: Optional[float] = None
    typical_heat_rate: Optional[float] = None


class TemplateResponse(BaseModel):
    """Response with template/default values from data_test."""
    zones: list[TemplateZone]
    technologies: list[TemplateTechnology]
    fuels: list[str]
    default_years: list[int]
    default_settings: dict[str, Any]
