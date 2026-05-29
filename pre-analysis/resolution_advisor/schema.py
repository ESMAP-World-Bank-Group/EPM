"""
Typed dataclasses for the resolution advisor YAML config.
Validates inputs and provides defaults.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal
import yaml


@dataclass
class CountryConfig:
    name: str
    area_km2: float
    n_bidding_zones: int = 1
    known_congestion_splits: int = 0
    re_cf_spread: float = 0.15          # fraction (0-1)
    distant_load_centers: bool = False
    hydro_concentration: bool = False
    data_quality: Literal["good", "medium", "limited"] = "medium"

    # Maximum zones allowed given data quality
    DATA_QUALITY_CAP: dict = field(default_factory=lambda: {
        "good": 6,
        "medium": 4,
        "limited": 1,
    }, repr=False)

    @property
    def data_cap(self) -> int:
        return self.DATA_QUALITY_CAP[self.data_quality]


@dataclass
class ModelUse:
    primary_questions: List[str] = field(default_factory=list)
    re_penetration_target: float = 0.30
    storage_relevance: Literal["low", "medium", "high"] = "medium"
    hydro_seasonality: Literal["low", "medium", "high"] = "medium"
    multi_period_years: List[int] = field(default_factory=lambda: [2030, 2040, 2050])
    n_scenarios: int = 1

    @property
    def n_periods(self) -> int:
        return len(self.multi_period_years)


@dataclass
class Constraints:
    max_runtime_hours: float = 4.0
    available_solver: str = "cplex"
    ram_gb: float = 32.0
    willing_to_collect_zonal_data: bool = False

    # Solver-specific variable budget heuristics (millions of variables)
    SOLVER_BUDGET: dict = field(default_factory=lambda: {
        "cplex":  8_000_000,
        "gurobi": 8_000_000,
        "glpk":   1_000_000,
    }, repr=False)

    @property
    def variable_budget(self) -> int:
        base = self.SOLVER_BUDGET.get(self.available_solver, 2_000_000)
        # Scale by runtime and RAM
        runtime_factor = min(self.max_runtime_hours / 4.0, 2.0)
        ram_factor = min(self.ram_gb / 32.0, 2.0)
        return int(base * runtime_factor * ram_factor)


@dataclass
class RegionConfig:
    name: str
    countries: dict[str, CountryConfig]


@dataclass
class AdvisorConfig:
    region: RegionConfig
    model_use: ModelUse
    constraints: Constraints

    @classmethod
    def from_yaml(cls, path: str) -> "AdvisorConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        countries = {
            iso: CountryConfig(**{k: v for k, v in data.items() if k != "name"},
                               name=data.get("name", iso))
            for iso, data in raw["region"]["countries"].items()
        }
        region = RegionConfig(name=raw["region"]["name"], countries=countries)

        mu_raw = raw.get("model_use", {})
        model_use = ModelUse(
            primary_questions=mu_raw.get("primary_questions", []),
            re_penetration_target=mu_raw.get("re_penetration_target", 0.30),
            storage_relevance=mu_raw.get("storage_relevance", "medium"),
            hydro_seasonality=mu_raw.get("hydro_seasonality", "medium"),
            multi_period_years=mu_raw.get("multi_period_years", [2030, 2040, 2050]),
            n_scenarios=mu_raw.get("n_scenarios", 1),
        )

        c_raw = raw.get("constraints", {})
        constraints = Constraints(
            max_runtime_hours=c_raw.get("max_runtime_hours", 4.0),
            available_solver=c_raw.get("available_solver", "cplex"),
            ram_gb=c_raw.get("ram_gb", 32.0),
            willing_to_collect_zonal_data=c_raw.get("willing_to_collect_zonal_data", False),
        )

        return cls(region=region, model_use=model_use, constraints=constraints)
