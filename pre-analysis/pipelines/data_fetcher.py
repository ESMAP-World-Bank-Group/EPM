"""Data validation and fetching utilities for open-data sources.

This module provides functions to:
- Validate that all required data sources exist
- Fetch automated data sources (ERA5, Renewables Ninja)
- Generate clear instructions for manual downloads
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

SHAREPOINT_URL = (
    "https://worldbankgroup.sharepoint.com/:f:/r/teams/PowerSystemPlanning-WBGroup/"
    "Shared%20Documents/2.%20Knowledge%20Products/19.%20Databases/EPM%20Prepare%20Data"
    "?csf=1&web=1&e=wig2nC"
)


def get_data_source_info() -> Dict[str, Dict]:
    """Return metadata about each data source (URL, description, manual/automated)."""
    return {
        "gap_excel": {
            "name": "Global Integrated Power Excel",
            "description": "Global Atlas Power database Excel file",
            "manual": True,
            "url": SHAREPOINT_URL,
            "example": "Global-Integrated-Power-September-2025-II.xlsx",
        },
        "irena_solar": {
            "name": "IRENA Solar PV MSR Data",
            "description": "IRENA Best MSRs to Cover 5% Country Area - Solar PV",
            "manual": True,
            "url": "https://www.irena.org/Data/Downloads",
            "example": "SolarPV_BestMSRsToCover5%CountryArea.csv",
        },
        "irena_wind": {
            "name": "IRENA Wind MSR Data",
            "description": "IRENA Best MSRs to Cover 5% Country Area - Wind",
            "manual": True,
            "url": "https://www.irena.org/Data/Downloads",
            "example": "Wind_BestMSRsToCover5%CountryArea.csv",
        },
        "toktarova_load": {
            "name": "Toktarova Load Profiles",
            "description": "Long-term load projection dataset (Toktarova et al. 2019)",
            "manual": True,
            "url": SHAREPOINT_URL,
            "example": "Toktarova2019_paper_LongtermLoadProjectioninHighREsolutionforallCountriesGlobally_ElecPowerEnergySys_supplementary_7_load_all_2020.csv",
        },
        "owid_energy": {
            "name": "OWID Energy Data",
            "description": "Our World in Data energy dataset",
            "manual": True,
            "url": "https://github.com/owid/energy-data",
            "example": "owid-energy-data.csv",
        },
        "hydro_reservoirs": {
            "name": "Global Hydro Reservoirs",
            "description": "Global Hydropower Reservoirs dataset",
            "manual": True,
            "url": SHAREPOINT_URL,
            "example": "GloHydroRes_vs1.csv",
        },
        "natural_earth_maps": {
            "name": "Natural Earth Shapefiles",
            "description": "Natural Earth country and city boundaries",
            "manual": True,
            "url": "https://www.naturalearthdata.com/downloads/",
            "example": "ne_110m_admin_0_countries.shp, ne_110m_populated_places.shp",
        },
        "era5_climate": {
            "name": "ERA5-Land Climate Data",
            "description": "ERA5-Land monthly reanalysis data (via CDS API)",
            "manual": False,
            "automated": True,
            "url": "https://cds.climate.copernicus.eu/",
            "requires": ["cdsapi", "CDS API credentials"],
        },
        "renewables_ninja": {
            "name": "Renewables Ninja Data",
            "description": "Renewables Ninja hourly generation profiles (via API)",
            "manual": False,
            "automated": True,
            "url": "https://www.renewables.ninja/",
            "requires": ["API token"],
        },
    }


def resolve_input_path(base_dir: Path, path_str: str) -> Path:
    """Resolve an input path relative to the dataset directory."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    # Handle paths prefixed with 'dataset/'
    if p.parts and p.parts[0] == "dataset":
        p = Path(*p.parts[1:])
    return base_dir / p


def validate_data_sources(config: Dict, base_dir: Path) -> Tuple[List[str], List[str]]:
    """Validate all required data sources from config.
    
    Returns:
        Tuple of (missing_files, missing_instructions) where:
        - missing_files: List of file paths that are missing
        - missing_instructions: List of formatted instruction strings for manual downloads
    """
    missing_files = []
    missing_instructions = []
    dataset_dir = base_dir / "dataset"
    source_info = get_data_source_info()
    
    # GAP Excel
    if "gap" in config:
        gap_cfg = config["gap"]
        if "excel" in gap_cfg:
            gap_path = resolve_input_path(dataset_dir, gap_cfg["excel"])
            if not gap_path.exists():
                missing_files.append(str(gap_path))
                info = source_info["gap_excel"]
                missing_instructions.append(
                    f"  - {info['name']}: {gap_path.name}\n"
                    f"    Download from: {info['url']}\n"
                    f"    Place in: {dataset_dir}"
                )
    
    # Generation map Excel (may be same as GAP or different)
    if "generation_map" in config:
        genmap_cfg = config.get("generation_map", {})
        if genmap_cfg.get("enabled", False):
            excel_path = genmap_cfg.get("excel") or config.get("gap", {}).get("excel")
            if excel_path:
                genmap_path = resolve_input_path(dataset_dir, excel_path)
                if not genmap_path.exists() and str(genmap_path) not in missing_files:
                    missing_files.append(str(genmap_path))
            
            # Check extra sources
            sources = genmap_cfg.get("sources", [])
            for source in sources[1:] if sources else []:  # Skip first (usually GAP)
                if isinstance(source, dict):
                    source_path = source.get("path") or source.get("excel")
                    if source_path:
                        src_path = resolve_input_path(dataset_dir, source_path)
                        if not src_path.exists() and str(src_path) not in missing_files:
                            missing_files.append(str(src_path))
                            missing_instructions.append(
                                f"  - Generation map source: {src_path.name}\n"
                                f"    Place in: {dataset_dir}"
                            )
    
    # IRENA files
    if "irena" in config:
        irena_cfg = config["irena"]
        input_dir = resolve_input_path(dataset_dir, irena_cfg.get("input_dir", "dataset"))
        input_files = irena_cfg.get("input_files", {})
        
        for tech, filename in input_files.items():
            irena_path = input_dir / filename
            if not irena_path.exists():
                missing_files.append(str(irena_path))
                info_key = f"irena_{tech}"
                if info_key in source_info:
                    info = source_info[info_key]
                    missing_instructions.append(
                        f"  - {info['name']}: {filename}\n"
                        f"    Download from: {info['url']}\n"
                        f"    Place in: {input_dir}"
                    )
    
    # Load profile (Toktarova)
    if "load_profile" in config:
        load_cfg = config["load_profile"]
        if "dataset" in load_cfg:
            load_path = resolve_input_path(dataset_dir, load_cfg["dataset"])
            if not load_path.exists():
                missing_files.append(str(load_path))
                info = source_info["toktarova_load"]
                missing_instructions.append(
                    f"  - {info['name']}: {load_path.name}\n"
                    f"    Download from: {info['url']}\n"
                    f"    Place in: {dataset_dir}"
                )
    
    # OWID Energy
    if "owid_energy" in config:
        owid_cfg = config.get("owid_energy", {})
        if owid_cfg.get("enabled", False) and "dataset" in owid_cfg:
            owid_path = resolve_input_path(dataset_dir, owid_cfg["dataset"])
            if not owid_path.exists():
                missing_files.append(str(owid_path))
                info = source_info["owid_energy"]
                missing_instructions.append(
                    f"  - {info['name']}: {owid_path.name}\n"
                    f"    Download from: {info['url']}\n"
                    f"    Place in: {dataset_dir}"
                )
    
    # Hydro Reservoirs
    if "hydro_reservoirs" in config:
        hydro_cfg = config.get("hydro_reservoirs", {})
        if hydro_cfg.get("enabled", False) and "dataset" in hydro_cfg:
            hydro_path = resolve_input_path(dataset_dir, hydro_cfg["dataset"])
            if not hydro_path.exists():
                missing_files.append(str(hydro_path))
                info = source_info["hydro_reservoirs"]
                missing_instructions.append(
                    f"  - {info['name']}: {hydro_path.name}\n"
                    f"    Download from: {info['url']}\n"
                    f"    Place in: {dataset_dir}"
                )
    
    # Socioeconomic maps - shapefiles and rasters
    if "socioeconomic_maps" in config:
        socio_cfg = config.get("socioeconomic_maps", {})
        if socio_cfg.get("enabled", False):
            # Check shapefiles
            for key in ["world_map_shapefile", "cities_shapefile"]:
                if key in socio_cfg:
                    shape_path = resolve_input_path(dataset_dir, socio_cfg[key])
                    if not shape_path.exists():
                        missing_files.append(str(shape_path))
                        missing_instructions.append(
                            f"  - Natural Earth shapefile: {shape_path.name}\n"
                            f"    Download from: {source_info['natural_earth_maps']['url']}\n"
                            f"    Place in: {shape_path.parent}"
                        )
            
            # Check raster files
            datasets = socio_cfg.get("datasets", [])
            for dataset in datasets:
                if isinstance(dataset, dict) and dataset.get("enabled", True):
                    raster = dataset.get("raster")
                    if raster:
                        raster_path = resolve_input_path(dataset_dir, raster)
                        if not raster_path.exists():
                            missing_files.append(str(raster_path))
                            missing_instructions.append(
                                f"  - Raster file: {raster_path.name}\n"
                                f"    Place in: {raster_path.parent}"
                            )
    
    return missing_files, missing_instructions


def list_missing_manual_sources(config: Dict, base_dir: Path) -> str:
    """Generate a formatted list of missing manual data sources with download instructions."""
    missing_files, missing_instructions = validate_data_sources(config, base_dir)
    
    if not missing_files:
        return "All required data sources are present.\n"
    
    message = f"Missing {len(missing_files)} required data source(s):\n\n"
    message += "\n".join(missing_instructions)
    message += f"\n\nFor most files, see WBG SharePoint:\n{SHAREPOINT_URL}\n"
    
    return message


def fetch_automated_sources(config: Dict, base_dir: Path, verbose: bool = True) -> Dict[str, bool]:
    """Fetch automated data sources (ERA5, Renewables Ninja).
    
    Note: This function validates prerequisites but actual fetching happens
    during workflow execution. Returns status of prerequisites.
    
    Returns:
        Dict mapping source name to whether prerequisites are met
    """
    results = {}
    
    # ERA5 - check if download is enabled and prerequisites exist
    if "climate_overview" in config:
        climate_cfg = config["climate_overview"]
        if climate_cfg.get("enabled", False) and climate_cfg.get("download", False):
            try:
                import cdsapi
                results["era5"] = True
                if verbose:
                    print("[data_fetcher] ERA5: CDS API available, downloads will occur during workflow")
            except ImportError:
                results["era5"] = False
                if verbose:
                    print("[data_fetcher] ERA5: cdsapi not installed. Install with: pip install cdsapi")
                    print("  Also ensure CDS API credentials are configured at ~/.cdsapirc")
    
    # Renewables Ninja - check if API token is available
    if "rninja" in config or "gap" in config:
        # Check for API token in environment or config file
        import os
        from configparser import ConfigParser
        
        token_available = False
        token_path_env = os.getenv("API_TOKENS_PATH")
        default_token_path = base_dir.parent / "config" / "api_tokens.ini"
        token_path = Path(token_path_env) if token_path_env else default_token_path
        
        if os.getenv("API_TOKEN_RENEWABLES_NINJA"):
            token_available = True
        elif token_path.exists():
            try:
                parser = ConfigParser()
                parser.read(token_path)
                if parser.has_section("api_tokens"):
                    token_available = parser.has_option("api_tokens", "renewables_ninja")
            except Exception:
                pass
        
        results["renewables_ninja"] = token_available
        if verbose:
            if token_available:
                print("[data_fetcher] Renewables Ninja: API token found")
            else:
                print("[data_fetcher] Renewables Ninja: API token not found")
                print("  Set API_TOKEN_RENEWABLES_NINJA environment variable or")
                print(f"  add 'renewables_ninja = <token>' to {token_path}")
    
    return results

