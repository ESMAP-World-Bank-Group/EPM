"""
Fetch and cache the Global Power Plant Database (WRI).
Source: https://github.com/wri/global-power-plant-database
~35,000 plants worldwide, columns: country, name, latitude, longitude,
primary_fuel, capacity_mw, commissioning_year, etc.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import requests
import pandas as pd

# Try multiple URLs in order (WRI has moved the file between releases)
GPPD_URLS = [
    # Direct CSV from WRI GitHub (v1.3.0 branch)
    "https://raw.githubusercontent.com/wri/global-power-plant-database/v1.3.0/global_power_plant_database.csv",
    # Main branch (may or may not still have the CSV)
    "https://raw.githubusercontent.com/wri/global-power-plant-database/master/global_power_plant_database.csv",
    # WRI official data portal (v1.3.0 zip)
    "https://datasets.wri.org/datafiles/global_power_plant_database_v_1_3_0.zip",
    # GitHub Releases zip
    "https://github.com/wri/global-power-plant-database/releases/download/v1.3.0/global_power_plant_database_v_1_3_0.zip",
    # ESRI ArcGIS Hub mirror (public open data)
    "https://opendata.arcgis.com/datasets/547d2f6bd4794da0bbb42c12e8b0d424_0.csv",
]
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "gppd"
CACHE_FILE = CACHE_DIR / "global_power_plant_database.csv"

# GPPD uses its own 3-letter codes -- mostly ISO_A3 but a few differ.
# Mapping for the countries we care about (all match ISO_A3 here).
GPPD_ISO_MAP: dict[str, str] = {
    "TUR": "TUR",
    "ROU": "ROU",
    "BGR": "BGR",
    "GEO": "GEO",
    "ARM": "ARM",
    "AZE": "AZE",
}


def load_gppd(countries: List[str] | None = None,
              force_download: bool = False) -> pd.DataFrame:
    """
    Load GPPD, downloading and caching if needed.

    countries: list of ISO_A3 codes to filter (None = all)
    Returns DataFrame with standardised columns.
    """
    if not CACHE_FILE.exists() or force_download:
        _download_gppd()

    if not CACHE_FILE.exists():
        return pd.DataFrame(columns=["country_iso", "country_name", "plant_name",
                                     "gppd_id", "capacity_mw", "lat", "lon",
                                     "fuel", "year_built"])

    df = pd.read_csv(CACHE_FILE, low_memory=False)
    df = _normalise(df)

    if countries:
        gppd_codes = [GPPD_ISO_MAP.get(c, c) for c in countries]
        df = df[df["country_iso"].isin(gppd_codes)]

    return df.reset_index(drop=True)


# -- internals -----------------------------------------------------------------

def _download_gppd():
    import io, zipfile
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in GPPD_URLS:
        print(f"  [GPPD] Trying {url.split('/')[-1]}...")
        try:
            resp = requests.get(url, timeout=120, verify=False,
                                headers={"User-Agent": "EPM-ResolutionAdvisor/1.0"})
            resp.raise_for_status()
            content = resp.content
            # Handle zip files
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                    if not csv_names:
                        continue
                    content = z.read(csv_names[0])
            CACHE_FILE.write_bytes(content)
            print(f"  [GPPD] Saved to {CACHE_FILE}")
            return
        except Exception as e:
            last_err = e
            print(f"  [GPPD] Failed: {e}")
    print(
        "\n  [GPPD] Auto-download failed (all URLs tried). GPPD data will be skipped.\n"
        "  To enable power plant data, manually download the CSV:\n"
        "    1. Go to: https://datasets.wri.org/dataset/globalpowerplantdatabase\n"
        "    2. Download 'global_power_plant_database.csv'\n"
        f"    3. Place it at: {CACHE_FILE}\n"
    )
    # Don't raise -- caller handles missing GPPD gracefully


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to a compact, consistent schema."""
    rename = {
        "country": "country_iso",
        "country_long": "country_name",
        "name": "plant_name",
        "gppd_idnr": "gppd_id",
        "capacity_mw": "capacity_mw",
        "latitude": "lat",
        "longitude": "lon",
        "primary_fuel": "fuel",
        "commissioning_year": "year_built",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["country_iso", "country_name", "plant_name", "gppd_id",
            "capacity_mw", "lat", "lon", "fuel", "year_built"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df
