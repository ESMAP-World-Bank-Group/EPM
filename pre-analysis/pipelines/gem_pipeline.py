"""
gem_pipeline.py — GEM / GIPT data fetcher and normaliser for EPM.

Downloads and caches the Global Integrated Power Tracker (GIPT) from:
  https://github.com/GlobalEnergyMonitor/gipt-dashboard

Main entry point:
    from pipelines.gem_pipeline import load_gipt_plants
    plants = load_gipt_plants(countries=["AZE", "ARM"])
    # -> list of dicts: {name, fuel, mw, country, status, year, lat, lon}

Adapted from regional-power-explorer/tools/prepare_gem.py.
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

_MODULE_DIR = Path(__file__).resolve().parent
_BASE_DIR   = _MODULE_DIR.parent
DEFAULT_CACHE_DIR = _BASE_DIR / "cache" / "gem"

# ── Constants ──────────────────────────────────────────────────────────────────

GIPT_REPO_API = "https://api.github.com/repos/GlobalEnergyMonitor/gipt-dashboard/contents/"
GIPT_RAW_BASE = "https://raw.githubusercontent.com/GlobalEnergyMonitor/gipt-dashboard/main/"

# Also try the regional-power-explorer cache as a fallback (avoids re-download)
# _BASE_DIR = pre-analysis/  → parent = EPM/  → parent = black_sea_2026/  → parent = EPM_Models/
_RPE_GEM_DIR = _BASE_DIR.parent.parent.parent / "regional-power-explorer" / "data-source" / "gem"

GIPT_FUEL_MAP = {
    "coal":       "coal",
    "oil/gas":    "gas",
    "solar":      "solar",
    "wind":       "wind",
    "hydropower": "hydro",
    "nuclear":    "nuclear",
    "bioenergy":  "biomass",
    "geothermal": "geothermal",
}

STATUS_MAP = {
    "operating":          "operating",
    "construction":       "construction",
    "under construction": "construction",
    "pre-construction":   "planned",
    "pre-permit":         "planned",
    "permitted":          "planned",
    "authorized":         "planned",
    "announced":          "planned",
    "discovered":         "planned",
    "proposed":           "planned",
    "shelved":            None,
    "mothballed":         None,
    "cancelled":          None,
    "retired":            None,
    "decommissioned":     None,
    "coal-to-gas":        "operating",
}

COUNTRY_ISO = {
    "afghanistan":"AFG","albania":"ALB","algeria":"DZA","angola":"AGO","argentina":"ARG",
    "armenia":"ARM","australia":"AUS","austria":"AUT","azerbaijan":"AZE","bahrain":"BHR",
    "bangladesh":"BGD","belarus":"BLR","belgium":"BEL","benin":"BEN","bhutan":"BTN",
    "bolivia":"BOL","bosnia and herzegovina":"BIH","botswana":"BWA","brazil":"BRA",
    "bulgaria":"BGR","burkina faso":"BFA","burundi":"BDI","cambodia":"KHM","cameroon":"CMR",
    "canada":"CAN","central african republic":"CAF","chad":"TCD","chile":"CHL","china":"CHN",
    "colombia":"COL","democratic republic of the congo":"COD","dr congo":"COD","drc":"COD",
    "congo":"COG","costa rica":"CRI","croatia":"HRV","cuba":"CUB","cyprus":"CYP",
    "czech republic":"CZE","czechia":"CZE","denmark":"DNK","djibouti":"DJI",
    "dominican republic":"DOM","ecuador":"ECU","egypt":"EGY","el salvador":"SLV",
    "eritrea":"ERI","estonia":"EST","eswatini":"SWZ","ethiopia":"ETH","finland":"FIN",
    "france":"FRA","gabon":"GAB","gambia":"GMB","georgia":"GEO","germany":"DEU",
    "ghana":"GHA","greece":"GRC","guatemala":"GTM","guinea":"GIN","guinea-bissau":"GNB",
    "haiti":"HTI","honduras":"HND","hungary":"HUN","india":"IND","indonesia":"IDN",
    "iran":"IRN","iran, islamic republic of":"IRN","iraq":"IRQ","ireland":"IRL",
    "israel":"ISR","italy":"ITA","ivory coast":"CIV","côte d'ivoire":"CIV",
    "cote d'ivoire":"CIV","jamaica":"JAM","japan":"JPN","jordan":"JOR","kazakhstan":"KAZ",
    "kenya":"KEN","kosovo":"XKX","kuwait":"KWT","kyrgyzstan":"KGZ","laos":"LAO",
    "latvia":"LVA","lebanon":"LBN","lesotho":"LSO","liberia":"LBR","libya":"LBY",
    "lithuania":"LTU","luxembourg":"LUX","madagascar":"MDG","malawi":"MWI","malaysia":"MYS",
    "mali":"MLI","mauritania":"MRT","mauritius":"MUS","mexico":"MEX","moldova":"MDA",
    "mongolia":"MNG","montenegro":"MNE","morocco":"MAR","mozambique":"MOZ","myanmar":"MMR",
    "namibia":"NAM","nepal":"NPL","netherlands":"NLD","new zealand":"NZL","nicaragua":"NIC",
    "niger":"NER","nigeria":"NGA","north korea":"PRK","north macedonia":"MKD","norway":"NOR",
    "oman":"OMN","pakistan":"PAK","palestine":"PSE","state of palestine":"PSE",
    "west bank and gaza":"PSE","panama":"PAN","papua new guinea":"PNG","paraguay":"PRY",
    "peru":"PER","philippines":"PHL","poland":"POL","portugal":"PRT","qatar":"QAT",
    "romania":"ROU","russia":"RUS","russian federation":"RUS","rwanda":"RWA",
    "saudi arabia":"SAU","senegal":"SEN","serbia":"SRB","sierra leone":"SLE",
    "somalia":"SOM","south africa":"ZAF","south korea":"KOR","korea, republic of":"KOR",
    "south sudan":"SSD","spain":"ESP","sri lanka":"LKA","sudan":"SDN","sweden":"SWE",
    "switzerland":"CHE","syria":"SYR","taiwan":"TWN","tajikistan":"TJK","tanzania":"TZA",
    "thailand":"THA","togo":"TGO","trinidad and tobago":"TTO","tunisia":"TUN",
    "turkey":"TUR","türkiye":"TUR","turkmenistan":"TKM","uganda":"UGA","ukraine":"UKR",
    "united arab emirates":"ARE","uae":"ARE","united kingdom":"GBR","uk":"GBR",
    "united states":"USA","usa":"USA","united states of america":"USA","uruguay":"URY",
    "uzbekistan":"UZB","venezuela":"VEN","viet nam":"VNM","vietnam":"VNM","yemen":"YEM",
    "zambia":"ZMB","zimbabwe":"ZWE","brunei":"BRN","brunei darussalam":"BRN",
    "timor-leste":"TLS","east timor":"TLS","singapore":"SGP",
}

ISO2_TO_ISO3 = {
    "AF":"AFG","AL":"ALB","DZ":"DZA","AO":"AGO","AR":"ARG","AM":"ARM","AU":"AUS",
    "AT":"AUT","AZ":"AZE","BH":"BHR","BD":"BGD","BY":"BLR","BE":"BEL","BJ":"BEN",
    "BT":"BTN","BO":"BOL","BA":"BIH","BW":"BWA","BR":"BRA","BG":"BGR","BF":"BFA",
    "BI":"BDI","KH":"KHM","CM":"CMR","CA":"CAN","CF":"CAF","TD":"TCD","CL":"CHL",
    "CN":"CHN","CO":"COL","KM":"COM","CG":"COG","CD":"COD","CR":"CRI","HR":"HRV",
    "CU":"CUB","CY":"CYP","CZ":"CZE","DK":"DNK","DJ":"DJI","DO":"DOM","EC":"ECU",
    "EG":"EGY","SV":"SLV","ER":"ERI","EE":"EST","SZ":"SWZ","ET":"ETH","FI":"FIN",
    "FR":"FRA","GA":"GAB","GM":"GMB","GE":"GEO","DE":"DEU","GH":"GHA","GR":"GRC",
    "GT":"GTM","GN":"GIN","GW":"GNB","HT":"HTI","HN":"HND","HU":"HUN","IN":"IND",
    "ID":"IDN","IR":"IRN","IQ":"IRQ","IE":"IRL","IL":"ISR","IT":"ITA","CI":"CIV",
    "JM":"JAM","JP":"JPN","JO":"JOR","KZ":"KAZ","KE":"KEN","XK":"XKX","KW":"KWT",
    "KG":"KGZ","LA":"LAO","LV":"LVA","LB":"LBN","LS":"LSO","LR":"LBR","LY":"LBY",
    "LT":"LTU","LU":"LUX","MG":"MDG","MW":"MWI","MY":"MYS","MV":"MDV","ML":"MLI",
    "MR":"MRT","MU":"MUS","MX":"MEX","MD":"MDA","MN":"MNG","ME":"MNE","MA":"MAR",
    "MZ":"MOZ","MM":"MMR","NA":"NAM","NP":"NPL","NL":"NLD","NZ":"NZL","NI":"NIC",
    "NE":"NER","NG":"NGA","KP":"PRK","MK":"MKD","NO":"NOR","OM":"OMN","PK":"PAK",
    "PS":"PSE","PA":"PAN","PG":"PNG","PY":"PRY","PE":"PER","PH":"PHL","PL":"POL",
    "PT":"PRT","QA":"QAT","RO":"ROU","RU":"RUS","RW":"RWA","SA":"SAU","SN":"SEN",
    "RS":"SRB","SL":"SLE","SO":"SOM","ZA":"ZAF","KR":"KOR","SS":"SSD","ES":"ESP",
    "LK":"LKA","SD":"SDN","SE":"SWE","CH":"CHE","SY":"SYR","TW":"TWN","TJ":"TJK",
    "TZ":"TZA","TH":"THA","TG":"TGO","TT":"TTO","TN":"TUN","TR":"TUR","TM":"TKM",
    "UG":"UGA","UA":"UKR","AE":"ARE","GB":"GBR","US":"USA","UY":"URY","UZ":"UZB",
    "VE":"VEN","VN":"VNM","YE":"YEM","ZM":"ZMB","ZW":"ZWE","BN":"BRN","SG":"SGP",
    "TL":"TLS",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _name_to_iso(name: str) -> Optional[str]:
    if not name:
        return None
    s = str(name).strip()
    if len(s) == 3 and s.isupper():
        return s
    if len(s) == 2 and s.isupper():
        return ISO2_TO_ISO3.get(s)
    return COUNTRY_ISO.get(s.lower())


def _map_status(raw) -> Optional[str]:
    if not raw or str(raw).strip().lower() in ("nan", "none", ""):
        return "operating"
    key = re.sub(r"\s*-\s*inferred\s+\d+\s*y.*$", "", str(raw).strip().lower()).strip()
    return STATUS_MAP.get(key)


def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    lc = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        match = lc.get(cand.strip().lower())
        if match:
            return match
    return None


# ── GIPT download ──────────────────────────────────────────────────────────────

def _find_rpe_gipt() -> Optional[Path]:
    """Return the cached GIPT file from regional-power-explorer if available."""
    if _RPE_GEM_DIR.exists():
        files = sorted(_RPE_GEM_DIR.glob("_gipt_Global Integrated Power*.xlsx"))
        if files:
            return files[-1]
    return None


def fetch_gipt(cache_dir: Path = DEFAULT_CACHE_DIR, force: bool = False) -> Optional[Path]:
    """Download (or return cached) GIPT xlsx file.

    Search order:
      1. EPM local cache_dir
      2. regional-power-explorer cache (reuse without re-download)
      3. GitHub download
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check EPM local cache
    local = sorted(cache_dir.glob("_gipt_*.xlsx"))
    if local and not force:
        print(f"  [gem] Using cached GIPT: {local[-1].name}")
        return local[-1]

    # Check regional-power-explorer cache (no copy needed — read in place)
    rpe = _find_rpe_gipt()
    if rpe and not force:
        print(f"  [gem] Reusing GIPT from regional-power-explorer: {rpe.name}")
        return rpe

    # Download from GitHub
    print("  [gem] Fetching GIPT file list from GitHub...")
    try:
        req = urllib.request.Request(GIPT_REPO_API, headers={"User-Agent": "EPM-gem_pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            files = json.loads(resp.read())
    except Exception as e:
        print(f"  [gem] GitHub API error: {e}")
        return None

    gipt_files = [
        f for f in files
        if f["name"].startswith("Global Integrated Power") and f["name"].endswith(".xlsx")
    ]
    if not gipt_files:
        print("  [gem] No GIPT xlsx in GitHub repo root.")
        return None

    latest = sorted(gipt_files, key=lambda f: f["name"])[-1]
    name = latest["name"]
    # Sanitize filename for Windows (remove non-ASCII chars like →)
    safe_name = name.encode("ascii", "ignore").decode("ascii").strip()
    dest = cache_dir / f"_gipt_{safe_name}"
    url  = GIPT_RAW_BASE + urllib.parse.quote(name)

    print(f"  [gem] Downloading {name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [gem] Saved {dest.stat().st_size / 1e6:.1f} MB → {dest.name}")
        return dest
    except Exception as e:
        print(f"  [gem] Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return None


# ── GIPT processing ────────────────────────────────────────────────────────────

def _process_gipt(path: Path) -> List[dict]:
    """Parse GIPT xlsx → list of normalised plant dicts."""
    df = pd.read_excel(path, sheet_name="Power facilities", engine="openpyxl")
    df.replace("not found", pd.NA, inplace=True)

    lat_col  = _find_col(df, "latitude", "lat")
    lon_col  = _find_col(df, "longitude", "lon", "long")
    mw_col   = _find_col(df, "capacity (mw)", "capacity_mw", "mw", "capacity")
    stat_col = _find_col(df, "status")
    name_col = _find_col(df, "plant / project name", "plant name", "project name", "name")
    year_col = _find_col(df, "start year", "commissioning year", "year")
    ctry_col = _find_col(df, "country/area", "country")
    type_col = _find_col(df, "type")

    if not lat_col or not lon_col or not type_col:
        raise ValueError("GIPT file missing required columns (lat, lon, or type).")

    plants = []
    for _, row in df.iterrows():
        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            if math.isnan(lat) or math.isnan(lon):
                continue
        except (ValueError, TypeError):
            continue

        raw_type = str(row[type_col]).strip().lower() if pd.notna(row[type_col]) else ""
        fuel = GIPT_FUEL_MAP.get(raw_type)
        if not fuel:
            continue

        status = _map_status(row[stat_col] if stat_col else None)
        if status is None:
            continue

        mw = None
        if mw_col:
            try:
                v = float(row[mw_col])
                if not math.isnan(v) and not math.isinf(v) and v > 0:
                    mw = round(v, 1)
            except (ValueError, TypeError):
                pass

        name = ""
        if name_col and pd.notna(row[name_col]):
            name = str(row[name_col]).strip()

        iso = _name_to_iso(str(row[ctry_col]).strip() if ctry_col and pd.notna(row[ctry_col]) else None)

        year = None
        if year_col:
            try:
                yv = float(row[year_col])
                if not math.isnan(yv) and 1900 <= yv <= 2050:
                    year = int(yv)
            except (ValueError, TypeError):
                pass

        plants.append({
            "lat":     round(lat, 3),
            "lon":     round(lon, 3),
            "name":    name,
            "fuel":    fuel,
            "mw":      mw,
            "country": iso,
            "status":  status,
            "year":    year,
        })

    return plants


# ── Public API ─────────────────────────────────────────────────────────────────

def load_gipt_plants(
    countries: Optional[List[str]] = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load GIPT plants, optionally filtered by ISO-3 country codes.

    Returns a DataFrame with columns:
        name, fuel, mw, country, status, year, lat, lon
    where status ∈ {operating, construction, planned}.
    """
    gipt_path = fetch_gipt(cache_dir=cache_dir, force=force)
    if gipt_path is None:
        raise RuntimeError(
            "GIPT file not available. Check internet connection or place "
            f"a manual GIPT xlsx in {cache_dir}."
        )

    if verbose:
        print(f"  [gem] Parsing {gipt_path.name} ...")
    plants = _process_gipt(gipt_path)

    df = pd.DataFrame(plants)
    if df.empty:
        return df

    if countries:
        df = df[df["country"].isin(countries)].copy()

    if verbose:
        total = len(df)
        op = (df["status"] == "operating").sum()
        co = (df["status"] == "construction").sum()
        pl = (df["status"] == "planned").sum()
        print(f"  [gem] {total} plants ({op} operating, {co} construction, {pl} planned)")

    return df.reset_index(drop=True)
