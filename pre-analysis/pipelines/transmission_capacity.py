"""
Transmission capacity estimation and corridors.csv management.

Replaces the crude voltage proxy with physics-based estimates:
  - Thermal limit  : P = sqrt(3) * V * I_rated * PF  (conductor heating)
  - Stability limit: P = V^2 * sin(delta_max) / (x * L)  (angular stability)
  - Capacity = min(thermal, stability) * n_circuits

Also handles reference data override (reference_lines.csv) and exports
to EPM-ready pTransferLimit.csv / pNewTransmission.csv.

corridors.csv columns
---------------------
z, zz            : zone pair (sorted alphabetically)
from_country     : ISO3 of zone z
to_country       : ISO3 of zone zz
voltage_kv       : max voltage among OSM lines on corridor (0 if no OSM data)
n_osm_lines      : number of OSM lines found
length_km        : mean line length from OSM geometry (0 if unknown)
mw_osm           : physics-based estimate (thermal+stability)
mw_override      : manually set capacity — takes priority over mw_osm if non-empty
status           : existing | under_construction | planned | candidate | long_term | cold_standby
earliest_entry   : first year line is active (for under_construction/planned)
cost_musd        : investment cost in M$ (for planned/candidate, optional)
note             : free text

Status → EPM mapping
---------------------
existing / under_construction → pTransferLimit.csv
planned / candidate / long_term  → pNewTransmission.csv
cold_standby                  → excluded
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Physical constants ────────────────────────────────────────────────────────

# Typical ACSR conductor thermal ratings and inductive reactance per km
# (I_rated_A, x_Ohm_per_km) — single circuit, overhead line
_THERMAL_PARAMS: dict[int, tuple[float, float]] = {
    110: (600,  0.40),
    154: (700,  0.40),
    220: (800,  0.40),
    330: (1000, 0.38),
    400: (1200, 0.38),
    500: (1400, 0.37),
    765: (1600, 0.36),
}
_PF        = 0.95           # power factor
_DELTA_MAX = math.radians(30)  # max angle difference for N-1 security


def _nearest_voltage(v: float) -> int:
    """Snap voltage to nearest entry in _THERMAL_PARAMS."""
    return min(_THERMAL_PARAMS.keys(), key=lambda k: abs(k - v))


def thermal_limit_mw(voltage_kv: float, n_circuits: int = 1) -> float:
    """Thermal capacity of a line in MW (conductor heating limit)."""
    if voltage_kv <= 0:
        return 0.0
    v = _nearest_voltage(voltage_kv)
    I_rated, _ = _THERMAL_PARAMS[v]
    return math.sqrt(3) * voltage_kv * I_rated * _PF / 1000 * n_circuits


def stability_limit_mw(voltage_kv: float, length_km: float, n_circuits: int = 1) -> float:
    """Angular stability limit in MW (N-1, delta_max = 30°)."""
    if voltage_kv <= 0 or length_km <= 0:
        return float("inf")  # unknown length → thermal limits
    v = _nearest_voltage(voltage_kv)
    _, x_km = _THERMAL_PARAMS[v]
    X_total = x_km * length_km  # total line reactance [Ω]
    p_per_circuit = (voltage_kv ** 2 * math.sin(_DELTA_MAX)) / X_total
    return p_per_circuit * n_circuits


def estimate_capacity_mw(voltage_kv: float, length_km: float, n_circuits: int = 1) -> int:
    """
    Best-estimate line capacity: min(thermal, stability) * n_circuits.

    For a 400 kV, 200 km, 1-circuit line:
      thermal  = sqrt(3) * 400 * 1200 * 0.95 / 1000 = 789 MW
      stability= 400^2 * sin(30°) / (0.38 * 200)    = 1053 MW
      → capacity = 789 MW  (thermally limited)

    For a 400 kV, 600 km line (stability-limited):
      stability= 400^2 * 0.5 / (0.38 * 600) = 351 MW
      → capacity = 351 MW
    """
    p_th = thermal_limit_mw(voltage_kv, n_circuits)
    p_st = stability_limit_mw(voltage_kv, length_km, n_circuits)
    return max(0, round(min(p_th, p_st)))


# ── Geodesic helpers ──────────────────────────────────────────────────────────

def _geodesic_km(coords: list) -> float:
    """Approximate great-circle length of a coordinate sequence [(lon, lat), ...]."""
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        total += 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return total


# ── Reference lines ───────────────────────────────────────────────────────────

def load_reference_lines(path) -> pd.DataFrame:
    """
    Load reference_lines.csv.  Comment lines starting with # are ignored.
    Returns empty DataFrame if file not found.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip().str.lower()
    for col in ("from_country", "to_country"):
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()
    return df


def _ref_index(ref_df: pd.DataFrame) -> dict[tuple, list]:
    """Index reference rows by sorted (from_country, to_country) pair."""
    idx: dict[tuple, list] = defaultdict(list)
    for _, row in ref_df.iterrows():
        fc = str(row.get("from_country", "")).upper()
        tc = str(row.get("to_country", "")).upper()
        idx[tuple(sorted([fc, tc]))].append(row)
    return idx


# ── Corridor builder ──────────────────────────────────────────────────────────

def build_corridors(
    interzone_lines: list[dict],
    zones_gdf,
    reference_lines_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build a corridors DataFrame from OSM inter-zone lines + optional reference data.

    Priority for mw_override:
      1. Reference lines with status=existing/under_construction → sum by country pair
      2. Physical estimate (thermal+stability) fills mw_osm for all corridors

    Planned/candidate lines in reference that have no OSM match are added as
    additional rows with mw_osm=0 and a zone-assignment warning.

    Parameters
    ----------
    interzone_lines : list of dicts with keys zone_from, zone_to, voltage_kv, coords
    zones_gdf       : GeoDataFrame with columns zone_name, ISO_A3
    reference_lines_path : path to reference_lines.csv (optional)
    """
    # ── Zone → country map ────────────────────────────────────────────────────
    zone_country: dict[str, str] = {}
    if zones_gdf is not None:
        for _, row in zones_gdf.iterrows():
            zone_country[row["zone_name"]] = str(row.get("ISO_A3", "")).upper()

    # ── Reference data ────────────────────────────────────────────────────────
    ref_df = load_reference_lines(reference_lines_path) if reference_lines_path else pd.DataFrame()
    ref_idx = _ref_index(ref_df) if not ref_df.empty else {}

    # ── Aggregate OSM lines by zone pair ──────────────────────────────────────
    pair_lines: dict[tuple, list] = defaultdict(list)
    for line in interzone_lines:
        zf, zt = line["zone_from"], line["zone_to"]
        pair = tuple(sorted([zf, zt]))
        pair_lines[pair].append(line)

    # ── Build corridor rows ───────────────────────────────────────────────────
    rows: list[dict] = []
    matched_country_pairs: set[tuple] = set()

    for pair, lines in pair_lines.items():
        z1, z2 = pair
        c1 = zone_country.get(z1, "")
        c2 = zone_country.get(z2, "")
        country_pair = tuple(sorted([c1, c2]))

        # Physical capacity: sum over all OSM lines on this corridor
        mw_osm = 0
        voltage_max = 0
        lengths = []
        for ln in lines:
            v = float(ln.get("voltage_kv") or 0)
            nc = int(ln.get("n_circuits") or 1)
            coords = ln.get("coords", [])
            length = _geodesic_km(coords) if len(coords) >= 2 else 0.0
            lengths.append(length)
            if v > 0:
                mw_osm += estimate_capacity_mw(v, length, nc)
                voltage_max = max(voltage_max, v)

        mean_length = round(sum(lengths) / len(lengths), 1) if lengths else 0

        # Reference override: sum existing/under_construction MW for country pair
        mw_override = ""
        note = f"{len(lines)} OSM line(s) · physics calc (thermal+stability)"
        if country_pair in ref_idx:
            matched_country_pairs.add(country_pair)
            ref_existing = [
                r for r in ref_idx[country_pair]
                if str(r.get("status", "")).lower() in ("existing", "under_construction")
            ]
            if ref_existing:
                total = 0.0
                for r in ref_existing:
                    val = str(r.get("mw_fwd", "0")).replace(",", ".")
                    try:
                        total += float(val)
                    except ValueError:
                        pass
                if total > 0:
                    mw_override = int(round(total))
                    subs = " + ".join(
                        f"{r.get('from_substation', '')}→{r.get('to_substation', '')}"
                        for r in ref_existing
                    )
                    note = f"Reference data: {subs}"

        rows.append({
            "z":             z1,
            "zz":            z2,
            "from_country":  c1,
            "to_country":    c2,
            "voltage_kv":    int(voltage_max) if voltage_max > 0 else "",
            "n_osm_lines":   len(lines),
            "length_km":     mean_length,
            "mw_osm":        mw_osm,
            "mw_override":   mw_override,
            "status":        "existing",
            "earliest_entry": "",
            "cost_musd":     "",
            "note":          note,
        })

    # ── Add planned/candidate from reference with no OSM match ───────────────
    for country_pair, ref_rows in ref_idx.items():
        planned = [
            r for r in ref_rows
            if str(r.get("status", "")).lower() in ("planned", "candidate", "long_term", "under_construction")
        ]
        for ref_row in planned:
            c1, c2 = country_pair
            # Find zones for each country
            zones_c1 = [z for z, c in zone_country.items() if c == c1]
            zones_c2 = [z for z, c in zone_country.items() if c == c2]
            if not zones_c1 or not zones_c2:
                continue  # country not in this study
            # Use first zone as placeholder — user should refine
            z1, z2 = sorted([zones_c1[0], zones_c2[0]])
            pair = (z1, z2)
            # Skip if OSM already found a corridor for this zone pair
            if pair in {(r["z"], r["zz"]) for r in rows}:
                continue

            status = str(ref_row.get("status", "planned")).lower()
            mw_val = ""
            try:
                mw_val = int(round(float(str(ref_row.get("mw_fwd", "0")).replace(",", "."))))
            except ValueError:
                pass

            rows.append({
                "z":             z1,
                "zz":            z2,
                "from_country":  c1,
                "to_country":    c2,
                "voltage_kv":    ref_row.get("voltage_kv", ""),
                "n_osm_lines":   0,
                "length_km":     ref_row.get("length_km", ""),
                "mw_osm":        0,
                "mw_override":   mw_val,
                "status":        status,
                "earliest_entry": ref_row.get("earliest_entry", ""),
                "cost_musd":     ref_row.get("cost_musd", ""),
                "note":          (
                    f"Reference ({ref_row.get('from_substation', '')} → "
                    f"{ref_row.get('to_substation', '')}) "
                    f"⚠ zone assignment approximate ({z1}/{z2}) — please verify"
                ),
            })

    return pd.DataFrame(rows) if rows else _empty_corridors()


def _empty_corridors() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "z", "zz", "from_country", "to_country",
        "voltage_kv", "n_osm_lines", "length_km",
        "mw_osm", "mw_override", "status", "earliest_entry", "cost_musd", "note",
    ])


# ── Safe save (preserve overrides) ───────────────────────────────────────────

def save_corridors(new_df: pd.DataFrame, path: Path) -> None:
    """
    Write corridors.csv, preserving mw_override and manually-added rows
    from any previously saved file.

    Rules:
    - Rows matched by (z, zz): keep existing mw_override if non-empty;
      update mw_osm with fresh calculation.
    - Rows in existing file but NOT in new_df (manually added): keep as-is.
    - Rows in new_df but NOT in existing file: add as new.
    """
    if not path.exists():
        new_df.to_csv(path, index=False)
        return

    old_df = pd.read_csv(path, dtype=str).fillna("")
    old_idx = {(r["z"], r["zz"]): r for _, r in old_df.iterrows()}
    new_idx = {(r["z"], r["zz"]): r for _, r in new_df.iterrows()}

    merged_rows = []
    seen = set()

    for pair, new_row in new_idx.items():
        seen.add(pair)
        if pair in old_idx:
            old_row = old_idx[pair]
            # Preserve mw_override if the user had set it
            if str(old_row.get("mw_override", "")).strip() not in ("", "0"):
                new_row = new_row.copy()
                new_row["mw_override"] = old_row["mw_override"]
        merged_rows.append(new_row)

    # Keep manually-added rows not present in new computation
    for pair, old_row in old_idx.items():
        if pair not in seen:
            merged_rows.append(old_row)

    pd.DataFrame(merged_rows).to_csv(path, index=False)


# ── EPM export ────────────────────────────────────────────────────────────────

_DEFAULT_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]


def export_epm_csvs(
    corridors_df: pd.DataFrame,
    output_dir: Path,
    years: Optional[list[int]] = None,
) -> dict[str, Path]:
    """
    Convert corridors.csv → EPM-ready pTransferLimit.csv + pNewTransmission.csv.

    pTransferLimit : existing + under_construction corridors, one row per (z, zz, y)
    pNewTransmission: planned + candidate + long_term corridors
    """
    years = years or _DEFAULT_YEARS
    paths: dict[str, Path] = {}

    # ── pTransferLimit ────────────────────────────────────────────────────────
    existing_mask = corridors_df["status"].str.lower().isin(["existing", "under_construction"])
    existing = corridors_df[existing_mask].copy()

    tl_rows = []
    for _, row in existing.iterrows():
        mw = _effective_mw(row)
        if mw == 0:
            continue
        entry = int(row["earliest_entry"]) if str(row.get("earliest_entry", "")).isdigit() else None
        for y in years:
            active_mw = mw if (entry is None or y >= entry) else 0
            tl_rows.append({"z": row["z"], "z2": row["zz"], "y": y, "pTransferLimit": active_mw})

    tl_path = output_dir / "pTransferLimit_estimated.csv"
    pd.DataFrame(tl_rows).to_csv(tl_path, index=False)
    paths["pTransferLimit_estimated.csv"] = tl_path

    # ── pNewTransmission ──────────────────────────────────────────────────────
    new_mask = corridors_df["status"].str.lower().isin(["planned", "candidate", "long_term"])
    new_lines = corridors_df[new_mask].copy()

    nt_rows = []
    for _, row in new_lines.iterrows():
        mw = _effective_mw(row)
        status_code = 2 if row["status"].lower() == "planned" else 3
        nt_rows.append({
            "z1":              row["z"],
            "z2":              row["zz"],
            "EarliestEntry":   row.get("earliest_entry", ""),
            "MaximumNumOfLines": 1,
            "CapacityPerLine": mw,
            "CostPerLine":     row.get("cost_musd", ""),
            "Life":            40,
            "Status":          status_code,
            "note":            row.get("note", ""),
        })

    nt_path = output_dir / "pNewTransmission_estimated.csv"
    pd.DataFrame(nt_rows).to_csv(nt_path, index=False)
    paths["pNewTransmission_estimated.csv"] = nt_path

    return paths


def _effective_mw(row) -> int:
    """Return mw_override if set, else mw_osm."""
    override = str(row.get("mw_override", "")).strip()
    if override not in ("", "0"):
        try:
            return int(round(float(override.replace(",", "."))))
        except ValueError:
            pass
    try:
        return int(round(float(str(row.get("mw_osm", 0)))))
    except ValueError:
        return 0


# ── GeoJSON export ────────────────────────────────────────────────────────────

def export_corridors_geojson(corridors_df: pd.DataFrame, zones_gdf, output_path) -> None:
    """
    Write corridors as centroid-to-centroid LineString GeoJSON.

    Properties per feature: zone_a, zone_b, mw, status, label.
    Existing corridors use status='existing'; planned use 'planned' etc.
    Skips any pair where zone centroids cannot be computed.
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    centroids: dict[str, list] = {}
    if zones_gdf is not None:
        for _, row in zones_gdf.iterrows():
            name = row["zone_name"]
            try:
                c = row.geometry.centroid
                centroids[name] = [round(c.x, 5), round(c.y, 5)]
            except Exception:
                pass

    features = []
    for _, row in corridors_df.iterrows():
        z1, z2 = str(row["z"]), str(row["zz"])
        if z1 not in centroids or z2 not in centroids:
            continue
        mw = _effective_mw(row)
        status = str(row.get("status", "existing")).lower()
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [centroids[z1], centroids[z2]],
            },
            "properties": {
                "zone_a": z1,
                "zone_b": z2,
                "mw": mw,
                "status": status,
                "label": f"{mw:,} MW" if mw > 0 else "",
            },
        })

    gj = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gj, f, separators=(",", ":"))
