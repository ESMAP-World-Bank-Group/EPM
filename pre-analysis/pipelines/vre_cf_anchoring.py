"""vre_cf_anchoring.py - Anchor VRE profile *levels* to a resource atlas.

Problem this solves
-------------------
The VRE pipeline samples Renewables Ninja at ONE point per zone (the polygon
centroid). A single coarse (~50 km MERRA-2) point is a poor estimate of a
zone's *developable* capacity factor (CF): it can land on a freak-good ridge
(e.g. Trakia ~0.49) or a low spot (e.g. Armenia ~0.14).

Method
------
Decompose each zone/tech profile into  shape (hourly pattern) x level (annual
mean CF).  Keep the Ninja *shape* (its strength: hourly timing), but reset the
*level* to a resource-atlas target:

    profile_final(h) = shape_ninja(h) / weighted_mean(shape_ninja)  x  cf_target

where cf_target = mean CF over the windiest `percentile` share of a zone's
eligible atlas pixels ("best X% of the land"), area-weighted (cos lat).

The rescaling is exact at the level of the model's representative slices when
the weighted mean uses pHours weights.

Generic by design
-----------------
- No hard-coded zone/country names: zones come from a GeoJSON, routed to their
  country raster via an ISO3 column.
- Raster-agnostic: works for any CF-like GeoTIFF (GWA wind today; Global Solar
  Atlas PVOUT later) -> same code path for wind and PV.
- Pure functions + a thin CLI; every number is written to an audit table.

Data source (auto-fetch): Global Wind Atlas v4 capacity-factor rasters
    GET https://globalwindatlas.info/api/gis/country/{ISO3}/{layer}
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask

GWA_URL = "https://globalwindatlas.info/api/gis/country/{iso3}/{layer}"


# ── Fetch ───────────────────────────────────────────────────────────────────
def fetch_gwa_raster(iso3: str, layer: str, cache_dir: Path,
                     log=print) -> Path:
    """Download a GWA GeoTIFF for one country/layer (idempotent; skips if cached)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{iso3}_{layer}.tif"
    if dst.exists() and dst.stat().st_size > 0:
        log(f"  [cache] {dst.name}")
        return dst
    url = GWA_URL.format(iso3=iso3, layer=layer)
    log(f"  [fetch] {iso3}/{layer} -> {dst.name}")
    urllib.request.urlretrieve(url, dst)  # 302 -> CDN GeoTIFF (urllib follows it)
    return dst


# ── Zonal target ────────────────────────────────────────────────────────────
def _top_share_mean(values: np.ndarray, weights: np.ndarray,
                    percentile: float) -> float:
    """Area-weighted mean of the top `percentile` share of `values`."""
    order = np.argsort(values)[::-1]                 # high CF first
    v, w = values[order], weights[order]
    cutoff = percentile * w.sum()
    keep = np.cumsum(w) <= cutoff
    if not keep.any():                               # zone smaller than one share step
        keep[0] = True
    return float(np.average(v[keep], weights=w[keep]))


def zone_cf_targets(zones: gpd.GeoDataFrame, rasters: Dict[str, Path],
                    percentile: float = 0.30, iso_col: str = "ISO_A3",
                    name_col: str = "zone_name", zero_is_nodata: bool = True,
                    log=print) -> pd.DataFrame:
    """Compute a developable CF target per zone from country rasters.

    Each zone polygon is clipped out of its country's raster; the target is the
    cos-lat-weighted mean CF over the windiest `percentile` of pixels.
    Zones whose ISO3 has no raster are skipped (returned in `skipped`).
    """
    rows, skipped = [], []
    for _, z in zones.iterrows():
        iso = str(z[iso_col]).upper()
        zname = z[name_col]
        if iso not in rasters:
            skipped.append(zname)
            continue
        with rasterio.open(rasters[iso]) as src:
            geom = [z.geometry.__geo_interface__]
            arr, transform = rio_mask(src, geom, crop=True, filled=True,
                                      nodata=np.nan)
            cf = arr[0].astype("float64")
        valid = np.isfinite(cf)
        if zero_is_nodata:
            valid &= cf > 0.0                        # GWA fills water/void with 0
        if valid.sum() == 0:
            skipped.append(zname)
            continue
        rr, cc = np.where(valid)
        # pixel latitude for cos-lat area weight
        ys = transform.f + (rr + 0.5) * transform.e  # transform.e < 0
        w = np.cos(np.deg2rad(ys))
        v = cf[rr, cc]
        rows.append({
            "zone": zname, "iso3": iso, "percentile": percentile,
            "cf_target": round(_top_share_mean(v, w, percentile), 4),
            "n_pixels": int(valid.sum()),
            "cf_min": round(float(v.min()), 4),
            "cf_mean_all": round(float(np.average(v, weights=w)), 4),
            "cf_max": round(float(v.max()), 4),
        })
    if skipped:
        log(f"  [skip] no raster / no valid pixels: {sorted(set(skipped))}")
    return pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)


# ── Rescale profiles ────────────────────────────────────────────────────────
def _weighted_mean_map(hours: pd.DataFrame) -> pd.Series:
    """Return a (season,daytype,slice)->hours weight Series from pHours.csv."""
    qcol = "q" if "q" in hours.columns else "season"
    dcol = "d" if "d" in hours.columns else "daytype"
    tcols = [c for c in hours.columns if c.startswith("t") and c[1:].isdigit()]
    m = hours.melt(id_vars=[qcol, dcol], value_vars=tcols,
                   var_name="t", value_name="w")
    m[qcol] = m[qcol].astype(str).str.upper()
    return m.set_index([qcol, dcol, "t"])["w"]


def rescale_profiles(profile: pd.DataFrame, targets: pd.DataFrame,
                     hours: pd.DataFrame, techs: Iterable[str],
                     zone_map: Optional[Dict[str, str]] = None,
                     log=print) -> pd.DataFrame:
    """Rescale profile rows for `techs` so each zone's weighted-mean CF == target.

    Shape is preserved exactly; only the level changes. Rows for other techs and
    zones without a target are left untouched. `zone_map` maps the *target* zone
    names to *profile* zone names (e.g. {'ARM_1': 'Armenia'}).
    """
    profile = profile.copy()
    zone_map = zone_map or {}
    tcols = [c for c in profile.columns if c.startswith("t") and c[1:].isdigit()]
    scol = "season" if "season" in profile.columns else "q"
    dcol = "daytype" if "daytype" in profile.columns else "d"
    zcol = profile.columns[0]
    tech_col = "tech" if "tech" in profile.columns else "fuel"

    wmap = _weighted_mean_map(hours)
    tgt = {zone_map.get(r.zone, r.zone): r.cf_target for r in targets.itertuples()}

    changed = []
    for (zname, tech), idx in profile.groupby([zcol, tech_col]).groups.items():
        if tech not in techs or zname not in tgt:
            continue
        block = profile.loc[idx]
        # per-slice weight aligned to this block's (season,daytype,t)
        long = block.melt(id_vars=[scol, dcol], value_vars=tcols,
                          var_name="t", value_name="cf")
        long["_key"] = list(zip(long[scol].astype(str).str.upper(),
                                long[dcol], long["t"]))
        long["w"] = long["_key"].map(lambda k: wmap.get(k, np.nan))
        wmean = np.average(long["cf"], weights=long["w"])
        if wmean <= 0:
            continue
        factor = tgt[zname] / wmean
        profile.loc[idx, tcols] = (block[tcols] * factor).clip(upper=1.0).values
        changed.append((zname, tech, round(float(wmean), 4),
                        round(tgt[zname], 4), round(float(factor), 3)))
    for zname, tech, old, new, f in changed:
        log(f"  [rescale] {zname:16s} {tech:12s} {old:.3f} -> {new:.3f}  (x{f})")
    return profile


def current_weighted_cf(profile: pd.DataFrame, hours: pd.DataFrame,
                        techs: Iterable[str]) -> pd.DataFrame:
    """Hour-weighted annual mean CF per (zone,tech) of an existing profile."""
    tcols = [c for c in profile.columns if c.startswith("t") and c[1:].isdigit()]
    scol = "season" if "season" in profile.columns else "q"
    dcol = "daytype" if "daytype" in profile.columns else "d"
    zcol = profile.columns[0]
    tech_col = "tech" if "tech" in profile.columns else "fuel"
    wmap = _weighted_mean_map(hours)
    out = []
    for (zname, tech), idx in profile.groupby([zcol, tech_col]).groups.items():
        if tech not in techs:
            continue
        b = profile.loc[idx].melt(id_vars=[scol, dcol], value_vars=tcols,
                                  var_name="t", value_name="cf")
        b["w"] = list(zip(b[scol].astype(str).str.upper(), b[dcol], b["t"]))
        b["w"] = b["w"].map(lambda k: wmap.get(k, np.nan))
        out.append({"zone": zname, "tech": tech,
                    "cf_now": round(float(np.average(b["cf"], weights=b["w"])), 4)})
    return pd.DataFrame(out)


# ── CLI ─────────────────────────────────────────────────────────────────────
def _default_zone_map() -> Dict[str, str]:
    """GeoJSON zone_name -> model/profile zone name (blacksea 9z study)."""
    return {"ARM_1": "Armenia", "AZE_1": "AzerbaijanMain",
            "GEO_1": "Georgia", "BGR_1": "Bulgaria", "ROU_1": "Romania"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zones", required=True, help="zones GeoJSON")
    ap.add_argument("--countries", nargs="+", required=True,
                    help="ISO3 codes to fetch/use (e.g. TUR ARM GEO AZE)")
    ap.add_argument("--layer", default="capacity-factor_IEC2")
    ap.add_argument("--percentile", type=float, default=0.30)
    ap.add_argument("--cache-dir", default="data/gwa")
    ap.add_argument("--profile-in", help="pVREProfile.csv to rescale")
    ap.add_argument("--hours", help="pHours.csv (weights)")
    ap.add_argument("--techs", nargs="+", default=["OnshoreWind"])
    ap.add_argument("--targets-out", default="output_vre/cf_targets.csv")
    ap.add_argument("--profile-out", help="rescaled pVREProfile.csv (omit = targets only)")
    args = ap.parse_args(argv)

    print("Fetching rasters:")
    rasters = {iso: fetch_gwa_raster(iso, args.layer, args.cache_dir)
               for iso in args.countries}

    zones = gpd.read_file(args.zones)
    print("Computing zone CF targets:")
    targets = zone_cf_targets(zones, rasters, percentile=args.percentile)

    # attach current ninja CF for comparison, if a profile is given
    if args.profile_in and args.hours:
        prof = pd.read_csv(args.profile_in)
        hrs = pd.read_csv(args.hours)
        now = current_weighted_cf(prof, hrs, args.techs)
        zmap = _default_zone_map()
        now["zone_target"] = now["zone"].map(
            {v: k for k, v in zmap.items()}).fillna(now["zone"])
        cmp = targets.merge(now, left_on="zone", right_on="zone_target",
                            how="left", suffixes=("", "_p"))
        cmp["delta_pct"] = ((cmp["cf_target"] - cmp["cf_now"]) /
                            cmp["cf_now"] * 100).round(1)
        targets = cmp

    Path(args.targets_out).parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.targets_out, index=False)
    print(f"\nWrote {args.targets_out}")
    print(targets.to_string(index=False))

    if args.profile_out:
        prof = pd.read_csv(args.profile_in)
        hrs = pd.read_csv(args.hours)
        out = rescale_profiles(prof, targets, hrs, techs=args.techs,
                               zone_map=_default_zone_map())
        out.to_csv(args.profile_out, index=False)
        print(f"\nWrote {args.profile_out}")


if __name__ == "__main__":
    main()