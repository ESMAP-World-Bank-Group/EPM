"""Build hourly VRE capacity-factor profiles for the five Greater Malé zones (request of 3 Nov 2026).

Five techs per zone (EPM tech names still to agree, VRE_LOG Q4):
  PV            fixed, optimal tilt        PVGIS-ERA5 trackingtype=0, optimal angles
  PV1AX         1-axis tracking            PVGIS-ERA5 trackingtype=1, horizontal N-S axis
  PV2AX         2-axis tracking            PVGIS-ERA5 trackingtype=2
  OnshoreWind   Gamesa G114 2.0 MW, 100 m  ERA5 100 m wind, level from the ESMAP Thulusdhoo lidar, ninja power curve
  OffshoreWind  Vestas V136 3.45 MW, 100 m same wind as onshore at the zone point (no measured or modelled offshore gain)

Solar level: Global Solar Atlas (satellite) is the reference for the long-term level; ERA5 (reanalysis)
gives the hour-by-hour shape. ERA5 has too much beam and too little diffuse light here (diffuse share 29%
vs 41% in GSA), which inflates tracker gains. So beam and diffuse are scaled separately:
  fb = GSA (GHI - DIF) / ERA5 horizontal beam,  fd = GSA DIF / ERA5 horizontal diffuse  (ERA5 2015-2023 means)
and each system's output is scaled by (fb*Gb(i) + fd*Gd(i) + fg*Gr(i)) / G(i) from its 2023 in-plane split
(fg = GHI ratio for ground-reflected light). Output is assumed proportional to in-plane irradiance.
The tracker profiles are then multiplied by the measured/modelled gain ratio from the ESMAP Malé station
(validate_esmap.py: tracker gain from measured beam/diffuse vs from calibrated ERA5, same sky model).
Wind level: ERA5 100 m speed x k, with k = (lidar / ERA5 at Thulusdhoo, concurrent 2017-2019) x
(GWA at zone / GWA at lidar) / (ERA5 Male cell / ERA5 Thulusdhoo cell). The first factor transfers the
measured level; the other two move it from the lidar to each zone (GWA for the 250 m pattern, net of what
ERA5 already has between its cells). Global Wind Atlas alone is 17% below the lidar, so it is not used for
the level. Then the turbine power curve as Renewables.ninja applies it (binned from ninja's own
speed/output pairs, raw/ninja/*_2023.json).

All times are Maldives local time (UTC+5). Inputs come from fetch_pvgis.py, fetch_atlas.py,
fetch_era5_wind.py, fetch_ninja.py, and out/validation/calibration.json from validate_esmap.py. Outputs go to out/.

The weather year is chosen in main() as the year closest to the 2015-2023 average.

Run:  python build_profiles.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sites import UTC_OFFSET, ZONES

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "out"
YEARS = range(2015, 2024)           # complete local years for all sources (2014 lacks its first 5 local hours)
PV_TECHS = {"PV": "fixed", "PV1AX": "1ax", "PV2AX": "2ax"}
WIND_TECHS = {"OnshoreWind": "onshore", "OffshoreWind": "offshore"}
TECHS = list(PV_TECHS) + list(WIND_TECHS)
SHIFT = pd.Timedelta(hours=UTC_OFFSET)


def pvgis(zone, system, field="P"):
    """Hourly PVGIS series for one zone and system, local time."""
    parts = []
    for f in sorted((RAW / "pvgis").glob(f"pvgis_{zone}_{system}_*.json")):
        h = json.loads(f.read_text())["outputs"]["hourly"]
        idx = pd.to_datetime([x["time"][:11] for x in h], format="%Y%m%d:%H")
        parts.append(pd.Series([x[field] for x in h], index=idx, dtype=float))
    s = pd.concat(parts).sort_index()
    s.index = s.index + SHIFT
    return s


def era5_wind(name):
    d = json.loads(next((RAW / "era5").glob(f"era5_{name}_*.json")).read_text())["hourly"]
    s = pd.Series(d["wind_speed_100m"], index=pd.to_datetime(d["time"]), dtype=float)
    s.index = s.index + SHIFT
    return s.interpolate(limit=3)


def pv_scale(zone, system, gsa):
    """Beam/diffuse calibration factor to Global Solar Atlas for one zone and PV system (see module notes)."""
    comp = RAW / "pvgis" / "components"
    hor = [json.loads((comp / f"pvgis_{zone}_ghi_{y}_components.json").read_text()) for y in YEARS]
    gb, gd = np.mean([h["Gb(i)"] for h in hor]), np.mean([h["Gd(i)"] for h in hor])
    fb, fd, fg = (gsa["GHI"] - gsa["DIF"]) / gb, gsa["DIF"] / gd, gsa["GHI"] / (gb + gd)
    c = json.loads((comp / f"pvgis_{zone}_{system}_{max(YEARS)}_components.json").read_text())
    tot = c["Gb(i)"] + c["Gd(i)"] + c["Gr(i)"]
    return (fb * c["Gb(i)"] + fd * c["Gd(i)"] + fg * c["Gr(i)"]) / tot, fb, fd


def ninja(path):
    d = json.loads(path.read_text())["data"]
    idx = pd.to_datetime([int(t) for t in d], unit="ms") + SHIFT
    return pd.DataFrame(list(d.values()), index=idx)


def power_curve(tech):
    """Ninja's power curve for this turbine: median output per 0.25 m/s bin of ninja's hub-height speed."""
    df = pd.concat([ninja(p) for p in (RAW / "ninja").glob(f"wind_*_{tech}_2023.json")])
    edges = np.arange(0, 30.25, 0.25)
    mid = (edges[:-1] + edges[1:]) / 2
    med = df.groupby(pd.cut(df["wind_speed"], edges), observed=False)["electricity"].median().to_numpy()
    ok = ~np.isnan(med)
    curve = np.interp(mid, mid[ok], med[ok], left=0.0, right=med[ok][-1])  # hold the last value above the observed range
    return mid, np.maximum.accumulate(np.clip(curve, 0, 1)), float(df["wind_speed"].max())


def local_years(s, years):
    return s[(s.index.year >= min(years)) & (s.index.year <= max(years))]


def main():
    OUT.mkdir(exist_ok=True)
    atlas = json.loads((RAW / "atlas_points.json").read_text())
    cal = json.loads((OUT / "validation" / "calibration.json").read_text())
    trk = {"fixed": 1.0, "1ax": cal["solar"]["tracker_factor_1ax"], "2ax": cal["solar"]["tracker_factor_2ax"]}
    w = cal["wind"]
    curves = {t: power_curve(t) for t in WIND_TECHS.values()}
    series, notes = {}, []

    for zone in ZONES:
        gsa = atlas[zone]["gsa"]
        for tech, system in PV_TECHS.items():
            f_pv, fb, fd = pv_scale(zone, system, gsa)
            f_pv *= trk[system]
            raw = pvgis(zone, system) / 1000
            series[(zone, tech)] = (raw * f_pv).clip(upper=1.0)
            notes.append({"zone": zone, "tech": tech, "raw_cf": local_years(raw, YEARS).mean(), "scale": f_pv,
                          "ref": f"beam x{fb:.3f}, diffuse x{fd:.3f} (GSA GHI {gsa['GHI']:.0f}, DIF {gsa['DIF']:.0f}), tracker x{trk[system]:.3f}; "
                                 f"GSA PVOUT fixed {gsa['PVOUT_csi']:.0f} kWh/kWp"})

        v = era5_wind(zone)
        k = w["lidar_over_era5"] * (atlas[zone]["gwa_ws100"] / w["gwa100_lidar"]) / w["era5_male_over_thu_cell"]
        for tech, kind in WIND_TECHS.items():
            mid, pc, vmax = curves[kind]
            series[(zone, tech)] = pd.Series(np.interp(v * k, mid, pc), index=v.index)
            notes.append({"zone": zone, "tech": tech, "raw_cf": np.interp(local_years(v, YEARS), mid, pc).mean(), "scale": k,
                          "ref": f"long-term 100 m mean {local_years(v, range(2014, 2024)).mean() * k:.2f} m/s "
                                 f"(ERA5 {local_years(v, range(2014, 2024)).mean():.2f} x {k:.3f}); GWA {atlas[zone]['gwa_ws100']:.2f}"})

    cols = [f"{z}_{t}" for z in ZONES for t in TECHS]
    wide = pd.DataFrame({f"{z}_{t}": s for (z, t), s in series.items()})[cols]
    wide.index.name = "local_time"

    # Weather year: the complete year whose annual CFs (all zones and techs) are closest to the 2015-2023
    # mean, in relative terms. 2023, the latest PVGIS year, has the lowest wind of the nine years.
    allyrs = local_years(wide, YEARS)
    annual = allyrs.groupby(allyrs.index.year).mean()
    dev = (annual / annual.mean() - 1).abs().mean(axis=1)
    WEATHER_YEAR = int(dev.idxmin())
    print("mean relative deviation of annual CF from the 2015-2023 mean, by year:", dev.round(3).to_dict())
    print("weather year:", WEATHER_YEAR)

    # 1) weather-year table, 2) all complete years, 3) wind for the load years 2024-2025
    wy = local_years(wide, [WEATHER_YEAR])
    wy = wy[~((wy.index.month == 2) & (wy.index.day == 29))]  # 8760 h, as the representative-days pipeline expects
    assert len(wy) == 8760 and not wy.isna().any().any(), "weather year incomplete"
    wy.round(4).to_csv(OUT / f"vre_hourly_{WEATHER_YEAR}.csv")
    assert not allyrs.isna().any().any(), "gaps in the multi-year table"
    allyrs.round(4).to_csv(OUT / f"vre_hourly_{min(YEARS)}_{max(YEARS)}.csv")
    wind_cols = [c for c in cols if c.endswith("Wind")]
    local_years(wide[wind_cols], [2024, 2025]).round(4).to_csv(OUT / "wind_hourly_2024_2025.csv")

    # 4) one file per tech in the representative-days pipeline format (zone, month, day, hour 0-23, value)
    rd = OUT / "repdays_input"
    rd.mkdir(exist_ok=True)
    for tech in TECHS:
        rows = []
        for zone in ZONES:
            s = wy[f"{zone}_{tech}"]
            rows.append(pd.DataFrame({"zone": zone, "month": s.index.month, "day": s.index.day,
                                      "hour": s.index.hour, "value": s.to_numpy().round(4)}))
        pd.concat(rows).to_csv(rd / f"{tech}_{WEATHER_YEAR}.csv", index=False)

    # 5) summary: annual CF by zone, tech and year
    annual = annual.T
    summ = pd.DataFrame({"zone": [c.rsplit("_", 1)[0] for c in annual.index], "tech": [c.rsplit("_", 1)[1] for c in annual.index]})
    summ[f"cf_{WEATHER_YEAR}"] = annual[WEATHER_YEAR].to_numpy()
    summ[f"cf_mean_{min(YEARS)}_{max(YEARS)}"] = annual.mean(axis=1).to_numpy()
    summ["cf_min_year"] = annual.min(axis=1).to_numpy()
    summ["cf_max_year"] = annual.max(axis=1).to_numpy()
    summ["full_load_hours"] = summ[f"cf_{WEATHER_YEAR}"] * 8760
    n = pd.DataFrame(notes)
    summ = summ.merge(n, on=["zone", "tech"])
    summ.round(4).to_csv(OUT / "summary_cf.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    for kind, (mid, pc, vmax) in curves.items():
        print(f"{kind}: power curve from ninja, observed ninja speeds up to {vmax:.1f} m/s; "
              f"CF at 4, 5, 6, 8 m/s = {[round(float(np.interp(x, mid, pc)), 3) for x in (4, 5, 6, 8)]}")
    pd.DataFrame({"wind_speed": mid, **{k: v[1] for k, v in curves.items()}}).round(4).to_csv(OUT / "power_curves_ninja.csv", index=False)


if __name__ == "__main__":
    main()
