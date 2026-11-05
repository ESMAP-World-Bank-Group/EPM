"""Validate the Greater Malé VRE inputs against the ESMAP measurement campaigns (files in raw/esmap/).

Wind:  ZephIR lidar at Thulusdhoo (4.37453 N, 73.65217 E), 10-min, local time (GMT+5), Apr 2017 - Apr 2019,
       100 m mean speed (9999 = missing). Compared with ERA5 (Open-Meteo; Thulusdhoo cell and the Malé cell),
       Renewables.ninja MERRA-2 and Global Wind Atlas, and converted to CF with the two power curves.
Solar: Tier 2 station at Velana airport, Hulhulé (4.19274 N, 73.52811 E), 1-min, UTC, Dec 2015 - Mar 2018.
       RSI GHI/DNI/DHI and pyranometer GHI with Solargis QC flags (0 night, 1 passed). Compared with the
       Solargis hourly satellite series at the station (ESMAP SatelliteTS, 1999 - Apr 2018) and PVGIS-ERA5,
       raw and with the beam/diffuse calibration used in build_profiles.py.
       Tracker gains are computed from each dataset's hourly beam and diffuse with the same simple isotropic
       sky model (sun position from the Solargis file), so the datasets are compared like for like. The
       isotropic model ignores circumsolar light, so its gains are a little below PVGIS's.

Writes out/validation/*.csv, out/validation/calibration.json (read by build_profiles.py) and prints the
results used in VRE_PROFILES.md. Run it before build_profiles.py.

Run:  python validate_esmap.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from build_profiles import power_curve, pv_scale

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
ESMAP = RAW / "esmap"
OUT = HERE / "out" / "validation"
LOCAL = pd.Timedelta(hours=5)
pd.set_option("display.width", 220)


# ----------------------------------------------------------------------------------------------- wind
def lidar_hourly():
    df = pd.read_csv(ESMAP / "wind-measurements_maldives_thulusdhoo_wb-esmap-qc.csv",
                     usecols=["time", "a50_wind_speed_mean", "a100_wind_speed_mean"], na_values=[9999, "9999"])
    df.index = pd.to_datetime(df.pop("time"), format="%d/%m/%Y %H:%M") - LOCAL          # to UTC
    df = df.where((df >= 0) & (df < 60))
    # hour h = 10-min records within h-30 min .. h+30 min, to match ERA5's on-the-hour values; >= 4 of 6 records
    g = df.resample("h", offset="-30min")
    hr = g.mean().where(g.count() >= 4)
    hr.index = hr.index + pd.Timedelta(minutes=30)
    return hr.rename(columns={"a50_wind_speed_mean": "lidar50", "a100_wind_speed_mean": "lidar100"})


def era5(path):
    d = json.loads(path.read_text())["hourly"]
    return pd.Series(d["wind_speed_100m"], index=pd.to_datetime(d["time"]), dtype=float)


def ninja_wind(years):
    parts = []
    for y in years:
        d = json.loads((RAW / "ninja" / f"wind_Thulusdhoo_onshore_{y}.json").read_text())["data"]
        parts.append(pd.DataFrame(list(d.values()), index=pd.to_datetime([int(t) for t in d], unit="ms")))
    return pd.concat(parts)


def wind():
    atl = json.loads((RAW / "atlas_validation.json").read_text())["Thulusdhoo_lidar"]
    curves = {k: power_curve(k)[:2] for k in ("onshore", "offshore")}
    cf = {k: (lambda v, k=k: np.interp(v, *curves[k])) for k in curves}
    e_thu = era5(RAW / "era5" / "validation_era5_Thulusdhoo_lidar_2017_2019.json")
    e_male = era5(next((RAW / "era5").glob("era5_Male_*.json")))
    e_all = era5(next((RAW / "era5").glob("era5_Male_*.json")))
    k_gwa_thu = atl["gwa_ws100"] / e_thu["2017":"2019"].mean()    # our method's scaling, applied at Thulusdhoo
    nj = ninja_wind((2017, 2018, 2019))
    df = lidar_hourly().join(pd.DataFrame({"era5_thu": e_thu, "era5_male": e_male, "ninja": nj["wind_speed"],
                                           "ninja_cf": nj["electricity"]}), how="inner").dropna()
    df["ours_thu"] = df["era5_thu"] * k_gwa_thu
    r = df["lidar100"].mean() / df["era5_thu"].mean()
    df["era5_thu_x_r"] = df["era5_thu"] * r
    n_days = df.index.normalize().nunique()
    print(f"WIND  concurrent hours {len(df)} ({df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}, {n_days} days)")
    rows = []
    for col, label in [("lidar100", "Lidar 100 m (measured)"), ("era5_thu", "ERA5, Thulusdhoo cell"),
                       ("era5_male", "ERA5, Male cell"), ("ours_thu", "First build at Thulusdhoo (ERA5 x GWA)"),
                       ("era5_thu_x_r", "ERA5 x (lidar / ERA5 mean), new method"),
                       ("ninja", "Renewables.ninja MERRA-2")]:
        s = df[col]
        rows.append({"series": label, "mean_ms": s.mean(), "ratio_to_lidar": s.mean() / df["lidar100"].mean(),
                     "std_over_mean": s.std() / s.mean(), "p10": s.quantile(.1), "p50": s.median(), "p90": s.quantile(.9),
                     "corr_hourly": s.corr(df["lidar100"]),
                     "corr_daily": s.resample("D").mean().corr(df["lidar100"].resample("D").mean()),
                     "cf_onshore": cf["onshore"](s).mean(), "cf_offshore": cf["offshore"](s).mean()})
    t = pd.DataFrame(rows)
    t.loc[len(t)] = {"series": "Renewables.ninja own CF (G114)", "cf_onshore": df["ninja_cf"].mean()}
    print(t.round(3).to_string(index=False))
    print(f"GWA 100 m at lidar {atl['gwa_ws100']} m/s; ESMAP 24-month report: measured 5.9 m/s at 100 m, long-term 5.9 m/s")
    print(f"lidar shear 50->100 m: alpha = {np.log(df.lidar100.mean() / df.lidar50.mean()) / np.log(2):.3f}")
    rm = df["era5_male"].mean() / df["era5_thu"].mean()
    print(f"lidar / ERA5(Thulusdhoo cell) = {r:.3f};  ERA5 Malé cell / Thulusdhoo cell (same hours) = {rm:.3f}")
    print(f"ERA5 Malé cell long-term 2014-2023 = {e_all['2014':'2023'].mean():.2f} m/s -> x {r:.3f} = {e_all['2014':'2023'].mean() * r:.2f} m/s")
    loc = df.copy()
    loc.index = loc.index + LOCAL
    mon = loc.groupby(loc.index.month)[["lidar100", "era5_thu", "ours_thu", "ninja"]].mean()
    diu = loc.groupby(loc.index.hour)[["lidar100", "era5_thu"]].mean()
    print("monthly mean speed (m/s):\n", mon.round(2).T.to_string())
    print("diurnal mean speed, local hour (m/s):\n", diu.round(2).T.to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(OUT / "wind_thulusdhoo_summary.csv", index=False)
    mon.round(3).to_csv(OUT / "wind_thulusdhoo_monthly.csv")
    diu.round(3).to_csv(OUT / "wind_thulusdhoo_diurnal.csv")
    return {"lidar_over_era5": r, "era5_male_over_thu_cell": rm, "gwa100_lidar": atl["gwa_ws100"],
            "lidar100_mean": df["lidar100"].mean(), "hours": len(df),
            "period": f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"}


# ---------------------------------------------------------------------------------------------- solar
def station_hourly():
    """1-min RSI/pyranometer data -> hourly means (W/m2 = Wh/m2) on [h:00, h+1:00) UTC, labelled h:30."""
    df = pd.read_csv(ESMAP / "solar-measurements_maldives_male_wb-esmap_qc.csv",
                     usecols=["time", "ghi_rsi", "dni_rsi", "dhi_rsi", "ghi_pyr", "comments"])
    t = pd.to_datetime(df.pop("time"))
    flags = df.pop("comments").astype(str)
    for c in ("ghi_rsi", "dni_rsi", "dhi_rsi", "ghi_pyr"):
        f = flags.str.extract(re.escape(c) + r"_flag:(\d)", expand=False).astype(float)
        df[c] = df[c].where(f.isin([0, 1]))
    df.index = (t - pd.Timedelta(minutes=1)).dt.floor("h")      # stamp hh:mm closes the minute (hh:mm-1, hh:mm]
    g = df.groupby(level=0)
    hr = g.mean().where(g.count() >= 54)
    hr.index = hr.index + pd.Timedelta(minutes=30)
    return hr


def solargis():
    f = ESMAP / "satellite" / "Solar-Measurements_Maldives_Male_WB-ESMAP_SatelliteTS.csv"
    df = pd.read_csv(f, sep=";", comment="#")
    df.index = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d.%m.%Y %H:%M")
    df = df[["GHI", "DNI", "DIF", "SE", "SA"]].astype(float)
    df.loc[~df[["GHI", "DNI", "DIF"]].ge(0).all(axis=1), ["GHI", "DNI", "DIF"]] = np.nan   # -9 = no data
    return df


def pvgis_station():
    parts = []
    for f in sorted((RAW / "pvgis" / "validation").glob("pvgis_MLE_solar_ghi_*_components.json")):
        h = json.loads(f.read_text())["outputs"]["hourly"]
        idx = pd.to_datetime([x["time"] for x in h], format="%Y%m%d:%H%M")
        parts.append(pd.DataFrame({"Gb": [x["Gb(i)"] for x in h], "Gd": [x["Gd(i)"] for x in h]}, index=idx))
    return pd.concat(parts)


def poa(ghi, dni, dhi, se, sa, rho=0.2):
    """Plane-of-array irradiance, isotropic sky: fixed 3 deg south, horizontal N-S single-axis, two-axis."""
    zen, az = np.radians(90 - se), np.radians(180 + sa)          # Solargis SA: from south, east negative
    sx, sy, sz = np.sin(zen) * np.sin(az), np.sin(zen) * np.cos(az), np.cos(zen)   # x east, y north, z up
    up = sz > 0
    b = np.radians(3.0)
    cos_fix = np.clip(-sy * np.sin(b) + sz * np.cos(b), 0, None)
    rot = np.arctan2(sx, sz)                                      # ideal N-S tracker rotation, no backtracking
    cos_1ax = np.sqrt(sx ** 2 + sz ** 2)
    out = {}
    for name, cosaoi, tilt in (("fixed", cos_fix, np.full_like(sz, b)), ("1ax", cos_1ax, np.abs(rot)), ("2ax", np.ones_like(sz), zen)):
        g = dni * cosaoi + dhi * (1 + np.cos(tilt)) / 2 + ghi * rho * (1 - np.cos(tilt)) / 2
        out[name] = np.where(up, g, 0.0)
    return out


def solar():
    st = station_hourly()
    sg = solargis()
    pv = pvgis_station()
    atl = json.loads((RAW / "atlas_validation.json").read_text())["MLE_solar"]["gsa"]
    _, fb, fd = pv_scale("Hulhumale", "fixed", atl)               # same beam/diffuse factors as build_profiles.py
    df = st.join(sg, how="inner").join(pv, how="inner").dropna()
    df = df[df["SE"] > -1]                                        # daytime hours (incl. sunrise/sunset hours)
    sinh = np.sin(np.radians(df["SE"].clip(lower=3)))
    df["era5_ghi"] = df["Gb"] + df["Gd"]
    df["era5_dni"] = np.where(df["SE"] > 3, df["Gb"] / sinh, 0.0)
    print(f"\nSOLAR  common daytime hours {len(df)} ({df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}); "
          f"station flagged hours dropped")
    sets = {
        "Measured (RSI)": (df["ghi_rsi"], df["dni_rsi"], df["dhi_rsi"]),
        "Solargis satellite": (df["GHI"], df["DNI"], df["DIF"]),
        "ERA5 (PVGIS), raw": (df["era5_ghi"], df["era5_dni"], df["Gd"]),
        "ERA5 (PVGIS), calibrated as in build": (df["Gb"] * fb + df["Gd"] * fd, df["era5_dni"] * fb, df["Gd"] * fd),
    }
    rows = []
    meas_ghi = df["ghi_rsi"].sum()
    for name, (g, dn, dh) in sets.items():
        p = poa(g.to_numpy(), dn.to_numpy(), dh.to_numpy(), df["SE"].to_numpy(), df["SA"].to_numpy())
        daily = lambda s: pd.Series(np.asarray(s), index=df.index).resample("D").sum()
        rows.append({"dataset": name, "ghi_ratio_to_measured": g.sum() / meas_ghi, "diffuse_share": dh.sum() / g.sum(),
                     "corr_hourly_ghi": np.corrcoef(g, df["ghi_rsi"])[0, 1],
                     "corr_daily_ghi": daily(g).corr(daily(df["ghi_rsi"])),
                     "gain_1ax": p["1ax"].sum() / p["fixed"].sum() - 1, "gain_2ax": p["2ax"].sum() / p["fixed"].sum() - 1})
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    m, c = t.iloc[0], t.iloc[3]
    calib = {"tracker_factor_1ax": (1 + m.gain_1ax) / (1 + c.gain_1ax), "tracker_factor_2ax": (1 + m.gain_2ax) / (1 + c.gain_2ax),
             "gain_1ax_measured": m.gain_1ax, "gain_1ax_era5_calibrated": c.gain_1ax,
             "gain_2ax_measured": m.gain_2ax, "gain_2ax_era5_calibrated": c.gain_2ax, "hours": len(df)}
    print(f"pyranometer GHI / RSI GHI = {df['ghi_pyr'].sum() / meas_ghi:.3f};  GSA long-term GHI {atl['GHI']:.0f}, DIF share {atl['DIF'] / atl['GHI']:.3f};  "
          f"calibration factors beam x{fb:.3f}, diffuse x{fd:.3f}")
    years = len(df) / (8760 / 2)                                   # rough: daytime hours per year ~ 4380
    print(f"measured GHI over common hours ~ {meas_ghi / 1000 / years:.0f} kWh/m2/yr (rough, coverage-weighted)")
    loc = df.copy()
    loc.index = loc.index + LOCAL
    diu = loc.groupby(loc.index.hour)[["ghi_rsi", "GHI", "era5_ghi"]].mean()
    mon = loc.groupby(loc.index.month)[["ghi_rsi", "GHI", "era5_ghi"]].mean()
    print("diurnal mean GHI, local hour, W/m2 (measured, Solargis, ERA5):\n", diu.round(0).T.to_string())
    print("monthly mean daytime GHI, W/m2:\n", mon.round(0).T.to_string())
    t.round(4).to_csv(OUT / "solar_male_summary.csv", index=False)
    diu.round(1).to_csv(OUT / "solar_male_diurnal.csv")
    mon.round(1).to_csv(OUT / "solar_male_monthly.csv")
    # Solargis long-term (1999-2017 full years) at the station, for the level
    sgl = sg["1999":"2017"]
    yr = sgl.groupby(sgl.index.year)[["GHI", "DIF"]].sum() / 1000
    print(f"Solargis 1999-2017 at station: GHI {yr.GHI.mean():.0f} kWh/m2/yr (range {yr.GHI.min():.0f}-{yr.GHI.max():.0f}), "
          f"diffuse share {yr.DIF.sum() / yr.GHI.sum():.3f}")
    return calib


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cal = {"wind": wind(), "solar": solar()}
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=1, default=float))
    print("calibration.json:", json.dumps(cal, default=lambda x: round(float(x), 4)))
