"""Deploy the 28-day (2023) representative-days outputs into data_blacksea.

Reformats pipeline outputs to the model schema, backs up replaced files (.bak),
propagates the 6->7 daytype change to pTradePrice, and maps VRE techs.
ROR is auto-filled by the model (EPM_FILL_ROR_FROM_AVAILABILITY=1) -> not written here.
"""
import argparse
from datetime import date
from pathlib import Path
import shutil
import pandas as pd

SRC = Path(__file__).resolve().parent / "output" / "blacksea"
DATA = Path(__file__).resolve().parents[2] / "epm" / "input" / "data_blacksea"

TMAP = {f"t{i}": f"t{i:02d}" for i in range(1, 25)}  # t1->t01 ... t9->t09, t10..t24 unchanged


LF_GAP_TOLERANCE = 0.05   # profile against forecast load factor, in points
PROFILE_MAX_FLOOR = 0.95  # same floor as the EPM input verification


def backup(p: Path):
    if p.exists():
        bak = p.with_suffix(p.suffix + ".bak_repdays")
        if bak.exists():  # never overwrite an earlier backup
            bak = p.with_suffix(p.suffix + f".bak_repdays_{date.today():%Y%m%d}")
        shutil.copy2(p, bak)
        print(f"  backup: {p.name} -> {bak.name}")


def deploy_phours():
    df = pd.read_csv(SRC / "pHours.csv").rename(columns={"season": "q", "daytype": "d", **TMAP})
    dst = DATA / "pHours.csv"
    backup(dst)
    df.to_csv(dst, index=False)
    print(f"  pHours -> {len(df)} slices, daytypes {sorted(df.d.unique())}")


def annual_load_max() -> pd.Series:
    """Maximum of the full-year hourly load fed to the pipeline, per zone."""
    path = next(SRC.glob("Load_*_season.csv"))
    df = pd.read_csv(path)
    value_col = [c for c in df.columns if c not in ("zone", "season", "day", "hour")][0]
    return df.groupby("zone")[value_col].max()


def rescale_to_annual_peak(df: pd.DataFrame, tcols: list) -> pd.DataFrame:
    """Put each zone on its own annual peak.

    EPM models the peak as profile max x Peak forecast. The selected days may miss the peak hour
    (kept as is), but the reference must be the annual peak of the zone, not another year or zone.
    """
    ref = annual_load_max()
    for zone, peak in ref.items():
        if peak < 0.999:
            rows = df.zone == zone
            df.loc[rows, tcols] = df.loc[rows, tcols] / peak
            print(f"  {zone}: annual max of the input was {peak:.4f}, profile rescaled, "
                  f"max now {df.loc[rows, tcols].max().max():.4f}")
    return df


def profile_load_factor(values, weights) -> float:
    """Weighted mean of a profile expressed against the annual peak (the reference of the Peak forecast)."""
    return float((values * weights).sum() / weights.sum())


def check_load_factor(df: pd.DataFrame, tcols: list, adjust_zones=()) -> pd.DataFrame:
    """Compare the load factor of each profile with the one implied by pDemandForecast.

    EPM closes any energy gap by lifting or lowering the off-peak hours, so a large gap distorts
    the shape (and can drive the minimum load negative). Zones in `adjust_zones` get their shape
    raised to an exponent, peak unchanged, so that the profile meets the mean forecast load factor.
    """
    hours = pd.read_csv(SRC / "pHours.csv").rename(columns=TMAP).set_index(["season", "daytype"])[tcols]
    forecast = pd.read_csv(DATA / "load" / "pDemandForecast.csv")
    years = [c for c in forecast.columns if str(c).isdigit()]
    peak = forecast[forecast["type"] == "Peak"].set_index("z")[years]
    energy = forecast[forecast["type"] == "Energy"].set_index("z")[years]
    target = (energy * 1e3 / (peak * 8760)).mean(axis=1)

    print("  load factor, profile against forecast (mean over forecast years):")
    for zone in df.zone.unique():
        if zone not in target.index:
            continue
        rows = df.zone == zone
        keys = pd.MultiIndex.from_frame(df.loc[rows, ["season", "daytype"]])
        w = hours.loc[keys].to_numpy()
        p = df.loc[rows, tcols].to_numpy()
        lf = profile_load_factor(p, w)
        gap = lf - target[zone]
        flag = "" if abs(gap) <= LF_GAP_TOLERANCE else "  <- gap above tolerance"
        print(f"    {zone}: {lf:.3f} against {target[zone]:.3f}{flag}")
        if zone in adjust_zones and abs(gap) > LF_GAP_TOLERANCE:
            pmax, lo, hi = p.max(), 0.05, 20.0
            for _ in range(60):  # bisection: the load factor decreases when the exponent grows
                mid = (lo + hi) / 2
                if profile_load_factor(pmax * (p / pmax) ** mid, w) > target[zone]:
                    lo = mid
                else:
                    hi = mid
            p = pmax * (p / pmax) ** mid
            df.loc[rows, tcols] = p
            print(f"      shape adjusted with exponent {mid:.3f}: load factor {profile_load_factor(p, w):.3f}, "
                  f"min {p.min():.3f}, max {p.max():.3f}")
    return df


def deploy_demand_profile(adjust_zones=(), dest: Path = None):
    df = pd.read_csv(SRC / "pDemandProfile.csv").rename(columns=TMAP)
    tcols = [c for c in df.columns if c.startswith("t")]
    dst = Path(dest) if dest else DATA / "load" / "pDemandProfile.csv"
    current = DATA / "load" / "pDemandProfile.csv"

    df = rescale_to_annual_peak(df, tcols)
    df = check_load_factor(df, tcols, adjust_zones)

    # Zones built outside this pipeline (iran_swap) are carried over untouched
    if current.exists():
        old = pd.read_csv(current)
        kept = old[~old.zone.isin(df.zone)]
        if len(kept):
            df = pd.concat([df, kept[df.columns]], ignore_index=True)
            print(f"  carried over from the deployed file: {sorted(kept.zone.unique())}")

    low = df.groupby("zone")[tcols].max().max(axis=1)
    low = low[low < PROFILE_MAX_FLOOR]
    if len(low):
        print(f"  WARNING: profile max below {PROFILE_MAX_FLOOR} for {low.round(3).to_dict()}")

    if dst == current:
        backup(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"  pDemandProfile -> {df.zone.nunique()} zones x {len(df)//max(1,df.zone.nunique())} (season,daytype) -> {dst}")


def deploy_vre_profile():
    df = pd.read_csv(SRC / "pVREProfile.csv").rename(columns={"fuel": "tech", **TMAP})
    # map tech values to model conventions
    df["tech"] = df["tech"].replace({"Wind": "OnshoreWind"})
    # OffshoreWind proxy = OnshoreWind (no offshore ninja data)
    off = df[df.tech == "OnshoreWind"].copy()
    off["tech"] = "OffshoreWind"
    df = pd.concat([df, off], ignore_index=True)
    dst = DATA / "supply" / "pVREProfile.csv"
    backup(dst)
    df.to_csv(dst, index=False)
    print(f"  pVREProfile -> techs {sorted(df.tech.unique())} (ROR auto-filled by model)")


def extend_tradeprice_d7():
    dst = DATA / "trade" / "pTradePrice.csv"
    df = pd.read_csv(dst)
    if "d7" in df["d"].unique():
        print("  pTradePrice: d7 already present, skip")
        return
    backup(dst)
    d6 = df[df["d"] == "d6"].copy()
    d6["d"] = "d7"
    out = pd.concat([df, d6], ignore_index=True)
    out.to_csv(dst, index=False)
    print(f"  pTradePrice -> added d7 (copy of d6), daytypes {sorted(out.d.unique())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demand-only", action="store_true", help="deploy pDemandProfile only")
    ap.add_argument("--adjust-shape", nargs="*", default=[], metavar="ZONE",
                    help="zones whose shape is adjusted to the forecast load factor (peak unchanged)")
    ap.add_argument("--dest", default=None,
                    help="write pDemandProfile to this path instead of data_blacksea (review before deploying)")
    args = ap.parse_args()

    print("Deploying representative-days (2023, 28 slices) to data_blacksea:")
    if args.demand_only or args.dest:
        deploy_demand_profile(args.adjust_shape, args.dest)
    else:
        deploy_phours()
        deploy_demand_profile(args.adjust_shape)
        deploy_vre_profile()
        extend_tradeprice_d7()
    print("Done.")
