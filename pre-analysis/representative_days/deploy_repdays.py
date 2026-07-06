"""Deploy the 28-day (2023) representative-days outputs into data_blacksea.

Reformats pipeline outputs to the model schema, backs up replaced files (.bak),
propagates the 6->7 daytype change to pTradePrice, and maps VRE techs.
ROR is auto-filled by the model (EPM_FILL_ROR_FROM_AVAILABILITY=1) -> not written here.
"""
from pathlib import Path
import shutil
import pandas as pd

SRC = Path(__file__).resolve().parent / "output" / "blacksea"
DATA = Path(__file__).resolve().parents[2] / "epm" / "input" / "data_blacksea"

TMAP = {f"t{i}": f"t{i:02d}" for i in range(1, 25)}  # t1->t01 ... t9->t09, t10..t24 unchanged


def backup(p: Path):
    if p.exists():
        shutil.copy2(p, p.with_suffix(p.suffix + ".bak_repdays"))
        print(f"  backup: {p.name} -> {p.name}.bak_repdays")


def deploy_phours():
    df = pd.read_csv(SRC / "pHours.csv").rename(columns={"season": "q", "daytype": "d", **TMAP})
    dst = DATA / "pHours.csv"
    backup(dst)
    df.to_csv(dst, index=False)
    print(f"  pHours -> {len(df)} slices, daytypes {sorted(df.d.unique())}")


def deploy_demand_profile():
    df = pd.read_csv(SRC / "pDemandProfile.csv").rename(columns=TMAP)
    dst = DATA / "load" / "pDemandProfile.csv"
    backup(dst)
    df.to_csv(dst, index=False)
    print(f"  pDemandProfile -> {df.zone.nunique()} zones x {len(df)//max(1,df.zone.nunique())} (season,daytype)")


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
    print("Deploying representative-days (2023, 28 slices) to data_blacksea:")
    deploy_phours()
    deploy_demand_profile()
    deploy_vre_profile()
    extend_tradeprice_d7()
    print("Done.")
