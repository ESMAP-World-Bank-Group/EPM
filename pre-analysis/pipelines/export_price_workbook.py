# -*- coding: utf-8 -*-
"""The EU border price hypothesis as one workbook people can open and check.

The slides state the assumption; this is what a counterpart opens to disagree
with it. Everything comes from output_prices, so the workbook cannot drift from
what the model was fed.

Run:
    python pre-analysis/pipelines/export_price_workbook.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PRICES = HERE / "output_prices"
OUT = HERE.parents[1] / "EU_border_prices.xlsx"     # next to the deck

ZONES = ["Romania", "Bulgaria", "Greece"]
YEARS = range(2024, 2041)

README = [
    ("What this is",
     "The price EPM applies at the EU border. The EU side is not modelled: "
     "these prices are given, and the project does not move them."),
    ("Formula",
     "buy = Shape x Level + W          sell = Shape x Level x (1 - lambda) "
     "- W - CBAM"),
    ("Level", "Annual level, real EUR 2024/MWh. Observed 2024, then ENTSO-E "
              "TYNDP 2024 scenarios. Sheet 'Level'."),
    ("Shape", "Hourly profile, mean 1 over the year. ENTSO-E day-ahead "
              "observed. Sheet 'Shape'."),
    ("W", "2 EUR/MWh, ITC perimeter (wheeling) fee. Subtracted from the "
          "export, added to the import."),
    ("lambda", "3 % losses, borne by the seller. EPM applies no loss factor "
               "on external links, so it is carried in the price."),
    ("CBAM", "EF(exporter) x ETS price, from 2026, export into the EU only. "
             "Sheet 'CBAM'."),
    ("Reference scenario", "eu_central. Ten price files in total: five levels "
                           "x {with, without CBAM}."),
]


def main() -> None:
    level = pd.read_csv(PRICES / "level_L.csv")
    level = level[level["zone"].isin(ZONES) & level["year"].isin(YEARS)]
    level = (level.pivot_table(index=["zone", "scenario"], columns="year",
                               values="L_eur2024")
                  .round(1).reset_index())

    shape = pd.read_csv(PRICES / "shape_S.csv")
    shape = shape[shape["zone"].isin(ZONES)].round(4)

    cbam = pd.read_csv(PRICES / "cbam_levy.csv")
    cbam = (cbam[cbam["year"].isin(YEARS)]
            [["zone", "exporter", "year", "ef_tco2_per_mwh",
              "ets_eur2024_per_t", "C_eur2024_per_mwh"]].round(2))

    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        pd.DataFrame(README, columns=["Item", "Definition"]).to_excel(
            xl, sheet_name="Read me", index=False)
        level.to_excel(xl, sheet_name="Level", index=False)
        shape.to_excel(xl, sheet_name="Shape", index=False)
        cbam.to_excel(xl, sheet_name="CBAM", index=False)

        xl.sheets["Read me"].column_dimensions["A"].width = 20
        xl.sheets["Read me"].column_dimensions["B"].width = 110

    print(f"  wrote {OUT}  {OUT.stat().st_size/1024:.0f} kB")
    print(f"  Level {len(level)} rows | Shape {len(shape)} | CBAM {len(cbam)}")


if __name__ == "__main__":
    main()
