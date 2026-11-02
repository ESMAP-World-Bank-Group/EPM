# -*- coding: utf-8 -*-
"""RoBg reference v5 (2026-11-02): the v4 input set with two changes. Every output is a NEW file;
nothing read by an existing scenario changes. The scenario columns are written by
tools/scenarios_build.py from internal_analysis/scenarios/scenario_grid.yaml (reference block).

External trade, back to v3
    The reference reads trade/pMaxAnnualExternalTradeShare_robg.csv again (3 for every
    country, caps nothing) instead of the _robg_cap20 file of v4. The border capacities are
    unchanged: v3 and v4 both read trade/pExtTransferLimit_robg.csv. The cap20 file stays on
    disk for the archive columns and a possible sensitivity. This is a grid yaml change only.

Bulgarian batteries, supply/pStorageDataInput_v5_{s5,2y}.csv
    = supply/pStorageDataInput_{s5,2y}.csv with Bulgaria_Battery_2026 at 3,118 MW instead of
    4,024, CapacityMWh unchanged (10,462.4). With Bulgaria_Battery (1,382 MW, 3,593.2 MWh) the
    fleet is 4,500 MW and 14,056 MWh, the 4.5 GW and 14 GWh connected by June 2026 given in the
    Bulgaria inception review. The 4,024 MW came from the ENTSO-E A68 installed capacity at the
    start of 2026 (5,406 MW) minus the start of 2025 (1,382 MW), above the figures observed in
    mid 2026. The generator files keep 4,024 on the same row: merge_storage_into_gendata
    (epm/input_treatment.py) copies the storage file values over the shared columns, so the
    storage file sets the capacity.

Usage: python build_robg_v5.py [path to data_blacksea]. Idempotent.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_robg_v4 as v4  # noqa: E402  (edit_rows, same ROOT convention and writers)

ROOT = v4.ROOT

# grid token: (source, v5 file)
STO_FILES = {
    "s5": ("supply/pStorageDataInput_s5.csv", "supply/pStorageDataInput_v5_s5.csv"),
    "2y": ("supply/pStorageDataInput_2y.csv", "supply/pStorageDataInput_v5_2y.csv"),
}
# storage unit: {column: (expected value in the source, new value)}
STO_EDITS = {
    "Bulgaria_Battery_2026": {"Capacity": ("4024", "3118")},
}


def build_storage():
    for src, dst in STO_FILES.values():
        v4.edit_rows(src, dst, STO_EDITS, "storage units")


if __name__ == "__main__":
    print("data folder: " + ROOT)
    build_storage()
