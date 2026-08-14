"""
build_bulgaria_fuelprice.py — generate Bulgaria pFuelPrice rows.

Sources:
  - Gas, ImportedCoal, Lignite: Kinesys WEM ELCFuel sheet (BG-CCDR)
    Units assumed $/GJ -> converted x1.055 to $/MMBtu
  - Uranium: Romania trajectory (same study, nuclear fuel cost reference)
  - Biomass: flat 5.0 $/MMBtu (Romania flat value)

Run:
  python pre-analysis/build_bulgaria_fuelprice.py           # preview
  python pre-analysis/build_bulgaria_fuelprice.py --write   # append to pFuelPrice.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

YEARS = list(range(2024, 2054))
F = 1.055  # $/GJ -> $/MMBtu

# ── Kinesys WEM milestones ($/GJ) → convert to $/MMBtu ──────────────────────
# Source: BG-CCDR Kinesys_2101_b2.20.26.xlsx, ELCFuel sheet, scenario cl_wb7-WEM.Nuc-Y.Clim-HDNucRet-Y

GAS_KY = {2024: 5.33, 2025: 5.33, 2030: 5.98, 2035: 5.40, 2040: 6.22, 2050: 8.10, 2053: 8.10}
COA_KY = {2025: 3.88, 2030: 3.73, 2035: 3.77, 2040: 3.78, 2045: 3.69, 2050: 3.65, 2053: 3.65}
LIG_KY = {2024: 2.14, 2025: 2.14, 2030: 2.20, 2053: 2.20}  # flat post-2030 (phase-out by 2033)


def interp_mmbtu(milestones):
    """Linear interpolation over YEARS, convert $/GJ -> $/MMBtu."""
    xs = sorted(milestones.keys())
    ys = [milestones[x] * F for x in xs]
    return [round(float(np.interp(y, xs, ys)), 4) for y in YEARS]


gas_vals = interp_mmbtu(GAS_KY)
coal_vals = interp_mmbtu(COA_KY)
lig_vals = interp_mmbtu(LIG_KY)

# ── Uranium: Romania trajectory (exact copy) ────────────────────────────────
ROMANIA_URANIUM = [
    1.504897274234244, 3.4033163183040847, 3.522029844084652,
    3.6213969695872814, 3.6213969695872814, 3.6213969695872814,
    3.1980942110374464, 3.9493818665606564, 4.175692192680371,
    4.238424426005719, 4.287974863746066, 4.3540446126956835,
    4.414066262820994, 4.527400338660385, 4.610486900920023,
    4.693573463179689, 4.776660025439355, 4.859746587698993,
    4.942833149958659, 5.025919712218325, 5.109006274477963,
    5.192092836737629, 5.275179398997295, 5.358265961256933,
    5.441352523516599, 5.524439085776265, 5.607525648035903,
    5.607525648035903, 5.607525648035903, 5.607525648035903,
]  # 2024-2053, $/MMBtu

# ── Biomass: flat 5.0 $/MMBtu ────────────────────────────────────────────────
bio_vals = [5.0] * len(YEARS)

# ── Assemble rows ─────────────────────────────────────────────────────────────
rows = [
    {'country': 'Bulgaria', 'fuel': 'Gas',          **dict(zip(YEARS, gas_vals))},
    {'country': 'Bulgaria', 'fuel': 'DomesticCoal',  **dict(zip(YEARS, lig_vals))},
    {'country': 'Bulgaria', 'fuel': 'ImportedCoal',  **dict(zip(YEARS, coal_vals))},
    {'country': 'Bulgaria', 'fuel': 'Uranium',       **dict(zip(YEARS, ROMANIA_URANIUM))},
    {'country': 'Bulgaria', 'fuel': 'Biomass',       **dict(zip(YEARS, bio_vals))},
]

df = pd.DataFrame(rows)
COLS = ['country', 'fuel'] + [str(y) for y in YEARS]
df.columns = [str(c) for c in df.columns]
df = df[COLS]

# ── Preview ───────────────────────────────────────────────────────────────────
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 160)
print("Bulgaria pFuelPrice preview ($/MMBtu, 2024-2053):")
print()
cols_show = ['country', 'fuel', '2024', '2025', '2030', '2035', '2040', '2045', '2050', '2053']
print(df[cols_show].to_string(index=False))
print()
print("Note: DomesticCoal = Bulgarian lignite price (Kinesys ELCCOB); flat from 2030 (Bulgaria_Agg_Lignite retires 2033)")
print("Note: Gas 2035 dip reflects Kinesys WEM policy scenario (increased domestic supply)")

# ── Write ─────────────────────────────────────────────────────────────────────
if '--write' in sys.argv:
    target = Path(__file__).resolve().parent.parent / \
             'epm' / 'input' / 'data_blacksea' / 'supply' / 'pFuelPrice.csv'
    existing = pd.read_csv(target, encoding='utf-8-sig', dtype=str)
    existing = existing[existing['country'] != 'Bulgaria']
    updated = pd.concat([existing, df.astype(str)], ignore_index=True)
    updated.to_csv(target, index=False, encoding='utf-8-sig')
    print(f'\nWritten 5 Bulgaria rows -> {target}  ({len(updated)} total rows)')
