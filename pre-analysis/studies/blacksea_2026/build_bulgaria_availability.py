"""
build_bulgaria_availability.py — generate Bulgaria pAvailabilityCustom rows.

Plants covered (4 entries / groups):
  1. Bulgaria_Agg_Lignite   — standard Turkiye lignite pattern
  2. Bulgaria_Agg_Coal      — standard ImpCoal pattern
  3. Bulgaria_Kozloduy_5/6  — IAEA PRIS EAF scaled by ENTSO-E seasonal shape
  4. Bulgaria_*_Hydro (13)  — ENTSO-E seasonal shape scaled to Kinesys annual level

Sources:
  - pAvailabilityCustom existing rows : standard patterns for lignite/coal
  - wna_pris_kozloduy       : IAEA PRIS / WNA annual EAF 2015-2024
                              K5 mean=88.4%, K6 mean=87.0%
                              (world-nuclear.org/nuclear-reactor-database/details/KOZLODUY-5/6)
  - entsoe_gen_bg_2019_2023 : ENTSO-E Transparency Platform, actual generation
                              by source for Bulgaria, hourly 2019-2023
                              query_generation('BG'), columns Nuclear + Hydro Water Reservoir
                              Used for: seasonal shape only (relative Q1/Q2/Q3/Q4 distribution)
  - bg_ccdr_kinesys_2026    : Kinesys WEM Bulgaria CCDR
                              Hydro annual generation 2025 = 2.930 TWh / 1719 MW → CF = 0.195
                              Used for: hydro annual level (more reliable than ENTSO-E 2019-2023
                              mean which reflects drier-than-average historical period)

Review notes:
  - Nuclear Q1 CF from ENTSO-E slightly >1.0 (1.02): capacity uprating of K5/K6 to 1003 MW each
    (total 2006 MW not 2080 MW) likely explains the discrepancy. Values scaled to PRIS EAF so
    the Q1 entry is capped at 1.0 after scaling. Conservative approach.
  - Hydro: ENTSO-E 2019-2023 mean CF = 0.146 vs Kinesys 2025 = 0.195. Gap (~25%) likely reflects
    drier-than-average historical period + Kinesys model optimisation. Kinesys used for level.
  - Hydro variability is high (2021 CF=0.26 vs 2023 CF=0.12). Single seasonal profile is a
    simplification; individual plant profiles not available for Bulgaria.
  - K5/K6 share the same seasonal shape (ENTSO-E has combined BG nuclear generation only).
    Their PRIS EAF values differ slightly (K5=88.4%, K6=87.0%) → different absolute values.
  - Bulgaria_Kozloduy_7/8 (committed): no custom entry needed — Generic_Nuclear 0.85 flat is
    appropriate for new VVER-1200 build.
  - Lignite Q2 dip (0.45) and Coal Q2 dip (0.60): reflects spring maintenance outages,
    consistent with all other zone patterns in the model.

Run:
  python pre-analysis/build_bulgaria_availability.py           # preview
  python pre-analysis/build_bulgaria_availability.py --write   # append to pAvailabilityCustom.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── 1. Standard patterns (copied from existing zone entries) ─────────────────
# Source: pAvailabilityCustom.csv — Trakia_Agg_Lignite (all zones identical)
LIGNITE = {'Q1': 0.50, 'Q2': 0.45, 'Q3': 0.50, 'Q4': 0.50}

# Source: pAvailabilityCustom.csv — SouthEast_Agg_ImpCoal
IMP_COAL = {'Q1': 0.85, 'Q2': 0.60, 'Q3': 0.85, 'Q4': 0.85}

# ── 2. Nuclear: ENTSO-E seasonal shape scaled to PRIS EAF ───────────────────
# ENTSO-E 5-year average (2019-2023) CF shape — combined K5+K6 (2080 MW basis):
#   Q1=1.0225, Q2=0.7758, Q3=0.9617, Q4=0.8462, mean=0.9016
# Source: ENTSO-E query_generation('BG'), columns 'Nuclear', hourly 2019-2023
ENTSOE_NUC_SHAPE = np.array([1.0225, 0.7758, 0.9617, 0.8462])
ENTSOE_NUC_MEAN  = ENTSOE_NUC_SHAPE.mean()

# IAEA PRIS / WNA annual EAF 2015-2024:
#   K5 mean = 88.4%, K6 mean = 87.0%
# Source: world-nuclear.org nuclear reactor database, Kozloduy-5 and Kozloduy-6
K5_EAF = 0.884
K6_EAF = 0.870


def nuc_profile(eaf: float) -> dict:
    """Scale ENTSO-E seasonal shape to target annual EAF, cap at 1.0."""
    vals = ENTSOE_NUC_SHAPE * (eaf / ENTSOE_NUC_MEAN)
    vals = np.minimum(vals, 1.0)
    return {f'Q{i+1}': round(float(v), 4) for i, v in enumerate(vals)}


# ── 3. Hydro: ENTSO-E seasonal shape scaled to Kinesys annual level ─────────
# ENTSO-E 5-year average (2019-2023) CF shape — Hydro Water Reservoir (1719 MW):
#   Q1=0.1609, Q2=0.1863, Q3=0.1205, Q4=0.1169, mean=0.1462
# Source: ENTSO-E query_generation('BG'), column 'Hydro Water Reservoir', hourly 2019-2023
ENTSOE_HYD_SHAPE = np.array([0.1609, 0.1863, 0.1205, 0.1169])
ENTSOE_HYD_MEAN  = ENTSOE_HYD_SHAPE.mean()   # 0.1462

# Annual level from Kinesys WEM (BG-CCDR, 2025 generation):
#   Hydro 2.930 TWh / 1719 MW / 8.76 TH = CF 0.195
# Source: bg_ccdr_kinesys_2026, sheet 'Ele generation and trades', WEM scenario
KINESYS_HYDRO_CF = 2.930257 / (1719 * 8.76 / 1000)  # = 0.1946

hyd_vals = ENTSOE_HYD_SHAPE * (KINESYS_HYDRO_CF / ENTSOE_HYD_MEAN)
HYDRO = {f'Q{i+1}': round(float(v), 4) for i, v in enumerate(hyd_vals)}

# ── 4. Hydro plants list ─────────────────────────────────────────────────────
HYDRO_PLANTS = [
    'Bulgaria_Belmeken_Hydro', 'Bulgaria_Devin_Hydro', 'Bulgaria_Ivailovgrad_Hydro',
    'Bulgaria_Kardjali_Hydro', 'Bulgaria_Krichim_Hydro', 'Bulgaria_Momina_Klisura_Hydro',
    'Bulgaria_Orfeus_Hydro',   'Bulgaria_Peshtera_Hydro','Bulgaria_Sestrimo_Hydro',
    'Bulgaria_Studen_Kladenets_Hydro', 'Bulgaria_Tsankov_Kamak_Hydro',
    'Bulgaria_Teshel_Hydro',   'Bulgaria_Aleko_Hydro',
    # Candidates (same profile — water availability unchanged)
    'Bulgaria_Turnu_Magurele_Hydro', 'Bulgaria_Batak_Hydro', 'Bulgaria_Dospat_Hydro',
]

# ── 5. Assemble rows ─────────────────────────────────────────────────────────
rows = [
    {'gen': 'Bulgaria_Agg_Lignite', **LIGNITE},
    {'gen': 'Bulgaria_Agg_Coal',    **IMP_COAL},
    {'gen': 'Bulgaria_Kozloduy_5',  **nuc_profile(K5_EAF)},
    {'gen': 'Bulgaria_Kozloduy_6',  **nuc_profile(K6_EAF)},
] + [{'gen': g, **HYDRO} for g in HYDRO_PLANTS]

df = pd.DataFrame(rows, columns=['gen', 'Q1', 'Q2', 'Q3', 'Q4'])

# ── 6. Preview ───────────────────────────────────────────────────────────────
print('Bulgaria pAvailabilityCustom preview:')
print()
print(df.to_string(index=False))
print()
print(f'Hydro profile (ENTSO-E shape × Kinesys level {KINESYS_HYDRO_CF:.3f}):')
print(f'  Q1={HYDRO["Q1"]}, Q2={HYDRO["Q2"]}, Q3={HYDRO["Q3"]}, Q4={HYDRO["Q4"]}  '
      f'(mean={sum(HYDRO.values())/4:.3f})')
print()
print(f'Nuclear K5 profile (PRIS EAF={K5_EAF:.1%}):')
p5 = nuc_profile(K5_EAF)
print(f'  Q1={p5["Q1"]}, Q2={p5["Q2"]}, Q3={p5["Q3"]}, Q4={p5["Q4"]}  '
      f'(mean={sum(p5.values())/4:.3f})')
print(f'Nuclear K6 profile (PRIS EAF={K6_EAF:.1%}):')
p6 = nuc_profile(K6_EAF)
print(f'  Q1={p6["Q1"]}, Q2={p6["Q2"]}, Q3={p6["Q3"]}, Q4={p6["Q4"]}  '
      f'(mean={sum(p6.values())/4:.3f})')
print()
print(f'Total rows: {len(df)}')

# ── 7. Write ─────────────────────────────────────────────────────────────────
if '--write' in sys.argv:
    target = Path(__file__).resolve().parent.parent / \
             'epm' / 'input' / 'data_blacksea' / 'supply' / 'pAvailabilityCustom.csv'
    existing = pd.read_csv(target, encoding='utf-8-sig')
    # Remove any existing Bulgaria entries
    existing = existing[~existing['gen'].str.startswith('Bulgaria_')]
    updated = pd.concat([existing, df], ignore_index=True)
    updated.to_csv(target, index=False, encoding='utf-8-sig')
    print(f'Written {len(df)} Bulgaria rows -> {target}  ({len(updated)} total rows)')
