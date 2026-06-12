"""
build_bulgaria_gendata.py — preview complete pGenDataInput for Bulgaria.
Dry-run only (no file writes). Combines GEM GIPT + Kinesys manual rows.
"""
import sys, math
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from pipelines.gem_pipeline import load_gipt_plants
import pandas as pd

df = load_gipt_plants(countries=['BGR'], verbose=False)

COLS = ['g','z','tech','f','Status','StYr','RetrYr','Capacity','BuildLimitperYear',
        'Life','HeatRate','RampUpRate','RampDnRate','ResLimShare','Capex',
        'FOMperMW','VOM','ReserveCost','UnitSize']

def R(g, z, tech, f, status, styr, retryr, cap, blpy='', capex='', unitsize='', hr=''):
    return {'g':g,'z':z,'tech':tech,'f':f,'Status':status,
            'StYr':styr,'RetrYr':retryr,'Capacity':cap,
            'BuildLimitperYear':blpy,'Life':'','HeatRate':hr,'RampUpRate':'',
            'RampDnRate':'','ResLimShare':'','Capex':capex,
            'FOMperMW':'','VOM':'','ReserveCost':'','UnitSize':unitsize}

# Heat rates derived from Kinesys WEM 2025 fuel consumption / generation:
#   Lignite 97.26 PJ / 8.34 TWh = 11.66 GJ/MWh  (vs generic 10.3)
#   Nuclear 164.77 PJ / 14.65 TWh = 11.25 GJ/MWh (vs generic 12.5)
#   Gas CCGT 15.70 PJ / 2.64 TWh = 5.94 GJ/MWh   (vs generic 6.4)
#   Coal = 4.5 GJ/MWh (CHP attribution artifact) -> use generic 8.5
HR_LIGNITE  = 11.7
HR_NUCLEAR  = 11.3
HR_GAS_CCGT = 5.9
HR_GAS_OCGT = 9.0   # generic (no Kinesys data for old OCGT)

Z = 'Bulgaria'
rows = []

# ── LIGNITE aggregate (Kinesys WEM 2025: 3966 MW; phase-out by 2033) ────
rows.append(R('Bulgaria_Agg_Lignite', Z, 'ST', 'Lignite', 1, 2000, 2033, 3966, hr=HR_LIGNITE))

# ── HARD COAL aggregate (Kinesys WEM 2025: 669 MW) ──────────────────────
rows.append(R('Bulgaria_Agg_Coal', Z, 'ST', 'Coal', 1, 2000, 2035, 669))

# ── GAS CCGT existing (GEM, post-2000 only: 303 MW) ─────────────────────
rows.append(R('Bulgaria_Plovdiv_North_CCGT', Z, 'CCGT', 'Gas', 1, 2011, 2041, 50,  hr=HR_GAS_CCGT))
rows.append(R('Bulgaria_Varna_CCGT',         Z, 'CCGT', 'Gas', 1, 2019, 2049, 210, hr=HR_GAS_CCGT))
rows.append(R('Bulgaria_Toplofikacia_CCGT',  Z, 'CCGT', 'Gas', 1, 2007, 2037, 43,  hr=HR_GAS_CCGT))

# ── GAS old aggregate (Kinesys 1357 - 303 GEM = 1054 MW) ────────────────
rows.append(R('Bulgaria_Agg_Gas_Old', Z, 'OCGT', 'Gas', 1, 2000, 2040, 1054, hr=HR_GAS_OCGT))

# ── GAS candidates (GEM planned) ────────────────────────────────────────
rows.append(R('Bulgaria_Sofia_Iztok_CCGT', Z, 'CCGT', 'Gas', 3, 2024, 2054, 240, 48, 1.0))
rows.append(R('Bulgaria_Rupite_CCGT',       Z, 'CCGT', 'Gas', 3, 2030, 2060, 276, 55, 1.0))
rows.append(R('Bulgaria_Bobov_Dol_CCGT',    Z, 'CCGT', 'Gas', 3, 2025, 2055,  42,  8, 1.0))

# ── NUCLEAR K5+K6 existing (GEM) ────────────────────────────────────────
rows.append(R('Bulgaria_Kozloduy_5', Z, 'Nuclear', 'Uranium', 1, 1993, 2053, 1040, unitsize=1040, hr=HR_NUCLEAR))
rows.append(R('Bulgaria_Kozloduy_6', Z, 'Nuclear', 'Uranium', 1, 1988, 2048, 1040, unitsize=1040, hr=HR_NUCLEAR))

# ── NUCLEAR K7+K8 committed (GEM planned -> Status=2 per Kinesys WEM) ───
rows.append(R('Bulgaria_Kozloduy_7', Z, 'Nuclear', 'Uranium', 2, 2033, 2093, 1000, capex=6.0, unitsize=1000, hr=HR_NUCLEAR))
rows.append(R('Bulgaria_Kozloduy_8', Z, 'Nuclear', 'Uranium', 2, 2036, 2096, 1000, capex=6.0, unitsize=1000, hr=HR_NUCLEAR))

# ── RESERVOIR HYDRO existing (GEM, Chaira moved to Storage) ─────────────
hydro_plants = [
    ('Belmeken',        375, 1974), ('Devin',          88, 1984),
    ('Ivailovgrad',     114, 1964), ('Kardjali',       110, 1963),
    ('Krichim',          80, 1972), ('Momina_Klisura', 120, 1975),
    ('Orfeus',          160, 1975), ('Peshtera',       135, 1959),
    ('Sestrimo',        240, 1974), ('Studen_Kladenets', 81, 1958),
    ('Tsankov_Kamak',    85, 2009), ('Teshel',          60, 1984),
    ('Aleko',            71, ''),
]
for name, mw, yr in hydro_plants:
    rows.append(R(f'Bulgaria_{name}_Hydro', Z, 'ReservoirHydro', 'Water', 1, yr, '', mw))

# ── CHAIRA reclassified -> Storage/Water (pumped hydro) ─────────────────
rows.append(R('Bulgaria_Chaira_Storage', Z, 'Storage', 'Water', 1, 1995, '', 864))

# ── HYDRO candidates (GEM: 3 large projects; note Kinesys hydro is flat) ─
rows.append(R('Bulgaria_Turnu_Magurele_Hydro', Z, 'ReservoirHydro', 'Water', 3, 2030, '', 840, 168, 3.5))
rows.append(R('Bulgaria_Batak_Hydro',           Z, 'ReservoirHydro', 'Water', 3, 2032, '', 800, 160, 3.5))
rows.append(R('Bulgaria_Dospat_Hydro',          Z, 'ReservoirHydro', 'Water', 3, 2032, '', 800, 160, 3.5))

# ── SOLAR PV existing (GEM operating, agg <10 MW) ───────────────────────
solar_op = df[(df['fuel'] == 'solar') & (df['status'] == 'operating')].copy()
small_pv  = solar_op[solar_op['mw'] < 10]
big_pv    = solar_op[solar_op['mw'] >= 10]
agg_small = round(small_pv['mw'].sum(), 1)

ctr = {}
for _, p in big_pv.iterrows():
    mw  = round(p['mw'], 1)
    yr  = int(p['year']) if pd.notna(p['year']) else ''
    rtr = (yr + 25) if yr else ''
    raw = str(p['name']).replace(' power station','').replace(' solar farm','') \
                        .replace(' solar park','').replace(' solar project','').strip()
    raw = ''.join(c if c.isalnum() or c in ' _-' else '' for c in raw) \
            .strip().replace(' ', '_')[:18]
    g = f'Bulgaria_{raw}_PV'
    if g in ctr: ctr[g] += 1; g = f'{g}_{ctr[g]}'
    else: ctr[g] = 0
    rows.append(R(g, Z, 'PV', 'Solar', 1, yr, rtr, mw))

if agg_small > 0:
    rows.append(R('Bulgaria_AGG_SmallPV', Z, 'PV', 'Solar', 1, '', '', agg_small))

# PV construction -> committed
for _, p in df[(df['fuel']=='solar') & (df['status']=='construction')].iterrows():
    mw  = round(p['mw'], 1)
    yr  = int(p['year']) if pd.notna(p['year']) else 2025
    rtr = yr + 25
    raw = str(p['name']).replace(' power station','').strip().replace(' ','_')[:18]
    g   = f'Bulgaria_{raw}_PV'
    rows.append(R(g, Z, 'PV', 'Solar', 2, yr, rtr, mw, capex=0.8))

# PV planned -> candidates
for _, p in df[(df['fuel']=='solar') & (df['status']=='planned')].iterrows():
    mw   = round(p['mw'], 1)
    yr   = int(p['year']) if pd.notna(p['year']) else 2030
    rtr  = yr + 25
    blpy = round(mw * 0.2, 1)
    raw  = str(p['name']).replace(' power station','').strip().replace(' ','_')[:18]
    g    = f'Bulgaria_{raw}_PV'
    rows.append(R(g, Z, 'PV', 'Solar', 3, yr, rtr, mw, blpy, 0.8))

# Generic solar candidate (fills Kinesys 5370-3941=1429 MW gap + future headroom)
rows.append(R('Generic_Solar_Bulgaria', Z, 'PV', 'Solar', 3, 2025, 2050, 10000, 1000, 0.8))

# ── WIND existing (GEM operating) ───────────────────────────────────────
for _, p in df[(df['fuel']=='wind') & (df['status']=='operating')].iterrows():
    mw  = round(p['mw'], 1)
    yr  = int(p['year']) if pd.notna(p['year']) else ''
    rtr = (yr + 25) if yr else ''
    raw = str(p['name']).replace(' power station','').replace(' wind farm','') \
                        .replace(' wind park','').strip()
    raw = ''.join(c if c.isalnum() or c in ' _-' else '' for c in raw) \
            .strip().replace(' ','_')[:18]
    g = f'Bulgaria_{raw}_Wind'
    if g in ctr: ctr[g] += 1; g = f'{g}_{ctr[g]}'
    else: ctr[g] = 0
    rows.append(R(g, Z, 'OnshoreWind', 'Wind', 1, yr, rtr, mw))

# Wind planned -> candidates
for _, p in df[(df['fuel']=='wind') & (df['status']=='planned')].iterrows():
    mw   = round(p['mw'], 1)
    yr   = int(p['year']) if pd.notna(p['year']) else 2030
    rtr  = yr + 25
    blpy = round(mw * 0.2, 1)
    raw  = str(p['name']).replace(' power station','').strip().replace(' ','_')[:18]
    g    = f'Bulgaria_{raw}_Wind'
    rows.append(R(g, Z, 'OnshoreWind', 'Wind', 3, yr, rtr, mw, blpy, 1.3))

# Generic wind (fills Kinesys 3556 MW 2040 target)
rows.append(R('Generic_Wind_Bulgaria', Z, 'OnshoreWind', 'Wind', 3, 2025, 2050, 10000, 600, 1.3))

# ── BATTERY (Kinesys WEM 2025: 728 MW, life ~10yr) ──────────────────────
rows.append(R('Bulgaria_Battery', Z, 'Storage', 'Battery', 1, 2024, 2034, 728))

# ── OUTPUT ──────────────────────────────────────────────────────────────
out = pd.DataFrame(rows, columns=COLS)

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 38)

print(out[['g','tech','f','Status','StYr','RetrYr','Capacity',
           'BuildLimitperYear','HeatRate','Capex','UnitSize']].to_string(index=False))
print()
print('=' * 65)
print('SUMMARY BY TECH / FUEL / STATUS (MW)')
print('=' * 65)
summary = out.groupby(['tech','f','Status'])['Capacity'] \
             .apply(lambda x: pd.to_numeric(x, errors='coerce').sum()) \
             .reset_index()
summary['Status'] = summary['Status'].map({1:'Existing',2:'Committed',3:'Candidate'})
print(summary.to_string(index=False))
print()
print(f'Total rows: {len(out)}')

# ── WRITE to pGenDataInput.csv ───────────────────────────────────────────
import argparse, sys as _sys
if '--write' in _sys.argv:
    target = __import__('pathlib').Path(__file__).resolve().parent.parent / \
             'epm' / 'input' / 'data_blacksea' / 'supply' / 'pGenDataInput.csv'
    existing = pd.read_csv(target, encoding='utf-8-sig')
    existing = existing[existing['z'] != 'Bulgaria']
    updated  = pd.concat([existing, out], ignore_index=True)
    updated.to_csv(target, index=False, encoding='utf-8-sig')
    print(f'\nWritten {len(out)} Bulgaria rows -> {target}  ({len(updated)} total rows)')
