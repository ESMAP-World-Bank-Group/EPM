"""Build a copy of an EPM input folder in which sub-zones are merged into one zone per country.

Written for data_sapp_nuc_rsa_moz -> data_sapp_nuc_rsa_moz_agg (Mozambique, Zimbabwe and Angola as single
zones, Mozal kept separate). Edit MAP and MERGE_UNITS below to use it for other folders.

Steps
1. Units listed in MERGE_UNITS are combined into one unit (sizes summed).
2. Rows that still use the country name as zone are deleted: they are leftovers that EPM ignores
   (zone outside zcmap or Status 0) and would otherwise become duplicates.
3. Sub-zones are renamed to the country, and the duplicates this creates are merged with one rule per
   file (sums for demand and transfer capacity, weighted means for profiles, first value for defaults).
   Lines between merged sub-zones are dropped.
4. Map layers are regenerated (geojson_to_epm.csv + epm/postprocessing/create_geojson.py).
Every deletion and merge is listed in <dst>/aggregation_log.md.

Only the files used by config.csv and input_scenarios.csv are converted (plus the constraint files read
from their default path, cplex/, resources/ and static/).

Usage (from the repository root)
    python tools/aggregate_zones.py --src data_sapp_nuc_rsa_moz --dst data_sapp_nuc_rsa_moz_agg [--force]
Then check the result with
    cd epm && python epm.py --folder_input data_sapp_nuc_rsa_moz_agg --check_inputs
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO / 'epm' / 'input'

MAP = {'Mozambique_Central': 'Mozambique', 'Mozambique_South': 'Mozambique', 'Mozambique_North': 'Mozambique',
       'Zimbabwe_East': 'Zimbabwe', 'Zimbabwe_West': 'Zimbabwe',
       'Angola_East': 'Angola', 'Angola_South': 'Angola', 'Angola_North': 'Angola', 'Angola_Central': 'Angola'}
# new unit -> (file parameter, units combined, columns summed); the other columns must be identical
MERGE_UNITS = {
    'New_BESS_MOZ': ('pStorageDataInput', ['New_BESS_MOZ_CE', 'New_BESS_MOZ_NO', 'New_BESS_MOZ_SO'],
                     ['Capacity', 'CapacityMWh', 'BuildLimitperYear']),
}
COUNTRIES = sorted(set(MAP.values()))
LOG = []

_args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_args.add_argument('--src', required=True, help='source folder (name in epm/input or path)')
_args.add_argument('--dst', required=True, help='new folder (name in epm/input or path)')
_args.add_argument('--force', action='store_true', help='delete --dst first if it exists')
_args.add_argument('--no-maps', action='store_true', help='skip the map layers')
ARGS = _args.parse_args()
SRC = Path(ARGS.src) if Path(ARGS.src).is_absolute() else INPUT_DIR / ARGS.src
DST = Path(ARGS.dst) if Path(ARGS.dst).is_absolute() else INPUT_DIR / ARGS.dst


def log(msg):
    LOG.append(msg)
    print(msg)


def read(rel):
    return pd.read_csv(SRC / rel, encoding='utf-8-sig', keep_default_na=False, na_values=[''], dtype=str)


def write(df, rel):
    out = df.copy()
    out.columns = ['' if str(c).startswith('Unnamed:') else c for c in out.columns]
    (DST / rel).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST / rel, index=False)


def num(df, cols):
    return df[cols].apply(pd.to_numeric, errors='coerce')


def fmt(x):
    if pd.isna(x):
        return np.nan
    return f'{x:.10g}'


def drop_and_rename(df, zcols, rel, keep=None):
    bad = df[zcols].isin(COUNTRIES).any(axis=1)
    if keep is not None:
        bad &= ~keep
    if bad.any():
        first = df.columns[0]
        log(f'- `{rel}`: deleted {bad.sum()} row(s) with a country-level zone name: '
            + ', '.join(pd.unique(df.loc[bad, first].astype(str))[:60]))
    df = df.loc[~bad].copy()
    orig = df[zcols].copy()
    for c in zcols:
        df[c] = df[c].replace(MAP)
    return df, orig


def merge_first(df, keys, rel):
    """Merge duplicate keys keeping the first non-empty value of each column; report conflicts."""
    vals = [c for c in df.columns if c not in keys]
    conflicts = []
    for k, g in df.groupby(keys, sort=False):
        if len(g) > 1:
            for c in vals:
                u = g[c].dropna().unique()
                if len(u) > 1:
                    conflicts.append(f'{"/".join(map(str, k))} {c}: {list(u)} -> {u[0]}')
    out = df.groupby(keys, sort=False, dropna=False).first().reset_index()[df.columns]
    if conflicts:
        log(f'- `{rel}`: values differ between sub-zones, first non-empty kept: ' + '; '.join(conflicts))
    return out


def merge_sum(df, keys, rel):
    vals = [c for c in df.columns if c not in keys]
    d = df.copy()
    d[vals] = num(d, vals)
    dmask = d.duplicated(keys, keep=False)
    out = d.groupby(keys, sort=False, dropna=False)[vals].sum(min_count=1).reset_index()
    for c in vals:
        out[c] = out[c].map(fmt)
    if dmask.any():
        merged = d.loc[dmask, keys].drop_duplicates()
        log(f'- `{rel}`: {dmask.sum()} rows summed into {len(merged)} key(s): '
            + ', '.join('/'.join(map(str, r)) for r in merged.drop_duplicates(keys[:2]).values[:30]))
    return out[df.columns]


def drop_internal(df, a, b, rel):
    m = df[a] == df[b]
    if m.any():
        log(f'- `{rel}`: dropped {m.sum()} internal row(s) between merged sub-zones')
    return df.loc[~m].copy()


# ------------------------------------------------------------------ files in use
cfg = pd.read_csv(SRC / 'config.csv')
sc = pd.read_csv(SRC / 'input_scenarios.csv').set_index('paramNames')
used = {}
for p, f in zip(cfg.paramNames, cfg.file):
    if str(f).endswith('.csv'):
        used.setdefault(f, p)
for p in sc.index:
    for f in sc.loc[p].dropna():
        if str(f).endswith('.csv'):
            used.setdefault(f, p)

if not (SRC / 'config.csv').exists():
    raise SystemExit(f'{SRC} has no config.csv')
if DST.exists():
    if not ARGS.force:
        raise SystemExit(f'{DST} exists: use --force to rebuild it')
    if not (DST / 'aggregation_log.md').exists():
        raise SystemExit(f'{DST} was not built by this script (no aggregation_log.md): not deleted')
    shutil.rmtree(DST)
DST.mkdir(parents=True)
for d in ('cplex', 'resources', 'static'):
    if (SRC / d).exists():
        shutil.copytree(SRC / d, DST / d)
for f in ('config.csv', 'input_scenarios.csv', 'mc_uncertainties.csv'):
    shutil.copy2(SRC / f, DST / f)
# read by input_readers.gms from their default path when config.csv does not list them
for f in ('constraint/pMinGenerationByFuel.csv', 'constraint/pMaxShareGenerationByTech.csv'):
    if (SRC / f).exists():
        (DST / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / f, DST / f)

# ------------------------------------------------------------------ generators (needed for weights and removed names)
removed_gens = set()
gen_files = [f for f, p in used.items() if p in ('pGenDataInput', 'pStorageDataInput')]
base_gen = read(cfg.set_index('paramNames').loc['pGenDataInput', 'file'])
w = base_gen[base_gen.Status.isin(['1', '2', '3'])].copy()
w['Capacity'] = pd.to_numeric(w.Capacity, errors='coerce').fillna(0)
VRE_WEIGHT = w.groupby(['z', 'tech']).Capacity.sum()          # sub-zone VRE capacity (existing + committed + candidate)

handled = set()
for rel, p in used.items():
    df = read(rel)
    if p in ('pGenDataInput', 'pStorageDataInput'):
        for new, (param, units, summed) in MERGE_UNITS.items():
            sel = df.g.isin(units)
            if param != p or not sel.any():
                continue
            if sel.sum() != len(units):
                raise SystemExit(f'{rel}: {new}: found {sorted(df.g[sel])}, expected {units}')
            part = df.loc[sel]
            other = [c for c in df.columns if c not in summed + ['g']]
            diff = [c for c in other if part[c].nunique(dropna=False) > 1]
            if diff:
                raise SystemExit(f'{rel}: {new}: {units} differ in {diff}, cannot merge')
            row = part.iloc[[0]].copy()
            row['g'] = new
            for c in summed:
                row[c] = fmt(pd.to_numeric(part[c], errors='coerce').sum(min_count=1))
            row['z'] = row['z'].replace(MAP)
            df = pd.concat([df.loc[~sel], row])
            log(f'- `{rel}`: {", ".join(units)} combined into {new} (zone {row.z.iloc[0]}, '
                + ', '.join(f'{c} {row[c].iloc[0]}' for c in summed) + ')')
        merged_new = df.g.isin(MERGE_UNITS)                 # combined units are kept by the country-name deletion
        bad = df.z.isin(COUNTRIES) & ~merged_new
        removed_gens |= set(df.loc[bad, 'g'])
        df, _ = drop_and_rename(df, ['z'], rel, keep=merged_new)
        dup = df.g[df.g.duplicated()]
        if len(dup):
            raise SystemExit(f'{rel}: duplicated generator names {list(dup)}')
        write(df, rel)
        handled.add(rel)
log(f'- generators/storage deleted (status 0, or zone not in zcmap): {len(removed_gens)}')

# ------------------------------------------------------------------ all other files
for rel, p in used.items():
    if rel in handled:
        continue
    df = read(rel)
    cols = list(df.columns)
    if p == 'zcmap':
        df['z'] = df.z.replace(MAP)
        df = df.drop_duplicates()
    elif p == 'pDemandForecast':
        df, _ = drop_and_rename(df, ['z'], rel)
        df = merge_sum(df, ['z', 'type'], rel)
    elif p == 'pDemandProfile':
        fc = read(cfg.set_index('paramNames').loc['pDemandForecast', 'file'])
        e = fc[fc.type.str.lower() == 'energy'].set_index('z')['2025'].astype(float)
        df, orig = drop_and_rename(df, ['Zone'], rel)
        tcols = [c for c in cols if c[:1] == 't' and c[1:].isdigit()]
        df[tcols] = num(df, tcols)
        df['w'] = orig['Zone'].map(e).fillna(1).to_numpy()
        rows = []
        for (z, m, d), g in df.groupby(['Zone', 'm', 'd'], sort=False):
            v = (g[tcols].mul(g.w, axis=0).sum() / g.w.sum())
            rows.append([z, m, d] + [float(x) for x in v])
        df = pd.DataFrame(rows, columns=cols)
        df[tcols] = df[tcols].astype(float)
        mx = df.groupby('Zone')[tcols].max().max(axis=1)
        for z in COUNTRIES:
            if z in mx.index and abs(mx[z] - 1) > 1e-6:
                df.loc[df.Zone == z, tcols] = df.loc[df.Zone == z, tcols] / mx[z]
        for c in tcols:
            df[c] = df[c].round(6).map(fmt)
        log(f'- `{rel}`: merged zones = 2025 energy-weighted mean of the sub-zone profiles, rescaled to max 1')
    elif p == 'pVREProfile':
        df, orig = drop_and_rename(df, ['zone'], rel)
        tcols = [c for c in cols if c[:1] == 't' and c[1:].isdigit()]
        df[tcols] = num(df, tcols)
        df['w'] = [VRE_WEIGHT.get((z, t), 0.0) for z, t in zip(orig['zone'], df.tech)]
        rows, used_w = [], {}
        for (z, tech, q, d), g in df.groupby(['zone', 'tech', 'q', 'd'], sort=False):
            ww = g.w if g.w.sum() > 0 else pd.Series(1.0, index=g.index)
            rows.append([z, tech, q, d] + [float(x) for x in g[tcols].mul(ww, axis=0).sum() / ww.sum()])
            if z in COUNTRIES:
                used_w[(z, tech)] = dict(zip(orig.loc[g.index, 'zone'], ww.round(0)))
        df = pd.DataFrame(rows, columns=cols)
        df[tcols] = df[tcols].astype(float)
        for c in tcols:
            df[c] = df[c].round(5).map(fmt)
        log(f'- `{rel}`: merged zones = VRE-capacity-weighted mean of the sub-zone profiles (equal weights if no plant): '
            + '; '.join(f'{z} {t}: {wt}' for (z, t), wt in sorted(used_w.items())))
    elif p in ('pGenDataInputDefault', 'pAvailabilityDefault', 'pCapexTrajectoriesDefault', 'pStorageDataInputDefault'):
        key = ['z', 'tech', 'f' if 'f' in cols else 'fuel']
        df, _ = drop_and_rename(df, ['z'], rel)
        df = merge_first(df, key, rel)
    elif p == 'pTransferLimit':
        df, _ = drop_and_rename(df, ['From', 'To'], rel)
        df = drop_internal(df, 'From', 'To', rel)
        df = merge_sum(df, ['From', 'To', 'q'], rel)
    elif p == 'pMinImport':
        a, b = cols[0], cols[1]
        df, _ = drop_and_rename(df, [a, b], rel)
        df = drop_internal(df, a, b, rel)
        df = merge_sum(df, [a, b], rel)
    elif p == 'pMaxGenerationByFuel':
        df, _ = drop_and_rename(df, ['z'], rel)
        df = merge_sum(df, ['z', 'tech', 'f'], rel)
    elif p == 'pNewTransmission':
        df = df.dropna(how='all')
        df, orig = drop_and_rename(df, ['From', 'To'], rel)
        df['orig'] = orig.From + '->' + orig.To
        df = drop_internal(df, 'From', 'To', rel)
        df['active'] = ~df.Status.isin(['0']) & df.Status.notna()
        keep = []
        for k, g in df.groupby(['From', 'To'], sort=False):
            g = g.sort_values('active', ascending=False, kind='stable')
            keep.append(g.index[0])
            if len(g) > 1:
                log(f'- `{rel}`: {k[0]}->{k[1]} has {len(g)} candidates ({", ".join(g.orig + " status " + g.Status.astype(str))}): kept {g.orig.iloc[0]}')
        df = df.loc[keep, cols]
        pairs = set(zip(df.From, df.To))
        both = sorted({tuple(sorted(x)) for x in pairs if (x[1], x[0]) in pairs})
        if both:
            log(f'- `{rel}`: corridors listed in both directions (as in the source): {both}')
    elif p == 'pLossFactorInternal':
        tl = read(cfg.set_index('paramNames').loc['pTransferLimit', 'file'])
        ycols = [c for c in tl.columns if c not in ('From', 'To', 'q')]
        tl['cap'] = num(tl, ycols).max(axis=1)
        cap = tl.groupby(['From', 'To']).cap.max()
        df, orig = drop_and_rename(df, ['z', 'z2'], rel)
        df['cap'] = [max(cap.get((a, b), 0), cap.get((b, a), 0)) for a, b in zip(orig.z, orig.z2)]
        df['orig'] = orig.z + '->' + orig.z2
        df = drop_internal(df, 'z', 'z2', rel)
        keep = []
        for k, g in df.groupby(['z', 'z2'], sort=False):
            g = g.sort_values('cap', ascending=False, kind='stable')
            keep.append(g.index[0])
            vals = g[cols[2:]].drop_duplicates()
            if len(g) > 1 and len(vals) > 1:
                log(f'- `{rel}`: {k[0]}->{k[1]} loss factors differ ({", ".join(g.orig + " " + g["2030"].astype(str))}): '
                    f'kept {g.orig.iloc[0]}, the corridor with the largest existing capacity')
        df = df.loc[keep, cols]
    else:
        # files without zones, or indexed by country / generator / external zone
        if p in ('pAvailability', 'pEvolutionAvailability', 'pCapexTrajectories', 'pVREgenProfile', 'pCSPData'):
            key = cols[0]
            m = df[key].isin(removed_gens)
            if m.any():
                log(f'- `{rel}`: deleted {m.sum()} row(s) of deleted generators: {sorted(df.loc[m, key].unique())}')
                df = df.loc[~m]
            else:
                (DST / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(SRC / rel, DST / rel)
                continue
        else:
            (DST / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / rel, DST / rel)
            continue
    write(df, rel)

# ------------------------------------------------------------------ checks
left = []
for f in DST.rglob('*.csv'):
    txt = f.read_text(encoding='utf-8-sig')
    hits = [z for z in MAP if z in txt]
    if hits:
        left.append(f'{f.relative_to(DST)}: {hits}')
log('- sub-zone names left in the new folder: ' + ('; '.join(left) if left else 'none'))

# ------------------------------------------------------------------ map layers
if not ARGS.no_maps:
    shared = REPO / 'epm' / 'resources' / 'postprocess' / 'geojson_to_epm.csv'
    src_map = SRC / 'geojson_to_epm.csv'
    gm = pd.read_csv(src_map if src_map.exists() else shared, encoding='utf-8-sig', dtype=str)
    for c in COUNTRIES:
        if not (gm.epm_zone == c).any():
            gm.loc[len(gm)] = [c, c] + [np.nan] * (gm.shape[1] - 2)
    gm.to_csv(DST / 'geojson_to_epm.csv', index=False)
    r = subprocess.run([sys.executable, str(REPO / 'epm' / 'postprocessing' / 'create_geojson.py'), '--folder', DST.name],
                       cwd=REPO, capture_output=True, text=True)
    ok = r.returncode == 0 and 'no map geometry' not in r.stdout
    log('- map layers: `geojson_to_epm.csv` (mapping + whole-country rows) and `zones*.geojson`, `linestring*.geojson` '
        + ('regenerated' if ok else 'FAILED, run `python epm/postprocessing/create_geojson.py --folder ' + DST.name + '`'))
    if not ok:
        print(r.stdout[-2000:], r.stderr[-2000:])

(DST / 'aggregation_log.md').write_text(
    f'# Aggregation log\n\nBuilt from `{SRC.name}` with `tools/aggregate_zones.py`: '
    + '; '.join(f'{c} ({", ".join(z for z, v in MAP.items() if v == c)})' for c in COUNTRIES)
    + ' merged into one zone each. Other zones (Mozal included) are unchanged.\n\n' + '\n'.join(LOG) + '\n',
    encoding='utf-8')
