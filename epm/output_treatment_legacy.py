"""
**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
Description:
    Converts results produced by a legacy EPM version (old generate_report:
    pCapacityPlan, pEnergyByPlant, pCostSummary, pDemandSupply, ...) into the
    current output format, so that they can be exploited by output_treatment.py
    (Tableau CSVs) and postprocessing.py (figures).

    For each scenario folder containing a legacy epmresults.gdx, it writes to a
    separate destination folder (the legacy results are never modified):

        <dest>/<scenario>/epmresults.gdx   current-format GDX (read by postprocessing.py)
        <dest>/<scenario>/output_csv/      CSVs as written by generate_report.gms,
                                           then treated by run_output_treatment()
        <dest>/<scenario>/legacy_generator_techfuel.csv
                                           inferred generator -> (tech, fuel) mapping

    What cannot be read directly from the legacy GDX is reconstructed:
      - Generator -> (tech, fuel): from the name suffix (_TPP, _PV, _HPP, ...),
        ambiguities resolved so that plant capacities and energies add up exactly
        to the legacy pCapacityByTechandFuel / pEnergyByTechandFuel.
      - pHours: from seasonal vs hourly demand (pDemandSupplySeason / pDemandTotalZ).
      - pWeightYear / pRR: same formulas as main.gms, discount rate from pScalars.
      - Legacy tech/fuel names are mapped to current names (LEGACY_TECHFUEL_MAP).

    Approximations (logged at runtime):
      - pNewCapacityPlant = positive year-on-year capacity increase (0 in first year).
      - pCapexInvestmentPlant: zonal legacy 'Capex: $m' split across plants
        pro rata of their annuity increase in the build year.
      - System reserve violations are only available as NPV totals (pSummary):
        they are added to pNetPresentCost* (allocated to zones by discounted
        demand) but not to the yearly pCosts.
      - pTransmissionCapacity derived from interchange / utilization.

Usage:
    python epm/output_treatment_legacy.py epm/output/Zimbabwe_SPD
    python epm/output_treatment_legacy.py epm/output/Zimbabwe_SPD --dest epm/output/Zimbabwe_SPD_converted
    # then, from epm/:
    python epm.py --postprocess output/Zimbabwe_SPD_converted --generate-figures

Author(s):
    ESMAP Modelling Team

Organization:
    World Bank

License:
    Creative Commons Zero v1.0 Universal
**********************************************************************
"""

import argparse
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gams.transfer as gt

try:
    # flat import - the module's own CLI runs with the epm/ directory on sys.path
    from output_treatment import run_output_treatment, TECHFUEL_PROCESSING_PATH
except ImportError:  # imported as epm.output_treatment_legacy
    from epm.output_treatment import run_output_treatment, TECHFUEL_PROCESSING_PATH


# =============================================================================
# CONFIGURATION
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TECHFUEL_PATH = os.path.join(_SCRIPT_DIR, 'resources', 'pTechFuel.csv')

# Legacy reports add 1e-6 to many values so that zeros are written; drop them
LEGACY_EPSILON = 1.5e-6

# Legacy (tech, fuel) -> current (tech, fuel). Current names must exist in
# resources/postprocess/pTechFuelProcessing.csv. Extend with --techfuel-map.
LEGACY_TECHFUEL_MAP = {
    ('Coal', '1.Coal'): ('ST', 'Coal'),
    ('ST', '1.Coal'): ('ST', 'Coal'),
    ('CoalCCS', '2.CoalCCS'): ('ST', 'Coal'),
    ('CCGT', '3.NaturalGas'): ('CCGT', 'Gas'),
    ('GT', '3.NaturalGas'): ('OCGT', 'Gas'),
    ('ICE', '3.NaturalGas'): ('ICE', 'Gas'),
    ('GasCCS', '4.NaturalGasCCS'): ('CCGT', 'Gas'),
    ('STO HY', '5.HydroStor'): ('ReservoirHydro', 'Water'),
    ('Storage', '5.HydroStor'): ('Storage', 'Water'),
    ('ROR', '6.RoR'): ('ROR', 'Water'),
    ('Wind', '7.Wind'): ('OnshoreWind', 'Wind'),
    ('PV', '8.PV'): ('PV', 'Solar'),
    ('PVwSTO', '8.PV'): ('PVwSTO', 'Solar'),
    ('STOPV', '9.BESS'): ('STOPV', 'Battery'),
    ('Storage', '9.BESS'): ('Storage', 'Battery'),
    ('BIOMAS', '10.Biomass'): ('BiomassPlant', 'Biomass'),
    ('NUC', '11.Uranium'): ('Nuclear', 'Uranium'),
    ('GEO', '12.Geothermal'): ('ST', 'Geothermal'),
}

# Legacy fuel -> current fuel (fuel-level outputs: pFuelCosts, pFuelConsumption)
LEGACY_FUEL_MAP = {
    '1.Coal': 'Coal', '2.CoalCCS': 'Coal', '3.NaturalGas': 'Gas', '4.NaturalGasCCS': 'Gas',
    '5.HydroStor': 'Water', '6.RoR': 'Water', '7.Wind': 'Wind', '8.PV': 'Solar',
    '9.BESS': 'Battery', '10.Biomass': 'Biomass', '11.Uranium': 'Uranium',
    '12.Geothermal': 'Geothermal',
}

# Generator name token -> plausible legacy technologies (in order of preference)
NAME_TOKEN_TECHS = {
    'BESS': ['Storage', 'STOPV'],
    'PV': ['PV', 'PVwSTO', 'STOPV'],
    'WPP': ['Wind'],
    'HPP': ['STO HY', 'ROR', 'Storage'],
    'BIO': ['BIOMAS', 'BIOGAS'],
    'GEO': ['GEO'],
    'NPP': ['NUC'],
    'CSP': ['CSP'],
    'TPP': ['Coal', 'CCGT', 'GT', 'ICE', 'ST', 'CHP', 'CoalCCS', 'GasCCS', 'BIOMAS'],
}

# Legacy pCostSummary labels -> current sumhdr labels (None = not a cost component)
COST_LABEL_MAP = {
    'Annualized capex: $m': 'Investment costs: $m',
    'Fixed O&M: $m': 'Fixed O&M: $m',
    'Variable O&M: $m': 'Variable O&M: $m',
    'Total fuel Costs: $m': 'Fuel costs: $m',
    'Transmission additions: $m': 'Transmission costs: $m',
    'Spinning Reserve costs: $m': 'Spinning reserve costs: $m',
    'Unmet demand costs: $m': 'Unmet demand costs: $m',
    'Excess generation: $m': 'Excess generation: $m',
    'VRE curtailment: $m': 'VRE curtailment: $m',
    'Import costs: $m': 'Import costs with external zones: $m',
    'Export revenue: $m': 'Export revenues with external zones: $m',
    'Trade shared benefits: $m': 'Trade shared benefits: $m',
    'Carbon costs: $m': 'Carbon costs: $m',
    'Climate backstop cost: $m': 'Unmet country CO2 backstop cost: $m',
    'Capex: $m': None,
    'Total Annual Cost by Zone: $m': None,
}

PLANT_COST_LABEL_MAP = {
    'Annuity Cost in MUSD': 'Investment costs: $m',
    'Fixed Cost in MUSD': 'Fixed O&M: $m',
    'Variable Cost in MUSD': 'Variable Cost: $m',
}

SOLVER_LABEL_MAP = {
    'Solver Status': 'modeltype Status',
    'Solver Time: ms': 'modeltype Time: ms',
}

STORAGE_LABEL_MAP = {
    'Storage Capacity in MW': 'Storage Capacity: MW',
    'Storage Capacity MWh': 'Storage Capacity: MWh',
}

# Legacy pScalars -> current pSettings abbreviations
SETTINGS_MAP = {
    'wACC': 'WACC', 'dr': 'DR', 'vOLL': 'VoLL', 'reserveVoLL': 'ReserveVoLL',
    'spinReserveVoLL': 'SpinReserveVoLL', 'costSurplus': 'CostSurplus',
    'costcurtail': 'CostCurtail', 'CO2backstop': 'CO2backstop', 'H2UnservedCost': 'H2UnservedCost',
}

# Symbols exported as CSV by generate_report.gms (embeddedCode Connect)
CSV_SYMBOLS = [
    "pGeneratorTechFuel", "pZoneCountry", "pDemandEnergyZone", "pDemandPeakZone",
    "pCapacityPlant", "pNewCapacityPlant", "pNewCapacityTechFuel", "pCapacityTechFuel",
    "pNewTransmissionCapacity", "pTransmissionCapacity",
    "pCostsPlant", "pCapexInvestmentComponent",
    "pHourlyPrice", "pCosts", "pDiscountedWeightedCosts", "pNetPresentCostSystem",
    "pNetPresentCostSystemPerMWh", "pFuelCosts",
    "pEnergyPlant", "pEnergyTechFuel", "pEnergyTechFuelComplete", "pEnergyBalance",
    "pUtilizationPlant", "pUtilizationTechFuel", "pReserveSpinningPlantZone",
    "pInterchange", "pInterconUtilization", "pCongestionShare", "pNetImport",
    "pEmissionsZone", "pEmissionsIntensityZone",
    "pDispatchTechFuel", "pDispatch",
    "pPlantAnnualLCOE", "pGenCostsPerMWh", "pNetPresentCostPerMWh", "pCostsPerMWh",
    "pSettings",
]

NPV_LABEL = 'NPV of system cost: $m'
SYSTEM_VIOLATIONS = {
    'Sys Plan Reserve violation: $m': 'Unmet system planning reserve costs: $m',
    'Sys Spin Reserve violation: $m': 'Unmet system spinning reserve costs: $m',
}


def _default_log(msg: str) -> None:
    print(msg)


# =============================================================================
# READING LEGACY GDX
# =============================================================================

def _get(c: gt.Container, name: str, cols: List[str]) -> pd.DataFrame:
    """Return legacy symbol records with positional column names, set columns as str."""
    if name not in c.data or c[name].records is None or len(c[name].records) == 0:
        return pd.DataFrame(columns=cols)
    df = c[name].records.copy()
    df.columns = cols
    for col in cols[:-1]:
        df[col] = df[col].astype(str)
    df['value'] = df['value'].astype(float)
    df = df[df['value'].abs() > LEGACY_EPSILON]
    return df.reset_index(drop=True)


def _label(df: pd.DataFrame, label: str, col: str = 'uni') -> pd.DataFrame:
    """Rows of a labelled legacy parameter, without the label column."""
    return df[df[col] == label].drop(columns=col)


def _sorted_years(years) -> List[str]:
    return sorted(set(years), key=int)


# =============================================================================
# RECONSTRUCTED INFORMATION
# =============================================================================

def infer_hours(c: gt.Container, log_func: Callable = _default_log) -> pd.DataFrame:
    """
    Recover pHours as a (y, q, d, t) weight table.

    Legacy pDemandSupplySeason (GWh per season) divided by the sum of hourly demand
    (pDemandTotalZ, MW) gives the weight of each block. Weights are year dependent
    in legacy runs (leap years). Requires one representative day per season;
    otherwise falls back to a least-squares fit on plant energy.
    """
    dz = _get(c, 'pDemandTotalZ', ['z', 'q', 'd', 't', 'y', 'value'])
    season = _label(_get(c, 'pDemandSupplySeason', ['z', 'uni', 'y', 'q', 'value']), 'Demand: GWh')
    blocks = dz[['y', 'q', 'd', 't']].drop_duplicates()

    if not season.empty and dz.groupby('q')['d'].nunique().max() == 1:
        hourly = dz.groupby(['z', 'y', 'q'])['value'].sum()
        ratio = season.set_index(['z', 'y', 'q'])['value'] * 1e3 / hourly
        ratio = ratio.groupby(['y', 'q']).mean().rename('w').reset_index()
        hours = blocks.merge(ratio, on=['y', 'q'], how='left')
        method = 'seasonal demand ratio'
    else:
        from scipy.optimize import nnls
        gen = _label(_get(c, 'pPlantDispatch', ['z', 'y', 'q', 'd', 'g', 't', 'uni', 'value']), 'Generation')
        energy = _get(c, 'pEnergyByPlant', ['z', 'g', 'y', 'value']).set_index(['g', 'y'])['value']
        rows = []
        for y, gy in gen.groupby('y'):
            a = gy.pivot_table(index='g', columns=['q', 'd'], values='value', aggfunc='sum', fill_value=0)
            b = energy.xs(y, level='y').reindex(a.index).fillna(0).values * 1e3
            w, _ = nnls(a.values, b)
            rows += [(y, q, d, wi) for (q, d), wi in zip(a.columns, w)]
        hours = blocks.merge(pd.DataFrame(rows, columns=['y', 'q', 'd', 'w']), on=['y', 'q', 'd'], how='left')
        method = 'least squares on plant energy'

    hours['w'] = hours['w'].fillna(0)

    # Check against plant energy
    gen = _label(_get(c, 'pPlantDispatch', ['z', 'y', 'q', 'd', 'g', 't', 'uni', 'value']), 'Generation')
    energy = _get(c, 'pEnergyByPlant', ['z', 'g', 'y', 'value']).groupby(['g', 'y'])['value'].sum()
    if not gen.empty and not energy.empty:
        gen = gen.merge(hours, on=['y', 'q', 'd', 't'])
        rebuilt = (gen['value'] * gen['w']).groupby([gen['g'], gen['y']]).sum() / 1e3
        err = (rebuilt.reindex(energy.index).fillna(0) - energy).abs().max() / max(energy.abs().max(), 1)
        log_func(f"[legacy]   pHours from {method}: {hours.groupby('y')['w'].sum().mean():.0f} h/year, "
                 f"max relative error on plant energy {err:.1e}")
        if err > 1e-3:
            log_func("[legacy]   WARNING: pHours reconstruction is imprecise - energy-weighted results are approximate")
    return hours


def year_weights(years: List[str], dr: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """pWeightYear and pRR, as in main.gms."""
    weight, rr = {}, {}
    for i, y in enumerate(years):
        weight[y] = 1.0 if i == 0 else float(int(y) - int(years[i - 1]))
    cumulated = 0.0
    for i, y in enumerate(years):
        rr[y] = 1.0 if i == 0 else 1 / (1 + dr) ** (cumulated - 1 + weight[y] / 2)
        cumulated += weight[y]
    return weight, rr


def _name_techs(generator: str) -> Optional[List[str]]:
    """Plausible legacy technologies from the generator name (last matching token)."""
    for token in reversed(generator.upper().split('_')):
        if token in NAME_TOKEN_TECHS:
            return NAME_TOKEN_TECHS[token]
    return None


def infer_generator_techfuel(c: gt.Container, overrides: Optional[pd.DataFrame] = None,
                             log_func: Callable = _default_log) -> pd.DataFrame:
    """
    Infer the legacy (tech, fuel) of each generator.

    Candidates come from the generator name; the assignment is then searched so
    that, in each zone, plant capacity and energy trajectories add up to the
    legacy pCapacityByTechandFuel and pEnergyByTechandFuel.

    Returns a DataFrame with columns z, g, tech_legacy, f_legacy, method.
    """
    cap = _get(c, 'pCapacityPlan', ['z', 'g', 'y', 'value'])
    energy = _get(c, 'pEnergyByPlant', ['z', 'g', 'y', 'value'])
    cap_tf = _get(c, 'pCapacityByTechandFuel', ['z', 'tech', 'f', 'y', 'value'])
    energy_tf = _get(c, 'pEnergyByTechandFuel', ['z', 'tech', 'f', 'y', 'value'])
    years = _sorted_years(cap['y'])

    plants = pd.concat([cap[['z', 'g']], energy[['z', 'g']]]).drop_duplicates()
    override = {}
    if overrides is not None:
        override = {r.g: (r.tech, r.f) for r in overrides.itertuples()}

    def vectors(df, keys):
        p = df.pivot_table(index=keys, columns='y', values='value', aggfunc='sum', fill_value=0)
        return p.reindex(columns=years, fill_value=0)

    cap_p, en_p = vectors(cap, ['z', 'g']), vectors(energy, ['z', 'g'])
    cap_t, en_t = vectors(cap_tf, ['z', 'tech', 'f']), vectors(energy_tf, ['z', 'tech', 'f'])

    result = []
    for z, zone_plants in plants.groupby('z'):
        def plant_vec(g):
            a = cap_p.loc[(z, g)].values if (z, g) in cap_p.index else np.zeros(len(years))
            b = en_p.loc[(z, g)].values if (z, g) in en_p.index else np.zeros(len(years))
            return np.concatenate([a, b])

        pairs = sorted(set(cap_t.loc[z].index if z in cap_t.index.get_level_values(0) else [])
                       | set(en_t.loc[z].index if z in en_t.index.get_level_values(0) else []))
        target = {}
        for pair in pairs:
            a = cap_t.loc[(z, *pair)].values if (z, *pair) in cap_t.index else np.zeros(len(years))
            b = en_t.loc[(z, *pair)].values if (z, *pair) in en_t.index else np.zeros(len(years))
            target[pair] = np.concatenate([a, b])
        tol = {k: 1e-3 + 1e-6 * np.abs(v) for k, v in target.items()}

        vec, cands, method = {}, {}, {}
        for g in zone_plants['g']:
            vec[g] = plant_vec(g)
            if g in override:
                cands[g], method[g] = [override[g]], 'override'
                continue
            techs = _name_techs(g)
            cand = [p for t in (techs or []) for p in pairs if p[0] == t]
            if not cand:
                cand, method[g] = list(pairs), 'aggregates only'
            else:
                method[g] = 'name' if len(cand) == 1 else 'name + aggregates'
            cands[g] = cand

        # Plants never built/dispatched are not constrained by the aggregates
        active = [g for g in vec if np.abs(vec[g]).max() > 1e-6]
        inactive = [g for g in vec if g not in active]
        order = sorted(active, key=lambda g: (len(cands[g]), -np.abs(vec[g]).sum()))
        resid = {k: v.copy() for k, v in target.items()}
        assign, budget = {}, [2_000_000]

        def dfs(i):
            budget[0] -= 1
            if budget[0] < 0:
                return False
            if i == len(order):
                return all(np.all(np.abs(resid[k]) <= tol[k]) for k in resid)
            g = order[i]
            for k in cands[g]:
                if k not in resid:
                    continue
                r = resid[k] - vec[g]
                if np.any(r < -tol[k]):
                    continue
                resid[k], assign[g] = r, k
                if dfs(i + 1):
                    return True
                resid[k] = r + vec[g]
            return False

        if not dfs(0):
            log_func(f"[legacy]   WARNING: zone {z}: no exact generator assignment matches the tech-fuel "
                     f"aggregates - using name-based guess. Check legacy_generator_techfuel.csv "
                     f"and provide --generator-map if needed.")
            assign = {g: cands[g][0] for g in active if cands[g]}
            for g in active:
                method[g] += ' (UNVERIFIED)'
        for g in inactive:
            if cands[g]:
                assign[g] = cands[g][0]
            else:
                techs = _name_techs(g) or []
                fuel = next((lf for (lt, lf) in LEGACY_TECHFUEL_MAP if lt in techs), None)
                assign[g] = (techs[0], fuel) if techs and fuel else ('Unknown', 'Unknown')
            method[g] += ' (not built)'

        for g, (tech, f) in assign.items():
            result.append((z, g, tech, f, method[g]))

    df = pd.DataFrame(result, columns=['z', 'g', 'tech_legacy', 'f_legacy', 'method'])
    n_ambiguous = df['method'].str.contains('aggregates').sum()
    log_func(f"[legacy]   Generator tech-fuel inferred for {len(df)} plants "
             f"({n_ambiguous} resolved with tech-fuel aggregates)")
    return df


def map_techfuel(gmap: pd.DataFrame, techfuel_map: Dict) -> pd.DataFrame:
    """Add current tech / f columns from the legacy names."""
    missing = sorted({(t, f) for t, f in zip(gmap['tech_legacy'], gmap['f_legacy'])} - set(techfuel_map))
    if missing:
        raise ValueError(f"Legacy tech-fuel pairs without a current equivalent: {missing}. "
                         f"Add them to LEGACY_TECHFUEL_MAP or pass --techfuel-map.")
    new = [techfuel_map[(t, f)] for t, f in zip(gmap['tech_legacy'], gmap['f_legacy'])]
    gmap = gmap.copy()
    gmap['tech'] = [t for t, _ in new]
    gmap['f'] = [f for _, f in new]
    return gmap


# =============================================================================
# CONVERSION TO CURRENT FORMAT
# =============================================================================

def _cols(domain: List[str]) -> List[str]:
    """Column names gams.transfer/Connect give to a domain."""
    names = ['uni' if d == '*' else d for d in domain]
    if len(set(names)) < len(names):
        names = [f"{n}_{i}" for i, n in enumerate(names)]
    return names + ['value']


def _group(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    return df.groupby(keys, as_index=False, observed=True)['value'].sum()


def _diff_positive(df: pd.DataFrame, keys: List[str], years: List[str]) -> pd.DataFrame:
    """Positive year-on-year increase of a (keys, y) trajectory; 0 in the first year."""
    p = df.pivot_table(index=keys, columns='y', values='value', aggfunc='sum', fill_value=0)
    p = p.reindex(columns=years, fill_value=0)
    d = p.diff(axis=1).clip(lower=0)
    d[years[0]] = 0
    return d.stack().rename('value').reset_index()


def convert_legacy_gdx(gdx_path: str, generator_map: Optional[pd.DataFrame] = None,
                       techfuel_map: Optional[Dict] = None,
                       log_func: Callable = _default_log) -> Tuple[Dict[str, Tuple[List[str], pd.DataFrame]], pd.DataFrame]:
    """
    Rebuild current-format result symbols from a legacy epmresults.gdx.

    Returns
    -------
    out : dict
        name -> (domain, records) for each current-format symbol
    gmap : pd.DataFrame
        Generator mapping (z, g, tech_legacy, f_legacy, tech, f, method)
    """
    techfuel_map = techfuel_map or LEGACY_TECHFUEL_MAP
    c = gt.Container(str(gdx_path))
    out: Dict[str, Tuple[List[str], pd.DataFrame]] = {}

    def put(name, domain, df, cols=None):
        """Store a symbol; df columns in domain order then 'value'."""
        cols = cols or [('uni' if d == '*' else d) for d in domain]
        df = df[cols + ['value']].copy()
        df = df[df['value'].notna() & (df['value'] != 0)]
        df.columns = _cols(domain)
        out[name] = (domain, df.reset_index(drop=True))

    # ---------------------------------------------------------
    # Mappings, years, hours and discounting
    # ---------------------------------------------------------
    cap_plant = _get(c, 'pCapacityPlan', ['z', 'g', 'y', 'value'])
    years = _sorted_years(cap_plant['y'])

    cap_country = _get(c, 'pCapacityPlanCountry', ['c', 'g', 'y', 'value'])
    zc = cap_plant[['z', 'g']].drop_duplicates().merge(cap_country[['c', 'g']].drop_duplicates(), on='g', how='left')
    zc['c'] = zc['c'].fillna(zc['z'])
    zc = zc[['z', 'c']].drop_duplicates()
    zones = sorted(zc['z'].unique())
    zone_country = zc.assign(value=1.0)
    put('pZoneCountry', ['z', 'c'], zone_country)

    gmap = map_techfuel(infer_generator_techfuel(c, generator_map, log_func), techfuel_map)
    gtf = gmap[['g', 'tech', 'f']].drop_duplicates()
    put('pGeneratorTechFuel', ['g', 'tech', 'f'], gtf.assign(value=1.0))

    hours = infer_hours(c, log_func)
    hours_year = hours.groupby('y')['w'].sum()

    scalars = _get(c, 'pScalars', ['sc', 'value']).set_index('sc')['value']
    dr = float(scalars.get('dr', 0.0))
    weight, rr = year_weights(years, dr)
    log_func(f"[legacy]   Discount rate {dr:.3f}, {len(years)} years ({years[0]}-{years[-1]})")
    put('pWeightYear', ['y'], pd.DataFrame({'y': years, 'value': [weight[y] for y in years]}))
    put('pRR', ['y'], pd.DataFrame({'y': years, 'value': [rr[y] for y in years]}))

    settings = pd.DataFrame([(SETTINGS_MAP[k], v) for k, v in scalars.items() if k in SETTINGS_MAP],
                            columns=['pSettingsHeader', 'value'])
    put('pSettings', ['pSettingsHeader'], settings)

    def with_tech(df):
        return df.merge(gtf, on='g', how='left')

    # ---------------------------------------------------------
    # 1. Capacity
    # ---------------------------------------------------------
    put('pCapacityPlant', ['z', 'g', 'y'], cap_plant)
    new_plant = _diff_positive(cap_plant, ['z', 'g'], years)
    put('pNewCapacityPlant', ['z', 'g', 'y'], new_plant)

    cap_tf = _group(with_tech(cap_plant), ['z', 'tech', 'f', 'y'])
    put('pCapacityTechFuel', ['z', 'tech', 'f', 'y'], cap_tf)
    put('pNewCapacityTechFuel', ['z', 'tech', 'f', 'y'], _group(with_tech(new_plant), ['z', 'tech', 'f', 'y']))

    storage = _get(c, 'pStorageComponents', ['g', 'uni', 'y', 'value'])
    storage['uni'] = storage['uni'].replace(STORAGE_LABEL_MAP)
    put('pStorageComponents', ['g', '*', 'y'], storage)

    # Transmission capacity: interchange / utilization over the year
    interchange = _get(c, 'pInterchange', ['z', 'z2', 'y', 'value'])
    utilization = _get(c, 'pInterconUtilization', ['z', 'z2', 'y', 'value'])
    tx = interchange.merge(utilization, on=['z', 'z2', 'y'], suffixes=('_e', '_u'))
    tx['value'] = (tx['value_e'] * 1e3 / tx['value_u'] / tx['y'].map(hours_year)).round(3)
    tx_pairs = pd.concat([interchange[['z', 'z2']],
                          _get(c, 'pAdditionalCapacity', ['z', 'z2', 'uni', 'y', 'value'])[['z', 'z2']]]).drop_duplicates()
    if not tx_pairs.empty:
        tx = tx.pivot_table(index=['z', 'z2'], columns='y', values='value', aggfunc='mean')
        tx = tx.reindex(index=pd.MultiIndex.from_frame(tx_pairs), columns=years).ffill(axis=1).bfill(axis=1)
        fallback = _label(_get(c, 'pAdditionalCapacity', ['z', 'z2', 'uni', 'y', 'value']), 'Total New TX Capacity')
        if not fallback.empty:
            fb = fallback.pivot_table(index=['z', 'z2'], columns='y', values='value', aggfunc='sum')
            tx = tx.fillna(fb.reindex(index=tx.index, columns=years))
        tx = tx.stack().rename('value').reset_index()
    else:
        tx = pd.DataFrame(columns=['z', 'z2', 'y', 'value'])
    put('pTransmissionCapacity', ['z', 'z2', 'y'], tx)
    put('pNewTransmissionCapacity', ['z', 'z2', 'y'], _diff_positive(tx, ['z', 'z2'], years) if not tx.empty else tx)

    # ---------------------------------------------------------
    # 3. Energy balance (needed by costs per MWh)
    # ---------------------------------------------------------
    energy_plant = _get(c, 'pEnergyByPlant', ['z', 'g', 'y', 'value'])
    put('pEnergyPlant', ['z', 'g', 'y'], energy_plant)
    energy_tf = _group(with_tech(energy_plant), ['z', 'tech', 'f', 'y'])
    put('pEnergyTechFuel', ['z', 'tech', 'f', 'y'], energy_tf)

    def hourly_to_gwh(name):
        df = _get(c, name, ['z', 'q', 'd', 't', 'y', 'value'])
        df = df.merge(hours, on=['y', 'q', 'd', 't'])
        df['value'] = df['value'] * df['w'] / 1e3
        return _group(df, ['z', 'y'])

    unmet = hourly_to_gwh('pUnmetZ')
    curtailed = hourly_to_gwh('pCurtailedVREZ')
    supply = _get(c, 'pDemandSupply', ['z', 'uni', 'y', 'value'])
    imports_int = _group(interchange.rename(columns={'z': 'from', 'z2': 'z'}), ['z', 'y'])
    exports_int = _group(interchange, ['z', 'y'])

    def label(name, df=supply):
        return _label(df, name)

    def ext(total, internal):
        m = label(total).merge(internal, on=['z', 'y'], how='left', suffixes=('', '_int')).fillna(0)
        m['value'] = (m['value'] - m['value_int']).clip(lower=0)
        return m[m['value'] > 1e-6][['z', 'y', 'value']]

    imports_ext = ext('Imports exchange: GWh', imports_int)
    exports_ext = ext('Exports exchange: GWh', exports_int)

    total_demand = label('Total Demand (Including P to H2): GWh')
    balance = pd.concat([
        label('Demand: GWh').assign(uni='Demand: GWh'),
        label('Electricity demand for H2 production: GWh').assign(uni='Demand H2: GWh'),
        (total_demand if not total_demand.empty else label('Demand: GWh')).assign(uni='Total Demand: GWh'),
        label('Total production: GWh').assign(uni='Total production: GWh'),
        curtailed.assign(uni='Total Curtailed VRE: GWh'),
        unmet.assign(uni='Unmet demand: GWh'),
        label('Surplus generation: GWh').assign(uni='Surplus generation: GWh'),
        label('Imports exchange: GWh').assign(uni='Imports exchange: GWh'),
        label('Exports exchange: GWh').assign(uni='Exports exchange: GWh'),
        label('Net interchange: GWh').assign(uni='Net interchange: GWh'),
        label('Net interchange Ratio: GWh').assign(uni='Net interchange Ratio: GWh'),
    ])
    put('pEnergyBalance', ['z', '*', 'y'], balance)

    complete = pd.concat([
        energy_tf,
        unmet.assign(tech='EnergyBalance', f='UnmetDemand'),
        label('Surplus generation: GWh').assign(tech='EnergyBalance', f='Surplus'),
        imports_int.assign(tech='EnergyBalance', f='Imports'),
        imports_ext.assign(tech='EnergyBalance', f='Imports External'),
        exports_int.assign(tech='EnergyBalance', f='Exports', value=-exports_int['value']),
        exports_ext.assign(tech='EnergyBalance', f='Exports External', value=-exports_ext['value']),
    ])
    put('pEnergyTechFuelComplete', ['z', '*', '*', 'y'], complete, cols=['z', 'tech', 'f', 'y'])

    put('pUtilizationPlant', ['z', 'g', 'y'], _get(c, 'pPlantUtilization', ['z', 'g', 'y', 'value']))
    util_tf = cap_tf.merge(energy_tf, on=['z', 'tech', 'f', 'y'], suffixes=('_c', '_e'))
    util_tf['value'] = util_tf['value_e'] / (util_tf['value_c'] * util_tf['y'].map(hours_year)) * 1000
    put('pUtilizationTechFuel', ['z', 'tech', 'f', 'y'], util_tf)

    demand = label('Demand: GWh')
    put('pDemandEnergyZone', ['z', 'y'], demand)
    reserve_margin = _get(c, 'pReserveMarginRes', ['z', 'uni', 'y', 'value'])
    put('pDemandPeakZone', ['z', 'y'], _label(reserve_margin, 'Peak demand: MW'))

    # ---------------------------------------------------------
    # 2. Costs
    # ---------------------------------------------------------
    costs_plant = _get(c, 'pCostsbyPlant', ['z', 'g', 'uni', 'y', 'value'])
    costs_plant = costs_plant[costs_plant['uni'].isin(PLANT_COST_LABEL_MAP)]
    costs_plant['uni'] = costs_plant['uni'].map(PLANT_COST_LABEL_MAP)
    put('pCostsPlant', ['z', 'g', '*', 'y'], costs_plant)

    # CAPEX: zonal legacy 'Capex: $m' split pro rata of annuity increase in the build year
    cost_summary = _get(c, 'pCostSummary', ['z', 'uni', 'y', 'value'])
    capex_zone = _label(cost_summary, 'Capex: $m').set_index(['z', 'y'])['value'] * 1e6
    annuity = _label(costs_plant, 'Investment costs: $m')
    if not annuity.empty:
        ann = annuity.pivot_table(index=['z', 'g'], columns='y', values='value', aggfunc='sum', fill_value=0)
        ann = ann.reindex(columns=years, fill_value=0)
        increase = ann.diff(axis=1).clip(lower=0)
        increase[years[0]] = ann[years[0]]
        increase = increase.stack().rename('inc').reset_index()
        increase = increase[increase['inc'] > 1e-9]
        increase['share'] = increase['inc'] / increase.groupby(['z', 'y'])['inc'].transform('sum')
        increase['value'] = increase['share'] * increase.set_index(['z', 'y']).index.map(capex_zone).fillna(0).values
    else:
        increase = pd.DataFrame(columns=['z', 'g', 'y', 'value'])
    capex_plant = with_tech(increase[['z', 'g', 'y', 'value']])
    capex_plant['uni'] = np.where(capex_plant['tech'] == 'Storage', 'StorageEnergy', 'Generation')
    put('pCapexInvestmentPlant', ['z', 'g', '*', 'y'], capex_plant)
    capex_comp = _group(capex_plant, ['z', 'uni', 'y'])
    unallocated = capex_zone.sub(capex_comp.groupby(['z', 'y'])['value'].sum(), fill_value=0)
    unallocated = unallocated[unallocated > 1].rename('value').reset_index().assign(uni='Generation')
    capex_comp = _group(pd.concat([capex_comp, unallocated]), ['z', 'uni', 'y'])
    put('pCapexInvestmentComponent', ['z', '*', 'y'], capex_comp)

    unknown = sorted(set(cost_summary['uni']) - set(COST_LABEL_MAP))
    if unknown:
        log_func(f"[legacy]   WARNING: legacy cost labels kept as is: {unknown}")
    costs = cost_summary[cost_summary['uni'].map(lambda u: COST_LABEL_MAP.get(u, u)).notna()].copy()
    costs['uni'] = costs['uni'].map(lambda u: COST_LABEL_MAP.get(u, u))
    costs = _group(costs, ['z', 'uni', 'y'])
    put('pCosts', ['z', '*', 'y'], costs)

    costs['w'] = costs['y'].map(lambda y: rr[y] * weight[y])
    discounted = costs.assign(value=costs['value'] * costs['w'])
    put('pDiscountedWeightedCosts', ['z', '*', 'y'], discounted)
    npv_zone = _group(discounted, ['z', 'uni'])
    put('pCostsSystem', ['sumhdr', 'y'], _group(costs, ['uni', 'y']), cols=['uni', 'y'])

    demand_mwh = demand.assign(value=demand['value'] * 1e3 * demand['y'].map(lambda y: rr[y] * weight[y]))
    disc_demand_zone = demand_mwh.groupby('z')['value'].sum()
    disc_demand_system = disc_demand_zone.sum()
    put('pDiscountedDemandZoneMWh', ['z'], disc_demand_zone.reset_index())

    # System reserve violations only exist as totals: reconcile with the legacy NPV
    summary = _get(c, 'pSummary', ['uni', 'value']).set_index('uni')['value']
    npv = summary.get(NPV_LABEL)
    if npv is not None:
        residual = npv - npv_zone['value'].sum()
        violations = {SYSTEM_VIOLATIONS[k]: summary[k] for k in SYSTEM_VIOLATIONS if k in summary}
        total_violation = sum(violations.values())
        if abs(residual) > 1e-6 * abs(npv) and total_violation > 0:
            share_zone = disc_demand_zone / disc_demand_system
            extra = pd.DataFrame([(z, lab, residual * v / total_violation * share_zone[z])
                                  for lab, v in violations.items() for z in share_zone.index],
                                 columns=['z', 'uni', 'value'])
            npv_zone = _group(pd.concat([npv_zone, extra]), ['z', 'uni'])
            log_func(f"[legacy]   NPV {npv:,.1f} $m = discounted costs {npv - residual:,.1f} + "
                     f"system reserve violations {residual:,.1f} (allocated to NPV outputs only)")
        elif abs(residual) > 1e-3 * abs(npv):
            log_func(f"[legacy]   WARNING: discounted costs differ from legacy NPV by {residual:,.1f} $m")
    put('pNetPresentCost', ['z', '*'], npv_zone)

    npv_system = _group(npv_zone, ['uni'])
    if npv is not None:
        npv_system = pd.concat([npv_system, pd.DataFrame({'uni': [NPV_LABEL], 'value': [npv]})])
    put('pNetPresentCostSystem', ['*'], npv_system)
    if disc_demand_system:
        put('pDiscountedDemandSystemMWh', [], pd.DataFrame({'value': [disc_demand_system]}))
        put('pNetPresentCostSystemPerMWh', ['*'],
            npv_system.assign(value=npv_system['value'] * 1e6 / disc_demand_system))
    npv_mwh = npv_zone.assign(value=npv_zone['value'] * 1e6 / npv_zone['z'].map(disc_demand_zone))
    put('pNetPresentCostPerMWh', ['z', 'sumhdr'], npv_mwh, cols=['z', 'uni'])

    per_mwh = costs.merge(demand, on=['z', 'y'], suffixes=('', '_d'))
    per_mwh['value'] = per_mwh['value'] * 1e6 / (per_mwh['value_d'] * 1e3)
    put('pCostsPerMWh', ['z', 'sumhdr', 'y'], per_mwh, cols=['z', 'uni', 'y'])
    system_mwh = _group(costs, ['uni', 'y']).merge(_group(demand, ['y']), on='y', suffixes=('', '_d'))
    system_mwh['value'] = system_mwh['value'] * 1e6 / (system_mwh['value_d'] * 1e3)
    put('pCostsSystemPerMWh', ['sumhdr', 'y'], system_mwh, cols=['uni', 'y'])

    gen_costs = _group(costs_plant[costs_plant['uni'].isin(['Investment costs: $m', 'Fixed O&M: $m', 'Variable Cost: $m'])],
                       ['z', 'uni', 'y'])
    gen_costs = gen_costs.merge(_group(energy_plant, ['z', 'y']), on=['z', 'y'], suffixes=('', '_e'))
    gen_costs['value'] = 1e6 * gen_costs['value'] / (gen_costs['value_e'] * 1e3)
    put('pGenCostsPerMWh', ['z', 'genCostCmp', 'y'], gen_costs, cols=['z', 'uni', 'y'])

    lcoe = _group(costs_plant, ['z', 'g', 'y']).merge(energy_plant, on=['z', 'g', 'y'], suffixes=('', '_e'))
    lcoe = lcoe[lcoe['value_e'] * 1e3 >= 1e3]
    lcoe['value'] = 1e6 * lcoe['value'] / (lcoe['value_e'] * 1e3)
    put('pPlantAnnualLCOE', ['z', 'g', 'y'], lcoe)

    for name in ['pFuelCosts', 'pFuelConsumption']:
        df = _get(c, name, ['z', 'f', 'y', 'value'])
        df['f'] = df['f'].map(lambda f: LEGACY_FUEL_MAP.get(f, f))
        put(name, ['z', 'f', 'y'], _group(df, ['z', 'f', 'y']))

    put('pHourlyPrice', ['z', 'q', 'd', 't', 'y'], _get(c, 'pPrice', ['z', 'q', 'd', 't', 'y', 'value']))
    put('pPrice', ['z', 'y'], _get(c, 'pAveragePrice', ['z', 'y', 'value']))

    # ---------------------------------------------------------
    # 4. Dispatch
    # ---------------------------------------------------------
    dispatch_plant = _get(c, 'pPlantDispatch', ['z', 'y', 'q', 'd', 'g', 't', 'uni', 'value'])
    put('pDispatchPlant', ['z', 'y', 'q', 'd', 'g', 't', '*'], dispatch_plant)
    generation = with_tech(_label(dispatch_plant, 'Generation'))
    put('pDispatchTechFuel', ['z', 'y', 'q', 'd', 'tech', 'f', 't'],
        _group(generation, ['z', 'y', 'q', 'd', 'tech', 'f', 't']))

    dispatch = _get(c, 'pDispatch', ['z', 'y', 'q', 'd', 'uni', 't', 'value'])
    dispatch = dispatch[dispatch['uni'] != 'Generation']
    if 'Unmet demand' not in set(dispatch['uni']):
        unmet_hourly = _get(c, 'pUnmetZ', ['z', 'q', 'd', 't', 'y', 'value']).assign(uni='Unmet demand')
        dispatch = pd.concat([dispatch, unmet_hourly])
    put('pDispatch', ['z', 'y', 'q', 'd', '*', 't'], dispatch)

    # ---------------------------------------------------------
    # 5. Reserves
    # ---------------------------------------------------------
    reserve = _label(dispatch_plant, 'Reserve').merge(hours, on=['y', 'q', 'd', 't'])
    reserve['value'] = reserve['value'] * reserve['w'] / 1e3
    put('pReserveSpinningPlantZone', ['z', 'g', 'y'], _group(reserve, ['z', 'g', 'y']))
    put('pReserveMargin', ['z', '*', 'y'], reserve_margin)

    # ---------------------------------------------------------
    # 6. Interconnections
    # ---------------------------------------------------------
    put('pInterchange', ['z', 'z2', 'y'], interchange)
    put('pInterconUtilization', ['z', 'z2', 'y'], utilization)

    flows = _get(c, 'pInterchangeH', ['z', 'z2', 'q', 'd', 't', 'y', 'value'])
    if not tx.empty:
        congestion = tx[['z', 'z2', 'y']].merge(hours, on='y')
        congestion = congestion.merge(flows, on=['z', 'z2', 'q', 'd', 't', 'y'], how='left').fillna({'value': 0})
        congestion = congestion.merge(tx, on=['z', 'z2', 'y'], suffixes=('', '_cap'))
        congested = (congestion['value'] >= congestion['value_cap'] - 1e-3) & (congestion['value_cap'] > 0)
        congestion['value'] = congested * congestion['w'] / congestion['y'].map(hours_year)
        put('pCongestionShare', ['z', 'z2', 'y'], _group(congestion, ['z', 'z2', 'y']))

    # Same convention as generate_report.gms (internal zones only in legacy runs)
    net_import = interchange.assign(value=-interchange['value'] / 1e3, isExternal='Internal')
    put('pNetImport', ['z', '*', 'y', 'isExternal'], net_import, cols=['z', 'z2', 'y', 'isExternal'])

    # ---------------------------------------------------------
    # 7. Emissions
    # ---------------------------------------------------------
    emissions = _label(_get(c, 'pEmissions', ['z', 'uni', 'y', 'value']), 'Emissions: mm tCO2eq')
    put('pEmissionsZone', ['z', 'y'], emissions)
    intensity = emissions.merge(label('Total production: GWh'), on=['z', 'y'], suffixes=('', '_p'))
    intensity['value'] = 1e-3 * intensity['value'] / intensity['value_p']
    put('pEmissionsIntensityZone', ['z', 'y'], intensity)

    # ---------------------------------------------------------
    # 11. Solver parameters
    # ---------------------------------------------------------
    solver = _get(c, 'pSolverParameters', ['uni', 'value'])
    solver['uni'] = solver['uni'].replace(SOLVER_LABEL_MAP)
    put('pSolverParameters', ['*'], solver)

    # ---------------------------------------------------------
    # Consistency checks against legacy aggregates
    # ---------------------------------------------------------
    def check(name, rebuilt, legacy):
        a, b = rebuilt['value'].sum(), legacy['value'].sum()
        flag = 'OK' if abs(a - b) <= 1e-4 * max(abs(b), 1) else 'MISMATCH'
        log_func(f"[legacy]   check {name:<28} rebuilt {a:>14,.1f}  legacy {b:>14,.1f}  {flag}")

    check('capacity by tech-fuel', cap_tf, _get(c, 'pCapacityByTechandFuel', ['z', 'tech', 'f', 'y', 'value']))
    check('energy by tech-fuel', energy_tf, _get(c, 'pEnergyByTechandFuel', ['z', 'tech', 'f', 'y', 'value']))
    check('new capacity', new_plant, _get(c, 'pNewCapacityFuel', ['z', 'f', 'y', 'value']))
    check('capex', capex_comp.assign(value=capex_comp['value'] / 1e6), _label(cost_summary, 'Capex: $m'))
    check('annual costs', costs, _label(cost_summary, 'Total Annual Cost by Zone: $m'))

    return out, gmap


# =============================================================================
# WRITING
# =============================================================================

def write_outputs(out: Dict[str, Tuple[List[str], pd.DataFrame]], scenario_dir: str,
                  log_func: Callable = _default_log) -> str:
    """Write the current-format epmresults.gdx and the raw output_csv files."""
    os.makedirs(scenario_dir, exist_ok=True)
    container = gt.Container()
    for name, (domain, df) in out.items():
        if not domain:
            gt.Parameter(container, name, records=float(df['value'].iloc[0]) if len(df) else None)
        else:
            gt.Parameter(container, name, domain=domain, records=df if len(df) else None)
    gdx_path = os.path.join(scenario_dir, 'epmresults.gdx')
    container.write(gdx_path)

    csv_dir = os.path.join(scenario_dir, 'output_csv')
    os.makedirs(csv_dir, exist_ok=True)
    for name in CSV_SYMBOLS:
        if name in out:
            out[name][1].to_csv(os.path.join(csv_dir, f'{name}.csv'), index=False)
    log_func(f"[legacy]   Wrote {gdx_path} ({len(out)} symbols) and {csv_dir}")
    return csv_dir


def _techfuel_inputs() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """(tech, f) pairs and 'tech-fuel' -> Processing mapping, as in run_output_treatment_gams."""
    techfuel_df = pd.read_csv(TECHFUEL_PATH, comment='#')[['tech', 'fuel']].rename(columns={'fuel': 'f'})
    processing = pd.read_csv(TECHFUEL_PROCESSING_PATH, comment='#')
    mapping = dict(zip(processing['tech'] + '-' + processing['fuel'], processing['Processing']))
    return techfuel_df.drop_duplicates(), mapping


def convert_folder(src: str, dest: Optional[str] = None, scenarios: Optional[List[str]] = None,
                   generator_map: Optional[str] = None, techfuel_map: Optional[str] = None,
                   treatment: bool = True, log_func: Callable = _default_log) -> str:
    """
    Convert every legacy scenario of a results folder.

    Parameters
    ----------
    src : str
        Results folder containing <scenario>/epmresults.gdx (legacy format)
    dest : str, optional
        Destination folder (default: <src>_converted)
    scenarios : list, optional
        Scenario folders to convert (default: all)
    generator_map : str, optional
        CSV (g, tech, f) with legacy names, forcing the mapping of some generators
    techfuel_map : str, optional
        CSV (legacy_tech, legacy_fuel, tech, fuel) extending LEGACY_TECHFUEL_MAP
    treatment : bool
        Run output_treatment on the CSVs (Tableau files)
    """
    src = os.path.normpath(src)
    dest = dest or f"{src}_converted"
    overrides = pd.read_csv(generator_map) if generator_map else None
    tf_map = dict(LEGACY_TECHFUEL_MAP)
    if techfuel_map:
        extra = pd.read_csv(techfuel_map)
        tf_map.update({(r.legacy_tech, r.legacy_fuel): (r.tech, r.fuel) for r in extra.itertuples()})

    found = sorted(d for d in os.listdir(src) if os.path.isfile(os.path.join(src, d, 'epmresults.gdx')))
    found = [s for s in found if not scenarios or s in scenarios]
    if not found:
        raise FileNotFoundError(f"No <scenario>/epmresults.gdx found in {src}")

    techfuel_df, techfuel_mapping = _techfuel_inputs()

    for scenario in found:
        log_func("=" * 60)
        log_func(f"[legacy] Scenario {scenario}")
        log_func("=" * 60)
        out, gmap = convert_legacy_gdx(os.path.join(src, scenario, 'epmresults.gdx'),
                                       generator_map=overrides, techfuel_map=tf_map, log_func=log_func)
        scenario_dir = os.path.join(dest, scenario)
        csv_dir = write_outputs(out, scenario_dir, log_func=log_func)
        gmap.to_csv(os.path.join(scenario_dir, 'legacy_generator_techfuel.csv'), index=False)

        if treatment:
            zone_country = out['pZoneCountry'][1][['z', 'c']]
            run_output_treatment(
                csv_dir,
                techfuel_df=techfuel_df,
                techfuel_mapping=techfuel_mapping,
                generator_techfuel_df=gmap[['g', 'tech', 'f']].drop_duplicates(),
                zone_country_df=zone_country,
                log_func=lambda msg: None,
            )
            log_func(f"[legacy]   output_treatment done: {sorted(os.listdir(csv_dir))}")

    log_func(f"\n[legacy] Converted {len(found)} scenario(s) to {dest}")
    return dest


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert legacy EPM results (old epmresults.gdx) to the current output format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python epm/output_treatment_legacy.py epm/output/Zimbabwe_SPD
  python epm/output_treatment_legacy.py epm/output/Zimbabwe_SPD --scenarios reference_zesa_modular
  # then, from epm/:
  python epm.py --postprocess output/Zimbabwe_SPD_converted --generate-figures
        """
    )
    
    parser.add_argument("src", help="Results folder with <scenario>/epmresults.gdx in legacy format")
    parser.add_argument("--dest", default=None, help="Destination folder (default: <src>_converted)")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Scenario folders to convert (default: all)")
    parser.add_argument("--generator-map", default=None,
                        help="CSV with columns g, tech, f (legacy names) forcing the mapping of generators")
    parser.add_argument("--techfuel-map", default=None,
                        help="CSV with columns legacy_tech, legacy_fuel, tech, fuel extending the name mapping")
    parser.add_argument("--no-treatment", action="store_true",
                        help="Only write the raw CSVs, skip output_treatment (Tableau files)")
    args = parser.parse_args()

    convert_folder(args.src, dest=args.dest, scenarios=args.scenarios,
                   generator_map=args.generator_map, techfuel_map=args.techfuel_map,
                   treatment=not args.no_treatment)
