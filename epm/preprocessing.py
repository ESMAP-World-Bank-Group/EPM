"""
**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
Description:
    This Python script is part of the GAMS-based Electricity Planning Model (EPM),
    designed for electricity system planning. It supports tasks such as capacity
    expansion, generation dispatch, and the enforcement of policy constraints,
    including renewable energy targets and emissions limits.

Author(s):
    ESMAP Modelling Team

Organization:
    World Bank

Version:
    (Specify version here)

License:
    Creative Commons Zero v1.0 Universal

Key Features:
    - Optimization of electricity generation and capacity planning
    - Inclusion of renewable energy integration and storage technologies
    - Multi-period, multi-region modeling framework
    - CO₂ emissions constraints and policy instruments

Notes:
    - Ensure GAMS is installed and the model has completed execution
      before running this script.
    - The model generates output files in the working directory
      which will be organized by this script.

Contact:
    Claire Nicolas — cnicolas@worldbank.org
**********************************************************************
"""
import importlib
import logging
import math
import subprocess
import sys
import numpy as np
import pandas as pd
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import chaospy  # optional dependency for Monte Carlo analysis
except ImportError:  # pragma: no cover - handled at runtime when Monte Carlo is requested
    chaospy = None

# Version pinned in requirements.txt (commented there because chaospy is optional).
CHAOSPY_VERSION = "4.3.18"


def _ensure_chaospy():
    """
    Return the chaospy module, installing it with pip if it is missing.

    chaospy is only needed for Monte Carlo analysis, so it is not a hard
    dependency. Every Monte Carlo call goes through this function: if the
    package is absent it is installed into the running interpreter
    (``sys.executable -m pip install chaospy==CHAOSPY_VERSION``), falling back
    to the latest release if the pinned version cannot be installed.

    Returns
    -------
    module
        The imported chaospy module.

    Raises
    ------
    ImportError
        If chaospy is missing and the automatic installation fails.
    """
    global chaospy
    if chaospy is not None:
        return chaospy

    logger.warning("chaospy is not installed; installing it for the Monte Carlo analysis...")
    errors = []
    # numpoly (a chaospy dependency) must come as a prebuilt wheel: building it from
    # source needs a C compiler (Microsoft C++ Build Tools on Windows).
    attempts = (
        ["--only-binary=numpoly", "chaospy"],
        ["--only-binary=numpoly", f"chaospy=={CHAOSPY_VERSION}"],
        ["chaospy"],
    )
    for args in attempts:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *args])
            importlib.invalidate_caches()
            chaospy = importlib.import_module("chaospy")
            logger.info("Installed chaospy %s.", getattr(chaospy, "__version__", ""))
            return chaospy
        except (subprocess.CalledProcessError, ImportError, OSError) as err:
            errors.append(f"pip install {' '.join(args)}: {err}")

    raise ImportError(
        "Monte Carlo analysis requires 'chaospy' and the automatic installation failed "
        f"({'; '.join(errors)}). Install it manually with "
        f"`pip install chaospy=={CHAOSPY_VERSION}` and rerun the command."
    )


class NamedJ:
    """
    A wrapper around a joint probability distribution created from multiple named distributions.

    This class allows accessing individual distributions by name and sampling from the joint distribution.
    """

    def __init__(self, distributions):
        """
        Initialize the joint distribution from a dictionary of named distributions.

        Parameters
        ----------
        distributions : dict
            Dictionary where each key is the name of a variable and the value is a dictionary with:
            - "type": name of a chaospy distribution (e.g., "Uniform")
            - "args": arguments passed to the distribution (e.g., lower and upper bounds)
        """
        _ensure_chaospy()
        self.J = self.J_from_dict(distributions.values())
        self.names = distributions.keys()
        self.mapping = {k: i for i, k in enumerate(self.names)}

    def __getitem__(self, attr):
        """
        Access a marginal distribution by name.

        Parameters
        ----------
        name : str
            Name of the variable (as defined in the input dictionary).
        """
        return self.J[self.mapping[attr]]

    def J_from_dict(self, values):
        cp = _ensure_chaospy()
        DD = []
        for v in values:
            D = getattr(cp, v["type"])
            DD.append(D(*v["args"]))
        return cp.J(*DD)

    def sample(self, size=100, rule="halton", fmt=3):
        """
        Sample from the joint distribution.

        Parameters
        ----------
        size : int
            Number of samples to generate.
        rule : str
            Sampling method used by chaospy (e.g., 'halton', 'random', 'sobol').
        fmt : int
            Number of decimal places to round the samples to.

        Returns
        -------
        pd.DataFrame
            Samples as a DataFrame with variable names as index and columns as samples.
        """
        samples = self.J.sample(size=size, rule=rule).round(fmt)
        if len(samples.shape) == 1:  # single feature when doing samples
            samples = samples.reshape(-1, samples.shape[0])
        index = [f"{n}" for n in self.names]
        return pd.DataFrame(samples, index=index)


def _with_assessment_suffix(scenario_name, suffix):
    """
    Append a sensitivity suffix while preserving project assessment markers (anything after '@').
    Uses '~' as the sensitivity separator to distinguish from assessment '@'.
    """
    if '@' in scenario_name:
        base, assessment = scenario_name.split('@', 1)
        return f"{base}~{suffix}@{assessment}"
    return f"{scenario_name}~{suffix}"


def _get_sensitivity_folder(base_path):
    """
    Get or create a sensitivity folder relative to the given base path.
    Uses pathlib for cross-platform compatibility.

    Parameters
    ----------
    base_path : str or Path
        Path to the base file (e.g., pSettings.csv)

    Returns
    -------
    Path
        Path to the sensitivity folder
    """
    base_path = Path(base_path)
    folder_sensi = base_path.parent / 'sensitivity'
    folder_sensi.mkdir(parents=True, exist_ok=True)
    return folder_sensi


def _generate_sensitivity_filepath(base_path, suffix):
    """
    Generate a sensitivity file path with proper cross-platform handling.

    Parameters
    ----------
    base_path : str or Path
        Path to the original file
    suffix : str
        Suffix to add to the filename (e.g., 'NoInterconnection')

    Returns
    -------
    str
        Full path to the new sensitivity file
    """
    base_path = Path(base_path)
    folder_sensi = _get_sensitivity_folder(base_path)
    # Create new filename: stem + suffix + extension
    new_filename = f"{base_path.stem}_{suffix}{base_path.suffix}"
    return str(folder_sensi / new_filename)


def _get_assessment_folder(base_path):
    """
    Get or create an assessment folder relative to the given base path.
    Uses pathlib for cross-platform compatibility.

    Parameters
    ----------
    base_path : str or Path
        Path to the base file (e.g., pGenDataInput.csv)

    Returns
    -------
    Path
        Path to the assessment folder
    """
    base_path = Path(base_path)
    folder_assessment = base_path.parent / 'assessment'
    folder_assessment.mkdir(parents=True, exist_ok=True)
    return folder_assessment


def _generate_assessment_filepath(base_path, suffix):
    """
    Generate an assessment file path with proper cross-platform handling.

    Parameters
    ----------
    base_path : str or Path
        Path to the original file
    suffix : str
        Suffix to add to the filename

    Returns
    -------
    str
        Full path to the new assessment file
    """
    base_path = Path(base_path)
    folder_assessment = _get_assessment_folder(base_path)
    # Create new filename: stem + suffix + extension
    new_filename = f"{base_path.stem}_{suffix}{base_path.suffix}"
    return str(folder_assessment / new_filename)


def _get_montecarlo_folder(base_path):
    """
    Get or create a montecarlo folder relative to the given base path.
    Uses pathlib for cross-platform compatibility.

    Parameters
    ----------
    base_path : str or Path
        Path to the base file

    Returns
    -------
    Path
        Path to the montecarlo folder
    """
    base_path = Path(base_path)
    folder_mc = base_path.parent / 'montecarlo'
    folder_mc.mkdir(parents=True, exist_ok=True)
    return folder_mc


def _generate_montecarlo_filepath(base_path, param, scenario_name):
    """
    Generate a montecarlo file path with proper cross-platform handling.

    The Monte Carlo scenario name is part of the file name so that two parent
    scenarios drawing the same sample value never overwrite each other's file.

    Parameters
    ----------
    base_path : str or Path
        Path to the original file
    param : str
        Parameter name (e.g., 'pFuelPrice')
    scenario_name : str
        Name of the Monte Carlo scenario the file belongs to

    Returns
    -------
    str
        Full path to the new montecarlo file
    """
    base_path = Path(base_path)
    folder_mc = _get_montecarlo_folder(base_path)
    new_filename = f"{param}_{scenario_name}.csv"
    return str(folder_mc / new_filename)


def multiindex2array(multiindex):
    """
    Convert a pandas MultiIndex to a NumPy array.
    """
    return np.array([np.array(row).astype(float) for row in multiindex]).T

def multiindex2df(multiindex):
    """
    Convert a pandas MultiIndex to a DataFrame.
    """
    return pd.DataFrame(multiindex2array(multiindex), index=multiindex.names)


# Continuous features: one value per sample drawn from the joint chaospy distribution.
CONTINUOUS_FEATURES = {'fossilfuel', 'fuel', 'demand', 'hydro'}
# Event features: random draws per sample - outages (which plants/lines, which seasons,
# how deep) and the weather year ('climate_year').
EVENT_FEATURES = {'thermal_outage', 'interco_outage', 'climate_year'}
# Features that can be sampled in the Monte Carlo uncertainty file.
MONTECARLO_FEATURES = CONTINUOUS_FEATURES | EVENT_FEATURES

# Technologies hit by 'thermal_outage' unless the 'techs' column says otherwise (case-insensitive).
THERMAL_TECHS = {'st', 'ccgt', 'ocgt', 'ice', 'nuclear'}

# Default settings of the event features (columns of the uncertainty file).
EVENT_DEFAULTS = {
    'thermal_outage': {'probability': 0.1, 'seasons': 1},
    'interco_outage': {'count': 1, 'seasons': 1},
}

# 'climate_year': model technology -> resource series of the climate-year dataset
# (pre-analysis/climate_years/fetch_climate_years.py, columns zone, tech, year, month, cf).
CLIMATE_TECH_MAP = {'pv': 'PV', 'cspplant': 'PV', 'csp': 'PV', 'solar': 'PV',
                    'onshorewind': 'Wind', 'offshorewind': 'Wind', 'wind': 'Wind'}

# Seed of the outage draws. Each sample uses its own stream, so a sample has the
# same outages in every parent scenario and adding samples does not change earlier ones.
MONTECARLO_SEED = 2026

# Base fossil fuel names. A fuel in pFuelPrice counts as fossil when its name, or
# the part before the first '_', is one of these (case-insensitive), so variants
# such as 'Gas_Moz' or 'Coal_Hwan' follow their base fuel.
FOSSIL_FUELS = {'coal', 'lignite', 'gas', 'lng', 'hfo', 'lfo', 'diesel', 'oil', 'crude', 'naphtha', 'kerosene', 'jet'}

# Hydro technologies scaled by the 'hydro' feature (case-insensitive, e.g. 'RoR').
HYDRO_TECHS = {'ror', 'reservoirhydro'}


def _parse_list(value):
    """Split a ';'-separated cell into a list; empty/NaN/'ALL' -> ['ALL']."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ['ALL']
    items = [v.strip() for v in str(value).split(';') if v.strip()]
    if not items or any(v.upper() == 'ALL' for v in items):
        return ['ALL']
    return items


def define_samples(df_uncertainties, nb_samples):
    """
    Generate a joint distribution and samples from a DataFrame defining uncertainty bounds.

    Parameters
    ----------
    df_uncertainties : pd.DataFrame
        Must contain columns: 'feature', 'type', 'lowerbound', 'upperbound'.
        Optional columns:
        - 'zones': ';'-separated model zones (or countries) the feature applies to; empty = all.
        - 'fuels': ';'-separated fuels the feature applies to (fuel features only); empty = default set.
        Event features ('thermal_outage', 'interco_outage') use 'lowerbound'/'upperbound' as the
        range of the share of capacity lost during an outage (0-1) and these optional columns:
        - 'techs': ';'-separated technologies hit by 'thermal_outage' (default: THERMAL_TECHS).
        - 'probability': chance that each plant has an outage in a sample ('thermal_outage').
        - 'lines': ';'-separated corridors 'ZoneA-ZoneB' that can fail ('interco_outage'; default all).
        - 'count': number of corridors hit per sample ('interco_outage').
        - 'seasons': number of seasons affected by each outage.
    nb_samples : int
        Number of samples to draw.

    Returns
    -------
    tuple
        (NamedJ distribution or None, dict of samples, dict feature -> zones, dict feature -> fuels,
        dict event feature -> settings). Samples are keyed 'sample_001', 'sample_002', ... and map
        each continuous feature to its value.
    """
    required = {'feature', 'type', 'lowerbound', 'upperbound'}
    missing = required - set(df_uncertainties.columns)
    if missing:
        raise ValueError(f"Uncertainty file is missing column(s) {sorted(missing)}.")

    def _number(row, column, default):
        value = row.get(column)
        if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == '':
            return default
        return float(value)

    uncertainties = {}
    zone_mapping = {}
    fuel_mapping = {}
    event_config = {}
    for _, row in df_uncertainties.iterrows():
        feature = str(row['feature']).strip()
        dist_type = str(row['type']).strip()
        if feature not in MONTECARLO_FEATURES:
            raise ValueError(
                f"Unknown Monte Carlo feature '{feature}'. Supported features: {sorted(MONTECARLO_FEATURES)}."
            )
        if feature in uncertainties or feature in event_config:
            raise ValueError(f"Monte Carlo feature '{feature}' is defined more than once.")
        lower, upper = float(row['lowerbound']), float(row['upperbound'])
        zone_mapping[feature] = _parse_list(row.get('zones'))
        fuel_mapping[feature] = _parse_list(row.get('fuels'))

        if feature == 'climate_year':
            path = row.get('file')
            if path is None or (isinstance(path, float) and pd.isna(path)):
                raise ValueError("'climate_year': give the climate-year dataset (vre_monthly_cf.csv) in the 'file' column.")
            season_months = row.get('season_months')
            event_config[feature] = {
                'file': str(path).strip(),
                # lowerbound/upperbound = first/last weather year to draw from (0 = no limit)
                'first_year': int(lower) if lower else None,
                'last_year': int(upper) if upper else None,
                'season_months': None if season_months is None or (isinstance(season_months, float)
                                                                   and pd.isna(season_months)) else str(season_months),
                'zones': zone_mapping[feature],
            }
            continue

        if feature in EVENT_FEATURES:
            if not (0 <= lower <= upper <= 1):
                raise ValueError(f"'{feature}': lowerbound/upperbound are shares of capacity lost, "
                                 f"0 <= lowerbound <= upperbound <= 1 (got {lower}, {upper}).")
            defaults = EVENT_DEFAULTS[feature]
            cfg = {'lower': lower, 'upper': upper,
                   'seasons': int(_number(row, 'seasons', defaults['seasons'])),
                   'zones': zone_mapping[feature]}
            if feature == 'thermal_outage':
                cfg['probability'] = _number(row, 'probability', defaults['probability'])
                techs = _parse_list(row.get('techs'))
                cfg['techs'] = THERMAL_TECHS if 'ALL' in techs else {t.lower() for t in techs}
                if not 0 <= cfg['probability'] <= 1:
                    raise ValueError(f"'thermal_outage': probability must be in [0, 1] (got {cfg['probability']}).")
            else:
                cfg['count'] = int(_number(row, 'count', defaults['count']))
                cfg['lines'] = _parse_list(row.get('lines'))
            if cfg['seasons'] < 1:
                raise ValueError(f"'{feature}': seasons must be >= 1.")
            event_config[feature] = cfg
            continue

        cp = _ensure_chaospy()
        dist = getattr(cp, dist_type, None)
        if not (isinstance(dist, type) and issubclass(dist, cp.Distribution)):
            raise ValueError(f"'{dist_type}' is not a chaospy distribution (e.g. Uniform, Normal, Triangle).")
        uncertainties[feature] = {'type': dist_type, 'args': (lower, upper)}

    if not uncertainties and not event_config:
        raise ValueError("Uncertainty file defines no feature.")

    width = max(3, len(str(nb_samples)))
    names = [f'sample_{i + 1:0{width}d}' for i in range(nb_samples)]
    distribution = None
    if uncertainties:
        distribution = NamedJ(uncertainties)
        samples_df = distribution.sample(size=nb_samples, rule='halton')
        samples = {
            name: {idx: round(float(samples_df.loc[idx, col]), 3) for idx in samples_df.index}
            for name, col in zip(names, samples_df.columns)
        }
    else:
        samples = {name: {} for name in names}
    return distribution, samples, zone_mapping, fuel_mapping, event_config


def _read_csv_list(path, column):
    """
    Unique values of a column of a CSV file, as strings (empty list if the file is missing).
    Falls back to the first column when ``column`` is absent: the reduced year file written
    by ``--simple`` has a '0' header instead of 'y'.
    """
    if path is None or not os.path.exists(str(path)):
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    values = df[column] if column in df.columns else df.iloc[:, 0]
    return [str(v) for v in values.dropna().unique()]


def _plant_base_availability(scenario, gen):
    """
    Seasonal availability of each plant as the model builds it: custom value
    (pAvailability), else zone/tech/fuel default (pAvailabilityDefault), else
    generic tech/fuel value (resources/pAvailabilityGeneric.csv), else 1.

    Returns a DataFrame indexed by generator with one column per season.
    """
    seasons = _read_csv_list(scenario.get('pHours'), 'q')
    g_col = 'g' if 'g' in gen.columns else gen.columns[0]
    z_col = 'z' if 'z' in gen.columns else gen.columns[1]
    f_col = 'fuel' if 'fuel' in gen.columns else ('f' if 'f' in gen.columns else gen.columns[3])
    keys = gen[[g_col, z_col, 'tech', f_col]].astype(str).drop_duplicates(subset=[g_col])
    keys.columns = ['g', 'z', 'tech', 'fuel']
    base = pd.DataFrame(np.nan, index=keys['g'], columns=seasons)

    default_path = scenario.get('pAvailabilityDefault')
    if default_path is not None and os.path.exists(str(default_path)):
        default = pd.read_csv(default_path)
        default.columns = ['z', 'tech', 'fuel'] + list(default.columns[3:])
        default[['z', 'tech', 'fuel']] = default[['z', 'tech', 'fuel']].astype(str)
        merged = keys.merge(default, on=['z', 'tech', 'fuel'], how='left').set_index('g')
        for q in seasons:
            if q in merged.columns:
                base[q] = merged[q]

    generic_path = Path(__file__).parent / 'resources' / 'pAvailabilityGeneric.csv'
    if generic_path.exists():
        generic = pd.read_csv(generic_path).astype({'tech': str, 'fuel': str})
        value = keys.merge(generic, on=['tech', 'fuel'], how='left').set_index('g')['value']
        for q in seasons:
            base[q] = base[q].fillna(value)

    custom_path = scenario.get('pAvailability')
    if custom_path is not None and os.path.exists(str(custom_path)):
        custom = pd.read_csv(custom_path)
        custom = custom.set_index(custom.columns[0])
        custom.index = custom.index.astype(str)
        for q in seasons:
            if q in custom.columns:
                values = custom[q].reindex(base.index)
                base[q] = values.where(values.notna(), base[q])
    return base.fillna(1.0)


def _draw_thermal_outages(scenario, cfg, rng, zones):
    """
    Random thermal outages for one sample, re-centred on the input availability.

    The input availability already includes average outages. Every plant in scope is
    first scaled by 1 / (1 - expected loss), with expected loss = probability x
    (seasons / number of seasons) x mean depth, so that the availability averaged over
    many samples equals the input. Plants hit in the sample then lose a share drawn
    in [lower, upper] of their availability in randomly selected season(s).

    Returns (availability DataFrame g x seasons for the plants in scope, events list).
    """
    gen = pd.read_csv(scenario['pGenDataInput'])
    g_col = 'g' if 'g' in gen.columns else gen.columns[0]
    z_col = 'z' if 'z' in gen.columns else gen.columns[1]
    mask = gen['tech'].astype(str).str.lower().isin(cfg['techs'])
    if zones is not None:
        mask &= gen[z_col].astype(str).isin(zones)
    scope = gen.loc[mask].drop_duplicates(subset=[g_col])
    if scope.empty:
        return pd.DataFrame(), []

    base = _plant_base_availability(scenario, gen).loc[scope[g_col].astype(str)]
    seasons = list(base.columns)
    n_hit_seasons = min(cfg['seasons'], len(seasons))
    expected_loss = cfg['probability'] * n_hit_seasons / len(seasons) * (cfg['lower'] + cfg['upper']) / 2
    availability = (base / (1 - expected_loss)).clip(upper=1) if expected_loss < 1 else base.copy()

    events = []
    zone_of = scope.set_index(scope[g_col].astype(str))[z_col].astype(str)
    for g in availability.index:
        if rng.random() >= cfg['probability']:
            continue
        hit = rng.choice(seasons, size=n_hit_seasons, replace=False)
        depth = float(rng.uniform(cfg['lower'], cfg['upper']))
        availability.loc[g, hit] = availability.loc[g, hit] * (1 - depth)
        for q in hit:
            events.append({'type': 'thermal_outage', 'element': g, 'zone': zone_of[g],
                           'season': q, 'depth': round(depth, 3)})
    return availability.round(4), events


def _draw_interco_outages(scenario, cfg, rng, zones):
    """
    Random interconnection outages for one sample: ``count`` corridors (existing or
    candidate, both directions) lose a share drawn in [lower, upper] of their transfer
    capacity in randomly selected season(s), in every model year.

    Returns (pTransferOutage DataFrame in CSV layout, events list).
    """
    corridors = set()
    for key in ('pTransferLimit', 'pNewTransmission'):
        path = scenario.get(key)
        if path is None or not os.path.exists(str(path)):
            continue
        df = pd.read_csv(path)
        if key == 'pTransferLimit':
            values = df.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').fillna(0)
            df = df.loc[values.sum(axis=1) > 0]
        for a, b in zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)):
            if a != b:
                corridors.add(tuple(sorted((a, b))))
    if zones is not None:
        corridors = {c for c in corridors if c[0] in zones and c[1] in zones}
    if 'ALL' not in cfg['lines']:
        wanted = set(cfg['lines'])
        corridors = {c for c in corridors if f'{c[0]}-{c[1]}' in wanted or f'{c[1]}-{c[0]}' in wanted}
    corridors = sorted(corridors)

    seasons = _read_csv_list(scenario.get('pHours'), 'q')
    years = _read_csv_list(scenario.get('y'), 'y')
    columns = ['From', 'To', 'q'] + years
    if not corridors or not seasons or not years:
        return pd.DataFrame(columns=columns), []

    n_lines = min(cfg['count'], len(corridors))
    n_hit_seasons = min(cfg['seasons'], len(seasons))
    rows, events = [], []
    for idx in rng.choice(len(corridors), size=n_lines, replace=False):
        a, b = corridors[idx]
        depth = float(rng.uniform(cfg['lower'], cfg['upper']))
        for q in rng.choice(seasons, size=n_hit_seasons, replace=False):
            for z1, z2 in ((a, b), (b, a)):
                rows.append([z1, z2, q] + [round(depth, 4)] * len(years))
            events.append({'type': 'interco_outage', 'element': f'{a}-{b}', 'zone': f'{a};{b}',
                           'season': q, 'depth': round(depth, 3)})
    return pd.DataFrame(rows, columns=columns), events


def _season_months(seasons, spec=None):
    """
    Months of each model season. ``spec`` like 'Q1:1-3;Q2:4-6;Q3:7-9;Q4:10-12' (months may
    also be listed, e.g. 'Q1:12,1,2'). Without a spec the 12 months are split evenly, in
    calendar order, over the seasons (Q1..Q4 -> quarters), with a warning.
    """
    if spec:
        mapping = {}
        for part in spec.split(';'):
            if not part.strip():
                continue
            name, months = part.split(':')
            values = []
            for item in months.split(','):
                if '-' in item:
                    a, b = (int(x) for x in item.split('-'))
                    values += list(range(a, b + 1))
                else:
                    values.append(int(item))
            mapping[name.strip()] = values
        missing = [q for q in seasons if q not in mapping]
        if missing:
            raise ValueError(f"'climate_year': season_months does not define season(s) {missing}.")
        return mapping
    if not seasons or 12 % len(seasons):
        raise ValueError(f"'climate_year': give 'season_months' (seasons {seasons} cannot split the year evenly).")
    size = 12 // len(seasons)
    logger.warning("'climate_year': no season_months given; assuming %s.",
                   '; '.join(f'{q}: months {i * size + 1}-{(i + 1) * size}' for i, q in enumerate(seasons)))
    return {q: list(range(i * size + 1, (i + 1) * size + 1)) for i, q in enumerate(seasons)}


def _climate_factors(cfg, seasons):
    """
    Seasonal factors of every weather year: mean capacity factor of the season's months in
    that year divided by its mean over all years of the range (the climatology).

    Returns (DataFrame indexed by (zone, tech, year) with one column per season, list of years).
    """
    data = pd.read_csv(cfg['file'])
    data[['zone', 'tech']] = data[['zone', 'tech']].astype(str)
    if cfg['first_year']:
        data = data.loc[data['year'] >= cfg['first_year']]
    if cfg['last_year']:
        data = data.loc[data['year'] <= cfg['last_year']]
    if data.empty:
        raise ValueError(f"'climate_year': no data in {cfg['file']} for the selected years.")
    months = _season_months(seasons, cfg['season_months'])
    season_of = {m: q for q, ms in months.items() for m in ms}
    data['season'] = data['month'].map(season_of)
    seasonal = data.dropna(subset=['season']).groupby(['zone', 'tech', 'year', 'season'])['cf'].mean().unstack()
    climatology = seasonal.groupby(level=['zone', 'tech']).mean()
    factors = seasonal / climatology.reindex(seasonal.index.droplevel('year')).values
    return factors.fillna(1.0), sorted(data['year'].unique())


def _apply_climate_year(scenario, factors, year, zones):
    """
    Scale pVREProfile and pVREgenProfile by the factors of weather ``year`` (capped at 1).
    Returns {param: DataFrame} for the files that changed and the list of zones without data.
    """
    year_factors = factors.xs(year, level='year')
    available = set(year_factors.index.get_level_values('zone'))
    out, no_data = {}, set()

    def factor_table(zone_series, tech_series, season_series):
        resource = tech_series.astype(str).str.lower().map(CLIMATE_TECH_MAP)
        keys = pd.MultiIndex.from_arrays([zone_series.astype(str), resource])
        stacked = year_factors.stack()
        values = [stacked.get((z, r, q), np.nan) if isinstance(r, str) else np.nan
                  for (z, r), q in zip(keys, season_series.astype(str))]
        ok = pd.Series(values, index=zone_series.index)
        if zones is not None:
            ok[~zone_series.astype(str).isin(zones)] = np.nan
        no_data.update(set(zone_series[resource.notna() & ~zone_series.astype(str).isin(available)].astype(str)))
        return ok

    path = scenario.get('pVREProfile')
    if path is not None and os.path.exists(str(path)):
        df = pd.read_csv(path)
        hours = list(df.columns[4:])
        factor = factor_table(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2])
        mask = factor.notna()
        if mask.any():
            df[hours] = df[hours].astype(float)
            df.loc[mask, hours] = df.loc[mask, hours].mul(factor[mask], axis=0).clip(upper=1).round(4)
            out['pVREProfile'] = df

    path = scenario.get('pVREgenProfile')
    if path is not None and os.path.exists(str(path)):
        df = pd.read_csv(path)
        gen = pd.read_csv(scenario['pGenDataInput'])
        g_col = 'g' if 'g' in gen.columns else gen.columns[0]
        z_col = 'z' if 'z' in gen.columns else gen.columns[1]
        info = gen.drop_duplicates(subset=[g_col]).set_index(gen.drop_duplicates(subset=[g_col])[g_col].astype(str))
        g = df.iloc[:, 0].astype(str)
        hours = list(df.columns[3:])
        factor = factor_table(g.map(info[z_col]).fillna(''), g.map(info['tech']).fillna(''), df.iloc[:, 1])
        mask = factor.notna()
        if mask.any():
            df[hours] = df[hours].astype(float)
            df.loc[mask, hours] = df.loc[mask, hours].mul(factor[mask], axis=0).clip(upper=1).round(4)
            out['pVREgenProfile'] = df
    return out, sorted(z for z in no_data if z)


def create_scenarios_montecarlo(samples, s, zone_mapping, fuel_mapping=None, event_config=None,
                                seed=MONTECARLO_SEED):
    """
    Generate new scenarios for Monte Carlo analysis by modifying baseline input files
    based on provided uncertainty samples.

    For every parent scenario in ``s`` and every sample, a scenario named
    ``<parent>_MC<k>`` is created. Its modified input files are written under a
    ``montecarlo/`` subdirectory next to the parent's files.

    Sample values are relative changes: a value of 0.2 multiplies the affected
    inputs by 1.2. Supported features:

    - ``fossilfuel``: pFuelPrice of fossil fuels (see FOSSIL_FUELS, variants like
      'Gas_Moz' included), or the fuels listed in the 'fuels' column.
    - ``fuel``: pFuelPrice of every fuel, or the fuels listed in the 'fuels' column.
    - ``demand``: pDemandForecast (energy and peak) of the selected zones.
    - ``hydro``: pAvailabilityDefault and pAvailability of ROR / ReservoirHydro
      plants in the selected zones, capped at 1.

    Event features (see ``define_samples``), drawn with a random generator seeded by
    ``seed`` and the sample number:

    - ``thermal_outage``: random seasonal outages of thermal plants, re-centred on
      the input availability, written to pAvailability (see ``_draw_thermal_outages``).
    - ``interco_outage``: random seasonal outages of corridors, written to pTransferOutage.

    Zones can be given as model zones or countries; fuel prices are defined by
    country, so zones are mapped to their country through zcmap.

    Parameters
    ----------
    samples : dict
        Output of ``define_samples``: sample name -> {feature: value}.
    s : dict
        Dictionary of scenarios (name -> pd.Series of parameter file paths).
    zone_mapping : dict
        Feature -> list of zones (['ALL'] for all zones).
    fuel_mapping : dict, optional
        Feature -> list of fuels (['ALL'] for the feature's default set).

    Returns
    -------
    tuple
        (s, scenarios_montecarlo, mc_mapping, mc_events) where ``mc_mapping`` is a DataFrame
        indexed by Monte Carlo scenario with columns 'parent', 'sample', one column per
        continuous feature and the number of outages per event feature. It is the explicit
        link to the parent scenario used to load the fixed investment plan, so parent names
        may contain any character. ``mc_events`` lists every drawn outage (scenario, type,
        element, zone, season, depth).
    """
    fuel_mapping = fuel_mapping or {}

    def _read_zcmap(scenario):
        path = scenario.get('zcmap') if hasattr(scenario, 'get') else None
        if path is None or not os.path.exists(str(path)):
            return pd.DataFrame(columns=['z', 'c'])
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        zone_col = 'z' if 'z' in df.columns else df.columns[0]
        country_col = 'c' if 'c' in df.columns else df.columns[1]
        return df.rename(columns={zone_col: 'z', country_col: 'c'})[['z', 'c']].astype(str)

    def _check_names(requested, available, feature, what):
        if 'ALL' in requested:
            return
        unknown = sorted(set(requested) - set(available))
        if unknown:
            logger.warning("Monte Carlo '%s': %s %s not found in the inputs; they are ignored.",
                           feature, what, unknown)
        if not set(requested) & set(available):
            raise ValueError(
                f"Monte Carlo feature '{feature}': none of the {what} {requested} exist in the inputs."
            )

    def _selected_zones(feature, zcmap):
        """Model zones targeted by a feature (None = all). Countries expand to their zones."""
        zones_req = zone_mapping.get(feature, ['ALL'])
        if 'ALL' in zones_req:
            return None
        return set(zcmap.loc[zcmap['z'].isin(zones_req) | zcmap['c'].isin(zones_req), 'z']) | set(zones_req)

    def _selected_countries(feature, zcmap):
        """Countries targeted by a feature (None = all); fuel prices are defined by country."""
        zones_req = zone_mapping.get(feature, ['ALL'])
        if 'ALL' in zones_req:
            return None
        return set(zcmap.loc[zcmap['z'].isin(zones_req) | zcmap['c'].isin(zones_req), 'c']) | set(zones_req)

    def _mask(series, selection):
        if selection is None:
            return pd.Series(True, index=series.index)
        return series.astype(str).isin(selection)

    def _scale_fuel_price(scenario, feature, val, zcmap):
        df = pd.read_csv(scenario['pFuelPrice'])
        country_col, fuel_col = df.columns[0], df.columns[1]
        year_cols = df.columns[2:]

        fuels_req = fuel_mapping.get(feature, ['ALL'])
        all_fuels = df[fuel_col].astype(str)
        if 'ALL' not in fuels_req:
            fuel_mask = all_fuels.isin(fuels_req)
        elif feature == 'fossilfuel':
            base = all_fuels.str.split('_').str[0].str.lower()
            fuel_mask = base.isin(FOSSIL_FUELS) | all_fuels.str.lower().isin(FOSSIL_FUELS)
        else:  # 'fuel': every fuel
            fuel_mask = pd.Series(True, index=df.index)

        mask = fuel_mask & _mask(df[country_col], _selected_countries(feature, zcmap))
        df[year_cols] = df[year_cols].astype(float)
        df.loc[mask, year_cols] = df.loc[mask, year_cols] * (1 + val)
        return df, int(mask.sum()), sorted(df.loc[mask, fuel_col].astype(str).unique())

    def _scale_demand(scenario, feature, val, zcmap):
        df = pd.read_csv(scenario['pDemandForecast'])
        value_cols = list(df.columns[2:])
        mask = _mask(df[df.columns[0]], _selected_zones(feature, zcmap))
        df[value_cols] = df[value_cols].astype(float)
        df.loc[mask, value_cols] = df.loc[mask, value_cols] * (1 + val)
        return df, int(mask.sum())

    def _scale_hydro_default(scenario, feature, val, zcmap):
        df = pd.read_csv(scenario['pAvailabilityDefault'])
        zone_col, tech_col = df.columns[0], df.columns[1]
        value_cols = list(df.columns[3:])
        mask = (df[tech_col].astype(str).str.lower().isin(HYDRO_TECHS)
                & _mask(df[zone_col], _selected_zones(feature, zcmap)))
        df[value_cols] = df[value_cols].astype(float)
        df.loc[mask, value_cols] = (df.loc[mask, value_cols] * (1 + val)).clip(upper=1)
        return df, int(mask.sum())

    def _scale_hydro_custom(scenario, feature, val, zcmap):
        df = pd.read_csv(scenario['pAvailability'])
        gen_col = df.columns[0]
        value_cols = list(df.columns[1:])
        gen = pd.read_csv(scenario['pGenDataInput'])
        g_col = 'g' if 'g' in gen.columns else gen.columns[0]
        z_col = 'z' if 'z' in gen.columns else gen.columns[1]
        hydro = gen.loc[gen['tech'].astype(str).str.lower().isin(HYDRO_TECHS), [g_col, z_col]]
        hydro = hydro.loc[_mask(hydro[z_col], _selected_zones(feature, zcmap)), g_col].astype(str)
        mask = df[gen_col].astype(str).isin(set(hydro))
        df[value_cols] = df[value_cols].astype(float)
        df.loc[mask, value_cols] = (df.loc[mask, value_cols] * (1 + val)).clip(upper=1)
        return df, int(mask.sum())

    def _has_custom_availability(scenario):
        return 'pAvailability' in scenario.index and os.path.exists(str(scenario['pAvailability']))

    def _validate(parent, scenario, zcmap):
        """Check zones/fuels once per parent scenario and log what each feature changes."""
        known_zones = set(zcmap['z']) | set(zcmap['c'])
        for feature in zone_mapping:
            _check_names(zone_mapping[feature], known_zones, feature, 'zones (zcmap)')
            if feature in ('fossilfuel', 'fuel'):
                fuels = pd.read_csv(scenario['pFuelPrice']).iloc[:, 1].astype(str).unique()
                _check_names(fuel_mapping.get(feature, ['ALL']), fuels, feature, 'fuels (pFuelPrice)')
                _, n, scaled = _scale_fuel_price(scenario, feature, 0.0, zcmap)
                if n == 0:
                    raise ValueError(f"Monte Carlo '{feature}': no pFuelPrice row matches the selected zones/fuels.")
                logger.info("Monte Carlo [%s] '%s': %d pFuelPrice row(s), fuels: %s",
                            parent, feature, n, ", ".join(scaled))
            elif feature == 'demand':
                _, n = _scale_demand(scenario, feature, 0.0, zcmap)
                if n == 0:
                    raise ValueError("Monte Carlo 'demand': no pDemandForecast row for the selected zones.")
                logger.info("Monte Carlo [%s] 'demand': %d pDemandForecast row(s).", parent, n)
            elif feature == 'hydro':
                _, n_def = _scale_hydro_default(scenario, feature, 0.0, zcmap)
                n_cus = _scale_hydro_custom(scenario, feature, 0.0, zcmap)[1] if _has_custom_availability(scenario) else 0
                if n_def + n_cus == 0:
                    logger.warning("Monte Carlo [%s] 'hydro': no hydro availability in the selected zones; "
                                   "the feature has no effect.", parent)
                logger.info("Monte Carlo [%s] 'hydro': %d pAvailabilityDefault row(s), %d pAvailability plant(s).",
                            parent, n_def, n_cus)

    def _save(df, scenario_name, param):
        path_file = _generate_montecarlo_filepath(scenarios_montecarlo[scenario_name][param], param, scenario_name)
        df.to_csv(path_file, index=False)
        scenarios_montecarlo[scenario_name][param] = path_file

    def _save_new(df, scenario_name, param, base_param):
        """Save an input the configuration may not define, next to ``base_param``'s file."""
        path_file = _generate_montecarlo_filepath(scenarios_montecarlo[scenario_name][base_param], param, scenario_name)
        df.to_csv(path_file, index=False)
        scenarios_montecarlo[scenario_name][param] = path_file

    def _save_availability(availability, scenario_name):
        """Merge plant availabilities (g x seasons) into the custom availability file."""
        current = scenarios_montecarlo[scenario_name]
        if _has_custom_availability(current):
            custom = pd.read_csv(current['pAvailability'])
            gen_col = custom.columns[0]
            custom[gen_col] = custom[gen_col].astype(str)
            custom = custom.loc[~custom[gen_col].isin(availability.index)]
            columns = list(custom.columns[1:])
        else:
            gen_col, custom, columns = 'g', pd.DataFrame(columns=['g']), list(availability.columns)
        new = availability.reindex(columns=columns).reset_index()
        new.columns = [gen_col] + columns
        merged = pd.concat([custom, new], ignore_index=True)
        if 'pAvailability' in current.index:
            _save(merged, scenario_name, 'pAvailability')
        else:
            _save_new(merged, scenario_name, 'pAvailability', base_param='pAvailabilityDefault')

    event_config = event_config or {}
    scenarios_montecarlo = {}
    mapping_rows = []
    event_rows = []
    first_parent = next(iter(s), None)

    climate_factors, climate_order = None, []
    if 'climate_year' in event_config and first_parent is not None:
        seasons = _read_csv_list(s[first_parent].get('pHours'), 'q')
        climate_factors, years = _climate_factors(event_config['climate_year'], seasons)
        climate_order = list(np.random.default_rng([seed, 2]).permutation(years))
        logger.info("Monte Carlo 'climate_year': %d weather years (%s-%s), %d zone(s) in %s.", len(years),
                    years[0], years[-1], climate_factors.index.get_level_values('zone').nunique(),
                    event_config['climate_year']['file'])
    for parent in list(s.keys()):
        zcmap = _read_zcmap(s[parent])
        _validate(parent, s[parent], zcmap)
        for sample_name, sample in samples.items():
            name_scenario = f"{parent}_MC{sample_name.split('_')[-1]}"
            scenarios_montecarlo[name_scenario] = s[parent].copy()
            current = scenarios_montecarlo[name_scenario]
            for feature, val in sample.items():
                if feature in ('fossilfuel', 'fuel'):
                    _save(_scale_fuel_price(current, feature, val, zcmap)[0], name_scenario, 'pFuelPrice')
                elif feature == 'demand':
                    _save(_scale_demand(current, feature, val, zcmap)[0], name_scenario, 'pDemandForecast')
                elif feature == 'hydro':
                    _save(_scale_hydro_default(current, feature, val, zcmap)[0], name_scenario, 'pAvailabilityDefault')
                    if _has_custom_availability(current):
                        _save(_scale_hydro_custom(current, feature, val, zcmap)[0], name_scenario, 'pAvailability')

            # Random outages, after the continuous features so they apply on top of them
            # (e.g. thermal outages on the hydro-scaled custom availability file).
            sample_events = []
            sample_index = int(sample_name.split('_')[-1])
            model_zones = set(zcmap['z']) or None  # input files may contain zones outside the model

            def _event_zones(feature):
                selected = _selected_zones(feature, zcmap)
                return model_zones if selected is None else selected

            if 'thermal_outage' in event_config:
                cfg = event_config['thermal_outage']
                rng = np.random.default_rng([seed, sample_index, 0])
                availability, events = _draw_thermal_outages(current, cfg, rng, _event_zones('thermal_outage'))
                if not availability.empty:
                    _save_availability(availability, name_scenario)
                sample_events += events
            if 'interco_outage' in event_config:
                cfg = event_config['interco_outage']
                rng = np.random.default_rng([seed, sample_index, 1])
                outage, events = _draw_interco_outages(current, cfg, rng, _event_zones('interco_outage'))
                if not outage.empty:
                    _save_new(outage, name_scenario, 'pTransferOutage', base_param='pTransferLimit')
                sample_events += events
            climate = {}
            if 'climate_year' in event_config:
                # Years are shuffled once and used in turn: every weather year is drawn
                # before any repeats, and a sample uses the same year in every parent.
                year = climate_order[(sample_index - 1) % len(climate_order)]
                changed, no_data = _apply_climate_year(current, climate_factors, year,
                                                       _event_zones('climate_year'))
                for param, df in changed.items():
                    _save(df, name_scenario, param)
                if no_data and parent == first_parent and sample_index == 1:
                    logger.warning("Monte Carlo 'climate_year': no climate data for zone(s) %s; "
                                   "their VRE profiles are not changed.", no_data)
                climate = {'climate_year': int(year)}
            for event in sample_events:
                event_rows.append({'scenario': name_scenario, 'parent': parent, 'sample': sample_name, **event})

            counts = {f'n_{feature}': sum(e['type'] == feature for e in sample_events)
                      for feature in event_config if feature != 'climate_year'}
            mapping_rows.append({'scenario': name_scenario, 'parent': parent, 'sample': sample_name,
                                 **sample, **climate, **counts})

    mc_mapping = pd.DataFrame(mapping_rows).set_index('scenario') if mapping_rows else pd.DataFrame(
        columns=['parent', 'sample'])
    mc_events = pd.DataFrame(event_rows, columns=['scenario', 'parent', 'sample', 'type', 'element',
                                                  'zone', 'season', 'depth'])
    return s, scenarios_montecarlo, mc_mapping, mc_events

def perform_sensitivity(sensitivity, s):

            
    param = 'interco'
    
    if sensitivity.get(param) and not np.isnan(float(sensitivity[param])):  # testing implications of interconnection mode
        
        # Creating a new folder
        folder_sensi = os.path.join(os.path.dirname(s['baseline']['pSettings']), 'sensitivity')
        if not os.path.exists(folder_sensi):
            os.mkdir(folder_sensi)
        

    # Work on the initial scenario set (including project assessments), avoid cascading combinations
    base_scenarios = list(s.keys())
    
    param = 'interco'
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):  # testing implications of interconnection mode

        
        df = pd.read_csv(s['baseline']['pSettings'])
        # Modifying the value if it's 1 put 0 and vice versa
        name = 'NoInterconnection'
        df.loc[df['Abbreviation'] == "fEnableInternalExchange", 'Value'] = 0

        path_file = _generate_sensitivity_filepath(s['baseline']['pSettings'], name)
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[f'baseline~{name}'] = s['baseline'].copy()
        s[f'baseline~{name}']['pSettings'] = path_file

        #----------------------------------------

        df = pd.read_csv(s['baseline']['pSettings'])
        # fAllowTransferExpansion
        name = 'NoInterconnectionExpansion'
        df.loc[df['Abbreviation'] == 'fAllowTransferExpansion', 'Value'] = 0

        path_file = _generate_sensitivity_filepath(s['baseline']['pSettings'], name)
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[f'baseline~{name}'] = s['baseline'].copy()
        s[f'baseline~{name}']['pSettings'] = path_file

        #----------------------------------------

        df = pd.read_csv(s['baseline']['pSettings'])
        # OptimalInterconnection with fRemoveInternalTransferLimit
        name = 'OptimalInterconnection'
        df.loc[df['Abbreviation'] == 'fRemoveInternalTransferLimit', 'Value'] = 1

        path_file = _generate_sensitivity_filepath(s['baseline']['pSettings'], name)
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[f'baseline~{name}'] = s['baseline'].copy()
        s[f'baseline~{name}']['pSettings'] = path_file
    
    param = 'ExternalExchange'
    if param in sensitivity and not (isinstance(sensitivity[param], float) and math.isnan(sensitivity[param])):
        new_scenarios = {}
        suffix = 'NoExternalExchange'
        for scenario in base_scenarios:
            df = pd.read_csv(s[scenario]['pSettings'])
            df.loc[df['Abbreviation'] == 'fEnableExternalExchange', 'Value'] = 0

            scenario_name = _with_assessment_suffix(scenario, suffix)
            path_file = _generate_sensitivity_filepath(s[scenario]['pSettings'], suffix)
            df.to_csv(path_file, index=False)

            new_scenarios[scenario_name] = s[scenario].copy()
            new_scenarios[scenario_name]['pSettings'] = path_file

        s.update(new_scenarios)
    
    param = 'RemoveGenericTechnologies'
    if param in sensitivity and not (isinstance(sensitivity[param], float) and math.isnan(sensitivity[param])):

        df = pd.read_csv(s['baseline']['pGenDataInput'])
        # Create a list of technologies to remove that are in a string separated by '&'
        # For example: 'WindOnshore&WindOffshore&SolarPV' will be converted to ['WindOnshore', 'WindOffshore', 'SolarPV']
        techs_to_remove = sensitivity['RemoveGenericTechnologies'].split('&')
        # For tech that equal to sensitivity['RemoveGenericTechnologies'], status 3, and Candidate in the name, put BuildLimitperYear to 0
        mask = df['tech'].isin(techs_to_remove) & (df['Status'] == 3) & (df['gen'].str.contains('Candidate'))
        df.loc[mask, 'BuildLimitperYear'] = 0

        name = 'RemoveGenericTechnologies'
        path_file = _generate_sensitivity_filepath(s['baseline']['pGenDataInput'], name)
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[f'baseline~{name}'] = s['baseline'].copy()
        s[f'baseline~{name}']['pGenDataInput'] = path_file

    param = 'RemoveCandidateFuel'
    if param in sensitivity and not (isinstance(sensitivity[param], float) and math.isnan(sensitivity[param])):
        # Create a list of fuels to remove that are in a string separated by '&'
        # For example: 'Biomass&Gas' will create two separate scenarios
        fuels_to_remove = [f.strip() for f in sensitivity['RemoveCandidateFuel'].split('&') if f.strip()]

        # Iterate over all existing scenarios
        new_scenarios = {}
        for scenario in base_scenarios:
            for fuel in fuels_to_remove:
                df = pd.read_csv(s[scenario]['pGenDataInput'])
                # For fuel that matches (case-insensitive) and Status is 2 or 3, set Status to 0 (candidate generators become unavailable)
                mask = (df['f'].str.lower() == fuel.lower()) & df['Status'].isin([2, 3])
                df.loc[mask, 'Status'] = 0

                scenario_name = _with_assessment_suffix(scenario, f"No{fuel}")
                path_file = _generate_sensitivity_filepath(s[scenario]['pGenDataInput'], f"No{fuel}")
                df.to_csv(path_file, index=False)

                # Put in the scenario dir
                new_scenarios[scenario_name] = s[scenario].copy()
                new_scenarios[scenario_name]['pGenDataInput'] = path_file

        s.update(new_scenarios)

    param = 'pSettings'
    if sensitivity.get(param) and not math.isnan(float(sensitivity[param])):  # testing implications of some setting parameters
        settings_sensi = {'CO2backstop': [500]}
        """ 'VoLL': [250],
                          'fApplyPlanningReserveConstraint': [0], 'sVREForecastErrorPct': [0, 0.3],
                          'zonal_spinning_reserve_constraints': [0],
                          'CostSurplus': [1, 5], 'CostCurtail': [1, 5], "fEnableInternalExchange": [0,1],
                          'fCountIntercoForReserves': [0,1], 'sIntercoReserveContributionPct': [0, 0.5]} """
        # Iterate over the Settings to change
        for k, vals in settings_sensi.items():
            # Iterate over the values
            for val in vals:

                # Reading the initial value
                df = pd.read_csv(s['baseline'][param])

                # Modifying the value
                df.loc[df['Abbreviation'] == k, 'Value'] = val

                val_name = str(val).replace('.', '')
                suffix = f"{k}_{val_name}"
                scenario_name = f'baseline~{param}~{suffix}'
                path_file = _generate_sensitivity_filepath(s['baseline'][param], suffix)
                df.to_csv(path_file, index=False)

                # Put in the scenario dir
                s[scenario_name] = s['baseline'].copy()
                s[scenario_name][param] = path_file

    param = 'y'
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):  # testing implications of year definition
        df = pd.read_csv(s['baseline'][param])
        # Check if all years have been include in the analysis
        if not (df[param].diff().dropna() == 1).all():
            t = pd.Series(range(df['y'].min(), df['y'].max() + 1, 1))

            name = f'baseline~{param}~full'
            path_file = _generate_sensitivity_filepath(s['baseline'][param], 'full')
            t.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[name] = s['baseline'].copy()
            s[name][param] = path_file

        # Make only first and last year simulation
        if len(df) > 2:
            t = pd.Series([df['y'].min(), df['y'].max()])

            name = f'baseline~{param}~reduced'
            path_file = _generate_sensitivity_filepath(s['baseline'][param], 'reduced')
            t.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[name] = s['baseline'].copy()
            s[name][param] = path_file

    param = 'pDemandForecast'  # testing implications of demand forecast
    if sensitivity.get(param) and not (isinstance(sensitivity[param], float) and math.isnan(sensitivity[param])):
        raw_val = sensitivity[param]
        if isinstance(raw_val, str):
            values = [v.strip() for v in raw_val.split('&') if v.strip()]
        elif isinstance(raw_val, numbers.Real):
            values = [raw_val]
        else:
            raise ValueError(f"Unsupported sensitivity value for {param}: {raw_val}")

        demand_forecast_sensi = [float(v) for v in values]

        # Iterate over all existing scenarios
        new_scenarios = {}
        for scenario in base_scenarios:
            for val in demand_forecast_sensi:
                df = pd.read_csv(s[scenario][param])

                cols = [i for i in df.columns if i not in ['z', 'type']]
                df[cols] = df[cols].astype(float)
                df.loc[:, cols] *= (1 + val)

                val_name = str(val).replace('.', '')
                scenario_name = _with_assessment_suffix(scenario, f"{param}_{val_name}")

                path_file = _generate_sensitivity_filepath(s[scenario][param], val_name)
                df.to_csv(path_file, index=False)

                # Put in the scenario dir
                new_scenarios[scenario_name] = s[scenario].copy()
                new_scenarios[scenario_name][param] = path_file

        s.update(new_scenarios)

    param = 'TradePrice'  # testing implications of external trade prices
    if sensitivity.get(param) and not (isinstance(sensitivity[param], float) and math.isnan(sensitivity[param])):
        raw_val = sensitivity[param]
        if isinstance(raw_val, str):
            values = [v.strip() for v in raw_val.split('&') if v.strip()]
        elif isinstance(raw_val, numbers.Real):
            values = [raw_val]
        else:
            raise ValueError(f"Unsupported sensitivity value for {param}: {raw_val}")

        trade_price_sensi = [float(v) for v in values]

        new_scenarios = {}
        for scenario in base_scenarios:
            for val in trade_price_sensi:
                df = pd.read_csv(s[scenario]['pTradePrice'])
                cols = [c for c in df.columns if c not in ['zext', 'q', 'd', 'y']]
                df[cols] = df[cols].astype(float)
                df.loc[:, cols] *= (1 + val)

                val_name = str(val).replace('.', '')
                scenario_name = _with_assessment_suffix(scenario, f"{param}_{val_name}")
                path_file = _generate_sensitivity_filepath(s[scenario]['pTradePrice'], val_name)
                df.to_csv(path_file, index=False)

                new_scenarios[scenario_name] = s[scenario].copy()
                new_scenarios[scenario_name]['pTradePrice'] = path_file

        s.update(new_scenarios)

    param = 'pDemandProfile'  # testing implications of having a flat profile
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):
        df = pd.read_csv(s['baseline'][param])

        cols = [i for i in df.columns if i not in ['zone', 'q', 'd', 't']]
        df.loc[:, cols] = 1 / 24

        name = f'baseline~{param}~flat'
        path_file = _generate_sensitivity_filepath(s['baseline'][param], 'flat')
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[name] = s['baseline'].copy()
        s[name][param] = path_file

    param = 'pAvailabilityDefault'  # testing implications of a change in availability for thermal power plants (default values, custom values stay the same)
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):
        availability_sensi = [0.3]

        for val in availability_sensi:
            df = pd.read_csv(s['baseline'][param])

            df.loc[df['f'].isin(['Coal', 'Gas', 'Diesel', 'HFO', 'LFO']), [i for i in df.columns if
                                                                              i not in ['zone', 'tech', 'f']]] = val

            val_name = str(val).replace('.', '')
            scenario_name = f'baseline~{param}~{val_name}'
            path_file = _generate_sensitivity_filepath(s['baseline'][param], val_name)
            df.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[scenario_name] = s['baseline'].copy()
            s[scenario_name][param] = path_file

    param = 'pCapexTrajectoriesDefault'  # testing implications of constant capex trajectories
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):

        df = pd.read_csv(s['baseline'][param])

        cols = [i for i in df.columns if i not in ['zone', 'tech', 'f']]
        df.loc[:, cols] = 1

        scenario_name = f'baseline~{param}~flat'
        path_file = _generate_sensitivity_filepath(s['baseline'][param], 'flat')
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[scenario_name] = s['baseline'].copy()
        s[scenario_name][param] = path_file

    param = 'pFuelPrice'  # testing implications of fuel prices
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):
        fuel_price_sensi = [-0.2, 0.2]

        for val in fuel_price_sensi:
            df = pd.read_csv(s['baseline'][param])

            cols = [i for i in df.columns if i not in ['c', 'f']]
            df.loc[:, cols] *= (1 + val)

            val_name = str(val).replace('.', '')
            scenario_name = f'baseline~{param}~{val_name}'
            path_file = _generate_sensitivity_filepath(s['baseline'][param], val_name)
            df.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[scenario_name] = s['baseline'].copy()
            s[scenario_name][param] = path_file

    param = 'ResLimShare'  # testing implications of contribution to reserves
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):
        parameter = 'pGenDataInputDefault'
        reslimshare_sensi = [-0.5, -1]
        for val in reslimshare_sensi:

            df = pd.read_csv(s['baseline'][parameter])
            df.loc[df['f'].isin(['Coal', 'Gas', 'HFO', 'LFO', 'Import']), param] *= (1 + val)

            val_name = str(val).replace('.', '')
            scenario_name = f'baseline~{parameter}~{param}~{val_name}'
            path_file = _generate_sensitivity_filepath(s['baseline'][parameter], f'{param}_{val_name}')
            df.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[scenario_name] = s['baseline'].copy()
            s[scenario_name][parameter] = path_file

    param = 'BuildLimitperYear'  # testing implications of limitations of build per year
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):
        parameter = 'pGenDataInput'

        df = pd.read_csv(s['baseline'][parameter])
        # Remove any built limitation per year
        df.loc[df.loc[:, 'Status'] == 3, param] = df.loc[df.loc[:, 'Status'] == 3, 'Capacity']

        scenario_name = f'baseline~{parameter}~{param}~removed'
        path_file = _generate_sensitivity_filepath(s['baseline'][parameter], f'{param}_removed')
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[scenario_name] = s['baseline'].copy()
        s[scenario_name][parameter] = path_file

        # For gen with Candidate name, status 3, fuel Solar, Wind, Battery, divide the BuildLimitperYear by 2
        df = pd.read_csv(s['baseline'][parameter])

        df.loc[(df['gen'].str.contains('Candidate')) & (df['Status'] == 3) & (df['f'].isin(['Solar', 'Wind', 'Battery'])), param] /= 2

        scenario_name = f'baseline~{parameter}~{param}~reduced'
        path_file = _generate_sensitivity_filepath(s['baseline'][parameter], f'{param}_reduced')
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[scenario_name] = s['baseline'].copy()
        s[scenario_name][parameter] = path_file
                
    param = 'delayedHydro'
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):  # testing implications of delayed hydro projects
        df = pd.read_csv(s['baseline']['pGenDataInput'])
        # Add 5 years delay to all fuel Water projects more than 1 GW Capacity if status is 2 or 3
        df.loc[(df['f'] == 'Water') & (df['Capacity'] > 1000) & (df['Status'].isin([2, 3])), 'StYr'] += 5

        scenario_name = f'baseline~{param}~5years'
        path_file = _generate_sensitivity_filepath(s['baseline']['pGenDataInput'], f'{param}_5years')
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[scenario_name] = s['baseline'].copy()
        s[scenario_name]['pGenDataInput'] = path_file

    param = 'delayedHydro_alt'
    if sensitivity.get(param) and not math.isnan(float(sensitivity[param])):  # testing implications of delayed hydro projects
        df = pd.read_csv(s['baseline']['pGenDataInput'])
        # Add 3 years delay to all fuel Water projects more than 400 MW Capacity if status is 2 or 3
        df.loc[(df['fuel'] == 'Water') & (df['Capacity'] > 400) & (df['Status'].isin([2, 3])), 'StYr'] += 3
        
        scenario_name = f'{param}'
        path_file = _generate_sensitivity_filepath(s['baseline']['pGenDataInput'], f'{param}_3years')
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[scenario_name] = s['baseline'].copy()
        s[scenario_name]['pGenDataInput'] = path_file


    param = 'delayedTransmission'
    if sensitivity.get(param) and not math.isnan(float(sensitivity[param])):  # testing implications of delayed transmission projects
        df = pd.read_csv(s['baseline']['pNewTransmission'])

        # Add 2 years delay to all transmission projects starting in 2030
        df.loc[(df['EarliestEntry'] > 2026), 'EarliestEntry'] += 2

        # Creating a new folder
        folder_sensi = os.path.join(os.path.dirname(s['baseline']['pNewTransmission']), 'sensitivity')
        if not os.path.exists(folder_sensi):
            os.mkdir(folder_sensi)
        scenario_name = f'{param}'
        path_file = os.path.basename(s['baseline']['pNewTransmission']).replace('pNewTransmission', scenario_name)
        path_file = os.path.join(folder_sensi, path_file)
        # Write the modified file
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[scenario_name] = s['baseline'].copy()
        s[scenario_name]['pNewTransmission'] = path_file

    
    param = 'pVREProfile'  # testing implications of a change in VRE production

    if sensitivity.get(param) and not math.isnan(sensitivity[param]):
        capacity_factor_sensi = [-0.2, 0.2]

        for val in capacity_factor_sensi:
            df = pd.read_csv(s['baseline'][param])
            cols = [i for i in df.columns if i not in ['z', 'tech', 'q', 'd']]
            df[cols] = (df[cols] * (1 + val)).clip(upper=1)

            val_name = str(val).replace('.', '')
            scenario_name = f'baseline~{param}~{val_name}'
            path_file = _generate_sensitivity_filepath(s['baseline'][param], val_name)
            df.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[scenario_name] = s['baseline'].copy()
            s[scenario_name][param] = path_file

    return s

def perform_assessment(generator_assessment, s):
    try:
        # Iterate over all scenarios to generate a counterfactual scenario without the project(s)
        new_s = {}
        for scenario in s.keys():

            # Reading the initial value
            df = pd.read_csv(s[scenario]['pGenDataInput'])

            # Remove project(s) in generator_assessment
            df = df.loc[~df['gen'].isin(generator_assessment)]

            # Write the modified file
            name = '-'.join(generator_assessment).replace(' ', '')
            path_file = _generate_assessment_filepath(s[scenario]['pGenDataInput'], name)
            df.to_csv(path_file, index=False)

            # Put in the scenario specification dictionary
            new_s[f'{scenario}_wo_{name}'] = s[scenario].copy()
            new_s[f'{scenario}_wo_{name}']['pGenDataInput'] = path_file

    except Exception:
        raise KeyError('Error in generator_assessment features')

    s.update(new_s)

    return s


def _merge_input_files(baseline_path: Path, project_path: Path, output_path: Path) -> Path:
    """
    Merge a project delta file into a baseline input file.
    Works for both pGenDataInput and pStorageDataInput.

    - Existing records (matching g,z,tech,f): overwritten with project values
    - New records: added

    Parameters
    ----------
    baseline_path : Path
        Path to the baseline CSV file
    project_path : Path
        Path to the project delta CSV file
    output_path : Path
        Path where the merged file will be written

    Returns
    -------
    Path
        Path to the merged output file
    """
    baseline_df = pd.read_csv(baseline_path)
    project_df = pd.read_csv(project_path)

    # Handle column name variations (g vs gen)
    g_col = 'g' if 'g' in baseline_df.columns else 'gen'
    if g_col not in project_df.columns:
        alt_col = 'gen' if g_col == 'g' else 'g'
        if alt_col in project_df.columns:
            project_df = project_df.rename(columns={alt_col: g_col})

    # Primary key for matching
    key_cols = [g_col, 'z', 'tech', 'f']

    merged = pd.concat([baseline_df, project_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=key_cols, keep='last')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path


def _merge_multiple_input_files(
    baseline_path: Path,
    project_paths: list,
    output_path: Path
) -> Path:
    """
    Merge multiple project delta files into a baseline input file.
    Projects are merged sequentially; later projects override earlier ones on conflicts.

    Parameters
    ----------
    baseline_path : Path
        Path to the baseline CSV file
    project_paths : list
        List of project delta CSV files to merge (in order)
    output_path : Path
        Path where the merged file will be written

    Returns
    -------
    Path
        Path to the merged output file
    """
    merged_df = pd.read_csv(baseline_path)
    g_col = 'g' if 'g' in merged_df.columns else 'gen'
    key_cols = [g_col, 'z', 'tech', 'f']

    for project_path in project_paths:
        project_df = pd.read_csv(project_path)
        if g_col not in project_df.columns:
            alt_col = 'gen' if g_col == 'g' else 'g'
            if alt_col in project_df.columns:
                project_df = project_df.rename(columns={alt_col: g_col})
        merged_df = pd.concat([merged_df, project_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=key_cols, keep='last')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    return output_path


def perform_project_assessment(project_assessment, s):
    """
    Build assessment scenarios using delta project files.

    Supports multiple projects (space-separated):
        --project_assessment "ers epbih ephzhb"

    For each project, merges:
        - pGenDataInput_{project}.csv into baseline pGenDataInput
        - pStorageDataInput_{project}.csv into baseline pStorageDataInput (if exists)

    When multiple projects are specified, also creates a combined scenario
    with all projects merged together (baseline@all).

    Parameters
    ----------
    project_assessment : str
        Space-separated list of project names (suffixes).
        Example: "ers epbih ephzhb"
    s : dict
        Scenario dictionary

    Returns
    -------
    dict
        Updated scenario dictionary with new project assessment scenarios
    """
    new_s = {}
    baseline_path = Path(s['baseline']['pGenDataInput'])
    baseline_dir = baseline_path.parent
    assessment_folder = _get_assessment_folder(baseline_path)

    # Support multiple projects (space-separated)
    projects = project_assessment.split()

    # Collect project file paths for combined scenario
    gen_project_paths = []
    storage_project_paths = []

    for project in projects:
        suffix = project if project.startswith('_') else f"_{project}"
        label = project.strip('_') or "project"

        # Find delta files
        gen_project = baseline_dir / f"pGenDataInput{suffix}.csv"
        storage_project = baseline_dir / f"pStorageDataInput{suffix}.csv"

        if not gen_project.exists():
            raise FileNotFoundError(f"Project file {gen_project.resolve()} not found.")

        # Collect paths for combined scenario
        gen_project_paths.append(gen_project)
        if storage_project.exists():
            storage_project_paths.append(storage_project)

        for scenario in list(s.keys()):
            scenario_name = f"{scenario}@{label}"
            new_s[scenario_name] = s[scenario].copy()

            # Merge pGenDataInput
            current_gen = Path(s[scenario]['pGenDataInput'])
            merged_gen = assessment_folder / f"pGenDataInput_merged_{label}.csv"
            _merge_input_files(current_gen, gen_project, merged_gen)
            new_s[scenario_name]['pGenDataInput'] = str(merged_gen)

            # Merge pStorageDataInput (if delta exists)
            if storage_project.exists() and 'pStorageDataInput' in s[scenario]:
                current_storage = Path(s[scenario]['pStorageDataInput'])
                merged_storage = assessment_folder / f"pStorageDataInput_merged_{label}.csv"
                _merge_input_files(current_storage, storage_project, merged_storage)
                new_s[scenario_name]['pStorageDataInput'] = str(merged_storage)

    # Create combined scenario when multiple projects specified
    if len(projects) > 1:
        combined_label = "all_combined" if "all" in [p.strip('_') for p in projects] else "all"

        for scenario in list(s.keys()):
            scenario_name = f"{scenario}@{combined_label}"
            new_s[scenario_name] = s[scenario].copy()

            # Merge all pGenDataInput files
            current_gen = Path(s[scenario]['pGenDataInput'])
            merged_gen = assessment_folder / f"pGenDataInput_merged_{combined_label}.csv"
            _merge_multiple_input_files(current_gen, gen_project_paths, merged_gen)
            new_s[scenario_name]['pGenDataInput'] = str(merged_gen)

            # Merge all pStorageDataInput files (if any exist)
            if storage_project_paths and 'pStorageDataInput' in s[scenario]:
                current_storage = Path(s[scenario]['pStorageDataInput'])
                merged_storage = assessment_folder / f"pStorageDataInput_merged_{combined_label}.csv"
                _merge_multiple_input_files(current_storage, storage_project_paths, merged_storage)
                new_s[scenario_name]['pStorageDataInput'] = str(merged_storage)

    s.update(new_s)
    return s

def perform_interco_assessment(interco_assessment, s, delay=5):
    try:
        # Iterate over all scenarios to generate a counterfactual scenario without the project(s)
        new_s = {}
        for scenario in s.keys():

            # Reading the initial value
            df = pd.read_csv(s[scenario]['pNewTransmission'])
            print(df.head())
            # Create a helper column with standardized "From-To" or "To-From" format
            df['interco_key'] = df.apply(lambda row: f"{row['From']}-{row['To']}", axis=1)
            print(df.head())
            # Remove project(s) in interco_assessment
            df_filtered = df[~df['interco_key'].isin(interco_assessment)].drop(columns='interco_key')
            print(df_filtered.head())
            # Write the modified file
            name = '-'.join(interco_assessment).replace(' ', '')
            path_file = _generate_assessment_filepath(s[scenario]['pNewTransmission'], name)
            df_filtered.to_csv(path_file, index=False)

            # Put in the scenario specification dictionary
            new_s[f'{scenario}_wo_{name}'] = s[scenario].copy()
            new_s[f'{scenario}_wo_{name}']['pNewTransmission'] = path_file

            if False:
                # Delayed project implementation
                df_delay = df.copy()
                df_delay.loc[df_delay['interco_key'].isin(interco_assessment), 'EarliestEntry'] += delay
                df_delay = df_delay.drop(columns='interco_key')
                path_file = _generate_assessment_filepath(s[scenario]['pNewTransmission'], f'{name}_delay{delay}')
                df_delay.to_csv(path_file, index=False)
                # Put in the scenario specification dictionary
                new_s[f'{scenario}_{name}_delay{delay}'] = s[scenario].copy()
                new_s[f'{scenario}_{name}_delay{delay}']['pNewTransmission'] = path_file

                # Reduce capacity of the interconnection to 50% of the original value
                df_reduced = df.copy()
                df_reduced.loc[df_reduced['interco_key'].isin(interco_assessment), 'CapacityPerLine'] *= 0.5
                df_reduced = df_reduced.drop(columns='interco_key')
                path_file = _generate_assessment_filepath(s[scenario]['pNewTransmission'], f'{name}_reduced')
                df_reduced.to_csv(path_file, index=False)
                # Put in the scenario specification dictionary
                new_s[f'{scenario}_{name}_reduced'] = s[scenario].copy()
                new_s[f'{scenario}_{name}_reduced']['pNewTransmission'] = path_file

    except Exception:
        raise KeyError('Error in interco_assessment features')

    s.update(new_s)

    return s
