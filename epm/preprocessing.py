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
import math
import numpy as np
import pandas as pd
import os
from pathlib import Path

try:
    import chaospy  # optional dependency for Monte Carlo analysis
except ImportError:  # pragma: no cover - handled at runtime when Monte Carlo is requested
    chaospy = None


def _ensure_chaospy():
    """
    Lazily import chaospy only when Monte Carlo features are requested.

    Returns
    -------
    module
        The imported chaospy module.

    Raises
    ------
    ImportError
        If chaospy is not installed and Monte Carlo is requested.
    """
    if chaospy is None:
        raise ImportError(
            "Monte Carlo analysis requires the optional dependency 'chaospy'. "
            "Install it with `pip install chaospy` and rerun the command."
        )
    return chaospy


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


def _with_assessment_suffix(scenario_name, suffix):
    """
    Append a sensitivity suffix while preserving project assessment markers (anything after '@').
    """
    if '@' in scenario_name:
        base, assessment = scenario_name.split('@', 1)
        return f"{base}_{suffix}@{assessment}"
    return f"{scenario_name}_{suffix}"


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


def _generate_montecarlo_filepath(base_path, param, val):
    """
    Generate a montecarlo file path with proper cross-platform handling.

    Parameters
    ----------
    base_path : str or Path
        Path to the original file
    param : str
        Parameter name (e.g., 'pFuelPrice')
    val : float
        Value used for this Monte Carlo sample

    Returns
    -------
    str
        Full path to the new montecarlo file
    """
    base_path = Path(base_path)
    folder_mc = _get_montecarlo_folder(base_path)
    # Create new filename with value (replace . with p for valid filename)
    val_str = str(val).replace('.', 'p')
    new_filename = f"{param}_{val_str}.csv"
    return str(folder_mc / new_filename)


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

def define_samples(df_uncertainties, nb_samples):
    """
    Generate a joint distribution and samples from a DataFrame defining uncertainty bounds.

    Parameters
    ----------
    df_uncertainties : pd.DataFrame
        Must contain columns: 'feature', 'type', 'lowerbound', 'upperbound'.
    nb_samples : int
        Number of samples to draw.

    Returns
    -------
    tuple
        (NamedJ distribution object, dict of samples keyed by a readable string for each sample)
    """
    cp = _ensure_chaospy()
    uncertainties = {}
    zone_mapping = {}
    chaospy_distributions = [
        name for name in dir(cp)
        if callable(getattr(cp, name)) and hasattr(getattr(cp, name), '__module__')
           and 'chaospy' in getattr(cp, name).__module__.lower()
    ]
    for _, row in df_uncertainties.iterrows():
        feature, type, lowerbound, upperbound = row['feature'], row['type'], row['lowerbound'], row['upperbound']
        assert type in chaospy_distributions, f'Distribution types is not allowed by the chaopsy package. Distribution type should belong to {chaospy_distributions}'
        uncertainties[feature] = {
            'type': type,
            'args': (lowerbound, upperbound)
        }
        # Getting zones concerned by the uncertainty
        zones = row['zones'] if pd.notna(row.get('zones', None)) else 'ALL'
        if isinstance(zones, str):
            zone_list = [z.strip() for z in zones.split(';')]
        else:
            zone_list = ['ALL']
        zone_mapping[feature] = zone_list
    distribution = NamedJ(uncertainties)

    samples = distribution.sample(size=nb_samples, rule='halton')
    samples = {
        f'{"_".join([f"{idx}{samples.loc[idx, col]:.3f}" for idx in samples.index])}': {
            idx: round(samples.loc[idx, col], 3) for idx in samples.index
        }
        for col in samples.columns
    }
    return distribution, samples, zone_mapping

def create_scenarios_montecarlo(samples, s, zone_mapping):
    """
    Generate new scenarios for Monte Carlo analysis by modifying baseline input files
    based on provided uncertainty samples.

    This function creates new input files (under a `montecarlo/` subdirectory) for each
    scenario sample, applies parameter-specific transformations (e.g., scaling demand or
    fuel prices), and updates the scenario dictionary accordingly.

    Parameters
    ----------
    samples : dict
        Dictionary of samples where keys are scenario names and values are dicts
        mapping uncertain variable names (e.g., 'fossilfuel', 'demand') to sample values.
    s : dict
        Dictionary of scenarios where 'baseline' must be defined. Each scenario is a
        dictionary of parameter file paths.

    Returns
    -------
    dict
        Updated dictionary of scenarios, including new scenarios generated from samples.
    """

    def save_new_dataframe(df, s, param, val, name='baseline'):
        """
        Helper function to save a modified DataFrame to a new file and update the scenario path.

        Parameters
        ----------
        df : pd.DataFrame
            The modified DataFrame to save.
        scenario_dict : dict
            The main scenario dictionary.
        param : str
            The name of the parameter being modified (e.g., 'pFuelPrice').
        val : float
            The value used for this Monte Carlo sample (used in naming).
        """
        path_file = _generate_montecarlo_filepath(s[name][param], param, val)
        df.to_csv(path_file, index=True)

        s[name_scenario][param] = path_file
        return s

    list_initial_scenarios = list(s.keys()).copy()
    scenarios_montecarlo = {}
    for name in list_initial_scenarios:

        for name_scenario, sample in samples.items():
            name_scenario = name_scenario.replace('.', 'p')
            name_scenario = name + '_' + name_scenario
            # Put in the scenario dir
            scenarios_montecarlo[name_scenario] = s[name].copy()
            for key, val in sample.items():
                affected_zones = zone_mapping.get(key)
                if key == 'fossilfuel':
                    param = 'pFuelPrice'
                    price_df = pd.read_csv(s[name][param], index_col=[0, 1]).copy()
                    price_df.columns = price_df.columns.astype(int)
                    tech_list = ["Diesel", "HFO", "Coal", "Gas", "LNG"]
                    idx = pd.IndexSlice
                    if 'ALL' in affected_zones:
                        price_df.loc[idx[:, tech_list], :] *= (1 + val)
                    else:
                        price_df.loc[idx[affected_zones, tech_list], :] *= (1 + val)
                    save_new_dataframe(price_df, scenarios_montecarlo, param, val, name=name_scenario)

                if key == 'demand':
                    param = 'pDemandForecast'
                    demand_df = pd.read_csv(s[name][param], index_col=[0, 1]).copy()
                    demand_df.columns = demand_df.columns.astype(int)

                    cols = [i for i in demand_df.columns if i not in ['zone', 'type']]
                    idx = pd.IndexSlice
                    if 'ALL' in affected_zones:
                        demand_df.loc[:, cols] *= (1 + val)
                    else:
                        demand_df.loc[idx[affected_zones, :], cols] *= (1 + val)

                    save_new_dataframe(demand_df, scenarios_montecarlo, param, val, name=name_scenario)

                if key == 'hydro':
                    # First handling default values
                    param = 'pAvailabilityDefault'
                    availability_default = pd.read_csv(s[name][param], index_col=[0, 1, 2]).copy()
                    # availability_default.columns = availability_default.columns.astype(float)
                    cols = [i for i in availability_default.columns if i not in ['zone', 'type', 'fuel']]
                    tech_list = ['ROR', 'ReservoirHydro']
                    if 'ALL' in affected_zones:
                        mask = availability_default.index.get_level_values('tech').isin(tech_list)

                    else:
                        mask = (availability_default.index.get_level_values('zone').isin(affected_zones)) & \
                               (availability_default.index.get_level_values('tech').isin(tech_list))

                    availability_default.loc[mask, cols] *= (1 + val)

                    save_new_dataframe(availability_default, scenarios_montecarlo, param, val, name=name_scenario)

                    # Then handling custom values
                    param = 'pAvailability'
                    param_to_merge = 'pGenDataInput'
                    availability_custom = pd.read_csv(s[name][param], index_col=[0]).copy()

                    gendata = pd.read_csv(s[name][param_to_merge], index_col=[0,1,2,3]).copy()
                    gendata = gendata.reset_index()[['gen', 'zone', 'tech', 'fuel']]
                    availability_custom = availability_custom.reset_index().merge(gendata, on=['gen'], how='left')
                    availability_custom.set_index(['gen', 'zone', 'tech', 'fuel'], inplace=True)

                    cols = [i for i in availability_custom.columns if i not in ['zone', 'type', 'fuel']]
                    if 'ALL' in affected_zones:
                        mask = availability_custom.index.get_level_values('tech').isin(tech_list)

                    else:
                        mask = (availability_custom.index.get_level_values('zone').isin(affected_zones)) & \
                               (availability_custom.index.get_level_values('tech').isin(tech_list))

                    availability_custom.loc[mask, cols] *= (1 + val)
                    availability_custom = availability_custom.droplevel(['zone', 'tech', 'fuel'], axis=0)

                    save_new_dataframe(availability_custom, scenarios_montecarlo, param, val, name=name_scenario)

    return s, scenarios_montecarlo

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
        s[name] = s['baseline'].copy()
        s[name]['pSettings'] = path_file

        #----------------------------------------

        df = pd.read_csv(s['baseline']['pSettings'])
        # fAllowTransferExpansion
        name = 'NoInterconnectionExpansion'
        df.loc[df['Abbreviation'] == 'fAllowTransferExpansion', 'Value'] = 0

        path_file = _generate_sensitivity_filepath(s['baseline']['pSettings'], name)
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[name] = s['baseline'].copy()
        s[name]['pSettings'] = path_file

        #----------------------------------------

        df = pd.read_csv(s['baseline']['pSettings'])
        # OptimalInterconnection with fRemoveInternalTransferLimit
        name = 'OptimalInterconnection'
        df.loc[df['Abbreviation'] == 'fRemoveInternalTransferLimit', 'Value'] = 1

        path_file = _generate_sensitivity_filepath(s['baseline']['pSettings'], name)
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[name] = s['baseline'].copy()
        s[name]['pSettings'] = path_file
    
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
        s[name] = s['baseline'].copy()
        s[name]['pGenDataInput'] = path_file

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
                mask = (df['fuel'].str.lower() == fuel.lower()) & df['Status'].isin([2, 3])
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
                scenario_name = f'{param}_{suffix}'
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

            name = f'{param}_full'
            path_file = _generate_sensitivity_filepath(s['baseline'][param], 'full')
            t.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[name] = s['baseline'].copy()
            s[name][param] = path_file

        # Make only first and last year simulation
        if len(df) > 2:
            t = pd.Series([df['y'].min(), df['y'].max()])

            name = f'{param}_reduced'
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

                cols = [i for i in df.columns if i not in ['zone', 'type']]
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
                cols = [c for c in df.columns if c not in ['zext', 'q', 'daytype', 'y']]
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

        name = f'{param}_flat'
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

            df.loc[df['fuel'].isin(['Coal', 'Gas', 'Diesel', 'HFO', 'LFO']), [i for i in df.columns if
                                                                              i not in ['zone', 'tech', 'fuel']]] = val

            val_name = str(val).replace('.', '')
            scenario_name = f'{param}_{val_name}'
            path_file = _generate_sensitivity_filepath(s['baseline'][param], val_name)
            df.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[scenario_name] = s['baseline'].copy()
            s[scenario_name][param] = path_file

    param = 'pCapexTrajectoriesDefault'  # testing implications of constant capex trajectories
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):

        df = pd.read_csv(s['baseline'][param])

        cols = [i for i in df.columns if i not in ['zone', 'tech', 'fuel']]
        df.loc[:, cols] = 1

        scenario_name = f'{param}_flat'
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

            cols = [i for i in df.columns if i not in ['country', 'fuel']]
            df.loc[:, cols] *= (1 + val)

            val_name = str(val).replace('.', '')
            scenario_name = f'{param}_{val_name}'
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
            df.loc[df['fuel'].isin(['Coal', 'Gas', 'HFO', 'LFO', 'Import']), param] *= (1 + val)

            val_name = str(val).replace('.', '')
            scenario_name = f'{parameter}_{param}_{val_name}'
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

        scenario_name = f'{parameter}_{param}_removed'
        path_file = _generate_sensitivity_filepath(s['baseline'][parameter], f'{param}_removed')
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[scenario_name] = s['baseline'].copy()
        s[scenario_name][parameter] = path_file

        # For gen with Candidate name, status 3, fuel Solar, Wind, Battery, divide the BuildLimitperYear by 2
        df = pd.read_csv(s['baseline'][parameter])

        df.loc[(df['gen'].str.contains('Candidate')) & (df['Status'] == 3) & (df['fuel'].isin(['Solar', 'Wind', 'Battery'])), param] /= 2

        scenario_name = f'{parameter}_{param}_reduced'
        path_file = _generate_sensitivity_filepath(s['baseline'][parameter], f'{param}_reduced')
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[scenario_name] = s['baseline'].copy()
        s[scenario_name][parameter] = path_file
                
    param = 'delayedHydro'
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):  # testing implications of delayed hydro projects
        df = pd.read_csv(s['baseline']['pGenDataInput'])
        # Add 5 years delay to all fuel Water projects more than 1 GW Capacity if status is 2 or 3
        df.loc[(df['fuel'] == 'Water') & (df['Capacity'] > 1000) & (df['Status'].isin([2, 3])), 'StYr'] += 5

        scenario_name = f'{param}_5years'
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
        
        scenario_name = f'{param}_3years'
        path_file = _generate_sensitivity_filepath(s['baseline']['pGenDataInput'], f'{param}_3years')
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[scenario_name] = s['baseline'].copy()
        s[scenario_name]['pGenDataInput'] = path_file


    param = 'delayedTransmission'
    if sensitivity.get(param) and not math.isnan(float(sensitivity[param])):  # testing implications of delayed transmission projects
        df = pd.read_csv(s['baseline']['pNewTransmission'])
        print(df.head())
        # Add 2 years delay to all transmission projects starting in 2030
        df.loc[(df['EarliestEntry'] > 2026), 'EarliestEntry'] += 2

        # Creating a new folder
        folder_sensi = os.path.join(os.path.dirname(s['baseline']['pNewTransmission']), 'sensitivity')
        if not os.path.exists(folder_sensi):
            os.mkdir(folder_sensi)
        scenario_name = f'{param}_2years'
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
            cols = [i for i in df.columns if i not in ['zone', 'tech', 'q', 'd']]
            df[cols] = (df[cols] * (1 + val)).clip(upper=1)

            val_name = str(val).replace('.', '')
            scenario_name = f'{param}_{val_name}'
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


def perform_project_assessment(project_assessment, s):
    """
    Build assessment scenarios using an existing pGenDataInput variant (suffix or filename).
    Always looks for the project file relative to baseline's pGenDataInput location.
    """
    try:
        new_s = {}
        # Always use baseline path to find the project assessment file
        baseline_path = Path(s['baseline']['pGenDataInput'])
        baseline_dir = baseline_path.parent

        if Path(project_assessment).is_absolute():
            candidate = Path(project_assessment)
        else:
            if project_assessment.endswith(".csv"):
                candidate = baseline_dir / project_assessment
            else:
                suffix = project_assessment
                if not suffix.startswith('_'):
                    suffix = f"_{suffix}"
                candidate_name = f"{baseline_path.stem}{suffix}{baseline_path.suffix}"
                candidate = baseline_dir / candidate_name

        if not candidate.exists():
            raise FileNotFoundError(f"Project assessment file {candidate.resolve()} not found.")

        label = candidate.stem.replace('pGenDataInput', '').strip('_') or "project"

        for scenario in list(s.keys()):
            scenario_name = f"{scenario}@{label}"
            new_s[scenario_name] = s[scenario].copy()
            new_s[scenario_name]['pGenDataInput'] = str(candidate)

    except Exception:
        raise KeyError('Error in project_assessment features')

    s.update(new_s)
    return s

def perform_interco_assessment(interco_assessment, s, delay=5):
    try:
        # Iterate over all scenarios to generate a counterfactual scenario without the project(s)
        new_s = {}
        for scenario in s.keys():

            # Reading the initial value
            df = pd.read_csv(s[scenario]['pNewTransmission'])

            # Create a helper column with standardized "From-To" or "To-From" format
            df['interco_key'] = df.apply(lambda row: f"{row['From']}-{row['To']}", axis=1)

            # Remove project(s) in interco_assessment
            df_filtered = df[~df['interco_key'].isin(interco_assessment)].drop(columns='interco_key')

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
