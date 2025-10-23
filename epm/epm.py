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
    Claire Nicolas — c.nicolas@worldbank.org
**********************************************************************
"""


import os
import subprocess
import pandas as pd
import datetime
from multiprocessing import Pool
import shutil
from gams import GamsWorkspace
import json
import argparse
import math
from postprocessing.utils import path_to_extract_results
from postprocessing.postprocessing import postprocess_output
import re
from pathlib import Path
import sys
import numpy as np

try:
    import chaospy  # optional dependency for Monte Carlo analysis
except ImportError:  # pragma: no cover - handled at runtime when Monte Carlo is requested
    chaospy = None

# TODO: Add all cplex option and other simulation parameters that were in Looping.py

PATH_GAMS = {
    'path_main_file': 'main.gms',
    'path_base_file': 'base.gms',
    'path_report_file': 'generate_report.gms',
    'path_reader_file': 'input_readers.gms',
    'path_demand_file': 'generate_demand.gms',
    'path_hydrogen_file': 'hydrogen_module.gms',
    'path_cplex_file': 'cplex.opt'
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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


def normalize_path(df):
    """
    Converts file paths in a DataFrame column to a cross-platform format.

    Parameters:
    df (pd.DataFrame): The DataFrame containing file paths.
    column_name (str): The column containing file paths.

    Returns:
    pd.DataFrame: A DataFrame with normalized file paths.
    """
    if (isinstance(df, pd.DataFrame)) or (isinstance(df, pd.Series)):
        return df.map(lambda x: str(Path(x).as_posix()) if isinstance(x, (Path, str)) else x)
    else:
        assert isinstance(df, str), 'Type of df is not correct.'
        return str(Path(df).as_posix()) if isinstance(df, (Path, str)) else df



def launch_epm_checkpoint(scenario,
               scenario_name='',
               path_main_file='main.gms',
               path_base_file='base.gms',
               path_report_file='generate_report.gms',
               path_reader_file='input_readers.gms',
               path_demand_file='generate_demand.gms',
               path_hydrogen_file='hydrogen_module.gms',
               path_cplex_file='cplex.opt',
               folder_input=None,
               prefix='' #'simulation_'
               ):
    """
    Version with GAMS Control Python API - NOT WORKING CURRENTLY, UNDER DEVELOPMENT
    """

    def read_gams_model(model_path):
        """
        Read the GAMS model file and return its contents as a string.
        """
        with open(model_path, 'r') as file:
            return file.read()

    # Initialize GAMS workspace
    def initialize_workspace(model_directory, sys_dir=None):
        """
        Initialize the GAMS workspace with the specified system and working directories.
        """
        return GamsWorkspace(system_directory=sys_dir, working_directory=model_directory)

    # Create and run initial job from the GAMS model
    def create_checkpoint(ws, gams_model, options=None):
        """
        Create a checkpoint and run the initial job from the GAMS model.
        """
        cp = ws.add_checkpoint()
        job = ws.add_job_from_string(gams_model)
        if options:
            opt = ws.add_options()
            for k,value in options.items():
                opt.defines[k] = value
            job.run(opt, checkpoint=cp)
        else:
            job.run(checkpoint=cp)
        return cp, job

    # Arguments for GAMS
    options = {k: i for k, i in scenario.items()}
    options.update({
        'BASE_FILE': path_base_file,
        'REPORT_FILE': path_report_file,
        'READER_FILE': path_reader_file,
        'DEMAND_FILE': path_demand_file,
        'HYDROGEN_FILE': path_hydrogen_file,
        'FOLDER_INPUT': folder_input,
    })

    # Defining the GAMS workspace
    new_dir = os.path.abspath(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), os.pardir))
    os.chdir(new_dir)
    model_path, model_directory =  os.path.join(new_dir, 'main.gms'), new_dir # Get the paths
    gams_model = read_gams_model(model_path) # Open model
    ws = initialize_workspace(model_directory) # Opening a gams workspace
    cp, job = create_checkpoint(ws, gams_model, options=options) # cp capture the current state of the model and job the work going on
    warmstart = job.out_db

    result = None

    return result


def launch_epm(scenario,
               scenario_name='',
               path_main_file='main.gms',
               path_base_file='base.gms',
               path_report_file='generate_report.gms',
               path_reader_file='input_readers.gms',
               path_demand_file='generate_demand.gms',
               path_hydrogen_file='hydrogen_module.gms',
               path_cplex_file='cplex.opt',
               solver='MIP',
               folder_input=None,
               dict_montecarlo=None,
               prefix=''  # 'simulation_'
               ):
    """
    Launch the EPM model with the given scenario

    Parameters
    ----------
    scenario: pd.DataFrame
        A DataFrame with the scenario to run the model
    scenario_name: str, optional, default ''
        The name of the scenario
    path_main_file: str
        The path to the GAMS file to run
    path_base_file: str
        The path to the GAMS base file
    path_report_file: str
        The path to the GAMS report file
    path_cplex_file: str
        The path to the CPLEX file
    folder_input: str, optional, default None
    dict_montecarlo: dict, optional, default None
        Correspondence for solution when running montecarlo scenarios

    Returns
    -------
    None
        Output files are written to the scenario-specific folder.
    """

    # If no scenario name is provided, use the current date and time
    if scenario_name == '':
        scenario_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Each simulation has its own subfolder to store the results
    folder = f'{prefix}{scenario_name}'
    if not os.path.exists(folder):
        os.mkdir(folder)
    cwd = os.path.join(os.getcwd(), folder)

    # Copy and paste cplex file to the simulation folder
    if '_' in path_cplex_file.split('/')[-1]:
        new_file_path = os.path.join(cwd, 'cplex.opt')
        shutil.copy(path_cplex_file, new_file_path)
    else:
        shutil.copy(path_cplex_file, cwd)

    # Arguments for GAMS
    if dict_montecarlo is not None:  # running in Monte-Carlo setting
        scenario['reportshort'] = 1  # shorter report to save memory
        scenario['solvemode'] = 2    # solve without checkpoint
    path_args = ['--{} {}'.format(k, i) for k, i in scenario.items()]

    # Define the logfile name
    #logfile = os.path.join(cwd, 'main.log')
    logfile = f'{scenario_name}_main.log'

    options = [
        "LogOption 4", # Write log to standard output and log file
        f"LogFile {logfile}" # Specify the name of the log file
        ]

    if dict_montecarlo is not None:
        loadsolpath = os.path.join(os.path.abspath(os.path.join(cwd, os.pardir)), dict_montecarlo[scenario_name])
        options.extend(["--LOADSOLPATH {}".format(loadsolpath)])

    command = ["gams", path_main_file] + options + ["--BASE_FILE {}".format(path_base_file),
                                                    "--REPORT_FILE {}".format(path_report_file),
                                                    "--READER_FILE {}".format(path_reader_file),
                                                    "--DEMAND_FILE {}".format(path_demand_file),
                                                    "--HYDROGEN_FILE {}".format(path_hydrogen_file),
                                                    "--FOLDER_INPUT {}".format(folder_input),
                                                    "--MODELTYPE {}".format(solver)
                                                    ] + path_args

    # Print the command
    print("Command to execute:", command)

    if sys.platform.startswith("win"):  # If running on Windows
        rslt = subprocess.run(' '.join(command), cwd=cwd, shell=True)
    else:  # For Linux or macOS
        rslt = subprocess.run(command, cwd=cwd)

    if rslt.returncode != 0:
        raise RuntimeError('GAMS Error: check GAMS logs file ')

    return None


def launch_epm_multiprocess(df, scenario_name, path_gams, folder_input=None,
                            solver='MIP', dict_montecarlo=None):
    return launch_epm(df, scenario_name=scenario_name, folder_input=folder_input,
                      dict_montecarlo=dict_montecarlo, **path_gams, solver=solver)

def launch_epm_multi_scenarios(config='config.csv',
                               scenarios_specification='scenarios.csv',
                               selected_scenarios=['baseline'],
                               cpu=1, path_gams=None,
                               sensitivity=None,
                               montecarlo=False,
                               montecarlo_nb_samples=10,
                               uncertainties=None,
                               folder_input=None,
                               project_assessment=None,
                               interco_assessment=None,
                               simple=None,
                               solver='MIP'):
    """
    Launch the EPM model with multiple scenarios based on scenarios_specification

    Parameters
    ----------
    config: str, optional, default 'config.csv'
        Path to the CSV file with the baseline scenario
    scenarios_specification: str, optional, default 'scenarios_specification.csv'
        Path to the CSV file with the scenarios specification
    cpu: int, optional, default 1
        Number of CPUs to use
    selected_scenarios: list, optional, default None
        List of scenarios to run
    folder_input: str, optional, default None
        Folder where data input files are stored
    """

    working_directory = os.getcwd()

    # Read the scenario CSV file
    if path_gams is not None:  # path for required gams file is provided
        path_gams = {k: os.path.join(working_directory, i) for k, i in path_gams.items()}
    else:  # use default configuration
        path_gams = {k: os.path.join(working_directory, i) for k, i in PATH_GAMS.items()}
        
    # Create the full path folder input
    folder_input = os.path.join(os.getcwd(), 'input', folder_input) if folder_input else os.path.join(os.getcwd(), 'input')

    # Read configuration file
    config_path = os.path.join(folder_input, 'config.csv')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Configuration file {os.path.abspath(config_path)} not found.')

    required_columns = ['paramNames', 'file']
    try:
        config = pd.read_csv(config_path, usecols=required_columns)
    except ValueError as err:
        if 'Usecols do not match columns' in str(err):
            header_columns = pd.read_csv(config_path, nrows=0).columns
            missing_columns = [col for col in required_columns if col not in header_columns]
            print(f"Error: Missing required columns {missing_columns} in {os.path.abspath(config_path)}.")
            raise ValueError(f"Missing required columns {missing_columns} in {os.path.abspath(config_path)}.") from None
        raise
    config = config.set_index('paramNames')['file']
    config = config.dropna()
    # Normalize path
    config = normalize_path(config)

    # Add the baseline scenario
    s = {'baseline': config}

    # Add scenarios if scenarios_specification is defined
    if scenarios_specification is not None:
        if not os.path.exists(scenarios_specification):
            raise FileNotFoundError(f'Scenarios specification file {os.path.abspath(scenarios_specification)} not found.')
        scenarios = pd.read_csv(scenarios_specification).set_index('paramNames')

        scenarios = normalize_path(scenarios)

        # Generate scenario pd.Series for alternative scenario
        s.update({k: config.copy() for k in scenarios})
        for k in s.keys():
            if k != 'baseline':
                s[k].update(scenarios[k].dropna())

    # Select useful scenarios if selected_scenarios is defined
    if selected_scenarios is not None:
        s = {k: s[k] for k in selected_scenarios}

    # Add full path to the files
    for k in s.keys():
        s[k] = s[k].apply(lambda i: os.path.join(folder_input, i) if '.csv' in i else i)

    # Run sensitivity analysis if activated
    if sensitivity is not None:
        s = perform_sensitivity(sensitivity, s)

    # Set-up project assessment scenarios if activated
    if project_assessment is not None:
        s = perform_assessment(project_assessment, s)
        
    # Set-up interconnection assessment scenarios if activated
    if interco_assessment is not None:
        s = perform_interco_assessment(interco_assessment, s)

    # Run montecarlo analysis if activated
    if montecarlo:
        assert uncertainties is not None, "Monte Carlo analysis is activated, but uncertainties is set to None."
        initial_scenarios = list(s.keys())
        df_uncertainties = pd.read_csv(uncertainties)
        distribution, samples, zone_mapping = define_samples(df_uncertainties, nb_samples=montecarlo_nb_samples)
        s, scenarios_montecarlo = create_scenarios_montecarlo(samples, s, zone_mapping)
        dict_montecarlo = {key: key.split('_')[0] for key in scenarios_montecarlo.keys()}  # getting the correspondence between montecarlo scenario and initial scenario

    # Reduce complexity if activated
    if simple is not None:

        for k in s.keys():
            if 'y' in simple:
                # Limit years to first and last
                df = pd.read_csv(s[k]['y'])

                # Make only first and last year simulation
                t = pd.Series([df['y'].min(), df['y'].max()])

                # Creating a new folder
                folder_sensi = os.path.join(os.path.dirname(s[k]['y']), 'sensitivity')
                if not os.path.exists(folder_sensi):
                    os.mkdir(folder_sensi)
                name = 'y_reduced'
                path_file = os.path.basename(s[k]['y']).replace('y', name)
                path_file = os.path.join(folder_sensi, path_file)
                # Write the modified file
                t.to_csv(path_file, index=False)

                # Put in the scenario dir
                s[k]['y'] = path_file

            if 'DiscreteCap' in simple:
                # Remove DiscreteCap
                df = pd.read_csv(s[k]['pGenDataInput'])

                df.loc[:, 'DiscreteCap'] = 0

                # Creating a new folder
                folder_sensi = os.path.join(os.path.dirname(s[k]['pGenDataInput']), 'sensitivity')
                if not os.path.exists(folder_sensi):
                    os.mkdir(folder_sensi)
                path_file = os.path.basename(s[k]['pGenDataInput']).split('.')[0] + '_linear.csv'
                path_file = os.path.join(folder_sensi, path_file)
                # Write the modified file
                df.to_csv(path_file, index=False)

                # Put in the scenario dir
                s[k]['pGenDataInput'] = path_file

    # Create dir for simulation and change current working directory
    if 'output' not in os.listdir():
        os.mkdir('output')

    pre = 'simulations_run'
    if sensitivity is not None:
        pre = 'sensitivity_run'
    if project_assessment is not None:
        pre = 'project_assessment_run'
    folder = '{}_{}'.format(pre, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder = os.path.join('output', folder)
    if not os.path.exists(folder):
        os.mkdir(folder)
        print('Folder created:', folder)
    os.chdir(folder)

    # Export scenario.csv file
    df = pd.DataFrame(s).copy()
    # Extracts everything after '/epm/' in a specified column of a Pandas DataFrame.
    def extract_path(path):
        match = re.search(r"/epm/(.*)", path)
        return match.group(1) if match else path
    df = df.astype(str).map(extract_path)
    df.to_csv('input_scenarios.csv')

    if montecarlo:
        samples_mc = pd.DataFrame(samples)
        samples_mc.columns = samples_mc.columns.map(lambda col: col.replace('.', 'p'))
        samples_mc.to_csv('samples_montecarlo.csv')

    # Run EPM in multiprocess
    if not montecarlo:
        with Pool(cpu) as pool:
            result = pool.starmap(launch_epm_multiprocess,
                                  [(s[k], k, path_gams, folder_input, solver) for k in s.keys()])
    else:
        # First, run initial scenarios
        # Ensure config file has extended setting to save output
        with Pool(cpu) as pool:
            for k in s.keys():
                assert s[k]['solvemode'] == '1', 'Parameter solvemode should be set to 1 in the configuration to obtain extended output for the baseline scenarios in the Monte-Carlo analysis.'
            result = pool.starmap(launch_epm_multiprocess,
                                  [(s[k], k, path_gams, folder_input, solver) for k in s.keys()])
        # Modify config file to ensure limited output saved
        with Pool(cpu) as pool:  # running montecarlo scenarios in multiprocessing
            result = pool.starmap(launch_epm_multiprocess,
                                  [(scenarios_montecarlo[k], k, path_gams, folder_input, solver, dict_montecarlo) for k in scenarios_montecarlo.keys()])

    os.chdir(working_directory)

    return folder, result

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
        folder_mc = os.path.join(os.path.dirname(s[name][param]), 'montecarlo')
        if not os.path.exists(folder_mc):
            os.mkdir(folder_mc)

        new_name = f'{param}_{val}'
        new_name = new_name.replace('.', 'p') + '.csv'
        path_file = os.path.join(folder_mc, new_name)
        # Write the modified file
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
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):  # testing implications of interconnection mode
        
        # Creating a new folder
        folder_sensi = os.path.join(os.path.dirname(s['baseline']['pSettings']), 'sensitivity')
        if not os.path.exists(folder_sensi):
            os.mkdir(folder_sensi)
        
        df = pd.read_csv(s['baseline']['pSettings'])
        # Modifying the value if it's 1 put 0 and vice versa
        name = 'NoInterconnection'
        df.loc[df['Abbreviation'] == "fEnableInternalExchange", 'Value'] = 0

        path_file = os.path.basename(s['baseline']['pSettings']).replace('pSettings', f'pSettings_{name}')
        path_file = os.path.join(folder_sensi, path_file)
        # Write the modified file
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[name] = s['baseline'].copy()
        s[name]['pSettings'] = path_file
        
        #----------------------------------------
        
        df = pd.read_csv(s['baseline']['pSettings'])
        # fAllowTransferExpansion
        name = 'NoInterconnectionExpansion'
        df.loc[df['Abbreviation'] == 'fAllowTransferExpansion', 'Value'] = 0
        
        path_file = os.path.basename(s['baseline']['pSettings']).replace('pSettings', f'pSettings_{name}')
        path_file = os.path.join(folder_sensi, path_file)
        # Write the modified file
        df.to_csv(path_file, index=False)
        
        # Put in the scenario dir
        s[name] = s['baseline'].copy()
        s[name]['pSettings'] = path_file
        
        #----------------------------------------
        
        df = pd.read_csv(s['baseline']['pSettings'])
        # OptimalInterconnection with fRemoveInternalTransferLimit
        name = 'OptimalInterconnection'
        df.loc[df['Abbreviation'] == 'fRemoveInternalTransferLimit', 'Value'] = 1
        
        path_file = os.path.basename(s['baseline']['pSettings']).replace('pSettings', f'pSettings_{name}')
        path_file = os.path.join(folder_sensi, path_file)
        # Write the modified file
        df.to_csv(path_file, index=False)
        # Put in the scenario dir
        s[name] = s['baseline'].copy()
        s[name]['pSettings'] = path_file
    
    param = 'RemoveGenericTechnologies'
    if param in sensitivity and not (isinstance(sensitivity[param], float) and math.isnan(sensitivity[param])):
        
        df = pd.read_csv(s['baseline']['pGenDataInput'])
        # Create a list of technologies to remove that are in a string separated by '&'
        # For example: 'WindOnshore&WindOffshore&SolarPV' will be converted to ['WindOnshore', 'WindOffshore', 'SolarPV']
        techs_to_remove = sensitivity['RemoveGenericTechnologies'].split('&')
        # For tech that equal to sensitivity['RemoveGenericTechnologies'], status 3, and Candidate in the name, put BuildLimitperYear to 0
        mask = df['tech'].isin(techs_to_remove) & (df['Status'] == 3) & (df['gen'].str.contains('Candidate'))
        df.loc[mask, 'BuildLimitperYear'] = 0
        # Creating a new folder
        folder_sensi = os.path.join(os.path.dirname(s['baseline']['pGenDataInput']), 'sensitivity')
        if not os.path.exists(folder_sensi):
            os.mkdir(folder_sensi)
        name = 'RemoveGenericTechnologies'
        path_file = os.path.basename(s['baseline']['pGenDataInput']).replace('pGenDataInput', f'pGenDataInput_{name}')
        path_file = os.path.join(folder_sensi, path_file)
        # Write the modified file
        df.to_csv(path_file, index=False)
        # Put in the scenario dir
        s[name] = s['baseline'].copy()
        s[name]['pGenDataInput'] = path_file
        
    param = 'pSettings'
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):  # testing implications of some setting parameters
        settings_sensi = {'VoLL': [250],
                          'fApplyPlanningReserveConstraint': [0], 'sVREForecastErrorPct': [0, 0.3],
                          'zonal_spinning_reserve_constraints': [0],
                          'CostSurplus': [1, 5], 'CostCurtail': [1, 5], "fEnableInternalExchange": [0,1],
                          'fCountIntercoForReserves': [0,1], 'sIntercoReserveContributionPct': [0, 0.5]}

        # Iterate over the Settings to change
        for k, vals in settings_sensi.items():
            # Iterate over the values
            for val in vals:

                # Reading the initial value
                df = pd.read_csv(s['baseline'][param])

                # Modifying the value
                df.loc[df['Abbreviation'] == k, 'Value'] = val

                # Creating a new folder
                folder_sensi = os.path.join(os.path.dirname(s['baseline'][param]), 'sensitivity')
                if not os.path.exists(folder_sensi):
                    os.mkdir(folder_sensi)
                name = str(val).replace('.', '')
                name = f'{param}_{k}_{name}'
                path_file = os.path.basename(s['baseline'][param]).replace(param, name)
                path_file = os.path.join(folder_sensi, path_file)
                # Write the modified file
                df.to_csv(path_file, index=False)

                # Put in the scenario dir
                s[name] = s['baseline'].copy()
                s[name][param] = path_file

    param = 'y'
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):  # testing implications of year definition
        df = pd.read_csv(s['baseline'][param])
        # Check if all years have been include in the analysis
        if not (df[param].diff().dropna() == 1).all():
            t = pd.Series(range(df['y'].min(), df['y'].max() + 1, 1))

            # Creating a new folder
            folder_sensi = os.path.join(os.path.dirname(s['baseline'][param]), 'sensitivity')
            if not os.path.exists(folder_sensi):
                os.mkdir(folder_sensi)
            name = f'{param}_full'
            path_file = os.path.basename(s['baseline'][param]).replace(param, name)
            path_file = os.path.join(folder_sensi, path_file)
            # Write the modified file
            t.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[name] = s['baseline'].copy()
            s[name][param] = path_file

        # Make only first and last year simulation
        if len(df) > 2:
            t = pd.Series([df['y'].min(), df['y'].max()])

            # Creating a new folder
            folder_sensi = os.path.join(os.path.dirname(s['baseline'][param]), 'sensitivity')
            if not os.path.exists(folder_sensi):
                os.mkdir(folder_sensi)
            name = f'{param}_reduced'
            path_file = os.path.basename(s['baseline'][param]).replace(param, name)
            path_file = os.path.join(folder_sensi, path_file)
            # Write the modified file
            t.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[name] = s['baseline'].copy()
            s[name][param] = path_file

    param = 'pDemandForecast'  # testing implications of demand forecast
    if sensitivity.get(param) and not (isinstance(sensitivity[param], float) and math.isnan(sensitivity[param])):
        demand_forecast_sensi = [float(i) for i in sensitivity[param].split('&')]
        for val in demand_forecast_sensi:
            df = pd.read_csv(s['baseline'][param])

            cols = [i for i in df.columns if i not in ['zone', 'type']]
            df[cols] = df[cols].astype(float)
            df.loc[:, cols] *= (1 + val)

            # Creating a new folder
            folder_sensi = os.path.join(os.path.dirname(s['baseline'][param]), 'sensitivity')
            if not os.path.exists(folder_sensi):
                os.mkdir(folder_sensi)
            name = str(val).replace('.', '')
            name = f'{param}_{name}'
            path_file = os.path.basename(s['baseline'][param]).replace(param, name)
            path_file = os.path.join(folder_sensi, path_file)
            # Write the modified file
            df.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[name] = s['baseline'].copy()
            s[name][param] = path_file

    param = 'pDemandProfile'  # testing implications of having a flat profile
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):
        df = pd.read_csv(s['baseline'][param])

        cols = [i for i in df.columns if i not in ['zone', 'q', 'd', 't']]
        df.loc[:, cols], name = 1 / 24, 'flat'
        folder_sensi = os.path.join(os.path.dirname(s['baseline'][param]), 'sensitivity')
        if not os.path.exists(folder_sensi):
            os.mkdir(folder_sensi)

        name = f'{param}_{name}'
        path_file = os.path.basename(s['baseline'][param]).replace(param, name)
        path_file = os.path.join(folder_sensi, path_file)
        # Write the modified file
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

            # Creating a new folder
            folder_sensi = os.path.join(os.path.dirname(s['baseline'][param]), 'sensitivity')
            if not os.path.exists(folder_sensi):
                os.mkdir(folder_sensi)
            name = str(val).replace('.', '')
            name = f'{param}_{name}'
            path_file = os.path.basename(s['baseline'][param]).replace(param, name)
            path_file = os.path.join(folder_sensi, path_file)
            # Write the modified file
            df.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[name] = s['baseline'].copy()
            s[name][param] = path_file

    param = 'pCapexTrajectoriesDefault'  # testing implications of constant capex trajectories
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):

        df = pd.read_csv(s['baseline'][param])

        cols = [i for i in df.columns if i not in ['zone', 'tech', 'fuel']]
        df.loc[:, cols], name = 1, 'flat'
        folder_sensi = os.path.join(os.path.dirname(s['baseline'][param]), 'sensitivity')
        if not os.path.exists(folder_sensi):
            os.mkdir(folder_sensi)

        name = f'{param}_{name}'
        path_file = os.path.basename(s['baseline'][param]).replace(param, name)
        path_file = os.path.join(folder_sensi, path_file)
        # Write the modified file
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[name] = s['baseline'].copy()
        s[name][param] = path_file

    param = 'pFuelPrice'  # testing implications of fuel prices
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):
        fuel_price_sensi = [-0.2, 0.2]

        for val in fuel_price_sensi:
            df = pd.read_csv(s['baseline'][param])

            cols = [i for i in df.columns if i not in ['country', 'fuel']]
            df.loc[:, cols] *= (1 + val)

            # Creating a new folder
            folder_sensi = os.path.join(os.path.dirname(s['baseline'][param]), 'sensitivity')
            if not os.path.exists(folder_sensi):
                os.mkdir(folder_sensi)
            name = str(val).replace('.', '')
            name = f'{param}_{name}'
            path_file = os.path.basename(s['baseline'][param]).replace(param, name)
            path_file = os.path.join(folder_sensi, path_file)
            # Write the modified file
            df.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[name] = s['baseline'].copy()
            s[name][param] = path_file

    param = 'ResLimShare'  # testing implications of contribution to reserves
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):
        parameter = 'pGenDataInputDefault'
        reslimshare_sensi = [-0.5, -1]
        for val in reslimshare_sensi:

            df = pd.read_csv(s['baseline'][parameter])
            df.loc[df['fuel'].isin(['Coal', 'Gas', 'HFO', 'LFO', 'Import']), param] *= (1 + val)

            # Creating a new folder
            folder_sensi = os.path.join(os.path.dirname(s['baseline'][parameter]), 'sensitivity')
            if not os.path.exists(folder_sensi):
                os.mkdir(folder_sensi)
            name = str(val).replace('.', '')
            name = f'{parameter}_{param}_{name}'
            path_file = os.path.basename(s['baseline'][parameter]).replace(parameter, name)
            path_file = os.path.join(folder_sensi, path_file)
            # Write the modified file
            df.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[name] = s['baseline'].copy()
            s[name][parameter] = path_file

    param = 'BuildLimitperYear'  # testing implications of limitations of build per year
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):
        parameter = 'pGenDataInput'

        df = pd.read_csv(s['baseline'][parameter])
        # Remove any built limitation per year
        df.loc[df.loc[:, 'Status'] == 3, param]  = df.loc[df.loc[:, 'Status'] == 3, 'Capacity']

        # Creating a new folder
        folder_sensi = os.path.join(os.path.dirname(s['baseline'][parameter]), 'sensitivity')
        if not os.path.exists(folder_sensi):
            os.mkdir(folder_sensi)
        name = f'{parameter}_{param}_removed'
        path_file = os.path.basename(s['baseline'][parameter]).replace(parameter, name)
        path_file = os.path.join(folder_sensi, path_file)
        # Write the modified file
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[parameter] = s['baseline'].copy()
        s[parameter][parameter] = path_file
        
        # For gen with Candidate name, status 3, fuel Solar, Wind, Battery, divide the BuildLimitperYear by 2
        df = pd.read_csv(s['baseline'][parameter])

        df.loc[(df['gen'].str.contains('Candidate')) & (df['Status'] == 3) & (df['fuel'].isin(['Solar', 'Wind', 'Battery'])), param] /= 2
        
        name = f'{parameter}_{param}_reduced'
        path_file = os.path.basename(s['baseline'][parameter]).replace(parameter, name)
        path_file = os.path.join(folder_sensi, path_file)
        # Write the modified file
        df.to_csv(path_file, index=False)
        
        # Put in the scenario dir
        s[name] = s['baseline'].copy()
        s[name][parameter] = path_file
                
    param = 'delayedHydro'
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):  # testing implications of delayed hydro projects
        df = pd.read_csv(s['baseline']['pGenDataInput'])
        # Add 5 years delay to all fuel Water projects more than 1 GW Capacity if status is 2 or 3
        df.loc[(df['fuel'] == 'Water') & (df['Capacity'] > 1000) & (df['Status'].isin([2, 3])), 'StYr'] += 5
        
        # Creating a new folder
        folder_sensi = os.path.join(os.path.dirname(s['baseline']['pGenDataInput']), 'sensitivity')
        if not os.path.exists(folder_sensi):
            os.mkdir(folder_sensi)
        name = f'{param}_5years'
        path_file = os.path.basename(s['baseline']['pGenDataInput']).replace('pGenDataInput', name)
        path_file = os.path.join(folder_sensi, path_file)
        # Write the modified file
        df.to_csv(path_file, index=False)
        
        # Put in the scenario dir
        s[name] = s['baseline'].copy()
        s[name]['pGenDataInput'] = path_file
    
    param  = 'pVREProfile'  # testing implications of a change in VRE production
    if sensitivity.get(param) and not math.isnan(sensitivity[param]):
        capacity_factor_sensi = [-0.2, 0.2]

        for val in capacity_factor_sensi:
            df = pd.read_csv(s['baseline'][param])
            cols = [i for i in df.columns if i not in ['zone', 'tech', 'q', 'd']]
            df[cols] = (df[cols] * (1 + val)).clip(upper=1)

            # Creating a new folder
            folder_sensi = os.path.join(os.path.dirname(s['baseline'][param]), 'sensitivity')
            if not os.path.exists(folder_sensi):
                os.mkdir(folder_sensi)
            name = str(val).replace('.', '')
            name = f'{param}_{name}'
            path_file = os.path.basename(s['baseline'][param]).replace(param, name)
            path_file = os.path.join(folder_sensi, path_file)
            # Write the modified file
            df.to_csv(path_file, index=False)

            # Put in the scenario dir
            s[name] = s['baseline'].copy()
            s[name][param] = path_file

    return s

def perform_assessment(project_assessment, s):
    try:

        # Iterate over all scenarios to generate a counterfactual scenario without the project(s)
        new_s = {}
        for scenario in s.keys():
            
             # Create a specific folder to store the counterfactual scenario
            folder_assessment = os.path.join(os.path.dirname(s[scenario]['pGenDataInput']), 'assessment')
            if not os.path.exists(folder_assessment):
                os.mkdir(folder_assessment)
                print('Folder created:', folder_assessment)
            
            
            # Reading the initial value
            df = pd.read_csv(s[scenario]['pGenDataInput'])
            
            # Remove project(s) in project_assessment
            df = df.loc[~df['gen'].isin(project_assessment)]

            # Write the modified file
            name = '-'.join(project_assessment).replace(' ', '')
            path_file = os.path.basename(s[scenario]['pGenDataInput']).split('.')[0] + '_' + name + '.csv'
            path_file = os.path.join(folder_assessment, path_file)
            df.to_csv(path_file, index=False)

            # Put in the scenario specification dictionary
            new_s[f'{scenario}_wo_{name}'] = s[scenario].copy()
            new_s[f'{scenario}_wo_{name}']['pGenDataInput'] = path_file
                

    except Exception:
        raise KeyError('Error in project_assessment features')

    s.update(new_s)

    return s

def perform_interco_assessment(interco_assessment, s, delay=5):
    try:
        
    
        # Iterate over all scenarios to generate a counterfactual scenario without the project(s)
        new_s = {}
        for scenario in s.keys():
            
             # Create a specific folder to store the counterfactual scenario
            folder_assessment = os.path.join(os.path.dirname(s[scenario]['pNewTransmission']), 'assessment')
            if not os.path.exists(folder_assessment):
                os.mkdir(folder_assessment)
                print('Folder created:', folder_assessment)
            
            # Reading the initial value
            df = pd.read_csv(s[scenario]['pNewTransmission'])
            
            # Create a helper column with standardized "From-To" or "To-From" format
            df['interco_key'] = df.apply(lambda row: f"{row['From']}-{row['To']}", axis=1)
            
            # Remove project(s) in interco_assessment
            df_filtered = df[~df['interco_key'].isin(interco_assessment)].drop(columns='interco_key')

            # Write the modified file
            name = '-'.join(interco_assessment).replace(' ', '')
            path_file = os.path.basename(s[scenario]['pNewTransmission']).split('.')[0] + '_' + name + '.csv'
            path_file = os.path.join(folder_assessment, path_file)
            df_filtered.to_csv(path_file, index=False)

            # Put in the scenario specification dictionary
            new_s[f'{scenario}_wo_{name}'] = s[scenario].copy()
            new_s[f'{scenario}_wo_{name}']['pNewTransmission'] = path_file
            
            if False:
                # Delayed project implementation
                df_delay = df.copy()
                df_delay.loc[df_delay['interco_key'].isin(interco_assessment), 'EarliestEntry'] += delay
                df_delay = df_delay.drop(columns='interco_key')
                path_file = os.path.basename(s[scenario]['pNewTransmission']).split('.')[0] + '_' + name + '.csv'
                path_file = os.path.join(folder_assessment, path_file)
                df_delay.to_csv(path_file, index=False)
                # Put in the scenario specification dictionary
                new_s[f'{scenario}_{name}_delay{delay}'] = s[scenario].copy()
                new_s[f'{scenario}_{name}_delay{delay}']['pNewTransmission'] = path_file
                
                # Reduce capacity of the interconnection to 50% of the original value
                df_reduced = df.copy()
                df_reduced.loc[df_reduced['interco_key'].isin(interco_assessment), 'CapacityPerLine'] *= 0.5
                df_reduced = df_reduced.drop(columns='interco_key')
                path_file = os.path.basename(s[scenario]['pNewTransmission']).split('.')[0]
                path_file = f'{path_file}_{name}.csv'
                path_file = os.path.join(folder_assessment, path_file)
                df_reduced.to_csv(path_file, index=False)
                # Put in the scenario specification dictionary
                new_s[f'{scenario}_{name}_reduced'] = s[scenario].copy()
                new_s[f'{scenario}_{name}_reduced']['pNewTransmission'] = path_file
               

    except Exception:
        raise KeyError('Error in interco_assessment features')

    s.update(new_s)

    return s

def main(test_args=None):
    parser = argparse.ArgumentParser(description="Process some configurations.")

    parser.add_argument(
        "--config",
        type=str,
        default="config.csv",
        help="Path to the configuration file from the folder_input"
    )

    parser.add_argument(
        "--folder_input",
        type=str,
        default="data_test",
        help="Input folder name (default: data_test)"
    )
    
    parser.add_argument(
        "--solver",
        type=str,
        default="MIP",
        help="Sover to use in GAMS (default: MIP)."
    )

    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Scenario file name (default: no filename)"
    )

    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Enable sensitivity analysis (default: False)"
    )

    parser.add_argument(
        "--montecarlo",
        action="store_true",
        help="Enable montecarlo analysis (default: False)"
    )

    parser.add_argument(
        "--montecarlo_samples",
        type=int,
        default=10,
        help="Number of samples to use in the Monte Carlo analysis (default: 10)"
    )

    parser.add_argument(
        "--uncertainties",
        type=str,
        default=None,
        help="Uncertainties file name (default: no filename)"
    )

    parser.add_argument(
        "--reduced_output",
        action="store_true",
        help="Enable reduced output (default: False)"
    )

    parser.add_argument(
        "--reduce_definition_csv",
        action="store_true",
        help="Enable reduced yearly definition for csv files used in Tableau visualization (default: False)"
    )

    parser.add_argument(
        "--selected_scenarios",
        nargs="+",  # Accepts one or more values
        type=str,
        default=None,
        help="List of selected scenarios (default: None). Example usage: --selected_scenarios baseline HighDemand"
    )

    parser.add_argument(
        "--cpu",
        type=int,
        default=1,
        help="Number of CPUs (default: 1)"
    )

    parser.add_argument(
        "--project_assessment",
        nargs="+",  # Accepts one or more values
        type=str,
        default=None,
        help="Name of the project to assess (default: None). Example usage: --project_assessment Solar"
    )
    
    parser.add_argument(
        "--interco_assessment",
        nargs="+",  # Accepts one or more values
        type=str,
        default=None,
        help="Name of the project to assess (default: None). Example usage: --interco_assessment Angola-Zambia"
    )

    parser.add_argument(
        "--simple",
        nargs="*",  # Accepts zero or more values
        default=None,
        help="List of simplified parameters. "
             "If omitted: nothing happens. "
             "If used without arguments: defaults to ['DiscreteCap', 'y']. "
             "If used with arguments: overrides default. "
             "Example: --simple DiscreteCap y"
    )

    parser.add_argument(
        "--postprocess",
        type=str,
        default=None,
        help="Run only postprocess with folder (default: None)"
    )

    parser.add_argument(
        "--output_zip",
        action="store_true",
        default=False,
        help="Zip the output folder and remove the original folder after processing (default: False)"
    )

    parser.add_argument(
        "--plot_selected_scenarios",
        nargs="+",  # Accepts one or more values
        type=str,
        default="all",
        help="List of selected scenarios (default: None). Example usage: --plot_selected_scenarios baseline HighDemand"
    )

    parser.add_argument(
        "--no_plot_dispatch",
        dest="plot_dispatch",
        action="store_false",
        help="Disable dispatch plots (default: True)"
    )
    
    parser.set_defaults(plot_dispatch=True)

    parser.add_argument(
        "--graphs_folder",
        type=str,
        default='img',
        help="Graphs folder to store postprocessing results (default: img)"
    )


    # If test_args is provided (for testing), use it instead of parsing from the command line
    if test_args:
        args = parser.parse_args(test_args)
    else:
        args = parser.parse_args()  # Normal command-line parsing

    # Custom logic
    if args.simple == []:
        args.simple = ['DiscreteCap', 'y']

    print(f"Config file: {args.config}")
    print(f"Folder input: {args.folder_input}")
    print(f"Solver: {args.solver}")
    print(f"Scenarios file: {args.scenarios}")
    print(f"Sensitivity: {args.sensitivity}")
    print(f"MonteCarlo: {args.montecarlo}")
    print(f"MonteCarlo samples: {args.montecarlo_samples}")
    print(f"Monte Carlo uncertainties file: {args.uncertainties}")
    print(f"Reduced output: {args.reduced_output}")
    print(f"Reduced definition csv: {args.reduce_definition_csv}")
    print(f"Selected scenarios: {args.selected_scenarios}")
    print(f"Simple: {args.simple}")

    if args.sensitivity:
        sensitivity = os.path.join('input', args.folder_input, 'sensitivity.csv')
        if not os.path.exists(sensitivity):
            print(f"Warning: sensitivity file {os.path.abspath(sensitivity)} does not exist. No sensitivity analysis will be performed.")
        sensitivity = pd.read_csv(sensitivity, index_col=0).to_dict()['sensitivity']
        print(f"Sensitivity analysis: {sensitivity}")

    else:
        sensitivity = None

    # If none do not run EPM
    if args.postprocess is None:
        folder, result = launch_epm_multi_scenarios(config=args.config,
                                                    folder_input=args.folder_input,
                                                    scenarios_specification=args.scenarios,
                                                    sensitivity=sensitivity,
                                                    montecarlo=args.montecarlo,
                                                    montecarlo_nb_samples=args.montecarlo_samples,
                                                    uncertainties=args.uncertainties,
                                                    selected_scenarios=args.selected_scenarios,
                                                    cpu=args.cpu,
                                                    project_assessment=args.project_assessment,
                                                    interco_assessment=args.interco_assessment,
                                                    simple=args.simple,
                                                    solver=args.solver)
    else:
        print(f"Project folder: {args.postprocess}")
        print("EPM does not run again but use the existing simulation within the folder" )
        folder = args.postprocess
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {os.path.abspath(folder)} does not exist. Please provide a valid folder with EPM results.")
        else:
            print(f"Find folder {os.path.abspath(folder)} for postprocessing.")


    postprocess_output(folder, reduced_output=args.reduced_output,
                       selected_scenario=args.plot_selected_scenarios, plot_dispatch=args.plot_dispatch,
                       graphs_folder=args.graphs_folder, montecarlo=args.montecarlo, 
                       reduce_definition_csv=args.reduce_definition_csv)

    # Zip the folder if it exists
    folder = path_to_extract_results(folder)
    if args.output_zip and folder and os.path.exists(folder):
        print(f"Compressing results folder {folder}")
        zip_path = folder + '.zip'
        shutil.make_archive(folder, 'zip', folder)
        shutil.rmtree(folder)  # Remove the original folder
        print(f"Folder {folder} zipped as {zip_path}")

if __name__ == '__main__':
    main()
