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
from zipfile import ZipFile, ZIP_DEFLATED
from requests import post, get
from requests.auth import HTTPBasicAuth
import gams.engine
from gams.engine.api import jobs_api
from gams import GamsWorkspace
import json
import argparse

from postprocessing.utils import postprocess_output, path_to_extract_results
import re
from pathlib import Path
import sys
import chaospy
import numpy as np
import shutil

# TODO: Add all cplex option and other simulation parameters that were in Looping.py

PATH_GAMS = {
    'path_main_file': 'main.gms',
    'path_base_file': 'base.gms',
    'path_report_file': 'generate_report.gms',
    'path_reader_file': 'input_readers.gms',
    'path_verification_file': 'input_verification.gms',
    'path_treatment_file': 'input_treatment.gms',
    'path_demand_file': 'generate_demand.gms',
    'path_cplex_file': 'cplex.opt'
}


URL_ENGINE = "https://engine.gams.com/api"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS = json.load(open(os.path.join(BASE_DIR, 'credentials_engine.json'), 'r'))

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


def get_auth_engine():
    user_name = CREDENTIALS['username']
    password = CREDENTIALS['password']
    auth = HTTPBasicAuth(user_name, password)
    return auth


def get_configuration():
    configuration = gams.engine.Configuration(
        host='https://engine.gams.com/api',
        username=CREDENTIALS['username'],
        password=CREDENTIALS['password'])
    return configuration


def post_job_engine(scenario_name, path_zipfile):
    """
    Post a job to the GAMS Engine.

    Parameters
    ----------
    scenario_name
    path_zipfile

    Returns
    -------

    """

    auth = get_auth_engine()

    # Send the job to the server
    query_params = {
        "model": 'engine_{}'.format(scenario_name),
        "namespace": "wb",
        "labels": "instance=GAMS_z1d.2xlarge_282_S"
    }
    job_files = {"model_data": open(path_zipfile, "rb")}
    req = post(
        URL_ENGINE + "/jobs/", params=query_params, files=job_files, auth=auth
    )
    return req


def launch_epm_checkpoint(scenario,
               scenario_name='',
               path_main_file='main.gms',
               path_base_file='base.gms',
               path_report_file='generate_report.gms',
               path_reader_file='input_readers.gms',
               path_verification_file='input_verification.gms',
               path_treatment_file='input_treatment.gms',
               path_demand_file='generate_demand.gms',
               path_cplex_file='cplex.opt',
               folder_input=None,
               path_engine_file=False,
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
        'VERIFICATION_FILE': path_verification_file,
        'TREATMENT_FILE': path_treatment_file,
        'DEMAND_FILE': path_demand_file,
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
               path_verification_file='input_verification.gms',
               path_treatment_file='input_treatment.gms',
               path_demand_file='generate_demand.gms',
               path_cplex_file='cplex.opt',
               folder_input=None,
               path_engine_file=False,
               dict_montecarlo=None,
               prefix='' #'simulation_'
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
    path_engine_file: str, optional, default False
        The path to the GAMS engine file
    dict_montecarlo: dict, optional, default None
        Correspondence for solution when running montecarlo scenarios

    Returns
    -------
    dict
        A dictionary with the name of the scenario, the path to the simulation folder and the token for the job
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
    if path_engine_file:
        print('Save file only to prepare running simulation on remote server')
        # Run GAMS with the updated environment

        options.extend(['a=c', 'xs=engine_{}'.format(scenario_name)])

    if dict_montecarlo is not None:
        loadsolpath = os.path.join(os.path.abspath(os.path.join(cwd, os.pardir)), dict_montecarlo[scenario_name])
        options.extend(["--LOADSOLPATH {}".format(loadsolpath)])

    command = ["gams", path_main_file] + options + ["--BASE_FILE {}".format(path_base_file),
                                                    "--REPORT_FILE {}".format(path_report_file),
                                                    "--READER_FILE {}".format(path_reader_file),
                                                    "--VERIFICATION_FILE {}".format(path_verification_file),
                                                    "--TREATMENT_FILE {}".format(path_treatment_file),
                                                    "--DEMAND_FILE {}".format(path_demand_file),
                                                    "--FOLDER_INPUT {}".format(folder_input)
                                                    ] + path_args

    # Print the command
    print("Command to execute:", command)

    if sys.platform.startswith("win"):  # If running on Windows
        rslt = subprocess.run(' '.join(command), cwd=cwd, shell=True)
    else:  # For Linux or macOS
        rslt = subprocess.run(command, cwd=cwd)

    if rslt.returncode != 0:
        raise RuntimeError('GAMS Error: check GAMS logs file ')

    result = None
    # Generate the command for Engine
    if path_engine_file:
        # Open Engine_Base.gms as text file and replace
        with open(path_engine_file, 'r') as file:
            filedata = file.read()

            # Replace the target string
            filedata = filedata.replace('Engine_Base', 'engine_{}'.format(scenario_name))

        # Store the new file in the simulation folder
        with open(os.path.join(cwd, 'engine_{}.gms'.format(scenario_name)), 'w') as file:
            file.write(filedata)

        # Make a ZipFile that can be sent to the server
        with ZipFile(os.path.join(cwd, 'engine_{}.zip'.format(scenario_name)), 'w', ZIP_DEFLATED) as files_ziped:
            files_ziped.write(os.path.join(cwd, 'engine_{}.gms'.format(scenario_name)), 'engine_{}.gms'.format(scenario_name))
            files_ziped.write(os.path.join(cwd, 'engine_{}.g00'.format(scenario_name)), 'engine_{}.g00'.format(scenario_name))

        path_zipfile = os.path.join(cwd, 'engine_{}.zip'.format(scenario_name))
        req = post_job_engine(scenario_name, path_zipfile)
        result = {'name': scenario_name, 'path': cwd, 'token': req.json()['token']}

    return result


def launch_epm_multiprocess(df, scenario_name, path_gams, folder_input=None, path_engine_file=False,
                            dict_montecarlo=None):
    return launch_epm(df, scenario_name=scenario_name, folder_input=folder_input,
                      path_engine_file=path_engine_file, dict_montecarlo=dict_montecarlo, **path_gams)


def launch_epm_multi_scenarios(config='config.csv',
                               scenarios_specification='scenarios.csv',
                               selected_scenarios=['baseline'],
                               cpu=1, path_gams=None,
                               sensitivity=None,
                               montecarlo=False,
                               montecarlo_nb_samples=10,
                               uncertainties=None,
                               path_engine_file=False,
                               folder_input=None,
                               project_assessment=None,
                               simple=None):
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
    path_engine_file: str, optional, default False
    folder_input: str, optional, default None
        Folder where data input files are stored
    """

    working_directory = os.getcwd()

    # Add the full path to the files
    if path_engine_file:
        path_engine_file = os.path.join(working_directory, path_engine_file)

    # Read the scenario CSV file
    if path_gams is not None:  # path for required gams file is provided
        path_gams = {k: os.path.join(working_directory, i) for k, i in path_gams.items()}
    else:  # use default configuration
        path_gams = {k: os.path.join(working_directory, i) for k, i in PATH_GAMS.items()}

    # Read configuration file
    if not os.path.exists(config):
        raise FileNotFoundError(f'Configuration file {os.path.abspath(config)} not found.')

    config = pd.read_csv(config).set_index('paramNames').squeeze()
    # Remove title section of the configuration file
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
    folder_input = os.path.join(os.getcwd(), 'input', folder_input) if folder_input else os.path.join(os.getcwd(), 'input')
    for k in s.keys():
        s[k] = s[k].apply(lambda i: os.path.join(folder_input, i) if '.csv' in i else i)

    # Run sensitivity analysis if activated
    if sensitivity is not None:
        s = perform_sensitivity(sensitivity, s)

    # Run montecarlo analysis if activated
    if montecarlo:
        assert uncertainties is not None, "Monte Carlo analysis is activated, but uncertainties is set to None."
        initial_scenarios = list(s.keys())
        df_uncertainties = pd.read_csv(uncertainties)
        distribution, samples, zone_mapping = define_samples(df_uncertainties, nb_samples=montecarlo_nb_samples)
        s, scenarios_montecarlo = create_scenarios_montecarlo(samples, s, zone_mapping)
        dict_montecarlo = {key: key.split('_')[0] for key in scenarios_montecarlo.keys()}  # getting the correspondence between montecarlo scenario and initial scenario

    # Set-up project assessment scenarios if activated
    if project_assessment is not None:
        s = perform_assessment(project_assessment, s)

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
                df = pd.read_csv(s[k]['pGenDataExcel'])

                df.loc[:, 'DiscreteCap'] = 0

                # Creating a new folder
                folder_sensi = os.path.join(os.path.dirname(s[k]['pGenDataExcel']), 'sensitivity')
                if not os.path.exists(folder_sensi):
                    os.mkdir(folder_sensi)
                path_file = os.path.basename(s[k]['pGenDataExcel']).split('.')[0] + '_linear.csv'
                path_file = os.path.join(folder_sensi, path_file)
                # Write the modified file
                df.to_csv(path_file, index=False)

                # Put in the scenario dir
                s[k]['pGenDataExcel'] = path_file

    # Create dir for simulation and change current working directory
    if 'output' not in os.listdir():
        os.mkdir('output')

    folder = 'simulations_run_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
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
                                  [(s[k], k, path_gams, folder_input, path_engine_file) for k in s.keys()])
    else:
        # First, run initial scenarios
        # Ensure config file has extended setting to save output
        with Pool(cpu) as pool:
            for k in s.keys():
                assert s[k]['solvemode'] == '1', 'Parameter solvemode should be set to 1 in the configuration to obtain extended output for the baseline scenarios in the Monte-Carlo analysis.'
            result = pool.starmap(launch_epm_multiprocess,
                                  [(s[k], k, path_gams, folder_input, path_engine_file) for k in s.keys()])
        # Modify config file to ensure limited output saved
        with Pool(cpu) as pool:  # running montecarlo scenarios in multiprocessing
            result = pool.starmap(launch_epm_multiprocess,
                                  [(scenarios_montecarlo[k], k, path_gams, folder_input, path_engine_file, dict_montecarlo) for k in scenarios_montecarlo.keys()])

    if path_engine_file:
        pd.DataFrame(result).to_csv('tokens_simulation.csv', index=False)

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
        DD = []
        for v in values:
            D = getattr(chaospy, v["type"])
            DD.append(D(*v["args"]))
        return chaospy.J(*DD)

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
    uncertainties = {}
    zone_mapping = {}
    chaospy_distributions = [
        name for name in dir(chaospy)
        if callable(getattr(chaospy, name)) and hasattr(getattr(chaospy, name), '__module__')
           and 'chaospy' in getattr(chaospy, name).__module__.lower()
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
                    param_to_merge = 'pGenDataExcel'
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
    param = 'pSettings'
    if sensitivity.get(param):  # testing implications of some setting parameters
        settings_sensi = {'VOLL': [250],
                          'planning_reserve_constraints': [0], 'VREForecastError': [0, 0.3],
                          'zonal_spinning_reserve_constraints': [0],
                          'costSurplus': [1, 5], 'costcurtail': [1, 5], 'interconMode': [0,1],
                          'includeIntercoReserves': [0,1], 'interco_reserve_contribution': [0, 0.5]}

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
    if sensitivity.get(param):  # testing implications of year definition
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
    if sensitivity.get(param):
        demand_forecast_sensi = [-0.25, -0.1, 0.1, 0.25]
        for val in demand_forecast_sensi:
            df = pd.read_csv(s['baseline'][param])

            cols = [i for i in df.columns if i not in ['zone', 'type']]
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
    if sensitivity.get(param):
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
    if sensitivity.get(param):
        availability_sensi = [0.3, 0.7]

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
    if sensitivity.get(param):

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
    if sensitivity.get(param):
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
    if sensitivity.get(param):
        parameter = 'pGenDataExcelDefault'
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
    if sensitivity.get(param):
        parameter = 'pGenDataExcel'

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

    param  = 'pVREProfile'  # testing implications of a change in VRE production
    if sensitivity.get(param):
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
            # Reading the initial value
            df = pd.read_csv(s[scenario]['pGenDataExcel'])
            # Multiple projects can be considered if separate by ' & '
            projects = project_assessment
            # Remove project(s) in project_assessment
            df = df.loc[~df['gen'].isin(projects)]

            # Create a specific folder to store the counterfactual scenario
            folder_assessment = os.path.join(os.path.dirname(s[scenario]['pGenDataExcel']), 'assessment')
            if not os.path.exists(folder_assessment):
                os.mkdir(folder_assessment)

            # Write the modified file
            name = '-'.join(project_assessment).replace(' ', '')
            path_file = os.path.basename(s[scenario]['pGenDataExcel']).split('.')[0] + '_' + name + '.csv'
            path_file = os.path.join(folder_assessment, path_file)
            df.to_csv(path_file, index=False)

            # Put in the scenario specification dictionary
            new_s[f'{scenario}_wo_{name}'] = s[scenario].copy()
            new_s[f'{scenario}_wo_{name}']['pGenDataExcel'] = path_file

    except Exception:
        raise KeyError('Error in project_assessment features')

    s.update(new_s)

    return s

def get_job_engine(tokens_simulation):
    # {'baseline': 'a241bf62-34db-436d-8f4f-113333d3c6b9'}

    tokens = pd.read_csv(tokens_simulation).set_index('path')['token']

    configuration = get_configuration()

    for scenario, token in tokens.items():

        with gams.engine.ApiClient(configuration) as api_client:
            if not os.path.exists(scenario):
                os.makedirs(scenario)

            job_api_instance = jobs_api.JobsApi(api_client)

        with ZipFile(job_api_instance.get_job_zip(token)) as zf:
            zf.extractall(path=scenario)

def main(test_args=None):
    parser = argparse.ArgumentParser(description="Process some configurations.")

    parser.add_argument(
        "--config",
        type=str,
        default="input/data_test_region/config.csv",
        help="Path to the configuration file (default: input/data_test_region/config.csv)"
    )

    parser.add_argument(
        "--folder_input",
        type=str,
        default="data_test",
        help="Input folder name (default: data_test)"
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
        help="Name of the project to assess (default: None). Example usage: --project_assessment Solar Project"
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
        "--engine",
        type=str,
        default=None,
        help="Name of the path engine file (default: None)"
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
    print(f"Scenarios file: {args.scenarios}")
    print(f"Sensitivity: {args.sensitivity}")
    print(f"MonteCarlo: {args.montecarlo}")
    print(f"MonteCarlo samples: {args.montecarlo_samples}")
    print(f"Monte Carlo uncertainties file: {args.uncertainties}")
    print(f"Reduced output: {args.reduced_output}")
    print(f"Selected scenarios: {args.selected_scenarios}")
    print(f"Simple: {args.simple}")

    if args.sensitivity:
        sensitivity = os.path.join('input', args.folder_input, 'sensitivity.csv')
        if not os.path.exists(sensitivity):
            print(f"Warning: sensitivity file {sensitivity} does not exist. No sensitivity analysis will be performed.")
        sensitivity = pd.read_csv(sensitivity, index_col=0).to_dict()['sensitivity']

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
                                                    simple=args.simple,
                                                    path_engine_file=args.engine)
    else:
        print(f"Project folder: {args.postprocess}")
        print("EPM does not run again but use the existing simulation within the folder" )
        folder = args.postprocess


    postprocess_output(folder, reduced_output=args.reduced_output, folder='postprocessing',
                       selected_scenario=args.plot_selected_scenarios, plot_dispatch=args.plot_dispatch,
                       graphs_folder=args.graphs_folder, montecarlo=args.montecarlo, reduce_definition_csv=True)

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

