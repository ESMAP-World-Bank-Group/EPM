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
import json
import argparse
from postprocessing.utils import postprocess_output
import re
from pathlib import Path
import sys

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
    return df.map(lambda x: str(Path(x).as_posix()) if isinstance(x, (Path, str)) else x)


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
               prefix=''#'simulation_'
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
    shutil.copy(path_cplex_file, cwd)

    # Arguments for GAMS
    path_args = ['--{} {}'.format(k, i) for k, i in scenario.items()]

    options = []
    if path_engine_file:
        print('Save file only to prepare running simulation on remote server')
        # Run GAMS with the updated environment

        options = ['a=c', 'xs=engine_{}'.format(scenario_name)]

    command = ["gams", path_main_file] + options + ["--BASE_FILE {}".format(path_base_file),
                                                    "--REPORT_FILE {}".format(path_report_file),
                                                    "--READER_FILE {}".format(path_reader_file),
                                                    "--VERIFICATION_FILE {}".format(path_verification_file),
                                                    "--TREATMENT_FILE {}".format(path_treatment_file),
                                                    "--DEMAND_FILE {}".format(path_demand_file),
                                                    "--FOLDER_INPUT {}".format(folder_input),
                                                    "--READER CONNECT_CSV_PYTHON"
                                                    ] + path_args

    # Print the command
    print("Command to execute:", command)

    if sys.platform.startswith("win"):  # If running on Windows
        subprocess.run(' '.join(command), cwd=cwd, shell=True)
    else:  # For Linux or macOS
        subprocess.run(command, cwd=cwd)

    result = None
    # Generate the command for Engine
    if True:
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


def launch_epm_multiprocess(df, scenario_name, path_gams, folder_input=None, path_engine_file=False):
    return launch_epm(df, scenario_name=scenario_name, folder_input=folder_input,
                      path_engine_file=path_engine_file, **path_gams)


def launch_epm_multi_scenarios(config='config.csv',
                               scenarios_specification='scenarios_specification.csv',
                               selected_scenarios=['baseline'],
                               cpu=1, path_gams=None,
                               sensitivity=None,
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
        path_engine_file = os.path.join(os.getcwd(), path_engine_file)

    # Read the scenario CSV file
    if path_gams is not None:  # path for required gams file is provided
        path_gams = {k: os.path.join(os.getcwd(), i) for k, i in path_gams.items()}
    else:  # use default configuration
        path_gams = {k: os.path.join(os.getcwd(), i) for k, i in PATH_GAMS.items()}

    # Read configuration file
    config = pd.read_csv(config).set_index('paramNames').squeeze()
    # Remove title section of the configuration file
    config = config.dropna()
    # Normalize path
    config = normalize_path(config)

    # Read scenarios specification
    if scenarios_specification is not None:
        scenarios = pd.read_csv(scenarios_specification).set_index('paramNames')

        scenarios = normalize_path(scenarios)

        # Generate scenario pd.Series for alternative scenario
        s = {k: config.copy() for k in scenarios}
        for k in s.keys():
            s[k].update(scenarios[k].dropna())
    else:
        s = {}

    # Add the baseline scenario
    s.update({'baseline': config})


    if selected_scenarios is not None:
        s = {k: s[k] for k in selected_scenarios}

    # Add full path to the files
    folder_input = os.path.join(os.getcwd(), 'input', folder_input) if folder_input else os.path.join(os.getcwd(), 'input')
    for k in s.keys():
        s[k] = s[k].apply(lambda i: os.path.join(folder_input, i))

    if sensitivity is not None:
        s = perform_sensitivity(sensitivity, s)

    if project_assessment is not None:
        s = perform_assessment(project_assessment, s)

    if simple:

        for k in s.keys():
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

            # Remove DescreteCap
            df = pd.read_csv(s[k]['pGenDataExcel'])

            df.loc[:, 'DescreteCap'] = 0

            # Creating a new folder
            folder_sensi = os.path.join(os.path.dirname(s[k]['pGenDataExcel']), 'sensitivity')
            if not os.path.exists(folder_sensi):
                os.mkdir(folder_sensi)
            name = 'pGenDataExcel_linear'
            path_file = os.path.basename(s[k]['pGenDataExcel']).replace('pGenDataExcel', name)
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
    df.to_csv('simulations_scenarios.csv')

    # Run EPM in multiprocess
    with Pool(cpu) as pool:
        result = pool.starmap(launch_epm_multiprocess,
                              [(s[k], k, path_gams, folder_input, path_engine_file) for k in s.keys()])

    if path_engine_file:
        pd.DataFrame(result).to_csv('tokens_simulation.csv', index=False)

    os.chdir(working_directory)

    return folder, result


def perform_sensitivity(sensitivity, s):
    param = 'pSettings'
    if sensitivity.get(param):
        settings_sensi = {'VOLL': [250],
                          'planning_reserve_constraints': [0], 'VREForecastError': [0, 0.3],
                          'zonal_spinning_reserve_constraints': [0],
                          'costSurplus': [1, 5], 'costcurtail': [1, 5], 'interconMode': [0,1]}
        # 'mingen_constraints': [1], # 'DR': [0.04, 0.08],

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
    if sensitivity.get(param):
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

    param = 'pDemandForecast'
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

    param = 'pDemandProfile'
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

    param = 'pAvailabilityDefault'
    if sensitivity.get(param):
        availability_sensi = [0.2, 0.4]

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

    param = 'pCapexTrajectoriesDefault'
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

    param = 'pFuelPrice'
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

    param = 'pGenDataExcelDefault'
    if sensitivity.get(param):

        df = pd.read_csv(s['baseline'][param])
        df.loc[df['fuel'].isin(['Coal', 'Gas', 'HFO', 'LFO']), 'ResLimShare'] *= (1 - 0.5)

        # Creating a new folder
        folder_sensi = os.path.join(os.path.dirname(s['baseline'][param]), 'sensitivity')
        if not os.path.exists(folder_sensi):
            os.mkdir(folder_sensi)
        name = f'{param}_ResLimShare_-05'
        path_file = os.path.basename(s['baseline'][param]).replace(param, name)
        path_file = os.path.join(folder_sensi, path_file)
        # Write the modified file
        df.to_csv(path_file, index=False)

        # Put in the scenario dir
        s[name] = s['baseline'].copy()
        s[name][param] = path_file

    param  = 'pVREProfile'
    if sensitivity.get(param):
        capacity_factor_sensi = [-0.2, 0.2]

        for val in capacity_factor_sensi:
            df = pd.read_csv(s['baseline'][param])
            df.loc[:, [i for i in df.columns if i not in ['zone', 'tech', 'q', 'd']]] *= (1 + val)

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
            projects = project_assessment.split(' & ')
            # Remove project(s) in project_assessment
            df = df.loc[~df['gen'].isin(projects)]

            # Create a specific folder to store the counterfactual scenario
            folder_assessment = os.path.join(os.path.dirname(s[scenario]['pGenDataExcel']), 'assessment')
            if not os.path.exists(folder_assessment):
                os.mkdir(folder_assessment)

            # Write the modified file
            name = project_assessment.replace(' ', '').replace('&', '_')
            path_file = 'pGenDataExcel' + name + '.csv'
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
        default="input/config.csv",
        help="Path to the configuration file (default: input/config.csv)"
    )

    parser.add_argument(
        "--folder_input",
        type=str,
        default="data_gambia",
        help="Input folder name (default: data_gambia)"
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
        "--reduced_output",
        action="store_false",
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
        type=str,
        default=None,
        help="Name of the project to assess (default: None)"
    )

    parser.add_argument(
        "--plot_all",
        action="store_true",
        help="Plot dispatch for all scenarios (default: False)"
    )

    parser.add_argument(
        "--simple",
        action="store_true",
        help="Make simplified run (default: False)"
    )


    # If test_args is provided (for testing), use it instead of parsing from the command line
    if test_args:
        args = parser.parse_args(test_args)
    else:
        args = parser.parse_args()  # Normal command-line parsing

    print(f"Config file: {args.config}")
    print(f"Folder input: {args.folder_input}")
    print(f"Scenarios file: {args.scenarios}")
    print(f"Sensitivity: {args.sensitivity}")
    print(f"Reduced output: {args.reduced_output}")
    print(f"Selected scenarios: {args.selected_scenarios}")
    print(f"Simple: {args.simple}")
    print(f"Plot options: {args.plot_all}")

    if args.sensitivity:
        sensitivity = {'pSettings': True, 'pDemandForecast': True,
                       'pFuelPrice': False, 'pCapexTrajectoriesDefault': True,
                       'pAvailabilityDefault': True, 'pDemandProfile': False,
                       'y': False, 'pGenDataExcelDefault': True, 'pVREProfile': True}
    else:
        sensitivity = None

    folder, result = launch_epm_multi_scenarios(config=args.config,
                                                folder_input=args.folder_input,
                                                scenarios_specification=args.scenarios,
                                                sensitivity=sensitivity,
                                                selected_scenarios=args.selected_scenarios,
                                                cpu=args.cpu,
                                                project_assessment=args.project_assessment,
                                                simple=args.simple)

    postprocess_output(folder, reduced_output=False, plot_all=args.plot_all, folder='postprocessing')

if __name__ == '__main__':

    # # Example test arguments
    # test_parameters = [
    #     "--config", "input/config.csv",
    #     "--folder_input", "data_gambia"
    # ]

    main()

