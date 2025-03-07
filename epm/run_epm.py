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
                               folder_input=None):
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

    # Read scenarios specification
    if scenarios_specification is not None:
        scenarios = pd.read_csv(scenarios_specification).set_index('paramNames')

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
        s = generate_sensitivity(sensitivity, s)

    # Create dir for simulation and change current working directory
    if 'output' not in os.listdir():
        os.mkdir('output')

    folder = 'simulations_run_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder = os.path.join('output', folder)
    if not os.path.exists(folder):
        os.mkdir(folder)
        print('Folder created:', folder)
    os.chdir(folder)

    with Pool(cpu) as pool:
        result = pool.starmap(launch_epm_multiprocess,
                              [(s[k], k, path_gams, folder_input, path_engine_file) for k in s.keys()])

    if path_engine_file:
        pd.DataFrame(result).to_csv('tokens_simulation.csv', index=False)

    os.chdir(working_directory)

    return folder, result


def generate_sensitivity(sensitivity, s):
    param = 'pSettings'
    if sensitivity.get(param):
        settings_sensi = {'VOLL': [250],
                          'planning_reserve_constraints': [0], 'VREForecastError': [0, 0.3],
                          'zonal_spinning_reserve_constraints': [0],
                          'costSurplus': [1, 5], 'costcurtail': [1, 5]}
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
        availability_sensi = [0.1, 0.4]

        for val in availability_sensi:
            df = pd.read_csv(s['baseline'][param])

            # 0.85 is usually the default value
            df.replace(0.85, val, inplace=True)

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


def main():
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
        "--full_output",
        action="store_false",
        help="Disable full output (default: True)"
    )

    args = parser.parse_args()

    print(f"Config file: {args.config}")
    print(f"Folder input: {args.folder_input}")
    print(f"Scenarios file: {args.scenarios}")
    print(f"Sensitivity: {args.sensitivity}")
    print(f"Full output: {args.full_output}")

    if args.sensitivity:
        sensitivity = {'pSettings': True, 'pDemandForecast': False,
                       'pFuelPrice': False, 'pCapexTrajectoriesDefault': False,
                       'pAvailabilityDefault': False, 'pDemandProfile': False}
    else:
        sensitivity = None

    folder, result = launch_epm_multi_scenarios(config=args.config,
                                                folder_input=args.folder_input,
                                                scenarios_specification=args.scenarios,
                                                sensitivity=sensitivity,
                                                selected_scenarios=None,
                                                cpu=3)
    postprocess_output(folder, full_output=True)

if __name__ == '__main__':
    main()

