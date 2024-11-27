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


# TODO: Add all cplex option and other simulation parameters that were in Looping.py

PATH_GAMS = {
    'path_main_file': 'WB_EPM_v8_5_main_country.gms',
    'path_base_file': 'WB_EPM_v8_5_base_country.gms',
    'path_report_file': 'WB_EPM_v8_5_Report_country.gms',
    'path_reader_file': 'WB_EPM_input_readers.gms',
    'path_cplex_file': 'cplex.opt'
}

URL_ENGINE = "https://engine.gams.com/api"


def get_auth_engine():
    user_name = "lucas"
    password = "Jo@d$293o45Q"
    auth = HTTPBasicAuth(user_name, password)
    return auth


def get_configuration():
    configuration = gams.engine.Configuration(
        host='https://engine.gams.com/api',
        username='lucas',
        password='Jo@d$293o45Q')
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
               path_main_file='WB_EPM_v8_5_main.gms',
               path_base_file='WB_EPM_v8_5_base.gms',
               path_report_file='WB_EPM_v8_5_Report.gms',
               path_reader_file='WB_EPM_input_readers.gms',
               path_cplex_file='cplex.opt',
               path_engine_file=False):
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
    folder = 'simulation_{}'.format(scenario_name)
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
                                                    "--READER CONNECT_CSV_PYTHON"
                                                    ] + path_args

    # Print the command
    print("Command to execute:", command)

    subprocess.run(command, cwd=cwd)

    result = None
    # Generate the command for Engine
    if False:
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


def launch_epm_multiprocess(df, scenario_name, path_gams, path_engine_file=False):
    return launch_epm(df, scenario_name=scenario_name, path_engine_file=path_engine_file, **path_gams)


def launch_epm_multi_scenarios(scenario_baseline='scenario_baseline.csv',
                               scenarios_specification='scenarios_specification.csv',
                               selected_scenarios=None,
                               cpu=1,
                               path_engine_file=False):
    """
    Launch the EPM model with multiple scenarios based on scenarios_specification

    Parameters
    ----------
    scenario_baseline: str, optional, default 'scenario_baseline.csv'
        Path to the CSV file with the baseline scenario
    scenarios_specification: str, optional, default 'scenarios_specification.csv'
        Path to the CSV file with the scenarios specification
    cpu: int, optional, default 1
        Number of CPUs to use
    selected_scenarios: list, optional, default None
        List of scenarios to run
    path_engine_file: str, optional, default False
    """

    # Add the full path to the files
    if path_engine_file:
        path_engine_file = os.path.join(os.getcwd(), path_engine_file)

    # Read the scenario CSV file
    path_gams = {k: os.path.join(os.getcwd(), i) for k, i in PATH_GAMS.items()}

    # Read scenario baseline
    scenario_baseline = pd.read_csv(scenario_baseline).set_index('paramNames').squeeze()

    # Read scenarios specification
    scenarios = pd.read_csv(scenarios_specification).set_index('paramNames')

    # Generate scenario pd.Series for alternative scenario
    s = {k: scenario_baseline.copy() for k in scenarios}
    for k in s.keys():
        s[k].update(scenarios[k].dropna())
    # Add the baseline scenario
    s.update({'baseline': scenario_baseline})

    if selected_scenarios is not None:
        s = {k: s[k] for k in selected_scenarios}

    # Add full path to the files
    for k in s.keys():
        s[k] = s[k].apply(lambda i: os.path.join(os.getcwd(), 'input', i))

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
                              [(s[k], k, path_gams, path_engine_file) for k in s.keys()])

    if path_engine_file:
        pd.DataFrame(result).to_csv('tokens_simulation.csv', index=False)
    return result


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


if __name__ == '__main__':

    if True:
        launch_epm_multi_scenarios(scenario_baseline='input/scenario_baseline.csv',
                                   scenarios_specification='input/scenarios_specification.csv',
                                   selected_scenarios=['DemandS3'],
                                   cpu=1,
                                   path_engine_file=None)
