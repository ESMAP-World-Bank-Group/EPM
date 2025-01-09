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
from pathlib import Path
import textwrap
import numpy as np
import chaospy
from math import factorial
from surrogate import NamedJ
from matplotlib import pyplot as plt


# TODO: Add all cplex option and other simulation parameters that were in Looping.py

PATH_GAMS = {
    'path_main_file': 'WB_EPM_v8_5_daily_storage_main.gms',#'WB_EPM_v8_5_main_V3_CONNECT_CSV.gms',
    'path_base_file': 'WB_EPM_v8_5_daily_storage_base.gms',
    'path_report_file': 'WB_EPM_v8_5_Report.gms',
    'path_reader_file': 'WB_EPM_daily_storage_input_readers.gms',
    'path_cplex_file': 'cplex.opt'
}

VARIABLE_SCENARIOS = {
    'demand': 'pDemandForecast',
    'hydro': 'pAvailabilityDaily',
    'import': 'pGenDataExcel',
    'hfoprice': 'pFuelPrice',
    'batterycost': 'pCapexTrajectory'
}



URL_ENGINE = "https://engine.gams.com/api"


def get_auth_engine():
    user_name = "CeliaEscribe"
    password = "cv86aWE30TG"
    auth = HTTPBasicAuth(user_name, password)
    return auth


def get_configuration():
    configuration = gams.engine.Configuration(
        host='https://engine.gams.com/api',
        username='CeliaEscribe',
        password='cv86aWE30TG')
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
        "labels": "instance=GAMS_z1d.3xlarge_282_S"
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
    if '_' in path_cplex_file.split('/')[-1]:
        new_file_name = path_cplex_file.split('/')[-1].split('_')[0] + '.opt'  # renaming file to cplex.opt, which is necessary for GAMS to use it as a solver option file
        new_file_path = os.path.join(cwd, new_file_name)
        shutil.copy(path_cplex_file, new_file_path)
    else:
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
    if True:
        if path_engine_file:
            with open(path_cplex_file, 'r') as cplex_file:
                cplex_content = cplex_file.read()

            with open(os.path.join(cwd, f'engine_{scenario_name}.gms'), 'w') as file:
                file.write("$onecho > cplex.opt\n")
                for line in cplex_content.splitlines():
                    file.write(f"{line.strip()}\n")  # Write each line with consistent indentation
                file.write("$offEcho\n")
                file.write(
                    f"$if 'x%gams.restart%'=='x' $call gams engine_{scenario_name}.gms r=engine_{scenario_name} lo=3 o=engine_{scenario_name}.lst\n")

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
                               cpu=1, path_gams=None,
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
    if path_gams is not None:  # path for required gams file is provided
        path_gams = {k: os.path.join(os.getcwd(), i) for k, i in path_gams.items()}
    else:  # use default configuration
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
        s[k] = s[k].apply(lambda i: os.path.join(os.getcwd(), i))

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

    pd.DataFrame(result).to_csv('tokens_simulation.csv', index=False)

    # TODO: change how the dict is saved to csv
    scenario_baseline.to_csv('scenario_baseline.csv', index=False)  # we want to remember what was the baseline configuration we used
    scenarios_specification.to_csv('scenario_baseline.csv', index=False)
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

        # Fetch the job ZIP file
        job_zip = job_api_instance.get_job_zip(token)

        # Open the ZIP file and extract only `epmresults.gdx`
        with ZipFile(job_zip) as zf:
            if "epmresults.gdx" in zf.namelist():  # Check if the file exists in the ZIP
                zf.extract("epmresults.gdx", path=scenario)
            else:
                print(f"'epmresults.gdx' not found in the ZIP for scenario '{scenario}'")

        # with ZipFile(job_api_instance.get_job_zip(token)) as zf:
        #     zf.extractall(path=scenario)


def new_demand_profile(alpha, alpha_mean, folder, file='input/pDemandForecast.csv', t0=2024):
    """Updates demand forecast by changing the linear growth rate of the peak demand."""
    demand_df = pd.read_csv(file, index_col=[0,1]).copy()
    demand_df.columns = demand_df.columns.astype(int)
    # modify file for t > t0 with new alpha
    multiplicative_factor = (demand_df.xs('Energy', level=1)[[t0]] / demand_df.xs('Peak', level=1)[[t0]]).values[0][0]

    cols_t0 = demand_df.loc[:, t0:].columns
    demand_df.loc[(slice(None), 'Peak'), cols_t0] += (alpha - alpha_mean) * (cols_t0 - t0).values
    demand_df.loc[(slice(None), 'Energy'), cols_t0] = multiplicative_factor * demand_df.loc[(slice(None), 'Peak'), cols_t0].values

    total_path = Path(folder) / Path(file.split('/')[-1].split('.csv')[0] + f'_alpha{alpha}.csv')
    demand_df.to_csv(total_path, float_format='%.3f')
    return total_path


def new_gendata(folder, import_capacity=None, file='input/pGenDataExcel.csv'):
    # TODO: add other modifications to the gendata file (if other modifications)
    gen_data = pd.read_csv(file).copy()
    gen_data['Capacity'] = gen_data['Capacity'].astype(float)
    path = file.split('/')[-1].split('.csv')[0]
    if import_capacity is not None:
        gen_data.loc[gen_data.Plants == 'CLSG_CIV', 'Capacity'] = import_capacity
        path += f'_import{import_capacity}'
    path += '.csv'
    total_path = Path(folder) / Path(path)
    gen_data.to_csv(total_path, float_format='%.3f', index=False)
    return total_path

def new_hydro(folder, hydro_multiplier, file='input/pAvailabilityDaily.csv'):
    """Updates hydro generation with by a multiplier."""
    hydro_df = pd.read_csv(file).copy()
    hydro_df.loc[:,'cf'] = np.where(
                                hydro_df['cf'] * hydro_multiplier < 1,
                                hydro_df['cf'] * hydro_multiplier,
                                hydro_df['cf']
                            )
    total_path = Path(folder) / Path(file.split('/')[-1].split('.csv')[0] + f'_multiplier{hydro_multiplier}.csv')
    hydro_df.to_csv(total_path, float_format='%.3f', index=False)
    return total_path

def new_price(price, folder, tech='HFO', t0=2024, file='input/pFuelPrice.csv'):
    """Updates fuel price with a new price."""
    price_df = pd.read_csv(file, index_col=[0,1]).copy()
    price_df.columns = price_df.columns.astype(int)
    cols_t0 = price_df.loc[:, t0:].columns
    price_df.loc[(slice(None), tech), cols_t0] = price

    total_path = Path(folder) / Path(file.split('/')[-1].split('.csv')[0] + f'_price{price}.csv')
    price_df.to_csv(total_path, float_format='%.3f')
    return total_path


def new_cost(cost_variation, folder, tech, t0=2024, file='input/pCapexTrajectory'):
    """Update capex evolution by a vertical translation."""
    cost_df = pd.read_csv(file, index_col=[0]).copy()
    cost_df.columns = cost_df.columns.astype(int)
    cols_t0 = cost_df.loc[:, t0:].columns
    if isinstance(tech, str):
        tech = [tech]
    cost_df.loc[tech, cols_t0] = cost_df.loc[tech, cols_t0] + cost_variation
    total_path = Path(folder) / Path(file.split('/')[-1].split('.csv')[0] + f'_variation{cost_variation}.csv')
    cost_df.to_csv(total_path, float_format='%.3f')
    return total_path



def create_scenario(scenario, sample, name_scenario, folder, baseline):

    for key, value in sample.items():
        if key == 'demand':
            folder_path = new_demand_profile(value, baseline['demand'], folder, file='input/pDemandForecast.csv')
            scenario.loc[VARIABLE_SCENARIOS[key], name_scenario] = folder_path  # update value
        if key == 'import':
            folder_path = new_gendata(folder, import_capacity=value, file='input/pGenDataExcel_storage.csv')
            scenario.loc[VARIABLE_SCENARIOS[key], name_scenario] = folder_path  # update value
        if key == 'hydro':
            folder_path = new_hydro(folder, hydro_multiplier=value, file='input/pAvailabilityDaily.csv')
            scenario.loc[VARIABLE_SCENARIOS[key], name_scenario] = folder_path  # update value
        if key == 'hfoprice':
            folder_path = new_price(price=value, folder=folder, t0=2024, file='input/pFuelPrice.csv')
            scenario.loc[VARIABLE_SCENARIOS[key], name_scenario] = folder_path  # update value
        if key == 'batterycost':
            folder_path = new_cost(cost_variation=value, folder=folder, tech=['BESS_4h', 'BESS_8h', 'BESS_2h', 'BESS_3h'],
                                   t0=2024, file='input/pCapexTrajectory.csv')
            scenario.loc[VARIABLE_SCENARIOS[key], name_scenario] = folder_path  # update value
    return scenario

def create_scenarios_sensitivity(samples, baseline):
    """Creation of scenario dictionary based on a samples dictionary specifying the scenario for each sample.
    Used for global sensitivity analysis.
    """
    folder = 'simulations_run_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder = os.path.join('input', folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    scenario_index = [
        "ftfindex", "mapGG", "pAvailability", "pAvailabilityDaily", "pAvailabilityH2",
        "pCapexTrajectory", "pCapexTrajectoryH2", "pCarbonPrice", "pCSPData",
        "pDemandData", "pDemandForecast", "pDemandProfile", "peak", "pEmissionsTotal",
        "pEmissionsCountry", "pEmissionsZone", "pEnergyEfficiencyFactor",
        "pExternalH2", "pExtTransferLimit", "pFuelData", "pFuelPrice", "pFuelTypeCarbonContent",
        "pGenDataExcel", "pH2DataExcel", "pHours", "pLossFactor", "pMaxExchangeShare",
        "pMaxFuellimit", "pMaxPriceImportShare", "pNewTransmission", "pReserveMargin",
        "pReserveReqLoc", "pReserveReqSys", "pScalars", "pStorDataExcel",
        "pTechDataExcel", "pTradePrice", "pTransferLimit", "pVREgenProfile",
        "pVREProfile", "pZoneIndex", "relevant", "reserve", "sTopology", "y",
        "zcmapExcel", "zext"
    ]

    # Create the DataFrame
    scenarios = []
    # iterate over all samples
    for name_scenario, sample in samples.items():
        name_scenario = name_scenario.replace('.', 'p')
        new_df = pd.DataFrame({name_scenario: np.nan for var in scenario_index}, index=scenario_index)
        new_df = new_df.astype('object')
        new_df = create_scenario(new_df, sample, name_scenario, folder, baseline)
        scenarios.append(new_df)
    scenarios = pd.concat(scenarios, axis=1)
    scenarios.index.names = ['paramNames']
    path_to_save = Path(folder) / Path('scenarios_spec.csv')
    scenarios.to_csv(path_to_save)
    return path_to_save


if __name__ == '__main__':

    if True:
        # launch_epm_multi_scenarios(scenario_baseline='input/scenario_baseline.csv',
        #                            scenarios_specification='input/scenarios_specification.csv',
        #                            selected_scenarios=['baseline'],
        #                            cpu=1,
        #                            path_engine_file=None)
        #
        # path_gams_storage = {
        #     'path_main_file': 'WB_EPM_v8_5_daily_storage_main.gms',  # 'WB_EPM_v8_5_main_V3_CONNECT_CSV.gms',
        #     'path_base_file': 'WB_EPM_v8_5_daily_storage_base.gms',
        #     'path_report_file': 'WB_EPM_v8_5_Report.gms',
        #     'path_reader_file': 'WB_EPM_daily_storage_input_readers.gms',
        #     'path_cplex_file': 'cplex.opt'
        # }
        #
        # launch_epm_multi_scenarios(scenario_baseline='input/scenario_hydrostorage_baseline.csv',
        #                            scenarios_specification='input/scenarios_hydrostorage_specification.csv',
        #                            selected_scenarios=['baseline'],
        #                            cpu=1, path_gams=path_gams_storage,
        #                            path_engine_file=None)

        # get_job_engine('output/simulations_run_20250103_150855/tokens_simulation.csv')

        # Run sensitivity analysis with surrogate model

        uncertainties = {
            # 'demand': {
            #     'type': 'Uniform',
            #     'args': (8, 20)
            # },
            'import': {
                'type': 'Uniform',
                'args': (0, 40)
            },
            'hydro': {
                'type': 'Uniform',
                'args': (0.7, 1.3)
            },
            'hfoprice': {
                'type': 'Uniform',
                'args': (12, 18)
            },
            'batterycost': {
                'type': 'Uniform',
                'args': (-0.5, 0)
            }
        }

        distribution = NamedJ(uncertainties)

        polynomial_degree = 4
        number_of_uncertainty = len(uncertainties.keys())
        cardinality = factorial(polynomial_degree + number_of_uncertainty) / (factorial(polynomial_degree) * factorial(number_of_uncertainty))
        OSR = 2
        number_of_samples = cardinality * OSR

        samples = distribution.sample(size=number_of_samples, rule='halton')

        # fig, axes = plt.subplots(figsize=(10, 5))
        # axes.scatter(*samples, marker='.', c='navy')
        # plt.title('Samples')
        # plt.xlabel('Demand')
        # plt.ylabel('Import')
        # plt.show()

        samples = {
            f'{"_".join([f"{idx}{samples.loc[idx, col]:.3f}" for idx in samples.index])}': {
                idx: round(samples.loc[idx, col], 3) for idx in samples.index
            }
            for col in samples.columns
        }

        baseline = {
            'demand': 11
        }
        path_scenario_spec = create_scenarios_sensitivity(samples, baseline)

        path_gams_storage = {
            'path_main_file': 'WB_EPM_v8_5_daily_storage_main.gms',  # 'WB_EPM_v8_5_main_V3_CONNECT_CSV.gms',
            'path_base_file': 'WB_EPM_v8_5_daily_storage_base.gms',
            'path_report_file': 'WB_EPM_v8_5_Report.gms',
            'path_reader_file': 'WB_EPM_daily_storage_input_readers.gms',
            'path_cplex_file': 'cplex.opt'
        }

        launch_epm_multi_scenarios(scenario_baseline='input/scenario_hydrostorage_baseline_SP2_candidate.csv',
                                   scenarios_specification=path_scenario_spec,
                                   selected_scenarios=None,
                                   cpu=1, path_gams=path_gams_storage,
                                   path_engine_file='Engine_Base.gms')

        # get_job_engine('output/simulations_run_20250107_141155/tokens_simulation.csv')