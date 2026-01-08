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

import os
import subprocess
import shlex
import pandas as pd
import datetime
import logging
from multiprocessing import Pool
import shutil
from gams import GamsWorkspace
import json
import argparse
import math
import re
from pathlib import Path
import sys
import numpy as np
from typing import Dict, List, Optional

from preprocessing import *

from postprocessing.utils import (
    parse_gams_solver_log_file,
    path_to_extract_results,
    set_utils_logger,
)
from postprocessing.postprocessing import postprocess_output

PATH_GAMS = {
    'path_main_file': 'main.gms',
    'path_base_file': 'base.gms',
    'path_report_file': 'generate_report.gms',
    'path_reader_file': 'input_readers.gms',
    'path_demand_file': 'generate_demand.gms',
    'path_hydrogen_file': 'hydrogen_module.gms'
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = Path(BASE_DIR) / "epm.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_ENV_FLAG = "EPM_LOG_INITIALIZED"


def _configure_logger():
    """Configure and return module logger with stream and file handlers."""
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("epm")
    if not os.environ.get(_ENV_FLAG):
        try:
            LOG_FILE_PATH.unlink()
        except FileNotFoundError:
            pass
        os.environ[_ENV_FLAG] = "1"

    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    return logger


logger = _configure_logger()
set_utils_logger(logger)


def copy_log_to_directory(target_directory, log_file_path=LOG_FILE_PATH):
    """Copy the log file into the target directory."""
    destination = Path(target_directory) / log_file_path.name
    try:
        shutil.copy2(log_file_path, destination)
    except (FileNotFoundError, PermissionError):
        logger.warning("Unable to copy log file to %s", destination)


def _log_solver_metrics_for_scenario(
    log_path: Path, scenario_name: str
) -> Optional[Dict[str, Optional[float]]]:
    """Parse a solver log file and append the summary to the main logger."""
    try:
        metrics = parse_gams_solver_log_file(log_path)
    except FileNotFoundError:
        logger.warning("Solver log missing for scenario %s: %s", scenario_name, log_path)
        return
    except OSError as exc:
        logger.warning(
            "Unable to read solver log for scenario %s at %s: %s",
            scenario_name,
            log_path,
            exc,
        )
        return
    except Exception as exc:  # pragma: no cover - defensive safeguard
        logger.warning(
            "Solver log parsing failed for scenario %s at %s: %s",
            scenario_name,
            log_path,
            exc,
        )
        return

    parts = []
    objective = metrics.get("Objective (Billion USD)")
    if objective is not None:
        parts.append(f"objective {objective:.3f} B$")

    elapsed = metrics.get("Time (s)")
    if elapsed is not None:
        parts.append(f"elapsed {elapsed:.3f} s")

    peak_memory = metrics.get("Peak Memory (Mb)")
    if peak_memory is not None:
        parts.append(f"peak memory {peak_memory:.1f} MB")

    summary = {
        "Objective (Billion USD)": objective,
        "Time (s)": elapsed,
        "Peak Memory (Mb)": peak_memory,
    }

    if parts:
        logger.info(
            "Scenario %s solver summary (%s): %s",
            scenario_name,
            log_path.name,
            ", ".join(parts),
        )
    else:
        logger.info(
            "Scenario %s solver summary (%s): metrics unavailable",
            scenario_name,
            log_path.name,
        )

    return summary


def _write_solver_metrics_table(
    output_folder: str, metrics_results: List[Optional[Dict[str, Optional[float]]]]
) -> None:
    """Persist solver metrics to a CSV file within the simulation folder."""
    rows = []
    for item in metrics_results or []:
        if not isinstance(item, dict):
            continue
        scenario = item.get("scenario")
        if scenario is None:
            continue
        rows.append(
            {
                "scenario": scenario,
                "Objective (Billion USD)": item.get("Objective (Billion USD)"),
                "Time (s)": item.get("Time (s)"),
                "Peak Memory (Mb)": item.get("Peak Memory (Mb)"),
            }
        )

    if not rows:
        logger.info("No solver metrics captured for %s", output_folder)
        return

    df_metrics = pd.DataFrame(rows)
    df_metrics = df_metrics[
        [
            "scenario",
            "Objective (Billion USD)",
            "Time (s)",
            "Peak Memory (Mb)",
        ]
    ]
    df_metrics["Objective (Billion USD)"] = df_metrics["Objective (Billion USD)"].round(2)

    def _format_time(seconds: Optional[float]) -> Optional[str]:
        if seconds is None or pd.isna(seconds):
            return None
        if seconds < 60:
            return f"{seconds:.2f} s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.2f} min"
        days = seconds / 86400
        return f"{days:.2f} j"

    df_metrics["Time"] = df_metrics["Time (s)"].apply(_format_time)
    df_metrics = df_metrics.drop(columns=["Time (s)"])
    df_metrics = df_metrics[
        [
            "scenario",
            "Objective (Billion USD)",
            "Time",
            "Peak Memory (Mb)",
        ]
    ]

    metrics_path = Path(output_folder) / "solver_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df_metrics.to_csv(metrics_path, index=False)
    except OSError as exc:
        logger.warning("Unable to write solver metrics table to %s: %s", metrics_path, exc)
    else:
        logger.info("Solver metrics table saved to %s", metrics_path)


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
               modeltype='MIP',
               folder_input=None,
               dict_montecarlo=None,
               prefix='',  # 'simulation_'
               debug=False,
               trace=False
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
    folder_input: str, optional, default None
    dict_montecarlo: dict, optional, default None
        Correspondence for solution when running montecarlo scenarios

    Returns
    -------
    dict
        Dictionary containing the scenario name, log filename, and solver metrics
        (objective in billions of dollars, elapsed time in seconds, and peak
        memory in megabytes). Output files are written to the
        scenario-specific folder.
    """

    # If no scenario name is provided, use the current date and time
    if scenario_name == '':
        scenario_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Each simulation has its own subfolder to store the results
    folder = f'{prefix}{scenario_name}'
    if not os.path.exists(folder):
        os.mkdir(folder)
    cwd = os.path.join(os.getcwd(), folder)

    # Arguments for GAMS
    if dict_montecarlo is not None:  # running in Monte-Carlo setting
        scenario['reportshort'] = 1  # shorter report to save memory
        scenario['solvemode'] = 2    # solve without checkpoint
    path_args = ['--{} {}'.format(k, i) for k, i in scenario.items()]

    # Define the logfile name
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
                                                    "--FOLDER_INPUT {}".format(folder_input)
                                                    ] + path_args

    if debug:
        command.append("--DEBUG 1")

    if trace:
        command.append("--TRACE 1")

    command_str = " ".join(shlex.quote(part) for part in command)
    logger.info("Command to execute: %s", command_str)

    if sys.platform.startswith("win"):  # If running on Windows
        rslt = subprocess.run(' '.join(command), cwd=cwd, shell=True)
    else:  # For Linux or macOS
        rslt = subprocess.run(command, cwd=cwd)

    if rslt.returncode != 0:
        raise RuntimeError('GAMS Error: check GAMS logs file ')

    metrics = _log_solver_metrics_for_scenario(Path(cwd) / logfile, scenario_name) or {}

    return {
        "scenario": scenario_name,
        "logfile": logfile,
        "Objective (Billion USD)": metrics.get("Objective (Billion USD)"),
        "Time (s)": metrics.get("Time (s)"),
        "Peak Memory (Mb)": metrics.get("Peak Memory (Mb)"),
    }


def launch_epm_multiprocess(df, scenario_name, path_gams, folder_input=None,
                            modeltype='MIP', dict_montecarlo=None, debug=False, trace=False):
    return launch_epm(df, scenario_name=scenario_name, folder_input=folder_input,
                      dict_montecarlo=dict_montecarlo, debug=debug, trace=trace, **path_gams, modeltype=modeltype)


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
                               generator_assessment=None,
                               interco_assessment=None,
                               simple=None,
                               simulation_label=None,
                               modeltype='MIP',
                               debug=False,
                               trace=False):
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
    debug: bool, optional, default False
        When True, passes ``--DEBUG 1`` to the GAMS command line.
    trace: bool, optional, default False
        When True, passes ``--TRACE 1`` to the GAMS command line.
    simulation_label: str, optional, default None
        Custom name for the simulation output folder that overrides the timestamp-based name.

    Returns
    -------
    tuple
        A tuple ``(folder, metrics)`` where ``folder`` is the path to the
        simulation output directory and ``metrics`` is a list of dictionaries
        with solver metrics per scenario.
    """

    working_directory = os.getcwd()

    # Read the scenario CSV file
    if path_gams is not None:  # path for required gams file is provided
        path_gams = {k: os.path.join(working_directory, i) for k, i in path_gams.items()}
    else:  # use default configuration
        path_gams = {k: os.path.join(working_directory, i) for k, i in PATH_GAMS.items()}
        
    # Create the full path folder input
    folder_input = os.path.join(os.getcwd(), folder_input)

    # Read configuration file
    config_path = os.path.join(folder_input, config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Configuration file {os.path.abspath(config_path)} not found.')

    required_columns = ['paramNames', 'file']
    try:
        config = pd.read_csv(config_path, usecols=required_columns)
    except ValueError as err:
        if 'Usecols do not match columns' in str(err):
            header_columns = pd.read_csv(config_path, nrows=0).columns
            missing_columns = [col for col in required_columns if col not in header_columns]
            logger.error(
                "Missing required columns %s in %s.",
                missing_columns,
                os.path.abspath(config_path)
            )
            raise ValueError(f"Missing required columns {missing_columns} in {os.path.abspath(config_path)}.") from None
        raise
    config = config.set_index('paramNames')['file']
    config = config.dropna()
    # Normalize path
    config = normalize_path(config)
    
    if modeltype is not None:
        config.at['modeltype'] = modeltype

    # Add the baseline scenario
    s = {'baseline': config}

    # Add scenarios if scenarios_specification is defined
    if scenarios_specification is not None:
        scenarios_specification = os.path.join(folder_input, scenarios_specification)
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
        s[k] = s[k].apply(
            lambda i: os.path.join(folder_input, i) if any(ext in i for ext in ['.csv', '.opt']) else i
        )

    # Set up project assessment scenarios if activated (before sensitivity so sensitivity applies to both)
    if project_assessment is not None:
        s = perform_project_assessment(project_assessment, s)

    # Run sensitivity analysis if activated (applies to all existing scenarios)
    if sensitivity is not None:
        s = perform_sensitivity(sensitivity, s)

    # Set up generator assessment scenarios if activated
    if generator_assessment is not None:
        s = perform_assessment(generator_assessment, s)
        
    # Set up interconnection assessment scenarios if activated
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
        mc_names = list(scenarios_montecarlo.keys())
        logger.info("Prepared %d Monte Carlo scenario(s): %s", len(mc_names), ", ".join(mc_names))

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

    scenario_names = list(s.keys())
    logger.info("Prepared %d scenario(s): %s", len(scenario_names), ", ".join(scenario_names))

    # Create dir for simulation and change current working directory
    if 'output' not in os.listdir():
        os.mkdir('output')

    pre = 'simulations_run'
    if sensitivity is not None:
        pre = 'sensitivity_run'
    if project_assessment is not None:
        pre = 'project_assessment_run'
    if generator_assessment is not None:
        pre = 'generator_assessment_run'
    if simulation_label:
        folder_name = simulation_label
    else:
        folder_name = '{}_{}'.format(pre, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder = os.path.join('output', folder_name)
    if os.path.exists(folder):
        logger.info('Folder exists, recreating: %s', folder)
        shutil.rmtree(folder)
    os.mkdir(folder)
    logger.info('Folder ready: %s', folder)
    os.chdir(folder)

    # Export scenario.csv file
    df = pd.DataFrame(s).copy()
    # Extracts everything after 'epm' folder in a path (cross-platform).
    def extract_path(path):
        match = re.search(r"[/\\]epm[/\\](.*)", path)
        return match.group(1).replace("\\", "/") if match else path
    df = df.astype(str).map(extract_path)
    df.to_csv('input_scenarios.csv')

    if montecarlo:
        samples_mc = pd.DataFrame(samples)
        samples_mc.columns = samples_mc.columns.map(lambda col: col.replace('.', 'p'))
        samples_mc.to_csv('samples_montecarlo.csv')

    # Run EPM in multiprocess
    metrics_results: List[Optional[Dict[str, Optional[float]]]] = []

    if not montecarlo:
        with Pool(cpu) as pool:
            metrics_results.extend(
                 pool.starmap(
                    launch_epm_multiprocess,
                    [(s[k], k, path_gams, folder_input, modeltype, None, debug, trace) for k in s.keys()],
                )
            )
    else:
        # First, run initial scenarios
        # Ensure config file has extended setting to save output
        with Pool(cpu) as pool:
            for k in s.keys():
                assert s[k]['solvemode'] == '1', 'Parameter solvemode should be set to 1 in the configuration to obtain extended output for the baseline scenarios in the Monte-Carlo analysis.'
            metrics_results.extend(
                pool.starmap(
                    launch_epm_multiprocess,
                    [(s[k], k, path_gams, folder_input, modeltype, None, debug, trace) for k in s.keys()],
                )
            )
        # Modify config file to ensure limited output saved
        with Pool(cpu) as pool:  # running montecarlo scenarios in multiprocessing
            metrics_results.extend(
                pool.starmap(
                    launch_epm_multiprocess,
                    [
                        (scenarios_montecarlo[k], k, path_gams, folder_input, modeltype, dict_montecarlo, debug, trace)
                        for k in scenarios_montecarlo.keys()
                    ],
                )
            )

    os.chdir(working_directory)

    _write_solver_metrics_table(folder, metrics_results)

    return folder, metrics_results


def main(test_args=None):
    """Parse command-line arguments and orchestrate an EPM simulation run.

    Args:
        test_args (list[str] | None): Optional list of arguments passed in tests.
    """
    logger.info("Starting EPM workflow")
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
        "--modeltype",
        type=str,
        default=None,
        help="Sover to use in GAMS (default: MIP)."
    )

    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="?",
        const="scenarios.csv",
        default=None,
        help="Scenario file name (default when flag used without value: scenarios.csv)"
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
        "--selected_scenarios", "--selected-scenarios", "-S",
        dest="selected_scenarios",
        nargs="+",  # Accepts one or more values
        type=str,
        default=None,
        help="List of selected scenarios (default: None). Example usage: -S baseline HighDemand"
    )

    parser.add_argument(
        "--cpu",
        type=int,
        default=1,
        help="Number of CPUs (default: 1)"
    )

    parser.add_argument(
        "--generator_assessment",
        nargs="+",  # Accepts one or more values
        type=str,
        default=None,
        help="Name of the project to assess (default: None). Example usage: --generator_assessment Solar"
    )
    parser.add_argument(
        "--project_assessment",
        type=str,
        default=None,
        help="Suffix or filename for an alternate pGenDataInput (default: None). Example: --project_assessment rehabilitation"
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

    parser.add_argument(
        "--simulation_label",
        type=str,
        default=None,
        help="Optional label for the simulation results folder (default: timestamp-based name)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable verbose DEBUG mode for GAMS (passes --DEBUG 1 to the solver)"
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Enable TRACE mode for GAMS (passes --TRACE 1 to the solver)"
    )

    # If test_args is provided (for testing), use it instead of parsing from the command line
    if test_args:
        args = parser.parse_args(test_args)
    else:
        args = parser.parse_args()  # Normal command-line parsing

    # Custom logic
    if args.simple == []:
        args.simple = ['DiscreteCap', 'y']
        

    folder_input = os.path.join("input", args.folder_input)
    folder = None
    results_folder = None

    try:
        logger.info("Config file: %s", args.config)
        logger.info("Folder input: %s", args.folder_input)
        logger.info("modeltype: %s", args.modeltype)
        logger.info("Scenarios file: %s", args.scenarios)
        logger.info("Sensitivity: %s", args.sensitivity)
        logger.info("MonteCarlo: %s", args.montecarlo)
        logger.info("MonteCarlo samples: %s", args.montecarlo_samples)
        logger.info("Monte Carlo uncertainties file: %s", args.uncertainties)
        logger.info("Reduced output: %s", args.reduced_output)
        logger.info("Reduced definition csv: %s", args.reduce_definition_csv)
        logger.info("Selected scenarios: %s", args.selected_scenarios)
        logger.info("Simple: %s", args.simple)
        logger.info("Simulation label: %s", args.simulation_label)
        logger.info("Debug flag: %s", args.debug)
        logger.info("Trace flag: %s", args.trace)
        
        if args.sensitivity:
            sensitivity = os.path.join(folder_input, 'sensitivity.csv')
            if not os.path.exists(sensitivity):
                logger.warning(
                    "Sensitivity file %s does not exist. No sensitivity analysis will be performed.",
                    os.path.abspath(sensitivity)
                )
            sensitivity = pd.read_csv(sensitivity, index_col=0).to_dict()['sensitivity']
            logger.info("Sensitivity analysis: %s", sensitivity)

        else:
            sensitivity = None

        # If none do not run EPM
        if args.postprocess is None:
            folder, result = launch_epm_multi_scenarios(config=args.config,
                                                        folder_input=folder_input,
                                                        scenarios_specification=args.scenarios,
                                                        sensitivity=sensitivity,
                                                        montecarlo=args.montecarlo,
                                                        montecarlo_nb_samples=args.montecarlo_samples,
                                                        uncertainties=args.uncertainties,
                                                        selected_scenarios=args.selected_scenarios,
                                                        cpu=args.cpu,
                                                        project_assessment=args.project_assessment,
                                                        generator_assessment=args.generator_assessment,
                                                        interco_assessment=args.interco_assessment,
                                                        simple=args.simple,
                                                        simulation_label=args.simulation_label,
                                                        modeltype=args.modeltype,
                                                        debug=args.debug,
                                                        trace=args.trace)
        else:
            logger.info("Project folder: %s", args.postprocess)
            logger.info("EPM does not run again but use the existing simulation within the folder")
            folder = args.postprocess
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Folder {os.path.abspath(folder)} does not exist. Please provide a valid folder with EPM results.")
            else:
                logger.info("Find folder %s for postprocessing.", os.path.abspath(folder))

        # Define scenario reference
        scenarios = [
            i
            for i in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, i))
            and 'epmresults.gdx' in os.listdir(os.path.join(folder, i))
        ]
        if not scenarios:
            raise RuntimeError(
                "No scenario folder contains epmresults.gdx; GAMS optimization appears to have failed for all launches."
            )
        scenario_reference = 'baseline'
        if scenario_reference not in scenarios:
            scenario_reference = scenarios[0]
                 
        # Launch postprocess
        logger.info("Starting postprocessing for folder: %s", folder)
        postprocess_output(folder, reduced_output=args.reduced_output, scenario_reference=scenario_reference,
                           selected_scenario=args.plot_selected_scenarios, plot_dispatch=args.plot_dispatch,
                           graphs_folder=args.graphs_folder, montecarlo=args.montecarlo, 
                           reduce_definition_csv=args.reduce_definition_csv, logger=logger)
        logger.info("Postprocessing completed for folder: %s", folder)

        # Zip the folder if it exists
        results_folder = path_to_extract_results(folder)
        if results_folder and os.path.exists(results_folder):
            copy_log_to_directory(results_folder)

        if args.output_zip and results_folder and os.path.exists(results_folder):
            logger.info("Compressing results folder %s", results_folder)
            zip_path = results_folder + '.zip'
            shutil.make_archive(results_folder, 'zip', results_folder)
            shutil.rmtree(results_folder)  # Remove the original folder
            logger.info("Folder %s zipped as %s", results_folder, zip_path)
    except Exception:
        logger.exception("EPM workflow failed")
        target = results_folder or folder
        if target and os.path.exists(target):
            copy_log_to_directory(target)
        raise

    logger.info("EPM workflow completed")


if __name__ == '__main__':
    main()
