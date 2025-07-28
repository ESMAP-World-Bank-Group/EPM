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
from pathlib import Path
# Relave imports as it's a submodule
from .utils import *
from .interactive_map import make_automatic_map

def process_simulation_results(FOLDER, SCENARIOS_RENAME=None, folder='postprocessing',
                               graphs_folder = 'img', keys_results=None):
        # Create the folder path
        def adjust_color(color, factor=0.1):
            """Adjusts the color slightly by modifying its HSL components."""
            rgb = mcolors.to_rgb(color)  # Convert to RGB
            h, l, s = colorsys.rgb_to_hls(*rgb)  # Convert to HLS

            # Adjust lightness slightly to differentiate (factor controls how much)
            l = min(1, max(0, l + factor * (0.5 - l)))

            # Convert back to RGB
            new_rgb = colorsys.hls_to_rgb(h, l, s)
            return mcolors.to_hex(new_rgb)

        RESULTS_FOLDER = path_to_extract_results(FOLDER)

        GRAPHS_FOLDER = os.path.join(RESULTS_FOLDER, graphs_folder)
        if not os.path.exists(GRAPHS_FOLDER):
            os.makedirs(GRAPHS_FOLDER)
            print(f'Created folder {GRAPHS_FOLDER}')

        # Read the plot specifications
        dict_specs = read_plot_specs(folder=folder)

        # Extract and process EPM inputs
        epm_input = extract_epm_folder(RESULTS_FOLDER, file='input.gdx')
        epm_input = process_epm_inputs(epm_input, dict_specs, scenarios_rename=SCENARIOS_RENAME)
        mapping_gen_fuel = epm_input['pGenDataExcel'].loc[:, ['scenario', 'generator', 'fuel']]
        mapping_zone_country = epm_input['zcmap'].loc[:, ['scenario', 'zone', 'country']]

        # Extract and process EPM results
        epm_results = extract_epm_folder(RESULTS_FOLDER, file='epmresults.gdx')
        epm_results = process_epm_results(epm_results, dict_specs, scenarios_rename=SCENARIOS_RENAME,
                                        mapping_gen_fuel=mapping_gen_fuel, mapping_zone_country=mapping_zone_country,
                                        keys=keys_results)

        # Update color dict with plant colors
        if True:
            if 'pCapacityPlan' in epm_results.keys():
                temp = epm_results['pCapacityPlan'].copy()
                plant_fuel_pairs = temp[['generator', 'fuel']].drop_duplicates()

                # Map base colors from fuel types
                plant_fuel_pairs['colors'] = plant_fuel_pairs['fuel'].map(dict_specs['colors'])

                # Generate slightly varied colors for each generator
                plant_fuel_pairs['colors'] = plant_fuel_pairs.apply(
                    lambda row: adjust_color(row['colors'], factor=0.2 * hash(row['generator']) % 5), axis=1
                )

                # Create the mapping
                plant_to_color = dict(zip(plant_fuel_pairs['generator'], plant_fuel_pairs['colors']))

                # Update dict_specs with the new colors
                dict_specs['colors'].update(plant_to_color)

        return RESULTS_FOLDER, GRAPHS_FOLDER, dict_specs, epm_input, epm_results, mapping_gen_fuel


def make_automatic_dispatch(epm_results, dict_specs, GRAPHS_FOLDER, selected_scenarios):

    dfs_to_plot_area = {
        'pPlantDispatch': filter_dataframe(epm_results['pPlantDispatch'], {'attribute': ['Generation']}),
        'pDispatch': filter_dataframe(epm_results['pDispatch'], {'attribute': ['Unmet demand', 'Exports', 'Imports', 'Storage Charge']})
    }

    dfs_to_plot_line = {
        'pDispatch': filter_dataframe(epm_results['pDispatch'], {'attribute': ['Demand']})
    }

    for selected_scenario in selected_scenarios:
        folder = f'{GRAPHS_FOLDER}/{selected_scenario}'
        if not os.path.exists(folder):
            os.mkdir(folder)
        for zone in epm_results['pDispatch'].loc[epm_results['pDispatch'].scenario == selected_scenario]['zone'].unique():
            years = epm_results['pDispatch']['year'].unique()

            # Select first and last years
            years = [min(years), max(years)]
            for year in years:
                filename = f'{folder}/dispatch/Dispatch_{selected_scenario}_{zone}_{year}.png'

                # Select season min and max
                conditions = {'scenario': 'baseline', 'zone': zone, 'year': year, 'attribute': 'Demand'}
                temp = epm_results['pDispatch'].copy()
                temp = filter_dataframe(temp, conditions)
                t = temp.groupby('season', observed=False)['value'].sum()
                s_max, s_min = t.idxmax(), t.idxmin()
                temp = filter_dataframe(temp, {'season': [s_min, s_max]})

                # Select the day with max demand
                d = temp.groupby(['day'], observed=False)['value'].sum().idxmax()

                select_time = {'season': [s_min, s_max], 'day': [d]}
                make_complete_fuel_dispatch_plot(dfs_to_plot_area, dfs_to_plot_line, dict_specs['colors'],
                                                 zone=zone, year=year, scenario=selected_scenario,
                                                 fuel_grouping=None, select_time=select_time, filename=filename,
                                                 bottom=None, legend_loc='bottom')
                select_time = {'season': [s_max]}
                make_complete_fuel_dispatch_plot(dfs_to_plot_area, dfs_to_plot_line, dict_specs['colors'],
                                                 zone=zone, year=year, scenario=selected_scenario,
                                                 fuel_grouping=None, select_time=select_time, filename=filename,
                                                 bottom=None, legend_loc='bottom')


def postprocess_montecarlo(epm_results, RESULTS_FOLDER, GRAPHS_FOLDER):
    simulations_scenarios = pd.read_csv(os.path.join(RESULTS_FOLDER, 'input_scenarios.csv'), index_col=0)
    samples_mc = pd.read_csv(os.path.join(RESULTS_FOLDER, 'samples_montecarlo.csv'), index_col=0)
    samples_mc_substrings = set(samples_mc.columns)

    def is_not_subset(col):
        return not any(sample in col for sample in samples_mc_substrings)
    original_scenarios = [c for c in simulations_scenarios.columns if is_not_subset(c)]

    df_summary = epm_results['pSummary'].copy()
    df_summary = df_summary.loc[df_summary.attribute.isin(['NPV of system cost: $m'])]
    df_summary_baseline = df_summary.loc[df_summary.scenario.isin(original_scenarios)]
    df_summary_baseline = df_summary_baseline.drop(columns=['attribute']).set_index('scenario')
    df_summary['scenario_mapping'] = df_summary.apply(lambda row: next(c for c in original_scenarios if c in row['scenario']), axis=1)
    df_summary = df_summary.groupby('scenario_mapping').value.describe()[['min', 'max']].reset_index().rename(columns={'scenario_mapping': 'scenario'})
    df_summary = df_summary.set_index('scenario').stack().to_frame().rename(columns={0: 'value'})
    df_summary.index.names = ['scenario', 'error']
    df_summary.reset_index(inplace=True)

    filename = f'{folder_comparison}/NPV_montecarlo.png'

    make_stacked_bar_subplots(df_summary_baseline, filename, dict_colors=None, df_errorbars=df_summary, selected_zone=None,
                                selected_year=None, column_xaxis=None, column_stacked=None, column_multiple_bars='scenario',
                                column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None,
                                dict_scenarios=None,
                                format_y=lambda y, _: '{:.0f} m$'.format(y), order_stacked=None, cap=2,
                                annotate=False, show_total=False, fonttick=12, rotation=45, title=None)

    df_cost_summary = epm_results['pCostSummary'].copy()
    # df_cost_summary = df_cost_summary.loc[df_cost_summary.attribute.isin(['Total Annual Cost by Zone: $m'])]
    df_cost_summary_baseline = df_cost_summary.loc[df_cost_summary.scenario.isin(original_scenarios)]
    df_cost_summary['scenario_mapping'] = df_cost_summary.apply(lambda row: next(c for c in original_scenarios if c in row['scenario']), axis=1)
    df_cost_summary = df_cost_summary.groupby(['scenario', 'scenario_mapping', 'zone', 'year']).value.sum().reset_index().groupby(['scenario_mapping', 'zone', 'year']).value.describe()[['min', 'max']].reset_index().rename(columns={'scenario_mapping': 'scenario'})
    df_cost_summary = df_cost_summary.set_index(['scenario', 'zone', 'year']).stack().to_frame().rename(columns={0: 'value'})
    df_cost_summary.index.names = ['scenario', 'zone', 'year', 'error']
    df_cost_summary.reset_index(inplace=True)

    costs_notrade = ["Annualized capex: $m", "Fixed O&M: $m", "Variable O&M: $m", "Total fuel Costs: $m", "Transmission additions: $m",
                        "Spinning Reserve costs: $m", "Unmet demand costs: $m", "Excess generation: $m",
                        "VRE curtailment: $m", "Import costs wiht external zones: $m", "Export revenues with external zones: $m",
                        # "Import costs with internal zones: $m", "Export revenues with internal zones: $m"
                        ]
    df_cost_summary_no_trade = epm_results['pCostSummaryFull'].copy()
    df_cost_summary_no_trade = df_cost_summary_no_trade.loc[df_cost_summary_no_trade.attribute.isin(costs_notrade)]
    df_cost_summary_baseline_notrade = df_cost_summary_no_trade.loc[(df_cost_summary_no_trade.scenario.isin(original_scenarios))]
    df_cost_summary_no_trade['scenario_mapping'] = df_cost_summary_no_trade.apply(lambda row: next(c for c in original_scenarios if c in row['scenario']), axis=1)
    df_cost_summary_no_trade = df_cost_summary_no_trade.groupby(['scenario', 'scenario_mapping', 'zone', 'year']).value.sum().reset_index().groupby(['scenario_mapping', 'zone', 'year']).value.describe()[['min', 'max']].reset_index().rename(columns={'scenario_mapping': 'scenario'})
    df_cost_summary_no_trade = df_cost_summary_no_trade.set_index(['scenario', 'zone', 'year']).stack().to_frame().rename(columns={0: 'value'})
    df_cost_summary_no_trade.index.names = ['scenario', 'zone', 'year', 'error']
    df_cost_summary_no_trade.reset_index(inplace=True)

    demand_supply = ["Unmet demand: GWh", "Imports exchange: GWh", "Exports exchange: GWh"
                        ]

    df_demandsupply = epm_results['pDemandSupply'].copy()
    df_demandsupply = df_demandsupply.loc[df_demandsupply.attribute.isin(demand_supply)]
    df_demandsupply_baseline = df_demandsupply.loc[(df_demandsupply.scenario.isin(original_scenarios))]
    df_demandsupply['scenario_mapping'] = df_demandsupply.apply(lambda row: next(c for c in original_scenarios if c in row['scenario']), axis=1)

    df_demandsupply = (
        df_demandsupply
        .groupby(['scenario_mapping', 'zone', 'year', 'attribute'])['value']
        .describe()[['min', 'max']]
        .reset_index()
        .rename(columns={'scenario_mapping': 'scenario'})
    )

    # Format to the format expected by the plotting function
    df_demandsupply = (
        df_demandsupply
        .set_index(['scenario', 'zone', 'year', 'attribute'])  # attribute is now part of the index!
        .stack()  # gives 'min' and 'max' as a new level in index
        .to_frame(name='value')
        .reset_index()
        .rename(columns={'level_4': 'error'})
    )

    full_index = df_demandsupply.set_index(['scenario', 'zone', 'year', 'attribute']).index.unique()
    df_demandsupply_baseline = (
        df_demandsupply_baseline
        .set_index(['scenario', 'zone', 'year', 'attribute'])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )  # adding elements which only have error bars

    years = sorted(epm_results['pCostSummary']['year'].unique())

    n = len(years)
    middle_index = n // 2
    years = sorted({years[0], years[middle_index], years[-1]})

    for year in years:

        filename = f'{folder_comparison}/AnnualCostWithTrade_montecarlo_{year}.png'

        make_stacked_bar_subplots(df_cost_summary_baseline, filename, dict_colors=None, df_errorbars=df_cost_summary, selected_zone=None,
                                    selected_year=year, column_xaxis='zone', column_stacked='attribute', column_multiple_bars='scenario',
                                    column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None,
                                    dict_scenarios=None,
                                    format_y=lambda y, _: '{:.0f} m$'.format(y), order_stacked=None, cap=2,
                                    annotate=False, show_total=False, fonttick=12, rotation=45, title=None)


        filename = f'{folder_comparison}/AnnualCost_montecarlo_{year}.png'

        make_stacked_bar_subplots(df_cost_summary_baseline_notrade, filename, dict_colors=None, df_errorbars=df_cost_summary_no_trade, selected_zone=None,
                                    selected_year=year, column_xaxis='zone', column_stacked='attribute', column_multiple_bars='scenario',
                                    column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None,
                                    dict_scenarios=None,
                                    format_y=lambda y, _: '{:.0f} m$'.format(y), order_stacked=None, cap=2,
                                    annotate=False, show_total=False, fonttick=12, rotation=45, title=None)

    zones = epm_results['pCostSummary']['zone'].unique()
    for zone in zones:
        # Only interested in subset of years
        df_cost_summary_baseline = df_cost_summary_baseline.loc[df_cost_summary_baseline.year.isin(years)]
        df_cost_summary = df_cost_summary.loc[df_cost_summary.year.isin(years)]
        df_cost_summary_baseline_notrade = df_cost_summary_baseline_notrade.loc[df_cost_summary_baseline_notrade.year.isin(years)]
        df_cost_summary_no_trade = df_cost_summary_no_trade.loc[df_cost_summary_no_trade.year.isin(years)]
        df_demandsupply_baseline = df_demandsupply_baseline.loc[df_demandsupply_baseline.year.isin(years)]
        df_demandsupply = df_demandsupply.loc[df_demandsupply.year.isin(years)]

        filename = f'{folder_comparison}/AnnualCostWithTrade_montecarlo_{zone}.png'

        make_stacked_bar_subplots(df_cost_summary_baseline, filename, dict_colors=None, df_errorbars=df_cost_summary, selected_zone=zone,
                                    selected_year=None, column_xaxis='year', column_stacked='attribute', column_multiple_bars='scenario',
                                    column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None,
                                    dict_scenarios=None,
                                    format_y=lambda y, _: '{:.0f} m$'.format(y), order_stacked=None, cap=2,
                                    annotate=False, show_total=False, fonttick=12, rotation=45, title=None)

        filename = f'{folder_comparison}/AnnualCost_montecarlo_{zone}.png'

        make_stacked_bar_subplots(df_cost_summary_baseline_notrade, filename, dict_colors=None, df_errorbars=df_cost_summary_no_trade, selected_zone=zone,
                                    selected_year=None, column_xaxis='year', column_stacked='attribute', column_multiple_bars='scenario',
                                    column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None,
                                    dict_scenarios=None,
                                    format_y=lambda y, _: '{:.0f} m$'.format(y), order_stacked=None, cap=2,
                                    annotate=False, show_total=False, fonttick=12, rotation=45, title=None)

        filename = f'{folder_comparison}/DemandSupply_montecarlo_{zone}.png'

        make_stacked_bar_subplots(df_demandsupply_baseline, filename, dict_colors=None, df_errorbars=df_demandsupply, selected_zone=zone,
                                    selected_year=None, column_xaxis='attribute', column_stacked='year', column_multiple_bars='scenario',
                                    column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None,
                                    dict_scenarios=None,
                                    format_y=lambda y, _: '{:.0f} GWh'.format(y), order_stacked=None, cap=2,
                                    annotate=False, show_total=False, fonttick=12, rotation=45, title=None, juxtaposed=True)


def postprocess_output(FOLDER, reduced_output=False, folder='', selected_scenario='all',
                       plot_dispatch=None, scenario_reference='baseline', graphs_folder='img',
                       montecarlo=False, reduce_definition_csv=False):
    
    def reduce_year_definition(folder_csv):
        """
        For each CSV in the folder, keeps only rows corresponding to the first, middle, and last year.

        Parameters:
        - folder_csv (str): Folder containing CSV files with a 'year' column.
        - output_suffix (str): Suffix to add to the output filenames.
        """

        for filename in os.listdir(folder_csv):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_csv, filename)
                df = pd.read_csv(file_path)

                if not df.empty:
                    if 'y' not in df.columns:
                        print(f"Skipping {filename}: 'year' column not found.")
                        continue

                    # Pick first, middle, and last years
                    years = sorted(df['y'].unique())
                    n = len(years)
                    middle_index = n // 2
                    years_to_keep = sorted({years[0], years[middle_index], years[-1]})

                    df_filtered = df[df['y'].isin(years_to_keep)]

                    df_filtered.to_csv(file_path, index=False)


    # Output csv are reduced in size to improve Tableau performance
    if reduce_definition_csv:  
        if 'output' not in FOLDER:
            results_folder = os.path.join('output', FOLDER)
        else:
            results_folder = FOLDER
        for scenario in [i for i in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, i))]:
            if 'output_csv' in os.listdir(os.path.join(results_folder, scenario)):
                folder_csv = os.path.join(results_folder, scenario, 'output_csv')
                reduce_year_definition(folder_csv=folder_csv)

    keys_results = None
    if montecarlo:
        keys_results = {'pSummary', 'pCostSummary', 'pCostSummaryFull', 'pDemandSupply'}

    # Process results
    RESULTS_FOLDER, GRAPHS_FOLDER, dict_specs, epm_input, epm_results, mapping_gen_fuel = process_simulation_results(
        FOLDER, SCENARIOS_RENAME=None, folder=folder, graphs_folder=graphs_folder, keys_results=keys_results)

    folder_comparison = os.path.join(GRAPHS_FOLDER, 'scenarios_comparison')
    if not os.path.exists(folder_comparison):
        os.mkdir(folder_comparison)

    # Specific postprocessing for Monte Carlo simulations
    if montecarlo:
        postprocess_montecarlo(epm_results, RESULTS_FOLDER, GRAPHS_FOLDER)

    # If not Monte Carlo, we proceed with the regular postprocessing
    if not montecarlo:

        if isinstance(selected_scenario, str):
            if selected_scenario == 'all':
                selected_scenarios = list(epm_results['pEnergyByFuel'].scenario.unique())  # we choose all scenarios
            else:
                selected_scenarios = [selected_scenario]
                assert selected_scenario in list(epm_results['pEnergyByFuel'].scenario.unique()), "Selected scenario does not belong to the set of scenarios."
        else:
            selected_scenarios = selected_scenario

        # Generate summary
        print('Generating summary...')
        generate_summary(epm_results, RESULTS_FOLDER, epm_input)

        # Generate detailed by plant to debug
        if not reduced_output:

            # Generate a detailed summary by Power Plant
            print('Generating detailed summary by Power Plant...')
            generate_summary_detailed(epm_results, RESULTS_FOLDER)

            # Define dataframes for capacity, energy, exchange
            df_capacityplan = epm_results['pCapacityPlan'].copy()
            df_capacityfuel = epm_results['pCapacityByFuel'].copy()
            df_energyplant = epm_results['pEnergyByPlant'].copy()
            df_energyfuel = epm_results['pEnergyByFuel'].copy()
            
            # Additionnal energy information not in pEnergyByFuel
            df_exchange = epm_results['pDemandSupply'].copy()
            df_exchange = df_exchange.loc[df_exchange['attribute'].isin(['Unmet demand: GWh', 'Exports exchange: GWh', 'Imports exchange: GWh'])]
            df_exchange = df_exchange.replace({'Unmet demand: GWh': 'Unmet demand',
                                              'Exports exchange: GWh': 'Exports',
                                              'Imports exchange: GWh': 'Imports'})
            # Put negative values when exports in colmun 'attribute'
            df_exchange['value'] = df_exchange.apply(lambda row: -row['value'] if row['attribute'] == 'Exports' else row['value'], axis=1)
            df_exchange.rename(columns={'attribute': 'fuel'}, inplace=True)
            # Define energyfuelfull to include exchange
            df_energyfuelfull = pd.concat([df_energyfuel, df_exchange], ignore_index=True)
            
            # Define dataframes with system costs 
            df_systemcost = epm_results['pSummary'].copy()
            df_systemcost = df_systemcost.loc[df_systemcost.scenario.isin(selected_scenarios)]
            attributes = [
                "Annualized capex: $m",
                "Fixed O&M: $m",
                "Variable O&M: $m",
                "Additional transmission costs: $m",
                "Fuel cost: $m",
                "Unmet demand cost: $m",
                "Trade Costs: $m",
                "Sys Spinning Reserve violation: $m",
                "Sys Planning Reserve violation: $m",
                "Zonal Spinning Reserve violation: $m",
                "Zonal Planning Reserve violation: $m"
            ] # "Trade Costs: $m", "NPV of system cost: $m",
            df_systemcost = df_systemcost.loc[df_systemcost.attribute.isin(attributes)]
            # Group reserve cost attributes into one unique attribute and sum the value
            reserve_cost_attributes = [
                "Sys Spinning Reserve violation: $m",
                "Sys Planning Reserve violation: $m",
                "Zonal Spinning Reserve violation: $m",
                "Zonal Planning Reserve violation: $m"
            ]
            df_reserve = df_systemcost[df_systemcost['attribute'].isin(reserve_cost_attributes)].copy()
            if not df_reserve.empty:
                df_reserve = df_reserve.groupby(
                    [col for col in df_reserve.columns if col not in ['attribute', 'value']],
                    as_index=False
                )['value'].sum()
                df_reserve['attribute'] = "Unmet reserve cost: $m"
            df_systemcost = pd.concat([
                df_systemcost[~df_systemcost['attribute'].isin(reserve_cost_attributes)],
                df_reserve
            ], ignore_index=True)
            df_systemcost = df_systemcost.replace({"Additional transmission costs: $m": 'Transmission additions: $m'})
            
            # ------------------------------------------------------------------------------------
            
            # Define selected scenario in case of no valid scenario names
            print('Generating scenario-specific figures...')
            for scenario in selected_scenarios:
                
                if scenario not in epm_results['pEnergyByPlant']['scenario'].unique():
                    print(f'No {scenario} in epm_results')
                    scenario = epm_results['pEnergyByPlant']['scenario'].unique()[0]
                    print(f'Selected scenario is set to: {scenario}')

                print('Creating folder for scenario:', scenario)
                folder_scenario = f'{GRAPHS_FOLDER}/{scenario}'
                if not os.path.exists(folder_scenario):
                    os.mkdir(folder_scenario)
                if not os.path.exists(Path(folder_scenario) / Path('capacity')):
                    os.mkdir(Path(folder_scenario) / Path('capacity'))
                if not os.path.exists(Path(folder_scenario) / Path('energy')):
                    os.mkdir(Path(folder_scenario) / Path('energy'))
                if not os.path.exists(Path(folder_scenario) / Path('dispatch')):
                    os.mkdir(Path(folder_scenario) / Path('dispatch'))
                if not os.path.exists(Path(folder_scenario) / Path('map')):
                    os.mkdir(Path(folder_scenario) / Path('map'))
                
                # ----------Scenario-specific capacity -----------------

                print('Generating scenario-specific capacity figures...')

                # Capacity evolution all zones
                filename = f'{folder_scenario}/capacity/CapacityEvolutionPerZone-{scenario}.png'
                make_stacked_bar_subplots(df_capacityfuel, filename, dict_specs['colors'], selected_zone=None, selected_year=None,
                                            column_xaxis='zone',
                                            column_stacked='fuel', column_multiple_bars='year',
                                            column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None,
                                            dict_scenarios=None,
                                            format_y=lambda y, _: '{:.0f} MW'.format(y), order_stacked=None, cap=2,
                                            annotate=False,
                                            show_total=False, fonttick=12, rotation=45, title=None)
                
                
                # Capacity mix in percentage by fuel by zone
                df_percentage = df_capacityfuel[df_capacityfuel['scenario'] == scenario]
                df_percentage = df_percentage.drop(columns=['scenario'])
                df_percentage = df_percentage.set_index(['zone', 'year', 'fuel']).squeeze()
                df_percentage = df_percentage / df_percentage.groupby(['zone', 'year']).sum()
                df_percentage = df_percentage.reset_index()
                
                filename = f'{folder_scenario}/capacity/CapacityMixClusteredStackedBarPlot.pdf'
                make_stacked_bar_subplots(df_percentage, filename, dict_specs['colors'], column_stacked='fuel',
                            column_xaxis='zone',
                            column_multiple_bars='year',
                            column_value='value',
                            format_y=lambda y, _: '{:.0f} %'.format(y * 100), rotation=45,
                            annotate=False,
                            title=f'Capacity Mix by Fuel {scenario}') 
                
                for year in [min(df_percentage['year']), max(df_percentage['year'])]:
                    df_year = df_percentage[df_percentage['year'] == year]
                    
                    filename = f'{folder_scenario}/capacity/CapacityMixStackedBarPlot-{year}.png'
                    make_stacked_bar_subplots(df_year, filename, dict_specs['colors'], column_stacked='fuel',
                                              column_xaxis=None,
                                              column_multiple_bars='zone',
                                              column_value='value',
                                              format_y=lambda y, _: '{:.0f} %'.format(y * 100), rotation=45,
                                              annotate=False,
                                              title=f'Capacity Mix by Fuel {year}',)
                
                
                # Capacity evolution per zone
                if len(df_capacityfuel.zone.unique()) > 1:
                    for zone in df_capacityfuel.zone.unique():
                        df_zone = df_capacityfuel.copy()
                        df_zone = df_zone[(df_zone['scenario'] == scenario) & (df_zone['zone'] == zone)]
                        filename = f'{folder_scenario}/capacity/CapacityEvolutionPerZone-{scenario}--{zone}.png'

                        make_stacked_bar_subplots(df_zone, filename, dict_specs['colors'], selected_zone=None, selected_year=None,
                                                  column_xaxis=None, column_stacked='fuel', column_multiple_bars='year',
                                                  column_value='value', select_xaxis=None, dict_grouping=None,
                                                  order_scenarios=None, dict_scenarios=None,
                                                  format_y=lambda y, _: '{:.0f} MW'.format(y), order_stacked=None, cap=2,
                                                  annotate=False, show_total=False, fonttick=12, rotation=45, title=None)
                
                
                
                # New capacity installed per zone
                for zone in df_capacityplan.zone.unique():
                    df_zone = df_capacityplan.copy()
                    df_zone = df_zone[(df_zone['scenario'] == scenario) & (df_zone['zone'] == zone)]
                    
                    filename = f'{folder_scenario}/capacity/NewCapacityInstalledTimeline-{scenario}-{zone}.png'
                    make_annotated_stacked_area_plot(df_zone, filename, dict_colors=dict_specs['colors'])
                        
                # ----------Scenario-specific energy -----------------

                print('Generating scenario-specific energy figures...')
                
                # Energy evolution all zones
                filename = f'{folder_scenario}/energy/EnergyEvolutionPerZone-{scenario}.png'
                make_stacked_bar_subplots(df_energyfuel, filename, dict_specs['colors'], selected_zone=None, selected_year=None,
                                            column_xaxis='zone',
                                            column_stacked='fuel', column_multiple_bars='year',
                                            column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None,
                                            dict_scenarios=None,
                                            format_y=lambda y, _: '{:.0f} GWh'.format(y), order_stacked=None, cap=2,
                                            annotate=False,
                                            show_total=False, fonttick=12, rotation=45, title=None)


                # Energy mix in percentage by fuel by zone
                df_percentage = df_energyfuelfull[df_energyfuelfull['scenario'] == scenario]
                df_percentage = df_percentage.drop(columns=['scenario'])
                df_percentage = df_percentage.set_index(['zone', 'year', 'fuel']).squeeze()
                df_percentage = df_percentage / df_percentage.groupby(['zone', 'year']).sum()
                df_percentage = df_percentage.reset_index()
                
                filename = f'{folder_scenario}/energy/EnergyMixClusteredStackedBarPlot.pdf'
                make_stacked_bar_subplots(df_percentage, filename, dict_specs['colors'], column_stacked='fuel',
                            column_xaxis='zone',
                            column_multiple_bars='year',
                            column_value='value',
                            format_y=lambda y, _: '{:.0f} %'.format(y * 100), rotation=45,
                            annotate=False,
                            title=f'Energy Mix by Fuel {scenario}') #TODO: show_total=['Imports', 'Exports']
                
                # Per year
                for year in [min(df_percentage['year']), max(df_percentage['year'])]:
                    df_year = df_percentage[df_percentage['year'] == year]
                    
                    filename = f'{folder_scenario}/energy/EnergyMixStackedBarPlot-{year}.png'
                    make_stacked_bar_subplots(df_year, filename, dict_specs['colors'], column_stacked='fuel',
                                            column_xaxis=None,
                                            column_multiple_bars='zone',
                                            column_value='value',
                                            format_y=lambda y, _: '{:.0f} %'.format(y * 100), rotation=45,
                                            annotate=False,
                                            title=f'Energy Mix by Fuel {scenario} {year}')
                    
            
                # Energy generation by fuel by zone
                if len(df_capacityfuel.zone.unique()) > 1:
                    for zone in df_energyplant.zone.unique():
                            filename = f'{folder_scenario}/energy/EnergyPlantsStackedAreaPlot-{scenario}-{zone}.png'
                            df_zone = df_energyplant.copy()
                            df_zone = df_zone[(df_zone['scenario'] == scenario) & (df_zone['zone'] == zone)]
                            if len(df_zone['generator'].unique()) < 20:
                                stacked_area_plot(df_zone, filename, dict_specs['colors'], x_column='year',
                                                y_column='value',
                                                stack_column='generator', title='Energy Generation by Plant',
                                                y_label='Generation (GWh)',
                                                legend_title='Energy sources', figsize=(10, 6),
                                                selected_scenario=scenario,
                                                sorting_column='fuel')
                
                if len(df_energyplant.zone.unique()) == 1 and len(epm_results['pEnergyByPlant']['generator'].unique()) < 20:
                    print('Generating energy figures for single zone by generators...')
                    temp = df_energyplant[df_energyplant['scenario'] == scenario]
                    filename = f'{folder_scenario}/energy/EnergyPlantsStackedAreaPlot-{scenario}.png'
                    stacked_area_plot(temp, filename, dict_specs['colors'], x_column='year',
                                        y_column='value',
                                        stack_column='generator', title='Energy Generation by Plant',
                                        y_label='Generation (GWh)',
                                        legend_title='Energy sources', figsize=(10, 6), selected_scenario=scenario,
                                        sorting_column='fuel')
                    
                # ---------------Scenario-specific interconnection----------------
                print('Generating interconnection figures...')
                    
                net_exchange = df_exchange[df_exchange['scenario'] == scenario]
                net_exchange = net_exchange.drop(columns=['scenario'])
                net_exchange = net_exchange.loc[net_exchange['fuel'].isin(['Exports', 'Imports']), :]
                net_exchange = net_exchange.set_index(['zone', 'year', 'fuel']).squeeze().unstack('fuel')
                net_exchange.columns.name = None
                # If there are no exports or imports, we set them to 0 to avoid errors
                net_exchange['Imports'] = net_exchange.get('Imports', 0)
                net_exchange['Exports'] = net_exchange.get('Exports', 0)
                net_exchange['value'] = net_exchange['Imports'] + net_exchange['Exports']
                net_exchange = net_exchange.reset_index()
                net_exchange = net_exchange.drop(columns=['Imports', 'Exports'])
                net_exchange['fuel'] = 'Net Exchange'
                filename = f'{GRAPHS_FOLDER}/{scenario}/NetExchangeHeatmap.pdf'
                simple_heatmap_plot(net_exchange, filename, fmt=",.0f", title=f"Net Exchange Heatmap {scenario} (GWh)")

                net_exchange = df_percentage.loc[df_percentage['fuel'].isin(['Exports', 'Imports']), :]
                net_exchange = net_exchange.set_index(['zone', 'year', 'fuel']).squeeze().unstack('fuel')
                net_exchange.columns.name = None
                net_exchange['Exports'] = net_exchange.get('Exports', 0)
                net_exchange['value'] = net_exchange['Imports'] + net_exchange['Exports']
                net_exchange = net_exchange.reset_index()
                net_exchange = net_exchange.drop(columns=['Imports', 'Exports'])
                net_exchange['fuel'] = 'Net Exchange'
                
                filename = f'{GRAPHS_FOLDER}/{scenario}/NetExchangeHeatmapPercent.pdf'
                simple_heatmap_plot(net_exchange, filename, fmt=".0%", title=f"Net Exchange Heatmap {scenario} (%)")

                # TODO: Finish the heatmap for interconnection utilization
                if False:
                    df = epm_results['pInterconUtilization'].copy()
                    df = df[df['scenario'] == scenario]
                    years = df['year'].unique()
                    years = [y for y in [2025, 2030, 2035, 2040] if y in years]
                    for year in years:
                        df_year = df[df['year'] == year].drop(columns=['year'])
                        filename = f'{GRAPHS_FOLDER}/{scenario}/InterconnectionUtilizationHeatmap_{year}.pdf'
                        simple_heatmap_plot(df_year, filename, fmt=".0%", title=f'Inteconnection Utilization {year} (%)', xcolumn='zone', ycolumn='z2', valuecolumn='value')

            
            if False:
                # ---------------------Scenario-specific interactive maps----------------
                if 'pAnnualTransmissionCapacity' in epm_results.keys():
                    print('Generating interactive map figures...')
                    if len(epm_results['pAnnualTransmissionCapacity'].zone.unique()) > 0:  # we have multiple zones
                        selected_scenarios = [s for s in selected_scenarios if 'baseline' in s]
                        make_automatic_map(epm_results, dict_specs, GRAPHS_FOLDER,
                                        selected_scenarios=selected_scenarios)


                # ---------------------Scenario-specific dispatch plots----------------
                if plot_dispatch:
                    print('Generating dispatch figures...')
                    # Perform automatic Energy DispatchFigures
                    make_automatic_dispatch(epm_results, dict_specs, GRAPHS_FOLDER,
                                            selected_scenarios=selected_scenarios)


            # ---------------Scenarios comparison----------------
            def calculate_diff(df, scenario_ref=scenario_reference):
                df_diff = df.pivot_table(index=['attribute'], columns='scenario', values='value', fill_value=0)
                df_diff = (df_diff.T - df_diff[scenario_reference]).T
                df_diff = df_diff.drop(scenario_ref, axis=1)
                df_diff = df_diff.stack().reset_index()
                df_diff.rename(columns={0: 'value'}, inplace=True)
                return df_diff
            
            print('Generating scenario comparison figures...')
            if len(selected_scenarios) < 10:
                df = df_capacityfuel.copy()
                df = df.loc[df.scenario.isin(selected_scenarios)]
                # MW to GW
                df['value'] = df['value'] / 1e3
                
                if False:
                    filename = f'{folder_comparison}/CapacityMixScenarioStackedAreaPlot.png'
                    make_stacked_bar_subplots(df, filename, dict_specs['colors'], column_stacked='fuel',
                                            column_xaxis='scenario',
                                            column_value='value', column_multiple_bars='year',
                                            format_y=lambda y, _: '{:.0f} GW'.format(y), rotation=45, format_label="{:.0f}")

                selected_years = df['year'][(df['year'] % 5 == 0) | (df['year'] == df['year'].min())].tolist()
                
                # TODO: Add year_ini=df['year'].min()
                filename = f'{folder_comparison}/CapacityMixClusteredStackedAreaPlot.pdf'
                make_stacked_bar_subplots(df, filename, dict_specs['colors'], column_stacked='fuel',
                                          column_xaxis='year',
                                          column_value='value', column_multiple_bars='scenario',
                                          select_xaxis=selected_years,
                                          format_y=lambda y, _: '{:.0f} GW'.format(y), rotation=45, format_label="{:.0f}")

                # ---------------Energy comparison----------------
                print('Generating energy figures...')

                df = df_energyfuel.copy()
                df = df.loc[df.scenario.isin(selected_scenarios)]
                # MWh to TWh
                df['value'] = df['value'] / 1e3
                
                selected_years = df['year'][(df['year'] % 5 == 0) | (df['year'] == df['year'].min())].tolist()
                filename = f'{folder_comparison}/EnergyMixClusteredStackedAreaPlot.pdf'
                make_stacked_bar_subplots(df, filename, dict_specs['colors'], column_stacked='fuel', column_xaxis='year',
                                          column_value='value', column_multiple_bars='scenario',
                                          select_xaxis=selected_years,
                                          format_y=lambda y, _: '{:.0f} TWh'.format(y), rotation=45)
                
                # ---------------System cost comparison----------------
                print('Generating system cost figures...')

                # Comparison of system cost breakdown by attribute for all scenarios
                filename = f'{folder_comparison}/CostStackedBarPlot.pdf'
                make_stacked_bar_subplots(df_systemcost, filename, dict_specs['colors'], column_stacked='attribute',
                                        column_xaxis=None,
                                        column_multiple_bars='scenario',
                                        column_value='value',
                                        format_y=lambda y, _: '${:,.0f}m'.format(y), rotation=45,
                                        annotate=False,
                                        title=f'Net Present System Cost ($m)', show_total=True)

                # System cost comparison between scenarios compare to the reference scenario
                df_systemcost_diff = calculate_diff(df_systemcost, scenario_reference)
                
                filename = f'{folder_comparison}/CostStackedBarPlotRelative.pdf'
                make_stacked_bar_subplots(df_systemcost_diff, filename, dict_specs['colors'], column_stacked='attribute',
                                        column_xaxis=None,
                                        column_multiple_bars='scenario',
                                        column_value='value',
                                        format_y=lambda y, _: '${:,.0f}m'.format(y), rotation=45,
                                        annotate=True,
                                        title=f'Net Present System Cost Comparison to the Baseline ($m)', show_total=True)
                
                # ---------------Zone cost comparison----------------
                print('Generating zone cost figures...')
                years = epm_results['pCostSummary']['year'].unique()
                final_year = max(years)
                # TODO: Cost comparison without trade
                df = epm_results['pCostSummary'].copy()
                df = df.loc[df.scenario.isin(selected_scenarios)]
                costs_comparison = ["Annualized capex: $m", "Fixed O&M: $m", "Variable O&M: $m", "Transmission additions: $m",
                                    "Spinning Reserve costs: $m", "Unmet demand costs: $m", "Excess generation: $m",
                                    "VRE curtailment: $m", "Import costs with external zones: $m", "Export revenues with external zones: $m"]
                
                df = df.loc[df.attribute.isin(costs_comparison)]
                df = df.loc[(df.year == final_year)]

                if scenario_reference in df['scenario'].unique() and len(df['scenario'].unique()) > 1:
                    df_diff = df.pivot_table(index=['zone', 'year', 'attribute'], columns='scenario', values='value', fill_value=0)
                    df_diff = (df_diff.T - df_diff[scenario_reference]).T
                    df_diff = df_diff.drop(scenario_reference, axis=1)
                    df_diff = df_diff.stack().reset_index()
                    df_diff.rename(columns={0: 'value'}, inplace=True)

                    filename = f'{folder_comparison}/CostsComparisonCountry_{final_year}.png'
                    make_stacked_bar_subplots(df_diff, filename, dict_colors=dict_specs['colors'], column_stacked='attribute',
                                            column_xaxis='zone', column_value='value', column_multiple_bars='scenario',
                                            format_y=lambda y, _: '{:.0f} $m'.format(y), annotate=False, rotation=90,
                                            show_total=True)


                # Cost comparison with trade
                df = epm_results['pCostSummary'].copy()
                df = df.loc[df.scenario.isin(selected_scenarios)]
                costs_comparison = ["Annualized capex: $m", "Fixed O&M: $m", "Variable O&M: $m", "Transmission additions: $m",
                                    "Spinning Reserve costs: $m", "Unmet demand costs: $m", "Excess generation: $m",
                                    "VRE curtailment: $m", "Trade Costs: $m", "Import costs wiht external zones: $m",
                                    "Export revenues with external zones: $m"]
                df = df.loc[df.attribute.isin(costs_comparison)]
                df = df.loc[(df.year == final_year)]

                if scenario_reference in df['scenario'].unique() and len(df['scenario'].unique()) > 1:
                    df_diff = df.pivot_table(index=['zone', 'year', 'attribute'], columns='scenario', values='value', fill_value=0)
                    df_diff = (df_diff.T - df_diff[scenario_reference]).T
                    df_diff = df_diff.drop(scenario_reference, axis=1)
                    df_diff = df_diff.stack().reset_index()
                    df_diff.rename(columns={0: 'value'}, inplace=True)

                    filename = f'{folder_comparison}/CostsComparisonWithTradeCountry.png'
                    make_stacked_bar_subplots(df_diff, filename, dict_colors=dict_specs['colors'], column_stacked='attribute',
                                            column_xaxis='zone', column_value='value', column_multiple_bars='scenario',
                                            format_y=lambda y, _: '{:.0f} $m'.format(y), annotate=False, rotation=90,
                                            show_total=True)

                # Capacity comparison
                df = df_capacityfuel.copy()
                df = df.loc[df.scenario.isin(selected_scenarios)]
                df = df.loc[(df.year == final_year)]

                if scenario_reference in df['scenario'].unique() and len(df['scenario'].unique()) > 1:
                    df_diff = df.pivot_table(index=['zone', 'year', 'fuel'], columns='scenario', values='value', fill_value=0)
                    df_diff = (df_diff.T - df_diff[scenario_reference]).T
                    df_diff = df_diff.drop(scenario_reference, axis=1)
                    df_diff = df_diff.stack().reset_index()
                    df_diff.rename(columns={0: 'value'}, inplace=True)

                    filename = f'{folder_comparison}/CapacityComparisonZone.png'
                    make_stacked_bar_subplots(df_diff, filename, dict_colors=dict_specs['colors'], column_stacked='fuel',
                                            column_xaxis='zone', column_value='value', column_multiple_bars='scenario',
                                            format_y=lambda y, _: '{:.0f} MW'.format(y), annotate=False, rotation=45)

            
            #----------------------- Project Economic Assessment -----------------------
            # Difference between scenarios with and without a project
            print('Generating project economic assessment figures...')
            
            # Cost assessment figures
            base_scenarios = df[~df['scenario'].str.contains('_wo_')].copy()
            wo_scenarios = df[df['scenario'].str.contains('_wo_')].copy()
            wo_scenarios['base_scenario'] = wo_scenarios['scenario'].str.extract(r'^(.*)_wo_')[0]
            wo_scenarios['removed'] = wo_scenarios['scenario'].str.extract(r'_wo_(.*)')[0]
            valid_bases = wo_scenarios['base_scenario'].unique()
            base_scenarios = base_scenarios[base_scenarios['scenario'].isin(valid_bases)]
            
            if not base_scenarios.empty:
                df_diff = pd.merge(
                    base_scenarios,
                    wo_scenarios,
                    left_on=['scenario', 'attribute'],
                    right_on=['base_scenario', 'attribute'],
                    suffixes=('', '_wo')
                )
                df_diff['diff'] = df_diff['value'] - df_diff['value_wo']
                df_diff = df_diff[['scenario', 'removed', 'attribute', 'diff']]
                df_diff = df_diff.rename(columns={'scenario': 'base_scenario'})
                filename = f'{folder_comparison}/AssessmentCostStackedBarPlotRelative.pdf'
                make_stacked_bar_subplots(df_diff, filename, dict_specs['colors'], column_stacked='attribute',
                                        column_xaxis=None,
                                        column_multiple_bars='base_scenario',
                                        column_value='diff',
                                        format_y=lambda y, _: '{:,.0f}'.format(y), rotation=45,
                                        annotate=False,
                                        title=f'Additional Cost of the Project', show_total=True)
                print(f'System cost assessment figures generated successfully: {filename}')

            # Energy assessment figures
            df = df_energyfuel.copy()
            year = max(df['year'].unique())
            df = df.loc[df['year'] == year]
            df = df.drop(columns=['year'])
            df = df.set_index(['scenario', 'zone', 'fuel']).squeeze().reset_index()
            df = df.groupby(['scenario', 'fuel'], as_index=False)['value'].sum()
            base_scenarios = df[~df['scenario'].str.contains('_wo_')].copy()
            wo_scenarios = df[df['scenario'].str.contains('_wo_')].copy()
            wo_scenarios['base_scenario'] = wo_scenarios['scenario'].str.extract(r'^(.*)_wo_')[0]
            wo_scenarios['removed'] = wo_scenarios['scenario'].str.extract(r'_wo_(.*)')[0]
            valid_bases = wo_scenarios['base_scenario'].unique()
            base_scenarios = base_scenarios[base_scenarios['scenario'].isin(valid_bases)]
            if not base_scenarios.empty:
                df_diff = pd.merge(
                    base_scenarios,
                    wo_scenarios,
                    left_on=['scenario', 'fuel'],
                    right_on=['base_scenario', 'fuel'],
                    suffixes=('', '_wo')
                )
                df_diff['diff'] = df_diff['value'] - df_diff['value_wo']
                df_diff = df_diff[['scenario', 'removed', 'fuel', 'diff']]
                df_diff = df_diff.rename(columns={'scenario': 'base_scenario'})
                
                filename = f'{folder_comparison}/AssessmentEnergyStackedBarPlotRelative_{year}.pdf'
                make_stacked_bar_subplots(df_diff, filename, dict_specs['colors'], column_stacked='fuel',
                                        column_xaxis=None,
                                        column_multiple_bars='base_scenario',
                                        column_value='diff',
                                        format_y=lambda y, _: '{:,.0f}'.format(y), rotation=45,
                                        annotate=False,
                                        title=f'Additional Energy with the Project {year}', show_total=True)
                print(f'Energy assessment figures generated successfully: {filename}')
            
            # Capacity assessment figures
            df = df_capacityfuel.copy()
            year = max(df['year'].unique())
            df = df.loc[df['year'] == year]
            df = df.drop(columns=['year'])
            df = df.set_index(['scenario', 'zone', 'fuel']).squeeze().reset_index()
            df = df.groupby(['scenario', 'fuel'], as_index=False)['value'].sum()
            base_scenarios = df[~df['scenario'].str.contains('_wo_')].copy()
            wo_scenarios = df[df['scenario'].str.contains('_wo_')].copy()
            wo_scenarios['base_scenario'] = wo_scenarios['scenario'].str.extract(r'^(.*)_wo_')[0]
            wo_scenarios['removed'] = wo_scenarios['scenario'].str.extract(r'_wo_(.*)')[0]
            valid_bases = wo_scenarios['base_scenario'].unique()
            base_scenarios = base_scenarios[base_scenarios['scenario'].isin(valid_bases)]
            if not base_scenarios.empty:
                df_diff = pd.merge(
                    base_scenarios,
                    wo_scenarios,
                    left_on=['scenario', 'fuel'],
                    right_on=['base_scenario', 'fuel'],
                    suffixes=('', '_wo')
                )
                df_diff['diff'] = df_diff['value'] - df_diff['value_wo']
                df_diff = df_diff[['scenario', 'removed', 'fuel', 'diff']]
                df_diff = df_diff.rename(columns={'scenario': 'base_scenario'})
                
                filename = f'{folder_comparison}/AssessmentCapacityStackedBarPlotRelative_{year}.pdf'
                make_stacked_bar_subplots(df_diff, filename, dict_specs['colors'], column_stacked='fuel',
                                        column_xaxis=None,
                                        column_multiple_bars='base_scenario',
                                        column_value='diff',
                                        format_y=lambda y, _: '{:,.0f}'.format(y), rotation=45,
                                        annotate=False,
                                        title=f'Additional Capacity with the Project {year}', show_total=True)
                print(f'Capacity assessment figures generated successfully: {filename}')

    else:
        return 0


def generate_summary_excel(results_folder, template_file="epm_results_summary_dis_template.xlsx"):

    # Get the data

    results_folder, graphs_folder, dict_specs, epm_input, epm_results, mapping_gen_fuel = process_simulation_results(
    results_folder, folder='')

    tabs_to_update=['pDemandSupply','pCapacityByFuel','pEnergyByFuel','pCostSummary','pCostSummaryCountry','pEmissions','pInterchange']

    output_file = f"{results_folder}_results_summary_dis.xlsx"

    # Create the file from the template
    shutil.copyfile(template_file, output_file)

    # Charge data
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        for tab in tabs_to_update:
            if tab in epm_results.keys():
                df_temp = epm_results[tab].copy()
                col_order = [col for col in df_temp.columns if col != "scenario"] + ["scenario"]
                df_temp = df_temp[col_order]
                df_temp.to_excel(writer, sheet_name=tab, index=False)
            else:
                print(f"No data for '{tab}' — ignored")

    print(f"Excel generated : {output_file}")