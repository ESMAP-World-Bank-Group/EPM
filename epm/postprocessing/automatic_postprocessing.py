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
import pandas as pd
# Relave imports as it's a submodule
from .utils import *
from .plots import *
from .interactive_map import make_automatic_map


KEYS_RESULTS = {
    # 1. Capacity expansion
    'pCapacityPlant', 
    'pCapacityFuel', 'pCapacityFuelCountry',
    'pNewCapacityFuel', 'pNewCapacityFuelCountry',
    'pAnnualTransmissionCapacity', 'pAdditionalCapacity',
    # 2. Cost
    'pPrice',
    'pCapexInvestment',
    'pCostsPlant',  
    'pYearlyCostsZone', 'pYearlyCostsCountry',
    'pCostsZone', 'pCostsSystem',
    'pFuelCosts', 'pFuelConsumption',
    # 3. Energy balance
    'pEnergyPlant', 'pEnergyFuel', 'pEnergyFuelCountry',
    'pEnergyBalance',
    'pUtilizationPlant', 'pUtilizationFuel',
    # 4. Energy dispatch
    'pDispatchPlant', 'pDispatch', 'pDispatchFuel',
    # 5. Reserves
    'pReserveSpinningPlantZone', 'pReserveSpinningPlantCountry',
    'pReserveMarginCountry',
    # 6. Interconnections
    'pInterchange', 'pInterconUtilization', 'pCongestionShare',
    'pInterchangeExternalExports', 'pInterchangeExternalImports',
    # 7. Emissions
    'pEmissionsZone', 'pEmissionsIntensityZone',
    # 8. Prices
    'pYearlyPriceHub',
    # 10. Metrics
    'pPlantAnnualLCOE',
    'pZonalAverageCost',
    'pZonalAverageGenCost',
    'pCountryAverageCost',
    'pCountryAverageGenCost', 
    'pSystemAverageCost',
    # 11. Other
    'pSolverParameters'
}

FIGURES_ACTIVATED = {
    # 1. Capacity figures
    'CapacityMixSystemEvolutionScenarios': False,
    'CapacityMixSystemEvolutionScenariosRelative': False,
    'CapacityMixEvolutionZone': False,
    'CapacityMixZoneScenarios': False,
    'NewCapacityInstalledTimeline': False,
    
    # 2. Cost figures
    'CostSystemScenarios': False,
    'CostSystemScenariosRelative': False,
    'CostBreakdownEvolutionScenarios': False,
    'CostBreakdownEvolutionScenariosRelative': False,
    'CostBreakdownEvolutionZone': False,
    'CostBreakdownEvolutionZonePercentage': False,
    'CostZoneScenarios': False,
                    
    # 3. Energy figures
    'EnergyMixSystemEvolutionScenarios': False,
    'EnergyMixSystemEvolutionScenariosRelative': False,
    'EnergyMixEvolutionZone': False,
    'EnergyMixZoneScenarios': False,
    'EnergyPlants': False,
    
    # 4. Interconnection figures
    'InterconnectionHeatmap': False,
    'InterconnectionHeatmapShare': False,
    'InterconnectionUtilizationHeatmap': False,
}

TRADE_ATTRS = [
    "Import costs with internal zones: $m",
    "Import costs with external zones: $m",
    "Export revenues with internal zones: $m",
    "Export revenues with external zones: $m",
    "Trade shared benefits: $m"
]

RESERVE_ATTRS = [
    "Unmet country spinning reserve costs: $m",
    "Unmet country planning reserve costs: $m",
    "Unmet country CO2 backstop cost: $m",
    "Unmet system planning reserve costs: $m",
    "Unmet system spinning reserve costs: $m"
]


def make_automatic_dispatch(epm_results, dict_specs, folder, selected_scenarios):

    def aggregate_to_system(df):
        if df is None or df.empty:
            return df

        df_system = df.copy()
        grouping_columns = [col for col in df_system.columns if col not in {'value', 'zone'}]

        if grouping_columns:
            df_system = df_system.groupby(
                grouping_columns,
                as_index=False,
                observed=False
            )['value'].sum()
        else:
            df_system = pd.DataFrame({'value': [df_system['value'].sum()]})

        df_system.insert(0, 'zone', 'System')

        ordered_cols = ['zone'] + [col for col in grouping_columns if col in df_system.columns] + ['value']
        return df_system[ordered_cols]

    dispatch_generation = filter_dataframe(epm_results['pDispatchPlant'], {'attribute': ['Generation']})
    dispatch_components = filter_dataframe(
        epm_results['pDispatch'],
        {'attribute': ['Unmet demand', 'Exports', 'Imports', 'Storage Charge']}
    )
    demand_df = filter_dataframe(epm_results['pDispatch'], {'attribute': ['Demand']})

    dfs_to_plot_area_zone = {
        'pDispatchPlant': dispatch_generation,
        'pDispatch': dispatch_components
    }

    dfs_to_plot_line_zone = {
        'pDispatch': demand_df
    }

    dispatch_full = epm_results['pDispatch']

    for selected_scenario in selected_scenarios:
        scenario_dispatch = dispatch_full[dispatch_full['scenario'] == selected_scenario]
        zones = scenario_dispatch['zone'].unique()
        if len(zones) == 0:
            continue

        years_available = sorted(scenario_dispatch['year'].unique())
        if not years_available:
            continue

        years_to_plot = [years_available[0], years_available[-1]] if len(years_available) > 1 else [years_available[0]]

        for zone in zones:
            for year in years_to_plot:
                zone_demand_year = demand_df[
                    (demand_df['scenario'] == selected_scenario) &
                    (demand_df['zone'] == zone) &
                    (demand_df['year'] == year)
                ]
                if zone_demand_year.empty:
                    continue

                season_totals = zone_demand_year.groupby('season', observed=False)['value'].sum()
                if season_totals.empty:
                    continue

                s_max = season_totals.idxmax()
                s_min = season_totals.idxmin()
                seasons_selection = [s_max] if s_max == s_min else [s_min, s_max]

                demand_subset = zone_demand_year[zone_demand_year['season'].isin(seasons_selection)]
                if demand_subset.empty:
                    continue

                day_totals = demand_subset.groupby('day', observed=False)['value'].sum()
                if day_totals.empty:
                    continue

                peak_day = day_totals.idxmax()

                filename = os.path.join(folder, f'Dispatch_{selected_scenario}_{zone}_{year}.pdf')

                select_time = {'season': seasons_selection, 'day': [peak_day]}
                make_complete_fuel_dispatch_plot(
                    dfs_to_plot_area_zone,
                    dfs_to_plot_line_zone,
                    dict_specs['colors'],
                    zone=zone,
                    year=year,
                    scenario=selected_scenario,
                    fuel_grouping=None,
                    select_time=select_time,
                    filename=filename,
                    bottom=None,
                    legend_loc='bottom'
                )

                select_time = {'season': [s_max]}
                make_complete_fuel_dispatch_plot(
                    dfs_to_plot_area_zone,
                    dfs_to_plot_line_zone,
                    dict_specs['colors'],
                    zone=zone,
                    year=year,
                    scenario=selected_scenario,
                    fuel_grouping=None,
                    select_time=select_time,
                    filename=filename,
                    bottom=None,
                    legend_loc='bottom'
                )

    dfs_to_plot_area_system = {
        'pDispatchPlant': aggregate_to_system(dispatch_generation),
        'pDispatch': aggregate_to_system(dispatch_components)
    }

    demand_system = aggregate_to_system(demand_df)
    dfs_to_plot_line_system = {
        'pDispatch': demand_system
    }

    if demand_system is None or demand_system.empty:
        return

    for selected_scenario in selected_scenarios:
        demand_scenario = demand_system[demand_system['scenario'] == selected_scenario]
        available_years = sorted(demand_scenario['year'].unique())
        if not available_years:
            continue

        years_to_plot = [available_years[0], available_years[-1]] if len(available_years) > 1 else [available_years[0]]

        for year in years_to_plot:
            df_year = demand_scenario[demand_scenario['year'] == year]
            if df_year.empty:
                continue

            season_totals = df_year.groupby('season', observed=False)['value'].sum()
            if season_totals.empty:
                continue

            s_max = season_totals.idxmax()
            s_min = season_totals.idxmin()
            seasons_selection = [s_max] if s_max == s_min else [s_min, s_max]

            df_season = df_year[df_year['season'].isin(seasons_selection)]
            if df_season.empty:
                continue

            day_totals = df_season.groupby('day', observed=False)['value'].sum()
            if day_totals.empty:
                continue

            peak_day = day_totals.idxmax()

            filename = os.path.join(folder, f'Dispatch_{selected_scenario}_system_{year}.pdf')

            select_time = {'season': seasons_selection, 'day': [peak_day]}
            make_complete_fuel_dispatch_plot(
                dfs_to_plot_area_system,
                dfs_to_plot_line_system,
                dict_specs['colors'],
                zone='System',
                year=year,
                scenario=selected_scenario,
                fuel_grouping=None,
                select_time=select_time,
                filename=filename,
                bottom=None,
                legend_loc='bottom'
            )

            select_time = {'season': [s_max]}
            make_complete_fuel_dispatch_plot(
                dfs_to_plot_area_system,
                dfs_to_plot_line_system,
                dict_specs['colors'],
                zone='System',
                year=year,
                scenario=selected_scenario,
                fuel_grouping=None,
                select_time=select_time,
                filename=filename,
                bottom=None,
                legend_loc='bottom'
            )


def postprocess_montecarlo(epm_results, RESULTS_FOLDER, GRAPHS_FOLDER):
    simulations_scenarios = pd.read_csv(os.path.join(RESULTS_FOLDER, 'input_scenarios.csv'), index_col=0)
    samples_mc = pd.read_csv(os.path.join(RESULTS_FOLDER, 'samples_montecarlo.csv'), index_col=0)
    samples_mc_substrings = set(samples_mc.columns)

    def is_not_subset(col):
        return not any(sample in col for sample in samples_mc_substrings)
    original_scenarios = [c for c in simulations_scenarios.columns if is_not_subset(c)]

    df_summary = epm_results['pCostsSystem'].copy()
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

    df_cost_summary = epm_results['pYearlyCostsZone'].copy()
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
    df_cost_summary_no_trade = epm_results['pYearlyCostsZoneFull'].copy()
    df_cost_summary_no_trade = df_cost_summary_no_trade.loc[df_cost_summary_no_trade.attribute.isin(costs_notrade)]
    df_cost_summary_baseline_notrade = df_cost_summary_no_trade.loc[(df_cost_summary_no_trade.scenario.isin(original_scenarios))]
    df_cost_summary_no_trade['scenario_mapping'] = df_cost_summary_no_trade.apply(lambda row: next(c for c in original_scenarios if c in row['scenario']), axis=1)
    df_cost_summary_no_trade = df_cost_summary_no_trade.groupby(['scenario', 'scenario_mapping', 'zone', 'year']).value.sum().reset_index().groupby(['scenario_mapping', 'zone', 'year']).value.describe()[['min', 'max']].reset_index().rename(columns={'scenario_mapping': 'scenario'})
    df_cost_summary_no_trade = df_cost_summary_no_trade.set_index(['scenario', 'zone', 'year']).stack().to_frame().rename(columns={0: 'value'})
    df_cost_summary_no_trade.index.names = ['scenario', 'zone', 'year', 'error']
    df_cost_summary_no_trade.reset_index(inplace=True)

    demand_supply = ["Unmet demand: GWh", "Imports exchange: GWh", "Exports exchange: GWh"
                        ]

    df_demandsupply = epm_results['pEnergyBalance'].copy()
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

    years = sorted(epm_results['pYearlyCostsZone']['year'].unique())

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

    zones = epm_results['pYearlyCostsZone']['zone'].unique()
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

    keys_results = KEYS_RESULTS
    if montecarlo:
        keys_results = {'pCostsSystem', 'pYearlyCostsZone', 'pYearlyCostsZoneFull', 'pEnergyBalance'}

    # Process results
    RESULTS_FOLDER, GRAPHS_FOLDER, dict_specs, epm_input, epm_results, mapping_gen_fuel = process_simulation_results(
        FOLDER, SCENARIOS_RENAME=None, folder=folder, graphs_folder=graphs_folder, keys_results=keys_results)

    # Specific postprocessing for Monte Carlo simulations
    if montecarlo:
        postprocess_montecarlo(epm_results, RESULTS_FOLDER, GRAPHS_FOLDER)

    # If not Monte Carlo, we proceed with the regular postprocessing
    if not montecarlo:

        if isinstance(selected_scenario, str):
            if selected_scenario == 'all':
                selected_scenarios = list(epm_results['pEnergyFuel'].scenario.unique())  # we choose all scenarios
            else:
                selected_scenarios = [selected_scenario]
                assert selected_scenario in list(epm_results['pEnergyFuel'].scenario.unique()), "Selected scenario does not belong to the set of scenarios."
        else:
            selected_scenarios = selected_scenario

        # Generate summary
        print('Generating summary...')
        generate_summary(epm_results, RESULTS_FOLDER, epm_input)

        # Generate detailed by plant to debug
        if not reduced_output:

            # Generate a detailed summary by Power Plant
            print('Generating detailed summary by Power Plant...')
            generate_plants_summary(epm_results, RESULTS_FOLDER)

            
            # ------------------------------------------------------------------------------------
            # Prepare dataframes for postprocessing
            print('Preparing dataframes for postprocessing...')
                
            # Define dataframes for capacity, energy, exchange
            df_energyplant = epm_results['pEnergyPlant'].copy()

            # Group reserve cost attributes into one unique attribute and sum the value
            def simplify_attributes(df, new_label, attributes_list=[]):
            

                df_reserve = df[df['attribute'].isin(attributes_list)].copy()
                if not df_reserve.empty:
                    df_reserve = df_reserve.groupby(
                        [col for col in df_reserve.columns if col not in ['attribute', 'value']],
                        as_index=False
                    )['value'].sum()
                    df_reserve['attribute'] = new_label
                df = pd.concat([
                    df[~df['attribute'].isin(attributes_list)],
                    df_reserve
                ], ignore_index=True)
                return df



                rows = []
                for scenario in df["scenario"].unique():
                    subset = df[df["scenario"] == scenario]

                    # Just sum directly since signs are already correct
                    trade_val = subset[subset["attribute"].isin(TRADE_ATTRS)]["value"].sum()

                    rows.append({
                        "scenario": scenario,
                        "attribute": "Trade costs: $m",
                        "value": trade_val
                    })

                # Create new rows
                trade_df = pd.DataFrame(rows)

                # Drop old attributes and add the simplified one
                df_final = pd.concat(
                    [df[~df["attribute"].isin(TRADE_ATTRS)].copy(), trade_df],
                    ignore_index=True
                )

                return df_final
                
                
            # ------------------------------------------------------------------------------------
            print('Creating folders for figures...')
            
            # Create subfolders directly under GRAPHS_FOLDER
            subfolders = {}
            for subfolder in ['1_capacity', '2_cost', '3_energy', '4_interonnection', '5_dispatch', '6_maps']:
                subfolders[subfolder] = Path(GRAPHS_FOLDER) / Path(subfolder)
                if not os.path.exists(subfolders[subfolder]):
                    os.mkdir(subfolders[subfolder])
            
            # Select main years for x-axis in some plots to simplify the reading
            df = epm_results['pCapacityFuel'].copy()
            selected_years = df['year'][(df['year'] % 5 == 0) | (df['year'] == df['year'].min())].tolist()

            scenarios_threshold = 10
            
            def calculate_diff(df, scenario_ref=scenario_reference, attribute='attribute'):
                """
                Calculate the difference in 'value' between each scenario and the reference scenario.
                Parameters:
                - df (pd.DataFrame): DataFrame containing 'scenario', 'attribute', and '
                - scenario_ref (str): The reference scenario to compare against.
                Returns:
                - pd.DataFrame: DataFrame with differences in 'value' for each scenario compared to
                the reference scenario.
                """
                df_diff = df.pivot_table(index=[attribute], columns='scenario', values='value', fill_value=0)
                df_diff = (df_diff.T - df_diff[scenario_reference]).T
                df_diff = df_diff.drop(scenario_ref, axis=1)
                df_diff = df_diff.stack().reset_index()
                df_diff.rename(columns={0: 'value'}, inplace=True)
                return df_diff


            # ------------------------------------------------------------------------------------
            # 1. Capacity figures
            # ------------------------------------------------------------------------------------

            print('Generating capacity figures...')
            
            # 1.1 Evolution of capacity mix for the system (all zones aggregated)
            if len(selected_scenarios) < scenarios_threshold:
                df = epm_results['pCapacityFuel'].copy()
                df = df.loc[df.scenario.isin(selected_scenarios)]
                # MW to GW
                df['value'] = df['value'] / 1e3
                
                # TODO: Add year_ini=df['year'].min()
                figure_name = 'CapacityMixSystemEvolutionScenarios'
                if FIGURES_ACTIVATED.get(figure_name, False):
                    filename = os.path.join(subfolders['1_capacity'], f'{figure_name}.pdf')
                    
                    make_stacked_bar_subplots(df, filename, dict_specs['colors'], 
                                            column_stacked='fuel',
                                            column_xaxis='year',
                                            column_value='value', 
                                            column_multiple_bars='scenario',
                                            select_xaxis=selected_years,
                                            format_y=make_auto_formatter("GW"), 
                                            rotation=45, 
                                            annotate=False,
                                            format_label="{:.0f}",
                                            title = 'Installed Capacity Mix by Fuel - System (GW)')
                    
                    
                if len(selected_scenarios) > 1 and scenario_reference in selected_scenarios:
                    # Capacity comparison between scenarios compare to the reference scenario
                    df_diff = calculate_diff(df, scenario_reference, attribute='fuel')
                    
                    figure_name = 'CapacityMixSystemEvolutionScenariosRelative'
                    if FIGURES_ACTIVATED.get(figure_name, False):
                        filename = os.path.join(subfolders['1_capacity'], f'{figure_name}.pdf')
                        
                        make_stacked_bar_subplots(df_diff, filename, dict_specs['colors'], column_stacked='attribute',
                                                column_xaxis=None,
                                                column_multiple_bars='scenario',
                                                column_value='value',
                                                format_y=make_auto_formatter("GW"), 
                                                rotation=45,
                                                annotate=False,
                                                title='Incremental Capacity Mix vs Baseline (GW)', 
                                                show_total=True)
                        

            # 1.2 Evolution of capacity mix per zone
            # Capacity mix in percentage by fuel by zone
            figure_name = 'CapacityMixEvolutionZone'
            if FIGURES_ACTIVATED.get(figure_name, False):
                for scenario in selected_scenarios:
                    df = epm_results['pCapacityFuel'].copy()
                    # MW to GW
                    df['value'] = df['value'] / 1e3
                    
                    df = df[df['scenario'] == scenario]
                    df = df.drop(columns=['scenario'])
                    
                    filename = os.path.join(subfolders['1_capacity'], f'{figure_name}-{scenario}.pdf')
                    
                    make_stacked_bar_subplots(df, 
                                                filename, 
                                                dict_specs['colors'], 
                                                column_stacked='fuel',
                                                column_xaxis='zone',
                                                column_multiple_bars='year',
                                                column_value='value',
                                                format_y=make_auto_formatter("GW"), 
                                                rotation=45,
                                                annotate=False,
                                                title=f'Installed Capacity Mix by Fuel - {scenario} (GW)'
                                                )                     
                    
                    
                    df_percentage = df.set_index(['zone', 'year', 'fuel']).squeeze()
                    df_percentage = df_percentage / df_percentage.groupby(['zone', 'year']).sum()
                    df_percentage = df_percentage.reset_index()
                    
                    filename = os.path.join(subfolders['1_capacity'], f'{figure_name}Percentage-{scenario}.pdf')
                    
                    make_stacked_bar_subplots(df_percentage, 
                                                filename, 
                                                dict_specs['colors'], 
                                                column_stacked='fuel',
                                                column_xaxis='zone',
                                                column_multiple_bars='year',
                                                column_value='value',
                                                format_y=make_auto_formatter("%"), rotation=45,
                                                annotate=False,
                                                title=f'Capacity Mix Shares by Zone - {scenario} (%)'
                                                ) 
            
            # 1.3 Capacity mix per zone for the first and last years available
            figure_name = 'CapacityMixZoneScenarios'
            for year in [min(epm_results['pCapacityFuel']['year']), max(epm_results['pCapacityFuel']['year'])]:
                df = epm_results['pCapacityFuel'].copy()
                # MW to GW
                df['value'] = df['value'] / 1e3
                df = df[df['year'] == year]                
                df = df.drop(columns=['year'])

                # TODO: percentage ?
                if FIGURES_ACTIVATED.get(figure_name, False):
                    filename = os.path.join(subfolders['1_capacity'], f'{figure_name}-{year}.pdf')        
                
                    make_stacked_bar_subplots(df, filename, dict_specs['colors'], 
                                            column_stacked='fuel',
                                                column_xaxis='zone',
                                                column_multiple_bars='scenario',
                                                column_value='value',
                                                format_y=make_auto_formatter("GW"), 
                                                rotation=45,
                                                annotate=False,
                                                title=f'Installed Capacity Mix by Fuel - {year} (GW)')
                
            # 1.4 New capacity installed per zone
            for scenario in selected_scenarios:
                for zone in epm_results['pCapacityPlant'].zone.unique():
                    df_zone = epm_results['pCapacityPlant'].copy()
                    df_zone = df_zone[(df_zone['scenario'] == scenario) & (df_zone['zone'] == zone)]
                    
                    figure_name = 'NewCapacityInstalledTimeline'
                    if FIGURES_ACTIVATED.get(figure_name, False):
                        filename = os.path.join(subfolders['1_capacity'], f'{figure_name}-{scenario}-{zone}.pdf')

                        
                        # Tranform in stacked bat plot
                        make_annotated_stacked_area_plot(df_zone, filename, dict_colors=dict_specs['colors'])
                
            # 1.5 Capex investment figures
            
            # ------------------------------------------------------------------------------------
            # 2. Cost figures
            # ------------------------------------------------------------------------------------

            print('Generating cost figures...')
             
            # 2.0 Total system cost
            if len(selected_scenarios) < scenarios_threshold:
                df = epm_results['pCostsSystem'].copy()
                # Remove rows with attribute == 'NPV of system cost: $m' to avoid double counting
                df = df.loc[df.attribute != 'NPV of system cost: $m']
                # Group reserve cost attributes into one unique attribute and sum the value
                df = simplify_attributes(df, "Unmet reserve costs: $m", RESERVE_ATTRS)
                # Group trade components into one unique attribute and sum the value
                df = simplify_attributes(df, "Trade costs: $m", TRADE_ATTRS)
                # Remove ": $m" from attribute names
                df['attribute'] = df['attribute'].str.replace(': $m', '', regex=False)
                # Keep only selected scenarios
                df = df.loc[df.scenario.isin(selected_scenarios)]
                
                
                figure_name = 'CostSystemScenarios'
                if FIGURES_ACTIVATED.get(figure_name, False):
                    filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')
                
                    make_stacked_bar_subplots(df, filename, dict_specs['colors'], column_stacked='attribute',
                                            column_xaxis=None,
                                            column_multiple_bars='scenario',
                                            column_value='value',
                                            format_y=make_auto_formatter("m$"), 
                                            rotation=45,
                                            annotate=False,
                                            title=f'Net Present System Cost by Scenario (million USD)', show_total=True)
                
                # System cost comparison between scenarios compare to the reference scenarios
                if scenario_reference in selected_scenarios and len(selected_scenarios) > 1:
                    figure_name = 'CostSystemScenariosRelative'
                    if FIGURES_ACTIVATED.get(figure_name, False):
                        filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')

                        df_diff = calculate_diff(df, scenario_reference)
                        
                        make_stacked_bar_subplots(df_diff, filename, dict_specs['colors'], column_stacked='attribute',
                                                column_xaxis=None,
                                                column_multiple_bars='scenario',
                                                column_value='value',
                                                format_y=make_auto_formatter("m$"), 
                                                rotation=45,
                                                annotate=True,
                                                title='Additional System Cost vs. Baseline (NPV, million USD)', 
                                                show_total=True)
            
            df_costzone = epm_results['pYearlyCostsZone'].copy()
            # Group reserve cost attributes into one unique attribute and sum the value
            df_costzone = simplify_attributes(df_costzone, "Unmet reserve costs: $m", RESERVE_ATTRS)
            # Group trade components into one unique attribute and sum the value
            df_costzone = simplify_attributes(df_costzone, "Trade costs: $m", TRADE_ATTRS)
            # Remove ": $m" from attribute names
            df_costzone['attribute'] = df_costzone['attribute'].str.replace(': $m', '', regex=False)
            
            # 2.1 Evolution of cost for the system (all zones aggregated)
            if len(selected_scenarios) < scenarios_threshold:
                df = df_costzone.copy()
                df = df.loc[df.scenario.isin(selected_scenarios)]
                
                # TODO: Add year_ini=df['year'].min()
                figure_name = 'CostBreakdownEvolutionScenarios'
                if FIGURES_ACTIVATED.get(figure_name, False):
                    filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')
                    
                    make_stacked_bar_subplots(df, filename, dict_specs['colors'], 
                                            column_stacked='attribute',
                                            column_xaxis='year',
                                            column_value='value', 
                                            column_multiple_bars='scenario',
                                            select_xaxis=selected_years,
                                            format_y=make_auto_formatter("m$"), 
                                            rotation=45, 
                                            annotate=True,
                                            format_label="{:.0f}",
                                            title='Annual Cost Breakdown by Scenario (million USD)')
                    
                    
                if len(selected_scenarios) > 1 and scenario_reference in selected_scenarios:
                    # Capacity comparison between scenarios compare to the reference scenario
                    df_diff = calculate_diff(df, scenario_reference, attribute='fuel')
                    
                    figure_name = 'CostBreakdownEvolutionScenariosRelative'
                    if FIGURES_ACTIVATED.get(figure_name, False):
                        filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')
                        
                        make_stacked_bar_subplots(df_diff, filename, dict_specs['colors'], column_stacked='attribute',
                                                column_xaxis=None,
                                                column_multiple_bars='scenario',
                                                column_value='value',
                                                format_y=make_auto_formatter("m$"), 
                                                rotation=45,
                                                annotate=True,
                                                title='Incremental Annual Cost vs. Baseline (million USD)', 
                                                show_total=True)
                        
            # 2.2 Evolution of capacity mix per zone
            # Capacity mix in percentage by fuel by zone
            figure_name = 'CostBreakdownEvolutionZone'
            if FIGURES_ACTIVATED.get(figure_name, False):
                for scenario in selected_scenarios:
                    df = df_costzone.copy()
                    df = df[df['scenario'] == scenario]
                    df = df.drop(columns=['scenario'])
                    
                    filename = os.path.join(subfolders['2_cost'], f'{figure_name}-{scenario}.pdf')
                    
                    make_stacked_bar_subplots(df, 
                                                filename, 
                                                dict_specs['colors'], 
                                                column_stacked='attribute',
                                                column_xaxis='zone',
                                                column_multiple_bars='year',
                                                column_value='value',
                                                format_y=make_auto_formatter("m$"), 
                                                rotation=45,
                                                annotate=False,
                                                title=f'Annual Cost Breakdown by Zone – {scenario} (million USD)'
                                                )                     
                    
                    figure_name = f'{figure_name}Percentage-{scenario}'
                    if FIGURES_ACTIVATED.get(figure_name, False):
                        filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')

                        df_percentage = df.set_index(['zone', 'year', 'fuel']).squeeze()
                        df_percentage = df_percentage / df_percentage.groupby(['zone', 'year']).sum()
                        df_percentage = df_percentage.reset_index()
                        
                    
                        make_stacked_bar_subplots(df_percentage, 
                                                    filename, 
                                                    dict_specs['colors'], 
                                                    column_stacked='attribute',
                                                    column_xaxis='zone',
                                                    column_multiple_bars='year',
                                                    column_value='value',
                                                    format_y=make_auto_formatter("%"), 
                                                    rotation=45,
                                                    annotate=False,
                                                    title=f'Cost Structure Shares by Zone – {scenario} (%)'
                                                    ) 
                
            # 2.3 Cost components per zone
            figure_name = 'CostZoneScenarios'
            if FIGURES_ACTIVATED.get(figure_name, False):
                filename = os.path.join(subfolders['2_cost'], f'{figure_name}-{year}.pdf')
        
                df = df_costzone.copy()
                
                df = df.loc[df.scenario.isin(selected_scenarios)]
                df = df.loc[(df.year == max(df['year'].unique()))]

                make_stacked_bar_subplots(df, 
                                          filename, dict_specs['colors'], 
                                          column_stacked='attribute',
                                            column_xaxis='zone',
                                            column_multiple_bars='scenario',
                                            column_value='value',
                                            format_y=make_auto_formatter("m$"), 
                                            rotation=45,
                                            annotate=False,
                                            title=f'Cost Composition by Zone in {year} (million USD)')
            
            # ------------------------------------------------------------------------------------
            # 3. Energy figures
            # ------------------------------------------------------------------------------------

            print('Generating energy figures...')
            
            # Prepare dataframes for energy
            df_energyfuel = epm_results['pEnergyFuel'].copy()
            
            # Additionnal energy information not in pEnergyFuel
            df_exchange = epm_results['pEnergyBalance'].copy()
            df_exchange = df_exchange.loc[df_exchange['attribute'].isin(['Unmet demand: GWh', 'Exports exchange: GWh', 'Imports exchange: GWh'])]
            df_exchange = df_exchange.replace({'Unmet demand: GWh': 'Unmet demand',
                                              'Exports exchange: GWh': 'Exports',
                                              'Imports exchange: GWh': 'Imports'})
            # Put negative values when exports in colmun 'attribute'
            df_exchange['value'] = df_exchange.apply(lambda row: -row['value'] if row['attribute'] == 'Exports' else row['value'], axis=1)
            df_exchange.rename(columns={'attribute': 'fuel'}, inplace=True)
            # Define energyfuelfull to include exchange
            df_energyfuelfull = pd.concat([df_energyfuel, df_exchange], ignore_index=True)         
            
            # 3.1 Evolution of capacity mix for the system (all zones aggregated)
            if len(selected_scenarios) < scenarios_threshold:
                df = df_energyfuelfull.copy()
                df = df.loc[df.scenario.isin(selected_scenarios)]
                
                figure_name = 'EnergyMixSystemEvolutionScenarios'
                if FIGURES_ACTIVATED.get(figure_name, False):
                    filename = os.path.join(subfolders['3_energy'], f'{figure_name}.pdf')
                    
                    make_stacked_bar_subplots(df, filename, dict_specs['colors'], 
                                            column_stacked='fuel',
                                            column_xaxis='year',
                                            column_value='value', 
                                            column_multiple_bars='scenario',
                                            select_xaxis=selected_years,
                                            format_y=make_auto_formatter("GWh"), 
                                            rotation=45, 
                                            format_label="{:.0f}",
                                            title = 'Energy Generation Mix by Fuel - System (GWh)',
                                            annotate=False)
                    
                    
                if len(selected_scenarios) > 1 and scenario_reference in selected_scenarios:
                    # Capacity comparison between scenarios compare to the reference scenario
                    df_diff = calculate_diff(df, scenario_reference, attribute='fuel')
                    
                    figure_name = 'EnergyMixSystemEvolutionScenariosRelative'
                    if FIGURES_ACTIVATED.get(figure_name, False):
                        filename = os.path.join(subfolders['3_energy'], f'{figure_name}.pdf')
                        
                        make_stacked_bar_subplots(df_diff, filename, dict_specs['colors'], column_stacked='attribute',
                                                column_xaxis=None,
                                                column_multiple_bars='scenario',
                                                column_value='value',
                                                format_y=make_auto_formatter("GWh"), rotation=45,
                                                annotate=True,
                                                title='Incremental Energy Mix vs Baseline (GWh)', show_total=True)
                        
            # 3.2 Evolution of capacity mix per zone
            figure_name = 'EnergyMixEvolutionZone'
            if FIGURES_ACTIVATED.get(figure_name, False):
                for scenario in selected_scenarios:
                    df = df_energyfuelfull.copy()
                    df = df[df['scenario'] == scenario]
                    df = df.drop(columns=['scenario'])
                    
                    filename = os.path.join(subfolders['3_energy'], f'{figure_name}-{scenario}.pdf')
                    
                    make_stacked_bar_subplots(df, 
                                                filename, 
                                                dict_specs['colors'], 
                                                column_stacked='fuel',
                                                column_xaxis='zone',
                                                column_multiple_bars='year',
                                                column_value='value',
                                                format_y=make_auto_formatter("GWh"), rotation=45,
                                                annotate=False,
                                                title=f'Energy Mix by Fuel - {scenario} (GWh)'
                                                )                     
                    
                    
                    # Energy mix in percentage by fuel by zone
                    df_percentage = df.set_index(['zone', 'year', 'fuel']).squeeze()
                    df_percentage = df_percentage / df_percentage.groupby(['zone', 'year']).sum()
                    df_percentage = df_percentage.reset_index()
                    
                    # Keeping for interconnection figures
                    df_exchange_percentage = df_percentage.loc[df_percentage['fuel'].isin(['Exports', 'Imports']), :]
                    
                    filename = os.path.join(subfolders['3_energy'], f'{figure_name}Percentage-{scenario}.pdf')
                    
                    make_stacked_bar_subplots(df_percentage, 
                                                filename, 
                                                dict_specs['colors'], 
                                                column_stacked='fuel',
                                                column_xaxis='zone',
                                                column_multiple_bars='year',
                                                column_value='value',
                                                format_y=make_auto_formatter("%"), rotation=45,
                                                annotate=False,
                                                title=f'Energy Mix Shares by Zone - {scenario} (%)'
                                                ) 
            
            # 3.3 Energy mix per zone for the first and last years available
            figure_name = 'EnergyMixZoneScenarios'
            for year in [min(df_energyfuelfull['year']), max(df_energyfuelfull['year'])]:
                df = df_energyfuelfull.copy()
                df = df[df['year'] == year]                
                df = df.drop(columns=['year'])

                # TODO: percentage ?
                if FIGURES_ACTIVATED.get(figure_name, False):
                    filename = os.path.join(subfolders['3_energy'], f'{figure_name}-{year}.pdf')        
                
                    make_stacked_bar_subplots(df, filename, dict_specs['colors'], 
                                            column_stacked='fuel',
                                                column_xaxis='zone',
                                                column_multiple_bars='scenario',
                                                column_value='value',
                                                format_y=make_auto_formatter("GWh"), 
                                                rotation=45,
                                                annotate=False,
                                                title=f'Energy Mix by Fuel - {year} (GWh)')

            # 3.4 Energy generation by plant
            figure_name = 'EnergyPlants'
            if FIGURES_ACTIVATED.get(figure_name, False):
                filename = os.path.join(subfolders['3_energy'], f'{figure_name}-{scenario_reference}.pdf')
                if len(df_energyplant.zone.unique()) == 1 and len(epm_results['pEnergyPlant']['generator'].unique()) < 20:
                    print('Generating energy figures for single zone by generators...')
                    temp = df_energyplant[df_energyplant['scenario'] == scenario_reference]
                    stacked_area_plot(temp, filename, dict_specs['colors'], column_xaxis='year',
                                        column_value='value',
                                        column_stacked='generator', title='Energy Generation by Plant',
                                        y_label='Generation (GWh)',
                                        legend_title='Energy sources', figsize=(10, 6), selected_scenario=scenario,
                                        sorting_column='fuel')
            
            # ------------------------------------------------------------------------------------
            # 4. Interconnection
            # ------------------------------------------------------------------------------------
                  
            print('Generating interconnection figures...')
            
            # 4.1 Net exchange heatmap [GWh and %]
            
            figure_name = 'InterconnectionHeatmap'
            if FIGURES_ACTIVATED.get(figure_name, False):
                filename = os.path.join(subfolders['4_interonnection'], f'{figure_name}.pdf')
            
                net_exchange = df_exchange[df_exchange['scenario'] == scenario_reference]
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
                simple_heatmap_plot(net_exchange, filename, unit="GWh", title=f"Net Imports by Zone over Time - {scenario_reference} [GWh]")

            figure_name = 'InterconnectionHeatmapShare'
            if FIGURES_ACTIVATED.get(figure_name, False):
                filename = os.path.join(subfolders['4_interonnection'], f'{figure_name}.pdf')
                
                net_exchange = df_exchange_percentage.set_index(['zone', 'year', 'fuel']).squeeze().unstack('fuel')
                net_exchange.columns.name = None
                net_exchange['Exports'] = net_exchange.get('Exports', 0)
                net_exchange['value'] = net_exchange['Imports'] + net_exchange['Exports']
                net_exchange = net_exchange.reset_index()
                net_exchange = net_exchange.drop(columns=['Imports', 'Exports'])
                net_exchange['fuel'] = 'Net Exchange'
                
                simple_heatmap_plot(net_exchange, filename, unit="%", title=f"Net Imports by Zone over Time - {scenario_reference} [% of energy generation]")

            figure_name = 'InterconnectionUtilizationHeatmap'
            if FIGURES_ACTIVATED.get(figure_name, False) and 'pInterconUtilization' in epm_results:
                df_utilization = epm_results['pInterconUtilization'].copy()
                df_utilization = df_utilization[df_utilization['scenario'].isin(selected_scenarios)]

                required_cols = {'zone', 'z2', 'year', 'scenario', 'value'}
                missing_cols = required_cols.difference(df_utilization.columns)
                if missing_cols:
                    print(f"Skipping {figure_name}: missing columns {sorted(missing_cols)}")
                else:
                    if scenario_reference not in df_utilization['scenario'].unique():
                        print(f"Skipping {figure_name}: scenario '{scenario_reference}' not available")
                    else:
                        df_utilization = df_utilization[df_utilization['scenario'] == scenario_reference]

                        if df_utilization.empty:
                            print(f"Skipping {figure_name}: no utilization data for scenario '{scenario_reference}'")
                        else:
                            df_utilization = df_utilization.groupby(
                                ['zone', 'z2', 'year'],
                                as_index=False,
                                observed=False
                            )['value'].mean()
                            df_utilization['interconnection'] = (
                                df_utilization['zone'].astype(str) + ' -> ' + df_utilization['z2'].astype(str)
                            )

                            df_plot = df_utilization[['year', 'interconnection', 'value']].copy()
                            df_plot = df_plot.sort_values(['interconnection', 'year'])

                            filename = os.path.join(
                                subfolders['4_interonnection'],
                                f'{figure_name}-{scenario_reference}.pdf'
                            )

                            title = f'Interconnection Utilization - {scenario_reference} (%)'
                            simple_heatmap_plot(
                                df_plot,
                                filename,
                                unit="%",
                                title=title,
                                xcolumn='interconnection',
                                ycolumn='year',
                                valuecolumn='value'
                            )
                            
            if plot_dispatch:
                print('Generating dispatch figures...')
                # Perform automatic Energy DispatchFigures
                make_automatic_dispatch(epm_results, dict_specs, subfolders['5_dispatch'],
                                        selected_scenarios=selected_scenarios)
            
            # ---------------------Scenario-specific interactive maps----------------
            
            if 'pAnnualTransmissionCapacity' in epm_results.keys():
                
                # Check if multiple zone
                if len(epm_results['pAnnualTransmissionCapacity'].zone.unique()) > 0:  

                    print('Generating interactive map figures...')
                    make_automatic_map(epm_results, dict_specs, subfolders['6_maps'],
                                    selected_scenarios=['baseline'])



            if False:
                #----------------------- Project Economic Assessment -----------------------
                # Difference between scenarios with and without a project
                print('Generating project economic assessment figures...')
                
                # TODO: Create folder_assessment
                
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
                                            title='Project Cost Impact by Component (million USD)', show_total=True)
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
