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
from .maps import make_automatic_map


KEYS_RESULTS = {
    # 1. Capacity expansion
    'pCapacityPlant', 
    'pCapacityFuel', 'pCapacityFuelCountry',
    'pNewCapacityFuel', 'pNewCapacityFuelCountry',
    'pAnnualTransmissionCapacity', 'pAdditionalTransmissionCapacity',
    # 2. Cost
    'pPrice',
    'pCapexInvestmentComponent', 'pCapexInvestmentPlant',
    'pCostsPlant',  
    'pYearlyCostsZone', 'pYearlyCostsCountry',
    'pCostsZone', 'pCostsSystem', 'pCostsSystemPerMWh',
    'pFuelCosts', 'pFuelCostsCountry', 'pFuelConsumption', 'pFuelConsumptionCountry',
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
    'pYearlySystemAverageCost',
    # 11. Other
    'pSolverParameters'
}

FIGURES_ACTIVATED = {
    
    'SummaryHeatmap': True,
    
    # 1. Capacity figures
    'CapacityMixSystemEvolutionScenarios': True,
    'CapacityMixSystemEvolutionScenariosRelative': True,
    'CapacityMixEvolutionZone': True,
    'CapacityMixZoneScenarios': True,
    'NewCapacityInstalledTimeline': True,
    
    # 2. Cost figures
    'CostSystemScenarios': True,
    'CostSystemScenariosRelative': True,
    'CostSystemEvolutionScenarios': True,
    'CostSystemEvolutionScenariosRelative': True,
    'CostZoneEvolution': True,
    'CostZoneEvolutionPercentage': True,
    'CostZoneScenarios': True,
    'CapexZoneEvolution': True,
                    
    # 3. Energy figures
    'EnergyMixSystemEvolutionScenarios': True,
    'EnergyMixSystemEvolutionScenariosRelative': True,
    'EnergyMixZoneEvolution': True,
    'EnergyMixZoneScenarios': True,
    'EnergyPlants': True,
    'EnergyPlantZoneTop10': True,
    
    # 4. Dispatch figures
    'DispatchZoneMaxLoadDay': True,
    'DispatchZoneMaxLoadSeason': True,
    'DispatchSystemMaxLoadDay': True,
    'DispatchSystemMaxLoadSeason': True,
    
    # 5. Interconnection figures
    'NetImportsZoneEvolution': True,
    'NetImportsZoneEvolutionZoneEvolutionShare': True,
    'InterconnectionExchangeHeatmap': True,
    'InterconnectionUtilizationHeatmap': True,

    # 6. Maps
    # 'TransmissionCapacityMap': False, 
    'TransmissionCapacityMapEvolution': True,
    # 'TransmissionUtilizationMap': False,
    'TransmissionUtilizationMapEvolution': True,
    # 'NetExportsMap': True, 
    
    'InteractiveMap': True
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


def make_automatic_dispatch(epm_results, dict_specs, folder, selected_scenarios, FIGURES_ACTIVATED):
    """
    Generate dispatch plots that highlight peak demand conditions for each scenario.

    The routine builds stacked dispatch charts for (i) the day with maximum demand
    within the highest and lowest load seasons, and (ii) the season with maximum load.
    Plots are produced for every zone and for the system aggregation whenever the
    corresponding entries in ``FIGURES_ACTIVATED`` are enabled.

    Parameters
    ----------
    epm_results : dict[str, pandas.DataFrame]
        Model result tables keyed by result name (e.g., ``'pDispatch'``).
    dict_specs : dict
        Plotting specifications, including a ``'colors'`` mapping per fuel.
    folder : str
        Destination directory for the generated figures.
    selected_scenarios : list[str]
        Scenarios for which the dispatch plots should be produced.
    FIGURES_ACTIVATED : dict[str, bool]
        Switches that control whether each dispatch figure is generated.
    """

    zone_max_load_day_active = FIGURES_ACTIVATED.get('DispatchZoneMaxLoadDay', False)
    zone_max_load_season_active = FIGURES_ACTIVATED.get('DispatchZoneMaxLoadSeason', False)
    system_max_load_day_active = FIGURES_ACTIVATED.get('DispatchSystemMaxLoadDay', False)
    system_max_load_season_active = FIGURES_ACTIVATED.get('DispatchSystemMaxLoadSeason', False)

    generate_zone_figures = zone_max_load_day_active or zone_max_load_season_active
    generate_system_figures = system_max_load_day_active or system_max_load_season_active

    if not (generate_zone_figures or generate_system_figures):
        return

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

    if generate_zone_figures:
        dfs_to_plot_area_zone = {
            'pDispatchPlant': dispatch_generation,
            'pDispatch': dispatch_components
        }

        dfs_to_plot_line_zone = {
            'pDispatch': demand_df
        }

        dispatch_full = epm_results['pDispatch'].copy()

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

                    max_load_season = season_totals.idxmax()
                    min_load_season = season_totals.idxmin()
                    extreme_seasons = [max_load_season] if max_load_season == min_load_season else [min_load_season, max_load_season]

                    if zone_max_load_day_active:
                        extreme_season_demand = zone_demand_year[zone_demand_year['season'].isin(extreme_seasons)]
                        if not extreme_season_demand.empty:
                            day_totals = extreme_season_demand.groupby('day', observed=False)['value'].sum()
                            if not day_totals.empty:
                                max_load_day = day_totals.idxmax()
                                filename = os.path.join(folder, f'Dispatch_{selected_scenario}_{zone}_max_load_day.pdf')
                                select_time = {'season': extreme_seasons, 'day': [max_load_day]}
                                make_fuel_dispatchplot(
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

                    if zone_max_load_season_active:
                        filename = os.path.join(folder, f'Dispatch_{selected_scenario}_{zone}_max_load_season.pdf')
                        select_time = {'season': [max_load_season]}
                        make_fuel_dispatchplot(
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

    if not generate_system_figures:
        return

    dfs_to_plot_area_system = {
        'pDispatchPlant': aggregate_to_system(dispatch_generation),
        'pDispatch': aggregate_to_system(dispatch_components)
    }

    demand_system = aggregate_to_system(demand_df)
    if demand_system is None or demand_system.empty:
        return

    dfs_to_plot_line_system = {
        'pDispatch': demand_system
    }

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

            max_load_season = season_totals.idxmax()
            min_load_season = season_totals.idxmin()
            extreme_seasons = [max_load_season] if max_load_season == min_load_season else [min_load_season, max_load_season]

            if system_max_load_day_active:
                df_extreme_seasons = df_year[df_year['season'].isin(extreme_seasons)]
                if not df_extreme_seasons.empty:
                    day_totals = df_extreme_seasons.groupby('day', observed=False)['value'].sum()
                    if not day_totals.empty:
                        max_load_day = day_totals.idxmax()
                        filename = os.path.join(folder, f'Dispatch_{selected_scenario}_system_max_load_day.pdf')
                        select_time = {'season': extreme_seasons, 'day': [max_load_day]}
                        make_fuel_dispatchplot(
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

            if system_max_load_season_active:
                filename = os.path.join(folder, f'Dispatch_{selected_scenario}_system_max_load_season.pdf')
                select_time = {'season': [max_load_season]}
                make_fuel_dispatchplot(
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

    make_stacked_barplot(df_summary_baseline, filename, dict_colors=None, df_errorbars=df_summary, column_subplot=None, column_stacked=None, column_xaxis='scenario',
                                column_value='value', select_subplot=None, stacked_grouping=None, order_scenarios=None,
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

    costs_notrade = ["Annualized capex: $m", "Fixed O&M: $m", "Variable O&M: $m", "Total fuel Costs: $m", "Transmission costs: $m",
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
        df = df_cost_summary_baseline[(df_cost_summary_baseline['year'] == year)]
        df = df.drop(columns=['year'])
        df_errorbars = df_cost_summary[(df_cost_summary['year'] == year)]
        df_errorbars = df_errorbars.drop(columns=['year'])
        
        make_stacked_barplot(df, filename, dict_colors=None, df_errorbars=df_errorbars, column_subplot='zone', column_stacked='attribute', column_xaxis='scenario',
                                    column_value='value', select_subplot=None, stacked_grouping=None, order_scenarios=None,
                                    dict_scenarios=None,
                                    format_y=lambda y, _: '{:.0f} m$'.format(y), order_stacked=None, cap=2,
                                    annotate=False, show_total=False, fonttick=12, rotation=45, title=None)


        filename = f'{folder_comparison}/AnnualCost_montecarlo_{year}.png'
        df = df_cost_summary_baseline_notrade[(df_cost_summary_baseline_notrade['year'] == year)]
        df = df.drop(columns=['year'])
        df_errorbars = df_cost_summary_no_trade[(df_cost_summary_no_trade['year'] == year)]
        df_errorbars = df_errorbars.drop(columns=['year'])

        make_stacked_barplot(df, filename, dict_colors=None, df_errorbars=df_errorbars, column_subplot='zone', column_stacked='attribute', column_xaxis='scenario',
                                    column_value='value', select_subplot=None, stacked_grouping=None, order_scenarios=None,
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

        # Select zone
        df = df_cost_summary_baseline[(df_cost_summary_baseline['zone'] == zone)]
        df = df.drop(columns=['zone'])
        df_errorbars = df_cost_summary[(df_cost_summary['zone'] == zone)]
        df_errorbars = df_errorbars.drop(columns=['zone'])

        make_stacked_barplot(df, filename, dict_colors=None, df_errorbars=df_errorbars, 
                                    column_subplot='year', column_stacked='attribute', column_xaxis='scenario',
                                    column_value='value', select_subplot=None, stacked_grouping=None, order_scenarios=None,
                                    dict_scenarios=None,
                                    format_y=lambda y, _: '{:.0f} m$'.format(y), order_stacked=None, cap=2,
                                    annotate=False, show_total=False, fonttick=12, rotation=45, title=None)

        filename = f'{folder_comparison}/AnnualCost_montecarlo_{zone}.png'
        
        # Select zone
        df = df_cost_summary_baseline_notrade[(df_cost_summary_baseline_notrade['zone'] == zone)]
        df = df.drop(columns=['zone'])
        df_errorbars = df_cost_summary_no_trade[(df_cost_summary_no_trade['zone'] == zone)]
        df_errorbars = df_errorbars.drop(columns=['zone'])

        make_stacked_barplot(df, filename, dict_colors=None, df_errorbars=df_errorbars, 
                                  column_subplot='year', column_stacked='attribute', column_xaxis='scenario',
                                    column_value='value', select_subplot=None, stacked_grouping=None, order_scenarios=None,
                                    dict_scenarios=None,
                                    format_y=lambda y, _: '{:.0f} m$'.format(y), order_stacked=None, cap=2,
                                    annotate=False, show_total=False, fonttick=12, rotation=45, title=None)

        filename = f'{folder_comparison}/DemandSupply_montecarlo_{zone}.png'

        # Select zone
        df = df_demandsupply_baseline[(df_demandsupply_baseline['zone'] == zone)]
        df = df.drop(columns=['zone'])
        df_errorbars = df_demandsupply[(df_demandsupply['zone'] == zone)]
        df_errorbars = df_errorbars.drop(columns=['zone'])

        make_stacked_barplot(df, filename, dict_colors=None, df_errorbars=df_errorbars,
                                    column_subplot='attribute', column_stacked='year', column_xaxis='scenario',
                                    column_value='value', select_subplot=None, stacked_grouping=None, order_scenarios=None,
                                    dict_scenarios=None,
                                    format_y=lambda y, _: '{:.0f} GWh'.format(y), order_stacked=None, cap=2,
                                    annotate=False, show_total=False, fonttick=12, rotation=45, title=None, juxtaposed=True)


def postprocess_output(FOLDER, reduced_output=False, selected_scenario='all',
                       plot_dispatch=True, scenario_reference='baseline', graphs_folder='img',
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
    RESULTS_FOLDER, dict_specs, epm_input, epm_results = process_simulation_results(
        FOLDER, keys_results=keys_results)


    GRAPHS_FOLDER = os.path.join(FOLDER, graphs_folder)
    if not os.path.exists(GRAPHS_FOLDER):
        os.makedirs(GRAPHS_FOLDER)
        print(f'Created folder {GRAPHS_FOLDER}')

    # Specific postprocessing for Monte Carlo simulations
    if montecarlo:
        postprocess_montecarlo(epm_results, RESULTS_FOLDER, GRAPHS_FOLDER)

    # If not Monte Carlo, we proceed with the regular postprocessing
    else:

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
            
            scenarios_threshold = 10

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
            
            
            
            # ------------------------------------------------------------------------------------
            
            # Generate a heatmap summary 
            print('Generating heatmap summary...')
            if len(selected_scenarios) < scenarios_threshold:
                figure_name = 'SummaryHeatmap'
                if FIGURES_ACTIVATED.get(figure_name, False):
                    
                    filename = os.path.join(RESULTS_FOLDER, f'{figure_name}.pdf')
                    make_heatmap_plot(epm_results, filename=filename, reference=scenario_reference)
            # ------------------------------------------------------------------------------------
            print('Creating folders for figures...')
            
            # Create subfolders directly under GRAPHS_FOLDER
            subfolders = {}
            for subfolder in ['1_capacity', '2_cost', '3_energy', '4_interconnection', '5_dispatch', '6_maps']:
                subfolders[subfolder] = Path(GRAPHS_FOLDER) / Path(subfolder)
                if not os.path.exists(subfolders[subfolder]):
                    os.mkdir(subfolders[subfolder])
            
            
            # Select main years for x-axis in some plots to simplify the reading
            df = epm_results['pCapacityFuel'].copy()
            selected_years = df['year'][(df['year'] % 5 == 0) | (df['year'] == df['year'].min())].tolist()

            
            
            def calculate_diff(df, scenario_ref=scenario_reference):
                """
                Calculate the difference in 'value' between each scenario and the reference scenario.
                Parameters:
                - df (pd.DataFrame): DataFrame containing 'scenario', 'attribute', and '
                - scenario_ref (str): The reference scenario to compare against.
                Returns:
                - pd.DataFrame: DataFrame with differences in 'value' for each scenario compared to
                the reference scenario.
                """
                df_diff = df.pivot_table(index=[i for i in df.columns if i not in ['scenario', 'value']], columns='scenario', values='value', fill_value=0)
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
                    
                    make_stacked_barplot(df, filename, dict_specs['colors'], 
                                            column_stacked='fuel',
                                            column_subplot='year',
                                            column_value='value', 
                                            column_xaxis='scenario',
                                            select_subplot=selected_years,
                                            format_y=make_auto_yaxis_formatter("GW"), 
                                            rotation=45, 
                                            annotate=False,
                                            format_label="{:.0f}",
                                            title = 'Installed Capacity Mix by Fuel - System (GW)')
                    
                    
                if len(selected_scenarios) > 1 and scenario_reference in selected_scenarios:
                    
                    figure_name = 'CapacityMixSystemEvolutionScenariosRelative'
                    if FIGURES_ACTIVATED.get(figure_name, False):
                        
                        # Capacity comparison between scenarios compare to the reference scenario
                        df_diff = calculate_diff(df, scenario_reference)
                        df_diff['value'] = df_diff['value'] * 1e3
                        
                        filename = os.path.join(subfolders['1_capacity'], f'{figure_name}.pdf')
                        
                        make_stacked_barplot(df_diff, filename, dict_specs['colors'], 
                                                  column_stacked='fuel',
                                                column_subplot='year',
                                                column_xaxis='scenario',
                                                column_value='value',
                                                format_y=make_auto_yaxis_formatter("MW"), 
                                                rotation=45,
                                                annotate=False,
                                                title='Incremental Capacity Mix vs Baseline (MW)', 
                                                show_total=True)
                        

            # 1.2 Evolution of capacity mix per zone
            # Capacity mix in percentage by fuel by zone
            figure_name = 'CapacityMixEvolutionZone'
            if FIGURES_ACTIVATED.get(figure_name, False):
                for scenario in selected_scenarios:
                    df = epm_results['pCapacityFuel'].copy()
                    # MW to GW
                    df['value'] /= 1e3
                    
                    df = df[df['scenario'] == scenario]
                    df = df.drop(columns=['scenario'])
                    
                    filename = os.path.join(subfolders['1_capacity'], f'{figure_name}-{scenario}.pdf')
                    
                    df_line = epm_input['pDemandForecast'].copy()
                    df_line = df_line[df_line['scenario'] == scenario]
                    df_line = df_line.drop(columns=['scenario'])
                    df_line = df_line[df_line['pe'] == 'peak']
                    df_line['value'] /= 1e3
                    
                    
                    make_stacked_barplot(df, 
                                                filename, 
                                                dict_specs['colors'],
                                                overlay_df=df_line,
                                                legend_label='Peak demand',
                                                column_stacked='fuel',
                                                column_subplot='zone',
                                                column_xaxis='year',
                                                column_value='value',
                                                format_y=make_auto_yaxis_formatter("GW"), 
                                                rotation=45,
                                                annotate=False,
                                                title=f'Installed Capacity Mix by Fuel - {scenario} (GW)'
                                                )                     
                    
                    
                    df_percentage = df.set_index(['zone', 'year', 'fuel']).squeeze()
                    df_percentage = df_percentage / df_percentage.groupby(['zone', 'year']).sum()
                    df_percentage = df_percentage.reset_index()
                    
                    filename = os.path.join(subfolders['1_capacity'], f'{figure_name}Percentage-{scenario}.pdf')
                    
                    make_stacked_barplot(df_percentage, 
                                                filename, 
                                                dict_specs['colors'], 
                                                column_stacked='fuel',
                                                column_subplot='zone',
                                                column_xaxis='year',
                                                column_value='value',
                                                format_y=make_auto_yaxis_formatter("%"), rotation=45,
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
                
                    make_stacked_barplot(df, filename, dict_specs['colors'], 
                                            column_stacked='fuel',
                                                column_subplot='zone',
                                                column_xaxis='scenario',
                                                column_value='value',
                                                format_y=make_auto_yaxis_formatter("GW"), 
                                                rotation=45,
                                                annotate=False,
                                                title=f'Installed Capacity Mix by Fuel - {year} (GW)')
            
            
            figure_name = 'NewCapacityInstalledTimeline'
            if FIGURES_ACTIVATED.get(figure_name, False):
                # 1.4 New capacity installed per zone
                for scenario in [scenario_reference]:
                                    
                    # System-level

                    for zone in epm_results['pCapacityPlant'].zone.unique():
                        df_generation = epm_results['pCapacityPlant'].copy()
                        df_generation = df_generation[
                            (df_generation['scenario'] == scenario) & (df_generation['zone'] == zone)
                        ]
                        df_generation = df_generation.copy()
                        df_generation['annotation_label'] = df_generation['generator'].astype(str).apply(
                            lambda name: ('Generation', name)
                        )

                        df_transmission_zone = pd.DataFrame()
                        if 'pAnnualTransmissionCapacity' in epm_results:
                            df_transmission = epm_results['pAnnualTransmissionCapacity'].copy()
                            if 'scenario' in df_transmission.columns:
                                df_transmission = df_transmission[df_transmission['scenario'] == scenario]
                            else:
                                df_transmission = df_transmission.assign(scenario=scenario)

                            df_transmission = df_transmission[
                                (df_transmission['zone'] == zone) | (df_transmission['z2'] == zone)
                            ]

                            if not df_transmission.empty:
                                line_names = (
                                    df_transmission[['zone', 'z2']]
                                    .astype(str)
                                    .apply(lambda row: '-'.join(sorted(row)), axis=1)
                                )
                                df_transmission = df_transmission.assign(
                                    generator=line_names,
                                    fuel='Transmission',
                                    zone=zone,
                                    annotation_label=line_names.apply(lambda name: ('Transmission', name))
                                )
                                df_transmission_zone = (
                                    df_transmission[
                                        ['scenario', 'zone', 'year', 'fuel', 'generator', 'annotation_label', 'value']
                                    ]
                                    .groupby(
                                        ['scenario', 'zone', 'year', 'fuel', 'generator', 'annotation_label'],
                                        observed=False
                                    )['value']
                                    .max()
                                    .reset_index()
                                )

                        df_zone = pd.concat(
                            [
                                df_generation,
                                df_transmission_zone
                            ],
                            ignore_index=True,
                            sort=False
                        )

                        if df_zone.empty:
                            continue
                    
                        filename = os.path.join(subfolders['1_capacity'], f'{figure_name}-{scenario}-{zone}.pdf')
                        make_stacked_areaplot(
                            df_zone,
                            filename,
                            colors=dict_specs['colors'],
                            column_xaxis='year',
                            column_value='value',
                            format_y=make_auto_yaxis_formatter("MW"),
                            column_stacked='fuel',
                            annotation_source='annotation_label',
                            annotation_template="{category}: {value:.0f} MW",
                            title=f'Installed Capacity {zone} - {scenario} [MW]'
                        )
                
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
                
                    make_stacked_barplot(df, filename, dict_specs['colors'], column_stacked='attribute',
                                            column_subplot=None,
                                            column_xaxis='scenario',
                                            column_value='value',
                                            format_y=make_auto_yaxis_formatter("m$"), 
                                            rotation=45,
                                            annotate=False,
                                            title=f'Net Present System Cost by Scenario (million USD)', show_total=True)
                
                # System cost comparison between scenarios compare to the reference scenarios
                if scenario_reference in selected_scenarios and len(selected_scenarios) > 1:
                    figure_name = 'CostSystemScenariosRelative'
                    if FIGURES_ACTIVATED.get(figure_name, False):
                        filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')

                        df_diff = calculate_diff(df, scenario_reference)
                        
                        make_stacked_barplot(df_diff, filename, dict_specs['colors'], column_stacked='attribute',
                                                column_subplot=None,
                                                column_xaxis='scenario',
                                                column_value='value',
                                                format_y=make_auto_yaxis_formatter("m$"), 
                                                rotation=45,
                                                annotate=False,
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
                figure_name = 'CostSystemEvolutionScenarios'
                if FIGURES_ACTIVATED.get(figure_name, False):
                    filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')
                    
                    make_stacked_barplot(df, filename, dict_specs['colors'], 
                                            column_stacked='attribute',
                                            column_subplot='year',
                                            column_value='value', 
                                            column_xaxis='scenario',
                                            select_subplot=selected_years,
                                            format_y=make_auto_yaxis_formatter("m$"), 
                                            rotation=45, 
                                            annotate=False,
                                            format_label="{:.0f}",
                                            show_total=True,
                                            title='System Cost Breakdown by Scenario (million USD)')
                    
                    
                if len(selected_scenarios) > 1 and scenario_reference in selected_scenarios:
                    # Capacity comparison between scenarios compare to the reference scenario
                    df_diff = calculate_diff(df, scenario_reference)
                    
                    figure_name = 'CostSystemEvolutionScenariosRelative'
                    if FIGURES_ACTIVATED.get(figure_name, False):
                        filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')
                        
                        make_stacked_barplot(df_diff, filename, dict_specs['colors'], column_stacked='attribute',
                                                column_subplot='year',
                                                column_xaxis='scenario',
                                                column_value='value',
                                                format_y=make_auto_yaxis_formatter("m$"), 
                                                format_label="{:.0f}",
                                                rotation=45,
                                                annotate=True,
                                                title='Incremental System Cost vs. Baseline (million USD)', 
                                                show_total=True)
                        
            # 2.2 Evolution of capacity mix per zone
            # Capacity mix in percentage by fuel by zone
            figure_name = 'CostZoneEvolution'
            if FIGURES_ACTIVATED.get(figure_name, False):
                for scenario in selected_scenarios:
                    df = df_costzone.copy()
                    df = df[df['scenario'] == scenario]
                    df = df.drop(columns=['scenario'])
                    
                    filename = os.path.join(subfolders['2_cost'], f'{figure_name}-{scenario}.pdf')
                    
                    make_stacked_barplot(df, 
                                                filename, 
                                                dict_specs['colors'], 
                                                column_stacked='attribute',
                                                column_subplot='zone',
                                                column_xaxis='year',
                                                column_value='value',
                                                format_y=make_auto_yaxis_formatter("m$"), 
                                                rotation=45,
                                                annotate=False,
                                                show_total=True,
                                                title=f'Annual Cost Breakdown by Zone – {scenario} (million USD)'
                                                )                     
                    
                    figure_name = f'{figure_name}Percentage-{scenario}'
                    if FIGURES_ACTIVATED.get(figure_name, False):
                        filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')

                        df_percentage = df.set_index(['zone', 'year', 'fuel']).squeeze()
                        df_percentage = df_percentage / df_percentage.groupby(['zone', 'year']).sum()
                        df_percentage = df_percentage.reset_index()
                        
                    
                        make_stacked_barplot(df_percentage, 
                                                    filename, 
                                                    dict_specs['colors'], 
                                                    column_stacked='attribute',
                                                    column_subplot='zone',
                                                    column_xaxis='year',
                                                    column_value='value',
                                                    format_y=make_auto_yaxis_formatter("%"), 
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

                make_stacked_barplot(df, 
                                          filename, dict_specs['colors'], 
                                          column_stacked='attribute',
                                            column_subplot='zone',
                                            column_xaxis='scenario',
                                            column_value='value',
                                            format_y=make_auto_yaxis_formatter("m$"), 
                                            rotation=45,
                                            annotate=False,
                                            show_total=True,
                                            title=f'Cost Composition by Zone in {year} (million USD)')
            
            # 2.4 Capex investment
            figure_name = 'CapexZoneEvolution'
            if FIGURES_ACTIVATED.get(figure_name, False):
                for scenario in selected_scenarios:
                        df = epm_results['pCapexInvestmentComponent'].copy()
                        df = df[df['scenario'] == scenario]
                        df = df.drop(columns=['scenario'])
                        df['value'] = df['value'] / 1e6
                        
                        filename = os.path.join(subfolders['2_cost'], f'{figure_name}-{scenario}.pdf')
                        
                        make_stacked_barplot(df, 
                                                    filename, 
                                                    dict_specs['colors'], 
                                                    column_stacked='attribute',
                                                    column_subplot='zone',
                                                    column_xaxis='year',
                                                    column_value='value',
                                                    format_y=make_auto_yaxis_formatter("m$"), 
                                                    rotation=45,
                                                    annotate=False,
                                                    title=f'Annual Capex by Zone – {scenario} (million USD)'
                                                    )   
                                
            
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
                df = df[~df['fuel'].isin(['Imports', 'Exports'])]
                df = df.loc[df.scenario.isin(selected_scenarios)]
                
                figure_name = 'EnergyMixSystemEvolutionScenarios'
                if FIGURES_ACTIVATED.get(figure_name, False):
                    filename = os.path.join(subfolders['3_energy'], f'{figure_name}.pdf')
                    
                    make_stacked_barplot(df, filename, dict_specs['colors'], 
                                            column_stacked='fuel',
                                            column_subplot='year',
                                            column_value='value', 
                                            column_xaxis='scenario',
                                            select_subplot=selected_years,
                                            format_y=make_auto_yaxis_formatter("GWh"), 
                                            rotation=45, 
                                            show_total=False,
                                            format_label="{:.0f}",
                                            title = 'System Energy Generation Mix by Fuel (GWh)',
                                            annotate=False)
                    
                    
                if len(selected_scenarios) > 1 and scenario_reference in selected_scenarios:
                    # Capacity comparison between scenarios compare to the reference scenario
                    df_diff = calculate_diff(df, scenario_reference)
                    
                    figure_name = 'EnergyMixSystemEvolutionScenariosRelative'
                    if FIGURES_ACTIVATED.get(figure_name, False):
                        filename = os.path.join(subfolders['3_energy'], f'{figure_name}.pdf')
                        
                        make_stacked_barplot(df_diff, filename, dict_specs['colors'], column_stacked='fuel',
                                                column_subplot='year',
                                                column_xaxis='scenario',
                                                column_value='value',
                                                format_y=make_auto_yaxis_formatter("GWh"), rotation=45,
                                                annotate=False,
                                                title='Incremental Energy Generation Mix vs Baseline (GWh)', show_total=True)
            
            # 3.2 Evolution of capacity mix per zone
            figure_name = 'EnergyMixZoneEvolution'
            if FIGURES_ACTIVATED.get(figure_name, False):
                for scenario in selected_scenarios:
                    df = df_energyfuelfull.copy()
                    df = df[df['scenario'] == scenario]
                    df = df.drop(columns=['scenario'])
                    
                    filename = os.path.join(subfolders['3_energy'], f'{figure_name}-{scenario}.pdf')
                    
                    make_stacked_barplot(df, 
                                                filename, 
                                                dict_specs['colors'], 
                                                column_stacked='fuel',
                                                column_subplot='zone',
                                                column_xaxis='year',
                                                column_value='value',
                                                format_y=make_auto_yaxis_formatter("GWh"), rotation=45,
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
                    
                    make_stacked_barplot(df_percentage, 
                                                filename, 
                                                dict_specs['colors'], 
                                                column_stacked='fuel',
                                                column_subplot='zone',
                                                column_xaxis='year',
                                                column_value='value',
                                                format_y=make_auto_yaxis_formatter("%"), rotation=45,
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
                
                    make_stacked_barplot(df, filename, dict_specs['colors'], 
                                            column_stacked='fuel',
                                                column_subplot='zone',
                                                column_xaxis='scenario',
                                                column_value='value',
                                                format_y=make_auto_yaxis_formatter("GWh"), 
                                                rotation=45,
                                                annotate=False,
                                                title=f'Energy Mix by Fuel - {year} (GWh)')

            # 3.4 Energy generation by plant (Top generators by country)
            figure_name = 'EnergyPlantZoneTop10'
            if FIGURES_ACTIVATED.get(figure_name, False):
                df_top = df_energyplant[df_energyplant['scenario'].isin(selected_scenarios)].copy()
                if 'zone' not in df_top.columns:
                    raise ValueError("Column 'zone' is required to build EnergyPlantsTop10 figures by zone.")

                for zone, df_zone in df_top.groupby('zone'):
                    df_zone = (df_zone
                                  .groupby(['scenario', 'year', 'generator'], observed=False)['value']
                                  .sum()
                                  .reset_index())
                    df_zone['rank'] = df_zone.groupby(['scenario', 'year'])['value'].rank(method='first', ascending=False)
                    df_zone = df_zone[df_zone['rank'] <= 10].drop(columns='rank')

                    if df_zone.empty:
                        continue

                    zone_slug = str(zone).replace(' ', '_')
                    filename = os.path.join(subfolders['3_energy'], f'{figure_name}-{zone_slug}.pdf')

                    make_stacked_barplot(
                        df_zone,
                        filename,
                        dict_specs['colors'],
                        column_stacked='generator',
                        column_subplot='year',
                        column_xaxis='scenario',
                        column_value='value',
                        format_y=make_auto_yaxis_formatter("GWh"),
                        rotation=45,
                        annotate=False,
                        title=f'Top 10 Plants - {zone} (GWh)'
                    ) 
                
            
            # 3.4 Energy generation by plant
            figure_name = 'EnergyPlants'
            if FIGURES_ACTIVATED.get(figure_name, False):
                filename = os.path.join(subfolders['3_energy'], f'{figure_name}-{scenario_reference}.pdf')
                if len(df_energyplant.zone.unique()) == 1 and len(epm_results['pEnergyPlant']['generator'].unique()) < 20:
                    print('Generating energy figures for single zone by generators... (not tested yet)')
                    temp = df_energyplant[df_energyplant['scenario'] == scenario_reference]
                    make_stacked_areaplot(
                        temp,
                        filename,
                        colors=dict_specs['colors'],
                        column_xaxis='year',
                        column_value='value',
                        column_stacked='generator',
                        title='Energy Generation by Plant',
                        y_label='Generation (GWh)',
                        legend_title='Energy sources',
                        figsize=(10, 6),
                        stack_sort_by='fuel'
                    )
            
            
            
            # ------------------------------------------------------------------------------------
            # 4. Dispatch
            # ------------------------------------------------------------------------------------                  
                        
            if plot_dispatch:
                print('Generating energy dispatch figures...')
                # Perform automatic Energy DispatchFigures
                make_automatic_dispatch(epm_results, dict_specs, subfolders['5_dispatch'],
                                        ['baseline'], FIGURES_ACTIVATED)
            
            
            # ------------------------------------------------------------------------------------
            # 5. Interconnection Heamap
            # ------------------------------------------------------------------------------------
                  
            print('Generating interconnection figures...')
            
            # 4.1 Net exchange heatmap [GWh and %] evolution
            
            figure_name = 'NetImportsZoneEvolution'
            if FIGURES_ACTIVATED.get(figure_name, False):
                filename = os.path.join(subfolders['4_interconnection'], f'{figure_name}.pdf')
            
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
                heatmap_plot(
                    net_exchange,
                    filename=filename,
                    unit="GWh",
                    title=f"Net Imports by Zone over Time - {scenario_reference} [GWh]",
                    x_column='zone',
                    y_column='year',
                    value_column='value'
                )
                
            figure_name = 'NetImportsZoneEvolutionZoneEvolutionShare'
            if FIGURES_ACTIVATED.get(figure_name, False):
                filename = os.path.join(subfolders['4_interconnection'], f'{figure_name}.pdf')
                
                net_exchange = df_exchange_percentage.set_index(['zone', 'year', 'fuel']).squeeze().unstack('fuel')
                net_exchange.columns.name = None
                net_exchange['Exports'] = net_exchange.get('Exports', 0)
                net_exchange['value'] = net_exchange['Imports'] + net_exchange['Exports']
                net_exchange = net_exchange.reset_index()
                net_exchange = net_exchange.drop(columns=['Imports', 'Exports'])
                net_exchange['fuel'] = 'Net Exchange'
                
                heatmap_plot(
                    net_exchange,
                    filename=filename,
                    unit="%",
                    title=f"Net Imports by Zone over Time - {scenario_reference} [% of energy demand]",
                    x_column='zone',
                    y_column='year',
                    value_column='value'
                )


            # 4.2 Exchange between zones (energy)
            figure_name = 'InterconnectionExchangeHeatmap'
            if FIGURES_ACTIVATED.get(figure_name, False) and 'pInterchange' in epm_results:
                df_interchange = epm_results['pInterchange'].copy()
                df_interchange = df_interchange[df_interchange['scenario'].isin(selected_scenarios)]

                required_cols = {'zone', 'z2', 'year', 'scenario', 'value'}
                missing_cols = required_cols.difference(df_interchange.columns)
                if missing_cols:
                    print(f"Skipping {figure_name}: missing columns {sorted(missing_cols)}")
                else:
                    if scenario_reference not in df_interchange['scenario'].unique():
                        print(f"Skipping {figure_name}: scenario '{scenario_reference}' not available")
                    else:
                        df_interchange = df_interchange[df_interchange['scenario'] == scenario_reference]
                        df_interchange = df_interchange[df_interchange['zone'] != df_interchange['z2']]
                        if df_interchange.empty:
                            print(f"Skipping {figure_name}: no interchange data for scenario '{scenario_reference}'")
                        else:
                            df_interchange = df_interchange.copy()
                            df_interchange['value'] = df_interchange['value'].abs()
                            for year in sorted(df_interchange['year'].unique()):
                                df_plot = df_interchange[df_interchange['year'] == year]
                                if df_plot.empty:
                                    continue
                                filename = os.path.join(
                                    subfolders['4_interconnection'],
                                    f'{figure_name}-{scenario_reference}-{year}.pdf'
                                )
                                title = f'Interconnection Energy Exchange - {scenario_reference} - {year} (GWh)'
                                heatmap_plot(
                                    df_plot,
                                    filename=filename,
                                    unit="GWh",
                                    title=title,
                                    x_column='zone',
                                    y_column='z2',
                                    value_column='value'
                                )
                                
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
                            for year in df_utilization['year'].unique():
                                
                                df_plot = df_utilization[df_utilization['year'] == year]
                                filename = os.path.join(
                                    subfolders['4_interconnection'],
                                    f'{figure_name}-{scenario_reference}-{year}.pdf'
                                )

                                title = f'Interconnection Utilization - {scenario_reference} - {year} (%)'
                                heatmap_plot(
                                    df_plot,
                                    filename=filename,
                                    unit="%",
                                    title=title,
                                    x_column='zone',
                                    y_column='z2',
                                    value_column='value'
                                )
            

            # ------------------------------------------------------------------------------------
            # 6. Interconnection Maps
            # ------------------------------------------------------------------------------------
            
            if (
                'pAnnualTransmissionCapacity' in epm_results
                and epm_results['pAnnualTransmissionCapacity'].zone.nunique() > 0
            ):

                    print('Generating interactive map figures...')
                    make_automatic_map(epm_results, dict_specs, subfolders['6_maps'],
                                       FIGURES_ACTIVATED)


            
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
                    make_stacked_barplot(df_diff, filename, dict_specs['colors'], column_stacked='attribute',
                                            column_subplot=None,
                                            column_xaxis='base_scenario',
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
                    make_stacked_barplot(df_diff, filename, dict_specs['colors'], column_stacked='fuel',
                                            column_subplot=None,
                                            column_xaxis='base_scenario',
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
                    make_stacked_barplot(df_diff, filename, dict_specs['colors'], column_stacked='fuel',
                                            column_subplot=None,
                                            column_xaxis='base_scenario',
                                            column_value='diff',
                                            format_y=lambda y, _: '{:,.0f}'.format(y), rotation=45,
                                            annotate=False,
                                            title=f'Additional Capacity with the Project {year}', show_total=True)
                    print(f'Capacity assessment figures generated successfully: {filename}')
