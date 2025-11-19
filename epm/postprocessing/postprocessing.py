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
import logging
import re
from pathlib import Path
from functools import wraps
import pandas as pd
# Relave imports as it's a submodule
from .utils import *
from .plots import *
from .maps import make_automatic_map


def _wrap_plot_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            label = kwargs.get('filename')
            if label is None:
                label = kwargs.get('title')
            log_warning(f'Failed to generate {label}: {err.args[0]}')
    return wrapper


make_stacked_barplot = _wrap_plot_function(make_stacked_barplot)
make_stacked_areaplot = _wrap_plot_function(make_stacked_areaplot)
make_fuel_dispatchplot = _wrap_plot_function(make_fuel_dispatchplot)
make_heatmap_plot = _wrap_plot_function(make_heatmap_plot)
heatmap_plot = _wrap_plot_function(heatmap_plot)
make_line_plot = _wrap_plot_function(make_line_plot)
#make_automatic_map = _wrap_plot_function(make_automatic_map)

# Used to not load all the parameters in epm_results.gdx for memory purpose
KEYS_RESULTS = {
    # 1. Capacity expansion
    'pCapacityPlant', 
    'pCapacityTechFuel', 'pCapacityTechFuelCountry',
    'pNewCapacityTechFuel', 'pNewCapacityTechFuelCountry',
    'pAnnualTransmissionCapacity', 'pNewTransmissionCapacity',
    # 2. Cost
    'pPrice', 'pYearlyPrice',
    'pCapexInvestmentComponent', 'pCapexInvestmentPlant',
    'pCostsPlant',  
    'pYearlyCostsZone', 'pYearlyCostsCountry',
    'pCostsZone', 'pCostsSystem', 'pCostsSystemPerMWh',
    'pCostsZonePerMWh', 'pCostsCountryPerMWh',
    'pFuelCosts', 'pFuelCostsCountry', 'pFuelConsumption', 'pFuelConsumptionCountry',
    'pYearlyGenCostZonePerMWh',
    # 3. Energy balance
    'pEnergyPlant', 'pEnergyTechFuel', 'pEnergyTechFuelCountry',
    'pEnergyBalance',
    'pUtilizationPlant', 'pUtilizationTechFuel',
    # 4. Energy dispatch
    'pDispatchPlant', 'pDispatch', 'pDispatchTechFuel',
    # 5. Reserves
    'pReserveSpinningPlantZone', 'pReserveSpinningPlantCountry',
    'pReserveMarginCountry',
    # 6. Interconnections
    'pInterchange', 'pInterconUtilization', 'pCongestionShare',
    'pInterchangeExternalExports', 'pInterchangeExternalImports',
    # 7. Emissions
    'pEmissionsZone', 'pEmissionsIntensityZone',
    # 10. Metrics
    'pPlantAnnualLCOE',
    'pCostsZonePerMWh',
    'pCostsCountryPerMWh',
    'pDiscountedDemandZoneMWh',
    'pDiscountedDemandCountryMWh',
    'pDiscountedDemandSystemMWh',
    'pYearlyCostsZonePerMWh',
    'pYearlyCostsCountryPerMWh',
    'pYearlyCostsSystemPerMWh',
    # 11. Other
    'pSolverParameters', 'pGeneratorTechFuel', 'pZoneCountry',
    'pDemandEnergyZone', 'pDemandPeakZone'
}

FIGURES_ACTIVATED = {
    
    'SummaryHeatmap': True,
    
    # 1. Capacity figures
    'CapacityMixSystemEvolutionScenarios': True,
    'CapacityMixSystemEvolutionScenariosRelative': True,
    'CapacityMixEvolutionZone': True,
    'CapacityMixZoneScenarios': True,
    'CapacityMixZoneScenariosRelative': True,
    'NewCapacityZoneInstalledTimeline': True,
    'NewCapacitySystemInstalledTimeline': True,
    
    # 2. Cost figures
    'NPVCostSystemScenarios': True,
    'NPVCostSystemScenariosRelative': True,
    'NPVCostZoneScenarios': True, 
    'NPVCostZoneScenariosRelative': True,
    'NPVCostMWhZoneScenarios': True, 
    'NPVCostMWhZoneScenariosRelative': True,
    'CostSystemEvolutionScenarios': True,
    'CostSystemEvolutionScenariosRelative': True,
    'CostZoneEvolution': True,
    'CostZoneEvolutionPercentage': True,
    'CostZoneScenarios': True,
    'CostMWhZoneEvolution': True,
    'CostMWhZoneEvolutionPercentage': True,
    'CostMWhZoneScenariosYear': True,
    'CostMWhZoneIni': True,
    'GenCostMWhZoneIni': True,
    'GenCostMWhZoneEvolution': True,
    'CapexZoneEvolution': True,
    'PriceBaselineByZone': True,
                    
    # 3. Energy figures
    'EnergyMixSystemEvolutionScenarios': True,
    'EnergyMixSystemEvolutionScenariosRelative': True,
    'EnergyMixZoneEvolution': True,
    'EnergyMixZoneScenarios': True,
    'EnergyPlants': True,
    'EnergyPlantZoneTop10': True,
    
    # 4. Dispatch figures
    'DispatchZoneMaxLoadDay': True,
    'DispatchZoneMaxLoadSeason': False,
    'DispatchZoneFullSeason': True,
    'DispatchSystemMaxLoadDay': True,
    'DispatchSystemMaxLoadSeason': False,
    
    # 5. Interconnection figures
    'NetImportsZoneEvolution': True,
    'NetImportsZoneEvolutionZoneEvolutionShare': True,
    'InterconnectionExchangeHeatmap': True,
    'InterconnectionUtilizationHeatmap': True,

    # 6. Maps
    # 'TransmissionCapacityMap': False, 
    'TransmissionCapacityMapEvolution': False,
    # 'TransmissionUtilizationMap': False,
    'TransmissionUtilizationMapEvolution': False,
    # 'NetExportsMap': True, 
    
    'InteractiveMap': True
}

FIGURE_CATEGORY_ENABLED = {
    'summary': False,
    'capacity': False,
    'costs': False,
    'energy': False,
    'dispatch': True,
    'interconnection': False,
    'maps': False,
}

FIGURE_CATEGORY_MAP = {
    'SummaryHeatmap': 'summary',
    'CapacityMixSystemEvolutionScenarios': 'capacity',
    'CapacityMixSystemEvolutionScenariosRelative': 'capacity',
    'CapacityMixEvolutionZone': 'capacity',
    'CapacityMixZoneScenarios': 'capacity',
    'CapacityMixZoneScenariosRelative': 'capacity',
    'NewCapacityZoneInstalledTimeline': 'capacity',
    'NewCapacitySystemInstalledTimeline': 'capacity',
    'NPVCostSystemScenarios': 'costs',
    'NPVCostSystemScenariosRelative': 'costs',
    'NPVCostZoneScenarios': 'costs',
    'NPVCostZoneScenariosRelative': 'costs',
    'NPVCostMWhZoneScenarios': 'costs',
    'NPVCostMWhZoneScenariosRelative': 'costs',
    'CostSystemEvolutionScenarios': 'costs',
    'CostSystemEvolutionScenariosRelative': 'costs',
    'CostZoneEvolution': 'costs',
    'CostZoneEvolutionPercentage': 'costs',
    'CostZoneScenarios': 'costs',
    'CostMWhZoneEvolution': 'costs',
    'CostMWhZoneEvolutionPercentage': 'costs',
    'CostMWhZoneScenariosYear': 'costs',
    'CostMWhZoneIni': 'costs',
    'GenCostMWhZoneIni': 'costs',
    'GenCostMWhZoneEvolution': 'costs',
    'CapexZoneEvolution': 'costs',
    'PriceBaselineByZone': 'costs',
    'EnergyMixSystemEvolutionScenarios': 'energy',
    'EnergyMixSystemEvolutionScenariosRelative': 'energy',
    'EnergyMixZoneEvolution': 'energy',
    'EnergyMixZoneScenarios': 'energy',
    'EnergyPlants': 'energy',
    'EnergyPlantZoneTop10': 'energy',
    'DispatchZoneMaxLoadDay': 'dispatch',
    'DispatchZoneMaxLoadSeason': 'dispatch',
    'DispatchZoneFullSeason': 'dispatch',
    'DispatchSystemMaxLoadDay': 'dispatch',
    'DispatchSystemMaxLoadSeason': 'dispatch',
    'NetImportsZoneEvolution': 'interconnection',
    'NetImportsZoneEvolutionZoneEvolutionShare': 'interconnection',
    'InterconnectionExchangeHeatmap': 'interconnection',
    'InterconnectionUtilizationHeatmap': 'interconnection',
    'TransmissionCapacityMap': 'maps',
    'TransmissionCapacityMapEvolution': 'maps',
    'TransmissionUtilizationMap': 'maps',
    'TransmissionUtilizationMapEvolution': 'maps',
    'NetExportsMap': 'maps',
    'InteractiveMap': 'maps',
}


def is_figure_active(figure_name: str) -> bool:
    """Return True when both the category and individual figure switches are active."""
    category = FIGURE_CATEGORY_MAP.get(figure_name)
    if category and not FIGURE_CATEGORY_ENABLED.get(category, True):
        return False
    return FIGURES_ACTIVATED.get(figure_name, False)


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

MAX_FULL_SEASON_DAYS = 15


def _day_sort_key(day):
    if pd.isna(day):
        return float('inf')
    try:
        return float(day)
    except (TypeError, ValueError):
        digits = re.sub(r'[^\d.]', '', str(day))
        if digits:
            try:
                return float(digits)
            except ValueError:
                pass
        return str(day)


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

    zone_max_load_day_active = is_figure_active('DispatchZoneMaxLoadDay')
    zone_max_load_season_active = is_figure_active('DispatchZoneMaxLoadSeason')
    zone_full_season_active = is_figure_active('DispatchZoneFullSeason')
    system_max_load_day_active = is_figure_active('DispatchSystemMaxLoadDay')
    system_max_load_season_active = is_figure_active('DispatchSystemMaxLoadSeason')

    generate_zone_figures = zone_max_load_day_active or zone_max_load_season_active or zone_full_season_active
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

    # dispatch_generation = filter_dataframe(epm_results['pDispatchPlant'], {'attribute': ['Generation']})
    dispatch_generation = epm_results['pDispatchTechFuel']
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

                    if zone_full_season_active and year == years_available[0]:
                        filename = os.path.join(folder, f'Dispatch_{selected_scenario}_{zone}_full_season.pdf')
                        full_season_filter = zone_demand_year[
                            (zone_demand_year['season'] == max_load_season)
                        ]
                        unique_days = pd.Index(full_season_filter['day']).dropna().unique()
                        sorted_days = sorted(unique_days, key=_day_sort_key)
                        days_to_plot = sorted_days[:MAX_FULL_SEASON_DAYS]
                        select_time = {'season': [max_load_season]}
                        if len(sorted_days) > 0:
                            select_time['day'] = days_to_plot
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

    costs_notrade = ["Generation costs: $m", "Fixed O&M: $m", "Variable O&M: $m", "Total fuel Costs: $m", "Transmission costs: $m",
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
                       montecarlo=False, reduce_definition_csv=False, logger=None):
    
    active_logger = logger or logging.getLogger("epm.postprocess")
    previous_logger = get_default_logger()
    set_default_logger(active_logger)
    set_utils_logger(active_logger)

    log_info(f"Postprocessing started for {FOLDER}", logger=active_logger)

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
                        log_warning(f"Skipping {filename}: 'year' column not found.", logger=active_logger)
                        continue

                    # Pick first, middle, and last years
                    years = sorted(df['y'].unique())
                    n = len(years)
                    middle_index = n // 2
                    years_to_keep = sorted({years[0], years[middle_index], years[-1]})

                    df_filtered = df[df['y'].isin(years_to_keep)]

                    df_filtered.to_csv(file_path, index=False)


    def simplify_attributes(df, new_label, attributes_list=[]):
        """
        Group attributes into one unique attribute and sum the value
        """
    
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
    RESULTS_FOLDER, dict_specs, epm_results = process_simulation_results(
        FOLDER, keys_results=keys_results)

    set_default_fuel_order(dict_specs.get('fuel_order'))


    GRAPHS_FOLDER = os.path.join(FOLDER, graphs_folder)
    if not os.path.exists(GRAPHS_FOLDER):
        os.makedirs(GRAPHS_FOLDER)
        log_info(f'Created folder {GRAPHS_FOLDER}', logger=active_logger)

    # Specific postprocessing for Monte Carlo simulations
    if montecarlo:
        postprocess_montecarlo(epm_results, RESULTS_FOLDER, GRAPHS_FOLDER)

    # If not Monte Carlo, we proceed with the regular postprocessing
    else:

        if isinstance(selected_scenario, str):
            if selected_scenario == 'all':
                selected_scenarios = list(epm_results['pEnergyTechFuel'].scenario.unique())  # we choose all scenarios
            else:
                selected_scenarios = [selected_scenario]
                assert selected_scenario in list(epm_results['pEnergyTechFuel'].scenario.unique()), "Selected scenario does not belong to the set of scenarios."
        else:
            selected_scenarios = selected_scenario

        # Generate summary
        log_info('Generating summary...', logger=active_logger)
        generate_summary(epm_results, RESULTS_FOLDER)

        # Generate detailed by plant to debug
        if not reduced_output:
            
            scenarios_threshold = 10
            
            # ------------------------------------------------------------------------------------

            # Generate a detailed summary by Power Plant
            log_info('Generating detailed summary by Power Plant...', logger=active_logger)
            try:
                generate_plants_summary(epm_results, RESULTS_FOLDER)
            except Exception as err:
                log_error(f'Failed to generate detailed summary by Power Plant: {err}', logger=active_logger)
            
            # ------------------------------------------------------------------------------------

            # Generate a heatmap summary 
            log_info('Generating heatmap summary...', logger=active_logger)
            if len(selected_scenarios) < scenarios_threshold:
                figure_name = 'SummaryHeatmap'
                if is_figure_active(figure_name):
                    
                    filename = os.path.join(RESULTS_FOLDER, f'{figure_name}.pdf')
                    make_heatmap_plot(epm_results, filename=filename, reference=scenario_reference)
            
            # ------------------------------------------------------------------------------------
            log_info('Creating folders for figures...', logger=active_logger)
            
            # Create subfolders directly under GRAPHS_FOLDER
            subfolders = {}
            for subfolder in ['1_capacity', '2_cost', '3_energy', '4_interconnection', '5_dispatch', '6_maps']:
                subfolders[subfolder] = Path(GRAPHS_FOLDER) / Path(subfolder)
                if not os.path.exists(subfolders[subfolder]):
                    os.mkdir(subfolders[subfolder])
            
            
            # ------------------------------------------------------------------------------------

            # Select main years for x-axis in some plots to simplify the reading
            df = epm_results['pCapacityTechFuel'].copy()
            
            selected_years = df['year'][(df['year'] % 5 == 0) | (df['year'] == df['year'].min())].tolist()
            
            nbr_zones = len(epm_results['pCapacityTechFuel']['zone'].unique())
            
            # ------------------------------------------------------------------------------------
            # 1. Capacity figures
            # ------------------------------------------------------------------------------------

            log_info('Generating capacity figures...', logger=active_logger)
                        
            # 1.1 Evolution of capacity mix for the system (all zones aggregated)
            if len(selected_scenarios) < scenarios_threshold:
                df = epm_results['pCapacityTechFuel'].copy()
                df = df.loc[df.scenario.isin(selected_scenarios)]
                # MW to GW
                df['value'] = df['value'] / 1e3
                
                # TODO: Add year_ini=df['year'].min()
                figure_name = 'CapacityMixSystemEvolutionScenarios'
                if is_figure_active(figure_name):
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
                    if is_figure_active(figure_name):
                        
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
            if nbr_zones > 1:
                if is_figure_active(figure_name):
                    for scenario in selected_scenarios:
                        df = epm_results['pCapacityTechFuel'].copy()
                        # MW to GW
                        df['value'] /= 1e3
                        
                        df = df[df['scenario'] == scenario]
                        df = df.drop(columns=['scenario'])
                        
                        filename = os.path.join(subfolders['1_capacity'], f'{figure_name}-{scenario}.pdf')
                        
                        df_line = epm_results['pDemandPeakZone'].copy()
                        df_line = df_line[df_line['scenario'] == scenario]
                        df_line = df_line.drop(columns=['scenario'])
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
            if nbr_zones > 1:
                for year in [min(epm_results['pCapacityTechFuel']['year']), max(epm_results['pCapacityTechFuel']['year'])]:
                    df = epm_results['pCapacityTechFuel'].copy()
                    # MW to GW
                    df['value'] = df['value'] / 1e3
                    df = df[df['year'] == year]                
                    df = df.drop(columns=['year'])

                    # TODO: percentage ?
                    if is_figure_active(figure_name):
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
                
                    if len(selected_scenarios) > 1 and scenario_reference in selected_scenarios:
                        
                        figure_name = 'CapacityMixZoneScenariosRelative'
                        if is_figure_active(figure_name):
                            
                            # Capacity comparison between scenarios compare to the reference scenario
                            df_diff = calculate_diff(df, scenario_reference)
                            df_diff['value'] = df_diff['value'] * 1e3
                            
                            filename = os.path.join(subfolders['1_capacity'], f'{figure_name}-{year}.pdf')        
                            
                            make_stacked_barplot(df_diff, filename, dict_specs['colors'], 
                            column_stacked='fuel',
                                column_subplot='zone',
                                column_xaxis='scenario',
                                column_value='value',
                                format_y=make_auto_yaxis_formatter("GW"), 
                                rotation=45,
                                annotate=False,
                                title=f'Installed Capacity Mix vs Baseline - {year} (GW)')
                                                
            # 1.4 New capacity timeline      
            figure_name_zone = 'NewCapacityZoneInstalledTimeline'
            if is_figure_active(figure_name_zone):
                df_capacity = epm_results['pCapacityPlant'].copy()
                unique_zones = df_capacity.zone.unique()

                for scenario in [scenario_reference]:
                    for zone in unique_zones:
                        df_generation = df_capacity[
                            (df_capacity['scenario'] == scenario) & (df_capacity['zone'] == zone)
                        ].copy()
                        if df_generation.empty:
                            continue

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

                        filename = os.path.join(subfolders['1_capacity'], f'{figure_name_zone}-{scenario}-{zone}.pdf')
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

            figure_name_system_base = 'NewCapacitySystemInstalledTimeline'
            for scenario in selected_scenarios:
                figure_name_system = f'{figure_name_system_base}-{scenario}'
                if not (
                    is_figure_active(figure_name_system)
                    or is_figure_active(figure_name_system_base)
                ):
                    continue

                df_generation_system = epm_results['pCapacityPlant'].copy()
                df_generation_system = df_generation_system[
                    df_generation_system['scenario'] == scenario
                ].copy()
                if df_generation_system.empty and 'pAnnualTransmissionCapacity' not in epm_results:
                    continue

                if not df_generation_system.empty:
                    df_generation_system['zone'] = 'System'
                    df_generation_system['annotation_label'] = df_generation_system['generator'].astype(str).apply(
                        lambda name: ('Generation', name)
                    )

                df_transmission_system = pd.DataFrame()
                if 'pAnnualTransmissionCapacity' in epm_results:
                    df_transmission = epm_results['pAnnualTransmissionCapacity'].copy()
                    if 'scenario' in df_transmission.columns:
                        df_transmission = df_transmission[df_transmission['scenario'] == scenario]
                    else:
                        df_transmission = df_transmission.assign(scenario=scenario)

                    if not df_transmission.empty:
                        line_names = (
                            df_transmission[['zone', 'z2']]
                            .astype(str)
                            .apply(lambda row: '-'.join(sorted(row)), axis=1)
                        )
                        df_transmission = df_transmission.assign(
                            generator=line_names,
                            fuel='Transmission',
                            zone='System',
                            annotation_label=line_names.apply(lambda name: ('Transmission', name))
                        )
                        df_transmission_system = (
                            df_transmission[
                                ['scenario', 'year', 'fuel', 'generator', 'annotation_label', 'value', 'zone']
                            ]
                            .groupby(
                                ['scenario', 'year', 'fuel', 'generator', 'annotation_label', 'zone'],
                                observed=False
                            )['value']
                            .max()
                            .reset_index()
                        )

                df_system = pd.concat(
                    [
                        df_generation_system,
                        df_transmission_system
                    ],
                    ignore_index=True,
                    sort=False
                )

                if df_system.empty:
                    continue

                filename = os.path.join(subfolders['1_capacity'], f'{figure_name_system}.pdf')
                make_stacked_areaplot(
                    df_system,
                    filename,
                    colors=dict_specs['colors'],
                    column_xaxis='year',
                    column_value='value',
                    format_y=make_auto_yaxis_formatter("MW"),
                    column_stacked='fuel',
                    annotation_source='annotation_label',
                    annotation_template="{category}: {value:.0f} MW",
                    title=f'Installed Capacity System - {scenario} [MW]'
                )
                
            
            # ------------------------------------------------------------------------------------
            # 2. Cost figures
            # ------------------------------------------------------------------------------------

            log_info('Generating cost figures...', logger=active_logger)
             
            figure_name = 'PriceBaselineByZone'
            if is_figure_active(figure_name):
                df_price = epm_results['pYearlyPrice'].copy()    
                df_price = df_price[df_price['scenario'] == scenario_reference]
                df_price = (
                    df_price[['year', 'zone', 'value']]
                    .dropna(subset=['year', 'value'])
                    .sort_values(['year', 'zone'])
                )
                filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')
                make_line_plot(
                    df=df_price,
                    filename=filename,
                    column_xaxis='year',
                    y_column='value',
                    series_column='zone',
                    format_y=make_auto_yaxis_formatter('$/MWh'),
                    title=f'Electricity Price by Zone – {scenario_reference} (USD/MWh)'
                )
            
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
                
                
                figure_name = 'NPVCostSystemScenarios'
                if is_figure_active(figure_name):
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
                    figure_name = 'NPVCostSystemScenariosRelative'
                    if is_figure_active(figure_name):
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

                
                df = epm_results['pCostsZone'].copy()
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
                
                
                figure_name = 'NPVCostZoneScenarios'
                if is_figure_active(figure_name):
                    filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')
                
                    make_stacked_barplot(df, filename, dict_specs['colors'], column_stacked='attribute',
                                            column_subplot='zone',
                                            column_xaxis='scenario',
                                            column_value='value',
                                            format_y=make_auto_yaxis_formatter("m$"), 
                                            rotation=45,
                                            annotate=False,
                                            title=f'Net Present System Cost by Scenario (million USD)', show_total=True) 
                    
                    # System cost comparison between scenarios compare to the reference scenarios
                    if scenario_reference in selected_scenarios and len(selected_scenarios) > 1:
                        figure_name = 'NPVCostZoneScenariosRelative'
                        if is_figure_active(figure_name):
                            filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')

                            df_diff = calculate_diff(df, scenario_reference)
                            
                            make_stacked_barplot(df_diff, filename, dict_specs['colors'], column_stacked='attribute',
                                                    column_subplot='zone',
                                                    column_xaxis='scenario',
                                                    column_value='value',
                                                    format_y=make_auto_yaxis_formatter("m$"), 
                                                    rotation=45,
                                                    annotate=False,
                                                    title='Additional System Cost vs. Baseline (NPV, million USD)', 
                                                    show_total=True)  
                
                df = epm_results['pCostsZonePerMWh'].copy()
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
                
                
                figure_name = 'NPVCostMWhZoneScenarios'
                if is_figure_active(figure_name):
                    filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')
                
                    make_stacked_barplot(df, filename, dict_specs['colors'], column_stacked='attribute',
                                            column_subplot='zone',
                                            column_xaxis='scenario',
                                            column_value='value',
                                            format_y=make_auto_yaxis_formatter("$/MWh"), 
                                            rotation=45,
                                            annotate=False,
                                            title='Discounted Cost per Scenario (USD/MWh)', show_total=True) 
                    
                    # System cost comparison between scenarios compare to the reference scenarios
                    if scenario_reference in selected_scenarios and len(selected_scenarios) > 1:
                        figure_name = 'NPVCostMWhZoneScenariosRelative'
                        if is_figure_active(figure_name):
                            filename = os.path.join(subfolders['2_cost'], f'{figure_name}.pdf')

                            df_diff = calculate_diff(df, scenario_reference)
                            
                            make_stacked_barplot(df_diff, filename, dict_specs['colors'], column_stacked='attribute',
                                                    column_subplot='zone',
                                                    column_xaxis='scenario',
                                                    column_value='value',
                                                    format_y=make_auto_yaxis_formatter("$/MWh"), 
                                                    rotation=45,
                                                    annotate=False,
                                                    title='Additional Cost per Scenario vs. Baseline (USD/MWh)', 
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
                if is_figure_active(figure_name):
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
                    if is_figure_active(figure_name):
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
            if nbr_zones > 1:
                if is_figure_active(figure_name):
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
                        if is_figure_active(figure_name):
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
            figure_name = 'CostZoneScenariosYear'
            if nbr_zones > 1:
                if is_figure_active(figure_name):
                    filename = os.path.join(subfolders['2_cost'], f'{figure_name}-{year}.pdf')

                    df = df_costzone.copy()

                    df = df.loc[df.scenario.isin(selected_scenarios)]
                    df = df.loc[(df.year == max(df['year'].unique()))]

                    make_stacked_barplot(
                        df,
                        filename,
                        dict_specs['colors'],
                        column_stacked='attribute',
                        column_subplot='zone',
                        column_xaxis='scenario',
                        column_value='value',
                        format_y=make_auto_yaxis_formatter("m$"),
                        rotation=45,
                        annotate=False,
                        show_total=True,
                        title=f'Cost Composition by Zone in {year} (million USD)'
                    )
 
            # Equivalent cost figures expressed in USD/MWh
            df_costzone_mwh = epm_results['pYearlyCostsZonePerMWh'].copy()
            df_costzone_mwh = simplify_attributes(df_costzone_mwh, "Unmet reserve costs: $m", RESERVE_ATTRS)
            df_costzone_mwh = simplify_attributes(df_costzone_mwh, "Trade costs: $m", TRADE_ATTRS)
            df_costzone_mwh['attribute'] = df_costzone_mwh['attribute'].str.replace(': $m', '', regex=False)
            df_costzone_mwh['attribute'] = df_costzone_mwh['attribute'].str.replace(': $/MWh', '', regex=False)

            figure_name = 'CostMWhZoneEvolution'
            if nbr_zones > 1:
                if is_figure_active(figure_name):
                    for scenario in selected_scenarios:
                        df = df_costzone_mwh.copy()
                        df = df[df['scenario'] == scenario]
                        df = df.drop(columns=['scenario'])

                        filename = os.path.join(subfolders['2_cost'], f'{figure_name}-{scenario}.pdf')

                        make_stacked_barplot(
                            df,
                            filename,
                            dict_specs['colors'],
                            column_stacked='attribute',
                            column_subplot='zone',
                            column_xaxis='year',
                            column_value='value',
                            format_y=make_auto_yaxis_formatter("$/MWh"),
                            rotation=45,
                            annotate=False,
                            show_total=True,
                            title=f'Annual Cost Breakdown by Zone – {scenario} (USD/MWh)'
                        )


            figure_name = 'CostMWhZoneScenariosYear'
            if nbr_zones > 1:
                if is_figure_active(figure_name):
                    filename = os.path.join(subfolders['2_cost'], f'{figure_name}-{year}.pdf')

                    df = df_costzone_mwh.copy()
                    df = df.loc[df.scenario.isin(selected_scenarios)]
                    df = df.loc[(df.year == max(df['year'].unique()))]

                    make_stacked_barplot(
                        df,
                        filename,
                        dict_specs['colors'],
                        column_stacked='attribute',
                        column_subplot='zone',
                        column_xaxis='scenario',
                        column_value='value',
                        format_y=make_auto_yaxis_formatter("$/MWh"),
                        rotation=45,
                        annotate=False,
                        show_total=True,
                        title=f'Cost Composition by Zone in {year} (USD/MWh)'
                    )

            figure_name = 'CostMWhZoneIni'
            if nbr_zones > 1 and scenario_reference in df_costzone_mwh['scenario'].unique():
                if is_figure_active(figure_name):
                    year_ini = df_costzone_mwh['year'].min()
                    df_ini = df_costzone_mwh[
                        (df_costzone_mwh['scenario'] == scenario_reference) &
                        (df_costzone_mwh['year'] == year_ini)
                    ].copy()
                    
                    filename = os.path.join(
                            subfolders['2_cost'],
                            f'{figure_name}-{scenario_reference}.pdf'
                        )

                    make_stacked_barplot(
                        df_ini,
                        filename,
                        dict_specs['colors'],
                        column_stacked='attribute',
                        column_subplot=None,
                        column_xaxis='zone',
                        column_value='value',
                        format_y=make_auto_yaxis_formatter('$/MWh'),
                        rotation=45,
                        annotate=False,
                        show_total=True,
                        title=f'Cost Composition by Zone in {year_ini} – {scenario_reference} (USD/MWh)'
                    )
            
            figure_name = 'GenCostMWhZoneIni'
            df_gen_cost_zone = epm_results['pYearlyGenCostZonePerMWh'].copy()
            df_gen_cost_zone['attribute'] = df_gen_cost_zone['attribute'].str.replace(': $m', '', regex=False)

            if nbr_zones > 1 and scenario_reference in df_gen_cost_zone['scenario'].unique():
                if is_figure_active(figure_name):
                    first_year = df_gen_cost_zone.loc[
                        df_gen_cost_zone['scenario'] == scenario_reference,
                        'year'
                    ].min()
                    df_gen_ini = df_gen_cost_zone[
                        (df_gen_cost_zone['scenario'] == scenario_reference)
                        & (df_gen_cost_zone['year'] == first_year)
                    ].copy()
                    filename = os.path.join(
                        subfolders['2_cost'],
                        f'{figure_name}-{scenario_reference}.pdf'
                    )

                    make_stacked_barplot(
                        df_gen_ini,
                        filename,
                        dict_specs['colors'],
                        column_stacked='attribute',
                        column_subplot=None,
                        column_xaxis='zone',
                        column_value='value',
                        format_y=make_auto_yaxis_formatter('$/MWh'),
                        rotation=45,
                        annotate=False,
                        show_total=True,
                        title=f'Generation Cost Composition by Zone in {first_year} – {scenario_reference} (USD/MWh)'
                    )

            figure_name = 'GenCostMWhZoneEvolution'
            if (
                nbr_zones > 1
                and scenario_reference in df_gen_cost_zone['scenario'].unique()
            ):
                if is_figure_active(figure_name):
                    df_gen_evo = df_gen_cost_zone[
                        df_gen_cost_zone['scenario'] == scenario_reference
                    ].copy()
                    filename = os.path.join(
                        subfolders['2_cost'],
                        f'{figure_name}-{scenario_reference}.pdf'
                    )

                    make_stacked_barplot(
                        df_gen_evo,
                        filename,
                        dict_specs['colors'],
                        column_stacked='attribute',
                        column_subplot='zone',
                        column_xaxis='year',
                        column_value='value',
                        format_y=make_auto_yaxis_formatter('$/MWh'),
                        rotation=45,
                        annotate=False,
                        show_total=True,
                        title=f'Generation Cost Composition by Zone – {scenario_reference} (USD/MWh)'
                    )




            # 2.4 Capex investment per zone
            figure_name = 'CapexZoneEvolution'
            if nbr_zones > 1:
                if is_figure_active(figure_name):
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

            log_info('Generating energy figures...', logger=active_logger)
            
            # Prepare dataframes for energy
            df_energyfuel = epm_results['pEnergyTechFuel'].copy()
            
            # Additionnal energy information not in pEnergyTechFuel
            # TODO: check zext for this one
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
            
            # 3.1 Evolution of energy mix for the system (all zones aggregated)
            if len(selected_scenarios) < scenarios_threshold:
                df = df_energyfuelfull.copy()
                df = df[~df['fuel'].isin(['Imports', 'Exports'])]
                df = df.loc[df.scenario.isin(selected_scenarios)]
                
                figure_name = 'EnergyMixSystemEvolutionScenarios'
                if is_figure_active(figure_name):
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
                    if is_figure_active(figure_name):
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
            if nbr_zones > 1:
                if is_figure_active(figure_name):
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
            if nbr_zones > 1:
                for year in [min(df_energyfuelfull['year']), max(df_energyfuelfull['year'])]:
                    df = df_energyfuelfull.copy()
                    df = df[df['year'] == year]                
                    df = df.drop(columns=['year'])

                    # TODO: percentage ?
                    if is_figure_active(figure_name):
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
            if is_figure_active(figure_name):
                df_energyplant = epm_results['pEnergyPlant'].copy()
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
            if is_figure_active(figure_name):
                filename = os.path.join(subfolders['3_energy'], f'{figure_name}-{scenario_reference}.pdf')
                df_energyplant = epm_results['pEnergyPlant'].copy()
                if nbr_zones == 1 and len(epm_results['pEnergyPlant']['generator'].unique()) < 20:
                    log_info('Generating energy figures for single zone by generators... (not tested yet)', logger=active_logger)
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
                log_info('Generating energy dispatch figures...', logger=active_logger)
                # Perform automatic Energy DispatchFigures
                try:
                    make_automatic_dispatch(
                        epm_results,
                        dict_specs,
                        subfolders['5_dispatch'],
                        ['baseline'],
                        FIGURES_ACTIVATED
                    )
                except Exception as err:
                    log_warning(f'Failed to generate dispatch figures: {err}', logger=active_logger)
            
            # ------------------------------------------------------------------------------------
            # 5. Interconnection Heamap
            # ------------------------------------------------------------------------------------
            
            if nbr_zones > 1:
                log_info('Generating interconnection figures...', logger=active_logger)
                
                # 4.1 Net exchange heatmap [GWh and %] evolution
                figure_name = 'NetImportsZoneEvolution'
                if is_figure_active(figure_name):
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
                if is_figure_active(figure_name):
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
                if is_figure_active(figure_name) and 'pInterchange' in epm_results:
                    df_interchange = epm_results['pInterchange'].copy()
                    df_interchange = df_interchange[df_interchange['scenario'].isin(selected_scenarios)]

                    required_cols = {'zone', 'z2', 'year', 'scenario', 'value'}
                    missing_cols = required_cols.difference(df_interchange.columns)
                    if missing_cols:
                        log_warning(f"Skipping {figure_name}: missing columns {sorted(missing_cols)}", logger=active_logger)
                    else:
                        if scenario_reference not in df_interchange['scenario'].unique():
                            log_warning(f"Skipping {figure_name}: scenario '{scenario_reference}' not available", logger=active_logger)
                        else:
                            df_interchange = df_interchange[df_interchange['scenario'] == scenario_reference]
                            df_interchange = df_interchange[df_interchange['zone'] != df_interchange['z2']]
                            if df_interchange.empty:
                                log_warning(f"Skipping {figure_name}: no interchange data for scenario '{scenario_reference}'", logger=active_logger)
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
                if is_figure_active(figure_name) and 'pInterconUtilization' in epm_results:
                    df_utilization = epm_results['pInterconUtilization'].copy()
                    df_utilization = df_utilization[df_utilization['scenario'].isin(selected_scenarios)]

                    required_cols = {'zone', 'z2', 'year', 'scenario', 'value'}
                    missing_cols = required_cols.difference(df_utilization.columns)
                    if missing_cols:
                        log_warning(f"Skipping {figure_name}: missing columns {sorted(missing_cols)}", logger=active_logger)
                    else:
                        if scenario_reference not in df_utilization['scenario'].unique():
                            log_warning(f"Skipping {figure_name}: scenario '{scenario_reference}' not available", logger=active_logger)
                        else:
                            df_utilization = df_utilization[df_utilization['scenario'] == scenario_reference]

                            if df_utilization.empty:
                                log_warning(f"Skipping {figure_name}: no utilization data for scenario '{scenario_reference}'", logger=active_logger)
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
            if nbr_zones > 1:
                if (
                    'pAnnualTransmissionCapacity' in epm_results
                    and epm_results['pAnnualTransmissionCapacity'].zone.nunique() > 0
                ):

                        log_info('Generating interactive map figures...', logger=active_logger)
                        make_automatic_map(
                            epm_results,
                            dict_specs,
                            subfolders['6_maps'],
                            FIGURES_ACTIVATED,
                            figure_is_active=is_figure_active,
                        )

            
            if False:
                #----------------------- Project Economic Assessment -----------------------
                # Difference between scenarios with and without a project
                log_info('Generating project economic assessment figures...', logger=active_logger)
                
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
                    log_info(f'System cost assessment figures generated successfully: {filename}', logger=active_logger)

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
                    log_info(f'Energy assessment figures generated successfully: {filename}', logger=active_logger)
                
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
                    log_info(f'Capacity assessment figures generated successfully: {filename}', logger=active_logger)

    log_info(f"Postprocessing finished for {FOLDER}", logger=active_logger)
    set_default_logger(previous_logger)
    set_utils_logger(previous_logger)
