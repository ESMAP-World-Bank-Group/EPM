from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('..')

from utils import create_folders, read_plot_specs, read_input_data, extract_epm_results, process_epmresults, \
    make_demand_plot, make_generation_plot, calculate_pRR, make_fuel_energy_mix_pie_plot, \
    make_fuel_capacity_mix_pie_plot, stacked_area_plot, dispatch_plot, make_fuel_dispatch_plot, \
    make_complete_fuel_dispatch_plot, stacked_bar_plot, make_capacity_plot


# TODO: Add interpolation for years for missing years in order to calculate total year
# TODO: Read .gdx input files instead of .xlsx
# TODO: Make fontsize for all figures
# TODO: Format figures in .pdf ?
# TODO: Stake area plot. Issues when df_2. Legend and display.
# TODO: Make summary.pdf with all figures

REGION_NAME = 'Guinea' #'Liberia'
RESULTS_FOLDER = 'EPM_Results/'  # where to find EPM results
GRAPHS_RESULTS = 'Results/'
SCENARIO = '1_Baseline'# '3.SP2 RoRwPVWBESS'
YEAR = 2035
DISCOUNT_RATE = 0.06


# TODO: Not necessary to remove.
# create_folders(GRAPHS_RESULTS, SCENARIO)

selected_scenario = SCENARIO

# TODO: Can be read directly by resource. Ideally, not required.
dict_specs = read_plot_specs()
epmresults = extract_epm_results(RESULTS_FOLDER, SCENARIO)
epm_dict = process_epmresults(epmresults, dict_specs)

fuel_grouping = {
    'Battery Storage 4h': 'Battery Storage',
    'Battery Storage 8h': 'Battery Storage',
    'Oil diesel': 'Oil',
    'Hydro MC': 'Hydro',
    'Hydro RoR': 'Hydro',
    'Hydro Storage': 'Hydro'
}
fuel_grouping = None

select_time = {
    'season': ['m1'],
    'day': ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
}
select_time = {
    'season': ['m1'],
    'day': ['d21', 'd22', 'd23', 'd24', 'd25', 'd26', 'd27', 'd28', 'd29', 'd30']
}


select_time = None

if True:

    # Plots example.

    # Total demand
    make_demand_plot(epm_dict['pDemandSupplyCountry'], GRAPHS_RESULTS, years=None, plot_option='bar',
                     selected_scenario=SCENARIO)
    # Total generation
    make_generation_plot(epm_dict['pEnergyByFuel'], GRAPHS_RESULTS, years=None, plot_option='bar',
                         selected_scenario=SCENARIO)
    # Fuel mix one year
    make_fuel_energy_mix_pie_plot(epm_dict['pEnergyByFuel'], [2035], GRAPHS_RESULTS, dict_specs['colors'],
                                  selected_scenario=SCENARIO)
    # Fuel mix multiple years
    make_fuel_energy_mix_pie_plot(epm_dict['pEnergyByFuel'], [2025, 2030, 2035], GRAPHS_RESULTS, dict_specs['colors'],
                                  selected_scenario=SCENARIO)
    make_fuel_capacity_mix_pie_plot(epm_dict['pCapacityByFuel'], [2025, 2030, 2035], GRAPHS_RESULTS,
                                    dict_specs['colors'], selected_scenario=SCENARIO)

    # Stacked plots
    # TODO: Only show the most important plants in the plot.
    filename = f'{GRAPHS_RESULTS}/PlantEnergyMixStackedAreaPlot_{selected_scenario}.png'
    stacked_area_plot(epm_dict['pEnergyByPlant'], filename, None, x_column='year', y_column='value',
                      stack_column='generator', title='Plant Energy by Fuel Type', y_label='Energy (GWh)',
                      legend_title='Energy sources', figsize=(10, 6))

    filename = f'{GRAPHS_RESULTS}/EnergyCapacityMixStackedAreaPlot_{selected_scenario}.png'
    stacked_area_plot(epm_dict['pCapacityByFuel'], filename, dict_specs['colors'], x_column='year', y_column='value',
                      stack_column='fuel', title='Energy Capacity by Fuel Type', y_label='Capacity (MW)',
                      legend_title='Energy sources', figsize=(10, 6))

    filename = f'{GRAPHS_RESULTS}/EnergyGenerationMixStackedAreaPlot_{selected_scenario}.png'
    stacked_area_plot(epm_dict['pEnergyByFuel'], filename, dict_specs['colors'], x_column='year', y_column='value',
                           stack_column='fuel', title='Energy Generation by Fuel Type', y_label='Generation (GWh)',
                           legend_title='Energy sources', figsize=(10, 6))

    filename = f'{GRAPHS_RESULTS}/EnergyGenerationMixStackedBarPlot_{selected_scenario}.png'
    stacked_bar_plot(epm_dict['pEnergyByFuel'], filename, dict_specs['colors'], x_column='year', y_column='value',
                     stack_column='fuel', title='Energy Generation by Fuel Type', y_label='Generation (GWh)',
                     legend_title='Energy sources', figsize=(10, 6))

    filename = f'{GRAPHS_RESULTS}/EnergyGenerationMixStackedAreaPlotEmission_{selected_scenario}.png'
    stacked_area_plot(epm_dict['pEnergyByFuel'], filename, dict_specs['colors'], x_column='year', y_column='value',
                           stack_column='fuel', title='Energy Generation by Fuel Type', y_label='Generation (GWh)',
                           legend_title='Energy sources', df_2=epm_dict['pEmissions'], figsize=(10, 6))



    temp = epm_dict['pCostSummary']
    temp = temp[~temp['attribute'].isin(['Capex: $m', 'Total Annual Cost by Zone: $m'])]
    #temp = temp.set_index([i for i in temp.columns if i != 'value'])
    filename = f'{GRAPHS_RESULTS}/CostSummary_{selected_scenario}.png'
    stacked_area_plot(temp, filename, None, x_column='year', y_column='value',
                           stack_column='attribute', title='Total Cost (m$/year)',
                           y_label='Total cost (m$/year)', figsize=(10, 6))

    # TODO: Make legend on the right side with on column.

    make_fuel_dispatch_plot(epm_dict['pFuelDispatch'], GRAPHS_RESULTS, dict_specs['colors'], zone=REGION_NAME, year=YEAR,
                            scenario=SCENARIO, fuel_grouping=fuel_grouping, select_time=select_time)

    dfs_to_plot_area = {
        'pFuelDispatch': epm_dict['pFuelDispatch'],
        'pDispatch': epm_dict['pDispatch'].loc[epm_dict['pDispatch'].attribute.isin(['Unmet demand', 'Exports'])]
    }

    dfs_to_plot_line = {
        'pDispatch': epm_dict['pDispatch'].loc[epm_dict['pDispatch'].attribute.isin(['Demand'])]
    }

    make_complete_fuel_dispatch_plot(dfs_to_plot_area, dfs_to_plot_line, GRAPHS_RESULTS, dict_specs['colors'],
                                     zone=REGION_NAME,
                                     year=YEAR, scenario=SCENARIO, fuel_grouping=fuel_grouping,
                                     select_time=select_time)

    make_capacity_plot(epm_dict['pCapacityByFuel'], GRAPHS_RESULTS, dict_specs['colors'], zone='Liberia',
                       select_stacked=[2023, 2027, 2028, 2029, 2030, 2033], fuel_grouping=fuel_grouping)
