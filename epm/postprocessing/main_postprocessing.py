import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('..')

from utils import create_folders, read_plot_specs, read_input_data, extract_epm_results, process_epmresults, make_demand_plot, make_generation_plot, calculate_pRR, make_fuel_energy_mix_pie_plot, make_fuel_capacity_mix_pie_plot, stacked_area_plot, dispatch_plot, make_fuel_dispatch_plot, make_complete_fuel_dispatch_plot, make_capacity_plot


if __name__ == '__main__':
    REGION_NAME = 'Liberia'  # name of the region
    RESULTS_FOLDER = 'EPM_Results/'  # where to find EPM results
    GRAPHS_RESULTS = 'Results/'
    SCENARIO = 'OldBaseline'
    # YEAR = 2035
    DISCOUNT_RATE = 0.06

    create_folders(GRAPHS_RESULTS, SCENARIO)

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

    make_demand_plot(epm_dict['pDemandSupplyCountry'], GRAPHS_RESULTS, years=None, plot_option='bar',
                     selected_scenario=SCENARIO)

    # Total generation
    make_generation_plot(epm_dict['pEnergyByFuel'], GRAPHS_RESULTS, years=None, plot_option='bar',
                         selected_scenario=SCENARIO)
    # Fuel mix one year
    make_fuel_energy_mix_pie_plot(epm_dict['pEnergyByFuel'], [2033], GRAPHS_RESULTS, dict_specs['colors'],
                                  BESS_included=True, selected_scenario=SCENARIO)

    # Fuel mix multiple years
    make_fuel_energy_mix_pie_plot(epm_dict['pEnergyByFuel'], [2023, 2027, 2033], GRAPHS_RESULTS, dict_specs['colors'],
                                  BESS_included=True, selected_scenario=SCENARIO)

    make_fuel_capacity_mix_pie_plot(epm_dict['pCapacityByFuel'], [2023, 2027, 2033], GRAPHS_RESULTS,
                                    dict_specs['colors'], selected_scenario=SCENARIO)

    select_time = {
        'season': ['m1'],
        'day': ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
    }

    make_fuel_dispatch_plot(epm_dict['pFuelDispatch'], GRAPHS_RESULTS, dict_specs['colors'], zone='Liberia', year=2027,
                            scenario=SCENARIO, fuel_grouping=fuel_grouping, select_time=select_time)

    make_capacity_plot(epm_dict['pCapacityByFuel'], GRAPHS_RESULTS, dict_specs['colors'], zone='Liberia',
                       select_stacked=[2023, 2027, 2033], fuel_grouping=fuel_grouping)

    subset_dispatch = epm_dict['pDispatch'].loc[
        epm_dict['pDispatch'].attribute.isin(['Unmet demand', 'Exports', 'Storage Charge'])]

    dfs_to_plot_area = {
        'pFuelDispatch': epm_dict['pFuelDispatch'],
        'pDispatch': subset_dispatch
    }

    subset_demand = epm_dict['pDispatch'].loc[epm_dict['pDispatch'].attribute.isin(['Demand'])]

    dfs_to_plot_line = {
        'pDispatch': subset_demand
    }

    select_time = {
        'season': ['m1'],
        'day': ['d21', 'd22', 'd23', 'd24', 'd25', 'd26', 'd27', 'd28', 'd29', 'd30']
    }

    make_complete_fuel_dispatch_plot(dfs_to_plot_area, dfs_to_plot_line, GRAPHS_RESULTS, dict_specs['colors'],
                                     zone='Liberia', year=2027, scenario=SCENARIO, fuel_grouping=fuel_grouping,
                                     select_time=select_time)

    make_capacity_plot(epm_dict['pCapacityByFuel'], GRAPHS_RESULTS, dict_specs['colors'], zone='Liberia',
                       select_stacked=[2023, 2027, 2033], fuel_grouping=fuel_grouping)