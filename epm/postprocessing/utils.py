# General packages to process and plot EPM results
# Docstring formatting should follow the NumPy/SciPy format: https://numpydoc.readthedocs.io/en/latest/format.html
# The first line should be a short and concise summary of the function's purpose.
# The second line should be blank, and any further lines should be wrapped at 72 characters.
# The docstring should describe the function's purpose, arguments, and return values, as well as any exceptions that the function may raise.
# The docstring should also provide examples of how to use the function.

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
import gams.transfer as gt
import seaborn as sns
from pathlib import Path
import geopandas as gpd
from matplotlib.ticker import MaxNLocator, FixedLocator
FUELS = 'static/fuels.csv'
TECHS = 'static/technologies.csv'
COLORS = 'static/colors.csv'
GEOJSON = 'static/countries.geojson'

NAME_COLUMNS = {
    'pFuelDispatch': 'fuel',
    'pDispatch': 'attribute',
    'pCostSummary': 'attribute',
    'pCapacityByFuel': 'fuel',
    'pEnergyByFuel': 'fuel'
}

UNIT = {
    'Capex: $m': 'M$/year',
    'Unmet demand costs: : $m': 'M$'
}


def create_folders_imgs(folder):
    """
    Creating folders for images
    
    Parameters:
    ----------
    folder: str
        Path to the folder where the images will be saved.
    
    """
    for p in [path for path in Path(folder).iterdir() if path.is_dir()]:
        if not (p / Path('images')).is_dir():
            os.mkdir(p / Path('images'))
    if not (Path(folder) / Path('images')).is_dir():
        os.mkdir(Path(folder) / Path('images'))


def read_plot_specs():
    """
    Read the specifications for the plots from the static files.
    
    Returns:
    -------
    dict_specs: dict
        Dictionary containing the specifications for the plots
    """

    colors = pd.read_csv(COLORS)
    fuel_mapping = pd.read_csv(FUELS)
    tech_mapping = pd.read_csv(TECHS)
    countries = gpd.read_file(GEOJSON)

    dict_specs = {
        'colors': colors.set_index('Processing')['Color'].to_dict(),
        'fuel_mapping': fuel_mapping.set_index('EPM_Fuel')['Processing'].to_dict(),
        'tech_mapping': tech_mapping.set_index('EPM_Tech')['Processing'].to_dict(),
        'map_countries': countries
    }
    return dict_specs


def extract_gdx(file):
    """
    Extract information as pandas DataFrame from a gdx file.

    Parameters
    ----------
    file: str
        Path to the gdx file

    Returns
    -------
    epm_result: dict
        Dictionary containing the extracted information
    """
    df = {}
    container = gt.Container(file)
    for param in container.getParameters():
        if container.data[param.name].records is not None:
            df[param.name] = container.data[param.name].records.copy()

    return df


def extract_epm_folder(results_folder, file='epmresults.gdx'):
    """
    Extract information from a folder containing multiple scenarios.
    
    Parameters
    ----------
    results_folder: str
        Path to the folder containing the scenarios
    file: str, optional, default='epmresults.gdx'
        Name of the gdx file to extract
        
    Returns
    -------
    inverted_dict: dict
        Dictionary containing the extracted information for each scenario
    """
    # Dictionary to store the extracted information for each scenario
    dict_df = {}
    for scenario in [i for i in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, i))]:
        if file in os.listdir(os.path.join(results_folder, scenario)):
            dict_df.update({scenario: extract_gdx(os.path.join(results_folder, scenario, file))})

    inverted_dict = {
        k: {outer: inner[k] for outer, inner in dict_df.items() if k in inner}
        for k in {key for inner in dict_df.values() for key in inner}
    }

    inverted_dict = {k: pd.concat(v, names=['scenario']).reset_index('scenario') for k, v in inverted_dict.items()}
    return inverted_dict


def standardize_names(dict_df, key, mapping, column='fuel'):
    """
    Standardize the names of fuels in the dataframes.

    Only works when dataframes have fuel and value (with numerical values) columns.
    
    Parameters
    ----------
    dict_df: dict
        Dictionary containing the dataframes
    key: str
        Key of the dictionary to modify
    mapping: dict
        Dictionary mapping the original fuel names to the standardized names
    column: str, optional, default='fuel'
        Name of the column containing the fuels
    """

    if key in dict_df.keys():
        temp = dict_df[key].copy()
        temp[column] = temp[column].replace(mapping)
        temp = temp.groupby([i for i in temp.columns if i != 'value']).sum().reset_index()

        new_fuels = [f for f in temp[column].unique() if f not in mapping.values()]
        if new_fuels:
            raise ValueError(f'New fuels found in {key}: {new_fuels}. '
                             f'Add fuels to the mapping in the /static folder and add in the colors.csv file.')

        dict_df[key] = temp.copy()
    else:
        print(f'{key} not found in epm_dict')


def process_epm_inputs(epm_input, dict_specs, scenarios_rename=None):
    """
    Processing EPM inputs to use in plots.  
    
    Parameters
    ----------
    epm_input: dict
        Dictionary containing the input data
    dict_specs: dict
        Dictionary containing the specifications for the plots
    """

    keys = ['ftfindex', 'pGenDataExcel', 'pTechDataExcel', 'pZoneIndex', 'pDemandProfile', 'pDemandForecast', 'pScalars']
    epm_input = {k: i for k, i in epm_input.items() if k in keys}
    if scenarios_rename is not None:
        for k, i in epm_input.items():
            if 'scenario' in i.columns:
                i['scenario'] = i['scenario'].replace(scenarios_rename)

    # t = epm_input['pGenDataExcel'].pivot(index=['scenario', 'Plants'], columns='uni', values='value')

    epm_input['pDemandForecast'].rename(columns={'uni_0': 'zone', 'uni_2': 'year'}, inplace=True)
    epm_input['pDemandForecast']['year'] = epm_input['pDemandForecast']['year'].astype(int)

    epm_input['ftfindex'].rename(columns={'uni_0': 'fuel1', 'uni_1': 'fuel2'}, inplace=True)
    mapping_fuel1 = epm_input['ftfindex'].loc[:, ['fuel1', 'value']].drop_duplicates().set_index(
        'value').squeeze().to_dict()
    mapping_fuel2 = epm_input['ftfindex'].loc[:, ['fuel2', 'value']].drop_duplicates().set_index(
        'value').squeeze().to_dict()

    temp = epm_input['pTechDataExcel']
    temp['uni'] = temp['uni'].astype(str)
    temp = temp[temp['uni'] == 'Assigned Value']
    mapping_tech = temp.loc[:, ['Abbreviation', 'value']].drop_duplicates().set_index(
        'value').squeeze()
    mapping_tech.replace(dict_specs['tech_mapping'], inplace=True)

    mapping_zone = epm_input['pZoneIndex'].loc[:, ['value', 'ZONE']].set_index('value').loc[:, 'ZONE'].to_dict()
    epm_input.update({'mapping_zone': mapping_zone})

    # Modify pGenDataExcel
    df = epm_input['pGenDataExcel'].pivot(index=['scenario', 'Plants'], columns='uni', values='value')
    df = df.loc[:, ['Type', 'fuel1']]
    df['fuel1'] = df['fuel1'].replace(mapping_fuel1)
    # df['fuel2'] = df['fuel2'].replace(mapping_fuel2)

    df['fuel1'] = df['fuel1'].replace(dict_specs['fuel_mapping'])
    # df['fuel2'] = df['fuel2'].replace(dict_specs['fuel_mapping'])

    df['Type'] = df['Type'].replace(mapping_tech)

    epm_input['pGenDataExcel'] = df.reset_index()

    epm_input['pGenDataExcel'].rename(columns={'Plants': 'generator'}, inplace=True)

    return epm_input


def remove_unused_tech(epm_dict, list_keys):
    """
    Remove rows that correspond to technologies that are never used across the whole time horizon

    Parameters
    ----------
    epm_dict: dict
        Dictionary containing the dataframes
    list_keys: list
        List of keys to remove unused technologies
        
    Returns
    -------
    epm_dict: dict
        Dictionary containing the dataframes with unused technologies removed
    """
    for key in list_keys:
        epm_dict[key] = epm_dict[key].where((epm_dict[key]['value'] > 2e-6) | (epm_dict[key]['value'] < -2e-6),np.nan)  # get rid of small values to avoid unneeded labels
        epm_dict[key] = epm_dict[key].dropna(subset=['value'])

    return epm_dict


def process_epm_results(epm_results, dict_specs, scenarios_rename=None, mapping_gen_fuel=None):
    """
    Processing EPM results to use in plots.
    
    Parameters
    ----------
    epm_results: dict
        Dictionary containing the results
    dict_specs: dict
        Dictionary containing the specifications for the plots
    scenarios_rename: dict, optional
        Dictionary mapping the original scenario names to the standardized names
    mapping_gen_fuel: pd.DataFrame, optional
        DataFrame containing the mapping between generators and fuels
        
    Returns
    -------
    epm_dict: dict
        Dictionary containing the processed results    
    """

    rename_columns = {'c': 'zone', 'country': 'zone', 'y': 'year', 'v': 'value', 's': 'scenario', 'uni': 'attribute',
                      'z': 'zone', 'g': 'generator', 'f': 'fuel', 'q': 'season', 'd': 'day', 't': 't'}

    keys = {'pDemandSupplyCountry', 'pDemandSupply', 'pEnergyByPlant', 'pEnergyByFuel', 'pCapacityByFuel', 'pCapacityPlan',
            'pPlantUtilization', 'pCostSummary', 'pCostSummaryCountry', 'pEmissions', 'pPrice', 'pHourlyFlow',
            'pDispatch', 'pFuelDispatch', 'pPlantFuelDispatch', 'pInterconUtilization',
            'pSpinningReserveByPlantCountry', 'InterconUtilization', 'pInterchange', 'Interchange', 'interchanges', 'pInterconUtilizationExt',
            'InterconUtilizationExt', 'pInterchangeExt', 'InterchangeExt', 'annual_line_capa', 'pAnnualTransmissionCapacity',
            'AdditiononalCapacity_trans', 'pDemandSupplySeason', 'pCurtailedVRET', 'pCurtailedStoHY',
            'pNewCapacityFuelCountry', 'pPlantAnnualLCOE', 'pStorageComponents', 'pNPVByYear',
            'pSpinningReserveByPlantCountry', 'pPlantDispatch'}

    rename_keys = {'pSpinningReserveByPlantCountry': 'pReserveByPlant'}

    # Rename columns
    epm_dict = {k: i.rename(columns=rename_columns) for k, i in epm_results.items() if k in keys and k in epm_results.keys()}
    epm_dict.update({rename_keys[k]: i for k, i in epm_dict.items() if k in rename_keys.keys()})

    if scenarios_rename is not None:
        for k, i in epm_dict.items():
            if 'scenario' in i.columns:
                i['scenario'] = i['scenario'].replace(scenarios_rename)

    list_keys = ['pSpinningReserveByPlantCountry', 'pPlantReserve', 'pCapacityPlan']
    list_keys = [i for i in list_keys if i in epm_dict.keys()]
    epm_dict = remove_unused_tech(epm_dict, list_keys)

    # Convert columns to the right type
    for k, i in epm_dict.items():
        if 'year' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'year': 'int'})
        if 'value' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'value': 'float'})
        if 'zone' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'zone': 'str'})
        if 'country' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'country': 'str'})
        if 'generator' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'generator': 'str'})
        if 'fuel' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'fuel': 'str'})
        if 'scenario' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'scenario': 'str'})
        if 'attribute' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'attribute': 'str'})

    # Standardize names
    standardize_names(epm_dict, 'pEnergyByFuel', dict_specs['fuel_mapping'])
    standardize_names(epm_dict, 'pCapacityByFuel', dict_specs['fuel_mapping'])
    standardize_names(epm_dict, 'pNewCapacityFuelCountry', dict_specs['fuel_mapping'])

    standardize_names(epm_dict, 'pFuelDispatch', dict_specs['fuel_mapping'])
    standardize_names(epm_dict, 'pPlantFuelDispatch', dict_specs['tech_mapping'])

    # Add fuel type to the results
    if mapping_gen_fuel is not None:
        # Add fuel type to the results
        plant_result = ['pSpinningReserveByPlantCountry', 'pPlantAnnualLCOE', 'pEnergyByPlant', 'pCapacityPlan', 'pPlantDispatch']
        for key in [k for k in plant_result if k in epm_dict.keys()]:
            epm_dict[key] = epm_dict[key].merge(mapping_gen_fuel, on=['scenario', 'generator'], how='left')

    return epm_dict


def format_ax(ax, linewidth=True):
    """
    Format the axis of a plot.
    

    Parameters:
    ----------
    ax: plt.Axes
        Axis to format
    linewidth: bool, optional, default=True
        If True, set the linewidth of the spines to 1
    """
    # Remove the background
    ax.set_facecolor('none')

    # Remove grid lines
    ax.grid(False)

    # Optionally, make spines more prominent (if needed)
    if linewidth:
        for spine in ax.spines.values():
            spine.set_color('black')  # Ensure spines are visible
            spine.set_linewidth(1)  # Adjust spine thickness

    # Remove ticks if necessary (optional, can comment out)
    ax.tick_params(top=False, right=False, left=True, bottom=True, direction='in', width=0.8)

    # Ensure the entire frame is displayed
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)


def calculate_total_system_cost(data, discount_rate, start_year=None, end_year=None):
    """
    Calculate the total system cost with interpolation and discounting.

    Parameters:
    ----------
    data: pd.DataFrame
        DataFrame with columns ['Scenario', 'Year', 'Cost']
    discount_rate: float
        Discount rate (e.g., 0.05 for 5%)
    start_year: int
        Starting year for cost calculation
    end_year: int
        Ending year for cost calculation

    Returns:
    -------
        pd.Series: Total system cost for each scenario
    """
    if start_year is None:
        start_year = data['year'].min()

    if end_year is None:
        end_year = data['year'].max()

    results = []

    # Group data by scenario
    for scenario, group in data.groupby('scenario'):
        # Sort data by year
        group = group.sort_values(by='year')

        # Create a full range of years
        years = np.arange(start_year, end_year + 1)

        # Interpolate costs
        interpolated_costs = np.interp(years, group['year'], group['value'])

        # Discount costs
        discounted_costs = [
            cost / ((1 + discount_rate) ** (year - start_year))
            for year, cost in zip(years, interpolated_costs)
        ]

        # Sum up discounted costs
        total_cost = sum(discounted_costs)

        # Store result
        results.append({'scenario': scenario, 'cost': total_cost})

    return pd.DataFrame(results).set_index('scenario').squeeze()


def line_plot(df, x, y, xlabel=None, ylabel=None, title=None, filename=None, figsize=(10, 6)):
    """Makes a line plot.

    Parameters:
    ----------
    df: pd.DataFrame
    x: str
        Column name for x-axis
    y: str
        Column name for y-axis
    xlabel: str, optional
        Label for x-axis
    ylabel: str, optional
        Label for y-axis
    title: str, optional
        Title of the plot
    filename: str, optional
        Path to save the plot
    figsize: tuple, optional, default=(10, 6)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[x], df[y])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if filename is not None:
        plt.savefig(filename)
        plt.close()

    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)
    return None


def bar_plot(df, x, y, xlabel=None, ylabel=None, title=None, filename=None, figsize=(8, 5)):
    """Makes a bar plot.

    Parameters:
    ----------
    df: pd.DataFrame
    x: str
        Column name for x-axis
    y: str
        Column name for y-axis
    xlabel: str, optional
        Label for x-axis
    ylabel: str, optional
        Label for y-axis
    title: str, optional
        Title of the plot
    filename: str, optional
        Path to save the plot
    figsize: tuple, optional, default=(10, 6)
    """
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(df[x], df[y], width=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:,.0f}', va='bottom', ha='center')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)
    return None


def make_demand_plot(pDemandSupplyCountry, folder, years=None, plot_option='bar', selected_scenario=None, unit='MWh'):
    """
    Depreciated. Makes a plot of demand for all countries.
    
    Parameters:
    ----------
    pDemandSupplyCountry: pd.DataFrame
        Contains demand data for all countries
    folder: str
        Path to folder where the plot will be saved
    years: list, optional
        List of years to include in the plot
    plot_option: str, optional, default='bar'
        Type of plot. Choose between 'line' and 'bar'
    selected_scenario: str, optional
        Name of the scenario
    unit: str, optional, default='GWh'
        Unit of the demand. Choose between 'GWh' and 'TWh'
    """
    # TODO: add scenario grouping, currently only works when selected_scenario is not None

    df_tot = pDemandSupplyCountry.loc[pDemandSupplyCountry['attribute'] == 'Demand: GWh']
    if selected_scenario is not None:
        df_tot = df_tot[df_tot['scenario'] == selected_scenario]
    df_tot = df_tot.groupby(['year']).agg({'value': 'sum'}).reset_index()

    if unit == 'TWh':
        df_tot['value'] = df_tot['value'] / 1000
    elif unit == '000 TWh':
        df_tot['value'] = df_tot['value'] / 1000000

    if years is not None:
        df_tot = df_tot.loc[df_tot['year'].isin(years)]

    if plot_option == 'line':
        line_plot(df_tot, 'year', 'value',
                       xlabel='Years',
                       ylabel=f'Demand {unit}',
                       title=f'Total demand - {selected_scenario} scenario',
                       filename=f'{folder}/TotalDemand_{plot_option}_{selected_scenario}.png')
    elif plot_option == 'bar':
        bar_plot(df_tot, 'year', 'value',
                      xlabel='Years',
                      ylabel=f'Demand {unit}',
                      title=f'Total demand - {selected_scenario} scenario',
                      filename=f'{folder}/TotalDemand_{plot_option}_{selected_scenario}.png')
    else:
        raise ValueError('Invalid plot_option argument. Choose between "line" and "bar"')


def make_generation_plot(pEnergyByFuel, folder, years=None, plot_option='bar', selected_scenario=None, unit='GWh',
                         BESS_included=True, Hydro_stor_included=True):
    """
    Makes a plot of demand for all countries.

    Parameters:
    ----------
    pDemandSupplyCountry: pd.DataFrame
        Contains demand data for all countries
    folder: str
        Path to folder where the plot will be saved
    years: list, optional
        List of years to include in the plot
    plot_option: str, optional, default='bar'
        Type of plot. Choose between 'line' and 'bar'
    selected_scenario: str, optional
        Name of the scenario
    unit: str, optional, default='GWh'
        Unit of the demand. Choose between 'GWh' and 'TWh'
    """
    # TODO: add scenario grouping, currently only works when selected_scenario is not None
    if selected_scenario is not None:
        pEnergyByFuel = pEnergyByFuel[pEnergyByFuel['scenario'] == selected_scenario]

    if not BESS_included:
        pEnergyByFuel = pEnergyByFuel[pEnergyByFuel['fuel'] != 'Battery Storage']

    if not Hydro_stor_included:
        pEnergyByFuel = pEnergyByFuel[pEnergyByFuel['fuel'] != 'Pumped-Hydro Storage']

    df_tot = pEnergyByFuel.groupby('year').agg({'value': 'sum'}).reset_index()

    if unit == 'TWh':
        df_tot['value'] = df_tot['value'] / 1000
    elif unit == '000 TWh':
        df_tot['value'] = df_tot['value'] / 1000000

    if years is not None:
        df_tot = df_tot.loc[df_tot['year'].isin(years)]

    if plot_option == 'line':
        line_plot(df_tot, 'year', 'value',
                       xlabel='Years',
                       ylabel=f'Generation {unit}',
                       title=f'Total generation - {selected_scenario} scenario',
                       filename=f'{folder}/TotalGeneration_{plot_option}_{selected_scenario}.png')
    elif plot_option == 'bar':
        bar_plot(df_tot, 'year', 'value',
                      xlabel='Years',
                      ylabel=f'Generation {unit}',
                      title=f'Total generation - {selected_scenario} scenario',
                      filename=f'{folder}/TotalGeneration_{plot_option}_{selected_scenario}.png')
    else:
        raise ValueError('Invalid plot_option argument. Choose between "line" and "bar"')


def subplot_pie(df, index, dict_colors, subplot_column=None, title='', figsize=(16, 4),
                percent_cap=1, filename=None, rename=None, bbox_to_anchor=(0.5, -0.1), loc='lower center'):
    """
    Creates pie charts for data grouped by a column, or a single pie chart if no grouping is specified.

    Parameters:
    ----------
    df: pd.DataFrame
        DataFrame containing the data
    index: str
        Column to use for the pie chart
    dict_colors: dict
        Dictionary mapping the index values to colors
    subplot_column: str, optional
        Column to use for subplots. If None, a single pie chart is created.
    title: str, optional
        Title of the plot
    figsize: tuple, optional, default=(16, 4)
        Size of the figure
    percent_cap: float, optional, default=1
        Minimum percentage to show in the pie chart
    filename: str, optional
        Path to save the plot
    bbox_to_anchor: tuple
        Position of the legend compared to the figure
    loc: str
        Localization of the legend
    """
    if rename is not None:
        df[index] = df[index].replace(rename)
    if subplot_column is not None:
        # Group by the column for subplots
        groups = df.groupby(subplot_column)

        # Calculate the number of subplots
        num_subplots = len(groups)
        ncols = min(3, num_subplots)  # Limit to 3 columns per row
        nrows = int(np.ceil(num_subplots / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0], figsize[1]*nrows))
        axes = np.array(axes).flatten()  # Ensure axes is iterable 1D array


        all_labels = set()  # Collect all labels for the combined legend
        for ax, (name, group) in zip(axes, groups):
            colors = [dict_colors[f] for f in group[index]]
            handles, labels = plot_pie_on_ax(ax, group, index, percent_cap, colors, title=f"{title} - {subplot_column}: {name}")
            all_labels.update(group[index])  # Collect unique labels

        # Hide unused subplots
        for j in range(len(groups), len(axes)):
            fig.delaxes(axes[j])

        # Create a shared legend below the graphs
        all_labels = sorted(all_labels)  # Sort labels for consistency
        handles = [plt.Line2D([0], [0], marker='o', color=dict_colors[label], linestyle='', markersize=10)
                   for label in all_labels]
        fig.legend(
            handles,
            all_labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=1,  # Adjust number of columns based on subplots
            frameon=False, fontsize=16
        )

        # Add title for the whole figure
        fig.suptitle(title, fontsize=16)

    else:  # Create a single pie chart if no subplot column is specified
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [dict_colors[f] for f in df[index]]
        handles, labels = plot_pie_on_ax(ax, df, index, percent_cap, colors, title)

    # Save the figure if filename is provided
    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pie_on_ax(ax, df, index, percent_cap, colors, title, radius=None, annotation_size=8):
    """Pie plot on a single axis."""
    if radius is not None:
        df.plot.pie(
            ax=ax,
            y='value',
            autopct=lambda p: f'{p:.0f}%' if p > percent_cap else '',
            startangle=140,
            legend=False,
            colors=colors,
            labels=None,
            radius=radius
        )
    else:
        df.plot.pie(
            ax=ax,
            y='value',
            autopct=lambda p: f'{p:.0f}%' if p > percent_cap else '',
            startangle=140,
            legend=False,
            colors=colors,
            labels=None
        )
    ax.set_ylabel('')
    ax.set_title(title)

    # Adjust annotation font sizes
    for text in ax.texts:
        if text.get_text().endswith('%'):  # Check if the text is a percentage annotation
            text.set_fontsize(annotation_size)

    # Generate legend handles and labels manually
    handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, df[index])]
    labels = list(df[index])
    return handles, labels


def stacked_area_plot(df, filename, dict_colors=None, x_column='year', y_column='value', stack_column='fuel',
                      df_2=None, title='', x_label='Years', y_label='',
                      legend_title='', y2_label='', figsize=(10, 6), selected_scenario=None,
                      annotate=None):
    """
    Generate a stacked area chart.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    filename : str
        Path to save the plot.
    dict_colors : dict, optional
        Dictionary mapping fuel types to colors.
    x_column : str, default 'year'
        Column for x-axis.
    y_column : str, default 'value'
        Column for y-axis.
    stack_column : str, default 'fuel'
        Column for stacking.
    legend_title : str
        Title for the legend.
    df_2 : pd.DataFrame, optional
        DataFrame containing data for the secondary y-axis.
    title : str
        Title of the plot.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the primary y-axis.
    y2_label : str
        Label for the secondary y-axis.
    figsize : tuple, default (10, 6)
        Size of the figure.
    selected_scenario : str, optional
        Name of the scenario.
    annotate : dict, optional
        Dictionary containing the annotations.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    if selected_scenario is not None:
        df = df[df['scenario'] == selected_scenario]

    # Plot stacked area for generation
    temp = df.groupby([x_column, stack_column])[y_column].sum().unstack(stack_column)
    temp.plot.area(ax=ax1, stacked=True, alpha=0.8, color=dict_colors)

    if annotate is not None:
        for key, value in annotate.items():
            x = key - 2
            y = temp.loc[key].sum() / 2
            ax1.annotate(value, xy=(x, y), xytext=(x, y * 1.1))

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    format_ax(ax1)
    
    years = temp.index  # Assuming the x-axis data corresponds to years
    ticks = [year for year in years if year % 5 == 0]
    ax1.xaxis.set_major_locator(FixedLocator(ticks))

    # Secondary y-axis
    if df_2 is not None:
        # Remove legend ax1
        ax1.get_legend().remove()

        temp = df_2.groupby([x_column])[y_column].sum()
        ax2 = ax1.twinx()
        line, = ax2.plot(temp.index, temp, color='brown', label=y2_label)
        ax2.set_ylabel(y2_label, color='brown')
        format_ax(ax2, linewidth=False)

        # Combine legends for ax1 and ax2
        handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()  # Collect from ax1
        handles_ax2, labels_ax2 = [line], [y2_label]  # Collect from ax2
        handles = handles_ax1 + handles_ax2  # Combine handles
        labels = labels_ax1 + labels_ax2  # Combine labels
        fig.legend(
            handles,
            labels,
            loc='center left',
            bbox_to_anchor=(1.1, 0.5),  # Right side, centered vertically
            frameon=False,
        )
    else:
        ax1.legend(
            loc='center left',
            bbox_to_anchor=(1.1, 0.5),  # Right side, centered vertically
            title=legend_title,
            frameon=False,
        )
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


def format_dispatch_ax(ax, pd_index):

    # Adding the representative days and seasons
    n_rep_days = len(pd_index.get_level_values('day').unique())
    dispatch_seasons = pd_index.get_level_values('season').unique()
    total_days = len(dispatch_seasons) * n_rep_days
    y_max = ax.get_ylim()[1]

    for d in range(total_days):
        x_d = 24 * d

        # Add vertical lines to separate days
        is_end_of_season = d % n_rep_days == 0
        linestyle = '-' if is_end_of_season else '--'
        ax.axvline(x=x_d, color='slategrey', linestyle=linestyle, linewidth=0.8)

        # Add day labels (d1, d2, ...)
        ax.text(
            x=x_d + 12,  # Center of the day (24 hours per day)
            y=y_max * 0.99,
            s=f'd{(d % n_rep_days) + 1}',
            ha='center',
            fontsize=7
        )

    # Add season labels
    season_x_positions = [24 * n_rep_days * s + 12 * n_rep_days for s in range(len(dispatch_seasons))]
    ax.set_xticks(season_x_positions)
    ax.set_xticklabels(dispatch_seasons, fontsize=8)
    ax.set_xlim(left=0, right=24 * total_days)
    ax.set_xlabel('Time')
    # Remove grid
    ax.grid(False)
    # Remove top spine to let days appear
    ax.spines['top'].set_visible(False)


def dispatch_plot(df_area=None, filename=None, dict_colors=None, df_line=None, figsize=(10, 6), legend_loc='bottom', bottom=0):
    """
    Generate and display or save a dispatch plot with area and line plots.
    
    
    Parameters
    ----------
    df_area : pandas.DataFrame, optional
        DataFrame containing data for the area plot. If provided, the area plot will be stacked.
    filename : str, optional
        Path to save the plot image. If not provided, the plot will be displayed.
    dict_colors : dict, optional
        Dictionary mapping column names to colors for the plot.
    df_line : pandas.DataFrame, optional
        DataFrame containing data for the line plot. If provided, the line plot will be overlaid on the area plot.
    figsize : tuple, default (10, 6)
        Size of the figure in inches.
    legend_loc : str, default 'bottom'
        Location of the legend. Options are 'bottom' or 'right'.
    ymin : int or float, default 0
        Minimum value for the y-axis.
    Raises
    ------
    ValueError
        If neither `df_area` nor `df_line` is provided.
    Notes
    -----
    The function will raise an assertion error if `df_area` and `df_line` are provided but do not share the same index.
    
    Examples
    --------
    >>> dispatch_plot(df_area=df_area, df_line=df_line, dict_colors=color_dict, filename='dispatch_plot.png')
    """    

    fig, ax = plt.subplots(figsize=figsize)

    if df_area is not None:
        df_area.plot.area(ax=ax, stacked=True, color=dict_colors, linewidth=0)
        pd_index = df_area.index
    if df_line is not None:
        if df_area is not None:
            assert df_area.index.equals(
                df_line.index), 'Dataframes used for area and line do not share the same index. Update the input dataframes.'
        df_line.plot(ax=ax, color=dict_colors)
        pd_index = df_line.index

    if (df_area is None) and (df_line is None):
        raise ValueError('No dataframes provided for the plot. Please provide at least one dataframe.')

    format_dispatch_ax(ax, pd_index)

    # Add axis labels and title
    ax.set_ylabel('Generation (MWh)', fontweight='bold')
    # ax.text(0, 1.2, f'Dispatch', fontsize=9, fontweight='bold', transform=ax.transAxes)
    # set ymin to 0
    if bottom is not None:
        ax.set_ylim(bottom=bottom)

    # Add legend bottom center
    if legend_loc == 'bottom':
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(df_area.columns), frameon=False)
    elif legend_loc == 'right':
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=1, frameon=False)

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def select_time_period(df, select_time):
    """Select a specific time period in a dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Columns contain season and day
    select_time: dict
        For each key, specifies a subset of the dataframe
        
    Returns
    -------
    pd.DataFrame: Dataframe with the selected time period
    str: String with the selected time period
    """
    temp = ''
    if 'season' in select_time.keys():
        df = df.loc[df.season.isin(select_time['season'])]
        temp += '_'.join(select_time['season'])
    if 'day' in select_time.keys():
        df = df.loc[df.day.isin(select_time['day'])]
        temp += '_'.join(select_time['day'])
    return df, temp


def clean_dataframe(df, zone, year, scenario, column_stacked, fuel_grouping=None, select_time=None):
    """
    Transforms a dataframe from the results GDX into a dataframe with season, day, and time as the index, and format ready for plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the results.
    zone : str
        The zone to filter the data for.
    year : int
        The year to filter the data for.
    scenario : str
        The scenario to filter the data for.
    column_stacked : str
        Column to use for stacking values in the transformed dataframe.
    fuel_grouping : dict, optional
        A dictionary mapping fuels to their respective groups and to sum values over those groups.
    select_time : dict or None, optional
        Specific time filter to apply (e.g., "summer").

    Returns
    -------
    pd.DataFrame
        A transformed dataframe with multi-level index (season, day, time).

    Example
    -------
    df = epm_dict['FuelDispatch']
    column_stacked = 'fuel'
    select_time = {'season': ['m1'], 'day': ['d21', 'd22', 'd23', 'd24', 'd25', 'd26', 'd27', 'd28', 'd29', 'd30']}
    df = clean_dataframe(df, zone='Liberia', year=2025, scenario='Baseline', column_stacked='fuel', fuel_grouping=None, select_time=select_time)
    """
    if 'zone' in df.columns:
        df = df[(df['zone'] == zone) & (df['year'] == year) & (df['scenario'] == scenario)]
        df = df.drop(columns=['zone', 'year', 'scenario'])
    else:
        df = df[(df['year'] == year) & (df['scenario'] == scenario)]
        df = df.drop(columns=['year', 'scenario'])

    if column_stacked == 'fuel':
        if fuel_grouping is not None:
            df['fuel'] = df['fuel'].replace(
                fuel_grouping)  # case-specific, according to level of preciseness for dispatch plot

    df = (df.groupby(['season', 'day', 't', column_stacked], observed=False).sum().reset_index())

    if select_time is not None:
        df, temp = select_time_period(df, select_time)
    else:
        temp = None

    df = df.set_index(['season', 'day', 't', column_stacked]).unstack(column_stacked)
    return df, temp


def remove_na_values(df):
    """Removes na values from a dataframe, to avoind unnecessary labels in plots."""
    df = df.where((df > 1e-6) | (df < -1e-6),
                                    np.nan)
    df = df.dropna(axis=1, how='all')
    return df


def make_complete_fuel_dispatch_plot(dfs_area, dfs_line, dict_colors, zone, year, scenario,
                                    filename=None, fuel_grouping=None, select_time=None, reorder_dispatch=None,
                                    legend_loc='bottom', bottom=0):
    """
    Generates and saves a fuel dispatch plot, including only generation plants.

    Parameters
    ----------
    dfs_area : dict
        Dictionary containing dataframes for area plots.
    dfs_line : dict
        Dictionary containing dataframes for line plots.
    graph_folder : str
        Path to the folder where the plot will be saved.
    dict_colors : dict
        Dictionary mapping fuel types to colors.
    fuel_grouping : dict
        Mapping to create aggregate fuel categories, e.g.,
        {'Battery Storage 4h': 'Battery Storage'}.
    select_time : dict
        Time selection parameters for filtering the data.
    dfs_line_2 : dict, optional
        Optional dictionary containing dataframes for a secondary line plot.

    Returns
    -------
    None

    Example
    -------
    Generate and save a fuel dispatch plot:
    dfs_to_plot_area = {
        'pFuelDispatch': epm_dict['pFuelDispatch'],
        'pCurtailedVRET': epm_dict['pCurtailedVRET'],
        'pDispatch': subset_dispatch
    }
    subset_demand = epm_dict['pDispatch'].loc[epm_dict['pDispatch'].attribute.isin(['Demand'])]
    dfs_to_plot_line = {
        'pDispatch': subset_demand
    }
    fuel_grouping = {
        'Battery Storage 4h': 'Battery Discharge',
        'Battery Storage 8h': 'Battery Discharge',
        'Battery Storage 2h': 'Battery Discharge',
        'Battery Storage 3h': 'Battery Discharge',
    }
    make_complete_fuel_dispatch_plot(dfs_to_plot_area, dfs_to_plot_line, folder_results / Path('images'), dict_specs['colors'],
                                 zone='Liberia', year=2030, scenario=scenario, fuel_grouping=fuel_grouping,
                                 select_time=select_time, reorder_dispatch=['MtCoffee', 'Oil', 'Solar'], season=False)
    """
    # TODO: Add ax2 to show other data. For example prices would be interesting to show in the same plot.

    tmp_concat_area = []
    for key in dfs_area:
        df = dfs_area[key]
        column_stacked = NAME_COLUMNS[key]
        df, temp = clean_dataframe(df, zone, year, scenario, column_stacked, fuel_grouping=fuel_grouping, select_time=select_time)
        tmp_concat_area.append(df)

    tmp_concat_line = []
    for key in dfs_line:
        df = dfs_line[key]
        column_stacked = NAME_COLUMNS[key]
        df, temp = clean_dataframe(df, zone, year, scenario, column_stacked, fuel_grouping=fuel_grouping, select_time=select_time)
        tmp_concat_line.append(df)

    df_tot_area = pd.concat(tmp_concat_area, axis=1)
    df_tot_area = df_tot_area.droplevel(0, axis=1)
    df_tot_area = remove_na_values(df_tot_area)

    df_tot_line = pd.concat(tmp_concat_line, axis=1)
    df_tot_line = df_tot_line.droplevel(0, axis=1)
    df_tot_line = remove_na_values(df_tot_line)

    if reorder_dispatch is not None:
        new_order = [col for col in reorder_dispatch if col in df_tot_area.columns] + [col for col in df_tot_area.columns if col not in reorder_dispatch]
        df_tot_area = df_tot_area[new_order]

    if select_time is None:
        temp = 'all'
    temp = f'{year}_{temp}'
    if filename is not None:
        filename = filename.split('.png')[0] + f'_{temp}.png'

    dispatch_plot(df_tot_area, filename, df_line=df_tot_line, dict_colors=dict_colors, legend_loc=legend_loc, bottom=bottom)


def stacked_bar_subplot(df, column_group, filename, dict_colors=None, year_ini=None,order_scenarios=None, order_columns=None,
                        dict_scenarios=None, rotation=0, fonttick=14, legend=True, format_y=lambda y, _: '{:.0f} GW'.format(y),
                        cap=6, annotate=True, show_total=False, title=None):
    """
    Create a stacked bar subplot from a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    column_group : str
        Column name to group by for the stacked bars.
    filename : str
        Path to save the plot image. If None, the plot is shown instead.
    dict_colors : dict, optional
        Dictionary mapping column names to colors for the bars. Default is None.
    figsize : tuple, optional
        Size of the figure (width, height). Default is (10, 6).
    year_ini : str, optional
        Initial year to highlight in the plot. Default is None.
    order_scenarios : list, optional
        List of scenario names to order the bars. Default is None.
    order_columns : list, optional
        List of column names to order the stacked bars. Default is None.
    dict_scenarios : dict, optional
        Dictionary mapping scenario names to new names for the plot. Default is None.
    rotation : int, optional
        Rotation angle for x-axis labels. Default is 0.
    fonttick : int, optional
        Font size for tick labels. Default is 14.
    legend : bool, optional
        Whether to display the legend. Default is True.
    format_y : function, optional
        Function to format y-axis labels. Default is a lambda function formatting as '{:.0f} GW'.
    cap : int, optional
        Minimum height of bars to annotate. Default is 6.
    annotate : bool, optional
        Whether to annotate each bar with its height. Default is True.
    show_total : bool, optional
        Whether to show the total value on top of each bar. Default is False.
    Returns
    -------
    None
    """
    
    list_keys = list(df.columns)
    n_scenario = df.index.get_level_values([i for i in df.index.names if i != column_group][0]).unique()
    num_subplots = int(len(list_keys))
    n_columns = min(3, num_subplots)  # Limit to 3 columns per row
    n_rows = int(np.ceil(num_subplots / n_columns))
    if year_ini is not None:
        width_ratios = [1] + [len(n_scenario)] * (n_columns - 1)
    else:
        width_ratios = [1] * n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(10, 6*n_rows), sharey='all',
                             gridspec_kw={'width_ratios': width_ratios})
    axes = np.array(axes).flatten()

    handles, labels = None, None
    for k, key in enumerate(list_keys):
        ax = axes[k]

        try:
            df_temp = df[key].unstack(column_group)

            if key == year_ini:
                df_temp = df_temp.iloc[0, :]
                df_temp = df_temp.to_frame().T
                df_temp.index = ['Initial']
            else:
                if dict_scenarios is not None:  # Renaming scenarios for plots
                    df_temp.index = df_temp.index.map(lambda x: dict_scenarios.get(x, x))
                if order_scenarios is not None:  # Reordering scenarios
                    df_temp = df_temp.loc[[c for c in order_scenarios if c in df_temp.index], :]
                if order_columns is not None:
                    new_order = [c for c in order_columns if c in df_temp.columns] + [c for c in df_temp.columns if c not in order_columns]
                    df_temp = df_temp.loc[:,new_order]

            df_temp.plot(ax=ax, kind='bar', stacked=True, linewidth=0,
                         color=dict_colors if dict_colors is not None else None)

            # Annotate each bar
            if annotate:
                for container in ax.containers:
                    for bar in container:
                        height = bar.get_height()
                        if height > cap:  # Only annotate bars with a height
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,  # X position: center of the bar
                                bar.get_y() + height / 2,  # Y position: middle of the bar
                                f"{height:.1f}",  # Annotation text (formatted value)
                                ha="center", va="center",  # Center align the text
                                fontsize=10, color="black"  # Font size and color
                            )

            if show_total:
                df_total = df_temp.sum(axis=1)
                for x, y in zip(df_temp.index, df_total.values):
                    # Put the total at the y-position equal to the total
                    ax.text(x, y * (1 + 0.02), f"{y:,.0f}", ha='center', va='bottom', fontsize=10,
                            color='black', fontweight='bold')
                ax.scatter(df_temp.index, df_total, color='black', s=20)

            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            # put tick label in bold
            ax.tick_params(axis='both', which=u'both', length=0)
            ax.set_xlabel('')

            if len(list_keys) > 1:
                title = key
                if isinstance(key, tuple):
                    title = '{}-{}'.format(key[0], key[1])
                ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)
            else:
                if title is not None:
                    if isinstance(title, tuple):
                        title = '{}-{}'.format(title[0], title[1])
                    ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)


            if k == 0:
                handles, labels = ax.get_legend_handles_labels()
                labels = [l.replace('_', ' ') for l in labels]
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
            if k > 0:
                ax.set_ylabel('')
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.get_legend().remove()

            # Add a horizontal line at 0
            ax.axhline(0, color='black', linewidth=0.5)

        except IndexError:
            ax.axis('off')

        if legend:
            fig.legend(handles[::-1], labels[::-1], loc='center left', frameon=False, ncol=1,
                       bbox_to_anchor=(1, 0.5))

    # Hide unused subplots
    for j in range(k + 1, len(axes)):
        fig.delaxes(axes[j])

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()



def make_stacked_bar_subplots(df, filename, dict_colors, selected_zone=None, selected_year=None, column_xaxis='year',
                              column_stacked='fuel', column_multiple_bars='scenario',
                              column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None, dict_scenarios=None,
                              format_y=lambda y, _: '{:.0f} MW'.format(y), order_stacked=None, cap=2, annotate=True,
                              show_total=False, fonttick=12, rotation=0, title=None):
    """
    Subplots with stacked bars. Can be used to explore the evolution of capacity over time and across scenarios.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with results.
    filename : str
        Path to save the figure.
    dict_colors : dict
        Dictionary with color arguments.
    selected_zone : str
        Zone to select.
    column_xaxis : str
        Column for choosing the subplots.
    column_stacked : str
        Column name for choosing the column to stack values.
    column_multiple_bars : str
        Column for choosing the type of bars inside a given subplot.
    column_value : str
        Column name for the values to be plotted.
    select_xaxis : list, optional
        Select a subset of subplots (e.g., a number of years).
    dict_grouping : dict, optional
        Dictionary for grouping variables and summing over a given group.
    order_scenarios : list, optional
        Order of scenarios for plotting.
    dict_scenarios : dict, optional
        Dictionary for renaming scenarios.
    format_y : function, optional
        Function for formatting y-axis labels.
    order_stacked : list, optional
        Reordering the variables that will be stacked.
    cap : int, optional
        Under this cap, no annotation will be displayed.
    annotate : bool, optional
        Whether to annotate the bars.
    show_total : bool, optional
        Whether to show the total value on top of each bar.

    Example
    -------
    Stacked bar subplots for capacity (by fuel) evolution:
    filename = Path(RESULTS_FOLDER) / Path('images') / Path('CapacityEvolution.png')
    fuel_grouping = {
        'Battery Storage 4h': 'Battery',
        'Battery Storage 8h': 'Battery',
        'Hydro RoR': 'Hydro',
        'Hydro Storage': 'Hydro'
    }
    scenario_names = {
        'baseline': 'Baseline',
        'HydroHigh': 'High Hydro',
        'DemandHigh': 'High Demand',
        'LowImport_LowThermal': 'LowImport_LowThermal'
    }
    make_stacked_bar_subplots(epm_dict['pCapacityByFuel'], filename, dict_specs['colors'], selected_zone='Liberia',
                              select_xaxis=[2025, 2028, 2030], dict_grouping=fuel_grouping, dict_scenarios=scenario_names,
                              order_scenarios=['Baseline', 'High Hydro', 'High Demand', 'LowImport_LowThermal'],
                              format_y=lambda y, _: '{:.0f} MW'.format(y))

    Stacked bar subplots for reserve evolution:
    filename = Path(RESULTS_FOLDER) / Path('images') / Path('ReserveEvolution.png')
    make_stacked_bar_subplots(epm_dict['pReserveByPlant'], filename, dict_colors=dict_specs['colors'], selected_zone='Liberia',
                              column_xaxis='year', column_stacked='fuel', column_multiple_bars='scenario',
                              select_xaxis=[2025, 2028, 2030], dict_grouping=dict_grouping, dict_scenarios=scenario_names,
                              order_scenarios=['Baseline', 'High Hydro', 'High Demand', 'LowImport_LowThermal'],
                              format_y=lambda y, _: '{:.0f} GWh'.format(y),
                              order_stacked=['Hydro', 'Oil'], cap=2)
    """
    if selected_zone is not None:
        df = df[(df['zone'] == selected_zone)]
        df = df.drop(columns=['zone'])

    if selected_year is not None:
        df = df[(df['year'] == selected_year)]
        df = df.drop(columns=['year'])

    if dict_grouping is not None:
        for key, grouping in dict_grouping.items():
            assert key in df.columns, f'Grouping parameter with key {key} is used but {key} is not in the columns.'
            df[key] = df[key].replace(grouping)  # case-specific, according to level of preciseness for dispatch plot

    if column_xaxis is not None:
        df = (df.groupby([column_xaxis, column_stacked, column_multiple_bars], observed=False)[column_value].sum().reset_index())
        df = df.set_index([column_stacked, column_multiple_bars, column_xaxis]).squeeze().unstack(column_xaxis)
    else:  # no subplots in this case
        df = (df.groupby([column_stacked, column_multiple_bars], observed=False)[column_value].sum().reset_index())
        df = df.set_index([column_stacked, column_multiple_bars])

    if select_xaxis is not None:
        df = df.loc[:, [i for i in df.columns if i in select_xaxis]]

    stacked_bar_subplot(df, column_stacked, filename, dict_colors, format_y=format_y,
                        rotation=rotation, order_scenarios=order_scenarios, dict_scenarios=dict_scenarios,
                        order_columns=order_stacked, cap=cap, annotate=annotate, show_total=show_total, fonttick=fonttick, title=title)


def scatter_plot_with_colors(df, column_xaxis, column_yaxis, column_color, color_dict, ymax=None, xmax=None, title='',
                             legend=None, filename=None, size_scale=None, annotate_thresh=None):
    """
    Creates a scatter plot with points colored based on the values in a specific column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    column_xaxis : str
        Column name for x-axis values.
    column_yaxis : str
        Column name for y-axis values.
    column_color : str
        Column name for categorical values determining color.
    color_dict : dict
        Dictionary mapping values in column_color to specific colors.
    size_proportional : bool, optional
        Whether to size points proportionally to x-axis values.
    size_scale : float, optional
        Scaling factor for point sizes if size_proportional is True.
    ymax : float, optional
        Maximum y-axis value.
    title : str, optional
        Title of the plot.
    legend_title : str, optional
        Title for the legend.
    filename : str, optional
        File name to save the plot. If None, the plot is displayed.

    Returns
    -------
    None
        Displays the scatter plot.
    """
    # Ensure all values in value_col have a defined color
    unique_values = df[column_color].unique()
    for val in unique_values:
        if val not in color_dict:
            raise ValueError(f"No color specified for value '{val}' in {column_color}")
    color_dict = {val: color_dict[val] for val in unique_values}

    # Determine sizes of points
    sizes = 50
    if size_scale is not None:
        sizes = df[column_xaxis] * size_scale

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    handles = []  # To store legend handles
    labels = []

    for value, color in color_dict.items():
        subset = df[df[column_color] == value]
        scatter = plt.scatter(
            subset[column_xaxis],
            subset[column_yaxis],
            label=value,
            color=color,
            alpha=0.7,
            s=sizes[subset.index] if size_scale else sizes)
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markersize=8))
        labels.append(value)  # Add the label for each unique group

        # Add the name of the 'generator' for the points with a value above the threshold
        if annotate_thresh is not None:
            for i, txt in enumerate(subset['generator']):
                if subset[column_xaxis].iloc[i] > annotate_thresh:
                    plt.annotate(txt, (subset[column_xaxis].iloc[i], subset[column_yaxis].iloc[i]), color='black')

    if ymax is not None:
        plt.ylim(0, ymax)

    if xmax is not None:
        plt.xlim(0, xmax)

    # Add labels and legend
    plt.xlabel(column_xaxis)
    plt.ylabel(column_yaxis)
    plt.title(title)

    # remove legend
    plt.legend().remove()
    if legend is not None:
        plt.legend(handles=handles, labels=labels, title=legend or column_color, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def subplot_scatter(df, column_xaxis, column_yaxis, column_color, color_dict, figsize=(12,8),
                             ymax=None, xmax=None, title='', legend=None, filename=None,
                             size_scale=None, annotate_thresh=None, subplot_column=None):
    """
    Creates scatter plots with points colored based on the values in a specific column.
    Supports optional subplots based on a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    column_xaxis : str
        Column name for x-axis values.
    column_yaxis : str
        Column name for y-axis values.
    column_color : str
        Column name for categorical values determining color.
    color_dict : dict
        Dictionary mapping values in column_color to specific colors.
    ymax : float, optional
        Maximum y-axis value.
    xmax : float, optional
        Maximum x-axis value.
    title : str, optional
        Title of the plot.
    legend : str, optional
        Title for the legend.
    filename : str, optional
        File name to save the plot. If None, the plot is displayed.
    size_scale : float, optional
        Scaling factor for point sizes.
    annotate_thresh : float, optional
        Threshold for annotating points with generator names.
    subplot_column : str, optional
        Column name to split the data into subplots.

    Returns
    -------
    None
        Displays the scatter plots.
    """
    # If subplots are required
    if subplot_column is not None:
        unique_values = df[subplot_column].unique()
        n_subplots = len(unique_values)
        ncols = min(3, n_subplots)  # Limit to 3 columns per row
        nrows = int(np.ceil(n_subplots / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), sharex=True, sharey=True)
        axes = np.array(axes).flatten()  # Ensure axes is an iterable 1D array

        for i, val in enumerate(unique_values):
            ax = axes[i]
            subset_df = df[df[subplot_column] == val]

            scatter_plot_on_ax(ax, subset_df, column_xaxis, column_yaxis, column_color, color_dict,
                               ymax, xmax, title=f"{title} - {subplot_column}: {val}",
                               legend=legend, size_scale=size_scale, annotate_thresh=annotate_thresh)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
    else:
        # If no subplots, plot normally
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter_plot_on_ax(ax, df, column_xaxis, column_yaxis, column_color, color_dict,
                           ymax, xmax, title=title, legend=legend,
                           size_scale=size_scale, annotate_thresh=annotate_thresh)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def scatter_plot_on_ax(ax, df, column_xaxis, column_yaxis, column_color, color_dict,
                       ymax=None, xmax=None, title='', legend=None,
                       size_scale=None, annotate_thresh=None):
    """
    Helper function to create a scatter plot on a given matplotlib Axes.
    """
    unique_values = df[column_color].unique()
    for val in unique_values:
        if val not in color_dict:
            raise ValueError(f"No color specified for value '{val}' in {column_color}")

    color_dict = {val: color_dict[val] for val in unique_values}

    # Determine sizes of points
    sizes = 50
    if size_scale is not None:
        sizes = df[column_xaxis] * size_scale

    # Plot each category separately
    for value, color in color_dict.items():
        subset = df[df[column_color] == value]
        scatter = ax.scatter(subset[column_xaxis], subset[column_yaxis],
                             label=value, color=color, alpha=0.7,
                             s=sizes[subset.index] if size_scale else sizes)

        # Annotate points above a certain threshold
        if annotate_thresh is not None:
            for i, txt in enumerate(subset['generator']):
                if subset[column_xaxis].iloc[i] > annotate_thresh:
                    x_value, y_value = subset[column_xaxis].iloc[i], subset[column_yaxis].iloc[i]
                    ax.annotate(
                        txt,
                        (x_value, y_value),  # Point location
                        xytext=(5, 10),  # Offset in points (x, y)
                        textcoords='offset points',  # Use an offset from the data point
                        fontsize=9,
                        color='black',
                        ha='left'
                    )
                    # ax.annotate(txt, (subset[column_xaxis].iloc[i], subset[column_yaxis].iloc[i]), color='black')

    if ymax is not None:
        ax.set_ylim(0, ymax)

    if xmax is not None:
        ax.set_xlim(0, xmax)

    ax.set_xlabel(column_xaxis)
    ax.set_ylabel(column_yaxis)
    ax.set_title(title)

    # Remove legend from each subplot to avoid redundancy
    if legend is not None:
        ax.legend(title=legend, frameon=False)

    ax.grid(True, linestyle='--', alpha=0.5)


def heatmap_plot(data, filename=None, percentage=False, baseline='Baseline'):
    """
    Plots a heatmap showing differences from baseline with color scales defined per column.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with scenarios as rows and metrics as columns.
    filename : str
        Path to save the plot.
    percentage : bool, optional
        Whether to show differences as percentages.
    """

    # Calculate differences from baseline
    baseline_values = data.loc[baseline, :]
    diff_from_baseline = data.subtract(baseline_values, axis=1)

    # Combine differences and baseline values for annotations
    annotations = data.map(lambda x: f"{x:,.0f}")  # Format baseline values
    # Format differences in percentage
    if percentage:
        diff_from_baseline = diff_from_baseline / baseline_values
        diff_annotations = diff_from_baseline.map(lambda x: f" ({x:+,.0%})")
    else:
        diff_annotations = diff_from_baseline.map(lambda x: f" ({x:+,.0f})")
    combined_annotations = annotations + diff_annotations  # Combine both

    # Normalize the color scale by column
    diff_normalized = diff_from_baseline.copy()
    for column in diff_from_baseline.columns:
        col_min = diff_from_baseline[column].min()
        col_max = diff_from_baseline[column].max()
        diff_normalized[column] = (diff_from_baseline[column] - col_min) / (col_max - col_min)

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the heatmap
    sns.heatmap(
        diff_normalized,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        annot=combined_annotations,  # Show baseline values and differences
        fmt="",  # Disable default formatting
        linewidths=0.5,
        ax=ax,
        cbar=False  # Remove color bar
    )

    # Customize the axes
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def rename_and_reoder(df, rename_index=None, rename_columns=None, order_index=None, order_columns=None):
    if rename_index is not None:
        df.index = df.index.map(lambda x: rename_index.get(x, x))
    if rename_columns is not None:
        df.columns = df.columns.map(lambda x: rename_columns.get(x, x))
    if order_index is not None:
        df = df.loc[order_index, :]
    if order_columns is not None:
        df = df.loc[:, order_columns]
    return df


def make_heatmap_regional_plot(epm_results, filename, percentage=False, scenario_order=None,
                               discount_rate=0, year=2050, required_keys=None, fuel_capa_list=None,
                               fuel_gen_list=None, summary_metrics_list=None, zone_list=None, rows_index='zone',
                               rename_columns=None):
    """
    Make a heatmap plot for the results of the EPM model.


    Parameters
    ----------
    epm_results: dict
    filename: str
    percentage: bool, optional, default is False
    scenario_order
    discount_rate
    """
    summary = []

    if required_keys is None:
        required_keys = ['pCapacityByFuel', 'pEnergyByFuel', 'pEmissions', 'pDemandSupplyCountry', 'pCostSummary', 'pNPVByYear']

    assert all(
        key in epm_results for key in required_keys), "Required keys for the summary are not included in epm_results"

    if fuel_capa_list is None:
        fuel_capa_list = ['Hydro', 'Solar', 'Wind']

    if fuel_gen_list is None:
        fuel_gen_list = ['Hydro', 'Oil']

    if summary_metrics_list is None:
        summary_metrics_list = ['Capex: $m']

    if 'pCapacityByFuel' in required_keys:
        temp = epm_results['pCapacityByFuel'].copy()
        temp = temp[(temp['year'] == year)]
        if zone_list is not None:
            temp = temp[temp['zone'].isin(zone_list)]
        temp = temp.pivot_table(index=[rows_index], columns=NAME_COLUMNS['pCapacityByFuel'], values='value')
        temp = temp.loc[:, fuel_capa_list]
        temp = rename_and_reoder(temp, rename_columns=rename_columns)
        temp.columns = [f'{col} (MW)' for col in temp.columns]
        temp = temp.round(0)
        summary.append(temp)

    if 'pEnergyByFuel' in required_keys:
        temp = epm_results['pEnergyByFuel'].copy()
        temp = temp[(temp['year'] == year)]
        if zone_list is not None:
            temp = temp[temp['zone'].isin(zone_list)]
        temp = temp.loc[:, fuel_gen_list]
        temp = temp.pivot_table(index=[rows_index], columns=NAME_COLUMNS['pEnergyByFuel'], values='value')
        temp.columns = [f'{col} (GWh)' for col in temp.columns]
        temp = temp.round(0)
        summary.append(temp)

    if 'pEmissions' in required_keys:
        temp = epm_results['pEmissions'].copy()
        temp = temp[(temp['year'] == year)]
        if zone_list is not None:
            temp = temp[temp['zone'].isin(zone_list)]
        temp = temp.set_index(['scenario'])['value']
        temp = temp * 1e3
        temp.rename('ktCO2', inplace=True).to_frame()
        summary.append(temp)

    if 'pDemandSupplyCountry' in required_keys:
        temp = epm_results['pDemandSupplyCountry'].copy()
        temp = temp[temp['attribute'] == 'Unmet demand: GWh']
        temp = temp.groupby(['scenario'])['value'].sum()

        t = epm_results['pDemandSupplyCountry'].copy()
        t = t[t['attribute'] == 'Demand: GWh']
        t = t.groupby(['scenario'])['value'].sum()
        temp = (temp / t) * 1e3
        temp.rename('Unmet (‰)', inplace=True).to_frame()
        summary.append(temp)
        
        temp = epm_results['pDemandSupplyCountry'].copy()
        temp = temp[(temp['zone'] == 'Guinea')]
        temp = temp[temp['attribute'] == 'Surplus generation: GWh']
        temp = temp.groupby(['scenario'])['value'].sum()

        t = epm_results['pDemandSupplyCountry'].copy()
        t = t[(t['zone'] == 'Guinea')]
        t = t[t['attribute'] == 'Demand: GWh']
        t = t.groupby(['scenario'])['value'].sum()
        temp = (temp / t) * 1000

        temp.rename('Surplus (‰)', inplace=True).to_frame()
        summary.append(temp)

    if 'pCostSummary' in required_keys:
        capex_metric = 'Capex: $m'
        temp = epm_results['pCostSummary'].copy()
        if zone_list is not None:
            temp = temp[temp['zone'].isin(zone_list)]

        duration = temp['year'].max() - temp['year'].min() + 1
        capex_tot = temp.loc[temp.attribute == capex_metric].groupby([rows_index])['value'].sum()
        capex_tot = capex_tot / duration

        temp = temp.pivot_table(index=[rows_index], columns=NAME_COLUMNS['pCostSummary'], values='value')
        filtered_metrics = [metric for metric in summary_metrics_list if metric != capex_metric]

        if len(filtered_metrics) > 0:
            temp = temp.loc[:, filtered_metrics]
            temp = pd.concat([temp,  capex_tot.to_frame().rename(columns={'value': capex_metric})], axis=1)
        else:
            temp = capex_tot.to_frame().rename(columns={'value': capex_metric})

        # temp.rename('Capex ($M/year)', inplace=True).to_frame()
        summary.append(temp)
        
    if 'pCostSummary' in required_keys:
        temp = epm_results['pCostSummary'].copy()
        temp = temp[temp['attribute'] == 'Total Annual Cost by Zone: $m']
        temp = calculate_total_system_cost(temp, discount_rate)

        t = epm_results['pDemandSupply'].copy()
        t = t[(t['zone'] == 'Guinea')]
        t = t[t['attribute'] == 'Demand: GWh']
        t = calculate_total_system_cost(t, discount_rate)

        temp = (temp * 1e6) / (t * 1e3)

        if isinstance(temp, (float, int)):
            temp = pd.Series(temp, index=[epm_results['pNPVByYear']['scenario'][0]])
        temp.rename('NPV ($/MWh)', inplace=True).to_frame()
        summary.append(temp)


    summary = pd.concat(summary, axis=1)

    if scenario_order is not None:
        scenario_order = [i for i in scenario_order if i in summary.index] + [i for i in summary.index if
                                                                              i not in scenario_order]
        summary = summary.loc[scenario_order]

    heatmap_plot(summary, filename, percentage=percentage, baseline=summary.index[0])





def create_zonemap(zone_map, selected_countries, map_epm_to_geojson):
    zone_map = zone_map[zone_map['ADMIN'].isin(selected_countries)]

    if zone_map.crs.is_geographic:
        zone_map = zone_map.to_crs(epsg=3857)

    # Get the coordinates of the centers of the zones, countries, region
    # centroids = zone_map.centroid
    centers = {row['ADMIN']: [row.geometry.centroid.x, row.geometry.centroid.y] for index, row in zone_map.iterrows()}

    latitudes = [coords[1] for coords in centers.values()]
    longitudes = [coords[0] for coords in centers.values()]
    center_latitude = sum(latitudes) / len(latitudes)
    center_longitude = sum(longitudes) / len(longitudes)
    # region_center = [center_latitude, center_longitude]

    # geojson_names = list(correspondence_Co['Geojson_Zone'])
    # model_names = list(correspondence_Co['EPM_Zone'])

    centers = {map_epm_to_geojson[c]: v for c, v in centers.items()}
    return zone_map, centers


def plot_zone_map_on_ax(ax, zone_map):
    zone_map.plot(ax=ax, color='white', edgecolor='black')

    # Adjusting the limits to better center the zone_map on the region
    ax.set_xlim(zone_map.bounds.minx.min() - 1, zone_map.bounds.maxx.max() + 1)
    ax.set_ylim(zone_map.bounds.miny.min() - 1, zone_map.bounds.maxy.max() + 1)


def make_capacity_mix_map(zone_map, pCapacityByFuel, dict_colors, centers, year, region, scenario, filename,
                          map_epm_to_geojson, index='fuel', list_reduced_size=None, figsize=(10, 6), bbox_to_anchor=(0.5, -0.1),
                          loc='center left', min_size=0.5, max_size =2.5, pie_sizing=True):
    """
    Plots a capacity mix map with pie charts overlaid on a regional map.

    Parameters:
    - zone_map: GeoDataFrame containing the map regions.
    - CapacityMix_scen: DataFrame containing the capacity mix data per zone.
    - fuels_list: List of fuels to include in the plot.
    - centers: Dictionary mapping zones to their center coordinates.
    - year: The target year for the plot.
    - region_name: Name of the region for the title.
    - scenario: Scenario name for the title.
    - graphs_folder: Path where the plot will be saved.
    - selected_scenario: The specific scenario being plotted.
    - geojson_names: List of country names in the GeoJSON file.
    - model_names: List of country names used in the model.
    - list_reduced_size: List of zones where pie size should be reduced.
    - colorf: Function mapping fuel names to colors.
    """

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Plot the base zone map
    plot_zone_map_on_ax(ax, zone_map)

    # Remove axes for a clean map
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(f'Capacity mix - {region} \n {scenario} - {year}', loc='center')

    # Compute pie sizes for each zone
    region_sizes = zone_map.copy()
    region_sizes['area'] = region_sizes.geometry.area
    region_sizes['Name'] = region_sizes['ADMIN'].replace(map_epm_to_geojson)

    def calculate_pie_size(zone, CapacityByFuel):
        """Calculate pie chart size based on region area."""
        # area = region_sizes.loc[region_sizes['Name'] == zone, 'area'].values[0]
        # normalized_area = (area - region_sizes['area'].min()) / (region_sizes['area'].max() - region_sizes['area'].min())
        area = pCapacityByFuel[(pCapacityByFuel['zone'] == zone) & (pCapacityByFuel['year'] == year)].value.sum()
        normalized_area = (area - CapacityByFuel.groupby('zone').value.sum().min()) / (CapacityByFuel.groupby('zone').value.sum().max() - CapacityByFuel.groupby('zone').value.sum().min())
        return min_size + normalized_area * (max_size - min_size)

    handles, labels = [], []
    # Plot pie charts for each zone
    for zone in pCapacityByFuel['zone'].unique():
        # Extract capacity mix for the given zone and year
        CapacityMix_plot = (pCapacityByFuel[(pCapacityByFuel['zone'] == zone) & (pCapacityByFuel['year'] == year)]
                            .set_index(index)['value']
                            .fillna(0)).reset_index()

        # Skip empty plots
        if CapacityMix_plot['value'].sum() == 0:
            continue

        # Get map coordinates
        coordinates = centers.get(zone, (0, 0))
        loc = fig.transFigure.inverted().transform(ax.transData.transform(coordinates))

        # Pie chart positioning and size
        size = [0.03, 0.07]
        if pie_sizing:
            if list_reduced_size is not None:
                pie_size = 0.7 if zone in list_reduced_size else calculate_pie_size(zone, pCapacityByFuel)
            else:
                pie_size = calculate_pie_size(zone, pCapacityByFuel)
        else:
            pie_size = None

        # Create inset pie chart
        ax_pie = fig.add_axes([loc[0] - 0.45 * size[0], loc[1] - 0.5 * size[1], size[0], size[1]])
        colors = [dict_colors[f] for f in CapacityMix_plot[index]]
        h, l = plot_pie_on_ax(ax_pie, CapacityMix_plot, index, 25, colors, None, radius= pie_size)
        ax_pie.set_axis_off()

        for handle, label in zip(h, l):
            if label not in labels:  # Avoid duplicates
                handles.append(handle)
                labels.append(label)

    fig.legend(handles, labels, loc=loc, frameon=False, ncol=1,
               bbox_to_anchor=bbox_to_anchor)

    # plt.tight_layout()

    # Save and show figure
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    print(0)
