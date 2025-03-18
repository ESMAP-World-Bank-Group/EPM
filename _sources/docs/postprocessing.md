
# Postprocessing of the results

This section describes the postprocessing of the results of the simulation. It enables the user to quickly visualize the results and generate standardized figures.

## 1. Structure

The postprocessing is divided into two main parts:
- `epm/postprocessing/utils.py`: Contains the functions to read the results and generate the figures.
- Jupyter notebooks to run the postprocessing. An example of the postprocessing is provided bellow. User need to adapt the code to their needs.

Jupyter notebooks can be downloaded from the documentation to be used for user case studies. The following video summarizes the process:
<video width="720" controls>
  <source src="videos/download_notebook.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 2. `utils` package overview

### Data Handling
- **`read_plot_specs(folder='')`**: Reads plot specifications (colors, fuel mappings, technology mappings, etc.).
- **`extract_gdx(file)`**: Extracts parameters and sets from a `.gdx` file into a dictionary of DataFrames.
- **`extract_epm_folder(results_folder, file='epmresults.gdx')`**: Extracts GDX results from multiple scenarios in a folder.
- **`standardize_names(dict_df, key, mapping, column='fuel')`**: Standardizes fuel names in results using a predefined mapping.
- **`filter_dataframe(df, conditions)`**: Filters a DataFrame based on a dictionary of conditions.
- **`filter_dataframe_by_index(df, conditions)`**: Filters a DataFrame by conditions applied to the index levels.
- **`process_epm_inputs(epm_input, dict_specs, scenarios_rename=None)`**: Processes EPM input files for plotting.
- **`process_epm_results(epm_results, dict_specs, scenarios_rename=None, mapping_gen_fuel=None)`**: Processes EPM output results for plotting.
- **`process_simulation_results(FOLDER, SCENARIOS_RENAME=None, folder='')`**: Extracts and processes both inputs and outputs.
- **`generate_summary(epm_results, folder, epm_input)`**: Generates a summary of key EPM results.

### Postrocessing & plotting
- **`postprocess_output(FOLDER, reduced_output=False, plot_all=False, folder='')`**: Processes simulation results, generates summaries, and creates visualizations.
- **`stacked_area_plot(df, filename, ...)`**: Creates a stacked area plot of generation by fuel type.
- **`bar_plot(df, x, y, ...)`**: Creates a bar plot.
- **`line_plot(df, x, y, ...)`**: Creates a line plot.
- **`dispatch_plot(df_area, df_line, ...)`**: Creates a dispatch plot with stacked areas and line plots.
- **`make_complete_fuel_dispatch_plot(dfs_area, dfs_line, ...)`**: Generates and saves a complete fuel dispatch plot.
- **`make_stacked_bar_subplots(df, filename, dict_colors, ...)`**: Creates stacked bar subplots for comparing capacity over time and across scenarios.
- **`subplot_pie(df, index, dict_colors, ...)`**: Creates pie chart subplots.
- **`make_capacity_mix_map(...)`**: Creates a capacity mix map with pie charts overlaid on a regional map.
- **`make_interconnection_map(...)`**: Generates an interconnection map showing transmission capacities.
- **`create_interactive_map(...)`**: Generates an interactive Folium-based map for energy dispatch and capacity visualization.

---

## 3.Data handling functions

#### `read_plot_specs`
`read_plot_specs(folder='')` 
Reads static plot specifications including color mappings, fuel mappings, and geographic data.

##### Parameters
- `folder` (str, optional): Path to the directory containing static files.

##### Returns
- `dict_specs` (dict): A dictionary containing:
  - Colors for fuels and technologies.
  - Fuel and technology mappings.
  - Geographic data for mapping.

##### Example Usage
```python
specs = read_plot_specs('static/')
print(specs['colors'])  # Displays color mapping for fuels
```

#### `extract_gdx`
`extract_gdx(file)`
Extracts parameters and sets from a `.gdx` file into a dictionary of DataFrames.

##### Parameters
- `file` (str): Path to the `.gdx` file.

##### Returns
- `epm_result` (dict): Dictionary where:
  - Keys are parameter or set names from the GDX file.
  - Values are Pandas DataFrames containing the extracted data.

##### Example Usage
```python
results = extract_gdx('output/epmresults.gdx')
print(results.keys())  # Displays the available parameters in the GDX file
```

#### `extract_epm_folder`
`extract_epm_folder(results_folder, file='epmresults.gdx')`
Extracts GDX results from multiple scenarios stored in a given folder.

##### Parameters
- `results_folder` (str): Path to the folder containing scenario outputs.
- `file` (str, optional, default=`'epmresults.gdx'`): Name of the `.gdx` file to extract.

##### Returns
- `inverted_dict` (dict): Dictionary where:
  - Keys are result categories (e.g., `pCapacityByFuel`, `pEnergyByFuel`).
  - Values are Pandas DataFrames with extracted results across multiple scenarios.

##### Example Usage
```python
epm_results = extract_epm_folder('output/simulations_run_20250317_132656')
print(epm_results['pCapacityByFuel'])  # Displays capacity by fuel for all scenarios
```


[TO CONTINUE]
### **`standardize_names(dict_df, key, mapping, column='fuel')`**
Standardizes fuel or technology names in DataFrames using a predefined mapping.

#### Parameters
- `dict_df` (dict): Dictionary of DataFrames containing EPM results.
- `key` (str): Key corresponding to the DataFrame to modify.
- `mapping` (dict): Dictionary mapping original names to standardized names.
- `column` (str, optional, default=`'fuel'`): Column to apply standardization.

#### Functionality
- Replaces values in the specified column according to the provided mapping.
- Aggregates values if multiple original names are mapped to the same standardized name.
- Raises an error if new, unmapped values are found in the column.

#### Example Usage
```python
fuel_mapping = {'Diesel': 'Oil', 'Heavy Fuel Oil': 'Oil'}
standardize_names(epm_results, 'pEnergyByFuel', fuel_mapping)
```

### **`filter_dataframe(df, conditions)`**
Filters a DataFrame based on specified conditions.

#### Parameters
- `df` (pd.DataFrame): The DataFrame to filter.
- `conditions` (dict): Dictionary where:
  - Keys are column names.
  - Values specify filter criteria (either a single value or a list of values).

#### Returns
- `pd.DataFrame`: Filtered DataFrame containing only rows that match the given conditions.

#### Example Usage
```python
conditions = {'scenario': 'Baseline', 'year': 2050}
filtered_df = filter_dataframe(epm_results['pCapacityByFuel'], conditions)
```

### **`filter_dataframe_by_index(df, conditions)`**
Filters a DataFrame based on conditions applied to its index levels.

#### Parameters
- `df` (pd.DataFrame): The DataFrame to filter, where conditions apply to index levels.
- `conditions` (dict): Dictionary where:
  - Keys are index level names.
  - Values specify filter criteria (either a single value or a list of values).

#### Returns
- `pd.DataFrame`: Filtered DataFrame containing only rows where the index levels match the given conditions.

#### Example Usage
```python
conditions = {'scenario': 'Baseline', 'year': 2050}
filtered_df = filter_dataframe_by_index(epm_results['pEnergyByFuel'], conditions)
```

### **`process_epm_inputs(epm_input, dict_specs, scenarios_rename=None)`**
Processes EPM input files for visualization by standardizing names and formatting columns.

#### Parameters
- `epm_input` (dict): Dictionary containing raw input data from the `.gdx` file.
- `dict_specs` (dict): Plot specifications obtained from `read_plot_specs()`, including mappings for fuels and technologies.
- `scenarios_rename` (dict, optional): Dictionary mapping old scenario names to new ones for consistency.

#### Returns
- `epm_dict` (dict): Processed input data where:
  - Columns are standardized using predefined mappings.
  - Data types are correctly formatted.
  - Scenario names are updated if a rename mapping is provided.

#### Example Usage
```python
processed_inputs = process_epm_inputs(epm_inputs, specs)
```

### **`process_epm_results(epm_results, dict_specs, scenarios_rename=None, mapping_gen_fuel=None)`**
Processes EPM model outputs for visualization by standardizing names, formatting data, and handling fuel mappings.

#### Parameters
- `epm_results` (dict): Raw GDX extraction results, containing multiple EPM output parameters.
- `dict_specs` (dict): Plot specifications obtained from `read_plot_specs()`, including color mappings and fuel/technology mappings.
- `scenarios_rename` (dict, optional): Dictionary mapping old scenario names to new ones for consistency in visualization.
- `mapping_gen_fuel` (pd.DataFrame, optional): DataFrame mapping generators to fuel types, used to standardize plant-level results.

#### Returns
- `epm_dict` (dict): Processed results where:
  - Fuel and technology names are standardized.
  - Scenario names are updated if a rename mapping is provided.
  - Data is cleaned, formatted, and ready for visualization.

#### Example Usage
```python
processed_results = process_epm_results(epm_results, specs)
```

### **`process_simulation_results(FOLDER, SCENARIOS_RENAME=None, folder='')`**
Processes both inputs and outputs from a simulation run, preparing them for analysis and visualization.

#### Parameters
- `FOLDER` (str): Path to the simulation output folder.
- `SCENARIOS_RENAME` (dict, optional): Dictionary mapping old scenario names to new ones for consistency.
- `folder` (str, optional): Path to the static folder containing additional mappings (e.g., color mappings, fuel classifications).

#### Returns
- `RESULTS_FOLDER` (str): Path to results folder.
- `GRAPHS_FOLDER` (str): Path to folder storing generated graphs.
- `dict_specs` (dict): Specifications for plots, including color mappings and technology classifications.
- `epm_input` (dict): Processed input data from the simulation.
- `epm_results` (dict): Processed output data, formatted for visualization.
- `mapping_gen_fuel` (pd.DataFrame): DataFrame mapping generators to fuels.

#### Example Usage
```python
RESULTS_FOLDER, GRAPHS_FOLDER, specs, epm_input, epm_results, mapping_gen_fuel = process_simulation_results('output/simulations_run_20250317_132656')
```

### **`generate_summary(epm_results, folder, epm_input)`**
Generates a summary of key EPM results, aggregating important indicators such as system costs, emissions, generation, and capacity.

#### Parameters
- `epm_results` (dict): Processed EPM output data containing key model results.
- `folder` (str): Path to the folder where the summary CSV file will be saved.
- `epm_input` (dict): Processed input data, used for context in the summary.

#### Functionality
- Extracts key indicators such as system costs, demand, capacity, generation, and emissions.
- Aggregates results across years and scenarios.
- Saves the summary as a CSV file in the specified folder.

#### Example Usage
```python
generate_summary(epm_results, 'output/simulations_run_20250317_132656', epm_input)
```

### **`postprocess_output(FOLDER, reduced_output=False, plot_all=False, folder='')`**
Runs the entire post-processing workflow, which includes processing simulation results, generating summary tables, and creating visualizations.

#### Parameters
- `FOLDER` (str): Path to the simulation output folder.
- `reduced_output` (bool, optional, default=`False`): If `True`, generates a reduced set of outputs to save processing time.
- `plot_all` (bool, optional, default=`False`): If `True`, generates all available plots instead of a limited selection.
- `folder` (str, optional): Path to the static mapping files, containing information such as color schemes and technology groupings.

#### Functionality
- Calls `process_simulation_results()` to extract and format inputs and outputs.
- Generates summary statistics via `generate_summary()`.
- Creates visualizations such as stacked area plots, dispatch plots, and bar charts.
- Saves results and graphs in the designated output folder.

#### Example Usage
```python
postprocess_output('output/simulations_run_20250317_132656', reduced_output=True, plot_all=False)
```
---

## 4.Plotting functions
### **`stacked_area_plot(df, filename, dict_colors=None, x_column='year', y_column='value', stack_column='fuel', ...)`**
Creates a stacked area plot, commonly used to visualize energy generation by fuel type over time.

#### Parameters
- `df` (pd.DataFrame): Input data containing the variables to be plotted.
- `filename` (str): Path to save the generated plot.
- `dict_colors` (dict, optional): Dictionary mapping categories (e.g., fuel types) to colors.
- `x_column` (str, default=`'year'`): Column name representing the x-axis (e.g., time).
- `y_column` (str, default=`'value'`): Column name representing the y-axis (e.g., energy produced).
- `stack_column` (str, default=`'fuel'`): Column name used to stack different categories in the plot.

#### Functionality
- Groups data by `x_column` and `stack_column`, summing values.
- Uses area plotting to visualize the contribution of different categories (e.g., fuel types).
- Colors are applied based on `dict_colors`, if provided.

#### Example Usage
```python
stacked_area_plot(df=energy_data, filename="generation.png", x_column="year", y_column="value", stack_column="fuel", dict_colors=color_dict)
```

### **`bar_plot(df, x, y, xlabel=None, ylabel=None, title=None, filename=None, figsize=(8, 5))`**
Creates a bar plot to visualize categorical data, such as capacity by fuel or cost breakdown.

#### Parameters
- `df` (pd.DataFrame): Input data containing the variables to be plotted.
- `x` (str): Column name for the x-axis categories.
- `y` (str): Column name for the y-axis values.
- `xlabel` (str, optional): Label for the x-axis.
- `ylabel` (str, optional): Label for the y-axis.
- `title` (str, optional): Title of the plot.
- `filename` (str, optional): Path to save the generated plot.
- `figsize` (tuple, default=`(8, 5)`): Size of the figure.

#### Functionality
- Creates a bar plot using the specified x and y variables.
- Adds annotations to bars for clarity.
- Saves the figure if `filename` is provided; otherwise, displays the plot.

#### Example Usage
```python
bar_plot(df=capacity_data, x="fuel", y="value", xlabel="Fuel Type", ylabel="Installed Capacity (MW)", title="Capacity by Fuel", filename="capacity_bar.png")
```
### **`line_plot(df, x, y, xlabel=None, ylabel=None, title=None, filename=None, figsize=(10, 6))`**
Creates a line plot to visualize trends over time, such as cost evolution or energy generation by year.

#### Parameters
- `df` (pd.DataFrame): Input data containing the variables to be plotted.
- `x` (str): Column name for the x-axis values.
- `y` (str): Column name for the y-axis values.
- `xlabel` (str, optional): Label for the x-axis.
- `ylabel` (str, optional): Label for the y-axis.
- `title` (str, optional): Title of the plot.
- `filename` (str, optional): Path to save the generated plot.
- `figsize` (tuple, default=`(10, 6)`): Size of the figure.

#### Functionality
- Plots a line chart using `x` and `y` columns from the DataFrame.
- Adds axis labels and title if provided.
- Saves the figure if `filename` is provided; otherwise, displays the plot.

#### Example Usage
```python
line_plot(df=cost_data, x="year", y="value", xlabel="Year", ylabel="Total System Cost ($M)", title="System Cost Evolution", filename="cost_evolution.png")
```

### **`dispatch_plot(df_area=None, filename=None, dict_colors=None, df_line=None, figsize=(10, 6), legend_loc='bottom', bottom=0)`**
Creates a dispatch plot combining stacked area and line plots to visualize electricity generation, demand, and storage dynamics.

#### Parameters
- `df_area` (pd.DataFrame, optional): DataFrame for stacked area plots (e.g., generation by fuel).
- `filename` (str, optional): Path to save the generated plot.
- `dict_colors` (dict, optional): Dictionary mapping categories (e.g., fuels) to colors.
- `df_line` (pd.DataFrame, optional): DataFrame for line plots (e.g., demand curve).
- `figsize` (tuple, default=`(10, 6)`): Size of the figure.
- `legend_loc` (str, default=`'bottom'`): Location of the legend (`'bottom'` or `'right'`).
- `bottom` (int, default=`0`): Minimum value for the y-axis.

#### Functionality
- Uses a stacked area plot to show generation by fuel type.
- Overlays line plots for demand and other attributes.
- Formats the plot with axis labels and a customizable legend position.

#### Example Usage
```python
dispatch_plot(df_area=generation_data, df_line=demand_data, dict_colors=fuel_colors, filename="dispatch_plot.png")
```

### **`make_complete_fuel_dispatch_plot(dfs_area, dfs_line, dict_colors, zone, year, scenario, filename=None, fuel_grouping=None, select_time=None, reorder_dispatch=None, legend_loc='bottom', bottom=0, figsize=(10,6))`**
Generates and saves a complete fuel dispatch plot, including generation, demand, and other dispatch components.

#### Parameters
- `dfs_area` (dict): Dictionary of DataFrames for stacked area plots (e.g., generation by fuel).
- `dfs_line` (dict): Dictionary of DataFrames for line plots (e.g., demand curve).
- `dict_colors` (dict): Dictionary mapping fuel types to colors.
- `zone` (str): The zone to filter the data for.
- `year` (int): The year to filter the data for.
- `scenario` (str): The scenario to filter the data for.
- `filename` (str, optional): Path to save the generated plot.
- `fuel_grouping` (dict, optional): Mapping of specific fuels to broader fuel categories.
- `select_time` (dict, optional): Dictionary specifying a subset of time periods to include in the plot.
- `reorder_dispatch` (list, optional): List specifying a custom order for stacking fuels.
- `legend_loc` (str, default=`'bottom'`): Location of the legend (`'bottom'` or `'right'`).
- `bottom` (int, default=`0`): Minimum value for the y-axis.
- `figsize` (tuple, default=`(10,6)`): Size of the figure.

#### Functionality
- Filters the input DataFrames based on the specified zone, year, and scenario.
- Processes data for stacked area and line plots.
- Generates a dispatch plot showing electricity generation, demand, and storage behavior.
- Saves the figure if `filename` is provided; otherwise, displays the plot.

#### Example Usage
```python
make_complete_fuel_dispatch_plot(dfs_area=generation_data, dfs_line=demand_data, dict_colors=fuel_colors, zone='Liberia', year=2030, scenario='Baseline', filename="dispatch_fuel_plot.png")
```

### **`make_stacked_bar_subplots(df, filename, dict_colors, selected_zone=None, selected_year=None, column_xaxis='year', column_stacked='fuel', column_multiple_bars='scenario', column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None, dict_scenarios=None, format_y=lambda y, _: '{:.0f} MW'.format(y), order_stacked=None, cap=2, annotate=True, show_total=False, fonttick=12, rotation=0, title=None)`**
Creates stacked bar subplots to analyze energy system evolution, such as capacity changes over time or across scenarios.

#### Parameters
- `df` (pd.DataFrame): DataFrame containing the results.
- `filename` (str): Path to save the figure.
- `dict_colors` (dict): Dictionary mapping categories (e.g., fuel types) to colors.
- `selected_zone` (str, optional): Filter the data for a specific zone.
- `selected_year` (int, optional): Filter the data for a specific year.
- `column_xaxis` (str, default=`'year'`): Column for subplot separation.
- `column_stacked` (str, default=`'fuel'`): Column to use for stacking bars.
- `column_multiple_bars` (str, default=`'scenario'`): Column defining multiple bars within a single subplot.
- `column_value` (str, default=`'value'`): Column containing numerical values to be plotted.
- `select_xaxis` (list, optional): Subset of values to display on the x-axis.
- `dict_grouping` (dict, optional): Mapping of categories for aggregation.
- `order_scenarios` (list, optional): Order in which scenarios should be displayed.
- `dict_scenarios` (dict, optional): Dictionary mapping scenario names to new labels.
- `format_y` (function, optional, default=`'{:.0f} MW'.format(y)`): Function to format y-axis labels.
- `order_stacked` (list, optional): Custom order for stacked categories.
- `cap` (int, default=`2`): Minimum value for displaying annotations.
- `annotate` (bool, default=`True`): Whether to annotate values on the bars.
- `show_total` (bool, default=`False`): Whether to display the total value on top of bars.
- `fonttick` (int, default=`12`): Font size for tick labels.
- `rotation` (int, default=`0`): Rotation angle for x-axis labels.
- `title` (str, optional): Title of the plot.

#### Functionality
- Groups data according to the selected columns and aggregates values.
- Generates multiple subplots for different x-axis categories (e.g., years).
- Stacks bars by fuel type or technology.
- Displays labels and annotations as specified.

#### Example Usage
```python
make_stacked_bar_subplots(df=capacity_data, filename="capacity_evolution.png", dict_colors=fuel_colors, selected_zone='Liberia', select_xaxis=[2025, 2030, 2040], order_scenarios=['Baseline', 'High Hydro', 'High Demand'], format_y=lambda y, _: '{:.0f} MW'.format(y))
```

### **`subplot_pie(df, index, dict_colors, filename=None, figsize=(10, 6), autopct='%1.1f%%', title=None, explode=None)`**
Creates pie chart subplots to visualize the share of different categories, such as capacity mix or energy generation by fuel type.

#### Parameters
- `df` (pd.DataFrame): DataFrame containing the data to be plotted.
- `index` (str): Column used to differentiate pie charts (e.g., scenario, year, or zone).
- `dict_colors` (dict): Dictionary mapping categories (e.g., fuel types) to colors.
- `filename` (str, optional): Path to save the figure.
- `figsize` (tuple, default=`(10, 6)`): Size of the figure.
- `autopct` (str, optional, default=`'%1.1f%%'`): Format for displaying percentages on the pie chart.
- `title` (str, optional): Title of the plot.
- `explode` (dict, optional): Dictionary specifying which slices should be exploded for emphasis.

#### Functionality
- Groups data by the specified `index` and normalizes values to calculate percentages.
- Generates multiple pie charts as subplots for different categories.
- Applies consistent colors using `dict_colors`.
- Saves the figure if `filename` is provided; otherwise, displays the plot.

#### Example Usage
```python
subplot_pie(df=capacity_mix, index="scenario", dict_colors=fuel_colors, filename="capacity_pie.png", title="Capacity Mix by Scenario")
```

### **`make_capacity_mix_map(zone_map, pCapacityByFuel, dict_colors, centers, year, scenario, filename=None, folder='')`**
Creates a capacity mix map with pie charts overlaid on a regional map, showing the share of different fuel types in each zone.

#### Parameters
- `zone_map` (GeoDataFrame): Geospatial map of the zones.
- `pCapacityByFuel` (pd.DataFrame): DataFrame containing capacity data by fuel type for each zone.
- `dict_colors` (dict): Dictionary mapping fuel types to colors.
- `centers` (dict): Dictionary mapping zone names to their geographic coordinates.
- `year` (int): Year to visualize.
- `scenario` (str): Scenario to visualize.
- `filename` (str, optional): Path to save the generated map.
- `folder` (str, optional): Directory where additional static files (e.g., shapefiles) are stored.

#### Functionality
- Filters `pCapacityByFuel` for the selected `year` and `scenario`.
- Normalizes values to compute capacity shares for each fuel type.
- Overlays pie charts on the map to represent the fuel mix in each zone.
- Saves the figure if `filename` is provided; otherwise, displays the map.

#### Example Usage
```python
make_capacity_mix_map(zone_map, capacity_data, dict_colors, centers, year=2030, scenario="Baseline", filename="capacity_mix_map.png")
```

### **`make_interconnection_map(zone_map, interconnection_data, dict_colors, centers, year, scenario, filename=None, folder='')`**
Generates an interconnection map showing transmission capacities between zones.

#### Parameters
- `zone_map` (GeoDataFrame): Geospatial map of the zones.
- `interconnection_data` (pd.DataFrame): DataFrame containing transmission capacities between zones.
- `dict_colors` (dict): Dictionary mapping different interconnection types to colors.
- `centers` (dict): Dictionary mapping zone names to their geographic coordinates.
- `year` (int): Year to visualize.
- `scenario` (str): Scenario to visualize.
- `filename` (str, optional): Path to save the generated map.
- `folder` (str, optional): Directory where additional static files (e.g., shapefiles) are stored.

#### Functionality
- Filters `interconnection_data` for the selected `year` and `scenario`.
- Plots transmission lines between zones, with thickness representing capacity.
- Uses `dict_colors` to differentiate interconnection types.
- Saves the figure if `filename` is provided; otherwise, displays the map.

#### Example Usage
```python
make_interconnection_map(zone_map, interconnection_data, dict_colors, centers, year=2030, scenario="Baseline", filename="interconnection_map.png")
```

### **`create_interactive_map(zone_map, centers, transmission_data, energy_data, year, scenario, filename=None, folder='')`**
Generates an interactive map using Folium to visualize energy capacity, dispatch, and interconnections.

#### Parameters
- `zone_map` (GeoDataFrame): Geospatial map of the zones.
- `centers` (dict): Dictionary mapping zone names to their geographic coordinates.
- `transmission_data` (pd.DataFrame): DataFrame containing transmission capacities between zones.
- `energy_data` (pd.DataFrame): DataFrame containing generation and demand data for each zone.
- `year` (int): Year to visualize.
- `scenario` (str): Scenario to visualize.
- `filename` (str, optional): Path to save the interactive map as an HTML file.
- `folder` (str, optional): Directory where additional static files (e.g., shapefiles) are stored.

#### Functionality
- Filters `transmission_data` and `energy_data` for the selected `year` and `scenario`.
- Creates an interactive map using Folium.
- Overlays pie charts representing generation mix and demand for each zone.
- Draws transmission lines with width proportional to capacity.
- Saves the map as an HTML file if `filename` is provided; otherwise, displays it in the browser.

#### Example Usage
```python
create_interactive_map(zone_map, centers, transmission_data, energy_data, year=2030, scenario="Baseline", filename="interactive_map.html")
```
---
