
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

### Processing EPM Inputs & Results
- **`process_epm_inputs(epm_input, dict_specs, scenarios_rename=None)`**: Processes EPM input files for plotting.
- **`process_epm_results(epm_results, dict_specs, scenarios_rename=None, mapping_gen_fuel=None)`**: Processes EPM output results for plotting.
- **`process_simulation_results(FOLDER, SCENARIOS_RENAME=None, folder='')`**: Extracts and processes both inputs and outputs.
- **`generate_summary(epm_results, folder, epm_input)`**: Generates a summary of key EPM results.

### Post-Processing & Plotting
- **`postprocess_output(FOLDER, reduced_output=False, plot_all=False, folder='')`**: Processes simulation results, generates summaries, and creates visualizations.
- **`stacked_area_plot(df, filename, ...)`**: Creates a stacked area plot of generation by fuel type.
- **`bar_plot(df, x, y, ...)`**: Creates a bar plot.
- **`line_plot(df, x, y, ...)`**: Creates a line plot.
- **`dispatch_plot(df_area, df_line, ...)`**: Creates a dispatch plot with stacked areas and line plots.
- **`make_complete_fuel_dispatch_plot(dfs_area, dfs_line, ...)`**: Generates and saves a complete fuel dispatch plot.
- **`make_stacked_bar_subplots(df, filename, dict_colors, ...)`**: Creates stacked bar subplots for comparing capacity over time and across scenarios.
- **`subplot_pie(df, index, dict_colors, ...)`**: Creates pie chart subplots.

### Geospatial & Mapping Functions
- **`make_capacity_mix_map(...)`**: Creates a capacity mix map with pie charts overlaid on a regional map.
- **`make_interconnection_map(...)`**: Generates an interconnection map showing transmission capacities.
- **`create_interactive_map(...)`**: Generates an interactive Folium-based map for energy dispatch and capacity visualization.

---

## 3. Postprocessing functions

### **`read_plot_specs(folder='')`**
Reads static plot specifications including color mappings, fuel mappings, and geographic data.

#### Parameters
- `folder` (str, optional): Path to the directory containing static files.

#### Returns
- `dict_specs` (dict): A dictionary containing:
  - Colors for fuels and technologies.
  - Fuel and technology mappings.
  - Geographic data for mapping.

#### Example Usage
```python
specs = read_plot_specs('static/')
print(specs['colors'])  # Displays color mapping for fuels
```
---
