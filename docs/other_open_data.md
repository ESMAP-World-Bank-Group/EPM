# Open-Source Data (in progress)

> This tables provides a list of open-source data sources that can be used in the Electricity Planning Model (EPM). The data is categorized by type and includes links to the sources.

**Note:** This table is a work in progress. If you have suggestions for additional data sources, please contact the EPM team.


<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

<table id="energy-table" class="display">
  <thead>
    <tr>
      <th>Data</th>
      <th>Resolution</th>
      <th>Source</th>
      <th>Description</th>
      <th>Comment</th>
      <th>Website</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Electricity demand</td>
      <td>Yearly</td>
      <td>—</td>
      <td>Historical data</td>
      <td>—</td>
      <td><a href="https://ourworldindata.org/energy" target="_blank">OWID</a>, <a href="https://github.com/owid/energy-data" target="_blank">GitHub</a></td>
    </tr>
    <tr>
      <td>Electricity demand</td>
      <td>Hourly</td>
      <td>ENTSO-E</td>
      <td>Monthly and hourly load values</td>
      <td>—</td>
      <td><a href="https://docstore.entsoe.eu/Documents/Publications/Statistics/Monthly-hourly-load-values_2006-2015.xlsx" target="_blank">Excel</a>, <a href="https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show" target="_blank">Transparency Platform</a></td>
    </tr>
    <tr>
      <td>Electricity demand</td>
      <td>Hourly</td>
      <td>Toktarova et al., 2019</td>
      <td>Long-term global load projections</td>
      <td>Synthetic profile</td>
      <td>—</td>
    </tr>
    <tr>
      <td>GDP</td>
      <td>—</td>
      <td>—</td>
      <td>GDP dataset</td>
      <td>—</td>
      <td><a href="https://datadryad.org/dataset/doi:10.5061/dryad.dk1j0" target="_blank">DataDryad</a></td>
    </tr>
    <tr>
      <td>Generation</td>
      <td>Europe</td>
      <td>PowerPlantMatching</td>
      <td>Downloadable CSV, used with PyPSA</td>
      <td>—</td>
      <td><a href="https://github.com/PyPSA/powerplantmatching" target="_blank">GitHub</a></td>
    </tr>
    <tr>
      <td>Generation</td>
      <td>—</td>
      <td>Global Energy Monitor</td>
      <td>Global Integrated Power Tracker: unit-level data worldwide</td>
      <td>—</td>
      <td><a href="https://globalenergymonitor.org/projects/global-integrated-power-tracker/" target="_blank">GEM</a></td>
    </tr>
    <tr>
      <td>Generation</td>
      <td>—</td>
      <td>Global Power Plant Database</td>
      <td>Plant details including capacity, fuel, ownership</td>
      <td>Not maintained anymore</td>
      <td><a href="https://github.com/wri/global-power-plant-database" target="_blank">GitHub</a></td>
    </tr>
    <tr>
      <td>Heating demand</td>
      <td>Hourly</td>
      <td>atlite</td>
      <td>Python library for deriving power systems data from weather</td>
      <td>—</td>
      <td><a href="https://atlite.readthedocs.io/en/latest/" target="_blank">atlite docs</a></td>
    </tr>
    <tr>
      <td>Hydro Power</td>
      <td>—</td>
      <td>EIA</td>
      <td>Total energy production of existing plants</td>
      <td>Calibration of runoff time series from reanalysis data</td>
      <td><a href="https://www.eia.gov/international/data/world" target="_blank">EIA</a></td>
    </tr>
    <tr>
      <td>Hydro Power</td>
      <td>Hourly</td>
      <td>atlite</td>
      <td>Power systems data derived from weather</td>
      <td>—</td>
      <td><a href="https://atlite.readthedocs.io/en/latest/" target="_blank">atlite docs</a></td>
    </tr>
    <tr>
      <td>Hydro Power</td>
      <td>Monthly</td>
      <td>GRDC</td>
      <td>Global river discharge and runoff data</td>
      <td>Estimate streamflow and hydropower generation</td>
      <td><a href="https://grdc.bafg.de/" target="_blank">GRDC</a></td>
    </tr>
    <tr>
      <td>Hydro Power</td>
      <td>—</td>
      <td>IRENA - Sterl et al., 2022</td>
      <td>Hydropower profiles across Africa</td>
      <td>Atlas for energy modelling</td>
      <td><a href="https://open-research-europe.ec.europa.eu/articles/1-29/v3" target="_blank">Paper</a>, <a href="https://www.hydroshare.org/resource/5e8ebdc3bfd24207852539ecf219d915/" target="_blank">Data</a></td>
    </tr>
    <tr>
      <td>Electricity demand</td>
      <td>Hourly</td>
      <td>synde</td>
      <td>Modelled demand under different SSPs using GEGIS</td>
      <td>—</td>
      <td><a href="https://github.com/euronion/synde" target="_blank">GitHub</a></td>
    </tr>
    <tr>
      <td>Solar PV</td>
      <td>Hourly</td>
      <td>Renewables.ninja</td>
      <td>—</td>
      <td>Not validated vs observed output</td>
      <td>—</td>
    </tr>
    <tr>
      <td>Solar PV</td>
      <td>—</td>
      <td>Global Solar Atlas</td>
      <td>—</td>
      <td>—</td>
      <td><a href="https://globalsolaratlas.info/" target="_blank">Atlas</a></td>
    </tr>
    <tr>
      <td>Solar PV</td>
      <td>—</td>
      <td>Autopilot for energy models</td>
      <td>—</td>
      <td>—</td>
      <td><a href="https://arxiv.org/abs/2003.01233" target="_blank">Paper</a></td>
    </tr>
    <tr>
      <td>Solar PV</td>
      <td>—</td>
      <td>pv-lib</td>
      <td>Python library for modeling PV systems</td>
      <td>—</td>
      <td><a href="https://github.com/pvlib/pvlib-python" target="_blank">GitHub</a></td>
    </tr>
    <tr>
      <td>Solar PV</td>
      <td>Hourly</td>
      <td>atlite</td>
      <td>—</td>
      <td>—</td>
      <td><a href="https://atlite.readthedocs.io/en/latest/" target="_blank">atlite docs</a></td>
    </tr>
    <tr>
      <td>Wind Power</td>
      <td>Hourly</td>
      <td>Renewables.ninja</td>
      <td>—</td>
      <td>Not validated vs observed output</td>
      <td>—</td>
    </tr>
    <tr>
      <td>Wind Power</td>
      <td>—</td>
      <td>Autopilot for energy models</td>
      <td>—</td>
      <td>—</td>
      <td><a href="https://arxiv.org/abs/2003.01233" target="_blank">Paper</a></td>
    </tr>
    <tr>
      <td>Wind Power</td>
      <td>—</td>
      <td>atlite</td>
      <td>—</td>
      <td>—</td>
      <td><a href="https://atlite.readthedocs.io/en/latest/" target="_blank">atlite docs</a></td>
    </tr>
    <tr>
      <td>Wind Power</td>
      <td>—</td>
      <td>Global Wind Atlas</td>
      <td>—</td>
      <td>—</td>
      <td><a href="https://globalwindatlas.info/en/" target="_blank">Atlas</a></td>
    </tr>
    <tr>
      <td>Solar PV + Wind</td>
      <td>—</td>
      <td>Sterl et al., 2022</td>
      <td>All-Africa dataset of supply regions for PV and wind</td>
      <td>Open-access for energy models</td>
      <td><a href="https://www.nature.com/articles/s41597-022-01786-5" target="_blank">Paper</a></td>
    </tr>
    <tr>
      <td>Solar PV + Wind</td>
      <td>—</td>
      <td>Sterl et al., 2020</td>
      <td>Smart renewable electricity portfolios in West Africa</td>
      <td>—</td>
      <td><a href="https://www.nature.com/articles/s41893-020-0539-0" target="_blank">Paper</a></td>
    </tr>
    <tr>
      <td>All</td>
      <td>—</td>
      <td>OSeMOSYS</td>
      <td>Data used in Brinkerink et al., 2021</td>
      <td>—</td>
      <td><a href="https://www.nature.com/articles/s41597-022-01737-0" target="_blank">Paper</a></td>
    </tr>
    <tr>
      <td>All</td>
      <td>—</td>
      <td>PyPSA-Earth</td>
      <td>Global electricity model workflow</td>
      <td>—</td>
      <td><a href="https://pypsa-earth.readthedocs.io/en/latest/data_workflow.html" target="_blank">Docs</a></td>
    </tr>
    <tr>
      <td>All</td>
      <td>Yearly</td>
      <td>—</td>
      <td>Global power data for 85 geographies</td>
      <td>Includes emissions and demand</td>
      <td><a href="https://ember-energy.org/data/" target="_blank">Ember</a></td>
    </tr>
    <tr>
      <td>All</td>
      <td>—</td>
      <td>ENERGYDATA.INFO</td>
      <td>Open data platform for the energy sector</td>
      <td>—</td>
      <td><a href="https://energydata.info/dataset/?organization=world-bank-grou&vocab_topics=Power+system+and+utilities" target="_blank">Data portal</a></td>
    </tr>
    <tr>
      <td>Hydro Power</td>
      <td>—</td>
      <td>FAO</td>
      <td>Geo-referenced database on dams</td>
      <td>—</td>
      <td><a href="http://www.fao.org/aquastat/en/databases/dams" target="_blank">FAO</a></td>
    </tr>
    <tr>
      <td>Hydro Power</td>
      <td>—</td>
      <td>Global Dam Watch</td>
      <td>Global Reservoir and Dam Database</td>
      <td>—</td>
      <td><a href="http://globaldamwatch.org/grand/" target="_blank">GRAND</a></td>
    </tr>
    <tr>
      <td>Hydro Power</td>
      <td>—</td>
      <td>Future Hydropower Reservoirs and Dams</td>
      <td>Dataset from Global Dam Watch</td>
      <td>—</td>
      <td><a href="http://globaldamwatch.org/fhred/" target="_blank">FHReD</a></td>
    </tr>
  </tbody>
</table>

<script>
  $(document).ready(function() {
    $('#energy-table').DataTable({
      "pageLength": 10
    });
  });
</script>