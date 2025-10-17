# Technology Overview

The main generation and storage technologies modeled in EPM are presented in the table below and the following paragraphs.

> In addition, we provide a [template](dwld/pGenDataInputDefault_CCDR.csv) to help you create your own technology data file. This template includes the required columns and example values for each technology. You can use it as a starting point to input your own data.

For more detailed guidelines on technology-specific assumptions, you can refer to the [CCDR EEX Methodology Note](dwld/CCDR_Note.docx).

---

### Technology data summary

<style>
  .scrollable-table {
    overflow-x: auto;
    max-width: 100%;
    margin-top: 1em;
  }

  table.tech-table {
    border-collapse: collapse;
    font-size: 0.85em;
    width: max-content;
    min-width: 100%;
  }

  table.tech-table th,
  table.tech-table td {
    border: 1px solid #ddd;
    padding: 6px 10px;
    text-align: center;
    white-space: nowrap;
    vertical-align: middle;
  }

  table.tech-table th {
    background-color: #f2f2f2;
    font-weight: 600;
  }

  table.tech-table thead tr:first-child th {
    text-align: center;
    vertical-align: bottom;
  }
</style>

<div class="scrollable-table">
  <table class="tech-table">
    <thead>
<tr>
  <th rowspan="2">Technology</th>
  <th rowspan="2">Fuel</th>
  <th colspan="2">Min. Generation (%)</th>
  <th colspan="2">Heat Rate (MMBTu/MWh)</th>
  <th colspan="2">Ramp-up Rate</th>
  <th colspan="2">Ramp-down Rate</th>
  <th colspan="2">Max. Reserve (%)</th>
  <th colspan="2">Fixed O&M ($/MW/year)</th>
  <th colspan="2">Variable O&M ($/MWh)</th>
  <th colspan="2">Capex (M$/MW)</th>
  <th rowspan="2">Lifetime</th>
  <th rowspan="2">Round-trip Efficiency</th>
</tr>

<tr>
  <th>Range</th><th>Std.</th>
  <th>Range</th><th>Std.</th>
  <th>Range</th><th>Std.</th>
  <th>Range</th><th>Std.</th>
  <th>Range</th><th>Std.</th>
  <th>Range</th><th>Std.</th>
  <th>Range</th><th>Std.</th>
  <th>Range</th><th>Std.</th>
</tr>
</thead>
    <tbody><tr><td>ST</td><td>Coal</td><td>25–40%</td><td>30%</td><td>7.7–9.4</td><td>8.5</td><td>30–100%</td><td>50%</td><td>30–100%</td><td>50%</td><td>0–0%</td><td>0%</td><td>30k–90k</td><td>60k</td><td>1.0–5.0</td><td>3</td><td>1.4–2.6</td><td>2</td><td>30</td><td>-</td></tr>
<tr><td>ST</td><td>Lignite</td><td>50–60%</td><td>55%</td><td>9.5–11.0</td><td>10.3</td><td>30–100%</td><td>50%</td><td>30–100%</td><td>50%</td><td>0–0%</td><td>0%</td><td>30k–90k</td><td>60k</td><td>1.0–5.0</td><td>3</td><td>1.4–2.6</td><td>2</td><td>30</td><td>-</td></tr>
<tr><td>OCGT</td><td>Gas</td><td>0–0%</td><td>0%</td><td>7.7–10.4</td><td>9</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>10–20%</td><td>20%</td><td>10k–30k</td><td>20k</td><td>3.0–5.0</td><td>4</td><td>0.56–1.04</td><td>0.8</td><td>30</td><td>-</td></tr>
<tr><td>OCGT</td><td>HFO</td><td>0–0%</td><td>0%</td><td>8.4–11.4</td><td>9.9</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>10–20%</td><td>20%</td><td>10k–30k</td><td>20k</td><td>3.3–5.5</td><td>4.4</td><td>0.56–1.04</td><td>0.8</td><td>30</td><td>-</td></tr>
<tr><td>OCGT</td><td>Diesel</td><td>0–0%</td><td>0%</td><td>8.4–11.4</td><td>9.9</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>10–20%</td><td>20%</td><td>10k–30k</td><td>20k</td><td>3.3–5.5</td><td>4.4</td><td>0.56–1.04</td><td>0.8</td><td>30</td><td>-</td></tr>
<tr><td>CCGT</td><td>Gas</td><td>40–50%</td><td>45%</td><td>5.1–7.7</td><td>6.4</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>3–6%</td><td>5%</td><td>15k–45k</td><td>30k</td><td>1.0–3.0</td><td>2</td><td>0.63–1.17</td><td>0.9</td><td>30</td><td>-</td></tr>
<tr><td>Stored Hydro</td><td>Water</td><td>0%</td><td>0%</td><td>-</td><td>-</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>5–50%</td><td>45%</td><td>25k–75k</td><td>50k</td><td>0.0–1.0</td><td>0.5</td><td>1.5–5.0</td><td>3.3</td><td>50</td><td>-</td></tr>
<tr><td>ROR</td><td>Water</td><td>0%</td><td>0%</td><td>-</td><td>-</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>5–50%</td><td>40%</td><td>20k–60k</td><td>40k</td><td>0.0–1.0</td><td>0.5</td><td>1.5–4.0</td><td>2.8</td><td>50</td><td>-</td></tr>
<tr><td>PV</td><td>Solar</td><td>0–0%</td><td>0%</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0–0%</td><td>0%</td><td>10k–20k</td><td>15k</td><td>0.0</td><td>0</td><td>0.6–1.2</td><td>0.8</td><td>25</td><td>-</td></tr>
<tr><td>Wind Onshore</td><td>Wind</td><td>0–0%</td><td>0%</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0–0%</td><td>0%</td><td>20k–60k</td><td>40k</td><td>0.0</td><td>0</td><td>1.0–3.0</td><td>1.3</td><td>30</td><td>-</td></tr>
<tr><td>Wind Offshore</td><td>Wind</td><td>0–0%</td><td>0%</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0–0%</td><td>0%</td><td>40k–100k</td><td>70k</td><td>0.0</td><td>0</td><td>2.0–4.1</td><td>3</td><td>30</td><td>-</td></tr>
<tr><td>Biomass</td><td>Biomass</td><td>0%</td><td>0%</td><td>10.0–15.0</td><td>12.5</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>3–6%</td><td>5%</td><td>50k–150k</td><td>100k</td><td>1.3–3.8</td><td>2.5</td><td>1.0–3.0</td><td>2</td><td>30</td><td>-</td></tr>
<tr><td>Geothermal</td><td>Geothermal</td><td>-</td><td>-</td><td>0.0</td><td>0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>50k–150k</td><td>100k</td><td>0.0</td><td>0</td><td>2.0–5.0</td><td>3.5</td><td>30</td><td>-</td></tr>
<tr><td>Nuclear</td><td>Uranium</td><td>50–100%</td><td>75%</td><td>10.0–15.0</td><td>12.5</td><td>10–20%</td><td>15%</td><td>10–20%</td><td>15%</td><td>0–0%</td><td>0%</td><td>100k–200k</td><td>150k</td><td>2.1–4.9</td><td>3.5</td><td>2.8–6.5</td><td>4</td><td>50</td><td>-</td></tr>
<tr><td>Storage</td><td>Battery</td><td>0%</td><td>0%</td><td>-</td><td>-</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>30–75%</td><td>50%</td><td>20k–60k</td><td>40k</td><td>0.0</td><td>0</td><td>0.20–0.40</td><td>0.3</td><td>20</td><td>85%</td></tr>
<tr><td>Storage</td><td>Pumped Hydro</td><td>0%</td><td>0%</td><td>-</td><td>-</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>50–100%</td><td>75%</td><td>25k–75k</td><td>50k</td><td>0.0–1.0</td><td>0.5</td><td>0.70–5.0</td><td>2.9</td><td>50</td><td>80%</td></tr>
</tbody>
  </table>
</div>

---

### ST (Steam Turbine)

**Description:** Historical thermal power plant design. Uses external combustion to heat water into steam (Rankine cycle), which drives a turbine. Less flexible but used for base-load operation.

**Compatible Fuels:** Coal, lignite, biomass.

**Fuel differences:** Coal and lignite are standard solid fuels requiring large boilers and operating with relatively low thermal efficiency. Biomass introduces variability in fuel composition and moisture content, which can affect combustion stability and ramping capability.

**Technical characteristics:**  
Steam turbines are generally inflexible due to their reliance on large boiler systems and thermal inertia. They typically operate with a high minimum generation level (30–60%), as shutting down and restarting the boiler is costly and slow. Ramp-up and ramp-down rates are modest for traditional designs, though some modern plants allow improved flexibility. Heat rates are moderate to high, reflecting relatively low efficiency, especially with lignite or biomass. Their contribution to reserves is limited unless equipped for flexible operation. Capital costs are high due to the scale of the installation, and O&M costs vary depending on fuel type, being higher for solid fuels. Plant lifetimes are long, often exceeding 40 years.

---

### OCGT (Open Cycle Gas Turbine)

**Description:** A simple gas turbine operating on the Brayton cycle. Air is compressed, mixed with fuel, combusted, and the resulting hot gases expand through a turbine to produce electricity. No heat recovery is included.

**Compatible Fuels:** Natural gas, LNG (regasified), diesel, HFO.

**Fuel differences:** Natural gas and LNG offer cleaner combustion, fast response, and low maintenance. Diesel is less clean but still suitable for fast-start backup. HFO requires preheating and atomization, resulting in slower ramping and more maintenance due to fouling and corrosion.

**Technical characteristics:**  
OCGTs are valued for their good operational flexibility and ability to provide reserve services. They can ramp up or down quickly, often within a few minutes, making them ideal for reserve and peaking applications. Minimum generation is close to zero, allowing shutdown when not needed. However, this flexibility comes at the expense of efficiency: heat rates are high (typically above 9 MMBtu/MWh), particularly when operating on liquid fuels like diesel or HFO. Despite low capital costs and relatively short construction times, O&M costs are higher when using HFO or diesel.

---

### CCGT (Combined Cycle Gas Turbine)

**Description:** Combines a gas turbine (Brayton cycle) with a steam turbine (Rankine cycle) that recovers heat from the exhaust, significantly increasing overall efficiency. Commonly used for base or mid-merit load.

**Compatible Fuels:** Natural gas, LNG (regasified).

**Fuel differences:** Only clean gaseous fuels are suitable due to the sensitivity of the heat recovery system. Use of liquid fuels is not compatible.

**Technical characteristics:**  
CCGTs offer a balance between efficiency and moderate flexibility. Their heat rates are lower than OCGTs (typically around 6.4 MMBtu/MWh), making them one of the most efficient fossil-based technologies. Minimum generation levels are lower than steam-only plants, especially in modern flexible configurations, and ramp rates can be relatively fast (up to 100%/h) depending on plant design. While they are less responsive than OCGTs, CCGTs can still contribute to secondary reserve services. Capital costs are higher due to the additional steam cycle, but O&M costs remain moderate. Plant lifetimes are typically 30–40 years.

---

### Reservoir Hydro

**Description:** Hydropower plants with an upstream reservoir allowing some control of water flow. This enables dispatchable and flexible electricity production.

**Fuel:** Water.

**Technical characteristics:**  
Reservoir hydro offers operational flexibility. Generation can be ramped up or down very quickly, typically within seconds to minutes, allowing the plant to contribute effectively to all types of reserve services. Minimum generation can be set to zero when the plant is not operating, and overload capacity is generally high. While capital costs are significant due to dam and civil works, O&M costs are low, and plant lifetimes can exceed 50 years. Reservoir hydro is fully renewable and often plays an important role in balancing variable generation. In EPM, reservoir hydro dispatch is optimized at the seasonal level, which can be defined by the user — typically monthly or quarterly (e.g., dry/wet seasons). Each season is assigned a capacity factor that constrains the total energy the plant can generate over that period. However, hourly dispatch within the season remains optimized to best match system needs.

---

### ROR (Run-of-River Hydro)

**Description:** Hydropower plants without significant storage. Electricity generation depends directly on river flow, with minimal control over timing.

**Fuel:** Water.

**Technical characteristics:**  
Run-of-river hydro is a renewable and not dispatchable, as generation follows the natural flow of the river. This limits flexibility and the ability to provide reserve services. Minimum generation is effectively determined by the river’s flow and cannot be adjusted. Capital costs are lower than reservoir-based hydro, and O&M costs are minimal. Lifetimes are long.

---

### Nuclear

**Description:** Uses nuclear fission to generate heat, which is converted to steam to drive a turbine.

**Fuel:** Uranium.

**Technical characteristics:**  
Nuclear plants are generally used for steady operation and have limited operational flexibility. Minimum generation levels are high (typically above 75%), and ramping is slow due to thermal constraints, economic and safety considerations. Heat rates are high compared to fossil gas technologies, a result of the thermodynamic limits of the steam cycle. While variable O&M costs are relatively low, capital costs are extremely high, with fixed O&M costs sometimes substantial due to regulatory and safety requirements, but plant lifetimes can exceed 60 years.

---

### Geothermal

**Description:** Converts underground heat into electricity, either by using dry steam directly or via steam generated from high-pressure hot water (flash or binary systems).

**Fuel:** Geothermal heat.

**Technical characteristics:**  
Geothermal plants are typically operated as base-load units. Due to the need to maintain stable reservoir pressure and avoid thermal stress on the wells, output modulation is limited. Minimum generation levels are high and ramping is slow. Thermal efficiency is relatively low, resulting in high heat rates. Contribution to reserves is minimal. While capital costs—especially for drilling—are significant, O&M costs are moderate and lifetimes are long. Geothermal energy is renewable but geographically constrained.

---

### Biomass

**Description:** Burns organic materials to generate steam and produce electricity, usually through a conventional steam turbine setup.

**Fuel:** Biomass (wood, agricultural residues, waste).

**Technical characteristics:**  
Biomass plants behave similarly to coal-based steam turbines, with high minimum generation levels and limited ramping capability. Heat rates are high due to combustion inefficiencies and the variability in biomass composition and moisture content. Reserve contribution is typically low. Capital costs are high and O&M costs are significant, particularly because of the need to handle and process fuel. Lifetimes are long. Biomass is considered renewable when sourced sustainably, but fuel supply logistics are a critical factor.

---

### PV (Photovoltaics)

**Description:** Converts solar radiation directly into electricity using photovoltaic cells. No moving parts and no combustion.

**Fuel:** Solar irradiance.

**Technical characteristics:**  
PV generation is renewable and fully dependent on sunlight and thus inherently variable and non-dispatchable. Minimum generation is zero (e.g., at night), and ramp rates are not controlled by the system but by cloud cover and solar angle. PV systems cannot contribute to reserves unless paired with battery storage. Capital costs have fallen significantly, O&M costs are very low, and typical asset life is 25–30 years.

---

### Wind (Onshore and Offshore)

**Description:** Converts kinetic energy of the wind into electricity via rotating blades connected to a generator. Offshore installations benefit from more stable wind conditions.

**Fuel:** Wind.

**Technical characteristics:**  
Wind turbines produce variable power depending on wind speed and direction. Minimum generation is zero and ramping cannot be controlled—power may drop or spike quickly based on weather. These renewable-sourced systems generally do not contribute to reserves unless supported by advanced controls or storage. Offshore wind offers higher capacity factors but entails higher capital and maintenance costs.

---

### Storage - Battery

**Description:** Electrochemical systems (typically lithium-ion) that store electricity and release it as needed. Ideal for short-duration flexibility and reserve provision.

**Fuel:** Electricity.

**Technical characteristics:**  
Battery storage offers flexibility. It can ramp up or down almost instantaneously, provide fast-response reserves, and operate at full flexibility down to zero output. Round-trip efficiency is high (typically 85–90%). However, storage duration is generally limited to a few hours. Capital costs per MWh stored remain high, although declining, while O&M costs are relatively low. Lifespans depend on usage and cycling patterns, usually ranging from 10 to 15 years.

---

### Storage - Pumped Hydro

**Description:** Uses electricity to pump water to a higher reservoir, then generates electricity by releasing the water through turbines.

**Fuel:** Electricity (stored mechanically).

**Technical characteristics:**  
Pumped hydro provides large-scale, long-duration energy storage. It can be fully modulated, with zero minimum generation and fast ramping both up and down. Its round-trip efficiency is typically between 75% and 80%. While capital costs are very high due to infrastructure, O&M costs are low, and lifetimes often exceed 50 years.
