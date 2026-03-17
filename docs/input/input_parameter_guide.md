# Typical Values

Reference values for technology and fuel parameters. For open data sources to populate these inputs, see [Open Data Sources](input_open_data.md).

---

## Technology Parameters

!!! info "Default values — CCDR methodology"
    The values below are drawn from the CCDR (Country Climate and Development Report) methodology. A [template file](dwld/pGenDataInputDefault_CCDR.csv) and the full [CCDR EEX Methodology Note](../dwld/CCDR_Methodology_Note.pdf) are available for reference.

    Work is ongoing to cross-validate and enrich these defaults against open datasets (IRENA, IEA, NREL ATB). See [Open Data Sources](input_open_data.md) for relevant resources.

<div class="scrollable-table">
  <table class="tech-table">
    <thead>
      <tr>
        <th rowspan="2">Technology</th>
        <th rowspan="2">Fuel</th>
        <th colspan="2">Min. Generation (%)</th>
        <th colspan="2">Heat Rate (MMBtu/MWh)</th>
        <th colspan="2">Ramp-up Rate</th>
        <th colspan="2">Ramp-down Rate</th>
        <th colspan="2">Max. Reserve (%)</th>
        <th colspan="2">Fixed O&M ($/MW/yr)</th>
        <th colspan="2">Variable O&M ($/MWh)</th>
        <th colspan="2">Capex (M$/MW)</th>
        <th rowspan="2">Lifetime</th>
        <th rowspan="2">Round-trip Eff.</th>
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
    <tbody>
      <tr><td>ST</td><td>Coal</td><td>25–40%</td><td>30%</td><td>7.7–9.4</td><td>8.5</td><td>30–100%</td><td>50%</td><td>30–100%</td><td>50%</td><td>0%</td><td>0%</td><td>30k–90k</td><td>60k</td><td>1.0–5.0</td><td>3</td><td>1.4–2.6</td><td>2</td><td>30</td><td>—</td></tr>
      <tr><td>ST</td><td>Lignite</td><td>50–60%</td><td>55%</td><td>9.5–11.0</td><td>10.3</td><td>30–100%</td><td>50%</td><td>30–100%</td><td>50%</td><td>0%</td><td>0%</td><td>30k–90k</td><td>60k</td><td>1.0–5.0</td><td>3</td><td>1.4–2.6</td><td>2</td><td>30</td><td>—</td></tr>
      <tr><td>OCGT</td><td>Gas</td><td>0%</td><td>0%</td><td>7.7–10.4</td><td>9</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>10–20%</td><td>20%</td><td>10k–30k</td><td>20k</td><td>3.0–5.0</td><td>4</td><td>0.56–1.04</td><td>0.8</td><td>30</td><td>—</td></tr>
      <tr><td>OCGT</td><td>HFO</td><td>0%</td><td>0%</td><td>8.4–11.4</td><td>9.9</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>10–20%</td><td>20%</td><td>10k–30k</td><td>20k</td><td>3.3–5.5</td><td>4.4</td><td>0.56–1.04</td><td>0.8</td><td>30</td><td>—</td></tr>
      <tr><td>OCGT</td><td>Diesel</td><td>0%</td><td>0%</td><td>8.4–11.4</td><td>9.9</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>10–20%</td><td>20%</td><td>10k–30k</td><td>20k</td><td>3.3–5.5</td><td>4.4</td><td>0.56–1.04</td><td>0.8</td><td>30</td><td>—</td></tr>
      <tr><td>CCGT</td><td>Gas</td><td>40–50%</td><td>45%</td><td>5.1–7.7</td><td>6.4</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>3–6%</td><td>5%</td><td>15k–45k</td><td>30k</td><td>1.0–3.0</td><td>2</td><td>0.63–1.17</td><td>0.9</td><td>30</td><td>—</td></tr>
      <tr><td>Reservoir Hydro</td><td>Water</td><td>0%</td><td>0%</td><td>—</td><td>—</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>5–50%</td><td>45%</td><td>25k–75k</td><td>50k</td><td>0.0–1.0</td><td>0.5</td><td>1.5–5.0</td><td>3.3</td><td>50</td><td>—</td></tr>
      <tr><td>ROR</td><td>Water</td><td>0%</td><td>0%</td><td>—</td><td>—</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>5–50%</td><td>40%</td><td>20k–60k</td><td>40k</td><td>0.0–1.0</td><td>0.5</td><td>1.5–4.0</td><td>2.8</td><td>50</td><td>—</td></tr>
      <tr><td>PV</td><td>Solar</td><td>0%</td><td>0%</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>0%</td><td>0%</td><td>10k–20k</td><td>15k</td><td>0</td><td>0</td><td>0.6–1.2</td><td>0.8</td><td>25</td><td>—</td></tr>
      <tr><td>Wind Onshore</td><td>Wind</td><td>0%</td><td>0%</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>0%</td><td>0%</td><td>20k–60k</td><td>40k</td><td>0</td><td>0</td><td>1.0–3.0</td><td>1.3</td><td>30</td><td>—</td></tr>
      <tr><td>Wind Offshore</td><td>Wind</td><td>0%</td><td>0%</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>0%</td><td>0%</td><td>40k–100k</td><td>70k</td><td>0</td><td>0</td><td>2.0–4.1</td><td>3</td><td>30</td><td>—</td></tr>
      <tr><td>Biomass</td><td>Biomass</td><td>0%</td><td>0%</td><td>10.0–15.0</td><td>12.5</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>3–6%</td><td>5%</td><td>50k–150k</td><td>100k</td><td>1.3–3.8</td><td>2.5</td><td>1.0–3.0</td><td>2</td><td>30</td><td>—</td></tr>
      <tr><td>Geothermal</td><td>Geothermal</td><td>—</td><td>—</td><td>0</td><td>0</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>50k–150k</td><td>100k</td><td>0</td><td>0</td><td>2.0–5.0</td><td>3.5</td><td>30</td><td>—</td></tr>
      <tr><td>Nuclear</td><td>Uranium</td><td>50–100%</td><td>75%</td><td>10.0–15.0</td><td>12.5</td><td>10–20%</td><td>15%</td><td>10–20%</td><td>15%</td><td>0%</td><td>0%</td><td>100k–200k</td><td>150k</td><td>2.1–4.9</td><td>3.5</td><td>2.8–6.5</td><td>4</td><td>50</td><td>—</td></tr>
      <tr><td>Battery</td><td>Battery</td><td>0%</td><td>0%</td><td>—</td><td>—</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>30–75%</td><td>50%</td><td>20k–60k</td><td>40k</td><td>0</td><td>0</td><td>0.20–0.40</td><td>0.3</td><td>20</td><td>85%</td></tr>
      <tr><td>Pumped Hydro</td><td>Pumped Hydro</td><td>0%</td><td>0%</td><td>—</td><td>—</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>50–100%</td><td>75%</td><td>25k–75k</td><td>50k</td><td>0.0–1.0</td><td>0.5</td><td>0.70–5.0</td><td>2.9</td><td>50</td><td>80%</td></tr>
    </tbody>
  </table>
</div>

### ST — Steam Turbine

Uses external combustion to heat water into steam (Rankine cycle). Less flexible, suited for base-load operation.

**Compatible fuels:** Coal, lignite, biomass.

High minimum generation (30–60%) due to thermal inertia of the boiler. Ramp rates are modest, heat rates moderate to high. Capital costs are high; lifetimes often exceed 40 years.

---

### OCGT — Open Cycle Gas Turbine

Simple gas turbine (Brayton cycle) with no heat recovery. Valued for fast-start flexibility and reserve capability.

**Compatible fuels:** Natural gas, LNG, diesel, HFO.

Near-zero minimum generation; can ramp fully within minutes. Heat rates are high (typically >9 MMBtu/MWh). Capital costs are low, but O&M is higher for liquid fuels (HFO requires pre-heating).

---

### CCGT — Combined Cycle Gas Turbine

Combines a gas turbine with a steam turbine that recovers exhaust heat, significantly improving efficiency.

**Compatible fuels:** Natural gas, LNG only (heat recovery system is incompatible with liquid fuels).

Best efficiency among fossil technologies (~6.4 MMBtu/MWh). Moderate flexibility with ramp rates up to 100%/h in modern designs. Capital costs higher than OCGT due to the steam cycle.

---

### ICE — Internal Combustion Engine

Reciprocating engines installed as modular blocks. Common for decentralized peaking and grid support.

**Compatible fuels:** Diesel, HFO, natural gas, LNG, biogas.

Excellent flexibility: very low minimum load per unit, fast start (minutes), frequent cycling. Heat rates ~8–9 MMBtu/MWh on gas. O&M is higher for heavy fuels.

---

### Reservoir Hydro

Dispatchable hydro with upstream storage reservoir allowing control over water flow.

Highly flexible — can ramp within seconds to minutes. In EPM, dispatch is optimized seasonally, with capacity factors constraining total energy per season. Lifetimes exceed 50 years; O&M is low.

---

### ROR — Run-of-River Hydro

Hydro plant without significant storage. Generation follows river flow with minimal control.

Non-dispatchable and cannot provide reserves. Treated as a renewable variable source in EPM. Lower capital cost than reservoir hydro; long lifetime.

---

### Nuclear

Uses nuclear fission to generate steam. Operated for steady base-load.

High minimum generation (>75%) and slow ramp rates due to thermal and safety constraints. Very high capital costs, low variable O&M. Lifetimes can exceed 60 years.

---

### Geothermal

Converts underground heat into electricity via flash or binary cycle systems.

Base-load operation only; output modulation is limited to avoid reservoir stress. High capital costs (drilling-intensive); moderate O&M; geographically constrained.

---

### Biomass

Burns organic materials through a steam turbine. Renewable if sustainably sourced.

Behaves similarly to coal-fired ST: high minimum generation, limited ramping. Fuel supply logistics are a critical planning factor.

---

### PV — Photovoltaics

Converts solar irradiance directly into electricity. Variable and non-dispatchable.

Zero minimum generation; output follows solar angle and cloud cover. Cannot provide reserves unless paired with storage. Costs have declined sharply; typical asset life 25–30 years.

---

### Wind — Onshore and Offshore

Converts wind kinetic energy into electricity. Variable and weather-dependent.

Zero minimum generation; ramping follows wind conditions, not system needs. Offshore wind offers higher capacity factors at higher capital and O&M cost.

---

### Battery Storage

Electrochemical storage (typically lithium-ion). Ideal for short-duration flexibility.

Near-instantaneous ramp response; round-trip efficiency ~85–90%. Duration limited to a few hours. Declining capital costs; lifespan 10–15 years depending on cycling patterns.

---

### Pumped Hydro Storage

Large-scale mechanical storage: pump water uphill when surplus power is available, generate when needed.

Full modulation capability with fast ramp. Round-trip efficiency 75–80%. Very high capital cost due to civil infrastructure; lifetimes exceed 50 years.

---

## Fuel Parameters

### Fuel Types

| Fuel | Energy Content | Notes |
|---|---|---|
| **Coal** | 20–28 MJ/kg | Solid, abundant, carbon-intensive |
| **Natural Gas** | 38–42 MJ/m³ | Cleanest fossil fuel; requires pipeline or LNG infra |
| **LNG** | ~50 MJ/kg | Natural gas liquefied at −162°C for transport; regasified before use |
| **HFO** | ~40 MJ/kg | Refinery residual; requires pre-heating; high emissions |
| **LFO** | ~42 MJ/kg | Lighter, cleaner than HFO; used in medium-scale generators |
| **Diesel** | ~43 MJ/kg | High-quality distillate; fast-start; most expensive liquid fuel |
| **Biomass** | 15–20 MJ/kg | Renewable if sustainably sourced; lower energy density |

### Fuel Price Methodology

EPM uses delivered fuel prices in **USD/MMBtu**. For both importing and exporting countries, the international benchmark is used — as the actual purchase cost or the opportunity cost of export, respectively.

**Steps to estimate delivered price:**

1. Start with an international benchmark (World Bank Pink Sheet, IEA, TradingEconomics)
2. Add transport and delivery: +1–3 USD/MMBtu for LNG shipping/regasification, +1–2 USD/MMBtu for local distribution
3. Include taxes, duties, or subsidies where applicable
4. Convert to USD/MMBtu: `1 MMBtu ≈ 293 kWh`, so `1 USD/MMBtu ≈ 3.4 USD/MWh`

### Reference Prices

| Fuel | Typical Source | Price (USD/MMBtu) | Equivalent (USD/MWh) |
|---|---|---:|---:|
| Coal | World Bank Pink Sheet (Australia thermal) | 3–4 | 10–14 |
| Natural Gas | Henry Hub / TTF / JKM | 6–10 | 20–34 |
| LNG | IEA or JKM index | 9–12 | 31–41 |
| HFO | IEA Oil Market Report | 12–16 | 41–55 |
| LFO | IEA or national market | 14–18 | 48–61 |
| Diesel | TradingEconomics, local prices | 18–22 | 61–75 |
| Biomass | IRENA, FAO, or local sources | 3–5 | 10–17 |

> Indicative mid-range 2026 values. Adjust for inflation, country context, and long-term escalation factors (IEA or World Bank forecasts).

---

## See also

- [Open Data Sources](input_open_data.md) — where to find technology costs, demand forecasts, and fuel prices
- [Data Preparation](../preprocessing/pre_overview.md) — pre-analysis workflows for building VRE profiles, demand profiles, and hydro availability from open datasets

