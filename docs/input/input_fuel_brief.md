# Fuel Overview

## 1. Fuel Types and Descriptions

### Coal
Coal is a solid fossil fuel formed from compressed plant matter over millions of years. It is mainly composed of carbon and used for electricity generation through combustion in steam turbines.  
- **Energy content:** 20–28 MJ/kg (~18–25 USD/MWh equivalent)  
- **Characteristics:** Abundant but carbon-intensive; local or imported depending on the country.  
- **Reference price:** [TradingEconomics – Coal](https://tradingeconomics.com/commodity/coal)  

---

### Natural Gas
Natural gas is a gaseous fossil fuel composed mainly of methane (CH₄). It is used in gas turbines, combined-cycle power plants, and sometimes in CHP systems.  
- **Energy content:** ~38–42 MJ/m³ (~1 MMBtu ≈ 293 kWh)  
- **Characteristics:** Cleaner than coal or oil; requires pipeline or LNG infrastructure.  
- **Reference price:** [TradingEconomics – Natural Gas](https://tradingeconomics.com/commodity/natural-gas)  

---

### Liquefied Natural Gas (LNG)
LNG is **natural gas cooled to -162°C** to convert it into liquid form for easier storage and transport. It must be regasified before use in power plants.  
- **Energy content:** ~50 MJ/kg (~1 MMBtu ≈ 26 kg LNG)  
- **Characteristics:** Enables international gas trade, often used where pipelines are unavailable.  
- **Difference vs. Natural Gas:** Same energy source, but includes higher delivery and infrastructure costs.  

---

### Heavy Fuel Oil (HFO)
HFO is a residual product from petroleum refining, with high viscosity and sulfur content. It is used in large thermal or diesel plants designed to burn heavy oils.  
- **Energy content:** ~40 MJ/kg  
- **Characteristics:** High CO₂ and pollutant emissions; used when gas is unavailable.  
- **Difference vs. Diesel/LFO:** Cheaper but dirtier and requires pre-heating for combustion.  

---

### Light Fuel Oil (LFO)
LFO (or light heating oil) is a refined petroleum product lighter and cleaner than HFO. It burns more efficiently and produces less particulate matter.  
- **Energy content:** ~42 MJ/kg  
- **Characteristics:** Used in small to medium-scale generators; more expensive than HFO.  
- **Difference vs. Diesel:** Similar chemical composition but lower cetane number and taxation profile; typically used for stationary applications.  

---

### Diesel
Diesel (automotive-grade distillate fuel) is used in small power plants or backup generators.  
- **Energy content:** ~43 MJ/kg  
- **Characteristics:** High-quality, clean-burning liquid fuel with a high cetane rating that enables rapid, reliable starts; very flexible but expensive.  
- **Difference vs. HFO/LFO:** Refined to tighter specifications, so it ignites instantly, runs cleaner, and requires less preheating, which is why backup generators favor it despite the higher cost.  

---

### Biomass
Biomass refers to **organic material used as fuel**, such as agricultural residues, wood chips, or dedicated energy crops.  
- **Energy content:** ~15–20 MJ/kg (depending on moisture content)  
- **Characteristics:** Renewable and potentially carbon-neutral if sustainably sourced; can be used in steam boilers or gasified for co-firing.  
- **Difference vs. Fossil Fuels:** Lower energy density and higher logistics costs, but much lower net CO₂ emissions.  

---

## 2. Fuel Price Methodology

The projection of fuel prices in EPM considers **international market prices** for each fuel type.  

- For **importing countries**, the international benchmark represents the **actual purchase cost** since the fuel must be bought on international markets.  
- For **exporting countries**, the same international price is used as a proxy for the **opportunity cost** — the value of selling the fuel abroad instead of consuming it domestically.  

### Steps to Estimate Delivered Fuel Price

1. **Start with international wholesale benchmark:**
   - Use reliable sources such as [TradingEconomics](https://tradingeconomics.com/), [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets), or [IEA](https://www.iea.org/).
2. **Add transport and delivery costs:**
   - +1–3 USD/MMBtu for LNG shipping, regasification, or long-distance pipelines.  
   - +1–2 USD/MMBtu for local distribution.  
3. **Include taxes, duties, or subsidies if applicable.**
4. **Convert to USD/MMBtu (EPM unit)** and optionally to **USD/MWh** for comparison:
   `1 MMBtu ≈ 293 kWh`, therefore `1 USD/MMBtu ≈ 3 USD/MWh`.

---

## 3. Reference Prices and Sources

| Fuel | Typical Source | Example Reference | Price (USD/MMBtu) | Equivalent (USD/MWh) |
|------|----------------|------------------:|------------------:|---------------------:|
| **Coal** | World Bank – Pink Sheet (Thermal Coal Australia) | [TradingEconomics – Coal](https://tradingeconomics.com/commodity/coal) | 3–4 | 10–14 |
| **Natural Gas** | TradingEconomics – Henry Hub / TTF / JKM | [TradingEconomics – Natural Gas](https://tradingeconomics.com/commodity/natural-gas) | 6–10 | 20–34 |
| **LNG** | IEA or JKM LNG Index | [IEA LNG Outlook](https://www.iea.org/reports/world-energy-outlook-2023) | 9–12 | 31–41 |
| **HFO** | IEA or national petroleum regulator | [IEA Oil Market Report](https://www.iea.org/reports/oil-market-report) | 12–16 | 41–55 |
| **LFO** | IEA or national market | [IEA Oil Market Report](https://www.iea.org/reports/oil-market-report) | 14–18 | 48–61 |
| **Diesel** | IEA, TradingEconomics, or local fuel prices | [TradingEconomics – Diesel](https://tradingeconomics.com/commodity/diesel) | 18–22 | 61–75 |
| **Biomass** | IRENA, FAO, or local sources | [IRENA – Bioenergy](https://www.irena.org/bioenergy) | 3–5 | 10–17 |

> These prices are indicative mid-range 2025 values and should be adjusted for inflation and local conditions.  
> For long-term projections, EPM uses escalation factors based on IEA or World Bank forecasts.
