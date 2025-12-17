# Electricity Planning Model (EPM)

## Status

[![Test workflows](https://github.com/ESMAP-World-Bank-Group/EPM/actions/workflows/gams_workflow.yml/badge.svg)](https://github.com/ESMAP-World-Bank-Group/EPM/actions/workflows/gams_workflow.yml)
[![GitHub Pages](https://github.com/ESMAP-World-Bank-Group/EPM/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/ESMAP-World-Bank-Group/EPM/actions/workflows/gh-pages.yml)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
![Size](https://img.shields.io/github/repo-size/ESMAP-World-Bank-Group/EPM)



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15591290.svg)](https://doi.org/10.5281/zenodo.15591290)


Full documentation is available [here](https://esmap-world-bank-group.github.io/EPM/home.html).

## What is EPM?

EPM (Electricity Planning Model) is a least-cost power system planning tool developed by the Power Systems Planning Group at the Energy Sector Management Assistance Program (ESMAP) of the World Bank. As power system planning is a key activity for energy ministries and utilities worldwide, EPM serves as an evolving and versatile decision-making tool for World Bank staff and clients.

## Objectives

EPM helps answer critical questions for power system planners:

### Capacity Planning

- **What generation capacity should be built?** - Optimal mix of technologies (solar, wind, hydro, gas, etc.)
- **When should new capacity be built?** - Investment timeline over the planning horizon
- **Where should it be located?** - Geographic distribution across zones/countries

### System Operations

- **How will generation be dispatched?** - Hourly operation of power plants
- **What are the electricity costs?** - Generation costs by zone and technology
- **How much will the system cost?** - Total NPV of capacity expansion and operations

### Policy Analysis

- **What is the impact of renewable targets?** - Achieving minimum RE shares
- **What are the benefits of interconnections?** - Cross-border power trade
- **What are the CO2 implications?** - Emissions under different scenarios

## How It Works

EPM is formulated in the General Algebraic Modeling System (GAMS). Input data is provided via CSV files, and results are generated as CSV outputs with automatic visualizations. Knowledge of GAMS programming is not required for basic runs.

EPM minimizes the costs of expanding and operating a power system while meeting technical, economic, and environmental requirements:

- **Fixed costs**: Annualized capital and fixed O&M
- **Variable costs**: Variable O&M and fuel costs
- **Dispatch optimization**: Hourly generation scheduling
- **Geographic scope**: Multi-zone with cross-border exchanges
- **Reserve co-optimization**: Spinning reserves allocation
- **Policy constraints**: Emissions limits, fuel constraints, renewable targets, carbon prices

## Example Outputs

> **Note**: These examples are for illustrative purposes only. Numbers should not be used quantitatively.

EPM produces comprehensive outputs including:

| Output Type | Description |
|-------------|-------------|
| Capacity Mix | Evolution of generation capacity by technology |
| System Costs | NPV analysis and cost comparisons across scenarios |
| Energy Generation | GWh production by technology and zone |
| Dispatch | Hourly generation scheduling |
| Interconnections | Cross-border power flows |
| Interactive Maps | Geographic visualization of the power system |

See the [Output Examples](https://esmap-world-bank-group.github.io/EPM/docs/output/examples.html) documentation for sample visualizations.

## Deployment Process

The practical deployment of EPM consists of a 7-step process, which is illustrated below.

<img src="https://i.postimg.cc/jdZH17w9/Screenshot-2022-01-14-102001.png" alt="drawing" width="300" align="center"/>

---

## Works deploying EPM

- Suski, A., Remy, T., Chattopadhyay, D., Song, C. S., Jaques, I., Keskes, T., & Li, Y. (2021). Analyzing Electric Vehicle Load Impact on Power Systems: Modeling Analysis and a Case Study for Maldives. IEEE Access, 9, 125640-125657.
- World Bank. 2021. The Value of Trade and Regional Investments in The Pan-Arab Electricity Market : Integrating Power Systems and Building Economies. World Bank, Washington, DC. © World Bank. https://openknowledge.worldbank.org/handle/10986/36614
- Timilsina, Govinda; Deluque Curiel, Ilka; Chattopadhyay, Deb. 2021. How Much Does Latin America Gain from Enhanced Cross-Border Electricity Trade in the Short Run?. Policy Research Working Paper;No. 9692. World Bank, Washington, DC. © World Bank. https://openknowledge.worldbank.org/handle/10986/35729
- Huang, Z., Smolenova, I., Chattopadhyay, D., Govindarajalu, C., De Wit, J., Remy, T., & Curiel, I. D. (2021). ACT on RE+ FLEX: Accelerating Coal Transition Through Repurposing Coal Plants Into Renewable and Flexibility Centers. IEEE Access, 9, 84811-84827.
- Chattopadhyay, D., Chitkara, P., Curiel, I. D., & Draugelis, G. (2020). Cross-Border Interconnectors in South Asia: Market-Oriented Dispatch and Planning. IEEE Access, 8, 120361-120374.
- Remy, T., & Chattopadhyay, D. (2020). Promoting better economics, renewables and CO2 reduction through trade: A case study for the Eastern Africa Power Pool. Energy for Sustainable Development, 57, 81-97.
- Islam, M. E., Khan, M. M. Z., Chattopadhyay, D., & Väyrynen, J. (2021). Impact of COVID-19 on dispatch and capacity plan: A case study for Bangladesh. The Electricity Journal, 34(5), 106955.
- Islam, M. E., Khan, M. M. Z., Chattopadhyay, D., & Draugelis, G. (2020, August). Economic benefits of cross border power trading: A case study for Bangladesh. In 2020 IEEE Power & Energy Society General Meeting (PESGM) (pp. 1-5). IEEE.
- Spyrou, E., Hobbs, B. F., Bazilian, M. D., & Chattopadhyay, D. (2019). Planning power systems in fragile and conflict-affected states. Nature energy, 4(4), 300-310.
- World Bank Group. Poland Energy Transition: The Path to Sustainability in the Electricity and Heating Sector. World Bank, 2018.


## Citing EPM

Please cite EPM as:

Chattopadhyay, D., De Sisternes, F., Oguah, S. K. W., World Bank Electricity Planning Model (EPM) Mathematical Formulation, 2018, Energy Sector Management Assistance Program (ESMAP), International Bank for Reconstruction and Development, The World Bank, Washington DC

```bibtex
@article{
  author = {Chattopadhyay, Debabrata and De Sisternes, Fernando and Oguah, Samuel Kofi Wilson},
  title = {World Bank Electricity Planning Model (EPM) Mathematical Formulation},
  year = {2018},
  institution = {Energy Sector Management Assistance Program (ESMAP), International Bank for Reconstruction and Development, The World Bank},
  address = {Washington DC}
}
```
--- 

## License

Licensed under the Areative Commons Zero v1.0 Universal (the "License"); you
may not use this file except in compliance with the License. You may
obtain a copy of the License at <https://creativecommons.org/publicdomain/zero/1.0/>
