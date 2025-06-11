
# Mathematical Formulation

## Introduction

The World Bank’s Electricity Planning Model (EPM) is a long-term, multi-year, multi-zone capacity expansion and dispatch model. The objective of the model is to minimize the sum of fixed (including annualized capital costs) and variable generation costs (discounted for time) for all zones and all years considered, subject to:

- Demand equals the sum of generation and non-served energy,
- Available capacity is existing capacity plus new capacity minus retired capacity,
- Generation does not exceed the max and min output limits of the units,
- Generation is constrained by ramping limits,
- Spinning reserves are committed every hour to compensate forecasting errors,
- renewable generation constrained by wind and solar hourly availability,
- Excess energy can be stored in storage units to be released later or traded between the other zones, and
- Transmission network topology and transmission line thermal limits.

The model is an abstract representation of the real power systems with certain limitations described in more detail in section 1.5.

EPM is used mostly to perform least cost expansion plans as well as dispatch analyses. Developed by ESMAP since 2015, EPM has been used to study dozens of countries. EPM uses a core mixed integer programming multi-zone model, implemented in the General Algebraic Modelling System ([GAMS](https://www.gams.com/)), to minimize the total discounted costs of capex and opex. The model optimizes the generation (and transmission if included) expansion over future years of the modeled time period as well as the underlying dispatch of generation and power flows on transmission lines for existing and new transmission assets. In addition to the core decision variables on generation and transmission capacity addition, dispatch, and flow, the model also co-optimizes spinning reserve provision from the generators. EPM had been parameterized for 87 client countries as of end-January 2022 (Table 1) and is being used for many types of analyses.

**Table 1: Regions and Countries with EPM Models**

| **Region/Country** | **Number of countries** | **Country names<sup>a</sup>** |
| --- | --- | --- |
| Central Asia + Pakistan | 7   | Afghanistan, Kazakhstan, Kyrgyz Rep., Tajikistan, Turkmenistan, Uzbekistan, Pakistan |
| Eastern Africa Power Pool + Zambia | 14  | Burundi; Congo, Dem. Rep.; Egypt, Arab Rep.; Libya; Djibouti; Eritrea; Ethiopia; Kenya; Rwanda; Tanzania; Uganda; South Sudan; Sudan, Zambia |
| Southern Africa Power Pool | 12  | Angola; Botswana; Congo, Dem. Rep.; Eswatini; Lesotho; Mozambique; Malawi; Namibia; South Africa; Tanzania; Zambia; Zimbabwe |
| Pan Arab Electricity Model | 17  | Algeria; Bahrain; Egypt, Arab Rep.; Iraq; Jordan; Kuwait; Lebanon; Libya; Morocco; Oman; Qatar; Saudi Arabia; Sudan; Syria; Tunisia; United Arab Emirates; West Bank and Gaza; Yemen |
| South Asia | 5   | Bangladesh, Bhutan, India, Nepal, Sri Lanka (BBINS) |
| West Africa Power Pool | 15  | Benin, Burkina Faso, Côte d’Ivoire, The Gambia, Guinea, Guinea Bissau, Ghana, Niger, Nigeria, Liberia, Mali, Mauritania, Togo, Senegal, Sierra Leone |
| Country-specific models<sup>b</sup> | 18  | Albania, Bosnia and Herzegovina, Bulgaria, Cameroon, Central African Republic, Chad, Comoros, Kosovo, Lao PDR, Madagascar, Mongolia, Myanmar, Papua New Guinea, Poland, Solomon Islands, Turkey, Ukraine, Vietnam |
| TOTAL | 88  |     |

a. The Democratic Republic of Congo is a member of both Southern and Eastern Africa Power Pools. Zambia is not a member of the Eastern Africa Power Pool but has been added to that group of countries.

b. These represent 18 stand-alone models for the individual countries.

This version is an update of the [previous documentation](https://www.researchgate.net/publication/325534590_World_Bank_Electricity_Planning_Model_EPM_Mathematical_Formulation_World_Bank_Electricity_Planning_Model). The main changes relate to the addition of new technologies (CCS, PV with storage) or to the improvement of existing technologies (storage in particular), and to the addition of new features regarding imports and exports with zones outside of the model and of transmission expansion capabilities, of VRE representation, etc. Additionally, the code has been updated to be more efficient (without changing the equations but rewriting it in a more efficient GAMS syntax).

### Modeling Assumptions

The model is derived based on the following assumptions:

1. Market participants are not strategic and they behave in a perfectly competitive manner, i.e., the power plant owners submit their true costs as bids.
2. The forecasted demand is considered perfectly inelastic, which implies that the maximization of the social welfare can be replaced by minimization of the system cost.
3. The trade among regions/countries is economically efficient (optimal), which translates to a single objective of minimization of cost for the whole region/country.
4. The pricing is efficient and does not provide incentives to market participants to deviate from the optimal behavior.

### Outcomes

- Identify investment needs (generation mostly) to meet the demand under various policy scenarios
- Determine where and by how much renewable resources should be deployed to maximize their value to the system.
- Determine the optimal capacity additions over time to complement renewable generation accounting for existing generating units, energy storage, demand-side response and/or a carbon constraint.
- Determine the optimal retirement schedule of the existing units over time.
- Assess the utilization of the transmission lines (important to design trade contracts).
- Determine the optimal transmission capacity additions.
- Determine hourly electricity prices with trade for different countries and zones, which is essential to value the energy traded.
- Determine the operations and expansion cost to the consumer.
- Determine the impact of different market conditions (e.g., fuel prices, fuel subsidies, carbon limits, etc.) and technology cost assumptions on the optimal capacity expansion plan and the optimal energy mix.
- Determine the cost of implementing specific environmental policies: renewable portfolio standard, cap on carbon emissions, tax on carbon emissions, and carbon emissions rate.
