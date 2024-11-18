
# Mathematical formulation

_Last update: 2023_

## Contents

- [Introduction](#introduction)
- [Modeling Assumptions](#modeling-assumptions)
- [Outcomes](#outcomes)
- [Notation](#notation)
  - [Indices and Sets](#indices-and-sets)
  - [Variables](#variables)
  - [Parameters](#parameters)
- [Model Formulation](#model-formulation)
- [Description of the Model](#description-of-the-model)
  - [Indices and Sets](#indices-and-sets)
  - [Objective Function](#objective-function)
  - [Load Approximation](#load-approximation)
  - [Value of Lost Load](#value-of-lost-load)
  - [Transmission Network Constraints](#transmission-network-constraints)
  - [System Requirements](#system-requirements)
  - [Generation Constraints](#generation-constraints)
  - [Renewable Generation Modeling](#renewable-generation-modeling)
  - [Concentrated Solar Power (CSP) Modeling](#concentrated-solar-power-csp-modeling)
  - [Time Consistency of Power System Additions and Retirements](#time-consistency-of-power-system-additions-and-retirements)
  - [Storage Modeling](#storage-modeling)
  - [Investment Constraints](#investment-constraints)
  - [Environmental Policy](#environmental-policy)
- [Limitations of the Model](#limitations-of-the-model)

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

EPM is used mostly to perform least cost expansion plans as well as dispatch analyses. Developed by ESMAP since 2015, EPM has been used to study dozens of countries. EPM uses a core mixed integer programming multi-zone model, implemented in the General Algebraic Modelling System ([GAMS](https://www.gams.com/)), to minimize the total discounted costs of capex and opex. The model optimizes the generation (and transmission if included) expansion over future years of the modeled time period as well as the underlying dispatch of generation and power flows on transmission lines for existing and new transmission assets. In addition to the core decision variables on generation and transmission capacity addition, dispatch, and flow, the model also co-optimizes spinning reserve provision from the generators. EPM had been parameterized for 87 client countries and economies as of end-January 2022 (Table 1) and is being used for many types of analyses.

**Table 1: Countries and Economies with EPM Models**

| **Region/Country** | **Number of economies** | **Country/economy names<sup>a</sup>** |
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

## Notation

### Indices and Sets

| Indices    | Set |
| --- | --- |
| $c \in C$   | where $C$ is the set of countries |
| $d \in D$     | where $D$ is the set of types of days or weeks |
| $f \in F$    | where $F$ is the set of fuels |
| $g \in G$     | where $G$ is the set of generators that can be built or the set of technology-specific types of aggregated generators |
|  $q \in Q$    | where $Q$ is the set of seasons or quarters |
|  $h \in H$    | where $H$ is the set of electrolyzers that can be built |
|  $y \in Y$   | where $Y$ is the set of years considered in the planning model |
| $z, z_2 \in Z$    | where  $Z$is the set of zones/regions modeled |
|  $sc \in S$   | where $S$ is the set of flags and penalties used to include/exclude certain features of the model |
| **Subsets considered** |     |
| $EN, NG \in G$    | where $EG$ and $NG$ is a partition of set $G$ and the former (EG) contains generators existing at the starting year of the planning horizon and the latter contains candidate generators<sup><sup>[\[1\]](#footnote-2)</sup></sup> |
| $DC and NDC \in G$    | where $DC$ and $NDC$ are partitions of set $G$ separating generators with a CAPEX cost constant over the modeling horizon and those generators that have costs varying over time. This feature is mainly developed to account for technologies that show cost reductions due to technological improvements and manufacturing advances. |
| $EH, NH \in H$    | where $EH$ and $NH$ is a partition of $H$ set and the former ($HG$) contains electrolyzers existing at the starting year of the planning horizon and the latter contains candidate electrolyzers<sup><sup>[\[2\]](#footnote-3)</sup></sup> |
| $DH and NDH \in H$    | where $DH$ and $NDH$ are partitions of set H separating electrolyzers with a CAPEX cost constant over the modeling horizon and those electrolyzers that have costs varying over time. This feature is mainly developed to account for technologies that show cost reductions due to technological improvements and manufacturing advances. |
| $so \in G$    | PV unit linked to a storage unit as a partition of set $G$ (this is the PV part of PV linked with storage) |
| $stp \in st$    | Storage unit linked to a PV plant as a partition of set $G$ (this is the storage part of PV linked with storage) |
| $MD \in D$    | where $MD$ is a subset of days the planner expects the minimum load levels to be binding |
| $RE \in F$    | where $F$ is a subset of set considered as renewable according to regulator’s criteria<sup><sup>[\[3\]](#footnote-4)</sup></sup> |
| $RG \in G$    | |
| $map_{c,z}$    | includes the subset of zones that correspond to each country |
|  $map_{g,f}$   | includes valid combinations of fuels and generators; subset of the set $G x F$|
| $map_{eg,ef}$    | includes combinations of existing and candidate generators; subset of set $EG x NG$ |
| $map_{g,z}$    | includes the subset of generators that correspond to each zone|
| $map_{h,z}$    | includes the subset of electrolyzers that correspond to each zone |


### Variables

| **Non-negative decision variables** |     |
| ---  | --- |
| $𝐴𝑑𝑑𝑖𝑡𝑖𝑜𝑛𝑎𝑙𝑇𝑟𝑎𝑛𝑠𝑓𝑒𝑟_{𝑧,𝑧2,𝑦}$    | Additional number of lines between z and z2 in year y |
|  $𝑏𝑢𝑖𝑙𝑑𝑔,𝑦$   | Generator investment in MW |
| $𝑏𝑢𝑖𝑙𝑑𝐻2ℎ,𝑦$    | Electrolyzer investment in MW |
| $𝑏𝑢𝑖𝑙𝑡𝐶𝑎𝑝𝑉𝑎𝑟𝑔,𝑦$    | Integer variable to model discrete unit capacity |
| $𝑏𝑢𝑖𝑙𝑡𝐶𝑎𝑝𝑉𝑎𝑟𝐻2ℎ,𝑦$    | Integer variable to model discrete unit capacity |
| $𝑏𝑢𝑖𝑙𝑡𝑇𝑟𝑎𝑛𝑠𝑚𝑖𝑠𝑠𝑖𝑜𝑛𝑧,𝑧2,𝑦$    | Integer variable to model number of new transmission lines added between z and z2 in year y |
| $𝑐𝑎𝑝𝑔,𝑦$    | Generator capacity available at year in MW |
| $𝑐𝑎𝑝𝐻2ℎ,𝑦$    | Electrolyzer capacity available at year in MW |
| $𝐶𝑂2𝑏𝑎𝑐𝑘𝑠𝑡𝑜𝑝𝑐,𝑦$    | Annual country CO2 emissions above the system emissions constraint cap |
| $𝑒𝑚𝑖𝑠𝑠𝑖𝑜𝑛𝑠𝑧,𝑦$    | Total system emissions of carbon dioxide in tons |
| $𝑒𝑚𝑖𝑠𝑠𝑖𝑜𝑛𝑠_𝑍𝑜𝑧,𝑦$    | Emissions of carbon dioxide in tons per zone |
| $𝑒𝑥𝑝𝑜𝑟𝑡𝑃𝑟𝑖𝑐𝑒𝑧,𝑞,𝑑,𝑡,𝑦$    | External price driven export in MW |
| $𝑓𝑢𝑒𝑙𝐻2𝑄𝑢𝑎𝑟𝑡𝑒𝑟𝑧,𝑞,𝑦$    | Hydrogen fuel consumption for electricity production hydrogen produced intended for re-circulation in the power sector in ( MMBTU per quarter and year) |
| $𝑓𝑢𝑒𝑙𝐻2𝑧,𝑞,𝑦$    | Hydrogen fuel consumption for electricity production hydrogen produced intended for re-circulation in the power sector in ( MMBTU per year) |
| $𝑓𝑢𝑒𝑙𝑧,𝑓,𝑦$    | Fuel consumption in MMBTU |
| $𝑔𝑒𝑛𝑔,𝑓,𝑞,𝑑,𝑡,𝑦$    | Generator output in MW |
| $𝑔𝑒𝑛𝐶𝑆𝑃𝑔,𝑧,𝑞,𝑑,𝑡,𝑦$    | Power output of the solar panel in MW |
| $𝑖𝑚𝑝𝑜𝑟𝑡𝑃𝑟𝑖𝑐𝑒𝑧,𝑞,𝑑,𝑡,𝑦$    | External price driven import in MW |
| $𝐻2𝑃𝑤𝑟𝐼𝑛ℎ,𝑞,𝑑,𝑡,𝑦$    | Power drawn by electrolyzer h in MW |
| $𝑟𝑒𝑡𝑖𝑟𝑒𝑔,𝑦$    | Capacity in MW retired |
| $𝑟𝑒𝑡𝑖𝑟𝑒𝐶𝑎𝑝𝑉𝑎𝑟𝑔,𝑦$    | Supplementary integer variable to model discrete unit capacity |
| $𝑟𝑒𝑡𝑖𝑟𝑒𝐶𝑎𝑝𝑉𝑎𝑟𝐻2ℎ,𝑦$    | Supplementary integer variable to model discrete unit capacity |
| $𝑟𝑒𝑠𝑒𝑟𝑣𝑒𝑔,𝑞,𝑑,𝑡,𝑦$    | Spinning reserve requirement met in MW |
| $𝑠𝑡𝑜𝑟𝑎𝑔𝑒𝑠𝑡,𝑞,𝑑,𝑡,𝑦$    | Level of energy in MWh stored in storage unit |
| $𝑠𝑡𝑜𝑟𝑎𝑔𝑒_𝑖𝑛𝑗𝑠𝑡,𝑞,𝑑,𝑡,𝑦$    | Power injected in MW in storage unit is charged during hour |
| $𝑠𝑡𝑜𝑟𝑎𝑔𝑒_𝑐𝑎𝑝𝑎𝑐𝑖𝑡𝑦𝑠𝑡,𝑞,𝑑,𝑡,𝑦$    | Total deployed energy storage capacity in MWh for storage unit |
| $𝑠𝑡𝑜𝑟𝑎𝑔𝑒𝐶𝑆𝑃𝑔,𝑧,𝑞,𝑑,𝑡,𝑦$    | Level of energy in MWh stored in CSP unit at zone |
| $𝑠𝑡𝑜𝑟𝑎𝑔𝑒𝐶𝑆𝑃𝑖𝑛𝑗𝑔,𝑧,𝑞,𝑑,𝑡,𝑦$    | Power level in MW at which the CSP storage unit is charged during hour |
| $𝑠𝑡𝑜𝑟𝑎𝑔𝑒𝐶𝑆𝑃𝑜𝑢𝑡𝑔,𝑧,𝑞,𝑑,𝑡,𝑦$    | Power level in MW at which the CSP storage unit is discharged during hour |
| $𝑠𝑢𝑟𝑝𝑙𝑢𝑠𝑧,𝑞,𝑑,𝑡,𝑦$    | Surplus generation in MW |
| $𝑢𝑛𝑚𝑒𝑡𝐷𝑒𝑚𝑧,𝑞,𝑑,𝑡,𝑦$    | Annual system CO2 emissions above the system emissions constraint cap |
| $𝑢𝑛𝑚𝑒𝑡𝐻2𝐸𝑥𝑡𝑒𝑟𝑛𝑎𝑙𝑧,𝑞,𝑦$    | Active power in MW flowing from to |
| $𝑢𝑛𝑚𝑒𝑡𝑅𝑒𝑠𝑧,𝑦$    | Unmet demand in MW (or equivalently violation of the load balance constraint) |
| $𝑢𝑛𝑚𝑒𝑡𝑆𝑅𝑒𝑠𝑍𝑜𝑧,𝑞,𝑑,𝑡,𝑦$    | Unmet quantity of hydrogen in MW (or equivalently violation of the external demand for hydrogen balance constraint) |
| $𝑢𝑛𝑚𝑒𝑡𝑆𝑅𝑒𝑠𝑆𝑌𝑞,𝑑,𝑡,𝑦$    | Violation of the planning reserve constraint in MW |
| $unmetSResZo$    | Violation of the zonal/regional spinning reserve constraint in MW |
| $unmetSResSY$    | Violation of the system-level spinning reserve constraint in MW |
| **Variables for modeling objective function** |     |
| $\text{npvcost} = \sum_{z,y} \text{ReturnRate}_y \cdot \text{WeightYear}_y \cdot \text{totalcost}_{z,y}$    | Annualized capex for new generation capacity installed with varying total constructions cost when built in year _y_ carried through planning horizon |
| $\text{totalcost}_{z,y} = \text{fixedcost}_{z,y} + \text{variablecost}_{z,y} + \text{reservecost}_{z,y} + \text{usecost}_{z,y} + \text{usrcost}_{z,y} + \text{carboncost}_{z,y} + \text{CO2backCost}_y + \text{TradeCost}_{z,y} + \text{CostTransmissionAdditions}_{z1,z2,y} + \text{CurtailmentCost}_{z,y} + \text{SurplusCost}_{z,y} + \text{usrcostH2}_{z,y}$ | Annualized capex for new electrolyzer capacity installed with varying total constructions cost when built in year _y_ carried through planning horizon |
|     | Carbon tax payments by generators |
|     | Amount of emissions above meeting target in tons |
|     | Penalty for missing to comply with carbon constraints in USD |
|     | Cost of new transmission capacity in USD |
|     | Curtailed energy in MWh |
|     | Fixed Operation and Maintenance Cost along with capital payments in constant prices |
|     | Net present value of power system cost over the whole planning horizon; objective function that optimization model tries to minimize |
|     | Cost to procure spinning reserves |
|     | Annual system cost in constant prices |
|     | Cost of surplus energy to the system in USE |
|     | Surplus energy in MW |
|     | Damage/economic loss in constant prices because of unmet demand |
|     | Penalty in constant prices for unmet spinning reserve requirements |
|     | Variable cost including fuel and variable operation and maintenance cost in constant prices |

### Parameters

|     | Availability of unit to generate power in quarter |
| --- | --- |
|     | Availability of electrolyzer to produce hydrogen in quarter |
|     | Maximum amount of MW allowed to be built per year |
|     | Capital cost of unit g in USD $ or other monetary unit per MW |
|     | Capital cost of electrolyzer h in USD $ or other monetary unit per MW |
|     | Percentage of generator’s stable (firm) capacity per year |
|     | Capital cost variation index for new generators to indicate how new construction costs change during the planning horizon |
|     | Capital cost variation index for new electrolyzers to indicate how new construction costs change during the planning horizon |
|     | Equivalent tons of emitted per MMBTU of fuel consumed |
|     | Carbon price in USD$ per equivalent tons of |
| CostH2Unserved | Penalty for unserved hydrogen ($/mmBTU) |
|     | Earliest commission year for generators |
|     | Earliest commission year for electrolyzers |
| _CO2backstopPenalty_ | Penalty per ton of CO2 above expected target |
|     | Cost of the last technology the system can build to reduce to the emissions caps set in the parameters, in $ per equivalent tons |
|     | Capital Recovery factor for new generators |
|     | Capital Recovery factor for new electrolyzers |
|     | Capital Recovery factor for new transmission lines |
|     | Maximum CSP storage capacity in MWh |
| CurtailmentPenalty | Penalty per MWh of VRE curtailed |
|     | Hourly load level in MW in hour t, day d, quarter q and year y |
|     | Discrete capacity of the unit for generator |
|     | Discount rate; real or nominal if cost parameters in real or nominal terms respectively |
|     | Duration of each time slice (block) in hours |
|     | Demand scaling factor for energy efficiency measures |
|     | Electrolyzer efficiency in mmBTU of hydrogen produced per MWh of electricity |
|     | External demand for hydrogen per quarter and year (hydrogen produced for use outside the power sector) (mmBTU) |
|     | Efficiency of the CSP solar field |
|     | Fixed Operation and Maintenance Cost in USD $ or other monetary unit per MW per generator capacity |
|     | Fixed Operation and Maintenance Cost in USD $ or other monetary unit per MW of electrolyzer capacity |
|     | Fuel price in USD $/MMBTU |
|     | Generator’s existing/available capacity in MW at initial year |
|     | Electrolyzer’s existing/available capacity in MW at initial year |
|     | Generation variable cost (fuel and VOM) in USD $ or other monetary unit per MWh |
|     | Contains the zone index of the zone the generator belongs to |
|     | Heat Rate in MMBTU/MWh |
|     | Operating life for new generators |
|     | Operating life for new electrolyzers |
|     | CAPEX of line connecting zones z1 and z2 |
|     | “Linearized” loss factor in % of active power flowing on transmission line |
|     | Maximum amount of annualized capital payments in USD$ billion over the horizon |
|     | Maximum share of the hourly price driven export in terms of demand |
|     | Maximum amount of fuel f (in million MMBTU) that can be consumed in year y |
|     | Maximum share of the hourly price driven import in terms of demand |
|     | Maximum capacity to be built over the horizon in MW |
|     | Maximum number of transmission lines built over the horizon |
|     | Minimum capacity factor (to reflect minimum load requirements) |
|     | Overload factor of generator _g_, as %, of capacity |
|     | Planning reserve margin per zone |
|     | Renewable power used for hydrogen production in MW |
|     | Renewable power which was used for electricity end use sectors (MW) |
|     | Ramp-down capability of generator _g_, as %, of capacity installed |
|     | Ramp-up capability of generator _g,_ as %, of capacity installed |
|     | Ramp-down capability of electrolyzer h, as %, of capacity installed |
|     | Ramp-up capability of generator h_,_ as %, of capacity installed |
|     | Renewable generation that is finally headed for hydrogen production (MW) |
|     | Renewable generation that is finally headed toward electricity end use sectors (MW) |
|     | Cost to provide reserves in USD $ or other monetary unit per MWh |
|     | Percentage of the generator’s unit qualify as a reserve offer |
|     | Violation penalty of planning reserve requirement in $/other monetary unit per MW |
|     | Latest retirement year for existing generators |
|     | Latest retirement year for existing electrolyzers |
|     | Discount factor at the starting year of stage ending at year |
|     | Renewable generation profile in % of installed (rated) capacity |
|     | CSP output to solar field ratio |
|     | System-level spinning reserve constraint in MW |
|     | Zonal/regional spinning reserve constraint in MW |
|     | Duration of a stage represented by year y in years |
|     | First year of the horizon |
|     | First year of generator’s operation |
|     | Efficiency of storage unit st (per charging cycle) |
|     | Maximum energy capability of storage unit |
| _SurplusPenalty_ | Penalty per MWh of surplus energy |
|     | Cap on emissions within the system at year in equivalent tons |
|     | Network topology: contains 0 for non-existing lines and 1 or -1 to define the direction of positive flow over the line |
|     | Transfer limits by quarter (seasonal) and year between zones |
|     | Transmission capacity of each new line in MW |
|     | Efficiency of the CSP power block |
|     | Variable Operation and Maintenance Cost in USD $ or other monetary unit per MWh electricity produced |
|     | Variable Operation and Maintenance Cost in USD $ or other monetary unit per mmBTU of hydrogen produced |
|     | Penalty/Economic loss consider per MWh of unmet demand |
| _VREforecastingerror_ | Percentage error in VRE forecast \[used to estimated required amount of spinning reserve\] |
|     | Weighted Average Cost of Capital |
|     | Weight on years |
|     | Cap on emissions within country c and year in equivalent tons |
|     | Index of zone , unique number assigned to zone _z_ |
|     |     |
|     |     |

## Model formulation

### Objective function and its component

$$
\text{npvcost} = \sum_{z,y} (\text{ReturnRate}_y \cdot \text{WeightYear}_y \cdot \text{totalcost}_{z,y})
$$

$$
\text{totalcost}_{z,y} = \text{fixedcost}_{z,y} + \text{variablecost}_{z,y} + \text{reservecost}_{z,y} + \text{usecost}_{z,y} + \text{usrcost}_{z,y} + \text{carboncost}_{z,y} + \text{CO2backCost}_y + \text{TradeCost}_{z,y} + \text{CostTransmissionAdditions}_{z1,z2,y} + \text{CurtailmentCost}_{z,y} + \text{SurplusCost}_{z,y} + \text{usrcostH2}_{z,y}
$$

$$
\text{fixedcost}_{z,y} = \sum_{g \in \text{NG} \cap \text{NDC}} (\text{CRF}_{\text{NG}} \cdot \text{CapCost}_{\text{NG},y} \cdot \text{cap}_{g,y}) +
\sum_{g \in \text{NG} \cap \text{DC}} \text{AnnCapex}_{\text{NG},y} +
\sum_g (\text{FixedOM}_{g,y} \cdot \text{cap}_{g,y}) +
\sum_{h \in \text{NH} \cap \text{NDH}} (\text{CRF}_{\text{NH}} \cdot \text{CapCostH2}_{\text{NH},y} \cdot \text{cap}_{h,y}) +
\sum_{h \in \text{NH} \cap \text{DH}} \text{AnnCapexH2}_{\text{NH},y} +
\sum_h (\text{FixedOMH2}_{h,y} \cdot \text{capH2}_{h,y})
$$

$$
\text{AnnCapex}_{g \in \text{DC},y} = \text{CFR}_{g \in \text{DC}} \cdot \text{CapCost}_{g \in \text{DC}} \cdot \text{CapexTrajectories}_{g \in \text{DC},y} \cdot \text{build}_{g \in \text{DC},y}
$$

$$
\text{AnnCapexH2}_{h \in \text{DH},y} = \text{CFR}_{h \in \text{DH}} \cdot \text{CapCost}_{h \in \text{DH}} \cdot \text{CapexTrajectories}_{h \in \text{DH},y} \cdot \text{build}_{h \in \text{DH},y}
$$

$$
\text{variablecost}_{z,y} = \sum_{g \in \text{G},f,q,d,t} (\text{GenCost}_{g,f,y} \cdot \text{Duration}_{q,d,t,y} \cdot \text{gen}_{g,f,q,d,t,y}) +
\sum_{h \in \text{H},q,d,t} (\text{VOM}_{\text{H2},h,y} \cdot \text{EfficiencyH2}_h \cdot \text{vH2PwrIn}_{h,g,q,d,t,y} \cdot \text{pHours}_{q,d,t,y})
$$

$$
\text{reservecost}_{z,y} = \sum_{g \in \text{Z},q,d,t} (\text{ResCost}_g \cdot \text{Duration}_{q,d,t,y} \cdot \text{reserve}_{g,q,d,t,y})
$$

$$
\text{usecost}_{z,y} = \sum_{q,d,t} (\text{VOLL} \cdot \text{Duration}_{q,d,t,y} \cdot \text{unmetDem}_{z,q,d,t,y})
$$

$$
\text{usrcost}_{z,y} = \sum_{q,d,t} (\text{RESVoLL} \cdot \text{unmetRes}_{z,y}) +
\sum_{z,q,d,t,y} (\text{Duration}_{q,d,t,y} \cdot \text{SRESVoLL} \cdot \text{unmetSResZo}_{z,q,d,t,y}) +
\sum_{q,d,t} (\text{Duration}_{q,d,t,y} \cdot \text{SRESVoLL} \cdot \text{unmetSResSY}_{q,d,t,y})
$$

$$
\text{carboncost}_{z,y} = \sum_{g \in \text{Z},f,q,d,t} (\text{Duration}_{q,d,t,y} \cdot \text{carbon\_tax}_y \cdot \text{HeatRate}_{g,f} \cdot \text{carbon\_emission}_f \cdot \text{gen}_{g,f,q,d,t,y}) +
\sum_{z \in \text{c}} (\text{CO2backstop}_{z,y} \cdot \text{CostOfCO2backstop}) +
\text{SysCO2backstop}_y \cdot \text{CostOfCO2backstop}
$$

$$
\text{CO2backCost}_{,y} = \text{CO2backstopPenalty} \cdot \text{CO2backstop}_y
$$

$$
\text{YearlyTradeCost}_{z,y} = \sum_{q,d,t} ((\text{ImportPrice}_{z,q,d,t,y} - \text{ExportPrice}_{z,q,d,t,y}) \cdot \text{TradePrice}_{z,q,d,t,y} \cdot \text{Duration}_{q,d,t,y})
$$

$$
\text{CurtailmentCost}_{z,y} = \text{CurtailmentPenalty} \cdot \text{Curtailment}_{z,y}
$$

$$
\text{CostTransmissionAdditions}_{z1,z2,y} = \sum_z (\text{AdditionalTransfer}_{z,z2,y} \cdot \text{CRF}_{\text{NL}} \cdot \text{LineCAPEX}_{z1,z2})
$$

$$
\text{SurplusCost}_{z,y} = \text{SurplusPenalty} \cdot \text{Surplus}_{z,y}
$$

$$
\text{usrcostH2}_{z,y} = \sum_q (\text{CostH2Unserved} \cdot \text{unmetH2External}_{z,q,y})
$$

### Transmission network constraints

$$
\sum_{g \in Z,f} \text{gen}_{g,f,q,d,t,y} - \sum_{z2} \text{trans}_{z,z2,q,d,t,y} +
\sum_{z2} \left( (1 - \text{LossFactor}_{z,z2,y}) \cdot \text{trans}_{z2,z,q,d,t,y} \right) +
\text{storage\_out}_{z,q,d,t,y} - \text{storage\_inj}_{z,q,d,t,y} +
\text{unmetDem}_{z,q,d,t,y} - \text{surplus}_{z,q,d,t,y} +
\text{importPrice}_{z,q,d,t,y} - \text{exportPrice}_{z,q,d,t,y} - \sum_{h \in \text{map}(h,z)} \text{H2PwrIn}_{h,q,d,t,y} =
\text{Demand}_{z,q,d,t,y} \cdot \text{EEfactor}_{z,y}
$$

$$
\text{trans}_{z,z2,q,d,t,y} \leq \text{TransLimit}_{z,z2,q,y} + \text{AdditionalTransfer}_{z,z2,y} \cdot \text{TransCapPerLine}_{z,z2}
$$

$$
\text{AdditionalTransfer}_{z,z2,y} = \text{AdditionalTransfer}_{z,z2,y-1} + \text{BuildTransmission}_{z,z2,y}
$$

$$
\text{AdditionalTransfer}_{z,z2,y} = \text{AdditionalTransfer}_{z2,z,y}
$$

$$
\sum_{z \in c,q,d,t} \left( \text{importPrice}_{z,q,d,t,y} \cdot \text{Duration}_{q,d,t,y} \right) \leq
\sum_{z \in c,q,d,t} \left( \text{Demand}_{z,q,d,t,y} \cdot \text{Duration}_{q,d,t,y} \cdot \text{EEfactor}_{z,y} \cdot \text{MaxPriceImShare}_{y,c} \right)
$$

$$
\sum_{z \in c} \text{importPrice}_{z,q,d,t,y} \leq \sum_{z \in c} \left( \text{Demand}_{z,q,d,t,y} \cdot \text{MaxImport} \right)
$$

$$
\sum_{z \in c} \text{exportPrice}_{z,q,d,t,y} \leq \sum_{z \in c} \left( \text{Demand}_{z,q,d,t,y} \cdot \text{MaxExport} \right)
$$

### System requirements

$$
\sum_{g \in z} \text{reserve}_{g,q,d,t,y} +
\text{unmetSResSY}_{z,q,d,t,y} +
\sum_{z2} \left( \text{TransLimit}_{z,z2,q,y} +
\text{AdditionalTransfer}_{z,z2,y} \cdot \text{TransCapPerLine}_{z,z2} - \text{trans}_{z2,z,q,d,t,y} \right) \geq
\text{SResSY}_{z,y} + 0.2 \cdot \sum_{VRE \in z} \text{gen}_{VRE,f,q,d,t,y} \quad \forall z,q,d,t,y
$$

$$
\text{reserve}_{g,q,d,t,y} \leq \text{cap}_{g,y} \cdot \text{ResOffer}_g
$$

$$
\sum_{g \in Z} \text{cap}_{g,y} \cdot \text{CapCredit}_{g,y} +
\text{unmetRes}_{z,y} +
\sum_{z2} \sum_q \left( \text{TransLimit}_{z,z2,q,y} +
\text{AdditionalTransfer}_{z,z2,y} \cdot \text{TransCapPerLine}_{z,z2} - \text{trans}_{z2,z,q,d,t,y} \right) \geq
(1 + \text{PRM}_z) \cdot \max_{q,d,t} \text{Demand}_{z,q,d,t,y} \quad \forall z,y
$$

### Generation constraints


$$
\sum_f \text{gen}_{g,f,q,d,t,y} + \text{reserve}_{g,q,d,t,y} \leq (1 + \text{OverLoadFactor}_g) \cdot \text{cap}_{g,y}
$$

$$
\text{reserve}_{g,q,d,t,y} \leq \text{cap}_{g,y} \cdot \text{ResOffer}_g
$$

$$
\sum_f \text{gen}_{g,f,q,d,t-1,y} - \sum_f \text{gen}_{g,f,q,d,t,y} \leq \text{cap}_{g,y} \cdot \text{RampDn}_g \quad \forall t > 1
$$

$$
\sum_f \text{gen}_{g,f,q,d,t,y} - \sum_f \text{gen}_{g,f,q,d,t-1,y} \leq \text{cap}_{g,y} \cdot \text{RampUp}_g \quad \forall t > 1
$$

$$
\sum_f \text{gen}_{g,f,q,d,t,y} \geq \text{MinCapFac}_g \cdot \text{cap}_{g,y} \quad \forall d \in M
$$

$$
\sum_{f,d,t} \left( \text{Duration}_{q,d,t,y} \cdot \text{gen}_{g,f,q,d,t,y} \right) \leq
\text{Availability}_{g,q} \cdot \sum_{d,t} \left( \text{Duration}_{q,d,t,y} \cdot \text{cap}_{g,y} \right)
$$

### Renewable generation

$$
\text{gen}_{g,f,q,d,t,y} \leq \text{RPprofile}_{g,\text{RE},q,d,y,t} \cdot \text{cap}_{g,y} \quad \forall \text{RE} \notin \text{CSP}
$$

### Concentrated Solar Power (CSP) Generation

$$
\text{storageCSP}_{g,z,q,d,t,y} \leq \text{cap}_{g,y} \cdot \text{CSP\_storage} \quad \forall \text{map}(g,\text{CSP})
$$

$$
\text{genCSP}_{g,z,q,d,t,y} = \text{RPprofile}_{z,\text{RE} \in \text{CSP},q,d,t} \cdot \text{cap}_{g,y} \cdot
\frac{\text{SolarMultipleCSP}}{\text{TurbineEfficiency\_CSP} \cdot \text{FieldEfficiency\_CSP}}
$$

$$
\sum_{f \in \text{CSP}} \text{gen}_{g,f,q,d,t,y} \leq \text{cap}_{g,y}
$$

$$
\sum_{f \in \text{CSP}} \left( \text{genCSP}_{g,z,q,d,t,y} \cdot \text{FieldEfficiency\_CSP} - \text{storageCSPinj}_{g,z,q,d,t,y} + \text{storageCSPout}_{g,z,q,d,t,y} \right) =
\frac{\text{gen}_{g,f,q,d,t,y}}{\text{TurbineEfficiency\_CSP}} \quad \forall g,z,q,d,t,y
$$

$$
\text{storageCSP}_{g,z,q,d,t,y} = \text{storageCSP}_{g,z,q,d,t-1,y} + \text{storageCSPinj}_{g,z,q,d,t,y} - \text{storageCSPout}_{g,z,q,d,t,y}
$$

### Time consistency of power system additions and retirements

$$
\text{cap}_{g \in \text{EG},y} = \text{cap}_{\text{EG},y-1} + \text{build}_{\text{EG},y} - \text{retire}_{\text{EG},y} \quad \forall (y, g \in \text{EG}), \forall \text{ord}(y) > 1
$$

$$
\text{cap}_{g \in \text{EG},y} = \text{GenCap}_g \quad \forall (y, g \in \text{EG}): (\text{ord}(y) = 1 \wedge \text{StartYear} > \text{Commission\_year}_g)
$$

$$
\text{cap}_{g \in \text{EG},y} = \text{GenCap}_g \quad \forall (y, g \in \text{EG}): (\text{Commission\_year}_g \geq \text{StartYear} \wedge \text{ord}(y) \geq \text{Commission\_year}_g \wedge \text{ord}(y) < \text{Retirement\_year}_{\text{EG}})
$$

$$
\text{cap}_{g \in \text{EG},y} = 0 \quad \forall (y, g \in \text{EG}): (\text{ord}(y) \geq \text{Retirement\_year}_{\text{EG}})
$$

$$
\text{cap}_{g \in \text{NG},y} = \text{cap}_{\text{NG},y-1} + \text{build}_{\text{NG},y} \quad \forall \text{ord}(y) > 1
$$

$$
\text{cap}_{g \in \text{NG},y} = \text{build}_{\text{NG},y} \quad \forall \text{ord}(y) = 1
$$

$$
\text{cap}_g \leq \text{GenCap}_g
$$

$$
\text{cap}_{g,y} = 0 \quad \forall (y, g): (\text{ord}(y) < \text{Commission\_year}_g)
$$

### Storage constraints

$$
\text{storage}_{\text{st},z,q,d,t=1,y} = \text{Storage\_efficiency}_{\text{st}} \cdot \text{storage\_inj}_{g,z,q,d,t=1,y} - \text{gen}_{\text{st},z,q,d,t=1,y} \quad \forall t=1
$$

$$
\text{storage}_{\text{st},z,q,d,t>1,y} = \text{storage}_{\text{st},z,q,d,t-1,y} +
\text{Storage\_efficiency}_{\text{st}} \cdot \text{storage\_inj}_{\text{st},z,q,d,t-1,y} - \text{gen}_{\text{st},z,q,d,t-1,y} \quad \forall t>1
$$

$$
\text{storage\_inj}_{\text{stp},q,d,t,y} = \text{Cap}_{\text{so},y} \cdot \text{pStoragePVProfile}_{\text{so},q,d,t,y}
$$

$$
\text{storage\_inj}_{\text{st},q,d,t,y} \leq \text{Cap}_{\text{st},y}
$$

$$
\text{storage}_{\text{st},q,d,t,y} \leq \text{Storage\_Capacity}_{\text{st},y}
$$

$$
\text{storage\_inj}_{\text{st},q,d,t,y} - \text{storage\_inj}_{z,q,d,t-1,y} \leq \text{Cap}_{\text{st},y} \cdot \text{RampDn}_{\text{st}} \quad \forall t>1
$$

$$
\text{storage\_inj}_{\text{st},q,d,t-1,y} - \text{storage\_inj}_{z,q,d,t,y} \leq \text{Cap}_{\text{st},y} \cdot \text{RampUp}_{\text{st}} \quad \forall t>1
$$

$$
\text{Storage\_Capacity}_{\text{EG},y} = \text{Storage\_Capacity}_{\text{EG},y-1} +
\text{Build\_Storage\_Capacity}_{\text{EG},y-1} - \text{Retire}_{\text{Storage\_Capacity}_{\text{EG},y-1}} \quad \forall y>1
$$

$$
\text{Storage\_Capacity}_{\text{NG},y} = \text{Storage\_Capacity}_{\text{NG},y-1} +
\text{Build\_Storage\_Capacity}_{\text{NG},y-1} \quad \forall y>1
$$

$$
\text{Storage\_Capacity}_{\text{NG},y} = \text{Build\_Storage\_Capacity}_{\text{NG},y-1} \quad \forall y=1
$$

$$
\text{StorageCapacity}_{\text{st},y} \geq \text{Cap}_{\text{st},y}
$$

$$
\text{StorageCapacity}_{\text{st},y} \leq \text{Maximum\_Storage\_Energy}_{\text{st},y} + \text{Maximum\_Storage\_Energy}_{\text{CSP},y}
$$

$$
\sum_{y} \text{Build\_Storage\_Capacity}_{\text{EG},y} \leq \text{Maximum\_Storage\_Energy}_{\text{EG},y}
$$

### Investment constraints

$$
\sum_y \text{build}_{g \in \text{NG},y} \leq \text{MaxNewCap}_{\text{NG}}
$$

$$
\sum_y \text{build}_{g \in \text{NG},y} \geq \text{MinNewCap}_{\text{NG}}
$$

$$
\sum_y \text{build}_{g \in \text{EG},y} \leq \text{GenCap}_{g \in \text{EG}} \quad \forall g: (\text{Commission\_year}_g > \text{StartYear})
$$

$$
\text{build}_{g \in \text{NG},y} \leq \text{Annual\_built\_limit}_y \cdot \text{WeightYear}_y
$$

$$
\text{build}_{g \in \text{NG},y} = \text{DiscreteCap}_{g \in \text{NG}} \cdot \text{builtCapVar}_{g \in \text{NG},y} \quad \forall g: (\text{DiscreteCap}_g \geq 0)
$$

$$
\text{retire}_{g \in \text{NG},y} = \text{DiscreteCap}_{g \in \text{NG}} \cdot \text{retireCapVar}_{g \in \text{NG},y} \quad \forall g: (\text{DiscreteCap}_g \geq 0)
$$

$$
\text{build}_{g,y} = 0 \quad \forall y: (\text{ord}(y) \leq \text{Commission\_year}_g)
$$

$$
\text{fuel}_{z,f,y} \leq \text{MaxFuelOff}_{f,y}
$$

$$
\text{fuel}_{z,f,y} = \sum_{g \in Z,q,d,t} \left( \text{Duration}_{q,d,t,y} \cdot \text{HeatRate}_{g,f} \cdot \text{gen}_{g,f,q,d,t,y} \right)
$$

$$
\sum_{y,g \in \text{NG}} \left( \text{ReturnRate}_y \cdot \text{pweight}_y \cdot \text{CRF}_{\text{NG}} \cdot \text{CapCost}_{\text{NG},y} \cdot \text{cap}_{g,y} \right) \leq \text{MaxCapital}
$$

### Environmental policy


$$
\text{emissions\_Zo}_{z,y} = \sum_{g \in Z,q,d,t} \left( \text{gen}_{g,f,q,d,t,y} \cdot \text{HeatRate}_{g,f} \cdot \text{carbon\_emission}_f \cdot \text{Duration}_{q,d,t,y} \right)
$$

$$
\text{emissions\_Zo}_{z,y} \leq \text{Zo\_emission\_cap}_{y,z}
$$

$$
\text{emissions}_{z,y} = \sum_{g,q,d,t} \left( \text{gen}_{g,f,q,d,t,y} \cdot \text{HeatRate}_{g,f} \cdot \text{carbon\_emission}_f \cdot \text{Duration}_{q,d,t,y} \right)
$$

$$
\text{emissions}_{z,y} \leq \text{Sy\_emission\_cap}_y
$$

### CCS retrofits


$$
\text{build}_{\text{ng},y} - \text{retire}_{\text{eg},y} \leq 0 \quad \forall \text{ng},\text{eg} \in \text{GG}
$$

### Green hydrogen production

$$
\text{capH2}_{\text{eh},y} = \text{capH2}_{\text{eh},y-1} + \text{buildH2}_{\text{eh},y} - \text{retireH2}_{\text{eh},y} \quad \forall \text{ord}(y) > 1
$$

$$
\text{capH2}_{\text{nh},y} = \text{capH2}_{\text{nh},y-1} + \text{buildH2}_{\text{nh},y} \quad \forall \text{ord}(y) > 1
$$

$$
\text{capH2}_{h,y} = \text{H2Cap}_{h \in \text{EH},y} + \text{buildH2}_{h,y} - \text{retireH2}_{\text{EH},y} \quad \forall \text{ord}(y) = 1
$$

$$
\sum_{y \in Y} \text{buildH2}_{\text{eh},y} \leq \text{H2Cap}_{\text{eh}}
$$

$$
\sum_{y \in Y} \text{buildH2}_{\text{nh},y} \leq \text{MaxNewCapH2}_{\text{nh}}
$$

$$
\text{buildH2}_{\text{nh},y} = \text{UnitSizeH2}_{\text{nh}} \cdot \text{buildCapVarH2}_{\text{nh},y}
$$

$$
\text{retireH2}_{\text{eh},y} = \text{UnitSizeH2}_{\text{eh}} \cdot \text{retireCapVarH2}_{\text{eh},y}
$$

$$
\sum_{d,t} \left( \text{H2PwrIn}_{h,q,d,t,y} \cdot \text{Duration}_{q,d,t,y} \right) \leq \text{AvailabilityH2}_{h,q} \cdot \text{CapH2}_{h,y} \cdot \sum_{d,t} \text{Duration}_{q,d,t,y}
$$

$$
\sum_z \left( \text{ExternalH2}_{z,q,y} - \text{unmetH2External}_{z,q,y} + \text{fuelH2Quarter}_{z,q,y} \right) \leq \sum_{z,h \in \text{map}(h,z)} \left( \text{H2PwrIn}_{h,q,d,t,y} \cdot \text{Duration}_{q,d,t,y} \cdot \text{EfficiencyH2}_h \right)
$$

$$
\sum_z \text{fuelH2Quarter}_{z,q,y} = \text{fuelH2}_{z,y}
$$

$$
\text{H2PwrIn}_{h,q,d,t,y} \leq \text{CapH2}_{h,y}
$$

$$
\text{H2PwrIn}_{h,q,d,t-1,y} - \text{H2PwrIn}_{h,q,d,t,y} \leq \text{CapH2}_{h,y} \cdot \text{RampDnH2}_h
$$

$$
\text{H2PwrIn}_{h,q,d,t,y} - \text{H2PwrIn}_{h,q,d,t-1,y} \leq \text{CapH2}_{h,y} \cdot \text{RampUpH2}_h
$$

$$
\text{Gen}_{\text{RE},f,d,t,y} = \text{REPwr2Grid}_{\text{RE},f,d,t,y} + \text{REPwr2H2}_{\text{RE},f,d,t,y}
$$

$$
\sum_{z,h \in \text{map}(h,z)} \text{H2PwrIn}_{h,q,d,t,y} = \text{PwrREH2}_{z,q,d,t,y}
$$

$$
\sum_{\text{RE},f,z \in (\text{map}(\text{RE},f),\text{map}(\text{RE},z))} \text{REPwr2H2}_{\text{RE},f,q,d,t,y} = \text{PwrREH2}_{z,q,d,t,y}
$$

$$
\sum_{\text{RE},f,z \in (\text{map}(\text{RE},f),\text{map}(\text{RE},z))} \text{REPwr2Grid}_{\text{RE},f,q,d,t,y} = \text{PwrREGrid}_{z,q,d,t,y}
$$

## Description of the model

EPM cand be used (i) as a least-cost capacity expansion model to optimize investments in the power sector over a specified horizon that could extend from a few years to decades or as (ii) an economic dispatch model to optimize the operation of a specified power system over a relatively short period compared to Capacity Expansion Model (1-week to 1-year). Like every optimization software the problem is mathematically described through an objective function and several accompanying equations (system constraints). System constraints are used to describe mathematically the physical part of the system, laws of physics, policy targets and other realities specific to the system (for example fuel availability). More detailed description of system constraints can be found on subsequent sections.

### Indices and Sets

All sets used in the formulation can be classified in two major categories: time and power system-related. Four sets belong to the first category: which represent different time scales considered in the model: days, quarters, hours and years. Hour is the smallest unit of time used in this formulation and we could use the same formulation using just one set for time containing as many hours as the set . It is convenient though to keep all the four sets since they reveal some fundamental assumptions of the model: (1) days are used to reflect the chronological sequence of the time slices used for ramping and storage constraints as we will further explain in the constraints section; (2) quarters are used to reflect seasonality in the load patterns, the availability of thermal power units and the thermal limits of transmission lines; (3) years are used to represent annual trends on demand growth and keep track of the lifetime of units; and finally (4) hours are commonly used as the smallest time unit in long term models since the day ahead scheduling models schedule generation units on an hourly basis.

Three sets are power system related. Set includes all generating units. Depending on the size of the system, we might decide to use set to model individual units of the power system for a small system or aggregated units that represent multiple units of the same technology for a large system. We use the term technology to refer to different technologies or different fuels used: e.g. coal steam turbines, natural gas combined cycle, natural gas combustion turbines, wind farms, solar photovoltaic panels, geothermal, hydropower and diesel generators. Depending on the resources available in a country, some technologies might not be present. As the model stands now, elements of set are mapped to sets and , which stand for fuel and zones respectively. Set is one of the major sets used in power systems since the power system is a network and physical laws (widely known as Kirchhoff’s laws) govern the flow of power over the transmission lines. Given that, a set such as set which captures the spatial dimension of the system is necessary. At the finest granularity, set might contain buses of the power system but in case we model larger systems, set might contain zones of a power system or even countries. Note that the modeler usually decides on the spatial granularity based on the presence of common regulatory rules or pricing schemes in a zone or/and based on the congestion observed on transmission lines connecting adjacent regions. Finally, set includes the different fuels used and we model it to keep track of the consumption of different fuels since for certain fuels domestic upper bounds on consumption might apply and/or issues of energy security might be involved in case of imported fuels. In addition, different types of fuels have different carbon content and lead to different emissions of carbon dioxide, which are important to track in case environmental policies exist.

### Objective function

The objective function in this model minimizes the total system cost including violation/penalty terms for constraints that are not met. All generation, transmission and emissions costs of the system are considered including: (1) Fixed costs including annualized capital cost payments for new generators<sup>[\[4\]](#footnote-5)</sup> and any additional transmission capacity and fixed operation and maintenance costs (2) variable costs including the fuel cost and any variable operation and maintenance costs (3) cost to procure spinning reserves (4) carbon tax payments and (5) penalties for unmet demand and unmet capacity and spinning reserve requirements, breaching the emissions and renewable rejection at the system or the zonal level.

In the case of a least-cost expansion model the objective function typically represents total system costs over the study horizon. Depending on the type of the problem (for example linear or non-linear) the user chooses an appropriate solver to minimize the value of the objective function. Total system costs are broken into several components the most important of which are the fixed and variable costs of electricity generation. CAPEX costs typically dominate fixed costs while fuel costs dominate variable costs of generation. In systems with very high shares of renewables VOM costs are higher than fuel costs. Other types of system costs included in the objective function are costs of reserve provision by generators, the cost of carbon (when applicable), the cost of electricity trading (it could be a net benefit if revenues from electricity trade exceed costs) and the cost of transmission enhancements.

When EPM is used as a dispatch model, there are no capex costs in the objective function. In dispatch mode, the model does not consider unit commitment related decisions in its current form, albeit minimum generation constraints and ramp rates are modeled.

### Load approximation

The size of the optimization problem (i.e. the number of decision variables to be determined) depends significantly on the study horizon and the optimization step. An EPM user needs to find the balance between precision and computational effort (practicality) to complete the analysis. This is especially important when little gains in precision impact significantly the running time. In most cases representing the full year at the hourly level (8760 hours) over a study horizon that extends to multiple years, usually 15 to 20 years, is neither practical nor required to assess in detail the capacity mix of most supply options. For capacity expansion analyses, the load is typically approximated through a number of representative days which are assumed to repeat several times throughout the calendar year. Such representative days are carefully selected to capture the hourly profile (including the peak) of electricity demand on a monthly or quarterly basis. It is common practice to select the following three types of representative days: (i) an average day which is assumed to be representative of the bulk of electricity production (repeats several times over a quarter (or month), (ii) a typical day of high demand (which also need to include the peak) and (iii) a typical day of low demand. The last two types of days typically represent a much smaller portion of the quarterly (or monthly) demand compared to the average day. The EPM input file includes tabs where a user can provide the hourly profile of each of the selected typical days on the “Demand” tab. Also, a user needs to provide the relative contribution of each typical day on total quarterly/monthly/seasonal demand represented as number of repeats over a typical quarter (or month), in the “Duration” tab.

The above-mentioned representation of the load has been widely used in power sector analysis to estimate the capacity mix of conventional supply options both efficiently and accurately. However, application of this methodology has weaknesses in power systems with very high shares of variable renewable energy (VRE) sources. In such systems detailed representation of chronology might have significant impact on results. VRE generation profiles are inputs on EPM. Selection of a representative day for load need to be accompanied by the respective VRE generation profile that corresponds to the selected day. This requires extensive data processing.

A weakness of selecting representative demand days is that the analysis might undersize electricity storage, the simulation of which requires full description of chronology. This is especially true in the case multi-hour and seasonal storage. For that reason, running EPM on dispatch mode (full chronology) of a full (selected) year might be required to supplement capacity expansion analysis.

An EPM user can represent demand in detail for each year of the study horizon or provide annual projections (single value) of demand and peak demand together with typical day profiles over a base (historical) year. When the latter option is activated, the model uses the above data to calculate the projected demand in detail.

### Value of Lost Load

The Value of Lost Load (VoLL) is an economic penalty used to ensure that the model will invest to meet the demand and avoid unserved energy if it is economic to do so. The VoLL represents the damage on the local economy from missing to supply 1MWh of demand. Without assigning VoLL matching supply and demand would be economically suboptimal as it is cheaper to fall short on supply rather than paying for the cost of electricity generation and associated investments. Thus, the selection of VoLL need to be based on research. There is no universally acceptable Value of Lost Load and different methods have been applied to estimate a reasonable value for the unserved energy. Typical values used in developed economies vary between 4,000 and 40,000$/MWh while in developing countries between 500 and 10,000$/MWh \[1\].

VoLL affects both the total-non served energy in the system and peaking supply investment decisions. In some cases, allowing for some loss of load might be preferable compared to investing into supply options that will only operate for a tiny amount of the annual time. The user will need to calibrate the model considering the local realities and system rules since the maximum allowed number of hours failing to meet demand is constrained by reliability criteria. When a value for VoLL is not available in global literature the EPM user can select a reported VoLL from a country with similar economic characteristics. Alternatively, the assessment of VoLL can be based on analytical expressions or based on the assumption that demand is sensitive to electricity price. In the latter case VoLL can take the value of cost of generation of the most expensive asset in the system which are usually backup diesel generators \[2\].

Typically, back-up generators are fueled with expensive diesel, and if they are used in the mode just described, the system VOLL would take the value of the variable cost of these generators (~500 $/MWh).

### Transmission network constraints

Kirchhoff’s laws are physical laws governing the flows over transmission lines in a network. According to first Kirchhoff law, also known as KCL (Kirchhoff’s Current Law), the sum of injections in a node should equal zero. In our formulation, KCL corresponds to equation _(14)_. There, we can see that the power provided by generators and storage should be equal to demand (minus the unmet demand) plus/min outflows/inflows from the node to adjacent nodes. This equation additionally includes price-driven import and export volumes from external zones. The demand is adjusted with the potential improvements in the energy efficiency using parameter.

|     | (14) |
| --- | --- |

Note that the second Kirchhoff Law (or widely known as KVL- Kirchhoff’s Voltage Law) is valid for power systems but for reduced power system, it might not apply depending on the method of network reduction followed. In this particular formulation, KVL is not considered.

Another important feature of our model relates to modeling of transmission losses. We model transmission losses as a percent reduction of the imported electricity at each node. In particular, term

at equation _(14)_ models injections to node _z_ and we can see how the loss factor reduces the amount of energy imported. On the contrary, the outflow is fully considered at the origin node of the network: .

Another common constraint for transmission networks refers to the capacity limits of transmission lines. In particular, as equation _(15)_ implies, the flow over a specific line cannot exceed a certain limit, which is defined either by thermal limit of the line or upper bounds imposed by reliability considerations. Note that we model flows over a particular transmission line with two positive variables, one for each direction. Please observe that the transmission limit parameter might change per year to reflect planned upgrades and the additions to the transmission network. There are also maximum built limits of additional transmission capacity between zones due to budget or technical constraints. Moreover, the transmission limit differs per season since ambient temperature affects the capacity available for power transfers.

| +   | (15) |
| --- | --- |

The modelled zones, countries and regions are usually not isolated from the external systems and markets not included in the model. Therefore, we allow trading with these zones at time-varying predefined price. Equations _(16)_, _(17)_ and _(18)_ define the limits of this exchange. Equation _(16)_ sets the annual limit on the energy imported from external zones as a percentage of the annual demand using parameter . Equations _(17)_ and _(18)_ introduce the hourly limit on the imported and exported power, defined as a percentage of hourly demand.

### System requirements

In our formulation, we model two products that the system operator might require generators to provide during operation: (1) energy and (2) spinning reserves. Per NERC’s definition, spinning reserves refers to “unloaded generation that is synchronized and ready to serve additional demand” \[3\]. Note that more products exist, especially in organized U.S. wholesale markets such as non-spinning reserves or flex ramp. Operating reserves (spinning and non-spinning reserves) provide the capability above firm system demand required to provide for regulation, load-forecasting error, equipment forced and scheduled outages and local area protection \[3\]. Moreover, under the spinning reserves different products might exist with respect to the response times required etc.

The amount of spinning reserve required depends on several factors that the planner/operator considers such as the load level and the associated forecasting error, the forecasting error attached to the renewable generation and the size of the largest unit committed on the system to be able to accommodate N-1 outages. Equation _(19)_ indicates that spinning reserve can be provided by interconnections. On top of system-wide reserve requirements, zonal requirements apply to accommodate for outages on transmission lines connecting adjacent regions/zones/nodes.

|     | (19) |
| --- | --- |

Planners usually consider a planning reserve margin (PRM) to account for forecasting error in demand projections. Typical values for the PRM vary between 10-15%. Equation _(21)_ indicates that interconnections can be accounted for as reserve margin. Note that intermittent units do not contribute towards the planning reserve constraint at their full capacity but at a fraction specified by the planner (using the Capacity Credit parameter), e.g. in the U.S. markets this fraction is calculated based on available historical data as the capacity factor during a set of peak hours \[4\].<sup>[\[5\]](#footnote-6)</sup>

|     | (21) |
| --- | --- |

### Generation constraints

Spinning reserve capacity is modeled as it is a very important dimension of a system flexibility and ability to integrate VRE.

|     | (22) |
| --- | --- |

Equation (22) assures that the power generated by the unit along with the spinning reserves provided by the same unit do not exceed the unit’s capacity.<sup>[\[6\]](#footnote-7)</sup> Note that the capacity is augmented by an overload factor. This factor is typically 10% for those generators that can handle overload conditions for a short period of time, and zero for those generators that cannot handle such conditions.

Given that spinning reserve products are usually defined with respect to response time of generator to a certain dispatch signal, only a certain percentage of the generator’s unit qualify as a reserve offer. We capture this characteristic in the model through equation (23).

|     | (23) |
| --- | --- |

Ramping constraints acknowledge that the generation units have inertia in changing their outputs and differences in generation outputs between consecutive hours should be constrained by the ramping up and down capabilities of the unit.

|     | (24) |
| --- | --- |
|     | (25) |

Another important feature of generators is the minimum load. The minimum load can either be determined based on technical specifications provided by the manufacturer or be calculated as an “economic” minimum beyond which the unit can provide energy economically. The minimum load constraint is really important for unit commitment and requires the use of binaries variables that make sure the constraint is enforced when the unit is on. However, in generation expansion models operations are approximated through a simple dispatch model for representative hours of the year. In this case, if deemed important, an approximation of the minimum load constraint can be applied for the thermal units for which it is relevant. The constraint can be applied for a subset of the days modeled and the constraint is activated only for those days or to all days. Constraint (26) is forcing all units to generate power equal to at least their minimum loading levels for specific days in the year.

|     | (26) |
| --- | --- |

Generating units require maintenance every year. Given that, we should consider the units as unavailable for certain periods during the year. In this particular application, we consider a uniform availability factor per quarter to account for maintenance.

|     | (27) |
| --- | --- |

### Renewable generation modeling

Renewable generation differs from conventional units in that its output is, to a certain extent, uncontrollable and intermittent. The power generated by renewables such as wind or solar depends on wind velocity or solar irradiation. Collecting historical data that records weather information (such as wind speed, temperature, wind direction, etc.) or the power generation output by installed renewables at specific locations, analysts usually employ statistical methods such as k-means to reduce the number of hours required to approximate the intermittent nature of renewables \[5\]. In this particular application, the generation profile for each renewable energy technology (such as wind or solar PV) is defined by the hourly capacity factor, in a year, of a generic power plant of each type, modeled at a specified location. Then, given this hourly profile for a year, we choose the amount of days modeled based on the days selected for the load approximation (see Section 1.4.3), i.e., the renewable profile during the 12 days in the year selected, for load, is maintained.

|     | (28) |
| --- | --- |

Note that the renewable profile is highly dependent on the region/location the resource is located. This formulation implicitly models that aspect since might have different elements for the same generation technology at different locations.

### Concentrated Solar Power (CSP) modeling

CSP technology modeling differs from other renewable technologies due to the complexity derived by its storage capabilities. The CSP configuration considered in this model consists of two integrated subsystems; these include the thermal storage system, and power cycle. Thermal storage is modeled using a simple energy balance approach that includes charging and discharging energy. The power cycle model provides a simple mechanism for modeling the conversion from thermal energy output from the solar field, and thermal storage into electrical energy.

|     | (29) |
| --- | --- |

Equation (29) indicate that at any time the CSP storage level cannot exceed its storage capability.

|     | (30) |
| --- | --- |

The power output of the solar panel is calculated by multiplying the nameplate capacity of the CSP power plant, the capacity factor of the system, and the solar multiple, then, dividing this by the turbine and solar field efficiencies (Equation (30)).

|     | (31) |
| --- | --- |

Equation (31) indicate that all the power output produced by CSP generators at any given zone, cannot exceed the nameplate capacity. Finally, Equations (32) and (33) detail the power balance formulations for the power cycle and thermal storage subsystems.

|     | (32) |
| --- | --- |

|     | (33) |
| --- | --- |

### Time consistency of power system additions and retirements

We use constraint (34) to track the capacity in consecutive years. In particular, generation capacity at year equals capacity at previous year plus any investment minus any retirement at year .

|     | (34) |
| --- | --- |

Several more constraints are formulated to fix the capacity at pre-specified levels in certain years.

- The first constraint sets the capacity of the existing units in the first year to the predefined capacity, if the commissioning year is earlier then the first year of the optimization horizon.

<table><tbody><tr><th></th><th><p>(35)</p></th></tr><tr><td></td><td></td></tr><tr><td colspan="3"><ul><li>The second constraint sets the capacity of the existing units to the predefined capacity, for all the years exceeding the commissioning year, but before the defined retirement year.</li></ul><table><tbody><tr><th></th><th><p>(36)</p></th></tr><tr><td></td><td></td></tr></tbody></table></td><td></td></tr></tbody></table>

- The third constraint forces the capacity of the existing units to 0, when the retirement year is reached.

|     | (37) |
| --- | --- |
|     |     |
|     |     |

- The fourth constraint states that the total capacity of a new generator equals the capacity of that new generator built the previous year plus the capacity to be built the current year:

|     | (38) |
| --- | --- |

- The fifth constraint forces the capacity at the first year of the horizon for new units to be equal to the capacity installed that year

|     | (39) |
| --- | --- |

- The sixth constraint limits the capacity to the predefined available or installed capacity of the unit.

|     | (40) |
| --- | --- |

- The seventh constraint forces the capacity of planned and candidate units at zero for years preceding the commission year of the unit.

|     | (41) |
| --- | --- |

### Storage modeling

Economically efficient storage in power systems has mainly been pumped hydro storage for a considerable number of years. Nowadays, more storage technologies such as battery energy storage systems are being added to the power system providing energy and/or reserve services.

Storage is modeled differently compared to conventional units since it requires two more variables: (1) one variable to keep track of the storage level (storage) and (2) one variable (_storageinj_) to model the injection of energy into the storage unit. The variable _gen_ used to track the generator output of conventional units is used to account for the output of a storage unit when it is discharged. Moreover, the chronological sequence of the time slices is important to make sure that the simulated operation is feasible e.g., we cannot discharge a storage unit if charging of the unit has not preceded. Finally, storage of energy requires the conversion of electricity to another form of energy e.g., mechanical for flywheels or chemical for fuel cells and common batteries. The conversion of one form of energy to another involves losses that we should consider in our models. Equations (30) and (31) represent the storage balance equations on the first hour of a representative day and any other hour of a representative day respectively:

|     | (42) |
| --- | --- |
|     | (43) |

The first equation (42) makes sure that the battery is discharged at the start of each representative day. The choice for not making a link between different days and thus having the battery discharged at the start of each day is because the standard formulation of the model works with representative days (e.g peak load, minimum load, and average load days) instead of actual days. Equation (43)in turn keeps track of the energy stored in the storage unit between consecutive hours of the same day: the energy stored in the unit at time slice _t_ (t > 1) equals the energy stored in the unit at time slice _t-1_ plus any injection reduced by the storage efficiency minus any discharge at time _t_. In case the storage unit is linked to a PV plant, the injection is given by the PV profile:

|     | (44) |
| --- | --- |

Two additional constraints are used to make sure that the operation of storage in each time slice is feasible considering the peak storage capacity (in MW):

|     | (45) |
| --- | --- |
|     | (46) |

The first constraint ensures that the hourly energy injection is lower or equal than the maximum hourly energy injection. The second constraint limits the total amount of energy (in MWh) stored to the maximum storage level (in MWh). Storage outputs are constrained by the general equation (22).

Furthermore, the storage units can be charged or discharged at a rate, which cannot exceed a specific value. To model this behavior, we include constraints (47) and (48) respectively.

|     | (47) |
| --- | --- |
|     | (48) |

Three equations keep track of the deployed energy storage capacity (in MWh) of the storage units:

|     | (49) |
| --- | --- |
|     | (50) |
|     | (51) |

Equation (49) keeps track of the energy storage capacity for existing storage units whereas equations (50) and (51) are used for new storage units.

Three final constraints limit the total deployed energy storage capacity (in MWh):

(52) (53)

(54)

The first constraint ensures that the deployed energy storage (in MWh) is larger than the peak storage capacity (in MW) at any time.<sup>[\[7\]](#footnote-8)</sup> The second constraint limits the total deployed energy storage (in MWh) to the maximum deployable energy storage (in MWh). Equation (54) constrains the deployment of existing storage units for which the commission year is later than the starting year of the modelling horizon.

### Investment constraints

Planners consider several constraints when they decide on a generation investment plan. Common constraints refer to budget, land use, scheduling of new construction and consumption of specific fuels for energy security considerations.

Constraint (55) usually reflects land use considerations, regulation that imposes an upper bound on capacity of specific technologies or simply resource potential (e.g., for wind there is a finite amount of locations where wind farms can provide the capacity factor modeled).

|     | (55) |
| --- | --- |

Constraint (58) is usually employed to reflect practical limitations on construction and spread the construction of new units more uniformly over time. For example, it seems pretty unrealistic that the whole system capacity can be built in one year.

|     | (58) |
| --- | --- |

Constraint (62) imposes an upper bound on fuel consumption. This upper bound might correspond to the fuel reserves a country might have at its disposal or the capacities of refinement units or importing units such as size of LNG terminals. Constraint (63) simply estimates the fuel consumption. Note that in case we want to reduce the number of variables in our model, we can get rid of the fuel variable since it is defined in terms of the generation variable.

|     | (62) |
| --- | --- |
|     | (63) |

Constraint (64) represents a budget constraint. It limits the capital expenses withdrawal to be lower than a pre-specified amount. In this formulation, we assume that the _MaxCapital_ parameter is similar to the maximum debt payments that a power system planner can do over the horizon.

|     | (64) |
| --- | --- |

An alternative constraint that addresses the same concern but relies on different information is expressed by constraint (69). In that case, the planner does not know the maximum amount of debt payments that the power plant owners might do but he has a good understanding of the maximum capital available to the system for investment. In that case, the sum of the overnight capital expenditure is not allowed to exceed this known budget.

|     | (69) |
| --- | --- |

### Environmental policy

Environmental concerns often lead policymakers to adoption of caps on the amount of emissions of certain pollutants. Several countries have announced targets to reduce their carbon dioxide emissions below particular amounts. Given that the power system is a big contributor to carbon dioxide emissions, power system caps are usually enforced at the country or at multi-country level. Constraints (65) and (66) impose the caps decided by regulators. Note that constraints (67) and (68) are modeled as “hard” constraints with no violation allowed. We would like to note that environmental policy might impose caps on more pollutants beyond carbon dioxide. For example, in the USA certain caps apply on and emissions. The dual variables of constraint (66) and (68) provide valuable information to the planner since they estimate the additional cost the planner would have to bear to decrease the amount of the pollutant by an infinitely small amount. In other words, the dual prices of (66) and (68) provide an approximation of the prices that the planner would pay per unit of pollutant if slightly stricter regulation would be applied.

|     | (65) |
| --- | --- |
|     | (66) |
|     | (67) |
|     | (68) |

Another policy mechanism related to carbon emissions is a carbon tax (less popular than the cap-and-trade system at present). We model the carbon tax as part of the objective function. Note that the carbon tax does not correspond to an actual cost for the society since it is a transfer payment for emitters to the government. It reflects an actual cost, though, only if it attempts to monetize the public health cost and the damage to the environment. However, it reflects an actual cost for power system since generators would probably have to pay the tax to the government and that explains why it is part of the objective function (8).

|     | (8) |
| --- | --- |

### Modelling CCS

CCS technology investments can be modelled on EPM as either greenfield or retrofits of existing coal plants. A new CCS plant is modelled similarly as any other coal plant with its techno-economic parameters adjusted. Typically, CCS plants are modelled to have lower efficiency and higher CAPEX and OPEX compared to conventional coal plants. The environmental impact of CO2 storage is modelled through setting fuel burned on a CCS plant to have reduced emissions factor compared to conventional coal (typically reduction 90% to 95%).

CCS retrofits are modeled as a sudden change on the techno-economic parameters of an existing plant subject to some CAPEX incurred. More specifically, this change is modelled as a decommissioning of the existing plant and a simultaneous commissioning of the retrofitted plan. The retrofit CAPEX and the updated techno-economic parameters need to be carefully selected based on international experience. The mathematical addition of CCS retrofit on EPM is done through equation 69 below:

(69)

Set mapGG containing the relevant pairs of the existing plant (candidate for retrofit) and the new plant (retrofitted plant) is automatically generated through the inputs development phase. The relevant connection (pairs of plants) is made by the EPM user on excel tab named LinkedComDecom

### Modelling of Green Hydrogen

Modeling of production of green hydrogen is possible on EPM. When the option is activated investments on electrolyzers are optimized similarly as with supply side technologies. The user provides the relevant technoeconomic inputs for electrolyzers (CAPEX, OPEX, efficiency, project lifetime etc). There are two potential uses of green hydrogen in EPM:

1. Re-circulation in the power sector. In that case green hydrogen acts as a carrier of renewable energy providing long-term storage to the system. Hydrogen can be burned in modified new CCGTs/OCGTs at 100% per volume or burned in existing dual fuel generators (future versions of EPM will include option for pre-specified per volume mix of hydrogen with natural gas to be burned in conventional turbines). The user needs to define hydrogen as an additional fuel with zero cost and zero emissions factor since it is produced within the system and not imported. Costs of hydrogen are indirectly accounted through the additional investments and relevant OPEX within the system. Also, techno-economic characteristics of new hydrogen turbines need to reflect published figures based on international experience
2. Used in other end-use sectors (residential, industry, transport). This is modeled as an external demand of green hydrogen to be satisfied subject to some penalty for each mmBTU of unserved hydrogen demand. EPM simulates power sector operation and investment requirements to satisfy external hydrogen demand but it does not model any subsequent use outside the power sector. The costs for external hydrogen production are accounted for in the model and thus the power sector is assumed to bear the costs of decarbonization of other sectors.

Description of green hydrogen formulation on EPM follows below:

Investments on electrolyzers are mathematically constrained to dictate time consistency of capacity additions and retirements, similarly as with generators:

Generation capacity of existing electrolyzer capacity at year equals capacity at previous year plus any investment minus any retirement at year where y is not the first year of the study horizon (equation 70)

|     | (70) |
| --- | --- |

Equation (71) is the same constraint as (70) modified for candidate capacity

|     | (71) |
| --- | --- |

Equation (72) constraints total electrolyzer capacity at the end of the first year of the optimization horizon. At the beginning of horizon existing capacity is the one defined in the inputs section and by the end of year the capacity is adjusted by any optimized additions or retirement throughout the year.

|     | (72) |
| --- | --- |

Total electrolyzer capacity that can be developed into mandatory projects (decision for development has been made) can not exceed the maximum electrolyzer capacity provided as input

|     | (73) |
| --- | --- |

Total new electrolyzer capacity that can be developed in candidate projects (decision to be optimized) cannot exceed the maximum electrolyzer capacity provided as input

|     | (74) |
| --- | --- |

Electolyzer candidate investments can also be set as integer decisions. In such cases the defined capacity can either be built in one step or stay out of the plan (equation 75). Similarly retirement of existing electrolyzer plant takes place in a single step (76). In equations (75) and (76), variables and can only take positive integer values

(75)

(76)

Equation 77 puts a constraint on operational time of electrolyzers so that it doesn’t exceed their technical availability

Equation 88 describes the thermal balance of hydrogen within the system (in mmBTU). Total mmBTU of hydrogen produced are equal to hydrogen demand for other sectors minus unmet hydrogen demand plus hydrogen recirculate in the power sector as fuel.

Equation 79 makes a connection between quarterly an annual amount of hydrogen fuel produced in the system

Equation 80 puts a constraint on total amount of electricity drawn by an electrolyzer. Max power absorbed can’t be more than the electrolyzer capacity

Equations 81 and 82 define ramping constraints for electrolyzer operation

Equations 83 sets an energy balance on renewable electricity. Total renewable electricity generated is equal to renewable electricity consumed by end-use sectors and renewable electricity consumed by electrolyzes

Equation 84 makes a connection between renewable power produced and electricity drawn by electrolyzers

Finally, equations 85 and 86 are mapping renewable electricity produced on a zonal basis which is necessary for the architecture of the hydrogen model

## Limitations of the model

As any model, this model also has some limitations in most practical system applications. Two major limitations of the model are the following:

(1) Transmission network representation: Although in theory, the model can be used to represent the high voltage transmission network down to individual substations, the size limitation of the underlying LP, typically restricts to a small (below 20) number zones and we assume zero congestion within each zone modeled. Transmission expansion is not always an option. The model does not consider voltage angles and DC approximation to power flow in its current form and as such may have unrealistic flows around a loop.

(2) As noted before, the model does not consider unit commitment (startup and shutdown) decisions. It also does not consider heat rate curves associated with thermal units and uses a single average heat rate for each generating unit.

(3) In its main version, the model is deterministic and no uncertainty with respect to assumed parameters etc. is considered.

(4) Hydropower: reservoir optimization is currently not a feature of EPM hence hydropower potential is represented as daily/seasonal/annual energy limits for individual hydro plant/unit. The main version model does not consider any representation of river chain.

(5) In the current model version, coal plant flexibilization or repurposing is not considered.

(6) When the model is used as a capacity expansion tool, the use of representative days and seasons limits the possibility to capture the VRE variability, hence the VRE integration costs are approximated. Similarly, the flexibility costs of the system are approximated as ancillary services are not all fully modeled.

## Bibliography

| \[1\] | A. Van Der Welle and B. Van Der Zwaan, "An overview of selected studies on the value of lost load (VOLL)," _Energy Research Centre of the Netherlands (ECN), Amsterdam,_ 2007. |
| --- | --- |
| \[2\] | F. De Sisternes, "Investment model for renewable electricity systems (IMRES): an electricity generation capacity expansion formulation with unit commitment constraints," _MIT, Cambridge MA,_ 2013. |
| \[3\] | NERC, "Glossary of Terms Used in NERC Reliability Standards," \[Online\]. Available: <http://www.nerc.com/pa/stand/glossary> of terms/glossary_of_terms.pdf. \[Accessed 12 December 2016\]. |
| \[4\] | B. Hobbs and C. Bothwell, "System Adequacy with Intermittent Resources : Capacity Value and Economic Distortions," ISO New England, 2016. \[Online\]. Available: <https://www.iso-ne.com/static-assets/documents/2016/09/PSPC09222016_A4_Cindy-Bothwell-Johns-Hopkins-University-System-Adequacy-with-Intermittent-Resources-Capacity-Value-and-Economic-Distortions.pdf>. \[Accessed 2 October 2016\]. |
| \[5\] | L. Baringo and A. J. Conejo, "Correlated wind-power production and electric load scenarios for investment decisions," _Applied Energy,_ vol. 101, pp. 475-482, 2013. |
| \[6\] | CESI, "Pan-Arab Regional Energy Trade Platform (PA-RETP). Power Market Study: Gas Comsumption Assessment for the Power Sector accross MENA Region," CESI, 2016. |
| \[7\] | CESI-Ramboll, "Feasibility Study of the Electrical Interconnection and Energy Trade between Arab Countries," Arab Fund for Economic and Social Development, 2014. |
| \[8\] | The World Bank, "Energy Efficiency Study in Lebanon," The World Bank, Washington, DC, 2009. |
| \[9\] | Presidency of the Council of Ministries Central Administration of Statistics, "Energy". |
| \[10\] | Palestinian Electricity Regulatory Council, "Annual Report 2011,," Palestinian Electricity Regulatory Council, 2011. |
| \[11\] | DSIRE, "N.C. Clean Energy Technology Center," DSIRE, \[Online\]. Available: <http://programs.dsireusa.org/system/program/tables>. \[Accessed 12 December 2016\]. |
| \[12\] | A. Hainoun, "Construction of the hourly load curves and detecting the annual peak load of future Syrian electric power demand using bottom-up approach," _International Journal of Electrical Power & Energy Systems,_ vol. 31, no. 1, pp. 1-12, 2009. |
| \[13\] | P. Khajavi, H. Monsef and H. Abniki, "Load profile reformation through demand response programs using smart grid," in _Proceedings of the International Symposium Modern Electric Power Systems (MEPS), 2010_, 2010. |
| \[14\] | A. Khazaee, M. Ghasempour, H. Hoseinzadeh and H. Hooshmandi, "DEMAND MANAGEMENT PROGRAM FOR LARGE INDUSTRIAL CONSUMERS BY USING THE AMI SYSTEM," in _23rd International Conference on Electricity Distribution_, 2015. |
| \[15\] | A. Salimi-Beni, D. Farrokhzad, M. Fotuhi-Firuzabad and S. Alemohammad, "A new approach to determine base, intermediate and peak-demand in an electric power system," in _International Conference on Power System Technology, 2006. PowerCon 2006._, 2006. |
| \[16\] | W. Short, P. Sullivan, T. Mai, M. Mowers, C. Uriarte, N. Blair, D. Heimiller and A. Martinez, "Regional Energy Deployment System ( ReEDS)," NREL, 2011. |




1. The generators already planned are included in any of the two sets depending on criteria such as their capacity, status of their construction process etc. [↑](#footnote-ref-2)

2. The electrolyzers already planned are included in any of the two sets depending on criteria such as their capacity, status of their construction process etc. [↑](#footnote-ref-3)

3. Type of resources considered as renewables might be different from country to country or state to state. For example, some states do not include hydropower towards their renewable targets (e.g. California does not count large hydropower towards the RPS (Renewable portfolio standard) while others such as Oregon do \[13\]. [↑](#footnote-ref-4)

4. Capital costs of existing generators are considered sunk costs and are not included in the objective function. [↑](#footnote-ref-5)

5. For the first model runs, capacity credit was considered at full capacity. This will be modified in the next runs but it is not expected to change significantly the model results. [↑](#footnote-ref-6)

6. Depending on the scope of the project, the dispatch constraint (14) might be slightly different. For example, ReEDS model implemented by NREL \[16\] treats quick start capacity service provided by a generator in the same way as spinning reserves under constraint (14) and on top of that, it accounts for planned and forced outages by considering average outage rates. [↑](#footnote-ref-7)

7. This constraint is necessary to ensure that the model does not simply build additional storage units just to meet planning or spinning reserve requirements. [↑](#footnote-ref-8)