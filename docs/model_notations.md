
# Notations

## Indices and Sets

| Indices    | Set |
| --- | --- |
| $c \in C$   | where $C$ is the set of countries |
| $d \in D$     | where $D$ is the set of types of days or weeks |
| $f \in F$    | where $F$ is the set of fuels |
| $g \in G$     | where $G$ is the set of generators that can be built or the set of technology-specific types of aggregated generators |
| $q \in Q$    | where $Q$ is the set of seasons or quarters |
| $h \in H$    | where $H$ is the set of electrolyzers that can be built |
| $y \in Y$   | where $Y$ is the set of years considered in the planning model |
| $z, z_2 \in Z$    | where $Z$ is the set of zones/regions modeled |
| $sc \in S$   | where $S$ is the set of flags and penalties used to include/exclude certain features of the model |
| **Subsets considered** |     |
| $EN, NG \in G$    | where $EG$ and $NG$ is a partition of set $G$ and the former (EG) contains generators existing at the starting year of the planning horizon and the latter contains candidate generators<sup><sup>[\[1\]](#footnote-2)</sup></sup> |
| $DC, NDC \in G$    | where $DC$ and $NDC$ are partitions of set $G$ separating generators with a CAPEX cost constant over the modeling horizon and those generators that have costs varying over time. This feature is mainly developed to account for technologies that show cost reductions due to technological improvements and manufacturing advances. |
| $EH, NH \in H$    | where $EH$ and $NH$ is a partition of $H$ set and the former ($HG$) contains electrolyzers existing at the starting year of the planning horizon and the latter contains candidate electrolyzers<sup><sup>[\[2\]](#footnote-3)</sup></sup> |
| $DH, NDH \in H$    | where $DH$ and $NDH$ are partitions of set H separating electrolyzers with a CAPEX cost constant over the modeling horizon and those electrolyzers that have costs varying over time. This feature is mainly developed to account for technologies that show cost reductions due to technological improvements and manufacturing advances. |
| $so \in G$    | PV unit linked to a storage unit as a partition of set $G$ (this is the PV part of PV linked with storage) |
| $stp \in st$    | Storage unit linked to a PV plant as a partition of set $G$ (this is the storage part of PV linked with storage) |
| $MD \in D$    | where $MD$ is a subset of days the planner expects the minimum load levels to be binding |
| $RE \in F$    | where $F$ is a subset of set considered as renewable according to regulator's criteria<sup><sup>[\[3\]](#footnote-4)</sup></sup> |
| $RG \in G$    | Subset of renewable generators |
| $map_{c,z}$    | includes the subset of zones that correspond to each country |
| $map_{g,f}$   | includes valid combinations of fuels and generators; subset of the set $G x F$ |
| $map_{eg,ef}$    | includes combinations of existing and candidate generators; subset of set $EG x NG$ |
| $map_{g,z}$    | includes the subset of generators that correspond to each zone |
| $map_{h,z}$    | includes the subset of electrolyzers that correspond to each zone |

### Variables

| **Non-negative decision variables** |     |
| ---  | --- |
| $𝐴𝑑𝑑𝑖𝑡𝑖𝑜𝑛𝑎𝑙𝑇𝑟𝑎𝑛𝑠𝑓𝑒𝑟_{𝑧,𝑧2,𝑦}$    | Additional number of lines between z and z2 in year y |
| $𝑏𝑢𝑖𝑙𝑑_{g,y}$   | Generator investment in MW |
| $𝑏𝑢𝑖𝑙𝑑𝐻2_{h,y}$    | Electrolyzer investment in MW |
| $𝑏𝑢𝑖𝑙𝑡𝐶𝑎𝑝𝑉𝑎𝑟_{g,y}$    | Integer variable to model discrete unit capacity |
| $𝑏𝑢𝑖𝑙𝑡𝐶𝑎𝑝𝑉𝑎𝑟𝐻2_{h,y}$    | Integer variable to model discrete unit capacity |
| $𝑏𝑢𝑖𝑙𝑡𝑇𝑟𝑎𝑛𝑠𝑚𝑖𝑠𝑠𝑖𝑜𝑛_{z,z2,y}$    | Integer variable to model number of new transmission lines added between z and z2 in year y |
| $𝑐𝑎𝑝_{g,y}$    | Generator capacity available at year in MW |
| $𝑐𝑎𝑝𝐻2_{h,y}$    | Electrolyzer capacity available at year in MW |
| $𝐶𝑂2𝑏𝑎𝑐𝑘𝑠𝑡𝑜𝑝_{c,y}$    | Annual country CO2 emissions above the system emissions constraint cap |
| $𝑒𝑚𝑖𝑠𝑠𝑖𝑜𝑛𝑠_{z,y}$    | Total system emissions of carbon dioxide in tons |
| $𝑒𝑚𝑖𝑠𝑠𝑖𝑜𝑛𝑠_𝑍𝑜_{z,y}$    | Emissions of carbon dioxide in tons per zone |
| $𝑒𝑥𝑝𝑜𝑟𝑡𝑃𝑟𝑖𝑐𝑒_{z,q,d,t,y}$    | External price driven export in MW |
| $𝑓𝑢𝑒𝑙𝐻2𝑄𝑢𝑎𝑟𝑡𝑒𝑟_{z,q,y}$    | Hydrogen fuel consumption for electricity production hydrogen produced intended for re-circulation in the power sector in (MMBTU per quarter and year) |
| $𝑓𝑢𝑒𝑙𝐻2_{z,q,y}$    | Hydrogen fuel consumption for electricity production hydrogen produced intended for re-circulation in the power sector in (MMBTU per year) |
| $𝑓𝑢𝑒𝑙_{z,f,y}$    | Fuel consumption in MMBTU |
| $𝑔𝑒𝑛_{g,f,q,d,t,y}$    | Generator output in MW |
| $𝑔𝑒𝑛𝐶𝑆𝑃_{g,z,q,d,t,y}$    | Power output of the solar panel in MW |
| $𝑖𝑚𝑝𝑜𝑟𝑡𝑃𝑟𝑖𝑐𝑒_{z,q,d,t,y}$    | External price driven import in MW |
| $𝐻2𝑃𝑤𝑟𝐼𝑛_{h,q,d,t,y}$    | Power drawn by electrolyzer h in MW |
| $𝑟𝑒𝑡𝑖𝑟𝑒_{g,y}$    | Capacity in MW retired |
| $𝑟𝑒𝑡𝑖𝑟𝑒𝐶𝑎𝑝𝑉𝑎𝑟_{g,y}$    | Supplementary integer variable to model discrete unit capacity |
| $𝑟𝑒𝑡𝑖𝑟𝑒𝐶𝑎𝑝𝑉𝑎𝑟𝐻2_{h,y}$    | Supplementary integer variable to model discrete unit capacity |
| $𝑟𝑒𝑠𝑒𝑟𝑣𝑒_{g,q,d,t,y}$    | Spinning reserve requirement met in MW |
| $𝑠𝑡𝑜𝑟𝑎𝑔𝑒_{st,q,d,t,y}$    | Level of energy in MWh stored in storage unit |
| $𝑠𝑡𝑜𝑟𝑎𝑔𝑒_𝑖𝑛𝑗_{st,q,d,t,y}$    | Power injected in MW in storage unit is charged during hour |
| $𝑠𝑡𝑜𝑟𝑎𝑔𝑒_𝑐𝑎𝑝𝑎𝑐𝑖𝑡𝑦_{st,q,d,t,y}$    | Total deployed energy storage capacity in MWh for storage unit |
| $𝑠𝑡𝑜𝑟𝑎𝑔𝑒𝐶𝑆𝑃_{g,z,q,d,t,y}$    | Level of energy in MWh stored in CSP unit at zone |
| $𝑠𝑡𝑜𝑟𝑎𝑔𝑒𝐶𝑆𝑃𝑖𝑛𝑗_{g,z,q,d,t,y}$    | Power level in MW at which the CSP storage unit is charged during hour |
| $𝑠𝑡𝑜𝑟𝑎𝑔𝑒𝐶𝑆𝑃𝑜𝑢𝑡_{g,z,q,d,t,y}$    | Power level in MW at which the CSP storage unit is discharged during hour |
| $𝑠𝑢𝑟𝑝𝑙𝑢𝑠_{z,q,d,t,y}$    | Surplus generation in MW |
| $𝑢𝑛𝑚𝑒𝑡𝐷𝑒𝑚_{z,q,d,t,y}$    | Annual system CO2 emissions above the system emissions constraint cap |
| $𝑢𝑛𝑚𝑒𝑡𝐻2𝐸𝑥𝑡𝑒𝑟𝑛𝑎𝑙_{z,q,y}$    | Active power in MW flowing from to |
| $𝑢𝑛𝑚𝑒𝑡𝑅𝑒𝑠_{z,y}$    | Unmet demand in MW (or equivalently violation of the load balance constraint) |
| $𝑢𝑛𝑚𝑒𝑡𝑆𝑅𝑒𝑠𝑍𝑜_{z,q,d,t,y}$    | Unmet quantity of hydrogen in MW (or equivalently violation of the external demand for hydrogen balance constraint) |
| $𝑢𝑛𝑚𝑒𝑡𝑆𝑅𝑒𝑠𝑆𝑌_{q,d,t,y}$    | Violation of the planning reserve constraint in MW |
| $unmetSResZo$    | Violation of the zonal/regional spinning reserve constraint in MW |
| $unmetSResSY$    | Violation of the system-level spinning reserve constraint in MW |

| **Variables for modeling objective function** |     |
| --- | --- |
| $npvcost = \sum_{z,y} ReturnRate_y \cdot WeightYear_y \cdot totalcost_{z,y}$ | Net present value of power system cost over the whole planning horizon; objective function that optimization model tries to minimize |
| $totalcost_{z,y} = fixedcost_{z,y} + variablecost_{z,y} + reservecost_{z,y} + usecost_{z,y} + usrcost_{z,y} + carboncost_{z,y} + CO2backCost_y + TradeCost_{z,y} + CostTransmissionAdditions_{z1,z2,y} + CurtailmentCost_{z,y} + SurplusCost_{z,y} + usrcostH2_{z,y}$ | Annual system cost in constant prices |

## Model formulation

### Objective function and its components

The objective function minimizes the net present value of total system costs:

$$
\text{npvcost} = \sum_{z,y} (\text{ReturnRate}_y \cdot \text{WeightYear}_y \cdot \text{totalcost}_{z,y})
$$

The total cost for each zone z and year y consists of:

$$
\text{totalcost}_{z,y} = \text{fixedcost}_{z,y} + \text{variablecost}_{z,y} + \text{reservecost}_{z,y} + \text{usecost}_{z,y} + \text{usrcost}_{z,y} + \text{carboncost}_{z,y} + \text{CO2backCost}_y + \text{TradeCost}_{z,y} + \text{CostTransmissionAdditions}_{z1,z2,y} + \text{CurtailmentCost}_{z,y} + \text{SurplusCost}_{z,y} + \text{usrcostH2}_{z,y}
$$

Fixed costs include capital recovery and fixed O&M for both generation and hydrogen units:

$$
\text{fixedcost}_{z,y} = \sum_{g \in \text{NG} \cap \text{NDC}} (\text{CRF}_{\text{NG}} \cdot \text{CapCost}_{\text{NG},y} \cdot \text{cap}_{g,y}) +
\sum_{g \in \text{NG} \cap \text{DC}} \text{AnnCapex}_{\text{NG},y} +
\sum_g (\text{FixedOM}_{g,y} \cdot \text{cap}_{g,y}) +
\sum_{h \in \text{NH} \cap \text{NDH}} (\text{CRF}_{\text{NH}} \cdot \text{CapCostH2}_{\text{NH},y} \cdot \text{cap}_{h,y}) +
\sum_{h \in \text{NH} \cap \text{DH}} \text{AnnCapexH2}_{\text{NH},y} +
\sum_h (\text{FixedOMH2}_{h,y} \cdot \text{capH2}_{h,y})
$$

Annualized capital expenditure for generation and hydrogen units:

$$
\text{AnnCapex}_{g \in \text{DC},y} = \text{CFR}_{g \in \text{DC}} \cdot \text{CapCost}_{g \in \text{DC}} \cdot \text{CapexTrajectories}_{g \in \text{DC},y} \cdot \text{build}_{g \in \text{DC},y}
$$

$$
\text{AnnCapexH2}_{h \in \text{DH},y} = \text{CFR}_{h \in \text{DH}} \cdot \text{CapCost}_{h \in \text{DH}} \cdot \text{CapexTrajectories}_{h \in \text{DH},y} \cdot \text{build}_{h \in \text{DH},y}
$$

Variable costs include generation fuel costs and hydrogen production costs:

$$
\text{variablecost}_{z,y} = \sum_{g \in \text{G},f,q,d,t} (\text{GenCost}_{g,f,y} \cdot \text{Duration}_{q,d,t,y} \cdot \text{gen}_{g,f,q,d,t,y}) +
\sum_{h \in \text{H},q,d,t} (\text{VOM}_{\text{H2},h,y} \cdot \text{EfficiencyH2}_h \cdot \text{vH2PwrIn}_{h,g,q,d,t,y} \cdot \text{pHours}_{q,d,t,y})
$$

Reserve costs for maintaining system reliability:

$$
\text{reservecost}_{z,y} = \sum_{g \in \text{Z},q,d,t} (\text{ResCost}_g \cdot \text{Duration}_{q,d,t,y} \cdot \text{reserve}_{g,q,d,t,y})
$$

Costs associated with unmet demand:

$$
\text{usecost}_{z,y} = \sum_{q,d,t} (\text{VOLL} \cdot \text{Duration}_{q,d,t,y} \cdot \text{unmetDem}_{z,q,d,t,y})
$$

Costs for unmet reserve requirements:

$$
\text{usrcost}_{z,y} = \sum_{q,d,t} (\text{RESVoLL} \cdot \text{unmetRes}_{z,y}) +
\sum_{z,q,d,t,y} (\text{Duration}_{q,d,t,y} \cdot \text{SRESVoLL} \cdot \text{unmetSResZo}_{z,q,d,t,y}) +
\sum_{q,d,t} (\text{Duration}_{q,d,t,y} \cdot \text{SRESVoLL} \cdot \text{unmetSResSY}_{q,d,t,y})
$$

Carbon-related costs including emissions tax and backstop measures:

$$
\text{carboncost}_{z,y} = \sum_{g \in \text{Z},f,q,d,t} (\text{Duration}_{q,d,t,y} \cdot \text{carbon\_tax}_y \cdot \text{HeatRate}_{g,f} \cdot \text{carbon\_emission}_f \cdot \text{gen}_{g,f,q,d,t,y}) +
\sum_{z \in \text{c}} (\text{CO2backstop}_{z,y} \cdot \text{CostOfCO2backstop}) +
\text{SysCO2backstop}_y \cdot \text{CostOfCO2backstop}
$$

Additional cost components:

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

Power balance constraint for each zone:

$$
\sum_{g \in Z,f} \text{gen}_{g,f,q,d,t,y} - \sum_{z2} \text{trans}_{z,z2,q,d,t,y} +
\sum_{z2} \left( (1 - \text{LossFactor}_{z,z2,y}) \cdot \text{trans}_{z2,z,q,d,t,y} \right) +
\text{storage\_out}_{z,q,d,t,y} - \text{storage\_inj}_{z,q,d,t,y} +
\text{unmetDem}_{z,q,d,t,y} - \text{surplus}_{z,q,d,t,y} +
\text{importPrice}_{z,q,d,t,y} - \text{exportPrice}_{z,q,d,t,y} - \sum_{h \in \text{map}(h,z)} \text{H2PwrIn}_{h,q,d,t,y} =
\text{Demand}_{z,q,d,t,y} \cdot \text{EEfactor}_{z,y}
$$

Transmission capacity constraints:

$$
\text{trans}_{z,z2,q,d,t,y} \leq \text{TransLimit}_{z,z2,q,y} + \text{AdditionalTransfer}_{z,z2,y} \cdot \text{TransCapPerLine}_{z,z2}
$$

$$
\text{AdditionalTransfer}_{z,z2,y} = \text{AdditionalTransfer}_{z,z2,y-1} + \text{BuildTransmission}_{z,z2,y}
$$

$$
\text{AdditionalTransfer}_{z,z2,y} = \text{AdditionalTransfer}_{z2,z,y}
$$

Import/export constraints:

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

System-wide reserve requirements:

$$
\sum_{g \in z} \text{reserve}_{g,q,d,t,y} +
\text{unmetSResSY}_{z,q,d,t,y} +
\sum_{z2} \left( \text{TransLimit}_{z,z2,q,y} +
\text{AdditionalTransfer}_{z,z2,y} \cdot \text{TransCapPerLine}_{z,z2} - \text{trans}_{z2,z,q,d,t,y} \right) \geq
\text{SResSY}_{z,y} + 0.2 \cdot \sum_{VRE \in z} \text{gen}_{VRE,f,q,d,t,y} \quad \forall z,q,d,t,y
$$

Reserve offer limits:

$$
\text{reserve}_{g,q,d,t,y} \leq \text{cap}_{g,y} \cdot \text{ResOffer}_g
$$

Planning reserve margin requirements:

$$
\sum_{g \in Z} \text{cap}_{g,y} \cdot \text{CapCredit}_{g,y} +
\text{unmetRes}_{z,y} +
\sum_{z2} \sum_q \left( \text{TransLimit}_{z,z2,q,y} +
\text{AdditionalTransfer}_{z,z2,y} \cdot \text{TransCapPerLine}_{z,z2} - \text{trans}_{z2,z,q,d,t,y} \right) \geq
(1 + \text{PRM}_z) \cdot \max_{q,d,t} \text{Demand}_{z,q,d,t,y} \quad \forall z,y
$$

### Generation constraints

Generation capacity and reserve constraints:

$$
\sum_f \text{gen}_{g,f,q,d,t,y} + \text{reserve}_{g,q,d,t,y} \leq (1 + \text{OverLoadFactor}_g) \cdot \text{cap}_{g,y}
$$

$$
\text{reserve}_{g,q,d,t,y} \leq \text{cap}_{g,y} \cdot \text{ResOffer}_g
$$

Ramping constraints:

$$
\sum_f \text{gen}_{g,f,q,d,t-1,y} - \sum_f \text{gen}_{g,f,q,d,t,y} \leq \text{cap}_{g,y} \cdot \text{RampDn}_g \quad \forall t > 1
$$

$$
\sum_f \text{gen}_{g,f,q,d,t,y} - \sum_f \text{gen}_{g,f,q,d,t-1,y} \leq \text{cap}_{g,y} \cdot \text{RampUp}_g \quad \forall t > 1
$$

Minimum capacity factor requirements:

$$
\sum_f \text{gen}_{g,f,q,d,t,y} \geq \text{MinCapFac}_g \cdot \text{cap}_{g,y} \quad \forall d \in M
$$

Availability constraints:

$$
\sum_{f,d,t} \left( \text{Duration}_{q,d,t,y} \cdot \text{gen}_{g,f,q,d,t,y} \right) \leq
\text{Availability}_{g,q} \cdot \sum_{d,t} \left( \text{Duration}_{q,d,t,y} \cdot \text{cap}_{g,y} \right)
$$

### Renewable generation

$$
\text{gen}_{g,f,q,d,t,y} \leq \text{RPprofile}_{g,\text{RE},q,d,y,t} \cdot \text{cap}_{g,y} \quad \forall \text{RE} \notin \text{CSP}
$$

### Concentrated Solar Power (CSP) Generation

The following equations govern the operation of CSP plants, including their thermal storage capabilities and generation limits.

$$
\text{storageCSP}_{g,z,q,d,t,y} \leq \text{cap}_{g,y} \cdot \text{CSP\_storage} \quad \forall \text{map}(g,\text{CSP})
$$

Storage level cannot exceed storage capacity.

$$
\text{genCSP}_{g,z,q,d,t,y} = \text{RPprofile}_{z,\text{RE} \in \text{CSP},q,d,t} \cdot \text{cap}_{g,y} \cdot
\frac{\text{SolarMultipleCSP}}{\text{TurbineEfficiency\_CSP} \cdot \text{FieldEfficiency\_CSP}}
$$

CSP generation is determined by solar resource profile and plant characteristics.

$$
\sum_{f \in \text{CSP}} \text{gen}_{g,f,q,d,t,y} \leq \text{cap}_{g,y}
$$

Total CSP generation cannot exceed capacity.

$$
\sum_{f \in \text{CSP}} \left( \text{genCSP}_{g,z,q,d,t,y} \cdot \text{FieldEfficiency\_CSP} - \text{storageCSPinj}_{g,z,q,d,t,y} + \text{storageCSPout}_{g,z,q,d,t,y} \right) =
\frac{\text{gen}_{g,f,q,d,t,y}}{\text{TurbineEfficiency\_CSP}} \quad \forall g,z,q,d,t,y
$$

Energy balance equation for CSP operation.

$$
\text{storageCSP}_{g,z,q,d,t,y} = \text{storageCSP}_{g,z,q,d,t-1,y} + \text{storageCSPinj}_{g,z,q,d,t,y} - \text{storageCSPout}_{g,z,q,d,t,y}
$$

Storage level evolution over time.

### Time Consistency of Power System Additions and Retirements

These equations ensure proper tracking of capacity additions and retirements over time.

$$
\text{cap}_{g \in \text{EG},y} = \text{cap}_{\text{EG},y-1} + \text{build}_{\text{EG},y} - \text{retire}_{\text{EG},y} \quad \forall (y, g \in \text{EG}), \forall \text{ord}(y) > 1
$$

Capacity evolution for existing generators.

$$
\text{cap}_{g \in \text{EG},y} = \text{GenCap}_g \quad \forall (y, g \in \text{EG}): (\text{ord}(y) = 1 \wedge \text{StartYear} > \text{Commission\_year}_g)
$$

Initial capacity for existing generators.

$$
\text{cap}_{g \in \text{EG},y} = \text{GenCap}_g \quad \forall (y, g \in \text{EG}): (\text{Commission\_year}_g \geq \text{StartYear} \wedge \text{ord}(y) \geq \text{Commission\_year}_g \wedge \text{ord}(y) < \text{Retirement\_year}_{\text{EG}})
$$

Capacity during operational period.

$$
\text{cap}_{g \in \text{EG},y} = 0 \quad \forall (y, g \in \text{EG}): (\text{ord}(y) \geq \text{Retirement\_year}_{\text{EG}})
$$

Zero capacity after retirement.

$$
\text{cap}_{g \in \text{NG},y} = \text{cap}_{\text{NG},y-1} + \text{build}_{\text{NG},y} \quad \forall \text{ord}(y) > 1
$$

Capacity evolution for new generators.

$$
\text{cap}_{g \in \text{NG},y} = \text{build}_{\text{NG},y} \quad \forall \text{ord}(y) = 1
$$

Initial capacity for new generators.

$$
\text{cap}_g \leq \text{GenCap}_g
$$

Maximum capacity constraint.

$$
\text{cap}_{g,y} = 0 \quad \forall (y, g): (\text{ord}(y) < \text{Commission\_year}_g)
$$

Zero capacity before commissioning.

### Storage Constraints

The following equations govern the operation of storage systems.

$$
\text{storage}_{\text{st},z,q,d,t=1,y} = \text{Storage\_efficiency}_{\text{st}} \cdot \text{storage\_inj}_{g,z,q,d,t=1,y} - \text{gen}_{\text{st},z,q,d,t=1,y} \quad \forall t=1
$$

Initial storage level.

$$
\text{storage}_{\text{st},z,q,d,t>1,y} = \text{storage}_{\text{st},z,q,d,t-1,y} +
\text{Storage\_efficiency}_{\text{st}} \cdot \text{storage\_inj}_{\text{st},z,q,d,t-1,y} - \text{gen}_{\text{st},z,q,d,t-1,y} \quad \forall t>1
$$

Storage level evolution.

$$
\text{storage\_inj}_{\text{stp},q,d,t,y} = \text{Cap}_{\text{so},y} \cdot \text{pStoragePVProfile}_{\text{so},q,d,t,y}
$$

Storage injection for PV storage.

$$
\text{storage\_inj}_{\text{st},q,d,t,y} \leq \text{Cap}_{\text{st},y}
$$

Maximum injection rate.

$$
\text{storage}_{\text{st},q,d,t,y} \leq \text{Storage\_Capacity}_{\text{st},y}
$$

Maximum storage level.

$$
\text{storage\_inj}_{\text{st},q,d,t,y} - \text{storage\_inj}_{z,q,d,t-1,y} \leq \text{Cap}_{\text{st},y} \cdot \text{RampDn}_{\text{st}} \quad \forall t>1
$$

Ramping down constraint.

$$
\text{storage\_inj}_{\text{st},q,d,t-1,y} - \text{storage\_inj}_{z,q,d,t,y} \leq \text{Cap}_{\text{st},y} \cdot \text{RampUp}_{\text{st}} \quad \forall t>1
$$

Ramping up constraint.

$$
\text{Storage\_Capacity}_{\text{EG},y} = \text{Storage\_Capacity}_{\text{EG},y-1} +
\text{Build\_Storage\_Capacity}_{\text{EG},y-1} - \text{Retire}_{\text{Storage\_Capacity}_{\text{EG},y-1}} \quad \forall y>1
$$

Storage capacity evolution for existing storage.

$$
\text{Storage\_Capacity}_{\text{NG},y} = \text{Storage\_Capacity}_{\text{NG},y-1} +
\text{Build\_Storage\_Capacity}_{\text{NG},y-1} \quad \forall y>1
$$

Storage capacity evolution for new storage.

$$
\text{Storage\_Capacity}_{\text{NG},y} = \text{Build\_Storage\_Capacity}_{\text{NG},y-1} \quad \forall y=1
$$

Initial storage capacity for new storage.

$$
\text{StorageCapacity}_{\text{st},y} \geq \text{Cap}_{\text{st},y}
$$

Minimum storage capacity requirement.

$$
\text{StorageCapacity}_{\text{st},y} \leq \text{Maximum\_Storage\_Energy}_{\text{st},y} + \text{Maximum\_Storage\_Energy}_{\text{CSP},y}
$$

Maximum storage capacity limit.

$$
\sum_{y} \text{Build\_Storage\_Capacity}_{\text{EG},y} \leq \text{Maximum\_Storage\_Energy}_{\text{EG},y}
$$

Total storage capacity build limit.

### Investment Constraints

These equations govern investment decisions for new capacity.

$$
\sum_y \text{build}_{g \in \text{NG},y} \leq \text{MaxNewCap}_{\text{NG}}
$$

Maximum total new capacity.

$$
\sum_y \text{build}_{g \in \text{NG},y} \geq \text{MinNewCap}_{\text{NG}}
$$

Minimum total new capacity.

$$
\sum_y \text{build}_{g \in \text{EG},y} \leq \text{GenCap}_{g \in \text{EG}} \quad \forall g: (\text{Commission\_year}_g > \text{StartYear})
$$

Maximum build limit for existing generators.

$$
\text{build}_{g \in \text{NG},y} \leq \text{Annual\_built\_limit}_y \cdot \text{WeightYear}_y
$$

Annual build limit.

$$
\text{build}_{g \in \text{NG},y} = \text{DiscreteCap}_{g \in \text{NG}} \cdot \text{builtCapVar}_{g \in \text{NG},y} \quad \forall g: (\text{DiscreteCap}_g \geq 0)
$$

Discrete capacity additions.

$$
\text{retire}_{g \in \text{NG},y} = \text{DiscreteCap}_{g \in \text{NG}} \cdot \text{retireCapVar}_{g \in \text{NG},y} \quad \forall g: (\text{DiscreteCap}_g \geq 0)
$$

Discrete capacity retirements.

$$
\text{build}_{g,y} = 0 \quad \forall y: (\text{ord}(y) \leq \text{Commission\_year}_g)
$$

No builds before commission year.

$$
\text{fuel}_{z,f,y} \leq \text{MaxFuelOff}_{f,y}
$$

Fuel availability constraint.

$$
\text{fuel}_{z,f,y} = \sum_{g \in Z,q,d,t} \left( \text{Duration}_{q,d,t,y} \cdot \text{HeatRate}_{g,f} \cdot \text{gen}_{g,f,q,d,t,y} \right)
$$

Fuel consumption calculation.

$$
\sum_{y,g \in \text{NG}} \left( \text{ReturnRate}_y \cdot \text{pweight}_y \cdot \text{CRF}_{\text{NG}} \cdot \text{CapCost}_{\text{NG},y} \cdot \text{cap}_{g,y} \right) \leq \text{MaxCapital}
$$

Capital investment constraint.

### Environmental Policy

These equations enforce environmental constraints.

$$
\text{emissions\_Zo}_{z,y} = \sum_{g \in Z,q,d,t} \left( \text{gen}_{g,f,q,d,t,y} \cdot \text{HeatRate}_{g,f} \cdot \text{carbon\_emission}_f \cdot \text{Duration}_{q,d,t,y} \right)
$$

Zone emissions calculation.

$$
\text{emissions\_Zo}_{z,y} \leq \text{Zo\_emission\_cap}_{y,z}
$$

Zone emissions cap.

$$
\text{emissions}_{z,y} = \sum_{g,q,d,t} \left( \text{gen}_{g,f,q,d,t,y} \cdot \text{HeatRate}_{g,f} \cdot \text{carbon\_emission}_f \cdot \text{Duration}_{q,d,t,y} \right)
$$

System emissions calculation.

$$
\text{emissions}_{z,y} \leq \text{Sy\_emission\_cap}_y
$$

System emissions cap.

### CCS Retrofits

$$
\text{build}_{\text{ng},y} - \text{retire}_{\text{eg},y} \leq 0 \quad \forall \text{ng},\text{eg} \in \text{GG}
$$

CCS retrofit constraint.

### Green Hydrogen Production

These equations govern green hydrogen production facilities.

$$
\text{capH2}_{\text{eh},y} = \text{capH2}_{\text{eh},y-1} + \text{buildH2}_{\text{eh},y} - \text{retireH2}_{\text{eh},y} \quad \forall \text{ord}(y) > 1
$$

Existing hydrogen capacity evolution.

$$
\text{capH2}_{\text{nh},y} = \text{capH2}_{\text{nh},y-1} + \text{buildH2}_{\text{nh},y} \quad \forall \text{ord}(y) > 1
$$

New hydrogen capacity evolution.

$$
\text{capH2}_{h,y} = \text{H2Cap}_{h \in \text{EH},y} + \text{buildH2}_{h,y} - \text{retireH2}_{\text{EH},y} \quad \forall \text{ord}(y) = 1
$$

Initial hydrogen capacity.

$$
\sum_{y \in Y} \text{buildH2}_{\text{eh},y} \leq \text{H2Cap}_{\text{eh}}
$$

Maximum existing hydrogen capacity.

$$
\sum_{y \in Y} \text{buildH2}_{\text{nh},y} \leq \text{MaxNewCapH2}_{\text{nh}}
$$

Maximum new hydrogen capacity.

$$
\text{buildH2}_{\text{nh},y} = \text{UnitSizeH2}_{\text{nh}} \cdot \text{buildCapVarH2}_{\text{nh},y}
$$

Discrete hydrogen capacity additions.

$$
\text{retireH2}_{\text{eh},y} = \text{UnitSizeH2}_{\text{eh}} \cdot \text{retireCapVarH2}_{\text{eh},y}
$$

Discrete hydrogen capacity retirements.

$$
\sum_{d,t} \left( \text{H2PwrIn}_{h,q,d,t,y} \cdot \text{Duration}_{q,d,t,y} \right) \leq \text{AvailabilityH2}_{h,q} \cdot \text{CapH2}_{h,y} \cdot \sum_{d,t} \text{Duration}_{q,d,t,y}
$$

Hydrogen production availability constraint.

$$
\sum_z \left( \text{ExternalH2}_{z,q,y} - \text{unmetH2External}_{z,q,y} + \text{fuelH2Quarter}_{z,q,y} \right) \leq \sum_{z,h \in \text{map}(h,z)} \left( \text{H2PwrIn}_{h,q,d,t,y} \cdot \text{Duration}_{q,d,t,y} \cdot \text{EfficiencyH2}_h \right)
$$

Hydrogen balance equation.

$$
\sum_z \text{fuelH2Quarter}_{z,q,y} = \text{fuelH2}_{z,y}
$$

Quarterly hydrogen fuel balance.

$$
\text{H2PwrIn}_{h,q,d,t,y} \leq \text{CapH2}_{h,y}
$$

Maximum hydrogen production rate.

$$
\text{H2PwrIn}_{h,q,d,t-1,y} - \text{H2PwrIn}_{h,q,d,t,y} \leq \text{CapH2}_{h,y} \cdot \text{RampDnH2}_h
$$

Hydrogen production ramping down constraint.

$$
\text{H2PwrIn}_{h,q,d,t,y} - \text{H2PwrIn}_{h,q,d,t-1,y} \leq \text{CapH2}_{h,y} \cdot \text{RampUpH2}_h
$$

Hydrogen production ramping up constraint.

$$
\text{Gen}_{\text{RE},f,d,t,y} = \text{REPwr2Grid}_{\text{RE},f,d,t,y} + \text{REPwr2H2}_{\text{RE},f,d,t,y}
$$

Renewable energy allocation between grid and hydrogen.

$$
\sum_{z,h \in \text{map}(h,z)} \text{H2PwrIn}_{h,q,d,t,y} = \text{PwrREH2}_{z,q,d,t,y}
$$

Power input to hydrogen production.

$$
\sum_{\text{RE},f,z \in (\text{map}(\text{RE},f),\text{map}(\text{RE},z))} \text{REPwr2H2}_{\text{RE},f,q,d,t,y} = \text{PwrREH2}_{z,q,d,t,y}
$$

Renewable power to hydrogen balance.

$$
\sum_{\text{RE},f,z \in (\text{map}(\text{RE},f),\text{map}(\text{RE},z))} \text{REPwr2Grid}_{\text{RE},f,q,d,t,y} = \text{PwrREGrid}_{z,q,d,t,y}
$$

Renewable power to grid balance.
