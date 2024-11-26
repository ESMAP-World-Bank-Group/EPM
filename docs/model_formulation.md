
# Model formulation

## Objective function and its components

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

## Transmission network constraints

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

## System requirements

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

## Renewable generation

$$
\text{gen}_{g,f,q,d,t,y} \leq \text{RPprofile}_{g,\text{RE},q,d,y,t} \cdot \text{cap}_{g,y} \quad \forall \text{RE} \notin \text{CSP}
$$

## Concentrated Solar Power (CSP) Generation

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

## Time consistency of power system additions and retirements

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

## Environmental policy


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

## CCS retrofits


$$
\text{build}_{\text{ng},y} - \text{retire}_{\text{eg},y} \leq 0 \quad \forall \text{ng},\text{eg} \in \text{GG}
$$

## Green hydrogen production

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
