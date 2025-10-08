**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
* Description:
* This GAMS-based model is designed for electricity system planning, 
* incorporating capacity expansion, generation dispatch, and policy 
* constraints such as renewable energy targets, emissions reductions, 
* and market mechanisms.
*
* Author(s): ESMAP Modelling Team
* Organization: World Bank
* Version: 
* License: Creative Commons Zero v1.0 Universal
*
* Key Features:
* - Optimization of electricity generation and capacity planning
* - Inclusion of renewable energy integration and storage technologies
* - Multi-period, multi-region modeling framework
* - CO2 emissions constraints and policy instruments
*
* Notes:
* - Ensure GAMS is installed before running this model.
* - The model requires input data in .GDX or Excel format.
*
* Contact:
* Claire Nicolas, c.nicolas@worldbank.org
**********************************************************************


$onMulti
* Report

Parameters

* ============================================================
* 1. CAPACITY
* ============================================================

  pCapacityPlant(z,g,y)                      'Installed capacity [MW] by plant, zone, and year'
  pCapacityTechFuel(z,tech,f,y)              'Installed capacity [MW] by technology, fuel, and zone'
  pCapacityFuel(z,f,y)                       'Installed capacity [MW] by fuel and zone'
  pCapacityTechFuelCountry(c,tech,f,y)       'Installed capacity [MW] by technology, fuel, and country'
  pCapacityFuelCountry(c,f,y)                'Installed capacity [MW] by fuel and country'

  pRetirementsPlant(z,g,y)                   'Retired capacity [MW] by plant, zone, and year'
  pRetirementsFuel(z,f,y)                    'Retired capacity [MW] by fuel and zone'
  pRetirementsCountry(c,y)                   'Total retired capacity [MW] by country and year'
  pRetirementsFuelCountry(c,f,y)             'Retired capacity [MW] by fuel and country'
  
  pNewCapacityFuel(z,f,y)                    'Newly added capacity [MW] by fuel and zone'
  pNewCapacityTech(z,tech,y)                 'Newly added capacity [MW] by technology and zone'
  pNewCapacityFuelCountry(c,f,y)             'Newly added capacity [MW] by fuel and country'
  pNewCapacityTechCountry(c,tech,y)          'Newly added capacity [MW] by technology and country'
  
  pAnnualTransmissionCapacity(z,z2,y)        'Total available transmission capacity [MW] between internal zones'
  pAdditionalCapacity(z,z2,y)                'Additional transmission capacity [MW] between internal zones'

  pCapacitySummary(z,*,y)                    'Summary of capacity indicators [MW] by zone and year'
  pCapacitySummaryCountry(c,*,y)             'Summary of capacity indicators [MW] by country and year'
  
  pCapacityPlantH2(z,hh,y)                   'Capacity plan of electrolyzers [MW] by zone'

* ============================================================
* 2. COSTS
* ============================================================

  pCostsPlant(z, g, *, y)                       'Yearly cost breakdown by plant and year'
  pCapexInvestment(z, y)                        'Annual CAPEX investment in USD by zone and year'

  pPrice(z, q, d, t, y)                         'Marginal cost [USD/MWh] by zone, time, and year'
  pImportCostsInternal(z, y)                    'Import costs with internal zones [USD] by zone and year'
  pExportRevenuesInternal(z, y)                 'Export revenues with internal zones [USD] by zone and year'
  pCongestionRevenues(z, z2, y)                 'Congestion rents [USD] from saturated lines z→z2 by year'
  pTradeSharedBenefits(z, y)                    'Congestion rents shared equally between countries [USD] by zone and year'

  pYearlyCostsZone(z, *, y)                      'Annual cost summary [million USD] by zone and year'
  pCostsZone(z, *)                          'Total cost [million USD] by zone and cost category'
  pYearlyCostsCountry(c, *, y)                   'Annual cost summary [million USD] by country and year'
  pCostAverageCountry(c, *)                     'Average annual cost [million USD] by country (undiscounted)'
  pYearlyCostsSystem
  pCostsSystem(*)                                'System-level cost summary [million USD], weighted and discounted'

  pFuelCosts(z, f, y)                           'Annual fuel costs [million USD] by fuel, zone, and year'
  pFuelCostsCountry(c, f, y)                    'Annual fuel costs [million USD] by fuel, country, and year'
  pFuelConsumption(z, f, y)                     'Annual fuel consumption [MMBtu] by fuel, zone, and year'
  pFuelConsumptionCountry(c, f, y)              'Annual fuel consumption [MMBtu] by fuel, country, and year'

* ============================================================
* 3. ENERGY BALANCE
* ============================================================

  pEnergyPlant(z,g,y)                     'Annual energy generation by plant [GWh]'
  pEnergyFuel(z,f,y)                      'Annual energy generation by fuel and zone [GWh]'
  pEnergyFuelCountry(c,f,y)               'Annual energy generation by fuel and country [GWh]'
  pEnergyTechFuel(z,tech,f,y)             'Annual energy generation by technology, fuel, and zone [GWh]'
  pEnergyTechFuelCountry(c,tech,f,y)      'Annual energy generation by technology, fuel, and country [GWh]'
  
  pEnergyBalance(z,*,y)                   'Annual supply-demand balance by zone [GWh]'
  pEnergyBalanceCountry(c,*,y)            'Annual supply-demand balance by country [GWh]'
  pEnergyBalanceH2(z,*,y)                 'Annual hydrogen supply-demand balance by zone [mmBTU]'
  pEnergyBalanceCountryH2(c,*,y)          'Annual hydrogen supply-demand balance by country [mmBTU]'
  
  pUtilizationPlant(z,g,y)                'Annual plant utilization factor'
  pUtilizationTech(z,tech,y)              'Annual technology utilization factor'
  pUtilizationFuel(z,f,y)                 'Annual average capacity factor by fuel'
  pUtilizationTechFuel(z,tech,f,y)        'Annual average capacity factor by technology and fuel'
  pUtilizationFuelCountry(c,f,y)          'Annual average capacity factor by fuel and country'
  pUtilizationTechFuelCountry(c,tech,f,y) 'Annual average capacity factor by technology, fuel, and country'

* ============================================================
* 4. ENERGY DISPATCH
* ============================================================

  pDispatchPlant(z, y, q, d, g, t, *)      'Plant-level hourly dispatch and reserve [MW]'
  pDispatchFuel(z, y, q, d, f, t)          'Fuel-level hourly dispatch [MW]'
  pDispatch(z, y, q, d, *, t)              'Zone-level hourly dispatch and flows [MW]'

* ============================================================
* 5. RESERVES
* ============================================================

  pReserveSpinningPlantZone(z,g,y)        'Spinning reserve provided by plant [MWh] per zone and year'
  pReserveSpinningFuelZone(z,f,y)         'Spinning reserve provided by fuel [MWh] per zone and year'
  pReserveSpinningPlantCountry(c,g,y)     'Spinning reserve provided by plant [MWh] per country and year'
  
  pReserveMargin(z,*,y)                   'Reserve margin indicators by zone and year'
  pReserveMarginCountry(c,*,y)            'Reserve margin indicators by country and year'

* ============================================================
* 6. INTERCONNECTIONS
* ============================================================

  pInterchange(z, z2, y)                'Annual energy exchanged [GWh] between internal zones'
  pInterconUtilization(z, z2, y)        'Interconnection utilization [%] between internal zones'
  pLossesTransmission(z, y)             'Transmission losses [MWh] per internal zone'
  pInterchangeCountry(c, c2, y)         'Annual energy exchanged [GWh] between countries'
  pLossesTransmissionCountry(c, y)      'Transmission losses [MWh] per country'
  isCongested(z, z2, q, d, t, y)        'Congestion indicator for line z-z2 at time q,d,t,y'
  pCongestionShare(z, z2, y)            'Share of time congested [%] for line z-z2'

  pHourlyInterchangeExternal(z, y, q, d, *, t)            'Hourly external trade [MW] per zone'
  pYearlyInterchangeExternal(z, *, y)                     'Annual external trade [GWh] per zone'
  pYearlyInterchangeExternalCountry(c, *, y)              'Annual external trade [GWh] per country'
  pHourlyInterchangeExternalCountry(c, y, q, d, *, t)     'Hourly external trade [MW] per country'

  pInterchangeExternalExports(z, zext, y)                 'Annual exports [GWh] from zone z to external zone zext'
  pInterchangeExternalImports(zext, z, y)                 'Annual imports [GWh] from external zone zext to zone z'
  pInterconUtilizationExternalExports(z, zext, y)         'External export line utilization [%] zone z to zext'
  pInterconUtilizationExternalImports(zext, z, y)         'External import line utilization [%] zext to zone z'

* ============================================================
* 7. EMISSIONS
* ============================================================
                                         
  pEmissionsZone(z,y)                    'CO2 emissions [Mt] by zone and year'
  pEmissionsIntensityZone(z,y)           'CO2 intensity [tCO2/GWh] by zone and year'
  pEmissionsCountrySummary(c,*,y)        'CO2 emissions [Mt] by country, type, and year'
  pEmissionsIntensityCountry(c,y)        'CO2 intensity [tCO2/GWh] by country and year'
  
  pEmissionMarginalCostsCountry(c,y)     'Marginal cost of country emission constraint [USD/tCO2]'
  pEmissionMarginalCosts(y)              'Marginal cost of system emission constraint [USD/tCO2]'

* ============================================================
* 8. PRICES
* ============================================================
   
  pYearlyPrice(z,y)                        'Demand-weighted average electricity price [USD/MWh] by zone and year'
  pYearlyPriceExport(z,y)                  'Flow-weighted average export price [USD/MWh] by zone and year'
  pYearlyPriceImport(z,y)                  'Flow-weighted average import price [USD/MWh] by zone and year'
  pYearlyPriceHub(z,y)                     'Flow-weighted hub price [USD/MWh] by zone and year'
  pYearlyPriceCountry(c,y)                 'Demand-weighted average electricity price [USD/MWh] by country and year'
  pYearlyPriceExportCountry(c,y)           'Flow-weighted average export price [USD/MWh] by country and year'
  pYearlyPriceImportCountry(c,y)           'Flow-weighted average import price [USD/MWh] by country and year'

* ============================================================
* 9. SPECIAL TECHNOLOGIES
* ============================================================

  pCSPBalance(y,g,q,d,*,t)                  'CSP hourly output by type [MW]'
  pCSPComponents(g,*,y)                     'CSP installed components and metrics'
  pPVwSTOBalance(y,q,d,g,*,t)               'PV+Storage hourly output by type [MW]'
  pPVwSTOComponents(g,*,y)                  'PV+Storage installed components and metrics'
  pStorageBalance(y,g,q,d,*,t)              'Generic storage hourly output by type [MW]'
  pStorageComponents(g,*,y)                 'Generic storage installed components and metrics'
  pSolarValueZone(z,y)                      'Average market value of solar [USD/MWh]'
  pSolarCost(z,y)                           'Levelized cost of solar [USD/MWh]'
  pSolarPower(z,q,d,t,y)                    'Solar hourly output [MWh]'

* ============================================================
* 10. METRICS
* ============================================================

  pPlantAnnualLCOE(z,g,y)          'Plant-level LCOE [USD/MWh] by year'
  pZonalAverageCost(z,y)           'Zone average total cost [USD/MWh] by year'
  pZonalAverageGenCost(z,y)        'Zone average generation cost [USD/MWh] by year'
  pCountryAverageCost(c,y)         'Country average total cost [USD/MWh] by year'
  pCountryAverageGenCost(c,y)      'Country average generation cost [USD/MWh] by year'
  pSystemAverageCost(y)            'System average cost [USD/MWh] by year'

* ============================================================
* 11. SOLVER PARAMETERS
* ============================================================
   pSolverParameters(*)                      'Solver parameters'                                                                                 
;

set sumhdr /
  "Annualized capex: $m",
  "Fixed O&M: $m",
  "Variable O&M: $m",
  "Fuel costs: $m",
  "Transmission additions: $m",
  "Spinning reserve costs: $m",
  "Unmet demand costs: $m",
  "Unmet country spinning reserve costs: $m",
  "Unmet country planning reserve costs: $m",
  "Unmet country CO2 backstop cost: $m",
  "Unmet system planning reserve costs: $m",
  "Unmet system spinning reserve costs: $m",
  "Unmet system CO2 backstop cost: $m",
  "Excess generation: $m",
  "VRE curtailment: $m",
  "Import costs with external zones: $m",
  "Export revenues with external zones: $m",
  "Import costs with internal zones: $m",
  "Export revenues with internal zones: $m",
  "Trade shared benefits: $m",
  "Carbon costs: $m",
  "NPV of system cost: $m"
/;

* Set of demand-related summary labels
set dshdr /
  "Demand: GWh",
  "Electricity demand for H2 production: GWh",
  "Total Demand (Including P to H2): GWh",
  "Total production: GWh",
  "Unmet demand: GWh",
  "Surplus generation: GWh",
  "Imports exchange: GWh",
  "Exports exchange: GWh",
  "Net interchange: GWh",
  "Net interchange Ratio: GWh"
/;

* H2-specific
set dsH2hdr /
  "Electricity demand for H2 production: GWh",
  "Total H2 Production: mmBTU",
  "External Demand of H2: mmBTU",
  "Unmet External Demand of H2: mmBTU",
  "H2 produced for power production: mmBTU"
/;

set zgmap(z,g); option zgmap<gzmap;
set zH2map(z,hh); option zH2map<h2zmap;

* ============================================================
* 1. CAPACITY
* ============================================================

* ---------------------------------------------------------
* Plant and technology-level capacity and utilization
* ---------------------------------------------------------
* Records installed capacity, retirements, and utilization
* factors for each plant and aggregated by technology:
*   - pCapacityPlant: installed capacity by plant
*   - pRetirementsPlant: retired capacity by plant

* ---------------------------------------------------------

pCapacityPlant(zgmap(z,g),y)                 = vCap.l(g,y);

pRetirementsPlant(zgmap(z,g),y)                  = vRetire.l(g,y);

* ---------------------------------------------------------
* Capacity, new builds, retirements, and utilization by fuel and technology
* ---------------------------------------------------------
* Tracks installed capacity, new capacity, and retirements
* disaggregated by fuel type and technology. 
* Also computes utilization factors:
*   - pCapacityTechFuel / pCapacityFuel: installed capacity
*   - pNewCapacityFuel / pNewCapacityTech: new builds
*   - pRetirementsFuel: retirements

* ---------------------------------------------------------

* Installed capacity by technology and fuel [MW]
pCapacityTechFuel(z, tech, f, y) =
  sum((gzmap(g, z), gtechmap(g, tech), gprimf(g, f)), vCap.l(g, y));

* Installed capacity by fuel [MW]
pCapacityFuel(z, f, y) =
  sum((gzmap(g, z), gprimf(g, f)), vCap.l(g, y));

* New capacity by fuel [MW]
pNewCapacityFuel(z, f, y) =
  sum((gzmap(g, z), gprimf(g, f)), vBuild.l(g, y));

* New capacity by technology [MW]
pNewCapacityTech(z, tech, y) =
  sum((gzmap(g, z), gtechmap(g, tech), gprimf(g, f)), vBuild.l(g, y));

* Retirements by fuel [MW]
pRetirementsFuel(z, f, y) =
  sum((gzmap(g, z), gprimf(g, f)), vRetire.l(g, y));

* ---------------------------------------------------------
* Transmission capacity expansion and annual limits
* ---------------------------------------------------------
* pAdditionalCapacity:
*   - New transmission capacity added between zones
*   - Based on build decision, per-line capacity, and allowed factor
*
* pAnnualTransmissionCapacity:
*   - Total available transfer capacity in a year
*   - Equals existing maximum transfer limit plus any new additions
* ---------------------------------------------------------

pAdditionalCapacity(sTopology(z,z2),y) = vNewTransferCapacity.l(z,z2,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine")*pAllowHighTransfer;                                                                                                        
pAnnualTransmissionCapacity(sTopology(z,z2),y) = pAdditionalCapacity(z,z2,y) + smax(q, pTransferLimit(z,z2,q,y)) ; 

* ---------------------------------------------------------
* Country-level capacity, new builds, retirements, and utilization
* ---------------------------------------------------------
* Aggregates zonal results to the country level:
*   - pCapacityPlantCountry: installed capacity by plant
*   - pCapacityTechFuelCountry / pCapacityFuelCountry: installed capacity by tech-fuel or fuel
*   - pNewCapacityFuelCountry / pNewCapacityTechCountry: new builds by fuel or tech
*   - pRetirementsFuelCountry / pRetirementsCountry: retirements by fuel and total
*   - pUtilizationFuelCountry / pUtilizationTechFuelCountry:
*       utilization factors (energy ÷ capacity × hours) by fuel or tech-fuel
* ---------------------------------------------------------

pCapacityTechFuelCountry(c, tech, f, y) =
  sum(zcmap(z, c), pCapacityTechFuel(z, tech, f, y));

pCapacityFuelCountry(c, f, y) =
  sum(zcmap(z, c), pCapacityFuel(z, f, y));

pNewCapacityFuelCountry(c, f, y) =
  sum(zcmap(z, c), pNewCapacityFuel(z, f, y));

pNewCapacityTechCountry(c, tech, y) =
  sum(zcmap(z, c), pNewCapacityTech(z, tech, y));

pRetirementsFuelCountry(c, f, y) =
  sum(zcmap(z, c), pRetirementsFuel(z, f, y));

pRetirementsCountry(c, y) =
  sum((zcmap(z, c), f), pRetirementsFuel(z, f, y));

* ---------------------------------------------------------
* Capacity summary at zone and country level [MW]
* ---------------------------------------------------------
* Provides aggregated indicators of system capacity:
*   - Available capacity
*   - Peak demand (with and without power-to-hydrogen load)
*   - New capacity builds
*   - Retired capacity
*   - Committed total transmission capacity
* Reported at both zonal and country aggregation levels.
* ---------------------------------------------------------

pCapacitySummary(z,"Available capacity: MW",y) = sum(gzmap(g,z), vCap.l(g,y));
pCapacitySummary(z,"Peak demand: MW"       ,y) = smax((q,d,t), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y));
pCapacitySummary(z,"Peak demand including P to H2: MW"       ,y) = smax((h2zmap(hh,z),q,d,t), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y) + vH2PwrIn.l(hh,q,d,t,y))$pIncludeH2+1e-6;
pCapacitySummary(z,"New capacity: MW"      ,y) = sum(gzmap(g,z), vBuild.l(g,y));
pCapacitySummary(z,"Retired capacity: MW"  ,y) = sum(gzmap(g,z), vRetire.l(g,y));
pCapacitySummary(z,"Committed Total TX capacity: MW"  ,y) = sum((z2,q), pTransferLimit(z,z2,q,y)/card(q));

pCapacitySummaryCountry(c,"Available capacity: MW",y) = sum(zcmap(z,c), pCapacitySummary(z,"Available capacity: MW",y));
pCapacitySummaryCountry(c,"Peak demand: MW"       ,y) = smax((q,d,t), sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y)));
pCapacitySummaryCountry(c,"New capacity: MW"      ,y) = sum(zcmap(z,c), pCapacitySummary(z,"New capacity: MW",y));
pCapacitySummaryCountry(c,"Retired capacity: MW"  ,y) = sum(zcmap(z,c), pCapacitySummary(z,"Retired capacity: MW",y));
pCapacitySummaryCountry(c,"Committed Total TX capacity: MW",y) = sum(zcmap(z,c), pCapacitySummary(z,"Committed Total TX capacity: MW",y));

* H2 model additions
pCapacityPlantH2(zh2map(z,hh),y) = vCapH2.l(hh,y)$pIncludeH2+1e-6;

* ============================================================
* 2. COSTS
* ============================================================

* Reporting costs by plant
* Plant-level cost reporting (prettier formatting)
pCostsPlant(z, g, "Annualized capex: $m", y) =
  vAnnGenCapex.l(g, y) / 1e6;

pCostsPlant(z, g, "Fixed O&M: $m", y) =
  (
    vCap.l(g, y)      * pGenData(g, "FOMperMW")
    + vCapStor.l(g, y)  * pStorData(g, "FixedOMMWh")
    + vCapStor.l(g, y)  * pCSPData(g, "Storage", "FixedOMMWh")
    + vCapTherm.l(g, y) * pCSPData(g, "Thermal field", "FixedOMMWh")
  ) / 1e6;

pCostsPlant(z, g, "Variable Cost: $m", y) =
  sum((gfmap(g, f), q, d, t),
    pVarCost(g, f, y) * vPwrOut.l(g, f, q, d, t, y) * pHours(q, d, t)
  ) / 1e6;

pCostsPlant(z, g, "Spinning Reserve Cost: $m", y) =
  sum((q, d, t),
    vSpinningReserve.l(g, q, d, t, y) * pGenData(g, "ReserveCost") * pHours(q, d, t)
  ) / 1e6;

* ---------------------------------------------------------
* Zone-level CAPEX investment flows [$]
* ---------------------------------------------------------
* Captures the annual capital expenditures for new builds
* in zone z and year y, before annualization.
*
* Components:
*   - Generation builds (MW × $/MW × cost trajectory)
*   - Storage builds (MWh × $/MWh × cost trajectory)
*   - CSP thermal field and storage builds (MWh × $/MWh × trajectory)
*   - Hydrogen builds (MW × $/MW × cost trajectory), optional
*   - Small epsilon added to avoid divide-by-zero issues
* ---------------------------------------------------------

pCapexInvestment(z, y) =
  1e6 * sum(gzmap(g, z),
    vBuild.l(g, y) * pGenData(g, "Capex") * pCapexTrajectories(g, y)
  )
  + 1e3 * sum(gzmap(st, z),
    vBuildStor.l(st, y) * pStorData(st, "CapexMWh") * pCapexTrajectories(st, y)
  )
  + 1e3 * sum(gzmap(cs, z),
    vBuildTherm.l(cs, y) * pCSPData(cs, "Thermal Field", "CapexMWh") * 1e3 * pCapexTrajectories(cs, y)
    + vBuildStor.l(cs, y) * pCSPData(cs, "Storage", "CapexMWh") * pCapexTrajectories(cs, y)
  )
  + 1e6 * sum(h2zmap(hh, z),
    vBuildH2.l(hh, y) * pH2Data(hh, "Capex") * pCapexTrajectoriesH2(hh, y)
  ) $ pIncludeH2
  + 1e-6;


* ---------------------------------------------------------
* Price calculation [$/MWh]
* ---------------------------------------------------------
* Market clearing price derived from the dual variable of the
* demand-supply balance constraint, scaled by operating hours,
* discount rate (pRR), and year weight.
* --------------------------------------------------------

pPrice(z,q,d,t,y)$(pHours(q,d,t)) = -eDemSupply.m(z,q,d,t,y)/pHours(q,d,t)/pRR(y)/pWeightYear(y);

* ---------------------------------------------------------
* Import and export costs with internal zones [$]
* ---------------------------------------------------------
* Import costs:
*   - Value of inflows from neighboring zones
*   - Calculated as local price × imported flow × hours
*
* Export revenues:
*   - Value of outflows to neighboring zones
*   - Negative sign ensures revenues reduce total cost
* ---------------------------------------------------------

pImportCostsInternal(z,y) = sum((sTopology(Zd,z),q,d,t), pPrice(z,q,d,t,y)*vFlow.l(Zd,z,q,d,t,y)*pHours(q,d,t));
pExportRevenuesInternal(z,y) = - sum((sTopology(z,Zd),q,d,t), pPrice(z,q,d,t,y)*vFlow.l(z,Zd,q,d,t,y)*pHours(q,d,t));

* ---------------------------------------------------------
* Congestion rents between zones [$]
* ---------------------------------------------------------
* Difference in marginal prices × traded flow × hours
* Negative sign ensures consistency with rent allocation
* ---------------------------------------------------------

pCongestionRevenues(z,Zd,y) = - sum((q,d,t), (pPrice(zD,q,d,t,y) - pPrice(z,q,d,t,y))*vFlow.l(z,Zd,q,d,t,y)*pHours(q,d,t));

* ---------------------------------------------------------
* Trade shared benefits [$]
* ---------------------------------------------------------
* Congestion rents are split equally (50/50) between
* importing and exporting zones as the allocation rule
* ---------------------------------------------------------

pTradeSharedBenefits(z,y) = 0.5*sum(sTopology(Zd,z), pCongestionRevenues(Zd,z,y)) + 0.5*sum(sTopology(z,Zd), pCongestionRevenues(z,Zd,y));

* ---------------------------------------------------------
* Net trade costs with internal zones [$]
* ---------------------------------------------------------
* Equivalent formulation where:
*   Trade cost = Import cost + Export revenues + Shared rents
* This matches pricing trades at the average marginal price
* between connected zones
* ---------------------------------------------------------

* pTradeCostsInternal(z,y) = pExportRevenuesInternal(z,y) + pImportCostsInternal(z,y) + pTradeSharedBenefits(z,y);

* ---------------------------------------------------------
* Demand allocation [GWh → TWh]
* ---------------------------------------------------------
* Zone-level demand:
*   - Sum of hourly demand × operating hours
*   - Adjusted by efficiency factor
*   - Converted from GWh to TWh
*
* Country-level demand:
*   - Aggregation of zonal demand by mapping zones to countries
* ---------------------------------------------------------
Parameter
  pDemandZone(z,y)                         'Total demand in GWh per zone'
  pDemandCountry(c,y)                      'Total demand in GWh per country'
  pDemand(y)
;

pDemandZone(z,y) = sum((q,d,t), 
    pDemandData(z,q,d,y,t) * pHours(q,d,t) * pEnergyEfficiencyFactor(z,y)
) / 1e3;
pDemandCountry(c,y) = sum(z$(zcmap(z,c)), pDemandZone(z,y));
pDemand(y) = sum(z, pDemandZone(z,y));

* ---------------------------------------------------------
* Zone-level cost components [$m]
* ---------------------------------------------------------
* Each entry in pYearlyCostsZone stores one annual cost component
* at the zonal level in million $.
*
* Categories included:
*   - Annualized CAPEX
*   - Fixed O&M
*   - Variable O&M
*   - Fuel costs
*   - Transmission additions
*   - Spinning reserve costs
*   - Unmet demand, reserve, and CO2 backstop costs
*       (allocated proportionally to zonal demand at country/system level)
*   - Excess generation
*   - VRE curtailment
*   - Import/export costs (external and internal zones)
*   - Trade shared benefits
*   - Carbon costs (system-wide total)
* ---------------------------------------------------------

* Investment-related costs
pYearlyCostsZone(z, "Annualized capex: $m", y) =
  vAnnCapex.l(z, y) / 1e6;

pYearlyCostsZone(z, "Transmission additions: $m", y) =
  vAnnualizedTransmissionCapex.l(z, y) / 1e6;

* Operation-related costs
pYearlyCostsZone(z, "Fixed O&M: $m", y) =
  vYearlyFOMCost.l(z, y) / 1e6;

pYearlyCostsZone(z, "Variable O&M: $m", y) =
  vYearlyVOMCost.l(z, y) / 1e6;

pYearlyCostsZone(z, "Fuel costs: $m", y) =
  vYearlyFuelCost.l(z, y) / 1e6;

* System balancing costs
pYearlyCostsZone(z, "Unmet demand costs: $m", y) =
  vYearlyUSECost.l(z, y) / 1e6;

pYearlyCostsZone(z, "Excess generation: $m", y) =
  vYearlySurplus.l(z, y) / 1e6;

pYearlyCostsZone(z, "VRE curtailment: $m", y) =
  vYearlyCurtailmentCost.l(z, y) / 1e6;

* Reserve-related costs
pYearlyCostsZone(z, "Spinning reserve costs: $m", y) =
  vYearlySpinningReserveCost.l(z, y) / 1e6;

pYearlyCostsZone(z, "Unmet country spinning reserve costs: $m", y) =
  sum(c$(zcmap(z, c) and pDemandCountry(c, y) > 0),
    vYearlyUnmetSpinningReserveCostCountry.l(c, y) * (pDemandZone(z, y) / pDemandCountry(c, y))
  ) / 1e6;

pYearlyCostsZone(z, "Unmet country planning reserve costs: $m", y) =
  sum(c$(zcmap(z, c) and pDemandCountry(c, y) > 0),
    vYearlyUnmetPlanningReserveCostCountry.l(c, y) * (pDemandZone(z, y) / pDemandCountry(c, y))
  ) / 1e6;

pYearlyCostsZone(z, "Unmet system planning reserve costs: $m", y) =
  vYearlyUnmetPlanningReserveCostSystem.l(y) * (pDemandZone(z, y) / pDemand(y)) / 1e6;

pYearlyCostsZone(z, "Unmet system spinning reserve costs: $m", y) =
  vYearlyUnmetSpinningReserveCostSystem.l(y) * (pDemandZone(z, y) / pDemand(y)) / 1e6;

* Carbon-related costs
pYearlyCostsZone(z, "Carbon costs: $m", y) =
  vYearlyCarbonCost.l(z, y) / 1e6;

pYearlyCostsZone(z, "Unmet country CO2 backstop cost: $m", y) =
  sum(c$(zcmap(z, c) and pDemandCountry(c, y) > 0),
    vYearlyCO2BackstopCostCountry.l(c, y) * (pDemandZone(z, y) / pDemandCountry(c, y))
  ) / 1e6;

pYearlyCostsZone(z, "Unmet system CO2 backstop cost: $m", y) =
  vYearlyCO2BackstopCostSystem.l(y) * (pDemandZone(z, y) / pDemand(y)) / 1e6;

* Trade-related costs
pYearlyCostsZone(z, "Import costs with internal zones: $m", y) =
  pImportCostsInternal(z, y) / 1e6;

pYearlyCostsZone(z, "Export revenues with internal zones: $m", y) =
  pExportRevenuesInternal(z, y) / 1e6;

pYearlyCostsZone(z, "Trade shared benefits: $m", y) =
  pTradeSharedBenefits(z, y) / 1e6;

pYearlyCostsZone(z, "Import costs with external zones: $m", y) =
  vYearlyImportExternalCost.l(z, y) / 1e6;

pYearlyCostsZone(z, "Export revenues with external zones: $m", y) =
  vYearlyExportExternalCost.l(z, y) / 1e6;

* Cost 
pCostsZone(z, sumhdr) =
    sum(y, pYearlyCostsZone(z,sumhdr,y) * pRR(y) * pWeightYear(y));

* ---------------------------------------------------------

* Cost by country and year 
pYearlyCostsCountry(c,sumhdr,y) = sum(z$(zcmap(z,c)), pYearlyCostsZone(z,sumhdr,y));                                        

* Cost average by country over the time horizon
pCostAverageCountry(c,sumhdr) = sum(y, pWeightYear(y) * pYearlyCostsCountry(c,sumhdr,y))/TimeHorizon;

* ---------------------------------------------------------
* System-level cost summary
* ---------------------------------------------------------
* Aggregates all zone-level costs into a single system-wide
* metric across the entire time horizon.
*
* - pCostsSystem(sumhdr):
*     Weighted sum of all zone-level costs
*     Units: million $ (after scaling)
*
* - "NPV of system cost: $m":
*     The model's computed Net Present Value of system costs
*     Directly taken from optimization variable vNPVCost
* ---------------------------------------------------------

pYearlyCostsSystem(sumhdr, y) = sum(z, pYearlyCostsZone(z, sumhdr, y))

* pCostsSystem(sumhdr) = sum((y,z), pYearlyCostsZone(z,sumhdr,y) * pRR(y) * pWeightYear(y));
pCostsSystem(sumhdr) = sum(z, pCostsZone(z,sumhdr));

pCostsSystem("NPV of system cost: $m") = vNPVCost.l/1e6;


* ---------------------------------------------------------
* Fuel costs and consumption [$m, PJ] by zone and country
* ---------------------------------------------------------
* Calculates annual fuel expenditures and consumption
* based on generation output, heat rates, and fuel prices.
* Results are stored at both zonal and country aggregation levels.
* ---------------------------------------------------------

pFuelCosts(z,f,y) = sum((gzmap(g,z),gfmap(g,f),zcmap(z,c),q,d,t), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t)*pFuelPrice(c,f,y)*pHeatRate(g,f))/1e6;
pFuelConsumption(z,f,y) = vFuel.l(z,f,y)/1e6;

pFuelCostsCountry(c,f,y) = sum(zcmap(z,c), pFuelCosts(z,f,y));
pFuelConsumptionCountry(c,f,y) = sum(zcmap(z,c), pFuelConsumption(z,f,y));

* ============================================================
* 3. ENERGY BALANCE
* ============================================================

* ---------------------------------------------------------
* Energy generation by plant, technology-fuel, and country
* ---------------------------------------------------------
* Calculates annual electricity generation (TWh) from model
* outputs at different aggregation levels:
*   - pEnergyPlant: plant-level generation
*   - pEnergyTechFuel / pEnergyTechFuelCountry: by technology and fuel, zonal and country
*   - pEnergyFuel / pEnergyFuelCountry: by fuel, zonal and country
* ---------------------------------------------------------


pEnergyPlant(zgmap(z,g),y) = sum((gfmap(g,f),q,d,t), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t))/1e3;
pEnergyFuel(z,f,y) = sum((gzmap(g,z),gfmap(g,f),q,d,t), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t))/1e3;
pEnergyFuelCountry(c,f,y) = sum(zcmap(z,c), pEnergyFuel(z,f,y));

* TODO: Optional
pEnergyTechFuel(z,tech,f,y) = sum((gzmap(g,z),gtechmap(g,tech),gfmap(g,f),q,d,t), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t))/1e3;
pEnergyTechFuelCountry(c,tech,f,y) = sum(zcmap(z,c), pEnergyTechFuel(z,tech,f,y));


* ---------------------------------------------------------
* Utilization factors by plant, technology, fuel, and country
* ---------------------------------------------------------
* Computes average utilization (capacity factor) as the ratio
* of generated energy to available capacity and hours:
*   - pUtilizationPlant / pUtilizationTech: plant-level 
*     and tech-level capacity factors
*   - pUtilizationFuel / pUtilizationTechFuel: by fuel and tech-fuel at zone level
*   - pUtilizationFuelCountry / pUtilizationTechFuelCountry: 
*     by fuel and tech-fuel at country level
* ---------------------------------------------------------


* Plant-level utilization (capacity factor, annual average)
pUtilizationPlant(zgmap(z,g),y)$vCap.l(g,y) =
  sum((gfmap(g,f), q, d, t), vPwrOut.l(g,f,q,d,t,y) * pHours(q,d,t))
  / vCap.l(g,y) / 8760;

* Technology-level utilization (capacity factor, annual average)
pUtilizationTech(z, tech, y)$sum((zgmap(z,g), gfmap(g,f), gtechmap(g,tech))$vCap.l(g,y), vCap.l(g,y)) =
  sum((zgmap(z,g), gfmap(g,f), gtechmap(g,tech), q, d, t)$vCap.l(g,y),
    vPwrOut.l(g,f,q,d,t,y) * pHours(q,d,t))
  / sum((zgmap(z,g), gfmap(g,f), gtechmap(g,tech))$vCap.l(g,y), vCap.l(g,y))
  / 8760;

* Fuel-level utilization (capacity factor, annual average)
pUtilizationFuel(z, f, y)$(pCapacityFuel(z, f, y)) =
  pEnergyFuel(z, f, y)
  / (pCapacityFuel(z, f, y) * sum((q, d, t), pHours(q, d, t)))
  * 1000;

* Technology-fuel-level utilization (capacity factor, annual average)
pUtilizationTechFuel(z, tech, f, y)$(pCapacityTechFuel(z, tech, f, y)) =
  pEnergyTechFuel(z, tech, f, y)
  / (pCapacityTechFuel(z, tech, f, y) * sum((q, d, t), pHours(q, d, t)))
  * 1000;

* Country-level fuel utilization (capacity factor, annual average)
pUtilizationFuelCountry(c, f, y)$(pCapacityFuelCountry(c, f, y)) =
  pEnergyFuelCountry(c, f, y)
  / (pCapacityFuelCountry(c, f, y) * sum((q, d, t), pHours(q, d, t)))
  * 1000;

* Country-level technology-fuel utilization (capacity factor, annual average)
pUtilizationTechFuelCountry(c, tech, f, y)$(pCapacityTechFuelCountry(c, tech, f, y)) =
  pEnergyTechFuelCountry(c, tech, f, y)
  / (pCapacityTechFuelCountry(c, tech, f, y) * sum((q, d, t), pHours(q, d, t)))
  * 1000;


* ---------------------------------------------------------
* Energy balance [GWh] by zone and country
* ---------------------------------------------------------
* Accounts for demand (electricity and H2), production,
* unmet demand, surplus generation, imports/exports,
* and net interchange, with aggregation to country level.
* ---------------------------------------------------------

pEnergyBalance(z, "Demand: GWh", y) =
  sum((q, d, t), pDemandData(z, q, d, y, t) * pHours(q, d, t) * pEnergyEfficiencyFactor(z, y)) / 1e3;

pEnergyBalance(z, "Demand H2: GWh", y) =
  (sum((h2zmap(hh, z), q, d, t), vH2PwrIn.l(hh, q, d, t, y) * pHours(q, d, t)) / 1e3) $ (pIncludeH2 + 1e-6);

pEnergyBalance(z, "Total Demand: GWh", y) =
  pEnergyBalance(z, "Demand: GWh", y) + pEnergyBalance(z, "Demand H2: GWh", y);

pEnergyBalance(z, "Total production: GWh", y) =
  sum(gzmap(g, z), pEnergyPlant(z, g, y));

pEnergyBalance(z, "Unmet demand: GWh", y) =
  sum((q, d, t), vUSE.l(z, q, d, t, y) * pHours(q, d, t)) / 1e3;

pEnergyBalance(z, "Surplus generation: GWh", y) =
  sum((q, d, t), vSurplus.l(z, q, d, t, y) * pHours(q, d, t)) / 1e3;

pEnergyBalance(z, "Imports exchange: GWh", y) =
  (sum((sTopology(z, z2), q, d, t), vFlow.l(z2, z, q, d, t, y) * pHours(q, d, t))
   + sum((zext, q, d, t), vYearlyImportExternal.l(z, zext, q, d, t, y) * pHours(q, d, t))) / 1e3;

pEnergyBalance(z, "Exports exchange: GWh", y) =
  (sum((sTopology(z, z2), q, d, t), vFlow.l(z, z2, q, d, t, y) * pHours(q, d, t))
   + sum((zext, q, d, t), vYearlyExportExternal.l(z, zext, q, d, t, y) * pHours(q, d, t))) / 1e3;

pEnergyBalance(z, "Net interchange: GWh", y) =
  pEnergyBalance(z, "Imports exchange: GWh", y) - pEnergyBalance(z, "Exports exchange: GWh", y);

pEnergyBalance(z, "Net interchange Ratio: GWh", y) $ pEnergyBalance(z, "Demand: GWh", y) =
  pEnergyBalance(z, "Net interchange: GWh", y) / pEnergyBalance(z, "Demand: GWh", y);

pEnergyBalanceCountry(c,dshdr,y) = sum(z$(zcmap(z,c)), pEnergyBalance(z,dshdr,y));


*--- H2 Demand-Supply Balance
pEnergyBalanceH2(z,"Demand H2: GWh",y)   =(sum( (h2zmap(hh,z), q,d,t), vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t))/1000)$pIncludeH2+1e-6; 
pEnergyBalanceH2(z,"Production H2: mmBTU",           y)        =sum( (h2zmap(hh,z), q,d,t), vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t)*pH2Data(hh,"HeatRate"))$pIncludeH2+1e-6;
pEnergyBalanceH2(z,"External Demand H2: mmBTU",         y)        =sum(q,pExternalH2(z,q,y)       )$pIncludeH2+1e-6;
pEnergyBalanceH2(z,"Unmet External Demand H2: mmBTU",   y)        =sum(q,vUnmetExternalH2.l(z,q,y))$pIncludeH2+1e-6;
pEnergyBalanceH2(z,"H2 Power: mmBTU",       y)        =sum(q,vFuelH2Quarter.l(z,q,y)  )$pIncludeH2+1e-6;

pEnergyBalanceCountryH2(c,dsH2hdr,y) = sum(z$(zcmap(z,c)), pEnergyBalanceH2(z,dsH2hdr,y));

* ============================================================
* 4. ENERGY DISPATCH
* ============================================================

* ---------------------------------------------------------
* Hourly dispatch reporting: plant, fuel, and zone level
* ---------------------------------------------------------
* - pDispatchPlant: plant-level hourly generation and reserve [MW]
* - pDispatchFuel: fuel-level hourly generation [MW]
* - pDispatch: zone-level hourly imports, exports, unmet demand, storage charge, and demand [MW]
* ---------------------------------------------------------

pDispatchPlant(z, y, q, d, g, t, "Generation")$zgmap(z, g) = sum(gfmap(g, f), vPwrOut.l(g, f, q, d, t, y));
pDispatchPlant(z, y, q, d, g, t, "Reserve")$zgmap(z, g) = vSpinningReserve.l(g, q, d, t, y);

pDispatchFuel(z, y, q, d, f, t) = sum((gzmap(g, z), gfmap(g, f)), vPwrOut.l(g, f, q, d, t, y));

pDispatch(z, y, q, d, "Imports", t) = sum(sTopology(z, z2), vFlow.l(z2, z, q, d, t, y)) + sum(zext, vYearlyImportExternal.l(z, zext, q, d, t, y));
pDispatch(z, y, q, d, "Exports", t) = -sum(sTopology(z, z2), vFlow.l(z, z2, q, d, t, y)) - sum(zext, vYearlyExportExternal.l(z, zext, q, d, t, y));
pDispatch(z, y, q, d, "Unmet demand", t) = vUSE.l(z, q, d, t, y);
pDispatch(z, y, q, d, "Storage Charge", t) = -sum(zgmap(z, st), vStorInj.l(st, q, d, t, y));
pDispatch(z, y, q, d, "Demand", t) = pDemandData(z, q, d, y, t) * pEnergyEfficiencyFactor(z, y);

* ============================================================
* 5. RESERVES
* ============================================================

* ---------------------------------------------------------
* Spinning reserve and reserve margin reporting
* ---------------------------------------------------------
* - pReserveSpinningPlantZone: annual spinning reserve provided by plant and zone [GWh]
* - pReserveSpinningFuelZone: annual spinning reserve provided by fuel and zone [GWh]
* - pReserveSpinningPlantCountry: annual spinning reserve provided by plant and country [GWh]
* - pReserveMargin / pReserveMarginCountry: peak demand, total firm capacity, and reserve margin by zone/country
* ---------------------------------------------------------

pReserveSpinningPlantZone(zgmap(z,g),y) = sum((q,d,t), vSpinningReserve.l(g,q,d,t,y)*pHours(q,d,t))/1e3 ;
pReserveSpinningFuelZone(z,f,y) = sum((gzmap(g,z),gfmap(g,f),q,d,t), vSpinningReserve.l(g,q,d,t,y)*pHours(q,d,t))/1e3 ;
pReserveSpinningPlantCountry(c,g,y)=  sum((zcmap(z,c),zgmap(z,g)), pReserveSpinningPlantZone(z,g,y));

pReserveMargin(z,"Peak demand: MW",y) = smax((q,d,t), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y));
pReserveMargin(z,"TotalFirmCapacity",y) = sum(zgmap(z,g), pCapacityCredit(g,y)* vCap.l(g,y));                  
pReserveMargin(z,"ReserveMargin",y)$(pReserveMargin(z,"TotalFirmCapacity",y)) = pReserveMargin(z,"TotalFirmCapacity",y)/pReserveMargin(z,"Peak demand: MW",y)  ;             

pReserveMarginCountry(c,"Peak demand: MW",y) =   pCapacitySummaryCountry(c,"Peak demand: MW"       ,y);
pReserveMarginCountry(c,"TotalFirmCapacity",y) = sum(zcmap(z,c), pReserveMargin(z,"TotalFirmCapacity",y));
pReserveMarginCountry(c,"ReserveMargin",y)$(pReserveMarginCountry(c,"TotalFirmCapacity",y)) = pReserveMarginCountry(c,"TotalFirmCapacity",y)/ pReserveMarginCountry(c,"Peak demand: MW",y)  ;   

* ============================================================
* 6. INTERCONNECTIONS
* ============================================================

* ---------------------------------------------------------
* Internal zones: interchange, utilization, losses, congestion
* ---------------------------------------------------------
* - pInterchange / pInterchangeCountry: net interchange flow [GWh]
* - pInterconUtilization: utilization of transfer capacity [%]
* - pLossesTransmission / pLossesTransmissionCountry: transmission losses [GWh]
* - isCongested / pCongestionShare: binary indicator and share of time congested
* ---------------------------------------------------------

* Annual interchange between internal zones [GWh]
pInterchange(sTopology(z, z2), y) =
  sum((q, d, t), vFlow.l(z, z2, q, d, t, y) * pHours(q, d, t)) / 1e3;

* Utilization of interconnection throughout modeling horizon [%]
pInterconUtilization(sTopology(z, z2), y)$pInterchange(z, z2, y) =
  1e3 * pInterchange(z, z2, y)
  / sum((q, d, t),
    (pTransferLimit(z, z2, q, y)
     + vNewTransferCapacity.l(z, z2, y)
       * max(pNewTransmission(z, z2, "CapacityPerLine"),
         pNewTransmission(z2, z, "CapacityPerLine"))
       * pAllowHighTransfer)
    * pHours(q, d, t)
    );

* Transmission losses between internal zones in MWh per zone
pLossesTransmission(z, y) =
  sum((sTopology(z, z2), q, d, t),
    vFlow.l(z2, z, q, d, t, y) * pLossFactor(z, z2, y) * pHours(q, d, t)
  );

* Country-level interchange and losses
alias (zcmap, zcmap2);

pInterchangeCountry(c, c2, y) =
  sum((zcmap(z, c), zcmap2(z2, c2), sMapConnectedZonesDiffCountries(z2, z)),
    pInterchange(z, z2, y)
  );

pLossesTransmissionCountry(c, y) =
  sum(zcmap(z, c), pLossesTransmission(z, y));

* Binary indicator for congestion per time step
isCongested(z, z2, q, d, t, y)$(
  sTopology(z, z2)
  and abs(
    vFlow.l(z, z2, q, d, t, y)
    - (
      pTransferLimit(z, z2, q, y)
      + vNewTransferCapacity.l(z, z2, y)
        * max(pNewTransmission(z, z2, "CapacityPerLine"),
          pNewTransmission(z2, z, "CapacityPerLine"))
        * pAllowHighTransfer
      )
    ) < 1e-5
) = 1;

* Percentage of time (weighted by pHours) when the line is congested
pCongestionShare(sTopology(z, z2), y) =
  sum((q, d, t), isCongested(z, z2, q, d, t, y) * pHours(q, d, t))
  / sum((q, d, t), pHours(q, d, t));


* ---------------------------------------------------------
* External zones: imports and exports
* ---------------------------------------------------------
* - pHourlyInterchangeExternal / pYearlyInterchangeExternal: hourly and annual imports/exports [GWh]
* - pInterchangeExternalExports / pInterconUtilizationExternalExports: exports and line utilization [%]
* - pInterchangeExternalImports / pInterconUtilizationExternalImports: imports and line utilization [%]
* - pHourlyInterchangeExternalCountry / pYearlyInterchangeExternalCountry: aggregated country-level interchange [GWh]
* ---------------------------------------------------------

set thrd / Imports, Exports /;

* Hourly interchange with external zones [MW]
pHourlyInterchangeExternal(z, y, q, d, "Imports", t) =
  sum(zext, vYearlyImportExternal.l(z, zext, q, d, t, y));

pHourlyInterchangeExternal(z, y, q, d, "Exports", t) =
  sum(zext, vYearlyExportExternal.l(z, zext, q, d, t, y));

* Annual interchange with external zones [GWh]
pYearlyInterchangeExternal(z, thrd, y) =
  sum((q, d, t), pHourlyInterchangeExternal(z, y, q, d, thrd, t) * pHours(q, d, t)) / 1e3;

* Annual exports and imports with external zones [GWh]
pInterchangeExternalExports(z, zext, y) =
  sum((q, d, t), vYearlyExportExternal.l(z, zext, q, d, t, y) * pHours(q, d, t)) / 1e3;

pInterchangeExternalImports(zext, z, y) =
  sum((q, d, t), vYearlyImportExternal.l(z, zext, q, d, t, y) * pHours(q, d, t)) / 1e3;

* Utilization of external interconnections [%]
pInterconUtilizationExternalExports(z, zext, y)$pInterchangeExternalExports(z, zext, y) =
  1e3 * pInterchangeExternalExports(z, zext, y)
  / sum((q, d, t), pExtTransferLimitOut(z, zext, q, y) * pHours(q, d, t));

pInterconUtilizationExternalImports(zext, z, y)$pInterchangeExternalImports(zext, z, y) =
  1e3 * pInterchangeExternalImports(zext, z, y)
  / sum((q, d, t), pExtTransferLimitIn(z, zext, q, y) * pHours(q, d, t));

* Country-level hourly and annual interchange with external zones
pHourlyInterchangeExternalCountry(c, y, q, d, thrd, t) =
  sum(zcmap(z, c), pHourlyInterchangeExternal(z, y, q, d, thrd, t));

pYearlyInterchangeExternalCountry(c, thrd, y) =
  sum(zcmap(z, c), pYearlyInterchangeExternal(z, thrd, y));

* ============================================================
* 7. EMISSIONS
* ============================================================

* ---------------------------------------------------------
* Emissions reporting: zone, country, and marginal costs
* ---------------------------------------------------------
* - pEmissionsZone: total CO2 emissions [Mt] by zone and year
* - pEmissionsIntensityZone: CO2 intensity [tCO2/GWh] by zone and year
* - pEmissionsCountrySummary: total and backstop CO2 emissions [Mt] by country and year
* - pEmissionsIntensityCountry: CO2 intensity [tCO2/GWh] by country and year
* - pEmissionMarginalCosts / pEmissionMarginalCostsCountry: marginal cost of emission constraints [USD/tCO2]
* ---------------------------------------------------------

pEmissionsZone(z,y)                                                     =  vZonalEmissions.l(z,y)/1e6 ;
pEmissionsIntensityZone(z,y)$pEnergyBalance(z,"Total production: GWh",y) = 1e-3*pEmissionsZone(z,y)/pEnergyBalance(z,"Total production: GWh",y);
pEmissionsCountrySummary(c,"Emissions: mm tCO2eq",y)                               = sum(zcmap(z,c), pEmissionsZone(z,y));
pEmissionsCountrySummary(c,"CO2backstopEmissions: mm tCO2eq",y)                    = vYearlyCO2backstop.l(c,y)/1e6;
pEmissionsIntensityCountry(c,y)$pEnergyBalanceCountry(c,"Total production: GWh",y) = 1e-3*sum(zcmap(z,c), vZonalEmissions.l(z,y))/pEnergyBalanceCountry(c,"Total production: GWh",y);

pEmissionMarginalCosts(y) $(pWeightYear(y))                  = -eTotalEmissionsConstraint.M(y)/pRR(y)/pWeightYear(y);
pEmissionMarginalCostsCountry(c,y) $(pWeightYear(y))         = -eEmissionsCountry.M(c,y)/pRR(y)/pWeightYear(y);  

* ============================================================
* 8. PRICES
* ============================================================
* ---------------------------------------------------------
* Average yearly electricity prices at zone and country level
* ---------------------------------------------------------
* Calculates demand-weighted consumer prices, export/import
* prices weighted by actual flows (internal & external),
* and hub prices. Aggregates results to country averages
* using zonal weights and interconnection flows.
* ---------------------------------------------------------

* Average yearly price paid by consumers, weighted by demand
pYearlyPrice(z,y)$pEnergyBalance(z,"Demand: GWh",y) = 1e-3*sum((q,d,t),pPrice(z,q,d,t,y)*pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))
                                                     /pEnergyBalance(z,"Demand: GWh",y) ;

Parameter
  pFlowMWSum(z,z2,y)    "Sum of hourly MW flows over the year (used for weights)"
  pFlowMWh(z,z2,y)      "Annual energy flow between zones [MWh]"
;

pFlowMWSum(z,z2,y) = sum(sFlow(z,z2,q,d,t,y),vFlow.l(z,z2,q,d,t,y));
pFlowMWh(z,z2,y) = sum(sFlow(z,z2,q,d,t,y),vFlow.l(z,z2,q,d,t,y)*pHours(q,d,t));

* Average price of exports and imports, weighted by actual flows on the lines
pYearlyPriceExport(z,y) $(sum(Zd, pFlowMWh(z,Zd,y))  > 0) = (sum((sTopology(z,Zd),q,d,t),  pPrice(z,q,d,t,y) *vFlow.l(z,Zd,q,d,t,y)*pHours(q,d,t))
                                                           + sum((zext,q,d,t), vYearlyExportExternal.l(z,zext,q,d,t,y) * pTradePrice(zext,q,d,y,t) * pHours(q,d,t))) 
                                                         /(sum(Zd, pFlowMWh(z,Zd,y))+ sum((zext,q,d,t), vYearlyExportExternal.l(z,zext,q,d,t,y)* pHours(q,d,t)));

pYearlyPriceImport(z,y) $(sum(Zd, pFlowMWh(Zd,z,y))  > 0) = (sum((sTopology(Zd,z),q,d,t),  pPrice(Zd,q,d,t,y)*vFlow.l(Zd,z,q,d,t,y)*pHours(q,d,t))
                                                            + sum((zext,q,d,t), vYearlyImportExternal.l(z,zext,q,d,t,y) * pTradePrice(zext,q,d,y,t) * pHours(q,d,t))) 
                                                         /(sum(Zd, pFlowMWh(Zd,z,y))+ sum((zext,q,d,t), vYearlyImportExternal.l(z,zext,q,d,t,y)  * pHours(q,d,t)));

* Average price at the hub, weighted by actual flows on the lines (TODO: check not with pHours)
pYearlyPriceHub(Zt,y)$(sum(Zd, pFlowMWSum(Zt,Zd,y))   > 0) = sum((sTopology(Zt,Zd),q,d,t), pPrice(Zt,q,d,t,y)*vFlow.l(Zt,Zd,q,d,t,y))
                                                         /sum(Zd, pFlowMWSum(Zt,Zd,y));

* Average yearly price paid by consumers, weighted by demand
pYearlyPriceCountry(c,y)$pEnergyBalanceCountry(c,"Demand: GWh",y) = sum(zcmap(z,c), pYearlyPrice(z,y)*pEnergyBalance(z,"Demand: GWh",y))/pEnergyBalanceCountry(c,"Demand: GWh",y);

* ---------------------------------------------------------
* Country-level interconnection energy flows [MWh]
* ---------------------------------------------------------
* Aggregated annual energy exchanged between a country and its 
* neighboring countries, used as weights to compute average 
* export and import prices:
*   - pCountryExportFlowMWh: total export energy from country c
*   - pCountryImportFlowMWh: total import energy into country c
* ---------------------------------------------------------

Parameter 
    pCountryExportFlowMWh(c,y) "Annual exported energy [MWh] from country c"
    pCountryImportFlowMWh(c,y) "Annual imported energy [MWh] into country c";

pCountryExportFlowMWh(c,y) = sum((zcmap(z,c), sMapConnectedZonesDiffCountries(z,Zd)), pFlowMWh(z,Zd,y));

pYearlyPriceExportCountry(c,y)$(pCountryExportFlowMWh(c,y) > 0) =
    sum((zcmap(z,c), sMapConnectedZonesDiffCountries(z,Zd),q,d,t),
        pPrice(z,q,d,t,y)*vFlow.l(z,Zd,q,d,t,y)*pHours(q,d,t))
    / pCountryExportFlowMWh(c,y);

pCountryImportFlowMWh(c,y) = sum((zcmap(z,c), sMapConnectedZonesDiffCountries(Zd,z)), pFlowMWh(Zd,z,y));

pYearlyPriceImportCountry(c,y)$(pCountryImportFlowMWh(c,y) > 0) =
    sum((zcmap(z,c), sMapConnectedZonesDiffCountries(Zd,z),q,d,t),
        pPrice(Zd,q,d,t,y)*vFlow.l(Zd,z,q,d,t,y)*pHours(q,d,t))
    / pCountryImportFlowMWh(c,y);



* ============================================================
* 9. SPECIAL TECHNOLOGIES
* ============================================================
* ---------------------------------------------------------
* CSP, PV+Storage, and Generic Storage balances and components
* ---------------------------------------------------------
* Tracks technology-specific balances and installed components:
*   - pCSPBalance: thermal, storage input/output, and power output by CSP
*   - pCSPComponents: thermal field, storage, power block, solar multiple, storage hours
*   - pPVwSTOBalance: PV output, storage input/output, storage losses and levels for PV+storage systems
*   - pPVwSTOComponents: PV capacity, storage MW/MWh, storage hours
*   - pStorageBalance: input, output, losses, net change, and storage level for generic storage
*   - pStorageComponents: installed MW, MWh, and storage hours
* ---------------------------------------------------------

pCSPBalance(y, g,q,d,"Thermal output",t) = vThermalOut.l(g,q,d,t,y);
pCSPBalance(y,cs,q,d,"Storage Input" ,t) = vStorInj.l(cs,q,d,t,y);
pCSPBalance(y,cs,q,d,"Storage Output",t) = vStorOut.l(cs,q,d,t,y);
pCSPBalance(y,cs,q,d,"Power Output"  ,t) = sum(gfmap(cs,f), vPwrOut.l(cs,f,q,d,t,y));

pCSPComponents(g, "Thermal Field: WM" ,y) = vCapTherm.l(g,y);
pCSPComponents(cs,"Storage: MWh"      ,y) = vCapStor.l(cs,y);
pCSPComponents(cs,"Power Block: MW"   ,y) = vCap.l(cs,y);
pCSPComponents(g, "Solar Multiple"    ,y) = vCapTherm.l(g,y)/max(vCap.l(g,y),1);
pCSPComponents(cs,"Storage Hours: hrs",y) = vCapStor.l(cs,y)/max(vCap.l(cs,y),1);

pPVwSTOBalance(y,q,d,so, "PV output"     ,t) = sum(gfmap(so,f), vPwrOut.l(so,f,q,d,t,y));
pPVwSTOBalance(y,q,d,stp,"Storage Input" ,t) = vStorInj.l(stp,q,d,t,y);
pPVwSTOBalance(y,q,d,stp,"Storage output",t) = sum(gfmap(stp,f), vPwrOut.l(stp,f,q,d,t,y));
pPVwSTOBalance(y,q,d,stp,"Storage Losses",t) = (1-pStorData(stp,"Efficiency"))*vStorInj.l(stp,q,d,t,y);
pPVwSTOBalance(y,q,d,stp,"Storage level" ,t) = vStorage.l(stp,q,d,t,y);

pPVwSTOComponents(so, "PV Plants"           ,y) = vCap.l(so,y);
pPVwSTOComponents(stp,"Storage Capacity MW" ,y) = vCap.l(stp,y);
pPVwSTOComponents(stp,"Storage Capacity MWh",y) = vCapStor.l(stp,y);
pPVwSTOComponents(stp,"Storage Hours"       ,y) = vCapStor.l(stp,y)/max(vCap.l(stp,y),1);

pStorageBalance(y,stg,q,d,"Storage Input"     ,t) = vStorInj.l(stg,q,d,t,y);
pStorageBalance(y,stg,q,d,"Storage Output"    ,t) = sum(gfmap(stg,f), vPwrOut.l(stg,f,q,d,t,y));
pStorageBalance(y,stg,q,d,"Storage Losses"    ,t) = (1-pStorData(stg,"Efficiency"))*vStorInj.l(stg,q,d,t,y);
pStorageBalance(y,stg,q,d,"Net Storage Change",t) = vStorNet.l(stg,q,d,t,y);
pStorageBalance(y,stg,q,d,"Storage Level"     ,t) = vStorage.l(stg,q,d,t,y);

pStorageComponents(stg,"Storage Capacity: MW",y) = vCap.l(stg,y);
pStorageComponents(stg,"Storage Capacity: MWh"  ,y) = vCapStor.l(stg,y);
pStorageComponents(stg,"Storage Hours"         ,y) = vCapStor.l(stg,y)/max(vCap.l(stg,y),1);

* ---------------------------------------------------------
* Zonal solar energy, value, and cost indicators
* ---------------------------------------------------------
* Computes solar power output, annual energy, market value,
* and levelized cost of solar at the zonal level:
*   - pSolarPower: hourly solar generation
*   - pSolarEnergyZone: total annual solar energy (MWh)
*   - pSolarValueZone: average market value of solar ($/MWh)
*   - pSolarCost: levelized cost of solar based on annualized CAPEX ($/MWh)
* ---------------------------------------------------------

set PVtech(tech) / PV, PVwSTO /;
Parameter pSolarEnergyZone(z,y);

pSolarPower(z,q,d,t,y) = sum((gzmap(g,z),gtechmap(g,PVtech),gfmap(g,f)), vPwrOut.l(g,f,q,d,t,y));
pSolarEnergyZone(z,y) = sum((q,d,t), pSolarPower(z,q,d,t,y)*pHours(q,d,t));
pSolarValueZone(z,y)$(pSolarEnergyZone(z,y) > 0) = sum((q,d,t), pPrice(z,q,d,t,y)*pSolarPower(z,q,d,t,y))/pSolarEnergyZone(z,y);
pSolarCost(z,y)$(pSolarEnergyZone(z,y) > 0) = (sum((gzmap(ng,z),gtechmap(ng,PVtech)), pCRF(ng)*vCap.l(ng,y)*pGenData(ng,"Capex")*pCapexTrajectories(ng,y))*1e6)/pSolarEnergyZone(z,y);


* ============================================================
* 10. METRICS
* ============================================================

* ---------------------------------------------------------
* LCOE by plant [$/MWh]
* ---------------------------------------------------------

Parameter
  pPlantEnergyMWh(z,g,y) "Annual energy production by plant [MWh]"
;

* Plant-level energy denominator
pPlantEnergyMWh(z,g,y) = 1e-6;
pPlantEnergyMWh(z,g,y)$pEnergyPlant(z,g,y) = pEnergyPlant(z,g,y)*1e3;

* LCOE for new capacity plants (without direct spinning reserve cost)
pPlantAnnualLCOE(z,g,y)$pPlantEnergyMWh(z,g,y) =
    ( pCostsPlant(z, g, "Annualized capex: $m", y)
    + pCostsPlant(z, g, "Variable Cost: $m", y) 
    + pCostsPlant(z, g, "Variable Cost: $m", y) ) / pPlantEnergyMWh(z,g,y);

* ---------------------------------------------------------
* Cost components by zone [$/year]
* ---------------------------------------------------------
* Each cost component represents one part of the total
* annual system cost in a given zone z and year y.
*
* Trade cost:
*   - External imports minus exports
*   - Valued at external trade price × operating hours
*
* Transmission cost:
*   - Interzonal flows
*   - Valued at zonal market price × operating hours
*
* Investment CAPEX:
*   - Annualized cost of generation, storage, CSP storage,
*     and CSP thermal field capacity
*   - Uses CRFs (capital recovery factors) to annualize
*   - Scaled to $ from MW or MWh basis
*
* Operating costs (O&M):
*   - Fixed O&M (FOM)
*   - Variable O&M (VOM)
*   - Fuel expenditures
*   - Spinning reserve provision
* ---------------------------------------------------------

Parameter
    pZoneTradeCost
    pZoneGenCost
    pZoneTotalCost
;

* Trade 
pZoneTradeCost(z,y) =
      pYearlyCostsZone(z,"Import costs with external zones: $m",y)
    + pYearlyCostsZone(z,"Export revenues with external zones: $m",y)
    + pYearlyCostsZone(z,"Import costs with internal zones: $m",y)
    + pYearlyCostsZone(z,"Export revenues with internal zones: $m",y)
    + pYearlyCostsZone(z,"Trade shared benefits: $m",y);

* Generation-only cost
pZoneGenCost(z,y) =
      pYearlyCostsZone(z,"Annualized capex: $m",y)
    + pYearlyCostsZone(z,"Fixed O&M: $m",y)
    + pYearlyCostsZone(z,"Variable O&M: $m",y)
    + pYearlyCostsZone(z,"Fuel costs: $m",y)
    + pYearlyCostsZone(z,"Spinning Reserve costs: $m",y);

* Total system cost at zone level (gen + trade + transmission)
pZoneTotalCost(z,y) = pZoneGenCost(z,y) + pZoneTradeCost(z,y) + pYearlyCostsZone(z,"Transmission additions: $m",y);


* ---------------------------------------------------------
* Energy denominators for normalization [MWh]
* ---------------------------------------------------------
* Zone energy basis:
*   - Derived from the zone-level energy balance
*   - Converts GWh → MWh
* Country energy basis:
*   - Aggregates the zonal energy bases into countries
* ---------------------------------------------------------

Parameter
    pZoneEnergyMWh(z,y)              'Annual energy output by zone [MWh]'
    pCountryEnergyMWh(c,y)           'Annual energy output by country [MWh]'
;

* Zone-level energy denominator
pZoneEnergyMWh(z,y)$pEnergyBalance(z,"Total production: GWh",y) =
    pEnergyBalance(z,"Total production: GWh",y)*1e3;

* Country-level energy denominator (aggregation)
pCountryEnergyMWh(c,y) = sum(zcmap(z,c), pZoneEnergyMWh(z,y));

* ---------------------------------------------------------
* Energy basis (MWh) used to normalize system costs
* ---------------------------------------------------------
* For zones: includes local production + net imports/exports 
*            + interzonal flows.
* For countries: aggregates the zonal bases and accounts 
*                for cross-border flows between countries.
* ---------------------------------------------------------

Parameter
    pZoneCostEnergyBasis
    pCountryCostEnergyBasis
;
    

pZoneCostEnergyBasis(z,y) =
      sum((zext,q,d,t),
          (vYearlyImportExternal.l(z,zext,q,d,t,y)
         - vYearlyExportExternal.l(z,zext,q,d,t,y)) * pHours(q,d,t))
    + sum((sTopology(Zd,z),q,d,t),
          vFlow.l(Zd,z,q,d,t,y) * pHours(q,d,t))
    + pZoneEnergyMWh(z,y);

pCountryCostEnergyBasis(c,y) =
      sum((zcmap(z,c),zext,q,d,t),
          (vYearlyImportExternal.l(z,zext,q,d,t,y)
         - vYearlyExportExternal.l(z,zext,q,d,t,y)) * pHours(q,d,t))
    + sum((zcmap(z,c),sMapConnectedZonesDiffCountries(Zd,z),q,d,t),
          vFlow.l(Zd,z,q,d,t,y) * pHours(q,d,t))
    + pCountryEnergyMWh(c,y);


* ---------------------------------------------------------
* Average costs [$/MWh]
* ---------------------------------------------------------
* Each average cost is defined as:
*   (Total annual cost in $) / (Energy basis in MWh)
*
* Zonal averages:
*   - Total average cost: includes generation, O&M, trade, transmission
*   - Generation-only average cost: includes only generation + O&M
*
* Country averages:
*   - Aggregated from zones belonging to the country
*   - Same distinction: total vs. generation-only
* ---------------------------------------------------------

* Zone-level average total system cost ($/MWh)
pZonalAverageCost(z,y)$pZoneEnergyMWh(z,y) =
    pZoneTotalCost(z,y) / pZoneCostEnergyBasis(z,y);

* Zone-level average generation cost only ($/MWh)
pZonalAverageGenCost(z,y)$pZoneEnergyMWh(z,y) =
    pZoneGenCost(z,y) / pZoneEnergyMWh(z,y);

* Country-level average total system cost ($/MWh)
pCountryAverageCost(c,y)$pCountryEnergyMWh(c,y) =
    sum(zcmap(z,c), pZoneTotalCost(z,y)) / pCountryCostEnergyBasis(c,y);

* Country-level average generation cost only ($/MWh)
pCountryAverageGenCost(c,y)$pCountryEnergyMWh(c,y) =
    sum(zcmap(z,c), pZoneGenCost(z,y)) / pCountryEnergyMWh(c,y);


* ---------------------------------------------------------
* System average cost [$ / MWh]
* ---------------------------------------------------------
* For each year y:
*   Numerator = trade costs + annualized CAPEX + O&M costs
*   Denominator = system-wide net imports + total energy produced
*
* Result = system average cost of supplying electricity
* ---------------------------------------------------------

pSystemAverageCost(y)$sum(z, pZoneEnergyMWh(z,y)) =
    (   sum((z,zext,q,d,t),
            (vYearlyImportExternal.l(z,zext,q,d,t,y)
           - vYearlyExportExternal.l(z,zext,q,d,t,y))
          * pTradePrice(zext,q,d,y,t) * pHours(q,d,t))

      + sum(ndc,
            pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex"))*1e6

      + sum(ndc$(not cs(ndc)),
            pCRFsst(ndc)*vCapStor.l(ndc,y)*pStorData(ndc,"CapexMWh"))*1e3

      + sum(ndc$(not st(ndc)),
            pCRFcst(ndc)*vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","CapexMWh"))*1e3

      + sum(ndc,
            pCRFcth(ndc)*vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal Field","CapexMWh"))*1e6

      + sum(dc, vAnnCapexGenTraj.l(dc,y))

      + sum(z, vYearlyFOMCost.l(z,y) + vYearlyVOMCost.l(z,y)
               + vYearlyFuelCost.l(z,y) + vYearlySpinningReserveCost.l(z,y))
    )
    /
    (   sum(z,
            sum((zext,q,d,t),
                (vYearlyImportExternal.l(z,zext,q,d,t,y)
               - vYearlyExportExternal.l(z,zext,q,d,t,y))
              * pHours(q,d,t))
          + pZoneEnergyMWh(z,y))
    );

* ============================================================
* 11. SOLVER PARAMETERS
* ============================================================

pSolverParameters("Solver Status")               = PA.modelstat + EPS;
pSolverParameters("Solver Time: ms")             = PA.etSolve + EPS;
pSolverParameters("Absolute gap")                = PA.objVal-PA.objEst + EPS;
pSolverParameters("Relative gap")$(PA.objVal >0) = pSolverParameters("Absolute gap")/PA.objVal + EPS;

*--- END RESULTS


$if not set OUTPUT_DIR $set OUTPUT_DIR output_csv
* create output directory

*$call /bin/sh -c "mkdir -p '%OUTPUT_DIR%'"

embeddedCode Connect:
- PythonCode:
    code: |
      import os
      os.makedirs(r"%OUTPUT_DIR%", exist_ok=True)
endEmbeddedCode


embeddedCode Connect:
- PythonCode:
    code: |
      symbols = [
      
        "pCapacityPlant",
        "pCapacityFuel",
        "pCapacityFuelCountry",
        "pAdditionalCapacity",
        "pNewCapacityFuel",
        "pNewCapacityFuelCountry",
        "pAnnualTransmissionCapacity",
        "pCapacitySummary",
        "pCapacitySummaryCountry",

        "pCostsPlant",
        "pCapexInvestment",
        "pPrice",
        "pYearlyCostsZone",
        "pYearlyCostsCountry",
        "pCostAverageCountry",
        "pCostsSystem",
        "pFuelCosts",
        "pFuelCostsCountry",
        
        "pEnergyPlant",
        "pEnergyFuel",
        "pEnergyFuelCountry",
        "pEnergyBalance",
        "pUtilizationPlant",
        "pUtilizationFuel",
        "pUtilizationTechFuel",
        "pUtilizationFuelCountry",
        "pUtilizationTechFuelCountry",
        
        "pInterchange",
        "pInterconUtilization",
        "pInterchangeCountry",
        "pCongestionShare",

        "pReserveSpinningPlantZone",
        "pReserveSpinningPlantCountry",
        "pReserveSpinningFuelZone",
        "pCapacityCredit",
        
        "pEmissionsZone",
        "pEmissionsIntensityZone",
        
        "pDispatchFuel",
        "pDispatch",
        
        "pPlantAnnualLCOE",
        "pZonalAverageCost",
        "zcmap",
        "pSettings"
        ]
      instructions.append(
        {'GAMSReader': {'symbols': [{'name': s} for s in symbols]}}
      )
      for s in symbols:
        instructions.append(
        {
          'CSVWriter':
          {
            'file': fr'%OUTPUT_DIR%%system.DirSep%{s}.csv',
            'name': s
          }
        })
endEmbeddedCode

* Additional outputs which can be included in epmresults according to the modelers' needs: pPVwSTOBalance,pPVwSTOComponents, pSolarValueZone, pSolarCost, pCapacityTechFuel, pUtilizationTech

$ifThenI.reportshort %REPORTSHORT% == 0
* Extensive reporting is used
    execute_unload 'epmresults',
      pSettings,
* 1. CAPACITY
      pCapacityPlant, pCapacityTechFuel, pCapacityFuel, pCapacityTechFuelCountry, pCapacityFuelCountry, pCapacityPlantH2,
      pRetirementsPlant, pRetirementsFuel, pRetirementsCountry, pRetirementsFuelCountry,
      pNewCapacityFuel, pNewCapacityTech, pNewCapacityFuelCountry, pNewCapacityTechCountry,
      pAnnualTransmissionCapacity, pAdditionalCapacity,
      pCapacitySummary, pCapacitySummaryCountry,
* 2. COSTS
      pCostsPlant, pCapexInvestment,
      pPrice, pImportCostsInternal, pExportRevenuesInternal, pCongestionRevenues, pTradeSharedBenefits,
      pYearlyCostsZone, pYearlyCostsCountry, pCostAverageCountry, pCostsZone, pCostsSystem, pYearlyCostsSystem
      pFuelCosts, pFuelCostsCountry, pFuelConsumption, pFuelConsumptionCountry,
* 3. ENERGY BALANCE
      pEnergyPlant, pEnergyTechFuel, pEnergyFuel, pEnergyTechFuelCountry, pEnergyFuelCountry,
      pUtilizationPlant, pUtilizationTech, pUtilizationFuel, pUtilizationTechFuel, pUtilizationFuelCountry, pUtilizationTechFuelCountry,
      pEnergyBalance, pEnergyBalanceCountry, pEnergyBalanceH2, pEnergyBalanceCountryH2,
* 4. ENERGY DISPATCH
      pDispatchPlant, pDispatchFuel, pDispatch,
* 5. RESERVES
      pReserveSpinningPlantZone, pReserveSpinningFuelZone, pReserveSpinningPlantCountry,
      pReserveMargin, pReserveMarginCountry,
* 5. INTERCONNECTIONS
      pInterchange, pInterconUtilization, pLossesTransmission, pInterchangeCountry, pLossesTransmissionCountry,
      pCongestionShare,
      pHourlyInterchangeExternal, pYearlyInterchangeExternal, pYearlyInterchangeExternalCountry, pHourlyInterchangeExternalCountry,
      pInterchangeExternalExports, pInterchangeExternalImports, pInterconUtilizationExternalExports, pInterconUtilizationExternalImports,
* 6. EMISSIONS
      pEmissionsZone, pEmissionsIntensityZone, pEmissionsCountrySummary, pEmissionsIntensityCountry,
      pEmissionMarginalCosts, pEmissionMarginalCostsCountry,
* 7. PRICES
      pYearlyPrice, pYearlyPriceExport, pYearlyPriceImport, pYearlyPriceHub,
      pYearlyPriceCountry, pYearlyPriceExportCountry, pYearlyPriceImportCountry,
* 8. SPECIAL TECHNOLOGIES
      pCSPBalance, pCSPComponents, pPVwSTOBalance, pPVwSTOComponents, pStorageBalance, pStorageComponents,
      pSolarPower, pSolarEnergyZone, pSolarValueZone, pSolarCost,
* 9. METRICS
      pPlantAnnualLCOE, pZonalAverageCost, pZonalAverageGenCost, pCountryAverageCost, pCountryAverageGenCost, pSystemAverageCost,
* 10. SOLVER PARAMETERS
      pSolverParameters,
* 11. ADDITIONAL OUTPUTS
      pVarCost, pCapacityCredit
;
$elseIfI.reportshort %REPORTSHORT% == 1
*  Limited reporting is used
    execute_unload 'epmresults', pYearlyCostsZone, pYearlyCostsZoneFull, pEnergyBalance
    ;
$endIf.reportshort