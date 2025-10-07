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

$offEolCom
$eolCom //

*-------------------------------------------------------------------
* DECLARATION OF SETS, PARAMETERS AND VARIABLES
*-------------------------------------------------------------------



alias (y,y2);
alias (f,f2);
alias (c,c2);



* Generators
Sets
   eg(g)                  'existing generators'
   ng(g)                  'new generators'
   commtransmission(z,z2) 'Committed transmission lines'
   cs(g)                  'concentrated solar power'
   so(g)                  'PV plants with storage'
   stp(g)                 'storage for PV plant'
   st(g)                  'all storage plants'
   dc(g)                  'candidate generators with capex trajectory'
   ndc(g)                 'candidate generators without capec trajectory'
   vre(g)                 'variable renewable generators'
   re(g)                  'renewable generators'
   RampRate(g)            'ramp rate constrained generator blocks' // Ramprate takes out inflexible generators for a stronger formulation so that it runs faster
   VRE_noROR(g)           'VRE generators that are not RoR generators - used to estimate spinning reserve needs'
;



* Mapping (fuel, storage...)
Sets
   gfmap(g,f)             'generator g mapped to fuel f'
   gzmap(g,z)             'generator g mapped to zone z'
   zfmap(z,f)             'fuel f available in zone z'
   gsmap(g2,g)            'generator storage map'
   sTopology(z,z2)        'network topology - to be assigned through network data'
   sMapConnectedZonesDiffCountries(z,z2)          'set of connecting zones belonging to different countries'   
;


* To check
Sets
   gstatus  'generator status' / Existing, Candidate, Committed /
   tstatus  'transmission status' / Candidate, Committed/
   mipline 'Solver option lines'
   mipopt(mipline<)                 'MIP solver options' / system.empty /
;



* Implicit variable domain
Sets
   sPwrOut(g,f,q,d,t,y)        'valid tuples for vPwrOut'
   sExportPrice(z,zext,q,d,t,y)     'valid tuples for vYearlyExportExternal'
   sImportPrice(z,zext,q,d,t,y)     'valid tuples for vYearlyImportExternal'
   sAdditionalTransfer(z,z2,y) 'valid tuples for vNewTransferCapacity'
   sFlow(z,z2,q,d,t,y)         'valid tuples for vFlow'
   sSpinningReserve(g,q,d,t,y)         'valid tuples for vSpinningReserve'
;

Singleton sets
   sStartYear(y)
   sFirstHour(t)
   sLastHour(t)
   sFirstDay(d)
;


* Additional parameters for results and reporting
Parameter
   pAllHours(q,d,y,t)                  'System peak hours'
   pFindSysPeak(y)                     'System peak by year'
   pSeasonalReporting                  'Seasonal reporting flag'
   pSystemResultReporting              'System reporting flag'
   pInterConMode                       'Interconnection mode flag'
   pNoTransferLim                      'Transfer limit flag'
   pAllowExports                       'Export permission flag'
   pVRECapacityCredits                 'VRE capacity credits'
   pDR                                 'Discount rate'
   pCaptraj                           'CAPEX trajectory flag'
   pIncludeEE                         'Energy efficiency flag'
   pSystem_CO2_constraints            'System CO2 constraint flag'
   pExtTransferLimitIn(z,zext,q,y)    'External import limits'
   pExtTransferLimitOut(z,zext,q,y)   'External export limits'
   pMaxLoadFractionCCCalc             'Load threshold for capacity credit calc'
   pVREForecastError                  'VRE forecast error percentage'
;

* Technology and mapping sets
Set 
   gprimf(g,f)          'Primary fuel mapping'
   gtechmap(g,tech)     'Generator-technology mapping'
   gstatusmap(g,gstatus) 'Generator status mapping'
   tstatusmap(z,z2,tstatus) 'Transmission status mapping'
   Zd(z)                'Zone definitions'
   Zt(z)                'Zone types'
   stg(g)               'Grid storage units'
   ror(g)               'Run of river units'
;


Parameters
   pCostOfCurtailment               'Cost of curtailment'
   pCostOfCO2backstop               'Cost of climate backstop techno in $ per ton of CO2'
   pWeightYear(y)                   'weight on years'
* Generators
   pGenData(g,pGenDataInputHeader)                 'generator data'
   pAvailability(g,q)               'Availability by generation type and season or quarter in percentage - need to reflect maintenance'
   pCapexTrajectories(g,y)          'capex trajectory  final'
* Exchanges
   pMaxExchangeShare(y,c)           'Max share of exchanges by country [same limit for imports or exports for now]'

   pAllowHighTransfer
   pAllowExports                    'Allow price based exports'   
   pTradePrice(zext,q,d,y,t)        'trade price - export or import driven by prices [assuming each zone in a country can only trade with one external zone]'
   pMaxImport                       'Maximum Hourly Imports, based on hourly country demand'   
   pMaxExport                       'Maximum Hourly Exports, based on hourly country demand'
   pExtTransferLimit(z,zext,q,*,y)  'external transfer limit'
   pExtTransferLimitIn(z,zext,q,y)  'transfer limit with external zone for import towards internal zone'
   pExtTransferLimitOut(z,zext,q,y) 'transfer limit with external zone for export towards external zone'
* Demand
   pDemandData(z,q,d,y,t)           'hourly load curves by quarter(seasonal) and year'
* Renewables and storage
   pCSPProfile(g,q,d,t)             'solar profile for CSP in pu'
   pStoPVProfile(g,q,d,t)           'solar profile for Pv with Storage in pu'
   pStorData(g,pStoreDataHeader)                'Storage data'
   pVREForecastError                'Percentage error in VRE forecast [used to estimated required amount of spinning reserve]'
   
* Reserves
   pCapacityCredit(g,y)             'Share of capacity counted towards planning reserves'
   pVREForecastError                'Spinning reserve needs for VRE (as a share of VRE generation)'
* CO2
   pCarbonPrice(y)                  'Carbon price in USD per ton of CO2eq'
   pFuelCarbonContent(f)            'Fuel carbon content in tCO2 per MMBTu'

   
* Economic parameters
   pRR(y)                          'accumulated return rate factor'
   pWACC                            'Weighted Average Cost of Capital'
   pCRF (G)                         'capital recovery factor'
   pCRFsst(g)                       'capital recovery factor storage'
   pCRFcst(g)                       'capital recovery factor CSP storage'
   pCRFcth(g)                       'capital recovery factor CSP thermal'
   pVOLL                            'VOLL'
   pPlanningReserveVoLL             'Planning Reserve VoLL per MW'
   pSpinningReserveVoLL             'Spinning Reserve VoLL per MWh'
   pMaxCapital                      'Capital limit in billion dollars'
   pVarCost(g,f,y)                  'Variable cost - fuel plus VOM'
   pFuelCost(g,f,y)                 'Fuel cost in USD per MMBTU'
   pVOMCost(g,f,y)
* Control parameters
   pramp_constraints                'Whether constraints on ramp up and down are included'
   pfuel_constraints
   pcapital_constraints             'Whether constraints on available capital for infrastructure are included'
   pmingen_constraints
   pincludeCSP
   pincludeStorage
   pIncludeCarbon                   'include the cost of carbon'
   pSurplusPenalty
   pMinRE
   pMinRETargetYr
   pzonal_CO2_constraints
   pSystem_CO2_constraints
   pzonal_spinning_reserve_constraints   'Whether constraints on spinning reserves at the country level are included'
   psystem_spinning_reserve_constraints  'Whether constraints on spinning reserves at the region level are included'
   pplanning_reserve_constraints         'Whether constraints on planning reserves are included'
   pinterco_reserve_contribution         'How much interconnections contribute to spinning reserve needs at the country level'
   pIncludeIntercoReserves               'Whether transmission lines are considered when assessing planning reserve needs at the country level'
   psystem_reserve_margin                'Share of peak demand that should be met with planning reserves'
   pHeatrate(g,f)                   'Heatrate of fuel f in generator g'
   pIncludeDecomCom                 'Include simultaneous commissioning'
;

Positive Variables
   vPwrOut(g,f,q,d,t,y)      'generation dispatch for aggregated generators in MW'
   vUSE(z,q,d,t,y)           'unserved demand'
   vSurplus(z,q,d,t,y)       'surplus generation'
   vYearlySurplus(z,y)
   
   vCap(g,y)                 'total capacity in place accounting for legacy, new and retired plants (MW)'
   vBuild(g,y)               'Build (MW)'
   vRetire(g,y)              'Retire (MW)'

   vFlow(z,z2,q,d,t,y)       'flow from z to z2 in MW'
   vYearlyImportExternal(z,zext,q,d,t,y)   'external (price-driven) import'
   vYearlyExportExternal(z,zext,q,d,t,y)   'external (price-driven) export'

   vFuel(z,f,y)              'annual fuel in MMBTU'

   vAnnCapexGenTraj(g,y)         'Annualized capex for capacity with capex trajectory'
   vAnnGenCapex(g,y)            'Annualized capex for capacity installed in Year y'
   vAnnCapex(z,y)            'Annualized capex for zone z in year y'

   vThermalOut(g,q,d,t,y)    'Generation from thermal element (CSP solar field in MW)'
   vCapStor(g,y)             'Total capacity of storage installed (MWh)'
   vCapTherm(g,y)            'Total thermal CSP solar field capacity installed (MW)'
   vStorage(g,q,d,t,y)       'Storage level  (MW)'
   vStorInj(g,q,d,t,y)       'Storage injection  (MW)'
   vStorOut(g,q,d,t,y)       'Storage output (MW)'
   vBuildStor(g,y)           'Build storage variable (MWh)'
   vRetireStor(g,y)          'Retire storage variable (MWh)'
   vBuildTherm(g,y)          'Build thermal elements (CSP solar field) variable (MW)'
   vRetireTherm(g,y)         'Retire thermal elemtns (CSP solar field) variable (MW)'

   vSpinningReserve(g,q,d,t,y)                 'Spinning reserve provision (MW)'
   vUnmetPlanningReserveCountry(c,y)              'Unserved zonal planning reserve'
   vUnmetPlanningReserveSystem(y)              'Unmet reserve for system -- capacity reserve'
   vUnmetSpinningReserveCountry(c,q,d,t,y)        'Unmet spinning reserve - local constraint'
   vUnmetSpinningReserveSystem(q,d,t,y)        'Unmet spinning reserve - system constraint'

   vZonalEmissions(z,y)      'average CO2eq emissions per year and zone in tons per MWh'
   vTotalEmissions(y)        'total regional CO2eq emissions per year in tons'
   vYearlyCO2backstop(c,y)      'CO2 emissions above the constraint by zone (t)'
   vYearlySysCO2backstop(y)     'system CO2 emissions above the constraint(t)'


   vNewTransferCapacity(z,z2,y)        'additional transfer limit'
   vAnnualizedTransmissionCapex(z,y)  'added transmission cost (not included in cost of generation)'
   vYearlyCurtailmentCost(z,y)
   vCurtailedVRE(z,g,q,d,t,y)

;

Free Variable
   vNPVCost                     'discounted total system cost'
   vStorNet(g,q,d,t,y)
   vYearlyTotalCost(c,y)
   vYearlyVariableCost(z,y)
   vYearlyUSECost(z,y)
   vYearlyExternalTradeCost(z,y)
   vYearlySpinningReserveCost(z,y)      'Yearly spinning reserve costs'
   vYearlyUnmetReserveCostCountry(c,y)         'Country unmet spinning and planning reserve'
   vYearlyCarbonCost(z,y)               'country carbon cost'
   vYearlyCO2BackstopCostCountry(c,y)              'ccost of CO2 backstop'
   vYearlyUnmetPlanningReserveCostSystem(y)              'system unmet planning reserve cost'
   vYearlyUnmetSpinningReserveCostSystem(y)              'system unmet spinning reserve cost'
   vYearlyCO2BackstopCostSystem(y)              'system CO2 backstop cost'
   vYearlyUnmetReserveCostCountry(c,y)              'country unmet reserve cost'
   vYearlyUnmetPlanningReserveCostCountry(c,y)              'country unmet planning reserve cost'
   vYearlyUnmetSpinningReserveCostCountry(c,y)              'country unmet spinning reserve cost'
   vYearlyFOMCost(z,y)                'country FOM cost'
   vYearlyFuelCost(z,y)               'country fuel cost'
   vYearlyVOMCost(z,y)                'country VOM cost'
   vYearlyImportExternalCost(z,y)        'cost of imports from external zones'
   vYearlyExportExternalCost(z,y)        'cost of exports to external zones'
   vSupply(z,q,d,t,y) "Total supply meeting demand at each node and time"
;

Integer variable
   vBuildTransmission(z,z2,y)
   vBuiltCapVar(g,y)
   vRetireCapVar(g,y)
;


Equations
   eNPVCost                        'objective function'
   eYearlyTotalCost(c,y)
   eYearlyVOMCost(z,y)
   eYearlySpinningReserveCost(z,y)
   eYearlyUSECost(z,y)
   eYearlyExternalTradeCost(z,y)
   eYearlyUnmetReserveCostCountry(c,y)
   eYearlyCarbonCost(z,y)
   eYearlyFOMCost(z,y)                'Total yearly FOM cost'
   eYearlyCO2BackstopCostCountry(c,y)     'cost of CO2 backstop'
   eYearlyVariableCost(z,y)               'Total yearly  cost'
   eYearlyFuelCost(z,y)                   'Total yearly fuel cost'
   eYearlImportExternalCost(z,y)        'Total yearly cost of imports from external zones'
   eYearlyExportExternalCost(z,y)        'Total yearly cost of exports to external zones'
   eYearlyUnmetSpinningReserveCostCountry(c,y)              'country unmet spinning reserve cost'
   eYearlyUnmetPlanningReserveCostCountry(c,y)              'country unmet planning reserve cost'
   eYearlyUnmetPlanningReserveCostSystem(y)              'system unmet planning reserve cost'
   eYearlyUnmetSpinningReserveCostSystem(y)              'system unmet spinning reserve'
   eYearlyCO2BackstopCostsSystem(y)              'system CO2 backstop cost'

   eTotalAnnualizedCapex(z,y)           'Total annualized capex for all generators in year y'
   eAnnualizedCapexInit(g,y)                  'Annualized capex'
   eAnnualizedCapexUpdate(g,y)
   eTotalAnnualizedGenCapex(g, y)            'Total annualized capex for all generators in year y'


   eDemSupply(z,q,d,t,y)           'demand balance'
   eDefineSupply(z,q,d,t,y) "Definition of total supply at each node"
   eMaxHourlyExportShareRevenue(c,q,d,t,y) 'max exports to an external zone (hourly limit)'
   eMaxHourlyImportShareCost(c,q,d,t,y) 'max imports from an external zone  (hourly limit)'

   eInitialCapacity(g,y)                'capacity balance'
   eCapacityEvolutionExist(g,y)               'capacity balance'
   eCapacityEvolutionNew(g,y)               'capacity balance'
   eInitialBuildLimit(g)
   eBuiltCap(g,y)                  'built capacity'
   eRetireCap(g,y)                 'retired capacity'


   eMinGenRE(c,y)                  'Min Generation of RE after a target year at country level'
   eMaxCF(g,q,y)                   'max capacity factor'
   eMinGen(g,q,d,t,y)              'Minimum generation limit for new generators'

   eFuel(z,f,y)                    'fuel balance'
   eFuelLimit(c,f,y)               'fuel limit at country level'

   eRampUpLimit(g,q,d,t,y)         'Ramp up limit'
   eRampDnLimit(g,q,d,t,y)         'Ramp down limit'

   eSpinningReserveLim(g,q,d,t,y)              'Reserve limit as a share of capacity'
   eSpinningReserveLimVRE(g,f,q,d,t,y)           'Reserve limit for VRE as a share of capacity adjusted for production profile'
   eJointResCap(g,q,d,t,y)                     'Joint reserve and generation limit'
   eSpinningReserveReqCountry(c,q,d,t,y)          'Country spinning reserve requirement'
   eSpinningReserveReqSystem(q,d,t,y)          'System spinning reserve requirement'
   ePlanningReserveReqSystem(y)                'Min system planning reserve requirement'
   ePlanningReserveReqCountry(c,y)                'Minimum capacity reserve over peak demand at country level'

   eTransferCapacityLimit(z,z2,q,d,t,y)    'Transfer limits'
   eMinImportRequirement(z2,z,q,d,t,y) 'Minimum transfer across some transmission line defined at the hourly scale'
   eVREProfile(g,f,z,q,d,t,y)      'VRE generation restricted to VRE profile'
   eMaxAnnualImportShareCost(c,y)            'import limits: max import from external zones'
   eMaxAnnualExportShareRevenue(c,y)            'export limits: max export to external zones'
   eYearlySurplusCost(z,y)
   eCumulativeTransferExpansion(z,z2,y)
   eSymmetricTransferBuild(z,z2,y)
   eAnnualizedTransmissionCapex (z,y)
   eExternalImportLimit(z,zext,q,d,t,y) 'import limits from external zone in MW'
   eExternalExportLimit(z,zext,q,d,t,y) 'export limits to external zone in MW'



   eCapitalConstraint              'capital limit expressed by pMaxCapital in billion USD'
   eZonalEmissions(z,y)            'CO2eq emissions by zone and year in tons'
   eEmissionsCountry(c,y)          'constraint on country CO2eq emissions'
   eTotalEmissions(y)              'total regional CO2eq emissions by year in tons'
   eTotalEmissionsConstraint(y) 	'constraint on total CO2eq emissions by year in tons'
  

   eChargeRampDownLimit(g,q,d,t,y)
   eChargeRampUpLimit(g,q,d,t,y)

   eYearlyCurtailmentCost(z,y)

   eStateOfChargeUpdate(g,q,d,t,y)
   eStateOfChargeInit(g,q,d,t,y)
* eSOCCycleClosure(g,q,d,t,y)
* eDailyStorageEnergyBalance(g,q,d,y)

   eSOCSupportsReserve(g,q,d,t,y)
   eChargeCapacityLimit(g,q,d,t,y)
   eChargeLimitWithPVProfile(g,q,d,t,y)
   eNetChargeBalance(g,q,d,t,y)

   eSOCUpperBound(g,q,d,t,y)
   eStorageCapMinConstraint(g,q,d,t,y)         'storage capacity (energy) must be at least 1 hour if installed'

   eCSPStorageCapacityLimit(g,q,d,t,y)
   eCSPStorageInjectionLimit(g,q,d,t,y)
   eCSPStorageInjectionCap(g,q,d,t,y)
   eCSPThermalOutputLimit(g,q,d,t,y)
   eCSPPowerBalance(g,q,d,t,y)
   eCSPStorageEnergyBalance(g,q,d,t,y)
   eCSPStorageInitialBalance(g,q,d,t,y)

   eCapacityStorLimit(g,y)
   eCapStorBalance(g,y)
   eCapStorAnnualUpdateEG(g,y)
   eCapStorAnnualUpdateNG(g,y)
   eCapStorInitialNG(g,y)
   eBuildStorNew(g)

   eCapacityThermLimit(g,y)
   eCapThermBalance1(g,y)
   eCapThermBalance2(g,y)
   eCapThermBalance3(g,y)
   eBuildThermNew(g)
;


*---    Objective function
* Adding system-level cost of reserves and CO2 backstop to the total cost
eNPVCost..
   vNPVCost =e= sum(y, pRR(y)*pWeightYear(y)*(sum(c, 
                                 vYearlyTotalCost(c,y)) + 
                                 vYearlyUnmetPlanningReserveCostSystem(y) + 
                                 vYearlyUnmetSpinningReserveCostSystem(y) +
                                 vYearlyCO2BackstopCostSystem(y))
                                 );

*---  Cost equations
*--- System-level costs
eYearlyUnmetPlanningReserveCostSystem(y)..
   vYearlyUnmetPlanningReserveCostSystem(y) =e= vUnmetPlanningReserveSystem(y)*pPlanningReserveVoLL;

eYearlyUnmetSpinningReserveCostSystem(y)..
   vYearlyUnmetSpinningReserveCostSystem(y) =e= sum((q,d,t), vUnmetSpinningReserveSystem(q,d,t,y)*pHours(q,d,t)*pSpinningReserveVoLL);

eYearlyCO2BackstopCostsSystem(y)..
   vYearlyCO2BackstopCostSystem(y) =e= vYearlySysCO2backstop(y)*pCostOfCO2backstop;

* Note capex is full capex in $m per MW. Also note VarCost includes fuel cost and VOM -
* essentially the short run marginal cost for the generator

* Adding country-level cost of reserves and CO2 backstop to the total cost
eYearlyTotalCost(c,y)..
   vYearlyTotalCost(c,y) =e= vYearlyUnmetReserveCostCountry(c,y)+ vYearlyCO2BackstopCostCountry(c,y) 
                           + sum(zcmap(z,c), vAnnCapex(z, y) 
                                           + vYearlyFOMCost(z, y)
                                           + vYearlyVariableCost(z,y)
                                           + vYearlySpinningReserveCost(z,y)
                                           + vYearlyUSECost(z,y)
                                           + vYearlyCarbonCost(z,y)
                                           + vYearlyExternalTradeCost(z,y)
                                           + vAnnualizedTransmissionCapex(z,y)
                                           + vYearlyCurtailmentCost(z,y)
                                           + vYearlySurplus(z,y));

*--- Country-level costs

* Yearly unmet reserve cost at the country level
eYearlyUnmetReserveCostCountry(c,y)..
   vYearlyUnmetReserveCostCountry(c,y) =e= vYearlyUnmetPlanningReserveCostCountry(c,y) + vYearlyUnmetSpinningReserveCostCountry(c,y);

* Yearly unmet spinning reserve cost at the country level
eYearlyUnmetSpinningReserveCostCountry(c,y)..
   vYearlyUnmetSpinningReserveCostCountry(c,y) =e= sum((q,d,t), vUnmetSpinningReserveCountry(c,q,d,t,y)*pHours(q,d,t)*pSpinningReserveVoLL);

* Yearly unmet planning reserve cost at the country level
eYearlyUnmetPlanningReserveCostCountry(c,y)..
   vYearlyUnmetPlanningReserveCostCountry(c,y) =e= vUnmetPlanningReserveCountry(c,y)*pPlanningReserveVoLL;

* Yearly CO2 backstop cost at the country level                    
eYearlyCO2BackstopCostCountry(c,y)..
   vYearlyCO2BackstopCostCountry(c,y) =e= vYearlyCO2backstop(c,y)*pCostOfCO2backstop;

*--- Zonal-level costs

* Annualized CAPEX for all zones in year y
eTotalAnnualizedCapex(z, y)..
   vAnnCapex(z,y) =e= sum(gzmap(g,z), vAnnGenCapex(g,y));

* Annualized CAPEX for all generators in year y
eTotalAnnualizedGenCapex(g,y)..
   vAnnGenCapex(g,y) =e=
       vAnnCapexGenTraj(g,y)$(dc(g))
     + pCRF(g)*vCap(g,y)*pGenData(g,"Capex")*1e6$(ndc(g))
     + pCRFsst(g)*vCapStor(g,y)*pStorData(g,"CapexMWh")*1e3$(ndc(g) and not cs(g))
     + pCRFcst(g)*vCapStor(g,y)*pCSPData(g,"Storage","CapexMWh")*1e3$(ndc(g) and not st(g))
     + pCRFcth(g)*vCapTherm(g,y)*pCSPData(g,"Thermal Field","CapexMWh")*1e6$(ndc(g));

* Annualized CAPEX accumulation (years after the start year) for generators with capex trajectory
eAnnualizedCapexUpdate(dc,y)$(not sStartYear(y))..
   vAnnCapexGenTraj(dc,y) =e= vAnnCapexGenTraj(dc,y-1)
                     + vBuild(dc,y)*pGenData(dc,"Capex")*pCapexTrajectories(dc,y)*pCRF(dc)*1e6
                     + vBuildStor(dc,y)*pStorData(dc,"CapexMWh")*pCapexTrajectories(dc,y)*pCRFsst(dc)*1e3
                     + vBuildStor(dc,y)*pCSPData(dc,"Storage","CapexMWh")*pCapexTrajectories(dc,y)*pCRFcst(dc)*1e3
                     + vBuildTherm(dc,y)*pCSPData(dc,"Thermal Field","CapexMWh")*pCapexTrajectories(dc,y)*pCRFcth(dc)*1e6;
                                                                          
* Initial annualized CAPEX in the start year for generators with capex trajectory
eAnnualizedCapexInit(dc,sStartYear(y))..
   vAnnCapexGenTraj(dc,y) =e= vBuild(dc,y)*pGenData(dc,"Capex")*pCapexTrajectories(dc,y)*pCRF(dc)*1e6
                     + vBuildStor(dc,y)*pStorData(dc,"CapexMWh")*pCapexTrajectories(dc,y)*pCRFsst(dc)*1e3
                     + vBuildStor(dc,y)*pCSPData(dc,"Storage","CapexMWh")*pCapexTrajectories(dc,y)*pCRFcst(dc)*1e3
                     + vBuildTherm(dc,y)*pCSPData(dc,"Thermal Field","CapexMWh")*pCapexTrajectories(dc,y)*pCRFcth(dc)*1e6;
                     

* FOM costs including fixed O&M costs for generators, storage, and CSP thermal field
eYearlyFOMCost(z,y)..
   vYearlyFOMCost(z,y) =e= sum(gzmap(g,z),  vCap(g,y)*pGenData(g,"FOMperMW"))
            + sum(gzmap(st,z), vCapStor(st,y)*pStorData(st,"FixedOMMWh"))
            + sum(gzmap(cs,z), vCapStor(cs,y)*pCSPData(cs,"Storage","FixedOMMWh"))
            + sum(gzmap(cs,z), vCapTherm(cs,y)*pCSPData(cs,"Thermal field","FixedOMMWh"));

* Variable costs equals fuel cost plus VOM cost
eYearlyVariableCost(z,y)..
   vYearlyVariableCost(z,y) =e= vYearlyFuelCost(z,y) + vYearlyVOMCost(z,y);

* Fuel costs
eYearlyFuelCost(z,y)..
   vYearlyFuelCost(z,y) =e= sum((gzmap(g,z),f,q,d,t), pFuelCost(g,f,y)*vPwrOut(g,f,q,d,t,y)*pHours(q,d,t));

* VOM
eYearlyVOMCost(z,y)..
   vYearlyVOMCost(z,y) =e= sum((gzmap(g,z),f,q,d,t), pVOMCost(g,f,y)*vPwrOut(g,f,q,d,t,y)*pHours(q,d,t));

* Note: ReserveCost is in $/MWh -- this is the DIRECT cost of holding reserve like wear and tear that a generator bids in a market
eYearlySpinningReserveCost(z,y)..
   vYearlySpinningReserveCost(z,y) =e= sum((gzmap(g,z),q,d,t), vSpinningReserve(g,q,d,t,y)*pGenData(g,"ReserveCost")*pHours(q,d,t));

* Unserved energy cost (USE)
eYearlyUSECost(z,y)..
   vYearlyUSECost(z,y) =e= sum((q,d,t), vUSE(z,q,d,t,y)*pVoLL*pHours(q,d,t));

* Surplus cost
eYearlySurplusCost(z,y)..
   vYearlySurplus(z,y) =e= sum((q,d,t), vSurplus(z,q,d,t,y)*pSurplusPenalty*pHours(q,d,t));

* Curtailment cost
eYearlyCurtailmentCost(z,y)..
   vYearlyCurtailmentCost(z,y) =e= sum((gzmap(g,z),q,d,t), vCurtailedVRE(z,g,q,d,t,y)*pCostOfCurtailment*pHours(q,d,t));

* Yearly trade cost
eYearlyExternalTradeCost(z,y)..
   vYearlyExternalTradeCost(z,y) =e= vYearlyImportExternalCost(z,y) - vYearlyExportExternalCost(z,y);

* Yearly import and export costs from external zones
eYearlImportExternalCost(z,y)..
   vYearlyImportExternalCost(z,y) =e= sum((zext,q,d,t), vYearlyImportExternal(z,zext,q,d,t,y)*pTradePrice(zext,q,d,y,t)*pHours(q,d,t));

* Yearly export cost to external zones
eYearlyExportExternalCost(z,y)..
   vYearlyExportExternalCost(z,y) =e= sum((zext,q,d,t), vYearlyExportExternal(z,zext,q,d,t,y)*pTradePrice(zext,q,d,y,t)*pHours(q,d,t));

* Yearly CO2 emissions cost for each zone
eYearlyCarbonCost(z,y)..
   vYearlyCarbonCost(z,y) =e= pIncludeCarbon*pCarbonPrice(y)
                            * Sum((gzmap(g,z),gfmap(g,f),q,d,t), vPwrOut(g,f,q,d,t,y)*pHeatRate(g,f)*pFuelCarbonContent(f)*pHours(q,d,t));


* Ensures that when accessing parameters like CostPerLine or Life for a connection between zones i and j, 
* the model always uses the maximum of both directions, regardless of ordering.
$macro symmax(s,i,j,h) max(s(i,j,h),s(j,i,h))

* Computes annualized investment cost of new transmission lines connected to zone z
* using the annuity formula and averaging (divided by 2) to avoid double-counting symmetric lines
eAnnualizedTransmissionCapex(z,y)$(pAllowHighTransfer and sum(sTopology(z,z2),1))..
   vAnnualizedTransmissionCapex(z,y) =e=
       sum(sTopology(z,z2),
           vNewTransferCapacity(z,z2,y)
         * symmax(pNewTransmission,z,z2,"CostPerLine")
         * 1e6)
     / 2
     * (pWACC / (1 - (1 / ((1 + pWACC) ** sum(sTopology(z,z2), symmax(pNewTransmission,z,z2,"Life"))))));

*--- Demand supply balance constraint
eDemSupply(z,q,d,t,y)..
   pDemandData(z,q,d,y,t) * pEnergyEfficiencyFactor(z,y) =e= vSupply(z,q,d,t,y);

eDefineSupply(z,q,d,t,y)..
   vSupply(z,q,d,t,y) =e=
     sum((gzmap(g,z),gfmap(g,f)), vPwrOut(g,f,q,d,t,y))
   - sum(sTopology(z,z2), vFlow(z,z2,q,d,t,y))
   + sum(sTopology(z,z2), vFlow(z2,z,q,d,t,y) * (1 - pLossFactor(z,z2,y)))
   - sum(gzmap(st,z), vStorInj(st,q,d,t,y))$(pincludeStorage)
   + sum(zext, vYearlyImportExternal(z,zext,q,d,t,y))
   - sum(zext, vYearlyExportExternal(z,zext,q,d,t,y))
   + vUSE(z,q,d,t,y)
   - vSurplus(z,q,d,t,y);


*--- Generator Capacity equations 
* Initial capacity balance in the first model year
eInitialCapacity(g,sStartYear(y))..
   vCap(g,y) =e= pGenData(g,"Capacity")$(eg(g) and (pGenData(g,"StYr") <= sStartYear.val))
               + vBuild(g,y) - vRetire(g,y);

* Year-on-year capacity tracking for existing generators
eCapacityEvolutionExist(eg,y)$(not sStartYear(y))..
   vCap(eg,y) =e= vCap(eg,y-1) + vBuild(eg,y) - vRetire(eg,y);

* Year-on-year capacity tracking for new generators (non-retirable)
eCapacityEvolutionNew(ng,y)$(not sStartYear(y))..
   vCap(ng,y) =e= vCap(ng,y-1) + vBuild(ng,y);

* Limit on initial build for new generators (starting after model year 1)
eInitialBuildLimit(eg)$(pGenData(eg,"StYr") > sStartYear.val)..
   sum(y, vBuild(eg,y)) =l= pGenData(eg,"Capacity");

eMinGenRE(c,y)$(pMinRE and y.val >= pMinRETargetYr)..
   sum((zcmap(z,c),gzmap(RE,z),gfmap(RE,f),q,d,t), vPwrOut(RE,f,q,d,t,y)*pHours(q,d,t)) =g=
   sum((zcmap(z,c),q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))*pMinRE;

eBuiltCap(ng,y)$pGenData(ng,"DescreteCap")..
   vBuild(ng,y) =e= pGenData(ng,"UnitSize")*vBuiltCapVar(ng,y);

eRetireCap(eg,y)$(pGenData(eg,"DescreteCap") and (y.val <= pGenData(eg,"RetrYr")))..
   vRetire(eg,y) =e= pGenData(eg,"UnitSize")*vRetireCapVar(eg,y);


*--- Production equations
eJointResCap(g,q,d,t,y)..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) + vSpinningReserve(g,q,d,t,y) =l= vCap(g,y)*(1+pGenData(g,"Overloadfactor"));

eMaxCF(g,q,y)..
   sum((gfmap(g,f),d,t), vPwrOut(g,f,q,d,t,y)*pHours(q,d,t)) =l= pAvailability(g,q)*vCap(g,y)*sum((d,t), pHours(q,d,t));

eFuel(zfmap(z,f),y)..
   vFuel(z,f,y) =e= sum((gzmap(g,z),gfmap(g,f),q,d,t), vPwrOut(g,f,q,d,t,y)*pHours(q,d,t)*pHeatRate(g,f));

eFuelLimit(c,f,y)$(pfuel_constraints and pMaxFuelLimit(c,f,y) > 0)..
   sum((zcmap(z,c),zfmap(z,f)), vFuel(z,f,y)) =l= pMaxFuelLimit(c,f,y)*1e6;

eMinGen(g,q,d,t,y)$(pmingen_constraints and pGenData(g,"MinLimitShare") > 0)..
    sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) =g= vCap(g,y)*pGenData(g,"MinLimitShare") ; 

eRampDnLimit(g,q,d,t,y)$(Ramprate(g) and not sFirstHour(t) and pramp_constraints)..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t-1,y)) - sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) =l= vCap(g,y)*pGenData(g,"RampDnRate");

eRampUpLimit(g,q,d,t,y)$(Ramprate(g) and not sFirstHour(t) and pramp_constraints)..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) - sum(gfmap(g,f), vPwrOut(g,f,q,d,t-1,y)) =l= vCap(g,y)*pGenData(g,"RampUpRate");

* Note that we are effectively assuming grid-connected RE generation to be dispatchable. Generally speaking, most RE will be
* dispatched anyway because they have zero cost (i.e., not a different outcome from net load approach, but this allows for
* RE generation to be rejected as well
eVREProfile(gfmap(VRE,f),z,q,d,t,y)$gzmap(VRE,z)..
   vPwrOut(VRE,f,q,d,t,y) + vCurtailedVRE(z,VRE,q,d,t,y) =e= pVREgenProfile(VRE,q,d,t)*vCap(VRE,y);


*--- Reserve equations
* Spinning reserve limit as a share of capacity
eSpinningReserveLim(g,q,d,t,y)$(pzonal_spinning_reserve_constraints or psystem_spinning_reserve_constraints)..
   vSpinningReserve(g,q,d,t,y) =l= vCap(g,y)*pGenData(g,"ResLimShare");
   
* Spinning reserve limit for VRE as a share of capacity adjusted for production profile
eSpinningReserveLimVRE(gfmap(VRE,f),q,d,t,y)$(pzonal_spinning_reserve_constraints or psystem_spinning_reserve_constraints)..
    vSpinningReserve(VRE,q,d,t,y) =l= vCap(VRE,y)*pGenData(VRE,"ResLimShare")* pVREgenProfile(VRE,q,d,t);

* This constraint increases solving time x3
* Reserve constraints include interconnections as reserves too
eSpinningReserveReqCountry(c,q,d,t,y)$pzonal_spinning_reserve_constraints..
   sum((zcmap(z,c),gzmap(g,z)),vSpinningReserve(g,q,d,t,y))
 + vUnmetSpinningReserveCountry(c,q,d,t,y)
 + pinterco_reserve_contribution * sum((zcmap(z,c),sMapConnectedZonesDiffCountries(z2,z)), pTransferLimit(z2,z,q,y)
                                        + vNewTransferCapacity(z2,z,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine")*pAllowHighTransfer
                                        - vFlow(z2,z,q,d,t,y))
   =g= pSpinningReserveReqCountry(c,y) + sum((zcmap(z,c),gzmap(VRE_noROR,z),gfmap(VRE_noROR,f)), vPwrOut(VRE_noROR,f,q,d,t,y))*pVREForecastError;
   
* System spinning reserve requirement
eSpinningReserveReqSystem(q,d,t,y)$psystem_spinning_reserve_constraints..
   sum(g, vSpinningReserve(g,q,d,t,y)) + vUnmetSpinningReserveSystem(q,d,t,y) =g= pSpinningReserveReqSystem(y) + sum(gfmap(VRE_noROR,f), vPwrOut(VRE_noROR,f,q,d,t,y))*pVREForecastError;

* Planning reserve requirement at the country level
ePlanningReserveReqCountry(c,y)$(pplanning_reserve_constraints and pPlanningReserveMargin(c))..
   sum((zcmap(z,c),gzmap(g,z)), vCap(g,y)*pCapacityCredit(g,y))
 + vUnmetPlanningReserveCountry(c,y)
 + (sum((zcmap(z,c),sMapConnectedZonesDiffCountries(z2,z)), sum(q,pTransferLimit(z2,z,q,y))/card(q) + vNewTransferCapacity(z2,z,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine")*pAllowHighTransfer))$pIncludeIntercoReserves
   =g= (1+pPlanningReserveMargin(c))*smax((q,d,t), sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y)));

* Planning reserve requirement at the system level
ePlanningReserveReqSystem(y)$(pplanning_reserve_constraints and psystem_reserve_margin)..
   sum(g, vCap(g,y)*pCapacityCredit(g,y)) + vUnmetPlanningReserveSystem(y)
   =g= (1+psystem_reserve_margin)*smax((q,d,t), sum(z, pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y)));


*--- Transfer equations
* Limits flow between zones to existing + expandable transmission capacity
eTransferCapacityLimit(sTopology(z,z2),q,d,t,y)..
   vFlow(z,z2,q,d,t,y) =l= pTransferLimit(z,z2,q,y) + vNewTransferCapacity(z,z2,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine")*pAllowHighTransfer;

* Enforces minimum import flow into a zone when specified
eMinImportRequirement(sTopology(z,z2),q,d,t,y)$pMinImport(z2,z,y)..
   vFlow(z2,z,q,d,t,y) =g= pMinImport(z2,z,y);   

* Cumulative build-out of new transfer capacity over time
eCumulativeTransferExpansion(sTopology(z,z2),y)$pAllowHighTransfer..
   vNewTransferCapacity(z,z2,y) =e=  vNewTransferCapacity(z,z2,y-1) + vBuildTransmission(z,z2,y);

* Ensures symmetry in bidirectional transmission investment
eSymmetricTransferBuild(sTopology(z,z2),y)$pAllowHighTransfer..
   vBuildTransmission(z,z2,y)  =e=  vBuildTransmission(z2,z,y);
   
* Caps total import cost based on annual demand and max share
eMaxAnnualImportShareCost(c,y)$(pallowExports)..
   sum((zcmap(z,c),zext,q,d,t), vYearlyImportExternal(z,zext,q,d,t,y)*pHours(q,d,t)) =l=
   sum((zcmap(z,c),q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))*pMaxExchangeShare(y,c);

* Caps total export value based on annual demand and max share
eMaxAnnualExportShareRevenue(c,y)$(pallowExports)..
   sum((zcmap(z,c),zext,q,d,t), vYearlyExportExternal(z,zext,q,d,t,y)*pHours(q,d,t)) =l=
   sum((zcmap(z,c),q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))*pMaxExchangeShare(y,c);

* Limits hourly import cost as a share of hourly demand
eMaxHourlyImportShareCost(c,q,d,t,y)$(pMaxImport<1 and pallowExports)..
   sum((zcmap(z,c), zext), vYearlyImportExternal(z,zext,q,d,t,y))  =l= sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pMaxImport * pEnergyEfficiencyFactor(z,y));

* Limits hourly export value as a share of hourly demand
eMaxHourlyExportShareRevenue(c,q,d,t,y)$(pMaxExport<1 and pallowExports)..
   sum((zcmap(z,c), zext),vYearlyExportExternal(z,zext,q,d,t,y)) =l= sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pMaxExport * pEnergyEfficiencyFactor(z,y));

* Caps import volume from an external zone to internal zone
eExternalImportLimit(z,zext,q,d,t,y)$pallowExports..
   vYearlyImportExternal(z,zext,q,d,t,y)=l= pExtTransferLimitIn(z,zext,q,y);

* Caps export volume from internal zone to an external zone
eExternalExportLimit(z,zext,q,d,t,y)$pallowExports..
   vYearlyExportExternal(z,zext,q,d,t,y)=l= pExtTransferLimitOut(z,zext,q,y);

*--- Storage-specific equations
* Limits state of charge (SOC) by capacit
eSOCUpperBound(st,q,d,t,y)$pincludeStorage..
   vStorage(st,q,d,t,y) =l= vCapStor(st,y);

* Prevents storage being used to meet reserves only
eStorageCapMinConstraint(st,q,d,t,y)$pincludeStorage..
   vCapStor(st,y) =g= vCap(st,y);

* Charging power ≤ power capacity
eChargeCapacityLimit(st,q,d,t,y)$pincludeStorage..
   vStorInj(st,q,d,t,y) =l= vCap(st,y);

* Charge cap from PV-following storage logic
eChargeLimitWithPVProfile(stp,q,d,t,y)$pincludeStorage..
   vStorInj(stp,q,d,t,y) =l= sum(gsmap(so,stp), vCap(so,y)*pStoPVProfile(so,q,d,t));

* Max rate of charge decrease (ramp-down)
eChargeRampDownLimit(st,q,d,t,y)$(not sFirstHour(t) and pincludeStorage and pramp_constraints)..
   vStorInj(st,q,d,t-1,y) - vStorInj(st,q,d,t,y) =l= pGenData(st,'RampDnRate')*vCap(st,y);

* Max rate of charge increase (ramp-up)
eChargeRampUpLimit(st,q,d,t,y)$(not sFirstHour(t) and pincludeStorage and pramp_constraints)..
   vStorInj(st,q,d,t,y) - vStorInj(st,q,d,t-1,y) =l= pGenData(st,'RampUpRate')*vCap(st,y);

* Defines net charge as output - input
eNetChargeBalance(st,q,d,t,y)$pincludeStorage..
   vStorNet(st,q,d,t,y) =e= sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y)) - vStorInj(st,q,d,t,y);

* SOC dynamics between time steps
eStateOfChargeUpdate(st,q,d,t,y)$(not sFirstHour(t) and pincludeStorage)..
   vStorage(st,q,d,t,y) =e= pStorData(st,"Efficiency")*vStorInj(st,q,d,t,y) - sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y)) + vStorage(st,q,d,t-1,y);

* SOC at first hour (no past state)
eStateOfChargeInit(st,q,d,sFirstHour(t),y)$pincludeStorage..
   vStorage(st,q,d,t,y) =e= pStorData(st,"Efficiency")*vStorInj(st,q,d,t,y) - sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y));

* Ensures SOC level can cover spinning reserve
eSOCSupportsReserve(st,q,d,t,y)$pincludeStorage..
   vSpinningReserve(st,q,d,t,y) =l= vStorage(st,q,d,t,y);

* Ensures that the state of charge at the end of the representative day equals the initial state
* This avoids artificial energy gains/losses over the daily cycle
* eSOCCycleClosure(st,q,d,sLastHour(t),y)$(pincludeStorage)..
*   vStorage(st,q,d,t,y) =e= vStorage(st,q,d,t-23,y) - (pStorData(st,"efficiency") * vStorInj(st,q,d,t-23,y) - sum(gfmap(st,f), vPwrOut(st,f,q,d,t-23,y)));

* Ensures energy conservation over the full representative day: total input × efficiency = total output
* eDailyStorageEnergyBalance(st,q,d,y)$(pincludeStorage)..
*   pStorData(st,"efficiency") * sum(t, vStorInj(st,q,d,t,y)) =e= sum((gfmap(st,f),t), vPwrOut(st,f,q,d,t,y));

*--- CSP-specific equations

* Limits CSP storage level to installed storage capacity.
eCSPStorageCapacityLimit(cs,q,d,t,y)$pincludeCSP..
   vStorage(cs,q,d,t,y) =l= vCapStor(cs,y);

* Limits CSP storage injection to thermal energy output * efficiency. Prevents unrealistic injection behavior.
eCSPStorageInjectionLimit(cs,q,d,t,y)$pincludeCSP..
   vStorInj(cs,q,d,t,y) =l= vThermalOut(cs,q,d,t,y)*pCSPData(cs,"Thermal Field","Efficiency");

* Prevents CSP storage injection exceeding installed capacity.
eCSPStorageInjectionCap(cs,q,d,t,y)$pincludeCSP..
   vStorInj(cs,q,d,t,y) =l= vCapStor(cs,y);

*Limits thermal energy output to installed thermal field capacity * hourly solar profile.
eCSPThermalOutputLimit(cs,q,d,t,y)$pincludeCSP..
   vThermalOut(cs,q,d,t,y) =l= vCapTherm(cs,y)*pCSPProfile(cs,q,d,t);

* Balances CSP thermal output, storage in/out, and generator dispatch.
eCSPPowerBalance(cs,q,d,t,y)$pincludeCSP..
   vThermalOut(cs,q,d,t,y)*pCSPData(cs,"Thermal Field","Efficiency")
 - vStorInj(cs,q,d,t,y) + vStorOut(cs,q,d,t,y)*pCSPData(cs,"Storage","Efficiency")
   =e= sum(gfmap(cs,f), vPwrOut(cs,f,q,d,t,y));

* Tracks CSP storage state of charge across time (except for first hour).
eCSPStorageEnergyBalance(cs,q,d,t,y)$(not sFirstHour(t) and pincludeCSP)..
   vStorage(cs,q,d,t,y) =e= vStorage(cs,q,d,t-1,y) + vStorInj(cs,q,d,t,y) - vStorOut(cs,q,d,t,y);

* Initializes CSP storage balance for the first hour of the day.
eCSPStorageInitialBalance(cs,q,d,sFirstHour(t),y)$pincludeCSP..
   vStorage(cs,q,d,t,y) =e= vStorInj(cs,q,d,t,y) - vStorOut(cs,q,d,t,y);

*Equation needed in dispatch mode but not for capacity expansion with representative days
*eStorageCSPBal2(cs,q,d,sFirstHour(t),y)$(not sFirstDay(d) and pincludeCSP)..
*   vStorage(cs,q,d,t,y) =e= vStorInj(cs,q,d,t,y) - vStorOut(cs,q,d,t,y) + vStorage(cs,q,d-1,sLastHour,y);

*--- Energy (storage) capacity limits

* Limits total installed storage capacity to predefined technical data from pStorData and CSP-related capacity from pCSPData. Only applies if storage is included (pincludeStorage).
eCapacityStorLimit(g,y)$pincludeStorage..
   vCapStor(g,y) =l= pStorData(g,"CapacityMWh") + pCSPData(g,"Storage","CapacityMWh");

* Sets initial year’s storage capacity equal to existing capacity (if online) plus new builds minus retirements.
eCapStorBalance(g, sStartYear(y))..
    vCapStor(g,y) =e= pStorData(g,"CapacityMWh")$(eg(g) and (pGenData(g,"StYr") <= sStartYear.val)) + vBuildStor(g,y) - vRetireStor(g,y);

* Tracks annual storage capacity changes for existing generators (EGs): previous year’s capacity + builds − retirements.
eCapStorAnnualUpdateEG(eg,y)$(not sStartYear(y) and pincludeStorage)..
   vCapStor(eg,y) =e= vCapStor(eg,y-1) + vBuildStor(eg,y) - vRetireStor(eg,y);

* Tracks annual storage capacity for new generators (NGs): previous year’s capacity + builds. Assumes no retirement.
eCapStorAnnualUpdateNG(ng,y)$(not sStartYear(y) and pincludeStorage)..
   vCapStor(ng,y) =e= vCapStor(ng,y-1) + vBuildStor(ng,y);

* Sets initial capacity for new generators at start year equal to build amount. Applies only in the first year.
eCapStorInitialNG(ng,sStartYear(y))$pincludeStorage..
   vCapStor(ng,y) =e= vBuildStor(ng,y);

eBuildStorNew(eg)$((pGenData(eg,"StYr") > sStartYear.val) and pincludeStorage)..
   sum(y, vBuildStor(eg,y)) =l= pStorData(eg,"CapacityMWh");
   
*--- Thermal elements (csp solar field) capacity limits
eCapacityThermLimit(g,y)$pincludeCSP..
   vCapTherm(g,y) =l= pCSPData(g,"Thermal Field","CapacityMWh");

eCapThermBalance1(eg,y)$(not sStartYear(y) and pincludeCSP)..
   vCapTherm(eg,y) =e= vCapTherm(eg,y-1) + vBuildTherm(eg,y) - vRetireTherm(eg,y);

eCapThermBalance2(ng,y)$(not sStartYear(y) and pincludeCSP)..
   vCapTherm(ng,y) =e= vCapTherm(ng,y-1) + vBuildTherm(ng,y);

eCapThermBalance3(ng,sStartYear(y))$pincludeCSP..
   vCapTherm(ng,y) =e= vBuildTherm(ng,y);

eBuildThermNew(eg)$((pGenData(eg,"StYr") > sStartYear.val) and pincludeCSP)..
   sum(y, vBuildTherm(eg,y)) =l= pCSPData(eg,"Thermal Field","CapacityMWh");

*---  Calculate capex for generators with reducing capex                                                                         ;

eCapitalConstraint$pcapital_constraints..
   sum(y, pRR(y)*pWeightYear(y)*sum(ng, pCRF(ng)*vCap(ng,y)*pGenData(ng,"Capex"))) =l= pMaxCapital*1e3;
   
*--- Emissions related equations

eZonalEmissions(z,y)..
   vZonalEmissions(z,y) =e=
   sum((gzmap(g,z),gfmap(g,f),q,d,t), vPwrOut(g,f,q,d,t,y)*pHeatRate(g,f)*pFuelCarbonContent(f)*pHours(q,d,t));

eEmissionsCountry(c,y)$pzonal_co2_constraints..
   sum(zcmap(z,c), vZonalEmissions(z,y))-vYearlyCO2backstop(c,y)=l= pEmissionsCountry(c,y);

eTotalEmissions(y)..
    sum(z, vZonalEmissions(z,y))=e= vTotalEmissions(y);
    
eTotalEmissionsConstraint(y)$pSystem_CO2_constraints..
    vTotalEmissions(y)-vYearlySysCO2backstop(y) =l= pEmissionsTotal(y);
   
*eSimultComDecom(eg,ng,y)$( pIncludeDecomCom and mapGG(eg,ng))..          vBuild(ng,y) -  vRetire(eg,y) =l= 0;


Model PA /
   eNPVCost
   eYearlyUnmetPlanningReserveCostSystem
   eYearlyUnmetSpinningReserveCostSystem
   eYearlyCO2BackstopCostsSystem
   eYearlyTotalCost
   eAnnualizedCapexUpdate
   eAnnualizedCapexInit
   eTotalAnnualizedGenCapex
   eTotalAnnualizedCapex
   eYearlyFOMCost
   eYearlyVariableCost
   eYearlyFuelCost
   eYearlyVOMCost
   eYearlySpinningReserveCost
   eYearlyUSECost
   eYearlyUnmetReserveCostCountry
   eYearlImportExternalCost
   eYearlyExportExternalCost
   eYearlyUnmetSpinningReserveCostCountry
   eYearlyUnmetPlanningReserveCostCountry
   eYearlyCarbonCost
   eYearlyCurtailmentCost
   eYearlyCO2BackstopCostCountry
   eYearlySurplusCost

   eDemSupply
   eDefineSupply
   eInitialCapacity
   eCapacityEvolutionExist
   eCapacityEvolutionNew
   eInitialBuildLimit

   eMinGenRE
   eMaxCF
   eMinGen
   
   eFuel
   eRampUpLimit
   eRampDnLimit
   eSpinningReserveLim
   eSpinningReserveLimVRE
   eJointResCap
   eSpinningReserveReqCountry
   eSpinningReserveReqSystem
   ePlanningReserveReqCountry
   ePlanningReserveReqSystem
   
   eVREProfile
   eFuelLimit
   eCapitalConstraint
   eZonalEmissions
   eEmissionsCountry
   eTotalEmissions
   eTotalEmissionsConstraint

   
   eBuiltCap
   eRetireCap
   
   eTransferCapacityLimit
   eMinImportRequirement
   eAnnualizedTransmissionCapex
   eCumulativeTransferExpansion
   eSymmetricTransferBuild
   eMaxAnnualImportShareCost
   eMaxAnnualExportShareRevenue  
   eMaxHourlyImportShareCost
   eMaxHourlyExportShareRevenue
   eYearlyExternalTradeCost
   eExternalImportLimit
   eExternalExportLimit   
   
   eSOCUpperBound
   eStorageCapMinConstraint
   eChargeCapacityLimit
   eChargeLimitWithPVProfile
   eChargeRampDownLimit
   eChargeRampUpLimit
   eNetChargeBalance
   eStateOfChargeUpdate
   eStateOfChargeInit

  
   eSOCSupportsReserve
   
   eCSPStorageCapacityLimit
   eCSPStorageInjectionLimit
   eCSPStorageInjectionCap
   eCSPThermalOutputLimit
   eCSPPowerBalance
  
   eCSPStorageEnergyBalance
   eCSPStorageInitialBalance
   
   eCapacityStorLimit
   eCapStorBalance
   eCapStorAnnualUpdateEG
   eCapStorAnnualUpdateNG
   eCapStorInitialNG
   eBuildStorNew
   
   eCapacityThermLimit
   eCapThermBalance1
   eCapThermBalance2
   eCapThermBalance3
   eBuildThermNew
   

   vPwrOut(sPwrOut)
   vYearlyExportExternal(sExportPrice)
   vYearlyImportExternal(sImportPrice)
   vNewTransferCapacity(sAdditionalTransfer)
   vFlow(sFlow)
   vSpinningReserve(sSpinningReserve)
   
  
/;
