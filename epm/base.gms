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



* -------------------------------------------------------------
* Generator classification sets
* -------------------------------------------------------------
Sets
   eg(g)                  'Existing generator fleet'
   ng(g)                  'New-build candidates'
   commtransmission(z,z2) 'Committed transmission corridors'
   cs(g)                  'Concentrated solar power units'
   so(g)                  'PV plants paired with storage'
   stp(g)                 'Dedicated storage blocks for PV plants'
   st(g)                  'All storage technologies (incl. standalone)'
   dc(g)                  'Candidates with CAPEX trajectory data'
   ndc(g)                 'Candidates without CAPEX trajectory data'
   vre(g)                 'Variable renewable technologies'
   re(g)                  'Renewable technologies (broader than VRE)'
   RampRate(g)            'Generators subject to ramp constraints (filters out inflexible units)'
   VRE_noROR(g)           'VRE assets excluding run-of-river (used for spinning reserve sizing)'
;


* -------------------------------------------------------------
* Mapping sets (generator-to-fuel/zone, network structure)
* -------------------------------------------------------------
Sets
   gfmap(g,f)             'Generator-to-fuel association'
   gzmap(g,z)             'Generator-to-zone association'
   zfmap(z,f)             'Fuel availability by zone'
   gsmap(g2,g)            'Storage-to-generator linkage'
   sTopology(z,z2)        'Internal network adjacency'
   sMapConnectedZonesDiffCountries(z,z2) 'Cross-country connections subset (topology filter)'
;


* -------------------------------------------------------------
* Status and modeltype-control sets
* -------------------------------------------------------------
Sets
   gstatus  'Generator status lookup'           / Existing, Candidate, Committed /
   tstatus  'Transmission project status'      / Candidate, Committed /
   mipline  'modeltype option line identifiers'
   mipopt(mipline<) 'MIP modeltype option key-value pairs' / system.empty /
;


* -------------------------------------------------------------
* Implicit domains (restrict variables to meaningful tuples)
* -------------------------------------------------------------
Sets
   sPwrOut(g,f,q,d,t,y)        'Active generator-fuel-time combinations'
   sExportPrice(z,zext,q,d,t,y) 'Valid export price records (external zones)'
   sImportPrice(z,zext,q,d,t,y) 'Valid import price records (external zones)'
   sAdditionalTransfer(z,z2,y)  'Transmission build decision domain'
   sFlow(z,z2,q,d,t,y)         'Feasible flow pairs'
   sSpinningReserve(g,q,d,t,y) 'Generators able to provide spinning reserve'
;

Singleton sets
   sStartYear(y)
   sFirstHour(t)
   sLastHour(t)
   sFirstDay(d)
;


* Additional parameters for results and reporting
* -------------------------------------------------------------
* Global flags and shared parameters for reporting/constraints
* -------------------------------------------------------------
Parameter
   pAllHours(q,d,y,t)           'All operating hours (used for peak tracking)'
   pFindSysPeak(y)              'Yearly system peak indicator'
   pSeasonalReporting           'Enable seasonal report outputs'
   fEnableInternalExchange      'Allow internal zone trading'
   fRemoveInternalTransferLimit 'Override internal transfer limits'
   fAllowTransferExpansion      'Permit expansion of internal transmission (set elsewhere too; redundant flag)'
   pDR                          'Discount rate applied in objective'
   fEnableCapexTrajectoryH2     'Toggle CAPEX trajectory for hydrogen assets'
   pfEnableEnergyEfficiency     'Enable demand-side efficiency adjustments'
   pfApplySystemCo2Constraint   'Apply system-wide CO₂ constraint'
   pExtTransferLimitIn(z,zext,q,y)  'Exogenous import limit from external zones'
   pExtTransferLimitOut(z,zext,q,y) 'Exogenous export limit to external zones'
   psVREForecastErrorPct        'Forecast error percentage for VRE (used in reserves)'
;

* Additional structure sets (technology/mapping)
Set 
   gprimf(g,f)            'Primary fuel used by each generator'
   gtechmap(g,tech)       'Generator-to-technology linkage'
   gstatusmap(g,gstatus)  'Generator status lookup (existing/candidate/committed)'
   tstatusmap(z,z2,tstatus) 'Transmission status lookup'
   Zd(z)                  'Demand-serving zones'
   Zt(z)                  'Zone type classification (often redundant with Zd)'
   stg(g)                 'Storage generators (grid-scale)'
   ror(g)                 'Run-of-river assets (subset of renewables)'
;


Parameters
   pCostOfCurtailment          'Penalty on curtailed VRE (check usage – may be redundant)'
   pCostOfCO2backstop          'CO₂ backstop cost ($/ton)'
   pWeightYear(y)              'Inter-year weighting factor'

* Generator data
   pGenData(g,pGenDataInputHeader) 'Generator characteristics (capex, heat rate, etc.)'
   pAvailability(g,q)             'Seasonal availability factors'
   pCapexTrajectories(g,y)        'CAPEX trajectory multipliers'

* External exchange data
   fEnableExternalExchange          'Permit external exchange'
   pMaxAnnualExternalTradeShare(y,c) 'Annual import/export share cap by country'
   pTradePrice(zext,q,d,y,t)      'Border price for external trades'
   sMaxHourlyImportExternalShare  'Hourly import cap (share of demand)'
   sMaxHourlyExportExternalShare  'Hourly export cap (share of demand)'
   pExtTransferLimit(z,zext,q,*,y) 'Full external transfer limit tensor'
   pExtTransferLimitIn(z,zext,q,y) 'Import limit (duplicate of earlier parameter)'
   pExtTransferLimitOut(z,zext,q,y)'Export limit (duplicate of earlier parameter)'

* Demand inputs
   pDemandData(z,q,d,y,t)         'Demand time series (seasonal, daily, hourly)'

* Renewable/storage inputs
   pCSPProfile(g,q,d,t)           'CSP solar profile (pu)'
   pStoPVProfile(g,q,d,t)         'PV-with-storage profile (pu)'
   pStorData(g,pStoreDataHeader)  'Storage technical data'

* Reserve and policy inputs
   pCapacityCredit(g,y)           'Capacity credit for planning reserves'
   psVREForecastErrorPct          'Reserve add-on for VRE (note: duplicate naming with parameter above)'
   pCarbonPrice(y)                'Carbon price trajectory'
   pFuelCarbonContent(f)          'Fuel carbon intensity (tCO₂/MMBtu)'

* Economic parameters
   pRR(y)                         'Discount factor (accumulated)'
   pWACC                          'Weighted average cost of capital'
   pCRF(g)                        'Capital recovery factor (generator)'
   pCRFsst(g)                     'Capital recovery factor (storage)'
   pCRFcst(g)                     'Capital recovery factor (CSP storage)'
   pCRFcth(g)                     'Capital recovery factor (CSP thermal field)'
   pVoLL                          'Value of lost load'
   pPlanningReserveVoLL           'Penalty for planning reserve shortfall'
   pSpinningReserveVoLL           'Penalty for spinning reserve shortfall'
   ssMaxCapitalInvestmentInvestment 'Total capital limit (USD) – consider renaming for clarity'
   pVarCost(g,f,y)                'Variable cost (fuel + VOM)'
   pFuelCost(g,f,y)               'Fuel price'
   pVOMCost(g,f,y)                'Variable O&M component'

* Control toggles
   fApplyRampConstraint           'Enable ramping constraints'
   fApplyFuelConstraint           'Enable fuel availability limits'
   fApplyCapitalConstraint        'Enable total capital constraint'
   fApplyMinGenerationConstraint  'Minimum generation constraint toggle'
   fEnableCSP                     'Enable CSP features'
   fEnableStorage                 'Enable storage operations'
   pIncludeCarbon                 'Apply carbon cost'
   pSurplusPenalty                'Penalty on surplus energy'
   pMinRE                         'Minimum RE share target'
   pMinsRenewableTargetYear       'Year from which RE target applies'
   fApplyCountryCo2Constraint     'Country-level CO₂ constraint toggle'
   pfApplySystemCo2Constraint     'System-level CO₂ constraint toggle (duplicate flag name earlier)'
   fApplyCountrySpinReserveConstraint 'Enable country spinning reserve constraint'
   pfApplySystemSpinReserveConstraint 'Enable system spinning reserve constraint'
   fApplyPlanningReserveConstraint 'Enable planning reserve constraint'
   sIntercoReserveContributionPct 'Share of interconnection capacity counted toward reserves'
   fCountIntercoForReserves       'Include interconnections in planning reserve assessment'
   sReserveMarginPct              'Planning reserve margin target'
   pHeatrate(g,f)                 'Generator heat rate'
;

* -------------------------------------------------------------
* Positive decision variables (continuous)
* -------------------------------------------------------------
Positive Variables
* Dispatch and demand balance
   vPwrOut(g,f,q,d,t,y)      'Generation output (MW)'
   vUSE(z,q,d,t,y)           'Unserved demand (MW)'
   vSurplus(z,q,d,t,y)       'Surplus generation (MW)'
   vYearlySurplus(z,y)       'Annual surplus energy (aggregated)'

* Capacity evolution
   vCap(g,y)                 'Installed capacity (MW)'
   vBuild(g,y)               'New build decision (MW)'
   vRetire(g,y)              'Retirement decision (MW)'

* Network flows and external trade
   vFlow(z,z2,q,d,t,y)       'Internal power flow (MW)'
   vYearlyImportExternal(z,zext,q,d,t,y)  'External imports (MW)'
   vYearlyExportExternal(z,zext,q,d,t,y)  'External exports (MW)'

* Fuel consumption
   vFuel(z,f,y)              'Annual fuel consumption (MMBtu)'

* Annualized CAPEX trackers (note potential overlap among these three)
   vAnnCapexGenTraj(g,y)     'Annualized CAPEX with trajectory (verify alongside vAnnGenCapex)'
   vAnnGenCapex(g,y)         'Annualized CAPEX for builds in year y'
   vAnnCapex(z,y)            'Annualized CAPEX aggregated by zone'

* Storage and CSP state variables
   vThermalOut(g,q,d,t,y)    'CSP thermal output (MW)'
   vCapStor(g,y)             'Installed storage energy capacity (MWh)'
   vCapTherm(g,y)            'Installed CSP thermal capacity (MW)'
   vStorage(g,q,d,t,y)       'State of charge (MW-equivalent)'
   vStorInj(g,q,d,t,y)       'Charging power (MW)'
   vStorOut(g,q,d,t,y)       'Discharging power (MW)'
   vBuildStor(g,y)           'New storage build (MWh)'
   vRetireStor(g,y)          'Storage retirement (MWh)'
   vBuildTherm(g,y)          'CSP thermal field build (MW)'
   vRetireTherm(g,y)         'CSP thermal field retirement (MW)'

* Reserve shortfalls
   vSpinningReserve(g,q,d,t,y)           'Spinning reserve provision (MW)'
   vUnmetPlanningReserveCountry(c,y)     'Country planning reserve shortfall'
   vUnmetPlanningReserveSystem(y)        'System planning reserve shortfall'
   vUnmetSpinningReserveCountry(c,q,d,t,y) 'Country spinning reserve shortfall'
   vUnmetSpinningReserveSystem(q,d,t,y)  'System spinning reserve shortfall'

* Emissions accounting
   vZonalEmissions(z,y)      'Zonal emissions intensity (tCO₂/MWh)'
   vTotalEmissions(y)        'System emissions (tCO₂)'
   vYearlyCO2backstop(c,y)   'Country CO₂ backstop usage (t)'
   vYearlySysCO2backstop(y)  'System CO₂ backstop usage (t)'

* Network expansion & curtailment
   vNewTransmissionLine(z,z2,y) 'New transmission capacity (MW)'
   vAnnualizedTransmissionCapex(z,y) 'Annualized transmission CAPEX (USD)'
   vYearlyCurtailmentCost(z,y)  'Annual curtailment penalty (USD)'
   vCurtailedVRE(z,g,q,d,t,y)   'Curtailment of VRE (MW)'
;

* -------------------------------------------------------------
* Free variables (objective & reporting)
* -------------------------------------------------------------
Free Variable
   vNPVCost                       'Net present value of total system cost'
   vStorNet(g,q,d,t,y)            'Net storage power (discharge - charge)'
   vYearlyTotalCost(c,y)          'Annual total cost by country'
   vYearlyVariableCost(z,y)       'Annual variable cost by zone'
   vYearlyUSECost(z,y)            'Cost of unserved energy'
   vYearlyExternalTradeCost(z,y)  'Net external trade cost'
   vYearlySpinningReserveCost(z,y)'Spinning reserve penalty'
   vYearlyUnmetReserveCostCountry(c,y) 'Country-level reserve shortfall cost'
   vYearlyCarbonCost(z,y)         'Zonal carbon cost'
   vYearlyCO2BackstopCostCountry(c,y) 'Country CO₂ backstop cost'
   vYearlyUnmetPlanningReserveCostSystem(y) 'System planning reserve cost'
   vYearlyUnmetSpinningReserveCostSystem(y) 'System spinning reserve cost'
   vYearlyCO2BackstopCostSystem(y) 'System CO₂ backstop cost'
   vYearlyUnmetPlanningReserveCostCountry(c,y) 'Country planning reserve cost'
   vYearlyUnmetSpinningReserveCostCountry(c,y) 'Country spinning reserve cost'
   vYearlyFOMCost(z,y)            'Zone fixed O&M cost'
   vYearlyFuelCost(z,y)           'Zone fuel cost'
   vYearlyVOMCost(z,y)            'Zone variable O&M cost'
   vYearlyImportExternalCost(z,y) 'Cost of imports from external zones'
   vYearlyExportExternalCost(z,y) 'Revenue from exports to external zones'
   vSupply(z,q,d,t,y)             'Total supply meeting demand (balance variable)'
;

* -------------------------------------------------------------
* Integer decision variables
* -------------------------------------------------------------
Integer variable
   vBuildTransmission(z,z2,y) 'Integer builds for transmission'
   vBuiltCapVar(g,y)          'Integer build decision (unit commitment for discrete capacity)'
   vRetireCapVar(g,y)         'Integer retirement decision'
;


* -------------------------------------------------------------
* Core equations (objective, balances, constraints)
* -------------------------------------------------------------
Equations

* ------------------------------
* Objective and cost accounting
* Summarizes NPV and annual cost components across system and countries.
* ------------------------------
   eNPVCost                        'Objective function – discounted system cost'
   eYearlyTotalCost(c,y)           'Aggregate annual cost at country level'
   eYearlyVOMCost(z,y)             'Variable O&M spend by zone'
   eYearlySpinningReserveCost(z,y) 'Cost of holding spinning reserves'
   eYearlyUSECost(z,y)             'Penalty on unserved demand'
   eYearlyExternalTradeCost(z,y)   'Net external trade cost (imports minus exports)'
   eYearlyUnmetReserveCostCountry(c,y) 'Combined reserve shortfall penalty'
   eYearlyCarbonCost(z,y)          'Carbon-price cost for zone emissions'
   eYearlyFOMCost(z,y)                'Total yearly FOM cost'
   eYearlyCO2BackstopCostCountry(c,y) 'Cost of CO₂ backstop'
   eYearlyVariableCost(z,y)        'Sum of fuel and VOM costs'
   eYearlyFuelCost(z,y)            'Fuel expenditure by zone'
   eYearlyImportExternalCost(z,y)  'Cost of imports from external zones'
   eYearlyExportExternalCost(z,y)  'Revenue from power exported externally'
   eYearlyUnmetSpinningReserveCostCountry(c,y) 'Country spinning reserve shortfall cost'
   eYearlyUnmetPlanningReserveCostCountry(c,y) 'Country planning reserve shortfall cost'
   eYearlyUnmetPlanningReserveCostSystem(y) 'System planning reserve shortfall cost'
   eYearlyUnmetSpinningReserveCostSystem(y) 'System spinning reserve shortfall cost'
   eYearlyCO2BackstopCostsSystem(y) 'System CO₂ backstop expenditure'

* ------------------------------
* Annualized CAPEX tracking
* Handles annualization of generator CAPEX for trajectory and flat-cost assets.
* ------------------------------
   eTotalAnnualizedCapex(z,y)     'Accumulate annualized CAPEX by zone'
   eAnnualizedCapexInit(g,y)      'Initial annualized CAPEX for trajectory assets'
   eAnnualizedCapexUpdate(g,y)    'Recursive update of annualized CAPEX'
   eTotalAnnualizedGenCapex(g,y)  'Annualized CAPEX per generator'

* ------------------------------
* Demand balance and supply definition
* Ensures zonal demand is met by generation, flows, storage, and trade.
* ------------------------------
   eDemSupply(z,q,d,t,y)          'Demand equals total supply'
   eDefineSupply(z,q,d,t,y)       'Supply components (gen/flows/storage)'
   eMaxHourlyExportShareEnergy(c,q,d,t,y) 'Limit hourly exports as demand share'
   eMaxHourlyImportShareEnergy(c,q,d,t,y) 'Limit hourly imports as demand share'

* ------------------------------
* Capacity evolution
* Updates installed capacity over time for existing and new units, including discrete builds.
* ------------------------------
   eInitialCapacity(g,y)          'Initial capacity accounting'
   eCapacityEvolutionExist(g,y)   'Existing unit capacity recursion'
   eCapacityEvolutionNew(g,y)     'New unit capacity recursion'
   eInitialBuildLimit(g)          'Initial build limit for existing assets'
   eBuiltCap(g,y)                 'Link integer build variable to capacity'
   eRetireCap(g,y)                'Link integer retirement variable to capacity'

* ------------------------------
* Generation operating limits
* Applies minimum output, availability, and VRE profile constraints.
* ------------------------------
   eMinGenRE(c,y)                 'Country minimum renewable generation'
   eMaxCF(g,q,y)                  'Capacity factor ceiling'
   eMinGen(g,q,d,t,y)             'Minimum dispatch requirement'

* ------------------------------
* Fuel and resource limits
* Tracks fuel usage and enforces optional availability caps by country.
* ------------------------------
   eFuel(z,f,y)                   'Fuel consumption accounting'
   eFuelLimit(c,f,y)              'Fuel availability constraint'

* ------------------------------
* Ramp and reserve constraints
* Applies ramp-rate limits and spinning/planning reserve requirements.
* ------------------------------
   eRampUpLimit(g,q,d,t,y)        'Ramp-up constraint'
   eRampDnLimit(g,q,d,t,y)        'Ramp-down constraint'
   eSpinningReserveLim(g,q,d,t,y) 'Reserve capability vs capacity'
   eSpinningReserveLimVRE(g,f,q,d,t,y) 'Reserve limit adjusted for VRE profile'
   eJointResCap(g,q,d,t,y)        'Joint dispatch-reserve envelope'
   eSpinningReserveReqCountry(c,q,d,t,y) 'Country spinning reserve requirement'
   eSpinningReserveReqSystem(q,d,t,y) 'System spinning reserve requirement'
   ePlanningReserveReqSystem(y)   'System planning reserve requirement'
   ePlanningReserveReqCountry(c,y) 'Country planning reserve requirement'

* ------------------------------
* Transmission and trade constraints
* Bounds internal transfers and external exchanges (hourly and annual).
* ------------------------------
   eTransferCapacityLimit(z,z2,q,d,t,y) 'Transmission capacity limit'
   eMinImportRequirement(z2,z,q,d,t,y) 'Minimum flow requirement if specified'
   eVREProfile(g,f,z,q,d,t,y)      'Follow VRE production profile with slack'
   eMaxAnnualImportShareEnergy(c,y) 'Annual import share cap'
   eMaxAnnualExportShareEnergy(c,y) 'Annual export share cap'
   eYearlySurplusCost(z,y)         'Penalty on surplus energy'
   eCumulativeTransferExpansion(z,z2,y) 'Cumulative new transfer capacity'
   eSymmetricTransferBuild(z,z2,y) 'Symmetric new build requirement'
   eAnnualizedTransmissionCapex (z,y) 'Annualized transmission CAPEX'
   eExternalImportLimit(z,zext,q,d,t,y) 'Import limit from external zone (MW)'
   eExternalExportLimit(z,zext,q,d,t,y) 'Export limit to external zone (MW)'

* ------------------------------
* Capital & emissions
* Enforces capital budget and CO₂ caps with backstop slack variables.
* ------------------------------
   eCapitalConstraint              'Enforce overall capital budget (legacy name)'
   eZonalEmissions(z,y)            'Compute zonal CO₂ emissions'
   eEmissionsCountry(c,y)          'Country CO₂ cap with backstop slack'
   eTotalEmissions(y)              'Aggregate emissions across zones'
   eTotalEmissionsConstraint(y)    'System-wide CO₂ cap with backstop'

* ------------------------------
* Storage operations
* Controls storage charging, SOC evolution, and reserve support obligations.
* ------------------------------
   eChargeRampDownLimit(g,q,d,t,y) 'Limit decrease in storage charging'
   eChargeRampUpLimit(g,q,d,t,y)   'Limit increase in storage charging'
   eYearlyCurtailmentCost(z,y)     'Curtailment penalty (zone)'
   eStateOfChargeUpdate(g,q,d,t,y) 'Storage SOC recurrence'
   eStateOfChargeInit(g,q,d,t,y)   'SOC initial condition per representative day'
*  eSOCCycleClosure(g,q,d,t,y)
*  eDailyStorageEnergyBalance(g,q,d,y)
   eSOCSupportsReserve(g,q,d,t,y)  'Ensure SOC can cover reserve commitment'
   eChargeCapacityLimit(g,q,d,t,y) 'Charging limited by power capacity'
   eChargeLimitWithPVProfile(g,q,d,t,y) 'PV-coupled storage charging limit'
   eNetChargeBalance(g,q,d,t,y)    'Net storage discharge minus charge'
   eSOCUpperBound(g,q,d,t,y)       'State of charge upper bound'
   eStorageCapMinConstraint(g,q,d,t,y) 'Minimum storage energy duration'

* ------------------------------
* CSP-specific and storage capacity evolution
* Governs CSP thermal/storage balances and long-term storage capacity updates.
* ------------------------------
   eCSPStorageCapacityLimit(g,q,d,t,y) 'CSP storage <= installed capacity'
   eCSPStorageInjectionLimit(g,q,d,t,y) 'Limit CSP charging to thermal output'
   eCSPStorageInjectionCap(g,q,d,t,y) 'CSP charging bounded by capacity'
   eCSPThermalOutputLimit(g,q,d,t,y) 'Thermal output limited by field capacity'
   eCSPPowerBalance(g,q,d,t,y)     'Thermal/storage/power balance for CSP'
   eCSPStorageEnergyBalance(g,q,d,t,y) 'CSP storage state-of-charge recursion'
   eCSPStorageInitialBalance(g,q,d,t,y) 'Initial CSP storage state'
   eCapacityStorLimit(g,y)         'Cap storage energy capacity'
   eCapStorBalance(g,y)            'Initial storage capacity balance'
   eCapStorAnnualUpdateEG(g,y)     'Storage capacity recursion (existing)'
   eCapStorAnnualUpdateNG(g,y)     'Storage capacity recursion (new)'
   eCapStorInitialNG(g,y)          'Initial storage for new plants'
   eBuildStorNew(g)                'Limit storage builds for entrants'
   eCapacityThermLimit(g,y)        'CSP thermal capacity cap'
   eCapThermBalance1(g,y)          'Thermal capacity recursion (existing)'
   eCapThermBalance2(g,y)          'Thermal capacity recursion (new)'
   eCapThermBalance3(g,y)          'Initial thermal capacity for new units'
   eBuildThermNew(g)               'Limit CSP thermal builds for entrants'
;


*-------------------------------------------------------------------
* OBJECTIVE FUNCTION
*-------------------------------------------------------------------
* Adding system-level cost of reserves and CO2 backstop to the total cost
eNPVCost..
   vNPVCost =e= sum(y, pRR(y)*pWeightYear(y)*(sum(c, 
                                 vYearlyTotalCost(c,y)) + 
                                 vYearlyUnmetPlanningReserveCostSystem(y) + 
                                 vYearlyUnmetSpinningReserveCostSystem(y) +
                                 vYearlyCO2BackstopCostSystem(y))
                                 );

*-------------------------------------------------------------------
* 1. COST
*-------------------------------------------------------------------
* ------------------------------
* System-level Cost accounting
* ------------------------------

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

* ------------------------------
* Country-level Cost accounting
* ------------------------------

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

* ------------------------------
* Zonal CAPEX accounting
* Aggregates generator-level annualized CAPEX to the zone level.
* ------------------------------
* Annualized CAPEX for all zones in year y
eTotalAnnualizedCapex(z, y)..
   vAnnCapex(z,y) =e= sum(gzmap(g,z), vAnnGenCapex(g,y));

* Annualized CAPEX for all generators in year y
* Separate treatment keeps time-varying CAPEX trajectories (dc) distinct
* from flat-cost assets (ndc) when annualizing capital costs.
* Assets tagged in set dc(g) follow time-varying CAPEX trajectories (stored
* in vAnnCapexGenTraj), while flat-cost assets (ndc(g)) use
* conventional CRF-based annualization. We keep both paths to preserve the
* different depreciation logic without double-counting.
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
eYearlyImportExternalCost(z,y)..
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
eAnnualizedTransmissionCapex(z,y)$(fAllowTransferExpansion and sum(sTopology(z,z2),1))..
   vAnnualizedTransmissionCapex(z,y) =e=
       sum(sTopology(z,z2),
           vNewTransmissionLine(z,z2,y)
         * symmax(pNewTransmission,z,z2,"CostPerLine")
         * 1e6)
     / 2
     * (pWACC / (1 - (1 / ((1 + pWACC) ** sum(sTopology(z,z2), symmax(pNewTransmission,z,z2,"Life"))))));

* ------------------------------
* Demand and supply balance
* Ensures zonal demand matches supply components.
* ------------------------------
eDemSupply(z,q,d,t,y)..
   pDemandData(z,q,d,y,t) * pEnergyEfficiencyFactor(z,y) =e= vSupply(z,q,d,t,y);

eDefineSupply(z,q,d,t,y)..
   vSupply(z,q,d,t,y) =e=
     sum((gzmap(g,z),gfmap(g,f)), vPwrOut(g,f,q,d,t,y))
   - sum(sTopology(z,z2), vFlow(z,z2,q,d,t,y))
   + sum(sTopology(z,z2), vFlow(z2,z,q,d,t,y) * (1 - pLossFactorInternal(z,z2,y)))
   - sum(gzmap(st,z), vStorInj(st,q,d,t,y))$(fEnableStorage)
   + sum(zext, vYearlyImportExternal(z,zext,q,d,t,y))
   - sum(zext, vYearlyExportExternal(z,zext,q,d,t,y))
   + vUSE(z,q,d,t,y)
   - vSurplus(z,q,d,t,y);


* ------------------------------
* Generator capacity equations
* Tracks installed capacity for existing/new units.
* ------------------------------
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

eMinGenRE(c,y)$(pMinRE and y.val >= pMinsRenewableTargetYear)..
   sum((zcmap(z,c),gzmap(RE,z),gfmap(RE,f),q,d,t), vPwrOut(RE,f,q,d,t,y)*pHours(q,d,t)) =g=
   sum((zcmap(z,c),q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))*pMinRE;

eBuiltCap(ng,y)$pGenData(ng,"DescreteCap")..
   vBuild(ng,y) =e= pGenData(ng,"UnitSize")*vBuiltCapVar(ng,y);

eRetireCap(eg,y)$(pGenData(eg,"DescreteCap") and (y.val <= pGenData(eg,"RetrYr")))..
   vRetire(eg,y) =e= pGenData(eg,"UnitSize")*vRetireCapVar(eg,y);


* ------------------------------
* Production equations
* Constrains generator dispatch, ramping, and minimum output.
* ------------------------------
eJointResCap(g,q,d,t,y)..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) + vSpinningReserve(g,q,d,t,y) =l= vCap(g,y)*(1+pGenData(g,"Overloadfactor"));

eMaxCF(g,q,y)..
   sum((gfmap(g,f),d,t), vPwrOut(g,f,q,d,t,y)*pHours(q,d,t)) =l= pAvailability(g,q)*vCap(g,y)*sum((d,t), pHours(q,d,t));

eFuel(zfmap(z,f),y)..
   vFuel(z,f,y) =e= sum((gzmap(g,z),gfmap(g,f),q,d,t), vPwrOut(g,f,q,d,t,y)*pHours(q,d,t)*pHeatRate(g,f));

eFuelLimit(c,f,y)$(fApplyFuelConstraint and pMaxFuelLimit(c,f,y) > 0)..
   sum((zcmap(z,c),zfmap(z,f)), vFuel(z,f,y)) =l= pMaxFuelLimit(c,f,y)*1e6;

eMinGen(g,q,d,t,y)$(fApplyMinGenerationConstraint and pGenData(g,"MinLimitShare") > 0)..
    sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) =g= vCap(g,y)*pGenData(g,"MinLimitShare") ; 

eRampDnLimit(g,q,d,t,y)$(Ramprate(g) and not sFirstHour(t) and fApplyRampConstraint)..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t-1,y)) - sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) =l= vCap(g,y)*pGenData(g,"RampDnRate");

eRampUpLimit(g,q,d,t,y)$(Ramprate(g) and not sFirstHour(t) and fApplyRampConstraint)..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) - sum(gfmap(g,f), vPwrOut(g,f,q,d,t-1,y)) =l= vCap(g,y)*pGenData(g,"RampUpRate");

* Note that we are effectively assuming grid-connected RE generation to be dispatchable. Generally speaking, most RE will be
* dispatched anyway because they have zero cost (i.e., not a different outcome from net load approach, but this allows for
* RE generation to be rejected as well
eVREProfile(gfmap(VRE,f),z,q,d,t,y)$gzmap(VRE,z)..
   vPwrOut(VRE,f,q,d,t,y) + vCurtailedVRE(z,VRE,q,d,t,y) =e= pVREgenProfile(VRE,q,d,t)*vCap(VRE,y);


* ------------------------------
* Reserve equations
* Enforces spinning and planning reserve requirements.
* ------------------------------
* Spinning reserve limit as a share of capacity
eSpinningReserveLim(g,q,d,t,y)$(fApplyCountrySpinReserveConstraint or pfApplySystemSpinReserveConstraint)..
   vSpinningReserve(g,q,d,t,y) =l= vCap(g,y)*pGenData(g,"ResLimShare");
   
* Spinning reserve limit for VRE as a share of capacity adjusted for production profile
eSpinningReserveLimVRE(gfmap(VRE,f),q,d,t,y)$(fApplyCountrySpinReserveConstraint or pfApplySystemSpinReserveConstraint)..
    vSpinningReserve(VRE,q,d,t,y) =l= vCap(VRE,y)*pGenData(VRE,"ResLimShare")* pVREgenProfile(VRE,q,d,t);

* This constraint increases solving time x3
* Reserve constraints include interconnections as reserves too
eSpinningReserveReqCountry(c,q,d,t,y)$fApplyCountrySpinReserveConstraint..
   sum((zcmap(z,c),gzmap(g,z)),vSpinningReserve(g,q,d,t,y))
 + vUnmetSpinningReserveCountry(c,q,d,t,y)
* + sIntercoReserveContributionPct * sum((zcmap(z,c),sMapConnectedZonesDiffCountries(z2,z)), pTransferLimit(z2,z,q,y)
*                                        + vNewTransmissionLine(z2,z,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine")*fAllowTransferExpansion
*                                        - vFlow(z2,z,q,d,t,y))
   =g= pSpinningReserveReqCountry(c,y) + sum((zcmap(z,c),gzmap(VRE_noROR,z),gfmap(VRE_noROR,f)), vPwrOut(VRE_noROR,f,q,d,t,y))*psVREForecastErrorPct;
   
* System spinning reserve requirement
eSpinningReserveReqSystem(q,d,t,y)$pfApplySystemSpinReserveConstraint..
   sum(g, vSpinningReserve(g,q,d,t,y)) + vUnmetSpinningReserveSystem(q,d,t,y) =g= pSpinningReserveReqSystem(y) + sum(gfmap(VRE_noROR,f), vPwrOut(VRE_noROR,f,q,d,t,y))*psVREForecastErrorPct;

* Planning reserve requirement at the country level
ePlanningReserveReqCountry(c,y)$(fApplyPlanningReserveConstraint and pPlanningReserveMargin(c))..
   sum((zcmap(z,c),gzmap(g,z)), vCap(g,y)*pCapacityCredit(g,y))
 + vUnmetPlanningReserveCountry(c,y)
 + (sum((zcmap(z,c),sMapConnectedZonesDiffCountries(z2,z)), sum(q,pTransferLimit(z2,z,q,y))/card(q) + vNewTransmissionLine(z2,z,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine")*fAllowTransferExpansion))$fCountIntercoForReserves
   =g= (1+pPlanningReserveMargin(c))*smax((q,d,t), sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y)));

* Planning reserve requirement at the system level
ePlanningReserveReqSystem(y)$(fApplyPlanningReserveConstraint and sReserveMarginPct)..
   sum(g, vCap(g,y)*pCapacityCredit(g,y)) + vUnmetPlanningReserveSystem(y)
   =g= (1+sReserveMarginPct)*smax((q,d,t), sum(z, pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y)));


* ------------------------------
* Transfer equations
* Limits internal flows and applies trade caps.
* ------------------------------
* Limits flow between zones to existing + expandable transmission capacity
eTransferCapacityLimit(sTopology(z,z2),q,d,t,y)..
   vFlow(z,z2,q,d,t,y) =l= pTransferLimit(z,z2,q,y) + vNewTransmissionLine(z,z2,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine")*fAllowTransferExpansion;

* Enforces minimum import flow into a zone when specified
eMinImportRequirement(sTopology(z,z2),q,d,t,y)$pMinImport(z2,z,y)..
   vFlow(z2,z,q,d,t,y) =g= pMinImport(z2,z,y);   

* Cumulative build-out of new transfer capacity over time
eCumulativeTransferExpansion(sTopology(z,z2),y)$fAllowTransferExpansion..
   vNewTransmissionLine(z,z2,y) =e=  vNewTransmissionLine(z,z2,y-1) + vBuildTransmission(z,z2,y);

* Ensures symmetry in bidirectional transmission investment
eSymmetricTransferBuild(sTopology(z,z2),y)$fAllowTransferExpansion..
   vBuildTransmission(z,z2,y)  =e=  vBuildTransmission(z2,z,y);

* External trade

* Caps total import energy based on annual demand and max external trade share
eMaxAnnualImportShareEnergy(c,y)$fEnableExternalExchange..
   sum((zcmap(z,c),zext,q,d,t), vYearlyImportExternal(z,zext,q,d,t,y)*pHours(q,d,t)) =l=
   sum((zcmap(z,c),q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))*pMaxAnnualExternalTradeShare(y,c);

* Caps total export energy based on annual demand and max external trade share
eMaxAnnualExportShareEnergy(c,y)$fEnableExternalExchange..
   sum((zcmap(z,c),zext,q,d,t), vYearlyExportExternal(z,zext,q,d,t,y)*pHours(q,d,t)) =l=
   sum((zcmap(z,c),q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))*pMaxAnnualExternalTradeShare(y,c);

* Limits hourly import energy as a share of hourly demand
eMaxHourlyImportShareEnergy(c,q,d,t,y)$(sMaxHourlyImportExternalShare<1 and fEnableExternalExchange)..
   sum((zcmap(z,c), zext), vYearlyImportExternal(z,zext,q,d,t,y))  =l= sum(zcmap(z,c), pDemandData(z,q,d,y,t)*sMaxHourlyImportExternalShare * pEnergyEfficiencyFactor(z,y));

* Limits hourly export energy as a share of hourly demand
eMaxHourlyExportShareEnergy(c,q,d,t,y)$(sMaxHourlyExportExternalShare<1 and fEnableExternalExchange)..
   sum((zcmap(z,c), zext),vYearlyExportExternal(z,zext,q,d,t,y)) =l= sum(zcmap(z,c), pDemandData(z,q,d,y,t)*sMaxHourlyExportExternalShare * pEnergyEfficiencyFactor(z,y));

* Caps import volume from an external zone to internal zone
eExternalImportLimit(z,zext,q,d,t,y)$fEnableExternalExchange..
   vYearlyImportExternal(z,zext,q,d,t,y)=l= pExtTransferLimitIn(z,zext,q,y);

* Caps export volume from internal zone to an external zone
eExternalExportLimit(z,zext,q,d,t,y)$fEnableExternalExchange..
   vYearlyExportExternal(z,zext,q,d,t,y)=l= pExtTransferLimitOut(z,zext,q,y);

* ------------------------------
* Storage operations
* Controls storage SOC, charging, and reserve support.
* ------------------------------
* Limits state of charge (SOC) by capacit
eSOCUpperBound(st,q,d,t,y)$fEnableStorage..
   vStorage(st,q,d,t,y) =l= vCapStor(st,y);

* Prevents storage being used to meet reserves only
eStorageCapMinConstraint(st,q,d,t,y)$fEnableStorage..
   vCapStor(st,y) =g= vCap(st,y);

* Charging power ≤ power capacity
eChargeCapacityLimit(st,q,d,t,y)$fEnableStorage..
   vStorInj(st,q,d,t,y) =l= vCap(st,y);

* Charge cap from PV-following storage logic
eChargeLimitWithPVProfile(stp,q,d,t,y)$fEnableStorage..
   vStorInj(stp,q,d,t,y) =l= sum(gsmap(so,stp), vCap(so,y)*pStoPVProfile(so,q,d,t));

* Max rate of charge decrease (ramp-down)
eChargeRampDownLimit(st,q,d,t,y)$(not sFirstHour(t) and fEnableStorage and fApplyRampConstraint)..
   vStorInj(st,q,d,t-1,y) - vStorInj(st,q,d,t,y) =l= pGenData(st,'RampDnRate')*vCap(st,y);

* Max rate of charge increase (ramp-up)
eChargeRampUpLimit(st,q,d,t,y)$(not sFirstHour(t) and fEnableStorage and fApplyRampConstraint)..
   vStorInj(st,q,d,t,y) - vStorInj(st,q,d,t-1,y) =l= pGenData(st,'RampUpRate')*vCap(st,y);

* Defines net charge as output - input
eNetChargeBalance(st,q,d,t,y)$fEnableStorage..
   vStorNet(st,q,d,t,y) =e= sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y)) - vStorInj(st,q,d,t,y);

* SOC dynamics between time steps
eStateOfChargeUpdate(st,q,d,t,y)$(not sFirstHour(t) and fEnableStorage)..
   vStorage(st,q,d,t,y) =e= pStorData(st,"Efficiency")*vStorInj(st,q,d,t,y) - sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y)) + vStorage(st,q,d,t-1,y);

* SOC at first hour (no past state)
eStateOfChargeInit(st,q,d,sFirstHour(t),y)$fEnableStorage..
   vStorage(st,q,d,t,y) =e= pStorData(st,"Efficiency")*vStorInj(st,q,d,t,y) - sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y));

* Ensures SOC level can cover spinning reserve
eSOCSupportsReserve(st,q,d,t,y)$fEnableStorage..
   vSpinningReserve(st,q,d,t,y) =l= vStorage(st,q,d,t,y);

* Ensures that the state of charge at the end of the representative day equals the initial state
* This avoids artificial energy gains/losses over the daily cycle
* eSOCCycleClosure(st,q,d,sLastHour(t),y)$(fEnableStorage)..
*   vStorage(st,q,d,t,y) =e= vStorage(st,q,d,t-23,y) - (pStorData(st,"efficiency") * vStorInj(st,q,d,t-23,y) - sum(gfmap(st,f), vPwrOut(st,f,q,d,t-23,y)));

* Ensures energy conservation over the full representative day: total input × efficiency = total output
* eDailyStorageEnergyBalance(st,q,d,y)$(fEnableStorage)..
*   pStorData(st,"efficiency") * sum(t, vStorInj(st,q,d,t,y)) =e= sum((gfmap(st,f),t), vPwrOut(st,f,q,d,t,y));

* ------------------------------
* CSP-specific equations
* Constrains CSP thermal and storage operations.
* ------------------------------

* Limits CSP storage level to installed storage capacity.
eCSPStorageCapacityLimit(cs,q,d,t,y)$fEnableCSP..
   vStorage(cs,q,d,t,y) =l= vCapStor(cs,y);

* Limits CSP storage injection to thermal energy output * efficiency. Prevents unrealistic injection behavior.
eCSPStorageInjectionLimit(cs,q,d,t,y)$fEnableCSP..
   vStorInj(cs,q,d,t,y) =l= vThermalOut(cs,q,d,t,y)*pCSPData(cs,"Thermal Field","Efficiency");

* Prevents CSP storage injection exceeding installed capacity.
eCSPStorageInjectionCap(cs,q,d,t,y)$fEnableCSP..
   vStorInj(cs,q,d,t,y) =l= vCapStor(cs,y);

*Limits thermal energy output to installed thermal field capacity * hourly solar profile.
eCSPThermalOutputLimit(cs,q,d,t,y)$fEnableCSP..
   vThermalOut(cs,q,d,t,y) =l= vCapTherm(cs,y)*pCSPProfile(cs,q,d,t);

* Balances CSP thermal output, storage in/out, and generator dispatch.
eCSPPowerBalance(cs,q,d,t,y)$fEnableCSP..
   vThermalOut(cs,q,d,t,y)*pCSPData(cs,"Thermal Field","Efficiency")
 - vStorInj(cs,q,d,t,y) + vStorOut(cs,q,d,t,y)*pCSPData(cs,"Storage","Efficiency")
   =e= sum(gfmap(cs,f), vPwrOut(cs,f,q,d,t,y));

* Tracks CSP storage state of charge across time (except for first hour).
eCSPStorageEnergyBalance(cs,q,d,t,y)$(not sFirstHour(t) and fEnableCSP)..
   vStorage(cs,q,d,t,y) =e= vStorage(cs,q,d,t-1,y) + vStorInj(cs,q,d,t,y) - vStorOut(cs,q,d,t,y);

* Initializes CSP storage balance for the first hour of the day.
eCSPStorageInitialBalance(cs,q,d,sFirstHour(t),y)$fEnableCSP..
   vStorage(cs,q,d,t,y) =e= vStorInj(cs,q,d,t,y) - vStorOut(cs,q,d,t,y);

*Equation needed in dispatch mode but not for capacity expansion with representative days
*eCSPStorageDayLink(cs,q,d,sFirstHour(t),y)$(not sFirstDay(d) and fEnableCSP)..
*   vStorage(cs,q,d,t,y) =e= vStorInj(cs,q,d,t,y) - vStorOut(cs,q,d,t,y) + vStorage(cs,q,d-1,sLastHour,y);

* ------------------------------
* Storage capacity evolution
* Tracks storage energy capacity builds/retirements.
* ------------------------------

* Limits total installed storage capacity to predefined technical data from pStorData and CSP-related capacity from pCSPData. Only applies if storage is included (fEnableStorage).
eCapacityStorLimit(g,y)$fEnableStorage..
   vCapStor(g,y) =l= pStorData(g,"CapacityMWh") + pCSPData(g,"Storage","CapacityMWh");

* Sets initial year’s storage capacity equal to existing capacity (if online) plus new builds minus retirements.
eCapStorBalance(g, sStartYear(y))..
    vCapStor(g,y) =e= pStorData(g,"CapacityMWh")$(eg(g) and (pGenData(g,"StYr") <= sStartYear.val)) + vBuildStor(g,y) - vRetireStor(g,y);

* Tracks annual storage capacity changes for existing generators (EGs): previous year’s capacity + builds − retirements.
eCapStorAnnualUpdateEG(eg,y)$(not sStartYear(y) and fEnableStorage)..
   vCapStor(eg,y) =e= vCapStor(eg,y-1) + vBuildStor(eg,y) - vRetireStor(eg,y);

* Tracks annual storage capacity for new generators (NGs): previous year’s capacity + builds. Assumes no retirement.
eCapStorAnnualUpdateNG(ng,y)$(not sStartYear(y) and fEnableStorage)..
   vCapStor(ng,y) =e= vCapStor(ng,y-1) + vBuildStor(ng,y);

* Sets initial capacity for new generators at start year equal to build amount. Applies only in the first year.
eCapStorInitialNG(ng,sStartYear(y))$fEnableStorage..
   vCapStor(ng,y) =e= vBuildStor(ng,y);

eBuildStorNew(eg)$((pGenData(eg,"StYr") > sStartYear.val) and fEnableStorage)..
   sum(y, vBuildStor(eg,y)) =l= pStorData(eg,"CapacityMWh");
   
* ------------------------------
* CSP thermal capacity limits
* Tracks CSP thermal field builds and limits.
* ------------------------------
eCapacityThermLimit(g,y)$fEnableCSP..
   vCapTherm(g,y) =l= pCSPData(g,"Thermal Field","CapacityMWh");

eCapThermBalance1(eg,y)$(not sStartYear(y) and fEnableCSP)..
   vCapTherm(eg,y) =e= vCapTherm(eg,y-1) + vBuildTherm(eg,y) - vRetireTherm(eg,y);

eCapThermBalance2(ng,y)$(not sStartYear(y) and fEnableCSP)..
   vCapTherm(ng,y) =e= vCapTherm(ng,y-1) + vBuildTherm(ng,y);

eCapThermBalance3(ng,sStartYear(y))$fEnableCSP..
   vCapTherm(ng,y) =e= vBuildTherm(ng,y);

eBuildThermNew(eg)$((pGenData(eg,"StYr") > sStartYear.val) and fEnableCSP)..
   sum(y, vBuildTherm(eg,y)) =l= pCSPData(eg,"Thermal Field","CapacityMWh");

* ------------------------------
* Capital budget constraint
* Enforces the global investment ceiling.
* ------------------------------

eCapitalConstraint$fApplyCapitalConstraint..
   sum(y, pRR(y)*pWeightYear(y)*sum(ng, pCRF(ng)*vCap(ng,y)*pGenData(ng,"Capex"))) =l= ssMaxCapitalInvestmentInvestment*1e3;
   
* ------------------------------
* Emissions related equations
* Computes CO₂ emissions and applies caps/backstops.
* ------------------------------

eZonalEmissions(z,y)..
   vZonalEmissions(z,y) =e=
   sum((gzmap(g,z),gfmap(g,f),q,d,t), vPwrOut(g,f,q,d,t,y)*pHeatRate(g,f)*pFuelCarbonContent(f)*pHours(q,d,t));

eEmissionsCountry(c,y)$fApplyCountryCo2Constraint..
   sum(zcmap(z,c), vZonalEmissions(z,y))-vYearlyCO2backstop(c,y)=l= pEmissionsCountry(c,y);

eTotalEmissions(y)..
    sum(z, vZonalEmissions(z,y))=e= vTotalEmissions(y);
    
eTotalEmissionsConstraint(y)$pfApplySystemCo2Constraint..
    vTotalEmissions(y)-vYearlySysCO2backstop(y) =l= pEmissionsTotal(y);
   

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
   eYearlyImportExternalCost
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
   eMaxAnnualImportShareEnergy
   eMaxAnnualExportShareEnergy  
   eMaxHourlyImportShareEnergy
   eMaxHourlyExportShareEnergy
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
   vNewTransmissionLine(sAdditionalTransfer)
   vFlow(sFlow)
   vSpinningReserve(sSpinningReserve)
   
  
/;
