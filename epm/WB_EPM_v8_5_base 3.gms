$onText
*test
*test2
December 1, 2018

$offText
$offEolCom
$eolCom //

*-------------------------------------------------------------------
* DECLARATION OF SETS, PARAMETERS AND VARIABLES
*-------------------------------------------------------------------

Sets
   g        'generators or technology-fuel types'
   f        'fuels'
   y        'years'
   q        'quarters or seasons'
   d        'day types'
   t        'hours of day'
   z        'zones'
   c        'countries'
   zext     'external zones'
*********Hydrogen specific addition***********
   hh        'Hydrogen production units'
;


alias (z,z2), (g,g1,g2);

*****Dispatch*******
alias (d,d2);
*****************

*
* Generators
Sets
   eg(g)       'existing generators'
   ng(g)       'new generators'
   cs(g)       'concentrated solar power'
   so(g)       'PV plants with storage'
   stp(g)      'storage for PV plant'
   st(g)       'all storage plants'
   dc(g)       'candidate generators with capex trajectory'
   ndc(g)      'candidate generators without capec trajectory'
   vre(g)      'variable renewable generators'
   re(g)       'renewable generators'
   RampRate(g) 'ramp rate constrained generator blocks' // Ramprate takes out inflexible generators for a stronger formulation so that it runs faster
   
*****************Dispatch*****************************
   MinGenPoint(g)   'Generators with minimum generation level constraints (only for running on dispatch mode)'
   MinUpTime(g)     'Generators with minimum UpTime constraints (only for running on dispatch mode)'
   MinDnTime(g)     'Generators with minimum DnTime constraints (only for running on dispatch mode)'   
   HoursOn(g)       'Generators with the Hours of being on ahead  of the first hour of the study horizon defined (only for running on dispatch mode)'
   HoursOff(g)      'Generators with the Hours of being turned off ahead  of the first hour of the study horizon defined (only for running on dispatch mode)'


************** H2 model specific sets ***************************
   eh(hh)           'existing hydrogen generation plants'
   nh(hh)           'new hydrogen generation plants'
   RampRateH2(hh)   'ramp rate constrained H2 generator blocks' // Ramprate takes out inflexible generators for a stronger formulation so that it runs faster
   dcH2(hh)         'candidate generators with capex trajectory'
   ndcH2(hh)        'candidate generators without capex trajectory'
   nRE(g)           'non RE generators'
   nVRE(g)          'non VRE generators'
   REH2(g)          'set of RE generators which are NOT VRE'
   nREH2(g)         'set of generators that are dont belong to subset REH2(g)'
   
;


*********************dispatch*************************
set AT all time steps (8760 hours) /AT1*AT8760/;
set FD(q,d,t) feasible days throughout the year;
alias (AT,AT2);
*********************************************** 

* Time
Sets
   ghdr         'Header for pGenData' / BuildLimitperYear, Capacity, Capex, DescreteCap, FOMperMW, MaxTotalBuild,
                                        MinLimitShare, MinTotalBuild, Overloadfactor, RampDnRate, RampUpRate, ReserveCost, ResLimShare, RetrYr, StYr, UnitSize, fuel1, CapacityCredit,
******************************************************Dispatch model**************************************************                                        
                MinGenPoint,StUpCost, minUT, minDT, HoursOn, HoursOff, InitialOn /
********************************************************************************************************************************************************
   shdr         'Header for pStorData' / Capacity, Capex, FixedOM, Efficiency/
   csphrd       'Header for pCSPData' / Storage, "Thermal Field" /
   thdr         'Header for pNewTransmission' / CapacityPerLine, CostPerLine, Life, MaximumNumOfLines /
   Relevant(d)  'relevant day and hours when MinGen limit is applied'
***********************Hydrogen model related sets*********************
   hhdr         'Header for pH2Data' / StYr, RetrYr,  Capacity, Capex, HeatRate,VOM, FOMperMW,  Efficiency,  BuildLimitperYear, MaxTotalBuild,  DescreteCap,
                                         RampUpRate, RampDnRate,   ResLimShare, UnitSize, Life, Type /
   h2Index      'Index of hydrogen fuels'    /HydrogenIndex/
;

alias(hh,hh1);

* Mapping (fuel, storage...)
Sets
   gfmap(g,f)             'generator g mapped to fuel f'
   gzmap(g,z)             'generator g mapped to zone z'
   zfmap(z,f)             'fuel f available in zone z'
   gsmap(g2,g)            'generator storage map'
   zcmap(z<,c<)           'map zones to countries'
   sTopology(z,z2)        'network topology - to be assigned through network data'
   sMapNCZ(z,z2)          'set of connecting zones belong to different countries'   
**************Hydrogen production model related sets*******************************
   h2zmap(hh,z)
   ft
;

* Implicit variable domain
Sets
   sPwrOut(g,f,q,d,t,y)        'valid tuples for vPwrOut'
   sExportPrice(z,zext,q,d,t,y)     'valid tuples for vExportPrice'
   sImportPrice(z,zext,q,d,t,y)     'valid tuples for vImportPrice'
   sAdditionalTransfer(z,z2,y) 'valid tuples for vAdditionalTransfer'
   sFlow(z,z2,q,d,t,y)         'valid tuples for vFlow'
   sSpinningReserve(g,q,d,t,y)         'valid tuples for vSpinningReserve'
************************H2 related set***************************************
   sH2PwrIn(hh,q,d,t,y)
;

Set MapGG(g,g1)  mapping of generators where simultaneous commissioning decommissioning exists;
Singleton sets
   sStartYear(y)
   
   sFirstHour(t)
   sLastHour(t)
   sFirstDay(d)
   
***********Dispatch*********************
   sFirstHourAT(AT)
   sLastDay(d)
   sFirstQuarter(q)
   sLastQuarter(q)

**************************

;
***************Dispatch****************
set mapTS(q,d,t,AT) set that maps days and time steps;

************************************

Parameters
   pCostOfCurtailment               'Cost of curtailment'
   pCostOfCO2backstop               'Cost of climate backstop techno in $ per ton of CO2'
$ifi     %mode%==MIRO   pHours(q<,d<,y<,t<) 'duration of each block'
$ifi not %mode%==MIRO   pHours(q<,d<,y ,t<) 'duration of each block'
   pWeightYear(y)                   'weight on years'
* Generators
   pGenData(g,ghdr)                 'generator data'
   pAvailability(g,q)               'Availability by generation type and season or quarter in percentage - need to reflect maintenance'
   pCapexTrajectories(g,y)          'capex trajectory  final'
* Exchanges
   pTransferLimit(z,z2,q,y)         'Transfer limits by quarter (seasonal) and year between zones'
   pMaxExchangeShare(y,c)           'Max share of exchanges by country [same limit for imports or exports for now]'

   pAllowHighTransfer
   pAllowExports                    'Allow price based exports'   
   pTradePrice(zext,q,d,y,t)        'trade price - export or import driven by prices [assuming each zone in a country can only trade with one external zone]'
   pMaxImport                       'Maximum Hourly Imports, based on hourly country demand'   
   pMaxExport                       'Maximum Hourly Exports, based on hourly country demand'
   pExtTransferLimit(z,zext,q,*,y)  'external transfer limit'
   pExtTransferLimitIn(z,zext,q,y)  'transfer limit with external zone for import towards internal zone'
   pExtTransferLimitOut(z,zext,q,y) 'transfer limit with external zone for export towards external zone'
* Fuels
   pMaxFuellimit(c,f,y)             'Fuel limit in MMBTU*1e6 (million) by country'
* Demand
   pDemandData(z,q,d,y,t)           'hourly load curves by quarter(seasonal) and year'
   pEnergyEfficiencyFactor(z,y)     'Scaling factor for energy efficiency measures'
* T&D
   pLossFactor(z,z2,y)              'loss factor in percentage'
   pNewTransmission(z,z2,thdr)      'new transmission lines'
* Renewables and storage
   pVREgenProfile(g,f,q,d,t)        'VRE generation profile by plant quarter day type and YEAR -- normalized (per MW of solar and wind capacity)'
   pCSPData(g,csphrd,shdr)
   pCSPProfile(g,q,d,t)             'solar profile for CSP in pu'
   pStoPVProfile(g,q,d,t)           'solar profile for Pv with Storage in pu'
   pStorData(g,shdr)                'Storage data'
   pVREForecastError                'Percentage error in VRE forecast [used to estimated required amount of spinning reserve]'
   
* Reserves
   pSpinningReserveReqCountry(c,y)     'Spinning reserve requirement local at country level (MW)  -- for isolated system operation scenarios'
   pSpinningReserveReqSystem(y)     'Spinning reserve requirement systemwide (MW) -- for integrated system operation scenarios'
   pPlanningReserveMargin(c)        'Country planning reserve margin'
   pCapacityCredit(g,y)             'Share of capacity counted towards planning reserves'
   pVREForecastError                'Spinning reserve needs for VRE (as a share of VRE generation)'
* CO2
   pCarbonPrice(y)                  'Carbon price in USD per ton of CO2eq'
   pFuelCarbonContent(f)            'Fuel carbon content in tCO2 per MMBTu'


   pEmissionsCountry(c,y)           'Maximum emissions allowed per country and year in tns'


   pEmissionsCountry(c,y)              'Maximum zonal emissions allowed per country and year in tns'

   pEmissionsTotal(y)               'Maximum system emissions allowed year in tns'
* Economic parameters
   pRR (y)                          'accumulated return rate factor'
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
* Control parameters
   pramp_constraints
   pfuel_constraints
   pcapital_constraints
   pmingen_constraints
   pincludeCSP
   pincludeStorage
   pIncludeCarbon                   'include the cost of carbon'
   pSurplusPenalty
   pMinRE
   pMinRETargetYr
   pzonal_CO2_constraints
   pSystem_CO2_constraints
   pzonal_spinning_reserve_constraints
   psystem_spinning_reserve_constraints
   pplanning_reserve_constraints
   psys_reserve_margin
   pHeatrate(g,f)                   'Heatrate of fuel f in generator g'
   pIncludeDecomCom                 'Include simultaneous commissioning'
     
**************H2 model related parameters
   pH2Data(hh,hhdr)                 'hydrogen generating units'
   pIncludeH2                       'Flag to activate hydrogen related equations'
   pAvailabilityH2(hh,q)            'Availability by generation type and season or quarter in percentage - need to reflect maintenance'
   pFuelData(f)
   pCRFH2(hh)                       'Capital recovery factor'
   pCapexTrajectoriesH2(hh,y)       'CAPEX trajectories for hydrogen generation units'
   pVarCostH2(hh,y)                 'Variable cost - H2 production'

   pExternalH2(z,q,y)               'mmBTUs of H2 as external demand that need to be met'
   pH2UnservedCost                  'Cost of external H2 unserved'
********************Dispatch****************************************
   pRunDispatch                    'Flag to run EPM on dispatch mode'
   pIncludeMUDT                    'Include minimum up/down constraints (only on dispatch mode)'
   pIncludeMinGenCon               'Include minimum generation level constraints (only on dispatch mode)'
   pIncludeStUp                     'Include Start up/shut down decisoins.'
   pIncludeDR                      'Include Demand Response'
   pMaxDR                          'Maximum level of DR in MW'
   pCostOfDR                       'cost of demand response in $ per MWh'
   pDays(q)                        'Number of days per month'
********************************************************************

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
   vImportPrice(z,zext,q,d,t,y)   'external price driven import'
   vExportPrice(z,zext,q,d,t,y)   'external price driven export'

   vFuel(z,f,y)              'annual fuel in MMBTU'

   vAnnCapex(g,y)            'Annualized capex for capacity vCap installed in Year y  carried through planning horizon'

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


   vAdditionalTransfer(z,z2,y)        'additional transfer limit'
   vYearlyTransmissionAdditions(z,y)  'added transmission cost (not included in cost of generation)'
   vYearlyCurtailmentCost(z,y)
   vCurtailedVRE(z,g,q,d,t,y)

*************H2 MODEL SPECIFIC VARIABLES***************************
   vCapH2(hh,y)                'total capacity in place accounting for legacy, new and retired plants (MW)'
   vBuildH2(hh,y)              'Build (MW)'
   vRetireH2(hh,y)             'Retire (MW)'
   vNetCurtailedVRE(z,q,d,t,y) 'Curtailed VRE as an outcome of not being used for H2 production'
   vH2PwrIn(hh,q,d,t,y)        'Power drawn by H2 plants for H2 production in MW'
   vFuelH2(z,y)                'Annual H2 production in MMBTU'
   vAnnCapexH2(hh,y)           'Annualized CAPEX of H2 producing technologies'
   vPwrREH2(z,q,d,t,y)          'Generation from RE generators that goes to H2 production'
   vPwrREGrid(z,q,d,t,y)       'Generation from RE generators that goes to the grid'
   vREPwr2H2(g,f,q,d,t,y)       'Generation from RE generators that goes to H2 production'
   vREPwr2Grid(g,f,q,d,t,y)    'Generation from VRE generators that goes to the grid'
   vFuelH2Quarter(z,q,y)        'H2 fuel saved for H2 electricity generationon a quarterly basis'
   vUnmetExternalH2(z,q,y)        'mmBTU of external H2 demand that cant be met'
   vYearlyH2UnservedCost(z,y)   'Annual Cost of external H2 unserved in USD'  

***********************Dispatch***************************
vStUpCost(g,q,d,t,y)
**********************
vDemandShed(z,q,d,t,y)            'Demand shed in MW'
vCostDemandShed(z,y)              ' Cost of demand shedding in USD'
vPwrOutDispatch(g,q,d,t,AT,y)

***********************************************

;

Free Variable
   vNPVCost                     'discounted total system cost'
   vStorNet(g,q,d,t,y)
   vYearlyTotalCost(c,y)
   vYearlyFixedCost(z,y)
   vYearlyVariableCost(z,y)
   vYearlyUSECost(z,y)
   vYearlyTradeCost(z,y)
   vYearlySpinningReserveCost(z,y)      'Yearly spinning reserve costs'
   vYearlyUnmetReserveCostCountry(c,y)         'Country unmet spinning and planning reserve'
   vYearlyCarbonCost(z,y)               'country carbon cost'
   vYearlyUnmetReserveCostSystem(y)                 'system unmet planing reserve'
   vYearlyCO2backCost(c,y)              'ccost of CO2 backstop'
   
;

Integer variable
   vBuildTransmission(z,z2,y)
   vBuiltCapVar(g,y)
   vRetireCapVar(g,y)
   

*******H2 production model integer variables******************
   vBuiltCapVarH2(hh,y)

   vRetireCapVarH2(hh,y)


********************************************
;


********************Dispatch model**********************88

Binary variable
vOn(g,q,d,t,AT,y)
vStUp(g,q,d,t,AT,y)
vShDn(g,q,d,t,AT,y)
********************************

;






Equations
   eNPVCost                        'objective function'
   eYearlyTotalCost(c,y)
   eYearlyFixedCost(z,y)
   eYearlyVariableCost(z,y)
   eYearlySpinningReserveCost(z,y)
   eYearlyUSECost(z,y)
   eYearlyTradeCost(z,y)
   eYearlyUnmetReserveCostCountry(c,y)
   eYearlyUnmetReserveCostSystem(y)            'unmet reserve system'
   eYearlyCarbonCost(z,y)

   eDemSupply(z,q,d,t,y)           'demand balance'
   eMaxhourlyExportsshare(c,q,d,t,y) 'max exports to an external zone (hourly limit)'
   eMaxhourlyImportsshare(c,q,d,t,y) 'max imports from an external zone  (hourly limit)'

   eCapBalance(g,y)                'capacity balance'
   eCapBalance1(g,y)               'capacity balance'
   eCapBalance2(g,y)               'capacity balance'
   eBuildNew(g)
   eBuiltCap(g,y)                  'built capacity'
   eRetireCap(g,y)                 'retired capacity'

   eMaxBuildTotal(g)               'max build over all years'
   



   eMinGenRE(c,y)                  'Min Generation of RE after a target year at country level'

   eMaxCF(g,q,y)                   'max capacity factor'
   eMinGen(g,q,d,t,y)              'Minimum generation limit for new generators'

   eFuel(z,f,y)                    'fuel balance'
   eFuelLimit(c,f,y)               'fuel limit at country level'

   eRampUpLimit(g,q,d,t,y)         'Ramp up limit'
   eRampDnLimit(g,q,d,t,y)         'Ramp down limit'

   eSpinningReserveLim(g,q,d,t,y)              'Reserve limit as a share of capacity'
   eJointResCap(g,q,d,t,y)                     'Joint reserve and generation limit'
   eSpinningReserveReqCountry(c,q,d,t,y)          'Country spinning reserve requirement'
   eSpinningReserveReqSystem(q,d,t,y)          'System spinning reserve requirement'
   ePlanningReserveReqSystem(y)                'Min system planning reserve requirement'
   ePlanningReserveReqCountry(c,y)                'Minimum capacity reserve over peak demand at country level'

   eTransferLimit(z,z2,q,d,t,y)    'Transfer limits'
   eVREProfile(g,f,z,q,d,t,y)      'VRE generation restricted to VRE profile'
   eMaxImportPrice(c,y)            'import limits: max import from external zones'
   eMaxExportPrice(c,y)            'export limits: max export to external zones'
   eYearlySurplusCost(z,y)
   eAdditionalTransfer(z,z2,y)
   eAdditionalTransfer2(z,z2,y)
   eYearlyTransmissionAdditions (z,y)
   eExtZoneLimitImport(z,zext,q,d,t,y) 'import limits from external zone in MW'
   eExtZoneLimitExport(z,zext,q,d,t,y) 'export limits to external zone in MW'

   eAnnCapex(g,y)                  'Annualized capex'
   eAnnCapex1(g,y)


   eCapitalConstraint              'capital limit expressed by pMaxCapital in billion USD'
   eZonalEmissions(z,y)            'CO2eq emissions by zone and year in tons'
   eEmissionsCountry(c,y)          'constraint on country CO2eq emissions'
   eTotalEmissions(y)              'total regional CO2eq emissions by year in tons'
   eEmissionsTotal(y)              'Total CO2eq emissions by year in tons'
   eYearlyCO2backstopCost(c,y)     'co2 backstop cost in USD'
   eTotalEmissionsConstraint(y) 	'constraint on total CO2eq emissions by year in tons'
  

   eRampDownInjLimit(g,q,d,t,y)
   eRampUpInjLimit(g,q,d,t,y)

   eYearlyCurtailmentCost(z,y)

   eStorBal(g,q,d,t,y)
   eStorBal1(g,q,d,t,y)
   eStorBal2(g,q,d,t,y)
   eStorBal3(g,q,d,y)




   eStorageOutput(g,q,d,t,y)
   eStorageInjection(g,q,d,t,y)
   eStorageInjection2(g,q,d,t,y)
   eStorageNet(g,q,d,t,y)

   eStorageCap(g,q,d,t,y)
   eStorageCap2(g,q,d,t,y)         'storage capacity (energy) must be at least 1 hour if installed'

   eStorageCSPCap(g,q,d,t,y)
   eInjCSP(g,q,d,t,y)
   eInjCSP1(g,q,d,t,y)
   eThermCSP(g,q,d,t,y)
   eGenCSP(g,q,d,t,y)
   eStorageCSPBal(g,q,d,t,y)
   eStorageCSPBal1(g,q,d,t,y)

   eCapacityStorLimit(g,y)
   eCapStorBalance1(g,y)
   eCapStorBalance2(g,y)
   eCapStorBalance3(g,y)
   eBuildStorNew(g)

   eCapacityThermLimit(g,y)
   eCapThermBalance1(g,y)
   eCapThermBalance2(g,y)
   eCapThermBalance3(g,y)
   eBuildThermNew(g)
   eSimultComDecom(g,g1,y)          Simultaneous commissioning and decommissioning of pair of units


**********************IAEA*************************************************************************
eStorageCap3(g,q,d,t,y)
eStorageCap4(g,q,d,t,y)
eStorageCap5(g,q,d,t,y)


***********Dispatch*************
eStorBal4(g,q,d,t,y)
eStorBal5(g,q,d,t,y)
eStorBal6(g,q,d,t,y)









eRampDispCon1(g,q,d,t,AT,y)
eRampUpLimitDisp(g,AT,y)
eRampDnLimitDisp(g,AT,y)


eSpinningReserveLim2(g,q,d,t,y)              'Reserve limit as a share of capacity'

eStUpCosts(g,q,d,t,y)
eDemandRespCon(q,d,t,y)
eYearlyCostDemandShed(z,y)
eOnCon(g,q,d,t,AT,y)
********************************




* Minimum up/down constraints


eMinUpTime_con1(g,y)              Minimum up time con
eMinUpTime_con2(g,AT2,y)                Minimum up time con


eMinDnTime_con1(g,y)              Minimum dn time con
eMinDnTime_con2(g,AT2,y)                Minimum dn time con


eStUp1(g,AT,y)
eStUp2(g,AT,y)
eStUp3(g,AT,y)

eMinGenPoint(g,q,d,t,y)
eMaxGenPoint(g,q,d,t,y)




******************************



***********************************Hydrogen production model**************************************
eH2UnservedCost(z,y)
eCapBalanceH2(hh,y)
eCapBalance1H2(hh,y)
eCapBalance2H2
eBuildNewH2(hh)
eMaxBuildTotalH2(hh)
eBuiltCapH2(hh,y)
eRetireCapH2(hh,y)
*eDemSupplyH2(z,q,d,t,y)
eMaxCF_H2(hh,q,y)
eFuel_H2(c,q,y)
eFuel_H2_2(c,z,y)
eRampDnLimitH2(hh,q,d,t,y)
eRampUpLimitH2(hh,q,d,t,y)
eFuelLimitH2(c,f,y)
eFuelLimitH2_2(c,f,y)
eAnnCapexH2_1(hh,y)
eAnnCapexH2(hh,y)

eVRE2H2_2(z,g,f,q,d,t,y)
eVRE2H2_3(z,q,d,t,y)
eVRE2H2_4(z,q,d,t,y)
eRE2H2(g,f,q,d,t,y)
eRE2H2_2(z,q,d,t,y)
eRE2H2_3(z,q,d,t,y)
eRE2H2_4(z,q,d,t,y)
eMaxH2PwrInjection(hh,q,d,t,y)

*eRE2H2_3(z,q,d,t,y)



;

*---    Objective function
eNPVCost..
   vNPVCost =e= sum(y, pRR(y)*pWeightYear(y)*(sum(c, vYearlyTotalCost(c,y)) + vYearlyUnmetReserveCostSystem(y)+vYearlySysCO2backstop(y)* pCostOfCO2backstop));

*---  Cost equations
* Note capex is full capex in $m per MW. Also note VarCost includes fuel cost and VOM -
* essentially the short run marginal cost for the generator

eYearlyTotalCost(c,y)..
   vYearlyTotalCost(c,y) =e= vYearlyUnmetReserveCostCountry(c,y)+ vYearlyCO2backCost(c,y) 
                           + sum(zcmap(z,c), vYearlyFixedCost(z,y)
                                           + vYearlyVariableCost(z,y)
                                           + vYearlySpinningReserveCost(z,y)
                                           + vYearlyUSECost(z,y)
                                           + vYearlyCarbonCost(z,y)
                                           + vYearlyTradeCost(z,y)
                                           + vYearlyTransmissionAdditions(z,y)
                                           + vYearlyCurtailmentCost(z,y)
                                           + vYearlySurplus(z,y)
********************************************IAEA***************************
                                           + vCostDemandShed(z,y)$(pIncludeDR)

*************************************H2 model*************************************************************                                           
                                           + vYearlyH2UnservedCost(z,y)$pIncludeH2);
***********************************************************************************************************
eYearlyFixedCost(z,y)..
   vYearlyFixedCost(z,y) =e= sum(gzmap(ndc,z), pCRF(ndc)*vCap(ndc,y)*pGenData(ndc,"Capex")*1e6)
                           + sum(gzmap(ndc,z)$(not cs(ndc)), pCRFsst(ndc)*vCapStor(ndc,y)*pStorData(ndc,"Capex")*1e3)
                           + sum(gzmap(ndc,z)$(not st(ndc)), pCRFcst(ndc)*vCapStor(ndc,y)*pCSPData(ndc,"Storage","Capex")*1e3)
                           + sum(gzmap(ndc,z), pCRFcth(ndc)*vCapTherm(ndc,y)*pCSPData(ndc,"Thermal Field","Capex")*1e6)
                           + sum(gzmap(dc,z), vAnnCapex(dc,y))
                           + sum(gzmap(g,z), vCap(g,y)*pGenData(g,"FOMperMW"))
                           + sum(gzmap(st,z),vCap(st,y)*pStorData(st,"FixedOM"))
                           + sum(gzmap(cs,z), vCapStor(cs,y)*pCSPData(cs,"Storage","FixedOM"))
                           + sum(gzmap(cs,z), vCapTherm(cs,y)*pCSPData(cs,"Thermal field","FixedOM"))
                           
***********************************************Hydrogen model related costs******************************************
                          + sum(h2zmap(ndcH2,z), pCRFH2(ndcH2)*vCapH2(ndcH2,y)*pH2Data(ndcH2,"Capex")*1e6)$pIncludeH2
                          + sum(h2zmap(dcH2,z), vAnnCapexH2(dcH2,y))$pIncludeH2
                          + sum(h2zmap(hh,z), vCapH2(hh,y)*pH2Data(hh,"FOMperMW"))$pIncludeH2;

*********************************************************************************************************************



eYearlyVariableCost(z,y)..

   vYearlyVariableCost(z,y) =e= sum((gzmap(g,z),f,q,d,t), pVarCost(g,f,y)*vPwrOut(g,f,q,d,t,y)*pHours(q,d,y,t))


****************************************************
   
*********************************************Hydrogen model related costs ************************************
***(Units for equation below)                              $/mmBTU_H2    x      mmBTU_H2/MWh_e  x      MW_e       x       Hrs
                              + sum((h2zmap(hh,z),q,d,t), pVarCostH2(hh,y)*pH2Data(hh,"Heatrate")*vH2PwrIn(hh,q,d,t,y)*pHours(q,d,y,t))$pIncludeH2
                              

*****************************Dispatch (do we want to have start up costs as part of variable costs?)*****************************   
+sum((gzmap(g,z),q,d,t),vStUpCost(g,q,d,t,y))$(pRunDispatch and pIncludeMinGenCon)                              
;
* Note: ReserveCost is in $/MWh -- this is the DIRECT cost of holding reserve like wear and tear that a generator bids in a market
eYearlySpinningReserveCost(z,y)..
   vYearlySpinningReserveCost(z,y) =e= sum((gzmap(g,z),q,d,t), vSpinningReserve(g,q,d,t,y)*pGenData(g,"ReserveCost")*pHours(q,d,y,t));

eYearlyUSECost(z,y)..
   vYearlyUSECost(z,y) =e= sum((q,d,t), vUSE(z,q,d,t,y)*pVoLL*pHours(q,d,y,t));

eYearlySurplusCost(z,y)..
   vYearlySurplus(z,y) =e= sum((q,d,t), vSurplus(z,q,d,t,y)*pSurplusPenalty*pHours(q,d,y,t));

eYearlyCurtailmentCost(z,y)..
   vYearlyCurtailmentCost(z,y) =e= sum((gzmap(g,z),q,d,t), vCurtailedVRE(z,g,q,d,t,y)*pCostOfCurtailment*pHours(q,d,y,t));

eYearlyTradeCost(z,y)..
   vYearlyTradeCost(z,y) =e= sum((zext,q,d,t), vImportPrice(z,zext,q,d,t,y)*pTradePrice(zext,q,d,y,t)*pHours(q,d,y,t))
                           - sum((zext,q,d,t), vExportPrice(z,zext,q,d,t,y)*pTradePrice(zext,q,d,y,t)*pHours(q,d,y,t));

eYearlyUnmetReserveCostCountry(c,y)..
   vYearlyUnmetReserveCostCountry(c,y) =e= vUnmetPlanningReserveCountry(c,y)*pPlanningReserveVoLL
                         + sum((q,d,t), vUnmetSpinningReserveCountry(c,q,d,t,y)*pHours(q,d,y,t)*pSpinningReserveVoLL);
                         
eYearlyCO2backstopCost(c,y)..
   vYearlyCO2backCost(c,y) =e= vYearlyCO2backstop(c,y)*pCostOfCO2backstop;

eYearlyUnmetReserveCostSystem(y)..
   vYearlyUnmetReserveCostSystem(y) =e= vUnmetPlanningReserveSystem(y)*pPlanningReserveVoLL
                          + sum((q,d,t), vUnmetSpinningReserveSystem(q,d,t,y)*pHours(q,d,y,t)*pSpinningReserveVoLL);

eYearlyCarbonCost(z,y)..
   vYearlyCarbonCost(z,y) =e= pIncludeCarbon*pCarbonPrice(y)
                            * Sum((gzmap(g,z),gfmap(g,f),q,d,t), vPwrOut(g,f,q,d,t,y)*pHeatRate(g,f)*pFuelCarbonContent(f)*pHours(q,d,y,t));

*****************************************H2 model***********************************************************
eH2UnservedCost(z,y)..
                vYearlyH2UnservedCost(z,y)       =e= sum(q, vUnmetExternalH2(z,q,y) )*pH2UnservedCost$(pIncludeH2);
***************************************************************************************************************

**************************************************Dispatch***************
eYearlyCostDemandShed(z,y)$(pIncludeDR)..
*                  --USD--                              -USD/MWh-                     --MW--                -Hrs-
               vCostDemandShed(z,y)             =e= pCostOfDR*sum((q,d,t)$FD(q,d,t),vDemandShed(z,q,d,t,y)*pHours(q,d,y,t));
***************************************************************************

*****************Dispatch**************

eStUpCosts(g,q,d,t,y)$(pRunDispatch and pIncludeMinGenCon and MinGenPoint(g) and FD(q,d,t)$pRunDispatch)..
vStUpCost(g,q,d,t,y)=e=sum(AT$mapTS(q,d,t,AT),vStUp(g,q,d,t,AT,y))*pGenData(g,"StUpCost");
       

$macro symmax(s,i,j,h) max(s(i,j,h),s(j,i,h))

eYearlyTransmissionAdditions(z,y)$(pAllowHighTransfer and sum(sTopology(z,z2),1))..
   vYearlyTransmissionAdditions(z,y) =e= sum(sTopology(z,z2), vAdditionalTransfer(z,z2,y)*symmax(pNewTransmission,z,z2,"CostPerLine")*1e6)
                                       / 2*(pWACC/(1-(1/((1+pWACC)**sum(sTopology(z,z2), symmax(pNewTransmission,z,z2,"Life"))))));

*--- Demand supply balance constraint
eDemSupply(z,q,d,t,y)$(FD(q,d,t)$pRunDispatch)..

**********************Demand supply equations has been redifined to account for H2 production************************************

*
   pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y) =e=sum((gzmap(g,z),gfmap(g,f)),vPwrOut(g,f,q,d,t,y))$(pIncludeH2=0)
*                                                            The row below is deactivated when H2 feature is on   
                                                        
***********************************************************H2 model equations below************************************************
                                                        +sum((gzmap(nRE,z),gfmap(nRE,f)),vPwrOut(nRE,f,q,d,t,y))$(pIncludeH2)
                                                          +vPwrREGrid(z,q,d,t,y)$pIncludeH2                                                         
*                                                         - SUM((h2zmap(hh,z)),vH2PwrIn(hh,q,d,t,y))$pIncludeH2
                                                        
*************************************************************************************************************************************
                                                         
                                                         - sum(sTopology(z,z2), vFlow(z,z2,q,d,t,y))
                                                         + sum(sTopology(z,z2), vFlow(z2,z,q,d,t,y)*(1-pLossFactor(z,z2,y)))
*                                                        + sum(gzmap(st,z), vStorOut(st,q,d,t,y))
                                                         - sum(gzmap(st,z), vStorInj(st,q,d,t,y))$(pincludeStorage)
                                                         + sum(zext, vImportPrice(z,zext,q,d,t,y)) 
                                                         - sum(zext, vExportPrice(z,zext,q,d,t,y)) 
                                                         + vUSE(z,q,d,t,y)
                                                         - vSurplus(z,q,d,t,y)
                                                         + vDemandShed(z,q,d,t,y)$(pIncludeDR)
                                                         ;



*--- Generator Capacity equations (check redundance with fix constraints in main.gms)
eCapBalance(g,sStartYear(y))..
   vCap(g,y) =e= pGenData(g,"Capacity")$(eg(g) and (pGenData(g,"StYr") <= sStartYear.val))
               + vBuild(g,y) - vRetire(g,y);

eCapBalance1(eg,y)$(not sStartYear(y))..
   vCap(eg,y) =e= vCap(eg,y-1) + vBuild(eg,y) - vRetire(eg,y);

eCapBalance2(ng,y)$(not sStartYear(y))..
   vCap(ng,y) =e= vCap(ng,y-1) + vBuild(ng,y);

eBuildNew(eg)$(pGenData(eg,"StYr") > sStartYear.val)..
   sum(y, vBuild(eg,y)) =l= pGenData(eg,"Capacity");

************Dispatch (it should be a general improvement on EPM)*********
eMaxBuildTotal(ng)$pGenData(ng,"MaxTotalBuild")..
   sum(y, vBuild(ng,y)) =l= pGenData(ng,"MaxTotalBuild");
*************************************************


eMinGenRE(c,y)$(pMinRE and y.val >= pMinRETargetYr)..
   sum((zcmap(z,c),gzmap(RE,z),gfmap(RE,f),q,d,t), vPwrOut(RE,f,q,d,t,y)*pHours(q,d,y,t)) =g=
   sum((zcmap(z,c),q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,y,t)*pEnergyEfficiencyFactor(z,y))*pMinRE;

eBuiltCap(ng,y)$pGenData(ng,"DescreteCap")..
   vBuild(ng,y) =e= pGenData(ng,"UnitSize")*vBuiltCapVar(ng,y);

eRetireCap(eg,y)$(pGenData(eg,"DescreteCap") and (y.val <= pGenData(eg,"RetrYr")))..
   vRetire(eg,y) =e= pGenData(eg,"UnitSize")*vRetireCapVar(eg,y);


*--- Production equations
eJointResCap(g,q,d,t,y)$(FD(q,d,t)$pRunDispatch)..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) + vSpinningReserve(g,q,d,t,y) =l= vCap(g,y)*(1+pGenData(g,"Overloadfactor"));

eMaxCF(g,q,y)..
   sum((gfmap(g,f),d,t), vPwrOut(g,f,q,d,t,y)*pHours(q,d,y,t)) =l= pAvailability(g,q)*vCap(g,y)*sum((d,t), pHours(q,d,y,t));

eFuel(zfmap(z,f),y)..
   vFuel(z,f,y) =e= sum((gzmap(g,z),gfmap(g,f),q,d,t), vPwrOut(g,f,q,d,t,y)*pHours(q,d,y,t)*pHeatRate(g,f));

eFuelLimit(c,f,y)$(pfuel_constraints and pMaxFuelLimit(c,f,y) > 0)..
   sum((zcmap(z,c),zfmap(z,f)), vFuel(z,f,y)) =l= pMaxFuelLimit(c,f,y)*1e6;

***Dispatch**********correction*****now this equation is only applied for non dispatch models
eMinGen(g,q,d,t,y)$(Relevant(d) and pmingen_constraints and pGenData(g,"MinLimitShare") > 0 and FD(q,d,t)$pRunDispatch)..
    sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) =g= vCap(g,y)*pGenData(g,"MinLimitShare") ; 
******************



eRampDnLimit(g,q,d,t,y)$(Ramprate(g) and not sFirstHour(t) and pramp_constraints and pRunDispatch=0)..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t-1,y)) - sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) =l= vCap(g,y)*pGenData(g,"RampDnRate");
   

eRampUpLimit(g,q,d,t,y)$(Ramprate(g) and not sFirstHour(t) and pramp_constraints and pRunDispatch=0)..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) - sum(gfmap(g,f), vPwrOut(g,f,q,d,t-1,y)) =l= vCap(g,y)*pGenData(g,"RampUpRate");
   


*******************************Dispatch*************************************************************
    


eRampDispCon1(g,q,d,t,AT,y)$(Ramprate(g) and pramp_constraints and pRunDispatch and mapTS(q,d,t,AT))..
sum(gfmap(g,f),vPwrOut(g,f,q,d,t,y))=e= vPwrOutDispatch(g,q,d,t,AT,y)$mapTS(q,d,t,AT);

eRampUpLimitDisp(g,AT,y)$(not sFirstHourAT(AT) and Ramprate(g) and pramp_constraints and pRunDispatch )..
sum((q,d,t)$mapTS(q,d,t,AT) ,vPwrOutDispatch(g,q,d,t,AT,y))-sum((q,d,t)$mapTS(q,d,t,AT-1) ,vPwrOutDispatch(g,q,d,t,AT-1,y)) =l= (sum((q,d,t)$mapTS(q,d,t,AT),vStUp(g,q,d,t,AT,y)))*pGenData(g,"Capacity")$(pIncludeMinGenCon and MinGenPoint(g))+ vCap(g,y)*pGenData(g,"RampUpRate");

eRampDnLimitDisp(g,AT,y)$(not sFirstHourAT(AT) and Ramprate(g) and pramp_constraints and pRunDispatch )..
sum((q,d,t)$mapTS(q,d,t,AT-1) ,vPwrOutDispatch(g,q,d,t,AT-1,y))-sum((q,d,t)$mapTS(q,d,t,AT) ,vPwrOutDispatch(g,q,d,t,AT,y)) =l= (sum((q,d,t)$mapTS(q,d,t,AT),vShDn(g,q,d,t,AT,y)))*pGenData(g,"Capacity")$(pIncludeMinGenCon and MinGenPoint(g))+vCap(g,y)*pGenData(g,"RampDnRate");






************************************************************************

* Note that we are effectively assuming grid-connected RE generation to be dispatchable. Generally speaking, most RE will be
* dispatched anyway because they have zero cost (i.e., not a different outcome from net load approach, but this allows for
* RE generation to be rejected as well
eVREProfile(gfmap(VRE,f),z,q,d,t,y)$(gzmap(VRE,z) AND FD(q,d,t)$pRunDispatch)..
   vPwrOut(VRE,f,q,d,t,y) + vCurtailedVRE(z,VRE,q,d,t,y) =e= pVREgenProfile(VRE,f,q,d,t)*vCap(VRE,y);


*--- Reserve equations
eSpinningReserveLim(g,q,d,t,y)$(pzonal_spinning_reserve_constraints or psystem_spinning_reserve_constraints)..
   vSpinningReserve(g,q,d,t,y) =l= vCap(g,y)*pGenData(g,"ResLimShare");


***********************************************Dispatch*************************************
eSpinningReserveLim2(g,q,d,t,y)$(pzonal_spinning_reserve_constraints or psystem_spinning_reserve_constraints and pIncludeMinGenCon and MinGenPoint(g) and pRunDispatch and FD(q,d,t))..
   vSpinningReserve(g,q,d,t,y) =l= sum(AT$mapTS(q,d,t,AT),vOn(g,q,d,t,AT,y))*9e15;
   
**************************************************************************
   
* This constraint increases solving time x3
* Reserve constraints include interconnections as reserves too


eSpinningReserveReqCountry(c,q,d,t,y)$(pzonal_spinning_reserve_constraints and FD(q,d,t)$pRunDispatch)..
   sum((zcmap(z,c),gzmap(g,z)),vSpinningReserve(g,q,d,t,y))
 + vUnmetSpinningReserveCountry(c,q,d,t,y)
 + sum((zcmap(z,c),sMapNCZ(z2,z)), pTransferLimit(z2,z,q,y)
                                + vAdditionalTransfer(z2,z,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine")
                                - vFlow(z2,z,q,d,t,y))
   =e= pSpinningReserveReqCountry(c,y) + sum((zcmap(z,c),gzmap(VRE,z),gfmap(VRE,f)), vPwrOut(VRE,f,q,d,t,y))*pVREForecastError;
   

eSpinningReserveReqSystem(q,d,t,y)$(psystem_spinning_reserve_constraints and FD(q,d,t)$pRunDispatch)..
   sum(g, vSpinningReserve(g,q,d,t,y)) + vUnmetSpinningReserveSystem(q,d,t,y) =e= pSpinningReserveReqSystem(y) + sum(gfmap(VRE,f), vPwrOut(VRE,f,q,d,t,y))*pVREForecastError;


ePlanningReserveReqCountry(c,y)$(pplanning_reserve_constraints and pPlanningReserveMargin(c))..
   sum((zcmap(z,c),gzmap(g,z)), vCap(g,y)*pCapacityCredit(g,y))
 + vUnmetPlanningReserveCountry(c,y)
 + sum((zcmap(z,c),sMapNCZ(z2,z)), sum(q,pTransferLimit(z2,z,q,y))/card(q) + vAdditionalTransfer(z2,z,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine"))
   =g= (1+pPlanningReserveMargin(c))*smax((q,d,t), sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y)));


ePlanningReserveReqSystem(y)$(pplanning_reserve_constraints and psys_reserve_margin)..
   sum(g, vCap(g,y)*pCapacityCredit(g,y)) + vUnmetPlanningReserveSystem(y)
   =g= (1+psys_reserve_margin)*smax((q,d,t), sum(z, pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y)));


*--- Transfer equations
eTransferLimit(sTopology(z,z2),q,d,t,y)$(FD(q,d,t)$pRunDispatch)..
   vFlow(z,z2,q,d,t,y) =l= pTransferLimit(z,z2,q,y) + vAdditionalTransfer(z,z2,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine")*pAllowHighTransfer;

eAdditionalTransfer(sTopology(z,z2),y)$pAllowHighTransfer..
   vAdditionalTransfer(z,z2,y) =e=  vAdditionalTransfer(z,z2,y-1) + vBuildTransmission(z,z2,y);

eAdditionalTransfer2(sTopology(z,z2),y)$pAllowHighTransfer..
   vBuildTransmission(z,z2,y)  =e=  vBuildTransmission(z2,z,y);
   

eMaxImportPrice(c,y)$(pallowExports)..
   sum((zcmap(z,c),zext,q,d,t), vImportPrice(z,zext,q,d,t,y)*pHours(q,d,y,t)) =l=
   sum((zcmap(z,c),q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,y,t)*pEnergyEfficiencyFactor(z,y))*pMaxExchangeShare(y,c);

eMaxExportPrice(c,y)$(pallowExports)..
   sum((zcmap(z,c),zext,q,d,t), vExportPrice(z,zext,q,d,t,y)*pHours(q,d,y,t)) =l=
   sum((zcmap(z,c),q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,y,t)*pEnergyEfficiencyFactor(z,y))*pMaxExchangeShare(y,c);


eMaxhourlyImportsshare(c,q,d,t,y)$(pMaxImport<1 and pallowExports and FD(q,d,t)$pRunDispatch)..

   sum((zcmap(z,c), zext), vImportPrice(z,zext,q,d,t,y))  =l= sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pMaxImport * pEnergyEfficiencyFactor(z,y));

eMaxhourlyExportsshare(c,q,d,t,y)$(pMaxExport<1 and pallowExports and FD(q,d,t)$pRunDispatch)..
   sum((zcmap(z,c), zext),vExportPrice(z,zext,q,d,t,y)) =l= sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pMaxExport * pEnergyEfficiencyFactor(z,y));




eExtZoneLimitImport(z,zext,q,d,t,y)$(pallowExports and FD(q,d,t)$pRunDispatch)..
   vImportPrice(z,zext,q,d,t,y)=l= pExtTransferLimitIn(z,zext,q,y);

eExtZoneLimitExport(z,zext,q,d,t,y)$(pallowExports and FD(q,d,t)$pRunDispatch)..
   vExportPrice(z,zext,q,d,t,y)=l= pExtTransferLimitOut(z,zext,q,y);

*--- Storage-specific equations
eStorageCap(st,q,d,t,y)$(pincludeStorage and FD(q,d,t)$pRunDispatch)..
   vStorage(st,q,d,t,y) =l= vCapStor(st,y);

*without this, it builds storage capacity to meet reserve
eStorageCap2(st,q,d,t,y)$(pincludeStorage and FD(q,d,t)$pRunDispatch)..
   vCapStor(st,y) =g= vCap(st,y);
   


*********************IAEA***************************
eStorageCap3(st,q,d,t,y)$(pincludeStorage and (pGenData(st,'fuel1')=16) and FD(q,d,t)$pRunDispatch)..
   vCapStor(st,y) =e= 2*vCap(st,y);
   
eStorageCap4(st,q,d,t,y)$(pincludeStorage and (pGenData(st,'fuel1')=17) and FD(q,d,t)$pRunDispatch)..
   vCapStor(st,y) =e= 4*vCap(st,y);
   
eStorageCap5(st,q,d,t,y)$(pincludeStorage and (pGenData(st,'fuel1')=15) and FD(q,d,t)$pRunDispatch)..
   vCapStor(st,y) =e= 8.9*vCap(st,y);

*****************************************************************

eStorageInjection(st,q,d,t,y)$( pincludeStorage and FD(q,d,t)$pRunDispatch)..
   vStorInj(st,q,d,t,y) =l= vCap(st,y);

*eStorageInjection2(stp,q,d,t,y)$pincludeStorage..
*   vStorInj(stp,q,d,t,y) =l= sum(so, vCap(so,y)*pStoPVProfile(so,q,d,t));

eStorageInjection2(stp,q,d,t,y)$(pincludeStorage and FD(q,d,t)$pRunDispatch)..
   vStorInj(stp,q,d,t,y) =l= sum(gsmap(so,stp), vCap(so,y)*pStoPVProfile(so,q,d,t));
   
*****************Dispatch*****Adding RampRate(St) which need be added in EPM

eRampDownInjLimit(st,q,d,t,y)$(not sFirstHour(t) and pincludeStorage and pramp_constraints and Ramprate(st) and FD(q,d,t)$pRunDispatch)..
   vStorInj(st,q,d,t-1,y) - vStorInj(st,q,d,t,y) =l= pGenData(st,'RampDnRate')*vCap(st,y);

eRampUpInjLimit(st,q,d,t,y)$(not sFirstHour(t) and pincludeStorage and pramp_constraints and Ramprate(st) and FD(q,d,t)$pRunDispatch)..
   vStorInj(st,q,d,t,y) - vStorInj(st,q,d,t-1,y) =l= pGenData(st,'RampUpRate')*vCap(st,y);
   
*********************************************************************




eStorageNet(st,q,d,t,y)$(pincludeStorage and FD(q,d,t)$pRunDispatch )..
   vStorNet(st,q,d,t,y) =e= sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y)) - vStorInj(st,q,d,t,y);

eStorBal(st,q,d,t,y)$(not sFirstHour(t) and pincludeStorage and FD(q,d,t)$pRunDispatch )..
   vStorage(st,q,d,t,y) =e= pStorData(st,"efficiency")*vStorInj(st,q,d,t,y) - sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y)) + vStorage(st,q,d,t-1,y);





* To ensure energy balance in one representative day (For this example only -dispatch- initial storage is at 65% however we could set an input on excel for this

eStorBal1(st,q,d,sFirstHour(t),y)$(pincludeStorage and FD(q,d,t)$pRunDispatch)..
   vStorage(st,q,d,t,y) =e= pStorData(st,"efficiency")*vStorInj(st,q,d,t,y) - sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y))  +0.65*vCapStor(st,y);

********Dispatch allagi parakatw
 eStorBal2(st,q,d,sLastHour(t),y)$(pincludeStorage and pRunDispatch=0)..    vStorage(st,q,d,t,y)  =e= vStorage(st,q,d,t-23,y)-(pStorData(st,"efficiency")*vStorInj(st,q,d,t-23,y) - sum(gfmap(st,f),vPwrOut(st,f,q,d,t-23,y))) ;

 eStorBal3(st,q,d,y)$(pincludeStorage and pRunDispatch=0)..  pStorData(st,"efficiency")* sum(t,vStorInj(st,q,d,t,y)) =e=  Sum((gfmap(st,f),t),vPwrOut(st,f,q,d,t,y));


*********

*********************************************Dispatch (correct initial storage at 65%)***********************************************************



eStorBal4(st,q,sFirstDay(d),sFirstHour(t),y)$(not sfirstquarter(q) and pincludeStorage and pRunDispatch and FD(q,d,t))..
vStorage(st,q,d,t,y) =e= pStorData(st,"efficiency")*vStorInj(st,q,d,t,y)
                               - sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y))+ sum(d2$(ord(d2)=pDays(q-1)),vStorage(st,q-1,d2,t+23,y));


                               
eStorBal5(st,q,d,sFirstHour(t),y)$(not sfirstday(d) and pincludeStorage   and pRunDispatch and FD(q,d,t))..
vStorage(st,q,d,t,y) =e= pStorData(st,"efficiency")*vStorInj(st,q,d,t,y)
                               - sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y))+ vStorage(st,q,d-1,t+23,y);
                               


eStorBal6(st,sLastQuarter(q),sLastDay(d),sLastHour(t),y)$(pincludeStorage and pRunDispatch and FD(q,d,t))..

vStorage(st,q,d,t,y)  =e=   0.65*vCapStor(st,y);



************************Dispatch********************************
eDemandRespCon(q,d,t,y)$(pIncludeDR and  FD(q,d,t)$pRunDispatch)..
sum(z,vDemandShed(z,q,d,t,y))  =l= pMaxDR;

*********************************************************************************************************************



eStorageOutput(st,q,d,t,y)$(pincludeStorage and FD(q,d,t)$pRunDispatch)..
   sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y)) + vSpinningReserve(st,q,d,t,y) =l= vStorage(st,q,d,t,y);

*--- CSP-specific equations
eStorageCSPCap(cs,q,d,t,y)$(pincludeCSP and FD(q,d,t)$pRunDispatch)..
   vStorage(cs,q,d,t,y) =l= vCapStor(cs,y);

eInjCSP(cs,q,d,t,y)$(pincludeCSP and FD(q,d,t)$pRunDispatch)..
   vStorInj(cs,q,d,t,y) =l= vThermalOut(cs,q,d,t,y)*pCSPData(cs,"Thermal Field","Efficiency");
*without this, there is storage injection without storage (injection and withdrawal occur at the same time)

eInjCSP1(cs,q,d,t,y)$(pincludeCSP and FD(q,d,t)$pRunDispatch)..
   vStorInj(cs,q,d,t,y) =l= vCapStor(cs,y);

eThermCSP(cs,q,d,t,y)$(pincludeCSP and FD(q,d,t)$pRunDispatch)..
   vThermalOut(cs,q,d,t,y) =l= vCapTherm(cs,y)*pCSPProfile(cs,q,d,t);

eGenCSP(cs,q,d,t,y)$(pincludeCSP and FD(q,d,t)$pRunDispatch)..
   vThermalOut(cs,q,d,t,y)*pCSPData(cs,"Thermal Field","Efficiency")
 - vStorInj(cs,q,d,t,y) + vStorOut(cs,q,d,t,y)*pCSPData(cs,"Storage","Efficiency")
   =e= sum(gfmap(cs,f), vPwrOut(cs,f,q,d,t,y));



eStorageCSPBal(cs,q,d,t,y)$(not sFirstHour(t) and pincludeCSP and FD(q,d,t)$pRunDispatch)..
   vStorage(cs,q,d,t,y) =e= vStorage(cs,q,d,t-1,y) + vStorInj(cs,q,d,t,y) - vStorOut(cs,q,d,t,y);

eStorageCSPBal1(cs,q,d,sFirstHour(t),y)$(pincludeCSP and FD(q,d,t)$pRunDispatch)..
   vStorage(cs,q,d,t,y) =e= vStorInj(cs,q,d,t,y) - vStorOut(cs,q,d,t,y) ;
**deleted estorageCSPbal2  



*Equation needed in dispatch mode but not for capacity expansion with representative days
*eStorageCSPBal2(cs,q,d,sFirstHour(t),y)$(not sFirstDay(d) and pincludeCSP)..
*   vStorage(cs,q,d,t,y) =e= vStorInj(cs,q,d,t,y) - vStorOut(cs,q,d,t,y) + vStorage(cs,q,d-1,sLastHour,y);


*--- Energy (storage) capacity limits
eCapacityStorLimit(g,y)$pincludeStorage..
   vCapStor(g,y) =l= pStorData(g,"Capacity") + pCSPData(g,"Storage","Capacity");

eCapStorBalance1(eg,y)$(not sStartYear(y) and pincludeStorage)..
   vCapStor(eg,y) =e= vCapStor(eg,y-1) + vBuildStor(eg,y) - vRetireStor(eg,y);

eCapStorBalance2(ng,y)$(not sStartYear(y) and pincludeStorage)..
   vCapStor(ng,y) =e= vCapStor(ng,y-1) + vBuildStor(ng,y);

eCapStorBalance3(ng,sStartYear(y))$pincludeStorage..
   vCapStor(ng,y) =e= vBuildStor(ng,y);

eBuildStorNew(eg)$((pGenData(eg,"StYr") > sStartYear.val) and pincludeStorage)..
   sum(y, vBuildStor(eg,y)) =l= pStorData(eg,"Capacity");

*--- Thermal elements (csp solar field) capacity limits
eCapacityThermLimit(g,y)$pincludeCSP..
   vCapTherm(g,y) =l= pCSPData(g,"Thermal Field","Capacity");

eCapThermBalance1(eg,y)$(not sStartYear(y) and pincludeCSP)..
   vCapTherm(eg,y) =e= vCapTherm(eg,y-1) + vBuildTherm(eg,y) - vRetireTherm(eg,y);

eCapThermBalance2(ng,y)$(not sStartYear(y) and pincludeCSP)..
   vCapTherm(ng,y) =e= vCapTherm(ng,y-1) + vBuildTherm(ng,y);

eCapThermBalance3(ng,sStartYear(y))$pincludeCSP..
   vCapTherm(ng,y) =e= vBuildTherm(ng,y);

eBuildThermNew(eg)$((pGenData(eg,"StYr") > sStartYear.val) and pincludeCSP)..
   sum(y, vBuildTherm(eg,y)) =l= pCSPData(eg,"Thermal Field","Capacity");

*---  Calculate capex for generators with reducing capex
eAnnCapex1(dc,y)$(not sStartYear(y))..
   vAnnCapex(dc,y) =e= vAnnCapex(dc,y-1)
                     + vBuild(dc,y)*pGenData(dc,"Capex")*pCapexTrajectories(dc,y)*pCRF(dc)*1e6
                     + vBuildStor(dc,y)*pStorData(dc,"Capex")*pCapexTrajectories(dc,y)*pCRFsst(dc)*1e3
                     + vBuildStor(dc,y)*pCSPData(dc,"Storage","Capex")*pCapexTrajectories(dc,y)*pCRFcst(dc)*1e3
                     + vBuildTherm(dc,y)*pCSPData(dc,"Thermal Field","Capex")*pCapexTrajectories(dc,y)*pCRFcth(dc)*1e6;
                                                                          ;

eAnnCapex(dc,sStartYear(y))..
   vAnnCapex(dc,y) =e= vBuild(dc,y)*pGenData(dc,"Capex")*pCapexTrajectories(dc,y)*pCRF(dc)*1e6
                     + vBuildStor(dc,y)*pStorData(dc,"Capex")*pCapexTrajectories(dc,y)*pCRFsst(dc)*1e3
                     + vBuildStor(dc,y)*pCSPData(dc,"Storage","Capex")*pCapexTrajectories(dc,y)*pCRFcst(dc)*1e3
                     + vBuildTherm(dc,y)*pCSPData(dc,"Thermal Field","Capex")*pCapexTrajectories(dc,y)*pCRFcth(dc)*1e6;                                                                          ;

*--- Emissions related equations
eCapitalConstraint$pcapital_constraints..
   sum(y, pRR(y)*pWeightYear(y)*sum(ng, pCRF(ng)*vCap(ng,y)*pGenData(ng,"Capex"))) =l= pMaxCapital*1e3;

eZonalEmissions(z,y)..
   vZonalEmissions(z,y) =e=
   sum((gzmap(g,z),gfmap(g,f),q,d,t), vPwrOut(g,f,q,d,t,y)*pHeatRate(g,f)*pFuelCarbonContent(f)*pHours(q,d,y,t));
 

eEmissionsCountry(c,y)$pzonal_co2_constraints..


   sum(zcmap(z,c), vZonalEmissions(z,y))-vYearlyCO2backstop(c,y)=l= pEmissionsCountry(c,y);



eTotalEmissions(y)..
    sum(z, vZonalEmissions(z,y))=e= vTotalEmissions(y);
    
eTotalEmissionsConstraint(y)$pSystem_CO2_constraints..
    vTotalEmissions(y)-vYearlySysCO2backstop(y) =l= pEmissionsTotal(y);
   
eSimultComDecom(eg,ng,y)$( pIncludeDecomCom and mapGG(eg,ng))..          vBuild(ng,y) -  vRetire(eg,y) =l= 0;




*******************Dispatch****************





eMinGenPoint(g,q,d,t,y)$(pRunDispatch and pIncludeMinGenCon and MinGenPoint(g) and FD(q,d,t) )..  
   sum(gfmap(g,f),vPwrOut(g,f,q,d,t,y))            =g= pGenData(g,"MinGenPoint")*pGenData(g,"Capacity")*sum(AT$mapTS(q,d,t,AT),vOn(g,q,d,t,AT,y));
   
eMaxGenPoint(g,q,d,t,y)$(pRunDispatch and pIncludeMinGenCon and MinGenPoint(g) and FD(q,d,t) )..  
   sum(gfmap(g,f),vPwrOut(g,f,q,d,t,y))            =l= pGenData(g,"Capacity")*sum(AT$mapTS(q,d,t,AT),vOn(g,q,d,t,AT,y));



eStUp1(g,AT,y)$(ord(AT)>1 and pRunDispatch and pIncludeMinGenCon and MinGenPoint(g) )..                    
   sum((q,d,t)$mapTS(q,d,t,AT),vStUp(g,q,d,t,AT,y)-vShDn(g,q,d,t,AT,y))   =e= sum((q,d,t)$mapTS(q,d,t,AT),vOn(g,q,d,t,AT,y) )- sum((q,d,t)$mapTS(q,d,t,AT-1),vOn(g,q,d,t,AT-1,y));

eStUp2(g,AT,y)$(ord(AT)=1 and pRunDispatch and pIncludeMinGenCon and MinGenPoint(g) )..                    
   sum((q,d,t)$mapTS(q,d,t,AT),vStUp(g,q,d,t,AT,y)-vShDn(g,q,d,t,AT,y))   =e= sum((q,d,t)$mapTS(q,d,t,AT),vOn(g,q,d,t,AT,y)) -pGenData(g,"InitialOn");

eStUp3(g,AT,y)$(pRunDispatch and pIncludeMinGenCon and MinGenPoint(g) )..                                   
   sum((q,d,t)$mapTS(q,d,t,AT),vStUp(g,q,d,t,AT,y)+vShDn(g,q,d,t,AT,y))   =l= 1;












*******************Dispatch****************

*  Minimum Up time Constraints

*  first day first month 


eMinUpTime_con1(g,y)$(pRunDispatch and pIncludeMUDT and MinGenPoint(g)  AND pGenData(g,"HoursOn") AND pGenData(g,"HoursOff")=0   AND pGenData(g,"minUT")>0.999  )..
             sum(AT$ (ord(AT) < (pGenData(g,"minUT")-pGenData(g,"HoursOn")+1)),sum((q,d,t)$mapTS(q,d,t,AT),  1-vOn(g,q,d,t,AT,y))) =e= 0;


eMinUpTime_con2(g,AT2,y)$(pRunDispatch and pIncludeMUDT and MinGenPoint(g)  AND pGenData(g,"minUT")>0.999 AND ord(AT2)>(pGenData(g,"minUT") -pGenData(g,"HoursOn")+1)$pGenData(g,"HoursOn")     AND ord(AT2)<(card(AT2)-pGenData(g,"minUT")+1) )..
 sum(AT$( (ord(AT) >= ord(AT2)) AND (ord(AT) < ord(AT2) + pGenData(g,"minUT"))),sum((q,d,t)$mapTS(q,d,t,AT), vOn(g,q,d,t,AT,y))) =g= pGenData(g,"minUT")*sum((q,d,t)$mapTS(q,d,t,AT2),vStUp(g,q,d,t,AT2,y));




eMinDnTime_con1(g,y)$(pRunDispatch and pIncludeMUDT and MinGenPoint(g)  AND pGenData(g,"HoursOn")=0 AND pGenData(g,"HoursOff")   AND pGenData(g,"minDT")>0.999  )..
             sum(AT$ (ord(AT) < (pGenData(g,"minDT")-pGenData(g,"HoursOff")+1)),sum((q,d,t)$mapTS(q,d,t,AT),  vOn(g,q,d,t,AT,y))) =e= 0;



eMinDnTime_con2(g,AT2,y)$(pRunDispatch and pIncludeMUDT and MinGenPoint(g)  AND pGenData(g,"minDT")>0.999 AND ord(AT2)>(pGenData(g,"minDT") -pGenData(g,"HoursOff")+1)$pGenData(g,"HoursOff")     AND ord(AT2)<(card(AT2)-pGenData(g,"minDT")+1) ) ..
 sum(AT$( (ord(AT) >= ord(AT2)) AND (ord(AT) < ord(AT2) + pGenData(g,"minDT"))),sum((q,d,t)$mapTS(q,d,t,AT),1- vOn(g,q,d,t,AT,y))) =g= pGenData(g,"minDT")*sum((q,d,t)$mapTS(q,d,t,AT2),vShDn(g,q,d,t,AT2,y));









*********************************************************************Hydrogen production equations********************************************************************

*  Total capacity of H2 plants at 1st year of optimization is equal to pre-existing capacity plus capacity being bulit minus retired capacity
*Checked
eCapBalanceH2(hh,sStartYear(y))$(pIncludeH2)..
   vCapH2(hh,y) =e= pH2Data(hh,"Capacity")$(eh(hh) and (pH2Data(hh,"StYr") <= sStartYear.val))
               + vBuildH2(hh,y) - vRetireH2(hh,y);

*  Total capacity of existing H2 palnts is equal to capacity over previous year plus capacity being built in current year minus retired capacity in current year
*Checked
eCapBalance1H2(eh,y)$(not sStartYear(y) and pIncludeH2)..
   vCapH2(eh,y) =e= vCapH2(eh,y-1) + vBuildH2(eh,y) - vRetireH2(eh,y);

*Total capacity of candidate H2 plants is equal to total capacity at previous year plus capacity being built in current year minus retired capacity in current year
*Checked
eCapBalance2H2(nh,y)$(not sStartYear(y) and pIncludeH2)..
   vCapH2(nh,y) =e= vCapH2(nh,y-1) + vBuildH2(nh,y);

* New H2 plants can be buuilt only after the StYr; the newly built capacity need to be less than declared capacity
*Checked
eBuildNewH2(eh)$(pH2Data(eh,"StYr") > sStartYear.val and pIncludeH2)..
    sum(y, vBuildH2(eh,y)) =l= pH2Data(eh,"Capacity");

* Total built H2 generation capacity need to be less than maxtotal built
*Checked
eMaxBuildTotalH2(nh)$(pIncludeH2)..
   sum(y, vBuildH2(nh,y)) =l= pH2Data(nh,"MaxTotalBuild");

* (Integer units ) Built capacity each year is equal to unit size
*Checked
eBuiltCapH2(nh,y)$(pH2Data(nh,"DescreteCap") and pIncludeH2)..
   vBuildH2(nh,y) =e= pH2Data(nh,"UnitSize")*vBuiltCapVarH2(nh,y);

* (Integer units ) Retired capacity each year is equal to unit size
*Checked
eRetireCapH2(eh,y)$(pH2Data(eh,"DescreteCap") and (y.val <= pH2Data(eh,"RetrYr")) and pIncludeH2)..
   vRetireH2(eh,y) =e= pH2Data(eh,"UnitSize")*vRetireCapVarH2(eh,y);

*  Maximum capacity factor of H2 production based on availability
*Checked
eMaxCF_H2(hh,q,y)$(pIncludeH2)..
   sum((d,t), vH2PwrIn(hh,q,d,t,y)*pHours(q,d,y,t)) =l= pAvailabilityH2(hh,q)*vCapH2(hh,y)*sum((d,t), pHours(q,d,y,t));

*       mmBTU of H2 produced
eFuel_H2(c,q,y)$(pIncludeH2)..
*                   mmBTU            mmBTU                    mmBTU                                                                     -MWe-                Hr                mmBTU/MWhe 
    sum(zcmap(z,c), pExternalH2(z,q,y)-vUnmetExternalH2(z,q,y)$pExternalH2(z,q,y)+vFuelH2Quarter(z,q,y)) =e= sum( (zcmap(z,c),h2zmap(hh,z), d,t), vH2PwrIn(hh,q,d,t,y)*pHours(q,d,y,t)*pH2Data(hh,"HeatRate"));

eFuel_H2_2(c,z,y)$(pIncludeH2)..
    sum((zcmap(z,c),q),vFuelH2Quarter(z,q,y)) =e=  sum(zcmap(z,c),vFuelH2(z,y));

eMaxH2PwrInjection(hh,q,d,t,y)$pIncludeH2..
    vH2PwrIn(hh,q,d,t,y)  =l= vCapH2(hh,y);

* The amount of hydrogen fuel that can be used for electricity generation can not be more than the amount of H2 that was produced from VRE curtailment
eFuelLimitH2(c,f,y)$(pFuelData(f) and pIncludeH2)..
   sum((zcmap(z,c)),  vFuel(z,f,y)) =e=  sum((zcmap(z,c)), vFuelH2(z,y));


*When the H2 production flag is off don't account for H2 fuel
eFuelLimitH2_2(c,f,y)$(pFuelData(f) and pIncludeH2=0)..
   sum((zcmap(z,c)),  vFuel(z,f,y)) =e=  0;


eRampDnLimitH2(hh,q,d,t,y)$(RamprateH2(hh) and not sFirstHour(t) and pramp_constraints and pIncludeH2 and FD(q,d,t)$pRunDispatch)..
    vH2PwrIn(hh,q,d,t-1,y) -  vH2PwrIn(hh,q,d,t,y) =l= vCapH2(hh,y)*pH2Data(hh,"RampDnRate");

eRampUpLimitH2(hh,q,d,t,y)$(RamprateH2(hh) and not sFirstHour(t) and pramp_constraints and pIncludeH2 and FD(q,d,t)$pRunDispatch)..
   vH2PwrIn(hh,q,d,t,y) -  vH2PwrIn(hh,q,d,t-1,y) =l= vCapH2(hh,y)*pH2Data(hh,"RampUpRate");

*Calculation of Annualized CAPEX for electrolyzers
eAnnCapexH2_1(dcH2,y)$(not sStartYear(y) and pIncludeH2)..
   vAnnCapexH2(dcH2,y) =e= vAnnCapexH2(dcH2,y-1)
                     + vBuildH2(dcH2,y)*pH2Data(dcH2,"Capex")*pCapexTrajectoriesH2(dcH2,y)*pCRFH2(dcH2)*1e6;

eAnnCapexH2(dcH2,sStartYear(y))$pIncludeH2..
   vAnnCapexH2(dcH2,y) =e= vBuildH2(dcH2,y)*pH2Data(dcH2,"Capex")*pCapexTrajectoriesH2(dcH2,y)*pCRFH2(dcH2)*1e6;                                                                          ;

eRE2H2_4(z,q,d,t,y)$(pIncludeH2 and FD(q,d,t)$pRunDispatch)..
  sum(h2zmap(hh,z),vH2PwrIn(hh,q,d,t,y))=e=vPwrREH2(z,q,d,t,y);
 

eRE2H2_3(z,q,d,t,y)$(pIncludeH2 and FD(q,d,t)$pRunDispatch)..
 vPwrREH2(z,q,d,t,y) =e= sum((gfmap(RE,f),gzmap(RE,z)),vREPwr2H2(RE,f,q,d,t,y));

eRE2H2_2(z,q,d,t,y)$(pIncludeH2 and FD(q,d,t)$pRunDispatch)..
vPwrREGrid(z,q,d,t,y)=e=sum((gfmap(RE,f),gzmap(RE,z)),vREPwr2Grid(RE,f,q,d,t,y));

eRE2H2(RE,f,q,d,t,y)$(pIncludeH2 and FD(q,d,t)$pRunDispatch)..
 vPwrOut(RE,f,q,d,t,y)=e=vREPwr2Grid(RE,f,q,d,t,y)+vREPwr2H2(RE,f,q,d,t,y);

*For example equation below is deactivated when H2 is on and is replaced by eVREProfile2

***********************************end of H2 equations section**************************************




Model PA /
   eNPVCost
   eYearlyTotalCost
   eYearlyFixedCost
   eYearlyVariableCost
   eYearlySpinningReserveCost
   eYearlyUSECost
   eYearlyUnmetReserveCostCountry
   eYearlyCarbonCost
   eDemSupply
   eCapBalance
   eCapBalance1
   eCapBalance2
   eBuildNew
   eMaxBuildTotal




   eMinGenRE
   eMaxCF
   eMinGen
   
   eFuel
   eRampUpLimit
   eRampDnLimit
   eSpinningReserveLim
*  eResLim_CSP
   eJointResCap
   eSpinningReserveReqCountry
   eSpinningReserveReqSystem
   ePlanningReserveReqCountry
   
   eVREProfile
   eFuelLimit
   eCapitalConstraint
   eZonalEmissions
   eEmissionsCountry
   eTotalEmissions
   eTotalEmissionsConstraint
   eYearlyCurtailmentCost
   eYearlyCO2backstopCost
   
   eBuiltCap
   eRetireCap
   
   eTransferLimit
   eYearlyTransmissionAdditions
   eAdditionalTransfer
   eAdditionalTransfer2
   eMaxImportPrice
   eMaxExportPrice  
   eMaxhourlyImportsshare
   eMaxhourlyExportsshare
   eYearlyTradeCost
   eExtZoneLimitImport
   eExtZoneLimitExport   


   ePlanningReserveReqSystem
   eYearlyUnmetReserveCostSystem
   eYearlySurplusCost

   
   eStorageCap
   eStorageCap2
   eStorageInjection
   eStorageInjection2
   eRampDownInjLimit
   eRampUpInjLimit
   eStorageNet
   eStorBal
   eStorBal1

  
   eStorageOutput
   
   eStorageCSPCap
   eInjCSP
   eInjCSP1
   eThermCSP
   eGenCSP
  
   eStorageCSPBal
   eStorageCSPBal1
   
   eCapacityStorLimit
   eCapStorBalance1
   eCapStorBalance2
   eCapStorBalance3
   eBuildStorNew
   
   eCapacityThermLimit
   eCapThermBalance1
   eCapThermBalance2
   eCapThermBalance3
   eBuildThermNew
   
   eAnnCapex1
   eAnnCapex
   

   eSimultComDecom

* variable limited domains
   vPwrOut(sPwrOut)
   vExportPrice(sExportPrice)
   vImportPrice(sImportPrice)
   vAdditionalTransfer(sAdditionalTransfer)
   vFlow(sFlow)
   vSpinningReserve(sSpinningReserve)
   
******************************IAEA*************************
eStorageCap3   
eStorageCap4
eStorageCap5




****************************Dispatch************************
eStorBal4
eStorBal5
eStorBal6


eRampDispCon1
eRampUpLimitDisp
eRampDnLimitDisp


eSpinningReserveLim2
eStUpCosts

eDemandRespCon
eYearlyCostDemandShed




eMinGenPoint
eMaxGenPoint
*


eStUp1
eStUp2
eStUp3

eMinUpTime_con1
eMinUpTime_con2
eMinDnTime_con1
eMinDnTime_con2


***************************Equations for H2 model*******************************
   vH2PwrIn(sH2PwrIn)

   eH2UnservedCost
   eBuildNewH2
   eCapBalance1H2
   eCapBalance2H2
   eCapBalanceH2
   eMaxBuildTotalH2
   eBuiltCapH2
   eRetireCapH2
*eDemSupplyH2
   eMaxCF_H2
   eFuel_H2
   eFuel_H2_2
   eRampDnLimitH2
   eRampUpLimitH2
   eFuelLimitH2
   eFuelLimitH2_2
*eYearlyCurtailmentCost2

   eRE2H2
   eRE2H2_2
   eRE2H2_3
   eRE2H2_4
   eMaxH2PwrInjection

   eAnnCapexH2_1
   eAnnCapexH2
  
/;
