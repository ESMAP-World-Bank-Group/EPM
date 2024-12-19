$onText

December 1, 2018

$offText
$offEolCom
$eolCom //
*
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
*********Hydrgen specific addition***********
   hh        'Hydrogen production units'
**************************************
    s       'scenarios'
;

alias (z,z2), (g,g2), (d,d2), (t,t2), (q,q2), (y,yy);


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
   nSt(g)      'non storage technologies'
   cg(g)       'coal and generators'
   ncg(g)      'not coal generators'
   PVoW(g)     'PV and onshore wind generators'
   cgg(g)      'coal and gas generators'

*sConversionPlants(g)
*sConversionPlantsMap(g,g2)
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

************************************************************************

;

* Time
Sets
   ghdr         'Header for pGenData' / BuildLimitperYear, Capacity, Capex, DescreteCap, FOMperMW, MaxTotalBuild,
                                        MinLimitShare, MinTotalBuild, Overloadfactor, RampDnRate, RampUpRate, ReserveCost, ResLimShare, RetrYr, StYr, UnitSize /
   shdr         'Header for pStorData' / Capacity, Capex, FixedOM, Efficiency /
   csphrd       'Header for pCSPData' / Storage, "Thermal Field" /
   thdr         'Header for pNewTransmission' / CapacityPerLine, CostPerLine, Life, MaximumNumOfLines /
   Relevant(d)  'relevant day and hours when MinGen limit is applied'


***********************Hydrogen model related sets*********************
   hhdr         'Header for pH2Data' / StYr, RetrYr,  Capacity, Capex, HeatRate,VOM, FOMperMW,  Efficiency,  BuildLimitperYear, MaxTotalBuild,  DescreteCap,
                                         RampUpRate, RampDnRate,   ResLimShare, UnitSize, Life /
   h2Index      'Index of hydrogen fuels'    /HydrogenIndex/
************************************************************************************************************
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
   szfuellim(z)
**************Hydrogen production model related sets*******************************   
    h2zmap(hh,z)
    ft
*************************************************************************************
    ;
    
* Implicit variable domain
Sets
   sPwrOut(g,f,q,d,t,y,s)        'valid tuples for vPwrOut'
   sExportPrice(z,q,d,t,y,s)     'valid tuples for vExportPrice'
   sImportPrice(z,q,d,t,y,s)     'valid tuples for vImportPrice'
   sAdditionalTransfer(z,z2,y) 'valid tuples for vAdditionalTransfer'
   sFlow(z,z2,q,d,t,y,s)         'valid tuples for vFlow'
   sReserve(g,q,d,t,y,s)         'valid tuples for vReserve'
   sH2PwrIn(hh,q,d,t,y)
;

Singleton sets
   sStartYear(y)
   sFirstHour(t)
   sLastHour(t)
   sFirstDay(d)
   sLastDay(d)
   sFirstQuarter(q)
   sLastQuarter(q)
;

Parameters
   pCostOfCurtailment               'Cost of curtailment'
   pCostOfCO2backstop               'Cost of climate backstop techno in $ per ton of CO2 '
   pHours(q<,d<,t<)              'duration of each block'
   pWeightYear(y)                   'weight on years'
* Generators
   pGenData(g,ghdr)                 'generator data'
   pAvailability(g,s,q)               'Availability by generation type and season or quarter in percentage - need to reflect maintenance'
   pCapexTrajectories(g,y)          'capex trajectory  final'
* Exchanges
   pTransferLimit(z,z2,q,y)         'Transfer limits by quarter (seasonal) and year between zones'
   pMaxPriceImportShare(y,c)        'Max share of imports by country'
   pAllowHighTransfer
   pTradePrice(z,q,d,y,t)           'trade price - export or import driven by prices [assuming each zone in a country can only trade with one external zone]'
   pMaxImport
   pMaxExport
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
* Reserves
   pReserveReqLoc(c,y)              'Spinning reserve requirement local at country level (MW)  -- for isolated system operation scenarios'
   pReserveReqSys(y)                'Spinning reserve requirement systemwide (MW) -- for integrated system operation scenarios'
   pReserveMargin(c)                'country reserve margin'
   pCapacityCredit(g,y)             'Share of capacity counted towards planning reserves'
* CO2
   pCarbonPrice(y)                  'Carbon price in USD per ton of CO2eq'
   pFuelCarbonContent(f)            'Fuel carbon content in tCO2 per MMBTu'
   pEmissionsZone(c,y)              'Maximum zonal emissions allowed per country and year in tns'
   pEmissionsTotal(y)               'Maximum system emissions allowed year in tns'
* Economic parameters
   pRR (y)                          'accumulated return rate factor'
   pWACC                            'Weighted Average Cost of Capital'
   pCRF (G)                         'capital recovery factor'
   pCRFsst(g)                       'capital recovery factor storage'
   pCRFcst(g)                       'capital recovery factor CSP storage'
   pCRFcth(g)                       'capital recovery factor CSP thermal'
   pVOLL                            'VOLL'
   pReserveVoLL                     'Reserve VoLL per MW'
   pSpinReserveVoLL                 'Spin Reserve VoLL per MWh'
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

***********Turkey model specific parameters***********************

*  pIncludeRELimit            'Include minimum annual RE share constraint'
*  pIncludeSNSP               'Include systm non synchronous penetration limit'
*  pIncludeMinCap             'Include minimum capacity limit'
  pIncludeMaxCap             'Include maximum capacity limit'
  pIncludeMaxAnnualBuild     'Include max annual build limit'
*  pIncludeEconRetire         'Include economic retirement'
*  pRELimit(y)                'Annual RE share values'
*  pSNSP(y)                   'System non synchronous penetration limit'
  pMinCap(c,f,y)             'Minimum capacity installed- country wide limit'
  pMaxCap(c,f,y)             'Maximum capacity installed- country wide limit'
  pAnnualMaxBuild(g,y)       'Maximum capacity installed per year'
*  pMaxRetire(c,y)            'Maximum retired capacity on country level'
*  pMinCF(g,q)                'Minimum capacity factor'
  pStorPenalty(g)            'Small penalty to avoid simultaneous charging/discharging of storage ($/MWh)'
  pDays(q)
  pIncludeAnnualProfiles     'Annual profiles for Availability and RE'
*  pAvailabilityY(g,y,q)      'Availability by generation type and season or quarter in percentage - need to reflect maintenance'

pAnnualFuelBuildLimit(f,y)
**********************************************************


**************H2 model related parameters

pH2Data(hh,hhdr)                 'hydrogen generating units'
pIncludeH2                       'Flag to activate hydrogen related equations'
pAvailabilityH2(hh,q)            'Availability by generation type and season or quarter in percentage - need to reflect maintenance'
pFuelData(f)
pCRFH2(hh)                       'capital recovery factor'
pCapexTrajectoriesH2(hh,y)       'CAPEX trajectories for hydrogen generation units'
pVarCostH2(hh,y)                  'Variable cost - H2 production'
pExternalH2(z,q,y)                'mmBTUs of H2 as external demand that need to be met'
pH2UnservedCost                  'Cost of external H2 unserved'

************************************************************


pProbaScenarios(s)

;
Positive Variables
   vPwrOut(g,f,q,d,t,y,s)      'generation dispatch for aggregated generators in MW'
   vUSE(z,q,d,t,y,s)           'unserved demand'
   vSurplus(z,q,d,t,y,s)       'surplus generation'
   vYearlySurplus(z,y)

   vCap(g,y)                 'total capacity in place accounting for legacy, new and retired plants (MW)'
   vBuild(g,y)               'Build (MW)'
   vRetire(g,y)              'Retire (MW)'

   vFlow(z,z2,q,d,t,y,s)       'flow from z to z2 in MW'
   vImportPrice(z,q,d,t,y,s)   'external price driven import'
   vExportPrice(z,q,d,t,y,s)   'external price driven export'

   vFuel(z,f,y,s)              'annual fuel in MMBTU'

   vAnnCapex(g,y)            'Annualized capex for capacity vCap installed in Year y  carried through planning horizon'

   vThermalOut(g,q,d,t,y,s)    'Generation from thermal element (CSP solar field in MW)'
   vCapStor(g,y)             'Total capacity of storage installed (MWh)'
   vCapTherm(g,y)            'Total thermal CSP solar field capacity installed (MW)'
   vStorage(g,q,d,t,y,s)       'Storage level  (MW)'
   vStorInj(g,q,d,t,y,s)       'Storage injection  (MW)'
   vStorOut(g,q,d,t,y,s)       'Storage output (MW)'
   vBuildStor(g,y)           'Build storage variable (MWh)'
   vRetireStor(g,y)          'Retire storage variable (MWh)'
   vBuildTherm(g,y)          'Build thermal elements (CSP solar field) variable (MW)'
   vRetireTherm(g,y)         'Retire thermal elemtns (CSP solar field) variable (MW)'

   vReserve(g,q,d,t,y,s)       'spinning reserve (MW)'
   vUnmetReserve(c,y,s)        'unserved reserve -- capacity reserve'
   vUnmetSysReserve(y)       'unmet reserve for system -- capacity reserve'
   vUnmetSpinLoc(c,q,d,t,y,s)  'unmet spinning reserve - local constraint'
   vUnmetSpin(q,d,t,y,s)       'unmet spinning reserve - system constraint'

   vZonalEmissions(z,y,s)      'average CO2eq emissions per year and zone in tons per MWh'
   vTotalEmissions(y,s)        'total regional CO2eq emissions per year in tons'
   vYearlyCO2backstop(c,y,s)      'CO2 emissions above the constraint by zone (t)'
   vYearlySysCO2backstop(y,s)     'system CO2 emissions above the constraint(t)'


   vAdditionalTransfer(z,z2,y)        'additional transfer limit'
   vYearlyTransmissionAdditions(z,y)  'added transmission cost (not included in cost of generation)'
   vYearlyCurtailmentCost(z,y)        ' Annual VRE curtailment in MWh'
   vCurtailedVRE(z,g,q,d,t,y,s)         'Curtailed power in MW'
   vTotalFuelBuildCapacity(c,y,f)

*************TURKEY MODEL SPECIFIC VARIABLES***************************
   vYearlyStorPenalty(z,y)

*********************************************

*************H2 MODEL SPECIFIC VARIABLES***************************


vCapH2(hh,y)                'total capacity in place accounting for legacy, new and retired plants (MW)'
vBuildH2(hh,y)              'Build (MW)'
vRetireH2(hh,y)             'Retire (MW)'
vNetCurtailedVRE(z,q,d,t,y) 'Curtailed VRE as an outcome of not being used for H2 production'
vH2PwrIn(hh,q,d,t,y)        'Power drawn by H2 plants for H2 production in MW'

vFuelH2(z,y)                'Annual H2 production in MMBTU'

vAnnCapexH2(hh,y)           'Annualized CAPEX of H2 producing technologies'


vPwrREH2(z,q,d,t,y)         'Generation from RE generators that goes to H2 production'
vPwrREGrid(z,q,d,t,y)       'Generation from RE generators that goes to the grid'

vREPwr2H2(g,f,q,d,t,y)      'Generation from RE generators that goes to H2 production'
vREPwr2Grid(g,f,q,d,t,y)    'Generation from VRE generators that goes to the grid'

vFuelH2Quarter(z,q,y)       'H2 fuel saved for H2 electricity generation on a quarterly basis'
vUnmetExternalH2(z,q,y)     'mmBTU of external H2 that cant be produced'
        
*****************************************************************
vBuildTransmission(z,z2,y)

;

Free Variable
   vNPVCost                     'discounted total system cost'
   vStorNet(g,q,d,t,y,s)
   vYearlyTotalCost(c,y)
   vYearlyFixedCost(z,y)
   vYearlyVariableCost(z,y)
   vYearlyUSECost(z,y)
   vYearlyTradeCost(z,y)
   vYearlyReserveCost(z,y)
   vYearlyUSRCost(c,y)          'country unmet spinning reserve'
   vYearlyCarbonCost(z,y)       'country carbon cost'
   vYearlySysUSRCost(y)         'system unmet planing reserve'
   vYearlyCO2backCost(c,y)      'cost of CO2 backstop'
   vYearlySysCO2backCost(y) 'cost of CO2 baskstop at system level'
   vYearlyH2UnservedCost        'Annual Cost of external H2 unserved in USD'
;

Integer variable
vBuiltCapVar(g,y)
vRetireCapVar(g,y)


;
Variable

*******H2 production model integer variables******************
   vBuiltCapVarH2(hh,y)

   vRetireCapVarH2(hh,y)


********************************************
;

Equations
   eNPVCost                        'objective function'
   eYearlyTotalCost(c,y)
   eYearlyFixedCost(z,y)
   eYearlyVariableCost(z,y)
   eYearlyReserveCost(z,y)
   eYearlyUSECost(z,y)
   eYearlyTradeCost(z,y)
   eYearlyUSRCost(c,y)
   eYearlySysUSRCost(y)            'unmet reserve system'
   eYearlyCarbonCost(z,y)

   eDemSupply(z,q,d,t,y,s)           'demand balance'
   eMaxhourlyExportsshare(c,q,d,t,y,s) 'max exports to an external zone (hourly limit)'
   eMaxhourlyImportsshare(c,q,d,t,y,s) 'max imports from an external zone  (hourly limit)'

   eCapBalance(g,y)                'capacity balance'
   eCapBalance1(g,y)               'capacity balance'
   eCapBalance2(g,y)               'capacity balance'
   eBuildNew(g)
   eBuiltCap(g,y)                  'built capacity'
   eRetireCap(g,y)                 'retired capacity'

   eMaxBuildTotal(g)               'max build over all years'
   eMinBuildTotal(g)
   eMinGenRE(c,y,s)                  'Min Generation of RE after a target year at country level'

   eMaxCF(g,q,y,s)                   'max capacity factor'
   eMinGen(g,q,d,t,y,s)              'Minimum generation limit for new generators'

   eFuel(z,f,y,s)                    'fuel balance'
   eFuelLimit(c,f,y,s)               'fuel limit at country level'

   eRampUpLimit(g,q,d,t,y,s)         'Ramp up limit'
   eRampDnLimit(g,q,d,t,y,s)         'Ramp down limit'

   eReslim(g,q,d,t,y,s)              'Reserve limit as a share of capacity'
   eJointResCap(g,q,d,t,y,s)         'Joint reserve and generation limit'
   eResReqLocal(c,q,d,t,y,s)         'Country spinning reserve requirement'
   eResReqSystem(q,d,t,y,s)          'System spinning reserve requirement'
   eSystemMinCapReserve(y)         'Min system planning reserve requirement'
   eZonalMinCapReserve(c,y,s)        'Minimum capacity reserve over peak demand at country level'

   eTransferLimit(z,z2,q,d,t,y,s)    'Transfer limits'
   eVREProfile(g,f,z,q,d,t,y,s)      'VRE generation restricted to VRE profile'
   eMaxImportPrice(c,y,s)            'import limits'
   eYearlySurplusCost(z,y)
   eAdditionalTransfer(z,z2,y)
   eAdditionalTransfer2(z,z2,y)
   eYearlyTransmissionAdditions (z,y)

   eAnnCapex(g,y)                  'Annualized capex'
   eAnnCapex1(g,y)


   eCapitalConstraint              'capital limit expressed by pMaxCapital in billion USD'
   eZonalEmissions(z,y,s)            'CO2eq emissions by zone and year in tons'
   eEmissionsCountry(c,y,s)          'constraint on country CO2eq emissions'
   eTotalEmissions(y,s)              'total regional CO2eq emissions by year in tons'
   eEmissionsTotal(y,s)              'Total CO2eq emissions by year in tons'
   eYearlyCO2backstopCost(c,y)     'co2 backstop cost in USD'
   eYearlySysCO2backstopCost(y)
   eTotalEmissionsConstraint(y,s)         'constraint on total CO2eq emissions by year in tons'


   eRampDownInjLimit(g,q,d,t,y,s)
   eRampUpInjLimit(g,q,d,t,y,s)

   eYearlyCurtailmentCost(z,y)
*  eYearlyCurtailmentCost2(z,y)

   eStorBal(g,q,d,t,y,s)
   eStorBal1(g,q,d,t,y,s)

   eStorageOutput(g,q,d,t,y,s)
   eStorageInjection(g,q,d,t,y,s)
   eStorageInjection2(g,q,d,t,y,s)
   eStorageNet(g,q,d,t,y,s)

   eStorageCap(g,q,d,t,y,s)
   eStorageCap2(g,q,d,t,y)         'storage capacity (energy) must be at least 1 hour if installed'

   eStorageCSPCap(g,q,d,t,y,s)
   eInjCSP(g,q,d,t,y,s)
   eInjCSP1(g,q,d,t,y,s)
   eThermCSP(g,q,d,t,y,s)
   eGenCSP(g,q,d,t,y,s)
   eStorageCSPBal(g,q,d,t,y,s)
   eStorageCSPBal1(g,q,d,t,y,s)
   eStorageCSPBal2(g,q,d,t,y,s)

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
   eH2UnservedCost(z,y)

*eConvertedPlants(g,y)
*******************************************************************

***********************************Hydrogen production model**************************************
eCapBalanceH2(hh,y)
eCapBalance1H2(hh,y)
eCapBalance2H2
eBuildNewH2(hh)
eMaxBuildTotalH2(hh)
eBuiltCapH2(hh,y)
eRetireCapH2(hh,y)
*eDemSupplyH2(z,q,d,t,y)
eMaxCF_H2(hh,q,y)
eFuel_H2(z,q,y)
eFuel_H2_2(z,y)
eRampDnLimitH2(hh,q,d,t,y)
eRampUpLimitH2(hh,q,d,t,y)
eFuelLimitH2(c,f,y)
eFuelLimitH2_2(c,f,y)
eAnnCapexH2_1(hh,y)
eAnnCapexH2(hh,y)
eVRE2H2(g,f,z,q,d,t,y)
eVRE2H2_2(z,g,f,q,d,t,y)
eVRE2H2_3(z,q,d,t,y)
eVRE2H2_4(z,q,d,t,y)
eRE2H2(g,f,q,d,t,y)
eRE2H2_2(z,q,d,t,y)
eRE2H2_3(z,q,d,t,y)
eRE2H2_4(z,q,d,t,y)
eMaxH2PwrInjection(hh,q,d,t,y)
*eSNSP2(q,d,t,y)
eFuel_H2_3(z,q,y)

eTotalFuelBuildCapacity(c,y,f)

eAnnualFuelBuildLimit(f,y)
***********************************************************************************************
;

*---    Objective function
* TOCHECK: I remove YearlySysCO2backCost(y) because of bugs
eNPVCost..
   vNPVCost =e= sum(y, pRR(y)*pWeightYear(y)*(sum(c, vYearlyTotalCost(c,y)) + vYearlySysUSRCost(y)));

*---  Cost equations
* Note capex is full capex in $m per MW. Also note VarCost includes fuel cost and VOM -
* essentially the short run marginal cost for the generator

eYearlyTotalCost(c,y)..
   vYearlyTotalCost(c,y) =e= vYearlyUSRCost(c,y)+ vYearlyCO2backCost(c,y)
                           + sum(zcmap(z,c), vYearlyFixedCost(z,y)
                                           + vYearlyVariableCost(z,y)
                                           + vYearlyReserveCost(z,y)
                                           + vYearlyUSECost(z,y)
                                           + vYearlyCarbonCost(z,y)
                                           + vYearlyTradeCost(z,y)
                                           + vYearlyTransmissionAdditions(z,y)
                                           + vYearlyCurtailmentCost(z,y)
                                           + vYearlySurplus(z,y)
**********************************************Turkey model specific variable******************
                                           + vYearlyStorPenalty(z,y)$pIncludeStorage
                                           + vYearlyH2UnservedCost(z,y))$pIncludeH2;
                                           
***********************************************************************************
eYearlyFixedCost(z,y)..
   vYearlyFixedCost(z,y) =e= sum(gzmap(ndc,z), pCRF(ndc)*vCap(ndc,y)*pGenData(ndc,"Capex")*1e6)
                           + sum(gzmap(ndc,z)$(not cs(ndc)), pCRFsst(ndc)*vCapStor(ndc,y)*pStorData(ndc,"Capex")*1e3)
                           + sum(gzmap(ndc,z)$(not st(ndc)), pCRFcst(ndc)*vCapStor(ndc,y)*pCSPData(ndc,"Storage","Capex")*1e3)
                           + sum(gzmap(ndc,z), pCRFcth(ndc)*vCapTherm(ndc,y)*pCSPData(ndc,"Thermal Field","Capex")*1e6)
                           + sum(gzmap(dc,z), vAnnCapex(dc,y))
                           + sum(gzmap(g,z),  vCap(g,y)*pGenData(g,"FOMperMW"))
                           + sum(gzmap(st,z), vCap(st,y)*pStorData(st,"FixedOM"))
                           + sum(gzmap(cs,z), vCapStor(cs,y)*pCSPData(cs,"Storage","FixedOM"))
                           + sum(gzmap(cs,z), vCapTherm(cs,y)*pCSPData(cs,"Thermal field","FixedOM"))
***********************************************Hydrogen model related costs******************************************
                          + sum(h2zmap(ndcH2,z), pCRFH2(ndcH2)*vCapH2(ndcH2,y)*pH2Data(ndcH2,"Capex")*1e6)$pIncludeH2
                          + sum(h2zmap(dcH2,z), vAnnCapexH2(dcH2,y))$pIncludeH2
                          + sum(h2zmap(hh,z), vCapH2(hh,y)*pH2Data(hh,"FOMperMW"))$pIncludeH2;;

*********************************************************************************************************************


eYearlyVariableCost(z,y)..
   vYearlyVariableCost(z,y) =e= sum((gzmap(g,z),f,q,d,t,s), pVarCost(g,f,y)*vPwrOut(g,f,q,d,t,y,s)*pProbaScenarios(s)*pHours(q,d,t))

*********************************************Hydrogen model related costs ************************************
***(Units for equation below)                              $/mmBTU_H2    x      mmBTU_H2/MWh_e  x      MW_e       x       Hrs
                              + sum((h2zmap(hh,z),q,d,t), pVarCostH2(hh,y)*pH2Data(hh,"Heatrate")*vH2PwrIn(hh,q,d,t,y)*pHours(q,d,t))$pIncludeH2;


*****************************************************************************************************************


* Note: ReserveCost is in $/MWh -- this is the DIRECT cost of holding reserve like wear and tear that a generator bids in a market
eYearlyReserveCost(z,y)..
   vYearlyReserveCost(z,y) =e= sum((gzmap(g,z),q,d,t,s), vReserve(g,q,d,t,y,s)*pProbaScenarios(s)*pGenData(g,"ReserveCost")*pHours(q,d,t));

eYearlyUSECost(z,y)..
   vYearlyUSECost(z,y) =e= sum((q,d,t,s), vUSE(z,q,d,t,y,s)*pProbaScenarios(s)*pVoLL*pHours(q,d,t));

eYearlySurplusCost(z,y)..
   vYearlySurplus(z,y) =e= sum((q,d,t,s), vSurplus(z,q,d,t,y,s)*pProbaScenarios(s)*pSurplusPenalty*pHours(q,d,t));

eYearlyCurtailmentCost(z,y)..
   vYearlyCurtailmentCost(z,y) =e= sum((gzmap(g,z),q,d,t,s), vCurtailedVRE(z,g,q,d,t,y,s)*pProbaScenarios(s)*pCostOfCurtailment*pHours(q,d,t));


eYearlyTradeCost(z,y)..
   vYearlyTradeCost(z,y) =e= sum((q,d,t,s), vImportPrice(z,q,d,t,y,s)*pTradePrice(z,q,d,y,t)*pHours(q,d,t))
                           - sum((q,d,t,s), vExportPrice(z,q,d,t,y,s)*pTradePrice(z,q,d,y,t)*pHours(q,d,t));

eYearlyUSRCost(c,y)..
   vYearlyUSRCost(c,y) =e= sum(s, vUnmetReserve(c,y,s)*pProbaScenarios(s)*pReserveVoLL)
                         + sum((q,d,t,s), vUnmetSpinLoc(c,q,d,t,y,s)*pProbaScenarios(s)*pHours(q,d,t)*pSpinReserveVoLL);

eYearlySysUSRCost(y)..
   vYearlySysUSRCost(y) =e= vUnmetSysReserve(y)*pReserveVoLL
                          + sum((q,d,t,s), vUnmetSpin(q,d,t,y,s)*pProbaScenarios(s)*pHours(q,d,t)*pSpinReserveVoLL);

eYearlyCO2backstopCost(c,y)..
   vYearlyCO2backCost(c,y) =e= sum(s, vYearlyCO2backstop(c,y,s)*pProbaScenarios(s)*pCostOfCO2backstop);
   
eYearlySysCO2backstopCost(y)..
   vYearlySysCO2backCost(y) =e= sum(s, vYearlySysCO2backstop(y,s)*pProbaScenarios(s)*pCostOfCO2backstop);


eYearlyCarbonCost(z,y)..
   vYearlyCarbonCost(z,y) =e= pIncludeCarbon*pCarbonPrice(y)
                            * Sum((gzmap(g,z),gfmap(g,f),q,d,t,s), vPwrOut(g,f,q,d,t,y,s)*pProbaScenarios(s)*pHeatRate(g,f)*pFuelCarbonContent(f)*pHours(q,d,t));


*****************Turkey model specific equation*****************************************************************
*eYearlyStorPenalty(z,y)..
*                vYearlyStorPenalty(z,y)          =e=  sum(gzmap(st,z),sum((q,d,t),pStorPenalty(st)*(vStorInj(st,q,d,t,y))))+ sum(gfmap(st,f), sum(gzmap(st,z),sum((q,d,t),pStorPenalty(st)*vPwrOut.l(st,f,q,d,t,y))));
                

eH2UnservedCost(z,y)..
                vYearlyH2UnservedCost(z,y)       =e= sum(q, vUnmetExternalH2(z,q,y) )*pH2UnservedCost;

**********************************************************************************************************

$macro symmax(s,i,j,h) max(s(i,j,h),s(j,i,h))

eYearlyTransmissionAdditions(z,y)$(pAllowHighTransfer and sum(sTopology(z,z2),1))..
   vYearlyTransmissionAdditions(z,y) =e= sum(sTopology(z,z2), vAdditionalTransfer(z,z2,y)*symmax(pNewTransmission,z,z2,"CostPerLine")*1e6)
                                       / 2*(pWACC/(1-(1/((1+pWACC)**sum(sTopology(z,z2), symmax(pNewTransmission,z,z2,"Life"))))));

*--- Demand equations
eDemSupply(z,q,d,t,y,s)..

**********************Demand supply equations has been redifined to account for H2 production************************************

*
   pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y) =e=sum((gzmap(g,z),gfmap(g,f)),vPwrOut(g,f,q,d,t,y,s))
*                                                            The row below is deactivated when H2 feature is on                                                           
***********************************************************H2 model equations below************************************************
*                                                        +sum((gzmap(nRE,z),gfmap(nRE,f)),vPwrOut(nRE,f,q,d,t,y))$(pIncludeH2)
*                                                          +vPwrREGrid(z,q,d,t,y)$pIncludeH2                                                         
                                                         - sum((h2zmap(hh,z)),vH2PwrIn(hh,q,d,t,y))$pIncludeH2                                                        
*************************************************************************************************************************************                                            
                                                         - sum(sTopology(z,z2), vFlow(z,z2,q,d,t,y,s))
                                                         + sum(sTopology(z,z2), vFlow(z2,z,q,d,t,y,s)*(1-pLossFactor(z,z2,y)))
*                                                        + sum(gzmap(st,z), vStorOut(st,q,d,t,y))
                                                         - sum(gzmap(st,z), vStorInj(st,q,d,t,y,s))
                                                         + vImportPrice(z,q,d,t,y,s)
                                                         - vExportPrice(z,q,d,t,y,s)
                                                         + vUSE(z,q,d,t,y,s)
                                                         - vSurplus(z,q,d,t,y,s)                                                
                                                         ;


eMaxImportPrice(c,y,s)..
   sum((zcmap(z,c),q,d,t), vImportPrice(z,q,d,t,y,s)*pHours(q,d,t)) =l=
   sum((zcmap(z,c),q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))*pMaxPriceImportShare(y,c);

eMaxhourlyImportsshare(c,q,d,t,y,s)$(pMaxImport<1)..
   sum(zcmap(z,c), vImportPrice(z,q,d,t,y,s))  =l= sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pMaxImport);

eMaxhourlyExportsshare(c,q,d,t,y,s)$(pMaxExport<1)..
   sum(zcmap(z,c), vExportPrice(z,q,d,t,y,s)) =l= sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pMaxExport);

*--- Generator Capacity equations (check redundance with fix constraints in main.gms)
eCapBalance(g,sStartYear(y))..
   vCap(g,y) =e= pGenData(g,"Capacity")$(eg(g) and (pGenData(g,"StYr") <= sStartYear.val))
               + vBuild(g,y) - vRetire(g,y);

eCapBalance1(eg,y)$(not sStartYear(y))..
   vCap(eg,y) =e= vCap(eg,y-1) + vBuild(eg,y) - vRetire(eg,y);

eCapBalance2(ng,y)$(not sStartYear(y))..
   vCap(ng,y) =e= vCap(ng,y-1) + vBuild(ng,y);

eBuildNew(eg)$(pGenData(eg,"StYr") > sStartYear.val )..
   sum(y, vBuild(eg,y)) =l= pGenData(eg,"Capacity");

eMaxBuildTotal(ng)..
   sum(y, vBuild(ng,y)) =l= pGenData(ng,"MaxTotalBuild");

eMinBuildTotal(ng)$pGenData(ng,"MinTotalBuild")..
   sum(y, vBuild(ng,y)) =g= pGenData(ng,"MinTotalBuild");

eMinGenRE(c,y,s)$(pMinRE and y.val >= pMinRETargetYr)..
   sum((zcmap(z,c),gzmap(RE,z),gfmap(RE,f),q,d,t), vPwrOut(RE,f,q,d,t,y,s)*pHours(q,d,t)) =g=
   sum((zcmap(z,c),q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))*pMinRE;

eBuiltCap(ng,y)$pGenData(ng,"DescreteCap")..
   vBuild(ng,y) =e= pGenData(ng,"UnitSize")*vBuiltCapVar(ng,y);

eRetireCap(eg,y)$(pGenData(eg,"DescreteCap") and (y.val <= pGenData(eg,"RetrYr")))..
   vRetire(eg,y) =e= pGenData(eg,"UnitSize")*vRetireCapVar(eg,y);

*eConvertedPlants(g,y)$(sConversionPlants(g))..
*   vRetireCapVar(g,y) =G= sum(g2$(sConversionPlantsMap(g,g2)), vBuiltCapVar(g2,y));

eTotalFuelBuildCapacity(c,y,f)..
   vTotalFuelBuildCapacity(c,y,f) =e= sum(yy$(yy.val<=y.val), sum(z$zcmap(z,c),  sum(g$(gfmap(g,f) AND gzmap(g,z)), vBuild(g,yy))));

*--- Production equations

eMaxCF(g,q,y,s)..
   sum((gfmap(g,f),d,t), vPwrOut(g,f,q,d,t,y,s)*pHours(q,d,t)) =l= pAvailability(g,s,q)*vCap(g,y)*sum((d,t), pHours(q,d,t));

eFuel(zfmap(z,f),y,s)..
   vFuel(z,f,y,s) =e= sum((gzmap(g,z),gfmap(g,f),q,d,t), vPwrOut(g,f,q,d,t,y,s)*pHours(q,d,t)*pHeatRate(g,f));

eFuelLimit(c,f,y,s)$(pfuel_constraints and pMaxFuelLimit(c,f,y) > 0)..
   sum((zcmap(z,c),zfmap(z,f)), vFuel(z,f,y,s)) =l= pMaxFuelLimit(c,f,y)*1e6;

eMinGen(g,q,d,t,y,s)$(Relevant(d) and pmingen_constraints and pGenData(g,"MinLimitShare") > 0 and (sum(z, pDemandData(z,q,d,y,t)) >0 ))..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y,s)) =g= vCap(g,y)*pGenData(g,"MinLimitShare") ;

eRampDnLimit(g,q,d,t,y,s)$(Ramprate(g) and not sFirstHour(t) and pramp_constraints)..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t-1,y,s)) - sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y,s)) =l= vCap(g,y)*pGenData(g,"RampDnRate");

eRampUpLimit(g,q,d,t,y,s)$(Ramprate(g) and not sFirstHour(t) and pramp_constraints)..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y,s)) - sum(gfmap(g,f), vPwrOut(g,f,q,d,t-1,y,s)) =l= vCap(g,y)*pGenData(g,"RampUpRate");

* Note that we are effectively assuming grid-connected RE generation to be dispatchable. Generally speaking, most RE will be
* dispatched anyway because they have zero cost (i.e., not a different outcome from net load approach, but this allows for
* RE generation to be rejected as well

eVREProfile(gfmap(VRE,f),z,q,d,t,y,s)$(gzmap(VRE,z))..
   vPwrOut(VRE,f,q,d,t,y,s) + vCurtailedVRE(z,VRE,q,d,t,y,s) =e= pVREgenProfile(VRE,f,q,d,t)*vCap(VRE,y);


*Equation below includes an additional part which is different than zero only for RE generators and only when H2 feature is on
eJointResCap(g,q,d,t,y,s)..
   sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y,s))  +vReserve(g,q,d,t,y,s) =l= vCap(g,y)*(1+pGenData(g,"Overloadfactor"));



*--- Reserve equations
eResLim(g,q,d,t,y,s)$(pzonal_spinning_reserve_constraints or psystem_spinning_reserve_constraints)..
   vReserve(g,q,d,t,y,s) =l= vCap(g,y)*pGenData(g,"ResLimShare");

* This constraint increases solving time x3
* Reserve constraints include interconnections as reserves too
eResReqLocal(c,q,d,t,y,s)$pzonal_spinning_reserve_constraints..
   sum((zcmap(z,c),gzmap(g,z)),vReserve(g,q,d,t,y,s))
 + vUnmetSpinLoc(c,q,d,t,y,s)
* + sum((zcmap(z,c),sMapNCZ(z2,z)), pTransferLimit(z2,z,q,y)
*                                + vAdditionalTransfer(z2,z,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine")
*                                - vFlow(z2,z,q,d,t,y,s))
   =g= pReserveReqLoc(c,y) + sum((zcmap(z,c),gzmap(VRE,z),gfmap(VRE,f)), vPwrOut(VRE,f,q,d,t,y,s))*0.15 ;

eResReqSystem(q,d,t,y,s)$psystem_spinning_reserve_constraints..
   sum(g, vReserve(g,q,d,t,y,s)) + vUnmetSpin(q,d,t,y,s) =g= pReserveReqSys(y) + sum(gfmap(VRE,f), vPwrOut(VRE,f,q,d,t,y,s))*0.15;

eZonalMinCapReserve(c,y,s)$(pplanning_reserve_constraints and pReserveMargin(c))..
   sum((zcmap(z,c),gzmap(g,z)), vCap(g,y)*pCapacityCredit(g,y))
 + vUnmetReserve(c,y,s)
 + sum((zcmap(z,c),sMapNCZ(z2,z)), sum(q,pTransferLimit(z2,z,q,y))/card(q) + vAdditionalTransfer(z2,z,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine"))
   =g= (1+pReserveMargin(c))*smax((q,d,t), sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y)));

$ontext
eZonalMinCapReserve(c,y)$(pplanning_reserve_constraints and pReserveMargin(c))..
   sum((zcmap(z,c),gzmap(g,z)), vCap(g,y)*pCapacityCredit(g,y))
 + vUnmetReserve(c,y)
 + sum((zcmap(z,c),sMapNCZ(z2,z),q), pTransferLimit(z2,z,q,y) + vAdditionalTransfer(z2,z,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine"))
   =g= (1+pReserveMargin(c))*smax((q,d,t), sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y)));
$offtext

eSystemMinCapReserve(y)$(pplanning_reserve_constraints and psys_reserve_margin)..
   sum(g, vCap(g,y)*pCapacityCredit(g,y)) + vUnmetSysReserve(y)
   =g= (1+psys_reserve_margin)*smax((q,d,t), sum(z, pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y)));


*pplanning_reserve_constraints planning_reserve_constraints: au moins 80%
*psys_reserve_margin: 0.1   system_reserve_margin :110 au niveau du systeme 
*pReserveMargin
* le fait de passser de 110 a 80 : enorme diff du full trade 

*--- Transfer equations
eTransferLimit(sTopology(z,z2),q,d,t,y,s)..
   vFlow(z,z2,q,d,t,y,s) =l= pTransferLimit(z,z2,q,y) + vAdditionalTransfer(z,z2,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine")*pAllowHighTransfer;

eAdditionalTransfer(sTopology(z,z2),y)$pAllowHighTransfer..
   vAdditionalTransfer(z,z2,y) =e=  vAdditionalTransfer(z,z2,y-1) + vBuildTransmission(z,z2,y);

eAdditionalTransfer2(sTopology(z,z2),y)$pAllowHighTransfer..
   vBuildTransmission(z,z2,y)  =e=  vBuildTransmission(z2,z,y);

*--- Storage-specific equations
eStorageCap(st,q,d,t,y,s)$pincludeStorage..
   vStorage(st,q,d,t,y,s) =l= vCapStor(st,y);

*without this, it builds storage capacity to meet reserve
eStorageCap2(st,q,d,t,y)$pincludeStorage..
   vCapStor(st,y) =g= vCap(st,y);

eStorageInjection(st,q,d,t,y,s)$pincludeStorage..
   vStorInj(st,q,d,t,y,s) =l= vCap(st,y);

*eStorageInjection2(stp,q,d,t,y)$pincludeStorage..
*   vStorInj(stp,q,d,t,y) =l= sum(so, vCap(so,y)*pStoPVProfile(so,q,d,t));

eStorageInjection2(stp,q,d,t,y,s)$pincludeStorage..
   vStorInj(stp,q,d,t,y,s) =l= sum(gsmap(so,stp), vCap(so,y)*pStoPVProfile(so,q,d,t));

eRampDownInjLimit(st,q,d,t,y,s)$(not sFirstHour(t) and pincludeStorage)..
   vStorInj(st,q,d,t-1,y,s) - vStorInj(st,q,d,t,y,s) =l= pGenData(st,'RampDnRate')*vCap(st,y);

eRampUpInjLimit(st,q,d,t,y,s)$(not sFirstHour(t) and pincludeStorage)..
   vStorInj(st,q,d,t,y,s) - vStorInj(st,q,d,t-1,y,s) =l= pGenData(st,'RampUpRate')*vCap(st,y);

eStorageNet(st,q,d,t,y,s)$pincludeStorage..
   vStorNet(st,q,d,t,y,s) =e= sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y,s)) - vStorInj(st,q,d,t,y,s);

eStorageOutput(st,q,d,t,y,s)$pincludeStorage..
   sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y,s)) + vReserve(st,q,d,t,y,s) =l= vStorage(st,q,d,t,y,s);

eStorBal(st,q,d,t,y,s)$(not sFirstHour(t) and pincludeStorage)..
   vStorage(st,q,d,t,y,s) =e= pStorData(st,"efficiency")*vStorInj(st,q,d,t,y,s)
                          - sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y,s)) + vStorage(st,q,d,t-1,y,s);

eStorBal1(st,q,d,sFirstHour(t),y,s)$pincludeStorage..
   vStorage(st,q,d,t,y,s) =e= pStorData(st,"efficiency")*vStorInj(st,q,d,t,y,s)
                          - sum(gfmap(st,f), vPwrOut(st,f,q,d,t,y,s));
*****************************************************************************************************

*--- CSP-specific equations
eStorageCSPCap(cs,q,d,t,y,s)$pincludeCSP..
   vStorage(cs,q,d,t,y,s) =l= vCapStor(cs,y);

eInjCSP(cs,q,d,t,y,s)$pincludeCSP..
   vStorInj(cs,q,d,t,y,s) =l= vThermalOut(cs,q,d,t,y,s)*pCSPData(cs,"Thermal Field","Efficiency");
*without this, there is storage injection without storage (injection and withdrawal occur at the same time)

eInjCSP1(cs,q,d,t,y,s)$pincludeCSP..
   vStorInj(cs,q,d,t,y,s) =l= vCapStor(cs,y);

eThermCSP(cs,q,d,t,y,s)$pincludeCSP..
   vThermalOut(cs,q,d,t,y,s) =l= vCapTherm(cs,y)*pCSPProfile(cs,q,d,t);

eGenCSP(cs,q,d,t,y,s)$pincludeCSP..
   vThermalOut(cs,q,d,t,y,s)*pCSPData(cs,"Thermal Field","Efficiency")
 - vStorInj(cs,q,d,t,y,s) + vStorOut(cs,q,d,t,y,s)*pCSPData(cs,"Storage","Efficiency")
   =e= sum(gfmap(cs,f), vPwrOut(cs,f,q,d,t,y,s));

eStorageCSPBal(cs,q,d,t,y,s)$(not sFirstHour(t) and pincludeCSP)..
   vStorage(cs,q,d,t,y,s) =e= vStorage(cs,q,d,t-1,y,s) + vStorInj(cs,q,d,t,y,s) - vStorOut(cs,q,d,t,y,s);

eStorageCSPBal1(cs,q,d,sFirstHour(t),y,s)$pincludeCSP..
   vStorage(cs,q,d,t,y,s) =e= vStorInj(cs,q,d,t,y,s) - vStorOut(cs,q,d,t,y,s) ;

eStorageCSPBal2(cs,q,d,sFirstHour(t),y,s)$(not sFirstDay(d) and pincludeCSP)..
   vStorage(cs,q,d,t,y,s) =e= vStorInj(cs,q,d,t,y,s) - vStorOut(cs,q,d,t,y,s) + vStorage(cs,q,d-1,sLastHour,y,s);


*--- Energy (storage) capacity limits
eCapacityStorLimit(g,y)$pincludeStorage..
   vCapStor(g,y) =l= pStorData(g,"Capacity") + pCSPData(g,"Storage","Capacity");

************************************************************

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
                     + vBuildTherm(dc,y)*pCSPData(dc,"Thermal Field","Capex")*pCapexTrajectories(dc,y)*pCRFcth(dc)*1e6;
                        ;

*--- Emissions related equations
eCapitalConstraint$pcapital_constraints..
   sum(y, pRR(y)*pWeightYear(y)*sum(ng, pCRF(ng)*vCap(ng,y)*pGenData(ng,"Capex"))) =l= pMaxCapital*1e3;

eZonalEmissions(z,y,s)..
   vZonalEmissions(z,y,s) =e=
   sum((gzmap(g,z),gfmap(g,f),q,d,t), vPwrOut(g,f,q,d,t,y,s)*pHeatRate(g,f)*pFuelCarbonContent(f)*pHours(q,d,t));

eEmissionsCountry(c,y,s)$pzonal_co2_constraints..
   sum(zcmap(z,c), vZonalEmissions(z,y,s))-vYearlyCO2backstop(c,y,s)=l= pEmissionsZone(c,y);

eTotalEmissions(y,s)..
    sum(z, vZonalEmissions(z,y,s))=e= vTotalEmissions(y,s);

eTotalEmissionsConstraint(y,s)$pSystem_CO2_constraints..
    vTotalEmissions(y,s)-vYearlySysCO2backstop(y,s) =l= pEmissionsTotal(y);

******************************************************

*********************Hydrogen production equations******************
*

*  Total capacity of H2 plants at 1st year of optimization is equal to pre-existing capacity plus capacity being bulit minus retired capacity
eCapBalanceH2(hh,sStartYear(y))$(pIncludeH2)..
   vCapH2(hh,y) =e= pH2Data(hh,"Capacity")$(eh(hh) and (pH2Data(hh,"StYr") <= sStartYear.val))
               + vBuildH2(hh,y) - vRetireH2(hh,y);


*  Total capacity of existing H2 palnts is equal to capacity over previous year plus capacity being built in current year minus retired capacity in current year
eCapBalance1H2(eh,y)$(not sStartYear(y) and pIncludeH2)..
   vCapH2(eh,y) =e= vCapH2(eh,y-1) + vBuildH2(eh,y) - vRetireH2(eh,y);


*Total capacity of candidate H2 plants is equal to total capacity at previous year plus capacity being built in current year minus retired capacity in current year
eCapBalance2H2(nh,y)$(not sStartYear(y) and pIncludeH2)..
   vCapH2(nh,y) =e= vCapH2(nh,y-1) + vBuildH2(nh,y);



* New H2 plants can be buuilt only after the StYr; the newly built capacity need to be less than declared capacity
eBuildNewH2(eh)$(pH2Data(eh,"StYr") > sStartYear.val and pIncludeH2)..
    sum(y, vBuildH2(eh,y)) =l= pH2Data(eh,"Capacity");



* Total built H2 generation capacity need to be less than maxtotal built
eMaxBuildTotalH2(nh)$(pIncludeH2)..
   sum(y, vBuildH2(nh,y)) =l= pH2Data(nh,"MaxTotalBuild");



* (Integer units ) Built capacity each year is equal to unit size
eBuiltCapH2(nh,y)$(pH2Data(nh,"DescreteCap") and pIncludeH2)..
   vBuildH2(nh,y) =e= pH2Data(nh,"UnitSize")*vBuiltCapVarH2(nh,y);



* (Integer units ) Retired capacity each year is equal to unit size
eRetireCapH2(eh,y)$(pH2Data(eh,"DescreteCap") and (y.val <= pH2Data(eh,"RetrYr")) and pIncludeH2)..
   vRetireH2(eh,y) =e= pH2Data(eh,"UnitSize")*vRetireCapVarH2(eh,y);


*  Maximum capacity factor of H2 production based on availability
eMaxCF_H2(hh,q,y)$(pIncludeH2)..
   sum((d,t), vH2PwrIn(hh,q,d,t,y)*pHours(q,d,t)) =l= pAvailabilityH2(hh,q)*vCapH2(hh,y)*sum((d,t), pHours(q,d,t));


*       mmBTU of H2 produced
*eFuel_H2(z,y)$(pIncludeH2)..

****(Units for equation below)       -MWe-                Hr                mmBTU/MWhe
*   vFuelH2(z,y) =e= sum((hh,q,d,t), vH2PwrIn(hh,q,d,t,y)*pHours(q,d,t)*pH2Data(hh,"HeatRate"));



*       mmBTU of H2 produced
eFuel_H2(z,q,y)$(pIncludeH2)..
                
****(Units for equation below)                                                                      -MWe-                Hr                mmBTU/MWhe
 pExternalH2(z,q,y)-vUnmetExternalH2(z,q,y)$pExternalH2(z,q,y)+vFuelH2Quarter(z,q,y) =l= sum( h2zmap(hh,z), sum((d,t), vH2PwrIn(hh,q,d,t,y)*pHours(q,d,t)*pH2Data(hh,"HeatRate")));

*

eFuel_H2_2(z,y)$(pIncludeH2)..
sum(q,vFuelH2Quarter(z,q,y)) =e=  vFuelH2(z,y);



*eFuel_H2_3(q,y)$(pIncludeH2 and pExternalH2(q,y))..                                    
* sum(z,vUnmetExternalH2(z,q,y)) =l=  pExternalH2(q,y);

eFuel_H2_3(z,q,y)$(pIncludeH2 and pExternalH2(z,q,y))..                                    
 vUnmetExternalH2(z,q,y) =l=  pExternalH2(z,q,y);


eMaxH2PwrInjection(hh,q,d,t,y)$pIncludeH2..
vH2PwrIn(hh,q,d,t,y)  =l= vCapH2(hh,y);


* to change if stochastic with H2 activated
* The amount of hydrogen fuel that can be used for electricity generation can not be more than the amount of H2 that was produced from VRE curtailment
eFuelLimitH2(c,f,y)$(pFuelData(f) and pIncludeH2)..
   sum((zcmap(z,c),s), pProbaScenarios(s)*vFuel(z,f,y,s)) =e=  sum((zcmap(z,c)), vFuelH2(z,y));


*When the H2 production flag is off don't account for H2 fuel
eFuelLimitH2_2(c,f,y)$(pFuelData(f) and pIncludeH2=0)..
   sum((zcmap(z,c),s), pProbaScenarios(s)*vFuel(z,f,y,s)) =l=  0;


eRampDnLimitH2(hh,q,d,t,y)$(RamprateH2(hh) and not sFirstHour(t) and pramp_constraints and pIncludeH2)..
    vH2PwrIn(hh,q,d,t-1,y) -  vH2PwrIn(hh,q,d,t,y) =l= vCapH2(hh,y)*pH2Data(hh,"RampDnRate");




eRampUpLimitH2(hh,q,d,t,y)$(RamprateH2(hh) and not sFirstHour(t) and pramp_constraints and pIncludeH2)..
   vH2PwrIn(hh,q,d,t,y) -  vH2PwrIn(hh,q,d,t-1,y) =l= vCapH2(hh,y)*pH2Data(hh,"RampUpRate");



*Calculation of Annualized CAPEX for electrolyzers
eAnnCapexH2_1(dcH2,y)$(not sStartYear(y) and pIncludeH2)..
   vAnnCapexH2(dcH2,y) =e= vAnnCapexH2(dcH2,y-1)
                     + vBuildH2(dcH2,y)*pH2Data(dcH2,"Capex")*pCapexTrajectoriesH2(dcH2,y)*pCRFH2(dcH2)*1e6;


eAnnCapexH2(dcH2,sStartYear(y))$pIncludeH2..
   vAnnCapexH2(dcH2,y) =e= vBuildH2(dcH2,y)*pH2Data(dcH2,"Capex")*pCapexTrajectoriesH2(dcH2,y)*pCRFH2(dcH2)*1e6;                                                                          ;




eRE2H2_4(z,q,d,t,y)$pIncludeH2..
  sum(h2zmap(hh,z),vH2PwrIn(hh,q,d,t,y))=e=vPwrREH2(z,q,d,t,y);
 

eRE2H2_3(z,q,d,t,y)$(pIncludeH2 )..
 vPwrREH2(z,q,d,t,y) =e= sum((gfmap(RE,f),gzmap(RE,z)),vREPwr2H2(RE,f,q,d,t,y));



eRE2H2_2(z,q,d,t,y)$(pIncludeH2)..
vPwrREGrid(z,q,d,t,y)=e=sum((gfmap(RE,f),gzmap(RE,z)),vREPwr2Grid(RE,f,q,d,t,y));


* To change if stochastic while H2 is activated
eRE2H2(RE,f,q,d,t,y)$(pIncludeH2)..
 sum(s, vPwrOut(RE,f,q,d,t,y,s)*pProbaScenarios(s))=e=vREPwr2Grid(RE,f,q,d,t,y)+vREPwr2H2(RE,f,q,d,t,y);

*eSNSP2(q,d,t,y)$(pIncludeH2)..
* sum(gfmap(vre,f) , vPwrOut(vre,f,q,d,t,y))=l= sum(st,vStorInj(st,q,d,t,y))+sum(gfmap(vre,f),vREPwr2H2(vre,f,q,d,t,y));

eAnnualFuelBuildLimit(f,y)$(pAnnualFuelBuildLimit(f,y))..
   sum(g$(gfmap(g,f) AND sum(z$(szfuellim(z) AND gzmap(g,z)),1)), vBuild(g,y)) =L= pAnnualFuelBuildLimit(f,y)*pWeightYear(y);


*For example equation below is deactivated when H2 is on and is replaced by eVREProfile2

***********************************end of H2 equations section**************************************



Model PA /
   eNPVCost
   eYearlyTotalCost
   eYearlyFixedCost
   eYearlyVariableCost
   eYearlyReserveCost
   eYearlyUSECost
   eYearlyUSRCost
   eYearlyCarbonCost
   eDemSupply
   eCapBalance
   eCapBalance1
   eCapBalance2
   eBuildNew
   eMaxBuildTotal
   eMinBuildTotal
   eMinGenRE
   eMaxCF
   eMinGen

   eFuel
   eRampUpLimit
   eRampDnLimit
   eResLim
*  eResLim_CSP
   eJointResCap
   eResReqLocal
   eResReqSystem
   eZonalMinCapReserve
   eTransferLimit
   eVREProfile
   eFuelLimit

   eCapitalConstraint
   eZonalEmissions
   eEmissionsCountry
   eTotalEmissions
eTotalEmissionsConstraint
   eYearlyCurtailmentCost
   eYearlyCO2backstopCost

   eTotalFuelBuildCapacity

   eBuiltCap
   eRetireCap

   eMaxImportPrice
   eMaxhourlyImportsshare
   eMaxhourlyExportsshare
   eYearlyTradeCost
   eSystemMinCapReserve
   eYearlySysUSRCost
   eYearlySurplusCost
   eYearlyTransmissionAdditions
   eAdditionalTransfer
   eAdditionalTransfer2

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
   eStorageCSPBal2
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

* variable limited domains
   vPwrOut(sPwrOut)
   vExportPrice(sExportPrice)
   vImportPrice(sImportPrice)
   vAdditionalTransfer(sAdditionalTransfer)
   vFlow(sFlow)
   vReserve(sReserve)

*eConvertedPlants

****************************************************************************

***************************Equations for H2 model*******************************
vH2PwrIn(sH2PwrIn)
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
eFuel_H2_3
eRampDnLimitH2
eRampUpLimitH2
eFuelLimitH2
eFuelLimitH2_2

eAnnCapexH2_1
eAnnCapexH2


eRE2H2
eRE2H2_2
eRE2H2_3
eRE2H2_4
eMaxH2PwrInjection
*eSNSP2
eH2UnservedCost

eAnnualFuelBuildLimit
*************************************************************************

/;




