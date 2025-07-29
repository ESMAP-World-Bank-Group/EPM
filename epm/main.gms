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

$offeolcom
$offinline
$inlinecom {  }
$eolcom //

$if not set DEBUG $set debug 0
$if not set EPMVERSION    $set EPMVERSION    9.0

* Turn on/off additional information to the listing file
Option limRow=0, limCol=0, sysOut=off, solPrint=off;
$if %DEBUG%==1 $onUELlist onUELXRef onListing 
$if %DEBUG%==1 Option limRow=1e9, limCol=1e9, sysOut=on, solPrint=on;

*-------------------------------------------------------------------------------------

* Only include base if we don't restart
$ifThen not set BASE_FILE
$set BASE_FILE "base.gms"
$endIf
$if set ROOT_FOLDER $set BASE_FILE %ROOT_FOLDER%/%BASE_FILE%

$log BASE_FILE is "%BASE_FILE%"

$ifThen not set REPORT_FILE
$set REPORT_FILE "generate_report.gms"
$endIf
$if set ROOT_FOLDER $set REPORT_FILE %ROOT_FOLDER%/%REPORT_FILE%
$log REPORT_FILE is "%REPORT_FILE%"



$ifThen not set READER_FILE
$set READER_FILE "input_readers.gms"
$endIf
$if set ROOT_FOLDER $set READER_FILE %ROOT_FOLDER%/%READER_FILE%
$log READER_FILE is "%READER_FILE%"


$ifThen not set VERIFICATION_FILE
$set VERIFICATION_FILE "input_verification.gms"
$endIf
$if set ROOT_FOLDER $set VERIFICATION_FILE %ROOT_FOLDER%/%VERIFICATION_FILE%
$log VERIFICATION_FILE is "%VERIFICATION_FILE%"


$ifThen not set TREATMENT_FILE
$set TREATMENT_FILE "input_treatment.gms"
$endIf
$if set ROOT_FOLDER $set TREATMENT_FILE %ROOT_FOLDER%/%TREATMENT_FILE%
$log TREATMENT_FILE is "%TREATMENT_FILE%"


$ifThen not set DEMAND_FILE
$set DEMAND_FILE "generate_demand.gms"
$endIf
$if set ROOT_FOLDER $set DEMAND_FILE %ROOT_FOLDER%/%DEMAND_FILE%
$log DEMAND_FILE is "%DEMAND_FILE%"


*-------------------------------------------------------------------------------------

* Define solver-related options
Scalar
   solverThreads 'Number of threads available to the solvers' /1/
   solverOptCR   'Relative gap for MIP Solver'                /0.05/
   solverResLim  'Solver time limit'                          /300000/
;   
Singleton Set
   NLPSolver 'Selected NLP Solver' / '' 'conopt4' /
   MIPSolver 'Selected MIP Solver' / '' 'cplex' /
;


* Evaluate and assign solver settings as macros
$eval     SOLVERTHREADS solverThreads
$eval     SOLVEROPTCR   solverOptCR
$eval     SOLVERRESLIM  solverResLim
$eval.set NLPSOLVER     NLPSolver.te
$eval.set MIPSOLVER     MIPSolver.te
* Apply solver settings to GAMS execution
option NLP=%NLPSOLVER%, MIP=%MIPSOLVER%, threads=%SOLVERTHREADS%, optCR=%SOLVEROPTCR%, resLim=%SOLVERRESLIM%;

*-------------------------------------------------------------------------------------


Sets
   g        'generators or technology-fuel types'
   f        'fuels'
   tech     'technologies'
   y        'years'
   q        'quarters or seasons'
   d        'day types'
   t        'hours of day'
   z        'zones'
   c        'countries'
   zext     'external zones'
   pGenDataInputHeader 'Generator data input headers'
   pSettingsHeader
   pStoreDataHeader
   pCSPDataHeader
   pTransmissionHeader
   pH2Header
   pTechDataHeader
   pe       'peak energy for demand forecast' /'peak', 'energy'/
*********Hydrogen specific addition***********
   hh        'Hydrogen production units'
;


Sets
   zcmap(z<,c<)           'map zones to countries'
   gmap(g,z,tech,f) 'Map generators to firms, zones, technologies and fuels'
   sRelevant(d)  'relevant day and hours when MinGen limit is applied'

;

alias (z,z2), (g,g1,g2);

* Input data parameters 
Parameter
* Generator data
   pGenDataInput(g<,z,tech<,f<,pGenDataInputHeader<)      'Generator data from Excel input'
   
   pGenDataInputDefault(z,tech,f,pGenDataInputHeader)     'Default generator data by zone/tech/fuel'
   pCapexTrajectoriesDefault(z,tech,f,y) 'Default CAPEX trajectories'
   pCapexTrajectories(g,y)             'Generator CAPEX trajectories'
   pAvailabilityDefault(z,tech,f,q)     'Default availability factors'
   
* Storage data
   pStorDataExcel(g,*,pStoreDataHeader<)             'Storage unit specifications'
   
* CSP and technology data
   pCSPData(g,pCSPDataHeader<,pStoreDataHeader)              'Concentrated solar power data'
   pTechData(tech,pTechDataHeader<)              'Technology specifications'
   
* Fuel data
   pFuelCarbonContent(f)                'Carbon content by fuel (tCO2/MMBtu)'
   pMaxFuellimit(c,f,y)             'Fuel limit in MMBTU*1e6 (million) by country'
   pFuelPrice(c,f,y)                   'Fuel price forecasts'

   
* Storage and transmission
   pStorDataInput(g,g2,pStoreDataHeader)            'Storage unit input data'
   pNewTransmission(z,z2,pTransmissionHeader<)          'New transmission line specifications'
   
* Trade parameters
   pTradePrice(zext,q,d,y,t)           'External trade prices'
   pMaxExchangeShare(y,c)              'Maximum trade share by country'
   
* Demand parameters
   pDemandProfile(z,q,d,t)             'Normalized demand profiles'
   pDemandForecast(z,pe,y)             'Peak/energy demand forecasts (MW/GWh)'
   pDemandData(z,q,d,y,t)              'Hourly load curves'
   
* Emissions and carbon
   pEmissionsCountry(c,y)              'Country emission limits (tons)'
   pEmissionsTotal(y)                  'System-wide emission limits (tons)'
   pCarbonPrice(y)                     'Carbon price (USD/ton CO2)'
   
* Time and transfer parameters
   pHours(q<,d<,t<)                    'Hours mapping'
   
   pTransferLimit(z,z2,q,y)            'Inter-zonal transfer limits'
   pMinImport(z2,z,y)                  'Minimum import requirements'
   pLossFactor(z,z2,y)                 'Transmission loss factors'
   
* VRE and availability
   pVREProfile(z,tech,q,d,t)           'VRE generation profiles by site'
   pVREgenProfile(g,q,d,t)             'VRE generation profiles by plant'
   pAvailability(g,q)                  'Seasonal availability factors'
   
* Reserve requirements
   pSpinningReserveReqCountry(c,y)     'Country spinning reserve requirements'
   pSpinningReserveReqSystem(y)        'System spinning reserve requirements'
   pPlanningReserveMargin(c)           'Planning reserve margins'
   
* Other parameters
   pSettings(pSettingsHeader<)         'Model settings and penalties'
   
   pEnergyEfficiencyFactor(z,y)        'Energy efficiency adjustment factors'
   pExtTransferLimit(z,zext,q,*,y)     'External transfer limits'
   
* Hydrogen parameters
   pH2Data(hh,pH2Header<)                    'Hydrogen production specifications'
   pH2DataExcel(hh<,*)                 'Hydrogen data from Excel'
   pAvailabilityH2(hh,q)               'H2 plant availability'
   pFuelDataH2(f)                      'Hydrogen fuel properties'
   pCapexTrajectoryH2(hh,y)            'H2 CAPEX trajectories'
   pExternalH2(z,q,y)               'mmBTUs of H2 as external demand that need to be met'

;   



* Allow multiple definitions of symbols without raising an error (use with caution)
$onMulti

Parameter
    ftfindex(f)
;
   
$if not errorfree $abort Error before reading input
*-------------------------------------------------------------------------------------
* Read inputs

* Include the external reader file defined by the macro variable %READER_FILE%
$include %READER_FILE%

* Open the specified GDX input file for reading
$gdxIn input.gdx

* Load domain-defining symbols (sets and indices)
$load zcmap pSettings y pHours
$load pGenDataInput gmap
$load pGenDataInputDefault pAvailabilityDefault pCapexTrajectoriesDefault
$load pTechData ftfindex

* Load demand data
$load pDemandData pDemandForecast pDemandProfile pEnergyEfficiencyFactor sRelevant

$load pFuelCarbonContent pCarbonPrice pEmissionsCountry pEmissionsTotal pFuelPrice

* Load constraints and technical data
$load pMaxFuellimit pTransferLimit pLossFactor pVREProfile pVREgenProfile pAvailability
$load pStorDataExcel pCSPData pCapexTrajectories pSpinningReserveReqCountry pSpinningReserveReqSystem 
$load pPlanningReserveMargin pEnergyEfficiencyFactor  

* Load trade data
$load zext
$load pExtTransferLimit, pNewTransmission, pMinImport
$load pTradePrice, pMaxExchangeShare

* Load Hydrogen model-related symbols
$load pH2DataExcel hh pAvailabilityH2 pFuelDataH2 pCAPEXTrajectoryH2 pExternalH2

* Close the GDX file after loading all required data
$gdxIn
$offmulti
$if not errorfree $abort CONNECT ERROR in input_readers.gms


*-------------------------------------------------------------------------------------

* Make input verification
$log ##########################
$log ### INPUT VERIFICATION ###
$log ##########################

$include %VERIFICATION_FILE%
$if not errorfree $abort PythonError in input_verification.gms

$log ##############################
$log ### INPUT VERIFICATION END ###
$log ##############################

*-------------------------------------------------------------------------------------
* Make input treatment

$log ###########################
$log ##### INPUT TREATMENT #####
$log ###########################

$onMulti
$include %TREATMENT_FILE%
$if not errorfree $abort PythonError in input_treatment.gms
$offMulti

$log ###############################
$log ##### INPUT TREATMENT END #####
$log ###############################

*-------------------------------------------------------------------------------------

$if not errorFree $abort Data errors.

*-------------------------------------------------------------------------------------

* Only include BASE_FILE if this is a fresh run (i.e., not using a restart file)
* This prevents reloading sets, parameters, or data already available in the restart
$if "x%gams.restart%" == "x" $include %BASE_FILE%

*-------------------------------------------------------------------------------------


pStorDataInput(g,g2,pStoreDataHeader) = pStorDataExcel(g,g2,pStoreDataHeader);
pStorDataInput(g,g,pStoreDataHeader)$pStorDataExcel(g,'',pStoreDataHeader) = pStorDataExcel(g,'',pStoreDataHeader);


* Generate gfmap and others from pGenDataInput
parameter gstatIndex(gstatus) / Existing 1, Candidate 3, Committed 2 /;
parameter tstatIndex(tstatus) / Candidate 3, Committed 2 /;

*H2 model parameter
parameter H2statIndex(H2status) / Existing 1, Candidate 3, Committed 2 /;


* TODO: Bug if removed, but never called?
set addHdr / fuel1, fuel2, Zone, Type, 'Assigned Value', status, Heatrate2,
             'RE Technology', 'Hourly Variation' /;
             

* Aggregate `gmap(g,z,tech,f)` over `tech` and `f` to get `gzmap(g,z)`,
* which represents the mapping of generator `g` to zone `z`.
gzmap(g,z) = sum((tech,f), gmap(g,z,tech,f));

* Aggregate `gmap(g,z,tech,f)` over `tech` and `z` to get `gfmap(g,f)`,
* which represents the mapping of generator `g` to fuel `f`.
gfmap(g,f) = sum((tech,z), gmap(g,z,tech,f));

* Compute `gprimf(g,f)`, which is similar to `gfmap(g,f)`,
* aggregating over `tech` and `z` to represent the primary fuel mapping.
gprimf(g,f) = sum((tech,z), gmap(g,z,tech,f));

* Aggregate `gmap(g,z,tech,f)` over `z` and `f` to get `gtechmap(g,tech)`,
* which represents the mapping of generator `g` to technology `tech`.
gtechmap(g,tech) = sum((z,f), gmap(g,z,tech,f));

* Update `gfmap(g,f)`, ensuring it includes additional mappings 
* based on `pGenDataInput(g,z,tech,f2,'fuel2')` when a condition is met.
gfmap(g,f) = gfmap(g,f) 
         or sum((z,tech,f2), (pGenDataInput(g,z,tech,f2,'fuel2') = ftfindex(f)));
         

* Map generator status from input data
gstatusmap(g,gstatus) = sum((z,tech,f),pGenDataInput(g,z,tech,f,'status')=gstatIndex(gstatus));


pHeatrate(gprimf(g,f)) = sum((z,tech), pGenDataInput(g,z,tech,f,"Heatrate"));
pHeatrate(g,f2)$(gfmap(g,f2) and not gprimf(g,f2)) = 
    sum((z,tech,f), pGenDataInput(g,z,tech,f,"Heatrate2") 
*  $(pGenDataInput(g,z,tech,f,"fuel2") = ftfindex(f2))
    );


pGenData(g,pGenDataInputHeader) = sum((z,tech,f),pGenDataInput(g,z,tech,f,pGenDataInputHeader));

***********************H2 model parameters***************************************************

pH2Data(hh,pH2Header)=pH2DataExcel(hh,pH2Header);
H2statusmap(hh,H2status) = pH2DataExcel(hh,'status')=H2statIndex(H2status);
* TODO: Check is that works for H2
* h2zmap(hh,z) = pH2DataExcel(hh,'Zone')=pZoneIndex(z);
h2zmap(hh,z) = pH2DataExcel(hh,'Zone');


execute_unload "input.gdx" y pHours pTechData pGenDataInput pGenDataInputDefault pAvailabilityDefault pCapexTrajectoriesDefault
zext ftfindex gmap gfmap gprimf zcmap sRelevant pDemandData pDemandForecast
pDemandProfile pFuelCarbonContent pCarbonPrice pEmissionsCountry
pEmissionsTotal pFuelPrice pMaxFuellimit pTransferLimit pLossFactor pVREProfile pVREgenProfile pAvailability
pStorDataExcel pCSPData pCapexTrajectories pSpinningReserveReqCountry pSpinningReserveReqSystem pSettings
pPlanningReserveMargin pEnergyEfficiencyFactor pTradePrice pMaxExchangeShare
pExtTransferLimit pNewTransmission pMinImport
pH2DataExcel hh pAvailabilityH2 pFuelDataH2 pCAPEXTrajectoryH2 pExternalH2 pHeatrate
;

*-------------------------------------------------------------------------------------

*--- Parameter initialisation for same demand profile for all years

$include %DEMAND_FILE%

*--- Part2: Start of initialisation of other parameters

$set zonal_spinning_reserve_constraints   -1
$set system_spinning_reserve_constraints  -1
$set planning_reserve_constraints         -1
$set ramp_constraints                     -1
$set fuel_constraints                     -1
$set capital_constraints                  -1
$set mingen_constraints                   -1
$set includeCSP                           -1
$set includeStorage                       -1
$set zonal_co2_constraints                -1
$set system_co2_constraints               -1
$set IncludeDecomCom                      -1
*Hydrogen model specific sets
$set IncludeH2                            -1


* Read main parameters from pSettings
pzonal_spinning_reserve_constraints  = pSettings("zonal_spinning_reserve_constraints");
psystem_spinning_reserve_constraints = pSettings("system_spinning_reserve_constraints");
psystem_reserve_margin               = pSettings("system_reserve_margin");
pplanning_reserve_constraints        = pSettings("planning_reserve_constraints");
pinterco_reserve_contribution        = pSettings("interco_reserve_contribution");
pramp_constraints                    = pSettings("ramp_constraints");
pfuel_constraints                    = pSettings("fuel_constraints");
pcapital_constraints                 = pSettings("capital_constraints");
pmingen_constraints                  = pSettings("mingen_constraints");
pincludeCSP                          = pSettings("includeCSP");
pincludeStorage                      = pSettings("includeStorage");
pMinRE                               = pSettings("MinREshare");
pMinRETargetYr                       = pSettings("RETargetYr");
pzonal_co2_constraints               = pSettings("zonal_co2_constraints");
psystem_co2_constraints              = pSettings("system_co2_constraints");
pAllowExports                        = pSettings("allowExports");
pSurplusPenalty                      = pSettings("costSurplus");
pAllowHighTransfer                   = pSettings("pAllowHighTransfer");
pCostOfCurtailment                   = pSettings("costcurtail");
pCostOfCO2backstop                   = pSettings("CO2backstop");
pMaxImport                           = pSettings("MaxImports");
pMaxExport                           = pSettings("MaxExports");
pVREForecastError                    = pSettings("VREForecastError");
pCaptraj                             = pSettings("Captraj");
pIncludeIntercoReserves              = pSettings("includeIntercoReserves");
pVRECapacityCredits                  = pSettings("VRECapacityCredits");
pSeasonalReporting                   = pSettings("Seasonalreporting");
pSystemResultReporting               = pSettings("Systemresultreporting");
pMaxLoadFractionCCCalc               = pSettings("MaxLoadFractionCCCalc");
*Related to hydrogen model
pIncludeH2                       = pSettings("IncludeH2");
pH2UnservedCost                  = pSettings("H2UnservedCost");


* Assign values to model parameters only if their corresponding macro variables are not set to "-1"
$if not "%zonal_spinning_reserve_constraints%"  == "-1" pzonal_spinning_reserve_constraints  = %zonal_spinning_reserve_constraints%;
$if not "%system_spinning_reserve_constraints%" == "-1" psystem_spinning_reserve_constraints = %system_spinning_reserve_constraints%;
$if not "%planning_reserve_constraints%"        == "-1" pplanning_reserve_constraints        = %planning_reserve_constraints%;
$if not "%ramp_constraints%"                    == "-1" pramp_constraints                    = %ramp_constraints%;
$if not "%fuel_constraints%"                    == "-1" pfuel_constraints                    = %fuel_constraints%;
$if not "%capital_constraints%"                 == "-1" pcapital_constraints                 = %capital_constraints%;
$if not "%mingen_constraints%"                  == "-1" pmingen_constraints                  = %mingen_constraints%;
$if not "%includeCSP%"                          == "-1" pincludeCSP                          = %includeCSP%;
$if not "%includeStorage%"                      == "-1" pincludeStorage                      = %includeStorage%;
$if not "%zonal_co2_constraints%"               == "-1" pzonal_co2_constraints               = %zonal_co2_constraints%;
$if not "%system_co2_constraints%"              == "-1" psystem_co2_constraints              = %system_co2_constraints%;
$if not "%IncludeDecomCom%"                     == "-1" pIncludeDecomCom                     = %IncludeDecomCom%;
$if not "%IncludeH2%"                           == "-1" pIncludeH2                           = %IncludeH2%;

singleton set sFinalYear(y);
scalar TimeHorizon;

sStartYear(y) = y.first;
sFinalYear(y) = y.last;
TimeHorizon = sFinalYear.val - sStartYear.val + 1;
sFirstHour(t) = t.first;
sLastHour(t) = t.last;
sFirstDay(d) = d.first;

pDR              = pSettings("DR");
pWACC            = pSettings("WACC");
pVOLL            = pSettings("VOLL");
pPlanningReserveVoLL     = pSettings("ReserveVoLL");
pMaxCapital      = pSettings("MaxCapital")*1e6;
pSpinningReserveVoLL = pSettings("SpinReserveVoLL");
pIncludeCarbon   = pSettings("includeCarbonPrice");
pinterconMode    = pSettings("interconMode");
pnoTransferLim   = pSettings("noTransferLim");
pincludeEE       = pSettings("includeEE");
pIncludeDecomCom = pSettings("IncludeDecomCom");

* Set external transfer limits to zero if exports are not allowed
pExtTransferLimit(z,zext,q,"Import",y)$(not pallowExports)  = 0 ;
pExtTransferLimit(z,zext,q,"Export",y)$(not pallowExports)  = 0 ;

* Assign import and export transfer limits only if exports are allowed
pExtTransferLimitIn(z,zext,q,y)$pallowExports   = pExtTransferLimit(z,zext,q,"Import",y) ;
pExtTransferLimitOut(z,zext,q,y)$pallowExports  = pExtTransferLimit(z,zext,q,"Export",y) ;

* Define `Zt(z)` to check if total demand in a zone `z` is zero
Zt(z) = sum((q,d,y,t),pDemandData(z,q,d,y,t)) = 0;
* Define `Zd(z)` as the complement of `Zt(z)`, indicating zones with demand
Zd(z) = not Zt(z);


* Assign storage data from `pStorDataInput` based on the generator-storage mapping
option gsmap<pStorDataInput;
loop(gsmap(g2,g), pStorData(g,pStoreDataHeader) = pStorDataInput(g,g2,pStoreDataHeader));

* Remove generator pairs (`g,g`) that correspond to standalone storage plants from `gsmap`
gsmap(g,g) = no;

* Identify candidate generators (`ng(g)`) based on their status in `gstatusmap`
ng(g)  = gstatusmap(g,'candidate');

* Define existing generators (`eg(g)`) as those that are not candidates, include comitted
eg(g)  = not ng(g);

* Identify variable renewable energy (VRE) generators (`vre(g)`) based on hourly variation data
vre(g) = sum(gtechmap(g,tech)$pTechData(tech,'Hourly Variation'),1);

* Identify renewable energy (RE) generators (`re(g)`) based on RE technology classification
re(g)  = sum(gtechmap(g,tech)$pTechData(tech,'RE Technology'),1);

* Identify concentrated solar power (CSP) technologies
cs(g)  = gtechmap(g,"CSPPlant");

* Identify PV with storage technologies
so(g)  = gtechmap(g,"PVwSTO");

* Identify solar thermal with PV (`STOPV`)
stp(g) = gtechmap(g,"STOPV");

* Identify storage technologies
stg(g) = gtechmap(g,"Storage");

* Define a general storage category (`st(g)`) as either `STOPV` or `STORAGE`
st(g)  = gtechmap(g,"STOPV") or gtechmap(g,"Storage");

* Define generators with capex trajectory data
dc(g)  = sum(y, pCapexTrajectories(g,y));

* Define generators without capex trajectory data
ndc(g) = not dc(g);

* Identify run-of-river hydro generators (`ror(g)`)
ror(g) = gtechmap(g,"ROR");

* Identify VRE generators excluding run-of-river hydro (`VRE_noROR(g)`)
VRE_noROR(g) = vre(g) and not ror(g);

* Define ramp-down rate for generators
RampRate(g) = pGenData(g,"RampDnRate");

* Map zones (`z`) to fuels (`f`) based on generator-fuel assignments (`gzmap` and `gfmap`)
zfmap(z,f) = sum((gzmap(g,z),gfmap(g,f)), 1);

* H2 model specific sets
nh(hh)  = H2statusmap(hh,'candidate');
eh(hh)  = not nh(hh);
RampRateH2(hh)= pH2Data(hh,"RampDnRate");
dcH2(hh)  = sum(y, pCapexTrajectoryH2(hh,y));
ndcH2(hh) = not dcH2(hh);
nRE(g) = not re(g);
nVRE(g)=not VRE(g);
REH2(g)= sum(gtechmap(g,tech)$pTechData(tech,'RE Technology'),1) - sum(gtechmap(g,tech)$pTechData(tech,'Hourly Variation'),1);
nREH2(g)= not REH2(g);

*-------------------------------------------------------------------
* TOPOLOGY DEFINITION
*-------------------------------------------------------------------

* Defining sTopology based on existing, committed and candidate transmission lines
sTopology(z,z2) = sum((q,y),pTransferLimit(z,z2,q,y)) + sum(pTransmissionHeader,pNewTransmission(z,z2,pTransmissionHeader)) + sum(pTransmissionHeader,pNewTransmission(z2,z,pTransmissionHeader));

* If not running in interconnected mode, set network to 0
sTopology(z,z2)$(not pinterconMode) = no;

* if ignore transfer limit, set limits to high value
pTransferLimit(sTopology,q,y)$pnoTransferLim = inf;

* Default life for transmission lines
pNewTransmission(sTopology,"Life")$(pNewTransmission(sTopology,"Life")=0 and pAllowHighTransfer) = 30; 

* Map transmission status from input data
tstatusmap(sTopology(z,z2),tstatus) = (pNewTransmission(z,z2, 'status')=tstatIndex(tstatus)) + (pNewTransmission(z2,z, 'status')=tstatIndex(tstatus));

* Identify candidate generators (`ng(g)`) based on their status in `gstatusmap`
commtransmission(sTopology(z,z2))  = tstatusmap(z,z2,'committed');

*-------------------------------------------------------------------
* CAPACITY CREDIT
*-------------------------------------------------------------------

* Identify the system peak demand for each year based on the highest total demand across all zones, times, and demand segments
pFindSysPeak(y)     = smax((t,d,q), sum(z, pDemandData(z,q,d,y,t)));


* Identify hours that are close to the peak demand for capacity credit calculations
pAllHours(q,d,y,t)  = 1$(abs(sum(z,pDemandData(z,q,d,y,t))/pFindSysPeak(y) - 1)<pMaxLoadFractionCCCalc);

* Default capacity credit for all generators is set to 1
pCapacityCredit(g,y)= 1;

* Protect against unintended changes while modifying `pVREgenProfile` with `pVREProfile` data
pVREgenProfile(VRE,q,d,t)$(not(pVREgenProfile(VRE,q,d,t))) = sum((z,tech)$(gzmap(VRE,z) and gtechmap(VRE,tech)),pVREProfile(z,tech,q,d,t));

* Set capacity credit for VRE based on predefined values or calculated generation-weighted availability
pCapacityCredit(VRE,y)$(pVRECapacityCredits =1) =  pGenData(VRE,"CapacityCredit")   ;
pCapacityCredit(VRE,y)$(pVRECapacityCredits =0) =  Sum((z,q,d,t)$gzmap(VRE,z),Sum(f$gfmap(VRE,f),pVREgenProfile(VRE,q,d,t)) * pAllHours(q,d,y,t)) * (Sum((z,f,q,d,t)$(gfmap(VRE,f) and gzmap(VRE,z) ),pVREgenProfile(VRE,q,d,t))/sum((q,d,t),1));

* Compute capacity credit for run-of-river hydro as an availability-weighted average
pCapacityCredit(ROR,y) =  sum(q,pAvailability(ROR,q)*sum((d,t),pHours(q,d,t)))/sum((q,d,t),pHours(q,d,t));

* Compute CSP and PV with storage generation profiles
pCSPProfile(cs,q,d,t)    = sum((z,tech)$(gtechmap(cs,tech) and gzmap(cs,z)), pVREProfile(z,tech,q,d,t));
pStoPVProfile(so,q,d,t)  =  sum((z,tech)$(gtechmap(so,tech) and gzmap(so,z)), pVREProfile(z,tech,q,d,t));

* H2 model parameters
pCapexTrajectoriesH2(hh,y) =1;
pCapexTrajectoriesH2(dch2,y)$pCaptraj = pCapexTrajectoryH2(dcH2,y);

*-------------------------------------------------------------------
* COST OF CAPITAL
*-------------------------------------------------------------------

* Set the weight of the start year to 1.0
pWeightYear(sStartYear) = 1.0;
* Compute weight for each year as the difference from the previous year's cumulative weight
pWeightYear(y)$(not sStartYear(y)) = y.val - sum(sameas(y2+1,y), y2.val) ;

* Compute the present value discounting factor considering mid-year adjustments
pRR(y) = 1.0;
pRR(y)$(ord(y)>1) = 1/((1+pDR)**(sum(y2$(ord(y2)<ord(y)),pWeightYear(y2))-1 + sum(sameas(y2,y), pWeightYear(y2)/2))) ;        
                                    
*-------------------------------------------------------------------
* Parameter Processing
*-------------------------------------------------------------------
pLossFactor(z2,z,y)$(pLossFactor(z,z2,y) and not pLossFactor(z2,z,y)) = pLossFactor(z,z2,y);

pEnergyEfficiencyFactor(z,y)$(not pincludeEE) = 1;
pEnergyEfficiencyFactor(z,y)$(pEnergyEfficiencyFactor(z,y)=0) = 1;

pVOMCost(gfmap(g,f),y) = pGenData(g,"VOM")
                       + pStorData(g, "VOMMWh")
                       + pCSPData(g, "Storage", "VOMMWh")
                       + pCSPData(g, "Thermal Field", "VOMMWh");

pFuelCost(g,f,y)$(gfmap(g,f)) = sum((gzmap(g,z),zcmap(z,c)),pFuelPrice(c,f,y)*pHeatRate(g,f));
pVarCost(g,f,y)$(gfmap(g,f)) = pFuelCost(g,f,y) + pVOMCost(g,f,y);
pVarCostH2(hh,y) = pH2Data(hh,"VOM");

* pCRF refers to the Capital Recovery Factor, which is used to calculate the annualized cost of capital for a project.
pCRF(g)$pGenData(g,'Life') = pWACC / (1 - (1 / ( (1 + pWACC)**pGenData(g,'Life'))));
pCRFH2(hh)$pH2Data(hh,'Life') = pWACC / (1 - (1 / ( (1 + pWACC)**pH2Data(hh,'Life'))));
pCRFsst(st)$pGenData(st,'Life') = pWACC / (1 - (1 / ( (1 + pWACC)**pGenData(st,'Life'))));
pCRFcst(cs)$pGenData(cs,'Life') = pWACC / (1 - (1 / ( (1 + pWACC)**pGenData(cs,'Life'))));
pCRFcth(cs)$pGenData(cs,'Life') = pWACC / (1 - (1 / ( (1 + pWACC)**pGenData(cs,'Life'))));

* Defines which connected zones belong to different countries. 
sMapConnectedZonesDiffCountries(sTopology(z,z2)) = sum(c$(zcmap(z,c) and zcmap(z2,c)), 1) = 0;

*** Simple bounds

* Set upper limit for generation capacity based on predefined data
vCap.up(g,y) = pGenData(g,"Capacity");

* Fix the build decision variable to zero for existing generation projects (started before or at the model start year)
vBuild.fx(eg,y)$(pGenData(eg,"StYr") <= sStartYear.val) = 0;

* Set the upper limit for new generation builds per year, accounting for the annual build limit and year weighting
vBuild.up(ng,y) = pGenData(ng,"BuildLimitperYear")*pWeightYear(y);

* Define the upper limit for additional transmission capacity, subject to high transfer allowance
vNewTransferCapacity.up(sTopology(z,z2),y)$pAllowHighTransfer = symmax(pNewTransmission,z,z2,"MaximumNumOfLines");

sAdditionalTransfer(sTopology(z,z2),y) = yes;
sAdditionalTransfer(sTopology(z,z2),y) $((y.val < pNewTransmission(z,z2,"EarliestEntry")) or (y.val < pNewTransmission(z2,z,"EarliestEntry"))) = no;

* Fix
vNewTransferCapacity.fx(commtransmission(z,z2),y)$((symmax(pNewTransmission,z,z2,"EarliestEntry") <= y.val) and pAllowHighTransfer) = symmax(pNewTransmission,z,z2,"MaximumNumOfLines");
vNewTransferCapacity.fx(commtransmission(z,z2),y)$(not sAdditionalTransfer(z,z2,y) and pAllowHighTransfer) = 0;

* Compute bounds 
vBuildTransmission.lo(sTopology(z,z2),y) = max(0,vNewTransferCapacity.lo(z,z2,y) - vNewTransferCapacity.up(z,z2,y-1));
vBuildTransmission.up(sTopology(z,z2),y) = max(0,vNewTransferCapacity.up(z,z2,y) - vNewTransferCapacity.lo(z,z2,y-1));

* Fix the storage build variable to zero if the project started before the model start year and storage is included
vBuildStor.fx(eg,y)$(pGenData(eg,"StYr") <= sStartYear.val and pincludeStorage) = 0;

* Fix the thermal build variable to zero if the project started before the model start year and CSP (Concentrated Solar Power) is included
vBuildTherm.fx(eg,y)$(pGenData(eg,"StYr") <= sStartYear.val and pincludeCSP) = 0;

*-------------------------------------------------------------------
* Fixed conditions
*-------------------------------------------------------------------

$ifthen set LOADSOLPATH
  execute_loadpoint "%LOADSOLPATH%%system.dirsep%PA_p.gdx", vCap.l, vRetire.l, vCapStor.l, vRetireStor.l

;

* first, handle the very first year
  vCap.fx(g,y)$(sStartYear(y)) = round(vCap.l(g,y),1);
* then enforce non–decreasing for all subsequent years
  loop(g,
    loop(y$(not sStartYear(y)),
        if( round(vCap.l(g,y),1) < round(vCap.l(g,y-1),1) ,
* bump up to the previous year’s level
          vCap.fx(g,y) = round(vCap.l(g,y-1),1) ;  
        else
* otherwise take its own rounded value
            vCap.fx(g,y) = round(vCap.l(g,y),1) ;    
        );
      );
    );
  
*  vCap.fx(g,y) = round(vCap.l(g,y), 1);
  vRetire.fx(g,y) = round(vRetire.l(g,y),1);
  vCapStor.fx(g,y) = round(vCapStor.l(g,y),1);
  vRetireStor.fx(g,y) = round(vRetireStor.l(g,y),1);

$endIf

* Fix capacity to zero for generation projects that have not yet started in a given year
vCap.fx(g,y)$(pGenData(g,"StYr") > y.val) = 0;

* Set the fixed capacity for existing generation projects at the start year, if they were commissioned before the model start year
vCap.fx(eg,sStartYear)$(pGenData(eg,"StYr") < sStartYear.val) = pGenData(eg,"Capacity");

* Set fixed capacity for generation projects in years where they are within their operational period
vCap.fx(eg,y)$((pGenData(eg,"StYr") <= y.val) and (pGenData(eg,"StYr") >= sStartYear.val)) = pGenData(eg,"Capacity");

* Retire capacity by setting it to zero in the year of retirement
vCap.fx(eg,y)$(pGenData(eg,"RetrYr") and (pGenData(eg,"RetrYr") <= y.val)) = 0;

* Set the initial thermal capacity for CSP plants at the model start year, if commissioned before that year
vCapTherm.fx(eg,sStartYear)$(pGenData(eg,"StYr") < sStartYear.val) = pCSPData(eg,"Thermal Field","CapacityMWh");

* Set the initial storage capacity at the model start year, considering both CSP and standalone storage units
vCapStor.fx(eg,sStartYear)$(pGenData(eg,"StYr") < sStartYear.val) = pCSPData(eg,"Storage","CapacityMWh") + pStorData(eg,"CapacityMWh");

* Prevent decommissioning of storage hours from existing storage when economic retirement is disabled
vCapStor.fx(eg,y)$((pSettings("econRetire") = 0) and (pGenData(eg,"StYr") <= y.val) and (pGenData(eg,"RetrYr") >= y.val)) = pStorData(eg,"CapacityMWh");

* Fix the retirement variable to zero, meaning no unit is retired by default unless specified otherwise
vRetire.fx(ng,y) = 0;

* Ensure plants with a lifetime of 99 years (considered effectively infinite) are not retired
vRetire.fx(eg,y)$(pGenData(eg,"Life") = 99) = 0;

* Ensure capacity remains unchanged when economic retirement is disabled and the plant is still within its operational lifetime
vCap.fx(eg,y)$((pSettings("econRetire") = 0 and pGenData(eg,"StYr") < y.val) and (pGenData(eg,"RetrYr") >= y.val)) = pGenData(eg,"Capacity");

* Prevent thermal capacity from appearing in years before the commissioning date
vCapTherm.fx(ng,y)$(pGenData(ng,"StYr") > y.val) = 0;

* Prevent storage capacity from appearing in years before the commissioning date
vCapStor.fx(ng,y)$(pGenData(ng,"StYr") > y.val) = 0;

* Ensure storage capacity is set to zero if storage is not included in the scenario
vCapStor.fx(ng,y)$(not pincludeStorage) = 0;


********************* Equations for hydrogen production**********************************************************
*Maximum capacity is equal to "Capacity"
vCapH2.up(hh,y) = pH2Data(hh,"Capacity");

*No new capacity can be built for existing generators
vBuildH2.fx(eh,y)$(pH2Data(eh,"StYr") <= sStartYear.val) = 0;

*The new capacity for new generators can not exceed the annual limit
vBuildH2.up(nh,y) = pH2Data(nh,"BuildLimitperYear")*pWeightYear(y);

*Electrolyzers can not operate at days with zero demand (applicable to dispatch model)
vH2PwrIn.fx(hh,q,d,t,y)$(sum(z,pDemandData(z,q,d,y,t))  =0)   =0;

*The maximum power drawn for H2 production can't be more than the Capacity of the electrolyzer
vH2PwrIn.up(hh,q,d,t,y)  =pH2Data(hh,"Capacity");


vCapH2.fx(hh,y)$(pH2Data(hh,"StYr") > y.val)=0;
vCapH2.fx(eh,sStartYear)$(pH2Data(eh,"StYr") < sStartYear.val) = pH2Data(eh,"Capacity");
vCapH2.fx(eh,y)$((pH2Data(eh,"StYr") <= y.val) and (pH2Data(eh,"StYr") >= sStartYear.val)) = pH2Data(eh,"Capacity");
vCapH2.fx(eh,y)$(pH2Data(eh,"RetrYr") and (pH2Data(eh,"RetrYr") <= y.val)) = 0;

vRetireH2.fx(nh,y) = 0;
vRetireH2.fx(eh,y)$(pH2Data(eh,"Life") = 99) = 0;

vCapH2.fx(eh,y)$((pSettings("econRetire") = 0 and pH2Data(eh,"StYr") < y.val) and (pH2Data(eh,"RetrYr") >= y.val)) = pH2Data(eh,"Capacity");

sH2PwrIn(hh,q,d,t,y) = yes;

vREPwr2H2.fx(nRE,f,q,d,t,y)=0;       
vREPwr2Grid.fx(nRE,f,q,d,t,y)=0;     

*******************************************************************************************************************
sPwrOut(gfmap(g,f),q,d,t,y) = yes;
sPwrOut(gfmap(st,f),q,d,t,y)$(not pincludeStorage) = yes;

* If price based export is not allowed, set to 0
sExportPrice(z,zext,q,d,t,y)$(pallowExports) = yes;
sImportPrice(z,zext,q,d,t,y)$(pallowExports) = yes;

sExportPrice(z,zext,q,d,t,y)$(not pallowExports) = no;
sImportPrice(z,zext,q,d,t,y)$(not pallowExports) = no;

vImportPrice.up(z,zext,q,d,t,y)$pallowExports = pExtTransferLimitIn(z,zext,q,y);
vExportPrice.up(z,zext,q,d,t,y)$pallowExports = pExtTransferLimitOut(z,zext,q,y);

* Do not allow imports and exports for a zone without import/export prices
sExportPrice(z,zext,q,d,t,y)$(pTradePrice(zext,q,d,y,t)= 0) = no;
sImportPrice(z,zext,q,d,t,y)$(pTradePrice(zext,q,d,y,t)= 0) = no;


sFlow(z,z2,q,d,t,y) = yes;
sFlow(z,z2,q,d,t,y)$(not sTopology(z,z2)) = no;

sSpinningReserve(g,q,d,t,y) = yes;
sSpinningReserve(g,q,d,t,y)$(not (pzonal_spinning_reserve_constraints or psystem_spinning_reserve_constraints) ) = no;

*To avoid bugs when there is no candidate transmission expansion line
pNewTransmission(z,z2,"EarliestEntry")$(not pAllowHighTransfer) = 2500;

*-------------------------------------------------------------------------------------
* Ensure that variables fixed (`.fx`) at specific values remain unchanged during the solve process  
PA.HoldFixed=1;

* Declare a file object `fmipopt` and set its name dynamically based on the solver
file fmipopt / %MIPSOLVER%.opt /;
* Check if the set `mipopt` contains any elements (i.e., if solver options exist)
if (card(mipopt),
 put fmipopt;
* Loop over each entry in `mipopt` and write its text content to the file
 loop(mipline, put mipopt.te(mipline) /);
 putclose;
); 

* Enable the solver to read an external solver option file
PA.optfile = 1;


* ############## SOLVE ##############
* SOLVEMODE == 2 solves as usual
* SOLVEMODE == 1 solves as usual but generates a savepoint file at the end
* SOLVEMODE == 0 uses a savepoint file to skip the solve (This speeds up development of post solve features)

$if not set SOLVEMODE $set SOLVEMODE 2 
$log LOG: Solving in SOLVEMODE = "%SOLVEMODE%"

* SOLVER TYPE
* MODELTYPE == MIP solves as a MIP
* MODELTYPE == RMIP forces to solve as an LP, even if there are integer variables

$if not set MODELTYPE $set MODELTYPE MIP
$log LOG: Solving with MODELTYPE = "%MODELTYPE%"

$ifThenI.solvemode %SOLVEMODE% == 2
*  Solve model as usual
   Solve PA using %MODELTYPE% minimizing vNPVcost;
*  Abort if model was not solved successfully
   abort$(not (PA.modelstat=1 or PA.modelstat=8)) 'ABORT: no feasible solution found.', PA.modelstat;
   execute_unload 'PA.gdx';
$elseIfI.solvemode %SOLVEMODE% == 1
*  Save model state at the end of execution (useful for debugging or re-running from a checkpoint)
   PA.savepoint = 1;
   Solve PA using %MODELTYPE% minimizing vNPVcost;
*  Abort if model was not solved successfully
   abort$(not (PA.modelstat=1 or PA.modelstat=8)) 'ABORT: no feasible solution found.', PA.modelstat;
$elseIfI.solvemode %SOLVEMODE% == 0
*  Only generate the model (no solve) 
   PA.JustScrDir = 1;
   Solve PA using %MODELTYPE% minimizing vNPVcost;
*  Use savepoint file to load state of the solve from savepoint file
   execute_loadpoint "PA_p.gdx";
$endIf.solvemode
* ####################################


$log ###############################
$log ##### GENERATING REPORT #####
$log ###############################

* Include the external report file specified by `%REPORT_FILE%`

$if not set REPORTSHORT $set REPORTSHORT 0
$log LOG: REPORTSHORT = "%REPORTSHORT%"

$include %REPORT_FILE%


