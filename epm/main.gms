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
* Claire Nicolas, cnicolas@worldbank.org
**********************************************************************

$offeolcom
$offinline
$inlinecom {  }
$eolcom //

$if not set DEBUG $set debug 0
$if not set EPMVERSION    $set EPMVERSION    9.0

* Turn on/off additional information to the listing file
option limRow=0, limCol=0, sysOut=off, solPrint=off;
$if %DEBUG%==1 $onUELlist onUELXRef onListing 
$if %DEBUG%==1 option limRow=2000, limCol=2000, solPrint=on, sysOut=off;

*-------------------------------------------------------------------------------------

* Useful for Python import
$setglobal modeldir %system.fp%

*-------------------------------------------------------------------------------------

* Folder input
$if not set FOLDER_INPUT $set FOLDER_INPUT "input/data_test"
$log FOLDER_INPUT is "%FOLDER_INPUT%"

*-------------------------------------------------------------------------------------

* By default modeltype is MIP
$if not set MODELTYPE $set MODELTYPE MIP
$log LOG: Solving with MODELTYPE = "%MODELTYPE%"
$if not set MODELTYPE   $set MODELTYPE MIP

* Use the relevant cplex file
$if not set CPLEXFILE   $set CPLEXFILE %FOLDER_INPUT%/cplex/cplex_baseline.opt
$ifi %MODELTYPE% == RMIP $set CPLEXFILE %FOLDER_INPUT%/cplex/cplex_baseline.opt

$log CPLEXFILE is "%CPLEXFILE%"
$call rm -f cplex.opt
$call cp "%CPLEXFILE%" cplex.opt

* Define modeltype-related options
Scalar
   modeltypeThreads 'Number of threads available to the modeltypes' /1/
   modeltypeOptCR   'Relative gap for MIP modeltype'                /0.05/
   modeltypeResLim  'modeltype time limit'                          /300000/
;   
Singleton Set
   NLPmodeltype 'Selected NLP modeltype' / '' 'conopt4' /
   MIPmodeltype 'Selected MIP modeltype' / '' 'cplex' /
;

* Evaluate and assign modeltype settings as macros
$eval     modeltypeTHREADS modeltypeThreads
$eval     modeltypeOPTCR   modeltypeOptCR
$eval     modeltypeRESLIM  modeltypeResLim
$eval.set NLPmodeltype     NLPmodeltype.te
$eval.set MIPmodeltype     MIPmodeltype.te

* Apply modeltype settings to GAMS execution
option NLP=%NLPmodeltype%, MIP=%MIPmodeltype%, threads=%modeltypeTHREADS%, optCR=%modeltypeOPTCR%, resLim=%modeltypeRESLIM%;

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


$ifThen not set DEMAND_FILE
$set DEMAND_FILE "generate_demand.gms"
$endIf
$if set ROOT_FOLDER $set DEMAND_FILE %ROOT_FOLDER%/%DEMAND_FILE%
$log DEMAND_FILE is "%DEMAND_FILE%"


$ifThen not set HYDROGEN_FILE
$set HYDROGEN_FILE "hydrogen_module.gms"
$endIf
$if set ROOT_FOLDER $set HYDROGEN_FILE %ROOT_FOLDER%/%HYDROGEN_FILE%
$log HYDROGEN_FILE is "%HYDROGEN_FILE%"

*-------------------------------------------------------------------------------------

* Core domain sets loaded from external files
Sets
   g        'Generators or technology-fuel types'
   f        'Fuels'
   tech     'Technologies'
   y        'Planning years'
   q        'Seasonal blocks (quarters)'
   d        'Representative days'
   t        'Hours of day'
   z        'System zones'
   c        'Countries'
   zext     'External trading zones'
   AT       'Full-year hourly chronology' /AT1*AT8760/
   pGenDataInputHeader 'Generator data input headers'
   pSettingsHeader
   pStorageDataHeader                                        'Storage data headers'
   pCSPDataHeader       'CSP data headers'                 / 'Storage', 'Thermal Field' /
   pTransmissionHeader  'Transmission data headers'
   pH2Header            'Hydrogen data headers'
   pTechFuelHeader      'Technology-fuel data headers'  / 'HourlyVariation', 'RETechnology', 'FuelIndex' /
   pe                   'Peak/energy tags for demand'      / 'peak', 'energy' /
   hh                   'Hydrogen production units'
;


* Relationship sets derived from the inputs
Sets
   zcmap(z<,c<) 'Zone-to-country mapping'
   gmap(g,z,tech,f) 'Generator-to-zone/technology/fuel mapping'
   sRelevant(d) 'Days where minimum generation limits apply'
   mapTS(q,d,t,AT) 'Mapping from season/day/hour tuples to chronological AT index'
;

alias (z,z2), (g,g1,g2);

* Input data parameters 
Parameter
* Generator data
   pTechFuel(tech<,f<,pTechFuelHeader)                    'Technology-fuel specifications'
   pGenDataInput(*,z,tech,f,pGenDataInputHeader)       'Generator data from Excel input'
   pGenDataInputDefault(z,tech,f,pGenDataInputHeader)    'Default generator data by zone/tech/fuel'
   pCapexTrajectoriesDefault(z,tech,f,y)                 'Default CAPEX trajectories'
   pCapexTrajectories(g,y)                               'Generator CAPEX trajectories'
   pAvailabilityDefault(z,tech,f,q)                      'Default availability factors'
   
* Storage data
   pStorageDataInput(*,z,tech,f,pStorageDataHeader)          'Storage unit specifications'
   
* CSP and technology data
   pCSPData(g,pCSPDataHeader,pStorageDataHeader)           'Concentrated solar power data'
   
* Fuel data
   pFuelCarbonContent(f)                                 'Carbon content by fuel (tCO2/MMBtu)'
   pMaxFuellimit(c,f,y)                                  'Fuel limit in MMBTU*1e6 (million) by country'
   pFuelPrice(c,f,y)                                     'Fuel price forecasts'

* Storage and transmission
   pNewTransmission(z,z2,pTransmissionHeader)           'New transmission line specifications'

* Trade parameters
   pTradePrice(zext,q,d,y,t)                             'External trade prices'
   pMaxAnnualExternalTradeShare(y,c)                     'Maximum trade share by country'
   
* Demand parameters
   pDemandProfile(z,q,d,t)                               'Normalized demand profiles'
   pDemandForecast(z,pe,y)                               'Peak/energy demand forecasts (MW/GWh)'
   pDemandData(z,q,d,y,t)                                'Hourly load curves'
   
* Emissions and carbon
   pEmissionsCountry(c,y)                                'Country emission limits (tons)'
   pEmissionsTotal(y)                                    'System-wide emission limits (tons)'
   pCarbonPrice(y)                                       'Carbon price (USD/ton CO2)'
   
* Time and transfer parameters
   pHours(q<,d<,t<)                                      'Hours mapping'
   pDays(q)                                              'Number of days represented by each period'
   pTransferLimit(z,z2,q,y)                              'Inter-zonal transfer limits'
   pMinImport(z2,z,y)                                    'Minimum import requirements'
   pLossFactorInternal(z,z2,y)                           'Transmission loss factors'
   
* VRE and availability
   pVREProfile(z,tech,q,d,t)                             'VRE generation profiles by site'
   pVREgenProfile(g,q,d,t)                               'VRE generation profiles by plant'
   pAvailabilityInput(g,q)                              'Seasonal availability factors (from CSV, no year)'
   pAvailability(g,y,q)                                  'Seasonal availability factors (expanded to years)'
   pEvolutionAvailability(g,y)                           'Year-dependent availability evolution factors'
   
* Reserve requirements
   pSpinningReserveReqCountry(c,y)                       'Country spinning reserve requirements'
   pSpinningReserveReqSystem(y)                          'System spinning reserve requirements'
   pPlanningReserveMargin(c)                             'Planning reserve margins'
   
* Other parameters
   pSettings(pSettingsHeader)                            'Model settings and penalties'
   pEnergyEfficiencyFactor(z,y)                          'Energy efficiency adjustment factors'
   pExtTransferLimit(z,zext,q,*,y)                       'External transfer limits'
   
* Hydrogen parameters
   pH2DataExcel(hh<,pH2Header)                           'Hydrogen data from Excel'
   pAvailabilityH2(hh,q)                                 'H2 plant availability'
   pFuelDataH2(f)                                        'Hydrogen fuel properties'
   pCapexTrajectoryH2(hh,y)                              'H2 CAPEX trajectories'
   pExternalH2(z,q,y)                                    'External H2 demand (MMBtu) to be met'
   
;   


* Add to pGenDataInputHeader the headers that are optional for users but necessary for the model

   
$if not errorfree $abort Error before reading input
*-------------------------------------------------------------------------------------
* Read inputs

* Include the external reader file defined by the macro variable %READER_FILE%
$include %READER_FILE%

* Open the specified GDX input file for reading
$gdxIn input.gdx

* Load domain-defining symbols (sets and indices)
$load zcmap pSettingsHeader y pHours pDays mapTS
$load pGenDataInputHeader, pTechFuel, pStorageDataHeader, 
$load g<pGenDataInput.Dim1
$load pGenDataInput gmap
$load pGenDataInputDefault pAvailabilityDefault pCapexTrajectoriesDefault
$load pSettings

* Load demand data
$load pDemandData pDemandForecast pDemandProfile pEnergyEfficiencyFactor sRelevant

$load pFuelCarbonContent pCarbonPrice pEmissionsCountry pEmissionsTotal pFuelPrice

* Load constraints and technical data
$load pMaxFuellimit pTransferLimit pLossFactorInternal pVREProfile pVREgenProfile pAvailabilityInput pEvolutionAvailability
* Use $loadM to merge storage units into set g (first dimension of pStorageDataInput)
$loadM g<pStorageDataInput.Dim1
$load pStorageDataInput pCSPData pCapexTrajectories pSpinningReserveReqCountry pSpinningReserveReqSystem 
$load pPlanningReserveMargin  

* Load trade data
$load zext, pTransmissionHeader
$load pExtTransferLimit, pNewTransmission, pMinImport
$load pTradePrice, pMaxAnnualExternalTradeShare

* Load Hydrogen model-related symbols
$load pH2Header, pH2DataExcel pAvailabilityH2 pFuelDataH2 pCAPEXTrajectoryH2 pExternalH2

* Close the GDX file after loading all required data
$gdxIn

$gdxunload afterReading.gdx 


$if not errorfree $abort CONNECT ERROR in input_readers.gms


*-------------------------------------------------------------------------------------
* Merge storage units from pStorageDataInput into generator structures
* This ensures all units (generators + storage) are in set g and have consistent data
*-------------------------------------------------------------------------------------

* Merge storage into gmap so storage units have zone/tech/fuel mappings
gmap(g,z,tech,f)$pStorageDataInput(g,z,tech,f,'Status') = yes;

* Fill pGenDataInput with storage data for common fields
* This ensures pGenData(g,header) includes storage units
* Loop over pGenDataInputHeader and copy values where header exists in both sets
loop(pGenDataInputHeader,
    pGenDataInput(g,z,tech,f,pGenDataInputHeader)$pStorageDataInput(g,z,tech,f,'Status')
        = sum(pStorageDataHeader$sameas(pStorageDataHeader,pGenDataInputHeader),
              pStorageDataInput(g,z,tech,f,pStorageDataHeader));
);

*-------------------------------------------------------------------------------------

* Make input verification
$log ##########################
$log ### INPUT VERIFICATION ###
$log ##########################


$onEmbeddedCode Python:
import sys, os

# Work from the original GDX on disk so zone/set pruning inside GAMS
# does not hide issues. "%cd%" points to the run directory where
# "input.gdx" already exists.
gms_dir = os.path.normpath(r"%modeldir%/")
if gms_dir not in sys.path:
    sys.path.insert(0, gms_dir)

from input_verification import run_input_verification_from_gdx
run_input_verification_from_gdx("input.gdx", verbose=False, log_func=gams.printLog)
$offEmbeddedCode 


$if not errorfree $abort PythonError in input_verification.py

*-------------------------------------------------------------------------------------
* Make input treatment

$log ########################
$log ### INPUT TREATMENT ####
$log ########################

$onMulti

$onEmbeddedCode Python:
import sys, os

# get directory of the .gms file
gms_dir = os.path.normpath(r"%modeldir%/")

# ensure it's in sys.path
if gms_dir not in sys.path:
    sys.path.insert(0, gms_dir)

from input_treatment import run_input_treatment
run_input_treatment(gams)
$offEmbeddedCode 

$if not errorfree $abort PythonError in input_treatment.py

$offMulti

*-------------------------------------------------------------------------------------

$if %DEBUG%==1 $log Debug mode active: exporting treated input to input_treated.gdx
$if %DEBUG%==1 execute_unload 'input_treated.gdx';

$if not errorFree $abort Data errors.

*-------------------------------------------------------------------------------------

* Only include BASE_FILE if this is a fresh run (i.e., not using a restart file)
* This prevents reloading sets, parameters, or data already available in the restart
$if "x%gams.restart%" == "x" $include %BASE_FILE%

$include %HYDROGEN_FILE%

*-------------------------------------------------------------------------------------
* Generate gfmap and others from pGenDataInput
parameter gstatIndex(gstatus) / Existing 1, Candidate 3, Committed 2 /;
parameter tstatIndex(tstatus) / Candidate 3, Committed 2 /;

*H2 model parameter
parameter H2statIndex(H2status) / Existing 1, Candidate 3, Committed 2 /;

             
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

* TODO: Update `gfmap(g,f)`, ensuring it includes additional mappings
* based on `pGenDataInput(g,z,tech,f2,'fuel2')` when a condition is met.
* gfmap(g,f) = gfmap(g,f) or sum((z,tech,f2), (pGenDataInput(g,z,tech,f2,'fuel2') = pTechFuel(tech,f,'FuelIndex')));
         

* Map generator status from input data
gstatusmap(g,gstatus) = sum((z,tech,f),pGenDataInput(g,z,tech,f,'status')=gstatIndex(gstatus));


pHeatrate(gprimf(g,f)) = sum((z,tech), pGenDataInput(g,z,tech,f,"Heatrate"));
pHeatrate(g,f2)$(gfmap(g,f2) and not gprimf(g,f2)) =
    sum((z,tech,f), pGenDataInput(g,z,tech,f,"Heatrate2")
*  $(pGenDataInput(g,z,tech,f,"fuel2") = pTechFuel(tech,f2,'FuelIndex'))
    );


pGenData(g,pGenDataInputHeader) = sum((z,tech,f),pGenDataInput(g,z,tech,f,pGenDataInputHeader));

***********************H2 model parameters***************************************************

pH2Data(hh,pH2Header)=pH2DataExcel(hh,pH2Header);
H2statusmap(hh,H2status) = pH2DataExcel(hh,'status')=H2statIndex(H2status);
* TODO: Check is that works for H2
* h2zmap(hh,z) = pH2DataExcel(hh,'Zone')=pZoneIndex(z);
h2zmap(hh,z) = pH2DataExcel(hh,'Zone');

*-------------------------------------------------------------------------------------

*--- Parameter initialisation for same demand profile for all years

$include %DEMAND_FILE%

*--- Part2: Start of initialisation of other parameters

* Read main parameters from pSettings

fEnableStorage                     = pSettings("fEnableStorage");
fDispatchMode                      = pSettings("fDispatchMode");

* --- Settings: Dispatch constraint switches
fApplyMinGenCommitment      = pSettings("fApplyMinGenCommitment");
fApplyStartupCost                  = pSettings("fApplyStartupCost");
fApplyRampConstraint               = pSettings("fApplyRampConstraint");
fApplyMUDT                       = pSettings("fApplyMUDT");

* --- Settings: Economic parameters and penalties
pDR                          = pSettings("DR");
pWACC                        = pSettings("WACC");
pVoLL                        = pSettings("VoLL");
pPlanningReserveVoLL         = pSettings("ReserveVoLL");
pSpinningReserveVoLL         = pSettings("SpinReserveVoLL");
pSurplusPenalty              = pSettings("CostSurplus");
pCostOfCurtailment           = pSettings("CostCurtail");
pCostOfCO2backstop           = pSettings("CO2backstop");
pH2UnservedCost              = pSettings("H2UnservedCost");
ssMaxCapitalInvestmentInvestment = pSettings("sMaxCapitalInvestment") * 1e6;

* --- Settings: Reserve requirements
fApplyPlanningReserveConstraint    = pSettings("fApplyPlanningReserveConstraint");
sReserveMarginPct                  = pSettings("sReserveMarginPct");
fApplyCountrySpinReserveConstraint = pSettings("fApplyCountrySpinReserveConstraint");
fApplySystemSpinReserveConstraint = pSettings("fApplySystemSpinReserveConstraint");
psVREForecastErrorPct              = pSettings("sVREForecastErrorPct");
sIntercoReserveContributionPct     = pSettings("sIntercoReserveContributionPct");
fCountIntercoForReserves           = pSettings("fCountIntercoForReserves");

* --- Settings: Policy and operational switches
fApplyMinGenShareAllHours      = pSettings("fApplyMinGenShareAllHours");
fApplyFuelConstraint               = pSettings("fApplyFuelConstraint");
fApplyCapitalConstraint            = pSettings("fApplyCapitalConstraint");
fEnableCSP                         = pSettings("fEnableCSP");
fEnableCapacityExpansion           = pSettings("fEnableCapacityExpansion");
pMinRE                             = pSettings("sMinRenewableSharePct");
pMinsRenewableTargetYear           = pSettings("sRenewableTargetYear");
fApplyCountryCo2Constraint         = pSettings("fApplyCountryCo2Constraint");
fApplySystemCo2Constraint         = pSettings("fApplySystemCo2Constraint");
fEnableCarbonPrice                     = pSettings("fEnableCarbonPrice");
fEnableEnergyEfficiency           = pSettings("fEnableEnergyEfficiency");


* --- Settings: Transmission and trade
fEnableInternalExchange            = pSettings("fEnableInternalExchange");
fRemoveInternalTransferLimit       = pSettings("fRemoveInternalTransferLimit");
fAllowTransferExpansion            = pSettings("fAllowTransferExpansion");
fAllowTransferExpansion            = fAllowTransferExpansion * fEnableCapacityExpansion;

fEnableExternalExchange            = pSettings("fEnableExternalExchange");
sMaxHourlyImportExternalShare                 = pSettings("sMaxHourlyImportExternalShare");
sMaxHourlyExportExternalShare                 = pSettings("sMaxHourlyExportExternalShare");

* --- Settings: Hydrogen options
fEnableCapexTrajectoryH2           = pSettings("fEnableCapexTrajectoryH2");
fEnableH2Production               = pSettings("fEnableH2Production");

* ----------------------------
singleton set sFinalYear(y);
scalar TimeHorizon;

sStartYear(y) = y.first;
sFinalYear(y) = y.last;
TimeHorizon = sFinalYear.val - sStartYear.val + 1;
sFirstHour(t) = t.first;
sLastHour(t) = t.last;
sFirstDay(d) = d.first;

sFirstHourAT(AT) = AT.first;

sLastDay(d) = yes$(ord(d) = card(d));

* ------------------------------
* Commitment initialization defaults
* Consolidates initial ON/OFF values so a single section shows which units start online.
* ------------------------------
pInitialOnStart(g) = 0;
pInitialOnStart(g)$pGenData(g,"InitialOn") = pGenData(g,"InitialOn");


pStorageInitShare = pSettings("InitialSOCforBattery");
if (pStorageInitShare < 0,
   pStorageInitShare = 0.5;
);

FD(q,d,t) = yes;
if (fDispatchMode,
   FD(q,d,t)$(ord(q) eq 2 and ord(d) > 28) = no;
   FD(q,d,t)$((ord(q) eq 4 or ord(q) eq 6 or ord(q) eq 9 or ord(q) eq 11) and ord(d) > 30) = no;
);

* Identify which generators carry minimum-generation commitments so that startup cost logic can focus on them.
MinGenPoint(g) = yes$(pGenData(g,"MinGenCommitment") > 0);

* ------------------------------


* Set external transfer limits to zero if exports are not allowed
pExtTransferLimit(z,zext,q,"Import",y)$(not fEnableExternalExchange)  = 0 ;
pExtTransferLimit(z,zext,q,"Export",y)$(not fEnableExternalExchange)  = 0 ;

* Assign import and export transfer limits only if exports are allowed
pExtTransferLimitIn(z,zext,q,y)$fEnableExternalExchange   = pExtTransferLimit(z,zext,q,"Import",y) ;
pExtTransferLimitOut(z,zext,q,y)$fEnableExternalExchange  = pExtTransferLimit(z,zext,q,"Export",y) ;

* Define `Zt(z)` to check if total demand in a zone `z` is zero
Zt(z) = sum((q,d,y,t),pDemandData(z,q,d,y,t)) = 0;
* Define `Zd(z)` as the complement of `Zt(z)`, indicating zones with demand
Zd(z) = not Zt(z);


* Aggregate storage data from the 4-index pStorageDataInput to pStorageData(g,header)
pStorageData(g,pStorageDataHeader) = sum((z,tech,f), pStorageDataInput(g,z,tech,f,pStorageDataHeader));

* gsmap is used for linked storage (PV+storage pairs)
* gsmap(g2,g) means storage g is linked to generator g2
* For standalone storage (empty "Linked plant"), gsmap remains empty
* Note: If linked storage is needed, the "Linked plant" column should contain the generator name
gsmap(g2,g) = no;

* Identify candidate generators (`ng(g)`) based on their status in `gstatusmap`
ng(g)  = gstatusmap(g,'candidate') or gstatusmap(g,'committed');

* Define existing generators (`eg(g)`) as those that are not candidates, include comitted
eg(g)  = not ng(g);

* Identify variable renewable energy (VRE) generators (`vre(g)`) based on hourly variation data
vre(g) = sum((gtechmap(g,tech),f)$pTechFuel(tech,f,'HourlyVariation'),1);

* Identify renewable energy (RE) generators (`re(g)`) based on RE technology classification
re(g)  = sum((gtechmap(g,tech),f)$pTechFuel(tech,f,'RETechnology'),1);

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
REH2(g)= sum((gtechmap(g,tech),f)$pTechFuel(tech,f,'RETechnology'),1) - sum((gtechmap(g,tech),f)$pTechFuel(tech,f,'HourlyVariation'),1);
nREH2(g)= not REH2(g);

*-------------------------------------------------------------------
* TOPOLOGY DEFINITION
*-------------------------------------------------------------------

* Defining sTopology based on existing, committed and candidate transmission lines
sTopology(z,z2) = sum((q,y),pTransferLimit(z,z2,q,y)) + sum(pTransmissionHeader,pNewTransmission(z,z2,pTransmissionHeader)) + sum(pTransmissionHeader,pNewTransmission(z2,z,pTransmissionHeader));

* If not running in interconnected mode, set network to 0
sTopology(z,z2)$(not fEnableInternalExchange) = no;

* if ignore transfer limit, set limits to high value
pTransferLimit(sTopology,q,y)$fRemoveInternalTransferLimit = inf;

* Default life for transmission lines
pNewTransmission(sTopology,"Life")$(pNewTransmission(sTopology,"Life")=0 and fAllowTransferExpansion) = 30; 

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
pAllHours(q,d,y,t)  = 1$(abs(sum(z,pDemandData(z,q,d,y,t))/pFindSysPeak(y) - 1)<pSettings("sPeakLoadProximityThreshold"));

* Default capacity credit for all generators is set to 1
pCapacityCredit(g,y)= 1;

* Protect against unintended changes while modifying `pVREgenProfile` with `pVREProfile` data
pVREgenProfile(VRE,q,d,t)$(not(pVREgenProfile(VRE,q,d,t))) = sum((z,tech)$(gzmap(VRE,z) and gtechmap(VRE,tech)),pVREProfile(z,tech,q,d,t));

* Set capacity credit for VRE based on predefined values or calculated generation-weighted availability
pCapacityCredit(VRE,y) =  Sum((z,q,d,t)$gzmap(VRE,z),Sum(f$gfmap(VRE,f),pVREgenProfile(VRE,q,d,t)) * pAllHours(q,d,y,t)) * (Sum((z,f,q,d,t)$(gfmap(VRE,f) and gzmap(VRE,z) ),pVREgenProfile(VRE,q,d,t))/sum((q,d,t),1));

* Compute capacity credit for run-of-river hydro as an availability-weighted average
pCapacityCredit(ROR,y) =  sum(q,pAvailability(ROR,y,q)*sum((d,t),pHours(q,d,t)))/sum((q,d,t),pHours(q,d,t));

* Compute CSP and PV with storage generation profiles
pCSPProfile(cs,q,d,t)    = sum((z,tech)$(gtechmap(cs,tech) and gzmap(cs,z)), pVREProfile(z,tech,q,d,t));
pStoPVProfile(so,q,d,t)  =  sum((z,tech)$(gtechmap(so,tech) and gzmap(so,z)), pVREProfile(z,tech,q,d,t));

* H2 model parameters
pCapexTrajectoriesH2(hh,y) =1;
pCapexTrajectoriesH2(dch2,y)$fEnableCapexTrajectoryH2 = pCapexTrajectoryH2(dcH2,y);

*-------------------------------------------------------------------
* Cost of capital
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
pLossFactorInternal(z2,z,y)$(pLossFactorInternal(z,z2,y) and not pLossFactorInternal(z2,z,y)) = pLossFactorInternal(z,z2,y);

pEnergyEfficiencyFactor(z,y)$(not fEnableEnergyEfficiency) = 1;
pEnergyEfficiencyFactor(z,y)$(pEnergyEfficiencyFactor(z,y)=0) = 1;

pVOMCost(gfmap(g,f),y) = pGenData(g,"VOM")
                       + pStorageData(g, "VOMMWh")
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
vNewTransmissionLine.up(sTopology(z,z2),y)$fAllowTransferExpansion = symmax(pNewTransmission,z,z2,"MaximumNumOfLines");

sAdditionalTransfer(sTopology(z,z2),y) = yes;
sAdditionalTransfer(sTopology(z,z2),y) $((y.val < pNewTransmission(z,z2,"EarliestEntry")) or (y.val < pNewTransmission(z2,z,"EarliestEntry"))) = no;

* Fix
vNewTransmissionLine.fx(commtransmission(z,z2),y)$((symmax(pNewTransmission,z,z2,"EarliestEntry") <= y.val) and fAllowTransferExpansion) = symmax(pNewTransmission,z,z2,"MaximumNumOfLines");
vNewTransmissionLine.fx(commtransmission(z,z2),y)$(not sAdditionalTransfer(z,z2,y) and fAllowTransferExpansion) = 0;

* Compute bounds 
vBuildTransmissionLine.lo(sTopology(z,z2),y) = max(0,vNewTransmissionLine.lo(z,z2,y) - vNewTransmissionLine.up(z,z2,y-1));
vBuildTransmissionLine.up(sTopology(z,z2),y) = max(0,vNewTransmissionLine.up(z,z2,y) - vNewTransmissionLine.lo(z,z2,y-1));

* Fix the storage build variable to zero if the project started before the model start year and storage is included
vBuildStor.fx(eg,y)$(pGenData(eg,"StYr") <= sStartYear.val and fEnableStorage) = 0;

* Fix the thermal build variable to zero if the project started before the model start year and CSP (Concentrated Solar Power) is included
vBuildTherm.fx(eg,y)$(pGenData(eg,"StYr") <= sStartYear.val and fEnableCSP) = 0;

* Disable all capacity expansion decisions when flag is off
if (fEnableCapacityExpansion = 0,
   vBuild.fx(g,y)            = 0;
   vBuiltCapVar.fx(g,y)      = 0;
   vBuildStor.fx(g,y)        = 0;
   vBuildTherm.fx(g,y)       = 0;
   vBuildTransmissionLine.fx(z,z2,y) = 0;
   vNewTransmissionLine.fx(z,z2,y)   = 0;
   vBuildH2.fx(hh,y)         = 0;
   vBuiltCapVarH2.fx(hh,y)   = 0;
);

*-------------------------------------------------------------------
* Fixed conditions
*-------------------------------------------------------------------


********************* Load a previously saved solution**********************************************************

* Load a previously saved solution from the specified path (if LOADSOLPATH is defined)
* This solution contains the .l (level) values of capacity-related variables from a previous run
$ifthen set LOADSOLPATH
  execute_loadpoint "%LOADSOLPATH%%system.dirsep%PA_p.gdx", vCap.l, vRetire.l, vCapStor.l, vRetireStor.l
;

* --- Fix capacity and retirement variables to values from the previous solution ---

* Step 1: Handle the first model year
* Fix vCap (generation capacity) for each generator g in the first year y to its rounded value
  vCap.fx(g,y)$(sStartYear(y)) = round(vCap.l(g,y),1);

* Step 2: Enforce non-decreasing capacity in all following years
  loop(g,
    loop(y$(not sStartYear(y)),
        if( round(vCap.l(g,y),1) < round(vCap.l(g,y-1),1) ,
* If the capacity in year y is less than the previous year (y-1), fix it to the previous year’s value
          vCap.fx(g,y) = round(vCap.l(g,y-1),1) ;  
        else
* Otherwise take its own rounded value
            vCap.fx(g,y) = round(vCap.l(g,y),1) ;    
        );
      );
    );
  
* Step 3: Fix other capacity-related variables to their rounded solution values
* Fix retirements and storage capacities from previous solution
  vRetire.fx(g,y) = round(vRetire.l(g,y),1);
  vCapStor.fx(g,y) = round(vCapStor.l(g,y),1);
  vRetireStor.fx(g,y) = round(vRetireStor.l(g,y),1);

$endIf

********************* Equations for generation capacity**********************************************************
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
vCapStor.fx(eg,sStartYear)$(pGenData(eg,"StYr") < sStartYear.val) = pCSPData(eg,"Storage","CapacityMWh") + pStorageData(eg,"CapacityMWh");

* Prevent decommissioning of storage hours from existing storage when economic retirement is disabled
vCapStor.fx(eg,y)$((pSettings("fEnableEconomicRetirement") = 0) and (pGenData(eg,"StYr") <= y.val) and (pGenData(eg,"RetrYr") >= y.val)) = pStorageData(eg,"CapacityMWh");

* Fix the retirement variable to zero, meaning no unit is retired by default unless specified otherwise
vRetire.fx(ng,y) = 0;

* Ensure plants with a lifetime of 99 years (considered effectively infinite) are not retired
vRetire.fx(eg,y)$(pGenData(eg,"Life") = 99) = 0;

* Ensure capacity remains unchanged when economic retirement is disabled and the plant is still within its operational lifetime
vCap.fx(eg,y)$((pSettings("fEnableEconomicRetirement") = 0 and pGenData(eg,"StYr") < y.val) and (pGenData(eg,"RetrYr") >= y.val)) = pGenData(eg,"Capacity");

* Prevent thermal capacity from appearing in years before the commissioning date
vCapTherm.fx(ng,y)$(pGenData(ng,"StYr") > y.val) = 0;

* Prevent storage capacity from appearing in years before the commissioning date
vCapStor.fx(ng,y)$(pGenData(ng,"StYr") > y.val) = 0;

* Ensure storage capacity is set to zero if storage is not included in the scenario
vCapStor.fx(ng,y)$(not fEnableStorage) = 0;


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

vCapH2.fx(eh,y)$((pSettings("fEnableEconomicRetirement") = 0 and pH2Data(eh,"StYr") < y.val) and (pH2Data(eh,"RetrYr") >= y.val)) = pH2Data(eh,"Capacity");

sH2PwrIn(hh,q,d,t,y) = yes;

vREPwr2H2.fx(nRE,f,q,d,t,y)=0;       
vREPwr2Grid.fx(nRE,f,q,d,t,y)=0;     

*******************************************************************************************************************

sPwrOut(gfmap(g,f),q,d,t,y) = yes;
sPwrOut(gfmap(st,f),q,d,t,y)$(not fEnableStorage) = yes;

* If price based export is not allowed, set to 0
sExportPrice(z,zext,q,d,t,y)$(fEnableExternalExchange) = yes;
sExportPrice(z,zext,q,d,t,y)$(not fEnableExternalExchange) = no;

* If price based import is not allowed, set to 0
sImportPrice(z,zext,q,d,t,y)$(fEnableExternalExchange) = yes;
sImportPrice(z,zext,q,d,t,y)$(not fEnableExternalExchange) = no;

vYearlyImportExternal.up(z,zext,q,d,t,y)$fEnableExternalExchange = pExtTransferLimitIn(z,zext,q,y);
vYearlyExportExternal.up(z,zext,q,d,t,y)$fEnableExternalExchange = pExtTransferLimitOut(z,zext,q,y);

* Do not allow imports and exports for a zone without import/export prices
sExportPrice(z,zext,q,d,t,y)$(pTradePrice(zext,q,d,y,t)= 0) = no;
sImportPrice(z,zext,q,d,t,y)$(pTradePrice(zext,q,d,y,t)= 0) = no;

* Define the flow of electricity between zones
sFlow(z,z2,q,d,t,y)$(sTopology(z,z2)) = yes;

* Define spinning reserve constraints based on the settings for zonal and system spinning reserves
sSpinningReserve(g,q,d,t,y)$((fApplyCountrySpinReserveConstraint or fApplySystemSpinReserveConstraint) ) = yes;

*To avoid bugs when there is no candidate transmission expansion line
pNewTransmission(z,z2,"EarliestEntry")$(not fAllowTransferExpansion) = 2500;

*-------------------------------------------------------------------------------------
* Ensure that variables fixed (`.fx`) at specific values remain unchanged during the solve process  
PA.HoldFixed=1;

* Declare a file object `fmipopt` and set its name dynamically based on the modeltype
file fmipopt / %MIPmodeltype%.opt /;
* Check if the set `mipopt` contains any elements (i.e., if modeltype options exist)
if (card(mipopt),
 put fmipopt;
* Loop over each entry in `mipopt` and write its text content to the file
 loop(mipline, put mipopt.te(mipline) /);
 putclose;
); 

* Enable the modeltype to read an external modeltype option file
PA.optfile = 1;


* ############## SOLVE ##############
* SOLVEMODE == 2 solves as usual
* SOLVEMODE == 1 solves as usual but generates a savepoint file at the end
* SOLVEMODE == 0 uses a savepoint file to skip the solve (This speeds up development of post solve features)

$if not set SOLVEMODE $set SOLVEMODE 2 
$log LOG: Solving in SOLVEMODE = "%SOLVEMODE%"


$ifThenI.solvemode %SOLVEMODE% == 2
*  Solve model as usual
   Solve PA using %MODELTYPE% minimizing vNPVcost;
*  Abort if model was not solved successfully
   abort$(not (PA.modelstat=1 or PA.modelstat=8)) 'ABORT: no feasible solution found.', PA.modelstat;
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
* Use savepoint file to load state of the solve from savepoint file
   $log LOG: USE_PA_LOADPOINT = "%USE_PA_LOADPOINT%"
   execute_loadpoint "PA_p.gdx";
$endIf.solvemode

*  Export model data only when debugging is enabled
$if %DEBUG%==1 execute_unload 'PA.gdx';
* ####################################

* Include the external report file specified by `%REPORT_FILE%`

$if not set REPORTSHORT $set REPORTSHORT 0
$log LOG: REPORTSHORT = "%REPORTSHORT%"

$include %REPORT_FILE%
