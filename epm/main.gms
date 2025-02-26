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
$if not set EPMVERSION    $set EPMVERSION    8.5

$ifThen.mode not set mode
$ set mode Excel
$ if not "x%gams.IDCGDXInput%"=="x" $set mode MIRO
$endIf.mode


***
*** Declarations
***


* Turn on/off additional information to the listing file
Option limRow=0, limCol=0, sysOut=off, solPrint=off;
$if %DEBUG%==1 $onUELlist onUELXRef onListing 
$if %DEBUG%==1 Option limRow=1e9, limCol=1e9, sysOut=on, solPrint=on;

*-------------------------------------------------------------------------------------

* Only include base if we don't restart
$ifThen not set BASE_FILE
$set BASE_FILE "base.gms"
$endIf
$log BASE_FILE is "%BASE_FILE%"

$if "x%gams.restart%" == "x" $include %BASE_FILE%

$ifThen not set REPORT_FILE
$set REPORT_FILE "generate_report.gms"
$endIf
$log REPORT_FILE is "%REPORT_FILE%"


$ifThen not set READER_FILE
$set READER_FILE "input_readers.gms"
$endIf
$log READER_FILE is "%READER_FILE%"

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

* Determine Excel-based reporting
$ifThenI.mode %mode%==Excel
*$set main Excel
$set DOEXCELREPORT 1

* Set input Excel file if not already defined
$ifThen not set XLS_INPUT
* If GDX input is set, derive XLS_INPUT from it (is it used ?)
$  if     set GDX_INPUT $set XLS_INPUT "%GDX_INPUT%.%ext%"
* Otherwise, set the default input file path
$  if not set GDX_INPUT $set XLS_INPUT input%system.dirsep%input_epm.xlsx
$endIf

* Extract file path, base name, and extension from XLS_INPUT
$setNames "%XLS_INPUT%" fp GDX_INPUT fe

* Set the default output Excel file if not already defined
$if not set XLS_OUTPUT $set XLS_OUTPUT %fp%EPMRESULTS.xlsx

* Append a timestamp to the output filename if USETIMESTAMP is enabled
$ifThen.timestamp set USETIMESTAMP
$  setNames "%XLS_OUTPUT%" fp fn fe
$  onembeddedCode Python:
   import datetime
   import os
   os.environ['MYDATE'] = datetime.datetime.now().strftime("%Y_%m_%d") # we can add hour, minute, or seconds if necessary: %H %M %S
$  offEmbeddedCode
$  set XLS_OUTPUT %fp%%fn%_%sysenv.MYDATE%%fe%
$endIf.timestamp

*-------------------------------------------------------------------------------------

$call 'rm -f miro.log'
file log / miro.log /;
put log '------------------------------------'/;
put log '        Data validation'/;
put log '------------------------------------'/;

Set
   tech     'technologies'
   gstatus  'generator status' / Existing, Candidate, Committed /
   H2status  'H2 generation plant status' / Existing, Candidate, Committed /
   techhdr  'techdata headers' / 'Construction Period (years)', 'RE Technology', 'Hourly Variation' /
   pe       'peak energy for demand forecast' /'peak', 'energy'/
   ft       'fuel types'
   mipline 'Solver option lines'
   sc       'settings' /
                        allowExports
                        altDemand
                        capCreditSolar
                        capCreditWind
                        capital_constraints
                        captraj
                        costSurplus
                        costcurtail
                        CO2backstop
                        dr
                        fuel_constraints
                        econRetire
                        includeCSP
                        includeCarbonPrice
                        includeEE
                        includeStorage
                        interconMode
                        maxCapital
                        maxExports
                        maxImports
                        VREForecastError
                        minREshare
                        mingen_constraints
                        noTransferLim
                        pAllowHighTransfer
                        planning_reserve_constraints
                        ramp_constraints
                        reTargetYr
                        reserveVoLL
                        seasonalreporting
                        spinReserveVoLL
                        system_co2_constraints
                        system_reserve_margin
                        system_spinning_reserve_constraints
                        systemresultreporting
                        vOLL
                        vRECapacityCredits
                        wACC
                        zonal_co2_constraints
                        zonal_spinning_reserve_constraints
                        IncludeDecomCom
                        MaxLoadFractionCCCalc                                   
                        IncludeH2
                        H2UnservedCost 
                       /
                       
;

alias (y,y2);
alias (f,f1,f2);
alias (c,c2);

$ifi %mode%==MIRO
$onExternalInput
Set
   ftfmap(ft<,f<)                   'map fuel types to fuels'
   zcmap(z,c<)                     'map zones to countries'
   sTopology(z,z2)                  'network topology - to be assigned through network data'
   Peak(t)                          'peak period hours'
   Relevant(d)                      'relevant day and hours when MinGen limit is applied'
   mipopt(mipline<)                 / system.empty /;
;   
Parameter
   pCSPData(g,csphrd,shdr)
   pTechData(tech<,techhdr)         'Technology data'
   pFuelTypeCarbonContent(ft)       'Fuel type carbon content in tCO2 per MMBTu'
   pStorDataInput(g,g2,shdr)        'Storage data'
   pNewTransmission(z,z2,thdr)      'new transmission lines'
   pTradePrice(zext,q,d,y,t)           'trade price - export or import driven by prices [assuming each zone in a country can only trade with one external zone]'
   pMaxExchangeShare(y,c)           'Max share of exchanges by country [same limit for imports or exports for now]'
   pDemandProfile(z,q,d,t)          'Demand profile in per unit'
   pDemandForecast(z,pe,y)          'Peak and Energy demand forecast in MW and GWh'
   pDemandData(z,q,d,y,t)           'hourly load curves by quarter(seasonal) and year'
   pEmissionsCountry(c,y)              'Maximum zonal emissions allowed per country and year in tns'
   pEmissionsTotal(y)               'Maximum total emissions allowed per year for the region in tns'
   pCarbonPrice(y)                  'Carbon price in USD per ton of CO2eq'
$ifi     %mode%==MIRO   pHours(q<,d<,t<) 'duration of each block'
$ifi not %mode%==MIRO   pHours(q<,d<,t<) 'duration of each block'
   pTransferLimit(z,z2,q,y)         'Transfer limits by quarter (seasonal) and year between zones'
   pMinImport(z2,z,y)               'Minimum trade constraint between zones defined at the yearly scale, and applied uniformly across each hour'
   pLossFactor(z,z2,y)              'loss factor in percentage'
   pVREProfile(z,*,q,d,t)           'VRE generation profile by site quarter day type and YEAR -- normalized (per MW of solar and wind capacity)'
   pVREgenProfile(g,f,q,d,t)        'VRE generation profile by plant quarter day type and YEAR -- normalized (per MW of solar and wind capacity)'
   pAvailability(g,q)               'Availability by generation type and season or quarter in percentage - need to reflect maintenance'
   pSpinningReserveReqCountry(c,y)     'Spinning reserve requirement local at country level (MW)  -- for isolated system operation scenarios'
   pSpinningReserveReqSystem(y)     'Spinning reserve requirement systemwide (MW) -- for integrated system operation scenarios'
   pScalars(sc)                     'Flags and penalties to load'
   pPlanningReserveMargin(c)        'Country planning reserve margin'
   pEnergyEfficiencyFactor(z,y)     'Scaling factor for energy efficiency measures'
   pExtTransferLimit(z,zext,q,*,y)  'transfer limits by quarter (seasonal) and year with external zones'
  pH2Data(hh,hhdr)                  'H2 production unit specs'
  pAvailabilityH2(hh,q)             'Availability by H2 generation plant and season or quarter in percentage - need to reflect maintenance'
  pFuelData(f)                     'Hydrogen fuels'
  pCapexTrajectoryH2(hh,y)          'CAPEX trajectory for hydrogen generation unit'
;   

$ifi %mode%==MIRO
$offExternalInput

Parameter
   pCapexTrajectories(g,y)          'capex trajectories for all generators (used in results)'
   pAllHours(q,d,y,t)               'Hour of system peak'
   pFuelPrice(c,f,y)                'Fuel price forecasts by country'
   pFindSysPeak(y)                  'System peak per year'
   pEmissionsTotal(y)               'Maximum total emissions allowed per year for the region in tns'
   pSeasonalReporting               'seasonal reporting flag'
   pSystemResultReporting           'system reporting file flag'
   pInterConMode                    'interconnected mode flag'
   pNoTransferLim                   'transfer limit flag'
   pAllowExports                    'Allow price based exports'
   pVRECapacityCredits              'User input capacity credits'
   pDR                              'discount rate'
   pCaptraj                         'allow capex trajectory'
   pIncludeEE                       'energy efficiency factor flag'
   pSystem_CO2_constraints
   pExtTransferLimitIn(z,zext,q,y)  'transfer limit with external zone for import towards internal zone'
   pExtTransferLimitOut(z,zext,q,y) 'transfer limit with external zone for export towards external zone'
   pMaxLoadFractionCCCalc           'maximum percentage difference between hourly load and peak load to consider in the capacity credit calculation' 
   pVREForecastError                'Percentage error in VRE forecast [used to estimated required amount of spinning reserve]'

;

Set gprimf(g,f)          'primary fuel f for generator g'
   tech / ROR            'Run of river hydro'
          CSP            'Concentrated Solar Power'
          PVwSTO         'Solar PV with Storage'
          STOPV          'Storage For PV'
          STORAGE        'Grid Connected Storage' /
   gtechmap(g,tech)      'Generator technology map'
   gstatusmap(g,gstatus) 'Generator status map'
   Offpeak(t)            'offpeak hours'
   Zd(z)
   Zt(z)
   stg(g)                'Grid tied storage'
   ror(g)                'ROR generators'   
   H2statusmap(hh,H2status)
;
$onmulti
Set
   ghdr         'Additional headers for pGenData' / CapacityCredit, Heatrate, Heatrate2, Life, VOM /
   shdr         'Additional headers for pStorData' / Life, VOM /
   thdr         'Additional header for pNewTransmission' / EarliestEntry /
;
$offmulti


Set
   zcmapExcel(z,c<)
   gmap(g,z,tech,f) 'Map generators to firms, zones, technologies and fuels'
   
;
Parameter
   pGenDataExcel(g<,z<,tech,f<,*)
   pStorDataExcel(g,*,shdr)   
   pH2DataExcel(hh<,*)
   pGenDataExcelDefault(z<,tech,f<,*)
   pCapexTrajectoriesDefault(z<,tech,f<,y)
   pAvailabilityDefault(z<,tech,f<,q)

* Allow multiple definitions of symbols without raising an error (use with caution)
$onMulti
   pTechDataExcel(tech<,*)
;

Parameter
   ftfindex(ft<,f)
   pZoneIndex(z)
;
   
   
*-------------------------------------------------------------------------------------
* Read inputs

* Include the external reader file defined by the macro variable %READER_FILE%
$include %READER_FILE%

* Open the specified GDX input file for reading
$gdxIn %GDX_INPUT%

* Load domain-defining symbols (sets and indices)
$load y pHours pTechDataExcel
$load pGenDataExcel pGenDataExcelDefault pAvailabilityDefault pCapexTrajectoriesDefault
$load zext ftfindex gmap zcmapExcel

* Load general model parameters related to demand and emissions
$load peak Relevant pDemandData pDemandForecast pDemandProfile
$load pFuelTypeCarbonContent pCarbonPrice pEmissionsCountry pEmissionsTotal pFuelPrice

* Load constraints and technical data
$load pMaxFuellimit pTransferLimit pLossFactor pVREProfile pVREgenProfile pAvailability
$load pStorDataExcel pCSPData pCapexTrajectories pSpinningReserveReqCountry pSpinningReserveReqSystem pScalars
$load sTopology pPlanningReserveMargin pEnergyEfficiencyFactor pTradePrice pMaxExchangeShare

* Load external transfer limits and transmission constraints
$load pExtTransferLimit
$load pNewTransmission
$load pMinImport

* Load Hydrogen model-related symbols
$load pH2DataExcel hh pAvailabilityH2 pFuelData pCAPEXTrajectoryH2 pExternalH2

* Close the GDX file after loading all required data
$gdxIn
$offmulti

*-------------------------------------------------------------------------------------
* Make input verification

$include input_verification.gms

*-------------------------------------------------------------------------------------
* Make input treatment
$onMulti
$include input_treatment.gms
$offMulti
*-------------------------------------------------------------------------------------



execute_unload "input.gdx" y pHours pTechDataExcel pGenDataExcel pGenDataExcelDefault pAvailabilityDefault pCapexTrajectoriesDefault
zext ftfindex gmap zcmapExcel
peak Relevant pDemandData pDemandForecast
pDemandProfile pFuelTypeCarbonContent pCarbonPrice pEmissionsCountry
pEmissionsTotal pFuelPrice pMaxFuellimit pTransferLimit pLossFactor pVREProfile pVREgenProfile pAvailability
pStorDataExcel pCSPData pCapexTrajectories pSpinningReserveReqCountry pSpinningReserveReqSystem pScalars
pStorDataExcel pCSPData pCapexTrajectories pSpinningReserveReqCountry pSpinningReserveReqSystem pScalars
sTopology pPlanningReserveMargin pEnergyEfficiencyFactor pTradePrice pMaxExchangeShare
pExtTransferLimit pNewTransmission pMinImport
pH2DataExcel hh pAvailabilityH2 pFuelData pCAPEXTrajectoryH2 pExternalH2
;

display pAvailability, pCapexTrajectories;

option ftfmap<ftfindex;
pStorDataInput(g,g2,shdr) = pStorDataExcel(g,g2,shdr);
pStorDataInput(g,g,shdr)$pStorDataExcel(g,'',shdr) = pStorDataExcel(g,'',shdr);

$if not errorFree $echo Data errors. Please inspect the listing file for details. > miro.log

* Generate gfmap and others from pGenDataExcel
parameter gstatIndex(gstatus) / Existing 1, Candidate 3, Committed 2 /;

*H2 model parameter
parameter H2statIndex(H2status) / Existing 1, Candidate 3, Committed 2 /;


set addHdr / fuel1, fuel2, Zone, Type, 'Assigned Value', status, Heatrate2,
             'RE Technology (Yes/No)', 'Hourly Variation? (Yes/No)' /;
             

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
* based on `pGenDataExcel(g,z,tech,f2,'fuel2')` when a condition is met.
gfmap(g,f) = gfmap(g,f) 
         or sum((z,tech,f2), (pGenDataExcel(g,z,tech,f2,'fuel2') = 
         sum(ftfmap(ft,f), ftfindex(ft,f))));
         

         
*gfmap(g,f) =   pGenDataExcel(g,'fuel1')=sum(ftfmap(ft,f),ftfindex(ft,f))
*            or pGenDataExcel(g,'fuel2')=sum(ftfmap(ft,f),ftfindex(ft,f));
*gzmap(g,z) = pGenDataExcel(g,'Zone')=pZoneIndex(z);
*gtechmap(g,tech) = pGenDataExcel(g,'Type')=pTechDataExcel(tech,'Assigned Value');
gstatusmap(g,gstatus) = sum((z,tech,f),pGenDataExcel(g,z,tech,f,'status')=gstatIndex(gstatus));
zcmap(z,c) = zcmapExcel(z,c);

*gprimf(gfmap(g,f)) = pGenDataExcel(g,'fuel1')=sum(ftfmap(ft,f),ftfindex(ft,f)); 
pHeatrate(gfmap(g,f)) = sum((z,tech),pGenDataExcel(g,z,tech,f,"Heatrate2"));
pHeatrate(gprimf(g,f)) = sum((z,tech),pGenDataExcel(g,z,tech,f,"Heatrate"));

pGenData(g,ghdr) = sum((z,tech,f),pGenDataExcel(g,z,tech,f,ghdr));
pTechData(tech,'RE Technology') = pTechDataExcel(tech,'RE Technology (Yes/No)');
pTechData(tech,'Hourly Variation') = pTechDataExcel(tech,'Hourly Variation? (Yes/No)');
pTechData(tech,'Construction Period (years)') = pTechDataExcel(tech,'Construction Period (years)');



***********************H2 model parameters***************************************************

pH2Data(hh,hhdr)=pH2DataExcel(hh,hhdr);
H2statusmap(hh,H2status) = pH2DataExcel(hh,'status')=H2statIndex(H2status);
* Check is that works for H2
* h2zmap(hh,z) = pH2DataExcel(hh,'Zone')=pZoneIndex(z);
h2zmap(hh,z) = pH2DataExcel(hh,'Zone')

$ifThen set generateMIROScenario
Parameter
   pGenDataMIRO(g,z,tech,gstatus,f,f,ghdr)
   pfuelConversion(ft,*)
   pMaxFuellimitMIRO(*,c,f,*,y)
   pFuelPriceMIRO(*,c,f,*,y)
;
Set actscen / BaseCase /;   

loop((gzmap(g,z),gtechmap(g,tech),gstatusmap(g,gstatus),gprimf(g,f)),
  if (sum(gfmap(g,f2),1)=1,
    pGenDataMIRO(g,z,tech,gstatus,f,f,ghdr) = pGenData(g,ghdr);
  else
    pGenDataMIRO(g,z,tech,gstatus,f,f2,ghdr)$(not sameas(f,f2) and gfmap(g,f2)) = pGenData(g,ghdr);
  )
);

pfuelConversion(ft,'mmbtu') = 1;
pMaxFuellimitMIRO('BaseCase',c,f,'mmbtu',y) = pMaxFuelLimit(c,f,y);
pFuelPriceMIRO('BaseCase',c,f,'mmbtu',y) = pFuelPrice(c,f,y);


execute_unload '%GDX_INPUT%_miro'
   zcmap
   ftfmap
   pHours
   pTechData
   pGenDataMIRO
   peak
   Relevant
   pDemandData
   pDemandForecast
   pDemandProfile
   pfuelConversion
   pFuelTypeCarbonContent
   pCarbonPrice
   pMaxFuellimitMIRO
   actscen
   pFuelPriceMIRO
   pTransferLimit
   pLossFactor
   pVREProfile
   pVREgenProfile
   pAvailability
   pStorDataInput
   pCSPData
   pSpinningReserveReqCountry
   pSpinningReserveReqSystem
   pEmissionsCountry
   pEmissionsTotal
   pScalars
   sTopology
   pPlanningReserveMargin
   pEnergyEfficiencyFactor
   pTradePrice
   pMaxExchangeShare
   pExtTransferLimit  
   pNewTransmission
   solverThreads
   solverOptCR
   solverResLim
   nlpsolver
   mipsolver
   mipopt
*   MapGG
*Hydrogen production related parameters
   pH2Data
   pAvailabilityH2
   pCapexTrajectoryH2
   pExternalH2
;

abort.noError 'Created %GDX_INPUT%_miro. Done!';
$endif


$else.mode
$if not set DOEXCELREPORT $set DOEXCELREPORT 0
$include WB_EPM_v8_5_miro
$endIf.mode

*--- Parameter initialisation for same demand profile for all years

$include generate_demand.gms

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

* Read main parameters from pScalars
pzonal_spinning_reserve_constraints  = pScalars("zonal_spinning_reserve_constraints");
psystem_spinning_reserve_constraints = pScalars("system_spinning_reserve_constraints");
psys_reserve_margin                  = pScalars("system_reserve_margin");
pplanning_reserve_constraints        = pScalars("planning_reserve_constraints");
pramp_constraints                    = pScalars("ramp_constraints");
pfuel_constraints                    = pScalars("fuel_constraints");
pcapital_constraints                 = pScalars("capital_constraints");
pmingen_constraints                  = pScalars("mingen_constraints");
pincludeCSP                          = pScalars("includeCSP");
pincludeStorage                      = pScalars("includeStorage");
pMinRE                               = pScalars("MinREshare");
pMinRETargetYr                       = pScalars("RETargetYr");
pzonal_co2_constraints               = pScalars("zonal_co2_constraints");
psystem_co2_constraints              = pScalars("system_co2_constraints");
pAllowExports                        = pScalars("allowExports");
pSurplusPenalty                      = pScalars("costSurplus");
pAllowHighTransfer                   = pScalars("pAllowHighTransfer");
pCostOfCurtailment                   = pScalars("costcurtail");
pCostOfCO2backstop                   = pScalars("CO2backstop");
pMaxImport                           = pScalars("MaxImports");
pMaxExport                           = pScalars("MaxExports");
pVREForecastError                    = pScalars("VREForecastError");
pCaptraj                             = pScalars("Captraj");
pVRECapacityCredits                  = pScalars("VRECapacityCredits");
pSeasonalReporting                   = pScalars("Seasonalreporting");
pSystemResultReporting               = pScalars("Systemresultreporting");
pMaxLoadFractionCCCalc               = pScalars("MaxLoadFractionCCCalc");
*Related to hydrogen model
pIncludeH2                       = pScalars("IncludeH2");
pH2UnservedCost                  = pScalars("H2UnservedCost");  

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

pDR              = pScalars("DR");
pWACC            = pScalars("WACC");
pVOLL            = pScalars("VOLL");
pPlanningReserveVoLL     = pScalars("ReserveVoLL");
pMaxCapital      = pScalars("MaxCapital")*1e6;
pSpinningReserveVoLL = pScalars("SpinReserveVoLL");
pIncludeCarbon   = pScalars("includeCarbonPrice");
pinterconMode    = pScalars("interconMode");
pnoTransferLim   = pScalars("noTransferLim");
pincludeEE       = pScalars("includeEE");
pIncludeDecomCom = pScalars("IncludeDecomCom");

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

* Compute fuel carbon content by aggregating data from fuel type mappings
pFuelCarbonContent(f) = sum(ftfmap(ft,f),pFuelTypeCarbonContent(ft));

* Assign storage data from `pStorDataInput` based on the generator-storage mapping
option gsmap<pStorDataInput;
loop(gsmap(g2,g), pStorData(g,shdr) = pStorDataInput(g,g2,shdr));

* Remove generator pairs (`g,g`) that correspond to standalone storage plants from `gsmap`
gsmap(g,g) = no;

* Write messages to the log file based on parameter values, indicating which data is being ignored
put / ;
if(pnoTransferLim                      = 1, put 'Ignoring transfer limits.'/);
if(pAllowExports                       = 0, put 'Ignoring trade prices data.'/);
if(pScalars("altDemand")               = 1, put 'Ignoring detailed demand and demand forecast data. Using same demand profile for all years instead.'/);
if(pincludeEE                          = 0, put 'Ignoring energy efficiency data.'/);
if(pCaptraj                            = 0, put 'Ignoring capex trajectory data.'/);
if(pfuel_constraints                   = 0, put 'Ignoring fuel limit data.'/);
if(pzonal_co2_constraints              = 0, put 'Ignoring CO2 emissions data.'/);
if(pzonal_spinning_reserve_constraints = 0, put 'Ignoring spinning reserve data.'/);
if(pincludeStorage                     = 0, put 'Ignoring storage data.'/);
if(pincludeCSP                         = 0, put 'Ignoring CSP characteristics data.'/);
if(pIncludeCarbon                      = 0, put 'Ignoring carbon price data.'/);
putclose log;

* Define `Offpeak(t)` as the complement of `peak(t)` (i.e., off-peak hours)
Offpeak(t) = not peak(t);

* Identify candidate generators (`ng(g)`) based on their status in `gstatusmap`
ng(g)  = gstatusmap(g,'candidate');

* Define existing generators (`eg(g)`) as those that are not candidates
eg(g)  = not ng(g);

* Identify variable renewable energy (VRE) generators (`vre(g)`) based on hourly variation data
vre(g) = sum(gtechmap(g,tech)$pTechData(tech,'Hourly Variation'),1);

* Identify renewable energy (RE) generators (`re(g)`) based on RE technology classification
re(g)  = sum(gtechmap(g,tech)$pTechData(tech,'RE Technology'),1);

* Identify concentrated solar power (CSP) technologies
cs(g)  = gtechmap(g,"CSP");

* Identify PV with storage technologies
so(g)  = gtechmap(g,"PVwSTO");

* Identify solar thermal with PV (`STOPV`)
stp(g) = gtechmap(g,"STOPV");

* Identify storage technologies
stg(g) = gtechmap(g,"STORAGE");

* Define a general storage category (`st(g)`) as either `STOPV` or `STORAGE`
st(g)  = gtechmap(g,"STOPV") or gtechmap(g,"STORAGE");

* Compute discounted capital expenditure (`dc(g)`) based on capex trajectory data
dc(g)  = sum(y, pCapexTrajectories(g,y));

* Identify non-discounted capital generators (`ndc(g)`) as those that do not have a capex trajectory
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


$offIDCProtect
* If not running in interconnected mode, set network to 0
sTopology(z,z2)$(not pinterconMode) = no;
* if ignore transfer limit, set limits to high value
pTransferLimit(sTopology,q,y)$pnoTransferLim = inf;
* Default life for transmission lines
pNewTransmission(sTopology,"Life")$(pNewTransmission(sTopology,"Life")=0 and pAllowHighTransfer) = 30; 
$onIDCProtect

* Identify the system peak demand for each year based on the highest total demand across all zones, times, and demand segments
pFindSysPeak(y)     = smax((t,d,q), sum(z, pDemandData(z,q,d,y,t)));

* Identify hours that are close to the peak demand for capacity credit calculations
pAllHours(q,d,y,t)  = 1$(abs(sum(z,pDemandData(z,q,d,y,t))/pFindSysPeak(y) - 1)<pMaxLoadFractionCCCalc);

* Default capacity credit for all generators is set to 1
pCapacityCredit(g,y)= 1;


* Protect against unintended changes while modifying `pVREgenProfile` with `pVREProfile` data
$offIDCProtect
pVREgenProfile(gfmap(VRE,f),q,d,t)$(not(pVREgenProfile(VRE,f,q,d,t))) = sum(gzmap(VRE,z),pVREProfile(z,f,q,d,t));
$onIDCProtect


* Set capacity credit for VRE based on predefined values or calculated generation-weighted availability
pCapacityCredit(VRE,y)$(pVRECapacityCredits =1) =  pGenData(VRE,"CapacityCredit")   ;
pCapacityCredit(VRE,y)$(pVRECapacityCredits =0) =  Sum((z,q,d,t)$gzmap(VRE,z),Sum(f$gfmap(VRE,f),pVREgenProfile(VRE,f,q,d,t)) * pAllHours(q,d,y,t)) * (Sum((z,f,q,d,t)$(gfmap(VRE,f) and gzmap(VRE,z) ),pVREgenProfile(VRE,f,q,d,t))/sum((q,d,t),1));

* Compute capacity credit for run-of-river hydro as an availability-weighted average
pCapacityCredit(ROR,y) =  sum(q,pAvailability(ROR,q)*sum((d,t),pHours(q,d,t)))/sum((q,d,t),pHours(q,d,t));

* TODO: REMOVE
*pCapacityCredit(RE,y) =  Sum((z,q,d,t)$gzmap(RE,z),Sum(f$gfmap(RE,f),pREProfile(z,f,q,d,t)) * pAllHours(q,d,y,t)) * (Sum((z,f,q,d,t)$(gfmap(RE,f) and gzmap(RE,z) ),pREProfile(z,f,q,d,t))/sum((q,d,t),1));

* Compute CSP and PV with storage generation profiles
pCSPProfile(cs,q,d,t)    = sum((z,f)$(gfmap(cs,f) and gzmap(cs,z)), pVREProfile(z,f,q,d,t));
pStoPVProfile(so,q,d,t)  =  sum((z,f)$(gfmap(so,f) and gzmap(so,z)), pVREProfile(z,f,q,d,t));


* H2 model parameters
pCapexTrajectoriesH2(hh,y) =1;
pCapexTrajectoriesH2(dch2,y)$pCaptraj = pCapexTrajectoryH2(dcH2,y);


* Set the weight of the start year to 1.0
pWeightYear(sStartYear) = 1.0;

* Compute weight for each year as the difference from the previous year's cumulative weight
pWeightYear(y)$(not sStartYear(y)) = y.val - sum(sameas(y2+1,y), y2.val) ;
*pWeightYear(sFinalYear) = sFinalYear.val - sum(sameas(y2+1,sFinalYear), y2.val)  ;


** (mid-year discounting)
*pRR(y) = 1/[(1+pDR)**(sum(y2$(ord(y2)<ord(y)),pWeightYear(y2)))];

* Compute the present value discounting factor considering mid-year adjustments
pRR(y) = 1.0;
pRR(y)$(ord(y)>1) = 1/((1+pDR)**(sum(y2$(ord(y2)<ord(y)),pWeightYear(y2))-1 + sum(sameas(y2,y), pWeightYear(y2)/2))) ;        
                                    

*-------------------------------------------------------------------
* Parameter Processing
*-------------------------------------------------------------------
$offIDCProtect
pLossFactor(z2,z,y)$(pLossFactor(z,z2,y) and not pLossFactor(z2,z,y)) = pLossFactor(z,z2,y);


pEnergyEfficiencyFactor(z,y)$(not pincludeEE) = 1;
pEnergyEfficiencyFactor(z,y)$(pEnergyEfficiencyFactor(z,y)=0) = 1;
$onIDCProtect
pVarCost(gfmap(g,f),y) = pGenData(g,"VOM")
                       + sum((gzmap(g,z),zcmap(z,c)),pFuelPrice(c,f,y)*pHeatRate(g,f) )
                       + pStorData(g, "VOM")
                       + pCSPData(g, "Storage", "VOM")
                       + pCSPData(g, "Thermal Field", "VOM");


pVarCostH2(hh,y) = pH2Data(hh,"VOM");
pCRF(g)$pGenData(g,'Life') = pWACC / (1 - (1 / ( (1 + pWACC)**pGenData(g,'Life'))));
pCRFH2(hh)$pH2Data(hh,'Life') = pWACC / (1 - (1 / ( (1 + pWACC)**pH2Data(hh,'Life'))));
pCRFsst(st)$pStorData(st,'Life') = pWACC / (1 - (1 / ( (1 + pWACC)**pStorData(st,'Life'))));
pCRFcst(cs)$pCSPData(cs,"Storage","Life") = pWACC / (1 - (1 / ( (1 + pWACC)**pCSPData(cs,"Storage","Life"))));
pCRFcth(cs)$pCSPData(cs,"Thermal Field","Life") = pWACC / (1 - (1 / ( (1 + pWACC)**pCSPData(cs,"Thermal Field","Life"))));

**Create set MapNCZ (neighbouring country zones)

*Zones_conn_in_Diff_country(z,zz)$((sum(c$gfmap(c,z),x(c,z)) ne sum(c$gfmap(c,zz),x(c,zz))) and conn(z,zz)) = 1;

sMapNCZ(sTopology(z,z2)) = sum(c$(zcmap(z,c) and zcmap(z2,c)), 1) = 0;

*** Simple bounds
*vImportPrice.up(z,q,d,t,y)$(pMaxImport>1) = pMaxImport;
*vExportPrice.up(z,q,d,t,y)$(pMaxExport>1) = pMaxExport;
vCap.up(g,y) = pGenData(g,"Capacity");
vBuild.fx(eg,y)$(pGenData(eg,"StYr") <= sStartYear.val) = 0;
vBuild.up(ng,y) = pGenData(ng,"BuildLimitperYear")*pWeightYear(y);
vAdditionalTransfer.up(sTopology(z,z2),y)$pAllowHighTransfer = symmax(pNewTransmission,z,z2,"MaximumNumOfLines");
vBuildStor.fx(eg,y)$(pGenData(eg,"StYr") <= sStartYear.val and pincludeStorage) = 0;
vBuildTherm.fx(eg,y)$(pGenData(eg,"StYr") <= sStartYear.val and pincludeCSP) = 0;
*vTotalEmissions.up(y)$psystem_CO2_constraints = pEmissionsTotal(y);

*-------------------------------------------------------------------
* Fixed conditions
*-------------------------------------------------------------------

* First year - all existing capacity
vCap.fx(g,y)$(pGenData(g,"StYr") > y.val)=0;
vCap.fx(eg,sStartYear)$(pGenData(eg,"StYr") < sStartYear.val) = pGenData(eg,"Capacity");
vCap.fx(eg,y)$((pGenData(eg,"StYr") <= y.val) and (pGenData(eg,"StYr") >= sStartYear.val)) = pGenData(eg,"Capacity");
vCap.fx(eg,y)$(pGenData(eg,"RetrYr") and (pGenData(eg,"RetrYr") <= y.val)) = 0;

vCapTherm.fx(eg,sStartYear)$(pGenData(eg,"StYr") < sStartYear.val) = pCSPData(eg,"Thermal Field","Capacity");
vCapStor.fx(eg,sStartYear)$(pGenData(eg,"StYr") < sStartYear.val) = pCSPData(eg,"Storage","Capacity")+pStorData(eg, "Capacity");

***This equation is needed to avoid decommissioning of hours of storage from existing storage
vCapStor.fx(eg,y)$((pScalars("econRetire") = 0 and pGenData(eg,"StYr") < y.val) and (pGenData(eg,"RetrYr") >= y.val)) = pStorData(eg,"Capacity");

vRetire.fx(ng,y) = 0;
vRetire.fx(eg,y)$(pGenData(eg,"Life") = 99) = 0;

*vRetire.fx(eg,y)$((pGenData(eg,"StYr") >= sStartYear.val) and (pGenData(eg,"RetrYr") > y.val)) = 0;

vCap.fx(eg,y)$((pScalars("econRetire") = 0 and pGenData(eg,"StYr") < y.val) and (pGenData(eg,"RetrYr") >= y.val)) = pGenData(eg,"Capacity");
vCapTherm.fx(ng,y)$(pGenData(ng,"StYr") > y.val) = 0;
vCapStor.fx(ng,y)$(pGenData(ng,"StYr") > y.val) = 0;
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

vCapH2.fx(eh,y)$((pScalars("econRetire") = 0 and pH2Data(eh,"StYr") < y.val) and (pH2Data(eh,"RetrYr") >= y.val)) = pH2Data(eh,"Capacity");

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


sAdditionalTransfer(z,z2,y) = yes;
sAdditionalTransfer(z,z2,y) $(y.val < pNewTransmission(z,z2,"EarliestEntry")) = no;

sFlow(z,z2,q,d,t,y) = yes;
sFlow(z,z2,q,d,t,y)$(not sTopology(z,z2)) = no;

sSpinningReserve(g,q,d,t,y) = yes;
sSpinningReserve(g,q,d,t,y)$(not (pzonal_spinning_reserve_constraints or psystem_spinning_reserve_constraints) ) = no;

*To avoid bugs when there is no candidate transmission expansion line
$offIDCProtect
pNewTransmission(z,z2,"EarliestEntry")$(not pAllowHighTransfer) = 2500;
$onIDCProtect

*******************************************************************************************************************

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
* Save model state at the end of execution (useful for debugging or re-running from a checkpoint)
option savepoint=1;

* Solve the MIP problem `PA`, minimizing the variable `vNPVcost`
Solve PA using MIP minimizing vNPVcost;

* Include the external report file specified by `%REPORT_FILE%`
$include %REPORT_FILE%

*-------------------------------------------------------------------------------------
* Check output

* $include output_verification.gms

*-------------------------------------------------------------------------------------



* If memory monitoring is enabled, execute embedded Python code to log memory usage details
$ifThen %gams.ProcTreeMemMonitor%==1
embeddedCode Python:
gams.printLog('')
gams.printLog('Domains:')
for s in [ 'g', 'f', 'y', 'q', 'd', 't', 'z', 'c']: # domains
  gams.printLog(f'{s.ljust(20)} {str(len(gams.db[s])).rjust(10)}')
gams.printLog('')
gams.printLog('Maps:')
for s in gams.db:
  if isinstance(s,GamsSet) and s.name.lower().find('map') >= 0 and len(s)>0:
    gams.printLog(f'{s.name.ljust(20)} {str(len(s)).rjust(10)}')
gs = []
for s in gams.db:
  if not isinstance(s,GamsSet) and len(s)>10000:
    gs.append((type(s),s.name,len(s)))
gs.sort(key=lambda x: x[2], reverse=True)
for t in zip([GamsParameter,GamsVariable,GamsEquation],['Parameter','Variable','Equation']):
  gams.printLog('')
  gams.printLog(f'{t[1]}:')
  for s in gs:
    if not s[0] == t[0]:
      continue
    gams.printLog(f'{s[1].ljust(20)} {str(s[2]).rjust(10)}')
endEmbeddedCode
$endif
