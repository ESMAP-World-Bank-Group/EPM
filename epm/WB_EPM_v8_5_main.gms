* Adjusted for SAPP model:
* (1) Min utilization of fossil fuel geneartors
* (2) Early retirement of young fossil fuel plants will not occur before 2030
*a=c xs=engine

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

* Only include base if we don't restart
$ifThen not set BASE_FILE
$set BASE_FILE "WB_EPM_v8_5_base.gms"
$endIf
$log BASE_FILE is "%BASE_FILE%"

$if "x%gams.restart%" == "x" $include %BASE_FILE%

$ifThen not set REPORT_FILE
$set REPORT_FILE "WB_EPM_v8_5_Report.gms"
$endIf
$log REPORT_FILE is "%REPORT_FILE%"


$ifThen not set READER_FILE
$set READER_FILE "WB_EPM_input_readers.gms"
$endIf

$call 'rm -f miro.log'
file log / miro.log /;
put log '------------------------------------'/;
put log '        Data validation'/;
put log '------------------------------------'/;

Set
   tech     'technologies'
   gstatus  'generator status' / Existing, Candidate, Committed /
   techhdr  'techdata headers' / 'Construction Period (years)', 'RE Technology', 'Hourly Variation' /
   pe       'peak energy for demand forecast' /'peak', 'energy'/
   ft       'fuel types'
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
						
                        pGasRetirement
						MinStorageHrs
						zonal_planning_Startyr
						zonal_planning_reserve_constraints
						
						RunMonteCarlo
						testHydrology
						AllowAllConnectionsLater
						EmissionBudget
						pDecarbLevel
                     
                       /
   mipline 'Solver option lines'
;

alias (y,y2);
alias (f,f2);
alias (c,c2);

$ifi %mode%==MIRO
$onExternalInput
Set
   ftfmap(ft<,f<)                   'map fuel types to fuels'
   zcmap(z<,c<)                     'map zones to countries'
   sTopology(z,z2)                  'network topology - to be assigned through network data'
   Peak(t)                          'peak period hours'
   Relevant(d)                      'relevant day and hours when MinGen limit is applied'
   mipopt(mipline<)                 / system.empty /;
;   
Parameter
   pCSPData(g,csphrd,shdr)
   pCapexTrajectory(tech,y)         'Capex trajectory'
   pTechData(tech<,techhdr)         'Technology data'
   pFuelTypeCarbonContent(ft)       'Fuel type carbon content in tCO2 per MMBTu'
   pStorDataInput(g,g2,shdr)        'Storage data'
   pNewTransmission(z,z2,thdr)      'new transmission lines'
   pTradePrice(zext,q,d,y,t)           'trade price - export or import driven by prices [assuming each zone in a country can only trade with one external zone]'
   pDemandProfile(z,q,d,t)          'Demand profile in per unit'
   pDemandForecast(z,pe,y)          'Peak and Energy demand forecast in MW and GWh'
   pDemandData(z,q,d,y,t)           'hourly load curves by quarter(seasonal) and year'
   pEmissionsCountry(c,y)              'Maximum zonal emissions allowed per country and year in tns'
   pEmissionsTotal(y)               'Maximum total emissions allowed per year for the region in tns'
   pCarbonPrice(y)                  'Carbon price in USD per ton of CO2eq'
$ifi     %mode%==MIRO   pHours(q<,d<,y<,t<) 'duration of each block'
$ifi not %mode%==MIRO   pHours(q<,d<,y ,t<) 'duration of each block'
   pTransferLimit(z,z2,q,y)         'Transfer limits by quarter (seasonal) and year between zones'
   pMinImport(z2,z,y)               'Minimum trade constraint between zones'

   pLossFactor(z,z2,y)              'loss factor in percentage'
   pVREProfile(z,*,q,d,t)           'VRE generation profile by site quarter day type and YEAR -- normalized (per MW of solar and wind capacity)'
   pVREgenProfile(g,f,q,d,t)        'VRE generation profile by plant quarter day type and YEAR -- normalized (per MW of solar and wind capacity)'
   pAvailability(g,q)               'Availability by generation type and season or quarter in percentage - need to reflect maintenance'
   pSpinningReserveReqCountry(c,y)     'Spinning reserve requirement local at country level (MW)  -- for isolated system operation scenarios'
   pSpinningReserveReqSystem(y)     'Spinning reserve requirement systemwide (MW) -- for integrated system operation scenarios'
   pScalars(sc)                     'Flags and penalties to load'
   pPlanningReserveMargin(c,y)        'Country planning reserve margin'
   pEnergyEfficiencyFactor(z,y)     'Scaling factor for energy efficiency measures'
   
   pHydroVar(g,y)                   'Total hydro capacity factor per generator and year'
    

;   
Scalar
   solverThreads 'Number of threads available to the solvers' /1/
   solverOptCR   'Relative gap for MIP Solver'                /0.001/
   solverResLim  'Solver time limit'                          /300000/
;   
Singleton Set
   NLPSolver 'Selected NLP Solver' / '' 'conopt4' /
   MIPSolver 'Selected MIP Solver' / '' 'cplex' /
;
$ifI %mode%==MIRO
$offExternalInput

Parameters
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
   pMaxHrImportShare
   pVRECapacityCredits              'User input capacity credits'
   pDR                              'discount rate'
   pCaptraj                         'allow capex trajectory'
   pIncludeEE                       'energy efficiency factor flag'
   pSystem_CO2_constraints
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

;
$onmulti
Sets
   ghdr         'Additional headers for pGenData' / CapacityCredit, Heatrate, Heatrate2, Life, VOM, MinUtilization, UnitSize /
   shdr         'Additional headers for pStorData' / Life, VOM /
   thdr         'Additional header for pNewTransmission' / EarliestEntry, MaximumNumOfLines /
;
$offmulti
    
$eval     SOLVERTHREADS solverThreads
$eval     SOLVEROPTCR   solverOptCR
$eval     SOLVERRESLIM  solverResLim
$eval.set NLPSOLVER     NLPSolver.te
$eval.set MIPSOLVER     MIPSolver.te
Option NLP=%NLPSOLVER%, MIP=%MIPSOLVER%, threads=%SOLVERTHREADS%, optCR=%SOLVEROPTCR%, resLim=%SOLVERRESLIM%;

$ifThenI.mode %mode%==Excel
$set main Excel
$set DOEXCELREPORT 1
$ifThen not set XLS_INPUT
$  if     set GDX_INPUT $set XLS_INPUT "%GDX_INPUT%.%ext%"
$  if not set GDX_INPUT $set XLS_INPUT data%system.dirsep%input%system.dirsep%WB_EPM_SAPP.xlsb
$endIf
$setNames "%XLS_INPUT%" fp GDX_INPUT fe
$if not set XLS_OUTPUT $set XLS_OUTPUT %fp%EPMRESULTS.xlsx

Set
   zcmapExcel(z,c<);
Parameter
   pGenDataExcel(g<,*)
   pStorDataExcel(g,*,shdr)
$onMulti   
   pTechDataExcel(tech<,*)
;

$include %READER_FILE%

Parameter
   ftfindex(ft<,f<)
   pZoneIndex(z<);

$gdxIn %GDX_INPUT%

* Domain defining symbols
$load pZoneIndex zcmapExcel ftfindex y pHours pTechDataExcel pGenDataExcel
$load zext
* Other symbols
$load peak Relevant pDemandData pDemandForecast pDemandProfile
$load pFuelTypeCarbonContent pCarbonPrice pEmissionsCountry pEmissionsTotal pFuelPrice
$load pMaxFuellimit pTransferLimit pLossFactor pVREProfile pVREgenProfile pAvailability pHydroVar
$load pStorDataExcel pCSPData pCapexTrajectory pSpinningReserveReqCountry pSpinningReserveReqSystem pScalars
$load sTopology pPlanningReserveMargin pEnergyEfficiencyFactor pTradePrice pMaxPriceImportShare 
$load pNewTransmission
$load pMinImport
$gdxIn
$offmulti

$exit

option ftfmap<ftfindex;
pStorDataInput(g,g2,shdr) = pStorDataExcel(g,g2,shdr);
pStorDataInput(g,g,shdr)$pStorDataExcel(g,'',shdr) = pStorDataExcel(g,'',shdr);

$if not errorFree $echo Data errors. Please inspect the listing file for details. > miro.log

* Generate gfmap and others from pGenDataExcel
parameter gstatIndex(gstatus) / Existing 1, Candidate 3, Committed 2 /;
set addHdr / fuel1, fuel2, Zone, Type, 'Assigned Value', status, Heatrate2,
             'RE Technology (Yes/No)', 'Hourly Variation? (Yes/No)' /;

gfmap(g,f) =   pGenDataExcel(g,'fuel1')=sum(ftfmap(ft,f),ftfindex(ft,f))
            or pGenDataExcel(g,'fuel2')=sum(ftfmap(ft,f),ftfindex(ft,f));
            


gzmap(g,z) = pGenDataExcel(g,'Zone')=pZoneIndex(z);
gtechmap(g,tech) = pGenDataExcel(g,'Type')=pTechDataExcel(tech,'Assigned Value');
gstatusmap(g,gstatus) = pGenDataExcel(g,'status')=gstatIndex(gstatus);
zcmap(z,c) = zcmapExcel(z,c);

gprimf(gfmap(g,f)) = pGenDataExcel(g,'fuel1')=sum(ftfmap(ft,f),ftfindex(ft,f)); 
pHeatrate(gfmap(g,f)) = pGenDataExcel(g,"Heatrate2");
pHeatrate(gprimf(g,f)) = pGenDataExcel(g,"Heatrate");

pGenData(g,ghdr) = pGenDataExcel(g,ghdr);
pTechData(tech,'RE Technology') = pTechDataExcel(tech,'RE Technology (Yes/No)');
pTechData(tech,'Hourly Variation') = pTechDataExcel(tech,'Hourly Variation? (Yes/No)');
pTechData(tech,'Construction Period (years)') = pTechDataExcel(tech,'Construction Period (years)');

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
   pCapexTrajectory
   pSpinningReserveReqCountry
   pSpinningReserveReqSystem
   pEmissionsCountry
   pEmissionsTotal
   pScalars
   sTopology
   pPlanningReserveMargin
   pEnergyEfficiencyFactor
   pTradePrice
   pMaxPriceImportShare
   pNewTransmission
   solverThreads
   solverOptCR
   solverResLim
   nlpsolver
   mipsolver
   mipopt
   pHydroVar
;
abort.noError 'Created %GDX_INPUT%_miro. Done!';
$endif

set cfmap(c,f); 
parameter Error01(c,f);
option cfmap<pFuelPrice, Error01<cfmap;
Error01(c,f)$(Error01(c,f)=1) = 0;
log.ap = 1;
put log
put 'Validating fuel mappings [Fuel Prices] ...';
if(card(Error01),
  put / 'pFuelPrice:: More than one fuel map per country!'/;
  put / 'SYMBOL' @25 'COUNTRY' @42 'FUEL';
  loop((c,f)$(Error01(c,f) > 0),
      put / 'pFuelPrice:: ' @25 c.tl @42 f.tl
    );
  abort "More than one fuel map per country [Fuel Prices]", Error01;
else
  put ' OK'/;
);

option cfmap<pMaxFuellimit, Error01<cfmap;
Error01(c,f)$(Error01(c,f)=1) = 0;
put 'Validating fuel mappings [Fuel Limits] ...';
if(card(Error01),
  put / 'pMaxFuellimit:: More than one fuel map per country!'/;
  put / 'SYMBOL' @25 'COUNTRY' @42 'FUEL';
  loop((c,f)$(Error01(c,f) > 0),
      put / 'pMaxFuellimit:: ' @25 c.tl @42 f.tl;
    );
  abort 'More than one fuel map per country [Fuel Limits]', Error01;
else
  put ' OK'/;
);
$else.mode
$if not set DOEXCELREPORT $set DOEXCELREPORT 0
$include WB_EPM_v8_5_miro
$endIf.mode

Set Error03(z,q,d,y,t);
option Error03<pDemandData;
Error03(z,q,d,y,t)$(pDemandData(z,q,d,y,t) >= 0) = 0;
if (pScalars("altDemand") <> 1,
  put 'Validating demand data ...';
  if(card(Error03),
    put / 'pDemandData: No negative demand data allowed!'/;
    put 'Note that the setting  "Use same demand profile for all years" has been set to true. '/;
    put 'Therefore, the demand data was calculated in the model and is not provided by the user. '/;
    put / 'SYMBOL' @25 'ZONE' @45 'QUARTER/SEASON' @65 'DAY TYPE' @85 'YEAR' @105 'HOUR OF DAY' @125 'VALUE';
    loop(Error03(z,q,d,y,t),
        put / 'pDemandData:: '@25 z.tl @45 q.tl @65 d.tl @85 y.tl @105 t.tl @125 pDemandData(z,q,d,y,t);
      );
    abort 'No negative demand data allowed', Error03;
  else
    put ' OK'/;
  );
);

Set Error04(q,d,y,t);
option Error04<pHours;
Error04(q,d,y,t)$(pHours(q,d,y,t) >= 0) = 0;
put 'Validating duration of each block data ...';
if(card(Error04),
  put / 'pHours:: No negative duration of each block data allowed!'/;
  put / 'SYMBOL' @25 'QUARTER/SEASON' @45 'DAY TYPE' @65 'YEAR' @85 'HOUR OF DAY' @105 'VALUE';
  loop(Error04(q,d,y,t),
      put / 'pHours:: '@25 q.tl @45 d.tl @65 y.tl @85 t.tl @105 pHours(q,d,y,t);
    );
  abort 'No negative duration of each block data allowed', Error04;
else
  put ' OK'/;
);

Set Error05(z,*,q,d,t);
option Error05<pVREProfile;
Error05(z,f,q,d,t)$(pVREProfile(z,f,q,d,t) < 1.01) = 0;
put 'Validating VRE generation profile by site data ...';
if(card(Error05),
  put / 'pVREProfile:: Capacity factor cannot be more than 1!'/;
  put / 'SYMBOL' @25 'ZONE' @45 'FUEL' @65 'QUARTER/SEASON' @85 'DAY TYPE' @105 'HOUR OF DAY' @125 'VALUE';
  loop(Error05(z,f,q,d,t),
      put / 'pVREProfile:: '@25 z.tl @45 f.tl @65 q.tl @85 d.tl @105 t.tl @125 pVREProfile(z,f,q,d,t);
    );
  abort 'Capacity factor cannot be greater than 1', Error05;
else
  put ' OK'/;
);

*Set Error06(g,q);
*option Error06<pAvailability;
*Error06(g,q)$(pAvailability(g,q) < 1.01) = 0;
*put 'Validating generator availability data ...';
*if(card(Error06),
*  put / 'pAvailability:: Generator availability factor cannot be more than 1!'/;
*  put / 'SYMBOL' @25 'GENERATOR TYPE' @45 'QUARTER/SEASON' @65 'VALUE';
*  loop(Error06(g,q),
*      put / 'pAvailability:: '@25 g.tl @45 q.tl @65 pAvailability(g,q);
*    );
*  abort 'Generator availability factor cannot be greater than 1', Error06;
*else
*  put ' OK'/;
*);

*--- Adjust peak and energy in case user selected same demand profile for all years

Parameters
   pmax(z,y)
   pmin(z,y)
   pdiff(z,y)
   ptotalenergy(z,y)
   ptemp2(y)
   ptemp(y)
   pTempDemand(z,q,d,y,t)
;

variable
   pyval(z,q,d,y,t)
   divisor(z,y)
   obj
;

divisor.lo(z,y) = card(q)*card(d)*card(t);

equation
   getDivisor(z,q,d,y,t)
   getArea(z,y)
   objFn
;

getDivisor(z,q,d,y,t)..  pyval(z,q,d,y,t) =e=  [2*pdiff(z,y)/divisor(z,y) * (pmax(z,y) - pDemandProfile(z,q,d,t))]/sqr(pmax(z,y) - pmin(z,y));
getArea(z,y)..           sum((q,d,t), pyval(z,q,d,y,t) * pHours(q,d,y,t)) =e=  pdiff(z,y);
objFn..                  obj =e= sum((z,q,d,y,t), pyval(z,q,d,y,t));

model demand / getDivisor, getArea, objFn /;

* If using alt demand:
if (pScalars("altDemand") = 1,
   pTempDemand(z,q,d,y,t) = pDemandProfile(z,q,d,t) * pDemandForecast(z,"Peak",y);

   pdiff(z,y) = ((pDemandForecast(z,"Energy",y)*1e3) - sum((q,d,t), pTempDemand(z,q,d,y,t)*pHours(q,d,y,t) )) ;

   pmax(z,y) = smax((q,d,t), pDemandProfile(z,q,d,t));
   pmin(z,y) = smin((q,d,t)$pDemandProfile(z,q,d,t), pDemandProfile(z,q,d,t));
   Solve demand using nlp min obj;

   abort$(demand.modelstat<>%modelstat.Optimal% and demand.modelstat<>%modelstat.locallyOptimal%) 'Demand model not solved successfully';
$offIDCProtect
   pDemandData(z,q,d,y,t) =  pTempDemand(z,q,d,y,t) + pyval.l(z,q,d,y,t);
$onIDCProtect
   ptemp(y) = sum((z,q,d,t), pdemanddata(z,q,d,y,t)*phours(q,d,y,t))/1000;
   ptemp2(y) = smax((q,d,t),sum(z, pdemanddata(z,q,d,y,t)));
);

ptotalenergy(z,y) =  sum((q,d,t), pdemanddata(z,q,d,y,t)*phours(q,d,y,t))/1e3;
*$include demand_adjustment.gms										 

*--- end of parameter initialisation for same demand profile for all years

*--- Start of initialisation of other parameters
***
*** Part2
***
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

*Main parameters
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
pMaxHrImportShare                    = pScalars("MaxImports");
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

pGasRetirement                       = pScalars("pGasRetirement");
pMinStorageHrs                       = pScalars("MinStorageHrs");
pzonal_planning_Startyr              = pScalars("zonal_planning_Startyr");
pzonal_planning_reserve_constraints  = pScalars("zonal_planning_reserve_constraints");
pEmissionBudget                      = pScalars("EmissionBudget");


$IF NOT "%zonal_spinning_reserve_constraints%"  == "-1" pzonal_spinning_reserve_constraints  = %zonal_spinning_reserve_constraints%;
$IF NOT "%system_spinning_reserve_constraints%" == "-1" psystem_spinning_reserve_constraints = %system_spinning_reserve_constraints%;
$IF NOT "%planning_reserve_constraints%"        == "-1" pplanning_reserve_constraints        = %planning_reserve_constraints%;
$IF NOT "%ramp_constraints%"                    == "-1" pramp_constraints                    = %ramp_constraints%;
$IF NOT "%fuel_constraints%"                    == "-1" pfuel_constraints                    = %fuel_constraints%;
$IF NOT "%capital_constraints%"                 == "-1" pcapital_constraints                 = %capital_constraints%;
$IF NOT "%mingen_constraints%"                  == "-1" pmingen_constraints                  = %mingen_constraints%;
$IF NOT "%includeCSP%"                          == "-1" pincludeCSP                          = %includeCSP%;
$IF NOT "%includeStorage%"                      == "-1" pincludeStorage                      = %includeStorage%;
$IF NOT "%zonal_co2_constraints%"               == "-1" pzonal_co2_constraints               = %zonal_co2_constraints%;
$IF NOT "%system_co2_constraints%"              == "-1" psystem_co2_constraints              = %system_co2_constraints%;

singleton set sFinalYear(y);
scalar TimeHorizon;

sStartYear(y) = y.first;
sFinalYear(y) = y.last;
TimeHorizon = sFinalYear.val - sStartYear.val + 1;

sFirstHour(t) = t.first;
sLastHour(t) = t.last;

sFirstDay(d) = d.first;
sLastDay(d)  = d.last;

sFirstQuarter(q)=q.first;					  

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

Zt(z) = sum((q,d,y,t),pDemandData(z,q,d,y,t)) = 0;
Zd(z) = not Zt(z);

pFuelCarbonContent(f) = sum(ftfmap(ft,f),pFuelTypeCarbonContent(ft));

Option gsmap<pStorDataInput;
loop(gsmap(g2,g), pStorData(g,shdr) = pStorDataInput(g,g2,shdr));
* Eliminate the g,g pairs which correspond to standalone storage plants from gsmap
gsmap(g,g) = no;

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
if(pMinStorageHrs = 0,  put 'Ignoring minimum storage hours requirement.'/;)
putclose log;

Offpeak(t) = not peak(t);
ng(g)  = gstatusmap(g,'candidate');
eg(g)  = not ng(g);
vre(g) = sum(gtechmap(g,tech)$pTechData(tech,'Hourly Variation'),1);
re(g)  = sum(gtechmap(g,tech)$pTechData(tech,'RE Technology'),1);
cs(g)  = gtechmap(g,"CSP");
so(g)  = gtechmap(g,"PVwSTO");
stp(g) = gtechmap(g,"STOPV");
stg(g) = gtechmap(g,"STORAGE");
st(g)  = gtechmap(g,"STOPV") or gtechmap(g,"STORAGE");
dc(g)  = sum(y, sum(tech$(gtechmap(g,tech)), pCapexTrajectory(tech,y)));
ndc(g) = not dc(g);
ror(g) = gtechmap(g,"ROR");
VRE_noROR(g) = vre(g) and not ror(g);

RampRate(g) = pGenData(g,"RampDnRate");

zfmap(z,f) = sum((gzmap(g,z),gfmap(g,f)), 1);

$offIDCProtect
* If not running in interconnected mode, set network to 0
sTopology(z,z2)$(not pinterconMode) = no;
* if ignore transfer limit, set limits to high value
pTransferLimit(sTopology,q,y)$pnoTransferLim = inf;
* Default life for transmission lines
pNewTransmission(sTopology,"Life")$(pNewTransmission(sTopology,"Life")=0 and pAllowHighTransfer) = 30; 

$onIDCProtect
* VRE does not fully contribute to reserve margin
pFindSysPeak(y)     = smax((t,d,q), sum(z, pDemandData(z,q,d,y,t)));
pAllHours(q,d,y,t)  = 1$(abs(sum(z,pDemandData(z,q,d,y,t))/pFindSysPeak(y) - 1)<1e-9);
pCapacityCredit(g,y)= 1;

$if %DEBUG%==1 display pVREgenProfile;
$offIDCProtect
pVREgenProfile(gfmap(VRE,f),q,d,t)$(not(pVREgenProfile(VRE,f,q,d,t))) = sum(gzmap(VRE,z),pVREProfile(z,f,q,d,t));
$onIDCProtect
pCapacityCredit(VRE,y)$(pVRECapacityCredits =1) =  pGenData(VRE,"CapacityCredit")   ;
pCapacityCredit(VRE,y)$(pVRECapacityCredits =0) =  Sum((z,q,d,t)$gzmap(VRE,z),Sum(f$gfmap(VRE,f),pVREgenProfile(VRE,f,q,d,t)) * pAllHours(q,d,y,t)) * (Sum((z,f,q,d,t)$(gfmap(VRE,f) and gzmap(VRE,z) ),pVREgenProfile(VRE,f,q,d,t))/sum((q,d,t),1));
pCapacityCredit(ROR,y) =  sum(q,pAvailability(ROR,q)*sum((d,t),pHours(q,d,y,t)))/sum((q,d,t),pHours(q,d,y,t))   ;           
*pCapacityCredit(RE,y) =  Sum((z,q,d,t)$gzmap(RE,z),Sum(f$gfmap(RE,f),pREProfile(z,f,q,d,t)) * pAllHours(q,d,y,t)) * (Sum((z,f,q,d,t)$(gfmap(RE,f) and gzmap(RE,z) ),pREProfile(z,f,q,d,t))/sum((q,d,t),1));


pCSPProfile(cs,q,d,t)    = sum((z,f)$(gfmap(cs,f) and gzmap(cs,z)), pVREProfile(z,f,q,d,t));
pStoPVProfile(so,q,d,t)  =  sum((z,f)$(gfmap(so,f) and gzmap(so,z)), pVREProfile(z,f,q,d,t));
$if %DEBUG%==1 display pVREgenProfile;
pCapexTrajectories(g,y) =  1;
* execute_unload 'debug_output.gdx', g, gtechmap, pCapexTrajectory;
pCapexTrajectories(dc, y)$pCaptraj = sum(tech$(pCapexTrajectory(tech, y) > 0 and gtechmap(dc, tech)), pCapexTrajectory(tech, y));


** (correcting the weights)
pWeightYear(sStartYear) = 1.0;
pWeightYear(y)$(not sStartYear(y)) = y.val - sum(sameas(y2+1,y), y2.val) ;
*pWeightYear(sFinalYear) = sFinalYear.val - sum(sameas(y2+1,sFinalYear), y2.val)  ;


** (mid-year discounting)
*pRR(y) = 1/[(1+pDR)**(sum(y2$(ord(y2)<ord(y)),pWeightYear(y2)))];

pRR(y) = 1.0;
pRR(y)$(ord(y)>1) = 1/((1+pDR)**(sum(y2$(ord(y2)<ord(y)),pWeightYear(y2))-1 + sum(sameas(y2,y), pWeightYear(y2)/2))) ;        
                                    
**

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


pCRF(g)$pGenData(g,'Life') = pWACC / (1 - (1 / ( (1 + pWACC)**pGenData(g,'Life'))));

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
*vCap.fx(eg,y)$((pGenData(eg,"StYr") <= y.val) and (pGenData(eg,"StYr") >= sStartYear.val)) = pGenData(eg,"Capacity");
vCap.fx(eg,y)$((pGenData(eg,"StYr") = y.val) and (pGenData(eg,"StYr") >= sStartYear.val)) = pGenData(eg,"Capacity");   //committed
vCap.up(eg,y)$((pGenData(eg,"StYr") < y.val) and (pGenData(eg,"StYr") >= sStartYear.val)) = pGenData(eg,"Capacity");   //committed
vCap.fx(eg,y)$((pGenData(eg,"StYr") < y.val) and (pGenData(eg,"StYr") >= sStartYear.val) and (gtechmap(eg,"STO HY") or gtechmap(eg,"WIND") or gtechmap(eg,"PV") or gtechmap(eg,"ROR") or gtechmap(eg,"CSP") or gtechmap(eg,"PVwSTO") or gtechmap(eg,"STOPV") or gtechmap(eg,"STORAGE"))) = pGenData(eg,"Capacity");
vCap.fx(eg,y)$(pGenData(eg,"RetrYr") and (pGenData(eg,"RetrYr") <= y.val)) = 0;

vCapTherm.fx(eg,sStartYear)$(pGenData(eg,"StYr") < sStartYear.val) = pCSPData(eg,"Thermal Field","Capacity");

vCapStor.fx(eg,sStartYear)$(pGenData(eg,"StYr") < sStartYear.val) = pCSPData(eg,"Storage","Capacity")+pStorData(eg, "Capacity");

***This equation is needed to avoid decommissioning of hours of storage from existing storage
vCapStor.fx(eg,y)$((pScalars("econRetire") = 0 and pGenData(eg,"StYr") < y.val) and (pGenData(eg,"RetrYr") >= y.val)) = pStorData(eg,"Capacity");

vRetire.fx(ng,y) = 0;
vRetire.fx(eg,y)$(pGenData(eg,"Life") = 99) = 0;
vRetire.fx(g,y)$((gtechmap(g,"STO HY") or gtechmap(g,"WIND") or gtechmap(g,"PV") or gtechmap(g,"ROR") or gtechmap(g,"CSP") or gtechmap(g,"PVwSTO") or gtechmap(g,"STOPV") or gtechmap(g,"STORAGE")) and pGenData(g, "RetrYr") gt y.val) = 0;

* constraint to avoid early retirement of young plants
vRetire.fx(eg,y)$(y.val<pGasRetirement and pGenData(eg,"RetrYr") > y.val) = 0;
vRetire.fx(eg,y)$(y.val<pGenData(eg,"StYr")+15 and pGenData(eg,"RetrYr") > y.val) = 0;
*vRetire.fx(eg,y)$((pGenData(eg,"StYr") >= sStartYear.val) and (pGenData(eg,"RetrYr") > y.val)) = 0;

vRetireStor.fx(ng,y) = 0;
vRetireStor.fx(eg,y)$(gtechmap(eg,"CSP") or gtechmap(eg,"STORAGE")) = 0;
vRetireTherm.fx(ng,y) = 0;
vRetireTherm.fx(eg,y)$(gtechmap(eg,"CSP")) = 0;
vCap.fx(eg,y)$((pScalars("econRetire") = 0 and pGenData(eg,"StYr") < y.val) and (pGenData(eg,"RetrYr") >= y.val)) = pGenData(eg,"Capacity");
vCapTherm.fx(ng,y)$(pGenData(ng,"StYr") > y.val) = 0;
vCapStor.fx(ng,y)$(pGenData(ng,"StYr") > y.val) = 0;
vCapStor.fx(ng,y)$(not pincludeStorage) = 0;

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

pMinUtilizationRate(g,y)=0;
*To avoid bugs when there is no candidate transmission expansion line
$offIDCProtect
pNewTransmission(z,z2,"EarliestEntry")$(not pAllowHighTransfer) = 2500;

pMinUtilizationRate(g,y)  = pGenData(g,"MinUtilization");
*pMinUtilizationRate(g,y)$(y.val >= 2025) = pGenData(g,"MinLimitShare");

if (pScalars("pDecarbLevel") <0.8,
vCap.fx(eg,y)$((pGenData(eg,"StYr") < y.val) and (pGenData(eg,"StYr") >= sStartYear.val)) = pGenData(eg,"Capacity")   //committed in low decarbonized scenario
);


*display pMaxPriceImportShare;


$onIDCProtect



PA.HoldFixed=1;

file fmipopt / %MIPSOLVER%.opt /;
if (card(mipopt),
 put fmipopt;
 loop(mipline, put mipopt.te(mipline) /);
 putclose;
); 

PA.optfile = 1;
option savepoint=1;


Solve PA using MIP minimizing vNPVcost;

display pRR, pWeightYear;

$include %REPORT_FILE%
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
$exit
* This demonstrates how to do scenarios of a base with many reports
scalar xDR;
scalar mcnt /1/, low, upp;
low = pDR-0.05;
upp = pDR+0.05;

for (xDR=low to upp by 0.02,
  pRR(y) = 1/[(1+xDR)**(sum(y2$(ord(y2)<ord(y)),pWeightYear(y2)))];
  pDR = xDR;
$offIDCProtect
  pScalars("DR") = xDR;
$onIDCProtect
  Solve PA using MIP minimizing vNPVcost;
  put_utility 'save' / 'main.g00';
  put_utility 'exec.checkErrorLevel' / 'gams WB_EPM_v8_5_Report lo=2 r=main.g00 idcGDXOutput=dummyout --DEBUG=0 --DOEXCELREPORT=0 IDCGenerateGDX=scenDR' mcnt:0:0;
  mcnt = mcnt+1;
);
