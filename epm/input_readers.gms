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


*$if not set FOLDER_INPUT $set FOLDER_INPUT "data_gambia"

$ifThen not set FOLDER_INPUT
$set FOLDER_INPUT "data_test"
$endIf
$log FOLDER_INPUT is "%FOLDER_INPUT%"

$if not set TRACE $set TRACE 0

* Define by default path
* SETTINGS
$if not set pSettings $set pSettings input/%FOLDER_INPUT%/pSettings.csv
$if not set zcmap $set zcmap input/%FOLDER_INPUT%/zcmap.csv
$if not set y $set y input/%FOLDER_INPUT%/y.csv
$if not set pHours $set pHours input/%FOLDER_INPUT%/pHours.csv

* LOAD DATA
$if not set pDemandForecast $set pDemandForecast input/%FOLDER_INPUT%/load/pDemandForecast.csv
$if not set pDemandProfile $set pDemandProfile input/%FOLDER_INPUT%/load/pDemandProfile.csv
$if not set pDemandData $set pDemandData input/%FOLDER_INPUT%/load/pDemandData.csv
$if not set sRelevant $set sRelevant input/%FOLDER_INPUT%/load/sRelevant.csv
$if not set pEnergyEfficiencyFactor $set pEnergyEfficiencyFactor input/%FOLDER_INPUT%/load/pEnergyEfficiencyFactor.csv

* SUPPLY DATA
$if not set pGenDataExcel $set pGenDataExcel input/%FOLDER_INPUT%/supply/pGenDataExcelCustom.csv
$if not set pGenDataExcelDefault $set pGenDataExcelDefault input/%FOLDER_INPUT%/supply/pGenDataExcelDefault.csv
$if not set pAvailability $set pAvailability input/%FOLDER_INPUT%/supply/pAvailabilityCustom.csv
$if not set pAvailabilityDefault $set pAvailabilityDefault input/%FOLDER_INPUT%/supply/pAvailabilityDefault.csv
$if not set pVREgenProfile $set pVREgenProfile input/%FOLDER_INPUT%/supply/pVREgenProfile.csv
$if not set pVREProfile $set pVREProfile input/%FOLDER_INPUT%/supply/pVREProfile.csv
$if not set pCapexTrajectories $set pCapexTrajectories input/%FOLDER_INPUT%/supply/pCapexTrajectoriesCustom.csv
$if not set pCapexTrajectoriesDefault $set pCapexTrajectoriesDefault input/%FOLDER_INPUT%/supply/pCapexTrajectoriesDefault.csv
$if not set pFuelPrice $set pFuelPrice input/%FOLDER_INPUT%/supply/pFuelPrice.csv

* OTHER SUPPLY OPTIONS
$if not set pCSPData $set pCSPData input/%FOLDER_INPUT%/supply/pCSPData.csv
$if not set pStorDataExcel $set pStorDataExcel input/%FOLDER_INPUT%/supply/pStorDataExcel.csv

* RESOURCES
$if not set ftfindex $set ftfindex input/%FOLDER_INPUT%/resources/ftfindex.csv
$if not set pFuelCarbonContent $set pFuelCarbonContent input/%FOLDER_INPUT%/resources/pFuelCarbonContent.csv
$if not set pTechData $set pTechData input/%FOLDER_INPUT%/resources/pTechData.csv

* RESERVE
$if not set pPlanningReserveMargin $set pPlanningReserveMargin input/%FOLDER_INPUT%/reserve/pPlanningReserveMargin.csv
$if not set pSpinningReserveReqCountry $set pSpinningReserveReqCountry input/%FOLDER_INPUT%/reserve/pSpinningReserveReqCountry.csv
$if not set pSpinningReserveReqSystem $set pSpinningReserveReqSystem input/%FOLDER_INPUT%/reserve/pSpinningReserveReqSystem.csv

* TRADE
$if not set zext $set zext input/%FOLDER_INPUT%/trade/zext.csv
$if not set pExtTransferLimit $set pExtTransferLimit input/%FOLDER_INPUT%/trade/pExtTransferLimit.csv
$if not set pLossFactor $set pLossFactor input/%FOLDER_INPUT%/trade/pLossFactor.csv
$if not set pMaxPriceImportShare $set pMaxPriceImportShare input/%FOLDER_INPUT%/trade/pMaxPriceImportShare.csv
$if not set pMaxExchangeShare $set pMaxExchangeShare input/%FOLDER_INPUT%/trade/pMaxExchangeShare.csv
$if not set pMinImport $set pMinImport input/%FOLDER_INPUT%/trade/pMinImport.csv
$if not set pNewTransmission $set pNewTransmission input/%FOLDER_INPUT%/trade/pNewTransmissionCommitted.csv
$if not set pTradePrice $set pTradePrice input/%FOLDER_INPUT%/trade/pTradePrice.csv
$if not set pTransferLimit $set pTransferLimit input/%FOLDER_INPUT%/trade/pTransferLimit.csv

* CONSTRAINT
$if not set pCarbonPrice $set pCarbonPrice input/%FOLDER_INPUT%/constraint/pCarbonPrice.csv
$if not set pEmissionsCountry $set pEmissionsCountry input/%FOLDER_INPUT%/constraint/pEmissionsCountry.csv
$if not set pEmissionsTotal $set pEmissionsTotal input/%FOLDER_INPUT%/constraint/pEmissionsTotal.csv
$if not set pMaxFuellimit $set pMaxFuellimit input/%FOLDER_INPUT%/constraint/pMaxFuellimit.csv

* H2 RELATED
$if not set pH2DataExcel $set pH2DataExcel input/%FOLDER_INPUT%/h2/pH2DataExcel.csv
$if not set pAvailabilityH2 $set pAvailabilityH2 input/%FOLDER_INPUT%/h2/pAvailabilityH2.csv
$if not set pFuelDataH2 $set pFuelDataH2 input/%FOLDER_INPUT%/h2/pFuelDataH2.csv
$if not set pCapexTrajectoryH2 $set pCapexTrajectoryH2 input/%FOLDER_INPUT%/h2/pCapexTrajectoryH2.csv
$if not set pH2DataExcel $set pH2DataExcel input/%FOLDER_INPUT%/h2/pH2DataExcel.csv
$if not set pExternalH2 $set pExternalH2 input/%FOLDER_INPUT%/h2/pExternalH2.csv


$log ### reading using Connect and CSV Input with Python

$onEmbeddedCode Connect:


- CSVReader:
    trace: %TRACE%
    file: %pSettings%
    name: pSettings
    valueSubstitutions: {0: .nan}
    indexColumns: [2]
    valueColumns: [3]
    type: par


- CSVReader:
    trace: %TRACE%
    file: %zcmap%
    name: zcmap
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    type: set

- CSVReader:
    trace: %TRACE%
    file: %y%
    name: y
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set
    

- CSVReader:
    trace: %TRACE%
    file: %pHours%
    name: pHours
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: EPS}
    header: [1]
    indexColumns: [1, 2]
    type: par


# LOAD DATA

- CSVReader:
    trace: %TRACE%
    file: %pDemandForecast%
    name: pDemandForecast
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pDemandProfile%
    name: pDemandProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par
    
- CSVReader:
    trace: %TRACE%
    file: %pDemandData%
    name: pDemandData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par
    
- CSVReader:
    trace: %TRACE%
    file: %sRelevant%
    name: sRelevant
    indexColumns: [1]
    type: set


- CSVReader:
    trace: %TRACE%
    file: %pEnergyEfficiencyFactor%
    name: pEnergyEfficiencyFactor
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    header: [1]
    indexColumns: [1]
    type: par


# SUPPLY DATA


- CSVReader:
    trace: %TRACE%
    file: %pTechData%
    name: pTechData
    indexSubstitutions: {.nan: ""}
    header: [1]
    indexColumns: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pGenDataExcel%
    name: gmap
    indexSubstitutions: {.nan: ""}
    indexColumns: [1,2,3,4]
    type: set

- CSVReader:
    trace: %TRACE%
    file: %pGenDataExcel%
    name: pGenDataExcel
    indexColumns: [1,2,3,4]
    valueSubstitutions: {0: EPS}
    header: [1]
    type: par

    
- CSVReader:
    trace: %TRACE%
    file: %pGenDataExcelDefault%
    name: pGenDataExcelDefault
    indexColumns: [1,2,3]
    valueSubstitutions: {0: EPS}
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pAvailability%
    name: pAvailability
    valueSubstitutions: {0: EPS}
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: %TRACE%
    file: %pAvailabilityDefault%
    name: pAvailabilityDefault
    indexColumns: [1, 2, 3]
    valueSubstitutions: {0: EPS}
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pVREgenProfile%
    name: pVREgenProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: EPS}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pVREProfile%
    name: pVREProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: EPS}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par


- CSVReader:
    trace: %TRACE%
    file: %pCapexTrajectories%
    name: pCapexTrajectories
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pCapexTrajectoriesDefault%
    name: pCapexTrajectoriesDefault
    valueSubstitutions: {0: EPS}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par
    
- CSVReader:
    file: %pFuelPrice%
    name: pFuelPrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    header: [1]
    indexColumns: [1, 2]
    type: par
    
# OTHER SUPLLY OPTIONS

- CSVReader:
    trace: %TRACE%
    file: %pCSPData%
    name: pCSPData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pStorDataExcel%
    name: pStorDataExcel
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

# RESOURCES

- CSVReader:
    file: %ftfindex%
    name: ftfindex
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pFuelCarbonContent%
    name: pFuelCarbonContent
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

# RESERVE

- CSVReader:
    trace: %TRACE%
    file: %pPlanningReserveMargin%
    name: pPlanningReserveMargin
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pSpinningReserveReqCountry%
    name: pSpinningReserveReqCountry
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pSpinningReserveReqSystem%
    name: pSpinningReserveReqSystem
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par


# TRADE

- CSVReader:
    trace: %TRACE%
    file: %zext%
    name: zext
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set

- CSVReader:
    trace: %TRACE%
    file: %pExtTransferLimit%
    name: pExtTransferLimit
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3,4]
    header: [1]
    type: par
    

- CSVReader:
    trace: %TRACE%
    file: %pMaxExchangeShare%
    name: pMaxExchangeShare
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pMaxPriceImportShare%
    name: pMaxPriceImportShare
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pTradePrice%
    name: pTradePrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pNewTransmission%
    name: pNewTransmission
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pLossFactor%
    name: pLossFactor
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pTransferLimit%
    name: pTransferLimit
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par
    
- CSVReader:
    trace: %TRACE%
    file: %pMinImport%
    name: pMinImport
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par


# ENVIRONMENTAL CONSTRAINT

- CSVReader:
    trace: %TRACE%
    file: %pCarbonPrice%
    name: pCarbonPrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pEmissionsCountry%
    name: pEmissionsCountry
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pEmissionsTotal%
    name: pEmissionsTotal
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par


- CSVReader:
    trace: %TRACE%
    file: %pMaxFuellimit%
    name: pMaxFuellimit
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par


# H2 RELATED

- CSVReader:
    trace: %TRACE%
    file: %pH2DataExcel%
    name: pH2DataExcel
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pAvailabilityH2%
    name: pAvailabilityH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pFuelDataH2%
    name: pFuelDataH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pCapexTrajectoryH2%
    name: pCapexTrajectoryH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par


- CSVReader:
    trace: %TRACE%
    file: %pH2DataExcel%
    name: hh
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set

- CSVReader:
    trace: %TRACE%
    file: %pExternalH2%
    name: pExternalH2
    indexColumns: [1, 2]
    header: [1]
    type: par

- GDXWriter:
    file: input.gdx
    symbols: all


$offEmbeddedCode

* Extract file path (`fp`), base filename (`GDX_INPUT`), and file extension (`fe`) from `%XLS_INPUT%`
$setNames "%XLS_INPUT%" fp GDX_INPUT fe
