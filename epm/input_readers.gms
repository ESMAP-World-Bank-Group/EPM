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

* Set the maximum number of rows for Excel processing
$set XLSXMAXROWS 1048576

* Set the default reader method to `CONNECT_EXCEL` if not already defined
$if not set READER $set READER CONNECT_CSV

$if not set FOLDER_INPUT $set FOLDER_INPUT "data_gambia"

* Log the selected reader method for debugging
$log ### READER = %READER%

* Remove the existing GDX file
$call rm %GDX_INPUT%.gdx

* Attempt to generate a new GDX file from the Excel input
$call test %GDX_INPUT%.gdx -nt "%XLS_INPUT%"

* Check for errors during the GDX file generation process
$ifThen.errorLevel errorlevel 1



$ifThenI.READER %READER% == CONNECT_EXCEL


$log ### Reading from %XLS_INPUT% using Connect and Excel Input is not enabled in this version.

$elseIfI.READER %READER% == CONNECT_CSV
$log ### reading using Connect and CSV Input


$onEmbeddedCode Connect:

# SETTINGS

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/pSettings.csv
    name: pSettings
    valueSubstitutions: {0: .nan}
    indexColumns: [2]
    valueColumns: [3]
    type: par


- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/zcmap.csv
    name: zcmap
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    type: set

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/y.csv
    name: y
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set
    

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/pHours.csv
    name: pHours
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: EPS}
    header: [1]
    indexColumns: [1, 2]
    type: par


# LOAD DATA

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/load/pDemandForecast.csv
    name: pDemandForecast
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/load/pDemandProfile.csv
    name: pDemandProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/load/pDemandData.csv
    name: pDemandData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/load/sRelevant.csv
    name: sRelevant
    indexColumns: [1]
    type: set


- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/load/pEnergyEfficiencyFactor.csv
    name: pEnergyEfficiencyFactor
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    header: [1]
    indexColumns: [1]
    type: par


# SUPPLY DATA


- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/resources/pTechData.csv
    name: pTechData
    indexSubstitutions: {.nan: ""}
    header: [1]
    indexColumns: [1]
    type: par


- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/supply/pGenDataExcelCustom.csv
    name: gmap
    indexSubstitutions: {.nan: ""}
    indexColumns: [1,2,3,4]
    type: set

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/supply/pGenDataExcelCustom.csv
    name: pGenDataExcel
    indexColumns: [1,2,3,4]
    valueSubstitutions: {0: EPS}
    header: [1]
    type: par

    
- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/supply/pGenDataExcelDefault.csv
    name: pGenDataExcelDefault
    indexColumns: [1,2,3]
    valueSubstitutions: {0: EPS}
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/supply/pAvailabilityCustom.csv
    name: pAvailability
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/supply/pAvailabilityDefault.csv
    name: pAvailabilityDefault
    indexColumns: [1, 2, 3]
    valueSubstitutions: {0: EPS}
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/supply/pVREgenProfile.csv
    name: pVREgenProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: EPS}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/supply/pVREProfile.csv
    name: pVREProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: EPS}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par


- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/supply/pCapexTrajectoriesCustom.csv
    name: pCapexTrajectories
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/supply/pCapexTrajectoriesDefault.csv
    name: pCapexTrajectoriesDefault
    valueSubstitutions: {0: EPS}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par
    
- CSVReader:
    file: input/%FOLDER_INPUT%/supply/pFuelPrice.csv
    name: pFuelPrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    header: [1]
    indexColumns: [1, 2]
    type: par
    
# OTHER SUPLLY OPTIONS

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/supply/pCSPData.csv
    name: pCSPData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/supply/pStorDataExcel.csv
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    name: pStorDataExcel
    indexColumns: [1, 2]
    header: [1]
    type: par



# RESOURCES

- CSVReader:
    file: input/%FOLDER_INPUT%/resources/ftfindex.csv
    name: ftfindex
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/resources/pFuelCarbonContent.csv
    name: pFuelCarbonContent
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

# RESERVE


- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/reserve/pPlanningReserveMargin.csv
    name: pPlanningReserveMargin
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/reserve/pSpinningReserveReqCountry.csv
    name: pSpinningReserveReqCountry
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/reserve/pSpinningReserveReqSystem.csv
    name: pSpinningReserveReqSystem
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par


# TRADE

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/trade/zext.csv
    name: zext
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/trade/pExtTransferLimit.csv
    name: pExtTransferLimit
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3,4]
    header: [1]
    type: par
    

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/trade/pMaxExchangeShare.csv
    name: pMaxExchangeShare
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/trade/pMaxPriceImportShare.csv
    name: pMaxPriceImportShare
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/trade/pTradePrice.csv
    name: pTradePrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/trade/pNewTransmission.csv
    name: pNewTransmission
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par


- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/trade/pLossFactor.csv
    name: pLossFactor
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/trade/pTransferLimit.csv
    name: pTransferLimit
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/trade/pMinImport.csv
    name: pMinImport
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par


# ENVIRONMENTAL CONSTRAINT

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/constraint/pCarbonPrice.csv
    name: pCarbonPrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/constraint/pEmissionsCountry.csv
    name: pEmissionsCountry
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/constraint/pEmissionsTotal.csv
    name: pEmissionsTotal
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par


- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/constraint/pMaxFuellimit.csv
    name: pMaxFuellimit
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par


# H2 RELATED

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/h2/pH2DataExcel.csv
    name: pH2DataExcel
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/h2/pAvailabilityH2.csv
    name: pAvailabilityH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/h2/pFuelDataH2.csv
    name: pFuelDataH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/h2/pCapexTrajectoryH2.csv
    name: pCapexTrajectoryH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par


- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/h2/pH2DataExcel.csv
    name: hh
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set


- CSVReader:
    trace: 0
    file: input/%FOLDER_INPUT%/h2/pExternalH2.csv
    name: pExternalH2
    indexColumns: [1, 2]
    header: [1]
    type: par


- GDXWriter:
    file: %GDX_INPUT%.gdx
    symbols: all
$offEmbeddedCode


$elseIfI.READER %READER% == CONNECT_CSV_PYTHON
$log ### reading using Connect and CSV Input with Python

$onEmbeddedCode Connect:


- CSVReader:
    trace: 0
    file: %pSettings%
    name: pSettings
    valueSubstitutions: {0: .nan}
    indexColumns: [2]
    valueColumns: [3]
    type: par


- CSVReader:
    trace: 0
    file: %zcmap%
    name: zcmap
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    type: set

- CSVReader:
    trace: 0
    file: %y%
    name: y
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set
    

- CSVReader:
    trace: 0
    file: %pHours%
    name: pHours
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: EPS}
    header: [1]
    indexColumns: [1, 2]
    type: par


# LOAD DATA

- CSVReader:
    trace: 0
    file: %pDemandForecast%
    name: pDemandForecast
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pDemandProfile%
    name: pDemandProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: %pDemandData%
    name: pDemandData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: %sRelevant%
    name: sRelevant
    indexColumns: [1]
    type: set


- CSVReader:
    trace: 0
    file: %pEnergyEfficiencyFactor%
    name: pEnergyEfficiencyFactor
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    header: [1]
    indexColumns: [1]
    type: par


# SUPPLY DATA


- CSVReader:
    trace: 0
    file: %pTechData%
    name: pTechData
    indexSubstitutions: {.nan: ""}
    header: [1]
    indexColumns: [1]
    type: par


- CSVReader:
    trace: 0
    file: %gmap%
    name: gmap
    indexSubstitutions: {.nan: ""}
    indexColumns: [1,2,3,4]
    type: set

- CSVReader:
    trace: 0
    file: %pGenDataExcel%
    name: pGenDataExcel
    indexColumns: [1,2,3,4]
    valueSubstitutions: {0: EPS}
    header: [1]
    type: par

    
- CSVReader:
    trace: 0
    file: %pGenDataExcelDefault%
    name: pGenDataExcelDefault
    indexColumns: [1,2,3]
    valueSubstitutions: {0: EPS}
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pAvailability%
    name: pAvailability
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: %pAvailabilityDefault%
    name: pAvailabilityDefault
    indexColumns: [1, 2, 3]
    valueSubstitutions: {0: EPS}
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pVREgenProfile%
    name: pVREgenProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: EPS}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pVREProfile%
    name: pVREProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: EPS}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par


- CSVReader:
    trace: 0
    file: %pCapexTrajectories%
    name: pCapexTrajectories
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
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
    trace: 0
    file: %pCSPData%
    name: pCSPData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
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
    trace: 0
    file: %pFuelCarbonContent%
    name: pFuelCarbonContent
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

# RESERVE

- CSVReader:
    trace: 0
    file: %pPlanningReserveMargin%
    name: pPlanningReserveMargin
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: %pSpinningReserveReqCountry%
    name: pSpinningReserveReqCountry
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pSpinningReserveReqSystem%
    name: pSpinningReserveReqSystem
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par


# TRADE

- CSVReader:
    trace: 0
    file: %zext%
    name: zext
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set

- CSVReader:
    trace: 0
    file: %pExtTransferLimit%
    name: pExtTransferLimit
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3,4]
    header: [1]
    type: par
    

- CSVReader:
    trace: 0
    file: %pMaxExchangeShare%
    name: pMaxExchangeShare
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pMaxPriceImportShare%
    name: pMaxPriceImportShare
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pTradePrice%
    name: pTradePrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pNewTransmission%
    name: pNewTransmission
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pLossFactor%
    name: pLossFactor
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pTransferLimit%
    name: pTransferLimit
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: %pMinImport%
    name: pMinImport
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par


# ENVIRONMENTAL CONSTRAINT

- CSVReader:
    trace: 0
    file: %pCarbonPrice%
    name: pCarbonPrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: %pEmissionsCountry%
    name: pEmissionsCountry
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pEmissionsTotal%
    name: pEmissionsTotal
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par


- CSVReader:
    trace: 0
    file: %pMaxFuellimit%
    name: pMaxFuellimit
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par


# H2 RELATED

- CSVReader:
    trace: 0
    file: %pH2DataExcel%
    name: pH2DataExcel
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pAvailabilityH2%
    name: pAvailabilityH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pFuelDataH2%
    name: pFuelDataH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: %pCapexTrajectoryH2%
    name: pCapexTrajectoryH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par


- CSVReader:
    trace: 0
    file: %hh%
    name: hh
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set


- CSVReader:
    trace: 0
    file: %pExternalH2%
    name: pExternalH2
    indexColumns: [1, 2]
    header: [1]
    type: par


- GDXWriter:
    file: %GDX_INPUT%.gdx
    symbols: all
$offEmbeddedCode

$else.READER
$abort 'No valid READER specified. Allowed are GDXXRW and CONNECT.'
$endif.READER
$endif.errorLevel

* Extract file path (`fp`), base filename (`GDX_INPUT`), and file extension (`fe`) from `%XLS_INPUT%`
$setNames "%XLS_INPUT%" fp GDX_INPUT fe
