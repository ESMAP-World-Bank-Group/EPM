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
* - The model requires input data in .GDX.
*
* Contact:
* Claire Nicolas, cnicolas@worldbank.org
**********************************************************************


$if not set TRACE $set TRACE 0

* Define by default path
* SETTINGS
$if not set pSettings $set pSettings %FOLDER_INPUT%/pSettings_startupcost.csv
$if not set zcmap $set zcmap %FOLDER_INPUT%/zcmap.csv
$if not set y $set y %FOLDER_INPUT%/y.csv
$if not set pHours $set pHours %FOLDER_INPUT%/pHours_dispatch.csv
$if not set pDays $set pDays %FOLDER_INPUT%/static/dispatch_month_days.csv
$if not set mapTS $set mapTS %FOLDER_INPUT%/static/dispatch_map_ts.csv

* LOAD DATA
$if not set pDemandForecast $set pDemandForecast %FOLDER_INPUT%/load/pDemandForecast.csv
$if not set pDemandProfile $set pDemandProfile %FOLDER_INPUT%/load/pDemandProfile_dispatch.csv
$if not set pDemandData $set pDemandData %FOLDER_INPUT%/load/pDemandData_dispatch.csv
$if not set sRelevant $set sRelevant %FOLDER_INPUT%/load/sRelevant.csv
$if not set pEnergyEfficiencyFactor $set pEnergyEfficiencyFactor %FOLDER_INPUT%/load/pEnergyEfficiencyFactor.csv

* SUPPLY DATA
$if not set pGenDataInput $set pGenDataInput %FOLDER_INPUT%/supply/pGenDataInput.csv
$if not set pGenDataInputDefault $set pGenDataInputDefault %FOLDER_INPUT%/supply/pGenDataInputDefault.csv
$if not set pAvailability $set pAvailability %FOLDER_INPUT%/supply/pAvailabilityCustom_dispatch.csv
$if not set pAvailabilityDefault $set pAvailabilityDefault %FOLDER_INPUT%/supply/pAvailabilityDefault_dispatch.csv
$if not set pVREgenProfile $set pVREgenProfile %FOLDER_INPUT%/supply/pVREgenProfile_dispatch.csv
$if not set pVREProfile $set pVREProfile %FOLDER_INPUT%/supply/pVREProfile_dispatch.csv
$if not set pCapexTrajectories $set pCapexTrajectories %FOLDER_INPUT%/supply/pCapexTrajectoriesCustom.csv
$if not set pCapexTrajectoriesDefault $set pCapexTrajectoriesDefault %FOLDER_INPUT%/supply/pCapexTrajectoriesDefault.csv
$if not set pFuelPrice $set pFuelPrice %FOLDER_INPUT%/supply/pFuelPrice.csv

* OTHER SUPPLY OPTIONS
$if not set pCSPData $set pCSPData %FOLDER_INPUT%/supply/pCSPData.csv
$if not set pStorDataExcel $set pStorDataExcel %FOLDER_INPUT%/supply/pStorDataExcel.csv

* RESOURCES
$if not set pSettingsHeader $set pSettingsHeader %FOLDER_INPUT%/resources/pSettingsHeader.csv
$if not set pGenDataInputHeader $set pGenDataInputHeader %FOLDER_INPUT%/resources/pGenDataInputHeader.csv
$if not set pStoreDataHeader $set pStoreDataHeader %FOLDER_INPUT%/resources/pStoreDataHeader.csv
$if not set pH2Header $set pH2Header %FOLDER_INPUT%/resources/pH2Header.csv

$if not set ftfindex $set ftfindex %FOLDER_INPUT%/resources/ftfindex.csv
$if not set pFuelCarbonContent $set pFuelCarbonContent %FOLDER_INPUT%/resources/pFuelCarbonContent.csv
$if not set pTechData $set pTechData %FOLDER_INPUT%/resources/pTechData.csv

* RESERVE
$if not set pPlanningReserveMargin $set pPlanningReserveMargin %FOLDER_INPUT%/reserve/pPlanningReserveMargin.csv
$if not set pSpinningReserveReqCountry $set pSpinningReserveReqCountry %FOLDER_INPUT%/reserve/pSpinningReserveReqCountry.csv
$if not set pSpinningReserveReqSystem $set pSpinningReserveReqSystem %FOLDER_INPUT%/reserve/pSpinningReserveReqSystem.csv

* TRADE
$if not set zext $set zext %FOLDER_INPUT%/trade/zext.csv
$if not set pTransmissionHeader $set pTransmissionHeader %FOLDER_INPUT%/resources/pTransmissionHeader.csv
$if not set pExtTransferLimit $set pExtTransferLimit %FOLDER_INPUT%/trade/pExtTransferLimit.csv
$if not set pLossFactorInternal $set pLossFactorInternal %FOLDER_INPUT%/trade/pLossFactorInternal.csv
$if not set pMaxPriceImportShare $set pMaxPriceImportShare %FOLDER_INPUT%/trade/pMaxPriceImportShare.csv
$if not set pMaxAnnualExternalTradeShare $set pMaxAnnualExternalTradeShare %FOLDER_INPUT%/trade/pMaxAnnualExternalTradeShare.csv
$if not set pMinImport $set pMinImport %FOLDER_INPUT%/trade/pMinImport.csv
$if not set pNewTransmission $set pNewTransmission %FOLDER_INPUT%/trade/pNewTransmission.csv
$if not set pTradePrice $set pTradePrice %FOLDER_INPUT%/trade/pTradePrice_dispatch.csv
$if not set pTransferLimit $set pTransferLimit %FOLDER_INPUT%/trade/pTransferLimit_dispatch.csv

* CONSTRAINT
$if not set pCarbonPrice $set pCarbonPrice %FOLDER_INPUT%/constraint/pCarbonPrice.csv
$if not set pEmissionsCountry $set pEmissionsCountry %FOLDER_INPUT%/constraint/pEmissionsCountry.csv
$if not set pEmissionsTotal $set pEmissionsTotal %FOLDER_INPUT%/constraint/pEmissionsTotal.csv
$if not set pMaxFuellimit $set pMaxFuellimit %FOLDER_INPUT%/constraint/pMaxFuellimit.csv

* H2 RELATED
$if not set pH2DataExcel $set pH2DataExcel %FOLDER_INPUT%/h2/pH2DataExcel.csv
$if not set pAvailabilityH2 $set pAvailabilityH2 %FOLDER_INPUT%/h2/pAvailabilityH2.csv
$if not set pFuelDataH2 $set pFuelDataH2 %FOLDER_INPUT%/h2/pFuelDataH2.csv
$if not set pCapexTrajectoryH2 $set pCapexTrajectoryH2 %FOLDER_INPUT%/h2/pCapexTrajectoryH2.csv
$if not set pH2DataExcel $set pH2DataExcel %FOLDER_INPUT%/h2/pH2DataExcel.csv
$if not set pExternalH2 $set pExternalH2 %FOLDER_INPUT%/h2/pExternalH2.csv


$log ### reading using Connect and CSV Input with Python

$onEmbeddedCode Connect:


- CSVReader:
    trace: %TRACE%
    file: %pSettingsHeader%
    name: pSettingsHeader
    indexColumns: [1]
    type: set


- CSVReader:
    trace: %TRACE%
    file: %pSettings%
    name: pSettings
    indexColumns: [2]
    valueColumns: [3]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pDays%
    name: pDays
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: EPS}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: %TRACE%
    file: %pGenDataInputHeader%
    name: pGenDataInputHeader
    indexColumns: [1]
    type: set
    
- CSVReader:
    trace: %TRACE%
    file: %pStoreDataHeader%
    name: pStoreDataHeader
    indexColumns: [1]
    type: set
    
- CSVReader:
    trace: %TRACE%
    file: %pH2Header%
    name: pH2Header
    indexColumns: [1]
    type: set


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
    file: %mapTS%
    name: mapTS
    indexSubstitutions: {.nan: ""}
    indexColumns: [1, 2, 3, 4]
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
    file: %pGenDataInput%
    name: gmap
    indexSubstitutions: {.nan: ""}
    indexColumns: [1,2,3,4]
    type: set

- CSVReader:
    trace: %TRACE%
    file: %pGenDataInput%
    name: pGenDataInput
    indexColumns: [1,2,3,4]
    header: [1]
    type: par

    
- CSVReader:
    trace: %TRACE%
    file: %pGenDataInputDefault%
    name: pGenDataInputDefault
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
    indexSubstitutions: {.nan: "", .nan: EPS}
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
    file: %pTransmissionHeader%
    name: pTransmissionHeader
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
    file: %pMaxAnnualExternalTradeShare%
    name: pMaxAnnualExternalTradeShare
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
    file: %pLossFactorInternal%
    name: pLossFactorInternal
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
    file: %pExternalH2%
    name: pExternalH2
    indexColumns: [1, 2]
    header: [1]
    type: par

- GDXWriter:
    file: input.gdx
    symbols: all


$offEmbeddedCode
