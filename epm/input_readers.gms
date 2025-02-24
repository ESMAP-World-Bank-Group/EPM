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
$if not set READER $set READER CONNECT_EXCEL

* Log the selected reader method for debugging
$log ### READER = %READER%

* Remove the existing GDX file
$call rm %GDX_INPUT%.gdx

* Attempt to generate a new GDX file from the Excel input
$call test %GDX_INPUT%.gdx -nt "%XLS_INPUT%"

* Check for errors during the GDX file generation process
$ifThen.errorLevel errorlevel 1

$ifThenI.READER %READER% == CONNECT_EXCEL
$log ### Reading from %XLS_INPUT% using Connect and Excel Input

$onEmbeddedCode Connect:
- ExcelReader:
    file: %XLS_INPUT%
    valueSubstitutions: {0: .nan}  # drop zeroes
    indexSubstitutions: {.nan: ""}  # keep empty labels
    symbols:
      - name: ftfindex
        range: FuelTechnologies!A3
        rowDimension: 2
        columnDimension: 0
      - name: pHours
        range: Duration!A6
        rowDimension: 2
        columnDimension: 1
      - name: pZoneIndex
        range: ZoneData!E7:F200
        rowDimension: 1
        columnDimension: 0
      - name: pGenDataExcel
        range: GenData!A6
        rowDimension: 4
        columnDimension: 1
      - name: gmap
        range: GenData!A6
        rowDimension: 4
        columnDimension: 0
        type: set
      - name: pTechDataExcel
        range: FuelTechnologies!B29:F50
        rowDimension: 1
        columnDimension: 1
        valueSubstitutions: {"NO": .nan, "YES": 1}  # drop "NO" and read "YES" as 1
      - name: zcmapExcel
        range: ZoneData!U7
        rowDimension: 2
        columnDimension: 0
        type: set
        ignoreText: True
      - name: y
        range: LoadDefinition!A6:A%XLSXMAXROWS%
        rowDimension: 1
        columnDimension: 0
        type: set
        ignoreText: True
      - name: sTopology
        range: Topology!A6
        rowDimension: 1
        columnDimension: 1
        type: set
        valueSubstitutions: {
          "": .nan,  # Some empty cells seem to contain "" (empty string), drop those
          1: ""  # Read 1 as empty string
        }
      - name: peak
        range: LoadDefinition!H6:H%XLSXMAXROWS%
        rowDimension: 1
        columnDimension: 0
        type: set
        ignoreText: True
      - name: Relevant
        range: LoadDefinition!E6:E%XLSXMAXROWS%
        rowDimension: 1
        columnDimension: 0
        type: set
        ignoreText: True
      - name: zext
        range: ZoneData!G7:G60
        rowDimension: 1
        columnDimension: 0
        type: set
      - name: pPlanningReserveMargin
        range: PlanningReserve!A6
        rowDimension: 1
        columnDimension: 0
      - name: pEnergyEfficiencyFactor
        range: EnergyEfficiency!A5
        rowDimension: 1
        columnDimension: 1
      - name: pScalars
        range: Settings1!B3:C70
        rowDimension: 1
        columnDimension: 0
        indexSubstitutions: {" IncludeH2": "IncludeH2"}
      - name: pAvailability
        range: GenAvailability!A6
        rowDimension: 1
        columnDimension: 1
      - name: pVREProfile
        range: REProfile!A6
        rowDimension: 4
        columnDimension: 1
      - name: pLossFactor
        range: LossFactor!A5
        rowDimension: 2
        columnDimension: 1
      - name: pTransferLimit
        range: TransferLimit!A5
        rowDimension: 3
        columnDimension: 1
      - name: pMinImport
        range: ImportShare!W3
        rowDimension: 2
        columnDimension: 1
      - name: pExtTransferLimit
        range: ExtTransferLimit!A5
        rowDimension: 4
        columnDimension: 1
      - name: pCarbonPrice
        range: EmissionFactors!A3:B24
        rowDimension: 1
        columnDimension: 0
      - name: pDemandData
        range: Demand!A6
        rowDimension: 4
        columnDimension: 1
      - name: pDemandForecast
        range: Demand_Forecast!A6
        rowDimension: 2
        columnDimension: 1
      - name: pDemandProfile
        range: DemandProfile!A6
        rowDimension: 3
        columnDimension: 1
      - name: pMaxExchangeShare
        range: ExchangeShare!A7
        rowDimension: 1
        columnDimension: 1
      - name: pTradePrice
        range: TradePrices!A6
        rowDimension: 4
        columnDimension: 1
      - name: pNewTransmission
        range: ZoneData!J6:Q107
        rowDimension: 2
        columnDimension: 1
      - name: pCapexTrajectory
        range: CapexTrajectories!A5
        rowDimension: 1
        columnDimension: 1
      - name: pCSPData
        range: CSP!A6
        ignoreColumns: E
        rowDimension: 2
        columnDimension: 1
      - name: pStorDataExcel
        range: Storage!A6
        rowDimension: 2
        columnDimension: 1
        indexSubstitutions: {"": .nan}
      - name: pFuelTypeCarbonContent
        range: EmissionFactors!J3
        rowDimension: 1
        columnDimension: 0
      - name: pSpinningReserveReqCountry
        range: SpinReserve!F5
        rowDimension: 1
        columnDimension: 1
      - name: pSpinningReserveReqSystem
        range: SpinReserve!A6
        rowDimension: 1
        columnDimension: 0
      - name: pEmissionsCountry
        range: Emissions!A17
        rowDimension: 1
        columnDimension: 1
      - name: pEmissionsTotal
        range: Emissions!A5
        rowDimension: 0
        columnDimension: 1
      - name: pFuelPrice
        range: FuelPrices!A6
        rowDimension: 2
        columnDimension: 1
      - name: pMaxFuellimit
        range: FuelLimit!A6
        rowDimension: 2
        columnDimension: 1
      - name: pVREgenProfile
        range: REgenProfile!A6
        rowDimension: 4
        columnDimension: 1
      - name: hh
        range: H2Data!A7:A%XLSXMAXROWS%
        rowDimension: 1
        columnDimension: 0
        type: set
        ignoreText: True
      - name: pH2DataExcel
        range: H2Data!A6
        rowDimension: 1
        columnDimension: 1
      - name: pAvailabilityH2
        range: H2Availability!A6
        rowDimension: 1
        columnDimension: 1
      - name: pFuelData
        range: FuelTechnologies!J3:K17  # range: FuelTechnologies!J3:K44
        columnDimension: 0
      - name: pCapexTrajectoryH2
        range: CapexTrajectoriesH2!A5
        rowDimension: 1
        columnDimension: 1
      - name: pExternalH2
        range: ExternalH2Demand!A5
        rowDimension: 2
        columnDimension: 1
      #- name: pRETargetSeriesYr
      #  range: REtargets!A6
      #  rowDimension: 1
      #  columnDimension: 1
- GDXWriter:
    file: %GDX_INPUT%.gdx
    symbols: all
$offEmbeddedCode

$elseIfI.READER %READER% == CONNECT_CSV
$log ### reading using Connect and CSV Input


$onEmbeddedCode Connect:

- CSVReader:
    trace: 0
    file: input/data/pAvailability.csv
    name: pAvailability
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pDemandData.csv
    name: pDemandData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pExternalH2.csv
    name: pExternalH2
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/relevant.csv
    name: relevant
    indexColumns: [1]
    type: set

- CSVReader:
    trace: 0
    file: input/data/peak.csv
    name: peak
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    type: set
    indexColumns: [1]

- CSVReader:
    file: input/data/pFuelPrice.csv
    name: pFuelPrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    header: [1]
    indexColumns: [1, 2]
    type: par

- CSVReader:
    file: input/data/ftfindex.csv
    name: ftfindex
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    valueColumns: [3]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pHours.csv
    name: pHours
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    header: [1]
    indexColumns: [1, 2]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pZoneIndex.csv
    name: pZoneIndex
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pTechDataExcel.csv
    name: pTechDataExcel
    indexSubstitutions: {.nan: ""}
    header: [1]
    indexColumns: [1]
    type: par
    valueSubstitutions: {"NO": .nan, "YES": 1}  # drop "NO" and read "YES" as 1

- CSVReader:
    trace: 0
    file: input/data/zcmapExcel.csv
    name: zcmapExcel
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    type: set

- CSVReader:
    trace: 0
    file: input/data/y.csv
    name: y
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set

- CSVReader:
    trace: 0
    file: input/data/pEnergyEfficiencyFactor.csv
    name: pEnergyEfficiencyFactor
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    header: [1]
    indexColumns: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pScalars.csv
    name: pScalars
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pVREProfile.csv
    name: pVREProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pLossFactor.csv
    name: pLossFactor
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pTransferLimit.csv
    name: pTransferLimit
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/data/pMinImport.csv
    name: pMinImport
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pCarbonPrice.csv
    name: pCarbonPrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pDemandForecast.csv
    name: pDemandForecast
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pDemandProfile.csv
    name: pDemandProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/data/pMaxExchangeShare.csv
    name: pMaxExchangeShare
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pMaxPriceImportShare.csv
    name: pMaxPriceImportShare
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pTradePrice.csv
    name: pTradePrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pNewTransmission.csv
    name: pNewTransmission
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pCapexTrajectory.csv
    name: pCapexTrajectory
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pCSPData.csv
    name: pCSPData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pStorDataExcel.csv
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    name: pStorDataExcel
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pFuelTypeCarbonContent.csv
    name: pFuelTypeCarbonContent
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pPlanningReserveMargin.csv
    name: pPlanningReserveMargin
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pSpinningReserveReqCountry.csv
    name: pSpinningReserveReqCountry
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pSpinningReserveReqSystem.csv
    name: pSpinningReserveReqSystem
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pEmissionsCountry.csv
    name: pEmissionsCountry
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pEmissionsTotal.csv
    name: pEmissionsTotal
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pMaxFuellimit.csv
    name: pMaxFuellimit
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pVREgenProfile.csv
    name: pVREgenProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par


- CSVReader:
    trace: 0
    file: input/data/pH2DataExcel.csv
    name: pH2DataExcel
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/data/pGenDataExcelDefault.csv
    name: pGenDataExcelDefault
    indexColumns: [1,2,3]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pGenDataExcelCustom.csv
    name: pGenDataExcel
    indexColumns: [1,2,3,4]
    valueSubstitutions: {.nan: 0}
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/data/pGenDataExcelCustom.csv
    name: gmap
    indexSubstitutions: {.nan: ""}
    indexColumns: [1,2,3,4]
    type: set


- CSVReader:
    trace: 0
    file: input/data/pAvailabilityH2.csv
    name: pAvailabilityH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pFuelData.csv
    name: pFuelData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/data/pCapexTrajectoryH2.csv
    name: pCapexTrajectoryH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par


- CSVReader:
    trace: 0
    file: input/data/pH2DataExcel.csv
    name: hh
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set

- CSVReader:
    file: input/data/sTopology.csv
    name: sTopology
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    type: set

- CSVReader:
    trace: 0
    file: input/data/zext.csv
    name: zext
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set

- CSVReader:
    trace: 0
    file: input/data/pExtTransferLimit.csv
    name: pExtTransferLimit
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3,4]
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
    file: %pAvailability%
    name: pAvailability
    indexColumns: [1]
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
    file: %pExternalH2%
    name: pExternalH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %relevant%
    name: relevant
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set

- CSVReader:
    trace: 0
    file: %peak%
    name: peak
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    type: set
    indexColumns: [1]

- CSVReader:
    file: %pFuelPrice%
    name: pFuelPrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    header: [1]
    indexColumns: [1, 2]
    type: par

- CSVReader:
    file: %ftfindex%
    name: ftfindex
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    valueColumns: [3]
    type: par

- CSVReader:
    trace: 0
    file: %pHours%
    name: pHours
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    header: [1]
    indexColumns: [1, 2, 3]
    type: par

- CSVReader:
    trace: 0
    file: %pZoneIndex%
    name: pZoneIndex
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: %pTechDataExcel%
    name: pTechDataExcel
    indexSubstitutions: {.nan: ""}
    header: [1]
    indexColumns: [1]
    type: par
    valueSubstitutions: {"NO": .nan, "YES": 1}  # drop "NO" and read "YES" as 1

- CSVReader:
    trace: 0
    file: %zcmapExcel%
    name: zcmapExcel
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
    file: %pEnergyEfficiencyFactor%
    name: pEnergyEfficiencyFactor
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    header: [1]
    indexColumns: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pScalars%
    name: pScalars
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: %pVREProfile%
    name: pVREProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
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
    file: %pCapexTrajectory%
    name: pCapexTrajectory
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par

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
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    name: pStorDataExcel
    indexColumns: [1, 2]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: %pFuelTypeCarbonContent%
    name: pFuelTypeCarbonContent
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

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

- CSVReader:
    trace: 0
    file: %pVREgenProfile%
    name: pVREgenProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par

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
    file: %pGenDataExcel%
    name: pGenDataExcel
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3,4]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: %pGenDataExcel%
    name: gmap
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1,2,3,4]
    type: set


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
    file: %pFuelData%
    name: pFuelData
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
    file: %pH2DataExcel%
    name: hh
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set


- CSVReader:
    file: %sTopology%
    name: sTopology
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    type: set

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
