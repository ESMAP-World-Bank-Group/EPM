$set XLSXMAXROWS 1048576

$if not set READER $set READER CONNECT_EXCEL

$log ### READER = %READER%

*ff remove gdx for testing
$call rm %GDX_INPUT%.gdx

$call test %GDX_INPUT%.gdx -nt "%XLS_INPUT%"
$ifThen.errorLevel errorlevel 1

$ifThenI.READER "%READER%" == "GDXXRW"

$onecho > gdxxrw.in
par=ftfindex                rdim=2 cdim=0 rng=FuelTechnologies!A3
par=pHours                  rdim=3 cdim=1 rng=Duration!A6
par=pZoneIndex              rdim=1 cdim=0 rng=ZoneData!E7:F200
par=pGenDataExcel           rdim=1 cdim=1 rng=GenData!A6
par=pTechDataExcel          rdim=1 cdim=1 rng=FuelTechnologies!B29:F50
set=zcmapExcel              rdim=2 cdim=0 rng=ZoneData!U7                           Values=NoData
set=y                       rdim=1 cdim=0 rng=LoadDefinition!A6:A%XLSXMAXROWS%      Values=NoData
set=sTopology               rdim=1 cdim=1 rng=Topology!A6
set=peak                    rdim=1 cdim=0 rng=LoadDefinition!H6:H%XLSXMAXROWS% Values=NoData
set=Relevant                rdim=1 cdim=0 rng=LoadDefinition!E6:E%XLSXMAXROWS% Values=NoData
set=zext                    rdim=1 cdim=0 rng=ZoneData!G7:G60
par=pReserveMargin          rdim=1 cdim=0 rng=Reserve!A6
par=pEnergyEfficiencyFactor rdim=1 cdim=1 rng=EnergyEfficiency!A5
par=pScalars                rdim=1 cdim=0 rng=Settings1!B3:C70
par=pAvailability           rdim=1 cdim=1 rng=GenAvailability!A6
par=pAvailabilityDaily      rdim=3 cdim=0 rng=GenAvailabilityDaily!A2
par=pVREProfile             rdim=4 cdim=1 rng=REProfile!A6
par=pLossFactor             rdim=2 cdim=1 rng=LossFactor!A5
par=pTransferLimit          rdim=3 cdim=1 rng=TransferLimit!A5
par=pExtTransferLimit       rdim=4 cdim=1 rng=ExternalLimits!A5
par=pCarbonPrice            rdim=1 cdim=0 rng=EmissionFactors!A3:B24
par=pDemandData             rdim=4 cdim=1 rng=Demand!A6
par=pDemandForecast         rdim=2 cdim=1 rng=Demand_Forecast!A6
par=pDemandProfile          rdim=3 cdim=1 rng=DemandProfile!A6
par=pMaxExchangeShare       rdim=1 cdim=1 rng=ExchangeShare!A7
par=pTradePrice             rdim=4 cdim=1 rng=TradePrices!A6
par=pNewTransmission        rdim=2 cdim=1 rng=ZoneData!J6:Q107
par=pCapexTrajectory        rdim=1 cdim=1 rng=CapexTrajectories!A5
par=pCSPData                rdim=2 cdim=1 rng=CSP!A6                                IgnoreColumns=E
par=pStorDataExcel          rdim=2 cdim=1 rng=Storage!A6
par=pFuelTypeCarbonContent  rdim=1 cdim=0 rng=EmissionFactors!J3

par=pReserveReqLoc          rdim=1 cdim=1 rng=SpinReserve!F5
par=pReserveReqSys          rdim=1 cdim=0 rng=SpinReserve!A6
par=pEmissionsCountry       rdim=1 cdim=1 rng=Emissions!A17
par=pEmissionsTotal         rdim=0 cdim=1 rng=Emissions!A5
par=pFuelPrice              rdim=2 cdim=1 rng=FuelPrices!A6
par=pMaxFuellimit           rdim=2 cdim=1 rng=FuelLimit!A6
par=pVREgenProfile          rdim=4 cdim=1 rng=REgenProfile!A6

set=MapGG                   rdim=1 cdim=1 rng=Retrofit!A6:ZW1000
*******************************************************************************************
set=hh                      rdim=1 cdim=0 rng=H2Data!A7:A%XLSXMAXROWS%      Values=NoData
par=pH2DataExcel            rdim=1 cdim=1 rng=H2Data!A6
par=pAvailabilityH2         rdim=1 cdim=1 rng=H2Availability!A6
par=pFuelData               rdim=1 cdim=0 rng=FuelTechnologies!J3:K44
par=pCapexTrajectoryH2      rdim=1 cdim=1 rng=CapexTrajectoriesH2!A5
par=pExternalH2             rDim=2 cdim=1 rng=ExternalH2Demand!A5
$offecho
$call.checkErrorLevel gdxxrw "%XLS_INPUT%" @gdxxrw.in
$call rm gdxxrw.in

$elseIfI.READER %READER% == CONNECT_EXCEL
$log ### Reading from %XLS_INPUT% using Connect and Excel Input

$onEmbeddedCode Connect:
- ExcelReader:
    file: %XLS_INPUT%
    symbols:
      - name: ftfindex
        range: FuelTechnologies!A3
        rowDimension: 2
        columnDimension: 0
      - name: pHours
        range: Duration!A6
        rowDimension: 3
        columnDimension: 1
      - name: pZoneIndex
        range: ZoneData!E7:F200
        rowDimension: 1
        columnDimension: 0
      - name: pGenDataExcel
        range: GenData!A6
        rowDimension: 1
        columnDimension: 1
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
      - name: pReserveMargin
        range: Reserve!A6
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
      - name: pAvailabilityDaily
        range: GenAvailabilityDaily!A2
        rowDimension: 3
        columnDimension: 0
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
      - name: pExtTransferLimit
        range: ExternalLimits!A5
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
        indexSubstitutions: {.nan: ""}  # keep empty labels
      - name: pFuelTypeCarbonContent
        range: EmissionFactors!J3
        rowDimension: 1
        columnDimension: 0
      - name: pReserveReqLoc
        range: SpinReserve!F5
        rowDimension: 1
        columnDimension: 1
      - name: pReserveReqSys
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
      - name: MapGG
        range: Retrofit!A6:ZW1000
        rowDimension: 1
        columnDimension: 1
        type: set
        valueSubstitutions: {1: ""}  # Read 1 as empty string
      - name: hh
        range: H2Data!A7:A%XLSXMAXROWS%
        rowDimension: 1
        columnDimension: 0
        type: set
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
        valueSubstitutions: {0: .nan}  # drop zeroes
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
    file: input/pDemandData.csv
    name: pDemandData
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par

- CSVReader:
    trace: 0
    file: input/pExternalH2.csv
    name: pExternalH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan} 
    indexColumns: [1, 2]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/relevant.csv
    name: relevant
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan} 
    indexColumns: [1]
    type: set
    
- CSVReader:
    trace: 0
    file: input/peak.csv
    name: peak
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan} 
    type: set
    indexColumns: [1]
    
- CSVReader:
    file: input/pFuelPrice.csv
    name: pFuelPrice
    indexSubstitutions: {.nan: ""}
    header: [1]
    indexColumns: [1, 2]
    type: par
    
- CSVReader:
    file: input/ftfindex.csv
    name: ftfindex
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan} 
    indexColumns: [1, 2]
    valueColumns: [3]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pHours.csv
    name: pHours
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan} 
    header: [1]
    indexColumns: [1, 2, 3]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pZoneIndex.csv
    name: pZoneIndex
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan} 
    indexColumns: [1]
    valueColumns: [2]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pTechDataExcel.csv
    name: pTechDataExcel
    indexSubstitutions: {.nan: ""}
    header: [1]
    indexColumns: [1]
    type: par
    valueSubstitutions: {"NO": .nan, "YES": 1}  # drop "NO" and read "YES" as 1
    
- CSVReader:
    trace: 0
    file: input/zcmapExcel.csv
    name: zcmapExcel
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan} 
    indexColumns: [1, 2]
    type: set
    
- CSVReader:
    trace: 0
    file: input/y.csv
    name: y
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan} 
    indexColumns: [1]
    type: set
    
- CSVReader:
    trace: 0
    file: input/pEnergyEfficiencyFactor.csv
    name: pEnergyEfficiencyFactor
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan} 
    header: [1]
    indexColumns: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pScalars.csv
    name: pScalars
    indexSubstitutions: {" IncludeH2": "IncludeH2"}
    valueSubstitutions: {0: .nan} 
    indexColumns: [1]
    valueColumns: [2]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pAvailability.csv
    name: pAvailability
    indexSubstitutions: {.nan: ""}
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pAvailabilityDaily.csv
    name: pAvailabilityDaily
    indexSubstitutions: {.nan: ""}
    indexColumns: [1,2,3]
    valueColumns: [4]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pVREProfile.csv
    name: pVREProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pLossFactor.csv
    name: pLossFactor
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pTransferLimit.csv
    name: pTransferLimit
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pCarbonPrice.csv
    name: pCarbonPrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pDemandForecast.csv
    name: pDemandForecast
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pDemandProfile.csv
    name: pDemandProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pTradePrice.csv
    name: pTradePrice
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pNewTransmission.csv
    name: pNewTransmission
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pCapexTrajectory.csv
    name: pCapexTrajectory
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pCSPData.csv
    name: pCSPData
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pStorDataExcel.csv
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan} 
    name: pStorDataExcel
    indexColumns: [1, 2]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pFuelTypeCarbonContent.csv
    name: pFuelTypeCarbonContent
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pReserveMargin.csv
    name: pReserveMargin
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: input/pReserveReqLoc.csv
    name: pReserveReqLoc
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pReserveReqSys.csv
    name: pReserveReqSys
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par


- CSVReader:
    trace: 0
    file: input/pEmissionsTotal.csv
    name: pEmissionsTotal
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pEmissionsCountry.csv
    name: pEmissionsCountry
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan, np.nan: 0}
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pMaxFuellimit.csv
    name: pMaxFuellimit
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pVREgenProfile.csv
    name: pVREgenProfile
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1, 2, 3, 4]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pH2DataExcel.csv
    name: pH2DataExcel
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pGenDataExcel_storage.csv
    name: pGenDataExcel
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pAvailabilityH2.csv
    name: pAvailabilityH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pFuelData.csv
    name: pFuelData
    valueSubstitutions: {0: .nan} 
    indexColumns: [1]
    valueColumns: [2]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pCapexTrajectoryH2.csv
    name: pCapexTrajectoryH2
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par
    
    
- CSVReader:
    trace: 0
    file: input/pH2DataExcel.csv
    name: hh
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set
    
- CSVReader:
    file: input/sTopology.csv
    name: sTopology
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {1: "", "": .nan}
    indexColumns: [1, 2]
    type: set
    
- CSVReader:
    trace: 0
    file: input/zext.csv
    name: zext
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    type: set

- CSVReader:
    trace: 0
    file: input/pMaxExchangeShare.csv
    name: pMaxExchangeShare
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/pExtTransferLimit.csv
    name: pExtTransferLimit
    valueSubstitutions: {0: .nan} 
    indexColumns: [1,2,3,4]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: input/MapGG.csv
    name: MapGG
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {1: ""}
    indexColumns: [1,2]
    type: set


- GDXWriter:
    file: %GDX_INPUT%.gdx
    symbols: all
$offEmbeddedCode


$elseIfI.READER %READER% == CONNECT_CSV_PYTHON
$log ### reading using Connect and CSV Input with Python


$onEmbeddedCode Connect:

- CSVReader:
    trace: 0
    file: %pDemandData%
    name: pDemandData
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
    indexSubstitutions: {" IncludeH2": "IncludeH2"}
    valueSubstitutions: {0: .nan} 
    indexColumns: [1]
    valueColumns: [2]
    type: par
    
- CSVReader:
    trace: 0
    file: %pAvailability%
    name: pAvailability
    indexSubstitutions: {.nan: ""}
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: %pAvailabilityDaily%
    name: pAvailabilityDaily
    indexSubstitutions: {.nan: ""}
    indexColumns: [1,2,3]
    valueColumns: [4]
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
    file: %pReserveMargin%
    name: pReserveMargin
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
    type: par

- CSVReader:
    trace: 0
    file: %pReserveReqLoc%
    name: pReserveReqLoc
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par
    
- CSVReader:
    trace: 0
    file: %pReserveReqSys%
    name: pReserveReqSys
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    valueColumns: [2]
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
    file: %pEmissionsCountry%
    name: pEmissionsCountry
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan, np.nan: 0}
    indexColumns: [1]
    header: [1]
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
    file: %pFuelData%
    name: pFuelData
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
    valueSubstitutions: {1: "", "": .nan}
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
    file: %pMaxExchangeShare%
    name: pMaxExchangeShare
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    indexColumns: [1]
    header: [1]
    type: par
    
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
    file: %MapGG%
    name: MapGG
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {1: ""}
    indexColumns: [1,2]
    type: set


- GDXWriter:
    file: %GDX_INPUT%.gdx
    symbols: all
$offEmbeddedCode


$else.READER
$abort 'No valid READER specified. Allowed are GDXXRW and CONNECT.'
$endif.READER
$endIf.errorLevel

$setNames "%XLS_INPUT%" fp GDX_INPUT fe
