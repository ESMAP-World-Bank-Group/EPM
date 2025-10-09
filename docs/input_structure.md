# Input Folder Structure

All input files must be placed inside the `epm/input` folder.

The input structure consists of:

- A folder containing all necessary input `.csv` files for the model. This folder is specified using the `--folder_input` argument when running the model with Python.
- A main configuration file (`config.csv`) that defines which input files correspond to each model parameter. This file is specified using the `--config` argument.

Below is an example of the structure based on the `data_test_region` folder.

To run the model with Python, use the following command:

```bash
python epm.py --folder_input data_test_region --config input/data_test_region/config.csv
```

This folder contains all the input files required to run EPM for the Southern Africa Power Pool (SAPP) region.

```plaintext
data_test_region/
│
├── config.csv                          # Main configuration file for baseline run
│
├── constraint/                         # Policy and emissions constraints
│   ├── pCarbonPrice.csv
│   ├── pEmissionsCountry.csv
│   ├── pEmissionsTotal.csv
│   └── pMaxFuellimit.csv
│
├── h2/                                 # Hydrogen-specific parameters
│   ├── pAvailabilityH2.csv
│   ├── pCapexTrajectoryH2.csv
│   ├── pExternalH2.csv
│   ├── pFuelDataH2.csv
│   └── pH2DataExcel.csv
│
├── load/                               # Load and demand data
│   ├── pDemandData.csv
│   ├── pDemandForecast.csv
│   ├── pDemandProfile.csv
│   ├── pEnergyEfficiencyFactor.csv
│   └── srelevant.csv
│
├── reserve/                            # Reserve requirements
│   ├── pPlanningReserveMargin.csv
│   ├── pSpinningReserveReqCountry.csv
│   └── pSpinningReserveReqSystem.csv
│
├── resources/                          # General model inputs
│   ├── ftfindex.csv
│   ├── pFuelCarbonContent.csv
│   └── pTechData.csv
│
├── supply/                             # Generation and availability data
│   ├── pAvailability.csv
│   ├── pAvailabilityCustom.csv
│   ├── pAvailabilityCustomUpdated.csv
│   ├── pAvailabilityDefault.csv
│   ├── pAvailabilityDefaultCCDR.csv
│   ├── pAvailabilityDefaultEmpty.csv
│   ├── pCapexTrajectories.csv
│   ├── pCapexTrajectoriesCustom.csv
│   ├── pCapexTrajectoriesDefault.csv
│   ├── pCapexTrajectoriesDefaultCCDR.csv
│   ├── pCapexTrajectoriesDefaultEmpty.csv
│   ├── pCSPData.csv
│   ├── pFuelPrice.csv
│   ├── pFuelPriceUpdated.csv
│   ├── pGenDataExcel.csv
│   ├── pGenDataExcelCustom.csv
│   ├── pGenDataExcelDefault.csv
│   ├── pGenDataExcelDefaultEmpty.csv
│   ├── pGenDataExcelDefaultRampHydro.csv
│   ├── pStorDataExcel.csv
│   ├── pVREgenProfile.csv
│   ├── pVREgenProfileFullROR.csv
│   ├── pVREgenProfileROR.csv
│   └── pVREProfile.csv
│
├── trade/                              # Cross-border trade and transmission
│   ├── pExtTransferLimit.csv
│   ├── pLossFactorInternal.csv
│   ├── pMaxAnnualExternalTradeShare.csv
│   ├── pMaxPriceImportShare.csv
│   ├── pMinImport.csv
│   ├── pNewTransmission.csv
│   ├── pTradePrice.csv
│   ├── pTransferLimit.csv
│   ├── pTransferLimitCurrent.csv
│   └── pTransferLimitRetradeUpdate.csv
│
├── scenarios_sapp.csv                  # Main scenario definition file
├── scenarios_sapp_small.csv            # Alternative scenario file (e.g., limited scope)
├── mapGG.csv                           # Country/zone mapping
├── pHours.csv                          # Time slice definitions
├── pSettings.csv                       # Simulation settings
├── y.csv                               # Year list (basic)
├── ydetailed.csv                       # Year list (detailed)
└── zcmap.csv                           # Zone-country mapping
```

---

### Example: Difference Between Scenario Files

- `scenarios_sapp.csv`: Full scenario set with detailed assumptions for demand growth, new capacity, and trade policies.
- `scenarios_sapp_small.csv`: Minimal variant for fast testing (e.g., fewer years or countries, no new builds).
