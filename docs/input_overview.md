# Input Overview

## Description

The following table lists the key input files used in the EPM model, along with a brief description and their structural dimensions. These files define model parameters, energy demand, supply data, constraints, and trade settings.

> In addition, we provide a [template](dwld/TemplateDataCollection.xlsx) to document the sources for each dataset used in EPM. This ensures clear traceability and consistency in the presentation of information across all inputs.

| File Name                          | Description                           | Dimensions       |
| ---------------------------------- | ------------------------------------- | ---------------- |
| **Configuration Files**            |                                       |                  |
| `pSettings.csv`                    | Global model parameters and settings. | 5 sections       |
| `y.csv`                            | List of model years.                  | 1 column         |
| `zcmap.csv`                        | Maps zones to countries.              | 2 columns        |
| **Resources**                      |                                       |                  |
| `ftfindex.csv`                     | Fuel-to-fuel indices.                 | 3 columns        |
| `pTechData.csv`                    | Technology parameters.                | Multiple columns |
| `pFuelCarbonContent.csv`           | Carbon content by fuel type.          | 2 columns        |
| **Load Data**                      |                                       |                  |
| `pDemandProfile.csv`               | Demand profiles over time.            | Multiple columns |
| `pDemandForecast.csv`              | Forecasted demand.                    | Multiple columns |
| `pDemandData.csv`                  | Historical demand data.               | Multiple columns |
| `pEnergyEfficiencyFactor.csv`      | Energy efficiency factors.            | 2 columns        |
| `sRelevants.csv`                   | Relevant scenarios.                   | 1 column         |
| **Supply Data**                    |                                       |                  |
| `pGenDataInputCustom.csv`          | Custom generation data.               | Multiple columns |
| `pGenDataInputDefault.csv`         | Default generation data.              | Multiple columns |
| `pAvailabilityCustom.csv`          | Custom generation availability.       | Multiple columns |
| `pAvailabilityDefault.csv`         | Default generation availability.      | Multiple columns |
| `pVREgenProfile.csv`               | Renewable energy generation profiles. | Multiple columns |
| `pVREProfile.csv`                  | Renewable variability profiles.       | Multiple columns |
| `pCapexTrajectoriesCustom.csv`     | Custom CAPEX trajectories.            | Multiple columns |
| `pCapexTrajectoriesDefault.csv`    | Default CAPEX trajectories.           | Multiple columns |
| `pFuelPrice.csv`                   | Fuel prices (historical/forecasted).  | Multiple columns |
| **Constraints**                    |                                       |                  |
| `pCarbonPrice.csv`                 | Carbon pricing.                       | 2 columns        |
| `pEmissionsCountry.csv`            | Emissions per country.                | Multiple columns |
| `pEmissionsTotal.csv`              | Total emission limits.                | Multiple columns |
| `pMaxFuelLimit.csv`                | Maximum fuel usage limits.            | Multiple columns |
| **Reserve**                        |                                       |                  |
| `pPlanningReserveMargin.csv`       | Reserve margin requirements.          | 2 columns        |
| `pSpinningReserveReqCountry.csv`   | Spinning reserve per country.         | Multiple columns |
| `pSpinningReserveReqTotal.csv`     | Total system reserve requirements.    | Multiple columns |
| **Trade**                          |                                       |                  |
| `pExtTransferLimit.csv`            | External transfer limits.             | Multiple columns |
| `pMaxAnnualExternalTradeShare.csv` | Max energy exchange share.            | Multiple columns |
| `pNewTransmission.csv`             | New transmission projects.            | Multiple columns |
| `pTradePrice.csv`                  | Energy trade prices.                  | Multiple columns |
| `pTransferLimit.csv`               | Transfer capacity limits.             | Multiple columns |
| `zext.csv`                         | External zone definitions.            | 1 column         |
| `pLossFactorInternal.csv`          | Transmission loss factors.            | 2 columns        |
| **Hydrogen (H2)**                  | No specific files listed.             |                  |

## Main changes

This update restructures how input data is provided to improve clarity and flexibility. All input data is now organized in CSV files, replacing the previous single Excel file. While the content is mostly unchanged, some usability improvements were introduced for modelers.

### Default data

For certain input types (e.g. generation data, seasonal availability, capex), the model now distinguishes between:

- `xxDefault.csv`: provides default values by zone, technology, and fuel.
- `xxCustom.csv`: allows for custom entries that override defaults when needed.

How it works:

| Case                      | What happens                      |
| ------------------------- | --------------------------------- |
| Field missing in `Custom` | Value from `Default` is used      |
| Field present in `Custom` | Value **overrides** the `Default` |

For example:
If `pAvailabilityDefault.csv` sets a default availability for coal in South Africa, and you do not specify a value in `pAvailabilityCustom.csv`, the default will be used. If you do specify one, it takes priority.

⚠️ Important Notes

- Required fields: For `pAvailabilityDefault.csv` and `pCapexTrajectoriesCustom.csv`, missing values for any combination of zone, tech, and fuel may cause errors. This is described in the Troubleshooting section below.
- Optional fields: For generation data `pGenDataInputCustom.csv` (e.g., Capex, vOM), missing values default to zero in GAMS, with no error.

✅ Recommended Workflow

- Copy the default files from the provided test folders (data_test, data_test_region).
- Adapt them to fit your model's zones, technologies, and fuels.
- Ensure all zone/tech/fuel combinations you need are covered in at least one file (Default or Custom).
