# Model Description

The Electricity Planning Model (EPM) is a long-term planning framework that optimizes generation, storage, and transmission portfolios while simulating hourly operations on a set of representative time slices. The formulation is implemented in `epm/base.gms`; this document provides a narrative description aligned with the current code base and links to the corresponding equations in [Model Formulation](model_formulation.md).

## Sets and Indices

EPM relies on two families of sets:

- **Temporal resolution**
  - `y`: years in the planning horizon (`sStartYear(y)` identifies the first one).
  - `q`: representative seasons/quarters.
  - `d`: representative days within each season.
  - `t`: intra-day time slices (generally hours); aliases capture special positions such as `sFirstHour(t)`.
- **Power-system structure**
  - `g`: generating resources with subsets for existing (`eg`), new build candidates (`ng`), storage (`st`), VRE (`vre`), CSP (`cs`), etc.
  - `z`: internal zones; `c` aggregates zones by country; `zext` holds external trading regions.
  - `f`: fuels mapping into generators via `gfmap(g,f)`.
  - `sTopology(z,z2)`: neighbouring zone pairs for the transmission network.
  - Additional mapping sets (`gzmap`, `gsmap`, `zfmap`, etc.) define which tuples are meaningful.

This structure mirrors the declarations at the top of `base.gms`, ensuring that documentation and implementation remain consistent.

## Decision Variables

Key variables include:

- `vPwrOut(g,f,q,d,t,y)`: dispatched power by generator, fuel, and time slice.
- `vCap(g,y)`, `vBuild(g,y)`, `vRetire(g,y)`: installed capacity and annual build/retirement decisions.
- `vFlow(z,z2,q,d,t,y)`: internal transmission flows; `vNewTransmissionLine(z,z2,y)` and `vBuildTransmissionLine(z,z2,y)` capture expansion choices.
- `vYearlyImportExternal`/`vYearlyExportExternal`: exchanges with external zones.
- `vStorage`, `vStorInj`, `vStorOut`, `vCapStor`: state variables for batteries and CSP storage.
- `vSpinningReserve`, `vUnmetSpinningReserve*`, `vUnmetPlanningReserve*`: reserve provision and slack variables.
- `vUSE`, `vSurplus`: unmet demand and surplus energy.
- Cost-reporting variables such as `vYearlyTotalCost`, `vYearlyFuelCost`, `vYearlyCarbonCost`, etc., which feed directly into the objective.

Integer variables (`vBuiltCapVar`, `vRetireCapVar`, `vBuildTransmissionLine`) enforce discrete investments when specified in the data.

## Objective Function

The optimization minimizes the discounted net present value of total system costs. The top-level expression and each cost component are detailed in [Objective Function and Cost Components](model_formulation.md#objective-function-and-cost-components). Cost categories cover:

- Annualized CAPEX for generation, storage, and transmission builds.
- Fixed and variable O&M (including fuel costs).
- Reserve holding costs.
- Penalties for unserved energy, unmet reserve margins, curtailment, and surplus energy.
- Carbon prices and optional CO₂ backstop payments.
- Net external trade costs.

Discount factors (`pRR`) and scenario-specific year weights (`pWeightYear`) translate annual totals into present value.

## Demand and Supply Balance

Zonal demand (after energy-efficiency adjustments) must equal supply in every time slice (`eDemSupply`). Supply is defined as the combination of local generation, net internal and external exchanges, storage actions, and the slack variables for unmet demand and surplus (`eDefineSupply`). The explicit expressions are available under [Demand and Supply Balance](model_formulation.md#demand-and-supply-balance).

## Capacity Evolution

Installed capacity evolves through recursive equations that add builds and subtract retirements for each asset (`eInitialCapacity`, `eCapacityEvolutionExist`, `eCapacityEvolutionNew`). Discrete unit sizes, commissioning schedules, and predefined life limits are controlled through integer variables and associated equations (`eBuiltCap`, `eRetireCap`, `eInitialBuildLimit`). Storage energy capacity and CSP thermal fields use analogous recursions (`eCapStor*`, `eCapTherm*`). See [Capacity Evolution](model_formulation.md#capacity-evolution).

## Generator Operations

Operational constraints ensure realistic dispatch:

- **Joint dispatch/reserve envelope** (`eJointResCap`): total output plus spinning reserve cannot exceed installed capacity times the overload factor.
- **Seasonal availability** (`eMaxCF`): enforces maintenance and resource availability via quarterly energy limits.
- **Minimum output** (`eMinGen`): optional floor on dispatch as a share of capacity.
- **Ramp limits** (`eRampUpLimit`, `eRampDnLimit`): cap inter-hour changes when enabled.
- **Fuel accounting** (`eFuel`, `eFuelLimit`) links dispatch to annual fuel consumption with optional caps.

Each relation is summarised under [Generator Operating Limits](model_formulation.md#generator-operating-limits) and [Ramp Rate Constraints](model_formulation.md#ramp-rate-constraints).

## Renewable Generation and Curtailment

Variable renewable generators follow exogenous profiles with explicit curtailment tracking (`eVREProfile`). Curtailment penalties feed into yearly cost reporting. Details appear in [Generator Operating Limits](model_formulation.md#renewable-generation-with-curtailment).

## Reserve Requirements

Spinning and planning reserve requirements are enforced at both country and system level when activated in the configuration:

- Generator-specific reserve limits (`eSpinningReserveLim`, `eSpinningReserveLimVRE`).
- Country and system spinning reserve requirements with VRE forecast-error adders (`eSpinningReserveReqCountry`, `eSpinningReserveReqSystem`).
- Planning reserve margins based on capacity credits (`ePlanningReserveReqCountry`, `ePlanningReserveReqSystem`).

Slack variables accumulate the penalty costs used in the objective. The equations are described in [Reserve Requirements](model_formulation.md#reserve-requirements).

## Transmission and External Trade

Internal transfers are limited by existing capacities plus optional expansion builds (`eTransferCapacityLimit`). Investment symmetry and accumulation are handled by `eCumulativeTransferExpansion` and `eSymmetricTransferBuild`. External exchanges observe annual and hourly share caps and point-to-point limits (`eMaxAnnualImportShareEnergy`, `eMaxAnnualExportShareEnergy`, `eMaxHourlyImportShareEnergy`, `eMaxHourlyExportShareEnergy`, `eExternalImportLimit`, `eExternalExportLimit`). Cost accounting equations record import and export expenditures. See [Transmission and Trade Constraints](model_formulation.md#transmission-and-trade-constraints).

## Storage and CSP Representation

Grid-scale storage units follow a standard state-of-charge formulation with efficiency losses, charging/discharging limits, and reserve contribution checks (`eSOCUpperBound`, `eChargeCapacityLimit`, `eNetChargeBalance`, `eStateOfChargeUpdate`, `eSOCSupportsReserve`). PV-coupled storage inherits charging caps from the associated PV profile. CSP technologies extend the storage equations with thermal balance, charging limits, and separate capacity tracking for the solar field and storage blocks. Refer to [Storage Operations](model_formulation.md#storage-operations) and [CSP Extensions](model_formulation.md#csp-extensions).

## Policy and Investment Constraints

Optional constraints activated via configuration flags include:

- Renewable energy share targets (`eMinGenRE`).
- Total capital investment caps (`fApplyCapitalConstraint`).
- Fuel availability limits (`eFuelLimit`).
- Minimum import requirements (`eMinImportRequirement`).
- Inclusion of interconnection capacity in reserve assessments (`fCountIntercoForReserves`).

The effect of each toggle is visible in the corresponding equation block inside `base.gms`.

## Using This Document

Each section above references the exact equation names from `base.gms` and links to the mathematical expressions in [Model Formulation](model_formulation.md). When extending the model or validating country data, consult these references to ensure the documentation and implementation remain aligned.

