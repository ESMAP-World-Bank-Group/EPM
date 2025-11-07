# Model Formulation

This document presents the core mathematical relations implemented in `epm/base.gms`. The notation mirrors the GAMS model to keep the documentation and the code synchronized.

## Notation

| Symbol | Description |
|--------|-------------|
| `y ∈ Y` | Planning years (`sStartYear(y)` marks the first model year). |
| `q ∈ Q` | Representative seasons or quarters. |
| `d ∈ D` | Representative days within each season. |
| `t ∈ T` | Intra-day time slices (typically hours). |
| `z ∈ Z` | Internal zones; `zext` denotes external trading zones. |
| `c ∈ C` | Countries (used for policy and reporting aggregation). |
| `g ∈ G` | Generating assets. Subsets `eg`, `ng`, `st`, `cs`, `vre`, `re`, `RampRate`, `VRE_noROR`, etc. capture technology groupings. |
| `f ∈ F` | Fuels; `gfmap(g,f)` links generators to fuels. |
| `gzmap(g,z)` | Generator-to-zone mapping. |
| `sTopology(z,z2)` | Internal transmission pairs; symmetric in `z` and `z2`. |
| `pHours(q,d,t)` | Weight of each time slice (hours represented). |

Unless otherwise noted, all summations are over the valid combinations defined by the mapping sets above. Variables beginning with `v` are decision or reporting variables; parameters beginning with `p`, `f`, or `s` are inputs or switches.

## Objective Function and Cost Components

The system-wide net present value minimization is written as

```math
vNPVCost = \sum_{y} pRR_{y} \cdot pWeightYear_{y} \cdot \Big(
  \sum_{c} vYearlyTotalCost_{c,y}
  + vYearlyUnmetPlanningReserveCostSystem_{y}
  + vYearlyUnmetSpinningReserveCostSystem_{y}
  + vYearlyCO2BackstopCostSystem_{y}
\Big).
```

The yearly country cost balance expands to

```math
\begin{aligned}
vYearlyTotalCost_{c,y} ={}& vYearlyUnmetReserveCostCountry_{c,y}
                         + vYearlyCO2BackstopCostCountry_{c,y} \\
&+ \sum_{z \in c} \Big(
      vAnnCapex_{z,y} + vYearlyFOMCost_{z,y}
    + vYearlyVariableCost_{z,y} + vYearlySpinningReserveCost_{z,y} \\
&\qquad\qquad\quad + vYearlyUSECost_{z,y} + vYearlyCarbonCost_{z,y}
    + vYearlyExternalTradeCost_{z,y} + vAnnualizedTransmissionCapex_{z,y} \\
&\qquad\qquad\quad + vYearlyCurtailmentCost_{z,y} + vYearlySurplus_{z,y}
  \Big).
\end{aligned}
```

Annual cost components are defined as follows:

- `vYearlyFOMCost`: fixed O&M costs for generators, storage, and CSP thermal fields.
- `vYearlyVariableCost = vYearlyFuelCost + vYearlyVOMCost`.
- `vYearlySpinningReserveCost`: reserve cost bids multiplied by reserve provision and time weights.
- `vYearlyUSECost`: value of lost load applied to unserved energy.
- `vYearlySurplus`: penalty on over-generation.
- `vYearlyCurtailmentCost`: penalty on curtailed VRE output.
- `vYearlyExternalTradeCost = vYearlyImportExternalCost - vYearlyExportExternalCost`.
- `vYearlyCarbonCost`: carbon price times the emissions implied by dispatched generation.
- `vAnnualizedTransmissionCapex`: annualized cost of new transmission builds associated with each zone.
- `vYearlyUnmetReserveCostCountry = vYearlyUnmetPlanningReserveCostCountry + vYearlyUnmetSpinningReserveCostCountry`.

Each of these relations is coded in `base.gms` as equations `eYearly*`.

## Demand and Supply Balance

The demand balance (`eDemSupply`) equates demand to the composite supply variable:

```math
pDemandData_{z,q,d,y,t} \cdot pEnergyEfficiencyFactor_{z,y} = vSupply_{z,q,d,t,y}.
```

`vSupply` is defined in `eDefineSupply`:

```math
\begin{aligned}
vSupply_{z,q,d,t,y} ={}&
  \sum_{g,f} vPwrOut_{g,f,q,d,t,y}
  - \sum_{z_2} vFlow_{z,z_2,q,d,t,y} \\
&+ \sum_{z_2} \bigl(1 - pLossFactorInternal_{z,z_2,y}\bigr) \cdot vFlow_{z_2,z,q,d,t,y} \\
&- \sum_{st} vStorInj_{st,q,d,t,y} \cdot \mathbb{1}_{\text{fEnableStorage}} \\
&+ \sum_{z_{\text{ext}}} vYearlyImportExternal_{z,z_{\text{ext}},q,d,t,y} \\
&- \sum_{z_{\text{ext}}} vYearlyExportExternal_{z,z_{\text{ext}},q,d,t,y} \\
&+ vUSE_{z,q,d,t,y} - vSurplus_{z,q,d,t,y}.
\end{aligned}
```

See [Demand and Supply Balance](#demand-and-supply-balance) for a detailed discussion.

## Capacity Evolution

Capacity trajectories for each generator are tracked through:

```math
vCap_{g,y_0} = pGenData_{g,\text{Capacity}} + vBuild_{g,y_0} - vRetire_{g,y_0}
```
for the first modelled year `y₀`, when the unit exists and starts before the study horizon.

```math
vCap_{eg,y} = vCap_{eg,y-1} + vBuild_{eg,y} - vRetire_{eg,y}
```
for existing assets (`eCapacityEvolutionExist`), and

```math
vCap_{ng,y} = vCap_{ng,y-1} + vBuild_{ng,y}
```
for new entrants without retirement (`eCapacityEvolutionNew`). Discrete build/retirement decisions are enforced through

```math
vBuild_{ng,y} = pGenData_{ng,\text{UnitSize}} \cdot vBuiltCapVar_{ng,y},
```

```math
vRetire_{eg,y} = pGenData_{eg,\text{UnitSize}} \cdot vRetireCapVar_{eg,y},
```

with additional bounds on cumulative builds and commissioning years (`eInitialBuildLimit`, `eBuiltCap`, `eRetireCap`). Storage and CSP thermal capacity follow analogous recursions (`eCapStor*`, `eCapTherm*`).

## Generator Operating Limits

### Dispatch and Reserve Envelope

The joint dispatch and reserve constraint (`eJointResCap`) caps the combined output and spinning reserve at the installed capacity with optional overloading:

```math
\sum_f vPwrOut_{g,f,q,d,t,y} + vSpinningReserve_{g,q,d,t,y}
  \le (1 + pGenData_{g,\text{Overloadfactor}}) \cdot vCap_{g,y}.
```

### Availability and Minimum Output

Seasonal availability (`eMaxCF`) limits the energy produced within each quarter:

```math
\sum_{f,d,t} vPwrOut_{g,f,q,d,t,y} \cdot pHours_{q,d,t}
 \le pAvailability_{g,q} \cdot vCap_{g,y} \cdot \sum_{d,t} pHours_{q,d,t}.
```

Optional minimum loading requirements (`eMinGen`) enforce:

```math
\sum_f vPwrOut_{g,f,q,d,t,y} \ge pGenData_{g,\text{MinLimitShare}} \cdot vCap_{g,y},
```
whenever the flag `fApplyMinGenerationConstraint` is active.

### Renewable Generation with Curtailment

Variable renewable energy (VRE) generators follow exogenous profiles (`eVREProfile`):

```math
vPwrOut_{g,f,q,d,t,y} + vCurtailedVRE_{z,g,q,d,t,y}
  = pVREgenProfile_{g,q,d,t} \cdot vCap_{g,y}.
```

Curtailment is tracked explicitly and priced through `vYearlyCurtailmentCost`.

### Fuel Consumption

Fuel use is accounted for via `eFuel`:

```math
vFuel_{z,f,y} = \sum_{g,q,d,t} vPwrOut_{g,f,q,d,t,y} \cdot pHours_{q,d,t} \cdot pHeatRate_{g,f}.
```

Optional limits (`eFuelLimit`) apply upper bounds by country and fuel.

## Ramp Rate Constraints

When the ramp flag is enabled, inter-hour changes must satisfy:

```math
\sum_f vPwrOut_{g,f,q,d,t,y} - \sum_f vPwrOut_{g,f,q,d,t-1,y}
  \le vCap_{g,y} \cdot pGenData_{g,\text{RampUpRate}},
```

```math
\sum_f vPwrOut_{g,f,q,d,t-1,y} - \sum_f vPwrOut_{g,f,q,d,t,y}
  \le vCap_{g,y} \cdot pGenData_{g,\text{RampDnRate}},
```
(`eRampUpLimit`, `eRampDnLimit`), with analogous bounds for storage charging (`eChargeRampUpLimit`, `eChargeRampDownLimit`).

## Reserve Requirements

### Generator-Level Reserve Limits

Maximum reserve provision is enforced through:

```math
vSpinningReserve_{g,q,d,t,y}
  \le vCap_{g,y} \cdot pGenData_{g,\text{ResLimShare}},
```
(`eSpinningReserveLim`), with an additional VRE-specific scaling (`eSpinningReserveLimVRE`) that multiplies by the hourly profile.

### Country and System Requirements

Spinning reserve requirements are applied at the country and system level:

```math
\sum_{z \in c} \sum_{g \in z} vSpinningReserve_{g,q,d,t,y}
  + vUnmetSpinningReserveCountry_{c,q,d,t,y}
  = pSpinningReserveReqCountry_{c,y}
    + \sum_{g \in \text{VRE\_noROR}} \sum_{f} vPwrOut_{g,f,q,d,t,y} \cdot psVREForecastErrorPct,
```

```math
\sum_{g} vSpinningReserve_{g,q,d,t,y}
  + vUnmetSpinningReserveSystem_{q,d,t,y}
  = pSpinningReserveReqSystem_{y}
    + \sum_{g \in \text{VRE\_noROR}} \sum_{f} vPwrOut_{g,f,q,d,t,y} \cdot psVREForecastErrorPct.
```

Planning reserve margins relate capacity credits to peak demand:

```math
\sum_{z \in c} \sum_{g \in z} vCap_{g,y} \cdot pCapacityCredit_{g,y}
  + vUnmetPlanningReserveCountry_{c,y}
  \ge \bigl(1 + pPlanningReserveMargin_{c}\bigr) \cdot \max_{q,d,t}
      \sum_{z \in c} pDemandData_{z,q,d,y,t} \cdot pEnergyEfficiencyFactor_{z,y},
```

```math
\sum_{g} vCap_{g,y} \cdot pCapacityCredit_{g,y}
  + vUnmetPlanningReserveSystem_{y}
  \ge \bigl(1 + sReserveMarginPct\bigr) \cdot \max_{q,d,t}
      \sum_z pDemandData_{z,q,d,y,t} \cdot pEnergyEfficiencyFactor_{z,y}.
```

Interconnection capacity can contribute to these margins when the relevant flags are enabled.

## Transmission and Trade Constraints

Internal transfers are bounded by existing and expandable capacities:

```math
vFlow_{z,z_2,q,d,t,y}
  \le pTransferLimit_{z,z_2,q,y}
     + vNewTransmissionLine_{z,z_2,y}
       \cdot pNewTransmission_{z,z_2,\text{CapacityPerLine}}
       \cdot fAllowTransferExpansion.
```

Symmetry and cumulative build constraints (`eCumulativeTransferExpansion`, `eSymmetricTransferBuild`) keep both directions consistent.

External exchange limits include:

- Annual import/export share caps (`eMaxAnnualImportShareEnergy`, `eMaxAnnualExportShareEnergy`).
- Hourly import/export share caps (`eMaxHourlyImportShareEnergy`, `eMaxHourlyExportShareEnergy`).
- Point-to-point limits (`eExternalImportLimit`, `eExternalExportLimit`).

Trade costs accumulate through `vYearlyImportExternalCost` and `vYearlyExportExternalCost`, each linked to `pTradePrice`.

## Storage Operations

Storage devices (`st`) follow these relations:

- **State-of-charge bound** (`eSOCUpperBound`):
  ```math
  vStorage_{st,q,d,t,y} \le vCapStor_{st,y}.
  ```
- **Power and energy limits** (`eChargeCapacityLimit`, `eStorageCapMinConstraint`):
  ```math
  vStorInj_{st,q,d,t,y} \le vCap_{st,y}, \qquad
  vCapStor_{st,y} \ge vCap_{st,y}.
  ```
- **Net charge definition** (`eNetChargeBalance`):
  ```math
  vStorNet_{st,q,d,t,y} = \sum_f vPwrOut_{st,f,q,d,t,y} - vStorInj_{st,q,d,t,y}.
  ```
- **Inter-hour SOC dynamics** (`eStateOfChargeUpdate`, `eStateOfChargeInit`):
  ```math
  vStorage_{st,q,d,t,y}
    = pStorData_{st,\text{Efficiency}} \cdot vStorInj_{st,q,d,t,y}
      - \sum_f vPwrOut_{st,f,q,d,t,y}
      + vStorage_{st,q,d,t-1,y},
  ```
  with an initialization equation for the first hour of each day.
- **Reserve contribution** (`eSOCSupportsReserve`):
  ```math
  vSpinningReserve_{st,q,d,t,y} \le vStorage_{st,q,d,t,y}.
  ```

PV-coupled batteries (`stp`) are additionally capped by the paired PV capacity profile (`eChargeLimitWithPVProfile`).

## CSP Extensions

CSP units (`cs`) use the same storage state variables with technology-specific limits:

- **Storage capacity** (`eCSPStorageCapacityLimit`) and **charging bounds** (`eCSPStorageInjectionCap`).
- **Thermal balance** (`eCSPStorageInjectionLimit`, `eCSPThermalOutputLimit`) relating the solar field, storage, and turbine output.
- **Energy recursion** (`eCSPStorageEnergyBalance`, `eCSPStorageInitialBalance`).
- **Capacity evolution** for storage and thermal subsystems (`eCapStor*`, `eCapacityThermLimit`, `eCapThermBalance*`, `eBuildStorNew`, `eBuildThermNew`).

## Investment and Policy Constraints

The model includes optional constraints on:

- Total capital expenditure (`ssMaxCapitalInvestmentInvestment` via `fApplyCapitalConstraint`).
- Renewable energy share (`eMinGenRE`).
- Fuel limits (`eFuelLimit`).
- Minimum import requirements (`eMinImportRequirement`).

These are activated through the corresponding Boolean flags in the configuration files.

## Integer Decisions

Discrete transmission builds (`vBuildTransmissionLine`) and generator builds/retirements (`vBuiltCapVar`, `vRetireCapVar`) are defined as integer variables. Their bounding equations ensure consistency between the integer choice and the continuous capacity variables.

---

Each equation name referenced above matches the label in `base.gms`, enabling cross-navigation between documentation, configuration, and the implementation.
