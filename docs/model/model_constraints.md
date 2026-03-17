# Constraints

All constraints are grouped by topic. Each equation name matches the label in `base.gms`. For notation and variable definitions, see [Objective Function](model_objective.md) and [Reference](equations.md).

> **[Dispatch mode only]** — active when `fDispatchMode=1`.
> **[Optional]** — activated by a flag in `pSettings.csv`.

---

## Demand and Supply Balance

The demand balance (`eDemSupply`) equates demand to the composite supply variable:

$$
pDemandData_{z,q,d,y,t} \cdot pEnergyEfficiencyFactor_{z,y} = vSupply_{z,q,d,t,y}.
$$

`vSupply` decomposes generation, storage, flows, trade, and slack (`eDefineSupply`):

$$
\begin{aligned}
vSupply_{z,q,d,t,y} ={}&
  \sum_{g,f} vPwrOut_{g,f,q,d,t,y}
  - \sum_{z_2} vFlow_{z,z_2,q,d,t,y} \\
&+ \sum_{z_2} \bigl(1 - pLossFactorInternal_{z,z_2,y}\bigr) \cdot vFlow_{z_2,z,q,d,t,y} \\
&- \sum_{st \in z} vStorInj_{st,q,d,t,y} \cdot \mathbb{1}_{fEnableStorage} \\
&+ \sum_{z_{\text{ext}}} vYearlyImportExternal_{z,z_{\text{ext}},q,d,t,y}
 - \sum_{z_{\text{ext}}} vYearlyExportExternal_{z,z_{\text{ext}},q,d,t,y} \\
&+ vUSE_{z,q,d,t,y} - vSurplus_{z,q,d,t,y}.
\end{aligned}
$$

---

## Capacity Evolution

Initial capacity for the first model year, where the installed capacity term applies only to existing generators (`eg`) commissioned before the horizon (`eInitialCapacity`):

$$
vCap_{g,y_0} = pGenData_{g,\text{Capacity}} \cdot \mathbb{1}_{eg(g)} + vBuild_{g,y_0} - vRetire_{g,y_0}.
$$

For subsequent years, existing assets follow (`eCapacityEvolutionExist`):

$$
vCap_{eg,y} = vCap_{eg,y-1} + vBuild_{eg,y} - vRetire_{eg,y}.
$$

New entrants have no retirement (`eCapacityEvolutionNew`):

$$
vCap_{ng,y} = vCap_{ng,y-1} + vBuild_{ng,y}.
$$

Discrete build and retirement decisions are enforced through integer variables (`eBuiltCap`, `eRetireCap`):

$$
vBuild_{ng,y} = pGenData_{ng,\text{UnitSize}} \cdot vBuiltCapVar_{ng,y}, \qquad
vRetire_{eg,y} = pGenData_{eg,\text{UnitSize}} \cdot vRetireCapVar_{eg,y}.
$$

Total builds for existing-fleet generators commissioned after the model start are capped (`eInitialBuildLimit`):

$$
\sum_y vBuild_{eg,y} \le pGenData_{eg,\text{Capacity}}.
$$

Storage and CSP thermal capacity follow analogous recursions (`eCapStor*`, `eCapTherm*`).

---

## Generator Operating Limits

### Dispatch and Reserve Envelope

Combined output and spinning reserve cannot exceed installed capacity (`eJointResCap`):

$$
\sum_f vPwrOut_{g,f,q,d,t,y} + vSpinningReserve_{g,q,d,t,y}
  \le \bigl(1 + pGenData_{g,\text{Overloadfactor}}\bigr) \cdot vCap_{g,y}.
$$

### Seasonal Availability

Quarterly energy output is limited by the seasonal availability factor (`eMaxCF`). Note that `pAvailability` carries a year index to allow for degradation or scheduled outages:

$$
\sum_{f,d,t} vPwrOut_{g,f,q,d,t,y} \cdot pHours_{q,d,t}
  \le pAvailability_{g,y,q} \cdot vCap_{g,y} \cdot \sum_{d,t} pHours_{q,d,t}.
$$

### Minimum Output **[Optional — `fApplyMinGenShareAllHours`]**

(`eMinGen`):

$$
\sum_f vPwrOut_{g,f,q,d,t,y} \ge vCap_{g,y} \cdot pGenData_{g,\text{MinGenShareAllHours}}.
$$

### Renewable Generation with Curtailment

VRE output follows the hourly profile scaled by seasonal availability (`eVREProfile`). Curtailment is tracked explicitly:

$$
vPwrOut_{g,f,q,d,t,y} + vCurtailedVRE_{z,g,q,d,t,y}
  = pVREgenProfile_{g,q,d,t} \cdot pAvailability_{g,y,q} \cdot vCap_{g,y}.
$$

Curtailment is penalised through `vYearlyCurtailmentCost` in the objective function.

### Fuel Consumption

(`eFuel`):

$$
vFuel_{z,f,y} = \sum_{g \in z,\, q,d,t} vPwrOut_{g,f,q,d,t,y} \cdot pHours_{q,d,t} \cdot pHeatRate_{g,f}.
$$

An upper bound on annual fuel consumption by country and fuel applies via `eFuelLimit` when `fApplyFuelConstraint=1`.

### Generation Phase-out **[Optional — `fApplyGenerationPhaseout`]**

`eMaxGenerationByFuel` caps annual generation by zone, technology group, and fuel (in GWh):

$$
\frac{1}{1000} \sum_{g \in tech \cap z,\, q,d,t} vPwrOut_{g,f,q,d,t,y} \cdot pHours_{q,d,t}
  \le pMaxGenerationByFuel_{z,tech,f,y}.
$$

---

## Ramp Rate Constraints

### Standard mode

Inter-hour ramp limits apply within each representative day (`eRampUpLimit`, `eRampDnLimit`):

$$
\sum_f vPwrOut_{g,f,q,d,t,y} - \sum_f vPwrOut_{g,f,q,d,t-1,y}
  \le vCap_{g,y} \cdot pGenData_{g,\text{RampUpRate}},
$$

$$
\sum_f vPwrOut_{g,f,q,d,t-1,y} - \sum_f vPwrOut_{g,f,q,d,t,y}
  \le vCap_{g,y} \cdot pGenData_{g,\text{RampDnRate}}.
$$

Analogous bounds apply to storage charging (`eChargeRampUpLimit`, `eChargeRampDownLimit`).

### Dispatch mode **[Dispatch mode only]**

In dispatch mode, ramp limits operate over chronological slots $AT$ using `vPwrOutDispatch`, linked to `vPwrOut` via `eRampContinuity`. Startup and shutdown events provide additional headroom when unit commitment is active (`eDispatchRampUpLimit`, `eDispatchRampDownLimit`):

$$
vPwrOutDispatch_{g,AT,y} - vPwrOutDispatch_{g,AT-1,y}
  \le vCap_{g,y} \cdot pGenData_{g,\text{RampUpRate}}
    + isStartup_{g,AT,y} \cdot pGenData_{g,\text{Capacity}} \cdot \mathbb{1}_{fApplyMinGenCommitment},
$$

$$
vPwrOutDispatch_{g,AT-1,y} - vPwrOutDispatch_{g,AT,y}
  \le vCap_{g,y} \cdot pGenData_{g,\text{RampDnRate}}
    + isShutdown_{g,AT,y} \cdot pGenData_{g,\text{Capacity}} \cdot \mathbb{1}_{fApplyMinGenCommitment}.
$$

---

## Reserve Requirements

### Generator reserve limit

Maximum spinning reserve provision per generator (`eSpinningReserveLim`):

$$
vSpinningReserve_{g,q,d,t,y} \le vCap_{g,y} \cdot pGenData_{g,\text{ResLimShare}}.
$$

For VRE generators the limit is further scaled by the hourly profile (`eSpinningReserveLimVRE`):

$$
vSpinningReserve_{g,q,d,t,y} \le vCap_{g,y} \cdot pGenData_{g,\text{ResLimShare}} \cdot pVREgenProfile_{g,q,d,t}.
$$

### Spinning reserve requirement **[Optional]**

Country level (`eSpinningReserveReqCountry`, flag `fApplyCountrySpinReserveConstraint`):

$$
\sum_{z \in c,\, g \in z} vSpinningReserve_{g,q,d,t,y}
  + vUnmetSpinningReserveCountry_{c,q,d,t,y}
  \ge pSpinningReserveReqCountry_{c,y}
    + \sum_{g \in \text{VREnoROR},\, f} vPwrOut_{g,f,q,d,t,y} \cdot psVREForecastErrorPct.
$$

System level (`eSpinningReserveReqSystem`, flag `fApplySystemSpinReserveConstraint`):

$$
\sum_{g} vSpinningReserve_{g,q,d,t,y}
  + vUnmetSpinningReserveSystem_{q,d,t,y}
  \ge pSpinningReserveReqSystem_{y}
    + \sum_{g \in \text{VREnoROR},\, f} vPwrOut_{g,f,q,d,t,y} \cdot psVREForecastErrorPct.
$$

### Planning reserve margin **[Optional — `fApplyPlanningReserveConstraint`]**

Country level (`ePlanningReserveReqCountry`). When `fCountIntercoForReserves=1`, available interconnection capacity contributes to the margin:

$$
\sum_{z \in c,\, g \in z} vCap_{g,y} \cdot pCapacityCredit_{g,y}
  + vUnmetPlanningReserveCountry_{c,y}
  + IC_{c,y}
  \ge \bigl(1 + pPlanningReserveMargin_{c}\bigr) \cdot
      \max_{q,d,t} \sum_{z \in c} pDemandData_{z,q,d,y,t} \cdot pEnergyEfficiencyFactor_{z,y},
$$

where the interconnection term $IC_{c,y}$ (zero when `fCountIntercoForReserves=0`) is:

$$
IC_{c,y} = \sum_{z \in c,\, z_2 \notin c} \left(
  \frac{\sum_q pTransferLimit_{z_2,z,q,y}}{|Q|}
  + vNewTransmissionLine_{z_2,z,y} \cdot pNewTransmission_{z_2,z,\text{CapacityPerLine}} \cdot fAllowTransferExpansion
\right).
$$

System level (`ePlanningReserveReqSystem`):

$$
\sum_{g} vCap_{g,y} \cdot pCapacityCredit_{g,y}
  + vUnmetPlanningReserveSystem_{y}
  \ge \bigl(1 + sReserveMarginPct\bigr) \cdot
      \max_{q,d,t} \sum_z pDemandData_{z,q,d,y,t} \cdot pEnergyEfficiencyFactor_{z,y}.
$$

---

## Transmission and Trade

Internal transfer capacity (`eTransferCapacityLimit`):

$$
vFlow_{z,z_2,q,d,t,y}
  \le pTransferLimit_{z,z_2,q,y}
     + vNewTransmissionLine_{z,z_2,y} \cdot pNewTransmission_{z,z_2,\text{CapacityPerLine}} \cdot fAllowTransferExpansion.
$$

Expansion symmetry and cumulative build tracking: `eCumulativeTransferExpansion`, `eSymmetricTransferBuild`.

External exchange limits **[Optional]**: annual and hourly import/export share caps (`eMaxAnnualImportShareEnergy`, `eMaxAnnualExportShareEnergy`, `eMaxHourlyImportShareEnergy`, `eMaxHourlyExportShareEnergy`), point-to-point limits (`eExternalImportLimit`, `eExternalExportLimit`), and a minimum import floor (`eMinImportRequirement`) when `pMinImport > 0`.

---

## Storage Operations

### Capacity and power bounds

State-of-charge upper bound (`eSOCUpperBound`):

$$
vStorage_{st,q,d,t,y} \le vCapStor_{st,y}.
$$

Power injection and energy-power consistency (`eChargeCapacityLimit`, `eStorageCapMinConstraint`):

$$
vStorInj_{st,q,d,t,y} \le vCap_{st,y}, \qquad vCapStor_{st,y} \ge vCap_{st,y}.
$$

When a fixed energy-to-power ratio is specified (`eStorageFixedDuration`):

$$
vCapStor_{st,y} = vCap_{st,y} \cdot pStorageData_{st,\text{StorageDuration}}.
$$

Net charge balance (`eNetChargeBalance`):

$$
vStorNet_{st,q,d,t,y} = \sum_f vPwrOut_{st,f,q,d,t,y} - vStorInj_{st,q,d,t,y}.
$$

### State-of-charge dynamics

Three equations govern SOC evolution depending on time position and run mode.

**First hour of a representative day** (`eStateOfChargeInitRep`) — standard mode only, no carry-over from the previous day:

$$
vStorage_{st,q,d,t_1,y} = pStorageData_{st,\text{Efficiency}} \cdot vStorInj_{st,q,d,t_1,y}
  - \sum_f vPwrOut_{st,f,q,d,t_1,y}.
$$

**All subsequent hours** (`eStorageHourTransition`):

$$
vStorage_{st,q,d,t,y} = vStorage_{st,q,d,t-1,y}
  + pStorageData_{st,\text{Efficiency}} \cdot vStorInj_{st,q,d,t,y}
  - \sum_f vPwrOut_{st,f,q,d,t,y}.
$$

**Day boundary — [Dispatch mode only]** (`eStorageDayWrap`) — links the first hour of each day to the last chronological slot $AT-1$:

$$
vStorage_{st,q,d,t_1,y} = vStorage_{st,AT-1,y}
  + pStorageData_{st,\text{Efficiency}} \cdot vStorInj_{st,q,d,t_1,y}
  - \sum_f vPwrOut_{st,f,q,d,t_1,y}.
$$

**Cycle anchoring — [Dispatch mode only]** (`eStorageSOCInitDispatch`, `eStorageSOCFinalDispatch`) — the first and last hours of the full chronological cycle are pinned to a prescribed share of capacity:

$$
vStorage_{st,q,d,t,y} = vCapStor_{st,y} \cdot pStorageInitShare.
$$

### Reserve contribution

Storage can provide spinning reserve up to its current state of charge (`eSOCSupportsReserve`):

$$
vSpinningReserve_{st,q,d,t,y} \le vStorage_{st,q,d,t,y}.
$$

PV-coupled batteries are additionally capped by the paired PV profile (`eChargeLimitWithPVProfile`).

---

## CSP Extensions

CSP plants use dedicated variables for the solar thermal field (`vThermalOut`, `vCapTherm`) alongside the standard storage variables.

Charging is limited to what the thermal field produces, scaled by its efficiency (`eCSPStorageInjectionLimit`):

$$
vStorInj_{cs,q,d,t,y} \le vThermalOut_{cs,q,d,t,y} \cdot \eta_{\text{thermal field}}.
$$

Thermal output is bounded by field capacity and the solar resource profile (`eCSPThermalOutputLimit`):

$$
vThermalOut_{cs,q,d,t,y} \le vCapTherm_{cs,y} \cdot pCSPProfile_{cs,q,d,t}.
$$

The power balance links thermal output, storage charge/discharge, and turbine dispatch (`eCSPPowerBalance`):

$$
vThermalOut_{cs,q,d,t,y} \cdot \eta_{\text{thermal}}
  - vStorInj_{cs,q,d,t,y}
  + vStorOut_{cs,q,d,t,y} \cdot \eta_{\text{storage}}
  = \sum_f vPwrOut_{cs,f,q,d,t,y}.
$$

Storage capacity and charging are also bounded by `eCSPStorageCapacityLimit` and `eCSPStorageInjectionCap`. SOC dynamics follow the same recursion as standard storage (`eCSPStorageEnergyBalance`, `eCSPStorageInitialBalance`). Capacity evolution for both subsystems uses `eCapStor*` and `eCapTherm*`.

---

## Investment and Policy

Capital budget **[Optional — `fApplyCapitalConstraint`]** (`eCapitalConstraint`):

$$
\sum_{y,\, ng} pRR_y \cdot pWeightYear_y \cdot pCRF_{ng} \cdot vCap_{ng,y} \cdot pGenData_{ng,\text{Capex}}
  \le sMaxCapitalInvestmentInvestment \times 10^3.
$$

Minimum renewable share **[Optional]** (`eMinGenRE`): a fraction `pMinRE` of total annual energy must come from renewable sources.

Fuel availability cap **[Optional — `fApplyFuelConstraint`]** (`eFuelLimit`): annual consumption by country and fuel cannot exceed `pMaxFuelLimit`.

Minimum import **[Optional]** (`eMinImportRequirement`): enforces a lower bound on imports from a given zone when `pMinImport > 0`.

Discrete investment decisions for transmission (`vBuildTransmissionLine`) and generators/retirements (`vBuiltCapVar`, `vRetireCapVar`) are integer variables bounded by capacity consistency equations.

---

## CO₂ Emissions

Zonal emissions are computed from dispatched generation, heat rates, and carbon content (`eZonalEmissions`):

$$
vZonalEmissions_{z,y} = \sum_{g \in z,\, f,\, q,d,t}
  vPwrOut_{g,f,q,d,t,y} \cdot pHours_{q,d,t} \cdot pHeatRate_{g,f} \cdot pFuelCarbonContent_f.
$$

System total (`eTotalEmissions`):

$$
vTotalEmissions_y = \sum_z vZonalEmissions_{z,y}.
$$

**Country cap [Optional — `fApplyCountryCo2Constraint`]** (`eEmissionsCountry`):

$$
\sum_{z \in c} vZonalEmissions_{z,y} - vYearlyCO2backstop_{c,y} \le pEmissionsCountry_{c,y}.
$$

**System cap [Optional — `fApplySystemCo2Constraint`]** (`eTotalEmissionsConstraint`):

$$
vTotalEmissions_y - vYearlySysCO2backstop_y \le pEmissionsTotal_y.
$$

The backstop slack variables allow the cap to be violated at a penalty cost, which feeds into the objective function via `vYearlyCO2BackstopCostCountry` and `vYearlyCO2BackstopCostSystem`.

---

## Unit Commitment **[Dispatch mode only]**

Unit commitment is active when `fDispatchMode=1` and `fApplyMinGenCommitment=1`. Binary variables `isOn`, `isStartup`, and `isShutdown` track generator state at each chronological slot $AT$.

Transition consistency (`eCommitmentConsistency`):

$$
isOn_{g,AT,y} - isOn_{g,AT-1,y} = isStartup_{g,AT,y} - isShutdown_{g,AT,y}.
$$

`eCommitmentInitialization` sets the state at the first slot; `eCommitmentSingleTransition` prevents simultaneous startup and shutdown.

Min/max generation tied to commitment state (`eDispatchMinGenPoint`, `eDispatchMaxGenPoint`):

$$
vCap_{g,y} \cdot pGenData_{g,\text{MinGenShareAllHours}} \cdot isOn_{g,AT,y}
  \le vPwrOutDispatch_{g,AT,y}
  \le vCap_{g,y} \cdot (1 + pGenData_{g,\text{Overloadfactor}}) \cdot isOn_{g,AT,y}.
$$

Minimum up/down time **[Optional — `fApplyMUDT`]**: `eMinUpInitial`, `eMinUpRolling`, `eMinDownInitial`, `eMinDownRolling` prevent generators from cycling faster than their physical limits.

Startup costs **[Optional — `fApplyStartupCost`]**: `eStartupCostConstraint` accumulates per-slot startup costs into `vYearlyStartupCost`, which enters the objective via `vYearlyVariableCost`.
