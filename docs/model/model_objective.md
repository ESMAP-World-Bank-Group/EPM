# Objective Function

EPM minimizes the net present value of total power system costs over the planning horizon. This page covers the cost structure and notation used throughout the formulation.

For the full list of sets, parameters, and variables, see [Reference](equations.md).

---

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

---

## Objective Function and Cost Components

The system-wide net present value minimization is written as

$$
vNPVCost = \sum_{y} pRR_{y} \cdot pWeightYear_{y} \cdot \Big(
  \sum_{c} vYearlyTotalCost_{c,y}
  + vYearlyUnmetPlanningReserveCostSystem_{y}
  + vYearlyUnmetSpinningReserveCostSystem_{y}
  + vYearlyCO2BackstopCostSystem_{y}
\Big).
$$

The yearly country cost balance expands to

$$
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
$$

Annual cost components are defined as follows:

- `vYearlyFOMCost`: fixed O&M costs for generators, storage, and CSP thermal fields.
- `vYearlyVariableCost = vYearlyFuelCost + vYearlyVOMCost + vYearlyStartupCost`. The startup cost term is non-zero only when `fDispatchMode=1` and `fApplyStartupCost=1`; it is forced to zero otherwise (`eYearlyStartupCostOff`).
- `vYearlySpinningReserveCost`: reserve cost bids multiplied by reserve provision and time weights.
- `vYearlyUSECost`: value of lost load applied to unserved energy.
- `vYearlySurplus`: penalty on over-generation.
- `vYearlyCurtailmentCost`: penalty on curtailed VRE output.
- `vYearlyExternalTradeCost = vYearlyImportExternalCost - vYearlyExportExternalCost`.
- `vYearlyCarbonCost`: carbon price times the emissions implied by dispatched generation.
- `vAnnualizedTransmissionCapex`: annualized cost of new transmission builds associated with each zone.
- `vYearlyUnmetReserveCostCountry = vYearlyUnmetPlanningReserveCostCountry + vYearlyUnmetSpinningReserveCostCountry`.

Each of these relations is coded in `base.gms` as equations `eYearly*`.
