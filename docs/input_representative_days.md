# Representative days

### Why representative days?

In long-term capacity expansion models, which typically simulate multiple years with hourly resolution, computational requirements can quickly become prohibitive. To address this, models often rely on a reduced number of representative days to approximate the temporal variability of demand and VRE generation.

Each representative day is selected from historical data and assigned a weight corresponding to the number of real days it stands for. The objective is to preserve key temporal dynamics—such as daily and seasonal load patterns, renewable variability, and the correlation between demand and VRE—while significantly reducing model size and run time.

However, this simplification must be handled carefully. A poorly selected set of representative days can lead to significant inaccuracies in system planning: misestimating investment needs, under- or overestimating flexibility requirements, or misrepresenting system costs. The method used to select representative days is therefore critical to ensure reliable model outcomes.

---

### Methods for selecting representative days

Several approaches are available to select representative days:

- **Heuristic selection**: manually select a small number of days based on predefined criteria (e.g. maximum load, minimum wind). Easy to implement but generally not robust or representative.
  
- **Clustering algorithms**: use data-driven methods such as K-means to group similar days, then select a real day closest to the centroid of each cluster. This method is computationally efficient and widely used, but may miss rare or extreme events.

- **Optimization-based selection**: explicitly formulate and solve a mathematical problem to identify the subset of days and associated weights that minimize the error in approximating key temporal characteristics. This method is the most rigorous and accurate.

The **optimization method** is described below and recommended for model accuracy.

---

### Representative day selection using optimization (Poncelet method)


This method is based on the work described in **Poncelet et al. (2017)** – [Selecting Representative Days for Capturing the Implications of Integrating Intermittent Renewables in Generation Expansion Planning Problems, IEEE Transactions on Power Systems](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/docs/dwld/Poncelet_et_al._-_2017_-_Selecting_Representative_Days.pdf)


#### Methodology

The approach formulates a mixed-integer linear programming (MILP) model that selects a fixed number of days and assigns them weights such that the reduced set best approximates the original time series. The optimization minimizes a weighted error across a set of metrics that reflect the most relevant temporal features for power system planning:
1. Relative Energy Error (REE) — Ensures the total annual energy (per time series) is preserved.
2. Normalized Root Mean Square Error on Duration Curves (NRMSE-DC) — Preserves the distribution of hourly values over the year.
3. Correlation Error (CE) — Maintains the correlation structure between series (e.g. between Load and PV).
4. Ramp Duration Curve Error (NRMSE-RDC) — Captures short-term variability relevant to storage and flexibility assessment.

These metrics are combined in a single objective function, allowing the optimization to trade off between them based on their relative importance.

---

#### Overall process

The selection process starts from hourly time series for load, solar PV, and wind generation over at least one year. These are first transformed into statistical representations—such as duration curves, ramp duration curves, and correlation matrices—that capture key temporal characteristics. A mixed-integer linear programming (MILP) problem is then formulated and solved to identify the optimal combination of representative days and their weights. Finally, the selected days and associated weights are exported into standardized CSV files (e.g. `pHours`, `pDemandProfile`, `pVREProfile`) for use in EPM.

**Inputs:**
- Hourly time series over at least one year (Load, PV, Wind).
- Desired number of representative days to be selected.
- Optional constraints (e.g., inclusion of specific “extreme” days).

**Outputs:**
- List of selected representative days with their corresponding season.
- Assigned weights for each selected day (number of real days it represents).
- Standardized CSV files for use in EPM:
  - `pHours.csv`: time weights,
  - `pDemandProfile.csv`: load profiles,
  - `pVREProfile.csv`: PV and wind profiles.

For additional information and implementation details, refer to the [notebook on representative days](https://esmap-world-bank-group.github.io/EPM/docs/representative_days.html).
