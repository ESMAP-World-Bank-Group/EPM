# Contributing postprocess

Our analysis framework is built around three key dimensions:

1. **Spatial** — e.g., zones, countries, or aggregated system-level data (EPM’s lower-resolution level).
2. **Temporal** — typically annual values, but hourly data is also possible for dispatch analyses.
3. **Uncertainty** — represented through different scenarios.

In addition to these dimensions, we also work with **components** (e.g., fuel types, capacity mix, energy mix, cost breakdown) to provide insight into how results are composed.

Since it’s only practical to visualize **two dimensions plus one component at a time**, we use a consistent plotting strategy based on **stacked bar subplots**:

- Each **subplot** represents one dimension (e.g., year or zone).
- Each **bar within a subplot** represents another dimension (e.g., scenario).
- The **stack within each bar** shows the breakdown by component.

## Plotting approach

We generally generate three types of figures:

1. **System-wide comparison across scenarios over time**

   - The spatial dimension is aggregated.
   - Each subplot is a year.
   - Bars within each subplot correspond to different scenarios.

2. **Spatial evolution within a single scenario**

   - One scenario is fixed.
   - We show the evolution across zones.

3. **Scenario comparison per zone at a fixed point in time**
   - The year is fixed.
   - Bars compare scenarios for each zone.

These three figure types are typically produced for **capacity**, **energy**, and **cost**.
