# Model Overview

Least-cost MIP capacity expansion and economic dispatch model, implemented in GAMS with a Python orchestration layer. Deployed across 100+ countries by ESMAP — see [Applications](../introduction/introduction_case_studies.md).

---

## Specifications

| | |
|---|---|
| **Formulation** | Mixed-integer program (MIP) or relaxed (RMIP) |
| **Objective** | Minimize NPV of total system cost (capex + opex) |
| **Horizon** | Multi-year, user-defined |
| **Time resolution** | Representative days · Full hourly chronology ([Dispatch Mode](../run/run_dispatch.md)) |
| **Spatial** | Multi-zone · Multi-country · Transmission network with losses and expansion |
| **Decision variables** | Capacity builds · Retirements · Dispatch · Transmission investment |
| **Scenarios** | Multi-scenario runs · Monte Carlo over uncertain parameters |

## Technologies

Thermal (gas, coal, oil, nuclear) · Solar PV · Wind · Hydropower · Battery storage · Pumped hydro · CSP with thermal storage · Hydrogen electrolyzers · CCS retrofits

## Features

| Feature | Notes |
|---|---|
| Unit commitment | On/off binaries, startup costs, min up/down time — Dispatch Mode only |
| Storage | SOC dynamics, charge/discharge efficiency, PV-coupled batteries |
| Reserves | Spinning reserve + planning reserve margin, country and system level |
| Emissions | CO₂ accounting per zone; country and system caps with backstop slack |
| Policy constraints | RE targets · Carbon tax · Fuel limits · Capital budget |

## Assumptions

Social planner model: perfect competition, inelastic demand, economically efficient trade across zones. Suitable for long-term planning — not for market dynamics or strategic bidding.

## Outputs

Capacity expansion plan · Generation dispatch · System NPV and cost breakdown · Transmission flows and utilization · Marginal prices (dual variables) · CO₂ emissions · Trade volumes
