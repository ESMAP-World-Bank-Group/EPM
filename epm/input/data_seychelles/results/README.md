# Seychelles 2035 – FCAS and minimum new thermal (EPM dispatch)

*Runs of 15 October 2026. Relaxed unit commitment (LP); full on/off (MIP) check pending.*

## Question

How much new thermal capacity does Mahé need in 2035 to hold the required fast frequency reserve (FCAS) in every
hour, and from which point does the system meet demand but not FCAS? The IRP base case plans 50.4 MW of new HFO.

## Model

- EPM in dispatch mode: two zones (Mahé, Praslin–La Digue) linked by the planned 40 MW cable, 8,760 hours of 2035,
  no capacity expansion. Energy and reserve are co-optimised at least cost.
- Unit commitment on: minimum load 30%, minimum up/down time, start-up cost. Engines give reserve only while
  running, up to 20% of MW and within headroom; batteries up to 80% of inverter MW, limited by stored energy.
- Unmet reserve is reported as FCAS shortfall (penalty USD 1,000 per MW-hour; unserved energy USD 1,000 per MWh).
- Inputs and settings: [`../README.md`](../README.md). Local model changes (`epm/base.gms`, `epm/main.gms`):
  reserve only from committed units, faster unit-commitment equations, interconnector-trip option.

## FCAS requirement

| Case | Requirement per hour |
|---|---|
| Mahé, cable in service | largest unit 8 MW + cable flow 40 MW (fixed) + 10% of solar and wind output = **48 MW + VRE** |
| PLD, cable in service | max(two largest units 4.5 MW, largest unit 2.5 MW + import from Mahé) + 10% of solar and wind output |
| Islanded (no cable) | two largest units + 10% of solar and wind output: Mahé 16 MW, PLD 4.5 MW |

Maximum Mahé capability with the IRP fleet, all engines online and batteries full:
0.8 × 35 MW (BESS) + 0.2 × 95.6 MW (engines ≥ 6 MW) = **47.1 MW**, below 48 MW + VRE. Each extra 8 MW unit adds 1.6 MW.

## Key assumptions (2035)

| Item | Value |
|---|---|
| Demand | IRP hourly profile as provided: Mahé 590 GWh / 96.2 MW peak; PLD 74 GWh / 12.8 MW |
| Mahé thermal | Victoria C A51/A61/B41/B51 (4 × 8 MW), extension (2 × 7.8 MW), new HFO 6 × 8 MW + 2.4 MW (IRP 50.4 MW) |
| PLD thermal | 14.8 MW LFO (no unit ≥ 6 MW, so no FCAS from PLD engines) |
| Solar / wind | Mahé 42.3 MW solar, 10 MW wind; PLD 29 MW solar, 70 MW wind; rooftop PV excluded |
| Batteries | Mahé 35 MW / 70 MWh; PLD 40 MW / 81 MWh |
| Waste-to-energy | 5 MW on Mahé, no FCAS |

## Scenarios

| Scenario | Description |
|---|---|
| `Base_U<k>` | k new 8 MW HFO units (plus the 2.4 MW remainder), batteries at 80% of inverter MW; `Baseline` = k = 6 |
| `BESS50_U<k>` | as Base, batteries counted at 50% of inverter MW; `BESS50` = k = 6 |
| `Islanded_U<k>` | no cable; requirement two largest units + VRE; `Islanded` = k = 6 |

## Results (Mahé, 2035)

| New 8 MW units | New thermal (MW) | Batteries 80%: FCAS shortfall (MWh/yr; hours; max MW) | Batteries 50%: FCAS shortfall (MWh/yr; hours; max MW) | Islanded: unserved energy (MWh/yr) |
|---|---|---|---|---|
| 2 | 18.4 | 71,142; all hours; 11.1 | n/r | **2,996** |
| 3 | 26.4 | 57,097; all hours; 9.5 | n/r | 0 |
| 4 | 34.4 | 43,046; all hours; 7.9 | n/r | 0 |
| 5 | 42.4 | 28,957; all hours; 6.3 | n/r | 0 |
| **6 (IRP)** | **50.4** | **14,810; all hours; 4.7** | 106,825; all hours; 15.2 | 0 |
| 7 | 58.4 | 3,980; 2,811 h (daytime); 3.0 | 92,616; all hours; 13.6 | n/r |
| 8 | 66.4 | 632; 1,130 h (10:00–16:00); 1.4 | 78,252; all hours; 11.9 | n/r |
| **9** | **74.4** | **0** | 63,829; all hours; 10.2 | n/r |
| 10 | 82.4 | n/r | 49,433; all hours; 8.7 | n/r |
| 11 | 90.4 | n/r | 35,075; all hours; 7.1 | n/r |
| 12 | 98.4 | n/r | 20,732; all hours; 5.4 | n/r |
| 13 | 106.4 | n/r | 6,422; all hours; 3.8 | n/r |
| 14 | 114.4 | n/r | 1,488; 1,441 h (midday); 2.2 | n/r |
| 15 | 122.4 | n/r | 48; 245 h (midday); 0.6 | n/r |
| **16** | **130.4** | n/r | **0** | n/r |

n/r = not run. Shortfalls fall steadily with each added unit (about 14,000 MWh per unit while the gap applies in
every hour, since each unit adds 1.6 MW of FCAS). Cases not run are not needed: adding units cannot increase the
shortfall, so batteries-80% cases above 9 units and islanded cases above 6 units would also show zero.

- **Minimum new thermal to meet FCAS: 74.4 MW** (9 units) with batteries at 80%; **130.4 MW** (16 units) at 50%
  (15 units leave only 48 MWh of midday shortfall, which curtailing a little solar would remove).
- **Minimum new thermal to meet demand** (islanded): **26.4 MW** (3 units). Between 3 and 8 units the system
  meets demand but not FCAS.
- Islanded FCAS (16 MW + VRE) is met at every level.
- Operating cost (excluding penalties, including fixed O&M, no capital cost): USD 73.4 m/yr at 6 units,
  73.5 m/yr at 9 units.

Full table: [`scenario_summary_2035_v2_sweep.csv`](scenario_summary_2035_v2_sweep.csv)
(columns: system cost, penalties, operating cost, requirement, imports, reserve by source, energy, curtailment,
unserved energy, FCAS shortfall).

## Caveats

- The 40 MW cable contingency is applied in every hour, including hours when the cable carries less.
- Unserved energy and FCAS shortfall have the same penalty, so in FCAS-short cases with the cable the model may shed
  some load instead of reserve; the islanded runs give the clean adequacy result.
- LP relaxation of unit commitment; key cases to be checked as MIP.
- Run outputs are not in the repository (`epm/output/` is ignored).

## Reproduce

```bash
cd epm
python epm.py --folder_input data_seychelles --scenarios --parallel 2 -S Baseline BESS50 Islanded Base_U9 BESS50_U16
```

Run at most 2 scenarios in parallel on a 32 GB machine (4 in parallel ran out of memory). The hourly FCAS balance
and the summary table are produced by `analyse_fcas.py` and `summarise_runs.py` (project scripts, not in this
repository).
