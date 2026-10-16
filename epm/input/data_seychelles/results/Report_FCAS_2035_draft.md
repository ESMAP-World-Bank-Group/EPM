# Seychelles: firm capacity and frequency reserve (FCAS) in 2035

*Draft for discussion – 16 October 2026*

## Key messages

1. **With the IRP fleet (50 MW of new HFO), Mahé cannot hold the required frequency reserve in any hour of 2035.**
   The requirement – largest unit (8 MW) plus a 40 MW interconnector contingency plus 10% of solar and wind output –
   is 48–52 MW; the most Mahé can hold, with every engine online and all batteries charged, is 47 MW.
2. **About 74 MW of new thermal (nine 8 MW units) is needed** to meet the requirement in every hour if batteries are
   counted at 80% of their inverter capacity – 24 MW more than the IRP. If batteries can be counted at only 50%,
   **about 130 MW** is needed.
3. **Energy needs are lower**: about **34 MW** of new thermal (four units) is enough for Mahé to meet demand even
   without the interconnector. Between roughly 34 and 66 MW of new thermal, the system meets demand but not the reserve
   requirement.
4. **The additional capacity is for reserve, not energy.** It barely changes operating costs (about USD 74–75 m/yr);
   its cost is the investment.
5. **Reserve stays scarce even at the threshold**: with 74 MW of new thermal the FCAS price on Mahé averages
   USD 50 per MW-h and exceeds USD 100 in about 1,200 hours.
6. **The gap could also be closed by other means**: about 6 MW of additional battery inverter on Mahé, or operating
   the interconnector so that a trip never removes more than about 35 MW.

![Graphical abstract](figures/fig1_graphical_abstract.png)

## 1. Question

The Seychelles IRP adds 50 MW of new HFO capacity on Mahé, a 40 MW Mahé–Praslin interconnector and large amounts of
solar, wind and battery storage by 2035. As the share of variable renewables grows, the system must keep enough fast
frequency reserve (FCAS) to withstand the loss of its largest elements. The question is how much new firm (thermal)
capacity Mahé needs for this, and how that compares with the capacity needed simply to meet demand.

## 2. Approach

**Hourly dispatch.** We used the World Bank Electricity Planning Model (EPM) to simulate all 8,760 hours of 2035 for
Mahé and Praslin–La Digue, linked by the interconnector. In each hour the model decides which engines run, how much
they produce, how batteries charge and discharge, how much power flows over the cable – and how much reserve each
unit holds. Energy and reserve are optimised together at least cost. If reserve cannot be met, the shortfall costs
USD 500 per MW-h; unserved energy costs USD 1,000 per MWh, so the model sheds reserve before load.

**Reserve requirement (per hour):**

| | Requirement |
|---|---|
| Mahé | largest unit (8 MW) + interconnector contingency (40 MW) + 10% of solar and wind output |
| Praslin–La Digue | two largest units (4.5 MW), or largest unit plus import from Mahé, + 10% of solar and wind output |
| Mahé without the interconnector | two largest units (16 MW) + 10% of solar and wind output |

**Who can provide reserve.** Engines of 6 MW or more, only while running, up to 20% of their rated capacity and within
the headroom left by their output; batteries up to 80% of inverter capacity (50% in a sensitivity), limited by the
energy stored at the start of each hour. Smaller engines, waste-to-energy and solar/wind provide no reserve.

**Availability.** Thermal plants and waste-to-energy are available at 90% of their capacity in every hour, solar and
wind at 95%. Batteries are fully available.

**Minimum new thermal.** Starting from the IRP fleet (six new 8 MW units plus 2.4 MW), we removed units to find where
demand can no longer be met, and added units one by one until the reserve requirement is met in every hour.

## 3. Main assumptions (2035)

| Item | Assumption | Source |
|---|---|---|
| Demand | Mahé 590 GWh, peak 96 MW; Praslin–La Digue 74 GWh, peak 13 MW (hourly profile) | IRP base case |
| Existing and committed thermal (Mahé) | Victoria C: 4 × 8 MW + extension 2 × 7.8 MW (older units retired) | PUC, ECA |
| New thermal (Mahé) | IRP: 50.4 MW, modelled as 6 × 8 MW + 2.4 MW | ECA IRP; unit size assumed |
| Thermal (Praslin–La Digue) | 14.8 MW of small diesel units | PUC, ECA |
| Solar / wind | Mahé 42 MW / 10 MW; Praslin–La Digue 29 MW / 70 MW; rooftop PV excluded | ECA IRP |
| Batteries | Mahé 35 MW / 70 MWh; Praslin–La Digue 40 MW / 81 MWh | ECA IRP |
| Interconnector | 40 MW | ECA IRP |
| Availability | thermal 90%, solar and wind 95%, every hour | Assumption |
| Engine operation | minimum load 30%; minimum up/down times; start-up costs | ECA; assumptions |
| Fuel | HFO 18.2, diesel 27.5 USD/MMBtu (2024 prices) | WB strategy note |

## 4. Results

### 4.1 IRP fleet

With 50 MW of new HFO, Mahé falls short of the reserve requirement **in every hour**, by up to 4.4 MW. At night the
gap is about 1 MW (48 MW required, 47 MW available); in the middle of the day it grows because 10% of solar output adds
to the requirement. Demand is met in every hour.

![When the shortfall occurs](figures/fig2_shortfall_heatmap.png)

### 4.2 Minimum new thermal

| New thermal on Mahé | Reserve shortfall (batteries at 80%) | Reserve shortfall (batteries at 50%) | Demand met without interconnector? |
|---|---|---|---|
| 18 MW (2 units) | not run | not run | **No** (8,700 MWh unserved) |
| 26 MW (3 units) | all hours, up to 20 MW | not run | **No** (about 400 MWh unserved, morning and evening peaks) |
| 34 MW (4 units) | all hours, up to 12 MW | not run | Yes |
| **50 MW (IRP)** | **all hours, up to 4.4 MW** | all hours, up to 15 MW | Yes |
| 58 MW (7 units) | 2,800 daytime hours, up to 2.8 MW | not run | Yes |
| 66 MW (8 units) | 1,000 midday hours, up to 1.2 MW | not run | Yes |
| **74 MW (9 units)** | **none** | not run | Yes |
| 82 MW (10 units) | none | all hours, up to 8.5 MW | Yes |
| 114 MW (14 units) | none | 1,400 midday hours, up to 2.0 MW | Yes |
| 122 MW (15 units) | none | 150 midday hours, up to 0.4 MW | Yes |
| **130 MW (16 units)** | none | **none** | Yes |

Cases not run lie beyond the thresholds; adding units can only reduce the shortfall.

- **Reserve, batteries at 80%: about 74 MW of new thermal**, 24 MW above the IRP.
- **Reserve, batteries at 50%: about 130 MW**, 80 MW above the IRP.
- **Demand only: about 34 MW**, even if the interconnector is out of service.

### 4.3 What the additional thermal does

The additional units are needed to hold reserve, not to produce energy. Mahé's thermal output changes little (about
436 GWh/yr with 6 or 9 new units), and operating cost stays at about USD 74–75 m/yr. Each extra 8 MW unit adds only
1.6 MW of reserve (20% of its capacity), so closing a 4.4 MW gap with engines takes three units.

### 4.4 Without the interconnector

If the interconnector is out of service, the reserve requirement on Mahé falls to its two largest units (16 MW) plus
the VRE share. The binding issue is then energy: Mahé meets demand on its own with four or more new units; with three
units about 400 MWh is unserved and reserve falls short by up to 5 MW in about 30 hours. With four units the spare
reserve just reaches zero in a few hours (below 2 MW in 15 hours with batteries at 80%, 470 hours at 50%).
Praslin–La Digue has to curtail about a fifth of its solar and wind output.

![Islanded headroom](figures/fig3_Islanded_U4_fcas_margin.png)

### 4.5 Value of reserve

The FCAS price (the model's marginal cost of the reserve requirement) shows how scarce reserve is:

| Case | Average (USD/MW-h) | Maximum | Hours above USD 100 |
|---|---|---|---|
| IRP fleet (50 MW new) | 500 (penalty, short every hour) | 500 | 8,760 |
| 74 MW new (9 units) | 50 | 445 | about 1,200 |
| Without interconnector, 34 MW new | 0 | 19 | 0 |

## 5. Comparison with the worst-hour spreadsheet

The earlier spreadsheet found a Mahé shortfall of 4.1 MW with the IRP fleet; the dispatch finds up to 4.4 MW. The two
are consistent. The small difference comes from the solar and wind share and from accounting for the reserve held by
the unit that trips. The dispatch adds the hourly picture: the shortfall occurs in every hour, is largest at midday,
and is not removed by realistic operation of engines and batteries.

## 6. Options to close the reserve gap (IRP fleet, batteries at 80%)

| Option | Size | Comment |
|---|---|---|
| More thermal units on Mahé | +3 × 8 MW (74 MW new in total) | Robust; additional investment and fixed O&M |
| More battery inverter on Mahé | about +6 MW | Cheaper capacity, but relies on batteries held charged for reserve |
| Limit the interconnector contingency | about 35 MW | Operate the cable so a trip never removes more than about 35 MW; some Praslin wind curtailed |
| Accept daytime curtailment of solar | – | Lowers the VRE share of the requirement; helps only with the last 1–3 MW |

If batteries cannot reliably be counted at 80% of inverter capacity, the thermal requirement rises steeply (130 MW at
50%), so the treatment of batteries in the reserve calculation is a key decision.

## 7. Caveats

- The 40 MW interconnector contingency is applied in every hour, including hours when the cable carries less power;
  this is conservative.
- Availability is a fixed 90% (thermal) and 95% (solar, wind) in every hour, not random outages; batteries are fully
  available.
- Engine operation is simplified (relaxed unit commitment); a full on/off check of the key cases is still to be done.
- The unit size of the new HFO plant (assumed 8 MW) and the availability of the Victoria B/C units in 2035 are not yet
  confirmed; both affect the result.
- Rooftop PV is excluded (demand treated as net of rooftop).

## 8. Next steps

1. Confirm with PUC: new plant unit size, Victoria B/C units kept in 2035, battery role in reserve.
2. Full on/off check of the key cases (9 units; 16 units with batteries at 50%).
3. Test the options in section 6 in the model (additional battery, interconnector contingency limit).

---

*Model inputs and results: EPM branch `seychelles_2026`, folder `epm/input/data_seychelles` (results in `results/`).*
