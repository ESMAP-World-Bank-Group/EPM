# P5 - staged trade prices vs the live data_blacksea file

Live file: every external zone flat across all hours and years - Bulgaria, Greece, Romania and Kazakhstan at 70 USD/MWh, Iran, Iraq and Syria at 40, Russia at 35.

Only Bulgaria, Greece and Romania are rewritten. The other five external zones have no liquid hub, no TYNDP entry and no ETS; they keep their cost-based values and are copied through unchanged.


## Annual mean, USD 2024 per MWh

Buy side (`pTradePrice`), which is the only file the model reads today.

| scenario | zone | live | 2026 | 2030 | 2040 | 2050 |
|---|---|---|---|---|---|---|
| **eu_central** | Bulgaria | 70 | 103.1 | 95.1 | 96.5 | 96.5 |
| **eu_central** | Greece | 70 | 103.9 | 96.1 | 101.5 | 101.5 |
| **eu_central** | Romania | 70 | 103.7 | 92.7 | 94.1 | 94.1 |
| eu_low | Bulgaria | 70 | 102.2 | 92.3 | 64.5 | 57.9 |
| eu_low | Greece | 70 | 99.0 | 81.3 | 58.4 | 46.0 |
| eu_low | Romania | 70 | 104.0 | 93.7 | 66.2 | 61.0 |
| eu_very_low | Bulgaria | 70 | 98.7 | 81.9 | 45.4 | 47.2 |
| eu_very_low | Greece | 70 | 97.4 | 76.4 | 43.9 | 46.2 |
| eu_very_low | Romania | 70 | 100.7 | 83.7 | 46.9 | 48.2 |
| eu_high | Bulgaria | 70 | 104.1 | 98.0 | 144.8 | 161.7 |
| eu_high | Greece | 70 | 109.1 | 113.7 | 177.6 | 227.2 |
| eu_high | Romania | 70 | 103.4 | 91.7 | 133.9 | 145.6 |
| eu_crisis | Bulgaria | 70 | 130.2 | 176.2 | 176.2 | 176.2 |
| eu_crisis | Greece | 70 | 137.8 | 197.8 | 197.8 | 197.8 |
| eu_crisis | Romania | 70 | 133.6 | 182.3 | 182.3 | 182.3 |
| eu_central_cbam | Bulgaria | 70 | 103.1 | 95.1 | 96.5 | 96.5 |
| eu_central_cbam | Greece | 70 | 103.9 | 96.1 | 101.5 | 101.5 |
| eu_central_cbam | Romania | 70 | 103.7 | 92.7 | 94.1 | 94.1 |
| eu_low_cbam | Bulgaria | 70 | 102.2 | 92.3 | 64.5 | 57.9 |
| eu_low_cbam | Greece | 70 | 99.0 | 81.3 | 58.4 | 46.0 |
| eu_low_cbam | Romania | 70 | 104.0 | 93.7 | 66.2 | 61.0 |
| eu_very_low_cbam | Bulgaria | 70 | 98.7 | 81.9 | 45.4 | 47.2 |
| eu_very_low_cbam | Greece | 70 | 97.4 | 76.4 | 43.9 | 46.2 |
| eu_very_low_cbam | Romania | 70 | 100.7 | 83.7 | 46.9 | 48.2 |
| eu_high_cbam | Bulgaria | 70 | 104.1 | 98.0 | 144.8 | 161.7 |
| eu_high_cbam | Greece | 70 | 109.1 | 113.7 | 177.6 | 227.2 |
| eu_high_cbam | Romania | 70 | 103.4 | 91.7 | 133.9 | 145.6 |
| eu_crisis_cbam | Bulgaria | 70 | 130.2 | 176.2 | 176.2 | 176.2 |
| eu_crisis_cbam | Greece | 70 | 137.8 | 197.8 | 197.8 | 197.8 |
| eu_crisis_cbam | Romania | 70 | 133.6 | 182.3 | 182.3 | 182.3 |

## What the sell side costs, if the export patch does not land

`base.gms:686` credits exports at the import price. Until `pTradePriceExport` exists, the model values every exported MWh at the buy price - the gap below is the per-MWh overstatement of export revenue.

| scenario | zone | mean buy | mean sell | overstatement |
|---|---|---|---|---|
| eu_central | Bulgaria | 97.3 | 90.2 | **+7.2** |
| eu_central | Greece | 100.8 | 93.5 | **+7.3** |
| eu_central | Romania | 95.5 | 88.4 | **+7.1** |
| eu_low | Bulgaria | 75.3 | 68.8 | **+6.5** |
| eu_low | Greece | 65.0 | 58.8 | **+6.2** |
| eu_low | Romania | 77.2 | 70.7 | **+6.5** |
| eu_very_low | Bulgaria | 62.4 | 56.3 | **+6.1** |
| eu_very_low | Greece | 59.0 | 53.0 | **+6.0** |
| eu_very_low | Romania | 63.9 | 57.7 | **+6.1** |
| eu_high | Bulgaria | 131.3 | 123.1 | **+8.2** |
| eu_high | Greece | 168.6 | 159.3 | **+9.3** |
| eu_high | Romania | 122.0 | 114.1 | **+7.9** |
| eu_crisis | Bulgaria | 168.1 | 158.8 | **+9.3** |
| eu_crisis | Greece | 187.3 | 177.4 | **+9.9** |
| eu_crisis | Romania | 173.7 | 164.3 | **+9.4** |
| eu_central_cbam | Bulgaria | 97.3 | 15.4 | **+82.0** |
| eu_central_cbam | Greece | 100.8 | 15.0 | **+85.8** |
| eu_central_cbam | Romania | 95.5 | 31.4 | **+64.1** |
| eu_low_cbam | Bulgaria | 75.3 | 11.9 | **+63.4** |
| eu_low_cbam | Greece | 65.0 | 9.6 | **+55.4** |
| eu_low_cbam | Romania | 77.2 | 21.6 | **+55.7** |
| eu_very_low_cbam | Bulgaria | 62.4 | 10.1 | **+52.3** |
| eu_very_low_cbam | Greece | 59.0 | 9.1 | **+49.9** |
| eu_very_low_cbam | Romania | 63.9 | 16.4 | **+47.5** |
| eu_high_cbam | Bulgaria | 131.3 | 32.3 | **+99.0** |
| eu_high_cbam | Greece | 168.6 | 57.9 | **+110.7** |
| eu_high_cbam | Romania | 122.0 | 54.0 | **+68.1** |
| eu_crisis_cbam | Bulgaria | 168.1 | 62.5 | **+105.6** |
| eu_crisis_cbam | Greece | 187.3 | 74.9 | **+112.4** |
| eu_crisis_cbam | Romania | 173.7 | 102.6 | **+71.1** |

## Untouched zones

| zone | value | basis |
|---|---|---|
| Iran | 40 | cost-based, unchanged |
| Iraq | 40 | cost-based, unchanged |
| Kazakhstan | 70 | cost-based, unchanged |
| Russia | 35 | cost-based, unchanged |
| Syria | 40 | cost-based, unchanged |
