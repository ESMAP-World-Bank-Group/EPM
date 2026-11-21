# TYNDP staging: what would change

Staged for the reference scenario Int_Base_5y. Nothing is in epm/input yet.

## Files

- `pExtTransferLimit_robg_tyndp26.csv` from `trade/pExtTransferLimit_robg.csv`
- `pTransferLimit_robg_hvdc_tyndp26.csv` from `trade/pTransferLimit_robg_hvdc.csv`
- `pNewTransmission_robg_tyndp26.csv` from `none, new candidate list`
- `pNewTransmissionExt_robg_tyndp26.csv` from `none, new candidate list`

## Border limits, MW

11 directions rise, 5 stay, 4 are held.

| From | To | Current 2040 | 2030 | 2032 | 2036 | 2040 | Change 2040 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Romania | Hungary | 1,000 | 2,650 | 2,650 | 3,700 | 3,700 | +2,700 | raised |
| Hungary | Romania | 1,000 | 1,450 | 1,450 | 3,600 | 3,600 | +2,600 | raised |
| Romania | Serbia | 800 | 2,650 | 2,650 | 2,650 | 2,650 | +1,850 | raised |
| Serbia | Romania | 800 | 2,250 | 2,250 | 2,250 | 2,250 | +1,450 | raised |
| Bulgaria | Serbia | 750 | 750 | 750 | 930 | 930 | +180 | raised |
| Serbia | Bulgaria | 750 | 750 | 750 | 750 | 750 | +0 | unchanged |
| Bulgaria | NorthMacedonia | 400 | 400 | 400 | 400 | 400 | +0 | unchanged |
| NorthMacedonia | Bulgaria | 400 | 400 | 400 | 400 | 400 | +0 | unchanged |
| Bulgaria | Greece | 650 | 1,200 | 1,200 | 1,200 | 1,200 | +550 | raised |
| Greece | Bulgaria | 650 | 1,000 | 1,000 | 1,000 | 1,000 | +350 | raised |
| Trakia | Greece | 100 | 166 | 166 | 166 | 166 | +66 | raised |
| Greece | Trakia | 166 | 218 | 218 | 218 | 218 | +52 | raised |
| Romania | Bulgaria | 1,400 | 2,100 | 2,100 | 2,100 | 2,100 | +700 | raised |
| Bulgaria | Romania | 1,400 | 2,200 | 2,200 | 2,200 | 2,200 | +800 | raised |
| Bulgaria | Trakia | 334 | 334 | 334 | 334 | 334 | +0 | unchanged |
| Trakia | Bulgaria | 200 | 200 | 200 | 200 | 200 | +0 | unchanged |
| Romania | Ukraine | 455 | 347 | 369 | 412 | 455 | +0 | held |
| Ukraine | Romania | 223 | 234 | 232 | 228 | 223 | +0 | held |
| Romania | Moldova | 547 | 436 | 458 | 503 | 547 | +0 | held |
| Moldova | Romania | 37 | 35 | 36 | 36 | 37 | +0 | held |

2025 is unchanged on every border. Seasons stay flat.

## Candidate lines

| Line | Project | Action | Earliest entry | MW | Cost, MUSD | MUSD per MW | Freeexp MW | Freeexp MUSD | Freeexp MUSD per MW |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| BG00-RS00 Real 1 | P342 | drop | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| HU00-RO00 Real 1 | P259 | drop | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| BG00-TR00 Real 1 | P1066 | candidate | 2040 | 700 | 64.8 | 0.093 | 775 | 250 | 0.323 |
| GR00-TR00 Real 1 | P1067 | candidate | 2035 | 600 | 94.6 | 0.158 | 775 | 250 | 0.323 |

## Read before promotion

- External zones are price takers, so higher limits give an upper bound on export volume and value.
- The gross external trade cap of 0.2 for Romania and Bulgaria still applies, so part of a higher external limit may stay unused.
- `fCountIntercoForReserves` is 1: the higher Romania-Bulgaria limit and a built Bulgaria-Trakia candidate also raise the planning reserve credit of those countries.
- Candidate capex comes from the 2024 edition, the one exception to no mixing. GRID.xlsx states no price year; converted at 1.08 USD per EUR.
- The TYNDP capex sits well below the cost of the freeexp line on the same border. The freeexp candidates are not merged here.
- `CapacityPerLine` holds one value: Bulgaria-Trakia uses 700 MW, the weaker direction, for the full cost of the 1,100/700 MW project.
- 2z variants and the corridor limit files (BSSC, GEC, onland) are not covered: these files replace the base limits only.
