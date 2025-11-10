# Transmission Line Capacity and Cost Calculator – Explanatory Notes

## Objective

In many developing-country projects, we do not have detailed engineering data about existing or planned transmission lines.  
The idea is to build a **very simplified tool** that can still provide a reasonable **order of magnitude** for:

- the **transfer capacity** of a transmission line (MW), and  
- the **investment cost** of constructing such a line (MUSD).

This tool is meant for **pre-feasibility** or **planning** purposes when limited information is available.  
Even approximate results are valuable to compare options or identify bottlenecks.

---

## 1. Basic User Inputs

Only three inputs are required:

1. **Number of circuits (N_circ)**  
   - A circuit is one complete three-phase system (three conductors).  
   - `N_circ = 1` → single-circuit line.  
   - `N_circ = 2` → double-circuit line (higher capacity and reliability).

2. **Voltage (V_kV)**  
   - Choose among 220, 330, 400, or 500 kV.  
   - Determines the technical parameters and unit costs used by the tool.

3. **Line length (L_km)**  
   - Approximate distance between substations, in kilometers.  
   - Can be estimated quickly using **Google Maps** or GIS tools.

Optional settings:
- **N-1 criterion (TRUE/FALSE)**  
  - If TRUE, the line must still transmit power when one circuit is lost.  
  - A single-circuit line cannot satisfy N-1.
- **Terrain factor**  
  - Multiplier to reflect construction difficulty:  
    1.0 = normal terrain, 1.2 = difficult, 1.5 = mountainous.

---

## 2. Understanding Transmission Line Capacity

The **capacity** of a transmission line (how much power it can carry) is not unlimited.  
It is typically constrained by three main physical limits:

1. **Thermal limit** – how much current the conductor can carry before overheating.  
2. **Stability limit** – how much real power can flow before the sending and receiving systems lose synchronism (angle stability).  
3. **Voltage limit** – how far voltage can be maintained along the line before it collapses due to reactive power imbalance.

Consequently:
- For **short or medium lines (< 200 km)**, the capacity is usually limited by **thermal heating** of conductors.  
- For **longer lines (200–400 km)**, **stability** (angle or voltage) starts to limit transfer.  
- Beyond **400–500 km**, **voltage control or compensation** becomes essential.

---

## 3. Step 1: Surge Impedance Loading (SIL)

**SIL** is a theoretical value representing the natural power level at which the line neither generates nor consumes reactive power.  
It depends on the **surge impedance (Zc)** of the line.

$$
SIL = \frac{V_{kV}^2}{Z_c}
$$

- **Zc (surge impedance)** is a property of the line determined by its geometry and conductor configuration.  
  For most high-voltage AC overhead lines, Zc ≈ 300 Ω is a reasonable average.  
- Example: for a 400 kV line, SIL ≈ (400² / 300) ≈ 533 MW per circuit.

SIL gives the **base scale** for how much power a line can carry under ideal conditions.  
Actual usable power will be below this due to stability and thermal limits.

---

## 4. Step 2: Stability Limit ($P_{stab}$)

As distance increases, the line reactance increases, and the angle between sending and receiving ends becomes harder to control.  
This defines the **stability limit**.

We approximate this by applying a **derating factor K(L)** that decreases with line length:

$$
K(L) = \frac{1}{1 + \frac{L_{km}}{L_0}}
$$

where:
- `L0` = "stability length" from the parameter table (e.g., 400 km for 220 kV, 250 km for 400 kV).

Then:

$$
P_{\text{stab}} = SIL \times K(L) \times N_{\text{circ}}
$$

- For short lines, K(L) ≈ 1 → almost full SIL available.  
- For long lines, K(L) < 1 → stability significantly reduces power transfer.

This is a **simplified way** to represent both **angle** and **voltage** stability effects.

---

## 5. Step 3: Thermal Limit (P_thermal)

The thermal limit corresponds to conductor heating due to electric current.

$$
P_{\text{thermal,circ}} = \frac{\sqrt{3} \times V_{kV} \times I_{\text{th,circ}}}{1000}
$$

where:
- `I_th,circ` = rated current per circuit (from the voltage table).

Examples:
- 220 kV line at 1500 A → 572 MW per circuit.  
- 400 kV line at 900 A → 624 MW per circuit.  
- 500 kV line at 2000 A → 1730 MW per circuit.

If there are multiple circuits:

$$
P_{\text{thermal,total}} = N_{\text{circ}} \times P_{\text{thermal,circ}}
$$

If N-1 is enforced:

$$
P_{\text{thermal,N-1}} = (N_{\text{circ}} - 1) \times P_{\text{thermal,circ}}
$$

Then the limit used is:
- If N-1 = TRUE → use $P_{\text{thermal,N-1}}$  
- If N-1 = FALSE → use $P_{\text{thermal,total}}$

---

## 6. Step 4: Voltage Stability (P_voltage)

For simplicity, the voltage limit is expressed as a multiplier of SIL:

$$
P_{\text{voltage}} = \beta \times SIL \times N_{\text{circ}}
$$

where β is a simple factor:
- β = 1.0 for normal, uncompensated lines,  
- β = 1.5–2.0 for strong or compensated systems.

This term can be ignored if the stability limit is already applied through K(L).

---

## 7. Step 5: Available Transfer Capacity

Finally, the model takes the **lowest** of all relevant limits:

$$
P_{\text{avail}} = \min(P_{\text{stab}}, P_{\text{thermal,limit}}, P_{\text{voltage}})
$$

This is the estimated **usable transfer capacity** in MW under the assumed conditions.

---

## 8. Step 6: Cost Estimation

### 8.1 Overhead Line Cost

$$
C_{\text{line}} = L_{km} \times N_{\text{circ}} \times C_{\text{unit,km}} \times \text{Terrain factor}
$$

- **C_unit,km** = cost per km per circuit from the parameter table (MUSD/km).  
- Includes towers, foundations, conductors, insulators, and installation.

Typical values:
- 220 kV: 0.35 MUSD/km  
- 330 kV: 0.45 MUSD/km  
- 400 kV: 0.50 MUSD/km  
- 500 kV: 0.80 MUSD/km  

These are average turnkey costs excluding substations and contingencies.

### 8.2 Substation Cost

$$
C_{\text{sub}} = N_{\text{sub}} \times C_{\text{unit,sub}}
$$

Usually two terminal substations (sending and receiving ends).  
Typical costs:
- 4–10 MUSD per 220–500 kV substation, depending on voltage and configuration.

### 8.3 Reactive Power Compensation

If applicable (for very long lines > 400 km):

$$
C_{\text{comp}} = N_{\text{comp}} \times C_{\text{unit,comp}}
$$

Typical compensation equipment:
- SVC, STATCOM, or series capacitors  
- 3–6 MUSD for 220–330 kV, 6–10 MUSD for 400–500 kV.

### 8.4 Total Investment Cost

$$
C_{\text{total}} = C_{\text{line}} + C_{\text{sub}} + C_{\text{comp}}
$$

Optional:
$$
\text{Cost per MW} = \frac{C_{\text{total}}}{P_{\text{avail}}}
$$

---

## 9. Interpreting the Results

The model outputs:
- **Available capacity (MW):** a rough estimate of how much power the line can carry considering physical limits and reliability.  
- **Total cost (MUSD):** an approximate investment cost including line, substations, and optional compensation.  

For **short lines**, the thermal limit usually binds (capacity increases with voltage and conductor size).  
For **longer lines**, stability constraints reduce the effective capacity even if conductors could carry more current.

These outputs should be seen as **order-of-magnitude indicators** — useful for comparing corridors, voltage choices, or reliability scenarios before more detailed engineering data becomes available.
