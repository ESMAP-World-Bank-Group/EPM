# Technology overview

This section quickly describes the main generation and storage technologies modeled in EPM.

---

## ST (Steam Turbine)

**Description:** Historical thermal power plant design. Uses external combustion to heat water into steam (Rankine cycle), which drives a turbine. Less flexible but robust for base-load operation.

**Compatible Fuels:** Coal, lignite, biomass.

**Fuel differences:** Coal and lignite require large boilers, have high emissions, and limited flexibility. Biomass introduces variability in combustion and may require dedicated infrastructure.

**Technical characteristics:** Steam turbines are relatively inflexible due to the inertia of the boiler-steam system. Minimum generation is typically high, around 30–60%, as it is costly and slow to cycle the plant on or off. Ramp rates are low to moderate, particularly for older designs. Heat rates are moderate to high, reflecting the relatively low thermodynamic efficiency of single-cycle steam setups. These plants generally cannot contribute much to reserve capacity unless explicitly designed for it. Investment costs are high, but lifetimes are long (over 40 years), and O&M costs are significant due to the fuel type and ash/fouling issues.

---

## OCGT (Open Cycle Gas Turbine)

**Description:** Simple gas turbine using the Brayton cycle. Air is compressed, fuel is injected and combusted, and the resulting hot gases expand through a turbine. No heat recovery. Widely used for peak load and reserves.

**Compatible Fuels:** Natural gas, LNG (regasified), diesel, HFO.

**Fuel differences:** Natural gas offers the cleanest and fastest operation. Diesel allows for flexibility in backup systems. HFO is more polluting and requires preheating, with slower response and more frequent maintenance.

**Technical characteristics:** OCGTs are characterized by their excellent flexibility, with very fast ramp-up and ramp-down rates (close to 100% of capacity per hour) and the ability to shut down completely when not needed. Minimum generation is nearly zero, making them ideal for standby or peaking purposes. Heat rates are relatively high (inefficient), especially compared to combined cycle systems, but acceptable given their role. Their contribution to reserve services is excellent due to their speed. Capex and O&M costs are moderate, and plant life is generally around 25–30 years.

---

## CCGT (Combined Cycle Gas Turbine)

**Description:** Combines a gas turbine with a steam cycle to recover heat from the exhaust and improve efficiency. The most efficient fossil-based technology, used for base or mid-merit load.

**Compatible Fuels:** Natural gas or LNG (regasified).

**Fuel differences:** Only clean gaseous fuels can be used, due to the sensitivity of the heat recovery system.

**Technical characteristics:** CCGTs offer a balance between efficiency and flexibility. Heat rates are low (i.e., efficiency is high, around 55–60%), especially for modern designs. Minimum generation can be reduced to around 10–15% for advanced units. Ramp rates are moderate to high, especially if designed for flexible operation. They can provide reserve services, though less responsively than OCGTs. Capital cost is higher than OCGTs due to the additional steam cycle, but O&M remains moderate. CCGTs have lifespans of around 30–40 years and are often used to balance variable renewables.

---

## Reservoir Hydro

**Description:** Hydropower plant with upstream reservoir allowing full control over water flow. Provides highly dispatchable renewable electricity.

**Fuel:** Water.

**Fuel differences:** None.

**Technical characteristics:** Reservoir hydro offers unmatched operational flexibility. It can start or stop quickly, making minimum generation effectively zero. Ramp rates are extremely fast, suitable for all types of reserve services. It contributes significantly to both energy and system stability. Although capital costs are high due to civil works, O&M costs are very low. Lifetimes exceed 50 years. It is considered fully renewable and often used for peak shaving or balancing.

---

## ROR (Run-of-River Hydro)

**Description:** Hydropower plant with little to no storage. Generation follows river flow conditions and is only partially controllable.

**Fuel:** Water.

**Fuel differences:** None.

**Technical characteristics:** ROR plants have minimal flexibility since their output depends on inflow conditions. Minimum generation is tied to river availability and can’t be adjusted. Ramp rates are limited, and these plants contribute very little to reserves. Capex is lower than large reservoirs, O&M is minimal, and lifetimes are long. Fully renewable, but seasonally variable.

---

## Nuclear

**Description:** Uses nuclear fission to generate heat and produce steam for a turbine. Designed to provide large-scale, stable base-load electricity.

**Fuel:** Uranium (or other fissile materials).

**Fuel differences:** None.

**Technical characteristics:** Nuclear plants are designed for stable operation, not flexibility. Minimum generation is typically high (often >75%) due to thermal and safety constraints. Ramp rates are low, and reserve contribution is limited. Heat rate is relatively high, reflecting the thermodynamic limitations of steam systems. Capital investment is extremely high, O&M costs are also significant, but plant lifetime is very long (40–60 years). Though not renewable, it is a low-carbon technology.

---

## Geothermal

**Description:** Utilizes underground heat to generate steam and produce electricity, either through direct steam or binary cycle plants.

**Fuel:** Geothermal heat.

**Fuel differences:** None.

**Technical characteristics:** Geothermal plants behave similarly to nuclear in operational terms: steady, base-load output with limited flexibility. Minimum generation is often high due to the need to maintain pressure in geothermal reservoirs. Ramp rates are limited, and heat rates are high due to low thermal efficiency. Reserve contribution is minimal. Capex is high (especially drilling), O&M is moderate, and lifetimes are long. Fully renewable but site-constrained.

---

## Biomass

**Description:** Burns organic material to generate steam, similar in process to coal-fired steam turbines.

**Fuel:** Biomass (wood, residues, organic waste).

**Fuel differences:** Moisture and composition can vary significantly, affecting combustion efficiency and fouling.

**Technical characteristics:** Biomass plants share many characteristics with coal-fired steam turbines: high minimum generation, low ramp rates, and high heat rates due to external combustion. Their ability to contribute to reserves is limited. Capex is significant (similar to coal), and O&M costs are high due to complex fuel logistics. Lifetimes are long. Renewable status depends on fuel origin and sustainability.

---

## PV (Photovoltaics)

**Description:** Converts solar radiation directly into electricity using semiconductor materials. No moving parts and no combustion.

**Fuel:** Solar irradiance.

**Fuel differences:** None.

**Technical characteristics:** PV is fully non-dispatchable and only produces during daylight hours. Minimum generation is zero, and ramping is not controllable—it depends entirely on solar availability. It cannot contribute to reserves unless coupled with storage. Capex has decreased significantly, O&M costs are very low, and lifetimes are around 25–30 years. Fully renewable.

---

## Wind (Onshore and Offshore)

**Description:** Converts wind energy into electricity using turbines. Offshore systems benefit from higher and more consistent wind speeds.

**Fuel:** Wind.

**Fuel differences:** Onshore is cheaper and easier to install, offshore is more productive but costlier.

**Technical characteristics:** Wind energy is non-dispatchable and variable. Minimum generation is zero, and ramping depends on weather. Wind cannot provide reserves on its own. Capex is moderate to high, O&M is moderate (higher for offshore), and lifetimes are around 20–25 years. Fully renewable, but requires balancing.

---

## Storage - Battery

**Description:** Electrochemical systems (e.g., lithium-ion) used to store electricity and provide rapid injection or withdrawal of power.

**Fuel:** Electricity (charged and discharged).

**Fuel differences:** None.

**Technical characteristics:** Batteries offer excellent flexibility. Minimum generation is zero, and ramp rates are nearly instantaneous. They are ideal for reserve provision, frequency control, and short-term balancing. Round-trip efficiency is high (85–90%). Capex per MWh stored is high, but O&M is low. Lifespans depend on cycling, typically 10–15 years. Essential for high-renewable systems.

---

## Storage - Pumped Hydro

**Description:** Stores electricity by pumping water to a higher reservoir and generating power when released.

**Fuel:** Electricity (stored mechanically via water elevation).

**Fuel differences:** None.

**Technical characteristics:** Pumped hydro offers large-scale, long-duration storage. It has near-zero minimum generation, fast ramp rates, and excellent reserve provision capability. Round-trip efficiency is around 75–80%. Capex is very high, but O&M is low and lifetime is very long (50+ years). Well suited for energy shifting and renewable integration.
