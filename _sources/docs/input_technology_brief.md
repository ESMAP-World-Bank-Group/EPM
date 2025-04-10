# Technology overview

This section describes the main generation and storage technologies modeled in EPM.

---

### ST (Steam Turbine)

**Description:** Historical thermal power plant design. Uses external combustion to heat water into steam (Rankine cycle), which drives a turbine. Less flexible but robust for base-load operation.

**Compatible Fuels:** Coal, lignite, biomass.

**Fuel differences:** Coal and lignite are standard solid fuels requiring large boilers and operating with relatively low thermal efficiency. Biomass introduces variability in fuel composition and moisture content, which can affect combustion stability and ramping capability.

**Technical characteristics:**  
Steam turbines are generally inflexible due to their reliance on large boiler systems and thermal inertia. They typically operate with a high minimum generation level (30–60%), as shutting down and restarting the boiler is costly and slow. Ramp-up and ramp-down rates are modest for traditional designs, though some modern plants allow improved flexibility. Heat rates are moderate to high, reflecting relatively low efficiency, especially with lignite or biomass. Their contribution to reserves is limited unless equipped for flexible operation. Capital costs are high due to the scale of the installation, and O&M costs vary depending on fuel type, being higher for solid fuels. Plant lifetimes are long, often exceeding 40 years.

---

### OCGT (Open Cycle Gas Turbine)

**Description:** A simple gas turbine operating on the Brayton cycle. Air is compressed, mixed with fuel, combusted, and the resulting hot gases expand through a turbine to produce electricity. No heat recovery is included.

**Compatible Fuels:** Natural gas, LNG (regasified), diesel, HFO.

**Fuel differences:** Natural gas and LNG offer clean combustion, fast response, and low maintenance. Diesel is slightly less clean but still suitable for fast-start backup. HFO requires preheating and atomization, resulting in slower ramping and more maintenance due to fouling and corrosion.

**Technical characteristics:**  
OCGTs are highly valued for their exceptional operational flexibility. They can ramp up or down extremely quickly, often within a few minutes, making them ideal for reserve and peaking applications. Minimum generation is close to zero, allowing full shutdown when not needed. However, this flexibility comes at the expense of efficiency: heat rates are high (typically above 9 MMBtu/MWh), particularly when operating on liquid fuels like diesel or HFO. Despite low capital costs and relatively short construction times, O&M costs are higher when using HFO or diesel. The ability to provide reserve services is excellent, especially when using gas, thanks to the fast start and response times.

---

### CCGT (Combined Cycle Gas Turbine)

**Description:** Combines a gas turbine (Brayton cycle) with a steam turbine (Rankine cycle) that recovers heat from the exhaust, significantly increasing overall efficiency. Commonly used for base or mid-merit load.

**Compatible Fuels:** Natural gas, LNG (regasified).

**Fuel differences:** Only clean gaseous fuels are suitable due to the sensitivity of the heat recovery system. Use of liquid fuels is not compatible.

**Technical characteristics:**  
CCGTs offer an optimal balance between efficiency and moderate flexibility. Their heat rates are significantly lower than OCGTs (typically around 6.4 MMBtu/MWh), making them one of the most efficient fossil-based technologies. Minimum generation levels are lower than steam-only plants, especially in modern flexible configurations, and ramp rates can be relatively fast (up to 100%/h) depending on plant design. While they are less responsive than OCGTs, CCGTs can still contribute to secondary reserve services. Capital costs are higher due to the additional steam cycle, but O&M costs remain moderate. Plant lifetimes are typically 30–40 years.

---

### Reservoir Hydro

**Description:** Hydroelectric plants with an upstream reservoir allowing full control of water flow. This enables highly dispatchable and flexible electricity production.

**Fuel:** Water.

**Technical characteristics:**  
Reservoir hydro offers outstanding operational flexibility. Generation can be ramped up or down very quickly, typically within seconds to minutes, allowing the plant to contribute effectively to all types of reserve services. Minimum generation can be set to zero when the plant is not operating, and overload capacity is generally high. While capital costs are significant due to dam and civil works, O&M costs are low, and plant lifetimes can exceed 50 years. Reservoir hydro is fully renewable and often plays a crucial role in balancing variable generation.

---

### ROR (Run-of-River Hydro)

**Description:** Hydropower plants without significant storage. Electricity generation depends directly on river flow, with minimal control over timing.

**Fuel:** Water.

**Technical characteristics:**  
Run-of-river hydro is not dispatchable, as generation follows the natural flow of the river. This limits flexibility and the ability to provide reserve services. Minimum generation is effectively determined by the river’s flow and cannot be adjusted. Ramp rates are very limited. Capital costs are lower than reservoir-based hydro, and O&M costs are minimal. Lifetimes are long. This technology is fully renewable but subject to significant seasonal and interannual variability.

---

### Nuclear

**Description:** Uses nuclear fission to generate heat, which is converted to steam to drive a turbine. Designed for large-scale, continuous base-load generation.

**Fuel:** Uranium.

**Technical characteristics:**  
Nuclear plants are optimized for steady, uninterrupted operation and have limited operational flexibility. Minimum generation levels are high (typically above 75%), and ramping is slow due to thermal constraints and safety considerations. Heat rates are high compared to fossil gas technologies, a result of the thermodynamic limits of the steam cycle. Nuclear plants generally cannot provide significant reserves, except with new load-following reactor designs. Capital costs are extremely high, O&M costs are substantial due to regulatory and safety requirements, but plant lifetimes can exceed 60 years.

---

### Geothermal

**Description:** Converts underground heat into electricity, either by using dry steam directly or via steam generated from high-pressure hot water (flash or binary systems).

**Fuel:** Geothermal heat.

**Technical characteristics:**  
Geothermal plants are typically operated as base-load units. Due to the need to maintain stable reservoir pressure and avoid thermal stress on the wells, output modulation is limited. Minimum generation levels are high and ramping is slow. Thermal efficiency is relatively low, resulting in high heat rates. Contribution to reserves is minimal. While capital costs—especially for drilling—are significant, O&M costs are moderate and lifetimes are long. Geothermal energy is fully renewable but geographically constrained.

---

### Biomass

**Description:** Burns organic materials to generate steam and produce electricity, usually through a conventional steam turbine setup.

**Fuel:** Biomass (wood, agricultural residues, waste).

**Technical characteristics:**  
Biomass plants behave similarly to coal-based steam turbines, with high minimum generation levels and limited ramping capability. Heat rates are high due to combustion inefficiencies and the variability in biomass composition and moisture content. Reserve contribution is typically low. Capital costs are high and O&M costs are significant, particularly because of the need to handle and process fuel. Lifetimes are long. Biomass is considered renewable when sourced sustainably, but fuel supply logistics are a critical factor.

---

### PV (Photovoltaics)

**Description:** Converts solar radiation directly into electricity using photovoltaic cells. No moving parts and no combustion.

**Fuel:** Solar irradiance.

**Technical characteristics:**  
PV generation is fully dependent on sunlight and thus inherently variable and non-dispatchable. Minimum generation is zero (e.g., at night), and ramp rates are not controlled by the system but by cloud cover and solar angle. PV systems cannot contribute to reserves unless paired with battery storage. Capital costs have fallen significantly, O&M costs are very low, and typical asset life is 25–30 years. PV is fully renewable and key for decarbonization strategies.

---

### Wind (Onshore and Offshore)

**Description:** Converts kinetic energy of the wind into electricity via rotating blades connected to a generator. Offshore installations benefit from more stable wind conditions.

**Fuel:** Wind.

**Technical characteristics:**  
Wind turbines produce variable power depending on wind speed and direction. Minimum generation is zero and ramping cannot be controlled—power may drop or spike quickly based on weather. These systems generally do not contribute to reserves unless supported by advanced controls or storage. Offshore wind offers higher capacity factors but entails higher capital and maintenance costs. Both types are fully renewable and essential components of the energy transition.

---

### Storage - Battery

**Description:** Electrochemical systems (typically lithium-ion) that store electricity and release it as needed. Ideal for short-duration flexibility and reserve provision.

**Fuel:** Electricity.

**Technical characteristics:**  
Battery storage offers excellent flexibility. It can ramp up or down almost instantaneously, provide fast-response reserves, and operate at full flexibility down to zero output. Round-trip efficiency is high (typically 85–90%). However, storage duration is generally limited to a few hours. Capital costs per MWh stored remain high, although declining, while O&M costs are relatively low. Lifespans depend on usage and cycling patterns, usually ranging from 10 to 15 years. Batteries are an essential enabler of high-renewable systems.

---

### Storage - Pumped Hydro

**Description:** Uses electricity to pump water to a higher reservoir, then generates electricity by releasing the water through turbines.

**Fuel:** Electricity (stored mechanically).

**Technical characteristics:**  
Pumped hydro provides large-scale, long-duration energy storage. It can be fully modulated, with zero minimum generation and fast ramping both up and down. Its round-trip efficiency is typically between 75% and 80%. While capital costs are very high due to infrastructure, O&M costs are low, and lifetimes often exceed 50 years. Pumped hydro is well suited for energy shifting, daily load balancing, and reserve provision, making it a cornerstone of flexible power systems.
