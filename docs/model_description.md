
# Description of the model

EPM cand be used (i) as a least-cost capacity expansion model to optimize investments in the power sector over a specified horizon that could extend from a few years to decades or as (ii) an economic dispatch model to optimize the operation of a specified power system over a relatively short period compared to Capacity Expansion Model (1-week to 1-year). Like every optimization software the problem is mathematically described through an objective function and several accompanying equations (system constraints). System constraints are used to describe mathematically the physical part of the system, laws of physics, policy targets and other realities specific to the system (for example fuel availability). More detailed description of system constraints can be found on subsequent sections.

## Indices and Sets

All sets used in the formulation can be classified in two major categories: time and power system-related. Four sets belong to the first category: which represent different time scales considered in the model: days, quarters, hours and years. Hour is the smallest unit of time used in this formulation and we could use the same formulation using just one set for time containing as many hours as the set . It is convenient though to keep all the four sets since they reveal some fundamental assumptions of the model: (1) days are used to reflect the chronological sequence of the time slices used for ramping and storage constraints as we will further explain in the constraints section; (2) quarters are used to reflect seasonality in the load patterns, the availability of thermal power units and the thermal limits of transmission lines; (3) years are used to represent annual trends on demand growth and keep track of the lifetime of units; and finally (4) hours are commonly used as the smallest time unit in long term models since the day ahead scheduling models schedule generation units on an hourly basis.

Three sets are power system related. Set includes all generating units. Depending on the size of the system, we might decide to use set to model individual units of the power system for a small system or aggregated units that represent multiple units of the same technology for a large system. We use the term technology to refer to different technologies or different fuels used: e.g. coal steam turbines, natural gas combined cycle, natural gas combustion turbines, wind farms, solar photovoltaic panels, geothermal, hydropower and diesel generators. Depending on the resources available in a country, some technologies might not be present. As the model stands now, elements of set are mapped to sets and , which stand for fuel and zones respectively. Set is one of the major sets used in power systems since the power system is a network and physical laws (widely known as Kirchhoff’s laws) govern the flow of power over the transmission lines. Given that, a set such as set which captures the spatial dimension of the system is necessary. At the finest granularity, set might contain buses of the power system but in case we model larger systems, set might contain zones of a power system or even countries. Note that the modeler usually decides on the spatial granularity based on the presence of common regulatory rules or pricing schemes in a zone or/and based on the congestion observed on transmission lines connecting adjacent regions. Finally, set includes the different fuels used and we model it to keep track of the consumption of different fuels since for certain fuels domestic upper bounds on consumption might apply and/or issues of energy security might be involved in case of imported fuels. In addition, different types of fuels have different carbon content and lead to different emissions of carbon dioxide, which are important to track in case environmental policies exist.

## Objective function

The objective function in this model minimizes the total system cost including violation/penalty terms for constraints that are not met. All generation, transmission and emissions costs of the system are considered including: (1) Fixed costs including annualized capital cost payments for new generators<sup>[\[4\]](#footnote-5)</sup> and any additional transmission capacity and fixed operation and maintenance costs (2) variable costs including the fuel cost and any variable operation and maintenance costs (3) cost to procure spinning reserves (4) carbon tax payments and (5) penalties for unmet demand and unmet capacity and spinning reserve requirements, breaching the emissions and renewable rejection at the system or the zonal level.

In the case of a least-cost expansion model the objective function typically represents total system costs over the study horizon. Depending on the type of the problem (for example linear or non-linear) the user chooses an appropriate solver to minimize the value of the objective function. Total system costs are broken into several components the most important of which are the fixed and variable costs of electricity generation. CAPEX costs typically dominate fixed costs while fuel costs dominate variable costs of generation. In systems with very high shares of renewables VOM costs are higher than fuel costs. Other types of system costs included in the objective function are costs of reserve provision by generators, the cost of carbon (when applicable), the cost of electricity trading (it could be a net benefit if revenues from electricity trade exceed costs) and the cost of transmission enhancements.

When EPM is used as a dispatch model, there are no capex costs in the objective function. In dispatch mode, the model does not consider unit commitment related decisions in its current form, albeit minimum generation constraints and ramp rates are modeled.

## Load approximation

The size of the optimization problem (i.e. the number of decision variables to be determined) depends significantly on the study horizon and the optimization step. An EPM user needs to find the balance between precision and computational effort (practicality) to complete the analysis. This is especially important when little gains in precision impact significantly the running time. In most cases representing the full year at the hourly level (8760 hours) over a study horizon that extends to multiple years, usually 15 to 20 years, is neither practical nor required to assess in detail the capacity mix of most supply options. For capacity expansion analyses, the load is typically approximated through a number of representative days which are assumed to repeat several times throughout the calendar year. Such representative days are carefully selected to capture the hourly profile (including the peak) of electricity demand on a monthly or quarterly basis. It is common practice to select the following three types of representative days: (i) an average day which is assumed to be representative of the bulk of electricity production (repeats several times over a quarter (or month)), (ii) a typical day of high demand (which also need to include the peak) and (iii) a typical day of low demand. The last two types of days typically represent a much smaller portion of the quarterly (or monthly) demand compared to the average day. The EPM input file includes tabs where a user can provide the hourly profile of each of the selected typical days on the “Demand” tab. Also, a user needs to provide the relative contribution of each typical day on total quarterly/monthly/seasonal demand represented as number of repeats over a typical quarter (or month), in the “Duration” tab.

The above-mentioned representation of the load has been widely used in power sector analysis to estimate the capacity mix of conventional supply options both efficiently and accurately. However, application of this methodology has weaknesses in power systems with very high shares of variable renewable energy (VRE) sources. In such systems detailed representation of chronology might have significant impact on results. VRE generation profiles are inputs on EPM. Selection of a representative day for load need to be accompanied by the respective VRE generation profile that corresponds to the selected day. This requires extensive data processing.

A weakness of selecting representative demand days is that the analysis might undersize electricity storage, the simulation of which requires full description of chronology. This is especially true in the case multi-hour and seasonal storage. For that reason, running EPM on dispatch mode (full chronology) of a full (selected) year might be required to supplement capacity expansion analysis.

An EPM user can represent demand in detail for each year of the study horizon or provide annual projections (single value) of demand and peak demand together with typical day profiles over a base (historical) year. When the latter option is activated, the model uses the above data to calculate the projected demand in detail.

### Value of Lost Load

The Value of Lost Load (VoLL) is an economic penalty used to ensure that the model will invest to meet the demand and avoid unserved energy if it is economic to do so. The VoLL represents the damage on the local economy from missing to supply 1MWh of demand. Without assigning VoLL matching supply and demand would be economically suboptimal as it is cheaper to fall short on supply rather than paying for the cost of electricity generation and associated investments. Thus, the selection of VoLL need to be based on research. There is no universally acceptable Value of Lost Load and different methods have been applied to estimate a reasonable value for the unserved energy. Typical values used in developed economies vary between 4,000 and 40,000$/MWh while in developing countries between 500 and 10,000$/MWh \[1\].

VoLL affects both the total-non served energy in the system and peaking supply investment decisions. In some cases, allowing for some loss of load might be preferable compared to investing into supply options that will only operate for a tiny amount of the annual time. The user will need to calibrate the model considering the local realities and system rules since the maximum allowed number of hours failing to meet demand is constrained by reliability criteria. When a value for VoLL is not available in global literature the EPM user can select a reported VoLL from a country with similar economic characteristics. Alternatively, the assessment of VoLL can be based on analytical expressions or based on the assumption that demand is sensitive to electricity price. In the latter case VoLL can take the value of cost of generation of the most expensive asset in the system which are usually backup diesel generators \[2\].

Typically, back-up generators are fueled with expensive diesel, and if they are used in the mode just described, the system VOLL would take the value of the variable cost of these generators (~500 $/MWh).

## Transmission network constraints

Kirchhoff’s laws are physical laws governing the flows over transmission lines in a network. According to first Kirchhoff law, also known as KCL (Kirchhoff’s Current Law), the sum of injections in a node should equal zero. In our formulation, KCL corresponds to equation _(14)_. There, we can see that the power provided by generators and storage should be equal to demand (minus the unmet demand) plus/min outflows/inflows from the node to adjacent nodes. This equation additionally includes price-driven import and export volumes from external zones. The demand is adjusted with the potential improvements in the energy efficiency using parameter.

|     | (14) |
| --- | --- |

Note that the second Kirchhoff Law (or widely known as KVL- Kirchhoff’s Voltage Law) is valid for power systems but for reduced power system, it might not apply depending on the method of network reduction followed. In this particular formulation, KVL is not considered.

Another important feature of our model relates to modeling of transmission losses. We model transmission losses as a percent reduction of the imported electricity at each node. In particular, term

at equation _(14)_ models injections to node _z_ and we can see how the loss factor reduces the amount of energy imported. On the contrary, the outflow is fully considered at the origin node of the network: .

Another common constraint for transmission networks refers to the capacity limits of transmission lines. In particular, as equation _(15)_ implies, the flow over a specific line cannot exceed a certain limit, which is defined either by thermal limit of the line or upper bounds imposed by reliability considerations. Note that we model flows over a particular transmission line with two positive variables, one for each direction. Please observe that the transmission limit parameter might change per year to reflect planned upgrades and the additions to the transmission network. There are also maximum built limits of additional transmission capacity between zones due to budget or technical constraints. Moreover, the transmission limit differs per season since ambient temperature affects the capacity available for power transfers.

| +   | (15) |
| --- | --- |

The modelled zones, countries and regions are usually not isolated from the external systems and markets not included in the model. Therefore, we allow trading with these zones at time-varying predefined price. Equations _(16)_, _(17)_ and _(18)_ define the limits of this exchange. Equation _(16)_ sets the annual limit on the energy imported from external zones as a percentage of the annual demand using parameter . Equations _(17)_ and _(18)_ introduce the hourly limit on the imported and exported power, defined as a percentage of hourly demand.

## System requirements

In our formulation, we model two products that the system operator might require generators to provide during operation: (1) energy and (2) spinning reserves. Per NERC’s definition, spinning reserves refers to “unloaded generation that is synchronized and ready to serve additional demand” \[3\]. Note that more products exist, especially in organized U.S. wholesale markets such as non-spinning reserves or flex ramp. Operating reserves (spinning and non-spinning reserves) provide the capability above firm system demand required to provide for regulation, load-forecasting error, equipment forced and scheduled outages and local area protection \[3\]. Moreover, under the spinning reserves different products might exist with respect to the response times required etc.

The amount of spinning reserve required depends on several factors that the planner/operator considers such as the load level and the associated forecasting error, the forecasting error attached to the renewable generation and the size of the largest unit committed on the system to be able to accommodate N-1 outages. Equation _(19)_ indicates that spinning reserve can be provided by interconnections. On top of system-wide reserve requirements, zonal requirements apply to accommodate for outages on transmission lines connecting adjacent regions/zones/nodes.

|     | (19) |
| --- | --- |

Planners usually consider a planning reserve margin (PRM) to account for forecasting error in demand projections. Typical values for the PRM vary between 10-15%. Equation _(21)_ indicates that interconnections can be accounted for as reserve margin. Note that intermittent units do not contribute towards the planning reserve constraint at their full capacity but at a fraction specified by the planner (using the Capacity Credit parameter), e.g. in the U.S. markets this fraction is calculated based on available historical data as the capacity factor during a set of peak hours \[4\].<sup>[\[5\]](#footnote-6)</sup>

|     | (21) |
| --- | --- |

## Generation constraints

Spinning reserve capacity is modeled as it is a very important dimension of a system flexibility and ability to integrate VRE.

|     | (22) |
| --- | --- |

Equation (22) assures that the power generated by the unit along with the spinning reserves provided by the same unit do not exceed the unit’s capacity.<sup>[\[6\]](#footnote-7)</sup> Note that the capacity is augmented by an overload factor. This factor is typically 10% for those generators that can handle overload conditions for a short period of time, and zero for those generators that cannot handle such conditions.

Given that spinning reserve products are usually defined with respect to response time of generator to a certain dispatch signal, only a certain percentage of the generator’s unit qualify as a reserve offer. We capture this characteristic in the model through equation (23).

|     | (23) |
| --- | --- |

Ramping constraints acknowledge that the generation units have inertia in changing their outputs and differences in generation outputs between consecutive hours should be constrained by the ramping up and down capabilities of the unit.

|     | (24) |
| --- | --- |
|     | (25) |

Another important feature of generators is the minimum load. The minimum load can either be determined based on technical specifications provided by the manufacturer or be calculated as an “economic” minimum beyond which the unit can provide energy economically. The minimum load constraint is really important for unit commitment and requires the use of binaries variables that make sure the constraint is enforced when the unit is on. However, in generation expansion models operations are approximated through a simple dispatch model for representative hours of the year. In this case, if deemed important, an approximation of the minimum load constraint can be applied for the thermal units for which it is relevant. The constraint can be applied for a subset of the days modeled and the constraint is activated only for those days or to all days. Constraint (26) is forcing all units to generate power equal to at least their minimum loading levels for specific days in the year.

|     | (26) |
| --- | --- |

Generating units require maintenance every year. Given that, we should consider the units as unavailable for certain periods during the year. In this particular application, we consider a uniform availability factor per quarter to account for maintenance.

|     | (27) |
| --- | --- |

## Renewable generation modeling

Renewable generation differs from conventional units in that its output is, to a certain extent, uncontrollable and intermittent. The power generated by renewables such as wind or solar depends on wind velocity or solar irradiation. Collecting historical data that records weather information (such as wind speed, temperature, wind direction, etc.) or the power generation output by installed renewables at specific locations, analysts usually employ statistical methods such as k-means to reduce the number of hours required to approximate the intermittent nature of renewables \[5\]. In this particular application, the generation profile for each renewable energy technology (such as wind or solar PV) is defined by the hourly capacity factor, in a year, of a generic power plant of each type, modeled at a specified location. Then, given this hourly profile for a year, we choose the amount of days modeled based on the days selected for the load approximation (see Section 1.4.3), i.e., the renewable profile during the 12 days in the year selected, for load, is maintained.

|     | (28) |
| --- | --- |

Note that the renewable profile is highly dependent on the region/location the resource is located. This formulation implicitly models that aspect since might have different elements for the same generation technology at different locations.

## Concentrated Solar Power (CSP) modeling

CSP technology modeling differs from other renewable technologies due to the complexity derived by its storage capabilities. The CSP configuration considered in this model consists of two integrated subsystems; these include the thermal storage system, and power cycle. Thermal storage is modeled using a simple energy balance approach that includes charging and discharging energy. The power cycle model provides a simple mechanism for modeling the conversion from thermal energy output from the solar field, and thermal storage into electrical energy.

|     | (29) |
| --- | --- |

Equation (29) indicate that at any time the CSP storage level cannot exceed its storage capability.

|     | (30) |
| --- | --- |

The power output of the solar panel is calculated by multiplying the nameplate capacity of the CSP power plant, the capacity factor of the system, and the solar multiple, then, dividing this by the turbine and solar field efficiencies (Equation (30)).

|     | (31) |
| --- | --- |

Equation (31) indicate that all the power output produced by CSP generators at any given zone, cannot exceed the nameplate capacity. Finally, Equations (32) and (33) detail the power balance formulations for the power cycle and thermal storage subsystems.

|     | (32) |
| --- | --- |

|     | (33) |
| --- | --- |

## Time consistency of power system additions and retirements

We use constraint (34) to track the capacity in consecutive years. In particular, generation capacity at year equals capacity at previous year plus any investment minus any retirement at year .

|     | (34) |
| --- | --- |

Several more constraints are formulated to fix the capacity at pre-specified levels in certain years.

- The first constraint sets the capacity of the existing units in the first year to the predefined capacity, if the commissioning year is earlier then the first year of the optimization horizon.

<table><tbody><tr><th></th><th><p>(35)</p></th></tr><tr><td></td><td></td></tr><tr><td colspan="3"><ul><li>The second constraint sets the capacity of the existing units to the predefined capacity, for all the years exceeding the commissioning year, but before the defined retirement year.</li></ul><table><tbody><tr><th></th><th><p>(36)</p></th></tr><tr><td></td><td></td></tr></tbody></table></td><td></td></tr></tbody></table>

- The third constraint forces the capacity of the existing units to 0, when the retirement year is reached.

|     | (37) |
| --- | --- |
|     |     |
|     |     |

- The fourth constraint states that the total capacity of a new generator equals the capacity of that new generator built the previous year plus the capacity to be built the current year:

|     | (38) |
| --- | --- |

- The fifth constraint forces the capacity at the first year of the horizon for new units to be equal to the capacity installed that year

|     | (39) |
| --- | --- |

- The sixth constraint limits the capacity to the predefined available or installed capacity of the unit.

|     | (40) |
| --- | --- |

- The seventh constraint forces the capacity of planned and candidate units at zero for years preceding the commission year of the unit.

|     | (41) |
| --- | --- |

## Storage modeling

Economically efficient storage in power systems has mainly been pumped hydro storage for a considerable number of years. Nowadays, more storage technologies such as battery energy storage systems are being added to the power system providing energy and/or reserve services.

Storage is modeled differently compared to conventional units since it requires two more variables: (1) one variable to keep track of the storage level (storage) and (2) one variable (_storageinj_) to model the injection of energy into the storage unit. The variable _gen_ used to track the generator output of conventional units is used to account for the output of a storage unit when it is discharged. Moreover, the chronological sequence of the time slices is important to make sure that the simulated operation is feasible e.g., we cannot discharge a storage unit if charging of the unit has not preceded. Finally, storage of energy requires the conversion of electricity to another form of energy e.g., mechanical for flywheels or chemical for fuel cells and common batteries. The conversion of one form of energy to another involves losses that we should consider in our models. Equations (30) and (31) represent the storage balance equations on the first hour of a representative day and any other hour of a representative day respectively:

|     | (42) |
| --- | --- |
|     | (43) |

The first equation (42) makes sure that the battery is discharged at the start of each representative day. The choice for not making a link between different days and thus having the battery discharged at the start of each day is because the standard formulation of the model works with representative days (e.g peak load, minimum load, and average load days) instead of actual days. Equation (43)in turn keeps track of the energy stored in the storage unit between consecutive hours of the same day: the energy stored in the unit at time slice _t_ (t > 1) equals the energy stored in the unit at time slice _t-1_ plus any injection reduced by the storage efficiency minus any discharge at time _t_. In case the storage unit is linked to a PV plant, the injection is given by the PV profile:

|     | (44) |
| --- | --- |

Two additional constraints are used to make sure that the operation of storage in each time slice is feasible considering the peak storage capacity (in MW):

|     | (45) |
| --- | --- |
|     | (46) |

The first constraint ensures that the hourly energy injection is lower or equal than the maximum hourly energy injection. The second constraint limits the total amount of energy (in MWh) stored to the maximum storage level (in MWh). Storage outputs are constrained by the general equation (22).

Furthermore, the storage units can be charged or discharged at a rate, which cannot exceed a specific value. To model this behavior, we include constraints (47) and (48) respectively.

|     | (47) |
| --- | --- |
|     | (48) |

Three equations keep track of the deployed energy storage capacity (in MWh) of the storage units:

|     | (49) |
| --- | --- |
|     | (50) |
|     | (51) |

Equation (49) keeps track of the energy storage capacity for existing storage units whereas equations (50) and (51) are used for new storage units.

Three final constraints limit the total deployed energy storage capacity (in MWh):

(52) (53)

(54)

The first constraint ensures that the deployed energy storage (in MWh) is larger than the peak storage capacity (in MW) at any time.<sup>[\[7\]](#footnote-8)</sup> The second constraint limits the total deployed energy storage (in MWh) to the maximum deployable energy storage (in MWh). Equation (54) constrains the deployment of existing storage units for which the commission year is later than the starting year of the modelling horizon.

## Investment constraints

Planners consider several constraints when they decide on a generation investment plan. Common constraints refer to budget, land use, scheduling of new construction and consumption of specific fuels for energy security considerations.

Constraint (55) usually reflects land use considerations, regulation that imposes an upper bound on capacity of specific technologies or simply resource potential (e.g., for wind there is a finite amount of locations where wind farms can provide the capacity factor modeled).

|     | (55) |
| --- | --- |

Constraint (58) is usually employed to reflect practical limitations on construction and spread the construction of new units more uniformly over time. For example, it seems pretty unrealistic that the whole system capacity can be built in one year.

|     | (58) |
| --- | --- |

Constraint (62) imposes an upper bound on fuel consumption. This upper bound might correspond to the fuel reserves a country might have at its disposal or the capacities of refinement units or importing units such as size of LNG terminals. Constraint (63) simply estimates the fuel consumption. Note that in case we want to reduce the number of variables in our model, we can get rid of the fuel variable since it is defined in terms of the generation variable.

|     | (62) |
| --- | --- |
|     | (63) |

Constraint (64) represents a budget constraint. It limits the capital expenses withdrawal to be lower than a pre-specified amount. In this formulation, we assume that the _MaxCapital_ parameter is similar to the maximum debt payments that a power system planner can do over the horizon.

|     | (64) |
| --- | --- |

An alternative constraint that addresses the same concern but relies on different information is expressed by constraint (69). In that case, the planner does not know the maximum amount of debt payments that the power plant owners might do but he has a good understanding of the maximum capital available to the system for investment. In that case, the sum of the overnight capital expenditure is not allowed to exceed this known budget.

|     | (69) |
| --- | --- |

## Environmental policy

Environmental concerns often lead policymakers to adoption of caps on the amount of emissions of certain pollutants. Several countries have announced targets to reduce their carbon dioxide emissions below particular amounts. Given that the power system is a big contributor to carbon dioxide emissions, power system caps are usually enforced at the country or at multi-country level. Constraints (65) and (66) impose the caps decided by regulators. Note that constraints (67) and (68) are modeled as “hard” constraints with no violation allowed. We would like to note that environmental policy might impose caps on more pollutants beyond carbon dioxide. For example, in the USA certain caps apply on and emissions. The dual variables of constraint (66) and (68) provide valuable information to the planner since they estimate the additional cost the planner would have to bear to decrease the amount of the pollutant by an infinitely small amount. In other words, the dual prices of (66) and (68) provide an approximation of the prices that the planner would pay per unit of pollutant if slightly stricter regulation would be applied.

|     | (65) |
| --- | --- |
|     | (66) |
|     | (67) |
|     | (68) |

Another policy mechanism related to carbon emissions is a carbon tax (less popular than the cap-and-trade system at present). We model the carbon tax as part of the objective function. Note that the carbon tax does not correspond to an actual cost for the society since it is a transfer payment for emitters to the government. It reflects an actual cost, though, only if it attempts to monetize the public health cost and the damage to the environment. However, it reflects an actual cost for power system since generators would probably have to pay the tax to the government and that explains why it is part of the objective function (8).

|     | (8) |
| --- | --- |

## Modelling CCS

CCS technology investments can be modelled on EPM as either greenfield or retrofits of existing coal plants. A new CCS plant is modelled similarly as any other coal plant with its techno-economic parameters adjusted. Typically, CCS plants are modelled to have lower efficiency and higher CAPEX and OPEX compared to conventional coal plants. The environmental impact of CO2 storage is modelled through setting fuel burned on a CCS plant to have reduced emissions factor compared to conventional coal (typically reduction 90% to 95%).

CCS retrofits are modeled as a sudden change on the techno-economic parameters of an existing plant subject to some CAPEX incurred. More specifically, this change is modelled as a decommissioning of the existing plant and a simultaneous commissioning of the retrofitted plan. The retrofit CAPEX and the updated techno-economic parameters need to be carefully selected based on international experience. The mathematical addition of CCS retrofit on EPM is done through equation 69 below:

(69)

Set mapGG containing the relevant pairs of the existing plant (candidate for retrofit) and the new plant (retrofitted plant) is automatically generated through the inputs development phase. The relevant connection (pairs of plants) is made by the EPM user on excel tab named LinkedComDecom

## Modelling of Green Hydrogen

Modeling of production of green hydrogen is possible on EPM. When the option is activated investments on electrolyzers are optimized similarly as with supply side technologies. The user provides the relevant technoeconomic inputs for electrolyzers (CAPEX, OPEX, efficiency, project lifetime etc). There are two potential uses of green hydrogen in EPM:

1. Re-circulation in the power sector. In that case green hydrogen acts as a carrier of renewable energy providing long-term storage to the system. Hydrogen can be burned in modified new CCGTs/OCGTs at 100% per volume or burned in existing dual fuel generators (future versions of EPM will include option for pre-specified per volume mix of hydrogen with natural gas to be burned in conventional turbines). The user needs to define hydrogen as an additional fuel with zero cost and zero emissions factor since it is produced within the system and not imported. Costs of hydrogen are indirectly accounted through the additional investments and relevant OPEX within the system. Also, techno-economic characteristics of new hydrogen turbines need to reflect published figures based on international experience
2. Used in other end-use sectors (residential, industry, transport). This is modeled as an external demand of green hydrogen to be satisfied subject to some penalty for each mmBTU of unserved hydrogen demand. EPM simulates power sector operation and investment requirements to satisfy external hydrogen demand but it does not model any subsequent use outside the power sector. The costs for external hydrogen production are accounted for in the model and thus the power sector is assumed to bear the costs of decarbonization of other sectors.

Description of green hydrogen formulation on EPM follows below:

Investments on electrolyzers are mathematically constrained to dictate time consistency of capacity additions and retirements, similarly as with generators:

Generation capacity of existing electrolyzer capacity at year equals capacity at previous year plus any investment minus any retirement at year where y is not the first year of the study horizon (equation 70)

|     | (70) |
| --- | --- |

Equation (71) is the same constraint as (70) modified for candidate capacity

|     | (71) |
| --- | --- |

Equation (72) constraints total electrolyzer capacity at the end of the first year of the optimization horizon. At the beginning of horizon existing capacity is the one defined in the inputs section and by the end of year the capacity is adjusted by any optimized additions or retirement throughout the year.

|     | (72) |
| --- | --- |

Total electrolyzer capacity that can be developed into mandatory projects (decision for development has been made) can not exceed the maximum electrolyzer capacity provided as input

|     | (73) |
| --- | --- |

Total new electrolyzer capacity that can be developed in candidate projects (decision to be optimized) cannot exceed the maximum electrolyzer capacity provided as input

|     | (74) |
| --- | --- |

Electolyzer candidate investments can also be set as integer decisions. In such cases the defined capacity can either be built in one step or stay out of the plan (equation 75). Similarly retirement of existing electrolyzer plant takes place in a single step (76). In equations (75) and (76), variables and can only take positive integer values

(75)

(76)

Equation 77 puts a constraint on operational time of electrolyzers so that it doesn’t exceed their technical availability

Equation 88 describes the thermal balance of hydrogen within the system (in mmBTU). Total mmBTU of hydrogen produced are equal to hydrogen demand for other sectors minus unmet hydrogen demand plus hydrogen recirculate in the power sector as fuel.

Equation 79 makes a connection between quarterly an annual amount of hydrogen fuel produced in the system

Equation 80 puts a constraint on total amount of electricity drawn by an electrolyzer. Max power absorbed can’t be more than the electrolyzer capacity

Equations 81 and 82 define ramping constraints for electrolyzer operation

Equations 83 sets an energy balance on renewable electricity. Total renewable electricity generated is equal to renewable electricity consumed by end-use sectors and renewable electricity consumed by electrolyzes

Equation 84 makes a connection between renewable power produced and electricity drawn by electrolyzers

Finally, equations 85 and 86 are mapping renewable electricity produced on a zonal basis which is necessary for the architecture of the hydrogen model

## Limitations of the model

As any model, this model also has some limitations in most practical system applications. Two major limitations of the model are the following:

(1) Transmission network representation: Although in theory, the model can be used to represent the high voltage transmission network down to individual substations, the size limitation of the underlying LP, typically restricts to a small (below 20) number zones and we assume zero congestion within each zone modeled. Transmission expansion is not always an option. The model does not consider voltage angles and DC approximation to power flow in its current form and as such may have unrealistic flows around a loop.

(2) As noted before, the model does not consider unit commitment (startup and shutdown) decisions. It also does not consider heat rate curves associated with thermal units and uses a single average heat rate for each generating unit.

(3) In its main version, the model is deterministic and no uncertainty with respect to assumed parameters etc. is considered.

(4) Hydropower: reservoir optimization is currently not a feature of EPM hence hydropower potential is represented as daily/seasonal/annual energy limits for individual hydro plant/unit. The main version model does not consider any representation of river chain.

(5) In the current model version, coal plant flexibilization or repurposing is not considered.

(6) When the model is used as a capacity expansion tool, the use of representative days and seasons limits the possibility to capture the VRE variability, hence the VRE integration costs are approximated. Similarly, the flexibility costs of the system are approximated as ancillary services are not all fully modeled.

## Bibliography

| \[1\] | A. Van Der Welle and B. Van Der Zwaan, "An overview of selected studies on the value of lost load (VOLL)," _Energy Research Centre of the Netherlands (ECN), Amsterdam,_ 2007. |
| --- | --- |
| \[2\] | F. De Sisternes, "Investment model for renewable electricity systems (IMRES): an electricity generation capacity expansion formulation with unit commitment constraints," _MIT, Cambridge MA,_ 2013. |
| \[3\] | NERC, "Glossary of Terms Used in NERC Reliability Standards," \[Online\]. Available: <http://www.nerc.com/pa/stand/glossary> of terms/glossary_of_terms.pdf. \[Accessed 12 December 2016\]. |
| \[4\] | B. Hobbs and C. Bothwell, "System Adequacy with Intermittent Resources : Capacity Value and Economic Distortions," ISO New England, 2016. \[Online\]. Available: <https://www.iso-ne.com/static-assets/documents/2016/09/PSPC09222016_A4_Cindy-Bothwell-Johns-Hopkins-University-System-Adequacy-with-Intermittent-Resources-Capacity-Value-and-Economic-Distortions.pdf>. \[Accessed 2 October 2016\]. |
| \[5\] | L. Baringo and A. J. Conejo, "Correlated wind-power production and electric load scenarios for investment decisions," _Applied Energy,_ vol. 101, pp. 475-482, 2013. |
| \[6\] | CESI, "Pan-Arab Regional Energy Trade Platform (PA-RETP). Power Market Study: Gas Comsumption Assessment for the Power Sector accross MENA Region," CESI, 2016. |
| \[7\] | CESI-Ramboll, "Feasibility Study of the Electrical Interconnection and Energy Trade between Arab Countries," Arab Fund for Economic and Social Development, 2014. |
| \[8\] | The World Bank, "Energy Efficiency Study in Lebanon," The World Bank, Washington, DC, 2009. |
| \[9\] | Presidency of the Council of Ministries Central Administration of Statistics, "Energy". |
| \[10\] | Palestinian Electricity Regulatory Council, "Annual Report 2011,," Palestinian Electricity Regulatory Council, 2011. |
| \[11\] | DSIRE, "N.C. Clean Energy Technology Center," DSIRE, \[Online\]. Available: <http://programs.dsireusa.org/system/program/tables>. \[Accessed 12 December 2016\]. |
| \[12\] | A. Hainoun, "Construction of the hourly load curves and detecting the annual peak load of future Syrian electric power demand using bottom-up approach," _International Journal of Electrical Power & Energy Systems,_ vol. 31, no. 1, pp. 1-12, 2009. |
| \[13\] | P. Khajavi, H. Monsef and H. Abniki, "Load profile reformation through demand response programs using smart grid," in _Proceedings of the International Symposium Modern Electric Power Systems (MEPS), 2010_, 2010. |
| \[14\] | A. Khazaee, M. Ghasempour, H. Hoseinzadeh and H. Hooshmandi, "DEMAND MANAGEMENT PROGRAM FOR LARGE INDUSTRIAL CONSUMERS BY USING THE AMI SYSTEM," in _23rd International Conference on Electricity Distribution_, 2015. |
| \[15\] | A. Salimi-Beni, D. Farrokhzad, M. Fotuhi-Firuzabad and S. Alemohammad, "A new approach to determine base, intermediate and peak-demand in an electric power system," in _International Conference on Power System Technology, 2006. PowerCon 2006._, 2006. |
| \[16\] | W. Short, P. Sullivan, T. Mai, M. Mowers, C. Uriarte, N. Blair, D. Heimiller and A. Martinez, "Regional Energy Deployment System ( ReEDS)," NREL, 2011. |




1. The generators already planned are included in any of the two sets depending on criteria such as their capacity, status of their construction process etc. [↑](#footnote-ref-2)

2. The electrolyzers already planned are included in any of the two sets depending on criteria such as their capacity, status of their construction process etc. [↑](#footnote-ref-3)

3. Type of resources considered as renewables might be different from country to country or state to state. For example, some states do not include hydropower towards their renewable targets (e.g. California does not count large hydropower towards the Renewable portfolio standard) while others such as Oregon do \[13\]. [↑](#footnote-ref-4)

4. Capital costs of existing generators are considered sunk costs and are not included in the objective function. [↑](#footnote-ref-5)

5. For the first model runs, capacity credit was considered at full capacity. This will be modified in the next runs but it is not expected to change significantly the model results. [↑](#footnote-ref-6)

6. Depending on the scope of the project, the dispatch constraint (14) might be slightly different. For example, ReEDS model implemented by NREL \[16\] treats quick start capacity service provided by a generator in the same way as spinning reserves under constraint (14) and on top of that, it accounts for planned and forced outages by considering average outage rates. [↑](#footnote-ref-7)

7. This constraint is necessary to ensure that the model does not simply build additional storage units just to meet planning or spinning reserve requirements. [↑](#footnote-ref-8)