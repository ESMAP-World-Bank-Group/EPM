**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
* Description:
* This GAMS-based model is designed for electricity system planning, 
* incorporating capacity expansion, generation dispatch, and policy 
* constraints such as renewable energy targets, emissions reductions, 
* and market mechanisms.
*
* Author(s): ESMAP Modelling Team
* Organization: World Bank
* Version: 
* License: Creative Commons Zero v1.0 Universal
*
* Key Features:
* - Optimization of electricity generation and capacity planning
* - Inclusion of renewable energy integration and storage technologies
* - Multi-period, multi-region modeling framework
* - CO2 emissions constraints and policy instruments
*
* Notes:
* - Ensure GAMS is installed before running this model.
* - The model requires input data in .GDX or Excel format.
*
* Contact:
* Claire Nicolas, c.nicolas@worldbank.org
**********************************************************************


$onMultiR
$gdxIn epmresults_EPS.gdx
$load pCapacityByFuel
$gdxIn
$offMulti

parameter pCapacityByFuel_EPS(z,f,y);
pCapacityByFuel_EPS(z,f,y) = pCapacityByFuel(z,f,y)

display pCapacityByFuel_EPS;

$onMultiR
$gdxIn epmresults.gdx
$load pCapacityByFuel
$gdxIn
$offMulti

display pCapacityByFuel;


parameter diff(z,f,y);
diff(z,f,y) = pCapacityByFuel(z,f,y) - pCapacityByFuel_EPS(z,f,y)

* Find the maximum absolute difference
scalar max_diff;
max_diff = smax((z,f,y), abs(diff(z,f,y)));

execute_unload 'check_output' diff, max_diff;