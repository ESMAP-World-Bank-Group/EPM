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
* Claire Nicolas, cnicolas@worldbank.org
**********************************************************************

$eval TIME jnow
$eval YEAR   gyear(%TIME%)
$eval MONTH  gmonth(%TIME%)
$eval DAY    gday(%TIME%)
$eval HOUR   ghour(%TIME%)
$eval MINUTE gminute(%TIME%)
$eval SECOND gsecond(%TIME%)
$set TIMESTAMP %YEAR%%MONTH%%DAY%_%HOUR%%MINUTE%%SECOND%

$set OUTPUTDIR output%system.DirSep%simulation_gmstudio_%TIMESTAMP%

$call mkdir %OUTPUTDIR%

$call cp cplex.opt %OUTPUTDIR%

* Add argument for main
$if not set FOLDER_INPUT $set FOLDER_INPUT data_capp

$call gams "../../main.gms" --ROOT_FOLDER "..%system.DirSep%.." --ROOT_INPUT "..%system.DirSep%..%system.DirSep%input" --FOLDER_INPUT=%FOLDER_INPUT% curdir=%OUTPUTDIR% LogOption=4

