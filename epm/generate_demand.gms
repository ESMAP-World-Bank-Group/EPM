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

*--- Adjust peak and energy in case user selected same demand profile for all years

Parameters
   pmax(z,y)
   pmin(z,y)
   pdiff(z,y)
   ptotalenergy(z,y)
   ptemp2(y)
   ptemp(y)
   pTempDemand(z,q,d,y,t)
;

variable
   pyval(z,q,d,y,t)
   divisor(z,y)
   obj
;

divisor.lo(z,y) = card(q)*card(d)*card(t);

equation
   getDivisor(z,q,d,y,t)
   getArea(z,y)
   objFn
;

getDivisor(z,q,d,y,t)..  pyval(z,q,d,y,t) =e=  [2*pdiff(z,y)/divisor(z,y) * (pmax(z,y) - pDemandProfile(z,q,d,t))]/sqr(pmax(z,y) - pmin(z,y));
getArea(z,y)..           sum((q,d,t), pyval(z,q,d,y,t) * pHours(q,d,t)) =e=  pdiff(z,y);
objFn..                  obj =e= sum((z,q,d,y,t), pyval(z,q,d,y,t));

model demand / getDivisor, getArea, objFn /;

* If using alt demand:
if (pSettings("altDemand") = 1,
   pTempDemand(z,q,d,y,t) = pDemandProfile(z,q,d,t) * pDemandForecast(z,"Peak",y);

   pdiff(z,y) = ((pDemandForecast(z,"Energy",y)*1e3) - sum((q,d,t), pTempDemand(z,q,d,y,t)*pHours(q,d,t) )) ;

   pmax(z,y) = smax((q,d,t), pDemandProfile(z,q,d,t));
   pmin(z,y) = smin((q,d,t)$pDemandProfile(z,q,d,t), pDemandProfile(z,q,d,t));
*   option limrow=1000; debugging
   Solve demand using nlp min obj;

   abort$(demand.modelstat<>%modelstat.Optimal% and demand.modelstat<>%modelstat.locallyOptimal%) 'Demand model not solved successfully';
$offIDCProtect
   pDemandData(z,q,d,y,t) =  pTempDemand(z,q,d,y,t) + pyval.l(z,q,d,y,t);
$onIDCProtect
   ptemp(y) = sum((z,q,d,t), pdemanddata(z,q,d,y,t)*pHours(q,d,t))/1000;
   ptemp2(y) = smax((q,d,t),sum(z, pdemanddata(z,q,d,y,t)));
);

ptotalenergy(z,y) =  sum((q,d,t), pdemanddata(z,q,d,y,t)*pHours(q,d,t))/1e3;
