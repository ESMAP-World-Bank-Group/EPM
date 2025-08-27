

Sets
   H2status  'H2 generation plant status' / Existing, Candidate, Committed /
   eh(hh)           'existing hydrogen generation plants'
   nh(hh)           'new hydrogen generation plants'
   RampRateH2(hh)   'ramp rate constrained H2 generator blocks' // Ramprate takes out inflexible generators for a stronger formulation so that it runs faster
   dcH2(hh)         'candidate generators with capex trajectory'
   ndcH2(hh)        'candidate generators without capex trajectory'
   nRE(g)           'non RE generators'
   nVRE(g)          'non VRE generators'
   REH2(g)          'set of RE generators which are NOT VRE'
   nREH2(g)         'set of generators that are dont belong to subset REH2(g)'
   h2zmap(hh,z)
   sH2PwrIn(hh,q,d,t,y)
;

Sets
   H2statusmap(hh,H2status) 'Hydrogen unit status'
;

Parameters
    pIncludeH2
   pCRFH2(hh)                       'Capital recovery factor'
   pCapexTrajectoriesH2(hh,y)       'CAPEX trajectories for hydrogen generation units'
   pVarCostH2(hh,y)                 'Variable cost - H2 production'
   pH2UnservedCost                  'Cost of external H2 unserved'
   pH2Data(hh,pH2Header)                    'Hydrogen production specifications'
;

Positive Variables
   vCapH2(hh,y)                'total capacity in place accounting for legacy, new and retired plants (MW)'
   vBuildH2(hh,y)              'Build (MW)'
   vRetireH2(hh,y)             'Retire (MW)'
   vH2PwrIn(hh,q,d,t,y)        'Power drawn by H2 plants for H2 production in MW'
   vFuelH2(z,y)                'Annual H2 production in MMBTU'
   vAnnCapexH2(hh,y)           'Annualized CAPEX of H2 producing technologies'
   vPwrREH2(z,q,d,t,y)          'Generation from RE generators that goes to H2 production'
   vPwrREGrid(z,q,d,t,y)       'Generation from RE generators that goes to the grid'
   vREPwr2H2(g,f,q,d,t,y)       'Generation from RE generators that goes to H2 production'
   vREPwr2Grid(g,f,q,d,t,y)    'Generation from VRE generators that goes to the grid'
   vFuelH2Quarter(z,q,y)        'H2 fuel saved for H2 electricity generationon a quarterly basis'
   vUnmetExternalH2(z,q,y)        'mmBTU of external H2 demand that cant be met'
   vYearlyH2UnservedCost(z,y)   'Annual Cost of external H2 unserved in USD'
;

Integer variable
   vBuiltCapVarH2(hh,y)
   vRetireCapVarH2(hh,y)
;

Equations
    eH2UnservedCost(z,y)
    eCapBalanceH2(hh,y)
    eCapBalance1H2(hh,y)
    eCapBalance2H2
    eBuildNewH2(hh)
    eBuiltCapH2(hh,y)
    eRetireCapH2(hh,y)
    eMaxCF_H2(hh,q,y)
    eFuel_H2(c,q,y)
    eFuel_H2_2(c,z,y)
    eRampDnLimitH2(hh,q,d,t,y)
    eRampUpLimitH2(hh,q,d,t,y)
    eFuelLimitH2(c,f,y)
    eFuelLimitH2_2(c,f,y)
    eAnnCapexH2_1(hh,y)
    eAnnCapexH2(hh,y)
    eRE2H2(g,f,q,d,t,y)
    eRE2H2_2(z,q,d,t,y)
    eRE2H2_3(z,q,d,t,y)
    eRE2H2_4(z,q,d,t,y)
    eMaxH2PwrInjection(hh,q,d,t,y)
;


* =============================
* Hydrogen Contributions to Main Equations
* =============================

* --- Hydrogen Contribution to Yearly Cost ---
Equation eYearlyTotalCost_H2Contribution;

eYearlyTotalCost_H2Contribution(c,y)..
*   vYearlyTotalCost(c,y) =e= vYearlyTotalCost(c,y)
    0 =e=
                          + sum(zcmap(z,c), vYearlyH2UnservedCost(z,y))$pIncludeH2;

* --- Hydrogen Contribution to Total Annualized CAPEX ---
Equation eTotalAnnualizedCapex_H2Contribution;

eTotalAnnualizedCapex_H2Contribution(z,y)..
*   vAnnCapex(z,y) =e= vAnnCapex(z,y)
    0 =e=
                   + sum(h2zmap(ndcH2,z), pCRFH2(ndcH2)*vCapH2(ndcH2,y)*pH2Data(ndcH2,"Capex")*1e6)$pIncludeH2
                   + sum(h2zmap(dcH2,z), vAnnCapexH2(dcH2,y))$pIncludeH2;

* --- Hydrogen Contribution to Yearly Variable O&M Cost ---
Equation eYearlyVOMCost_H2Contribution;

eYearlyVOMCost_H2Contribution(z,y)..
*   vYearlyVOMCost(z,y) =e= vYearlyVOMCost(z,y)
    0 =e=
                         + sum((h2zmap(hh,z),q,d,t), pVarCostH2(hh,y)*pH2Data(hh,"Heatrate")*vH2PwrIn(hh,q,d,t,y)*pHours(q,d,t))$pIncludeH2;

* --- Hydrogen Unserved Demand Cost (can stay modular as-is) ---
Equation eH2UnservedCost;

eH2UnservedCost(z,y)..
   vYearlyH2UnservedCost(z,y) =e= sum(q, vUnmetExternalH2(z,q,y)) * pH2UnservedCost $pIncludeH2;

* --- Hydrogen Supply Contribution to Demand Balance ---
Equation eSupply_H2Contribution;

* TODO: Change that
eSupply_H2Contribution(z,q,d,t,y)..
*   vSupply(z,q,d,t,y) =e=vSupply(z,q,d,t,y)
*        -sum((gzmap(g,z),gfmap(g,f)),vPwrOut(g,f,q,d,t,y))
    0 =e=
     + sum((gzmap(nRE,z),gfmap(nRE,f)),vPwrOut(nRE,f,q,d,t,y))$(pIncludeH2)
     + vPwrREGrid(z,q,d,t,y)$pIncludeH2;

* =============================
* Hydrogen Specific Equations
* =============================


* Total capacity of H2 plants at 1st year of optimization is equal to pre-existing capacity plus capacity being bulit minus retired capacity
eCapBalanceH2(hh,sStartYear(y))$(pIncludeH2)..
   vCapH2(hh,y) =e= pH2Data(hh,"Capacity")$(eh(hh) and (pH2Data(hh,"StYr") <= sStartYear.val))
               + vBuildH2(hh,y) - vRetireH2(hh,y);

* Total capacity of existing H2 palnts is equal to capacity over previous year plus capacity being built in current year minus retired capacity in current year
*Checked
eCapBalance1H2(eh,y)$(not sStartYear(y) and pIncludeH2)..
   vCapH2(eh,y) =e= vCapH2(eh,y-1) + vBuildH2(eh,y) - vRetireH2(eh,y);

* Total capacity of candidate H2 plants is equal to total capacity at previous year plus capacity being built in current year minus retired capacity in current year
*Checked
eCapBalance2H2(nh,y)$(not sStartYear(y) and pIncludeH2)..
   vCapH2(nh,y) =e= vCapH2(nh,y-1) + vBuildH2(nh,y);

* New H2 plants can be buuilt only after the StYr; the newly built capacity need to be less than declared capacity
*Checked
eBuildNewH2(eh)$(pH2Data(eh,"StYr") > sStartYear.val and pIncludeH2)..
    sum(y, vBuildH2(eh,y)) =l= pH2Data(eh,"Capacity");


* (Integer units ) Built capacity each year is equal to unit size
*Checked
eBuiltCapH2(nh,y)$(pH2Data(nh,"DescreteCap") and pIncludeH2)..
   vBuildH2(nh,y) =e= pH2Data(nh,"UnitSize")*vBuiltCapVarH2(nh,y);

* (Integer units ) Retired capacity each year is equal to unit size
*Checked
eRetireCapH2(eh,y)$(pH2Data(eh,"DescreteCap") and (y.val <= pH2Data(eh,"RetrYr")) and pIncludeH2)..
   vRetireH2(eh,y) =e= pH2Data(eh,"UnitSize")*vRetireCapVarH2(eh,y);

*  Maximum capacity factor of H2 production based on availability
*Checked
eMaxCF_H2(hh,q,y)$(pIncludeH2)..
   sum((d,t), vH2PwrIn(hh,q,d,t,y)*pHours(q,d,t)) =l= pAvailabilityH2(hh,q)*vCapH2(hh,y)*sum((d,t), pHours(q,d,t));

*       mmBTU of H2 produced
eFuel_H2(c,q,y)$(pIncludeH2)..
*                   mmBTU            mmBTU                    mmBTU                                                                     -MWe-                Hr                mmBTU/MWhe 
    sum(zcmap(z,c), pExternalH2(z,q,y)-vUnmetExternalH2(z,q,y)$pExternalH2(z,q,y)+vFuelH2Quarter(z,q,y)) =e= sum( (zcmap(z,c),h2zmap(hh,z), d,t), vH2PwrIn(hh,q,d,t,y)*pHours(q,d,t)*pH2Data(hh,"HeatRate"));

eFuel_H2_2(c,z,y)$(pIncludeH2)..
    sum((zcmap(z,c),q),vFuelH2Quarter(z,q,y)) =e=  sum(zcmap(z,c),vFuelH2(z,y));

eMaxH2PwrInjection(hh,q,d,t,y)$pIncludeH2..
    vH2PwrIn(hh,q,d,t,y)  =l= vCapH2(hh,y);

* The amount of hydrogen fuel that can be used for electricity generation can not be more than the amount of H2 that was produced from VRE curtailment
eFuelLimitH2(c,f,y)$(pFuelDataH2(f) and pIncludeH2)..
   sum((zcmap(z,c)),  vFuel(z,f,y)) =e=  sum((zcmap(z,c)), vFuelH2(z,y));


*When the H2 production flag is off don't account for H2 fuel
eFuelLimitH2_2(c,f,y)$(pFuelDataH2(f) and pIncludeH2=0)..
   sum((zcmap(z,c)),  vFuel(z,f,y)) =e=  0;


eRampDnLimitH2(hh,q,d,t,y)$(RamprateH2(hh) and not sFirstHour(t) and pramp_constraints and pIncludeH2)..
    vH2PwrIn(hh,q,d,t-1,y) -  vH2PwrIn(hh,q,d,t,y) =l= vCapH2(hh,y)*pH2Data(hh,"RampDnRate");

eRampUpLimitH2(hh,q,d,t,y)$(RamprateH2(hh) and not sFirstHour(t) and pramp_constraints and pIncludeH2)..
   vH2PwrIn(hh,q,d,t,y) -  vH2PwrIn(hh,q,d,t-1,y) =l= vCapH2(hh,y)*pH2Data(hh,"RampUpRate");

*Calculation of Annualized CAPEX for electrolyzers
eAnnCapexH2_1(dcH2,y)$(not sStartYear(y) and pIncludeH2)..
   vAnnCapexH2(dcH2,y) =e= vAnnCapexH2(dcH2,y-1)
                     + vBuildH2(dcH2,y)*pH2Data(dcH2,"Capex")*pCapexTrajectoriesH2(dcH2,y)*pCRFH2(dcH2)*1e6;

eAnnCapexH2(dcH2,sStartYear(y))$pIncludeH2..
   vAnnCapexH2(dcH2,y) =e= vBuildH2(dcH2,y)*pH2Data(dcH2,"Capex")*pCapexTrajectoriesH2(dcH2,y)*pCRFH2(dcH2)*1e6;                                                                          ;

eRE2H2_4(z,q,d,t,y)$pIncludeH2..
  sum(h2zmap(hh,z),vH2PwrIn(hh,q,d,t,y))=e=vPwrREH2(z,q,d,t,y);
 

eRE2H2_3(z,q,d,t,y)$pIncludeH2..
 vPwrREH2(z,q,d,t,y) =e= sum((gfmap(RE,f),gzmap(RE,z)),vREPwr2H2(RE,f,q,d,t,y));

eRE2H2_2(z,q,d,t,y)$pIncludeH2..
vPwrREGrid(z,q,d,t,y)=e=sum((gfmap(RE,f),gzmap(RE,z)),vREPwr2Grid(RE,f,q,d,t,y));

eRE2H2(RE,f,q,d,t,y)$pIncludeH2..
 vPwrOut(RE,f,q,d,t,y)=e=vREPwr2Grid(RE,f,q,d,t,y)+vREPwr2H2(RE,f,q,d,t,y);

*For example equation below is deactivated when H2 is on and is replaced by eVREProfile2

* =============================
* Updating Hydrogen Model
* =============================


$onMulti
Model PA /
   
    eYearlyTotalCost_H2Contribution,
    eTotalAnnualizedCapex_H2Contribution,
    eYearlyVOMCost_H2Contribution,
    eH2UnservedCost,
    eSupply_H2Contribution

   vH2PwrIn(sH2PwrIn)
   eBuildNewH2
   eCapBalance1H2
   eCapBalance2H2
   eCapBalanceH2
   eBuiltCapH2
   eRetireCapH2
*eDemSupplyH2
   eMaxCF_H2
   eFuel_H2
   eFuel_H2_2
   eRampDnLimitH2
   eRampUpLimitH2
   eFuelLimitH2
   eFuelLimitH2_2
*eYearlyCurtailmentCost2

   eRE2H2
   eRE2H2_2
   eRE2H2_3
   eRE2H2_4
   eMaxH2PwrInjection

   eAnnCapexH2_1
   eAnnCapexH2
   /
;
$offMulti