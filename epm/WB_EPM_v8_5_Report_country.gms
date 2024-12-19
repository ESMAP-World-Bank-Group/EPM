
$onMulti
*Report

Parameters
*---- Summary
$ifi %mode%==MIRO
$onExternalOutput
   pSummary(*)                               'Summary of total results with costs in million USD'
$ifI %mode%==MIRO
$offExternalOutput
   pCapex(z,y)                               'Capex  in USD'
   pAnncapex(z,y)                            'Annualized capex in USD'
   pFOM(z,y)                                 'FOM in USD'
   pVOM(z,y)                                 'VOM  in USD'
   pFuel(z,y)                                'Fuel costs in USD'
   pImportCosts(z,y)                         'Net import costs from trade with external zones in USD'
   pExportRevenue(z,y)                       'Export revenue from trade with external zones in USD'
   pNewTransmissionCosts(z,y)                'Added transmission costs in USD'
   pUSECosts(z,y)                            'Unmet demand costs in USD'
   pCO2backstopCosts(c,y)                    'CO2backstop costs in USD'
   
   pSurplusCosts(z,y)                        'Surplus generation costs  in USD'
   pVRECurtailment(z,y)                      'VRE curtailment costs'
   pUSRSysCosts(y)                           'Unmet system spinning reserve violation costs'
   pUPRSysCosts(y)                           'Unmet system planning reserve violation costs'
   pSpinResCosts(z,y)                        'Spinning res costs  by zone in USD'
   pUSRLocCosts(c,y)                         'Unmet locational spinning reserve costs per country'
   pCountryPlanReserveCosts(c,y)             'Planning reserve violation cost by zone in USD'

   pCostSummary(z,*,y)                       'Summary of costs in millions USD  (unweighted and undiscounted) by zone'


****************Turkey model related parameter ********************************************************
    pCostSummaryM(z,*,y,q)                   'Monthly summary of costs in millions USD  (unweighted and undiscounted) by zone'
    pVOMm(z,y,q)                              'Monthly VOM costs'
    pFuelM(z,y,q)                            'Monthly fuel costs'
    pImportCostsM(z,y,q)                     'Monthly Import Costs'
    pExportRevenueM(z,y,q)                   'Monthly electricity export revenues'
    pUSECostsM(z,y,q)                        'Monthly unserved energy penalty'
    pVRECurtailmentM(z,y,q)                  'Monthly VRE curtailment penalty'
    pSurplusCostsM(z,y,q)                    'Monthly surplus energy penalty'
    pSpinResCostsM(z,y,q)                    'Monthly spinning reserve violation penalty'

*********************************************************************************************************
   pCostSummaryCountry(c,*,y)                'Summary of costs in millions USD  (unweighted and undiscounted) by country'
   pCostSummaryWeighted(z,*,y)               'Summary of costs in millions USD  (weighted and undiscounted) by zone'
   pCostSummaryWeightedCountry(c,*,y)        'Summary of costs in millions USD  (weighted and undiscounted) by country'
   pCostSummaryWeightedAverageCountry(c,*,*) 'Summary of average costs (undiscounted) by country'
   pCostSummaryWeightedAverageCtry(c,*)      'Summary of average costs (undiscounted) by country (modified pCostSummaryWeightedAverageCountry for MIRO)'
   pFuelCosts(z,f,y)                         'Fuel costs in millions USD by zone'
   pFuelCostsCountry(c,f,y)                  'Fuel costs in millions USD by zone'
   pFuelConsumption(z,f,y)                   'Fuel consumed in unit defined by user (millions of unit)  per zone'
   pFuelConsumptionCountry(c,f,y)            'Fuel consumed in unit defined by user (millions of unit)  per country'

   pEnergyByPlant(z,g,y)                     'Energy by plant in GWh'
   pEnergyByFuel(z,f,y)                      'Energy by fuel in GWh per zone'
   pEnergyByFuelCountry(c,f,y)               'Energy by fuel in GWh per country'
   pEnergyMix(c,f,y)                         'Energy mix by country'

   pDemandSupply(z,*,y)                      'Supply demand in GWh per zone'
   pDemandSupplyCountry(c,*,y)               'Supply demand in GWh per country'

   pInterchange(z,z2,y)                      'Total exchange in GWh between internal zones  per zone'
   pInterconUtilization(z,z2,y)              'Utilization of interconnection throughout modeling horizon (between internal zones)'
   pLossesTransmission(z,y)                  'Transmission losses between internal zones in MWh per zone'
   pInterchangeCountry(c,c2,y)               'Total exchange in GWh between countries per country'
   pLossesTransmissionCountry(c,y)           'Total Transmission losses in MWh per country'

   pYearlyTrade(z,*,y)                       'Trade with external zones by year in GWh per zone'
   pHourlyTrade(z,y,q,d,*,t)                 'Trade with external zones in MW per zone'
   pYearlyTradeCountry(c,*,y)                'Trade with external zones by year in GWh per country'
   pHourlyTradeCountry(c,y,q,d,*,t)          'Trade with external zones in MW per country'

   pPrice(z,q,d,t,y)                         'Marginal cost in USD per MWh  per zone'
   pAveragePrice(z,y)                        'Average price by internal zone'
   pAveragePriceHub(z,y)                     'Average price hub'
   pAveragePriceExp(z,y)                     'Average Marginal cost of exports to internal zones in USD per MWh'
   pAveragePriceImp(z,y)                     'Average Marginal cost of imports to internal zones in USD per MWh'
   pAveragePriceCountry(c,y)                 'Average price by internal zone per country'
   pAveragePriceExpCountry(c,y)              'Average Marginal cost of exports to internal zones in USD per MWh for each country'
   pAveragePriceImpCountry(c,y)              'Average Marginal cost of imports to internal zones in USD per MWh for each country'
   pAveragePriceExp1
   pAveragePriceImp1

   pPeakCapacity(z,*,y)                      'Peak capacity in MW per zone'
   pCapacityByFuel(z,f,y)                    'Peak capacity by primary fuel in MW per zone'
   pNewCapacityFuel(z,f,y)                   'New capacity by fuel in MW per zone'
   pPlantUtilization(z,g,y)                  'Plant utilization'
   pRetirements(z,g,y)                       'Retirements in MW per zone'
   pRetirementsByFuel(f,y)
   pCapacityPlan(z,g,y)                      'Capacity plan MW per zone'
   pPeakCapacityCountry(c,*,y)               'Peak capacity in MW  per country'
   pCapacityByFuelCountry(c,f,y)             'Peak capacity by primary fuel in MW per country'
   pNewCapacityFuelCountry(c,f,y)            'New capacity by fuel in MW per country'
   pCapacityPlanCountry(c,g,y)               'Capacity plan MW per country'
   pAdditionalCapacity(z,z2,y)               'Additional transmission capacity between internal zones in MW'
   pAnnualTransmissionCapacity(z,z2,y)       'Total annual transmission capacity between internal zones in MW'


   pReserveCosts(z,g,y)                      'Cost of reserves by plant in dollars per zone'
   pReserveByPlant(z,g,y)                    'Reserve contribution by plant in MWh per zone'
   pReserveCostsCountry(c,g,y)               'Cost of reserves by plant in dollars per country'
   pReserveByPlantCountry(c,g,y)             'Reserve contribution by plant in MWh per country'

   pEmissions(z,y)                           'Emissions in Megaton CO2 by zone'
   pEmissionsIntensity(z,y)                  'Emissions intensity tCO2 per GWh by zone'
   pEmissionsCountry(c,y)                    'Emissions in Megaton CO2 by country'
   pEmissionsIntensityCountry(c,y)           'Emissions intensity tCO2 per GWh by country'
   pEmissionMarginalCosts(y)                 'Marginal costs of Emission Limit Constraint eEmissionsCountry'
   pDenom(z,g,y)                             'Energy by plant in MWh'
   pDenom2(z,y)                              'Energy by zone in MWh'
   pDenom3(c,y)                              'Energy by country in MWh'
   pPlantAnnualLCOE(z,g,y)                   'Plant levelized cost by year USD per MWh'
   pZonalAverageCost(z,y)                    'Zonal annual cost of generation+ import by year USD per MWh'
   pZonalAverageGenCost(z,y)                 'Zonal annual cost of generation by year USD per MWh'
   pCountryAverageCost(c,y)                  'Country annual cost of generation + trade by year USD per MWh'
   pCountryAverageGenCost(c,y)               'Country annual cost of generation by year USD per MWh'
   pSystemAverageCost(y)                     'System annual cost of generation by year USD per MWh'

   pPlantDispatch(z,y,q,d,g,t)               'Detailed dispatch by plant in MW'
   pPlantDispatchScenarios(z,y,q,d,g,t,s)    'Detailed dispatch by plant in MW scenario-specific'

   pDispatch(z,y,q,d,*,t)                    'Detailed dispatch and flows'
   pDispatchScenarios(z,y,q,d,*,t,s)         'Detailed dispatch and flows scenario-specific'

   pCSPBalance(y,g,q,d,*,t)                  'in MW'
   pCSPComponents(g,*, y)                    'CSP specific output'
   pPVwSTOBalance(y,q,d,g,*,t)               'pVwithStorage specific output'
   pPVwSTOComponents(g,*,y)                  'pVwithStorage specific output'
   pStorageBalance(y,g, q,d,*,t)             'in MW'
   pStorageComponents(g,*,y)                 'Storage specific output'

   pSolarValue(z,y)                          'Value of solar energy in USD'
   pSolarCost(z,y)
   pSolarEnergy(z,q,d,t,y)                   'Solar output in MWh'
$ifI %mode%==MIRO
$onExternalOutput
   pSolverParameters(*)                      'Solver parameters'
$ifI %mode%==MIRO
$offExternalOutput
   pDemandSupplySeason(z,*,y,q)              'Seasonal demand supply parameters per zone'

   pEnergyByPlantSeason(z,g,y,q)             'Energy by plant in GWh per season  per zone'
   pInterchangeSeason(z,z2,y,q)              'Total exchange in GWh between internal zones per season per zone'
   pSeasonTrade(z,*,y,q)                     'Trade with external zones by season in GWh per zone'
   pInterchangeSeasonCountry(c,c2,y,q)       'Total exchange in GWh between internal zones per season per country'
   pSeasonTradeCountry(c,*,y,q)              'Trade with external zones by season in GWh per country'

   pZonesperCountry(c)                       'Number of zones per country'



****************Turkey model specific parameters********************

pCurtailedVRE(q,d,t,y,g)
pCurtailedVRET(q,d,t,y)
pCurtailedVREM(y,q)                           'Monthly VRE curtailment'
pCurtailedVREperY(g,y)
pCurtailedVRETperY(y)
pVREdispatchT(q,d,t,y)
pVREpenetration(q,d,t,y)
pREpenetration2(y)
pDemandTotalE(q,d,t,y)
pDemandTotalP(q,d,t,y)

pLCOEperFuelAnnual(c,f,y)
pAnnCapexperFuel(c,f,y)

pMargPrice(z,y,q,d,t)                             'Variable cost of most expensive generator'
pVarGenCost(z,g,y,q,d,f,t)                        'Variable cost of generators'
pGenMarkUp(z,g,y,q,d,f,t)                         'MarkUp of generators'
pPresentValue(z,g,f,y)                            'Present value of markup'
pGenRevenue(z,g,f,y)                              'Revenue of generator in absolute number'
pPresentValueWeighted(z,g,f,y)                    'Weighted NPV'
pNormalRetireYear(z,g,f)                          'Year a generator is scheduled to retire'
pNominalCapacity(z,g,f)                           'The nominal capacity of generators'
pFuelDispatch(z,q,d,t,y,f)                        'Detailed Dispatch by fuel'
pUnmetP(q,d,t,y)                                  'Unmet demand for each hour (Power)'
pSurplusP(q,d,t,y)                                'Surplus generation for each hour (Power)'
pStorLevel(q,d,t,y)                               'Level of storage (ΜW)'
pStorInj(q,d,t,y)                                 'Storage injection (MW)'
pStorOut(q,d,t,y)                                 'Storage output (MW)'
pStorInjD(y,q,d,t)                                 'Storage injection (MW)'
pStorOutD(y,q,d,t)                                 'Storage output (MW)'


*****************************************************************
****************Hydrogen model specific parameters*****************************

pH2perY(y)                                       'The mmBTU of H2 produced per year'
pH2perG(q,d,t,y,hh)                              'The mmBTU of H2 produced per electrolyzer'
*pH2T(q,d,t,y)                                    'Total mmBTU of H2 produced'

pH2PwrInPerGen(q,d,t,y,hh)                     'Power injected for H2 production per electrolyzer(MW)'
pH2PwrInT(q,d,t,y)                          'Total Power injected for H2 production (MW)'

pCurtailedVREnet(q,d,t,y)                    'Net VRE curtailment'

pCapacityPlanH2(z,hh,y)                      'Capacity plan of electrolyzers in MW per zone'



pAnnCapexH2(hh,y)           'Annualized CAPEX of H2 producing technologies'
pCapexH2(hh,y)              'CAPEX of electrolyzers'
pAnnCapexH2T(y)           'Annualized CAPEX of H2 producing technologies'
pCapexH2T(y)              'CAPEX of electrolyzers'


pPwrRE(q,d,t,y)           'Total Generation from RE generators'
pPwrREH2(q,d,t,y)         'Total Generation from RE generators that goes to H2 production'
pPwrREGrid(q,d,t,y)       'Total Generation from RE generators that goes to the grid'
pUnmetH2 (z,q,y)          'Total Unmet H2 demand'

pSpinningReserveByFuelByCountry(f,c,y)

pNPVbyCountry(c,y)
pNPVCostSpecific(*)
pCAPEXByFuel(f)
pCAPEXByFuelByYear(y,f)
pNPVByYear(y)

pREAnnualCF(z,f,y)

pHoursLineCongested(z,z2,y)
pHoursLineCongestedPercent(z,z2,y)
pCurtailedVRETperYZ(y,z)

;

pCurtailedVRETperY(y)                         =sum(gzmap(VRE,z),sum((q,d,t,s), pProbaScenarios(s)*vCurtailedVRE.l(z,VRE,q,d,t,y,s)*pHours(q,d,t)))/1000;
pCurtailedVRETperYZ(y,z)                         =sum((q,d,t,VRE,s), pProbaScenarios(s)*vCurtailedVRE.l(z,VRE,q,d,t,y,s)*pHours(q,d,t))/1000;

*****************************************************************************************

* TOCHANGE
pHoursLineCongested(z,z2,y)$(sTopology(z,z2)) = sum((q,d,t,s)$(eTransferLimit.M(z,z2,q,d,t,y,s)<-0.0001 AND vFlow.L(z,z2,q,d,t,y,s)>0.0001), pProbaScenarios(s)*pHours(q,d,t));
pHoursLineCongested(z,z2,y) = pHoursLineCongested(z,z2,y) + pHoursLineCongested(z2,z,y);

*pHoursLineCongested(z,z2,y)$(pHoursLineCongested(z2,z,y)) = NO;
pHoursLineCongestedPercent(z,z2,y) = pHoursLineCongested(z,z2,y)/sum((q,d,t), pHours(q,d,t));


pREAnnualCF(z,f,y)$(sum((q,d,t),pVREProfile(z,f,q,d,t))) =  sum((q,d,t),pVREProfile(z,f,q,d,t)*pHours(q,d,t))/sum((q,d,t), pHours(q,d,t));


Scalar MaxZonesperCountry;

*--- START of results

*--- Cost Items and Penalty Costs
******************************H2 model************************************************************
pAnnCapexH2(dch2,y)  =  vAnnCapexH2.l(dch2,y)$pIncludeH2;


pAnnCapexH2(ndch2,y)  =  (1e6* pCRFH2(ndcH2)*vCapH2.l(ndcH2,y)*pH2Data(ndcH2,"Capex"))$pIncludeH2;

pCapexH2(hh,y)    = (1e6* vBuildH2.l(hh,y)*pH2Data(hh,"Capex")*pCapexTrajectoriesH2(hh,y))$pIncludeH2 ;
pAnnCapexH2T(y)   =  (sum(h2zmap(dch2,z), vAnnCapexH2.l(dch2,y))
                    +1e6*sum(h2zmap(ndcH2,z), pCRFH2(ndcH2)*vCapH2.l(ndcH2,y)*pH2Data(ndcH2,"Capex")))$pIncludeH2;
pCapexH2T(y)      = (1e6*sum(h2zmap(hh,z), vBuildH2.l(hh,y)*pH2Data(hh,"Capex")*pCapexTrajectoriesH2(hh,y)))$pIncludeH2;

************************************************************************************************

* zonal level
pCapex(z,y) = 1e6*sum(gzmap(g,z),  vBuild.l(g,y)*pGenData(g,"Capex")*pCapexTrajectories(g,y))
            + 1e3*sum(gzmap(st,z), vBuildStor.l(st,y)*pStorData(st,"Capex")*pCapexTrajectories(st,y))
            + 1e3*sum(gzmap(cs,z), vBuildTherm.l(cs,y)*pCSPData(cs,"Thermal Field","Capex")*1e3*pCapexTrajectories(cs,y)
                                 + vBuildStor.l(cs,y)*pCSPData(cs,"Storage","Capex")*pCapexTrajectories(cs,y))
**************************H2 model addition**************************************************************
            +1e6*sum(h2zmap(hh,z), vBuildH2.l(hh,y)*pH2Data(hh,"Capex")*pCapexTrajectoriesH2(hh,y))$pIncludeH2;
****************************************************************************************

pAnncapex(z,y) = 1e6*sum(gzmap(ndc,z),               pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex"))
               + 1e3*sum(gzmap(ndc,z)$(not cs(ndc)), pCRFsst(ndc)*vCapStor.l(ndc,y)*pStorData(ndc,"Capex"))
               + 1e3*sum(gzmap(ndc,z)$(not st(ndc)), pCRFcst(ndc)*vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","Capex"))
               + 1e6*sum(gzmap(ndc,z),               pCRFcth(ndc)*vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal Field","Capex"))
               +     sum(gzmap(dc,z),                vAnnCapex.l(dc,y))
**************************H2 model addition**************************************************************
               +     sum(h2zmap(dch2,z),                vAnnCapexH2.l(dch2,y))
               +1e6*sum(h2zmap(ndcH2,z),              pCRFH2(ndcH2)*vCapH2.l(ndcH2,y)*pH2Data(ndcH2,"Capex"))$pIncludeH2;
****************************************************************************************

pFOM(z,y) = sum(gzmap(g,z),  vCap.l(g,y)*pGenData(g,"FOMperMW"))
          + sum(gzmap(st,z), vCapStor.l(st,y)*pStorData(st,"FixedOM"))
          + sum(gzmap(cs,z), vCapStor.l(cs,y)*pCSPData(cs,"Storage","FixedOM"))
          + sum(gzmap(cs,z), vCapTherm.l(cs,y)*pCSPData(cs,"Thermal field","FixedOM"))
**************************H2 model addition**************************************************************
          + sum(h2zmap(hh,z),  vCapH2.l(hh,y)*pH2Data(hh,"FOMperMW"))$pIncludeH2;
****************************************************************************************


pSpinResCosts(z,y) = vYearlyReserveCost.l(z,y);

pVOM(z,y) = sum((q,d,t,s), pProbaScenarios(s) * pHours(q,d,t)*(
                           sum((gzmap(g,z),gfmap(g,f)),   pGenData(g,"VOM")*vPwrOut.l(g,f,q,d,t,y,s))
                         + sum((gzmap(st,z),gfmap(st,f)), pStorData(st,"VOM")*vPwrOut.l(st,f,q,d,t,y,s))
                         + sum((gzmap(cs,z),gfmap(cs,f)), pCSPData(cs,"Storage","VOM")*vPwrOut.l(cs,f,q,d,t,y,s))
                         + sum((gzmap(cs,z),gfmap(cs,f)), pCSPData(cs,"Thermal Field","VOM")*vPwrOut.l(cs,f,q,d,t,y,s))
**************************H2 model addition**************************************************************
***(units for equation below)**********       $/mMBTU_H2          mmBTU_H2/MWh_e        MW_e
                        + sum((h2zmap(hh,z)), pH2Data(hh,"VOM")*pH2Data(hh,"Heatrate")*vH2PwrIn.l(hh,q,d,t,y))$pIncludeH2));
****************************************************************************************

*******************Turkey related model parameter********************************

pVOMm(z,y,q)=sum((d,t,s), pProbaScenarios(s) * pHours(q,d,t)*(
                           sum((gzmap(g,z),gfmap(g,f)),   pGenData(g,"VOM")*vPwrOut.l(g,f,q,d,t,y,s))
                         + sum((gzmap(st,z),gfmap(st,f)), pStorData(st,"VOM")*vPwrOut.l(st,f,q,d,t,y,s))
                         + sum((gzmap(cs,z),gfmap(cs,f)), pCSPData(cs,"Storage","VOM")*vPwrOut.l(cs,f,q,d,t,y,s))
                         + sum((gzmap(cs,z),gfmap(cs,f)), pCSPData(cs,"Thermal Field","VOM")*vPwrOut.l(cs,f,q,d,t,y,s))
**************************H2 model addition**************************************************************
***(units for equation below)**********       $/mMBTU_H2          mmBTU_H2/MWh_e        MW_e
                        + sum((h2zmap(hh,z)), pH2Data(hh,"VOM")*pH2Data(hh,"Heatrate")*vH2PwrIn.l(hh,q,d,t,y))$pIncludeH2));
****************************************************************************************


pFuelM(z,y,q)= sum((gzmap(g,z),gfmap(g,f),zcmap(z,c),d,t,s), pProbaScenarios(s)*vPwrOut.l(g,f,q,d,t,y,s)*pHours(q,d,t)*pFuelPrice(c,f,y)*pHeatRate(g,f));


*******************Dispath model equations*******************
pImportCostsM(z,y,q)   =  sum((d,t,s), pProbaScenarios(s)*vImportPrice.l(z,q,d,t,y,s)*pTradePrice(z,q,d,y,t)*pHours(q,d,t));

pExportRevenueM(z,y,q) =  sum((d,t,s), pProbaScenarios(s)*vExportPrice.l(z,q,d,t,y,s)*pTradePrice(z,q,d,y,t)*pHours(q,d,t));

pUSECostsM(z,y,q)       =    sum((d,t,s), pProbaScenarios(s)*vUSE.l(z,q,d,t,y,s)*pVoLL*pHours(q,d,t));

pVRECurtailmentM(z,y,q) = sum((gzmap(g,z),d,t,s), pProbaScenarios(s)*vCurtailedVRE.l(z,g,q,d,t,y,s)*pCostOfCurtailment*pHours(q,d,t));

pSurplusCostsM(z,y,q)    = sum((d,t,s), pProbaScenarios(s)*vSurplus.l(z,q,d,t,y,s)*pSurplusPenalty*pHours(q,d,t));

pSpinResCostsM(z,y,q)  =  sum((d,t,s), pProbaScenarios(s)*vUnmetSpin.l(q,d,t,y,s)*pHours(q,d,t)*pSpinReserveVoLL);


**************************************************************************

********************************H2 model equations**********************************************
pH2perY(y)                 = (sum((hh,q,d,t),vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t)*pH2Data(hh,"HeatRate")))$pIncludeH2;
pH2perG(q,d,t,y,hh)         = (vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t)*pH2Data(hh,"HeatRate"))$pIncludeH2;
*pH2T(q,d,t,y)               = (sum(hh,vH2PwrIn.l(hh,q,d,t,y)*pH2Data(hh,"HeatRate")))$pIncludeH2;
pUnmetH2 (z,q,y)            = vUnmetExternalH2.l(z,q,y);

************************************************************************************************************************************

pFuel(z,y) = sum((gzmap(g,z),gfmap(g,f),zcmap(z,c),q,d,t,s), pProbaScenarios(s)*vPwrOut.l(g,f,q,d,t,y,s)*pHours(q,d,t)*pFuelPrice(c,f,y)*pHeatRate(g,f));

parameter pSpinningCheck(f,c,y);

pSpinningReserveByFuelByCountry(f,c,y) = sum((g,q,d,t,z,s)$(gfmap(g,f) AND gzmap(g,z) AND zcmap(z,c)), pProbaScenarios(s)*vReserve.L(g,q,d,t,y,s)*pHours(q,d,t));
pSpinningCheck(f,c,y) = sum((g,q,d,t,z,s)$(gfmap(g,f) AND gzmap(g,z) AND zcmap(z,c)),pProbaScenarios(s)*vReserve.L(g,q,d,t,y,s));


pImportCosts(z,y) = sum((q,d,t,s), pProbaScenarios(s)*vImportPrice.l(z,q,d,t,y,s)*pTradePrice(z,q,d,y,t)*pHours(q,d,t));

pExportRevenue(z,y) = sum((q,d,t,s), pProbaScenarios(s)*vExportPrice.l(z,q,d,t,y,s)*pTradePrice(z,q,d,y,t)*pHours(q,d,t));

pNewTransmissionCosts(z,y) = vYearlyTransmissionAdditions.l(z,y);
pUSECosts(z,y) = vYearlyUSECost.l(z,y);
pCO2backstopCosts(c,y) = vYearlyCO2backCost.l(c,y);


pSurplusCosts(z,y) = vYearlySurplus.l(z,y);
pVRECurtailment(z,y) = vYearlyCurtailmentCost.l(z,y);

* country level
pCountryPlanReserveCosts(c,y) = sum(s, pProbaScenarios(s)*vUnmetReserve.l(c,y,s)*pReserveVoLL);
pUSRLocCosts(c,y) = sum((q,d,t,s), pProbaScenarios(s)*vUnmetSpinLoc.l(c,q,d,t,y,s)*pHours(q,d,t)*pSpinReserveVoLL);

* System level
pUSRSysCosts(y) = sum((q,d,t,s), pProbaScenarios(s)*vUnmetSpin.l(q,d,t,y,s)*pHours(q,d,t)*pSpinReserveVoLL);
pUPRSysCosts(y) = vUnmetSysReserve.l(y)*pReserveVoLL;

set sumhdr /
   "Capex: $m"
   "Annualized capex: $m"
   "Fixed O&M: $m"
   "Variable O&M: $m"
   "Total fuel Costs: $m"
   "Transmission additions: $m"
   "Spinning Reserve costs: $m"
   "Unmet demand costs: $m"
   "Excess generation: $m"
   "VRE curtailment: $m"
   "Import costs: $m"
   "Export revenue: $m" /;
set avgsumhdr /
   "Average Capex: $m"
   "Average Annualized capex: $m"
   "Average Fixed O&M: $m"
   "Average Variable O&M: $m"
   "Average Total fuel Costs: $m"
   "Average Transmission additions: $m"
   "Average Spinning Reserve costs: $m"
   "Average Unmet demand costs: $m"
   "Average Excess generation: $m"
   "Average VRE curtailment: $m"
   "Average Import costs: $m"
   "Average Export revenue: $m" /;
set sumhdrmap(avgsumhdr,sumhdr) / #avgsumhdr:#sumhdr /;

*--- Cost Summary Unweighted by zone
pCostSummary(z,"Capex: $m"                    ,y) = pCapex(z,y)/1e6;
pCostSummary(z,"Annualized capex: $m"         ,y) = pAnncapex(z,y)/1e6;
pCostSummary(z,"Fixed O&M: $m"                ,y) = pFOM(z,y)/1e6;
pCostSummary(z,"Variable O&M: $m"             ,y) = pVOM(z,y)/1e6;
pCostSummary(z,"Total fuel Costs: $m"         ,y) = pFuel(z,y)/1e6;
pCostSummary(z,"Transmission additions: $m"   ,y) = pNewTransmissionCosts(z,y)/1e6;
pCostSummary(z,"Spinning Reserve costs: $m"   ,y) = pSpinResCosts(z,y)/1e6;
pCostSummary(z,"Unmet demand costs: $m"       ,y) = pUSECosts(z,y)/1e6;
pCostSummary(z,"Excess generation: $m"        ,y) = pSurplusCosts(z,y)/1e6;
pCostSummary(z,"VRE curtailment: $m"          ,y) = pVRECurtailment(z,y)/1e6;
pCostSummary(z,"Import costs: $m"             ,y) = pImportCosts(z,y)/1e6;
pCostSummary(z,"Export revenue: $m"           ,y) = pExportRevenue(z,y)/1e6;
pCostSummary(z,"Total Annual Cost by Zone: $m",y) = ( pAnncapex(z,y) + pNewTransmissionCosts(z,y) + pFOM(z,y) + pVOM(z,y) + pFuel(z,y)
                                                    + pImportCosts(z,y) - pExportRevenue(z,y) + pUSECosts(z,y) + pVRECurtailment(z,y)
                                                    + pSurplusCosts(z,y) + pSpinResCosts(z,y))/1e6;

pCostSummary(z,"Total H2 costs by Zone: $m",y) = sum(h2zmap(dch2,z), vAnnCapexH2.l(dch2,y))
                                                +1e6*sum(h2zmap(ndcH2,z),pCRFH2(ndcH2)*vCapH2.l(ndcH2,y)*pH2Data(ndcH2,"Capex"))$pIncludeH2
                                                +sum(h2zmap(hh,z),  vCapH2.l(hh,y)*pH2Data(hh,"FOMperMW"))$pIncludeH2
                                                + sum((q,d,t), pHours(q,d,t)*(sum((h2zmap(hh,z)), pH2Data(hh,"VOM")*pH2Data(hh,"Heatrate")*vH2PwrIn.l(hh,q,d,t,y))$pIncludeH2));




******************Turkey model related output**************************

*pCostSummaryM(z,"Monthly fuel Costs: $m",y,q)            = pFuelM(z,y,q)/1e6;
*pCostSummaryM(z,"Monthly Variable O&M: $m"  ,y,q)        = pVOMm(z,y,q)/1e6;
*pCostSummaryM(z,"Fixed O&M: $m" ,y,q)                    = pFOM(z,y)/12/1e6;
*pCostSummaryM(z,"Monthly Unmet demand costs: $m"       ,y,q) = pUSECostsM(z,y,q)/1e6;
*pCostSummaryM(z,"Monthly VRE curtailment: $m"          ,y,q) = pVRECurtailmentM(z,y,q)/1e6;
*pCostSummaryM(z,"Total Monthly Cost by Zone: $m",y,q)=((pAnncapex(z,y)+pNewTransmissionCosts(z,y) + pFOM(z,y))/12+pVOMm(z,y,q)+pFuelM(z,y,q)+pImportCostsM(z,y,q)
*                                                                     -pExportRevenueM(z,y,q) + pUSECostsM(z,y,q) + pVRECurtailmentM(z,y,q)
*                                                                     + pSurplusCostsM(z,y,q) +pSpinResCostsM(z,y,q))/1e6;



**********************************************************************************

*--- Cost Summary Unweighted by country
pCostSummaryCountry(c,sumhdr,y) = sum(zcmap(z,c), pCostSummary(z,sumhdr,y));
pCostSummaryCountry(c,"Country Spinning Reserve violation: $m",y) = pUSRLocCosts(c,y)/1e6;
pCostSummaryCountry(c,"Country Planning Reserve violation: $m",y) = pCountryPlanReserveCosts(c,y)/1e6;
pCostSummaryCountry(c,"Total Annual Cost by Country: $m"      ,y) = sum(zcmap(z,c), pCostSummary(z,"Total Annual Cost by Zone: $m",y))
                                                                  + (pUSRLocCosts(c,y) + pCountryPlanReserveCosts(c,y)+pCO2backstopCosts(c,y))/1e6 ;
pCostSummaryCountry(c,"Total CO2 backstop cost by Country: $m",y) = pCO2backstopCosts(c,y)/1e6 ;




*--- Cost Summary Weighted by zone
pCostSummaryWeighted(z,sumhdr,y) = pWeightYear(y)*pCostSummary(z,sumhdr,y);
pCostSummaryWeighted(z,"Total Annual Cost by Zone: $m",y) = pWeightYear(y)*(pAnncapex(z,y) + pNewTransmissionCosts(z,y) + pFOM(z,y) + pVOM(z,y)
                                                                          + pFuel(z,y) + pImportCosts(z,y) - pExportRevenue(z,y) + pUSECosts(z,y)
                                                                          + pVRECurtailment(z,y) + pSurplusCosts(z,y) + pSpinResCosts(z,y))/1e6;

*--- Cost Summary Weighted by country
pCostSummaryWeightedCountry(c,sumhdr,y) = sum(zcmap(z,c), pWeightYear(y)*pCostSummary(z,sumhdr,y));
pCostSummaryWeightedCountry(c,"Country Spinning Reserve violation: $m",y) = pWeightYear(y)*pUSRLocCosts(c,y)/1e6;
pCostSummaryWeightedCountry(c,"Country Planning Reserve violation: $m",y) = pWeightYear(y)*pCountryPlanReserveCosts(c,y)/1e6;
pCostSummaryWeightedCountry(c,"Total Annual Cost by Country: $m"      ,y) = pWeightYear(y)*sum(zcmap(z,c), pCostSummaryWeighted(z,"Total Annual Cost by Zone: $m",y))
                                                                          + pWeightYear(y)*(pCountryPlanReserveCosts(c,y) + pUSRLocCosts(c,y))/1e6;
pCostSummaryWeightedCountry(c,"Total CO2 backstop cost by Country: $m"      ,y) = pWeightYear(y)*pCO2backstopCosts(c,y)/1e6 ;

*--- Cost Summary Weighted Averages by country
pCostSummaryWeightedAverageCountry(c,avgsumhdr,'Summary') = sum((sumhdrmap(avgsumhdr,sumhdr),y), pCostSummaryWeightedCountry(c,sumhdr,y))/TimeHorizon;
pCostSummaryWeightedAverageCountry(c,"Average Spinning Reserve violation: $m","Summary") = sum(y, pCostSummaryWeightedCountry(c,"Country Spinning Reserve violation: $m",y))/TimeHorizon;
pCostSummaryWeightedAverageCountry(c,"Average Planning Reserve violation: $m","Summary") = sum(y, pCostSummaryWeightedCountry(c,"Country Planning Reserve violation: $m",y))/TimeHorizon;
pCostSummaryWeightedAverageCountry(c,"Average Total Annual Cost: $m"         ,"Summary") = sum(y, pCostSummaryWeightedCountry(c,"Total Annual Cost by Country: $m",y))/TimeHorizon;
pCostSummaryWeightedAverageCountry(c,"Total CO2 backstop cost by Country: $m" ,"Summary") = sum(y, pCostSummaryWeightedCountry(c,"Total CO2 backstop cost by Country: $m",y))/TimeHorizon;


pCostSummaryWeightedAverageCtry(c,"Average Capex: $m")                      = pCostSummaryWeightedAverageCountry(c,"Average Capex: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Annualized capex: $m")           = pCostSummaryWeightedAverageCountry(c,"Average Annualized capex: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Transmission additions: $m")     = pCostSummaryWeightedAverageCountry(c,"Average Transmission additions: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Fixed O&M: $m")                  = pCostSummaryWeightedAverageCountry(c,"Average Fixed O&M: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Variable O&M: $m")               = pCostSummaryWeightedAverageCountry(c,"Average Variable O&M: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Total fuel Costs: $m")           = pCostSummaryWeightedAverageCountry(c,"Average Total fuel Costs: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Spinning Reserve costs: $m")     = pCostSummaryWeightedAverageCountry(c,"Average Spinning Reserve costs: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Unmet demand costs: $m")         = pCostSummaryWeightedAverageCountry(c,"Average Unmet demand costs: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Excess generation: $m")          = pCostSummaryWeightedAverageCountry(c,"Average Excess generation: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average VRE curtailment: $m")            = pCostSummaryWeightedAverageCountry(c,"Average VRE curtailment: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Import costs: $m")               = pCostSummaryWeightedAverageCountry(c,"Average Import costs: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Export revenue: $m")             = pCostSummaryWeightedAverageCountry(c,"Average Export revenue: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Spinning Reserve violation: $m") = pCostSummaryWeightedAverageCountry(c,"Average Spinning Reserve violation: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Planning Reserve violation: $m") = pCostSummaryWeightedAverageCountry(c,"Average Planning Reserve violation: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Average Total Annual Cost: $m")          = pCostSummaryWeightedAverageCountry(c,"Average Total Annual Cost: $m","Summary");
pCostSummaryWeightedAverageCtry(c,"Total CO2 backstop cost by Country: $m" )= pCostSummaryWeightedAverageCountry(c,"Total CO2 backstop cost by Country: $m" ,"Summary");

*--- Cost and consumption by fuel

* By zone
pFuelCosts(z,f,y) = sum((gzmap(g,z),gfmap(g,f),zcmap(z,c),q,d,t,s), pProbaScenarios(s)*vPwrOut.l(g,f,q,d,t,y,s)*pHours(q,d,t)*pFuelPrice(c,f,y)*pHeatRate(g,f))/1e6;
pFuelConsumption(z,f,y) = sum(s, pProbaScenarios(s)*vFuel.l(z,f,y,s)/1e6);

* By country
pFuelCostsCountry(c,f,y) = sum(zcmap(z,c), pFuelCosts(z,f,y));
pFuelConsumptionCountry(c,f,y) = sum(zcmap(z,c), pFuelConsumption(z,f,y));


*--- Energy Results (Energy by Fuel, by Plant and Energy mix)
set zgmap(z,g); option zgmap<gzmap;


**************************Hydrogen model****************************
set zH2map(z,hh); option zH2map<h2zmap;
************************************************************************

pEnergyByFuel(z,f,y) = sum((gzmap(g,z),gfmap(g,f),q,d,t,s), pProbaScenarios(s)*vPwrOut.l(g,f,q,d,t,y,s)*pHours(q,d,t))/1e3;
pEnergyByFuelCountry(c,f,y) = sum(zcmap(z,c), pEnergyByFuel(z,f,y));
*pEnergyMix(c,f,y) = pEnergyByFuelCountry(c,f,y)/(sum(f2, pEnergyByFuelCountry(c,f2,y)));

pEnergyByPlant(zgmap(z,g),y) = sum((gfmap(g,f),q,d,t,s), pProbaScenarios(s)*vPwrOut.l(g,f,q,d,t,y,s)*pHours(q,d,t))/1e3;

*--- Demand-Supply Balance

*A. Demand - Production Balance

* By zone
pDemandSupply(z,"Demand: GWh"            ,y) = sum((q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))/1e3;
pDemandSupply(z,"Total production: GWh"  ,y) = sum(gzmap(g,z), pEnergyByPlant(z,g,y));
pDemandSupply(z,"Unmet demand: GWh"      ,y) = sum((q,d,t,s), pProbaScenarios(s)*vUSE.l(z,q,d,t,y,s)*pHours(q,d,t))/1e3;
pDemandSupply(z,"Surplus generation: GWh",y) = sum((q,d,t,s), pProbaScenarios(s)*vSurplus.l(z,q,d,t,y,s)*pHours(q,d,t))/1e3;

pDemandSupply(z,"Imports exchange: GWh"     ,y) = (sum((sTopology(z,z2),q,d,t,s), pProbaScenarios(s)*vFlow.l(z2,z,q,d,t,y,s)*pHours(q,d,t)) + sum((q,d,t,s), pProbaScenarios(s)*vImportPrice.l(z,q,d,t,y,s)*pHours(q,d,t)))/1e3;
pDemandSupply(z,"Exports exchange: GWh"     ,y) = (sum((sTopology(z,z2),q,d,t,s), pProbaScenarios(s)*vFlow.l(z,z2,q,d,t,y,s)*pHours(q,d,t)) + sum((q,d,t,s), pProbaScenarios(s)*vExportPrice.l(z,q,d,t,y,s)*pHours(q,d,t)))/1e3;
pDemandSupply(z,"Net interchange: GWh"      ,y) = pDemandSupply(z,"Imports exchange: GWh",y) - pDemandSupply(z,"Exports exchange: GWh",y);
pDemandSupply(z,"Net interchange Ratio: GWh",y)$pDemandSupply(z,"Demand: GWh",y) = pDemandSupply(z,"Net interchange: GWh",y)/pDemandSupply(z,"Demand: GWh",y);

$if %DEBUG%==1 display  pDemandSupply;

* By Country
pDemandSupplyCountry(c,"Demand: GWh"            ,y) = sum(zcmap(z,c), pDemandSupply(z,"Demand: GWh"            ,y));
pDemandSupplyCountry(c,"Total production: GWh"  ,y) = sum(zcmap(z,c), pDemandSupply(z,"Total production: GWh"  ,y));
pDemandSupplyCountry(c,"Unmet demand: GWh"      ,y) = sum(zcmap(z,c), pDemandSupply(z,"Unmet demand: GWh"      ,y));
pDemandSupplyCountry(c,"Surplus generation: GWh",y) = sum(zcmap(z,c), pDemandSupply(z,"Surplus generation: GWh",y));

pDemandSupplyCountry(c,"Imports exchange: GWh"     ,y) = (sum((zcmap(z,c),sMapNCZ(z2,z),q,d,t,s), pProbaScenarios(s)*vFlow.l(z2,z,q,d,t,y,s)*pHours(q,d,t)) + sum((zcmap(z,c),q,d,t,s), pProbaScenarios(s)*vImportPrice.l(z,q,d,t,y,s)*pHours(q,d,t)))/1e3;
pDemandSupplyCountry(c,"Exports exchange: GWh"     ,y) = (sum((zcmap(z,c),sMapNCZ(z,z2),q,d,t,s), pProbaScenarios(s)*vFlow.l(z,z2,q,d,t,y,s)*pHours(q,d,t)) + sum((zcmap(z,c),q,d,t,s), pProbaScenarios(s)*vExportPrice.l(z,q,d,t,y,s)*pHours(q,d,t)))/1e3;
pDemandSupplyCountry(c,"Net interchange: GWh"      ,y) = pDemandSupplyCountry(c,"Imports exchange: GWh",y) - pDemandSupplyCountry(c,"Exports exchange: GWh",y);
pDemandSupplyCountry(c,"Net interchange Ratio: GWh",y)$pDemandSupplyCountry(c,"Demand: GWh",y) = pDemandSupplyCountry(c,"Net interchange: GWh",y)/pDemandSupplyCountry(c,"Demand: GWh",y);

pNPVbyCountry(c,y) = pRR(y)*pWeightYear(y)*(vYearlyTotalCost.L(c,y));

pNPVByYear(y) =  pRR(y)*pWeightYear(y)*(sum(c, vYearlyTotalCost.L(c,y)) + vYearlySysUSRCost.L(y));
*I remove +vYearlySysCO2backCost.L(y)

pNPVCostSpecific('Unserved Spinning Reserve Cost') = sum(y, pRR(y)*pWeightYear(y)*(sum(c, vYearlyUSRCost.L(c,y))));
pNPVCostSpecific('CO2 Backstop Cost') = sum(y, pRR(y)*pWeightYear(y)*(sum(c, vYearlyCO2backCost.L(c,y))));
pNPVCostSpecific('Fixed Cost') = sum(y, pRR(y)*pWeightYear(y)*(sum(c, sum(zcmap(z,c), vYearlyFixedCost.L(z,y)))));
pNPVCostSpecific('Variable Cost') = sum(y, pRR(y)*pWeightYear(y)*(sum(c, sum(zcmap(z,c), vYearlyVariableCost.L(z,y)))));
pNPVCostSpecific('Reserve Cost') = sum(y, pRR(y)*pWeightYear(y)*(sum(c, sum(zcmap(z,c), vYearlyReserveCost.L(z,y)))));
pNPVCostSpecific('Unserved Energy Cost') = sum(y, pRR(y)*pWeightYear(y)*(sum(c, sum(zcmap(z,c), vYearlyUSECost.L(z,y)))));
pNPVCostSpecific('Carbon Cost') = sum(y, pRR(y)*pWeightYear(y)*(sum(c, sum(zcmap(z,c), vYearlyCarbonCost.L(z,y)))));
pNPVCostSpecific('Trade Cost') = sum(y, pRR(y)*pWeightYear(y)*(sum(c, sum(zcmap(z,c), vYearlyTradeCost.L(z,y)))));
pNPVCostSpecific('Transmission Cost') = sum(y, pRR(y)*pWeightYear(y)*(sum(c, sum(zcmap(z,c), vYearlyTransmissionAdditions.L(z,y)))));
pNPVCostSpecific('Curtailment Cost') = sum(y, pRR(y)*pWeightYear(y)*(sum(c, sum(zcmap(z,c), vYearlyCurtailmentCost.L(z,y)))));
pNPVCostSpecific('Surplus Cost') = sum(y, pRR(y)*pWeightYear(y)*(sum(c, sum(zcmap(z,c), vYearlySurplus.L(z,y)))));
pNPVCostSpecific('System Unserved Reserve Cost') = sum(y, pRR(y)*pWeightYear(y)*vYearlySysUSRCost.L(y));
pNPVCostSpecific('System Backstop Cost') = sum((y,s), pRR(y)*pWeightYear(y)*pProbaScenarios(s)*vYearlySysCO2backstop.L(y,s)* pCostOfCO2backstop);

pCAPEXByFuel(f) = sum((g,y)$(ndc(g) AND gprimf(g,f)), vBuild.l(g,y)*pGenData(g,"Capex")+ vBuildStor.l(g,y)*(pStorData(g,"Capex")+pCSPData(g,"Storage","Capex"))/1e3+vBuildTherm.l(g,y)*pCSPData(g,"Thermal Field","Capex")) + sum((g,y)$(dc(g) AND gprimf(g,f)), vBuild.l(g,y)*pGenData(g,"Capex")*pCapexTrajectories(g,y)+(vBuildStor.l(g,y)*(pStorData(g,"Capex")+pCSPData(g,"Storage","Capex"))/1e3+vBuildTherm.l(g,y)*pCSPData(g,"Thermal Field","Capex"))*pCapexTrajectories(g,y));

pCAPEXByFuelByYear(y,f) = sum((g)$(ndc(g) AND gprimf(g,f)), vBuild.l(g,y)*pGenData(g,"Capex")+ vBuildStor.l(g,y)*(pStorData(g,"Capex")+pCSPData(g,"Storage","Capex"))/1e3+vBuildTherm.l(g,y)*pCSPData(g,"Thermal Field","Capex")) + sum((g)$(dc(g) AND gprimf(g,f)), vBuild.l(g,y)*pGenData(g,"Capex")*pCapexTrajectories(g,y)+(vBuildStor.l(g,y)*(pStorData(g,"Capex")+pCSPData(g,"Storage","Capex"))/1e3+vBuildTherm.l(g,y)*pCSPData(g,"Thermal Field","Capex"))*pCapexTrajectories(g,y));

parameter
pCAPEXByFuelByZoneByYear(y,f,z),pCAPEXByFuelByCountryByYear(y,f,c);

pCAPEXByFuelByZoneByYear(y,f,z) = sum((g)$(ndc(g) AND gzmap(g,z)and gprimf(g,f)  ), vBuild.l(g,y)*pGenData(g,"Capex")+ vBuildStor.l(g,y)*(pStorData(g,"Capex")+pCSPData(g,"Storage","Capex"))/1e3+vBuildTherm.l(g,y)*pCSPData(g,"Thermal Field","Capex")) + sum((g)$(dc(g) AND gprimf(g,f)), vBuild.l(g,y)*pGenData(g,"Capex")*pCapexTrajectories(g,y)+(vBuildStor.l(g,y)*(pStorData(g,"Capex")+pCSPData(g,"Storage","Capex"))/1e3+vBuildTherm.l(g,y)*pCSPData(g,"Thermal Field","Capex"))*pCapexTrajectories(g,y));

pCAPEXByFuelByCountryByYear(y,f,c)  = sum(zcmap(z,c), pCAPEXByFuelByZoneByYear(y,f,z) );

*--- Summary of results
pSummary("NPV of system cost: $m"          ) = vNPVCost.l/1e6;
pSummary("Total Generation: GWh"           ) = sum((gzmap(g,z),y), pWeightYear(y)*pEnergyByPlant(z,g,y));
pSummary("Total Demand: GWh"               ) = sum((z,y), pWeightYear(y)*pDemandSupply(z,"Demand: GWh",y));
pSummary("Total Capacity Added: MW"        ) = sum((g,y), vBuild.l(g,y));
*pSummary("Total Investment: $m"            ) = sum((g,y)$(ndc(g)), vBuild.l(g,y)*pGenData(g,"Capex")) + sum((g,y)$(dc(g)), vBuild.l(g,y)*pGenData(g,"Capex")*pCapexTrajectories(g,y));
pSummary("Total Investment: $m"            ) = sum((g,y)$(ndc(g)), vBuild.l(g,y)*pGenData(g,"Capex")+ vBuildStor.l(g,y)*(pStorData(g,"Capex")+pCSPData(g,"Storage","Capex"))/1e3+vBuildTherm.l(g,y)*pCSPData(g,"Thermal Field","Capex")) + sum((g,y)$(dc(g)), vBuild.l(g,y)*pGenData(g,"Capex")*pCapexTrajectories(g,y)+(vBuildStor.l(g,y)*(pStorData(g,"Capex")+pCSPData(g,"Storage","Capex"))/1e3+vBuildTherm.l(g,y)*pCSPData(g,"Thermal Field","Capex"))*pCapexTrajectories(g,y));
pSummary("Total USE: GWh"                  ) = sum((z,y), pWeightYear(y)*pDemandSupply(z,"Unmet demand: GWh",y));
pSummary("Sys Spin Reserve violation: $m"  ) = sum(y, pWeightYear(y)*pRR(y)*pUSRSysCosts(y))/1e6;
pSummary("Sys Plan Reserve violation: $m"  ) = sum(y, pWeightYear(y)*pRR(y)*pUPRSysCosts(y))/1e6;
pSummary("Zonal Spin Reserve violation: $m") = sum((c,y), pWeightYear(y)*pRR(y)*pUSRLocCosts(c,y))/1e6;
pSummary("Zonal Plan Reserve violation: $m") = sum((c,y), pWeightYear(y)*pRR(y)*pCountryPlanReserveCosts(c,y))/1e6;
pSummary("Excess Generation Costs: $m"     ) = sum((z,y), pWeightYear(y)*pRR(y)*eYearlySurplusCost.l(z,y))/1e6;
pSummary("Carbon costs: $m"                ) = sum((z,y), pWeightYear(y)*pRR(y)*vYearlyCarbonCost.l(z,y))/1e6;
pSummary("Trade costs: $m"                 ) = sum((z,y), pWeightYear(y)*pRR(y)*vYearlyTradeCost.l( z,y))/1e6;
pSummary("VRE curtailment: $m"             ) = sum((z,y), pWeightYear(y)*pRR(y)*pVRECurtailment(z,y))/1e6;
pSummary("Spinning reserve costs: $m"      ) = sum((z,y), pWeightYear(y)*pRR(y)*pSpinResCosts(z,y))/1e6;
pSummary("Total Emission: mt"              ) = sum((z,y,s), pWeightYear(y)*pProbaScenarios(s)*vZonalEmissions.l(z,y,s))/1e6;
pSummary("Climate backstop cost: $m"       ) = sum((c,y), pWeightYear(y)*pRR(y)*pCO2backstopCosts(c,y))/1e6+sum((y,s), pWeightYear(y)*pRR(y)*pCostOfCO2backstop*vYearlySysCO2backstop.l(y,s))/1e6;
pSummary("Total curtailed VRE: GWh"        ) = sum(y,pCurtailedVRETperY(y));
pSummary("Total H2 production: mmBTU"      ) = sum(y,pH2perY(y));

*B. Interchange and Losses with INTERNAL zones

PARAMETER 

pInterchangeHourly(z,z2,q,d,t,y);
pInterchangeHourly(z,z2,q,d,t,y) = sum(s, pProbaScenarios(s)*vFlow.l(z,z2,q,d,t,y,s));


* By zone
pInterchange(sTopology(z,z2),y) = sum((q,d,t,s), pProbaScenarios(s)*vFlow.l(z,z2,q,d,t,y,s)*pHours(q,d,t))/1e3;
pInterconUtilization(sTopology(z,z2),y)$pInterchange(z,z2,y) = 1e3*(sum((q,d,t,s),pProbaScenarios(s)*(vFlow.l(z,z2,q,d,t,y,s) + vFlow.l(z2,z,q,d,t,y,s))*pHours(q,d,t))/1e3 )
                                                              /sum((q,d,t),( max(pTransferLimit(z,z2,q,y),pTransferLimit(z2,z,q,y)) + vAdditionalTransfer.l(z,z2,y)
                                                                                                     * max(pNewTransmission(z,z2,"CapacityPerLine"),
                                                                                                           pNewTransmission(z2,z,"CapacityPerLine"))
                                                      )*pHours(q,d,t));
pLossesTransmission(z,y) = sum((sTopology(z,z2),q,d,t,s), pProbaScenarios(s)*vFlow.l(z2,z,q,d,t,y,s)*pLossFactor(z,z2,y)*pHours(q,d,t));

**By country
alias (zcmap, zcmap2);
pInterchangeCountry(c,c2,y)= sum((zcmap(z,c),zcmap2(z2,c2),sMapNCZ(z2,z)), pInterchange(z,z2,y));
pLossesTransmissionCountry(c,y) = sum(zcmap(z,c), pLossesTransmission(z,y));

PARAMETER pNetInterchange(c,c2,y), pLineInterchange(c,c2,y);
pLineInterchange(c,c2,y) = 0;
pNetInterchange(c,c2,y) = pInterchangeCountry(c,c2,y) - pInterchangeCountry(c2,c,y);
pNetInterchange(c,c2,y)$(pNetInterchange(c,c2,y)<0) = 0;

pLineInterchange(c,c2,y)$(not pLineInterchange(c2,c,y)) = pInterchangeCountry(c,c2,y) + pInterchangeCountry(c2,c,y);

*C. Trade with EXTERNAL zones

* By Zone
set thrd / Imports, Exports /;
pHourlyTrade(z,y,q,d,"Imports",t) =  sum(s, pProbaScenarios(s)*vImportPrice.l(z,q,d,t,y,s));
pHourlyTrade(z,y,q,d,"Exports",t) =  sum(s, pProbaScenarios(s)*vExportPrice.l(z,q,d,t,y,s));
pYearlyTrade(z,thrd,y) = sum((q,d,t), pHourlyTrade(z,y,q,d,thrd,t)*pHours(q,d,t))/1e3;

* By Country
pHourlyTradeCountry(c,y,q,d,thrd,t) = sum(zcmap(z,c), pHourlyTrade(z,y,q,d,thrd,t));
pYearlyTradeCountry(c,thrd,y) = sum(zcmap(z,c), pYearlyTrade(z,thrd,y));

*---   Prices Costs -- Marginal costs

* By zone
pPrice(z,q,d,t,y) = -sum(s, eDemSupply.m(z,q,d,t,y,s)*pProbaScenarios(s))/pHours(q,d,t)/pRR(y)/pWeightYear(y);
pAveragePrice(z,y)$pDemandSupply(z,"Demand: GWh",y) = 1e-3*sum((q,d,t),pPrice(z,q,d,t,y)*pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))
                                                     /pDemandSupply(z,"Demand: GWh",y) ;

Parameter zzFlow(z,z2,y), zzFlowpH(z,z2,y);
zzFlow(z,z2,y) = sum(sFlow(z,z2,q,d,t,y,s),vFlow.l(z,z2,q,d,t,y,s)*pProbaScenarios(s));
zzFlowpH(z,z2,y) = sum(sFlow(z,z2,q,d,t,y,s),vFlow.l(z,z2,q,d,t,y,s)*pProbaScenarios(s)*pHours(q,d,t));
pAveragePriceExp(z,y) $(sum(Zd, zzFlowpH(z,Zd,y))  > 0) = sum((sTopology(z,Zd),q,d,t,s),  pPrice(z,q,d,t,y) *vFlow.l(z,Zd,q,d,t,y,s)*pProbaScenarios(s)*pHours(q,d,t))
                                                         /sum(Zd, zzFlowpH(z,Zd,y));
pAveragePriceImp(z,y) $(sum(Zd, zzFlowpH(Zd,z,y))  > 0) = sum((sTopology(Zd,z),q,d,t,s),  pPrice(Zd,q,d,t,y)*vFlow.l(Zd,z,q,d,t,y,s)*pProbaScenarios(s)*pHours(q,d,t))
                                                         /sum(Zd, zzFlowpH(Zd,z,y));
pAveragePriceHub(Zt,y)$(sum(Zd, zzFlow(Zt,Zd,y))   > 0) = sum((sTopology(Zt,Zd),q,d,t,s), pPrice(Zt,q,d,t,y)*vFlow.l(Zt,Zd,q,d,t,y,s)*pProbaScenarios(s))
                                                         /sum(Zd, zzFlow(Zt,Zd,y));

* By Country
pAveragePriceCountry(c,y)$pDemandSupplyCountry(c,"Demand: GWh",y) = sum(zcmap(z,c), pAveragePrice(z,y)*pDemandSupply(z,"Demand: GWh",y))/pDemandSupplyCountry(c,"Demand: GWh",y);

Parameter cFlowpH(c,y);
cFlowpH(c,y) = sum((zcmap(z,c),sMapNCZ(z,Zd)), zzFlowpH(z,Zd,y));
pAveragePriceExpCountry(c,y)$(cFlowpH(c,y) > 0) = sum((zcmap(z,c),sMapNCZ(z,Zd),q,d,t,s), pPrice(z,q,d,t,y)* pProbaScenarios(s)*vFlow.l(z,Zd,q,d,t,y,s)*pHours(q,d,t))
                                                 /cFlowpH(c,y);
cFlowpH(c,y) = sum((zcmap(z,c),sMapNCZ(Zd,z)), zzFlowpH(Zd,z,y));
pAveragePriceImpCountry(c,y)$(cFlowpH(c,y) > 0) = sum((zcmap(z,c),sMapNCZ(Zd,z),q,d,t,s), pPrice(Zd,q,d,t,y)*pProbaScenarios(s)*vFlow.l(Zd,z,q,d,t,y,s)*pHours(q,d,t))
                                                 /cFlowpH(c,y);
*pAveragePriceHubCountry(Zt,y)$(sum((Zd,q,d,t),vFlow.l(Zt,Zd,q,d,t,y))>0) = sum((Zd,q,d,t)$sTopology(Zt,Zd), pPrice(Zt,q,d,t,y) * vFlow.l(Zt,Zd,q,d,t,y))/ sum((Zd,q,d,t),vFlow.l(Zt,Zd,q,d,t,y)) ;

pAveragePriceExp1(z,z2,y) = sum((sTopology(z,z2),q,d,t,s),  pPrice(z,q,d,t,y) * pProbaScenarios(s)*vFlow.l(z,z2,q,d,t,y,s)*pHours(q,d,t));                                                                                                            
pAveragePriceImp1(z,z2,y) = sum((sTopology(z2,z),q,d,t,s),  pPrice(z2,q,d,t,y) * pProbaScenarios(s)*vFlow.l(z2,z,q,d,t,y,s)*pHours(q,d,t));

*--- Capacity and Utilization Results

* By Zone
pPeakCapacity(z,"Available capacity: MW",y) = sum(gzmap(g,z), vCap.l(g,y));
pPeakCapacity(z,"Peak demand: MW"       ,y) = smax((q,d,t), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y));
pPeakCapacity(z,"New capacity: MW"      ,y) = sum(gzmap(g,z), vBuild.l(g,y));
pPeakCapacity(z,"Retired capacity: MW"  ,y) = sum(gzmap(g,z), vRetire.l(g,y));
pCapacityByFuel(z,f,y)  = sum((gzmap(g,z),gprimf(g,f)), vCap.l(g,y));
pNewCapacityFuel(z,f,y) = sum((gzmap(g,z),gprimf(g,f)), vBuild.l(g,y));

* By Country
pPeakCapacityCountry(c,"Available capacity: MW",y) = sum(zcmap(z,c), pPeakCapacity(z,"Available capacity: MW",y));
pPeakCapacityCountry(c,"Peak demand: MW"       ,y) = smax((q,d,t), sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y)));
pPeakCapacityCountry(c,"New capacity: MW"      ,y) = sum(zcmap(z,c), pPeakCapacity(z,"New capacity: MW",y));
pPeakCapacityCountry(c,"Retired capacity: MW"  ,y) = sum(zcmap(z,c), pPeakCapacity(z,"Retired capacity: MW",y));
pCapacityByFuelCountry(c,f,y)  = sum(zcmap(z,c), pCapacityByFuel(z,f,y));
pNewCapacityFuelCountry(c,f,y) = sum(zcmap(z,c), pNewCapacityFuel(z,f,y));
pCapacityPlanCountry(c,g,y)    = sum((zcmap(z,c),gzmap(g,z)), vCap.l(g,y));

* By Plant
pCapacityPlan(zgmap(z,g),y)                 = vCap.l(g,y);
pRetirements(zgmap(z,g),y)                  = vRetire.l(g,y);
pRetirementsByFuel(f,y)                     = sum(g$(gfmap(g,f)), vRetire.l(g,y));

pPlantUtilization(zgmap(z,g),y)$vCap.l(g,y) = sum((gfmap(g,f),q,d,t,s), pProbaScenarios(s)*vPwrOut.l(g,f,q,d,t,y,s)*pHours(q,d,t))/vCap.l(g,y)/8760;

*--- New TX Capacity by zone
pAdditionalCapacity(z,z2,y) = vAdditionalTransfer.l(z,z2,y)*pNewTransmission(z,z2,"CapacityPerLine");
$if %DEBUG%==1 display pAdditionalCapacity;

pAnnualTransmissionCapacity(z,z2,y) = smax(q,pTransferLimit(z,z2,q,y)) + vAdditionalTransfer.l(z,z2,y)*pNewTransmission(z,z2,"CapacityPerLine");

*--- Reserve Results
* By Zone
pReserveCosts(zgmap(z,g),y)   = sum((q,d,t,s), pProbaScenarios(s)*vReserve.l(g,q,d,t,y,s)*pGenData(g,"ReserveCost")*pHours(q,d,t))/1e6 ;
pReserveByPlant(zgmap(z,g),y) = sum((q,d,t,s), pProbaScenarios(s)*vReserve.l(g,q,d,t,y,s)*pHours(q,d,t))/1e3 ;

* By Country
pReserveCostsCountry(c,g,y)  =  sum((zcmap(z,c),zgmap(z,g)), pReserveCosts(z,g,y));
pReserveByPlantCountry(c,g,y)=  sum((zcmap(z,c),zgmap(z,g)), pReserveByPlant(z,g,y));

*--- Emission Results
* By Zone
pEmissions(z,y)                                                     =  sum(s, pProbaScenarios(s)*vZonalEmissions.l(z,y,s))/1e6 ;
pEmissionsIntensity(z,y)$pDemandSupply(z,"Total production: GWh",y) = 1e-3*sum(s, pProbaScenarios(s)*vZonalEmissions.l(z,y,s))/pDemandSupply(z,"Total production: GWh",y);

pEmissionsCountry(c,y)                                                            = sum(zcmap(z,c), pEmissions(z,y));
pEmissionsIntensityCountry(c,y)$pDemandSupplyCountry(c,"Total production: GWh",y) = 1e-3*sum(zcmap(z,c), sum(s, pProbaScenarios(s)*vZonalEmissions.l(z,y,s)))/pDemandSupplyCountry(c,"Total production: GWh",y);

*--- Emission marginal costs
pEmissionMarginalCosts(y) = -sum(s, pProbaScenarios(s)*eTotalEmissions.M(y,s))/pRR(y)/pWeightYear(y);
*--- Solar Energy (zonal results)
set PVtech(tech) / PV, PVwSTO /;
Parameter zSolar(z,y), zSolarpH(z,y);
pSolarEnergy(z,q,d,t,y) = sum((gzmap(g,z),gtechmap(g,PVtech),gfmap(g,f),s), pProbaScenarios(s)*vPwrOut.l(g,f,q,d,t,y,s));

zSolar(z,y) = sum((q,d,t), pSolarEnergy(z,q,d,t,y));
pSolarValue(z,y)$(zSolar(z,y) > 0) = sum((q,d,t), pPrice(z,q,d,t,y)*pSolarEnergy(z,q,d,t,y))/zSolar(z,y);

zSolarpH(z,y) = sum((q,d,t), pSolarEnergy(z,q,d,t,y)*pHours(q,d,t));
pSolarCost(z,y)$(zSolarpH(z,y) > 0) = (sum((gzmap(ng,z),gtechmap(ng,PVtech)), pCRF(ng)*vCap.l(ng,y)*pGenData(ng,"Capex")*pCapexTrajectories(ng,y))*1e6)/zSolarpH(z,y);

*--- Dispatch Results
pPlantDispatchScenarios(z,y,q,d,g,t,s)$zgmap(z,g) = sum(gfmap(g,f), vPwrOut.l(g,f,q,d,t,y,s));
pPlantDispatch(z,y,q,d,g,t)$zgmap(z,g) = sum((gfmap(g,f),s), pProbaScenarios(s)*vPwrOut.l(g,f,q,d,t,y,s));


pDispatch(z,y,q,d,"Generation"    ,t) = sum(zgmap(z,g), pPlantDispatch(z,y,q,d,g,t));
pDispatch(z,y,q,d,"Imports"       ,t) =      sum((sTopology(z,z2),s),pProbaScenarios(s)*(vFlow.l(z2,z,q,d,t,y,s)) + vImportPrice.l(z,q,d,t,y,s));
pDispatch(z,y,q,d,"Exports"       ,t) = 0 - (sum((sTopology(z,z2), s), pProbaScenarios(s)*vFlow.l(z,z2,q,d,t,y,s)) + sum(s, pProbaScenarios(s)*vExportPrice.l(z,q,d,t,y,s)));
pDispatch(z,y,q,d,"Net Exchange"  ,t) =  pDispatch(z,y,q,d,"Imports"       ,t) + pDispatch(z,y,q,d,"Exports"       ,t);
pDispatch(z,y,q,d,"Unmet demand"  ,t) = sum(s, pProbaScenarios(s)*vUSE.l(z,q,d,t,y,s));
pDispatch(z,y,q,d,"Storage Charge",t) = 0 - sum((zgmap(z,st),s), pProbaScenarios(s)*vStorInj.l(st,q,d,t,y,s));
pDispatch(z,y,q,d,"Demand"        ,t) = pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y);

pDispatchScenarios(z,y,q,d,"Generation"    ,t,s) = sum(zgmap(z,g), pPlantDispatchScenarios(z,y,q,d,g,t,s));
pDispatchScenarios(z,y,q,d,"Imports"       ,t,s) =      sum(sTopology(z,z2),vFlow.l(z2,z,q,d,t,y,s) + vImportPrice.l(z,q,d,t,y,s));
pDispatchScenarios(z,y,q,d,"Exports"       ,t,s) = 0 - (sum(sTopology(z,z2), vFlow.l(z,z2,q,d,t,y,s)) + vExportPrice.l(z,q,d,t,y,s));
pDispatchScenarios(z,y,q,d,"Unmet demand"  ,t,s) = vUSE.l(z,q,d,t,y,s);
pDispatchScenarios(z,y,q,d,"Storage Charge",t,s) = 0 - sum(zgmap(z,st), vStorInj.l(st,q,d,t,y,s));



*--- LCOE and Average Cost Results

pDenom(z,g,y) = 1;
pDenom2(z,y)  = 1;
pDenom(z,g,y)$pEnergyByPlant(z,g,y) = pEnergyByPlant(z,g,y)*1e3;
pDenom2(z,y)$pDemandSupply(z,"Total production: GWh",y) = pDemandSupply(z,"Total production: GWh",y)*1e3;
pDenom3(c,y)  = sum(zcmap(z,c), pDenom2(z,y));


pSystemAverageCost(y)$sum(z, pDenom2(z,y)) =
                        (sum((z,q,d,t,s), pProbaScenarios(s)*(vImportPrice.l(z,q,d,t,y,s)-vExportPrice.l(z,q,d,t,y,s))*pTradePrice(z,q,d,y,t)*pHours(q,d,t))
*                 max(1,Sum((z,q,d,t),vImportPrice.l(z,q,d,t,y)*pHours(q,d,t))));
                       + sum(ndc, pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex"))*1e6
                       + sum(ndc$(not cs(ndc)), pCRFsst(ndc)*vCapStor.l(ndc,y)*pStorData(ndc,"Capex"))*1e3
                       + sum(ndc$(not st(ndc)), pCRFcst(ndc)*vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","Capex"))*1e3
                       + sum(ndc, pCRFcth(ndc)*vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal Field","Capex"))*1e6
                       + sum(dc, vAnnCapex.l(dc,y))
                       + sum(z, pFOM(z,y) + pVOM(z,y) + pFuel(z,y) + pSpinResCosts(z,y))
                       )/sum(z, sum((q,d,t,s), pProbaScenarios(s)*(vImportPrice.l(z,q,d,t,y,s) - vExportPrice.l(z,q,d,t,y,s))*pHours(q,d,t)) + pDenom2(z,y));


pPlantAnnualLCOE(zgmap(z,ndc),y)$pDenom(z,ndc,y) = (pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex")*1e6
                                                  + vCap.l(ndc,y)*pGenData(ndc,"FOMperMW")
                                                  + vCapStor.l(ndc,y)*pStorData(ndc,"FixedOM")
                                                  + vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","FixedOM")
                                                  + vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal field","FixedOM")
                                                  + sum((gfmap(ndc,f),q,d,t,s), pVarCost(ndc,f,y)*pProbaScenarios(s)*vPwrOut.l(ndc,f,q,d,t,y,s)*pHours(q,d,t))
*                                                 + sum((q,d,t)$(gfmap(g,f)),  vReserve(g,q,d,t,y)*pGenData(g,"ReserveCost")*pHours(q,d,t))
                                                   )/pDenom(z,ndc,y);

pPlantAnnualLCOE(zgmap(z,dc),y)$pDenom(z,dc,y) =  (vAnnCapex.l(dc,y)
                                                 + vCap.l(dc,y)*pGenData(dc,"FOMperMW")
                                                 + vCapStor.l(dc,y)*pStorData(dc,"FixedOM")
                                                 + vCapStor.l(dc,y)*pCSPData(dc,"Storage","FixedOM")
                                                 + vCapTherm.l(dc,y)*pCSPData(dc,"Thermal field","FixedOM")
                                                 + sum((gfmap(dc,f),q,d,t,s), pVarCost(dc,f,y)*pProbaScenarios(s)*vPwrOut.l(dc,f,q,d,t,y,s)*pHours(q,d,t))
*                                                + sum((q,d,t)$(gfmap(g,f)),  vReserve(g,q,d,t,y)*pGenData(g,"ReserveCost")*pHours(q,d,t))
                                                  )/pDenom(z,dc,y);


parameter zctmp(z,y);
zctmp(z,y) = sum((q,d,t,s), pProbaScenarios(s)*(vImportPrice.l(z,q,d,t,y,s) - vExportPrice.l(z,q,d,t,y,s))*pTradePrice(z,q,d,y,t)*pHours(q,d,t))
           + sum((sTopology(Zd,z),q,d,t,s), pPrice(Zd,q,d,t,y)*pProbaScenarios(s)*vFlow.l(Zd,z,q,d,t,y,s)*pHours(q,d,t))
           + sum(zgmap(z,dc), vAnnCapex.l(dc,y))
           + sum(zgmap(z,ndc), pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex"))*1e6
           + sum(zgmap(z,ndc)$(not cs(ndc)), pCRFsst(ndc)*vCapStor.l(ndc,y)*pStorData(ndc,"Capex"))*1e3
           + sum(zgmap(z,ndc)$(not st(ndc)), pCRFcst(ndc)*vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","Capex"))*1e3
           + sum(zgmap(z,ndc), pCRFcth(ndc)*vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal Field","Capex"))*1e6
           + pFOM(z,y) + pVOM(z,y)+ pFuel(z,y) + pSpinResCosts(z,y);

pZonalAverageCost(z,y)$pDenom2(z,y) = zctmp(z,y)/(sum((q,d,t,s), pProbaScenarios(s)*(vImportPrice.l(z,q,d,t,y,s) - vExportPrice.l(z,q,d,t,y,s))*pHours(q,d,t))
                                                + sum((sTopology(Zd,z),q,d,t,s), pProbaScenarios(s)*vFlow.l(Zd,z,q,d,t,y,s)*pHours(q,d,t)) + pDenom2(z,y));

pCountryAverageCost(c,y)$pDenom3(c,y) = sum(zcmap(z,c), zctmp(z,y))/(sum((zcmap(z,c),q,d,t,s), pProbaScenarios(s)*(vImportPrice.l(z,q,d,t,y,s) - vExportPrice.l(z,q,d,t,y,s))*pHours(q,d,t))
                                                                   + pDenom3(c,y)
                                                                   + sum((zcmap(z,c),sMapNCZ(Zd,z),q,d,t,s), pProbaScenarios(s)*vFlow.l(Zd,z,q,d,t,y,s)*pHours(q,d,t)));

Parameter NetTradeValue(c,y);

NetTradeValue(c,y) = sum(zcmap(z,c),  sum((sTopology(z,Zd),q,d,t,s),  pPrice(z,q,d,t,y)*pProbaScenarios(s)*vFlow.l(z,Zd,q,d,t,y,s)*pHours(q,d,t)) - sum((sTopology(Zd,z),q,d,t,s), pPrice(Zd,q,d,t,y)*pProbaScenarios(s)*vFlow.l(Zd,z,q,d,t,y,s)*pHours(q,d,t)));

set 
netimporters(c,y)
netexporters(c,y);

netimporters(c,y)$(NetTradeValue(c,y) < 0) = YES;
netexporters(c,y)$(NetTradeValue(c,y) >= 0) = YES;

Parameter zctmp2(z,y) 
pCountryAverageCost2(c,y);

zctmp2(z,y) = sum(zcmap(z,c),  sum((q,d,t,s), pProbaScenarios(s)*(vImportPrice.l(z,q,d,t,y,s) - vExportPrice.l(z,q,d,t,y,s))*pTradePrice(z,q,d,y,t)*pHours(q,d,t))
           + NetTradeValue(c,y)$(netimporters(c,y))
           + sum(zgmap(z,dc), vAnnCapex.l(dc,y))
           + sum(zgmap(z,ndc), pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex"))*1e6
           + sum(zgmap(z,ndc)$(not cs(ndc)), pCRFsst(ndc)*vCapStor.l(ndc,y)*pStorData(ndc,"Capex"))*1e3
           + sum(zgmap(z,ndc)$(not st(ndc)), pCRFcst(ndc)*vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","Capex"))*1e3
           + sum(zgmap(z,ndc), pCRFcth(ndc)*vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal Field","Capex"))*1e6
           + pFOM(z,y) + pVOM(z,y)+ pFuel(z,y) + pSpinResCosts(z,y) );

pCountryAverageCost2(c,y)$pDenom3(c,y) = sum(zcmap(z,c), zctmp(z,y))/(sum((zcmap(z,c),q,d,t,s), (vImportPrice.l(z,q,d,t,y,s) - vExportPrice.l(z,q,d,t,y,s))*pProbaScenarios(s)*pHours(q,d,t))
                                                                   + pDenom3(c,y)
                                                                   + (sum((zcmap(z,c),sMapNCZ(Zd,z),q,d,t,s), vFlow.l(Zd,z,q,d,t,y,s)*pProbaScenarios(s)*pHours(q,d,t)) - sum((zcmap(z,c),sMapNCZ(z,Zd),q,d,t,s), pProbaScenarios(s)*vFlow.l(z,Zd,q,d,t,y,s)*pHours(q,d,t)))$(netimporters(c,y)));




option clear=zctmp;
zctmp(z,y) = sum(zgmap(z,dc), vAnnCapex.l(dc,y))
           + sum(zgmap(z,ndc), pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex"))*1e6
           + sum(zgmap(z,ndc)$(not cs(ndc)), pCRFsst(ndc)*vCapStor.l(ndc,y)*pStorData(ndc,"Capex"))*1e3
           + sum(zgmap(z,ndc)$(not st(ndc)), pCRFcst(ndc)*vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","Capex"))*1e3
           + sum(zgmap(z,ndc), pCRFcth(ndc)*vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal Field","Capex"))*1e6
           + pFOM(z,y) + pVOM(z,y) + pFuel(z,y) + pSpinResCosts(z,y);
pZonalAverageGenCost(z,y)$pDenom2(z,y) = zctmp(z,y)/pDenom2(z,y);
pCountryAverageGenCost(c,y)$pDenom3(c,y) = sum(zcmap(z,c),zctmp(z,y))/pDenom3(c,y);


*---  CSP and storage
pCSPBalance(y, g,q,d,"Thermal output",t) = sum(s, pProbaScenarios(s)*vThermalOut.l(g,q,d,t,y,s));
pCSPBalance(y,cs,q,d,"Storage Input" ,t) = sum(s, pProbaScenarios(s)*vStorInj.l(cs,q,d,t,y,s));
pCSPBalance(y,cs,q,d,"Storage Output",t) = sum(s, pProbaScenarios(s)*vStorOut.l(cs,q,d,t,y,s));
pCSPBalance(y,cs,q,d,"Power Output"  ,t) = sum((gfmap(cs,f),s), pProbaScenarios(s)*vPwrOut.l(cs,f,q,d,t,y,s));

pCSPComponents(g, "Thermal Field: WM" ,y) = vCapTherm.l(g,y);
pCSPComponents(cs,"Storage: MWh"      ,y) = vCapStor.l(cs,y);
pCSPComponents(cs,"Power Block: MW"   ,y) = vCap.l(cs,y);
pCSPComponents(g, "Solar Multiple"    ,y) = vCapTherm.l(g,y)/max(vCap.l(g,y),1);
pCSPComponents(cs,"Storage Hours: hrs",y) = vCapStor.l(cs,y)/max(vCap.l(cs,y),1);

pPVwSTOBalance(y,q,d,so, "PV output"     ,t) = sum((gfmap(so,f),s), pProbaScenarios(s)*vPwrOut.l(so,f,q,d,t,y,s));
pPVwSTOBalance(y,q,d,stp,"Storage Input" ,t) = sum(s, pProbaScenarios(s)*vStorInj.l(stp,q,d,t,y,s));
pPVwSTOBalance(y,q,d,stp,"Storage output",t) = sum((gfmap(stp,f),s), pProbaScenarios(s)*vPwrOut.l(stp,f,q,d,t,y,s));
pPVwSTOBalance(y,q,d,stp,"Storage Losses",t) = pStorData(stp,"efficiency")*sum(s, pProbaScenarios(s)*vStorInj.l(stp,q,d,t,y,s));
pPVwSTOBalance(y,q,d,stp,"Storage level" ,t) = sum(s, pProbaScenarios(s)*vStorage.l(stp,q,d,t,y,s));

pPVwSTOComponents(so, "PV Plants"           ,y) = vCap.l(so,y);
pPVwSTOComponents(stp,"Storage Capacity MW" ,y) = vCap.l(stp,y);
pPVwSTOComponents(stp,"Storage Capacity MWh",y) = vCapStor.l(stp,y);
pPVwSTOComponents(stp,"Storage Hours"       ,y) = vCapStor.l(stp,y)/max(vCap.l(stp,y),1);

pStorageBalance(y,stg,q,d,"Storage Input"     ,t) = sum(s, pProbaScenarios(s)*vStorInj.l(stg,q,d,t,y,s));
pStorageBalance(y,stg,q,d,"Storage Output"    ,t) = sum((gfmap(stg,f),s), pProbaScenarios(s)*vPwrOut.l(stg,f,q,d,t,y,s));
pStorageBalance(y,stg,q,d,"Storage Losses"    ,t) = pStorData(stg,"efficiency")*sum(s, pProbaScenarios(s)*vStorInj.l(stg,q,d,t,y,s));
pStorageBalance(y,stg,q,d,"Net Storage Change",t) = sum(s, pProbaScenarios(s)*vStorNet.l(stg,q,d,t,y,s));
pStorageBalance(y,stg,q,d,"Storage Level"     ,t) = sum(s, pProbaScenarios(s)*vStorage.l(stg,q,d,t,y,s));

pStorageComponents(stg,"Storage Capacity in MW",y) = vCap.l(stg,y);
pStorageComponents(stg,"Storage Capacity MWh"  ,y) = vCapStor.l(stg,y);
pStorageComponents(stg,"Storage Hours"         ,y) = vCapStor.l(stg,y)/max(vCap.l(stg,y),1);

*pLossesStorage(z,y) = (1-pStorage(z,y,"Efficiency"))*Sum((q,d,t),vStorInj.l(z,q,d,t,y)*pHours(q,d,t));

*--- Solver parameters
pSolverParameters("Solver Status")               = PA.modelstat + EPS;
pSolverParameters("Solver Time: ms")             = PA.etSolve + EPS;
pSolverParameters("Absolute gap")                = PA.objVal-PA.objEst + EPS;
pSolverParameters("Relative gap")$(PA.objVal >0) = pSolverParameters("Absolute gap")/PA.objVal + EPS;

*--- Seasonal Results

* A. Seasonal Demand-Supply Balance Results
pDemandSupplySeason(z,"Demand: GWh"            ,y,q) = sum((d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))/1e3;
pDemandSupplySeason(z,"Total production: GWh"  ,y,q) = sum((zgmap(z,g),gfmap(g,f),d,t,s), pProbaScenarios(s)*vPwrOut.l(g,f,q,d,t,y,s)*pHours(q,d,t))/1e3;
pDemandSupplySeason(z,"Unmet demand: GWh"      ,y,q) = sum((d,t,s), pProbaScenarios(s)*vUSE.l(z,q,d,t,y,s)*pHours(q,d,t))/1e3;
pDemandSupplySeason(z,"Surplus generation: GWh",y,q) = sum((d,t,s), pProbaScenarios(s)*vSurplus.l(z,q,d,t,y,s)*pHours(q,d,t))/1e3;

pDemandSupplySeason(z,"Imports exchange: GWh",y,q) = (sum((sTopology(z,z2),d,t,s), pProbaScenarios(s)*vFlow.l(z2,z,q,d,t,y,s)*pHours(q,d,t)) + sum((d,t,s), pProbaScenarios(s)*vImportPrice.l(z,q,d,t,y,s)*pHours(q,d,t)))/1e3;
pDemandSupplySeason(z,"Exports exchange: GWh",y,q) = (sum((sTopology(z,z2),d,t,s), pProbaScenarios(s)*vFlow.l(z,z2,q,d,t,y,s)*pHours(q,d,t)) + sum((d,t,s), pProbaScenarios(s)*vExportPrice.l(z,q,d,t,y,s)*pHours(q,d,t)))/1e3;

* B. Seasonal Energy by Plant
pEnergyByPlantSeason(zgmap(z,g),y,q) = sum((gfmap(g,f),d,t,s),pProbaScenarios(s)*vPwrOut.l(g,f,q,d,t,y,s)*pHours(q,d,t))/1e3;


* C.Interchange

* By Zone
pInterchangeSeason(sTopology(z,z2),y,q) = sum((d,t,s), pProbaScenarios(s)*vFlow.l(z,z2,q,d,t,y,s)*pHours(q,d,t))/1e3;
pSeasonTrade(z,"External Zone Imports",y,q) = sum((d,t,s), pProbaScenarios(s)*vImportPrice.l(z,q,d,t,y,s)*pHours(q,d,t))/1e3;
pSeasonTrade(z,"External Zone Exports",y,q) = sum((d,t,s), pProbaScenarios(s)*vExportPrice.l(z,q,d,t,y,s)*pHours(q,d,t))/1e3;

* By Country
pInterchangeSeasonCountry(c,c2,y,q) = sum((zcmap(z,c),zcmap2(z2,c2),sMapNCZ(z,z2)), pInterchangeSeason(z,z2,y,q));
pSeasonTradeCountry(c,"External Zone Imports",y,q) = sum(zcmap(z,c), pSeasonTrade(z,"External Zone Imports",y,q));
pSeasonTradeCountry(c,"External Zone Exports",y,q) = sum(zcmap(z,c), pSeasonTrade(z,"External Zone Exports",y,q));



*******************************************Parameters related to Turkey model*****************************************************
pCurtailedVRE(q,d,t,y,VRE)                    =sum((gzmap(VRE,z),s),pProbaScenarios(s)*vCurtailedVRE.l(z,VRE,q,d,t,y,s)*pHours(q,d,t));

pCurtailedVRET(q,d,t,y)                       =sum((gzmap(VRE,z),s),pProbaScenarios(s)*vCurtailedVRE.l(z,VRE,q,d,t,y,s)*pHours(q,d,t));

*pCurtailedVREM(y,q)                           =sum(gzmap(VRE,z),sum((d,t),vCurtailedVRE.l(z,VRE,q,d,t,y)*pHours(q,d,t)));

pCurtailedVREperY(VRE,y)                      =sum(gzmap(VRE,z),sum((q,d,t,s), pProbaScenarios(s)*vCurtailedVRE.l(z,VRE,q,d,t,y,s)*pHours(q,d,t)))/1000 ;

pCurtailedVRETperY(y)                         =sum(gzmap(VRE,z),sum((q,d,t,s), pProbaScenarios(s)*vCurtailedVRE.l(z,VRE,q,d,t,y,s)*pHours(q,d,t)))/1000;

pVREdispatchT(q,d,t,y)                        =sum((VRE,f)$gfmap(VRE,f),sum(s, pProbaScenarios(s)*vPwrOut.l(VRE,f,q,d,t,y,s)*pHours(q,d,t)))/1000;

pVREpenetration(q,d,t,y)                      =sum((VRE,f)$gfmap(VRE,f),sum(s, pProbaScenarios(s)*vPwrOut.l(VRE,f,q,d,t,y,s)))/(sum(z,pDemandData(z,q,d,y,t))+sum((stg,s),vStorInj.l(stg,q,d,t,y,s)));

*pREpenetration2(y)                            =sum((RE,f)$gfmap(RE,f),sum((q,d,t) , vPwrOut.l(RE,f,q,d,t,y)))/(sum((nST,f)$gfmap(nSt,f),sum((q,d,t) , vPwrOut.l(nSt,f,q,d,t,y))));

pDemandTotalE(q,d,t,y)                        =sum(z,pDemandData(z,q,d,y,t)*pHours(q,d,t));

pDemandTotalP(q,d,t,y)                        =sum(z,pDemandData(z,q,d,y,t));


pLCOEperFuelAnnual(c,f,y)$(sum(z$zcmap(z,c),sum(g$gfmap(g,f),pDenom(z,g,y))))                     =sum(z$zcmap(z,c),sum(g$gfmap(g,f),pPlantAnnualLCOE(z,g,y)*pDenom(z,g,y)))/ sum(z$zcmap(z,c),sum(g$gfmap(g,f),pDenom(z,g,y)));

pAnnCapexperFuel(c,f,y)                       =sum(ndc$gfmap(ndc,f),(pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex")))+sum(dc$gfmap(dc,f),(vAnnCapex.l(dc,y)/1000000));




*Calculate the variable cost of generators that operate
pVarGenCost(z,cg,y,q,d,f,t)$(gzmap(cg,z) and  sum(s, pProbaScenarios(s)*vPwrOut.l(cg,f,q,d,t,y,s))*pHours(q,d,t)>1)          =sum(gfmap(cg,f),  pVarCost(cg,f,y));

pVarGenCost(z,cg,y,q,d,f,t)$(gzmap(cg,z) and  sum(s, pProbaScenarios(s)*vPwrOut.l(cg,f,q,d,t,y,s))*pHours(q,d,t)<1)         =0;


*Remove the effect of divizion by zero
pVarGenCost(z,cg,y,q,d,f,t)$(pVarGenCost(z,cg,y,q,d,f,t)=Undf) =0;



*Calculate the vabiable cost of most expensive generator
pMargPrice(z,y,q,d,t)                           =smax(gfmap(cg,f),pVarGenCost(z,cg,y,q,d,f,t));

*Estimate the markup of generators that operate
pGenMarkUp(z,cg,y,q,d,f,t)$(gzmap(cg,z) and (sum(s, pProbaScenarios(s)*vPwrOut.l(cg,f,q,d,t,y,s))*pHours(q,d,t)>1))            =sum(gfmap(cg,f), (pMargPrice(z,y,q,d,t)-pVarCost(cg,f,y)));
*Remobe the effect of divizion by zero
pGenMarkUp(z,cg,y,q,d,f,t)$(pGenMarkUp(z,cg,y,q,d,f,t)=Undf) =0;

pGenMarkUp(z,cg,y,q,d,f,t)$(gzmap(cg,z) and (sum(s, pProbaScenarios(s)*vPwrOut.l(cg,f,q,d,t,y,s))*pHours(q,d,t)<1))        =0;

*Calculate the Revenue of coal and gas Generators
pGenRevenue(z,cg,f,y)$(gzmap(cg,z) and gfmap(cg,f))= sum((gfmap(cg,f),q,d,t,s),pProbaScenarios(s)*vPwrOut.l(cg,f,q,d,t,y,s)*pHours(q,d,t)*pGenMarkUp(z,cg,y,q,d,f,t))/1000000;
*Calculate the Present value of Revenue of coal and gas generators
pPresentValue(z,cg,f,y)$(gzmap(cg,z) and gfmap(cg,f))= pRR(y)*sum((gfmap(cg,f),q,d,t,s),pProbaScenarios(s)*vPwrOut.l(cg,f,q,d,t,y,s)*pHours(q,d,t)*pGenMarkUp(z,cg,y,q,d,f,t))/1000000;
pPresentValueWeighted(z,cg,f,y)$(gzmap(cg,z) and gfmap(cg,f))= pPresentValue(z,cg,f,y)*pWeightYear(y);

pNormalRetireYear(z,cg,f)$(gzmap(cg,z) and gfmap(cg,f)) =  pGenData(cg,"RetrYr");
pNominalCapacity(z,cg,f)$(gzmap(cg,z) and gfmap(cg,f))  =  pGenData(cg,"Capacity");

pFuelDispatch(z,q,d,t,y,f)  =sum(gfmap(g,f)$(gzmap(g,z)),sum(s, pProbaScenarios(s)*vPwrOut.l(g,f,q,d,t,y,s)));

PARAMETER pDispatchByTypeAndFuel(z,y,q,d,*,t);

pDispatchByTypeAndFuel(z,y,q,d,"Generation",t) = pDispatch(z,y,q,d,"Generation"    ,t);
pDispatchByTypeAndFuel(z,y,q,d,"Imports",t) = pDispatch(z,y,q,d,"Imports"       ,t);
pDispatchByTypeAndFuel(z,y,q,d,"Exports",t) = pDispatch(z,y,q,d,"Exports"       ,t);
pDispatchByTypeAndFuel(z,y,q,d,"Net Exchange",t) = pDispatch(z,y,q,d,"Net Exchange"  ,t);
pDispatchByTypeAndFuel(z,y,q,d,"Unmet demand",t) = pDispatch(z,y,q,d,"Unmet demand"  ,t);
pDispatchByTypeAndFuel(z,y,q,d,"Storage Charge",t) = pDispatch(z,y,q,d,"Storage Charge",t);
pDispatchByTypeAndFuel(z,y,q,d,"Demand",t) = pDispatch(z,y,q,d,"Demand"        ,t);

loop(f,

pDispatchByTypeAndFuel(z,y,q,d,f,t) = pFuelDispatch(z,q,d,t,y,f);
   );


pUnmetP(q,d,t,y)            =sum((z,s),pProbaScenarios(s)*vUSE.l(z,q,d,t,y,s));
pSurplusP(q,d,t,y)          =sum((z,s),pProbaScenarios(s)*vSurplus.l(z,q,d,t,y,s));

pStorLevel(q,d,t,y)       =sum((st,s),pProbaScenarios(s)*vStorage.l(st,q,d,t,y,s));
pStorInj(q,d,t,y)         =sum((st,s),pProbaScenarios(s)*vStorInj.l(st,q,d,t,y,s));
pStorOut(q,d,t,y)         =sum(gfmap(st,f),sum(s,pProbaScenarios(s)*vPwrOut.l(st,f,q,d,t,y,s)));

pStorInjD(y,q,d,t)  = pStorInj(q,d,t,y);
pStorOutD(y,q,d,t)  = pStorOut(q,d,t,y);

********************************************************************************************************************
*********************Additions for H2 model***************************************************************
*pCurtailedVREnet(q,d,t,y)                       =sum(z,vNetCurtailedVRE.l(z,q,d,t,y)*pHours(q,d,t))$pIncludeH2;
pCapacityPlanH2(zH2map(z,hh),y)                 = vCapH2.l(hh,y)$pIncludeH2;


pPwrREH2(q,d,t,y)         =sum(z,vPwrREH2.l(z,q,d,t,y))$pIncludeH2;
pPwrREGrid(q,d,t,y)       =sum(z,vPwrREGrid.l(z,q,d,t,y))$pIncludeH2;
pH2PwrInPerGen(q,d,t,y,hh)  =vH2PwrIn.l(hh,q,d,t,y)$pIncludeH2;
pH2PwrInT(q,d,t,y)      =sum(hh,vH2PwrIn.l(hh,q,d,t,y))$pIncludeH2;


**************************************************************************************************************



*--- END RESULTS

pZonesperCountry(c) = sum(zcmap(z,c), 1);
MaxZonesperCountry = smax(c,pZonesperCountry(c));

display pNominalCapacity;
$ifthen.excelreport %DOEXCELREPORT%==1
execute_unload 'epmresults',     pScalars, pSummary, pSystemAverageCost, pZonalAverageCost,pCountryAverageCost
                                 pAveragePrice, pAveragePriceExp, pAveragePriceImp, pPrice, pAveragePriceHub,
                                 pAveragePriceCountry, pAveragePriceExpCountry, pAveragePriceImpCountry,pCO2backstopCosts,
                                 pCostSummary, pCostSummaryCountry, pCostSummaryWeighted, pCostSummaryWeightedCountry,
                                 pCostSummaryWeightedAverageCountry, pFuelCosts,pFuelCostsCountry,pFuelConsumption,pFuelConsumptionCountry
                                 pEnergyByPlant, pEnergyByFuel,pEnergyByFuelCountry, pEnergyMix, pDemandSupply,  pDemandSupplyCountry
                                 pInterchange, pInterconUtilization, pLossesTransmission, pInterchangeCountry,pLossesTransmissionCountry,
                                 pYearlyTrade,pHourlyTrade,pYearlyTradeCountry,pHourlyTradeCountry,pDEnergyCheck,
                                 pPeakCapacity, pCapacityByFuel, pNewCapacityFuel, pCapacityPlan,pAdditionalCapacity,pRetirements,
                                 pPeakCapacityCountry, pCapacityByFuelCountry, pNewCapacityFuelCountry,pCapacityPlanCountry,
                                 pReserveByPlant, pReserveCosts,pReserveByPlantCountry, pReserveCostsCountry,
                                 pplanning_reserve_constraints,pReserveMargin,psys_reserve_margin,vReserve,gfmap,
                                 pEmissions, pEmissionsIntensity,pEmissionsCountry, pEmissionsIntensityCountry,pEmissionMarginalCosts,
                                 pPlantDispatch, pPlantDispatchScenarios, pDispatch, pDispatchScenarios,pPlantUtilization, pPlantAnnualLCOE,
                                 pCSPBalance, pCSPComponents,pPVwSTOBalance,pPVwSTOComponents,pStorageBalance,pStorageComponents
                                 pSolarValue, pSolarCost,vFuel.l,vPwrOut.l,pMaxFuelLimit,pAveragePriceExp1,pAveragePriceImp1,
                                 pdiff,pDemandForecast, pTempDemand,pHours, pDemandProfile, vYearlyCurtailmentCost.l,
                                 pSolverParameters,pDemandSupplySeason,pEnergyByPlantSeason, pNPVbyCountry, pNPVCostSpecific, pCAPEXByFuel, pCAPEXByFuelByYear,pCAPEXByFuelByCountryByYear
                                 pInterchangeSeason,pSeasonTrade,pInterchangeSeasonCountry,pSeasonTradeCountry, pSpinningReserveByFuelByCountry, pRetirementsByFuel,pNPVByYear, pDispatchByTypeAndFuel, pInterchangeHourly,pVRECurtailment


***************************************************Rarameters related to Turkey model**********************************
                                pCurtailedVRET,pCurtailedVREperY,pCurtailedVRETperYZ,pCurtailedVRETperY,pVREdispatchT,pVREpenetration,pREpenetration2,pDemandTotalE, pDemandTotalP
                                pCurtailedVRE, pLCOEperFuelAnnual, pAnnCapexperFuel, pVarGenCost, pMargPrice, pGenMarkUp, pPresentValue, pPresentValueWeighted
                                pGenRevenue, pNormalRetireYear, pNominalCapacity, pFuelDispatch, pUnmetP, pSurplusP, pStorLevel, pStorInj, pStorOut , pStorInjD, pStorOutD, pGenData, pCountryAverageCost2, netimporters, netexporters
*                                pCapacityCreditM,pFindSysPeakM,pCurtailedVREM, pCostSummaryM,

************************************************************************************************************
******************************************Parameters related to H2 production******************************************************************
                                 pCapacityPlanH2, pAnnCapexH2, pAnnCapexH2T, pCapexH2,pCapexH2T, pPwrRE,  pPwrREH2,
                                pPwrREGrid, pH2PwrInPerGen,pH2PwrInT,pH2perY,pH2perG, pUnmetH2, pNetInterchange, pLineInterchange, pAnnualTransmissionCapacity, pREAnnualCF, pHoursLineCongested, pHoursLineCongestedPercent

*pCurtailedVREnet

************************************************************************************************************************************************
;

file fgdxxrw / 'gdxxrw.out' /;
file fxlsxrep / 'xlsxReport.cmd' /;
singleton set execPlatform(*) / '' /;
put_utility 'assignText' / 'execPlatform' / system.platform;
scalar isWindows; loop(execPlatform, isWindows=ord(execPlatform.te,1)=ord('W',1));
put$(not isWindows) fxlsxrep 'rem Run to create Excel files' / 'cd "%gams.workdir%"';
$setNames "%gams.input%" fp fn fe

if (MaxZonesperCountry>1,
   put_utility fgdxxrw 'ren' / 'WriteZonalandCountry.txt';

$  onPut
   par=pScalars                           rng=Settings_raw!A6                     rdim=1
   par=pSummary                           rng=Summary!A6                          rdim=1
   par=pSystemAverageCost                 rng=SysAverageCost!A5
   par=pZonalAverageCost  rDim=2          rng=ZonalAverageCost!A5
   par=pAveragePrice                      rng=ZonalAverageCost!U5
   par=pCO2backstopCosts rDim=2            rng=CO2backstop!A5
   par=pCountryAverageCost  rDim=2              rng=CountryAverageCost!A5
   par=pAveragePriceCountry               rng=CountryAverageCost!U5
   par=pPrice                             rng=MarginalCost!A5
   par=pAveragePriceExp                   rng=ExchangeCost!A5
   par=pAveragePriceImp                   rng=ExchangeCost!Z5
   par=pAveragePriceExpCountry            rng=ExchangeCostCountry!A5
   par=pAveragePriceImpCountry            rng=ExchangeCostCountry!Z5
   par=pAveragePriceHub                   rng=TradeHubPrice!A5
   par=pCostSummary    rDim=3                   rng=CostSummary!A5
   par=pCostSummaryCountry  rDim=3              rng=CostSummaryCountry!A5
   par=pCostSummaryWeighted  rDim=3             rng=CostWeighted!A5
   par=pCostSummaryWeightedCountry rDim=3       rng=CostWeigthedCountry!A5
   par=pCostSummaryWeightedAverageCountry rDim=3 rng=CostAverage!A5
   par=pFuelCosts     rDim=3                    rng=FuelCosts!A5
   par=pFuelCostsCountry  rDim=3                rng=FuelCostsCountry!A5
   par=pFuelConsumption rDim=3                  rng=FuelConsumption!A5
   par=pFuelConsumptionCountry  rDim=3          rng=FuelConsumptionCountry!A5
   par=pDemandSupply     rDim=3                 rng=DemandSupply!A5
   par=pDemandSupplyCountry    rDim=3           rng=DemandSupplyCountry!A5
   par=pEnergyByFuel          rDim=3            rng=EnergyByFuel!A5
   par=pEnergyByFuelCountry   rDim=3            rng=EnergyByFuelCountry!A5
   par=pEnergyMix          rDim=3               rng=EnergyMix!A5
   par=pEnergyByPlant    rDim=3                 rng=EnergyByPlant!A5
   par=pInterchange                       rng=Interchange!A5
   par=pInterchangeCountry                rng=InterchangeCountry!A5
   par=pInterconUtilization               rng=InterconUtilization!A5
   par=pLossesTransmission                rng=LossesTransmission!A5
   par=pLossesTransmissionCountry         rng=LossesTransmissionCountry!A5
   par=pAdditionalCapacity                rng=AddedTransCap!A5
   par=pYearlyTrade                       rng=PriceTrade!A5
   par=pHourlyTrade                       rng=HourlyTrade!A5
   par=pYearlyTradeCountry                rng=PriceTradeCountry!A5
   par=pHourlyTradeCountry                rng=HourlyTradeCountry!A5
   par=pPeakCapacity       rDim=3               rng=PeakAndCapacity!A5
   par=pPeakCapacityCountry  rDim=3             rng=PeakAndCapacityCountry!A5
   par=pCapacityByFuel   rDim=3                 rng=CapacityByFuel!A5
   par=pCapacityByFuelCountry  rDim=3           rng=CapacityByFuelCountry!A5
   par=pNewCapacityFuel    rDim=3               rng=NewCapacityFuel!A5
   par=pNewCapacityFuelCountry   rDim=3         rng=NewCapacityFuelCountry!A5
   par=pRetirements        rDim=3               rng=Retirements!A5
   par=pCapacityPlan       rDim=3               rng=CapacityPlan!A5
   par=pCapacityPlanCountry   rDim=3            rng=CapacityPlanCountry!A5
   par=pReserveCosts                      rng=ReserveCosts!A5
   par=pReserveCostsCountry               rng=ReserveCostsCountry!A5
   par=pReserveByPlant                    rng=ReserveByPlant!A5
   par=pReserveByPlantCountry             rng=ReserveByPlantCountry!A5
   par=pEmissions       rDim=2                  rng=Emissions!A5
   par=pEmissionsCountry   rDim=2               rng=EmissionsCountry!A5
   par=pEmissionsIntensity    rDim=2            rng=EmissionsIntensity!A5
   par=pEmissionsIntensityCountry  rDim=2       rng=EmissionsIntensityCountry!A5
   par=pPlantDispatch      rDim=6               rng=PlantDispatch!A5
   par=pDispatch        rDim=6                  rng=Dispatch!A5
   par=pPlantUtilization  rDim=3                rng=PlantUtilization!A5
   par=pPlantAnnualLCOE   rDim=3                rng=PlantAnnualLCOE!A5
   par=pCSPBalance                        rng=CSPBalance!A5
   par=pCSPComponents                     rng=CSPComponents!A5
   par=pPVwSTOBalance                     rng=PVSTOBalance!A5
   par=pPVwSTOComponents                  rng=PVSTOComponents!A5
   par=pStorageBalance                    rng=StorageBalance!A5
   par=pStorageComponents                 rng=StorageComponents!A5
   par=pSolarValue                        rng=SolarValue!A5
   par=pSolarCost                         rng=SolarCost!A5
   par=pSolverParameters                  rng=SolverParameters!A5                 rdim=1
   
******************************************************************************************
*********************H2 model related parameters******************************************
*   par=pCurtailedVREnet                      rng=CurtVRET!AA5
    par=pH2perY                               rng=H2perYear!A5
   par=pH2perG    rDim=4 cDim=1              rng=H2perGenerator!a5
   par=pCapacityPlanH2                       rng=CapacityPlanH2!A5
   par=pAnnCapexH2                           rng=H2AnnCapexPerGen!A5
   par=pCapexH2                              rng=H2CapexPerGen!A5
   par=pCapexH2T            cDim=1           rng=H2CapexT!A5
   par=pPwrREH2      rDim=3 cDim=1           rng=H2REGenT!A5
   par=pPwrREGrid      rDim=3 cDim=1         rng=REGridGenT!A5
   par=pH2PwrInPerGen  rDim=4 cDim=1         rng=Power2H2!A5
   par=pUnmetH2         rDim=2 cDim=1        rng=UnmetH2!A5

*******************************************************************************************








$  offPut
   putclose;
   if (isWindows,
      execute.checkErrorLevel 'gdxxrw epmresults.gdx output="%XLS_OUTPUT%" @WriteZonalandCountry.txt';
   else
      put fxlsxrep / 'gdxxrw "%fn%\epmresults.gdx" output="%XLS_OUTPUT%" @"%fn%\WriteZonalandCountry.txt"';
   );
else
   put_utility fgdxxrw 'ren' / 'WriteZonal.txt';
$  onPut
   par=pScalars                           rng=Settings_raw!A6                     rdim=1
   par=pSummary                           rng=Summary!A6                          rdim=1
   par=pSystemAverageCost                 rng=SysAverageCost!A5
   par=pZonalAverageCost                  rng=ZonalAverageCost!A5
   par=pAveragePrice                      rng=ZonalAverageCost!V5
   par=pPrice                             rng=MarginalCost!A5
   par=pAveragePriceExp                   rng=ExchangeCost!A5
   par=pAveragePriceImp                   rng=ExchangeCost!U5
   par=pAveragePriceHub                   rng=TradeHubPrice!A5
   par=pCostSummary                       rng=CostSummary!A5
   par=pCostSummaryWeighted               rng=CostWeighted!A5
   par=pCostSummaryWeightedAverageCountry rng=CostAverage!A5
   par=pFuelCosts                         rng=FuelCosts!A5
   par=pFuelConsumption                   rng=FuelConsumption!A5
   par=pDemandSupply                      rng=DemandSupply!A5
   par=pEnergyByFuel                      rng=EnergyByFuel!A5
   par=pEnergyMix                         rng=EnergyMix!A5
   par=pEnergyByPlant                     rng=EnergyByPlant!A5
   par=pInterchange                       rng=Interchange!A5
   par=pInterconUtilization               rng=InterconUtilization!A5
   par=pLossesTransmission                rng=LossesTransmission!A5
   par=pAdditionalCapacity                rng=AddedTransCap!A5
   par=pYearlyTrade                       rng=PriceTrade!A5
   par=pHourlyTrade                       rng=HourlyTrade!A5
   par=pPeakCapacity                      rng=PeakAndCapacity!A5
   par=pCapacityByFuel                    rng=CapacityByFuel!A5
   par=pNewCapacityFuel                   rng=NewCapacityFuel!A5
   par=pRetirements                       rng=Retirements!A5
   par=pCapacityPlan                      rng=CapacityPlan!A5
   par=pReserveCosts                      rng=ReserveCosts!A5
   par=pReserveByPlant                    rng=ReserveByPlant!A5
   par=pEmissions                         rng=Emissions!A5
   par=pEmissionsIntensity                rng=EmissionsIntensity!A5
   par=pEmissionMarginalCosts             rng=EmissionMarginalCosts!A5
   par=pPlantDispatch                     rng=PlantDispatch!A5
   par=pDispatch                          rng=Dispatch!A5
   par=pPlantUtilization                  rng=PlantUtilization!A5
   par=pPlantAnnualLCOE                   rng=PlantAnnualLCOE!A5
   par=pCSPBalance                        rng=CSPBalance!A5
   par=pCSPComponents                     rng=CSPComponents!A5
   par=pPVwSTOBalance                     rng=PVSTOBalance!A5
   par=pPVwSTOComponents                  rng=PVSTOComponents!A5
   par=pStorageBalance                    rng=StorageBalance!A5
   par=pStorageComponents                 rng=StorageComponents!A5
   par=pSolarValue                        rng=SolarValue!A5
   par=pSolarCost                         rng=SolarCost!A5
   par=pSolverParameters                  rng=SolverParameters!A5                 rdim=1

*********************************Turkey related parameters ************************************************

   par=pCurtailedVRE        rDim=3 cDim=2    rng=CurtVREG!A5
   par=pCurtailedVRET                        rng=CurtVRET!A5
   par=pCurtailedVREM                        rng=CurtVREM!A5
   par=pCurtailedVREperY                     rng=CurtVREGT!A5
   par=pCurtailedVRETperY                    rng=CurtVREY!A5
   par=pVREdispatchT                         rng=VREDispT!A5
   par=pVREpenetration                       rng=VREpenetrT!A5
   par=pREpenetration2                       rng=REpenetrTY!A5
   par=pDemandTotalE                         rng=EnergyDemand!A5
   par=pDemandTotalP                         rng=PowerDemand!A5

   par=pLCOEperFuelAnnual                    rng=LCOEperFuelA!A5
   par=pVarGenCost                           rng=GenVarCost!A5
   par=pMargPrice                            rng=ZonMargCost!A5
   par=pGenMarkUp                            rng=GenMarkUp!A5
   par=pGenRevenue                           rng=GenRevenue!A5
   par=pPresentValue                         rng=NPVGenMU!A5
   par=pPresentValueWeighted                 rng=NPVGenMUW!A5
   par=pNormalRetireYear   rdim=3            rng=NPVGenMU!Z6
   par=pNominalCapacity    rdim=3            rng=GenRevenue!AC6
   par=pFuelDispatch       rdim=4 cDim=2     rng=FuelDispath!A6
   par=pUnmetP                               rng=UnmetDemandP!a6
   par=pSurplusP                             rng=SurplusPower!A6
   par=pStorLevel          rDim=3 cDim=1     rng=StorageOper!a6
   par=pStorInj            rDim=3 cDim=1     rng=StorageOper!l6
   par=pStorOut            rDim=3 cDim=1     rng=StorageOper!s6
   par=pStorInjD            rDim=3 cDim=1    rng=StorageInjD!a1
   par=pStorOutD            rDim=3 cDim=1    rng=StorageOutD!a1
   par=pCapacityCreditM    rDim=1 cDim=2     rng=CapacityCredit!a6
   par=pFindSysPeakM                         rng=MonthlyPeak!a1
   par=pCostSummaryM                         rng=CostSummaryM!A5
******************************************************************************************
*********************H2 model related parameters******************************************
*   par=pCurtailedVREnet                      rng=CurtVRET!AA5

   par=pH2perY                               rng=H2perYear!A5
   par=pH2perG    rDim=4 cDim=1              rng=H2perGenerator!a5
   par=pCapacityPlanH2                       rng=CapacityPlanH2!A5
   par=pAnnCapexH2                           rng=H2AnnCapexPerGen!A5
   par=pCapexH2                              rng=H2CapexPerGen!A5
   par=pCapexH2T            cDim=1           rng=H2CapexT!A5
   par=pPwrREH2      rDim=3 cDim=1           rng=H2REGenT!A5
   par=pPwrREGrid      rDim=3 cDim=1         rng=REGridGenT!A5
   par=pH2PwrInPerGen  rDim=4 cDim=1         rng=Power2H2!A5
   par=pUnmetH2         rDim=2 cDim=1        rng=UnmetH2!A5

*******************************************************************************************




$  offPut
   putclose;
   if (isWindows,
      execute.checkErrorLevel 'gdxxrw epmresults.gdx output="%XLS_OUTPUT%" @WriteZonal.txt';
   else
      put fxlsxrep / 'gdxxrw "%fn%\epmresults.gdx" output="%XLS_OUTPUT%" @"%fn%\WriteZonal.txt"';
   );
);

if (pSeasonalReporting > 0,
   put_utility fgdxxrw 'ren' / 'WriteSeasonal.txt';
$  onPut
   par=pDemandSupplySeason          rng=DemandSupplySeason!A5
   par=pEnergyByPlantSeason         rng=EbyPlantSeason!A5
   par=pInterchangeSeason           rng=InterchangeSeason!A5
   par=pInterchangeSeasonCountry    rng=InterchangeSeasonCountry!A5
   par=pSeasonTrade                 rng=TradeSeason!A5
   par=pSeasonTradeCountry          rng=TradeSeasonCountry!A5
$  offPut
   putclose;
   if (isWindows,
      execute.checkErrorLevel 'gdxxrw epmresults.gdx output=EPMRESULTSSEASON.xlsx @WriteSeasonal.txt';
   else
      put fxlsxrep / 'gdxxrw "%fn%epmresults.gdx" output=EPMRESULTSSEASON.xlsx @"%fn%\WriteSeasonal.txt"';
   );
);
$endif.excelreport


*--- Summary Report  -- only generated if System Result Report option is set to YES
Parameter
   Capacity(*,f,y)                  'Capacity (MW)'
   CapacitySummary(*,*,y)           'Capacity summary (MW)'
   Energy(*,f,y)                    'Energy (GWh)'
   Costs(*,*,y)                     'Costs summary ($m)'
   Utilization(*,f,y)               'Utilization'
   CapacityType(*,f,tech,y)         'Capacity by type (MW)'
   AverageCost(*,*,y)               'Average cost ($/MWh)'
;
*Alternative summary report declarations (remove one superflous index) for MIRO
$ifI %mode%==MIRO
$onExternalOutput
Parameter
   CapacityMIRO(f,y)                'Capacity (MW)'
   CapacitySummaryMIRO(*,y)         'Capacity summary (MW)'
   EnergyMIRO(f,y)                  'Energy (GWh)'
   CostsMIRO(*,y)                   'Costs summary ($m)'
   UtilizationMIRO(f,y)             'Utilization'
   CapacityTypeMIRO(f,tech,y)       'Capacity by type (MW)'
   AverageCostMIRO(y)               'Average cost ($/MWh)'

;
$ifI %mode%==MIRO
$offExternalOutput

Capacity("Capacity MW",f,y) =  Sum(gprimf(g,f), vCap.l(g,y));
CapacityMIRO(f,y) = Capacity("Capacity MW",f,y);

CapacitySummary(" ","Peak"               ,y) = smax(c, pPeakCapacityCountry(c,"Peak demand: MW",y));
CapacitySummary(" ","New capacity MW"    ,y) = sum(g, vBuild.l(g,y));
CapacitySummary(" ","Retired capacity MW",y) = sum(g, vRetire.l(g,y));
CapacitySummaryMIRO("Peak"            ,y) = CapacitySummary(" ","Peak"               ,y);
CapacitySummaryMIRO("New capacity"    ,y) = CapacitySummary(" ","New capacity MW"    ,y);
CapacitySummaryMIRO("Retired capacity",y) = CapacitySummary(" ","Retired capacity MW",y);

Energy("Energy GWh",f,y) = sum(c, pEnergyByFuelCountry(c,f,y));

EnergyMIRO(f,y) = Energy("Energy GWh",f,y);

Costs("Costs","Capex: $m"       ,y)= sum(z, pCapex(z,y))/1e6;
Costs("Costs","Fixed O&M: $m"   ,y)= sum(z, pFOM(z,y))/1e6;
Costs("Costs","Variable O&M: $m",y)= sum(z, pVOM(z,y))/1e6;
Costs("Costs","Total fuel: $m"  ,y)= sum(z, pFuel(z,y))/1e6;
Costs("Costs","Net Import: $m"  ,y)= sum(z, pImportCosts(z,y) - pExportRevenue(z,y))/1e6;

CostsMIRO("Capex: $m"       ,y) = Costs("Costs","Capex: $m"       ,y);
CostsMIRO("Fixed O&M: $m"   ,y) = Costs("Costs","Fixed O&M: $m"   ,y);
CostsMIRO("Variable O&M: $m",y) = Costs("Costs","Variable O&M: $m",y);
CostsMIRO("Total fuel: $m"  ,y) = Costs("Costs","Total fuel: $m"  ,y);
CostsMIRO("Net Import: $m"  ,y) = Costs("Costs","Net Import: $m"  ,y);

Utilization("Utilization",f,y)$Capacity("Capacity MW",f,y) = Energy("Energy GWh",f,y)/Capacity("Capacity MW",f,y);
UtilizationMIRO(f,y) = Utilization("Utilization",f,y);

CapacityType("Capacity by Type MW",f,tech,y) = sum((gprimf(g,f),gtechmap(g,tech)), vCap.l(g,y));
CapacityTypeMIRO(f,tech,y) = CapacityType("Capacity by Type MW",f,tech,y);

AverageCost(" ","Average cost $/MWh",y) = (sum(ndc, pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex"))*1e6
                                         + sum(dc, vAnnCapex.l(dc,y))
                                         + sum(z, pFOM(z,y) + pVOM(z,y) + pSpinResCosts(z,y) + pFuel(z,y))
                                          )/sum(z, sum((q,d,t,s), pProbaScenarios(s)*(vImportPrice.l(z,q,d,t,y,s)-vExportPrice.l(z,q,d,t,y,s))*pHours(q,d,t))
                                                 + pDenom2(z,y));
AverageCostMIRO(y) = AverageCost(" ","Average cost $/MWh",y);


pDemandSupply(z,"Unmet demand: GWh"      ,y) = sum((q,d,t,s), pProbaScenarios(s)*vUSE.l(z,q,d,t,y,s)*pHours(q,d,t))/1e3




$ifthen.excelreport %DOEXCELREPORT%==1
if (pSystemResultReporting > 0,
   put_utility fgdxxrw 'ren' / 'SystemResults.txt';
$  onPut
   par=Capacity        rng=Summary2!A5:y44
   par=CapacitySummary rng=Summary2!A45:y59
   par=Energy          rng=Summary2!A60:y94
   par=Costs           rng=Summary2!A95:y104
   par=Utilization     rng=Summary2!A105:y139
   par=AverageCost     rng=Summary2!A140:y149
   par=CapacityType    rng=Summary2!A150
$  offPut
   putclose;

   execute_unload 'systemresults', Capacity, CapacitySummary, Energy, Costs, Utilization, CapacityType, AverageCost;
   if (isWindows,
      execute.checkErrorLevel 'gdxxrw systemresults.gdx output=SystemResults.xlsx @SystemResults.txt';
   else
      put fxlsxrep / 'gdxxrw "%fn%\epmresults.gdx" output=SystemResults.xlsx @"%fn%\SystemResults.txt"';
   );
); // end of if statement producing system results file;

put_utility$isWindows 'shell' / 'rm -f WriteZonalandCountry.txt WriteZonal.txt WriteSeasonal.txt SystemResults.txt'
$endif.excelreport

* MIRO
Set pSymbols /
   "Summary of TOT results ($M)",
   "Capex ($)",
   "Annualized capex ($)",
   "FOM ($)",
   "VOM ($)",
   "Fuel costs ($)",
   "Net IMP costs ($)",
   "EXP revenue ($)",
   "Added TRANS costs ($)",
   "Unmet demand costs ($)",
   "Surplus GEN costs ($)",
   "VRE curtailment costs",
   "Unmet sys spin res vio costs",
   "Unmet sys. plan. res vio. costs",
   "Spin. res costs by zone ($)",
   "Unmet loc. spin. res costs",
   "Planning res vio. cost ($)",
   "Costs ($1M, unweight.) by zone",
   "Costs ($1M, unweight.) by ctry",
   "Costs ($1M, weighted) by zone",
   "Costs ($1M, weighted) by ctry",
   "Av. costs (undisc.) by ctry",
   "Fuel costs ($M) by zone",
   "Fuel costs ($M) by ctry",
   "Fuel consumed (1M) per zone",
   "Fuel consumed (1M) per ctry",
   "Energy by plant (GWh)",
   "Energy by fuel per zone (GWh)",
   "Energy by fuel per ctry (GWh)",
   "Energy mix by ctry",
   "Supply demand per zone (GWh)",
   "Supply demand per ctry (GWh)",
   "TOT EXCH per zone (GWh)",
   "UTIL. of interconnection",
   "TRANS losses per zone (MWh)",
   "TOT EXCH per ctry (GWh)",
   "TOT TRANS losses per ctry (MWh)",
   "Trade by year per zone (GWh)",
   "Trade in MW per zone",
   "Trade by year per ctry (GWh)",
   "Trade in MW per ctry",
   "MRGL cost per MWh per zone ($)",
   "Av. price by int. zone",
   "Av. price hub",
   "Av. MRGL cost EXPto int ($/MWh)",
   "Av. MRGL cost IMPto int ($/MWh)",
   "Av. price by int. zone per ctry",
   "Av. MRGL cost EXPint ctry $/MWh",
   "Av. MRGL cost IMPint ctry $/MWh",
   "Peak CAP in MW per zone",
   "Peak CAP by PRIM fuel (MW/zone)",
   "New CAP by fuel (MW/zone)",
   "Plant utilization",
   "retirements in MW per zone",
   "CAP plan MW per zone",
   "Peak CAP in MW per ctry",
   "Peak CAP by PRIM fuel (MW/ctry)",
   "New CAP by fuel (MW/ctry)",
   "CAP plan (MW/ctry)",
   "Add. TRANS CAP b/w zones (MW)",
   "Cost reserves by plant ($/zone)",
   "RES contrib by plant (MW/zone)",
   "Cost reserves by plant ($/ctry)",
   "RES contrib by plant (MWh/ctry)",
   "EMIS CO2 by zone (MT)",
   "EMIS intensity tCO2/GWh by zone",
   "EMIS CO2 by ctry (MT)",
   "EMIS intensity tCO2/GWh by ctry",
   "Energy by plant (MWh)",
   "Energy by zone (MWh)",
   "Energy by ctry (MWh)",
   "Plant LCOE by year ($/MWh)",
   "Zonal cost of GEN + IMP($/MWh)",
   "Zonal cost of GEN ($/MWh)",
   "Ctry cost of GEN+ trade ($/MWh)",
   "Ctry cost of GEN ($/MWh)",
   "System cost of GEN ($/MWh)",
   "Detailed dispatch by plant (MW)",
   "Detailed dispatch and flows","CSP Balance (MW)",
   "CSP specific output","PV with Storage balance",
   "PV with Storage components","Storage balance (MW)",
   "Storage components",
   "Solar energy value ($)",
   "Solar energy cost",
   "Solar output (MWh)",
   "Solver parameters",
   "Seasonal demand supply per zone",
   "IMP by seas. per zone (GWh)",
   "EXP by seas. per zone (GWh)",
   "Energy by plant per seas. (GWh)",
   "TOT EXCH per seas per zone(GWh)",
   "Trade by seas. per zone (GWh)",
   "TOT EXCH per seas per ctry(GWh)",
   "Trade by seas. per ctry (GWh)",
   "Zones per ctry",
   "System Capacity (MW)",
   "System Capacity summary (MW)",
   "System Energy (GWh)",
   "System Costs ($m)",
   "System Utilization",
   "System Capacity by type (MW)",
   "System Average cost ($/MWh)" /;

$ifI %mode%==MIRO
$onExternalOutput
parameter cubeOutput(pSymbols,*,*,*,*,*,*,*,*,*,*,*,*);
$ifI %mode%==MIRO
$offExternalOutput

$ifI not %mode%==MIRO $exit

*Write veda file
File veda_file / out.vdd /;
put veda_file;
$onPut
[DataBaseName]
myveda

[Dimensions]
Attribute                           attr
subgroup                            gr
zones                               z
zones                               z2
years                               y
countries                           c
countries                           c2
fuels                               f
generatorsOrTechnologyFuelFypes     g
quartersOrSeasons                   q
dayTypes                            d
hoursOfDay                          t
technologies                        tech

[DataEntries]
* VEDA Attr                           GAMS                                                - indexes -
*** Variables & Parameters
"Summary of TOT results ($M)"         pSummary                                            s
"Costs ($1M, unweight.) by zone"      pCostSummary                                        z gr y
"Costs ($1M, unweight.) by ctry"      pCostSummaryCountry                                 c gr y
"Costs ($1M, weighted) by zone"       pCostSummaryWeighted                                z gr y
"Costs ($1M, weighted) by ctry"       pCostSummaryWeightedCountry                         c gr y
"CO2 costs ($M)by country"            pCO2backstopCosts                                   c y
"Av. costs (undisc.) by ctry"         pCostSummaryWeightedAverageCtry                     c s
"Fuel costs ($M) by zone"             pFuelCosts                                          z f y
"Fuel costs ($M) by ctry"             pFuelCostsCountry                                   c f y
"Fuel consumed (1M) per zone"         pFuelConsumption                                    z f y
"Fuel consumed (1M) per ctry"         pFuelConsumptionCountry                             c f y
"Energy by plant (GWh)"               pEnergyByPlant                                      z g y
"Energy by fuel per zone (GWh)"       pEnergyByFuel                                       z f y
"Energy by fuel per ctry (GWh)"       pEnergyByFuelCountry                                c f y
"Energy mix by ctry"                  pEnergyMix                                          c f y
"Supply demand per zone (GWh)"        pDemandSupply                                       z gr y
"Supply demand per ctry (GWh)"        pDemandSupplyCountry                                c gr y
"TOT EXCH per zone (GWh)"             pInterchange                                        z z2 y
"UTIL. of interconnection"            pInterconUtilization                                z z2 y
"TRANS losses per zone (MWh)"         pLossesTransmission                                 z y
"TOT EXCH per ctry (GWh)"             pInterchangeCountry                                 c c2 y
"TOT TRANS losses per ctry (MWh)"     pLossesTransmissionCountry                          c y
"Trade by year per zone (GWh)"        pYearlyTrade                                        z gr y
"Trade in MW per zone"                pHourlyTrade                                        z y q d gr t
"Trade by year per ctry (GWh)"        pYearlyTradeCountry                                 c gr y
"Trade in MW per ctry"                pHourlyTradeCountry                                 c y q d gr t
"MRGL cost per MWh per zone ($)"      pPrice                                              z q d t y
"Av. price by int. zone"              pAveragePrice                                       z y
"Av. price hub"                       pAveragePriceHub                                    z y
"Av. MRGL cost EXPto int ($/MWh)"     pAveragePriceExp                                    z y
"Av. MRGL cost IMPto int ($/MWh)"     pAveragePriceImp                                    z y
"Av. price by int. zone per ctry"     pAveragePriceCountry                                c y
"Av. MRGL cost EXPint ctry $/MWh"     pAveragePriceExpCountry                             c y
"Av. MRGL cost IMPint ctry $/MWh"     pAveragePriceImpCountry                             c y
"Peak CAP in MW per zone"             pPeakCapacity                                       z gr y
"Peak CAP by PRIM fuel (MW/zone)"     pCapacityByFuel                                     z f y
"New CAP by fuel (MW/zone)"           pNewCapacityFuel                                    z f y
"Plant utilization"                   pPlantUtilization                                   z g y
"retirements in MW per zone"          pRetirements                                        z g y
"CAP plan MW per zone"                pCapacityPlan                                       z g y
"Peak CAP in MW per ctry"             pPeakCapacityCountry                                c gr y
"Peak CAP by PRIM fuel (MW/ctry)"     pCapacityByFuelCountry                              c f y
"New CAP by fuel (MW/ctry)"           pNewCapacityFuelCountry                             c f y
"CAP plan (MW/ctry)"                  pCapacityPlanCountry                                c g y
"Add. TRANS CAP b/w zones (MW)"       pAdditionalCapacity                                 z z2 y
"Cost reserves by plant ($/zone)"     pReserveCosts                                       z g y
"RES contrib by plant (MW/zone)"      pReserveByPlant                                     z g y
"Cost reserves by plant ($/ctry)"     pReserveCostsCountry                                c g y
"RES contrib by plant (MWh/ctry)"     pReserveByPlantCountry                              c g y
"EMIS CO2 by zone (MT)"               pEmissions                                          z y
"EMIS intensity tCO2/GWh by zone"     pEmissionsIntensity                                 z y
"EMIS CO2 by ctry (MT)"               pEmissionsCountry                                   c y
"EMIS intensity tCO2/GWh by ctry"     pEmissionsIntensityCountry                          c y
"Plant LCOE by year ($/MWh)"          pPlantAnnualLCOE                                    z g y
"Zonal cost of GEN + IMP($/MWh)"      pZonalAverageCost                                   z y
"Ctry cost of GEN+ trade ($/MWh)"     pCountryAverageCost                                 c y
"System cost of GEN ($/MWh)"          pSystemAverageCost                                  y
"Detailed dispatch by plant (MW)"     pPlantDispatch                                      z y q d g t
"Detailed dispatch and flows"         pDispatch                                           z y q d gr t
"CSP Balance (MW)"                    pCSPBalance                                         y g q d gr t
"CSP specific output"                 pCSPComponents                                      g gr y
"PV with Storage balance"             pPVwSTOBalance                                      y q d g gr t
"PV with Storage components"          pPVwSTOComponents                                   g gr y
"Storage balance (MW)"                pStorageBalance                                     y g  q d gr t
"Storage components"                  pStorageComponents                                  g gr y
"Solar energy value ($)"              pSolarValue                                         z y
"Solar energy cost"                   pSolarCost                                          z y
"Seasonal demand supply per zone"     pDemandSupplySeason                                 z gr y q
"Energy by plant per seas. (GWh)"     pEnergyByPlantSeason                                z g y q
"TOT EXCH per seas per zone(GWh)"     pInterchangeSeason                                  z z2 y q
"Trade by seas. per zone (GWh)"       pSeasonTrade                                        z gr y q
"TOT EXCH per seas per ctry(GWh)"     pInterchangeSeasonCountry                           c c2 y q
"Trade by seas. per ctry (GWh)"       pSeasonTradeCountry                                 c gr y q
"System Capacity (MW)"                CapacityMIRO                                        f y
"System Capacity summary (MW)"        CapacitySummaryMIRO                                 gr y
"System Energy (GWh)"                 EnergyMIRO                                          f y
"System Costs ($m)"                   CostsMIRO                                           gr y
"System Utilization"                  UtilizationMIRO                                     f y
"System Capacity by type (MW)"        CapacityTypeMIRO                                    f tech y
"System Average cost ($/MWh)"         AverageCostMIRO                                     y
$offPut
putclose;

*prepare output cube
execute_unload 'cubeData';

put_utility 'shell' / 'gdx2veda cubeData.gdx out.vdd';

option zerotoeps=on, clear=cubeoutput;
set dummy 'veda uel for missing index' / '-' /;

embeddedCode Python:
import pandas as pd
import gams2numpy
gams.wsWorkingDir = '.'

arr = pd.read_csv('cubeData.vd', delimiter=',', quotechar='"', engine='c', low_memory=False, skiprows=12, header=None).values
g2np = gams2numpy.Gams2Numpy(gams.ws.system_directory)
g2np.gmdFillSymbolStr(gams.db, gams.db['cubeOutput'], arr)
endEmbeddedCode cubeOutput
put_utility 'shell' / 'rm -f cubeData.gdx'


display pNominalCapacity;
