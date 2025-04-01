$onMulti
* Report

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
   pFuelCostsZone(z,y)                       'Fuel costs in USD'
   pImportCostsExternal(z,y)                 'Net import costs from trade with external zones in USD'
   pExportRevenuesExternal(z,y)               'Export revenue from trade with external zones in USD'
   pImportCostsTopology(z,y)                 'Import costs with internal zones in USD'
   pExportRevenuesTopology(z,y)              'Export revenues with internal zones in USD'
   pCongestionRevenues(z,z2,y)               'Congestion revenues from saturation of direction z,Zd in USD'
   pTradeCostsTopology(z,y)                  'Trade costs from imports, exports and sharing congestion rent among countries in USD'
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
   pCostsbyPlant(z,g,*,y)                    'Yearly Costs by Plant'
   pCostSummary(z,*,y)                       'Summary of costs in millions USD  (unweighted and undiscounted) by zone'
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
   pEnergyByTechandFuel(z,tech,f,y)          'Energy by technology and fuel in GWh per zone'
   pEnergyByTechandFuelCountry(c,tech,f,y)   'Energy by technology and fuel in GWh per country'
   pEnergyMix(c,f,y)                         'Energy mix by country'
                                             
   pDemandSupply(z,*,y)                      'Supply demand in GWh per zone'
   pDemandSupplyCountry(c,*,y)               'Supply demand in GWh per country'
 

************************************H2  model additions***************************************************  
   pDemandSupplyH2(z,*,y)                      'H2 Supply demand in mmBTU of H2 per zone'
   pDemandSupplyCountryH2(c,*,y)               'H2 Supply demand in mmBTU of H2 per country'
   pCapacityPlanH2(z,hh,y)                      'Capacity plan of electrolyzers in MW per zone'

***********************************************************************************************************

   pInterchange(z,z2,y)                      'Total exchange in GWh between internal zones  per zone'
   pInterchangeExtExp(z, zext, y)                'Total exchange in GWh between internal and external zones  per zone for exportd'
   pInterchangeExtImp(zext, z, y)                'Total exchange in GWh between internal and external zones  per zone for imports'
   pInterconUtilization(z,z2,y)              'Utilization of interconnection throughout modeling horizon (between internal zones)'
   pInterconUtilizationExtExp(z,zext,y)         'Utilization of interconnection throughout modeling horizon (between external zones for exports)'
   pInterconUtilizationExtImp(zext,z,y)         'Utilization of interconnection throughout modeling horizon (between external zones) for imports'
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
   pPeakCapacity(z,*,y)                      'Peak capacity in MW per zone'
   pCapacityByFuel(z,f,y)                    'Peak capacity by primary fuel in MW per zone'
   pNewCapacityFuel(z,f,y)                   'New capacity by fuel in MW per zone'
   pNewCapacityTech(z,tech,y)                'New capacity by technology in MW per zone'
   pPlantUtilization(z,g,y)                  'Plant utilization'
   pPlantUtilizationTech(z,tech,y)           'Plant utilization with Technology'
   pFuelUtilization(z,f,y)                   'Plant utilization with fuel'  
   pRetirements(z,g,y)                       'Retirements in MW per zone'
   pRetirementsFuel(z,f,y)                   'Retirements by fuel in MW per zone'
   pRetirementsCountry(c,y)                'Retirements in MW per country'
   pRetirementsFuelCountry(c,f,y)            'Retirements by fuel in MW per country'
   pCapacityPlan(z,g,y)                      'Capacity plan MW per zone'
   pPeakCapacityCountry(c,*,y)               'Peak capacity in MW  per country'
   pCapacityByFuelCountry(c,f,y)             'Peak capacity by primary fuel in MW per country'
   pNewCapacityFuelCountry(c,f,y)            'New capacity by fuel in MW per country'
   pNewCapacityTechCountry(c,tech,y)         'New capacity by technology in MW per country'
   pCapacityPlanCountry(c,g,y)               'Capacity plan MW per country'
   pAdditionalCapacity(z,z2,y)             'Additional transmission capacity between internal zones in MW'
   pAnnualTransmissionCapacity(z,z2,y)       'Annual transmission capacity including base and new ones'
   pAdditionalCapacityCountry(c,*,y)         'Additional transmission capacity between internal countries in MW'
   pCapacityByTechandFuel(z,tech,f,y)        'Peak capacity by technology and primary fuel in MW per zone'
   pCapacityByTechandFuelCountry(c,tech,f,y) 'Peak capacity by technology and primary fuel in MW per country'
   pUtilizationByFuel(z,f,y)                 'Average capacity factor by fuel'
   pUtilizationByTechandFuel(z,tech,f,y)     'Average capacity factor by technology and fuel'
   pUtilizationByFuelCountry(c,f,y)          'Average capacity factor by fuel in country'
   pUtilizationByTechandFuelCountry(c,tech,f,y) 'Average capacity factor by technology and fuel in country'
                                         
   
   pSpinningReserveCostsZone(z,g,y)                      'Cost of reserves by plant in dollars per zone'
   pSpinningReserveByPlantZone(z,g,y)                    'Reserve contribution by plant in MWh per zone'
   pSpinningReserveCostsCountry(c,g,y)               'Cost of reserves by plant in dollars per country'
   pSpinningReserveByPlantCountry(c,g,y)             'Reserve contribution by plant in MWh per country'

   pReserveMarginRes(z,*,y)                     'Resulting reserve margin calculations by zone'
   pReserveMarginResCountry(c,*,y)              'Resulting reserve margin calculations by country'

                                         
   pEmissions(z,y)                           'Emissions in Megaton CO2 by zone'
   pEmissionsIntensity(z,y)                  'Emissions intensity tCO2 per GWh by zone'
   pEmissionsCountry1(c,*,y)                  'Emissions in Megaton CO2 by country'
   pEmissionsIntensityCountry(c,y)           'Emissions intensity tCO2 per GWh by country'
   pEmissionMarginalCostsCountry(c,y)        'Marginal costs of Emission Limit Constraint eEmissionsCountry'        
   pEmissionMarginalCosts(y)                 'Marginal costs of Emission Limit Constraint eTotalEmissionsConstraint'                                                                                    
   
   pDenom(z,g,y)                             'Energy by plant in MWh'
   pDenom2(z,y)                              'Energy by zone in MWh'
   pDenom3(c,y)                              'Energy by country in MWh'
   pPlantAnnualLCOE(z,g,y)                   'Plant levelized cost by year USD per MWh'
   pZonalAverageCost(z,y)                    'Zonal annual cost of generation+ import by year USD per MWh'
   pZonalAverageGenCost(z,y)                 'Zonal annual cost of generation by year USD per MWh'
   pCountryAverageCost(c,y)                  'Country annual cost of generation + trade by year USD per MWh'
   pCountryAverageGenCost(c,y)               'Country annual cost of generation by year USD per MWh'
   pSystemAverageCost(y)                     'System annual cost of generation by year USD per MWh'
                                             
   pPlantDispatch(z,y,q,d,g,t,*)             'Detailed dispatch and reserve by plant in MW'
   pDispatch(z,y,q,d,*,t)                    'Detailed dispatch and flows'
   

                                             
   pCSPBalance(y,g,q,d,*,t)                  'in MW'
   pCSPComponents(g,*, y)                    'CSP specific output'
   pPVwSTOBalance(y,q,d,g,*,t)               'pVwithStorage specific output'
   pPVwSTOComponents(g,*,y)                  'pVwithStorage specific output'
   pStorageBalance(y,g,q,d,*,t)              'in MW'
   pStorageComponents(g,*,y)                 'Storage specific output'
                                             
   pSolarValue(z,y)                          'Value of solar energy in USD'
   pSolarCost(z,y)                           
   pSolarEnergy(z,q,d,t,y)                   'Solar output in MWh'                                        
   pSolverParameters(*)                      'Solver parameters'                                                                                 
   pDemandSupplySeason(z,*,y,q)              'Seasonal demand supply parameters per zone'
***********************H2 related addition*********************************************
   pDemandSupplySeasonH2(z,*,y,q)              'Seasonal demand supply parameters per zone'
******************************************************************************************
   
   pEnergyByPlantSeason(z,g,y,q)             'Energy by plant in GWh per season  per zone'
   pInterchangeSeason(z,z2,y,q)              'Total exchange in GWh between internal zones per season per zone'
   pSeasonTrade(z,*,y,q)                     'Trade with external zones by season in GWh per zone'
   pInterchangeSeasonCountry(c,c2,y,q)       'Total exchange in GWh between internal zones per season per country'
   pSeasonTradeCountry(c,*,y,q)              'Trade with external zones by season in GWh per country'
   pZonesperCountry(c)                       'Number of zones per country'
   pTotalHoursperYear(y)                     'Number of hours in a given year'       
;

Scalar MaxZonesperCountry;

*--- START of results

pTotalHoursperYear(y) = sum((q,d,t), pHours(q,d,t));
pSpinningReserveCostsZone(z,g,y)          = 0;
pSpinningReserveCostsCountry(c,g,y)   = 0;

*--- Cost Items and Penalty Costs

* zonal level
pCapex(z,y) = 1e6*sum(gzmap(g,z),  vBuild.l(g,y)*pGenData(g,"Capex")*pCapexTrajectories(g,y))
            + 1e3*sum(gzmap(st,z), vBuildStor.l(st,y)*pStorData(st,"CapexMWh")*pCapexTrajectories(st,y))
            + 1e3*sum(gzmap(cs,z), vBuildTherm.l(cs,y)*pCSPData(cs,"Thermal Field","CapexMWh")*1e3*pCapexTrajectories(cs,y)
                                 + vBuildStor.l(cs,y)*pCSPData(cs,"Storage","CapexMWh")*pCapexTrajectories(cs,y))
            +1e6*sum(h2zmap(hh,z), vBuildH2.l(hh,y)*pH2Data(hh,"Capex")*pCapexTrajectoriesH2(hh,y))$pIncludeH2+1e-5 ;

pAnncapex(z,y) = 1e6*sum(gzmap(ndc,z),               pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex"))
               + 1e3*sum(gzmap(ndc,z)$(not cs(ndc)), pCRFsst(ndc)*vCapStor.l(ndc,y)*pStorData(ndc,"CapexMWh"))
               + 1e3*sum(gzmap(ndc,z)$(not st(ndc)), pCRFcst(ndc)*vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","CapexMWh"))
               + 1e6*sum(gzmap(ndc,z),               pCRFcth(ndc)*vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal Field","CapexMWh"))
               +     sum(gzmap(dc,z),                vAnnCapex.l(dc,y))
               +     sum(h2zmap(dch2,z),                vAnnCapexH2.l(dch2,y))
               +1e6*sum(h2zmap(ndcH2,z),              pCRFH2(ndcH2)*vCapH2.l(ndcH2,y)*pH2Data(ndcH2,"Capex"))$pIncludeH2;

pFOM(z,y) = sum(gzmap(g,z),  vCap.l(g,y)*pGenData(g,"FOMperMW"))
          + sum(gzmap(st,z), vCapStor.l(st,y)*pStorData(st,"FixedOMMWh"))
          + sum(gzmap(cs,z), vCapStor.l(cs,y)*pCSPData(cs,"Storage","FixedOMMWh"))
          + sum(gzmap(cs,z), vCapTherm.l(cs,y)*pCSPData(cs,"Thermal field","FixedOMMWh"))
**************************H2 model addition**************************************************************
          + sum(h2zmap(hh,z),  vCapH2.l(hh,y)*pH2Data(hh,"FOMperMW"))$pIncludeH2;


pSpinResCosts(z,y) = vYearlySpinningReserveCost.l(z,y);


pVOM(z,y) = sum((q,d,t), pHours(q,d,t)*(
                           sum((gzmap(g,z),gfmap(g,f)),   pGenData(g,"VOM")*vPwrOut.l(g,f,q,d,t,y))
                         + sum((gzmap(st,z),gfmap(st,f)), pStorData(st,"VOMMWh")*vPwrOut.l(st,f,q,d,t,y))
                         + sum((gzmap(cs,z),gfmap(cs,f)), pCSPData(cs,"Storage","VOMMWh")*vPwrOut.l(cs,f,q,d,t,y))
                         + sum((gzmap(cs,z),gfmap(cs,f)), pCSPData(cs,"Thermal Field","VOMMWh")*vPwrOut.l(cs,f,q,d,t,y))
**************************H2 model addition**************************************************************
***(units for equation below)**********       $/mMBTU_H2          mmBTU_H2/MWh_e        MW_e
                        + sum((h2zmap(hh,z)), pH2Data(hh,"VOM")*pH2Data(hh,"Heatrate")*vH2PwrIn.l(hh,q,d,t,y))$pIncludeH2));


pFuelCostsZone(z,y) = sum((gzmap(g,z),gfmap(g,f),zcmap(z,c),q,d,t), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t)*pFuelPrice(c,f,y)*pHeatRate(g,f));


pImportCostsExternal(z,y) = sum((zext,q,d,t), vImportPrice.l(z,zext,q,d,t,y)*pTradePrice(zext,q,d,y,t)*pHours(q,d,t));
pExportRevenuesExternal(z,y) = sum((zext,q,d,t), vExportPrice.l(z,zext,q,d,t,y)*pTradePrice(zext,q,d,y,t)*pHours(q,d,t));

pPrice(z,q,d,t,y)$(pHours(q,d,t)) = -eDemSupply.m(z,q,d,t,y)/pHours(q,d,t)/pRR(y)/pWeightYear(y);

pImportCostsTopology(z,y) = sum((sTopology(Zd,z),q,d,t), pPrice(z,q,d,t,y)*vFlow.l(Zd,z,q,d,t,y)*pHours(q,d,t));
pExportRevenuesTopology(z,y) = - sum((sTopology(z,Zd),q,d,t), pPrice(z,q,d,t,y)*vFlow.l(z,Zd,q,d,t,y)*pHours(q,d,t));
pCongestionRevenues(z,Zd,y) = - sum((q,d,t), (pPrice(zD,q,d,t,y) - pPrice(z,q,d,t,y))*vFlow.l(z,Zd,q,d,t,y)*pHours(q,d,t));
* Choosing one allocation rule for the congestion rent
pTradeCostsTopology(z,y) = pExportRevenuesTopology(z,y) + pImportCostsTopology(z,y) + 0.5*sum(sTopology(Zd,z), pCongestionRevenues(Zd,z,y)) + 0.5*sum(sTopology(z,Zd), pCongestionRevenues(z,Zd,y));

* Dividing pNewTransmissionCosts by 2 to avoid double-counting for the two countries involved in transmission line
pNewTransmissionCosts(z,y) = vYearlyTransmissionAdditions.l(z,y) / 2;
pUSECosts(z,y) = vYearlyUSECost.l(z,y);
pCO2backstopCosts(c,y) = vYearlyCO2backstop.l(c,y)*pCostOfCO2backstop;
pSurplusCosts(z,y) = vYearlySurplus.l(z,y);
pVRECurtailment(z,y) = vYearlyCurtailmentCost.l(z,y);
        

* country level
pCountryPlanReserveCosts(c,y) = vUnmetPlanningReserveCountry.l(c,y)*pPlanningReserveVoLL;
pUSRLocCosts(c,y) = sum((q,d,t), vUnmetSpinningReserveCountry.l(c,q,d,t,y)*pHours(q,d,t)*pSpinningReserveVoLL);

* System level
pUSRSysCosts(y) = sum((q,d,t), vUnmetSpinningReserveSystem.l(q,d,t,y)*pHours(q,d,t)*pSpinningReserveVoLL);
pUPRSysCosts(y) = vUnmetPlanningReserveSystem.l(y)*pPlanningReserveVoLL;

 

set sumhdr /

   "Annualized capex: $m"      
   "Fixed O&M: $m"             
   "Variable O&M: $m"          
   "Total fuel Costs: $m"      
   "Transmission additions: $m"
   "Spinning Reserve costs: $m"
   "Unmet demand costs: $m"    
   "Excess generation: $m"     
   "VRE curtailment: $m"       
   "Import costs wiht external zones: $m"          
   "Export revenues with external zones: $m"
   "Min Gen penalty cost: $m"/;
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
   "Average Export revenue: $m"
   "Average Min Gen penalty cost: $m" /;
set sumhdrmap(avgsumhdr,sumhdr) / #avgsumhdr:#sumhdr /;
   
*--- Cost Summary Unweighted by zone
pCostSummary(z,"Capex: $m"                    ,y) = pCapex(z,y)/1e6;
pCostSummary(z,"Annualized capex: $m"         ,y) = pAnncapex(z,y)/1e6;
pCostSummary(z,"Fixed O&M: $m"                ,y) = pFOM(z,y)/1e6;
pCostSummary(z,"Variable O&M: $m"             ,y) = pVOM(z,y)/1e6;

pCostSummary(z,"Total fuel Costs: $m"         ,y) = pFuelCostsZone(z,y)/1e6;

* Should we keep O to be sure we are considering all cost components ?
pCostSummary(z,"Transmission additions: $m"              ,y) = pNewTransmissionCosts(z,y)/1e6;
pCostSummary(z,"Spinning Reserve costs: $m"              ,y) = pSpinResCosts(z,y)/1e6;
pCostSummary(z,"Unmet demand costs: $m"                  ,y) = pUSECosts(z,y)/1e6;
pCostSummary(z,"Excess generation: $m"                   ,y) = pSurplusCosts(z,y)/1e6;
pCostSummary(z,"VRE curtailment: $m"                     ,y) = pVRECurtailment(z,y)/1e6;
pCostSummary(z,"Import costs wiht external zones: $m"    ,y) = pImportCostsExternal(z,y)/1e6;
pCostSummary(z,"Export revenues with external zones: $m" ,y) = pExportRevenuesExternal(z,y)/1e6;
pCostSummary(z,"Import costs with internal zones: $m"    ,y) = pImportCostsTopology(z,y)/1e6;
pCostSummary(z,"Export revenues with internal zones: $m" ,y) = pExportRevenuesTopology(z,y)/1e6;
pCostSummary(z,"Trade Costs: $m"                         ,y) = pTradeCostsTopology(z,y)/1e6;


pCostSummary(z,"Total Annual Cost by Zone: $m",y) = ( pAnncapex(z,y) + pNewTransmissionCosts(z,y) + pFOM(z,y) + pVOM(z,y) + pFuelCostsZone(z,y)
                                                    + pImportCostsExternal(z,y) - pExportRevenuesExternal(z,y) + pUSECosts(z,y) + pVRECurtailment(z,y)
                                                    + pSurplusCosts(z,y) + pSpinResCosts(z,y))/1e6;
pCostSummary(z,"Total Annual Cost by Zone with trade: $m",y) = (pTradeCostsTopology(z,y) + pAnncapex(z,y) + pNewTransmissionCosts(z,y) + pFOM(z,y) + pVOM(z,y) + pFuelCostsZone(z,y)
                                                    + pImportCostsExternal(z,y) - pExportRevenuesExternal(z,y) + pUSECosts(z,y) + pVRECurtailment(z,y)
                                                    + pSurplusCosts(z,y) + pSpinResCosts(z,y))/1e6;


*--- Cost Summary Unweighted by country

pCostSummaryCountry(c,"Capex: $m",y)= sum(zcmap(z,c), pCapex(z,y))/1e6 ;                     

pCostSummaryCountry(c,sumhdr,y) = sum(zcmap(z,c), pCostSummary(z,sumhdr,y));

pCostSummaryCountry(c,"Country Spinning Reserve violation: $m",y) = pUSRLocCosts(c,y)/1e6;
pCostSummaryCountry(c,"Country Planning Reserve violation: $m",y) = pCountryPlanReserveCosts(c,y)/1e6;
pCostSummaryCountry(c,"Total CO2 backstop cost by Country: $m",y) = pCO2backstopCosts(c,y)/1e6 ;

pCostSummaryCountry(c,"Total Annual Cost by Country: $m"      ,y) = sum(zcmap(z,c), pCostSummary(z,"Total Annual Cost by Zone: $m",y))
                                                                  + (pUSRLocCosts(c,y) + pCountryPlanReserveCosts(c,y)
                                                                  + pCO2backstopCosts(c,y))/1e6 ;

                                                      


*--- Cost Summary Weighted by zone
pCostSummaryWeighted(z,sumhdr,y) = pWeightYear(y)*pCostSummary(z,sumhdr,y);


pCostSummaryWeighted(z,"Total Annual Cost by Zone: $m",y) = pWeightYear(y)*(pAnncapex(z,y) + pNewTransmissionCosts(z,y) + pFOM(z,y) + pVOM(z,y)
                                                                          + pFuelCostsZone(z,y) + pImportCostsExternal(z,y) - pExportRevenuesExternal(z,y) + pUSECosts(z,y)
                                                                          + pVRECurtailment(z,y) + pSurplusCosts(z,y) + pSpinResCosts(z,y))/1e6;

*--- Cost Summary Weighted by country

pCostSummaryWeightedCountry(c,"Capex: $m",y)= pCostSummaryCountry(c,"Capex: $m",y);                     

pCostSummaryWeightedCountry(c,sumhdr,y) = sum(zcmap(z,c), pWeightYear(y)*pCostSummary(z,sumhdr,y));

pCostSummaryWeightedCountry(c,"Country Spinning Reserve violation: $m",y) = pWeightYear(y)*pUSRLocCosts(c,y)/1e6;
pCostSummaryWeightedCountry(c,"Country Planning Reserve violation: $m",y) = pWeightYear(y)*pCountryPlanReserveCosts(c,y)/1e6;
pCostSummaryWeightedCountry(c,"Total CO2 backstop cost by Country: $m"      ,y) = pWeightYear(y)*pCO2backstopCosts(c,y)/1e6 ;
pCostSummaryWeightedCountry(c,"Total Annual Cost by Country: $m"      ,y) = sum(zcmap(z,c), pCostSummaryWeighted(z,"Total Annual Cost by Zone: $m",y))
                                                                          + pWeightYear(y)*(pCountryPlanReserveCosts(c,y) + pUSRLocCosts(c,y)+pCO2backstopCosts(c,y))/1e6;


*--- Cost Summary Weighted Averages by country
pCostSummaryWeightedAverageCountry(c,avgsumhdr,'Summary') = sum((sumhdrmap(avgsumhdr,sumhdr),y), pCostSummaryWeightedCountry(c,sumhdr,y))/TimeHorizon;

pCostSummaryWeightedAverageCountry(c,"Average Capex: $m","Summary")=sum(y, pCostSummaryWeightedCountry(c,"Capex: $m",y))/TimeHorizon;;                     

pCostSummaryWeightedAverageCountry(c,"Average Spinning Reserve violation: $m","Summary")    = sum(y, pCostSummaryWeightedCountry(c,"Country Spinning Reserve violation: $m",y))/TimeHorizon;
pCostSummaryWeightedAverageCountry(c,"Average Planning Reserve violation: $m","Summary")    = sum(y, pCostSummaryWeightedCountry(c,"Country Planning Reserve violation: $m",y))/TimeHorizon;
pCostSummaryWeightedAverageCountry(c,"Average CO2 backstop cost by Country: $m" ,"Summary") = sum(y, pCostSummaryWeightedCountry(c,"Total CO2 backstop cost by Country: $m",y))/TimeHorizon;
pCostSummaryWeightedAverageCountry(c,"Average Total Annual Cost: $m"         ,"Summary")    = sum(y, pCostSummaryWeightedCountry(c,"Total Annual Cost by Country: $m",y))/TimeHorizon;


* TODO: Why that ? pCostSummaryWeightedAverageCtry ?
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
pCostSummaryWeightedAverageCtry(c,"Average CO2 backstop cost by Country: $m" )= pCostSummaryWeightedAverageCountry(c,"Average CO2 backstop cost by Country: $m" ,"Summary"); 
pCostSummaryWeightedAverageCtry(c,"Average Total Annual Cost: $m")          = pCostSummaryWeightedAverageCountry(c,"Average Total Annual Cost: $m","Summary");         


*--- Cost and consumption by fuel

* By zone
pFuelCosts(z,f,y) = sum((gzmap(g,z),gfmap(g,f),zcmap(z,c),q,d,t), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t)*pFuelPrice(c,f,y)*pHeatRate(g,f))/1e6;
pFuelConsumption(z,f,y) = vFuel.l(z,f,y)/1e6;

* By country
pFuelCostsCountry(c,f,y) = sum(zcmap(z,c), pFuelCosts(z,f,y));
pFuelConsumptionCountry(c,f,y) = sum(zcmap(z,c), pFuelConsumption(z,f,y));


*--- Energy Results (Energy by Fuel, by Plant and Energy mix)
set zgmap(z,g); option zgmap<gzmap;

**************************Hydrogen model****************************
set zH2map(z,hh); option zH2map<h2zmap;

pEnergyByFuel(z,f,y) = sum((gzmap(g,z),gfmap(g,f),q,d,t), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t))/1e3;
pEnergyByFuelCountry(c,f,y) = sum(zcmap(z,c), pEnergyByFuel(z,f,y));
pEnergyByTechandFuel(z,tech,f,y) = sum((gzmap(g,z),gtechmap(g,tech),gfmap(g,f),q,d,t), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t))/1e3;
pEnergyByTechandFuelCountry(c,tech,f,y) = sum(zcmap(z,c), pEnergyByTechandFuel(z,tech,f,y));

pEnergyMix(c,f,y) = pEnergyByFuelCountry(c,f,y)/(sum(f2, pEnergyByFuelCountry(c,f2,y)) + 1e-5);

pEnergyByPlant(zgmap(z,g),y) = sum((gfmap(g,f),q,d,t), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t))/1e3;

*--- Demand-Supply Balance

*A. Demand - Production Balance

* By zone
pDemandSupply(z,"Demand: GWh"            ,y) = sum((q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))/1e3;
*********************************************************H2 model addition****************************************************************
pDemandSupply(z,"Electricity demand for H2 production: GWh",         y) =(sum((h2zmap(hh,z),q,d,t),vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t) )/1e3)$pIncludeH2+0.000001;
pDemandSupply(z,"Total Demand (Including P to H2): GWh"      ,y) =(sum((q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))/1e3+sum((h2zmap(hh,z),q,d,t),vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t) )/1e3)$pIncludeH2+0.000001;

pDemandSupply(z,"Total production: GWh"  ,y) = sum(gzmap(g,z), pEnergyByPlant(z,g,y));
pDemandSupply(z,"Unmet demand: GWh"      ,y) = sum((q,d,t), vUSE.l(z,q,d,t,y)*pHours(q,d,t))/1e3;
pDemandSupply(z,"Surplus generation: GWh",y) = sum((q,d,t), vSurplus.l(z,q,d,t,y)*pHours(q,d,t))/1e3;

pDemandSupply(z,"Imports exchange: GWh"     ,y) = (sum((sTopology(z,z2),q,d,t), vFlow.l(z2,z,q,d,t,y)*pHours(q,d,t)) + sum((zext,q,d,t), vImportPrice.l(z,zext,q,d,t,y)*pHours(q,d,t)))/1e3;
pDemandSupply(z,"Exports exchange: GWh"     ,y) = (sum((sTopology(z,z2),q,d,t), vFlow.l(z,z2,q,d,t,y)*pHours(q,d,t)) + sum((zext,q,d,t), vExportPrice.l(z,zext,q,d,t,y)*pHours(q,d,t)))/1e3;
pDemandSupply(z,"Net interchange: GWh"      ,y) = pDemandSupply(z,"Imports exchange: GWh",y) - pDemandSupply(z,"Exports exchange: GWh",y);
pDemandSupply(z,"Net interchange Ratio: GWh",y)$pDemandSupply(z,"Demand: GWh",y) = pDemandSupply(z,"Net interchange: GWh",y)/pDemandSupply(z,"Demand: GWh",y);



**************************************************************H2 model addition**************************************************************************
pDemandSupplyH2(z,"Electricity demand for H2 production: GWh",y)   =(sum( (h2zmap(hh,z), q,d,t), vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t))/1000)$pIncludeH2+1e-6; 
pDemandSupplyH2(z,"Total H2 Production: mmBTU",           y)        =sum( (h2zmap(hh,z), q,d,t), vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t)*pH2Data(hh,"HeatRate"))$pIncludeH2+1e-6;
pDemandSupplyH2(z,"External Demand of H2: mmBTU",         y)        =sum(q,pExternalH2(z,q,y)       )$pIncludeH2+1e-6;
pDemandSupplyH2(z,"Unmet External Demand of H2: mmBTU",   y)        =sum(q,vUnmetExternalH2.l(z,q,y))$pIncludeH2+1e-6;
pDemandSupplyH2(z,"H2 produced for power production: mmBTU",       y)        =sum(q,vFuelH2Quarter.l(z,q,y)  )$pIncludeH2+1e-6;

$if %DEBUG%==1 display  pDemandSupply;
*********************H2 model addition*****************************
$if %DEBUG%==1 display  pDemandSupplyH2;
*******************************************************************

* By Country
pDemandSupplyCountry(c,"Demand: GWh"            ,y) = sum(zcmap(z,c), pDemandSupply(z,"Demand: GWh"            ,y));
*********************************************************H2 model addition****************************************************************
pDemandSupplyCountry(c,"Electricity demand for H2 production: GWh",         y) =(sum((zcmap(z,c),h2zmap(hh,z),q,d,t),vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t) )/1e3)$pIncludeH2+0.000001;
pDemandSupplyCountry(c,"Total Demand (Including P to H2): GWh"      ,y) =(sum((zcmap(z,c),q,d,t), pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))/1e3+sum((zcmap(z,c),h2zmap(hh,z),q,d,t),vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t) )/1e3)$pIncludeH2+0.000001;

pDemandSupplyCountry(c,"Total production: GWh"  ,y) = sum(zcmap(z,c), pDemandSupply(z,"Total production: GWh"  ,y));
pDemandSupplyCountry(c,"Unmet demand: GWh"      ,y) = sum(zcmap(z,c), pDemandSupply(z,"Unmet demand: GWh"      ,y));
pDemandSupplyCountry(c,"Surplus generation: GWh",y) = sum(zcmap(z,c), pDemandSupply(z,"Surplus generation: GWh",y));

pDemandSupplyCountry(c,"Imports exchange: GWh"     ,y) = (sum((zcmap(z,c),sMapConnectedZonesDiffCountries(z2,z),q,d,t), vFlow.l(z2,z,q,d,t,y)*pHours(q,d,t)) + sum((zcmap(z,c),zext,q,d,t), vImportPrice.l(z,zext,q,d,t,y)*pHours(q,d,t)))/1e3;
pDemandSupplyCountry(c,"Exports exchange: GWh"     ,y) = (sum((zcmap(z,c),sMapConnectedZonesDiffCountries(z,z2),q,d,t), vFlow.l(z,z2,q,d,t,y)*pHours(q,d,t)) + sum((zcmap(z,c),zext,q,d,t), vExportPrice.l(z,zext,q,d,t,y)*pHours(q,d,t)))/1e3;
pDemandSupplyCountry(c,"Net interchange: GWh"      ,y) = pDemandSupplyCountry(c,"Imports exchange: GWh",y) - pDemandSupplyCountry(c,"Exports exchange: GWh",y);
pDemandSupplyCountry(c,"Net interchange Ratio: GWh",y)$pDemandSupplyCountry(c,"Demand: GWh",y) = pDemandSupplyCountry(c,"Net interchange: GWh",y)/pDemandSupplyCountry(c,"Demand: GWh",y);

****************************************************H2 model additions*******************************************************************
pDemandSupplyCountryH2(c,"Total electricity demand for H2 production: GWh",y)   =(sum( (zcmap(z,c),h2zmap(hh,z), q,d,t), vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t))/1000)$pIncludeH2+1e-6; 
pDemandSupplyCountryH2(c,"Total H2 Production: mmBTU",           y) =sum((zcmap(z,c),h2zmap(hh,z), q,d,t), vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t)*pH2Data(hh,"HeatRate"))$pIncludeH2+1e-6;
pDemandSupplyCountryH2(c,"External Demand of H2: mmBTU",         y) =sum((zcmap(z,c),q),pExternalH2(z,q,y)       )$pIncludeH2+1e-6;
pDemandSupplyCountryH2(c,"Unmet External Demand of H2: mmBTU",   y) =sum((zcmap(z,c),q),vUnmetExternalH2.l(z,q,y) )$pIncludeH2+1e-6;
pDemandSupplyCountryH2(c,"H2 for power production: mmBTU",       y) =sum((zcmap(z,c),q),vFuelH2Quarter.l(z,q,y)   )$pIncludeH2+1e-6;

*--- Summary of results
pSummary("NPV of system cost: $m"          ) = vNPVCost.l/1e6;
pSummary("Total Generation: GWh"           ) = sum((gzmap(g,z),y), pWeightYear(y)*pEnergyByPlant(z,g,y));
pSummary("Total Demand: GWh"               ) = sum((z,y), pWeightYear(y)*pDemandSupply(z,"Demand: GWh",y));
pSummary("Total Capacity Added: MW"        ) = sum((g,y), vBuild.l(g,y));
pSummary("Total Investment: $m"            ) = sum((g,y)$(ndc(g)), vBuild.l(g,y)*pGenData(g,"Capex")+ vBuildStor.l(g,y)*(pStorData(g,"CapexMWh")+pCSPData(g,"Storage","CapexMWh"))/1e3+vBuildTherm.l(g,y)*pCSPData(g,"Thermal Field","CapexMWh")) + sum((g,y)$(dc(g)), vBuild.l(g,y)*pGenData(g,"Capex")*pCapexTrajectories(g,y)+(vBuildStor.l(g,y)*(pStorData(g,"CapexMWh")+pCSPData(g,"Storage","CapexMWh"))/1e3+vBuildTherm.l(g,y)*pCSPData(g,"Thermal Field","CapexMWh"))*pCapexTrajectories(g,y));
pSummary("Total Trade: GWh"                ) = sum((z,y), pDemandSupply(z,"Imports exchange: GWh" ,y));
pSummary("Total USE: GWh"                  ) = sum((z,y), pWeightYear(y)*pDemandSupply(z,"Unmet demand: GWh",y));
pSummary("Carbon costs: $m"                ) = sum((z,y), pWeightYear(y)*pRR(y)*vYearlyCarbonCost.l(z,y))/1e6;
pSummary("Trade costs: $m"                 ) = sum((z,y), pWeightYear(y)*pRR(y)*vYearlyTradeCost.l( z,y))/1e6;
pSummary("VRE curtailment: $m"             ) = sum((z,y), pWeightYear(y)*pRR(y)*pVRECurtailment(z,y))/1e6;
pSummary("Total Emission: mt"              ) = sum((z,y), pWeightYear(y)*vZonalEmissions.l(z,y))/1e6;
pSummary("Climate backstop cost: $m"       ) = sum((c,y), pWeightYear(y)*pRR(y)*vYearlyCO2backstop.l(c,y)*pCostOfCO2backstop)/1e6+sum(y, pWeightYear(y)*pRR(y)*pCostOfCO2backstop*vYearlySysCO2backstop.l(y))/1e6;

pSummary("Sys Spin Reserve violation: $m"  ) = sum(y, pWeightYear(y)*pRR(y)*pUSRSysCosts(y))/1e6;
pSummary("Sys Plan Reserve violation: $m"  ) = sum(y, pWeightYear(y)*pRR(y)*pUPRSysCosts(y))/1e6;
pSummary("Zonal Spin Reserve violation: $m") = sum((c,y), pWeightYear(y)*pRR(y)*pUSRLocCosts(c,y))/1e6;
pSummary("Zonal Plan Reserve violation: $m") = sum((c,y), pWeightYear(y)*pRR(y)*pCountryPlanReserveCosts(c,y))/1e6;
pSummary("Excess Generation Costs: $m"     ) = sum((z,y), pWeightYear(y)*pRR(y)*eYearlySurplusCost.l(z,y))/1e6;                                      
pSummary("Spinning reserve costs: $m"      ) = sum((z,y), pWeightYear(y)*pRR(y)*pSpinResCosts(z,y))/1e6;

**************************************************************H2 model additions******************************************************************************************************
pSummary("Total Demand for H2 production: GWh") =(sum((h2zmap(hh,z),y,q,d,t),vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t) )/1e3)$pIncludeH2+0.000001;
pSummary("Total Demand including demand for H2 production: GWh") =(sum((h2zmap(hh,z),y,q,d,t),vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t) )/1e3+ sum((z,y), pWeightYear(y)*pDemandSupply(z,"Demand: GWh",y))  )$pIncludeH2+0.000001;
pSummary("Total H2 produced: mmBTU" )                   =sum((h2zmap(hh,z),y, q,d,t), vH2PwrIn.l(hh,q,d,t,y)*pHours(q,d,t)*pH2Data(hh,"HeatRate"))$pIncludeH2+1e-6;
pSummary("Total External Demand of H2: mmBTU")          =sum((z,q,y),pExternalH2(z,q,y)        )$pIncludeH2+1e-6;
pSummary("Total Unmet External Demand of H2: mmBTU")    =sum((z,q,y),vUnmetExternalH2.l(z,q,y) )$pIncludeH2+1e-6;
pSummary("Total H2 for power production: mmBTU")        =sum((z,q,y),vFuelH2Quarter.l(z,q,y)   )$pIncludeH2+1e-6;

*B. Interchange and Losses with INTERNAL zones

* By zone
pInterchange(sTopology(z,z2),y) = sum((q,d,t), vFlow.l(z,z2,q,d,t,y)*pHours(q,d,t))/1e3;
pInterconUtilization(sTopology(z,z2),y)$pInterchange(z,z2,y) = 1e3*pInterchange(z,z2,y)
                                                              /sum((q,d,t),(pTransferLimit(z,z2,q,y) + vAdditionalTransfer.l(z,z2,y)
                                                                                                     * max(pNewTransmission(z,z2,"CapacityPerLine"),
                                                                                                           pNewTransmission(z2,z,"CapacityPerLine"))*pAllowHighTransfer
                                                      )*pHours(q,d,t));
pLossesTransmission(z,y) = sum((sTopology(z,z2),q,d,t), vFlow.l(z2,z,q,d,t,y)*pLossFactor(z,z2,y)*pHours(q,d,t));

**By country
alias (zcmap, zcmap2);
pInterchangeCountry(c,c2,y)= sum((zcmap(z,c),zcmap2(z2,c2),sMapConnectedZonesDiffCountries(z2,z)), pInterchange(z,z2,y));
pLossesTransmissionCountry(c,y) = sum(zcmap(z,c), pLossesTransmission(z,y));


*C. Trade with EXTERNAL zones

* By Zone
set thrd / Imports, Exports /;
pHourlyTrade(z,y,q,d,"Imports",t) =   sum(zext,vImportPrice.l(z,zext,q,d,t,y)); 
pHourlyTrade(z,y,q,d,"Exports",t) =   sum(zext,vExportPrice.l(z,zext,q,d,t,y)); 
pYearlyTrade(z,thrd,y) = sum((q,d,t), pHourlyTrade(z,y,q,d,thrd,t)*pHours(q,d,t))/1e3;
pInterchangeExtExp(z,zext,y) = sum((q,d,t), vExportPrice.l(z,zext,q,d,t,y)*pHours(q,d,t))/1e3;
pInterconUtilizationExtExp(z,zext,y)$pInterchangeExtExp(z,zext,y) = 1e3*pInterchangeExtExp(z,zext,y) /sum((q,d,t),pExtTransferLimitOut(z,zext,q,y)*pHours(q,d,t));
pInterchangeExtImp(zext,z,y) = sum((q,d,t), vImportPrice.l(z,zext,q,d,t,y)*pHours(q,d,t))/1e3;
pInterconUtilizationExtImp(zext,z,y)$pInterchangeExtImp(zext,z,y) = 1e3*pInterchangeExtImp(zext,z,y) /sum((q,d,t),pExtTransferLimitIn(z,zext,q,y)*pHours(q,d,t));


* By Country
pHourlyTradeCountry(c,y,q,d,thrd,t) = sum(zcmap(z,c), pHourlyTrade(z,y,q,d,thrd,t));
pYearlyTradeCountry(c,thrd,y) = sum(zcmap(z,c), pYearlyTrade(z,thrd,y));

*---   Prices Costs -- Marginal costs

* By zone


pAveragePrice(z,y)$pDemandSupply(z,"Demand: GWh",y) = 1e-3*sum((q,d,t),pPrice(z,q,d,t,y)*pDemandData(z,q,d,y,t)*pHours(q,d,t)*pEnergyEfficiencyFactor(z,y))
                                                     /pDemandSupply(z,"Demand: GWh",y) ;

Parameter zzFlow(z,z2,y), zzFlowpH(z,z2,y);
zzFlow(z,z2,y) = sum(sFlow(z,z2,q,d,t,y),vFlow.l(z,z2,q,d,t,y));
zzFlowpH(z,z2,y) = sum(sFlow(z,z2,q,d,t,y),vFlow.l(z,z2,q,d,t,y)*pHours(q,d,t));

pAveragePriceExp(z,y) $(sum(Zd, zzFlowpH(z,Zd,y))  > 0) = (sum((sTopology(z,Zd),q,d,t),  pPrice(z,q,d,t,y) *vFlow.l(z,Zd,q,d,t,y)*pHours(q,d,t))
                                                           + sum((zext,q,d,t), vExportPrice.l(z,zext,q,d,t,y) * pHours(q,d,t))) 
                                                         /(sum(Zd, zzFlowpH(z,Zd,y))+ sum((zext,q,d,t), vExportPrice.l(z,zext,q,d,t,y)* pHours(q,d,t)));
pAveragePriceImp(z,y) $(sum(Zd, zzFlowpH(Zd,z,y))  > 0) = (sum((sTopology(Zd,z),q,d,t),  pPrice(Zd,q,d,t,y)*vFlow.l(Zd,z,q,d,t,y)*pHours(q,d,t))
                                                            + sum((zext,q,d,t), vImportPrice.l(z,zext,q,d,t,y)* pTradePrice(zext,q,d,y,t)  * pHours(q,d,t))) 
                                                         /(sum(Zd, zzFlowpH(Zd,z,y))+ sum((zext,q,d,t), vImportPrice.l(z,zext,q,d,t,y)  * pHours(q,d,t)));

pAveragePriceHub(Zt,y)$(sum(Zd, zzFlow(Zt,Zd,y))   > 0) = sum((sTopology(Zt,Zd),q,d,t), pPrice(Zt,q,d,t,y)*vFlow.l(Zt,Zd,q,d,t,y))
                                                         /sum(Zd, zzFlow(Zt,Zd,y));

* By Country
pAveragePriceCountry(c,y)$pDemandSupplyCountry(c,"Demand: GWh",y) = sum(zcmap(z,c), pAveragePrice(z,y)*pDemandSupply(z,"Demand: GWh",y))/pDemandSupplyCountry(c,"Demand: GWh",y);

Parameter cFlowpH(c,y);
cFlowpH(c,y) = sum((zcmap(z,c),sMapConnectedZonesDiffCountries(z,Zd)), zzFlowpH(z,Zd,y));
pAveragePriceExpCountry(c,y)$(cFlowpH(c,y) > 0) = sum((zcmap(z,c),sMapConnectedZonesDiffCountries(z,Zd),q,d,t), pPrice(z,q,d,t,y)* vFlow.l(z,Zd,q,d,t,y)*pHours(q,d,t))
                                                 /cFlowpH(c,y);
cFlowpH(c,y) = sum((zcmap(z,c),sMapConnectedZonesDiffCountries(Zd,z)), zzFlowpH(Zd,z,y));
pAveragePriceImpCountry(c,y)$(cFlowpH(c,y) > 0) = sum((zcmap(z,c),sMapConnectedZonesDiffCountries(Zd,z),q,d,t), pPrice(Zd,q,d,t,y)*vFlow.l(Zd,z,q,d,t,y)*pHours(q,d,t))
                                                 /cFlowpH(c,y);
*pAveragePriceHubCountry(Zt,y)$(sum((Zd,q,d,t),vFlow.l(Zt,Zd,q,d,t,y))>0) = sum((Zd,q,d,t)$sTopology(Zt,Zd), pPrice(Zt,q,d,t,y) * vFlow.l(Zt,Zd,q,d,t,y))/ sum((Zd,q,d,t),vFlow.l(Zt,Zd,q,d,t,y)) ;


*--- Capacity and Utilization Results

* By Zone
pPeakCapacity(z,"Available capacity: MW",y) = sum(gzmap(g,z), vCap.l(g,y));
pPeakCapacity(z,"Peak demand: MW"       ,y) = smax((q,d,t), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y));


*********************H2 model addition***************************************************************************************************
pPeakCapacity(z,"Peak demand including P to H2: MW"       ,y) = smax((h2zmap(hh,z),q,d,t), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y) + vH2PwrIn.l(hh,q,d,t,y))$pIncludeH2+1e-6;

pPeakCapacity(z,"New capacity: MW"      ,y) = sum(gzmap(g,z), vBuild.l(g,y));
pPeakCapacity(z,"Retired capacity: MW"  ,y) = sum(gzmap(g,z), vRetire.l(g,y));
pPeakCapacity(z,"Committed Total TX capacity: MW"  ,y) = sum((z2,q), pTransferLimit(z,z2,q,y)/card(q));

pCapacityByFuel(z,f,y)  = sum((gzmap(g,z),gprimf(g,f)), vCap.l(g,y));
pCapacityByTechandFuel(z,tech,f,y)  = sum((gzmap(g,z),gtechmap(g,tech),gprimf(g,f)), vCap.l(g,y));   
pNewCapacityFuel(z,f,y) = sum((gzmap(g,z),gprimf(g,f)), vBuild.l(g,y));

pNewCapacityTech(z,tech,y)       = sum((gzmap(g,z),gtechmap(g,tech),gprimf(g,f)), vBuild.l(g,y));
pRetirementsFuel(z,f,y)          = sum((gzmap(g,z),gprimf(g,f)),vRetire.l(g,y));





pUtilizationByFuel(z,f,y)$(pCapacityByFuel(z,f,y)) = pEnergyByFuel(z,f,y)/(pCapacityByFuel(z,f,y) * pTotalHoursperYear(y))*1000;
pUtilizationByTechandFuel(z,tech,f,y)$(pCapacityByTechandFuel(z,tech,f,y)) = pEnergyByTechandFuel(z,tech,f,y)/(pCapacityByTechandFuel(z,tech,f,y) * pTotalHoursperYear(y))*1000;

* By Country
pPeakCapacityCountry(c,"Available capacity: MW",y) = sum(zcmap(z,c), pPeakCapacity(z,"Available capacity: MW",y));
pPeakCapacityCountry(c,"Peak demand: MW"       ,y) = smax((q,d,t), sum(zcmap(z,c), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y)));
pPeakCapacityCountry(c,"New capacity: MW"      ,y) = sum(zcmap(z,c), pPeakCapacity(z,"New capacity: MW",y));
pPeakCapacityCountry(c,"Retired capacity: MW"  ,y) = sum(zcmap(z,c), pPeakCapacity(z,"Retired capacity: MW",y));
pPeakCapacityCountry(c,"Committed Total TX capacity: MW",y) = sum(zcmap(z,c), pPeakCapacity(z,"Committed Total TX capacity: MW",y));

pCapacityByFuelCountry(c,f,y)  = sum(zcmap(z,c), pCapacityByFuel(z,f,y));
pCapacityByTechandFuelCountry(c,tech,f,y)  = sum(zcmap(z,c), pCapacityByTechandFuel(z,tech,f,y));      
pNewCapacityFuelCountry(c,f,y) = sum(zcmap(z,c), pNewCapacityFuel(z,f,y));

pNewCapacityTechCountry(c,tech,y)=sum(zcmap(z,c), pNewCapacityTech(z,tech,y));

pCapacityPlanCountry(c,g,y)    = sum((zcmap(z,c),gzmap(g,z)), vCap.l(g,y));
pUtilizationByFuelCountry(c,f,y)$(pCapacityByFuelCountry(c,f,y)) = pEnergyByFuelCountry(c,f,y)/(pCapacityByFuelCountry(c,f,y)* pTotalHoursperYear(y))*1000;
pUtilizationByTechandFuelCountry(c,tech,f,y)$(pCapacityByTechandFuelCountry(c,tech,f,y)) = pEnergyByTechandFuelCountry(c,tech,f,y)/(pCapacityByTechandFuelCountry(c,tech,f,y) * pTotalHoursperYear(y))*1000;

pRetirementsCountry(c,y)          = sum((zcmap(z,c),f),pRetirementsFuel(z,f,y));
pRetirementsFuelCountry(c,f,y)      = sum(zcmap(z,c),pRetirementsFuel(z,f,y));


* By Plant
pCapacityPlan(zgmap(z,g),y)                 = vCap.l(g,y);
pRetirements(zgmap(z,g),y)                  = vRetire.l(g,y);
pPlantUtilization(zgmap(z,g),y)$vCap.l(g,y) = sum((gfmap(g,f),q,d,t), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t))/vCap.l(g,y)/8760;
pPlantUtilizationTech(z,tech,y)$sum((zgmap(z,g),gfmap(g,f),gtechmap(g,tech))$vCap.l(g,y),vCap.l(g,y)) = sum((zgmap(z,g),gfmap(g,f),gtechmap(g,tech),q,d,t)$vCap.l(g,y), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t))
                                                                                                        /sum((zgmap(z,g),gfmap(g,f),gtechmap(g,tech))$vCap.l(g,y),vCap.l(g,y))/8760;
pFuelUtilization(z,f,y)$sum((zgmap(z,g),gfmap(g,f))$vCap.l(g,y),vCap.l(g,y)) = sum((zgmap(z,g),gfmap(g,f),q,d,t)$vCap.l(g,y), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t))/sum((zgmap(z,g),gfmap(g,f))$vCap.l(g,y), vCap.l(g,y))/8760;      

*--- New TX Capacity by zone

pAdditionalCapacity(sTopology(z,z2),y) = vAdditionalTransfer.l(z,z2,y)*symmax(pNewTransmission,z,z2,"CapacityPerLine")*pAllowHighTransfer;
                                                                                                           

pAnnualTransmissionCapacity(sTopology(z,z2),y) = pAdditionalCapacity(z,z2,y) + smax(q, pTransferLimit(z,z2,q,y)) ; 

$if %DEBUG%==1 display pAdditionalCapacity;

*--- Reserve Results
* By Zone
pSpinningReserveCostsZone(zgmap(z,g),y)   = sum((q,d,t), vSpinningReserve.l(g,q,d,t,y)*pGenData(g,"ReserveCost")*pHours(q,d,t))/1e6 ;
pSpinningReserveByPlantZone(zgmap(z,g),y) = sum((q,d,t), vSpinningReserve.l(g,q,d,t,y)*pHours(q,d,t))/1e3 ;

pReserveMarginRes(z,"Peak demand: MW",y) = smax((q,d,t), pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y));
pReserveMarginRes(z,"TotalFirmCapacity",y) = sum(zgmap(z,g), pCapacityCredit(g,y)* vCap.l(g,y));                  
pReserveMarginRes(z,"ReserveMargin",y)$(pReserveMarginRes(z,"TotalFirmCapacity",y)) = pReserveMarginRes(z,"TotalFirmCapacity",y)/pReserveMarginRes(z,"Peak demand: MW",y)  ;             



* By Country
pSpinningReserveCostsCountry(c,g,y)  =  sum((zcmap(z,c),zgmap(z,g)), pSpinningReserveCostsZone(z,g,y));
pSpinningReserveByPlantCountry(c,g,y)=  sum((zcmap(z,c),zgmap(z,g)), pSpinningReserveByPlantZone(z,g,y));


pReserveMarginResCountry(c,"Peak demand: MW",y) =   pPeakCapacityCountry(c,"Peak demand: MW"       ,y);
pReserveMarginResCountry(c,"TotalFirmCapacity",y) = sum(zcmap(z,c), pReserveMarginRes(z,"TotalFirmCapacity",y));
pReserveMarginResCountry(c,"ReserveMargin",y)$(pReserveMarginResCountry(c,"TotalFirmCapacity",y)) = pReserveMarginResCountry(c,"TotalFirmCapacity",y)/ pReserveMarginResCountry(c,"Peak demand: MW",y)  ;   




*--- Emission Results
* By Zone
pEmissions(z,y)                                                     =  vZonalEmissions.l(z,y)/1e6 ;
pEmissionsIntensity(z,y)$pDemandSupply(z,"Total production: GWh",y) = 1e-3*vZonalEmissions.l(z,y)/pDemandSupply(z,"Total production: GWh",y);

pEmissionsCountry1(c,"Emissions: mm tCO2eq",y)                               = sum(zcmap(z,c), pEmissions(z,y)  );
pEmissionsCountry1(c,"CO2backstopEmissions: mm tCO2eq",y)                    = vYearlyCO2backstop.l(c,y)/1e6;
pEmissionsIntensityCountry(c,y)$pDemandSupplyCountry(c,"Total production: GWh",y) = 1e-3*sum(zcmap(z,c), vZonalEmissions.l(z,y))/pDemandSupplyCountry(c,"Total production: GWh",y);

*--- Emission marginal costs
 
pEmissionMarginalCosts(y) $(pWeightYear(y))                  = -eTotalEmissionsConstraint.M(y)/pRR(y)/pWeightYear(y);
pEmissionMarginalCostsCountry(c,y) $(pWeightYear(y))         = -eEmissionsCountry.M(c,y)/pRR(y)/pWeightYear(y);  


*--- Solar Energy (zonal results)
set PVtech(tech) / PV, PVwSTO /;
Parameter zSolar(z,y), zSolarpH(z,y);
pSolarEnergy(z,q,d,t,y) = sum((gzmap(g,z),gtechmap(g,PVtech),gfmap(g,f)), vPwrOut.l(g,f,q,d,t,y));

zSolar(z,y) = sum((q,d,t), pSolarEnergy(z,q,d,t,y));
pSolarValue(z,y)$(zSolar(z,y) > 0) = sum((q,d,t), pPrice(z,q,d,t,y)*pSolarEnergy(z,q,d,t,y))/zSolar(z,y);

zSolarpH(z,y) = sum((q,d,t), pSolarEnergy(z,q,d,t,y)*pHours(q,d,t));
pSolarCost(z,y)$(zSolarpH(z,y) > 0) = (sum((gzmap(ng,z),gtechmap(ng,PVtech)), pCRF(ng)*vCap.l(ng,y)*pGenData(ng,"Capex")*pCapexTrajectories(ng,y))*1e6)/zSolarpH(z,y);

*--- Dispatch Results

pPlantDispatch(z,y,q,d,g,t,"Generation")$zgmap(z,g) = sum(gfmap(g,f), vPwrOut.l(g,f,q,d,t,y));
pPlantDispatch(z,y,q,d,g,t,"Reserve")$zgmap(z,g) = vSpinningReserve.l(g,q,d,t,y);


pDispatch(z,y,q,d,"Generation"    ,t) = sum(zgmap(z,g), pPlantDispatch(z,y,q,d,g,t,"Generation")); 
pDispatch(z,y,q,d,"Imports"       ,t) =      sum(sTopology(z,z2), vFlow.l(z2,z,q,d,t,y)) + sum(zext,vImportPrice.l(z,zext,q,d,t,y));
pDispatch(z,y,q,d,"Exports"       ,t) = 0 - (sum(sTopology(z,z2), vFlow.l(z,z2,q,d,t,y)) + sum(zext,vExportPrice.l(z,zext,q,d,t,y)));
pDispatch(z,y,q,d,"Unmet demand"  ,t) = vUSE.l(z,q,d,t,y);
pDispatch(z,y,q,d,"Storage Charge",t) = 0 - sum(zgmap(z,st), vStorInj.l(st,q,d,t,y));
pDispatch(z,y,q,d,"Demand"        ,t) = pDemandData(z,q,d,y,t)*pEnergyEfficiencyFactor(z,y);

*--- LCOE and Average Cost Results

pDenom(z,g,y) = 1;
pDenom2(z,y)  = 1;
pDenom(z,g,y)$pEnergyByPlant(z,g,y) = pEnergyByPlant(z,g,y)*1e3;
pDenom2(z,y)$pDemandSupply(z,"Total production: GWh",y) = pDemandSupply(z,"Total production: GWh",y)*1e3;
pDenom3(c,y)  = sum(zcmap(z,c), pDenom2(z,y));


pSystemAverageCost(y)$sum(z, pDenom2(z,y)) =
                        (sum((z,zext,q,d,t), (vImportPrice.l(z,zext,q,d,t,y)-vExportPrice.l(z,zext,q,d,t,y))*pTradePrice(zext,q,d,y,t)*pHours(q,d,t))
*                 max(1,Sum((z,q,d,t),vImportPrice.l(z,q,d,t,y)*pHours(q,d,t))));
                       + sum(ndc, pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex"))*1e6
                       + sum(ndc$(not cs(ndc)), pCRFsst(ndc)*vCapStor.l(ndc,y)*pStorData(ndc,"CapexMWh"))*1e3
                       + sum(ndc$(not st(ndc)), pCRFcst(ndc)*vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","CapexMWh"))*1e3
                       + sum(ndc, pCRFcth(ndc)*vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal Field","CapexMWh"))*1e6
                       + sum(dc, vAnnCapex.l(dc,y))
                       + sum(z, pFOM(z,y) + pVOM(z,y) + pFuelCostsZone(z,y) + pSpinResCosts(z,y))
                       )/sum(z, sum((zext,q,d,t), (vImportPrice.l(z,zext,q,d,t,y) - vExportPrice.l(z,zext,q,d,t,y))*pHours(q,d,t)) + pDenom2(z,y));



pPlantAnnualLCOE(zgmap(z,ndc),y)$pDenom(z,ndc,y) = (pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex")*1e6
                                                  + vCap.l(ndc,y)*pGenData(ndc,"FOMperMW")
                                                  + vCapStor.l(ndc,y)*pStorData(ndc,"FixedOMMWh")
                                                  + vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","FixedOMMWh")
                                                  + vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal field","FixedOMMWh")
                                                  + sum((gfmap(ndc,f),q,d,t), pVarCost(ndc,f,y)*vPwrOut.l(ndc,f,q,d,t,y)*pHours(q,d,t))
*                                                 + sum((q,d,t)$(gfmap(g,f)),  vSpinningReserve(g,q,d,t,y)*pGenData(g,"ReserveCost")*pHours(q,d,t))
                                                   )/pDenom(z,ndc,y);

pPlantAnnualLCOE(zgmap(z,dc),y)$pDenom(z,dc,y) =  (vAnnCapex.l(dc,y)
                                                 + vCap.l(dc,y)*pGenData(dc,"FOMperMW")
                                                 + vCapStor.l(dc,y)*pStorData(dc,"FixedOMMWh")
                                                 + vCapStor.l(dc,y)*pCSPData(dc,"Storage","FixedOMMWh")
                                                 + vCapTherm.l(dc,y)*pCSPData(dc,"Thermal field","FixedOMMWh")
                                                 + sum((gfmap(dc,f),q,d,t), pVarCost(dc,f,y)*vPwrOut.l(dc,f,q,d,t,y)*pHours(q,d,t))
*                                                + sum((q,d,t)$(gfmap(g,f)),  vSpinningReserve(g,q,d,t,y)*pGenData(g,"ReserveCost")*pHours(q,d,t))
                                                  )/pDenom(z,dc,y);


parameter zctmp(z,y);
zctmp(z,y) = sum((zext,q,d,t), (vImportPrice.l(z,zext,q,d,t,y) - vExportPrice.l(z,zext,q,d,t,y))*pTradePrice(zext,q,d,y,t)*pHours(q,d,t))
           + sum((sTopology(Zd,z),q,d,t), pPrice(Zd,q,d,t,y)*vFlow.l(Zd,z,q,d,t,y)*pHours(q,d,t))
           + sum(zgmap(z,dc), vAnnCapex.l(dc,y))
           + sum(zgmap(z,ndc), pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex"))*1e6
           + sum(zgmap(z,ndc)$(not cs(ndc)), pCRFsst(ndc)*vCapStor.l(ndc,y)*pStorData(ndc,"CapexMWh"))*1e3
           + sum(zgmap(z,ndc)$(not st(ndc)), pCRFcst(ndc)*vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","CapexMWh"))*1e3
           + sum(zgmap(z,ndc), pCRFcth(ndc)*vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal Field","CapexMWh"))*1e6
           + pFOM(z,y) + pVOM(z,y)+ pFuelCostsZone(z,y) + pSpinResCosts(z,y);

pZonalAverageCost(z,y)$pDenom2(z,y) = zctmp(z,y)/(sum((zext,q,d,t), (vImportPrice.l(z,zext,q,d,t,y) - vExportPrice.l(z,zext,q,d,t,y))*pHours(q,d,t))
                                                + sum((sTopology(Zd,z),q,d,t), vFlow.l(Zd,z,q,d,t,y)*pHours(q,d,t)) + pDenom2(z,y));

pCountryAverageCost(c,y)$pDenom3(c,y) = sum(zcmap(z,c), zctmp(z,y))/(sum((zcmap(z,c),zext,q,d,t), (vImportPrice.l(z,zext,q,d,t,y) - vExportPrice.l(z,zext,q,d,t,y))*pHours(q,d,t))
                                                                   + pDenom3(c,y)
                                                                   + sum((zcmap(z,c),sMapConnectedZonesDiffCountries(Zd,z),q,d,t), vFlow.l(Zd,z,q,d,t,y)*pHours(q,d,t)));

option clear=zctmp;

zctmp(z,y) = sum(zgmap(z,dc), vAnnCapex.l(dc,y))
           + sum(zgmap(z,ndc), pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex"))*1e6
           + sum(zgmap(z,ndc)$(not cs(ndc)), pCRFsst(ndc)*vCapStor.l(ndc,y)*pStorData(ndc,"CapexMWh"))*1e3
           + sum(zgmap(z,ndc)$(not st(ndc)), pCRFcst(ndc)*vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","CapexMWh"))*1e3
           + sum(zgmap(z,ndc), pCRFcth(ndc)*vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal Field","CapexMWh"))*1e6
           + pFOM(z,y) + pVOM(z,y) + pFuelcostsZone(z,y) + pSpinResCosts(z,y);

pZonalAverageGenCost(z,y)$pDenom2(z,y) = zctmp(z,y)/pDenom2(z,y);
pCountryAverageGenCost(c,y)$pDenom3(c,y) = sum(zcmap(z,c),zctmp(z,y))/pDenom3(c,y);



*CostSummarybyPlant

pCostsbyPlant(zgmap(z,ndc), "Annuity Cost in MUSD",y)= (pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex")*1e6
                                                 + pCRF(ndc)*vCapStor.l(ndc,y)*pStorData(ndc,"CapexMWh")*1e3
                                                 + pCRFcst(ndc)*vCapStor.l(ndc,y)*pCSPData(ndc,"Storage","CapexMWh")*1e3
                                                 + pCRFcth(ndc)*vCapTherm.l(ndc,y)*pCSPData(ndc,"Thermal Field","CapexMWh")*1e6)/1e6;
                                                 
pCostsbyPlant(zgmap(z,dc), "Annuity Cost in MUSD",y)=   vAnnCapex.l(dc,y)/1e6;


pCostsbyPlant(zgmap(z,g),"Fixed Cost in MUSD",y)=       (  vCap.l(g,y)*pGenData(g,"FOMperMW")
                                                 + vCapStor.l(g,y)*pStorData(g,"FixedOMMWh")
                                                 + vCapStor.l(g,y)*pCSPData(g,"Storage","FixedOMMWh")
                                                 + vCapTherm.l(g,y)*pCSPData(g,"Thermal field","FixedOMMWh"))/1e6;

pCostsbyPlant(zgmap(z,g),"Variable Cost in MUSD",y)= sum((gfmap(g,f),q,d,t), pVarCost(g,f,y)*vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t))/1e6;
pCostsbyPlant(zgmap(z,g),"Spinning Reserve Cost in MUSD",y)= pSpinningReserveCostsZone(z,g,y);
pCostsbyPlant(zgmap(z,g),"Capacity Factor",y)= pPlantUtilization(z,g,y);




*---  CSP and storage
pCSPBalance(y, g,q,d,"Thermal output",t) = vThermalOut.l(g,q,d,t,y);
pCSPBalance(y,cs,q,d,"Storage Input" ,t) = vStorInj.l(cs,q,d,t,y);
pCSPBalance(y,cs,q,d,"Storage Output",t) = vStorOut.l(cs,q,d,t,y);
pCSPBalance(y,cs,q,d,"Power Output"  ,t) = sum(gfmap(cs,f), vPwrOut.l(cs,f,q,d,t,y));

pCSPComponents(g, "Thermal Field: WM" ,y) = vCapTherm.l(g,y);
pCSPComponents(cs,"Storage: MWh"      ,y) = vCapStor.l(cs,y);
pCSPComponents(cs,"Power Block: MW"   ,y) = vCap.l(cs,y);
pCSPComponents(g, "Solar Multiple"    ,y) = vCapTherm.l(g,y)/max(vCap.l(g,y),1);
pCSPComponents(cs,"Storage Hours: hrs",y) = vCapStor.l(cs,y)/max(vCap.l(cs,y),1);

pPVwSTOBalance(y,q,d,so, "PV output"     ,t) = sum(gfmap(so,f), vPwrOut.l(so,f,q,d,t,y));
pPVwSTOBalance(y,q,d,stp,"Storage Input" ,t) = vStorInj.l(stp,q,d,t,y);
pPVwSTOBalance(y,q,d,stp,"Storage output",t) = sum(gfmap(stp,f), vPwrOut.l(stp,f,q,d,t,y));
pPVwSTOBalance(y,q,d,stp,"Storage Losses",t) = (1-pStorData(stp,"Efficiency"))*vStorInj.l(stp,q,d,t,y);
pPVwSTOBalance(y,q,d,stp,"Storage level" ,t) = vStorage.l(stp,q,d,t,y);

pPVwSTOComponents(so, "PV Plants"           ,y) = vCap.l(so,y);
pPVwSTOComponents(stp,"Storage Capacity MW" ,y) = vCap.l(stp,y);
pPVwSTOComponents(stp,"Storage Capacity MWh",y) = vCapStor.l(stp,y);
pPVwSTOComponents(stp,"Storage Hours"       ,y) = vCapStor.l(stp,y)/max(vCap.l(stp,y),1);

pStorageBalance(y,stg,q,d,"Storage Input"     ,t) = vStorInj.l(stg,q,d,t,y);
pStorageBalance(y,stg,q,d,"Storage Output"    ,t) = sum(gfmap(stg,f), vPwrOut.l(stg,f,q,d,t,y));
pStorageBalance(y,stg,q,d,"Storage Losses"    ,t) = (1-pStorData(stg,"Efficiency"))*vStorInj.l(stg,q,d,t,y);
pStorageBalance(y,stg,q,d,"Net Storage Change",t) = vStorNet.l(stg,q,d,t,y);
pStorageBalance(y,stg,q,d,"Storage Level"     ,t) = vStorage.l(stg,q,d,t,y);

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
pDemandSupplySeason(z,"Total production: GWh"  ,y,q) = sum((zgmap(z,g),gfmap(g,f),d,t), vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t))/1e3;
pDemandSupplySeason(z,"Unmet demand: GWh"      ,y,q) = sum((d,t), vUSE.l(z,q,d,t,y)*pHours(q,d,t))/1e3;
pDemandSupplySeason(z,"Surplus generation: GWh",y,q) = sum((d,t), vSurplus.l(z,q,d,t,y)*pHours(q,d,t))/1e3;

pDemandSupplySeason(z,"Imports exchange: GWh",y,q) = (sum((sTopology(z,z2),d,t), vFlow.l(z2,z,q,d,t,y)*pHours(q,d,t)) + sum((zext,d,t), vImportPrice.l(z,zext,q,d,t,y)*pHours(q,d,t)))/1e3;
pDemandSupplySeason(z,"Exports exchange: GWh",y,q) = (sum((sTopology(z,z2),d,t), vFlow.l(z,z2,q,d,t,y)*pHours(q,d,t)) + sum((zext,d,t), vExportPrice.l(z,zext,q,d,t,y)*pHours(q,d,t)))/1e3;

* B. Seasonal Energy by Plant
pEnergyByPlantSeason(zgmap(z,g),y,q) = sum((gfmap(g,f),d,t),vPwrOut.l(g,f,q,d,t,y)*pHours(q,d,t))/1e3;


* C.Interchange

* By Zone
pInterchangeSeason(sTopology(z,z2),y,q) = sum((d,t), vFlow.l(z,z2,q,d,t,y)*pHours(q,d,t))/1e3;
pSeasonTrade(z,"External Zone Imports",y,q) = sum((zext,d,t), vImportPrice.l(z,zext,q,d,t,y)*pHours(q,d,t))/1e3;
pSeasonTrade(z,"External Zone Exports",y,q) = sum((zext,d,t), vExportPrice.l(z,zext,q,d,t,y)*pHours(q,d,t))/1e3;

* By Country
pInterchangeSeasonCountry(c,c2,y,q) = sum((zcmap(z,c),zcmap2(z2,c2),sMapConnectedZonesDiffCountries(z,z2)), pInterchangeSeason(z,z2,y,q));
pSeasonTradeCountry(c,"External Zone Imports",y,q) = sum(zcmap(z,c), pSeasonTrade(z,"External Zone Imports",y,q));
pSeasonTradeCountry(c,"External Zone Exports",y,q) = sum(zcmap(z,c), pSeasonTrade(z,"External Zone Exports",y,q));

*******************************H2 model additions****************************************************

pCapacityPlanH2(zh2map(z,hh),y)                 = vCapH2.l(hh,y)$pIncludeH2+1e-6;
****************************************************************************************************

*--- END RESULTS

pZonesperCountry(c) = sum(zcmap(z,c), 1);
MaxZonesperCountry = smax(c,pZonesperCountry(c));

$ifthen.excelreport %DOEXCELREPORT%==1
execute_unload 'epmresults',     pSettings, pSummary, pSystemAverageCost, pZonalAverageCost,pCountryAverageCost
                                 pAveragePrice, pAveragePriceExp, pAveragePriceImp, pPrice, pAveragePriceHub,
                                 pAveragePriceCountry, pAveragePriceExpCountry, pAveragePriceImpCountry,
                                 pCostSummary, pCostSummaryCountry, pCostSummaryWeighted, pCostSummaryWeightedCountry,
                                 pCostSummaryWeightedAverageCountry, pCongestionRevenues, pFuelCosts,pFuelCostsCountry,pFuelConsumption,pFuelConsumptionCountry
                                 pEnergyByPlant, pEnergyByFuel,pEnergyByFuelCountry, pEnergyByTechandFuel,pEnergyByTechandFuelCountry,pEnergyMix,
                                 pDemandSupply,  pDemandSupplyCountry, pVarCost,
                                 pInterchange, pInterchangeExtExp, pInterchangeExtImp, pInterconUtilization, pInterconUtilizationExtExp, pInterconUtilizationExtImp, pLossesTransmission, pInterchangeCountry,pLossesTransmissionCountry,
                                 pYearlyTrade,pHourlyTrade,pYearlyTradeCountry,pHourlyTradeCountry,
                                 pPeakCapacity, pCapacityByFuel, pCapacityByTechandFuel, pNewCapacityFuel, pCapacityPlan,pAdditionalCapacity, pAnnualTransmissionCapacity, pRetirements,  
                                 pPeakCapacityCountry, pCapacityByFuelCountry, pCapacityByTechandFuelCountry, pNewCapacityFuelCountry,pCapacityPlanCountry,
                                 pNewCapacityTech, pNewCapacityTechCountry,
                                 pReserveMarginRes, pReserveMarginResCountry,
                                 pCostsbyPlant,pRetirementsFuel,pRetirementsCountry,pRetirementsFuelCountry,
                                 pAdditionalCapacityCountry,
                                 pUtilizationByFuel,pUtilizationByTechandFuel,pUtilizationByFuelCountry,pUtilizationByTechandFuelCountry,
                                 pSpinningReserveByPlantZone, pSpinningReserveCostsZone,pSpinningReserveByPlantCountry, pSpinningReserveCostsCountry,pCapacityCredit,
                                 pEmissions, pEmissionsIntensity,pEmissionsCountry1, pEmissionsIntensityCountry,pEmissionMarginalCosts,pEmissionMarginalCostsCountry,  
                                 pPlantDispatch, pDispatch, pPlantUtilization, pPlantAnnualLCOE,
                                 pPlantUtilizationTech,pFuelUtilization,
                                 pCSPBalance, pCSPComponents,pPVwSTOBalance,pPVwSTOComponents,pStorageBalance,pStorageComponents
                                 pSolarValue, pSolarCost,
                                 pSolverParameters,pDemandSupplySeason,pEnergyByPlantSeason,
                                 pInterchangeSeason,pSeasonTrade,pInterchangeSeasonCountry,pSeasonTradeCountry,
                                pDemandSupplyH2,pDemandSupplyCountryH2, pCapacityPlanH2

;



file fgdxxrw / 'gdxxrw.out' /;
file fxlsxrep / 'xlsxReport.cmd' /;
singleton set execPlatform(*) / '' /;
put_utility 'assignText' / 'execPlatform' / system.platform;
scalar isWindows; loop(execPlatform, isWindows=ord(execPlatform.te,1)=ord('W',1));
put$(not isWindows) fxlsxrep 'rem Run to create Excel files' / 'cd "%gams.workdir%"';
$setNames "%gams.input%" fp fn fe

*the below lines allow to save a dedicated result file by scenario
*Parameter XlsTalkResult;
*execute 'xlstalk -M -V    data%system.dirsep%input%system.dirsep%EPMRESULTS_%RunScenario%_%mydate%.xlsb';
*         XlsTalkResult = errorlevel;
*         if(              XlsTalkResult > 0,
*                          execute 'xlstalk -S -V    data%system.dirsep%input%system.dirsep%EPMRESULTS_%RunScenario%_%mydate%.xlsb';
*         );
*execute "copy data%system.dirsep%input%system.dirsep%EPMRESULTS.xlsb  data%system.dirsep%input%system.dirsep%EPMRESULTS_%RunScenario%_%mydate%.xlsb /y"



   put_utility fgdxxrw 'ren' / 'WriteZonalandCountry.txt';
   
$  onPut 
   par=pSettings                           rng=Settings_raw!A6                     rdim=1
   par=pSummary                           rng=Summary!A6                          rdim=1
   par=pSystemAverageCost                 rng=SysAverageCost!A5
   par=pZonalAverageCost                  rng=ZonalAverageCost!A5
   par=pAveragePrice                      rng=ZonalAverageCost!U5
   par=pCountryAverageCost                rng=CountryAverageCost!A5
   par=pAveragePriceCountry               rng=CountryMarginalCost!A5
   par=pPrice                             rng=MarginalCost!A5
   par=pAveragePriceExp                   rng=ExchangeCost!A5
   par=pAveragePriceImp                   rng=ExchangeCost!Z5
   par=pAveragePriceExpCountry            rng=ExchangeCostCountry!A5
   par=pAveragePriceImpCountry            rng=ExchangeCostCountry!Z5
   par=pAveragePriceHub                   rng=TradeHubPrice!A5
   par=pCostSummary                       rng=CostSummary!A5
   par=pCostSummaryCountry                rng=CostSummaryCountry!A5
   par=pCostSummaryWeighted               rng=CostWeighted!A5
   par=pCostSummaryWeightedCountry        rng=CostWeigthedCountry!A5
   par=pCostSummaryWeightedAverageCountry rng=CostAverage!A5
   par=pFuelCosts                         rng=FuelCosts!A5
   par=pFuelCostsCountry                  rng=FuelCostsCountry!A5
   par=pFuelConsumption                   rng=FuelConsumption!A5
   par=pFuelConsumptionCountry            rng=FuelConsumptionCountry!A5
   par=pDemandSupply                      rng=DemandSupply!A5
   par=pDemandSupplyCountry               rng=DemandSupplyCountry!A5
   par=pEnergyByFuel                      rng=EnergyByFuel!A5
   par=pEnergyByFuelCountry               rng=EnergyByFuelCountry!A5
   par=pEnergyByTechandFuelCountry        rng=EnergyByTechandFuelCountry!A5   
   par=pEnergyMix                         rng=EnergyMix!A5
   par=pEnergyByPlant                     rng=EnergyByPlant!A5
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
   par=pPeakCapacity                      rng=PeakAndCapacity!A5
   par=pPeakCapacityCountry               rng=PeakAndCapacityCountry!A5
   par=pCapacityByFuel                    rng=CapacityByFuel!A5
   par=pCapacityByFuelCountry             rng=CapacityByFuelCountry!A5
   par=pCapacityByTechandFuel             rng=CapacityByTechandFuel!A5
   par=pCapacityByTechandFuelCountry      rng=CapacityByTechandFuelCountry!A5
   par=pNewCapacityFuel                   rng=NewCapacityFuel!A5
   par=pNewCapacityFuelCountry            rng=NewCapacityFuelCountry!A5
   par=pUtilizationByFuel                 rng=UtilizationbyFuel!A5
   par=pUtilizationByTechandFuel          rng=UtilizationbyTechandFuel!A5
   par=pUtilizationByFuelCountry          rng=UtilizationbyFuelCountry!A5
   par=pUtilizationByTechandFuelCountry   rng=UtilizationbyTechandFuelCountry!A5
   par=pRetirements                       rng=Retirements!A5
   par=pCapacityPlan                      rng=CapacityPlan!A5
   par=pCapacityPlanCountry               rng=CapacityPlanCountry!A5
   par=pSpinningReserveCostsZone          rng=ReserveCosts!A5
   par=pSpinningReserveCostsCountry       rng=ReserveCostsCountry!A5
   par=pSpinningReserveByPlantZone        rng=ReserveByPlant!A5
   par=pSpinningReserveByPlantCountry     rng=ReserveByPlantCountry!A5
   par=pEmissions                         rng=Emissions!A5
   par=pEmissionsCountry1                  rng=EmissionsCountry!A5
   par=pEmissionsIntensity                rng=EmissionsIntensity!A5
   par=pEmissionsIntensityCountry         rng=EmissionsIntensityCountry!A5
   par=pEmissionMarginalCosts             rng=EmissionMarginalCosts!A5
   par=pEmissionMarginalCostsCountry      rng=EmissionMarginalCostsCountry!A5 
   par=pPlantDispatch                     rng=PlantDispatch!A5
   par=pDispatch                          rng=Dispatch!A5
   par=pPlantUtilization                  rng=PlantUtilization!A5
   par=pPlantUtilizationTech              rng=PlantUtilizationTech!A5
   par=pPlantAnnualLCOE                   rng=PlantAnnualLCOE!A5
   par=pCSPBalance                        rng=CSPBalance!A5
   par=pCSPComponents                     rng=CSPComponents!A5
   par=pPVwSTOBalance                     rng=PVSTOBalance!A5
   par=pPVwSTOComponents                  rng=PVSTOComponents!A5
   par=pStorageBalance                    rng=StorageBalance!A5
   par=pStorageComponents                 rng=StorageComponents!A5
   par=pSolarValue                        rng=SolarValue!A5
   par=pSolarCost                         rng=SolarCost!A5
   par=pCapacityCredit                    rng=CapacityCreditsPlants!A5   
   par=pSolverParameters                  rng=SolverParameters!A5                 rdim=1
   
   par=pNewCapacityTech                   rng=NewCapacityTech!A5
   par=pNewCapacityTechCountry            rng=NewCapacityTechCountry!A5
   par=pReserveMarginRes                  rng=ReserveMargin!A5
   par=pReserveMarginResCountry           rng=ReserveMarginCountry!A5
   par=pCostsbyPlant                      rng=CostsbyPlant!A5
   par=pRetirementsFuel                   rng=RetirementsFuel!A5
   par=pRetirementsCountry                rng=RetirementsCountry!A5
   par=pRetirementsFuelCountry            rng=RetirementsFuelCountry!A5
   par=pAdditionalCapacityCountry         rng=AddedTransCapCountry!A5
   
***************************H2 model additions************************************************************
   par=pDemandSupplyH2                      rng=DemandSupplyH2!A5
   par=pCapacityPlanH2                      rng=CapacityPlanH2!A5
   par=pDemandSupplyCountryH2               rng=DemandSupplyCountryH2!A5
********************************************************************************************************   
$  offPut
   putclose;
   if (isWindows,
      execute.checkErrorLevel 'gdxxrw epmresults.gdx output="%XLS_OUTPUT%" @WriteZonalandCountry.txt';
   else
      put fxlsxrep / 'gdxxrw "%fn%\epmresults.gdx" output="%XLS_OUTPUT%" @"%fn%\WriteZonalandCountry.txt"';
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

Capacity("Capacity MW",f,y) =  Sum(gprimf(g,f), vCap.l(g,y));

CapacitySummary(" ","Peak"               ,y) = smax(c, pPeakCapacityCountry(c,"Peak demand: MW",y));
CapacitySummary(" ","New capacity MW"    ,y) = sum(g, vBuild.l(g,y));
CapacitySummary(" ","Retired capacity MW",y) = sum(g, vRetire.l(g,y));

Energy("Energy GWh",f,y) = sum(c, pEnergyByFuelCountry(c,f,y));

Costs("Costs","Capex: $m"       ,y)= sum(z, pCapex(z,y))/1e6;
Costs("Costs","Fixed O&M: $m"   ,y)= sum(z, pFOM(z,y))/1e6;
Costs("Costs","Variable O&M: $m",y)= sum(z, pVOM(z,y))/1e6;

Costs("Costs","Total fuel: $m"  ,y)= sum(z, pFuelCostsZone(z,y))/1e6;

Costs("Costs","Net Import: $m"  ,y)= sum(z, pImportCostsExternal(z,y) - pExportRevenuesExternal(z,y))/1e6;


Utilization("Utilization",f,y)$Capacity("Capacity MW",f,y) = Energy("Energy GWh",f,y)/Capacity("Capacity MW",f,y);

CapacityType("Capacity by Type MW",f,tech,y) = sum((gprimf(g,f),gtechmap(g,tech)), vCap.l(g,y));


AverageCost(" ","Average cost $/MWh",y) = (sum(ndc, pCRF(ndc)*vCap.l(ndc,y)*pGenData(ndc,"Capex"))*1e6
                                         + sum(dc, vAnnCapex.l(dc,y))
                                         + sum(z, pFOM(z,y) + pVOM(z,y) + pSpinResCosts(z,y) + pFuelCostsZone(z,y))
                                          )/sum(z, sum((zext,q,d,t), (vImportPrice.l(z,zext,q,d,t,y)-vExportPrice.l(z,zext,q,d,t,y))*pHours(q,d,t))
                                                 + pDenom2(z,y));
                                                 
$ifthen.excelreport %DOEXCELREPORT%==1
if (pSystemResultReporting > 0,
   put_utility fgdxxrw 'ren' / 'SystemResults.txt';
$  onPut
   par=Capacity        rng=Summary!A5
   par=CapacitySummary rng=Summary!A15
   par=Energy          rng=Summary!A25
   par=Costs           rng=Summary!A35
   par=Utilization     rng=Summary!A45
   par=AverageCost     rng=Summary!A55
   par=CapacityType    rng=Summary!A65
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
put_utility fgdxxrw 'ren' / 'gdxxrw.out'
$endif.excelreport
