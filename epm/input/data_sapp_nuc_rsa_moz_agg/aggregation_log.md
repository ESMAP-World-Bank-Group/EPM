# Aggregation log

Built from `data_sapp_nuc_rsa_moz` with `tools/aggregate_zones.py`: Angola (Angola_East, Angola_South, Angola_North, Angola_Central); Mozambique (Mozambique_Central, Mozambique_South, Mozambique_North); Zimbabwe (Zimbabwe_East, Zimbabwe_West) merged into one zone each. Other zones (Mozal included) are unchanged.

- `pHours.csv` (pHours) taken from `pre-analysis/representative_days/output/sapp_2024/epm_16d/pHours.csv` (--override)
- `load/pDemandProfile.csv` (pDemandProfile) taken from `pre-analysis/representative_days/output/sapp_2024/epm_16d/pDemandProfile.csv` (--override)
- `supply/pVREgenProfile.csv` (pVREgenProfile) taken from `pre-analysis/representative_days/output/sapp_2024/epm_16d/pVREgenProfile.csv` (--override)
- `supply/pVREProfile.csv` (pVREProfile) taken from `pre-analysis/representative_days/output/sapp_2024/epm_16d/pVREProfile.csv` (--override)
- `trade/pTradePrice.csv` (pTradePrice) taken from `pre-analysis/representative_days/output/sapp_2024/epm_16d/pTradePrice.csv` (--override)
- `supply/pGenDataInput_SAPP_v5.csv`: deleted 40 row(s) with a country-level zone name: ExistingSmallHydro2012_HY_AGO, LuandaOCGT_TH_AGO, MCT_Morro_Bento_TH_AGO, Caraculo_OffGrid_PV_AGO, RochaPinto_TH_AGO, New_CSP_AGO, ZhongJinHeli_TH_ZWE, ZhongXinIPP_Phase2_TH_ZWE_WE, Hwange_Rehab_TH_ZWE_WE, ZimbabweNG_TH_ZWE, ZZEE_TH_ZWB, Import_MOZ_EDM_ZWE, Import_SAPP_ZWE, Thuli_TH_ZWE, Hwange9_10_TH_ZWE, Centrachine_TH_ZWE, CloverNewEnergy_TH_ZWE, Afrochine_PV_ZWE, Karibafloating_Sunspire, Karibafloating_Sunspire_Ph2, Karibafloating_Sunspire_Ph3, NewGlovers_Phase1_PV_ZWE, NewGlovers_Phase2_PV_ZWE, Guruve_PV_ZWE, Grove_PV_ZWE, Sabi_PV_ZWE, Sunports_PV_ZWE, Mapanzure_PV_ZWE, Zimplats_Phase2_PV_ZWE, Skypower_PV_ZWE, New_OnshoreWind_ZWE, Import_MOZ_HCB_ZWE, Import_ZAM_ZWE, Export_NAM_ZWE, Import_MOZ_ZWE_new, Import_BWA_ZWE_new, Import_ZAM_ZWE_new, GowkeNorth_TH_ZWE, Chisumbanje_BIO_ZWE, Afrochine_TH_ZWE
- `supply/pStorageDataInput.csv`: New_BESS_MOZ_CE, New_BESS_MOZ_NO, New_BESS_MOZ_SO combined into New_BESS_MOZ (zone Mozambique, Capacity 10000, CapacityMWh 40000, BuildLimitperYear 300)
- `supply/pStorageDataInput.csv`: deleted 1 row(s) with a country-level zone name: NewPSP_MOZ
- generators/storage deleted (status 0, or zone not in zcmap): 41
- `load/pDemandProfile.csv`: merged zones = 2025 energy-weighted mean of the sub-zone profiles, rescaled to max 1
- `load/pDemandForecast_sapp_poolplan_v5_mozalout_corrected.csv`: 18 rows summed into 6 key(s): Angola/Energy, Mozambique/Energy, Zimbabwe/Energy, Angola/Peak, Mozambique/Peak, Zimbabwe/Peak
- `supply/pAvailabilityCustom.csv`: deleted 1 row(s) of deleted generators: ['ExistingSmallHydro2012_HY_AGO']
- `supply/pAvailabilityDefault.csv`: deleted 2 row(s) with a country-level zone name: Angola, Mozambique
- `supply/pCSPData.csv`: deleted 2 row(s) of deleted generators: ['New_CSP_AGO']
- `supply/pVREProfile.csv`: deleted 128 row(s) with a country-level zone name: Angola, Mozambique, Zimbabwe
- `supply/pVREProfile.csv`: merged zones = VRE-capacity-weighted mean of the sub-zone profiles (equal weights if no plant): Angola OffshoreWind: {'Angola_Central': 0.0, 'Angola_East': 0.0, 'Angola_North': 0.0, 'Angola_South': 8000.0}; Angola OnshoreWind: {'Angola_Central': 0.0, 'Angola_East': 1000.0, 'Angola_North': 106.0, 'Angola_South': 8000.0}; Angola PV: {'Angola_Central': 1228.0, 'Angola_East': 122.0, 'Angola_North': 0.0, 'Angola_South': 8081.0}; Mozambique OffshoreWind: {'Mozambique_South': 4000.0}; Mozambique OnshoreWind: {'Mozambique_Central': 4000.0, 'Mozambique_North': 0.0, 'Mozambique_South': 4180.0}; Mozambique PV: {'Mozambique_Central': 3620.0, 'Mozambique_North': 4361.0, 'Mozambique_South': 0.0}; Zimbabwe OnshoreWind: {'Zimbabwe_East': 0.0, 'Zimbabwe_West': 400.0}; Zimbabwe PV: {'Zimbabwe_East': 5106.0, 'Zimbabwe_West': 772.0}
- `supply/pStorageDataInputDefault.csv`: deleted 3 row(s) with a country-level zone name: Zimbabwe, Mozambique
- `constraint/pMaxGenerationByFuel.csv`: deleted 1 row(s) with a country-level zone name: Angola
- `trade/pLossFactorInternal.csv`: deleted 7 row(s) with a country-level zone name: Angola, Namibia, DRC, Mozambique, Zambia
- `trade/pLossFactorInternal.csv`: dropped 12 internal row(s) between merged sub-zones
- `trade/pLossFactorInternal.csv`: Zambia->Zimbabwe loss factors differ (Zambia->Zimbabwe_East 0.02, Zambia->Zimbabwe_West 0.07): kept Zambia->Zimbabwe_East, the corridor with the largest existing capacity
- `trade/pLossFactorInternal.csv`: Zimbabwe->Zambia loss factors differ (Zimbabwe_East->Zambia 0.02, Zimbabwe_West->Zambia 0.07): kept Zimbabwe_East->Zambia, the corridor with the largest existing capacity
- `trade/pNewTransmission_vnov26.csv`: deleted 5 row(s) with a country-level zone name: Namibia, Mozal, Angola, DRC, Zambia
- `trade/pNewTransmission_vnov26.csv`: dropped 5 internal row(s) between merged sub-zones
- `trade/pNewTransmission_vnov26.csv`: DRC->Angola has 2 candidates (DRC->Angola_North status 3, DRC->Angola_East status 0): kept DRC->Angola_North
- `trade/pTransferLimit.csv`: dropped 24 internal row(s) between merged sub-zones
- `trade/pTransferLimit.csv`: 16 rows summed into 8 key(s): Mozambique/South_Africa/Q1, South_Africa/Mozambique/Q1
- sub-zone names left in the new folder: none
- map layers: `geojson_to_epm.csv` (mapping + whole-country rows) and `zones*.geojson`, `linestring*.geojson` regenerated
