"""Sample points for the Greater Malé VRE profiles (one per EPM zone, plus offshore candidates).

Zone names are clean IDs without the trailing spaces that zcmap.csv still has.
Thilafushi and Villingili use the ESMAP 2015 candidate wind-mast sites (#25, #26), as requested.
Malé, Hulhumalé and Gulhifalhu have no ESMAP site, so they use a point on land near the island centre
(checked against OpenStreetMap island outlines).

Maldives local time is UTC+5, no daylight saving.
"""
UTC_OFFSET = 5

ZONES = {
    # zone: (lat, lon, how the point was chosen)
    "Male":       (4.1755, 73.5093, "island centre"),
    "Hulhumale":  (4.2160, 73.5420, "island centre (Phase 1/2)"),
    "Villingili": (4.17088, 73.48658, "ESMAP 2015 site #26 Vilingilli"),
    "Gulhifalhu": (4.1789, 73.4689, "island centre"),
    "Thilafushi": (4.18277, 73.43135, "ESMAP 2015 site #25 Thilafushi"),
}

# Offshore wind: the lagoon / reef around Greater Malé. ESMAP #24 (Kodhdhipparufinolhu) is the
# best-scoring ESMAP site near the study area, 6 km NW of Thilafushi.
OFFSHORE = {
    "ESMAP24": (4.25915, 73.37954, "ESMAP 2015 site #24 Kodhdhipparufinolhu (reef island, outside the 5 zones)"),
}

# ESMAP measurement sites (energydata.info), used only to validate the profiles (validate_esmap.py)
VALIDATION = {
    "MLE_solar": (4.19274, 73.52811, "ESMAP Tier 2 solar station, Velana airport, Hulhule (Suntrace; Dec 2015 - Mar 2018)"),
    "Thulusdhoo_lidar": (4.37453, 73.65217, "ESMAP ZephIR lidar, Thulusdhoo, North Male atoll (DNV GL; Apr 2017 - Apr 2019)"),
}
