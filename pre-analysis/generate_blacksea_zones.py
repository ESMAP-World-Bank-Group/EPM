import json

SRC = r'C:\Users\wb590892\Documents\EPM_Models\epm-data-explorer\public\data\zones\blacksea_turkey9z_zones_hd.geojson'
DST = r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\EPM\epm\input\data_blacksea\zones.geojson'

KEEP = {
    'TUR': {
        'zones': ['EastMed','WestMed','SouthEast','WestAna','CenterAna','EastAna','NorthWest','CenterBlack','Trakia'],
        'c': 'Turkiye',
    },
    'ARM': {
        'zones': ['ARM_1'],
        'c': 'Armenia',
        'z_override': 'Armenia',
    },
}

with open(SRC) as f:
    src_gj = json.load(f)

features = []
for ft in src_gj['features']:
    zone_name = ft['properties'].get('zone_name', '')
    iso = ft['properties'].get('ISO_A3', '')
    if iso not in KEEP:
        continue
    cfg = KEEP[iso]
    if zone_name not in cfg['zones']:
        continue
    z = cfg.get('z_override', zone_name)
    features.append({
        'type': 'Feature',
        'properties': {'z': z, 'c': cfg['c'], 'ISO_A3': iso},
        'geometry': ft['geometry'],
    })

out = {'type': 'FeatureCollection', 'features': features}
with open(DST, 'w') as f:
    json.dump(out, f, separators=(',', ':'))

print("Written", len(features), "features to", DST)
for ft in features:
    print(" ", ft['properties'])
