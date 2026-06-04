import json
import math

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
    'GEO': {
        'zones': ['GEO_1'],
        'c': 'Georgia',
        'z_override': 'Georgia',
    },
}


def ring_area(ring):
    """Shoelace formula — returns unsigned area."""
    n = len(ring)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += ring[i][0] * ring[j][1]
        area -= ring[j][0] * ring[i][1]
    return abs(area) / 2.0


def simplify_geometry(geom):
    """Keep only the largest polygon piece and drop all interior rings (holes).

    For EPM map display, sub-zone islands and polygon holes create visual
    artifacts — we only need the main territory outline.
    """
    if geom['type'] == 'Polygon':
        # Keep only the outer ring (index 0), drop holes (index 1+)
        return {'type': 'Polygon', 'coordinates': [geom['coordinates'][0]]}

    elif geom['type'] == 'MultiPolygon':
        # Each element is a list of rings; index 0 is the outer ring
        # Keep only the polygon with the largest outer ring
        best_poly = max(geom['coordinates'], key=lambda poly: ring_area(poly[0]))
        return {'type': 'Polygon', 'coordinates': [best_poly[0]]}

    return geom  # passthrough for other types


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
    simplified = simplify_geometry(ft['geometry'])
    features.append({
        'type': 'Feature',
        'properties': {'z': z, 'c': cfg['c'], 'ISO_A3': iso},
        'geometry': simplified,
    })

out = {'type': 'FeatureCollection', 'features': features}
with open(DST, 'w') as f:
    json.dump(out, f, separators=(',', ':'))

print("Written", len(features), "features to", DST)
for ft in features:
    geom = ft['geometry']
    n_pts = len(geom['coordinates'][0])
    print(f"  {ft['properties']['z']:20s} {geom['type']:12s} {n_pts} pts (outer ring only)")
