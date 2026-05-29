"""Generate an interactive HTML map of the current Turkey zone configuration."""
import json
from pathlib import Path

HERE = Path(__file__).parent

zones_path   = HERE / "zones_turkiye.geojson"
topology_path = HERE / "sTopology.csv"
output_path  = HERE / "zone_map_turkiye.html"

import folium
import pandas as pd

# Load data
with open(zones_path, encoding="utf-8") as f:
    zones = json.load(f)

topo = pd.read_csv(topology_path)

# Zone colors (one per zone)
COLORS = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
          "#edc948","#b07aa1","#ff9da7","#9c755f"]

zone_names = [f["properties"]["ADMIN"] for f in zones["features"]]
color_map  = {z: COLORS[i % len(COLORS)] for i, z in enumerate(zone_names)}

# Map centered on Turkey
m = folium.Map(location=[39.0, 35.0], zoom_start=6, tiles="CartoDB positron")

# Zone polygons
centroids = {}
for feat in zones["features"]:
    z = feat["properties"]["ADMIN"]
    color = color_map[z]

    folium.GeoJson(
        feat,
        style_function=lambda f, c=color: {
            "fillColor": c, "color": "#333",
            "weight": 1.5, "fillOpacity": 0.35,
        },
        tooltip=z,
    ).add_to(m)

    # Compute centroid for label + topology lines
    coords = feat["geometry"]["coordinates"]
    geom_type = feat["geometry"]["type"]
    all_pts = []
    if geom_type == "Polygon":
        all_pts = coords[0]
    elif geom_type == "MultiPolygon":
        for poly in coords:
            all_pts.extend(poly[0])
    if all_pts:
        cx = sum(p[0] for p in all_pts) / len(all_pts)
        cy = sum(p[1] for p in all_pts) / len(all_pts)
        centroids[z] = (cy, cx)

# Zone labels
for z, (lat, lon) in centroids.items():
    folium.Marker(
        [lat, lon],
        icon=folium.DivIcon(
            html=f'<div style="font-size:11px;font-weight:bold;color:#111;'
                 f'background:rgba(255,255,255,0.75);padding:2px 5px;'
                 f'border-radius:3px;white-space:nowrap">{z}</div>',
            icon_size=(120, 24), icon_anchor=(60, 12),
        ),
    ).add_to(m)

# Topology links
for _, row in topo.iterrows():
    z1, z2 = row.iloc[0], row.iloc[1]
    if z1 in centroids and z2 in centroids:
        folium.PolyLine(
            [centroids[z1], centroids[z2]],
            color="#666", weight=2, opacity=0.6, dash_array="5 4",
            tooltip=f"{z1} ↔ {z2}",
        ).add_to(m)

m.save(str(output_path))
print(f"Saved: {output_path}")
