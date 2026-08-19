"""Reference geographic data for EPM.

The single source of truth for every boundary EPM draws or cuts zones from:
the World Bank Official Boundaries, built into a dated artifact by
`wb_boundaries.py`. Everything downstream -- the zone polygons in
`epm/resources/postprocess/`, the per-model `epm/input/data_*/zones*.geojson`,
and the basemap of the data explorer -- derives from that one artifact, so the
maps can never disagree with each other or with Bank cartographic policy.
"""
