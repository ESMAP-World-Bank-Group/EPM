"""Reference geographic data for EPM: the boundaries its zones are cut from.

The single source of truth for every boundary EPM draws or cuts zones from:
the World Bank Official Boundaries, built into a dated artifact by
`wb_boundaries.py`. Everything downstream -- the zone polygons in
`epm/resources/postprocess/`, the per-model `epm/input/data_*/zones*.geojson`,
and the basemap of the data explorer -- derives from that one artifact, so the
maps can never disagree with each other or with Bank cartographic policy.

Nothing here depends on model results. A zone layer is a pure function of the
zoning (zcmap), the mapping from admin areas to zones, and those reference
polygons -- so it can be built before a run, during one, or with no run at all.
"""
