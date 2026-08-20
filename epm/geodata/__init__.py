"""Geographic data for EPM: the boundaries its zones are cut from.

Nothing in this package depends on model results. A zone layer is a pure
function of the zoning (zcmap), the mapping from admin areas to zones, and the
reference polygons -- so it can be built before a run, during one, or with no
run at all.
"""
