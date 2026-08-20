"""Backwards-compatible name for `epm.geodata.recipe`.

The provenance and freshness tracking moved to `epm/geodata/`, next to the code
that builds the layers it describes; it never depended on post-processing.
Imports through this module keep working.
"""

from epm.geodata.recipe import *      # noqa: F401,F403
from epm.geodata.recipe import (      # noqa: F401  - names `import *` skips
    Issue,
    _check_pair,
    _dump_geojson,
    _rel,
)
