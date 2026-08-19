"""
Provenance and freshness tracking for the generated zone GeoJSON files.

The map layers of EPM View are *derived* artefacts committed under
``epm/input/data_*/``:

    zones_{stem}.geojson        one polygon per EPM zone
    linestring_{stem}.geojson   centroid-to-centroid lines between zones

They are built by ``create_geojson.py`` from up to four sources:

    epm/input/{folder}/{stem}.csv                       zone -> country (zcmap)
    epm/input/{folder}/geojson_to_epm.csv               admin name -> zone (+ split rules),
      or epm/resources/postprocess/geojson_to_epm.csv     falling back to the shared mapping
    epm/resources/postprocess/zones.geojson             admin-0 polygons
    epm/input/{folder}/zones_custom.geojson             hand-drawn areas no admin polygon
      or epm/resources/postprocess/zones_custom.geojson   supplies, in the same ADMIN schema

Nothing regenerates them automatically, so editing any source silently leaves
the map showing the previous zoning. To make that detectable, every file
written by ``create_geojson.py`` carries an ``epm_source`` member recording the
SHA-256 of each source it was built from. ``check_folder`` recomputes those
hashes and reports the files that no longer match.

``epm_source`` is a GeoJSON foreign member: GDAL/geopandas and the browser both
ignore it, so stamping is invisible to every consumer.

This module depends on the standard library and pandas only -- no geopandas --
so the check can run inside a model run without pulling in the plotting stack.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Repo layout. Mirrors the constants in epm/postprocessing/utils.py, restated
# here so that importing this module does not drag in geopandas.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_DIR = REPO_ROOT / 'epm' / 'resources' / 'postprocess'
SHARED_GEOJSON_TO_EPM = RESOURCES_DIR / 'geojson_to_epm.csv'
SHARED_ZONE_MAP = RESOURCES_DIR / 'zones.geojson'
SHARED_ZONES_CUSTOM = RESOURCES_DIR / 'zones_custom.geojson'
INPUT_DIR = REPO_ROOT / 'epm' / 'input'

STAMP_KEY = 'epm_source'
GENERATOR = 'epm/postprocessing/create_geojson.py'

# Per-folder overrides. A data folder that uses its own zoning drops these files
# next to its zcmap; otherwise the shared resources apply.
FOLDER_GEOJSON_TO_EPM = 'geojson_to_epm.csv'
FOLDER_ZONES_CUSTOM = 'zones_custom.geojson'


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

def resolve_geojson_to_epm(folder):
    """Mapping file that applies to `folder`: its own override, else the shared one."""
    local = Path(folder) / FOLDER_GEOJSON_TO_EPM
    return local if local.exists() else SHARED_GEOJSON_TO_EPM


def resolve_zones_custom(folder):
    """Custom-zone overlay for `folder`, or None when neither override nor shared file exists."""
    local = Path(folder) / FOLDER_ZONES_CUSTOM
    if local.exists():
        return local
    return SHARED_ZONES_CUSTOM if SHARED_ZONES_CUSTOM.exists() else None


def zcmap_files(folder):
    """Every zcmap*.csv sitting at the root of a data folder, base zcmap.csv first."""
    found = sorted(Path(folder).glob('zcmap*.csv'))
    return sorted(found, key=lambda p: (p.stem != 'zcmap', p.stem))


def zcmap_zones(zcmap_path):
    """Zone names declared by a zcmap, tolerating the z/zone and c/country spellings."""
    df = pd.read_csv(zcmap_path)
    col = 'zone' if 'zone' in df.columns else 'z'
    return list(dict.fromkeys(df[col].dropna().astype(str)))


def output_names(stem):
    """(zones file, linestring file) for a zcmap stem."""
    return f'zones_{stem}.geojson', f'linestring_{stem}.geojson'


def legacy_names():
    """The unsuffixed pair the base zcmap.csv also owns, kept for backward compatibility."""
    return 'zones.geojson', 'linestring_countries.geojson'


# ---------------------------------------------------------------------------
# Stamping
# ---------------------------------------------------------------------------

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _rel(path):
    """Repo-relative posix path, for a stamp that reads the same on every machine."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def source_fingerprint(zcmap_path, geojson_to_epm_path, zone_map_path=None, zones_custom_path=None):
    """{role: {path, sha256}} for every source a generated file was built from."""
    entries = {
        'zcmap': zcmap_path,
        'geojson_to_epm': geojson_to_epm_path,
        'zone_map': zone_map_path or SHARED_ZONE_MAP,
        'zones_custom': zones_custom_path,
    }
    return {
        role: {'path': _rel(p), 'sha256': file_sha256(p)}
        for role, p in entries.items()
        if p is not None and os.path.exists(p)
    }


def _dump_geojson(data, path):
    """Write a GeoJSON with one feature per line, the way GDAL does.

    Keeps the file reviewable: a zone whose geometry changed then shows up as a
    single changed line instead of a whole-file rewrite.
    """
    features = data.get('features') or []
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('{\n')
        for key, value in data.items():
            if key == 'features':
                continue
            fh.write(f'{json.dumps(key)}: {json.dumps(value, ensure_ascii=False)},\n')
        fh.write('"features": [\n')
        for index, feature in enumerate(features):
            fh.write(json.dumps(feature, ensure_ascii=False))
            fh.write(',\n' if index < len(features) - 1 else '\n')
        fh.write(']\n}\n')


def stamp(geojson_path, fingerprint, zones):
    """Inject the `epm_source` member into an already-written GeoJSON file."""
    path = Path(geojson_path)
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    data[STAMP_KEY] = {
        'generator': GENERATOR,
        'generated_at': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        'zones': sorted(zones),
        'inputs': fingerprint,
    }
    _dump_geojson(data, path)


def read_stamp(geojson_path):
    """The `epm_source` member of a generated file, or None if absent/unreadable."""
    try:
        with open(geojson_path, encoding='utf-8') as fh:
            return json.load(fh).get(STAMP_KEY)
    except (OSError, ValueError):
        return None


def feature_zones(geojson_path):
    """Set of zone names actually carried by a GeoJSON's features."""
    try:
        with open(geojson_path, encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return set()
    zones = set()
    for feature in data.get('features') or []:
        props = feature.get('properties') or {}
        for key in ('z', 'z_other'):
            if props.get(key):
                zones.add(props[key])
    return zones


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------

class Issue:
    """One reason a generated GeoJSON no longer reflects its sources."""

    #: file absent although its zcmap exists and the folder opted into map layers
    MISSING = 'missing'
    #: file predates provenance stamping, so freshness cannot be established
    UNSTAMPED = 'unstamped'
    #: a source changed since the file was generated
    STALE = 'stale'
    #: zones declared in zcmap that the file draws nothing for
    ZONES_MISSING = 'zones-missing'

    def __init__(self, path, kind, detail):
        self.path = _rel(path)
        self.kind = kind
        self.detail = detail

    def __str__(self):
        return f'{self.path}: {self.kind} - {self.detail}'

    __repr__ = __str__


def _check_pair(folder, zcmap_path, zones_name, linestring_name, require_exists):
    folder = Path(folder)
    geojson_to_epm = resolve_geojson_to_epm(folder)
    zones_custom = resolve_zones_custom(folder)
    expected = source_fingerprint(zcmap_path, geojson_to_epm, zones_custom_path=zones_custom)
    declared_zones = set(zcmap_zones(zcmap_path))

    issues = []
    for name in (zones_name, linestring_name):
        path = folder / name
        if not path.exists():
            if require_exists:
                issues.append(Issue(path, Issue.MISSING,
                                    f'generated from {_rel(zcmap_path)}, never written'))
            continue

        found = read_stamp(path)
        if found is None:
            issues.append(Issue(path, Issue.UNSTAMPED,
                                'no epm_source member - predates provenance stamping, '
                                'or was hand-edited'))
        else:
            recorded = found.get('inputs') or {}
            changed = sorted(role for role, exp in expected.items()
                             if recorded.get(role, {}).get('sha256') != exp['sha256'])
            dropped = sorted(set(recorded) - set(expected))
            if changed or dropped:
                what = ', '.join(changed + [f'{r} (no longer used)' for r in dropped])
                issues.append(Issue(path, Issue.STALE,
                                    f'built {found.get("generated_at", "?")}, '
                                    f'source changed since: {what}'))

        missing = sorted(declared_zones - feature_zones(path))
        if missing:
            issues.append(Issue(path, Issue.ZONES_MISSING,
                                f'{len(missing)} zone(s) of {_rel(zcmap_path)} have no geometry: '
                                f'{", ".join(missing)}'))
    return issues


def check_folder(folder):
    """Every freshness issue of one epm/input/data_* folder.

    A folder is only checked when it already carries map layers (a zones
    GeoJSON exists); folders that never opted in are left alone.
    """
    folder = Path(folder)
    legacy_zones, legacy_lines = legacy_names()
    if not (folder / legacy_zones).exists() and not list(folder.glob('zones_*.geojson')):
        return []

    issues = []
    for zcmap_path in zcmap_files(folder):
        zones_name, linestring_name = output_names(zcmap_path.stem)
        issues += _check_pair(folder, zcmap_path, zones_name, linestring_name,
                              require_exists=(folder / zones_name).exists())
        if zcmap_path.stem == 'zcmap' and (folder / legacy_zones).exists():
            issues += _check_pair(folder, zcmap_path, legacy_zones, legacy_lines,
                                  require_exists=False)
    return issues


def check_all(input_dir=None):
    """{folder: issues} over every epm/input/data_* folder that carries map layers."""
    root = Path(input_dir or INPUT_DIR)
    out = {}
    for folder in sorted(root.glob('data_*')):
        if not folder.is_dir():
            continue
        issues = check_folder(folder)
        if issues:
            out[folder] = issues
    return out


def format_issues(issues_by_folder):
    """Human-readable report ending with the command that fixes it."""
    if not issues_by_folder:
        return 'Zone GeoJSON files are up to date with their sources.'
    lines = ['Zone GeoJSON files are out of date with their sources:']
    for folder, issues in issues_by_folder.items():
        lines.append(f'  {_rel(folder)}')
        for issue in issues:
            lines.append(f'    - {Path(issue.path).name}: {issue.kind} - {issue.detail}')
    lines.append('')
    lines.append('  Fix: python epm/postprocessing/create_geojson.py --all')
    return '\n'.join(lines)


def warn_if_stale(folder, log_func=None):
    """Report a single folder's issues without ever interrupting a model run.

    Called from epm.py at config load: a stale map must not stop a solve, but it
    must not stay invisible either.
    """
    try:
        issues = check_folder(folder)
    except Exception as exc:  # a broken check must never break the run
        if log_func is not None:
            log_func(f'Could not verify zone GeoJSON freshness for {folder}: {exc}')
        return []
    if not issues:
        return []
    (log_func or print)(format_issues({Path(folder): issues}))
    return issues
