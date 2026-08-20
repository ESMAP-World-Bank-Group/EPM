"""Build the zone layers EPM View draws, with or without a model run.

Two files describe a model's map:

    zones_{stem}.geojson        one polygon per EPM zone      (z, ISO_A3, c)
    linestring_{stem}.geojson   centroid-to-centroid lines    (z, z_other, c, c2)

Both are a pure function of four sources -- the zoning, the admin-area-to-zone
mapping, the reference polygons, and the hand-drawn overlay -- so neither needs
a solve. `recipe.py` resolves which four apply to a folder; this module cuts
them.

    python -m epm.geodata.zone_layers --folder data_casa
    python -m epm.geodata.zone_layers --folder data_casa --out epm/output/run_0820
    python -m epm.geodata.zone_layers --check

`epm/postprocessing/create_geojson.py` calls in here, so the layers a run
writes and the layers this command writes come out of the same code.
"""

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

if __package__ in (None, ''):        # `python epm/geodata/zone_layers.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from epm.geodata import recipe
from epm.geodata.zone_geometry import create_zonemap, get_json_data

# Centroids of a geographic CRS are good enough to hang a line off, and the
# warning fires once per zone.
warnings.filterwarnings('ignore', message='.*Geometry is in a geographic CRS.*',
                        category=UserWarning)


@dataclass
class Sources:
    """The four files one pair of layers is cut from, plus what to cut.

    `stem` names the output pair: `zones_{stem}.geojson`, or the unsuffixed
    `zones.geojson` / `linestring_countries.geojson` when it is None.
    """

    zcmap: Path
    geojson_to_epm: Path
    zones_custom: Optional[Path]
    zone_map: Path
    zones: Sequence[str]
    stem: Optional[str] = None

    def fingerprint(self):
        """Hashes of the sources, for the `epm_source` stamp on the outputs."""
        return recipe.source_fingerprint(
            self.zcmap, self.geojson_to_epm,
            zone_map_path=self.zone_map, zones_custom_path=self.zones_custom)

    def names(self):
        return (recipe.output_names(self.stem) if self.stem
                else recipe.legacy_names())


def resolve(folder, zcmap=None, zones=None, stem=..., zone_map=None):
    """Which sources apply to a data folder.

    A folder may ship its own `geojson_to_epm.csv` and `zones_custom.geojson`;
    otherwise the shared resources apply. `stem` defaults to the zcmap's own
    name, so `zcmap_robg.csv` writes `zones_zcmap_robg.geojson` and cannot
    overwrite the layers of the base zoning; pass None for the unsuffixed pair.
    """
    folder = Path(folder)
    zcmap_path = Path(zcmap) if zcmap else folder / 'zcmap.csv'
    if not zcmap_path.is_absolute() and not zcmap_path.exists():
        zcmap_path = folder / zcmap_path
    if not zcmap_path.exists():
        raise FileNotFoundError(f'zcmap not found: {os.path.abspath(zcmap_path)}')
    return Sources(
        zcmap=zcmap_path,
        geojson_to_epm=recipe.resolve_geojson_to_epm(folder),
        zones_custom=recipe.resolve_zones_custom(folder),
        zone_map=Path(zone_map) if zone_map else recipe.SHARED_ZONE_MAP,
        zones=list(zones) if zones else recipe.zcmap_zones(zcmap_path),
        stem=zcmap_path.stem if stem is ... else stem,
    )


def _zone_country(zcmap):
    """(zone -> country) from a zcmap, tolerating the z/zone and c/country spellings."""
    df = pd.read_csv(zcmap) if isinstance(zcmap, (str, Path)) else zcmap.copy()
    zone_col = 'zone' if 'zone' in df.columns else 'z'
    country_col = 'country' if 'country' in df.columns else 'c'
    return df.set_index(zone_col)[country_col]


def _polygons(zone_map_gdf, geojson_to_epm_dict, zone_country):
    """One polygon per zone, with the columns EPM View reads."""
    zones = zone_map_gdf.copy()
    zones['z'] = zones['ADMIN'].map(geojson_to_epm_dict)
    zones = zones[zones['z'].notna()].copy()
    zones['c'] = zones['z'].map(zone_country)
    if 'ISO_A3' not in zones.columns:
        zones['ISO_A3'] = None
    return gpd.GeoDataFrame(
        zones[['z', 'ISO_A3', 'c', 'geometry']].reset_index(drop=True),
        geometry='geometry', crs=zone_map_gdf.crs)


def _lines(centers, zone_country):
    """A LineString between every ordered pair of zone centroids.

    Both directions are kept: the explorer reads flow on a directed pair, so
    A->B and B->A are different rows carrying different values.
    """
    points = gpd.GeoDataFrame(
        {'z': list(centers.keys())},
        geometry=[Point(c) for c in centers.values()],
        crs='EPSG:4326').reset_index(drop=True)

    rows = []
    for i, a in points.iterrows():
        for j, b in points.iterrows():
            if i == j:
                continue
            rows.append({**a.to_dict(),
                         **{f'{k}_other': v for k, v in b.to_dict().items()}})

    df = pd.DataFrame(rows)
    df['country_ini_lat'] = df['geometry'].apply(lambda p: p.y)
    df['country_ini_lon'] = df['geometry'].apply(lambda p: p.x)
    df['geometry'] = df.apply(
        lambda r: LineString([r['geometry'], r['geometry_other']]), axis=1)
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df.crs = points.crs
    df.drop(columns=['geometry_other'], inplace=True)

    centroid = df['geometry'].centroid
    df['lat_linestring'] = centroid.apply(lambda p: p.y)
    df['lon_linestring'] = centroid.apply(lambda p: p.x)

    df = df.set_index('z')
    df['c'] = zone_country
    df = df.reset_index().set_index('z_other')
    df['c2'] = zone_country
    return df


def build(sources=None, zone_country=None, dict_specs=None, selected_zones=None,
          log=None):
    """Cut one pair of layers. Returns (zones, lines); writes nothing.

    `dict_specs` and `zone_country` are the run's way in: post-processing
    already holds the reference polygons and the zoning the solve actually
    used, and passes them rather than re-reading the folder.
    """
    warn = log or (lambda m: print(m))
    zones_wanted = list(selected_zones if selected_zones is not None else sources.zones)

    if dict_specs is not None:
        zone_map_gdf, mapping = get_json_data(
            selected_zones=zones_wanted, dict_specs=dict_specs)
    else:
        zone_map_gdf, mapping = get_json_data(
            selected_zones=zones_wanted,
            geojson_to_epm=str(sources.geojson_to_epm),
            zone_map=str(sources.zone_map),
            zones_custom=str(sources.zones_custom) if sources.zones_custom else None)

    zone_map_gdf, centers = create_zonemap(zone_map_gdf, map_geojson_to_epm=mapping)

    missing = [z for z in zones_wanted if z not in centers]
    if missing:
        warn(f'{len(missing)} zone(s) have no map geometry: {missing}\n'
             f'  Add a row for each to the geojson_to_epm.csv that applies to this\n'
             f'  folder (epm/input/<folder>/geojson_to_epm.csv, else\n'
             f'  epm/resources/postprocess/geojson_to_epm.csv). A zone that matches no\n'
             f'  admin area at all must first be drawn in zones_custom.geojson under an\n'
             f'  ADMIN property, and that name mapped.')

    if not centers:
        warn('No zone has map geometry - the layers will be empty.')
        empty = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
        return empty, empty

    if zone_country is None:
        zone_country = _zone_country(sources.zcmap)
    elif not isinstance(zone_country, pd.Series):
        zone_country = _zone_country(zone_country)

    return (_polygons(zone_map_gdf, mapping, zone_country),
            _lines(centers, zone_country))


def write(zones_gdf, lines_gdf, out_dir, stem=None, fingerprint=None):
    """Write a pair of layers into `out_dir`. Returns the two paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zones_name, lines_name = (recipe.output_names(stem) if stem
                              else recipe.legacy_names())
    zones_path, lines_path = out_dir / zones_name, out_dir / lines_name

    lines_gdf.to_file(str(lines_path), driver='GeoJSON')
    zones_gdf.to_file(str(zones_path), driver='GeoJSON')
    if fingerprint:
        drawn = sorted(set(zones_gdf['z'])) if 'z' in zones_gdf.columns else []
        for path in (lines_path, zones_path):
            recipe.stamp(path, fingerprint, drawn)
    return zones_path, lines_path


def build_and_write(sources, out_dir, log=None, **kwargs):
    """Cut one pair of layers and write it, stamped with its sources."""
    zones_gdf, lines_gdf = build(sources, log=log, **kwargs)
    return write(zones_gdf, lines_gdf, out_dir, stem=sources.stem,
                 fingerprint=sources.fingerprint())


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

HELP = """
Build the zone layers of a model without running it.

  python -m epm.geodata.zone_layers --folder data_casa
      Rewrite epm/input/data_casa's layers from today's sources: one pair per
      zcmap*.csv it carries, plus the unsuffixed pair the base zcmap owns.

  python -m epm.geodata.zone_layers --folder data_casa --zcmap zcmap_robg.csv
      Only that zoning.

  python -m epm.geodata.zone_layers --folder data_casa --out epm/output/run_0820
      Somewhere else -- a run folder, or a scratch directory to look at first.

  python -m epm.geodata.zone_layers --check
      Report every folder whose layers no longer match their sources and write
      nothing. Exits 1 when anything is out of date, so CI can run it.

Sources, each resolved per folder with a fallback to the shared resources:
  epm/input/<folder>/zcmap*.csv               the zones of the model
  epm/input/<folder>/geojson_to_epm.csv       admin area -> zone, and split rules
    else epm/resources/postprocess/geojson_to_epm.csv
  epm/resources/postprocess/zones.geojson     admin-0 polygons
  epm/input/<folder>/zones_custom.geojson     zones no admin area can supply
    else epm/resources/postprocess/zones_custom.geojson
"""


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='python -m epm.geodata.zone_layers',
        description='Build the zone GeoJSON layers EPM View draws.',
        epilog=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folder', help='a folder under epm/input/, or a path to one')
    ap.add_argument('--zcmap', help='one zcmap of that folder (default: every zcmap*.csv)')
    ap.add_argument('--out', help='where to write (default: the folder itself)')
    ap.add_argument('--zones', nargs='+',
                    help='restrict to these zones; the result is left unstamped, '
                         'since it does not correspond to the whole zcmap')
    ap.add_argument('--check', action='store_true',
                    help='report layers that no longer match their sources, write nothing')
    args = ap.parse_args(argv)

    if args.check:
        issues = recipe.check_all()
        print(recipe.format_issues(issues))
        return 1 if issues else 0

    if not args.folder:
        ap.error('--folder is required (or --check)')

    folder = Path(args.folder)
    if not folder.exists():
        folder = recipe.INPUT_DIR / args.folder
    if not folder.exists():
        ap.error(f'no such folder: {args.folder}')

    out_dir = Path(args.out) if args.out else folder
    zcmaps = [folder / args.zcmap] if args.zcmap else recipe.zcmap_files(folder)
    if not zcmaps:
        ap.error(f'{folder} carries no zcmap*.csv')

    for zcmap in zcmaps:
        # The unsuffixed pair belongs to the base zcmap, and is what a folder
        # with a single zoning actually publishes; write both for it.
        stems = [zcmap.stem] + ([None] if zcmap.stem == 'zcmap' else [])
        for stem in stems:
            sources = resolve(folder, zcmap=zcmap, zones=args.zones, stem=stem)
            if args.zones:
                # A subset does not match the zcmap the check compares against,
                # so stamping it would make a partial layer look authoritative.
                zones_gdf, lines_gdf = build(sources)
                paths = write(zones_gdf, lines_gdf, out_dir, stem=stem)
            else:
                paths = build_and_write(sources, out_dir)
            for path in paths:
                print(f'  wrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
