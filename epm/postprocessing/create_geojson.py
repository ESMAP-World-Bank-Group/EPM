"""
**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
Description:
    This Python script is part of the GAMS-based Electricity Planning Model (EPM),
    designed for electricity system planning. It supports tasks such as capacity
    expansion, generation dispatch, and the enforcement of policy constraints,
    including renewable energy targets and emissions limits.

Author(s):
    ESMAP Modelling Team

Organization:
    World Bank

Version:
    (Specify version here)

License:
    Creative Commons Zero v1.0 Universal

Key Features:
    - Optimization of electricity generation and capacity planning
    - Inclusion of renewable energy integration and storage technologies
    - Multi-period, multi-region modeling framework
    - CO₂ emissions constraints and policy instruments

Notes:
    - Ensure GAMS is installed and the model has completed execution
      before running this script.
    - The model generates output files in the working directory
      which will be organized by this script.

Contact:
    Claire Nicolas — c.nicolas@worldbank.org
**********************************************************************
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

# Suppress warning about centroid on geographic CRS - acceptable for visualization
warnings.filterwarnings('ignore', message=".*Geometry is in a geographic CRS.*", category=UserWarning)

# If this script runs directly from `epm/postprocessing`, make sure the
# repository root is on `sys.path` so package imports succeed.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing utility functions for data processing using the package path
from epm.geodata import recipe, zone_layers
from epm.geodata.zone_geometry import get_json_data, create_zonemap
from epm.postprocessing.utils import log_warning, log_info

# Historical name for the module that resolves and fingerprints the sources.
freshness = recipe


def create_geojson_for_tableau(geojson_to_epm, zcmap, selected_zones, folder='tableau',
                               zone_map=None, output_path=None, dict_specs=None,
                               output_stem=None, custom_zones=None, stamp_sources=None):
    """Generate the linestring and zones GeoJSON layers for a set of EPM zones.

    Kept for the callers that already use it -- post-processing, and the
    per-model rebuild scripts under tools/. The layers themselves are cut by
    `epm.geodata.zone_layers`, which is also what `python -m
    epm.geodata.zone_layers` runs, so a layer built during a run and one built
    on demand come out of the same code.

    Parameters
    ----------
    geojson_to_epm : str or None
        Path to the admin-area-to-zone mapping CSV. None falls back to the
        mapping that applies to `output_path`.
    zcmap : str or pandas.DataFrame
        Path to a zcmap, or a frame with zone/country (or z/c) columns.
    selected_zones : list of str
        The zones to draw.
    folder, zone_map, output_path, dict_specs, output_stem, custom_zones,
    stamp_sources
        As before: `output_path` is where the pair is written, `output_stem`
        names it, `dict_specs` lets a run pass the polygons it already holds,
        and `stamp_sources` the fingerprint to record in `epm_source`.

    Returns
    -------
    geopandas.GeoDataFrame
        The linestring layer.
    """
    out_dir = Path(output_path) if output_path else Path('..') / 'output' / folder

    sources = None
    if dict_specs is None:
        mapping = (Path(geojson_to_epm) if geojson_to_epm
                   else recipe.resolve_geojson_to_epm(out_dir))
        sources = zone_layers.Sources(
            zcmap=zcmap if isinstance(zcmap, (str, Path)) else None,
            geojson_to_epm=mapping,
            zones_custom=Path(custom_zones) if custom_zones
            else recipe.resolve_zones_custom(out_dir),
            zone_map=Path(zone_map) if zone_map else recipe.SHARED_ZONE_MAP,
            zones=selected_zones,
            stem=output_stem,
        )

    zones_gdf, lines_gdf = zone_layers.build(
        sources=sources,
        zone_country=zcmap if not isinstance(zcmap, (str, Path)) else None,
        dict_specs=dict_specs,
        selected_zones=selected_zones,
        log=log_warning,
    )
    zone_layers.write(zones_gdf, lines_gdf, out_dir, stem=output_stem,
                      fingerprint=stamp_sources)
    zones_name, lines_name = (recipe.output_names(output_stem) if output_stem
                              else recipe.legacy_names())
    log_info(f'Linestring GeoJSON written: {out_dir / lines_name} ({len(lines_gdf)} lines)')
    log_info(f'Zones GeoJSON written: {out_dir / zones_name} ({len(zones_gdf)} zones)')
    return lines_gdf


def regen_zones_from_run(run_folder, dict_specs=None):
    """
    Generate zones.geojson for an existing run folder from its linestring_countries.geojson.

    Reads zone names from the linestring file, rebuilds polygon geometries via
    get_json_data, and writes zones.geojson alongside. Use this for runs that
    predate automatic zones.geojson generation.

    Parameters
    ----------
    run_folder : str or Path
        Path to the run folder (e.g. epm/output_view/RETRADE_0626/).
    dict_specs : dict, optional
        If provided, use for loading zone data. Otherwise read_plot_specs() is called.
    """
    ls_path = Path(run_folder) / 'linestring_countries.geojson'
    if not ls_path.exists():
        raise FileNotFoundError(f"linestring_countries.geojson not found in {run_folder}")

    with open(ls_path, encoding='utf-8') as f:
        ls_data = json.load(f)

    zone_set = set()
    for feature in ls_data['features']:
        p = feature.get('properties', {})
        if p.get('z'):
            zone_set.add(p['z'])
        if p.get('z_other'):
            zone_set.add(p['z_other'])

    if not zone_set:
        raise ValueError("No zone identifiers found in linestring_countries.geojson properties")

    selected_zones = sorted(zone_set)
    log_info(f"regen_zones_from_run: rebuilding {len(selected_zones)} zones from {run_folder}")

    if dict_specs is None:
        from epm.postprocessing.postprocessing import read_plot_specs
        dict_specs = read_plot_specs()

    zone_map_gdf, geojson_to_epm_dict = get_json_data(
        selected_zones=selected_zones,
        dict_specs=dict_specs
    )
    zone_map_gdf, _ = create_zonemap(zone_map_gdf, map_geojson_to_epm=geojson_to_epm_dict)

    zones = zone_map_gdf.copy()
    zones['z'] = zones['ADMIN'].map(geojson_to_epm_dict)
    zones = zones[zones['z'].notna()].copy()
    zones_gdf = gpd.GeoDataFrame(
        zones[['z', 'ISO_A3', 'geometry']].reset_index(drop=True),
        geometry='geometry',
        crs=zone_map_gdf.crs
    )

    out_path = Path(run_folder) / 'zones.geojson'
    zones_gdf.to_file(str(out_path), driver='GeoJSON')
    log_info(f"zones.geojson written: {out_path} ({len(zones_gdf)} zones)")
    return zones_gdf




def generate_for_data_folder(folder, zcmap_path, legacy=False):
    """Generate one (zones, linestring) pair for an epm/input/data_* folder.

    Sources are resolved per folder: a data folder may ship its own
    `geojson_to_epm.csv` and `zones_custom.geojson`, otherwise the shared
    resources apply. Both outputs are stamped with the fingerprint of the
    sources they were built from.

    `legacy` writes the unsuffixed `zones.geojson` / `linestring_countries.geojson`
    pair instead of the stem-named one; the base zcmap.csv owns both.
    """
    folder = Path(folder)
    zcmap_path = Path(zcmap_path)
    geojson_to_epm = freshness.resolve_geojson_to_epm(folder)
    custom_zones = freshness.resolve_zones_custom(folder)
    selected_zones = freshness.zcmap_zones(zcmap_path)
    fingerprint = freshness.source_fingerprint(
        zcmap_path, geojson_to_epm, zones_custom_path=custom_zones
    )
    return create_geojson_for_tableau(
        geojson_to_epm=str(geojson_to_epm),
        zcmap=str(zcmap_path),
        selected_zones=selected_zones,
        output_path=str(folder),
        output_stem=None if legacy else zcmap_path.stem,
        custom_zones=str(custom_zones) if custom_zones else None,
        stamp_sources=fingerprint,
    )


def iter_generation_targets(input_dir=None, create_missing=False):
    """Every (folder, zcmap_path, legacy) pair `--all` should regenerate.

    A data folder is only touched when it already carries map layers, i.e. a
    zones GeoJSON exists. Folders that never opted in are skipped: the shared
    mapping does not necessarily cover their zones, so generating there would
    write empty files over usable ones. `create_missing` lifts that guard.
    """
    root = Path(input_dir or freshness.INPUT_DIR)
    legacy_zones, _ = freshness.legacy_names()
    for folder in sorted(root.glob('data_*')):
        if not folder.is_dir():
            continue
        opted_in = (folder / legacy_zones).exists() or any(folder.glob('zones_*.geojson'))
        if not opted_in and not create_missing:
            continue
        for zcmap_path in freshness.zcmap_files(folder):
            zones_name, _ = freshness.output_names(zcmap_path.stem)
            if (folder / zones_name).exists() or create_missing:
                yield folder, zcmap_path, False
            if zcmap_path.stem == 'zcmap' and ((folder / legacy_zones).exists() or create_missing):
                yield folder, zcmap_path, True


if __name__ == '__main__':
    HELP_TEXT = """
Generate the zone GeoJSON layers (linestring + zones) consumed by EPM View.

Refresh every data folder that already has map layers (run from the repo root):
    python epm/postprocessing/create_geojson.py --all

Report which layers no longer match their sources, without writing anything
(exit code 1 when something is out of date, so CI or a pre-commit hook can use it):
    python epm/postprocessing/create_geojson.py --check

Refresh a single folder:
    python epm/postprocessing/create_geojson.py --folder data_sapp --zcmap zcmap.csv

Generates in epm/input/{folder}/:
    linestring_{zcmap_stem}.geojson   - LineStrings between zone centroids
    zones_{zcmap_stem}.geojson        - Polygon per EPM zone
plus the unsuffixed zones.geojson / linestring_countries.geojson for the base zcmap.csv.

Sources, each resolved per folder with a fallback to the shared resources:
    epm/input/{folder}/zcmap*.csv               zones of the model
    epm/input/{folder}/geojson_to_epm.csv       admin area -> zone, and split rules
      else epm/resources/postprocess/geojson_to_epm.csv
    epm/input/{folder}/zones_custom.geojson     zones no admin area can supply
      else epm/resources/postprocess/zones_custom.geojson

To regenerate zones.geojson for an existing run folder without re-running GAMS:
    python epm/postprocessing/create_geojson.py --regen-zones --run-folder epm/output_view/RETRADE_0626

Note: also runs automatically during postprocessing.py for multi-zone models.
"""

    parser = argparse.ArgumentParser(
        description="Generate GeoJSON files for EPM zone visualization.",
        epilog=HELP_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--all", action="store_true",
                        help="Regenerate every data folder that already carries map layers")
    parser.add_argument("--check", action="store_true",
                        help="Report layers that no longer match their sources; write nothing. "
                             "Exits 1 when anything is out of date")
    parser.add_argument("--create-missing", action="store_true",
                        help="With --all, also generate layers for folders that have none yet")
    parser.add_argument("--zones", nargs="+", default=None,
                        help="List of EPM zone names (default: all zones from zcmap.csv)")
    parser.add_argument("--folder", type=str, default=None,
                        help="Folder name in epm/input/ where zcmap.csv is located")
    parser.add_argument("--zcmap", type=str, default="zcmap.csv",
                        help="Filename of zone-country mapping CSV (default: zcmap.csv)")
    parser.add_argument("--regen-zones", action="store_true",
                        help="Regenerate zones.geojson for an existing run folder from its linestring file")
    parser.add_argument("--run-folder", type=str, default=None,
                        help="Run folder path for --regen-zones mode (e.g. epm/output_view/RETRADE_0626)")

    args = parser.parse_args()

    if args.check:
        issues = freshness.check_all()
        print(freshness.format_issues(issues))
        sys.exit(1 if issues else 0)

    if args.regen_zones:
        if not args.run_folder:
            print("Error: --regen-zones requires --run-folder to be specified.")
            sys.exit(1)
        print(f"Regenerating zones.geojson for: {os.path.abspath(args.run_folder)}")
        zones = regen_zones_from_run(args.run_folder)
        print(f"Done: zones.geojson written with {len(zones)} zones.")
        sys.exit(0)

    if args.all:
        targets = list(iter_generation_targets(create_missing=args.create_missing))
        if not targets:
            print("No data folder carries map layers yet. Use --create-missing to bootstrap one.")
            sys.exit(0)
        for folder, zcmap_path, legacy in targets:
            names = freshness.legacy_names() if legacy else freshness.output_names(zcmap_path.stem)
            print(f"\n=== {folder.name} / {zcmap_path.name} -> {', '.join(names)}")
            generate_for_data_folder(folder, zcmap_path, legacy=legacy)
        remaining = freshness.check_all()
        print()
        print(freshness.format_issues(remaining))
        sys.exit(0)

    if not args.folder:
        parser.error("one of --all, --check, --folder or --regen-zones is required")

    # Single-folder mode: honours --zones so a subset can be generated on demand.
    zcmap_stem = Path(args.zcmap).stem

    if os.path.isabs(args.zcmap) or os.path.exists(args.zcmap):
        zcmap_path = Path(args.zcmap)
    else:
        zcmap_path = Path('epm') / 'input' / args.folder / args.zcmap

    output_path = Path('epm') / 'input' / args.folder
    if not output_path.exists():
        print(f"Error: Output folder does not exist: {os.path.abspath(output_path)}")
        sys.exit(1)
    if not zcmap_path.exists():
        print(f"Error: zcmap not found at {os.path.abspath(zcmap_path)}.")
        sys.exit(1)

    print(f"Output folder: {os.path.abspath(output_path)}")

    if args.zones is None:
        generate_for_data_folder(output_path, zcmap_path)
        if zcmap_stem == 'zcmap':
            generate_for_data_folder(output_path, zcmap_path, legacy=True)
    else:
        # Explicit zone subset: no provenance stamp, since the result does not
        # correspond to the full zcmap the check would compare it against.
        print(f"Generating GeoJSON for zones: {args.zones}")
        create_geojson_for_tableau(
            geojson_to_epm=str(freshness.resolve_geojson_to_epm(output_path)),
            zcmap=str(zcmap_path),
            selected_zones=args.zones,
            output_path=str(output_path),
            output_stem=zcmap_stem,
            custom_zones=(lambda p: str(p) if p else None)(freshness.resolve_zones_custom(output_path)),
        )

    print(f"\nDone: {args.folder}")
