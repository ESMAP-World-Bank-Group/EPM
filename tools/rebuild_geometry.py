"""Rebuild the zone geometry of every model under epm/input/ from the World Bank
boundary artifact, and check that it lines up with the map it will be drawn on.

Each model folder carries its own copy of the polygons the explorer paints:

    zones.geojson              one polygon per EPM zone, joined to model results
    linestring_countries.geojson   the lines between zone centroids
    zones_<stem>.geojson       the same pair per zcmap, when a model has several
    zones_offgrid.geojson      territory of a modelled country that no zone covers

None of them is a GAMS input -- the model reads CSVs only -- so rebuilding them
changes no model state. What it changes is whether the zones coincide with the
boundaries the Bank publishes, which is what the explorer now draws underneath.

The geometry itself comes from epm/resources/postprocess/, which is rebuilt from
the artifact by rebuild_reference_zones.py and rebuild_custom_zones.py. This
script only replays the per-model cut and reports what moved.

    python tools/rebuild_geometry.py                 # check every model
    python tools/rebuild_geometry.py --folder data_casa
    python tools/rebuild_geometry.py --apply

The exit code is non-zero when a model fails the alignment test, so it can be
run in CI: no zone may overlap another, and no zone may stick out past the
countries it belongs to.
"""
from __future__ import annotations

import argparse
import inspect
import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from epm.geodata.wb_boundaries import resolve_cache  # noqa: E402
from tools.rebuild_reference_zones import find_artifact  # noqa: E402

INPUT = _REPO / "epm" / "input"
AREA_CRS = "EPSG:6933"

# The floor the artifact itself sits at: the country outlines and the units are
# written at the same precision but not by the same operation, so a shared edge
# can land some ten metres off. Anything under this is that, not a zone drawn
# against the wrong map. See check_tiling in epm/geodata/wb_boundaries.py.
NOISE_KM2 = 5.0
# how much a zone may change before it is worth a line in the report
REPORT_SHARE = 0.005


def km2(geom, crs="EPSG:4326"):
    if geom is None or geom.is_empty:
        return 0.0
    return gpd.GeoSeries([geom], crs=crs).to_crs(AREA_CRS).iloc[0].area / 1e6


def targets(data_dir):
    """The (stem, zones file, linestring file) triples a model folder holds.

    A model always has the default pair, and one more pair per zcmap it defines
    -- some studies carry an alternative zoning next to the main one.
    """
    out = [(None, data_dir / "zones.geojson",
            data_dir / "linestring_countries.geojson")]
    for zcmap in sorted(data_dir.glob("zcmap*.csv")):
        stem = zcmap.stem
        out.append((stem, data_dir / f"zones_{stem}.geojson",
                    data_dir / f"linestring_{stem}.geojson"))
    return out


def zcmap_for(data_dir, stem):
    """The zone-country mapping a target is cut from."""
    path = data_dir / f"{stem or 'zcmap'}.csv"
    return path if path.exists() else data_dir / "zcmap.csv"


def regenerate(data_dir, stem, out_dir):
    """Cut one pair of layers into out_dir, from today's resources.

    Returns False when this model's postprocessing cannot cut the target, which
    is not a failure: create_geojson_for_tableau only grew `output_stem` partway
    through, so a branch that has not picked that up yet can still have its
    default pair rebuilt. Requiring every model to update its postprocessing
    first would make the geometry migration wait on an unrelated one.
    """
    from epm.postprocessing.create_geojson import create_geojson_for_tableau

    zcmap = zcmap_for(data_dir, stem)
    table = pd.read_csv(zcmap)
    zone_col = "zone" if "zone" in table.columns else "z"
    kwargs = dict(geojson_to_epm=None, zcmap=str(zcmap),
                  selected_zones=table[zone_col].unique().tolist(),
                  folder=data_dir.name, output_path=str(out_dir))
    if "output_stem" in inspect.signature(create_geojson_for_tableau).parameters:
        kwargs["output_stem"] = stem
    elif stem:
        return False
    create_geojson_for_tableau(**kwargs)
    return True


def zone_key(*frames):
    """The column that names a zone, in every frame given.

    Models do not agree on it -- some layers carry `z`, the ones cut from the
    reference polygons carry `ADMIN` -- and a layer written years ago may not
    use the same one as the layer replacing it.
    """
    for candidate in ("z", "ADMIN", "zone"):
        if all(candidate in f.columns for f in frames):
            return candidate
    return None


def compare(old_path, new_path):
    """What changed between the zone layer on disk and the one just cut."""
    new = gpd.read_file(new_path)
    if not old_path.exists():
        return new, [f"    new file, {len(new)} zones"], []

    old = gpd.read_file(old_path)
    key = zone_key(old, new)
    if key is None:
        # Nothing names the zones in both files, so there is no way to say which
        # zone moved. That is a schema change, not a geometry one, and it is the
        # model owner's call -- report it and compare nothing.
        return new, [f"    cannot compare: on disk {sorted(old.columns)}, "
                     f"rebuilt {sorted(new.columns)}"], []
    notes, problems = [], []
    gone = sorted(set(old[key]) - set(new[key]))
    added = sorted(set(new[key]) - set(old[key]))
    if gone:
        problems.append(f"    zones no longer drawn: {', '.join(gone)}")
    if added:
        notes.append(f"    zones now drawn: {', '.join(added)}")

    o = old.set_index(key).geometry
    for z, geom in new.set_index(key).geometry.items():
        if z not in o.index:
            continue
        before, after = km2(o[z]), km2(geom)
        if before and abs(after - before) / before > REPORT_SHARE:
            notes.append(f"    {z}: {before:,.0f} -> {after:,.0f} km2 "
                         f"({(after - before) / before:+.1%})")
    return new, notes, problems


def align(zones, countries):
    """The alignment test: zones tile, and stay inside the countries they claim.

    A zone that overlaps its neighbour double-counts whatever is drawn on it. A
    zone that reaches past its country's outline is drawn against a different
    map from the one under it, which is the whole failure this exercise is
    about.
    """
    problems = []
    key = zone_key(zones)
    if key is None:
        # The overlap and containment tests are about geometry, so they still
        # mean something on a layer whose zones are not named. Number them, and
        # do not fall back to some other column: picking `geometry` makes the
        # spatial join below join a frame to itself twice over.
        zones = zones.copy()
        key = "_row"
        zones[key] = [f"row {i}" for i in range(len(zones))]
    broken = zones[~zones.geometry.is_valid]
    if len(broken):
        problems.append(f"    invalid geometry: {', '.join(broken[key])}")

    hits = gpd.sjoin(zones[[key, "geometry"]], zones[[key, "geometry"]],
                     predicate="overlaps", how="inner")
    seen = set()
    for left, right in zip(hits.index, hits.index_right):
        pair = (min(left, right), max(left, right))
        if left == right or pair in seen:
            continue
        seen.add(pair)
        area = km2(zones.geometry[left].intersection(zones.geometry[right]))
        if area > NOISE_KM2:
            problems.append(f"    {zones[key][left]} and {zones[key][right]} "
                            f"overlap by {area:,.0f} km2")

    if "ISO_A3" in zones.columns and countries is not None:
        outline = countries.set_index("ISO_A3").geometry
        for r in zones.itertuples():
            iso = getattr(r, "ISO_A3", None)
            if iso not in outline.index:
                continue
            out = km2(r.geometry.difference(outline[iso]))
            if out > NOISE_KM2:
                problems.append(f"    {getattr(r, key)} reaches {out:,.0f} km2 "
                                f"outside {iso}")
    return problems


def run(artifact, data_dir, apply=False):
    """Rebuild and check one model folder. Returns the problems found."""
    countries = gpd.read_file(artifact / "countries_10m.geojson")
    if "STATUS" in countries:
        countries = countries[countries.STATUS.fillna("") != "non-determined"]
    countries = countries.dissolve("ISO_A3").reset_index()

    print(f"{data_dir.name}")
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for stem, zones_path, lines_path in targets(data_dir):
            if not zones_path.exists() and not lines_path.exists():
                continue
            if not regenerate(data_dir, stem, tmp):
                print(f"  {zones_path.name}: this model's create_geojson cannot "
                      "cut a named zcmap - skipped")
                continue
            fresh_zones = tmp / zones_path.name
            fresh_lines = tmp / lines_path.name
            zones, notes, found = compare(zones_path, fresh_zones)
            found += align(zones, countries)
            print(f"  {zones_path.name}: {len(zones)} zones")
            for line in notes + found:
                print(line)
            problems += [f"{data_dir.name}/{zones_path.name}: {p.strip()}"
                         for p in found]
            for src, dst in ((fresh_zones, zones_path),
                             (fresh_lines, lines_path)):
                if not src.exists():
                    continue
                if not dst.exists():
                    # Only ever replace a layer a model already ships. Adding
                    # one is a decision about what that model publishes, and
                    # data_test in particular is a fixture whose contents tests
                    # are written against.
                    print(f"    {dst.name} does not exist here - not created")
                elif apply and not found:
                    shutil.copyfile(src, dst)
                    print(f"    wrote {dst.relative_to(_REPO)}")

    # The off-grid layer is cut from the same units, but only after the zones
    # are: it is what the modelled zones leave over, so it has to be subtracted
    # from the ones just written, not from the ones they replaced.
    if (data_dir / "zones_offgrid.geojson").exists() and not problems:
        offgrid().build(artifact, data_dir, apply=apply)
    return problems


def offgrid():
    """The off-grid builder, which lives in a directory a hyphen keeps out of
    the import path."""
    import importlib.util
    path = _REPO / "pre-analysis" / "build_offgrid_zones.py"
    spec = importlib.util.spec_from_file_location("build_offgrid_zones", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", help="boundary artifact (default: the newest built)")
    ap.add_argument("--cache", help="where artifacts live (see wb_boundaries)")
    ap.add_argument("--folder", nargs="*", metavar="NAME",
                    help="model folders under epm/input/ (default: all with a zcmap)")
    ap.add_argument("--apply", action="store_true",
                    help="write the layers; without it nothing is written")
    args = ap.parse_args()

    artifact = find_artifact(resolve_cache(args.cache), args.artifact)
    print(f"artifact: {artifact}\n")

    folders = ([INPUT / n for n in args.folder] if args.folder else
               [p for p in sorted(INPUT.glob("data_*"))
                if (p / "zcmap.csv").exists()])
    problems = []
    for data_dir in folders:
        if not data_dir.exists():
            raise SystemExit(f"no such model folder: {data_dir}")
        problems += run(artifact, data_dir, apply=args.apply)
        print()

    if problems:
        print("alignment problems, nothing written for the layers concerned:")
        for line in problems:
            print(f"  {line}")
        sys.exit(1)
    if not args.apply:
        print("(report only; pass --apply to write)")


if __name__ == "__main__":
    main()
