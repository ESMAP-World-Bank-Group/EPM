"""Rebuild a layer of sub-national model zones from World Bank administrative
units -- zones_custom.geojson by default, eapp_zones.geojson with --layer.

A custom zone is a group of administrative units: KAZ_N is eight Kazakh oblasts,
PAK_KAR is the district of Karachi. Until now those groups existed only as
polygons cut from Natural Earth, with the member units written down in a free
text property, and for the nine Turkish zones not written down at all. That is
not a recipe anyone can replay, and it is why the zones do not line up with the
boundaries the Bank publishes.

So the membership is made explicit, in two tables next to the layer:

    zones_custom_recipe.csv   one row per (zone, administrative unit), keyed on
                              the HASC code, which is stable across releases in
                              a way names are not
    zones_custom_meta.csv     one row per zone: the editorial fields that are
                              not derivable from geometry (why the split, which
                              grid, which plants)

and the layer becomes a build product: units dissolved, nothing simplified, so
a zone ends exactly where the country under it ends on the map.

    python tools/rebuild_custom_zones.py --derive   # recover the tables from
                                                    # today's polygons, once
    python tools/rebuild_custom_zones.py            # report
    python tools/rebuild_custom_zones.py --apply    # rebuild the layer
    python tools/rebuild_custom_zones.py --layer epm/resources/postprocess/eapp_zones.geojson --derive

--derive is the migration step, and its output is meant to be read before it is
trusted: it assigns a unit to the zone covering most of it, reports how much,
and flags every unit it had to cut. Zones drawn to a grid rather than to an
administrative limit will not come back exactly.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import make_valid

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from epm.geodata.wb_boundaries import resolve_cache  # noqa: E402
from tools.rebuild_reference_zones import find_artifact  # noqa: E402

RESOURCES = _REPO / "epm" / "resources" / "postprocess"
DEFAULT_LAYER = RESOURCES / "zones_custom.geojson"


def tables(layer):
    """The two tables that describe a layer, named after it."""
    return (layer.with_name(layer.stem + "_recipe.csv"),
            layer.with_name(layer.stem + "_meta.csv"))

# equal-area, so that "most of this unit" means what it says
AREA_CRS = "EPSG:6933"

# a unit belongs to the zone covering at least this much of it
CLAIM = 0.5
# below this it is noise along a shared edge, not a claim
TOUCH = 0.02
# a zone matching one country this closely, both ways, is that country
WHOLE = 0.95

META_FIELDS = ("epm_zone", "epm_country", "split_rationale", "grid_area",
               "model_plants", "caveat")
RECIPE_FIELDS = ("zone", "iso_a3", "level", "code", "name", "coverage")


def load_units(artifact):
    """The units zones may be built from -- whole countries, adm1, and the adm2
    subset -- in one frame. adm2 is only there for the countries whose zones cut
    below the first level, today Karachi.

    Whole countries are in there because most zones are one: an EAPP zone like
    Kenya is the country. Taken from the display layer it is exactly the polygon
    zones.geojson and the explorer basemap already use. Dissolving its counties
    would reach the same outline only by accident of the fitting rule, at ten
    times the size.
    """
    countries = gpd.read_file(artifact / "countries_10m.geojson")
    if "STATUS" in countries:
        # areas the Bank attributes to no country are not something to zone
        countries = countries[countries.STATUS.fillna("") != "non-determined"]
    countries = countries.copy()
    countries["CODE"] = countries.ISO_A3
    countries["NAME"] = countries.WB_NAME
    countries["GAUL"] = 0
    countries["LEVEL"] = "country"
    frames = [countries[["ISO_A3", "CODE", "NAME", "GAUL", "LEVEL", "geometry"]]]

    adm1 = gpd.read_file(artifact / "adm1.geojson")
    adm1 = adm1.rename(columns={"HASC_1": "CODE", "NAM_1": "NAME", "GAUL_1": "GAUL"})
    adm1["LEVEL"] = "adm1"
    frames.append(adm1[["ISO_A3", "CODE", "NAME", "GAUL", "LEVEL", "geometry"]])

    path = artifact / "adm2_subset.geojson"
    if path.exists():
        adm2 = gpd.read_file(path)
        adm2["PARENT"] = adm2.HASC_1
        adm2 = adm2.rename(columns={"HASC_2": "CODE", "NAM_2": "NAME",
                                    "GAUL_2": "GAUL"})
        adm2["LEVEL"] = "adm2"
        frames.append(adm2[["ISO_A3", "CODE", "NAME", "GAUL", "LEVEL", "PARENT",
                            "geometry"]])
    units = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=adm1.crs)
    if "PARENT" not in units:
        units["PARENT"] = ""
    units["PARENT"] = units.PARENT.fillna("")
    return valid(key_units(units).to_crs(AREA_CRS))


def slug(text):
    return "".join(c if c.isalnum() else "-" for c in (text or "").strip())[:24]


def key_units(units):
    """Give every unit a code a recipe can name it by, and make it unique.

    HASC is the code to use where it exists, but the Bank's layer leaves it
    blank on 550 of 3439 first-level units and GAUL on 709, so neither alone
    can key the table. The chain falls back through GAUL to the unit's own
    name, which is always present. Units that end up sharing a code are the
    same unit arriving in several pieces -- Luanda twice -- and are merged.
    """
    code = units.CODE.fillna("").astype(str).str.strip()
    # GAUL arrives as a number, and 0 is its way of saying "none"
    gaul = units.GAUL.fillna(0).astype("int64").astype(str)
    units["CODE"] = [
        c if c else (f"{i}.G{g}" if g and g != "0" else f"{i}.N{slug(n)}")
        for c, g, i, n in zip(code, gaul, units.ISO_A3.fillna(""), units.NAME)
    ]
    # a code shared by units of different names is not a key: keep them apart
    clash = units.groupby("CODE").NAME.transform("nunique") > 1
    units.loc[clash, "CODE"] = (units.loc[clash, "CODE"] + "."
                                + units.loc[clash, "NAME"].map(slug))

    merged = units.dissolve(by=["LEVEL", "CODE"], aggfunc="first").reset_index()
    if len(merged) < len(units):
        print(f"  merged {len(units) - len(merged)} unit(s) arriving in several "
              f"pieces under one code")
    return merged


def valid(gdf):
    """Repair self-intersections before any overlay.

    The zones being read are hand-drawn Natural Earth cut-outs and a few of them
    are not clean; an overlay against them fails outright rather than degrading.
    """
    broken = ~gdf.geometry.is_valid
    if broken.any():
        print(f"  repaired {int(broken.sum())} invalid geometries")
        gdf.loc[broken, "geometry"] = gdf.loc[broken, "geometry"].apply(make_valid)
    return gdf


def coverage(units, zone_geom):
    """How much of each unit the zone covers, for the units it touches at all."""
    idx = list(units.sindex.query(zone_geom, predicate="intersects"))
    if not idx:
        return pd.DataFrame(columns=["pos", "share"])
    sub = units.iloc[idx]
    inter = sub.geometry.intersection(zone_geom).area
    share = (inter / sub.geometry.area).fillna(0.0)
    return pd.DataFrame({"pos": idx, "share": share.values})


def whole_country(countries, geom):
    """The country this zone is, if it is one.

    Judged both ways -- the zone covers the country and the country covers the
    zone -- so a zone that is only part of a country, or straddles two, falls
    through to the units it is really made of.
    """
    best = None
    for r in coverage(countries, geom).itertuples():
        if r.share < WHOLE:
            continue
        unit = countries.iloc[r.pos]
        inside = geom.intersection(unit.geometry).area / geom.area
        if inside >= WHOLE and (best is None or r.share > best[1]):
            best = (unit, r.share)
    return best


def derive(artifact, layer):
    """Recover the recipe from the polygons that exist today.

    Only from the hand-made ones. Deriving from a layer this tool built would
    read back its own output -- every unit fits perfectly, the flags that make
    the tables worth reading go quiet, and a zone the recipe lost stays lost.
    """
    doc = json.loads(layer.read_text(encoding="utf-8"))
    if "artifact" in (doc.get("source") or ""):
        raise SystemExit(
            f"{layer.name} was already built from {doc['source']}."
            " Deriving from it would just read back the recipe it came"
            " from. Restore the original layer first:"
            f" git checkout HEAD -- {layer.name}")
    units = load_units(artifact)
    countries = units[units.LEVEL == "country"].reset_index()
    adm1 = units[units.LEVEL == "adm1"].reset_index()
    adm2 = units[units.LEVEL == "adm2"].reset_index()
    zones = valid(gpd.read_file(layer).to_crs(AREA_CRS))

    rows, notes = [], []
    for zone in zones.itertuples():
        name = zone.ADMIN
        geom = zone.geometry

        entire = whole_country(countries, geom)
        if entire is not None:
            unit, share = entire
            rows.append({"zone": name, "iso_a3": unit.ISO_A3, "level": "country",
                         "code": unit.CODE, "name": unit.NAME,
                         "coverage": f"{share:.3f}"})
            continue

        picked, cut = [], []
        for r in coverage(adm1, geom).itertuples():
            unit = adm1.iloc[r.pos]
            if r.share >= CLAIM:
                picked.append((unit, r.share))
            elif r.share > TOUCH:
                cut.append((unit, r.share))

        # A unit the zone only half covers means the zone cuts below adm1. If
        # the second level was downloaded for that country, use it there.
        for unit, share in cut:
            children = adm2[adm2.PARENT == unit.CODE]
            if children.empty:
                notes.append(f"    {name}: {unit.NAME} ({unit.CODE}) is "
                             f"{share:.0%} inside and has no adm2 to cut with")
                if share >= CLAIM:
                    picked.append((unit, share))
                continue
            kids = children.reset_index(drop=True)
            for r in coverage(kids, geom).itertuples():
                child = kids.iloc[r.pos]
                if r.share >= CLAIM:
                    picked.append((child, r.share))
                elif r.share > TOUCH:
                    # partly in, so it goes to whichever neighbouring zone holds
                    # most of it -- worth reading, it is a modelling choice
                    notes.append(f"    {name}: {child.NAME} ({child.CODE}) is "
                                 f"{r.share:.0%} inside, left to another zone")

        if not picked:
            notes.append(f"    {name}: no administrative unit is mostly inside it")
        iso = (pd.Series([u.ISO_A3 for u, _ in picked]).mode()
               if picked else pd.Series(dtype=str))
        for unit, share in sorted(picked, key=lambda p: (p[0].ISO_A3, p[0].NAME)):
            rows.append({"zone": name, "iso_a3": unit.ISO_A3, "level": unit.LEVEL,
                         "code": unit.CODE, "name": unit.NAME,
                         "coverage": f"{share:.3f}"})
        # how much of the zone the chosen units fail to cover: the zone was
        # drawn to something other than an administrative limit
        if picked:
            covered = gpd.GeoSeries([u.geometry for u, _ in picked],
                                    crs=units.crs).union_all()
            missed = geom.difference(covered).area / geom.area
            if missed > 0.02:
                notes.append(f"    {name}: {missed:.0%} of the zone is outside "
                             f"the units picked for it"
                             + (f" (iso {iso.iloc[0]})" if len(iso) else ""))
    rows = dedupe(rows, notes)
    rows = resolve_overlaps(rows, units, notes)
    audit(rows, units, notes)
    return rows, notes, zones


def dedupe(rows, notes):
    """A zone drawn twice is one zone.

    eapp_zones.geojson carries Somalia as two identical features; both derive
    the same members, and a recipe that lists a unit twice would just union it
    with itself.
    """
    seen, out = set(), []
    for r in rows:
        key = (r["zone"], r["level"], r["code"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    if len(out) < len(rows):
        notes.append(f"    dropped {len(rows) - len(out)} membership(s) listed "
                     f"twice by a zone the layer draws more than once")
    return out


def resolve_overlaps(rows, units, notes):
    """Stop a zone from containing another one.

    A zone that cuts below adm1 takes adm2 children, while its neighbour takes
    the whole parent -- PAK_S claims Sindh, PAK_KAR claims the Karachi
    districts inside it. Left alone the neighbour would swallow the zone. So a
    whole-unit claim whose children are spoken for is expanded into the
    children nobody took.
    """
    adm2 = units[units.LEVEL == "adm2"]
    owner = {(r["level"], r["code"]): r["zone"] for r in rows}
    out = []
    for r in rows:
        kids = adm2[adm2.PARENT == r["code"]] if r["level"] == "adm1" else []
        contested = [k for k in getattr(kids, "itertuples", list)()
                     if owner.get(("adm2", k.CODE), r["zone"]) != r["zone"]]
        if not contested:
            out.append(r)
            continue
        free = [k for k in kids.itertuples() if ("adm2", k.CODE) not in owner]
        notes.append(f"    {r['zone']}: {r['name']} ({r['code']}) split -- "
                     f"{len(contested)} district(s) belong to "
                     f"{', '.join(sorted({owner[('adm2', k.CODE)] for k in contested}))}, "
                     f"the other {len(free)} stay here")
        for k in free:
            out.append({"zone": r["zone"], "iso_a3": k.ISO_A3, "level": "adm2",
                        "code": k.CODE, "name": k.NAME, "coverage": r["coverage"]})
    return out


def audit(rows, units, notes):
    """Two things a set of zones must satisfy, reported rather than assumed."""
    seen = {}
    for r in rows:
        seen.setdefault((r["level"], r["code"]), []).append(r["zone"])
    for key, zones in sorted(seen.items()):
        if len(zones) > 1:
            notes.append(f"    {key[1]} is claimed by {', '.join(zones)}")

    # a unit of a country the zones cover, that no zone took and whose children
    # no zone took either: a hole in the country's zoning
    claimed = set(seen)
    parents = {units.loc[units.CODE == c, "PARENT"].iloc[0]
               for lvl, c in claimed if lvl == "adm2"
               and (units.CODE == c).any()}
    # A country taken whole and also cut into pieces is two ways of drawing the
    # same place -- the EAPP layer carries Somalia and its split side by side.
    # They overlap by construction and the mapping picks one set, so it is worth
    # saying once per country rather than once per unit.
    entire = {r["iso_a3"] for r in rows if r["level"] == "country"}
    split = {r["iso_a3"] for r in rows if r["level"] != "country"}
    for iso in sorted(entire & split):
        def zones_at(whole):
            return ", ".join(sorted({r["zone"] for r in rows
                                     if r["iso_a3"] == iso
                                     and (r["level"] == "country") == whole}))
        notes.append(f"    {iso}: {zones_at(True)} covers the same ground as "
                     f"{zones_at(False)} -- alternative representations")

    # a unit of a country the zones cut up, that no zone took and whose children
    # no zone took either: a hole in that country's zoning
    for iso in sorted(split):
        for u in units[(units.ISO_A3 == iso) & (units.LEVEL == "adm1")].itertuples():
            if ("adm1", u.CODE) not in claimed and u.CODE not in parents:
                notes.append(f"    {iso}: {u.NAME} ({u.CODE}) belongs to no zone")


def derive_meta(layer):
    """The fields no geometry can give back, taken off today's polygons."""
    out = {}
    for f in json.loads(layer.read_text(encoding="utf-8"))["features"]:
        p = f["properties"]
        row = {"zone": p.get("ADMIN", "")}
        row.update({k: (p.get(k) or "") for k in META_FIELDS})
        # a zone the layer draws twice keeps whichever copy says something
        kept = out.get(row["zone"])
        if kept is None or not any(kept[k] for k in META_FIELDS):
            out[row["zone"]] = row
    return list(out.values())


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.relative_to(_REPO)} ({len(rows)} rows)")


def read_csv(path):
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(_REPO)} -- run --derive first")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def rebuild(artifact, layer):
    """Dissolve the units of each zone into the layer."""
    recipe_path, meta_path = tables(layer)
    units = load_units(artifact).set_index(["LEVEL", "CODE"])
    recipe = read_csv(recipe_path)
    meta = {r["zone"]: r for r in read_csv(meta_path)}

    features, missing = [], []
    for zone in dict.fromkeys(r["zone"] for r in recipe):
        members = [r for r in recipe if r["zone"] == zone]
        geoms, names = [], []
        for r in members:
            key = (r["level"], r["code"])
            if key not in units.index:
                missing.append(f"{zone}: {r['level']} {r['code']} ({r['name']})")
                continue
            geoms.append(units.loc[key].geometry)
            names.append(r["name"])
        if not geoms:
            continue
        geom = gpd.GeoSeries(geoms, crs=units.crs).union_all()
        geom = gpd.GeoSeries([geom], crs=units.crs).to_crs("EPSG:4326").iloc[0]
        levels = {r["level"] for r in members}
        entire = levels == {"country"}
        props = {"ADMIN": zone, "ISO_A3": members[0]["iso_a3"],
                 "admin_source": ("World Bank Official Boundaries" if entire
                                  else "World Bank GAD")
                                 + f", artifact {artifact.name}"}
        if not entire:
            # for a whole country this would only repeat ADMIN
            props["admin_units"] = ", ".join(sorted(names))
        props.update({k: v for k, v in meta.get(zone, {}).items()
                      if k != "zone" and v})
        features.append({"type": "Feature", "properties": props,
                         "geometry": json.loads(gpd.GeoSeries([geom]).to_json())
                                          ["features"][0]["geometry"]})
    return features, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact")
    ap.add_argument("--cache")
    ap.add_argument("--layer", default=str(DEFAULT_LAYER),
                    help="the layer to rebuild; its two tables are named after "
                         "it (default: zones_custom.geojson)")
    ap.add_argument("--derive", action="store_true",
                    help="recover the recipe tables from today's polygons")
    ap.add_argument("--apply", action="store_true",
                    help="write the layer; without it nothing is written")
    args = ap.parse_args()

    artifact = find_artifact(resolve_cache(args.cache), args.artifact)
    layer = Path(args.layer).resolve()
    recipe_path, meta_path = tables(layer)
    print(f"artifact: {artifact}")
    print(f"layer:    {layer.relative_to(_REPO)}")

    if args.derive:
        rows, notes, zones = derive(artifact, layer)
        print(f"  {len(zones)} zones -> {len(rows)} unit memberships")
        if notes:
            print("  read these before trusting the tables:")
            print("\n".join(notes))
        if not args.apply:
            print("\n(report only; pass --apply to write the tables)")
            return
        write_csv(recipe_path, rows, RECIPE_FIELDS)
        write_csv(meta_path, derive_meta(layer), ("zone",) + META_FIELDS)
        return

    features, missing = rebuild(artifact, layer)
    print(f"  {len(features)} zones rebuilt")
    if missing:
        raise SystemExit("units the artifact does not have:\n  "
                         + "\n  ".join(missing))
    if not args.apply:
        print("\n(report only; pass --apply to write)")
        return
    doc = {"type": "FeatureCollection", "name": layer.stem,
           "source": f"World Bank GAD, artifact {artifact.name}",
           "crs": {"type": "name",
                   "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
           "features": features}
    layer.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False),
                     encoding="utf-8")
    print(f"  wrote {layer.relative_to(_REPO)} "
          f"({layer.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
