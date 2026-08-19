"""Rebuild the reference zone polygons in epm/resources/postprocess/ from a
World Bank boundary artifact.

    zones.geojson               one polygon per country, the layer every
                                country-level EPM zone is cut from
    zones_undetermined.geojson  the areas the Bank attributes to no country

The artifact is built by `python -m epm.geodata.wb_boundaries`; see that module
for what is in it and why. This script only re-cuts the reference layers from
it, which is the step that moves EPM off Natural Earth.

What is deliberately preserved
------------------------------
The property schema (ADMIN, ISO_A3, ISO_A2) and, above all, the ADMIN strings.
`geojson_to_epm.csv` joins zones to polygons by that name -- 'Democratic
Republic of the Congo', not the Bank's 'Congo, Dem. Rep.' -- so the names are
carried over from the file being replaced, keyed on ISO_A3. Only the geometry
changes. A country the Bank knows and the old file did not gets the Bank's own
name, and is reported.

Countries that disappear are reported too, and are an error if any mapping
still refers to them: the old Natural Earth layer carried entities the Bank
does not recognise as countries (Northern Cyprus, Siachen Glacier, Somaliland,
...). Those are a modelling decision, not something this script may silently
drop -- see docs and the --allow-dropped escape hatch.

Usage (from the EPM repo root):
    python tools/rebuild_reference_zones.py               # report, write nothing
    python tools/rebuild_reference_zones.py --apply
    python tools/rebuild_reference_zones.py --artifact PATH --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from epm.geodata.wb_boundaries import resolve_cache  # noqa: E402

RESOURCES = _REPO / "epm" / "resources" / "postprocess"
ZONES = RESOURCES / "zones.geojson"
UNDETERMINED = RESOURCES / "zones_undetermined.geojson"


def find_artifact(cache, explicit=None):
    """The artifact to cut from: an explicit one, else the most recent built."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "source.json").exists():
            raise SystemExit(f"not a boundary artifact: {path}")
        return path
    built = sorted(p for p in cache.glob("wb_boundaries_*")
                   if (p / "source.json").exists())
    if not built:
        raise SystemExit(
            f"no boundary artifact under {cache}\n"
            "build one first: python -m epm.geodata.wb_boundaries")
    return built[-1]


def read_features(path):
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def load_legacy(path):
    """ISO_A3 -> (ADMIN, ISO_A2) from the file being replaced, plus the
    unattributed entries, which have no code to be keyed on."""
    by_iso, unattributed = {}, []
    for f in read_features(path):
        p = f["properties"]
        iso = (p.get("ISO_A3") or "").strip()
        if not iso or iso == "-99":
            unattributed.append(p.get("ADMIN", ""))
        else:
            by_iso.setdefault(iso, (p.get("ADMIN", ""), p.get("ISO_A2", "")))
    return by_iso, unattributed


def referenced_names():
    """What the mappings ask for, and what the other layers already supply.

    A name only has to survive in zones.geojson if some mapping asks for it and
    nothing else provides it. The custom layers are merged into the same name
    space -- zones_custom.geojson by read_plot_specs(), eapp_zones.geojson by
    studies that pass it as zone_map -- so a name they carry is theirs to keep,
    and its geometry is their rebuild's problem, not this one's.
    """
    asked = {}
    for csv_path in sorted(RESOURCES.glob("*geojson_to_epm*.csv")):
        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("source_name") or "").strip()
                if name:
                    asked.setdefault(name, set()).add(csv_path.name)
    supplied = {}
    for extra in ("zones_custom.geojson", "eapp_zones.geojson"):
        path = RESOURCES / extra
        if path.exists():
            for f in read_features(path):
                name = f["properties"].get("ADMIN", "")
                if name:
                    supplied.setdefault(name, set()).add(extra)
    return asked, supplied


def build(artifact, legacy_by_iso, iso_a2_fallback):
    """Country features carrying the legacy names, plus the unattributed areas."""
    zones, undetermined, added = [], [], []
    for f in read_features(artifact / "countries_10m.geojson"):
        p = f["properties"]
        if p.get("STATUS") == "non-determined":
            undetermined.append({
                "type": "Feature",
                "properties": {"ADMIN": p["WB_NAME"], "STATUS": "non-determined"},
                "geometry": f["geometry"],
            })
            continue
        iso = p["ISO_A3"]
        if iso in legacy_by_iso:
            admin, iso_a2 = legacy_by_iso[iso]
        else:
            admin, iso_a2 = p["WB_NAME"], iso_a2_fallback.get(iso, "")
            added.append(f"{iso} {admin}")
        zones.append({
            "type": "Feature",
            "properties": {"ADMIN": admin, "ISO_A3": iso, "ISO_A2": iso_a2},
            "geometry": f["geometry"],
        })
    return zones, undetermined, added


def document(name, features, artifact):
    return {
        "type": "FeatureCollection",
        "name": name,
        "source": f"World Bank Official Boundaries, artifact {artifact.name}",
        "crs": {"type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", help="boundary artifact directory "
                                       "(default: the most recent in the cache)")
    ap.add_argument("--cache", help="where artifacts live (see wb_boundaries)")
    ap.add_argument("--apply", action="store_true",
                    help="write the files; without it nothing is written")
    ap.add_argument("--allow-dropped", nargs="*", default=[], metavar="NAME",
                    help="names that may disappear even though a mapping uses them")
    args = ap.parse_args()

    artifact = find_artifact(resolve_cache(args.cache), args.artifact)
    print(f"artifact: {artifact}")

    legacy_by_iso, legacy_unattributed = load_legacy(ZONES)
    # ISO_A2 is not carried by the display layer; take it from the full-detail
    # adm0 for any country the old file did not have.
    iso_a2_fallback = {f["properties"]["ISO_A3"]: f["properties"].get("ISO_A2", "")
                       for f in read_features(artifact / "adm0.geojson")}

    zones, undetermined, added = build(artifact, legacy_by_iso, iso_a2_fallback)
    kept = {f["properties"]["ADMIN"] for f in zones}
    kept_iso = {f["properties"]["ISO_A3"] for f in zones}
    # An entry the old file carried without a code can still come back under
    # the same name once the Bank gives it one -- Kosovo does. Losing the
    # feature and keeping the name is not a drop.
    dropped = [name for name in
               ([n for iso, (n, _) in sorted(legacy_by_iso.items())
                 if iso not in kept_iso] + legacy_unattributed)
               if name not in kept]

    print(f"  {len(zones)} countries, {len(undetermined)} non-determined areas")
    if added:
        print(f"  {len(added)} added, named by the Bank: {', '.join(sorted(added))}")

    asked, supplied = referenced_names()
    blocking = []
    if dropped:
        print(f"  {len(dropped)} dropped:")
        for name in sorted(dropped):
            where, by = asked.get(name), supplied.get(name)
            note = ""
            if where:
                note = f"   <-- asked for by {', '.join(sorted(where))}"
                if by:
                    note += f", supplied by {', '.join(sorted(by))}"
            print(f"    {name}{note}")
            if where and not by and name not in args.allow_dropped:
                blocking.append(name)
    if blocking:
        raise SystemExit(
            "\nStop: the mappings still refer to polygons the Bank's layer does not\n"
            "have. Each one is a modelling decision -- rebuild the zone from the\n"
            "sub-national units of the country it sits in, or rename it -- not\n"
            "something this script may drop on its own.\n"
            f"  {', '.join(sorted(blocking))}")

    if not args.apply:
        print("\n(report only; pass --apply to write)")
        return

    ZONES.write_text(json.dumps(document("zones", zones, artifact),
                                separators=(",", ":"), ensure_ascii=False),
                     encoding="utf-8")
    UNDETERMINED.write_text(
        json.dumps(document("zones_undetermined", undetermined, artifact),
                   separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    for path in (ZONES, UNDETERMINED):
        print(f"  wrote {path.relative_to(_REPO)} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
