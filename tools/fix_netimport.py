"""Repair pTransmissionMerged.csv for a run that has already been solved.

Bridge, not a fix. The fix is in generate_report.gms and output_treatment.py; this
brings runs that predate it up to the same content without re-solving, using the GDX
those runs already wrote. Delete it once every run of interest has been regenerated.

What was wrong, in generate_report.gms before the correction:

  * external net import read `imports - exports / 1e3`. Both legs are already in GWh,
    and the precedence bound the division to the export term alone, annihilating it --
    so a net exporter was published as a net importer. Turkiye showed +264 GWh from
    Bulgaria in 2035 against a true -1256.
  * internal net import carried a spurious `/ 1e3`, writing TWh under a GWh label.
  * internal net import took `-pInterchange(z, z2)`, one direction only, ignoring the
    return flow. That is a gross flow reported as a net: Armenia|Georgia published
    -1.26 where the true net was -1223.9 GWh.
  * pInterchangeExternalImports / …Exports were unloaded to the GDX but left out of the
    CSV `symbols` list, so no consumer could see per-direction external exchange and
    all of them fell back on the lossy pNetImport.

Usage:  python tools/fix_netimport.py <run_dir>
        (run_dir holds one directory per scenario, each with epmresults.gdx)

Idempotent: the untouched file is kept beside as .csv.orig and restored before each
pass, so running twice gives the same result as running once.
"""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

CSV_NAME = "pTransmissionMerged.csv"
IMP_ATTR = "InterchangeExternalImports"
EXP_ATTR = "InterchangeExternalExports"


def gdx(gdx_path, symbol):
    """(key1, key2, year) -> value, for a three-dimensional GAMS parameter."""
    r = subprocess.run(["gdxdump", str(gdx_path), "symb=%s" % symbol, "format=csv"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    out = {}
    for row in csv.reader(r.stdout.splitlines()[1:]):
        if len(row) >= 4:
            try:
                out[(row[0], row[1], row[2])] = float(row[3])
            except ValueError:
                pass
    return out


def patch(scen_dir):
    gdx_path = scen_dir / "epmresults.gdx"
    csv_path = scen_dir / "output_csv" / CSV_NAME
    if not gdx_path.exists() or not csv_path.exists():
        return "%s: skipped (no gdx or no csv)" % scen_dir.name

    # Restore first, so a second run starts from the same place the first one did.
    bak = csv_path.with_suffix(".csv.orig")
    if bak.exists():
        shutil.copy2(bak, csv_path)
    else:
        shutil.copy2(csv_path, bak)

    imp = gdx(gdx_path, "pInterchangeExternalImports")   # (zext, z, y) -> GWh into z
    exp = gdx(gdx_path, "pInterchangeExternalExports")   # (z, zext, y) -> GWh out of z

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "%s: skipped (empty csv)" % scen_dir.name
    fields = list(rows[0].keys())

    # Interchange(z, uni, y) is the flow z -> uni, in GWh. The country of each internal
    # zone is taken from the file itself so the rebuilt rows carry the same `c`.
    flow, country = {}, {}
    for r in rows:
        country.setdefault(r["z"], r["c"])
        if r["attribute"] == "Interchange":
            flow[(r["z"], r["uni"], r["y"])] = float(r["value"] or 0)

    ext_zones = set(k[0] for k in imp) | set(k[1] for k in exp)

    fixed_ext = fixed_int = 0
    for r in rows:
        if r["attribute"] != "NetImport":
            continue
        z, other, y = r["z"], r["uni"], r["y"]
        if other in ext_zones:
            net = imp.get((other, z, y), 0.0) - exp.get((z, other, y), 0.0)
            fixed_ext += 1
        else:
            # A real net: what came in from `other`, less what went out to it.
            net = flow.get((other, z, y), 0.0) - flow.get((z, other, y), 0.0)
            fixed_int += 1
        r["value"] = repr(net)

    # The per-direction rows the CSV never carried. Keyed the way output_treatment now
    # writes them: z is the internal zone, uni the external one, for both attributes.
    added = 0
    for (zext, z, y), v in sorted(imp.items()):
        rows.append({"c": country.get(z, z), "z": z, "attribute": IMP_ATTR,
                     "y": y, "uni": zext, "value": repr(v)})
        added += 1
    for (z, zext, y), v in sorted(exp.items()):
        rows.append({"c": country.get(z, z), "z": z, "attribute": EXP_ATTR,
                     "y": y, "uni": zext, "value": repr(v)})
        added += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    return ("%s: NetImport %d ext / %d int, +%d directional rows, %d external zones"
            % (scen_dir.name, fixed_ext, fixed_int, added, len(ext_zones)))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip().splitlines()[-4].strip())
    run = Path(sys.argv[1])
    scens = sorted(d for d in run.iterdir()
                   if d.is_dir() and (d / "epmresults.gdx").exists())
    if not scens:
        sys.exit("no scenario with a gdx under %s" % run)
    for d in scens:
        print(patch(d), flush=True)


if __name__ == "__main__":
    main()
