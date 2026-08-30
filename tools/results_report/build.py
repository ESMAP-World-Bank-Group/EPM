# -*- coding: utf-8 -*-
"""Turn an EPM run into the results report.

    python build.py                                  # latest run, default scopes
    python build.py --run simulations_run_20260819_204446
    python build.py --scenarios LC_Baseline,LC_BSTN --countries Georgia,Armenia
    python build.py --no-extract                     # reuse the cache, redraw only

Extraction is the slow half (it streams ~110 MB of hourly dispatch per
scenario), so it is skipped when a cache for the run already exists unless
--force is passed.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import extract          # noqa: E402
import render           # noqa: E402

DEFAULT_OUT = (HERE.parents[2] / "Data" / "results" / "results_review.html")


def latest_run():
    runs = sorted(p for p in extract.OUTVIEW.glob("simulations_run_*")
                  if p.is_dir() and (p / "summary.csv").exists())
    if not runs:
        sys.exit("aucun run dans %s" % extract.OUTVIEW)
    return runs[-1].name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="nom du dossier de run")
    ap.add_argument("--scenarios", default="baseline,LC_Iso")
    ap.add_argument("--countries", default="Georgia",
                    help="pays a detailler, ou 'all'")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-extract", action="store_true",
                    help="reutilise le cache existant")
    ap.add_argument("--force", action="store_true",
                    help="reconstruit le cache meme s'il existe")
    a = ap.parse_args()

    run = a.run or latest_run()
    cache = HERE / "cache" / ("%s.json" % run)
    countries = None if a.countries == "all" else \
        [c.strip() for c in a.countries.split(",") if c.strip()]

    if not a.no_extract and (a.force or not cache.exists()):
        argv = ["--run", run, "--scenarios", a.scenarios]
        if countries:
            argv += ["--countries", ",".join(countries)]
        sys.argv = ["extract.py"] + argv
        extract.main()
    elif not cache.exists():
        sys.exit("pas de cache pour %s ; relance sans --no-extract" % run)
    else:
        print("cache reutilise : %s" % cache.name)

    shown = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    out = render.build(cache, a.out, countries=countries,
                       scenarios=shown)
    print("rapport : %s (%.1f Mo)" % (out, out.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
