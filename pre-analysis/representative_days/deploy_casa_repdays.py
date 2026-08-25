"""Move a representative-days run into the CASA build, and stop there.

WHY THIS DOES NOT WRITE INTO data_casa, which is what the Black Sea deploy script does.
The CASA deployment folder is generated: build.py reads build_casa.yaml, copies the 2020
reference and applies the declared verbs, and three of those verbs pull pHours,
pDemandProfile and pVREProfile straight out of data_build/extracted/. Writing into
data_casa directly would put hand-made files in a folder the next build.py --apply
overwrites without noticing -- the work would vanish, silently, at the next run of a
command nobody thinks of as destructive.

So this writes into data_build/extracted/, which is where the build looks, and leaves the
deployment to build.py. That also means the change is visible to tracker.py, docs.py and
assumptions.py like every other source, instead of appearing in the model from nowhere.

WHAT IT REFUSES TO DO. It stops if the run's hours do not add up to 8760, and it stops if
the season names do not match seasons_months.csv. Both are cheap to check here and
expensive to find later: input_verification.py compares the sum of pHours to 8760 with no
tolerance at all, and a season the rest of the build has never heard of would be carried
into every seasonal table by name.

WHAT CAN UNDO THIS, and it is the mirror of the trap above. extracted/pHours.csv and
extracted/pDemandProfile.csv are the two files build_demand.py writes: staging here
replaces its medoid selection, and re-running build_demand.py afterwards replaces ours
back, just as quietly. The order that holds is deploy last -- run the extractors, then
this, then build.py. The .bak_repdays copies are there so a mistaken order is
recoverable rather than merely detectable.

ENVIRONMENT. This runs in gams_env, not epm_env: the pipeline needs scikit-learn, scipy
and seaborn for the clustering and gams.transfer for the weight optimisation, and epm_env
carries none of them.

    conda run -n gams_env python deploy_casa_repdays.py --apply

Usage
    python deploy_casa_repdays.py            # show what would change
    python deploy_casa_repdays.py --apply    # write, then run build.py --apply
"""
from pathlib import Path
import argparse
import csv
import shutil

import pandas as pd

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
DATA_BUILD = REPO / "data_build"
EXTRACTED = DATA_BUILD / "extracted"
SRC = BASE / "output" / "casa"

FULL_YEAR = 8760

# The pipeline names its columns season/daytype; the model reads q/d. Everything else
# passes through untouched.
RENAME = {"season": "q", "daytype": "d"}

FILES = [
    ("pHours.csv", "pHours.csv"),
    ("pDemandProfile.csv", "pDemandProfile.csv"),
    ("pVREProfile.csv", "pVREProfile.csv"),
]


def declared_seasons():
    path = DATA_BUILD / "mappings" / "seasons_months.csv"
    with open(path, encoding="utf-8-sig") as fh:
        return [row["season"].strip() for row in csv.DictReader(fh)]


def check_hours(frame, seasons):
    """The two things that are cheap here and expensive downstream."""
    hour_columns = [c for c in frame.columns
                    if c.startswith("t") and c[1:].lstrip("0").isdigit()]
    total = frame[hour_columns].to_numpy().sum()
    if round(total) != FULL_YEAR:
        raise SystemExit(
            "pHours sums to {0:.0f} h, not {1}. input_verification.py refuses this with "
            "no tolerance; fix the weights before deploying.".format(total, FULL_YEAR))

    found = list(dict.fromkeys(frame["q"]))
    if set(found) != set(seasons):
        raise SystemExit(
            "the run carries seasons {0} but seasons_months.csv declares {1}. Re-run the "
            "driver against the current mapping.".format(found, seasons))
    print("  hours   {0:.0f} h over {1} season(s), {2} day type(s)".format(
        total, len(found), frame["d"].nunique()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="write into data_build/extracted (default: dry run)")
    args = parser.parse_args()

    if not SRC.exists():
        raise SystemExit("no run at {0}: run run_casa_repdays.py first".format(SRC))

    seasons = declared_seasons()
    staged = []
    for source_name, target_name in FILES:
        source = SRC / source_name
        if not source.exists():
            print("  skip    {0} (not produced by this run)".format(source_name))
            continue
        frame = pd.read_csv(source).rename(columns=RENAME)
        if source_name == "pHours.csv":
            check_hours(frame, seasons)
        target = EXTRACTED / target_name
        old = len(pd.read_csv(target)) if target.exists() else 0
        print("  stage   {0:<20} {1} rows (was {2})".format(target_name, len(frame), old))
        staged.append((frame, target))

    if not args.apply:
        print("\nDry run. Nothing written. Re-run with --apply to stage these, then:\n"
              "  cd {0} && python build.py --config build_casa.yaml --apply".format(
                  DATA_BUILD))
        return

    for frame, target in staged:
        if target.exists():
            backup = target.with_suffix(target.suffix + ".bak_repdays")
            shutil.copy2(target, backup)
            print("  backup  {0}".format(backup.name))
        frame.to_csv(target, index=False)
        print("  written {0}".format(target))

    print("\nStaged into data_build/extracted. The model does not see this yet. Now:\n"
          "  cd {0}\n"
          "  python build.py --config build_casa.yaml --apply\n"
          "  python docs.py --config build_casa.yaml\n"
          "  python assumptions.py --config build_casa.yaml".format(DATA_BUILD))
    print("\nDo not run build_demand.py after this without re-deploying: it writes the\n"
          "same two files and would put its own medoid days back.")


if __name__ == "__main__":
    main()
