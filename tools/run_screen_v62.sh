#!/usr/bin/env bash
# Sensitivity screen on the v6.2 key cases, five-year grid: 76 runs in six lots, one lot after the other.
#
#     bash tools/run_screen_v62.sh              every lot, in the order below
#     bash tools/run_screen_v62.sh A1 C2        only the lots named, in the order given
#     bash tools/run_screen_v62.sh --list       print the plan and what is already there, run nothing
#
# For the server, inside tmux or screen, with the conda environment of the model active. Each lot is one
# call of tools/run_parallel.py: one zip per scenario, epm/output/v62s5_<scenario>_<date>.zip, then one zip
# per lot, epm/output/v62s5_lot<lot>_<date>.zip, that holds the scenario zips of the lot.
#
# Resumable: a scenario whose zip is complete is skipped, so the same command can be relaunched. A scenario
# that fails does not stop the others and a lot that fails does not stop the next lot. The last lines give
# the count of zips per lot. Logs and status.csv: epm/output/_parallel/v62s5_<date>.
#
# Lots. A scenario column reads <Scope>_<Case>_<Sensitivity>_v62_5y, four cases per sensitivity
# (Base, BSSC, GECO, Onland2R).
#   A1  Ext   4 runs  EUCBAM                                                    scenarios_v62_screen1.csv
#   A2  Ext  24 runs  EU24NT EU24DE EU24GA EU24NTCBAM EU26LEV EU26HEV           scenarios_v62_screen2.csv
#   C2  Int   8 runs  EU24NT GeorgiaPolicy                                      scenarios_v62_screen2.csv
#   B   Int  12 runs  ArmeniaGasV61 NoCTN RussiaOpen                            scenarios_v62_screen1.csv
#   C1  Int  12 runs  EUDataCentres EUNuclearDelay EUREPlan                     scenarios_v62_screen1.csv
#   D   Int  16 runs  WorldGasHigh WorldGasLow CaucasusGasNetback CaucasusRESigned  scenarios_v62_screen1.csv
#
# SCREEN_JOBS (default 3) sets the scenarios alive at the same time. SCREEN_DATE (default 20261127) is the
# date of the labels: keep the date of the first launch when relaunching, or finished runs start again.
# PYTHON (default python) is the interpreter.
set -u
cd "$(dirname "$0")/.." || exit 2

DATE=${SCREEN_DATE:-20261127}
JOBS=${SCREEN_JOBS:-3}
PY=${PYTHON:-python}
PREFIX=v62s5
FOLDER=epm/input/data_blacksea
CASES="Base BSSC GECO Onland2R"
ORDER="A1 A2 C2 B C1 D"

define() {  # sets SCOPE, FILE and SENS for the lot named
    case "$1" in
        A1) SCOPE=Ext; FILE=scenarios_v62_screen1.csv; SENS="EUCBAM" ;;
        A2) SCOPE=Ext; FILE=scenarios_v62_screen2.csv; SENS="EU24NT EU24DE EU24GA EU24NTCBAM EU26LEV EU26HEV" ;;
        C2) SCOPE=Int; FILE=scenarios_v62_screen2.csv; SENS="EU24NT GeorgiaPolicy" ;;
        B)  SCOPE=Int; FILE=scenarios_v62_screen1.csv; SENS="ArmeniaGasV61 NoCTN RussiaOpen" ;;
        C1) SCOPE=Int; FILE=scenarios_v62_screen1.csv; SENS="EUDataCentres EUNuclearDelay EUREPlan" ;;
        D)  SCOPE=Int; FILE=scenarios_v62_screen1.csv; SENS="WorldGasHigh WorldGasLow CaucasusGasNetback CaucasusRESigned" ;;
        *)  echo "unknown lot: $1 (lots: $ORDER)"; exit 2 ;;
    esac
}

columns() {  # sets COLS and ZIPS for the lot defined, sensitivity by sensitivity
    COLS=""; ZIPS=""
    for s in $SENS; do
        for c in $CASES; do
            COLS="$COLS ${SCOPE}_${c}_${s}_v62_5y"
            ZIPS="$ZIPS epm/output/${PREFIX}_${SCOPE}_${c}_${s}_v62_5y_${DATE}.zip"
        done
    done
}

LIST=""
LOTS=""
for arg in "$@"; do
    if [ "$arg" = "--list" ]; then LIST="--list"; else LOTS="$LOTS $arg"; fi
done
[ -n "$LOTS" ] || LOTS=$ORDER

# Before anything runs: every lot is known and every column is in its scenario file.
gap=0
for lot in $LOTS; do
    define "$lot"
    columns
    if [ ! -f "$FOLDER/$FILE" ]; then
        echo "lot $lot: scenario file missing, $FOLDER/$FILE (git pull, then dvc pull -r r2)"
        gap=1
        continue
    fi
    header=",$(head -n 1 "$FOLDER/$FILE" | tr -d '\r'),"
    for col in $COLS; do
        case "$header" in
            *",$col,"*) ;;
            *) echo "lot $lot: column not in $FILE: $col"; gap=1 ;;
        esac
    done
done
if [ "$gap" -ne 0 ]; then
    echo "STOPPED before any run: the input folder does not hold the screen (see the lines above)"
    exit 2
fi

echo "screen v6.2, lots:$LOTS, $JOBS at a time, labels ${PREFIX}_<scenario>_${DATE}, $(date '+%Y-%m-%d %H:%M:%S')"
summary=""
bad=0
for lot in $LOTS; do
    define "$lot"
    columns
    echo ""
    echo "=== lot $lot ($SCOPE, $FILE), $(date '+%Y-%m-%d %H:%M:%S')"
    # shellcheck disable=SC2086
    "$PY" tools/run_parallel.py --scenarios "$FILE" --prefix "$PREFIX" --date "$DATE" --jobs "$JOBS" $LIST -S $COLS
    code=$?
    [ -n "$LIST" ] && continue
    found=""; have=0; want=0
    for z in $ZIPS; do
        want=$((want + 1))
        if [ -f "$z" ]; then found="$found $z"; have=$((have + 1)); fi
    done
    lotzip="epm/output/${PREFIX}_lot${lot}_${DATE}.zip"
    if [ "$have" -gt 0 ]; then
        # shellcheck disable=SC2086
        "$PY" -m zipfile -c "$lotzip" $found
    fi
    [ "$have" -eq "$want" ] && [ "$code" -eq 0 ] || bad=1
    summary="$summary
lot $lot: $have of $want scenario zips, launcher exit $code, $lotzip"
done

if [ -z "$LIST" ]; then
    echo ""
    echo "=== end of the screen, $(date '+%Y-%m-%d %H:%M:%S')$summary"
    [ "$bad" -eq 0 ] && echo "every lot complete" || echo "at least one lot is incomplete: relaunch the same command, finished runs are skipped"
fi
exit "$bad"
