"""Model capacity against the published national plan, on the plan milestones.

The deck carried this comparison as a screenshot of the HTML report, so it could
not be refreshed with the run.  This redraws it: one group of stacked bars per
plan year, the plan on the left and the model next to it in the same colours,
so the gap reads per technology and not only on the total.

Where the CESI Green Energy Corridor study modelled the country (Georgia and
Azerbaijan), its own long term fleet is added as a third bar, read from the
CESI register kept under DVC (no study figure lives in this file).  The study
reports 2032, 2036 and 2040, the three link years, so each study bar sits in
the nearest plan group and carries its own year.  Task 6 Figure 5-1
(Azerbaijan) excludes the export hub, Figure 5-3 (Georgia) includes the
additional PV and wind of Figure 5-4; electrolysers are not generation and are
left out.

The plan reports Hydro and Solar as single lines while the model splits them, so
the model technologies are merged into the plan's vocabulary before stacking.

    python slide_ndp.py --scope Georgia
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runcfg  # noqa: E402

PLANS = HERE / "reference" / "national_plans.json"
STUDY = runcfg.ROOT / "epm" / "input" / "data_blacksea" / "cesi" / "cesi_register.yaml"
STUDY_YEARS = ["2032", "2036", "2040"]  # the GEC link years, the study's anchors
STUDY_LABEL = "CESI"
STUDY_ZONE = {"Georgia": "Georgia", "Azerbaijan": "AzerbaijanMain"}
STUDY_ELEMENTS = {"VRE ceilings": None, "Gas fleet": "Gas", "Hydro": "Hydro",
                  "BESS": "Battery", "BESS, PSH": "Battery"}
STUDY_TECH = {"PV": "Solar", "OnshoreWind": "Wind", "OffshoreWind": "Wind"}

COLORS = {
    "Nuclear": "#C8A8F0", "Coal": "#808890", "Gas": "#9A7040",
    "Biomass": "#52C860", "Geothermal": "#D4A820", "Hydro": "#1E9AF5",
    "Solar": "#FFD700", "Wind": "#44DAEC", "Battery": "#6A7BC8",
}
# The model splits what the plan aggregates.
MERGE = {"Reservoir": "Hydro", "ROR": "Hydro", "PSH": "Hydro", "PV": "Solar",
         "Onshore Wind": "Wind", "Offshore Wind": "Wind"}
ORDER = ["Nuclear", "Coal", "Gas", "Biomass", "Geothermal", "Hydro", "Wind",
         "Solar", "Battery"]

PEAK = runcfg.ROOT / "epm" / "input" / "data_blacksea" / "load" / "pDemandForecast.csv"
PEAK_ZONES = {
    "Turkiye": ["Trakia", "NorthWest", "CenterBlack", "CenterAna", "EastAna",
                "WestAna", "WestMed", "EastMed", "SouthEast"],
    "Azerbaijan": ["AzerbaijanMain", "Nakhchivan"],
}
PEAK_COLOR = "#D9534F"

INK = "#67788f"
SOFT = "#8a97a8"
GRID = "#e2e7ee"
LEGEND_IN = 1.05


def solid(c):
    return dict(facecolor=to_rgba(c, .92), edgecolor=to_rgba(c, .92),
                linewidth=.2)


def model_capacity(d, scope, scenario, pyears):
    """Cache capacity, merged into the plan's technology vocabulary."""
    cap = d["annual"][scope][scenario]["cap"]
    out = {}
    for tech, series in cap.items():
        k = MERGE.get(tech, tech)
        row = out.setdefault(k, [0.0] * len(pyears))
        for j, y in enumerate(pyears):
            i = d["years"].index(y) if y in d["years"] else -1
            if i >= 0:
                row[j] += series[i] or 0.0
    return out


def study_capacity(register, scope):
    """CESI Task 6 national fleet on STUDY_YEARS, in the plan's vocabulary.

    Empty when the study did not model the scope or the register is absent.
    """
    zone = STUDY_ZONE.get(scope)
    if not zone or not register or not Path(register).exists():
        return {}
    entries = yaml.safe_load(Path(register).read_text(encoding="utf-8"))["entries"]
    out = {}

    def add(k, vals):
        row = out.setdefault(k, [0.0] * len(STUDY_YEARS))
        for j, v in enumerate(vals):
            row[j] += float(v or 0.0)

    for e in entries:
        if e.get("zone") != zone or e.get("element") not in STUDY_ELEMENTS:
            continue
        v = e["cesi_value"]
        if e["element"] == "VRE ceilings":
            for tech, vals in v.items():
                add(STUDY_TECH[tech], vals)
        elif isinstance(v, dict):
            add(STUDY_ELEMENTS[e["element"]],
                [v.get(int(y), v.get(y, 0.0)) for y in STUDY_YEARS])
        else:
            add(STUDY_ELEMENTS[e["element"]], [v] * len(STUDY_YEARS))
    return out


def peak_demand(path, scope, pyears):
    """Country peak demand in GW on the plan years, from the model's own
    demand forecast (sum of the zone peaks, so a shade above the coincident
    peak for the multi-zone countries).  Empty when the file is absent."""
    if not path or not Path(path).exists():
        return {}
    import csv
    zones = PEAK_ZONES.get(scope, [scope])
    out = {y: 0.0 for y in pyears}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["z"] in zones and row["type"] == "Peak":
                for y in pyears:
                    out[y] += float(row.get(y) or 0.0) / 1000.0
    return out if any(out.values()) else {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(runcfg.cache_path()))
    p.add_argument("--scope", default="Georgia")
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--study", default=str(STUDY),
                   help="CESI register; an empty string drops the study bar")
    p.add_argument("--width", type=float, default=4.60)
    p.add_argument("--height", type=float, default=2.30)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--title", default=None)
    p.add_argument("--peak", default=str(PEAK),
                   help="demand forecast file for the peak marker; an empty "
                        "string drops it")
    p.add_argument("--source", default=None,
                   help="small grey line under the chart, e.g. the plan document")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    d = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    plans = json.loads(PLANS.read_text(encoding="utf-8"))
    plan = plans["capacity_gw"].get(a.scope) or {}
    pyears = [str(y) for y in plans["years"]]
    model = model_capacity(d, a.scope, a.scenario, pyears)
    study = study_capacity(a.study, a.scope)
    peak = peak_demand(a.peak, a.scope, pyears)

    # One list of (label, {tech: GW}) per plan year.  The plan bar exists only
    # on the years the plan document covers (null in the reference file
    # otherwise, or no entry at all for a country without a plan).  A study
    # year lands in the group of the nearest plan year and keeps its own year
    # in the label.
    groups = []
    for j in range(len(pyears)):
        g = []
        if any(v[j] is not None for v in plan.values()):
            g.append(("Plan", {k: v[j] or 0.0 for k, v in plan.items()}))
        g.append(("Model", {k: v[j] for k, v in model.items()}))
        groups.append(g)
    for i, sy in enumerate(STUDY_YEARS):
        if not study:
            break
        j = min(range(len(pyears)), key=lambda j: abs(int(pyears[j]) - int(sy)))
        groups[j].append(("%s\n%s" % (STUDY_LABEL, sy),
                          {k: v[i] for k, v in study.items()}))

    # A technology under 50 MW is invisible on a GW axis and only clutters
    # the legend, so it is folded into the stack without a legend entry.
    def have(k):
        return max([0.0] + [bar.get(k) or 0.0 for g in groups for _, bar in g]) >= .05
    cats = [k for k in ORDER if have(k)]
    cats += sorted({k for g in groups for _, bar in g for k in bar
                    if k not in cats and have(k)})

    fs = 6.5 if a.width >= 4 else max(4.4, 6.5 * a.width / 4.4)
    legend_in = max(.55, min(LEGEND_IN, a.width * .26))
    two_line = any("\n" in name for g in groups for name, _ in g)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": fs,
        "axes.edgecolor": "#ccd4de", "axes.linewidth": .6,
        "text.color": INK, "axes.labelcolor": SOFT,
        "xtick.color": SOFT, "ytick.color": SOFT,
        "xtick.major.size": 0, "ytick.major.size": 2,
        "ytick.major.width": .6, "ytick.major.pad": 2,
        "xtick.major.pad": 2, "xtick.labelsize": fs, "ytick.labelsize": fs,
    })
    fig, ax = plt.subplots(figsize=(a.width, a.height), dpi=a.dpi)

    n_max = max(len(g) for g in groups)
    gap = .04 if n_max == 2 else .03
    bw = min(.34, (.84 - gap * (n_max - 1)) / n_max)
    for j, g in enumerate(groups):
        n = len(g)
        for si, (name, bar) in enumerate(g):
            x = j - (n * bw + (n - 1) * gap) / 2 + bw / 2 + si * (bw + gap)
            base, total = 0.0, 0.0
            for k in cats:
                v = bar.get(k) or 0.0
                if v <= 0:
                    continue
                ax.bar(x, v, bottom=base, width=bw, zorder=2,
                       **solid(COLORS.get(k, "#9aa5b4")))
                base += v
                total += v
            if total <= 0:
                ax.text(x, .2, "n/a", fontsize=fs - .5, color="#b7c0cc",
                        ha="center", va="bottom")
            else:
                ax.text(x, total, "%.1f" % total, fontsize=fs - .5,
                        color=SOFT, ha="center", va="bottom")
            ax.text(x, -.035, name, fontsize=fs - 1, color=SOFT, ha="center",
                    va="top", linespacing=1.1, clip_on=False,
                    transform=ax.get_xaxis_transform())
            # Peak demand as a red tick across the bar: the same demand faces
            # the plan and the model, so every bar of the year carries it.
            pk = peak.get(pyears[j])
            if pk:
                ax.plot([x - bw / 2 - .015, x + bw / 2 + .015], [pk, pk],
                        color=PEAK_COLOR, linewidth=1.1, solid_capstyle="butt",
                        zorder=4)

    ax.set_xticks(range(len(pyears)))
    ax.set_xticklabels(pyears, fontsize=fs + .5, color=INK, fontweight="bold")
    ax.tick_params(axis="x", pad=22 if two_line else 15)
    ax.set_ylabel("GW", fontsize=fs, labelpad=2)
    ax.yaxis.grid(True, color=GRID, linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-.6, len(pyears) - .4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    against = " vs ".join(["national plan"] * bool(plan) + ["CESI study"] * bool(study))
    title = a.title or "%s installed capacity, model%s" % (
        a.scope, (" vs " + against) if against else "")
    fig.suptitle(title, fontsize=fs + 1, fontweight="bold", color=INK,
                 x=.012, y=.995, ha="left", va="top")

    right = 1 - legend_in / a.width
    bottom = 0.0
    if a.source:
        bottom = 11 / (a.height * 72)   # one small line of text
        fig.text(.012, .012, a.source, fontsize=fs - 1, color=SOFT,
                 ha="left", va="bottom", style="italic")
    fig.tight_layout(pad=.3, rect=(.012, bottom, right, .90))
    lg = [Patch(label=k, **solid(COLORS.get(k, "#9aa5b4"))) for k in cats]
    if peak:
        lg.append(Line2D([], [], color=PEAK_COLOR, linewidth=1.1,
                         label="Peak demand"))
    fig.legend(handles=lg, loc="center left", ncol=1, fontsize=fs,
               frameon=False, handlelength=.9, handleheight=.8,
               handletextpad=.35, labelspacing=.34, borderpad=0,
               bbox_to_anchor=(right + .01, .5))

    out = Path(a.out) if a.out else (
        HERE.parents[2] / "Data" / "results" / "slides"
        / ("ndp_%s.png" % a.scope.lower()))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    print("%s  %.2f x %.2f in" % (out, a.width, a.height))


if __name__ == "__main__":
    main()
