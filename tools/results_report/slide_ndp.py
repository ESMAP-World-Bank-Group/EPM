"""Model capacity against the published national plan, on the plan milestones.

The deck carried this comparison as a screenshot of the HTML report, so it could
not be refreshed with the run.  This redraws it: one pair of stacked bars per
plan year, the plan hollowed out and dashed on the left, the model solid on the
right, so the gap reads per technology and not only on the total.

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
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runcfg  # noqa: E402

PLANS = HERE / "reference" / "national_plans.json"

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

INK = "#67788f"
SOFT = "#8a97a8"
GRID = "#e2e7ee"
LEGEND_IN = 1.05


def solid(c):
    return dict(facecolor=to_rgba(c, .92), edgecolor=to_rgba(c, .92),
                linewidth=.2)


def hollow(c):
    """The plan: same colour, hollowed out and dashed, so it reads as a target."""
    return dict(facecolor=to_rgba(c, .20), edgecolor=c, linewidth=.7,
                linestyle=(0, (2, 1.2)))


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(runcfg.cache_path()))
    p.add_argument("--scope", default="Georgia")
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--width", type=float, default=4.60)
    p.add_argument("--height", type=float, default=2.30)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    d = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    plans = json.loads(PLANS.read_text(encoding="utf-8"))
    plan = plans["capacity_gw"].get(a.scope)
    if not plan:
        raise SystemExit("no published plan for %s" % a.scope)
    pyears = [str(y) for y in plans["years"]]
    model = model_capacity(d, a.scope, a.scenario, pyears)

    # A technology under 50 MW is invisible on a GW axis and only clutters
    # the legend, so it is folded into the stack without a legend entry.
    def have(k):
        return max([0.0] + (plan.get(k) or []) + (model.get(k) or [])) >= .05
    cats = [k for k in ORDER if have(k)]
    cats += sorted(k for k in (set(plan) | set(model))
                   if k not in cats and have(k))

    fs = 6.5 if a.width >= 4 else max(4.4, 6.5 * a.width / 4.4)
    legend_in = max(.55, min(LEGEND_IN, a.width * .26))

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

    bw = .34
    for j, yr in enumerate(pyears):
        for si, (name, src, style) in enumerate(
                (("Plan", plan, hollow), ("Model", model, solid))):
            x = j - bw / 2 - .02 + si * (bw + .04)
            base, total = 0.0, 0.0
            for k in cats:
                v = (src.get(k) or [0] * len(pyears))[j] or 0.0
                if v <= 0:
                    continue
                ax.bar(x, v, bottom=base, width=bw, zorder=2,
                       **style(COLORS.get(k, "#9aa5b4")))
                base += v
                total += v
            if total <= 0:
                ax.text(x, .2, "n/a", fontsize=fs - .5, color="#b7c0cc",
                        ha="center", va="bottom")
            else:
                ax.text(x, total, "%.1f" % total, fontsize=fs - .5,
                        color=SOFT, ha="center", va="bottom")
            ax.text(x, 0, name, fontsize=fs - 1, color=SOFT, ha="center",
                    va="top", clip_on=False,
                    transform=ax.get_xaxis_transform())

    ax.set_xticks(range(len(pyears)))
    ax.set_xticklabels(pyears, fontsize=fs + .5, color=INK, fontweight="bold")
    ax.tick_params(axis="x", pad=12)
    ax.set_ylabel("GW", fontsize=fs, labelpad=2)
    ax.yaxis.grid(True, color=GRID, linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-.6, len(pyears) - .4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    title = a.title or "%s installed capacity, model vs national plan" % a.scope
    fig.suptitle(title, fontsize=fs + 1, fontweight="bold", color=INK,
                 x=.012, y=.995, ha="left", va="top")

    right = 1 - legend_in / a.width
    fig.tight_layout(pad=.3, rect=(.012, 0, right, .90))
    lg = [Patch(label=k, **solid(COLORS.get(k, "#9aa5b4"))) for k in cats]
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
