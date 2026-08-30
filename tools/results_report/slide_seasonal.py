"""Seasonal net exchange, one group of season bars per year.

The annual balance nets out to near zero, which hides the real pattern: the zone
buys through the winter and sells through the hydro season.  Weighting each slot
by pHours turns the hourly Imports/Exports series into GWh per season.

    python slide_seasonal.py --scope Georgia
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent

# Cool for the buying seasons, warm for the selling ones is misleading here
# (the sign already says it), so the seasons get one ramp and the sign is read
# off the axis.
SEASON = {"Q1": "#1B6CA8", "Q2": "#36B5B5", "Q3": "#E8C547", "Q4": "#4DA6FF"}
INK = "#67788f"   # blue grey: titles and axis text, never black
SOFT = "#8a97a8"
NET = "#2f3f57"
GRID = "#e2e7ee"

LEGEND_IN = 1.25


def face(c):
    return dict(facecolor=to_rgba(c, .88), edgecolor=to_rgba(c, .88),
                linewidth=.2)


def pale(c):
    """Exports: same season colour, hollowed out so the direction reads."""
    return dict(facecolor=to_rgba(c, .22), edgecolor=c, linewidth=.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(HERE / "cache" / "simulations_run_20260825.json"))
    p.add_argument("--scope", default="Georgia")
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--width", type=float, default=2.26)
    p.add_argument("--height", type=float, default=1.61)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--years", default="2025,2035")
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    d = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    blk = d["dispatch"][a.scope][a.scenario]
    axis, H = blk["axis"], d["hours"]
    years = [y.strip() for y in a.years.split(",") if y.strip() in blk["years"]]
    # The panel is sized to the slide box, which can be a third of the full
    # width: type and legend column follow it instead of overflowing.
    fs = 6.5 if a.width >= 4 else max(4.4, 6.5 * a.width / 4.4)
    legend_in = max(.52, min(LEGEND_IN, a.width * .28))

    def weight(s):
        return float(H["%s|%s|%s" % (s["q"], s["d"], s["t"])])

    seasons = []
    for s in axis:
        if s["q"] not in seasons:
            seasons.append(s["q"])

    # Gross imports and gross exports per season, in GWh.  The partner split of
    # card 4 is annual only (external flows carry no q/d/t dimension), so at
    # seasonal resolution the bar segments are the two directions, not the
    # counterparties.
    gross = {}
    for y in years:
        uni = blk["years"][y]
        imp = uni.get("Imports") or [0] * len(axis)
        exp = uni.get("Exports") or [0] * len(axis)
        acc = {q: [0.0, 0.0] for q in seasons}
        for i, s in enumerate(axis):
            w = weight(s) / 1000.0
            acc[s["q"]][0] += w * imp[i]
            acc[s["q"]][1] += w * exp[i]     # already negative in the output
        gross[y] = acc

    # A multi-zone scope carries its internal corridors in the same two series;
    # the extraction already collapsed them into the net position, so the bars
    # read the same way for every scope and the chart keeps one single look.

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

    nb = len(seasons)
    bw = .80 / nb
    for j, q in enumerate(seasons):
        c = SEASON.get(q, "#9aa5b4")
        xs = [i - .40 + bw * (j + .5) for i in range(len(years))]
        imp = [gross[y][q][0] for y in years]
        exp = [gross[y][q][1] for y in years]
        ax.bar(xs, imp, width=bw * .86, zorder=2, **face(c))
        ax.bar(xs, exp, width=bw * .86, zorder=2, **pale(c))
        # Net position of the season, as a tick across the bar.
        for x, i0, e0 in zip(xs, imp, exp):
            ax.plot([x - bw * .43, x + bw * .43], [i0 + e0] * 2,
                    color=NET, linewidth=1.1, solid_capstyle="butt", zorder=5)

    ax.axhline(0, color="#8b96a5", linewidth=.6, zorder=3)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=fs + .5, color=INK, fontweight="bold")
    ax.set_ylabel("GWh", fontsize=fs, labelpad=2)
    ax.yaxis.grid(True, color=GRID, linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-.6, len(years) - .4)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo * 1.12, hi * 1.12)       # room for the value labels
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Which side of the axis is which, without a caption underneath.
    ax.text(-.56, hi * 1.05, "imports",
            fontsize=fs - .5, color=SOFT, ha="left", va="top")
    ax.text(-.56, lo * 1.05, "exports",
            fontsize=fs - .5, color=SOFT, ha="left", va="bottom")

    title = a.title or "%s seasonal exchange, %s" % (a.scope, a.scenario)
    fig.suptitle(title, fontsize=fs + 1, fontweight="bold", color=INK,
                 x=.012, y=.995, ha="left", va="top")

    right = 1 - legend_in / a.width
    fig.tight_layout(pad=.3, rect=(.012, 0, right, .90))
    lg = [Patch(label=q, **face(SEASON.get(q, "#9aa5b4"))) for q in seasons]
    lg.append(Line2D([], [], color=NET, lw=1.1, label="Net"))
    fig.legend(handles=lg,
               loc="center left", ncol=1, fontsize=fs, frameon=False,
               handlelength=.9, handleheight=.8, handletextpad=.35,
               labelspacing=.38, borderpad=0, bbox_to_anchor=(right + .01, .5))

    out = Path(a.out) if a.out else (
        HERE.parents[2] / "Data" / "results" / "slides"
        / ("seasonal_%s.png" % a.scope.lower()))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    print("%s  %.2f x %.2f in" % (out, a.width, a.height))


if __name__ == "__main__":
    main()
