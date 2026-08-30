"""Slide-ready trade evolution bars, drawn from the report cache.

Same series as card 4 of the HTML page: imports above the axis, exports below,
one colour per counterparty, partners outside the model hatched, and the net
position as a line.  Sized for a half-slide block, legend in a right-hand
column so the bars keep the full width.

    python slide_trade.py --scope Georgia
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

MAPBAND = ["#1B6CA8", "#36B5B5", "#E8C547", "#4DA6FF", "#4169E1",
           "#85C1E9", "#2E9EC8", "#5EBCBA", "#1A5276", "#7EC8E3",
           "#14A094", "#4CAFE8", "#EDD770", "#AED6F1", "#1F618D"]
NETCOLOR = "#2f3f57"
INK = "#67788f"   # blue grey: titles and axis text, never black
SOFT = "#8a97a8"
GRID = "#e2e7ee"

LEGEND_IN = 1.25


def face(c, hatched):
    if hatched:
        return dict(facecolor=to_rgba(c, .28), edgecolor=c, linewidth=.4,
                    hatch="///")
    return dict(facecolor=to_rgba(c, .88), edgecolor=to_rgba(c, .88),
                linewidth=.2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(HERE / "cache" / "simulations_run_20260825.json"))
    p.add_argument("--scope", default="Georgia")
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--width", type=float, default=5.6)
    p.add_argument("--height", type=float, default=2.2)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    d = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    blk = d["annual"][a.scope][a.scenario]
    years = d["years"]
    ext = set(blk.get("ext_partners") or [])
    partners = sorted(blk["trade"], key=lambda k: (k in ext, k))
    fs = 6.5

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
    x = list(range(len(years)))
    up = [0.0] * len(years)
    dn = [0.0] * len(years)
    handles = []
    for i, pn in enumerate(partners):
        kw = face(MAPBAND[i % len(MAPBAND)], pn in ext)
        imp = blk["trade"][pn].get("imp") or [0] * len(years)
        exp = blk["trade"][pn].get("exp") or [0] * len(years)
        ax.bar(x, imp, bottom=up, width=.7, zorder=2, **kw)
        ax.bar(x, [-v for v in exp], bottom=dn, width=.7, zorder=2, **kw)
        up = [b + v for b, v in zip(up, imp)]
        dn = [b - v for b, v in zip(dn, exp)]
        handles.append(Patch(label=pn + (" *" if pn in ext else ""), **kw))

    net = [u + v for u, v in zip(up, dn)]
    ax.plot(x, net, color=NETCOLOR, linewidth=1.3, zorder=4)
    handles.append(Line2D([], [], color=NETCOLOR, lw=1.3, label="Net position"))

    ax.axhline(0, color="#8b96a5", linewidth=.6, zorder=3)
    # Every second year only: sixteen labels do not fit at a readable size.
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(years[i]) for i in x[::2]], fontsize=fs)
    ax.set_ylabel("TWh", fontsize=fs, labelpad=2)
    ax.yaxis.grid(True, color=GRID, linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-.7, len(years) - .3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    title = a.title or "%s trade, %s" % (a.scope, a.scenario)
    fig.suptitle(title, fontsize=fs + 1, fontweight="bold", color=INK,
                 x=.012, y=.995, ha="left", va="top")
    right = 1 - LEGEND_IN / a.width
    fig.tight_layout(pad=.3, rect=(.012, 0, right, .90))
    fig.legend(handles=handles, loc="center left", ncol=1, fontsize=fs,
               frameon=False, handlelength=1.1, handleheight=.9,
               handletextpad=.45, labelspacing=.45, borderpad=0,
               bbox_to_anchor=(right + .015, .44))

    out = Path(a.out) if a.out else (
        HERE.parents[2] / "Data" / "results" / "slides"
        / ("trade_%s.png" % a.scope.lower()))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    print("%s  %.2f x %.2f in" % (out, a.width, a.height))


if __name__ == "__main__":
    main()
