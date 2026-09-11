# -*- coding: utf-8 -*-
"""Charts for the EU price and CBAM sensitivity (waves 2 and 3).

Three charts, all built on two runs side by side: the central-price run and the
CBAM / VeryLow run.

    python slide_pricesens.py --chart price
    python slide_pricesens.py --chart exports
    python slide_pricesens.py --chart benefit

Unlike the other generators this one does not read the JSON cache. The cache is
built per run and carries no objective value, while every chart here is a
comparison ACROSS runs of either the objective or the external export volume.
Both come straight out of two small CSVs, so reading them is cheaper and clearer
than teaching the cache to span runs.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runcfg  # noqa: E402

OUTVIEW = runcfg.OUTVIEW
TRADE = runcfg.ROOT / "epm" / "input" / "data_blacksea" / "trade"
OUTDIR = HERE.parents[2] / "Data" / "results" / "slides"

# The three price worlds now sit in one run folder: the wave 2/3 scenarios were
# folded into the central run, which they only ever differed from by the trade
# price files. Kept as two names so a future split reads as a one-line change.
RUN_CENTRAL = "simulations_run_20260906"
RUN_SENS = RUN_CENTRAL

YEARS = [str(y) for y in range(2025, 2041)]
TICKS = ["2025", "2030", "2035", "2040"]

# The EU members the region can reach. Russia, Iran, Iraq, Syria and Kazakhstan
# are external too but sit outside the ETS, so CBAM does not touch them.
EU = ("Bulgaria", "Greece", "Romania")

# A price world is (key, label, colour, export-price file). That file is what
# actually differs between the three: every other input is identical.
WORLDS = [
    ("central", "EU central", "#1B6CA8", "pTradePriceExport_eu_central.csv"),
    ("verylow", "EU very low", "#59A9DE", "pTradePriceExport_eu_very_low.csv"),
    ("cbam", "EU central + CBAM", "#C0392B", "pTradePriceExport_eu_central_cbam.csv"),
]

# Topology label, colour, then its scenario name in each of the three worlds.
# Base is the reference every other topology is measured against, inside its
# own world.
TOPO = [
    ("Base", "#8a97a8", "baseline", "LC_Base_VeryLow", "LC_Base_CBAM"),
    ("BSSC", "#36B5B5", "LC_BSSC", "LC_BSSC_VeryLow", "LC_BSSC_CBAM"),
    ("All Projects", "#1B6CA8", "LC_AllProjects", "LC_AllProjects_VeryLow",
     "LC_AllProjects_CBAM"),
    ("Free Exp.", "#C8A8F0", "LC_FreeExpAll", "LC_FreeExpAll_VeryLow",
     "LC_FreeExpAll_CBAM"),
]

INK = "#67788f"
SOFT = "#8a97a8"
GRID = "#e2e7ee"


def rc(fs):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": fs,
        "axes.edgecolor": "#ccd4de", "axes.linewidth": .6,
        "text.color": INK, "axes.labelcolor": SOFT,
        "xtick.color": SOFT, "ytick.color": SOFT,
        "xtick.major.size": 0, "ytick.major.size": 2,
        "ytick.major.width": .6, "ytick.major.pad": 2,
        "xtick.major.pad": 2, "xtick.labelsize": fs, "ytick.labelsize": fs,
    })


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=.6)
    ax.set_axisbelow(True)


def save(fig, a, name):
    out = Path(a.out) if a.out else OUTDIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    print("%s  %.2f x %.2f in" % (out, a.width, a.height))
    return out


# ------------------------------------------------------------------- readers

def run_of(world):
    return OUTVIEW / (RUN_CENTRAL if world == "central" else RUN_SENS)


def scen_of(topo_row, world):
    central, verylow, cbam = topo_row[2], topo_row[3], topo_row[4]
    return {"central": central, "verylow": verylow, "cbam": cbam}[world]


def objectives(run):
    """scenario -> NPV of system cost in billion USD.

    Read from summary.csv rather than from the solver objective. The two agree
    to about 5 million on a 223 billion total, but summary.csv is the number the
    tables on the deck quote, and one source keeps a chart and a table beside it
    from rounding to different digits.
    """
    df = pd.read_csv(run / "summary.csv", low_memory=False)
    r = df[(df["country"] == "System")
           & (df["attribute"] == "NPV of system cost: $m")]
    skip = ("country", "zone", "attribute", "resolution", "year")
    return {c: float(r.iloc[0][c]) / 1000.0 for c in r.columns if c not in skip}


def eu_exports(run, scen):
    """Annual exports to the EU, TWh, one value per year of YEARS.

    pTransmissionMerged carries the modelled zone in `z` and the external
    partner in `uni`, so the EU filter goes on `uni`, not on the zone.
    """
    df = pd.read_csv(run / scen / "output_csv" / "pTransmissionMerged.csv")
    m = df[(df["attribute"] == "InterchangeExternalExports")
           & (df["uni"].astype(str).isin(EU))]
    g = m.groupby(m["y"].astype(str))["value"].sum() / 1000.0
    return [float(g.get(y, 0.0)) for y in YEARS]


def eu_price(fname):
    """Mean price offered for an exported MWh, USD/MWh, by year.

    The file is hourly by season and day type. The plain mean over the 24 hourly
    columns is what revenue per MWh converges to, which is the number the story
    is about, so no load weighting is applied here.
    """
    df = pd.read_csv(TRADE / fname)
    df.columns = [str(c).strip() for c in df.columns]
    hours = [c for c in df.columns if c.startswith("t")]
    m = df[df[df.columns[0]].astype(str).isin(EU)]
    g = m.groupby(m["year"].astype(str))[hours].mean().mean(axis=1)
    return [float(g.get(y, float("nan"))) for y in YEARS]


# -------------------------------------------------------------------- charts

def chart_price(a):
    """What the three worlds do to the price of an exported MWh."""
    rc(a.fs)
    fig, ax = plt.subplots(figsize=(a.width, a.height))
    x = list(range(len(YEARS)))
    for key, label, colour, fname in WORLDS:
        ax.plot(x, eu_price(fname), color=colour, linewidth=1.8, label=label,
                marker="o", markersize=2.4, markevery=5)
    tidy(ax)
    ax.set_xticks([YEARS.index(y) for y in TICKS])
    ax.set_xticklabels(TICKS)
    ax.set_ylim(0, None)
    ax.set_ylabel("USD/MWh", fontsize=a.fs)
    ax.set_title("Price offered for an exported MWh, EU partners",
                 fontsize=a.fs + 1.4, color=INK, loc="left", pad=6)
    # The three lines sweep the whole panel, so the legend goes under the axes
    # rather than hunting for a corner that stays empty as the data changes.
    ax.legend(frameon=False, fontsize=a.fs - .4, loc="upper center",
              bbox_to_anchor=(.5, -.09), ncol=len(WORLDS),
              handlelength=1.6, borderpad=.2, columnspacing=1.6)
    fig.tight_layout(pad=.4)
    return save(fig, a, "pricesens_price.png")


def chart_exports(a):
    """Export volume by topology, one panel per price world, shared y scale."""
    rc(a.fs)
    fig, axes = plt.subplots(1, len(WORLDS), figsize=(a.width, a.height),
                             sharey=True)
    x = list(range(len(YEARS)))
    top = 0.0
    for ax, world in zip(axes, WORLDS):
        key, label, colour = world[0], world[1], world[2]
        run = run_of(key)
        for row in TOPO:
            v = eu_exports(run, scen_of(row, key))
            top = max(top, max(v))
            # Free Expansion sits exactly on Base, so Base goes on top and
            # dashed: the overlap is the finding, not a missing series.
            base = row[0] == "Base"
            ax.plot(x, v, color=row[1], linewidth=2.0 if base else 1.6,
                    linestyle=(0, (2.4, 1.6)) if base else "-",
                    zorder=4 if base else 3)
        tidy(ax)
        ax.set_xticks([YEARS.index(y) for y in TICKS])
        ax.set_xticklabels(TICKS)
        ax.set_title(label, fontsize=a.fs + .8, color=colour, loc="left", pad=4)
    axes[0].set_ylabel("TWh exported to the EU", fontsize=a.fs)
    axes[0].set_ylim(0, top * 1.12)
    handles = [Line2D([], [], color=r[1], linewidth=1.8, label=r[0],
                      linestyle=(0, (2.4, 1.6)) if r[0] == "Base" else "-")
               for r in TOPO]
    fig.legend(handles=handles, frameon=False, fontsize=a.fs - .2,
               loc="lower center", ncol=len(TOPO), bbox_to_anchor=(.5, -.015))
    fig.tight_layout(pad=.4, rect=(0, .10, 1, 1))
    # Widened after tight_layout, which would otherwise pack the panels close
    # enough for the 2040 of one to touch the 2025 of the next.
    fig.subplots_adjust(wspace=.20)
    return save(fig, a, "pricesens_exports.png")


def chart_benefit(a):
    """NPV saving of each topology against the Base case of its own world.

    Measured inside the world, never across worlds: the three Base cases differ
    from one another, so a cross-world difference would mix the value of the
    corridors with the value of the price assumption.
    """
    rc(a.fs)
    fig, ax = plt.subplots(figsize=(a.width, a.height))
    obj = {w[0]: objectives(run_of(w[0])) for w in WORLDS}
    rows = [r for r in TOPO if r[0] != "Base"]
    n = len(WORLDS)
    width = .78 / n
    for j, world in enumerate(WORLDS):
        key, label, colour = world[0], world[1], world[2]
        base = obj[key][scen_of(TOPO[0], key)]
        vals = [base - obj[key][scen_of(r, key)] for r in rows]
        pos = [i + (j - (n - 1) / 2.0) * width for i in range(len(rows))]
        ax.bar(pos, vals, width=width * .92, label=label,
               facecolor=to_rgba(colour, .90), edgecolor=to_rgba(colour, .90),
               linewidth=.2)
        for p, v in zip(pos, vals):
            ax.text(p, v + .10, "%.1f" % v, ha="center", va="bottom",
                    fontsize=a.fs - 1.0, color=colour)
    tidy(ax)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0, None)
    ax.set_ylabel("NPV saving vs Base, billion USD", fontsize=a.fs)
    ax.set_title("Value of the corridors, inside each price world",
                 fontsize=a.fs + 1.4, color=INK, loc="left", pad=6)
    ax.legend(frameon=False, fontsize=a.fs - .4, loc="upper right",
              handlelength=1.1, borderpad=.2, labelspacing=.3)
    fig.tight_layout(pad=.4)
    return save(fig, a, "pricesens_benefit.png")


CHARTS = {"price": chart_price, "exports": chart_exports, "benefit": chart_benefit}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chart", required=True, choices=sorted(CHARTS))
    p.add_argument("--width", type=float, default=5.56)
    p.add_argument("--height", type=float, default=2.45)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--fs", type=float, default=6.4)
    p.add_argument("--out")
    a = p.parse_args()
    CHARTS[a.chart](a)


if __name__ == "__main__":
    main()
