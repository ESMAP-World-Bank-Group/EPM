# -*- coding: utf-8 -*-
"""Slide-grade figures for the EU border price hypothesis.

The QC figures in output_prices are built to be read at a desk: five legends,
a 2053 horizon, matplotlib defaults. A slide is read from the back of a room,
so these are the same numbers redrawn for that - deck palette, model horizon,
one shared legend, no chart junk.

Outputs (300 dpi PNG, into output_prices/slides/):
    level_band.png     one panel: a line per scenario, a band for the three zones
    shape_day.png      one panel: a line per quarter, a band for the day-types
    shape_spark.png    one 24-hour profile, no axes, to sit beside the formula

No titles: they are written on the slide, in the deck's own type. The shape is
measured on 2023, the year the representative days were built on - say so on
the slide, because nothing in the figure can.

Both charts use the same grammar - the line is an average, the shading is the
spread around it - and in both the country dimension is the one that gets
collapsed, because it is measurably the smallest of the three:

    level, same year   scenarios 101-135 EUR/MWh apart | zones 3-20 apart
    shape, same hour   day-types 0.91 apart | quarters 0.43 | zones 0.17

So the zone is never a line, and the bands being thin is itself the finding:
this is one EU price assumption, not three national ones.

Run:
    python pre-analysis/pipelines/render_price_slides.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parents[1]
PRICES = HERE / "output_prices"
PHOURS = HERE / "representative_days" / "output" / "blacksea" / "pHours.csv"
OUT = PRICES / "slides"

SHAPE_YEAR = 2023             # the year the representative days were built on

ZONES = ["Romania", "Bulgaria", "Greece"]
YEARS = (2015, 2040)          # the model horizon ends in 2040; L runs to 2053
YMAX = 320                    # tall enough to show the 2022 peak whole (287)

# the deck's own palette, read off the existing slides
NAVY, GOLD, TEAL = "#1A3A5C", "#C9A84C", "#2A7F7F"
SLATE, INK, GRID = "#8497B0", "#3A3A3A", "#D6DCE5"
RUST = "#9C4221"

# Only three of the five are published TYNDP trajectories - see
# eu_price_level.PUBLISHED_TRAJECTORIES. The upper two are ours, because the
# published family holds nothing above National Trends, so the legend has to say
# where they come from instead of borrowing ENTSO-E's name for them.
SCEN = [("CENTRAL", "Central (TYNDP NT)", NAVY, 2.0),
        ("HIGH", "High (mirror of Low)", GOLD, 1.5),
        ("EU_HIGH", "EU high (2021-23 crisis)", RUST, 1.5),
        ("LOW", "Low (TYNDP DE)", TEAL, 1.5),
        ("VERY_LOW", "Very low (TYNDP GA)", SLATE, 1.5)]

QUARTERS = [("Q1", NAVY), ("Q2", TEAL), ("Q3", GOLD), ("Q4", RUST)]

# Near-square axes with the legend beside them, rather than a ribbon with the
# legend underneath. FIGSIZE is the whole canvas; the axes stop at LEGEND_X.
FIGSIZE = (5.6, 2.75)
LEGEND_X = 0.70


def _legend(fig, handles, labels):
    """A single column down the right-hand side.

    Under the chart, the legend has to be wide, which forces the whole figure
    wide and the plot into a ribbon. Beside it, the legend spends width that
    would otherwise be empty and the axes come out close to square.

    fig.legend, not ax.legend: tight_layout cannot see a legend anchored
    outside its axes and would leave a quarter of the canvas empty.
    """
    fig.legend(handles, labels, ncol=1, fontsize=7.5, frameon=False,
               loc="center left", bbox_to_anchor=(LEGEND_X, 0.52),
               handlelength=1.7, handletextpad=0.6, labelspacing=0.85,
               labelcolor=INK)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5, colors=INK, length=2, width=0.6)


def level_band(level: pd.DataFrame, observed: pd.DataFrame) -> Path:
    """Scenario as the line, the three zones as the band around it."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=300)
    _style(ax)

    obs = observed[[z for z in ZONES if z in observed.columns]]
    mean = obs.mean(axis=1)
    ax.fill_between(obs.index, obs.min(axis=1), obs.max(axis=1),
                    color=INK, alpha=0.13, lw=0)
    ax.plot(obs.index, mean, color=INK, lw=1.7, label="Observed",
            solid_capstyle="round")

    # the crisis peak is the reason the axis goes to 320; label it so the empty
    # top half is paying for something. Below and left of the apex, not above:
    # above, the label would sit across the 300 gridline.
    peak = mean.idxmax()
    ax.annotate(f"{mean[peak]:.0f}", xy=(peak, mean[peak]), xytext=(-4, -2),
                textcoords="offset points", ha="right", va="top",
                fontsize=7.5, color=INK)

    ax.axvline(2024, color=GRID, lw=0.9)          # where observed stops
    fwd = level[level["zone"].isin(ZONES) & level["year"].between(2024, YEARS[1])]
    for name, lab, colour, lw in SCEN:
        g = fwd[fwd["scenario"] == name].groupby("year")["L_eur2024"]
        ax.fill_between(g.min().index, g.min(), g.max(), color=colour,
                        alpha=0.16, lw=0)
        ax.plot(g.mean().index, g.mean(), color=colour, lw=lw, ls=(0, (4, 2)),
                label=lab)

    ax.set_xlim(*YEARS)
    ax.set_ylim(0, YMAX)
    ax.set_xticks([2015, 2020, 2025, 2030, 2035, 2040])
    ax.set_yticks([0, 100, 200, 300])
    ax.set_ylabel("€2024 / MWh", fontsize=8, color=INK, labelpad=2)

    # the shading has to be named somewhere or it reads as a confidence
    # interval; the legend is the one place a reader already looks
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=INK, alpha=0.2, lw=0))
    labels.append("Range across\nRO / BG / GR")
    _legend(fig, handles, labels)
    fig.subplots_adjust(left=0.105, right=LEGEND_X - 0.02, top=0.965,
                        bottom=0.115)
    path = OUT / "level_band.png"
    fig.savefig(path, dpi=300, transparent=True)
    plt.close(fig)
    return path


def level_band_cbam(level: pd.DataFrame, observed: pd.DataFrame) -> Path:
    """level_band, plus what each scenario is worth once CBAM is taken off it.

    Same five scenarios, twice: dashed is the European price, dotted is that
    price net of the border levy - the two halves of the slide's formula, on
    one axis.

    ONE exporter, named in the legend, not a levy averaged over both. The levy
    is set by the seller (0.718 Turkiye, 0.440 Georgia) while every other
    country dimension on this chart is the buyer, so drawing both would put two
    different meanings of "country" in the same picture. Turkiye is the one
    drawn because it is the binding case: it is the levy that reaches the price.

    The dotted family carries no band. Five more shaded envelopes on an axis
    that now has to reach below zero is mush, and the spread was never the
    point of the CBAM line - the crossing is.
    """
    lv = pd.read_csv(PRICES / "cbam_levy.csv")
    lv = lv[lv["exporter"] == "Türkiye"].groupby("year")["C_eur2024_per_mwh"].first()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=300)
    _style(ax)

    obs = observed[[z for z in ZONES if z in observed.columns]]
    mean = obs.mean(axis=1)
    ax.fill_between(obs.index, obs.min(axis=1), obs.max(axis=1),
                    color=INK, alpha=0.13, lw=0)
    ax.plot(obs.index, mean, color=INK, lw=1.7, solid_capstyle="round")
    peak = mean.idxmax()
    ax.annotate(f"{mean[peak]:.0f}", xy=(peak, mean[peak]), xytext=(-4, -2),
                textcoords="offset points", ha="right", va="top",
                fontsize=7.5, color=INK)

    ax.axhline(0, color=INK, lw=0.8)              # the line the dotted family crosses
    ax.axvline(2024, color=GRID, lw=0.9)
    fwd = level[level["zone"].isin(ZONES) & level["year"].between(2024, YEARS[1])]
    for name, _, colour, lw in SCEN:
        g = fwd[fwd["scenario"] == name].groupby("year")["L_eur2024"]
        ax.fill_between(g.min().index, g.min(), g.max(), color=colour,
                        alpha=0.16, lw=0)
        ax.plot(g.mean().index, g.mean(), color=colour, lw=lw, ls=(0, (4, 2)))
        net = g.mean() - lv.reindex(g.mean().index).fillna(0.0)
        ax.plot(net.index, net, color=colour, lw=lw, ls=(0, (1, 1.6)))

    ax.set_xlim(*YEARS)
    ax.set_xticks([2015, 2020, 2025, 2030, 2035, 2040])
    ax.set_ylabel("€2024 / MWh", fontsize=8, color=INK, labelpad=2)

    # the legend names the scenarios once and the two line styles once, rather
    # than ten entries in which every colour appears twice
    handles = [Line2D([], [], color=INK, lw=1.7)]
    labels = ["Observed"]
    for _, lab, colour, lw in SCEN:
        handles.append(Line2D([], [], color=colour, lw=lw))
        labels.append(lab)
    handles += [Line2D([], [], color=SLATE, lw=1.5, ls=(0, (4, 2))),
                Line2D([], [], color=SLATE, lw=1.5, ls=(0, (1, 1.6)))]
    labels += ["– – –  without CBAM", "· · ·  net of CBAM\n        (Türkiye)"]
    _legend(fig, handles, labels)
    fig.subplots_adjust(left=0.115, right=LEGEND_X - 0.02, top=0.965,
                        bottom=0.115)
    path = OUT / "level_band_cbam.png"
    fig.savefig(path, dpi=300, transparent=True)
    plt.close(fig)
    return path


def day_weights() -> pd.Series:
    """Hours per year carried by each (q, d), from pHours.

    Not decoration: the day-types run from 48 h/year to 552 h/year, a factor of
    11, so a plain mean of the seven would let a two-day-a-year day-type set the
    seasonal profile. The same weights are what normalise S to a mean of 1, so
    weighting here is also what makes the 1.0 line mean what it says.
    """
    h = pd.read_csv(PHOURS)
    tcols = [c for c in h.columns if c[:1].lower() == "t" and c[1:].isdigit()]
    w = h.set_index(["season", "daytype"])[tcols].sum(axis=1)
    w.index.names = ["q", "d"]
    return w


def shape_day(shape: pd.DataFrame) -> Path:
    """The 24-hour profile: the season is the line, the day-type is the band.

    The quarter carries the colour because it is the structural half of the
    variation - the summer midday collapse is the same story every year - while
    the day-type is the volatile half and belongs in a single grey envelope
    behind everything. One band, not four: four wide overlapping bands is mush.

    S is a multiplier, not a share: it is normalised to a mean of 1 over the
    whole year, so 1.8 is an hour worth 80 % more than the annual average. The
    seasonal lines are pHours-weighted; the envelope is a plain min-max, which
    is what a range should be.
    """
    hours = [c for c in shape.columns if c.startswith("t")]
    z = shape[shape["zone"].isin(ZONES)]
    days = z.groupby(["q", "d"])[hours].mean()      # 28 day-types, zones averaged
    w = day_weights().reindex(days.index)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=300)
    _style(ax)
    x = range(1, len(hours) + 1)

    ax.fill_between(x, days.min(), days.max(), color=INK, alpha=0.10, lw=0)
    ax.axhline(1.0, color=GRID, lw=0.8, ls=(0, (3, 3)))   # the annual average
    for q, colour in QUARTERS:
        d, wq = days.loc[q], w.loc[q]
        ax.plot(x, d.mul(wq, axis=0).sum() / wq.sum(), color=colour, lw=1.7,
                label=q, solid_capstyle="round")

    ax.set_xlim(1, len(hours))
    ax.set_ylim(0, max(3.2, float(days.max().max()) + 0.1))
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_xticklabels([f"{h:02d}h" for h in (1, 6, 12, 18, 24)])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("× annual level", fontsize=8, color=INK, labelpad=2)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=INK, alpha=0.16, lw=0))
    labels.append("Range across the 28\nrepresentative days\n"
                  "(RO / BG / GR averaged)")
    _legend(fig, handles, labels)
    fig.subplots_adjust(left=0.105, right=LEGEND_X - 0.02, top=0.965,
                        bottom=0.115)
    path = OUT / "shape_day.png"
    fig.savefig(path, dpi=300, transparent=True)
    plt.close(fig)
    return path


def shape_spark(shape: pd.DataFrame) -> Path:
    """The mean Romanian day, normalised - the Shape term, shown not described."""
    hours = [c for c in shape.columns if c.startswith("t")]
    d = shape[shape["zone"] == "Romania"].set_index(["q", "d"])[hours]
    w = day_weights().reindex(d.index)
    y = (d.mul(w, axis=0).sum() / w.sum()).values

    fig, ax = plt.subplots(figsize=(1.55, 0.52), dpi=300)
    ax.plot(range(1, 25), y, color=TEAL, lw=1.4, solid_capstyle="round")
    ax.axhline(1.0, color=GRID, lw=0.7, ls=(0, (2, 2)))
    ax.set_xlim(1, 24)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    path = OUT / "shape_spark.png"
    fig.savefig(path, dpi=300, transparent=True)
    plt.close(fig)
    return path


def show(path: Path) -> None:
    """Open the figure in the default viewer - these are made to be looked at."""
    if sys.platform == "win32":
        os.startfile(path)                                   # noqa: S606
    else:
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open",
                        str(path)], check=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    level = pd.read_csv(PRICES / "level_L.csv")
    shape = pd.read_csv(PRICES / "shape_S.csv")
    observed = pd.read_csv(PRICES / "observed_annual_real.csv", index_col=0)
    observed.index = observed.index.astype(int)

    slide_figs = (level_band(level, observed), shape_day(shape))
    for p in slide_figs + (shape_spark(shape),):
        print(f"  wrote {p.relative_to(HERE)}  {p.stat().st_size/1024:.0f} kB")
    for p in slide_figs:
        show(p)


if __name__ == "__main__":
    main()
