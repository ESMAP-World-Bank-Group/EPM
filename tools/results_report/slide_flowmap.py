"""Country flow map, one panel per year, cropped on the country and its partners.

The deck carried these as screenshots of the HTML report and they could not be
refreshed with the run.  This redraws them with the same cartographic toolkit as
the regional map (arrow direction is the net energy, colour is the utilisation,
width is the volume), cropped on the country's own zones plus every zone it
actually trades with, so a small country is not lost inside the regional frame.

    python slide_flowmap.py --scope Georgia
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runcfg  # noqa: E402
import slides_regional as R  # noqa: E402

PAD = 1.4          # degrees of margin around the zones kept in frame


def zones_of(zcmap, country):
    return {z for z, c in zcmap.items() if c == country}


def keep(cor, mine, idx):
    """Corridors with at least one end in the country, plus the zones they reach.

    A border that carries nothing and has no capacity in any displayed year is
    dropped rather than framed: the Georgia-Romania placeholder alone would drag
    the crop 15 degrees west for an arrow that is never drawn."""
    live, reach = {}, set(mine)
    for key, c in cor.items():
        if c["a"] not in mine and c["b"] not in mine:
            continue
        alive = any(c["ntc"][i] > 0 or c["fwd"][i] > .005 or c["rev"][i] > .005
                    for i in idx)
        if not alive:
            continue
        live[key] = c
        reach.add(c["a"])
        reach.add(c["b"])
    return live, reach


def crop(ax, geo, reach, w_in, h_in):
    """Frame the reached zones, then widen to the panel's aspect ratio."""
    # External neighbours are whole countries: Russia alone would blow the frame
    # open. Their label position is enough to keep the arrow's target on screen.
    pts = []
    for name in reach:
        rings = geo["zones"].get(name)
        if rings:
            for ring in rings:
                pts.extend(ring)
        elif name in geo["centroids"]:
            pts.append(geo["centroids"][name])
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs) - PAD, max(xs) + PAD
    y0, y1 = min(ys) - PAD, max(ys) + PAD
    k = math.cos(math.radians((y0 + y1) / 2))
    # The axes keep a true aspect, so only one dimension can be honoured; grow
    # the other around its centre rather than letting matplotlib letterbox it.
    want = (h_in / w_in) * (x1 - x0) * k
    if want >= y1 - y0:
        cy = (y0 + y1) / 2
        y0, y1 = cy - want / 2, cy + want / 2
    else:
        wl = (w_in / h_in) * (y1 - y0) / k
        cx = (x0 + x1) / 2
        x0, x1 = cx - wl / 2, cx + wl / 2
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1.0 / k)
    # Let the frame grow into the panel instead of letterboxing inside it: the
    # aspect is still true, the surplus goes into more map rather than white.
    ax.set_adjustable("datalim")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(runcfg.cache_path()))
    p.add_argument("--scope", default="Georgia")
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--years", default="2025,2035,2040")
    p.add_argument("--width", type=float, default=6.60)
    p.add_argument("--height", type=float, default=2.30)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    d = R.cache()
    geo = d["geo"]
    cor = d["corridors"][a.scenario]
    mine = zones_of(d["zcmap"], a.scope)
    if not mine:
        raise SystemExit("no zone maps to %s" % a.scope)
    years = [y.strip() for y in a.years.split(",") if y.strip() in d["years"]]
    live, reach = keep(cor, mine, [d["years"].index(y) for y in years])

    # A country whose zones are all off-frame would have its label anchored on
    # an off-screen centroid, so only the zones actually in frame are labelled.
    shown = {z: c for z, c in d["zcmap"].items() if z in reach}
    pseudo = sorted(n for n in reach
                    if n not in geo["zones"] and n not in geo["ext"]
                    and n in geo["centroids"])

    fs = 6.0
    R.rc(fs)
    fig, axes = plt.subplots(1, len(years), figsize=(a.width, a.height),
                             dpi=a.dpi)
    axes = list(axes) if len(years) > 1 else [axes]
    panel = a.width * .96 / len(years)
    sc = min(1.0, panel / 4.4)
    fsp = max(4.2, fs * (.55 + .45 * sc))
    for ax, y in zip(axes, years):
        i = d["years"].index(y)
        R.draw_base(ax, geo, fsp)
        crop(ax, geo, reach, panel, a.height * .80)
        R.draw_flows(ax, geo, live, i, fs, scale=sc)
        # Zoomed in on two or three countries, the outlines alone are ambiguous.
        R.label_countries(ax, geo, shown, fsp + 1.2)
        # draw_base names the external polygons; a pseudo zone such as the Iran
        # swap has a centroid but no geometry, so its arrow would end nowhere.
        for name in pseudo:
            lon, lat = geo["centroids"][name]
            ax.text(lon, lat, name, fontsize=fsp - 1.2, color="#8d97a6",
                    ha="center", va="center", zorder=3, clip_on=True)
        ax.set_title(y, fontsize=fs + 1.5, fontweight="bold", loc="left",
                     pad=1.5, color=R.INK)
        ax.set_anchor("N")
    if a.title:
        fig.suptitle(a.title, fontsize=fs + 1, fontweight="bold", color=R.INK,
                     x=.012, y=.995, ha="left", va="top")
    top = .93 if a.title else .995
    fig.tight_layout(pad=.2, w_pad=.3, rect=(0, .085, .995, top))
    h = [Line2D([], [], color=R.FLOW_COOL, lw=1.6, label="< 60% used"),
         Line2D([], [], color=R.FLOW_WARM, lw=1.6, label="60-85%"),
         Line2D([], [], color=R.FLOW_HOT, lw=1.6, label=">= 85%"),
         Line2D([], [], color=R.SOFT, lw=2.6, label="thicker = more energy")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=fs - .5,
               frameon=False, handlelength=1.3, handletextpad=.4,
               columnspacing=1.6, borderpad=0, bbox_to_anchor=(.5, -.005))

    out = Path(a.out) if a.out else (
        HERE.parents[2] / "Data" / "results" / "slides"
        / ("flowmap_%s.png" % a.scope.lower()))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    print("%s  %.2f x %.2f in" % (out, a.width, a.height))


if __name__ == "__main__":
    main()
