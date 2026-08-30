"""Slide-ready dispatch panels, drawn straight from the report cache.

The HTML charts are inline SVG painted by the browser, so they cannot be
exported from here.  This redraws the same series with matplotlib at the size
the slide actually needs, which also lets the type be sized against the printed
width rather than against a 1400 px canvas.

Default target: slide 3 of the results deck, the block left of the Notes box and
under the NDP comparison picture (about 5.6 x 2.5 in).

    python slide_dispatch.py --scope Georgia --years 2025,2030
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
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent

# Same palette as templates/report.js, so page and slide agree.
COLORS = {
    "Nuclear": "#C8A8F0", "Coal": "#808890", "Gas": "#9A7040",
    "CCGT": "#B8921A", "OCGT": "#C4A820", "Diesel": "#6A7888",
    "HFO": "#7A7068", "Oil": "#7A7068", "Biomass": "#52C860",
    "Waste": "#8A9098", "Geothermal": "#D4A820", "Reservoir": "#1E9AF5",
    "ROR": "#5DADE2", "Hydro": "#1E9AF5", "PSH": "#0D7680",
    "Solar": "#FFD700", "PV": "#FFD700", "CSP": "#E8C547", "RPV": "#FFD700",
    "Onshore Wind": "#44DAEC", "Wind": "#44DAEC", "Offshore Wind": "#7CC8FA",
    "Battery": "#6A7BC8", "Storage": "#8290CE",
    "Imports": "#2E9EC8", "Exports": "#E8C547", "Storage Charge": "#0D7680",
    "Unmet demand": "#D9534F", "Demand": "#8B0000",
}
NETIMPORT = "#12356e"
HATCHED = {"Imports", "Exports"}          # trade is not own generation
INK = "#67788f"   # blue grey: titles and axis text, never black
SOFT = "#8a97a8"
GRID = "#e2e7ee"

LEGEND_IN = 1.25                          # width reserved for the legend column


def face(k, hatched):
    """Fill kwargs.  Matplotlib draws hatch lines in the edge colour, so a
    hatched band needs a pale face and a solid edge or the pattern vanishes."""
    c = COLORS.get(k, "#9aa5b4")
    if hatched:
        return dict(facecolor=to_rgba(c, .28), edgecolor=c, linewidth=.4,
                    hatch="///")
    return dict(facecolor=to_rgba(c, .88), edgecolor=to_rgba(c, .88),
                linewidth=.2)


def internal_heavy(d, scope, scenario, block):
    """True when the scope groups several zones and the Imports/Exports series
    are mostly traffic between those zones.  Zone-level output carries internal
    corridors in the same series as the external ones, so on a scope like
    Turkiye the raw bands are several times the fleet and sit above demand."""
    if scope in d.get("dispatch_netted", []):
        return True          # extract.py already collapsed the two series
    if "dispatch_netted" in d or len(d["scopes"].get(scope, [scope])) < 2:
        return False
    H, axis = d["hours"], block["axis"]
    w = [float(H["%s|%s|%s" % (s["q"], s["d"], s["t"])]) for s in axis]
    tr = d["annual"][scope][scenario].get("trade", {})
    yi = [str(v) for v in d["years"]]
    for y, uni in block["years"].items():
        k = yi.index(y)
        gross = sum(wi * (abs((uni.get("Imports") or [0] * len(w))[i])
                          + abs((uni.get("Exports") or [0] * len(w))[i]))
                    for i, wi in enumerate(w)) / 1e6
        ext = sum(abs(v.get(side, [0] * len(yi))[k])
                  for v in tr.values() for side in ("imp", "exp"))
        if gross > 1.6 * max(ext, .001):
            return True
    return False


def net_trade(uni):
    """Collapse Imports and Exports into the net position of the scope."""
    imp = uni.get("Imports") or []
    exp = uni.get("Exports") or []
    n = max(len(imp), len(exp))
    if not n:
        return uni
    net = [(imp[i] if i < len(imp) else 0) + (exp[i] if i < len(exp) else 0)
           for i in range(n)]
    out = dict(uni)
    out["Imports"] = [v if v > 0 else 0.0 for v in net]
    out["Exports"] = [v if v < 0 else 0.0 for v in net]
    return out


def stack(ax, axis, uni, up, down, fs, netline=True):
    """Step-filled stack, generation up and exports/charging down."""
    n = len(axis)
    x = list(range(n + 1))

    def edges(v):                          # repeat the last slot to close the step
        return list(v) + [v[-1]]

    for keys, sign in ((up, 1), (down, -1)):
        base = [0.0] * (n + 1)
        for k in keys:
            v = uni.get(k)
            if not v or not any(v):
                continue
            v = edges([sign * abs(s) for s in v])
            top = [b + s for b, s in zip(base, v)]
            ax.fill_between(x, base, top, step="post", zorder=2,
                            **face(k, k in HATCHED))
            base = top

    dem = uni.get("Demand")
    if dem:
        ax.step(x, edges(dem), where="post", color=COLORS["Demand"],
                linewidth=1.0, zorder=4)
    imp, exp = uni.get("Imports"), uni.get("Exports")
    if netline and (imp or exp):
        # Exports come out of the model already negative, so the two series add.
        net = [(imp[i] if imp else 0) + (exp[i] if exp else 0) for i in range(n)]
        ax.step(x, edges(net), where="post", color=NETIMPORT,
                linewidth=1.0, zorder=5)

    # One tick per season block, labelled with the season.
    bounds, labels = [], []
    for i, a in enumerate(axis):
        if i == 0 or a["q"] != axis[i - 1]["q"]:
            bounds.append(i)
            labels.append(a["q"])
    ax.set_xticks([b + (n / len(bounds)) / 2 for b in bounds])
    ax.set_xticklabels(labels, fontsize=fs)
    for b in bounds[1:]:
        ax.axvline(b, color="#ccd4de", linewidth=.5, zorder=3)
    ax.axhline(0, color="#8b96a5", linewidth=.6, zorder=3)
    ax.set_xlim(0, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(HERE / "cache" / "simulations_run_20260825.json"))
    p.add_argument("--scope", default="Georgia")
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--years", default="2025,2030")
    p.add_argument("--width", type=float, default=5.6)
    p.add_argument("--height", type=float, default=2.5)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    d = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    block = d["dispatch"][a.scope][a.scenario]
    years = [y.strip() for y in a.years.split(",")]
    up = d["fuel_order"] + ["Imports", "Unmet demand"]
    down = ["Exports", "Storage Charge"]
    fs = 6.5
    # Netted scopes still show the two bands and the same legend as everyone
    # else: the caption explains the netting, the chart stays comparable.
    netted = internal_heavy(d, a.scope, a.scenario, block)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": fs,
        "axes.edgecolor": "#ccd4de", "axes.linewidth": .6,
        "text.color": INK, "axes.labelcolor": SOFT,
        "xtick.color": SOFT, "ytick.color": SOFT,
        "xtick.major.size": 0, "ytick.major.size": 2,
        "ytick.major.width": .6, "ytick.major.pad": 2,
        "xtick.major.pad": 2, "xtick.labelsize": fs, "ytick.labelsize": fs,
    })
    fig, axes = plt.subplots(len(years), 1, figsize=(a.width, a.height),
                             dpi=a.dpi, sharex=True)
    if len(years) == 1:
        axes = [axes]

    used = []
    for ax, y in zip(axes, years):
        uni = block["years"][y]
        if netted:
            uni = net_trade(uni)
        stack(ax, block["axis"], uni, up, down, fs, netline=True)
        ax.set_ylabel("MW", fontsize=fs, labelpad=2)
        ax.set_title(y, fontsize=fs + 1.5, fontweight="bold", loc="left",
                     pad=2, color=INK)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.grid(True, color=GRID, linewidth=.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for k in up + down:
            v = uni.get(k)
            if v and any(v) and k not in used:
                used.append(k)

    handles = [Patch(label=k, **face(k, k in HATCHED)) for k in used]
    handles.append(Line2D([], [], color=COLORS["Demand"], lw=1.1,
                          label="Demand"))
    handles.append(Line2D([], [], color=NETIMPORT, lw=1.1, label="Net imports"))

    title = a.title or "%s hourly dispatch, %s" % (a.scope, a.scenario)
    fig.suptitle(title, fontsize=fs + 1, fontweight="bold", color=INK,
                 x=.012, y=.995, ha="left", va="top")

    # Legend as a single right-hand column: the plots keep the full width of the
    # slide block and the labels never wrap.
    right = 1 - LEGEND_IN / a.width
    fig.tight_layout(pad=.3, h_pad=.9, rect=(.012, 0, right, .955))
    # A wide fleet (Turkiye) brings twice as many entries as Armenia, so the
    # column is shrunk until it fits rather than running off the bottom.
    lfs, lsp = fs, .45
    while len(handles) * lfs * (1 + lsp) > a.height * 72 * .93 and lfs > 4.2:
        lfs -= .2
        lsp = max(.18, lsp - .03)
    fig.legend(handles=handles, loc="center left", ncol=1, fontsize=lfs,
               frameon=False, handlelength=1.1, handleheight=.9,
               handletextpad=.45, labelspacing=lsp, borderpad=0,
               bbox_to_anchor=(right + .015, .48))

    out = Path(a.out) if a.out else (
        HERE.parents[2] / "Data" / "results" / "slides"
        / ("dispatch_%s_%s.png" % (a.scope.lower(), "_".join(years))))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    print("%s  %.2f x %.2f in" % (out, a.width, a.height))


if __name__ == "__main__":
    main()
