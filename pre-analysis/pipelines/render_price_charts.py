# -*- coding: utf-8 -*-
"""Render the section-3 charts of calibration_review.html from the price pipeline.

The report holds hand-written inline SVG and no charting library, so the charts
are emitted here as SVG text rather than drawn by matplotlib and linked as
images. Two reasons: the report must stay a single self-contained file, and the
existing 23 charts already follow that idiom -- a PNG would read as a foreign
body next to them.

Everything is derived from the pipeline outputs, never typed by hand, so a rerun
of eu_price.py / eu_price_level.py followed by a rerun of this script keeps the
report in step with the numbers. Hand-typed coordinates would silently go stale
the first time a level moved, which is worse than having no chart at all.

No prose is placed inside the SVG. The report is bilingual through .lf/.le spans
that rely on `display:contents`, which does not survive inside an SVG subtree, so
only language-neutral tokens (years, integers, scenario identifiers) are drawn;
every sentence lives in the surrounding HTML wrappers.

Usage
-----
    python render_price_charts.py            # write the fragment
    python render_price_charts.py --splice   # write it and inject into the report
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

# Imported rather than restated so the milestone table can never drift from the
# horizons the level pipeline actually reads.
from eu_price_level import TYNDP_SCENARIOS

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "output_prices"
PHOURS = HERE.parent.parent / "epm" / "input" / "data_blacksea" / "pHours.csv"
REPORT = (HERE.parent.parent.parent / "Data" / "calibration"
          / "calibration_review.html")
FRAGMENT = OUT / "charts_section3.html"

BEGIN = "<!--CHARTS3:BEGIN-->"
END = "<!--CHARTS3:END-->"

ZONES = ["Romania", "Bulgaria", "Greece"]
ZONE_FR = {"Romania": "Roumanie", "Bulgaria": "Bulgarie", "Greece": "Gr&egrave;ce"}

# Colour encodes the level, dash encodes the provenance: the three published
# ENTSO-E trajectories are solid, the two built in-house are dashed. A reader who
# only looks at the picture still sees which lines the study invented.
SCEN = [
    ("VERY_LOW", "#2b6b2e", False, "GA"),
    ("LOW", "#4f9e8a", False, "DE"),
    ("CENTRAL", "#1E6DB8", False, "NT+"),
    ("HIGH", "#c0682a", True, "maison"),
    ("EU_HIGH", "#a5381f", True, "maison"),
]
SCEN_COLOR = {s: c for s, c, _, _ in SCEN}
SCEN_DASH = {s: d for s, _, d, _ in SCEN}
OBS_COLOR = "#8b96a8"
GRID = "#eef1f6"
AXIS = "#9aa5b5"

SEASON_COLOR = {"Q1": "#1E6DB8", "Q2": "#4f9e8a", "Q3": "#c0682a", "Q4": "#8b96a8"}
TCOLS = [f"t{h:02d}" for h in range(1, 25)]


# --------------------------------------------------------------------------- #
# tiny SVG helpers
# --------------------------------------------------------------------------- #
def f(v: float) -> str:
    """Trim float noise so the emitted markup stays readable in a diff."""
    return f"{v:.1f}".rstrip("0").rstrip(".")


class Frame:
    """A linear/log plot box mapping data coordinates to SVG user units."""

    def __init__(self, x0, x1, ytop, ybot, xlo, xhi, ylo, yhi, ylog=False):
        self.x0, self.x1, self.ytop, self.ybot = x0, x1, ytop, ybot
        self.xlo, self.xhi, self.ylo, self.yhi = xlo, xhi, ylo, yhi
        self.ylog = ylog

    def px(self, v):
        return self.x0 + (v - self.xlo) / (self.xhi - self.xlo) * (self.x1 - self.x0)

    def py(self, v):
        if self.ylog:
            a, b, c = math.log(self.ylo), math.log(self.yhi), math.log(max(v, 1e-9))
            return self.ybot - (c - a) / (b - a) * (self.ybot - self.ytop)
        return self.ybot - (v - self.ylo) / (self.yhi - self.ylo) * (self.ybot - self.ytop)

    def hgrid(self, ticks, label=True, unit=""):
        out = []
        for t in ticks:
            y = self.py(t)
            out.append(f'<line x1="{f(self.x0)}" y1="{f(y)}" x2="{f(self.x1)}" '
                       f'y2="{f(y)}" stroke="{GRID}" stroke-width="1"/>')
            if label:
                out.append(f'<text x="{f(self.x0 - 3)}" y="{f(y + 2.5)}" font-size="7" '
                           f'fill="{AXIS}" text-anchor="end">{t}{unit}</text>')
        return "".join(out)

    def xticks(self, ticks, fmt=str):
        out = []
        for t in ticks:
            x = self.px(t)
            out.append(f'<text x="{f(x)}" y="{f(self.ybot + 11)}" font-size="7" '
                       f'fill="{AXIS}" text-anchor="middle">{fmt(t)}</text>')
        return "".join(out)

    def poly(self, xs, ys, color, width=2.0, dash=False, opacity=1.0):
        pts = " ".join(f"{f(self.px(x))},{f(self.py(y))}" for x, y in zip(xs, ys))
        d = ' stroke-dasharray="4,2.5"' if dash else ""
        o = f' opacity="{opacity}"' if opacity < 1 else ""
        return (f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
                f'stroke-linejoin="round" stroke-linecap="round"{d}{o} points="{pts}"/>')


def svg(w, h, body, extra=""):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'preserveAspectRatio="xMidYMid meet"{extra}>{body}</svg>')


def chart(title_fr, title_en, inner):
    return (f'<div class="chart"><div class="ct"><span class="lf">{title_fr}</span>'
            f'<span class="le">{title_en}</span></div>{inner}</div>')


def legend(items):
    """items: (colour, label_fr, label_en, dashed)."""
    out = []
    for color, lf, le, dash in items:
        style = (f"background:repeating-linear-gradient(90deg,{color} 0 3px,"
                 f"transparent 3px 5px)" if dash else f"background:{color}")
        out.append(f'<span><i style="{style}"></i><span class="lf">{lf}</span>'
                   f'<span class="le">{le}</span></span>')
    return f'<div class="leg">{"".join(out)}</div>'


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load():
    lvl = pd.read_csv(OUT / "level_L.csv")
    obs = pd.read_csv(OUT / "observed_annual_real.csv", index_col=0)
    shape = pd.read_csv(OUT / "shape_S.csv")
    qcL = json.loads((OUT / "qc_level_L.json").read_text(encoding="utf-8"))
    qcS = json.loads((OUT / "qc_shape_S.json").read_text(encoding="utf-8"))
    hours = pd.read_csv(PHOURS)
    return lvl, obs, shape, qcL, qcS, hours


# --------------------------------------------------------------------------- #
# A. the fan chart -- observed history plus the five forward trajectories
# --------------------------------------------------------------------------- #
def chart_fan(lvl, obs, qcL):
    """One panel per zone, 2015-2053.

    The y axis is clipped at 220 EUR/MWh. The 2022 observation sits at 286-302,
    so an un-clipped axis would compress the whole 40-110 band -- where every
    trajectory actually lives -- into the bottom third. The spike is kept
    visible as an annotated arrow rather than dropped.
    """
    ycap = 220.0
    panels = []
    for z in ZONES:
        fr = Frame(30, 374, 12, 128, 2015, 2053, 0, ycap)
        body = [fr.hgrid([0, 50, 100, 150, 200])]

        # anchor separator: observed to the left, projected to the right
        xa = fr.px(2024)
        body.append(f'<line x1="{f(xa)}" y1="12" x2="{f(xa)}" y2="128" '
                    f'stroke="#c8d2e0" stroke-width="1" stroke-dasharray="3,3"/>')

        o = obs[z].dropna()
        body.append(fr.poly(list(o.index), [min(v, ycap) for v in o.values],
                            OBS_COLOR, 1.8))
        peak_y = o.idxmax()
        body.append(f'<text x="{f(fr.px(peak_y))}" y="9" font-size="7" '
                    f'fill="{OBS_COLOR}" text-anchor="middle">&#8593;{o.max():.0f}</text>')

        for name, color, dash, _ in SCEN:
            s = lvl[(lvl.zone == z) & (lvl.scenario == name)].sort_values("year")
            if s.empty:
                continue
            body.append(fr.poly(list(s.year), [min(v, ycap) for v in s.L_eur2024],
                                color, 2.0 if name != "CENTRAL" else 2.4, dash))

        a = qcL["zones"][z]["anchor_real_eur2024"]
        body.append(f'<circle cx="{f(xa)}" cy="{f(fr.py(a))}" r="2.6" fill="#12355b"/>')
        body.append(fr.xticks([2015, 2025, 2035, 2045], str))
        panels.append(chart(f"{ZONE_FR[z]} &#183; &#8364;/MWh r&eacute;els 2024",
                            f"{z} &#183; real 2024 EUR/MWh",
                            svg(380, 144, "".join(body))))

    leg = legend([
        (OBS_COLOR, "Observ&eacute; (r&eacute;el)", "Observed (real)", False),
        ("#2b6b2e", "VERY_LOW &#183; GA", "VERY_LOW &#183; GA", False),
        ("#4f9e8a", "LOW &#183; DE", "LOW &#183; DE", False),
        ("#1E6DB8", "CENTRAL &#183; NT+", "CENTRAL &#183; NT+", False),
        ("#c0682a", "HIGH &#183; maison", "HIGH &#183; in-house", True),
        ("#a5381f", "EU_HIGH &#183; maison", "EU_HIGH &#183; in-house", True),
    ])
    return leg + f'<div class="cols3">{"".join(panels)}</div>'


# --------------------------------------------------------------------------- #
# B. the ladder at 2040 -- published versus in-house, on a log axis
# --------------------------------------------------------------------------- #
def chart_ladder(lvl, year=2040):
    """Horizontal bars on a logarithmic axis.

    Log is deliberate here and nowhere else: HIGH is built as CENTRAL^2/LOW, so
    on a log axis LOW and HIGH fall at equal distance either side of CENTRAL.
    The chart therefore demonstrates the mirror rule instead of asserting it.
    """
    lo, hi = 30.0, 260.0
    panels = []
    for z in ZONES:
        s = lvl[(lvl.zone == z) & (lvl.year == year)].set_index("scenario")
        central = s.loc["CENTRAL", "L_eur2024"]
        body = []

        def bx(v):
            return 74 + (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) * 270

        for t in (40, 60, 100, 160, 240):
            x = bx(t)
            body.append(f'<line x1="{f(x)}" y1="8" x2="{f(x)}" y2="110" '
                        f'stroke="{GRID}" stroke-width="1"/>')
            body.append(f'<text x="{f(x)}" y="141" font-size="7" fill="{AXIS}" '
                        f'text-anchor="middle">{t}</text>')

        xc = bx(central)
        body.append(f'<line x1="{f(xc)}" y1="8" x2="{f(xc)}" y2="130" '
                    f'stroke="#12355b" stroke-width="1" stroke-dasharray="3,2"/>')

        for i, (name, color, dash, src) in enumerate(SCEN):
            if name not in s.index:
                continue
            v = float(s.loc[name, "L_eur2024"])
            y = 14 + i * 20
            x = bx(v)
            fill = "url(#hatch)" if dash else color
            body.append(f'<rect x="74" y="{y}" width="{f(x - 74)}" height="13" rx="2" '
                        f'fill="{fill}" stroke="{color}" stroke-width="1"/>')
            body.append(f'<text x="70" y="{y + 9.5}" font-size="7.2" fill="#33445e" '
                        f'text-anchor="end" font-weight="700">{name}</text>')
            body.append(f'<text x="{f(x + 4)}" y="{y + 9.5}" font-size="7.2" '
                        f'fill="#5a6678">{v:.0f} &#215;{v / central:.2f}</text>')

        # The mirror bracket. HIGH = CENTRAL^2/LOW means CENTRAL/LOW and
        # HIGH/CENTRAL are the same factor, so on a log axis the two spans have
        # the same width. Drawing them makes the construction rule self-evident
        # rather than something the reader must take on faith from the prose.
        if {"LOW", "HIGH"} <= set(s.index):
            low = float(s.loc["LOW", "L_eur2024"])
            xl, xh = bx(low), bx(float(s.loc["HIGH", "L_eur2024"]))
            ratio = central / low
            yb = 122
            for xa, xz, lab in ((xl, xc, f"&#247;{ratio:.2f}"),
                                (xc, xh, f"&#215;{ratio:.2f}")):
                body.append(f'<line x1="{f(xa)}" y1="{yb}" x2="{f(xz)}" y2="{yb}" '
                            f'stroke="#8a94a6" stroke-width="1"/>')
                for e in (xa, xz):
                    body.append(f'<line x1="{f(e)}" y1="{yb - 3}" x2="{f(e)}" '
                                f'y2="{yb + 3}" stroke="#8a94a6" stroke-width="1"/>')
                body.append(f'<text x="{f((xa + xz) / 2)}" y="{yb - 5}" font-size="7.2" '
                            f'fill="#5a6678" text-anchor="middle">{lab}</text>')

        defs = ('<defs><pattern id="hatch" width="4" height="4" '
                'patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
                '<rect width="4" height="4" fill="#fff"/>'
                '<line x1="0" y1="0" x2="0" y2="4" stroke="#c0682a" '
                'stroke-width="2.4"/></pattern></defs>')
        panels.append(chart(f"{ZONE_FR[z]} &#183; {year} &#183; &#8364;/MWh",
                            f"{z} &#183; {year} &#183; EUR/MWh",
                            svg(380, 150, defs + "".join(body))))

    leg = legend([
        ("#1E6DB8", "Plein = publi&eacute; ENTSO-E", "Solid = ENTSO-E published", False),
        ("#c0682a", "Hachur&eacute; = construit maison", "Hatched = built in-house", True),
    ])
    return leg + f'<div class="cols3">{"".join(panels)}</div>'


# --------------------------------------------------------------------------- #
# C. the hourly shape S -- weighted mean day per season
# --------------------------------------------------------------------------- #
def chart_shape(shape, hours):
    """The 28 day-types collapse to 4 seasonal mean days, weighted by pHours.

    An unweighted mean would over-represent the three extreme days, which carry
    small weights by construction, and would show a shape the model never sees.
    """
    w = hours.groupby(["q", "d"])["t01"].first()
    panels = []
    for z in ZONES:
        sz = shape[shape.zone == z]
        fr = Frame(30, 374, 10, 108, 1, 24, 0, 2.2)
        body = [fr.hgrid([0, 0.5, 1.0, 1.5, 2.0])]
        y1 = fr.py(1.0)
        body.append(f'<line x1="30" y1="{f(y1)}" x2="374" y2="{f(y1)}" '
                    f'stroke="#12355b" stroke-width="1" stroke-dasharray="2,2" '
                    f'opacity="0.55"/>')
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            sub = sz[sz.q == q]
            if sub.empty:
                continue
            wt = sub.d.map(lambda d: w.get((q, d), 1.0)).astype(float)
            prof = (sub[TCOLS].mul(wt, axis=0).sum() / wt.sum())
            body.append(fr.poly(range(1, 25), list(prof), SEASON_COLOR[q], 1.9))
        body.append(fr.xticks([1, 6, 12, 18, 24], lambda t: f"{t:02d}h"))
        panels.append(chart(f"{ZONE_FR[z]} &#183; S, moyenne = 1",
                            f"{z} &#183; S, mean = 1",
                            svg(380, 124, "".join(body))))

    leg = legend([
        ("#1E6DB8", "Q1 hiver", "Q1 winter", False),
        ("#4f9e8a", "Q2 printemps", "Q2 spring", False),
        ("#c0682a", "Q3 &eacute;t&eacute;", "Q3 summer", False),
        ("#8b96a8", "Q4 automne", "Q4 autumn", False),
    ])
    return leg + f'<div class="cols3">{"".join(panels)}</div>'


# --------------------------------------------------------------------------- #
# D. the price duration curve -- where the reduction loses the low tail
# --------------------------------------------------------------------------- #
def chart_duration(qcS):
    """True versus reconstructed duration curve, both rescaled to the true mean.

    The rescaling is the point. S carries the shape only; the level comes from
    L. Plotting raw reconstructed values would show the +10 % level bias on top
    of the shape error and make the shape error unreadable. Rescaling isolates
    exactly what the G2bis gate tests.
    """
    order = ["P99", "P95", "P90", "P75", "P50", "P25", "P10", "P5", "P1"]
    panels = []
    for z in ZONES:
        g = qcS["zones"][z]["G2bis_duration_curve"]
        mean = g["annual_mean_true"]
        xs = [100 - int(p[1:]) for p in order]
        true = [g[p]["true_norm"] * mean for p in order]
        rec = [g[p]["reconstructed_norm"] * mean for p in order]
        fr = Frame(30, 374, 10, 106, 0, 100, 0, 320)
        body = [fr.hgrid([0, 100, 200, 300])]

        # Shade the whole area between the two curves. It collapses to a hairline
        # everywhere the reconstruction works, so the one place it opens up --
        # the bottom decile -- is the only thing the eye lands on. A single
        # annotated segment made the reader hunt for it.
        ring = ([f"{f(fr.px(x))},{f(fr.py(v))}" for x, v in zip(xs, true)] +
                [f"{f(fr.px(x))},{f(fr.py(v))}" for x, v in zip(reversed(xs), reversed(rec))])
        body.append(f'<polygon points="{" ".join(ring)}" fill="#a5381f" '
                    f'opacity="0.22"/>')
        body.append(fr.poly(xs, true, "#12355b", 2.0))
        body.append(fr.poly(xs, rec, "#c0682a", 2.0, dash=True))

        # the accepted miss: the bottom decile the representative days never reach
        xp = fr.px(90)
        yt = fr.py(g["P10"]["true_norm"] * mean)
        yr = fr.py(g["P10"]["reconstructed_norm"] * mean)
        body.append(f'<line x1="{f(xp)}" y1="{f(yt)}" x2="{f(xp)}" y2="{f(yr)}" '
                    f'stroke="#a5381f" stroke-width="2.6"/>')
        # the label is pulled clear of the curve and tied back with a leader:
        # sitting it on the gap put it straight on top of the two lines
        lx, ly = fr.px(78), fr.py(190)
        body.append(f'<line x1="{f(lx + 12)}" y1="{f(ly + 2)}" x2="{f(xp - 1)}" '
                    f'y2="{f((yt + yr) / 2)}" stroke="#a5381f" stroke-width="0.7" '
                    f'opacity="0.7"/>')
        # no white-halo stroke here: paint-order is not honoured everywhere, and
        # where it is ignored the halo paints over the glyphs and hides them
        body.append(f'<text x="{f(lx)}" y="{f(ly)}" font-size="7.4" fill="#a5381f" '
                    f'text-anchor="middle" font-weight="700">'
                    f'P10 +{g["P10"]["dev_pct_of_mean"]:.0f}%</text>')
        body.append(fr.xticks([0, 25, 50, 75, 100], lambda t: f"{t}%"))
        panels.append(chart(f"{ZONE_FR[z]} &#183; &#8364;/MWh &#183; % des heures",
                            f"{z} &#183; EUR/MWh &#183; % of hours",
                            svg(380, 122, "".join(body))))

    leg = legend([
        ("#12355b", "Observ&eacute; 2023 (8 760 h)", "Observed 2023 (8,760 h)", False),
        ("#c0682a", "Reconstruit (28 jours &#215; 24 h)", "Reconstructed (28 days &#215; 24 h)", True),
        ("#a5381f", "&Eacute;cart au d&eacute;cile bas", "Bottom-decile gap", False),
    ])
    return leg + f'<div class="cols3">{"".join(panels)}</div>'


# --------------------------------------------------------------------------- #
# E. anchor sensitivity -- how much the ladder moves if the anchor moves
# --------------------------------------------------------------------------- #
def chart_sensitivity(qcL):
    """Mean absolute deviation over 2024-2053 when the anchor becomes 2023-2024.

    This is the honest robustness statement: the anchor is one observed year and
    2024 was unusually calm, so the reader is entitled to know what a different
    anchor would have done to each trajectory.
    """
    rows = []
    for z in ZONES:
        sc = qcL["zones"][z]["anchor_sensitivity"]["scenarios"]
        rows.append((z, {k: v["mean_abs_dev_pct"] for k, v in sc.items()}))
    vmax = max(max(d.values()) for _, d in rows) * 1.15

    W, H = 760, 176
    x0, x1, ytop, ybot = 40, 742, 14, 132
    gw = (x1 - x0) / len(SCEN)
    body = []
    for t in [0, 1, 2, 3]:
        if t > vmax:
            continue
        y = ybot - t / vmax * (ybot - ytop)
        body.append(f'<line x1="{x0}" y1="{f(y)}" x2="{x1}" y2="{f(y)}" '
                    f'stroke="{GRID}" stroke-width="1"/>')
        body.append(f'<text x="{x0 - 4}" y="{f(y + 2.5)}" font-size="8" fill="{AXIS}" '
                    f'text-anchor="end">{t}%</text>')

    zcol = {"Romania": "#1E6DB8", "Bulgaria": "#4f9e8a", "Greece": "#c0682a"}
    for i, (name, _, _, _) in enumerate(SCEN):
        gx = x0 + i * gw
        bw = gw * 0.20
        for j, (z, d) in enumerate(rows):
            v = d.get(name, 0.0)
            h = v / vmax * (ybot - ytop)
            bxx = gx + gw * 0.14 + j * bw
            body.append(f'<rect x="{f(bxx)}" y="{f(ybot - h)}" width="{f(bw * 0.86)}" '
                        f'height="{f(h)}" rx="1.5" fill="{zcol[z]}"/>')
            body.append(f'<text x="{f(bxx + bw * 0.43)}" y="{f(ybot - h - 3)}" '
                        f'font-size="7.5" fill="#5a6678" text-anchor="middle">'
                        f'{v:.1f}</text>')
        body.append(f'<text x="{f(gx + gw / 2)}" y="{ybot + 13}" font-size="8.5" '
                    f'fill="#33445e" text-anchor="middle" font-weight="700">{name}</text>')

    leg = legend([(zcol[z], ZONE_FR[z], z, False) for z in ZONES])
    return leg + chart(
        "&Eacute;cart moyen absolu 2024&#8211;2053 si l&#39;ancrage devient la moyenne 2023&#8211;2024",
        "Mean absolute deviation 2024&#8211;2053 if the anchor becomes the 2023&#8211;2024 mean",
        svg(W, H, "".join(body)))


# --------------------------------------------------------------------------- #
# F. the controls table -- one row per control, question beside its answer
# --------------------------------------------------------------------------- #
def gate_matrix(qcL):
    """One row per control rather than one row per zone.

    Gate names are internal identifiers: a column headed "G3ter ladder order"
    tells the reader nothing, and putting the identifiers in the headers leaves
    the question in a different place from its answer. Each row now carries the
    question it asks and its tolerance, which also lets the note underneath drop
    to the two diagnoses a table cannot express.
    """
    def cell(text, kind):
        c = {"ok": "#2b6b2e", "warn": "#8a5a12", "bad": "#a5381f",
             "info": "#5a6678"}[kind]
        d = ("" if kind == "info" else
             f'<span class="dot" style="background:{c}"></span>')
        w = "700" if kind != "info" else "400"
        return (f'<td style="text-align:right;vertical-align:top;color:{c};'
                f'font-weight:{w}">{d}{text}</td>')

    def q_cell(fr, en):
        # the question wraps over two or three lines; top-aligning the whole row
        # keeps each result on the same line as the question it answers
        return (f'<td style="text-align:left;vertical-align:top;font-weight:400;'
                f'color:#54617a">'
                f'<span class="lf">{fr}</span><span class="le">{en}</span></td>')

    def name_cell(code, fr, en):
        return (f'<td style="text-align:left;vertical-align:top"><b>{code}</b><br>'
                f'<span style="font-size:.72rem;color:#7a869c">'
                f'<span class="lf">{fr}</span><span class="le">{en}</span></span></td>')

    head = ('<tr><th style="text-align:left"><span class="lf">Contr&#244;le</span>'
            '<span class="le">Control</span></th>'
            '<th style="text-align:left"><span class="lf">Ce qu&#39;il demande'
            '</span><span class="le">What it asks</span></th>')
    for z in ZONES:
        head += (f'<th style="text-align:right"><span class="lf">{ZONE_FR[z]}</span>'
                 f'<span class="le">{z}</span></th>')
    head += '</tr>'

    rows = []

    # 1. currency
    cur = qcL["currency_gate_G3bis"]["pass"]
    r = (name_cell("G3bis", "Devise", "Currency") +
         q_cell("Les s&eacute;ries longues sont-elles <b>toutes en euros</b> ? "
                "Roumanie cot&eacute;e en RON jusqu&#39;en 2021, Bulgarie en BGN : "
                "sinon le niveau roumain d&#39;avant-crise ressort "
                "<b>4,7&#215; trop haut</b>.",
                "Are the long series <b>all in euros</b>? Romania quoted in RON "
                "through 2021, Bulgaria in BGN: otherwise the Romanian pre-crisis "
                "level comes out <b>4.7&#215; too high</b>."))
    for _ in ZONES:
        r += cell("OK", "ok" if cur else "bad")
    rows.append(f"<tr>{r}</tr>")

    # 2. back-cast
    r = (name_cell("G3", "R&eacute;tro-test 2023", "2023 back-cast") +
         q_cell("Rembobin&eacute;e jusqu&#39;&#224; 2023, la trajectoire centrale "
                "<b>retombe-t-elle sur le prix observ&eacute;</b> ? "
                "Tol&eacute;rance <b>&#177;2 %</b>.",
                "Rewound to 2023, does the central trajectory <b>land on the "
                "observed price</b>? Tolerance <b>&#177;2 %</b>."))
    for z in ZONES:
        bc = qcL["zones"][z]["G3_backcast"]
        dev = bc["dev_pct"]
        # a residual of -0.001 % formats as "-0.0%", which reads like a real miss
        # typographic minus, as used everywhere else in the report
        txt = ("0,0 %" if abs(dev) < 0.05
               else f"{dev:+.1f} %".replace(".", ",").replace("-", "&#8722;"))
        kind = "ok" if bc["pass"] else ("warn" if abs(dev) < 5 else "bad")
        r += cell(txt, kind)
    rows.append(f"<tr>{r}</tr>")

    # 3. ladder ordering
    r = (name_cell("G3ter", "Ordre de l&#39;&eacute;chelle", "Ladder order") +
         q_cell("Les trois trajectoires publi&eacute;es restent-elles <b>dans "
                "l&#39;ordre</b> (VERY_LOW &#8804; LOW &#8804; CENTRAL), les bornes "
                "hautes au-dessus ? Tol&eacute;rance <b>1 %</b>.",
                "Do the three published trajectories stay <b>in order</b> "
                "(VERY_LOW &#8804; LOW &#8804; CENTRAL), upper bounds above? "
                "Tolerance <b>1 %</b>."))
    for z in ZONES:
        lad = qcL["zones"][z]["G3ter_ladder_order"]
        txt = "OK" if lad["pass"] else ("+%.2f %%" % lad["max_inversion_pct"]).replace(".", ",")
        r += cell(txt, "ok" if lad["pass"] else "warn")
    rows.append(f"<tr>{r}</tr>")

    # 4. mirror -- reported, never gated
    r = (name_cell("&#8212;", "Miroir 2040 &#183; indicatif",
                   "2040 mirror &#183; indicative") +
         q_cell("Recoupement seul : EU_HIGH tombe-t-il pr&#232;s du miroir de "
                "VERY_LOW ? <b>Jamais bloquant</b> &#8212; les deux bornes hautes ne "
                "suivent pas la m&#234;me logique.",
                "Cross-check only: does EU_HIGH land near the mirror of VERY_LOW? "
                "<b>Never blocking</b> &#8212; the two upper bounds follow different "
                "logics."))
    for z in ZONES:
        mir = qcL["zones"][z]["upper_bound_mirror_check"]["years"]["2040"]["dev_pct"]
        r += cell(f"{mir:+.0f} %", "info")
    rows.append(f"<tr>{r}</tr>")

    return ('<div class="gridtbl"><table class="cal"><thead>' + head +
            '</thead><tbody>' + "".join(rows) + '</tbody></table></div>')


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def level_table(lvl):
    years = [2030, 2040, 2050]
    head = ('<tr><th><span class="lf">Trajectoire</span><span class="le">Trajectory</span></th>'
            '<th><span class="lf">Source</span><span class="le">Source</span></th>')
    for z in ZONES:
        head += (f'<th colspan="3" style="text-align:center">'
                 f'<span class="lf">{ZONE_FR[z]}</span><span class="le">{z}</span></th>')
    head += '</tr><tr><th></th><th></th>' + ''.join(
        f'<th style="text-align:right">{y}</th>' for _ in ZONES for y in years) + '</tr>'

    badge = {"GA": '<span class="srcbadge sb-ndp">ENTSO-E GA</span>',
             "DE": '<span class="srcbadge sb-ndp">ENTSO-E DE</span>',
             "NT+": '<span class="srcbadge sb-ndp">ENTSO-E NT+</span>',
             "maison": '<span class="srcbadge sb-team">'
                       '<span class="lf">maison</span><span class="le">in-house</span></span>'}
    rows = []
    for name, color, _, src in SCEN:
        cells = ""
        for z in ZONES:
            for y in years:
                v = lvl[(lvl.zone == z) & (lvl.scenario == name)
                        & (lvl.year == y)].L_eur2024
                cells += f'<td style="text-align:right">{float(v.iloc[0]):.0f}</td>'
        rows.append(f'<tr><td style="text-align:left"><span class="dot" '
                    f'style="background:{color}"></span><b>{name}</b></td>'
                    f'<td style="text-align:left">{badge[src]}</td>{cells}</tr>')
    return ('<div class="gridtbl"><table class="cal"><thead>' + head +
            '</thead><tbody>' + "".join(rows) + '</tbody></table></div>')


def build():
    lvl, obs, shape, qcL, qcS, hours = load()
    fx = qcL["eur_usd_2024"]
    fx_fr = str(fx).replace(".", ",")   # the French side of the report uses a comma
    p = []

    # ---- Level L -------------------------------------------------------- #
    p.append('<div class="subh"><span class="lf">Niveau L &#8212; l&#39;&eacute;chelle '
             '2024&#8594;2053</span><span class="le">Level L &#8212; the ladder '
             '2024&#8594;2053</span></div>')
    p.append('<div class="body" style="margin-bottom:10px"><span class="lf">Cinq '
             'trajectoires, une seule grandeur : le <b>prix annuel moyen</b> de la zone, '
             'en <b>&#8364;/MWh r&eacute;els 2024</b>. Trois sont lues dans les livrables '
             'ENTSO-E, deux sont construites ici &#8212; le trait plein contre le trait '
             'pointill&eacute; le dit sur chaque graphique.</span><span class="le">Five '
             'trajectories, a single quantity: the zone&#39;s <b>annual mean price</b>, in '
             '<b>real 2024 EUR/MWh</b>. Three are read from ENTSO-E deliverables, two are '
             'built here &#8212; solid versus dashed says which on every chart.</span></div>')
    p.append(chart_fan(lvl, obs, qcL))
    p.append('<div class="pn"><span class="lf">Axe vertical <b>tronqu&eacute; &#224; 220 '
             '&#8364;/MWh</b> : le pic 2022 (286&#8211;302 selon la zone, fl&#232;che en '
             'haut) &#233;craserait sinon toute la bande 40&#8211;110 o&#249; vivent les '
             'trajectoires. Le trait vertical pointill&eacute; marque l&#39;<b>ancrage 2024'
             '</b> (point noir) : &#224; gauche de l&#39;observ&eacute;, &#224; droite du '
             'projet&eacute;.</span><span class="le">Vertical axis <b>clipped at 220 '
             'EUR/MWh</b>: the 2022 spike (286&#8211;302 depending on the zone, arrow at the '
             'top) would otherwise crush the entire 40&#8211;110 band where the trajectories '
             'live. The dashed vertical marks the <b>2024 anchor</b> (black dot): observed to '
             'its left, projected to its right.</span></div>')

    # ---- provenance ----------------------------------------------------- #
    p.append('<div class="subh"><span class="lf">D&#39;o&#249; viennent ces cinq niveaux'
             '</span><span class="le">Where the five levels come from</span></div>')
    p.append('<div class="cols3">')
    p.append('<div class="sec"><div class="sn"><span class="lf">Trois lectures publi&eacute;es'
             '</span><span class="le">Three published reads</span></div><div class="body">'
             '<span class="lf">CENTRAL, LOW et VERY_LOW sont les sc&eacute;narios <b>NT+, DE '
             'et GA</b> du <b>TYNDP 2024</b> (ann&eacute;e climatique 2009). Une seule '
             'grandeur est retenue : le co&#251;t marginal horaire moyen, lu <b>ligne 10 de '
             'la feuille horaire</b> et rep&eacute;r&eacute; par le code zone en ligne 12 '
             '&#8212; jamais dans l&#39;onglet de synth&#232;se, o&#249; DE et GA ne '
             'renseignent que 38 colonnes sur 224 et laissent BG00 et GR00 vides. Lu ainsi, '
             'NT+ <b>reproduit exactement</b> ses propres chiffres de synth&#232;se : c&#39;est '
             'ce qui prouve la comparabilit&eacute; des trois.</span><span class="le">CENTRAL, '
             'LOW and VERY_LOW are the <b>NT+, DE and GA</b> scenarios of the <b>TYNDP 2024</b> '
             '(climate year 2009). A single quantity is kept: the average hourly marginal cost, '
             'read from <b>row 10 of the hourly sheet</b> and located by the zone code on row 12 '
             '&#8212; never from the summary tab, where DE and GA populate only 38 of 224 '
             'columns and leave BG00 and GR00 empty. Read this way, NT+ <b>exactly reproduces</b> '
             'its own summary figures: that is what proves the three are comparable.</span>'
             '</div></div>')
    p.append('<div class="sec"><div class="sn"><span class="lf">Le pi&#232;ge de '
             'l&#39;ann&eacute;e mon&eacute;taire</span><span class="le">The money-year trap'
             '</span></div><div class="body"><span class="lf">L&#39;ann&eacute;e mon&eacute;taire '
             'des co&#251;ts marginaux TYNDP <b>n&#39;appara&#238;t ni dans le classeur, ni '
             'dans son readme, ni dans les lignes directrices, ni dans ERAA</b>. Elle n&#39;est '
             '&eacute;nonc&eacute;e que dans le <i>TYNDP 2024 Scenarios Methodology Report</i> '
             '(version finale, janvier 2025), <b>p. 73</b> : &#171; with prices in real terms '
             '(in &#8364; 2022) &#187;, d&#39;apr&#232;s le WEO 2022 de l&#39;AIE. D&#39;o&#249; '
             'un d&eacute;flateur <b>&#215;1,0792</b> vers 2024. Le choix n&#39;est pas cosm&eacute;tique : '
             'passer de &#8364;2020 &#224; &#8364;2022 d&eacute;place le niveau 2030 '
             'd&#39;environ <b>20 %</b> et am&eacute;liore le r&eacute;tro-test des trois zones.'
             '</span><span class="le">The money year of the TYNDP marginal costs <b>appears '
             'neither in the workbook, nor its readme, nor the implementation guidelines, nor '
             'ERAA</b>. It is stated only in the <i>TYNDP 2024 Scenarios Methodology Report</i> '
             '(final version, January 2025), <b>p. 73</b>: &#8220;with prices in real terms (in '
             '&#8364; 2022)&#8221;, after the IEA WEO 2022. Hence a <b>&#215;1.0792</b> deflator '
             'to 2024. The choice is not cosmetic: moving from &#8364;2020 to &#8364;2022 shifts '
             'the 2030 level by about <b>20 %</b> and improves the back-cast in all three zones.'
             '</span></div></div>')
    p.append('<div class="sec"><div class="sn"><span class="lf">Deux bornes hautes, deux '
             'logiques</span><span class="le">Two upper bounds, two logics</span></div>'
             '<div class="body"><span class="lf">La famille publi&eacute;e '
             'n&#39;ouvre que <b>vers le bas</b> : NT+ est le plus haut des trois. Il faut donc '
             'construire le haut. <b>HIGH</b> est le <b>miroir g&eacute;om&eacute;trique</b> de '
             'LOW autour de CENTRAL (CENTRAL&#178;/LOW) &#8212; l&#39;incertitude de prix est '
             'sym&eacute;trique <b>en logarithme</b>, pas en niveau. <b>EU_HIGH</b> est tout '
             'autre chose : la <b>moyenne observ&eacute;e 2021&#8211;2023</b>, maintenue plate '
             '&#8212; non pas une pr&eacute;vision mais un <b>test de r&eacute;sistance</b>, la '
             'crise qui dure. Les deux ne sont donc <b>pas ordonn&eacute;es entre elles</b>, et '
             'le contr&#244;le d&#39;ordre ne le pr&eacute;tend pas.</span><span class="le">The '
             'published family only opens <b>downwards</b>: NT+ is the highest of the three. The '
             'top must therefore be built. <b>HIGH</b> is the <b>geometric mirror</b> of LOW '
             'about CENTRAL (CENTRAL&#178;/LOW) &#8212; price uncertainty is symmetric <b>in '
             'logs</b>, not in levels. <b>EU_HIGH</b> is a different object: the <b>observed '
             '2021&#8211;2023 mean</b>, held flat &#8212; not a forecast but a <b>stress test</b>, '
             'the crisis that lasts. The two are therefore <b>not ranked against each other</b>, '
             'and the ordering gate does not claim they are.</span></div></div>')
    p.append('</div>')

    # ---- how the yearly series is filled in ------------------------------ #
    # This is the one construction step that is entirely ours and shows up in a
    # gate result (the Romanian crossing), so it is written out rather than left
    # for a reader to infer from the fan chart.
    p.append('<div class="sec"><div class="sn"><span class="lf">Remplir les '
             'ann&eacute;es interm&eacute;diaires &#8212; et pourquoi les courbes se '
             'croisent avant 2031</span><span class="le">Filling the years in between '
             '&#8212; and why the curves cross before 2031</span></div>'
             '<div class="body"><span class="lf">EPM demande <b>une valeur par '
             'ann&eacute;e</b> de 2024 &#224; 2053 ; le TYNDP n&#39;en publie que trois ou '
             'quatre. Chaque trajectoire part donc du <b>prix observ&eacute; 2024</b> '
             '&#8212; ancrage commun aux cinq &#8212; puis suit une droite jusqu&#39;&#224; '
             'son premier point publi&eacute;, puis de point publi&eacute; en point '
             'publi&eacute;. Sans cet ancrage la s&eacute;rie sauterait sans transition du '
             'march&eacute; d&#39;aujourd&#39;hui au monde du TYNDP.</span>'
             '<span class="le">EPM needs <b>one value per year</b> from 2024 to 2053; the '
             'TYNDP publishes only three or four. Each trajectory therefore starts from the '
             '<b>observed 2024 price</b> &#8212; the anchor is common to all five &#8212; '
             'then runs straight to its first published point, then from published point to '
             'published point. Without that anchor the series would jump without transition '
             'from today&#39;s market into the TYNDP world.</span></div></div>')

    mile = [("VERY_LOW", "GA"), ("LOW", "DE"), ("CENTRAL", "NT")]
    rows = []
    for name, code in mile:
        yrs = TYNDP_SCENARIOS[code]
        label = "NT+" if code == "NT" else code
        rows.append(f'<tr><td><b>{name}</b></td><td>ENTSO-E {label}</td>'
                    f'<td style="text-align:right">{", ".join(str(y) for y in yrs)}</td>'
                    f'<td style="text-align:right">2024 &#8594; {yrs[0]}</td></tr>')
    p.append('<div class="gridtbl"><table class="cal"><thead><tr>'
             '<th><span class="lf">Trajectoire</span><span class="le">Trajectory</span></th>'
             '<th><span class="lf">Source</span><span class="le">Source</span></th>'
             '<th style="text-align:right"><span class="lf">Points publi&eacute;s</span>'
             '<span class="le">Published points</span></th>'
             '<th style="text-align:right"><span class="lf">Premier segment '
             'interpol&eacute;</span><span class="le">First interpolated segment</span>'
             '</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>')

    p.append('<div class="pn"><span class="lf">Les jalons <b>ne tombent pas aux m&#234;mes '
             'dates</b>, et c&#39;est de l&#224; que vient l&#39;inversion relev&eacute;e '
             'par le contr&#244;le d&#39;ordre. CENTRAL doit atteindre sa cible '
             '2030 en <b>six ans</b> : descente raide. LOW et VERY_LOW &#233;talent la leur '
             'sur <b>onze ans</b> : pente douce. Sur la fen&#234;tre 2025&#8211;2030 la '
             'droite molle de LOW passe donc l&eacute;g&#232;rement <b>au-dessus</b> de '
             'CENTRAL &#8212; au plus <b>+1,11 %</b> en Roumanie en 2030 &#8212; puis '
             'l&#39;ordre attendu se r&eacute;tablit d&#232;s 2031 et LOW reste franchement '
             'en dessous ensuite (&#8722;13,8 % en 2035). Cons&eacute;quence de '
             'l&#39;asym&eacute;trie des jalons, <b>pas</b> d&#39;une incoh&eacute;rence des '
             'donn&eacute;es ENTSO-E. HIGH et EU_HIGH n&#39;ont aucun jalon propre : elles '
             'sont d&eacute;riv&eacute;es ann&eacute;e par ann&eacute;e des trois '
             'pr&eacute;c&eacute;dentes.</span>'
             '<span class="le">The milestones <b>do not fall on the same dates</b>, and that '
             'is where the inversion flagged by the ordering gate comes from. CENTRAL '
             'must reach its 2030 target in <b>six years</b>: a steep descent. LOW and '
             'VERY_LOW spread theirs over <b>eleven years</b>: a gentle slope. Over the '
             '2025&#8211;2030 window the slack LOW line therefore sits slightly <b>above</b> '
             'CENTRAL &#8212; at most <b>+1.11 %</b> in Romania in 2030 &#8212; then the '
             'expected order is restored from 2031 and LOW stays firmly below afterwards '
             '(&#8722;13.8 % in 2035). A consequence of the milestone asymmetry, <b>not</b> '
             'of any inconsistency in the ENTSO-E data. HIGH and EU_HIGH have no milestones '
             'of their own: they are derived year by year from the three above.</span></div>')

    p.append('<div class="subh"><span class="lf">L&#39;&eacute;chelle &#224; 2040 &#8212; '
             'le publi&eacute; n&#39;ouvre que vers le bas</span><span class="le">The ladder at '
             '2040 &#8212; the published family only opens downwards</span></div>')
    p.append(chart_ladder(lvl))
    p.append('<div class="pn"><span class="lf">Axe <b>logarithmique</b>, uniquement sur ce '
             'graphique : puisque HIGH = CENTRAL&#178;/LOW, LOW et HIGH y tombent &#224; '
             '<b>distance &eacute;gale</b> de part et d&#39;autre de CENTRAL (trait vertical). '
             'Le graphique <b>montre</b> la r&#232;gle du miroir au lieu de l&#39;affirmer. '
             'Multiplicateur indiqu&eacute; par rapport &#224; CENTRAL.</span><span class="le">'
             '<b>Logarithmic</b> axis, on this chart only: since HIGH = CENTRAL&#178;/LOW, LOW '
             'and HIGH fall at <b>equal distance</b> either side of CENTRAL (vertical line). The '
             'chart <b>shows</b> the mirror rule instead of asserting it. Multiplier shown '
             'relative to CENTRAL.</span></div>')

    p.append(level_table(lvl))
    p.append(f'<div class="pn"><span class="lf">&#8364;/MWh <b>r&eacute;els 2024</b>. Ancrage '
             f'2024 observ&eacute; : RO 103,5 &#183; BG 101,8 &#183; GR 100,9. Tout est '
             f'assembl&eacute; en euros r&eacute;els via l&#39;IPCH zone euro, avec une '
             f'<b>unique</b> conversion finale en USD 2024 (&#215;{fx_fr}) &#8212; jamais de '
             f'change intercal&eacute; en cours de cha&#238;ne. Livrable : '
             f'<code>level_L.csv</code> (450 lignes).</span><span class="le">Real <b>2024 '
             f'EUR/MWh</b>. Observed 2024 anchor: RO 103.5 &#183; BG 101.8 &#183; GR 100.9. '
             f'Everything is assembled in real euros via euro-area HICP, with a <b>single</b> '
             f'final conversion to USD 2024 (&#215;{fx}) &#8212; never an FX step mid-chain. '
             f'Deliverable: <code>level_L.csv</code> (450 rows).</span></div>')

    # ---- Shape S -------------------------------------------------------- #
    p.append('<div class="subh"><span class="lf">Forme S &#8212; le profil horaire, '
             'normalis&eacute; &#224; moyenne 1</span><span class="le">Shape S &#8212; the '
             'hourly profile, normalised to mean 1</span></div>')
    p.append('<div class="body" style="margin-bottom:10px"><span class="lf">S est '
             '<b>sans dimension</b> : sa moyenne pond&eacute;r&eacute;e par <code>pHours</code> '
             'vaut <b>exactement 1</b>, donc S ne porte que la forme et le niveau vient '
             'enti&#232;rement de L. Les 28 jours-types sont ici agr&eacute;g&eacute;s en '
             '<b>4 journ&eacute;es saisonni&#232;res moyennes</b>, pond&eacute;r&eacute;es par '
             'le nombre de jours r&eacute;els que chacune repr&eacute;sente.</span>'
             '<span class="le">S is <b>dimensionless</b>: its <code>pHours</code>-weighted mean '
             'is <b>exactly 1</b>, so S carries shape only and the level comes entirely from L. '
             'The 28 day-types are aggregated here into <b>4 seasonal mean days</b>, weighted by '
             'the number of real days each represents.</span></div>')
    p.append(chart_shape(shape, hours))
    p.append('<div class="pn"><span class="lf">Trait horizontal &#224; 1,0 = moyenne annuelle. '
             'La <b>double bosse</b> matin/soir et le <b>creux solaire</b> de milieu de '
             'journ&eacute;e en Q2&#8211;Q3 sont exactement ce que le prix plat &#224; 70 $/MWh '
             'effa&#231;ait.</span><span class="le">Horizontal line at 1.0 = annual mean. The '
             '<b>morning/evening double peak</b> and the midday <b>solar trough</b> in Q2&#8211;Q3 '
             'are precisely what the flat 70 $/MWh erased.</span></div>')

    p.append('<div class="subh"><span class="lf">Courbe de dur&eacute;e &#8212; ce que la '
             'r&eacute;duction temporelle perd</span><span class="le">Duration curve &#8212; what '
             'the temporal reduction loses</span></div>')
    p.append(chart_duration(qcS))
    p.append('<div class="pn"><span class="lf">Les deux courbes sont <b>remises au m&#234;me '
             'niveau moyen</b> (celui de l&#39;observ&eacute;) pour isoler l&#39;erreur de '
             '<b>forme</b> du biais de niveau. Lecture : les <b>28 jours-types n&#39;atteignent '
             'jamais le d&eacute;cile bas</b> &#8212; 886 heures reconstruites sous 60 &#8364;/MWh '
             'contre 1 407 r&eacute;elles en Roumanie. C&#39;est un <b>&eacute;chec assum&eacute;'
             '</b> du contr&#244;le G2bis : les jours-types ont &eacute;t&eacute; choisis sur la '
             'charge et le VRE, pas sur le prix, et les rares heures &#224; prix quasi nul ne '
             'survivent pas au clustering. <b>Cons&eacute;quence &#224; porter au stade '
             'r&eacute;sultats : une sensibilit&eacute; de &#8722;11 &#8364;/MWh sur le d&eacute;cile '
             'bas</b>, qui sous-estime l&#39;attrait des imports aux heures creuses.</span>'
             '<span class="le">Both curves are <b>rescaled to the same mean</b> (the observed one) '
             'to isolate the <b>shape</b> error from the level bias. Reading: the <b>28 day-types '
             'never reach the bottom decile</b> &#8212; 886 reconstructed hours below 60 EUR/MWh '
             'against 1,407 real ones in Romania. This is an <b>accepted failure</b> of the G2bis '
             'gate: the day-types were selected on load and VRE, not on price, and the few '
             'near-zero-price hours do not survive the clustering. <b>To be carried to the results '
             'stage: a &#8722;11 EUR/MWh sensitivity on the bottom decile</b>, which understates how '
             'attractive imports are in off-peak hours.</span></div>')

    # ---- Controls ------------------------------------------------------- #
    p.append('<div class="subh"><span class="lf">Contr&#244;les &#8212; ce qui passe et ce qui '
             'ne passe pas</span><span class="le">Controls &#8212; what passes and what does not'
             '</span></div>')
    p.append(gate_matrix(qcL))
    # the table now carries each question and its tolerance, so the note keeps
    # only the three things a table cannot express: how the currency break was
    # found, and the diagnosis behind the two misses
    p.append('<div class="pn"><span class="lf"><b>Comment la contamination se voit</b> : '
             'march&eacute;s coupl&eacute;s, donc la m&eacute;diane annuelle du ratio brut '
             'RO/HU <b>est</b> le taux de change (4,424 contre 4,4454 BCE en 2015). '
             '<b>Gr&#232;ce</b> : 2023 y reste une ann&eacute;e tendue, gaz marginal quasi '
             'en permanence, que le TYNDP ne reproduit pas &#8212; r&eacute;serve '
             'report&eacute;e, pas corrig&eacute;e. <b>Roumanie</b> : asym&eacute;trie des '
             'jalons publi&eacute;s (voir plus haut). L&#39;&eacute;cart existe sur '
             '2025&#8211;2030 mais ne <b>d&eacute;passe la tol&eacute;rance qu&#39;en '
             '2030</b>, et s&#39;inverse d&#232;s 2031.</span>'
             '<span class="le"><b>How the contamination shows up</b>: the markets are coupled, '
             'so the yearly median of the raw RO/HU ratio <b>is</b> the exchange rate (4.424 '
             'against an ECB 4.4454 in 2015). <b>Greece</b>: 2023 remains a tight year there, '
             'gas almost permanently marginal, which the TYNDP does not reproduce &#8212; a '
             'reservation carried forward, not corrected. <b>Romania</b>: published-milestone '
             'asymmetry (see above). The gap runs over 2025&#8211;2030 but <b>only breaches '
             'the tolerance in 2030</b>, and reverses from 2031.</span></div>')

    p.append('<div class="subh"><span class="lf">Robustesse &#224; l&#39;ancrage</span>'
             '<span class="le">Robustness to the anchor</span></div>')
    p.append(chart_sensitivity(qcL))
    p.append('<div class="pn"><span class="lf">L&#39;ancrage est <b>une seule ann&eacute;e '
             'observ&eacute;e</b> (2024). En le rempla&#231;ant par la moyenne 2023&#8211;2024, '
             'l&#39;&eacute;cart moyen sur 2024&#8211;2053 reste <b>sous 1 %</b> pour la '
             'Roumanie et la Bulgarie et <b>sous 2,6 %</b> pour la Gr&#232;ce &#8212; o&#249; '
             '2023 fut nettement plus cher que 2024. Toutes les trajectoires convergent vers '
             'les points publi&eacute;s, donc l&#39;effet s&#39;&eacute;teint apr&#232;s 2035 : '
             'l&#39;ancrage pilote le d&eacute;but de p&eacute;riode, pas la cible.</span>'
             '<span class="le">The anchor is <b>a single observed year</b> (2024). Replacing it '
             'with the 2023&#8211;2024 mean keeps the mean deviation over 2024&#8211;2053 <b>under '
             '1 %</b> for Romania and Bulgaria and <b>under 2.6 %</b> for Greece &#8212; where 2023 '
             'was markedly dearer than 2024. All trajectories converge to the published points, so '
             'the effect dies out after 2035: the anchor drives the early years, not the target.'
             '</span></div>')

    p.append('<div class="keybox"><span class="lf"><b>&#201;tat de l&#39;hypoth&#232;se de '
             'niveau.</b> L est <b>clos</b> : cinq trajectoires &#215; trois zones &#215; '
             '2024&#8211;2053, en euros r&eacute;els 2024 et en USD 2024, dont <b>trois '
             'tra&#231;ables ligne &#224; ligne</b> jusqu&#39;aux classeurs ENTSO-E. Deux '
             'r&eacute;serves report&eacute;es et non masqu&eacute;es : le r&eacute;tro-test '
             'grec &#224; &#8722;15,7 % (le TYNDP ne reproduit pas la tension gazi&#232;re '
             'hell&eacute;nique de 2023) et le d&eacute;cile bas manquant dans S '
             '(&#8722;11 &#8364;/MWh). Le net-back est d&#233;sormais construit et '
             'promu dans <code>data_blacksea</code>.</span>'
             '<span class="le"><b>Status of the level assumption.</b> L is <b>closed</b>: five '
             'trajectories &#215; three zones &#215; 2024&#8211;2053, in real 2024 EUR and in USD '
             '2024, of which <b>three are traceable line by line</b> back to the ENTSO-E workbooks. '
             'Two reservations carried forward and not hidden: the Greek back-cast at &#8722;15.7 % '
             '(the TYNDP does not reproduce Greece&#39;s 2023 gas tightness) and the missing bottom '
             'decile in S (&#8722;11 EUR/MWh). The net-back is now built and promoted into '
             '<code>data_blacksea</code>.</span></div>')

    p.append('<div class="keybox"><span class="lf"><b>Ce qui a &#233;t&#233; livr&#233;.</b> '
             'Net-back par lien (&#955; = 3&#160;%, W = 2&#160;&#8364;/MWh, C = CBAM) &#183; '
             'assemblage P = L &#215; S &#215; (1&#8722;&#955;) &#8722; W &#8722; C &#183; '
             '<b>15 CSV</b> dans <code>data_blacksea/trade/</code> couvrant 10 sc&#233;narios '
             '(5 niveaux &#215; REF/CBAM ; le fichier d&#39;achat est partag&#233; par chaque '
             'paire, v&#233;rifi&#233; par hachage). <code>eu_central</code> est c&#226;bl&#233; '
             'comme r&#233;f&#233;rence. La s&#233;paration achat/vente a exig&#233; un '
             'param&#232;tre GAMS, <code>pTradePriceExport</code> &#8212; sans lui les '
             'sc&#233;narios CBAM auraient rendu les r&#233;sultats de leurs jumeaux.</span>'
             '<span class="le"><b>What was delivered.</b> Per-link net-back (&#955; = 3%, '
             'W = 2&#160;&#8364;/MWh, C = CBAM) &#183; assembly P = L &#215; S &#215; '
             '(1&#8722;&#955;) &#8722; W &#8722; C &#183; <b>15 CSVs</b> in '
             '<code>data_blacksea/trade/</code> covering 10 scenarios (5 levels &#215; '
             'REF/CBAM; the buy file is shared within each pair, verified by hash). '
             '<code>eu_central</code> is wired as the reference. Splitting buy from sell '
             'required a GAMS parameter, <code>pTradePriceExport</code> &#8212; without it '
             'the CBAM scenarios would have returned their twins&#39; results.</span></div>')

    return BEGIN + "\n" + "\n".join(p) + "\n" + END


def splice(fragment: str):
    """Replace the marked block, or the legacy block on the first run."""
    html = REPORT.read_text(encoding="utf-8")
    if BEGIN in html and END in html:
        a = html.index(BEGIN)
        b = html.index(END) + len(END)
    else:
        # first run: the section still holds the hand-written prose block, which
        # runs from the "Niveau L" subheading to just before the section-4 heading
        a = html.index('<div class="subh"><span class="lf">Niveau L')
        b = html.index('<h3 class="roman"><span class="lf">4 &#183; Conclusion')
    REPORT.write_text(html[:a] + fragment + "\n" + html[b:], encoding="utf-8")
    print(f"spliced into {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splice", action="store_true",
                    help="inject the fragment into calibration_review.html")
    args = ap.parse_args()

    frag = build()
    FRAGMENT.write_text(frag, encoding="utf-8")
    print(f"wrote {FRAGMENT} ({len(frag):,} chars)")
    if args.splice:
        splice(frag)
