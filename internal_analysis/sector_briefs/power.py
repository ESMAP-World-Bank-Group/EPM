"""Energy > Power pane of the Black Sea sector briefs.

Figures come from the model inputs in epm/input/data_blacksea, so the brief and
the model agree. Physical cross-border ratings come from
pre-analysis/data/reference_lines.csv, the single source for interconnections.
The ground map draws OpenStreetMap lines of 220 kV and above and the Global
Energy Monitor plant tracker, extracted to data/gem_plants.json by
extract_gem.py.
"""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path

from mapkit import (DATA_BLACKSEA, DIGITAL_DATA, FOCUS, PALETTE, ROOT, CountryIndex,
                    Map, lines_of, load_json)

HERE = Path(__file__).resolve().parent
YEAR = "2025"
REFERENCE_LINES = ROOT / "pre-analysis" / "data" / "reference_lines.csv"

# Wider than the shared window: Bulgaria, Greece and Romania close the western
# end of every corridor drawn here.
BBOX = (22.0, 35.4, 52.6, 47.6)
W, H = 1000, 540

STUDY = ["Turkiye", "Georgia", "Armenia", "Azerbaijan"]

# Fuel groups, ordered for the stacked bars, with the colours of the CASA brief.
GROUPS = [
    ("Hydro", "#5389AE"), ("Gas", "#A98A62"), ("Coal", "#767B80"),
    ("Solar", "#DCB755"), ("Wind", "#84B3BB"), ("Nuclear", "#A896C2"),
    ("Oil", "#8B7E72"), ("Other", "#7FA36B"),
]
GROUP_COLOUR = dict(GROUPS)
FUEL_GROUP = {
    "Water": "Hydro", "Gas": "Gas", "DomesticCoal": "Coal", "ImportedCoal": "Coal",
    "Coal": "Coal", "Lignite": "Coal", "Solar": "Solar", "Wind": "Wind",
    "Uranium": "Nuclear", "HFO": "Oil", "LFO": "Oil", "Diesel": "Oil", "Oil": "Oil",
}

PLANT_COLOUR = {
    "hydropower": "#5389AE", "oil/gas": "#A98A62", "coal": "#767B80",
    "utility-scale solar": "#DCB755", "wind": "#84B3BB", "nuclear": "#A896C2",
    "bioenergy": "#7FA36B", "geothermal": "#C47F5A",
}
PLANT_LABEL = {
    "hydropower": "Hydro", "oil/gas": "Gas and oil", "coal": "Coal",
    "utility-scale solar": "Solar", "wind": "Wind", "nuclear": "Nuclear",
    "bioenergy": "Bioenergy", "geothermal": "Geothermal",
}
GRID = "#c99a2e"

# Synchronous areas. Armenia runs in parallel with Iran, which is why the
# Georgia to Armenia project needs a back to back converter at Ayrum.
SYNC = {
    "Continental Europe": ("#CFDDE7", {"Turkey", "Bulgaria", "Greece", "Romania",
                                       "Moldova", "Ukraine", "Republic of Serbia",
                                       "Hungary", "Macedonia", "Albania"}),
    "IPS/UPS (Russia)": ("#E6DCC2", {"Russia", "Georgia", "Azerbaijan", "Kazakhstan",
                                     "Belarus"}),
    "Iran": ("#E9D6CD", {"Iran", "Armenia", "Turkmenistan"}),
}
OTHER_FILL = "#ECEAE4"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def read_csv(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def zone_country():
    return {r["z"]: r["c"] for r in read_csv(DATA_BLACKSEA / "zcmap.csv")
            if r["c"] in STUDY}


def num(text, default=None):
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def model_capacity(zc):
    """Existing capacity in service in the base year, by country and fuel group."""
    out = {c: defaultdict(float) for c in STUDY}
    y = int(YEAR)
    for name in ("pGenDataInput.csv", "pStorageDataInput.csv"):
        for r in read_csv(DATA_BLACKSEA / "supply" / name):
            if r["z"] not in zc or r["Status"] != "1":
                continue
            start, end = num(r["StYr"], 0), num(r["RetrYr"], 9999)
            if not (start <= y < end):
                continue
            group = "Hydro" if r["tech"] == "Storage" and r["f"] == "Water" else (
                FUEL_GROUP.get(r["f"], "Other"))
            out[zc[r["z"]]][group] += num(r["Capacity"], 0.0)
    return out


def model_demand(zc):
    """Annual energy and coincident peak by country.

    The forecast gives a peak per zone. Turkiye has nine zones, and their peaks
    do not fall in the same hour, so the national peak is the highest hourly sum
    of zone peak times zone profile over the representative days.
    """
    peak, energy = defaultdict(float), defaultdict(float)
    zpeak = {}
    for r in read_csv(DATA_BLACKSEA / "load" / "pDemandForecast.csv"):
        if r["z"] not in zc:
            continue
        if r["type"] == "Peak":
            zpeak[r["z"]] = float(r[YEAR])
        else:
            energy[zc[r["z"]]] += float(r[YEAR])
    hours = defaultdict(dict)
    for r in read_csv(DATA_BLACKSEA / "load" / "pDemandProfile.csv"):
        for h in range(1, 25):
            hours[(r["season"], r["daytype"], h)][r["zone"]] = float(r[f"t{h:02d}"])
    for c in STUDY:
        zones = [z for z in zc if zc[z] == c]
        peak[c] = max(sum(zpeak[z] * v.get(z, 0.0) for z in zones) for v in hours.values())
    return peak, energy


ISO = {"TUR": "Turkiye", "GEO": "Georgia", "ARM": "Armenia", "AZE": "Azerbaijan",
       "BGR": "Bulgaria", "GRC": "Greece", "ROU": "Romania", "MDA": "Moldova",
       "RUS": "Russia", "IRN": "Iran", "IRQ": "Iraq", "SYR": "Syria", "UKR": "Ukraine",
       "KAZ": "Kazakhstan", "SRB": "Serbia", "HUN": "Hungary"}


def reference_lines():
    with open(REFERENCE_LINES, encoding="utf-8-sig") as fh:
        rows = [line for line in fh if not line.startswith("#")]
    return list(csv.DictReader(rows))


# Pairs whose lines exist on the ground but have carried nothing since the
# early 1990s. Every other zero rating in the inventory is a line kept out on
# purpose (occupied territories, rehabilitation) and is not drawn.
IDLE = {("Armenia", "Turkiye"), ("Armenia", "Azerbaijan")}


def physical_links(lines):
    """Existing cross-border ratings by country pair, Nakhchivan kept apart.

    The Igdir line to Nakhchivan has no physical rating in the inventory, only
    its 50 MW operating NTC, which stands in for it.
    """
    out = defaultdict(float)
    for r in lines:
        if r["status"] != "existing":
            continue
        a = "Nakhchivan" if r["from_zone"] == "Nakhchivan" else ISO.get(r["from_country"])
        b = ISO.get(r["to_country"])
        if not a or not b or a == b:
            continue
        mw = num(r["mw_fwd"], 0.0)
        if not mw and a == "Nakhchivan":
            mw = num(r["ntc_op"], 0.0)
        out[tuple(sorted((a, b)))] += mw
    return {k: v for k, v in out.items() if v > 0 or k in IDLE}


# ---------------------------------------------------------------------------
# small renderers
# ---------------------------------------------------------------------------
def esc(text) -> str:
    return html.escape(str(text))


def fmt_gw(mw: float) -> str:
    return f"{mw / 1000:.1f}"


def mixbar(mix, width=150):
    total = sum(mix.values()) or 1.0
    segs = "".join(
        f'<i style="width:{100 * mix.get(g, 0) / total:.2f}%;background:{col}" '
        f'title="{g} {mix.get(g, 0):,.0f} MW"></i>'
        for g, col in GROUPS if mix.get(g, 0) > 0
    )
    return f'<span class="mixbar" style="width:{width}px">{segs}</span>'


def donut(mix, size=210, hole=0.6, centre=None, sub=None) -> str:
    total = sum(mix.values()) or 1.0
    r = size / 2
    rin = r * hole
    cx = cy = r
    parts = []
    angle = -math.pi / 2
    for g, col in GROUPS:
        v = mix.get(g, 0.0)
        if v <= 0:
            continue
        sweep = 2 * math.pi * v / total
        a0, a1 = angle, angle + sweep
        large = 1 if sweep > math.pi else 0
        p = [
            (cx + r * math.cos(a0), cy + r * math.sin(a0)),
            (cx + r * math.cos(a1), cy + r * math.sin(a1)),
            (cx + rin * math.cos(a1), cy + rin * math.sin(a1)),
            (cx + rin * math.cos(a0), cy + rin * math.sin(a0)),
        ]
        parts.append(
            f'<path d="M{p[0][0]:.1f},{p[0][1]:.1f} A{r:.1f},{r:.1f} 0 {large} 1 '
            f'{p[1][0]:.1f},{p[1][1]:.1f} L{p[2][0]:.1f},{p[2][1]:.1f} '
            f'A{rin:.1f},{rin:.1f} 0 {large} 0 {p[3][0]:.1f},{p[3][1]:.1f}Z" '
            f'fill="{col}" stroke="#fff" stroke-width="1"><title>{g}: '
            f'{v:,.0f} MW, {100 * v / total:.0f}%</title></path>'
        )
        angle = a1
    if centre:
        parts.append(f'<text x="{cx}" y="{cy + 2}" text-anchor="middle" font-size="22" '
                     f'font-weight="700" fill="{PALETTE["ink"]}">{esc(centre)}</text>')
    if sub:
        parts.append(f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" font-size="11" '
                     f'fill="#6f6a61">{esc(sub)}</text>')
    return (f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
            f'font-family="Segoe UI, Arial, sans-serif">{"".join(parts)}</svg>')


def html_legend(rows) -> str:
    """rows: list of (heading, [(swatch_html, text), ...])."""
    out = ['<div class="imap-legend">']
    for head, items in rows:
        out.append(f'<div class="lgrow"><span class="lgh">{esc(head)}</span>')
        out += [f'<span class="lg">{sw}{esc(t)}</span>' for sw, t in items]
        out.append("</div>")
    out.append("</div>")
    return "".join(out)


def sw_dot(col, ring=False):
    if ring:
        return f'<i class="d" style="background:#fff;border:2px solid {col}"></i>'
    return f'<i class="d" style="background:{col}"></i>'


def sw_line(col, h=3, dash=None):
    if dash:
        return (f'<svg class="lsw" width="22" height="8"><line x1="1" y1="4" x2="21" y2="4" '
                f'stroke="{col}" stroke-width="2.6" stroke-dasharray="{dash}"/></svg>')
    return f'<i class="ln" style="background:{col};height:{h}px"></i>'


def sw_box(col):
    return f'<i class="d sq" style="background:{col};border:1px solid #b0bec5"></i>'


def imap(svg: str, caption: str, legend: str, plants: bool = False) -> str:
    slider = ('<span class="psz-w"><span class="psz-l">plants</span><input class="psz" '
              'type="range" min="0.4" max="2.4" step="0.1" value="1"></span>') if plants else ""
    return (
        '<div class="imap-box"><div class="imap-main"><div class="imap-btns">'
        '<button class="zb" data-z="in">+</button><button class="zb" data-z="out">'
        '&minus;</button><button class="zb" data-z="reset">reset</button>'
        f'{slider}</div><div class="imap-wrap">{svg}</div>'
        f'<div class="mapcap">{caption}</div>'
        '<div class="mapcap disc">Country boundaries, colours, denominations and other '
        'information shown on this map do not imply any judgment on the legal status of '
        'any territory, or any endorsement or acceptance of such boundaries.</div>'
        f'</div><aside class="imap-side">{legend}</aside></div>'
    )


def svg_of(m: Map) -> str:
    return m.render().replace("<svg ", '<svg class="imap" ', 1)


# ---------------------------------------------------------------------------
# 1. primer
# ---------------------------------------------------------------------------
def primer_svg() -> str:
    ink, acc, mut = PALETTE["ink"], PALETTE["accent"], "#6f6a61"
    steps = [
        ("Generation", "Plants, 10 to 25 kV"),
        ("Step up", "Transformer"),
        ("Transmission", "220 to 500 kV"),
        ("Substation", "Step down"),
        ("Distribution", "6 to 110 kV"),
        ("Consumers", "Homes, industry"),
    ]
    bw, gap, y = 118, 22, 18
    out = ['<svg viewBox="0 0 860 300" width="100%" font-family="Segoe UI, Arial, sans-serif">',
           '<defs><marker id="pa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
           f'markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="{acc}"/></marker></defs>']
    for i, (head, sub) in enumerate(steps):
        x = 12 + i * (bw + gap)
        fill = "#eaf1f8" if head == "Transmission" else "#ffffff"
        out.append(f'<rect x="{x}" y="{y}" width="{bw}" height="56" rx="7" fill="{fill}" '
                   f'stroke="{ink}" stroke-width="1.2"/>')
        out.append(f'<text x="{x + bw / 2}" y="{y + 24}" text-anchor="middle" font-size="13" '
                   f'font-weight="700" fill="{ink}">{head}</text>')
        out.append(f'<text x="{x + bw / 2}" y="{y + 42}" text-anchor="middle" font-size="10.5" '
                   f'fill="{mut}">{sub}</text>')
        if i < len(steps) - 1:
            out.append(f'<line x1="{x + bw + 2}" y1="{y + 28}" x2="{x + bw + gap - 2}" '
                       f'y2="{y + 28}" stroke="{acc}" stroke-width="2" marker-end="url(#pa)"/>')
    out.append(f'<text x="{12 + 2 * (bw + gap) + bw / 2}" y="{y + 74}" text-anchor="middle" '
               f'font-size="10.5" fill="{acc}" font-weight="600">Interconnectors sit here</text>')

    def panel(x, title, lines, b2b):
        py = 120
        out.append(f'<rect x="{x}" y="{py}" width="410" height="168" rx="8" fill="#fbfaf6" '
                   f'stroke="#e6e6ea"/>')
        out.append(f'<text x="{x + 16}" y="{py + 24}" font-size="13" font-weight="700" '
                   f'fill="{ink}">{title}</text>')
        for k, (cx, lab) in enumerate([(x + 70, "System A"), (x + 340, "System B")]):
            out.append(f'<circle cx="{cx}" cy="{py + 70}" r="30" fill="#fff" stroke="{ink}" '
                       f'stroke-width="1.2"/>')
            wave_y = py + 70
            amp = 8
            period = 22 if (b2b and k == 1) else 26
            d = "M" + " L".join(
                f"{cx - 20 + t:.1f},{wave_y - amp * math.sin(2 * math.pi * t / period):.1f}"
                for t in range(0, 41, 2))
            out.append(f'<path d="{d}" fill="none" stroke="{acc}" stroke-width="1.6"/>')
            out.append(f'<text x="{cx}" y="{py + 116}" text-anchor="middle" font-size="10.5" '
                       f'fill="{mut}">{lab}</text>')
        if b2b:
            out.append(f'<line x1="{x + 100}" y1="{py + 70}" x2="{x + 180}" y2="{py + 70}" '
                       f'stroke="{ink}" stroke-width="2"/>')
            out.append(f'<line x1="{x + 230}" y1="{py + 70}" x2="{x + 310}" y2="{py + 70}" '
                       f'stroke="{ink}" stroke-width="2"/>')
            out.append(f'<rect x="{x + 180}" y="{py + 52}" width="50" height="36" rx="4" '
                       f'fill="{ink}"/>')
            out.append(f'<text x="{x + 205}" y="{py + 75}" text-anchor="middle" font-size="10" '
                       f'font-weight="700" fill="#fff">AC/DC/AC</text>')
        else:
            out.append(f'<line x1="{x + 100}" y1="{py + 70}" x2="{x + 310}" y2="{py + 70}" '
                       f'stroke="{ink}" stroke-width="2"/>')
            out.append(f'<text x="{x + 205}" y="{py + 62}" text-anchor="middle" font-size="10" '
                       f'fill="{mut}">AC line</text>')
        for j, t in enumerate(lines):
            out.append(f'<text x="{x + 16}" y="{py + 138 + j * 15}" font-size="11" '
                       f'fill="#2b2926">{t}</text>')

    panel(12, "Synchronous AC link",
          ["One frequency across both systems. Flows split by physics.",
           "Cheap and simple, but a disturbance travels across."], b2b=False)
    panel(438, "Back to back or HVDC link",
          ["Each side keeps its own frequency. The operator sets the flow.",
           "The only way to join two systems that are not synchronised."], b2b=True)
    out.append("</svg>")
    return "".join(out)


def primer() -> str:
    bullets = [
        ("Balance every second.",
         "Supply must equal demand at every instant. Frequency (50 Hz here) is the "
         "signal: it drops when demand exceeds supply."),
        ("Capacity is not energy.",
         "Capacity is the MW a plant can deliver at once. Energy is what it produces "
         "over a year, in MWh. A peak is met with capacity, a bill is paid in energy."),
        ("Trade is capped by transfer capacity.",
         "The MW that can cross a border at once, set by the lines and by what each "
         "grid can absorb behind them."),
        ("Here, synchronism decides who can trade.",
         "Turkiye runs with ENTSO-E, Georgia and Azerbaijan with Russia, "
         "Armenia with Iran. Every link across these lines needs a converter, as at "
         "Akhaltsikhe today and Ayrum tomorrow."),
    ]
    items = "".join(f"<li><b>{esc(h)}</b> {esc(t)}</li>" for h, t in bullets)
    return (f'<div class="primer-fig">{primer_svg()}</div>'
            f'<ul class="primer-list">{items}</ul>')


# ---------------------------------------------------------------------------
# 2. key figures
# ---------------------------------------------------------------------------
LINK_TYPE = {
    ("Georgia", "Turkiye"): "b2b", ("Iran", "Turkiye"): "b2b",
    ("Iraq", "Turkiye"): "radial", ("Syria", "Turkiye"): "radial",
    ("Nakhchivan", "Turkiye"): "radial", ("Armenia", "Georgia"): "radial",
    ("Azerbaijan", "Iran"): "radial", ("Iran", "Nakhchivan"): "radial",
}

SYNC_OF = {
    "Turkiye": "ENTSO-E",
    "Georgia": "IPS/UPS, with Russia and Azerbaijan",
    "Armenia": "In parallel with Iran",
    "Azerbaijan": "IPS/UPS. Nakhchivan apart, fed from Iran and Turkiye",
}


def key_figures(cap, peak, energy, links) -> str:
    total = sum(sum(m.values()) for m in cap.values())
    mix = defaultdict(float)
    for m in cap.values():
        for g, v in m.items():
            mix[g] += v
    tr_share = sum(cap["Turkiye"].values()) / total
    inner = {k: v for k, v in links.items()
             if k[0] in STUDY + ["Nakhchivan"] and k[1] in STUDY + ["Nakhchivan"]}
    inner_mw = sum(inner.values())
    via_ge = sum(v for k, v in inner.items() if "Georgia" in k)
    tiles = [
        (f"{total / 1000:.0f} GW", f"Installed capacity, {YEAR}. Turkiye holds "
                                   f"{100 * tr_share:.0f}%."),
        (f"{sum(energy.values()) / 1000:.0f} TWh", f"Annual demand, {YEAR}"),
        (f"{sum(peak.values()) / 1000:.0f} GW", "Peak load, sum of the four national peaks"),
        (f"{inner_mw / 1000:.1f} GW", "Transfer capacity between the four countries. "
                                      f"{100 * via_ge / inner_mw:.0f}% of it touches "
                                      "Georgia."),
    ]
    stats = "".join(f'<div class="stat"><div class="v">{esc(v)}</div>'
                    f'<div class="l">{esc(lab)}</div></div>' for v, lab in tiles)
    legend = "".join(
        f'<span class="lg">{sw_dot(col)}{g} {100 * mix[g] / total:.0f}%</span>'
        for g, col in GROUPS if mix.get(g, 0) > 0
    )
    head = (
        '<div class="headrow">'
        f'<div class="hr-kpi"><div class="stats">{stats}</div></div>'
        '<div class="hr-mix"><div class="mixcard">'
        '<div class="mixhead">Capacity mix, four countries</div>'
        f'{donut(mix, centre=f"{total / 1000:.0f} GW", sub="installed")}'
        f'<div class="mixleg">{legend}</div></div></div></div>'
    )

    def nb(country):
        rows = []
        for (a, b), mw in sorted(links.items(), key=lambda kv: -kv[1]):
            names = {a, b}
            me = country if country in names else (
                "Nakhchivan" if country == "Azerbaijan" and "Nakhchivan" in names else None)
            if not me:
                continue
            other = (names - {me}).pop() if len(names) > 1 else me
            tag = {"b2b": " (B2B)", "radial": " (radial)"}.get(
                LINK_TYPE.get(tuple(sorted((me, other))), ""), "")
            label = other if me == country else f"{other} (Nakhchivan)"
            rows.append(f"{label} {mw:,.0f}{tag}" if mw else f"{label}: idle")
        return "; ".join(rows)

    body = "".join(
        f"<tr><td><b>{c}</b></td><td class='n'>{fmt_gw(sum(cap[c].values()))}</td>"
        f"<td>{mixbar(cap[c])}</td><td class='n'>{energy[c] / 1000:.1f}</td>"
        f"<td class='n'>{fmt_gw(peak[c])}</td><td>{esc(SYNC_OF[c])}</td>"
        f"<td class='small'>{esc(nb(c))}</td></tr>"
        for c in STUDY
    )
    table = (
        '<table class="kt"><thead><tr><th>Country</th><th class="n">GW</th><th>Mix</th>'
        '<th class="n">TWh</th><th class="n">Peak GW</th><th>Synchronous system</th>'
        '<th>Cross-border links, MW</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )
    note = (
        f'<p class="note">Model inputs, {YEAR}, existing plants. Turkiye peak: coincident '
        "over its nine zones. Links: physical ratings, not NTC. B2B: back to back "
        "converter. Radial: fed in island mode. Idle: out of service since 1993.</p>"
    )
    return head + table + note


# ---------------------------------------------------------------------------
# 3. background
# ---------------------------------------------------------------------------
TIMELINE = [
    ("1993", "Lines from Turkiye and Azerbaijan to Armenia fall idle. Armenia turns to "
             "Iran and Georgia.", "Study line inventory"),
    ("1995", "Metsamor unit 2 restarts after the 1988 earthquake shutdown. Armenia's "
             "only nuclear unit, still about a third of its output.", "IAEA PRIS"),
    ("2013", "Black Sea Transmission Network: 700 MW back to back at Akhaltsikhe, the "
             "first bridge between the Caucasus and Continental Europe.", "GSE, KfW"),
    ("2015", "Turkiye becomes a permanent synchronous part of Continental Europe after "
             "trial operation from 2010.", "ENTSO-E"),
    ("Mar 2022", "Ukraine and Moldova synchronise with Continental Europe. The European "
                 "grid now wraps the north and west of the Black Sea.", "ENTSO-E"),
    ("Dec 2022", "Azerbaijan, Georgia, Romania and Hungary sign the Green Energy Corridor "
                 "agreement in Bucharest, launching the BSSC.", "Government of Romania"),
    ("Nov 2024", "Azerbaijan, Kazakhstan and Uzbekistan agree at COP29 to extend the "
                 "corridor across the Caspian.", "COP29 presidency"),
    ("Aug 2025", "Washington summit: Armenia and Azerbaijan endorse TRIPP, a route "
                 "through southern Armenia that includes power lines.", "White House"),
    ("Dec 2025", "BSSC enters the EU list of Projects of Mutual Interest.",
     "European Commission"),
    ("Feb 2026", "Transelectrica and GSE sign a memorandum on joint BSSC studies, "
                 "surveys and financing.", "Transelectrica"),
]


def background() -> str:
    rows = "".join(
        f'<li><span class="tl-y">{esc(y)}</span><span class="tl-t">{esc(t)}</span>'
        f'<span class="tl-s">{esc(s)}</span></li>'
        for y, t, s in TIMELINE
    )
    return f'<ol class="tl">{rows}</ol>'


# ---------------------------------------------------------------------------
# shared map pieces
# ---------------------------------------------------------------------------
COUNTRY_LABELS = [
    (34.0, 39.2, "TURKIYE"), (43.35, 42.25, "GEORGIA"), (44.75, 40.45, "ARMENIA"),
    (47.9, 40.65, "AZERBAIJAN"), (45.2, 39.05, "Nakhchivan"), (25.4, 42.75, "BULGARIA"),
    (25.2, 45.6, "ROMANIA"), (23.0, 39.6, "GREECE"), (41.5, 46.6, "RUSSIA"),
    (49.0, 37.0, "IRAN"), (43.2, 36.1, "IRAQ"), (38.3, 35.7, "SYRIA"),
    (31.5, 47.2, "UKRAINE"), (28.6, 47.35, "MOLDOVA"), (51.6, 46.9, "KAZAKHSTAN"),
]


def label_countries(m: Map, size=10):
    for lon, lat, name in COUNTRY_LABELS:
        focus = name.title() in {"Turkiye", "Georgia", "Armenia", "Azerbaijan"}
        m.label(lon, lat, name, size=size + (1 if focus else 0),
                colour=PALETTE["ink"] if focus else "#8a857c",
                weight="700" if focus else "500", halo=True)
    for lon, lat, name in [(34.0, 43.2, "Black Sea"), (50.7, 42.0, "Caspian Sea"),
                           (31.5, 35.8, "Mediterranean")]:
        m.label(lon, lat, name, size=11, colour="#7aa6bd", weight="400")


def voltage_of(props):
    text = str(props.get("voltage") or "")
    try:
        return int(text.split(";")[0]) // 1000
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# 4. the region on the ground
# ---------------------------------------------------------------------------
def ground_map(index: CountryIndex) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    hv = load_json(DIGITAL_DATA / "osm_hv_lines.geojson")
    for feat in sorted(hv["features"], key=lambda f: voltage_of(f["properties"])):
        kv = voltage_of(feat["properties"])
        width = 2.2 if kv >= 500 else 1.4 if kv >= 380 else 1.0 if kv >= 300 else 0.6
        for line in lines_of(feat["geometry"]):
            m.add(f'<path d="{m.path(line, min_px=1.2)}" fill="none" stroke="{GRID}" '
                  f'stroke-width="{width}" stroke-linecap="round" stroke-opacity="0.85"/>')
    plants = load_json(HERE / "data" / "gem_plants.json")
    keep = [p for p in plants if p.get("operating", 0) + p.get("construction", 0) >= 20]
    keep.sort(key=lambda p: -(p.get("operating", 0) + p.get("construction", 0)))
    for p in keep:
        mw_op, mw_c = p.get("operating", 0), p.get("construction", 0)
        col = PLANT_COLOUR.get(p["type"], "#999")
        x, y = m.xy(p["lon"], p["lat"])
        r = max(1.3, 0.13 * math.sqrt(mw_op + mw_c))
        tip = (f'{p["name"]}, {PLANT_LABEL.get(p["type"], p["type"])}, '
               + (f"{mw_op:,.0f} MW operating" if mw_op else "")
               + (", " if mw_op and mw_c else "")
               + (f"{mw_c:,.0f} MW under construction" if mw_c else ""))
        if mw_op:
            m.add(f'<circle class="pl" data-r="{r:.1f}" cx="{x:.1f}" cy="{y:.1f}" '
                  f'r="{r:.1f}" fill="{col}" fill-opacity="0.82" stroke="#fff" '
                  f'stroke-width="0.5"><title>{esc(tip)}</title></circle>')
        else:
            m.add(f'<circle class="pl" data-r="{r:.1f}" cx="{x:.1f}" cy="{y:.1f}" '
                  f'r="{r:.1f}" fill="#fff" fill-opacity="0.6" stroke="{col}" '
                  f'stroke-width="1.6"><title>{esc(tip)}</title></circle>')
    label_countries(m)
    # Names on the plants that carry each study country, plus Akkuyu.
    named = {"Enguri hydroelectric plant": "Enguri", "Armenian nuclear power plant": "Metsamor",
             "Azerbaijan thermal power plant": "Mingachevir",
             "Akkuyu nuclear power plant": "Akkuyu (building)",
             "Atatürk hydroelectric plant": "Ataturk dam"}
    # Offsets keep the plant names clear of the country names.
    nudge = {"Enguri": (-0.15, 0.05, "end"), "Mingachevir": (-0.2, -0.38, "end")}
    for p in keep:
        if p["name"] in named:
            text = named[p["name"]]
            dx, dy, anc = nudge.get(text, (0.12, -0.18, "start"))
            m.label(p["lon"] + dx, p["lat"] + dy, text, size=9.5,
                    colour="#403b35", anchor=anc, weight="600", halo=True)
    m.add("</g>")
    svg = svg_of(m)
    legend = html_legend([
        ("Grid, OpenStreetMap", [(sw_line(GRID, 4), "500 kV and above"),
                                 (sw_line(GRID, 2.6), "380 to 400 kV"),
                                 (sw_line(GRID, 2), "330 kV"),
                                 (sw_line(GRID, 1.2), "220 kV")]),
        ("Plants, 20 MW and above", [(sw_dot(c), PLANT_LABEL[t])
                                     for t, c in PLANT_COLOUR.items()]
         + [(sw_dot("#767B80", ring=True), "under construction")]),
    ])
    cap = ("Grid: OpenStreetMap, 220 kV and above. Plants: Global Energy Monitor, "
           "August 2026. Hover a plant for its name.")
    return imap(svg, cap, legend, plants=True)


# ---------------------------------------------------------------------------
# 5. capacity, synchronous zones and interconnections
# ---------------------------------------------------------------------------
ANCHOR = {
    "Turkiye": (35.2, 39.0), "Georgia": (43.6, 42.05), "Armenia": (44.9, 40.35),
    "Azerbaijan": (48.0, 40.35), "Nakhchivan": (45.35, 39.25), "Bulgaria": (25.3, 42.6),
    "Greece": (23.2, 40.3), "Romania": (25.5, 45.4), "Moldova": (28.6, 47.0),
    "Russia": (44.0, 45.6), "Iran": (47.4, 37.2), "Iraq": (43.4, 36.2),
    "Syria": (38.6, 35.9), "Ukraine": (31.5, 47.0),
}
# Turkiye is wide: its links leave from the side that faces the neighbour.
TR_SIDE = {"Bulgaria": (27.3, 41.5), "Greece": (26.9, 40.9), "Syria": (38.4, 37.4),
           "Iraq": (41.6, 37.7), "Iran": (42.4, 39.2), "Georgia": (41.6, 40.9),
           "Nakhchivan": (43.4, 39.6), "Armenia": (42.6, 40.3)}


def anchor(country, other):
    if country == "Turkiye" and other in TR_SIDE:
        return TR_SIDE[other]
    return ANCHOR[country]


RING_AT = {"Turkiye": (34.2, 38.9), "Georgia": (43.0, 42.25), "Armenia": (44.95, 40.25),
           "Azerbaijan": (48.3, 40.25)}


def ring(m: Map, lon, lat, mix, r):
    total = sum(mix.values())
    cx, cy = m.xy(lon, lat)
    rin = r * 0.55
    angle = -math.pi / 2
    for g, col in GROUPS:
        v = mix.get(g, 0.0)
        if v <= 0:
            continue
        sweep = 2 * math.pi * v / total
        a0, a1 = angle, angle + sweep
        large = 1 if sweep > math.pi else 0
        p = [(cx + r * math.cos(a0), cy + r * math.sin(a0)),
             (cx + r * math.cos(a1), cy + r * math.sin(a1)),
             (cx + rin * math.cos(a1), cy + rin * math.sin(a1)),
             (cx + rin * math.cos(a0), cy + rin * math.sin(a0))]
        m.add(f'<path d="M{p[0][0]:.1f},{p[0][1]:.1f} A{r:.1f},{r:.1f} 0 {large} 1 '
              f'{p[1][0]:.1f},{p[1][1]:.1f} L{p[2][0]:.1f},{p[2][1]:.1f} '
              f'A{rin:.1f},{rin:.1f} 0 {large} 0 {p[3][0]:.1f},{p[3][1]:.1f}Z" '
              f'fill="{col}" stroke="#fff" stroke-width="0.8"><title>{g}: {v:,.0f} MW'
              f'</title></path>')
        angle = a1
    m.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rin - 0.5:.1f}" fill="#fff"/>')
    m.add(f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-size="'
          f'{max(9, min(15, r * 0.42)):.0f}" font-weight="700" fill="{PALETTE["ink"]}" '
          f'font-family="Segoe UI, Arial, sans-serif">{total / 1000:.1f}</text>')


def structure_map(index: CountryIndex, cap, links) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.add(f'<rect width="{W}" height="{H}" fill="{PALETTE["sea"]}"/>')
    m.add('<g class="zoomable">')
    for name, poly, *_ in index.entries:
        fill = OTHER_FILL
        for _, (col, members) in SYNC.items():
            if name in members:
                fill = col
        d = " ".join(r + "Z" for r in (m.path(ring_, min_px=0.5) for ring_ in poly) if r)
        if d:
            stroke = "#6f6a61" if name in FOCUS else "#ffffff"
            sw = 1.0 if name in FOCUS else 0.8
            m.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
                  f'fill-rule="evenodd"/>')
    # links, drawn between country anchors
    for (a, b), mw in sorted(links.items(), key=lambda kv: kv[1]):
        if a not in ANCHOR or b not in ANCHOR:
            continue
        kind = LINK_TYPE.get((a, b), "sync")
        x1, y1 = m.xy(*anchor(a, b))
        x2, y2 = m.xy(*anchor(b, a))
        if mw <= 0:
            m.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                  f'stroke="#9e9a91" stroke-width="1.6" stroke-dasharray="1 4" '
                  f'stroke-linecap="round"/>')
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            m.add(f'<text x="{mx:.1f}" y="{my - 5:.1f}" text-anchor="middle" font-size="10" '
                  f'fill="#8a857c" font-style="italic" stroke="#fff" stroke-width="3" '
                  f'paint-order="stroke" font-family="Segoe UI, Arial, sans-serif">idle'
                  f'</text>')
            continue
        width = 1.5 + 4.5 * min(1.0, mw / 1500)
        col = PALETTE["ink"] if kind != "radial" else "#7f8c95"
        m.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
              f'stroke="{col}" stroke-width="{width:.1f}" stroke-linecap="round" '
              f'stroke-opacity="0.85"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if kind == "b2b":
            m.add(f'<rect x="{mx - 7:.1f}" y="{my - 7:.1f}" width="14" height="14" rx="2" '
                  f'fill="#fff" stroke="{PALETTE["ink"]}" stroke-width="2"/>')
            m.add(f'<path d="M{mx - 4:.1f},{my + 4:.1f}L{mx + 4:.1f},{my - 4:.1f}" '
                  f'stroke="{PALETTE["ink"]}" stroke-width="1.4"/>')
            ly = my - 12
        else:
            ly = my - 6
        m.add(f'<text x="{mx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="10.5" '
              f'font-weight="700" fill="{PALETTE["ink"]}" stroke="#fff" stroke-width="3" '
              f'paint-order="stroke" font-family="Segoe UI, Arial, sans-serif">'
              f'{mw:,.0f}</text>')
    for c, (lon, lat) in RING_AT.items():
        total = sum(cap[c].values())
        ring(m, lon, lat, cap[c], max(11, 2.6 * math.sqrt(total / 1000) * 2.2))
    for lon, lat, name in COUNTRY_LABELS:
        if name.title() in RING_AT:
            continue
        m.label(lon, lat, name, size=10, colour="#6f6a61", weight="500", halo=True)
    for c, (lon, lat) in RING_AT.items():
        dy = {"Turkiye": 2.25, "Georgia": 0.75, "Armenia": -0.5, "Azerbaijan": 0.75}[c]
        m.label(lon, lat + dy, c.upper(), size=11, colour=PALETTE["ink"], weight="700",
                halo=True)
    m.add("</g>")
    zones = [(sw_box(col), name) for name, (col, _) in SYNC.items()]
    zones.append((sw_box(OTHER_FILL), "Other or national"))
    legend = html_legend([
        ("Synchronous systems", zones),
        ("Interconnections, physical MW", [
            (sw_line(PALETTE["ink"], 4), "synchronous AC, width by MW"),
            ('<i class="d sq" style="background:#fff;border:2px solid #256081"></i>',
             "back to back converter"),
            (sw_line("#7f8c95", 3), "radial, island mode"),
            (sw_line("#9e9a91", 3, "1 4"), "idle since the 1990s"),
        ]),
        ("Rings", [(sw_dot(col), g) for g, col in GROUPS if g != "Other"]
         + [(sw_dot(GROUP_COLOUR["Other"]), "Biomass, geothermal")]),
    ])
    cap_txt = (f"Rings: installed GW, {YEAR}, model inputs. Links: physical MW, drawn "
               "country to country, not on routes. Nakhchivan is not tied to Azerbaijan.")
    return imap(svg_of(m), cap_txt, legend)


# ---------------------------------------------------------------------------
# 6. projects and corridors
# ---------------------------------------------------------------------------
# Land routes are not published for most projects. They are drawn as smooth
# curves through the named end substations and labelled schematic. The two
# subsea legs follow the surveyed Caucasus Cable System, with the announced
# landfalls, as in the digital note.
CCS = "Caucasus Cable System"
ANAKLIA = (41.573, 42.395)
CONSTANTA = [(28.75, 43.90), (28.66, 44.17)]
TRIPP_BBOX = (44.6, 38.7, 47.6, 39.95)

PROJECTS = [
    # map: the pill text on the map. A project without one shares its neighbour's pill.
    {"name": "BSSC", "map": "BSSC", "status": "planned", "route": "bssc", "mw": "1,300",
     "year": "2031", "label": (33.0, 43.75, "middle"),
     "corridor": "Caspian and Caucasus to EU", "volt": "HVDC, ±525 kV",
     "link": "Georgia to Romania, subsea, Anaklia to Constanta",
     "stage": "Feasibility done. EU PMI list. World Bank support (ESPIRE).",
     "stake": "First direct Caucasus to EU link."},
    {"name": "GEC", "map": "GEC", "status": "planned", "route": "gec", "mw": "up to 3,900",
     "year": "2036 to 2037", "label": (33.0, 41.95, "middle"),
     "corridor": "Caspian and Caucasus to EU", "volt": "HVDC",
     "link": "Azerbaijan to Georgia to Romania, three subsea cables",
     "anchors": [(49.6, 40.45), (47.1, 40.75), (45.1, 41.45), (43.9, 41.85),
                 (42.6, 42.15), (41.65, 42.25)],
     "stage": "EU TYNDP listed. Feasibility ongoing.",
     "stake": "Caspian offshore wind for Europe. Needs the AZ to GE backbone."},
    {"name": "Trans-Caspian", "map": "Trans-Caspian", "status": "planned", "mw": "1,000",
     "year": "2035 or later",
     "anchors": [(49.55, 40.35), (50.6, 41.6), (51.0, 42.9), (51.17, 43.65)],
     "label": (50.75, 42.75, "end"), "corridor": "Caspian and Caucasus to EU",
     "volt": "HVDC", "link": "Kazakhstan to Azerbaijan, subsea (Green Corridor Alliance)",
     "stage": "Feasibility ongoing. ADB and AIIB.",
     "stake": "Reaches Central Asian wind and solar. Least mature leg."},
    {"name": "BSTN extension", "map": "BSTN ext.", "status": "planned", "mw": "350",
     "year": "2032", "anchors": [(42.98, 41.64), (42.2, 40.9), (41.55, 40.3)],
     "label": (41.3, 40.95, "end"), "corridor": "Georgia to Turkiye",
     "volt": "400 kV AC, B2B",
     "link": "Third back to back unit at Akhaltsikhe, Tortum line",
     "stage": "In the GSE ten year plan.",
     "stake": "Georgia to Turkiye from 700 to 1,050 MW."},
    {"name": "EWTC", "map": "EWTC", "status": "planned", "mw": "775 + 775",
     "year": "2033 and 2034",
     "anchors": [(27.77, 41.57), (26.9, 41.95), (25.92, 42.15)],
     "extra": [[(27.09, 41.43), (26.5, 41.2), (25.85, 41.05)]],
     "label": (27.7, 42.3, "start"), "corridor": "Turkiye to EU", "volt": "400 kV AC",
     "link": "Turkiye to Bulgaria (Vize to Maritsa Iztok) and Greece (Babaeski to Nea Santa)",
     "stage": "Eight TSO consortium. USTDA funded study.",
     "stake": "Turkiye's export door to the EU. CBAM sets the price."},
    {"name": "CTN", "map": "CTN", "status": "planned", "mw": "350", "year": "2029",
     "anchors": [(44.81, 41.48), (44.85, 41.33), (44.87, 41.19)],
     "label": (45.15, 41.38, "start"), "corridor": "Iran to Russia, north to south",
     "volt": "400 and 500 kV, B2B",
     "link": "Georgia to Armenia, back to back converter at Ayrum",
     "stage": "KfW led. Design ready.",
     "stake": "First controllable link between the Russian and Iranian systems."},
    {"name": "ArTur", "map": "ArTur", "status": "planned", "mw": "300", "year": "2030",
     "anchors": [(43.85, 40.79), (43.5, 40.75), (43.10, 40.60)],
     "label": (43.3, 40.25, "end"), "corridor": "Nakhchivan and TRIPP", "volt": "n/a",
     "link": "Armenia to Turkiye, Gyumri to Kars direction",
     "stage": "Model scenario only.",
     "stake": "Reopens the line idle since 1993. Needs normalisation."},
    {"name": "TRIPP, Azerbaijan to Nakhchivan", "map": "TRIPP", "status": "committed",
     "route": "tripp", "mw": "800", "year": "2028", "label": (45.75, 38.2, "middle"),
     "corridor": "Nakhchivan and TRIPP", "volt": "330 kV",
     "link": "Jabrayil to Nakhchivan, double circuit, Aras valley",
     "stage": "Under construction.",
     "stake": "Reconnects Nakhchivan to Azerbaijan, first time since the 1990s."},
    {"name": "TRIPP, Nakhchivan to Turkiye", "status": "planned", "mw": "1,000",
     "year": "2032", "anchors": [(45.41, 39.21), (44.7, 39.5), (44.04, 39.92)],
     "corridor": "Nakhchivan and TRIPP", "volt": "400 kV",
     "link": "Nakhchivan to Igdir",
     "stage": "",
     "stake": "With the leg above, ties Azerbaijan to Turkiye's grid."},
    {"name": "AGIR", "map": "AGIR", "status": "committed", "mw": "850", "year": "2027",
     "anchors": [(46.24, 38.9), (46.15, 39.2), (46.08, 39.53)],
     "label": (46.75, 39.75, "start"), "corridor": "Iran to Russia, north to south",
     "volt": "400 kV", "link": "Iran to Armenia, third line to Noravan",
     "stage": "Under construction.",
     "stake": "Triples Armenia to Iran capacity. Gas for power swap."},
    {"name": "Reyhanli to Harim", "map": "Reyhanli to Harim", "status": "committed",
     "mw": "500", "year": "2027",
     "anchors": [(36.57, 36.27), (36.55, 36.24), (36.52, 36.21)],
     "label": (36.75, 36.05, "start"), "corridor": "Turkiye to Middle East",
     "volt": "400 kV", "link": "Turkiye to Syria",
     "stage": "World Bank Syria emergency project.",
     "stake": "Restores supply to northern Syria."},
    {"name": "Romania to Moldova", "map": "RO to MD", "status": "committed", "mw": "630",
     "year": "2026", "anchors": [(28.40, 45.68), (28.75, 46.35), (28.86, 47.01)],
     "label": (29.05, 46.4, "start"), "corridor": "Romania and Moldova", "volt": "400 kV",
     "link": "Vulcanesti to Chisinau, phase I",
     "stage": "Under construction. World Bank, EBRD, EIB, EU.",
     "stake": "Ends Moldova's reliance on the Transnistrian plant."},
]
STATUS_STYLE = {
    "committed": (PALETTE["building"], "7 4", "Committed"),
    "planned": (PALETTE["planned"], "1.5 4", "Planned"),
}
# Pill text colour: mustard is too pale to read, so planned pills write in ink.
PILL_TEXT = {"committed": PALETTE["building"], "planned": PALETTE["ink"]}


def ccs_route(cables, offset=0.0):
    for feat in cables["features"]:
        if feat["properties"].get("name") != CCS:
            continue
        line = [tuple(p) for p in max(lines_of(feat["geometry"]), key=len)]
        if line[0][0] < line[-1][0]:
            line.reverse()
        mid = [(lon, lat - offset) for lon, lat in line[1:-1]]
        return [ANAKLIA] + mid + CONSTANTA
    return []


def tripp_alignment(railways, index):
    out = []
    x0, y0, x1, y1 = TRIPP_BBOX
    for feat in railways["features"]:
        if feat["properties"].get("layer") != "corridor":
            continue
        for line in lines_of(feat["geometry"]):
            lon = sum(p[0] for p in line) / len(line)
            lat = sum(p[1] for p in line) / len(line)
            if not (x0 <= lon <= x1 and y0 <= lat <= y1):
                continue
            if index.at(lon, lat) not in ("Armenia", "Azerbaijan"):
                continue
            out.append(line)
    return out


def pill(m: Map, lon, lat, anchor, text, status):
    """Rounded label box, border in the status colour."""
    x, y = m.xy(lon, lat)
    w = 7.0 * len(text) + 14
    left = {"start": x, "end": x - w, "middle": x - w / 2}[anchor]
    col, _, _ = STATUS_STYLE[status]
    m.add(f'<rect x="{left:.1f}" y="{y - 11:.1f}" width="{w:.1f}" height="18" rx="5" '
          f'fill="#fff" fill-opacity="0.95" stroke="{col}" stroke-width="1.6"/>')
    m.add(f'<text x="{left + w / 2:.1f}" y="{y + 2:.1f}" text-anchor="middle" '
          f'font-size="11" font-weight="700" fill="{PILL_TEXT[status]}" '
          f'font-family="Segoe UI, Arial, sans-serif">{esc(text)}</text>')


def project_paths(m: Map, index: CountryIndex, p, cables, railways):
    """SVG paths of one project and its end nodes, routed as on the projects map."""
    paths, ends = [], []
    if p.get("route") == "bssc":
        route = ccs_route(cables)
        paths.append(m.path(route, min_px=0.5))
        ends += [route[0], route[-1]]
    elif p.get("route") == "gec":
        paths.append(m.smooth(p["anchors"]))
        route = ccs_route(cables, offset=0.28)
        paths.append(m.path(route, min_px=0.5))
        ends += [p["anchors"][0], route[-1]]
    elif p.get("route") == "tripp":
        paths += [m.path(line, min_px=0.5) for line in tripp_alignment(railways, index)]
    else:
        for line in [p["anchors"]] + p.get("extra", []):
            paths.append(m.smooth(line))
            ends += [line[0], line[-1]]
    return paths, ends


def projects_map(index: CountryIndex) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    hv = load_json(DIGITAL_DATA / "osm_hv_lines.geojson")
    for feat in hv["features"]:
        if voltage_of(feat["properties"]) < 380:
            continue
        for line in lines_of(feat["geometry"]):
            m.add(f'<path d="{m.path(line, min_px=1.5)}" fill="none" stroke="#c9d3d9" '
                  f'stroke-width="0.8"/>')
    cables = load_json(DIGITAL_DATA / "submarine_cables.geojson")
    railways = load_json(DIGITAL_DATA / "osm_railways.geojson")
    label_countries(m, size=9)
    nodes = []
    for p in PROJECTS:
        col, dash, _ = STATUS_STYLE[p["status"]]
        width = 3.2 if p["status"] == "planned" else 2.8
        paths, ends = project_paths(m, index, p, cables, railways)
        nodes += [(e, col) for e in ends]
        for d in paths:
            m.add(f'<path d="{d}" fill="none" stroke="#fff" stroke-width="{width + 2.4}" '
                  f'stroke-linecap="round" stroke-opacity="0.9"/>')
            m.add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{width}" '
                  f'stroke-dasharray="{dash}" stroke-linecap="round"><title>'
                  f'{esc(p["name"])}, {esc(p["mw"])} MW, {esc(p["year"])}</title></path>')
    for (lon, lat), col in nodes:
        x, y = m.xy(lon, lat)
        m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="#fff" stroke="{col}" '
              f'stroke-width="2"/>')
    for p in PROJECTS:
        if p.get("map"):
            pill(m, *p["label"], p["map"], p["status"])
    m.add("</g>")
    legend = html_legend([
        ("Status", [(sw_line(PALETTE["building"], 3, "7 4"), "committed"),
                    (sw_line(PALETTE["planned"], 3, "1.5 4"), "planned"),
                    (sw_line("#c9d3d9", 2), "existing grid, 380 kV and above")]),
    ])
    cap = ("Subsea legs: Caucasus Cable System route. TRIPP: Aras valley alignment. "
           "Other routes are schematic. MW and years: study inputs.")
    rows = []
    for corridor in dict.fromkeys(p["corridor"] for p in PROJECTS):
        group = [p for p in PROJECTS if p["corridor"] == corridor]
        for i, p in enumerate(group):
            col, _, word = STATUS_STYLE[p["status"]]
            first = (f'<td rowspan="{len(group)}" class="corr">{esc(corridor)}</td>'
                     if i == 0 else "")
            rows.append(
                f"<tr>{first}<td><b>{esc(p['name'])}</b><br><span class='small'>"
                f"{esc(p['link'])}</span></td><td>{esc(p['volt'])}</td>"
                f"<td class='n'>{esc(p['mw'])}</td><td>{esc(p['year'])}</td>"
                f"<td><span class='st' style='border-color:{col};"
                f"color:{col if p['status'] == 'committed' else '#8a6d00'}'>{word}</span>"
                f"<br><span class='small'>{esc(p['stage'])}</span></td>"
                f"<td class='small'>{esc(p['stake'])}</td></tr>"
            )
    table = ('<table class="kt pt"><thead><tr><th>Corridor</th><th>Project</th>'
             '<th>Voltage</th><th class="n">MW</th><th>Entry</th><th>Status</th>'
             '<th>What is at stake</th>'
             f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')
    return imap(svg_of(m), cap, legend) + table


# ---------------------------------------------------------------------------
# pane
# ---------------------------------------------------------------------------
def pane(index: CountryIndex, heads) -> str:
    """heads: the block titles, in outline order, from the pane spec."""
    zc = zone_country()
    cap = model_capacity(zc)
    peak, energy = model_demand(zc)
    links = physical_links(reference_lines())
    primer_title, map_b, map_c = heads
    return "".join([
        '<details class="primer">',
        f'<summary>{esc(primer_title)}</summary>',
        f'<div class="primer-body">{primer()}</div></details>',
        '<h4 class="mh">Key figures</h4>',
        key_figures(cap, peak, energy, links),
        '<h4 class="mh">Background</h4>',
        background(),
        '<h4 class="mh">The region on the ground</h4>',
        ground_map(index),
        f'<h4 class="mh">{esc(map_b)}</h4>',
        structure_map(index, cap, links),
        f'<h4 class="mh">{esc(map_c)}</h4>',
        projects_map(index),
    ])
