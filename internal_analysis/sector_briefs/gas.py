"""Energy > Gas pane of the Black Sea sector briefs.

Volumes are national and company statistics, quoted with their year: the model
has no gas balance to draw on. The ground map draws the OpenStreetMap pipeline
extract of the digital note; trunk names, terminals, storage and fields are
placed by hand from operator sites.
"""

from __future__ import annotations

import math

from mapkit import DIGITAL_DATA, FOCUS, PALETTE, CountryIndex, Map, lines_of, load_json
from power import (BBOX, H, HERE, STATUS_STYLE, W, esc, html_legend, imap, label_countries,
                   pill, svg_of, sw_box, sw_dot, sw_line, tripp_alignment)

PIPE = "#b0703a"     # gas trunk lines on the ground map
PIPE_LIGHT = "#d9b48f"

# ---------------------------------------------------------------------------
# figures, each with its year and source
# ---------------------------------------------------------------------------
# Turkiye imports, 2024, EPDK: 52.2 bcm, shares by supplier.
TR_IMPORTS = 52.2
TR_SHARES = [("Russia", 0.42, "#8B7E72"), ("Azerbaijan", 0.22, "#5389AE"),
             ("Iran", 0.14, "#A98A62"), ("LNG", 0.22, "#84B3BB")]

# Azerbaijan, 2025, Ministry of Energy: production and exports by destination.
AZ_PRODUCTION = 51.5
AZ_EXPORTS = {"Europe": 12.8, "Turkiye": 9.6, "Georgia": 2.3, "Syria": 0.5}

COUNTRY_ROWS = [
    # country, production, imports or exports, suppliers, gas share of power 2025,
    # storage, role
    ("Turkiye", "3.5 (Sakarya, 2025)", "Imports 52.2 (2024)",
     "Russia 42%, Azerbaijan 22%, Iran 14%, LNG 22%", "22%",
     "About 6 bcm (Silivri, Tuz Golu)", "Importer. Transit to Europe, Syria, Nakhchivan."),
    ("Azerbaijan", "51.5 (2025)", "Exports 25.2 (2025)",
     "Europe 12.8, Turkiye 9.6, Georgia 2.3, Syria 0.5", "88%",
     "Garadagh, Kalmaz", "Exporter. Source of the Southern Gas Corridor."),
    ("Georgia", "None", "About 2.7 (2024)",
     "Azerbaijan 2.4, plus SCP transit gas in kind", "20%",
     "None", "Transit for SCP and for Russian gas to Armenia."),
    ("Armenia", "None", "Imports 2.7 (2024)",
     "Russia 2.3, Iran 0.45 (gas for power swap)", "34%",
     "Abovyan, small", "Importer. Gazprom Armenia owns the network."),
]


# ---------------------------------------------------------------------------
# 1. primer
# ---------------------------------------------------------------------------
def primer_svg() -> str:
    ink, acc, mut = PALETTE["ink"], PALETTE["accent"], "#6f6a61"
    out = ['<svg viewBox="0 0 860 210" width="100%" font-family="Segoe UI, Arial, sans-serif">',
           '<defs><marker id="ga" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
           f'markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="{acc}"/></marker></defs>']

    def box(x, y, head, sub, fill="#fff"):
        out.append(f'<rect x="{x}" y="{y}" width="118" height="54" rx="7" fill="{fill}" '
                   f'stroke="{ink}" stroke-width="1.2"/>')
        out.append(f'<text x="{x + 59}" y="{y + 23}" text-anchor="middle" font-size="13" '
                   f'font-weight="700" fill="{ink}">{head}</text>')
        out.append(f'<text x="{x + 59}" y="{y + 40}" text-anchor="middle" font-size="10.5" '
                   f'fill="{mut}">{sub}</text>')

    def arrow(x1, y1, x2, y2):
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{acc}" '
                   f'stroke-width="2" marker-end="url(#ga)"/>')

    top = [("Field", "Wells, processing"), ("Trunk pipeline", "Compressors, 80 bar"),
           ("Border", "Transit, fees"), ("Storage", "Winter swing"),
           ("Distribution", "Low pressure"), ("Consumers", "Power, heat, industry")]
    for i, (h, s) in enumerate(top):
        x = 12 + i * 140
        box(x, 18, h, s, "#eaf1f8" if h == "Trunk pipeline" else "#fff")
        if i < len(top) - 1:
            arrow(x + 120, 45, x + 138, 45)
    bottom = [("Liquefaction", "Exporter coast"), ("LNG carrier", "Any buyer"),
              ("Regasification", "Terminal or FSRU")]
    for i, (h, s) in enumerate(bottom):
        x = 12 + i * 140
        box(x, 130, h, s)
        if i < len(bottom) - 1:
            arrow(x + 120, 157, x + 138, 157)
    arrow(71, 74, 71, 128)
    arrow(351, 128, 351, 74)
    out.append(f'<text x="80" y="104" font-size="10.5" fill="{acc}" font-weight="600">'
               'or by sea</text>')
    out.append(f'<text x="460" y="150" font-size="10.5" fill="{mut}">Pipelines tie a buyer '
               'to a seller.</text>')
    out.append(f'<text x="460" y="166" font-size="10.5" fill="{mut}">LNG costs more but '
               'can come from anywhere.</text>')
    out.append("</svg>")
    return "".join(out)


def primer() -> str:
    bullets = [
        ("Pipelines lock in partners.",
         "A trunk line is built against long contracts. Its route fixes who sells to whom "
         "for decades."),
        ("LNG buys flexibility.",
         "Liquefaction, shipping and regasification add cost, but any seller can reach any "
         "terminal."),
        ("Transit pays.",
         "Countries crossed earn fees, often in gas. Georgia takes part of its SCP fee "
         "in kind."),
        ("Storage covers winter.",
         "Demand peaks in the cold months. Storage and LNG absorb the swing."),
        ("Here, gas and power are tied.",
         "Gas sets the power price in Azerbaijan and Turkiye. Armenia swaps Iranian gas "
         "for electricity."),
    ]
    items = "".join(f"<li><b>{esc(h)}</b> {esc(t)}</li>" for h, t in bullets)
    return (f'<div class="primer-fig">{primer_svg()}</div>'
            f'<ul class="primer-list">{items}</ul>')


# ---------------------------------------------------------------------------
# 2. key figures
# ---------------------------------------------------------------------------
def donut(parts, size=210, hole=0.6, centre=None, sub=None) -> str:
    total = sum(v for _, v, _ in parts)
    r = size / 2
    rin = r * hole
    out = []
    angle = -math.pi / 2
    for name, v, col in parts:
        sweep = 2 * math.pi * v / total
        a0, a1 = angle, angle + sweep
        large = 1 if sweep > math.pi else 0
        p = [(r + r * math.cos(a0), r + r * math.sin(a0)),
             (r + r * math.cos(a1), r + r * math.sin(a1)),
             (r + rin * math.cos(a1), r + rin * math.sin(a1)),
             (r + rin * math.cos(a0), r + rin * math.sin(a0))]
        out.append(f'<path d="M{p[0][0]:.1f},{p[0][1]:.1f} A{r:.1f},{r:.1f} 0 {large} 1 '
                   f'{p[1][0]:.1f},{p[1][1]:.1f} L{p[2][0]:.1f},{p[2][1]:.1f} '
                   f'A{rin:.1f},{rin:.1f} 0 {large} 0 {p[3][0]:.1f},{p[3][1]:.1f}Z" '
                   f'fill="{col}" stroke="#fff" stroke-width="1"><title>{esc(name)}: '
                   f'{100 * v / total:.0f}%</title></path>')
        angle = a1
    if centre:
        out.append(f'<text x="{r}" y="{r + 2}" text-anchor="middle" font-size="22" '
                   f'font-weight="700" fill="{PALETTE["ink"]}">{esc(centre)}</text>')
    if sub:
        out.append(f'<text x="{r}" y="{r + 20}" text-anchor="middle" font-size="11" '
                   f'fill="#6f6a61">{esc(sub)}</text>')
    return (f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
            f'font-family="Segoe UI, Arial, sans-serif">{"".join(out)}</svg>')


def key_figures() -> str:
    exported = sum(AZ_EXPORTS.values())
    tiles = [
        (f"{AZ_PRODUCTION:.1f} bcm", "Azerbaijan production, 2025. The only producer of "
                                     "scale among the four."),
        (f"{exported:.1f} bcm", f"Azerbaijan exports, 2025. "
                                f"{100 * AZ_EXPORTS['Europe'] / exported:.0f}% to Europe."),
        (f"{TR_IMPORTS:.0f} bcm", "Turkiye imports, 2024. Russia still the first supplier."),
        ("58 bcm/y", "Turkiye LNG regasification capacity, 2025. Above its total imports."),
    ]
    stats = "".join(f'<div class="stat"><div class="v">{esc(v)}</div>'
                    f'<div class="l">{esc(lab)}</div></div>' for v, lab in tiles)
    legend = "".join(f'<span class="lg">{sw_dot(col)}{name} {100 * s:.0f}%</span>'
                     for name, s, col in TR_SHARES)
    head = (
        '<div class="headrow">'
        f'<div class="hr-kpi"><div class="stats">{stats}</div></div>'
        '<div class="hr-mix"><div class="mixcard">'
        '<div class="mixhead">Turkiye gas imports by supplier, 2024</div>'
        f'{donut(TR_SHARES, centre=f"{TR_IMPORTS:.0f} bcm", sub="imported")}'
        f'<div class="mixleg">{legend}</div></div></div></div>'
    )
    body = "".join(
        f"<tr><td><b>{esc(c)}</b></td><td>{esc(prod)}</td><td>{esc(trade)}</td>"
        f"<td class='small'>{esc(sup)}</td><td class='n'>{esc(share)}</td>"
        f"<td class='small'>{esc(sto)}</td><td class='small'>{esc(role)}</td></tr>"
        for c, prod, trade, sup, share, sto, role in COUNTRY_ROWS
    )
    table = (
        '<table class="kt"><thead><tr><th>Country</th><th>Production, bcm</th>'
        '<th>Trade, bcm</th><th>Suppliers or buyers</th><th class="n">Gas in power</th>'
        '<th>Storage</th><th>Role</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )
    note = ('<p class="note">Turkiye: EPDK. Azerbaijan: Ministry of Energy. Georgia: '
            "Geostat. Armenia: Interfax. Gas in power: Ember, 2025. Turkish and Azerbaijani "
            "statistics differ on their bilateral flow.</p>")
    return head + table + note


# ---------------------------------------------------------------------------
# 3. background
# ---------------------------------------------------------------------------
TIMELINE = [
    ("1987", "First Soviet gas reaches Turkiye through the Trans-Balkan pipeline.", "BOTAS"),
    ("2001", "Tabriz to Ankara pipeline brings Iranian gas to Turkiye.", "BOTAS"),
    ("2003", "Blue Stream links Russia to Samsun under the Black Sea.", "Gazprom"),
    ("Dec 2006", "South Caucasus Pipeline carries the first Shah Deniz gas through "
                 "Georgia to Turkiye.", "bp"),
    ("2009", "Armenia starts swapping Iranian gas for electricity, 3 kWh per cubic "
             "metre.", "Interfax"),
    ("Jun 2018", "TANAP opens across Turkiye.", "TANAP"),
    ("Jan 2020", "TurkStream starts. Russian gas to Turkiye no longer crosses Ukraine.",
     "Gazprom"),
    ("Dec 2020", "TAP delivers the first Azerbaijani gas to Italy. The Southern Gas "
                 "Corridor is complete.", "TAP AG"),
    ("Jul 2022", "EU and Azerbaijan agree to double supply to 20 bcm a year by 2027.",
     "European Commission"),
    ("Apr 2023", "First gas from Sakarya, Turkiye's Black Sea field, lands at Filyos.",
     "TPAO"),
    ("Mar 2025", "Igdir to Nakhchivan pipeline opens. Nakhchivan no longer depends on "
                 "Iranian gas.", "SOCAR"),
    ("Aug 2025", "Azerbaijani gas reaches Syria through Kilis to Aleppo.", "Anadolu"),
    ("Jan 2026", "TAP adds 1.2 bcm a year, its first expansion step.", "TAP AG"),
]


def background() -> str:
    rows = "".join(
        f'<li><span class="tl-y">{esc(y)}</span><span class="tl-t">{esc(t)}</span>'
        f'<span class="tl-s">{esc(s)}</span></li>'
        for y, t, s in TIMELINE
    )
    return f'<ol class="tl">{rows}</ol>'


# ---------------------------------------------------------------------------
# 4. the region on the ground
# ---------------------------------------------------------------------------
# Points placed by hand from operator sites. Approximate to a few kilometres.
LNG = [("Marmara Ereglisi", 27.95, 40.97), ("Aliaga", 26.95, 38.82),
       ("Dortyol FSRU", 36.17, 36.85), ("Saros FSRU", 26.65, 40.6),
       ("Alexandroupolis FSRU", 25.97, 40.75), ("Revithoussa", 23.40, 37.96)]
STORAGE = [("Silivri", 28.1, 41.05), ("Tuz Golu", 33.4, 38.8), ("Garadagh", 49.6, 40.17),
           ("Kalmaz", 49.35, 40.05), ("Abovyan", 44.6, 40.27)]
FIELDS = [("Shah Deniz", 50.35, 39.95), ("Absheron", 50.6, 40.15), ("Sakarya", 31.3, 42.45)]
# Trunk names, at a point on or beside each line.
TRUNKS = [
    (29.2, 42.75, "TurkStream", "start"), (36.6, 42.6, "Blue Stream", "start"),
    (45.4, 41.95, "SCP", "start"), (37.5, 39.35, "TANAP", "middle"),
    (44.35, 42.65, "North South (Russia to Armenia)", "start"),
    (40.4, 38.75, "Tabriz to Ankara", "middle"), (24.6, 40.55, "TAP", "middle"),
]


def ground_map(index: CountryIndex) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    pipes = load_json(DIGITAL_DATA / "osm_pipelines.geojson")
    for feat in pipes["features"]:
        gas = feat["properties"].get("substance") in ("gas", "natural_gas")
        if not gas:
            continue
        for line in lines_of(feat["geometry"]):
            m.add(f'<path d="{m.path(line, min_px=1.2)}" fill="none" stroke="{PIPE}" '
                  f'stroke-width="1.3" stroke-linecap="round" stroke-opacity="0.8"/>')
    plants = load_json(HERE / "data" / "gem_plants.json")
    for p in plants:
        if p["type"] != "oil/gas" or p.get("operating", 0) < 100:
            continue
        x, y = m.xy(p["lon"], p["lat"])
        r = max(1.5, 0.11 * math.sqrt(p["operating"]))
        m.add(f'<circle class="pl" data-r="{r:.1f}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
              f'fill="#A98A62" fill-opacity="0.35" stroke="#A98A62" stroke-width="0.8">'
              f'<title>{esc(p["name"])}, gas plant, {p["operating"]:,.0f} MW</title></circle>')
    label_countries(m)
    for name, lon, lat in FIELDS:
        x, y = m.xy(lon, lat)
        m.add(f'<path d="M{x:.1f},{y - 7:.1f}L{x + 6:.1f},{y + 4:.1f}L{x - 6:.1f},{y + 4:.1f}Z" '
              f'fill="{PALETTE["ink"]}" stroke="#fff" stroke-width="1"><title>{esc(name)} '
              f'field</title></path>')
        m.label(lon + 0.15, lat - 0.25, name, size=9.5, colour="#403b35", anchor="start",
                halo=True)
    for name, lon, lat in LNG:
        x, y = m.xy(lon, lat)
        m.add(f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" '
              f'transform="rotate(45 {x:.1f} {y:.1f})" fill="#84B3BB" stroke="#fff" '
              f'stroke-width="1"><title>{esc(name)}, LNG terminal</title></rect>')
    for name, lon, lat in STORAGE:
        x, y = m.xy(lon, lat)
        m.add(f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" rx="2" '
              f'fill="#fff" stroke="{PIPE}" stroke-width="2"><title>{esc(name)}, '
              f'underground storage</title></rect>')
    for lon, lat, text, anchor in TRUNKS:
        m.label(lon, lat, text, size=10, colour=PIPE, anchor=anchor, weight="700", halo=True)
    m.add("</g>")
    legend = html_legend([
        ("Network", [(sw_line(PIPE, 3), "gas pipeline, OpenStreetMap")]),
        ("Points", [
            ('<i class="d" style="background:#84B3BB;border-radius:1px;transform:rotate(45deg)">'
             '</i>', "LNG terminal"),
            ('<i class="d sq" style="background:#fff;border:2px solid #b0703a"></i>',
             "underground storage"),
            ('<svg class="lsw" width="12" height="11"><path d="M6,0L12,11L0,11Z" '
             'fill="#256081"/></svg>', "gas field"),
            (sw_dot("#A98A62"), "gas power plant, 100 MW and above"),
        ]),
    ])
    cap = ("Pipelines: OpenStreetMap, gas only. Gas plants: Global Energy Monitor, August "
           "2026. Terminals, storage and fields placed from operator sites.")
    return imap(svg_of(m), cap, legend, plants=True)


# ---------------------------------------------------------------------------
# 5. supply sources and flows
# ---------------------------------------------------------------------------
ROLE = {
    "Producer and exporter": ("#E6DCC2", {"Russia", "Azerbaijan", "Iran", "Turkmenistan",
                                          "Kazakhstan"}),
    "Importer and transit": ("#CFDDE7", {"Turkey", "Georgia"}),
    "Importer": ("#E9D6CD", {"Armenia", "Syria"}),
    "EU market": ("#DDE7D3", {"Bulgaria", "Greece", "Romania", "Hungary",
                              "Republic of Serbia", "Macedonia", "Albania", "Moldova"}),
}

# Flows: name, bcm, year, path through (lon, lat). Width scales with bcm.
FLOWS = [
    ("Russia", TR_SHARES[0][1] * TR_IMPORTS, "2024",
     [(37.6, 44.9), (35.0, 43.3), (33.2, 41.2)]),
    ("Iran", TR_SHARES[2][1] * TR_IMPORTS, "2024",
     [(46.6, 37.9), (43.6, 38.7), (40.5, 39.1)]),
    ("LNG", TR_SHARES[3][1] * TR_IMPORTS, "2024",
     [(27.5, 36.2), (28.6, 37.3), (29.8, 38.4)]),
    ("Azerbaijan to Turkiye", AZ_EXPORTS["Turkiye"], "2025",
     [(49.2, 40.55), (45.6, 41.4), (43.2, 41.45), (41.4, 40.45), (38.5, 39.9)]),
    ("Azerbaijan to Europe", AZ_EXPORTS["Europe"], "2025",
     [(49.2, 40.35), (45.6, 41.15), (43.0, 41.05), (40.0, 39.55), (33.0, 39.6),
      (28.0, 40.6), (25.0, 41.0), (22.6, 40.9)]),
    ("Azerbaijan to Georgia", AZ_EXPORTS["Georgia"], "2025",
     [(47.6, 41.35), (46.0, 41.75), (44.7, 41.9)]),
    ("Russia to Armenia", 2.3, "2024",
     [(44.6, 43.4), (44.85, 42.3), (44.8, 41.4), (44.65, 40.55)]),
    ("Iran to Armenia", 0.45, "2024", [(46.6, 38.7), (46.2, 39.15), (45.3, 39.85)]),
    ("Turkiye to Nakhchivan", 0.5, "2025", [(43.5, 39.8), (44.3, 39.55), (45.0, 39.3)]),
    ("Azerbaijan to Syria", AZ_EXPORTS["Syria"], "2025",
     [(37.2, 37.4), (37.1, 36.9), (37.15, 36.3)]),
]
# Label text and position for each flow.
FLOW_LABEL = {
    "Russia": (35.6, 42.6, "Russia {v}"), "Iran": (43.2, 38.25, "Iran {v}"),
    "LNG": (27.7, 37.35, "LNG {v}"),
    "Azerbaijan to Turkiye": (39.6, 40.55, "Azerbaijan {v}"),
    "Azerbaijan to Europe": (30.5, 40.35, "Azerbaijan to EU {v}"),
    "Azerbaijan to Georgia": (46.2, 42.15, "{v}"),
    "Russia to Armenia": (44.4, 42.75, "Russia {v}"),
    "Iran to Armenia": (47.0, 39.2, "Iran {v}"),
    "Turkiye to Nakhchivan": (43.9, 39.25, "{v}"),
    "Azerbaijan to Syria": (37.65, 36.75, "{v}"),
}


def structure_map(index: CountryIndex) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.add(f'<rect width="{W}" height="{H}" fill="{PALETTE["sea"]}"/>')
    m.add('<g class="zoomable">')
    m.add('<defs><marker id="fa" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="2.6" '
          'markerHeight="2.6" orient="auto"><path d="M0,0L10,5L0,10z" fill="#7a4a22"/>'
          '</marker></defs>')
    for name, poly, *_ in index.entries:
        fill = "#ECEAE4"
        for col, members in ROLE.values():
            if name in members:
                fill = col
        d = " ".join(r + "Z" for r in (m.path(ring, min_px=0.5) for ring in poly) if r)
        if d:
            stroke = "#6f6a61" if name in FOCUS else "#ffffff"
            m.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
                  f'stroke-width="{1.0 if name in FOCUS else 0.8}" fill-rule="evenodd"/>')
    for name, bcm, year, path in sorted(FLOWS, key=lambda f: -f[1]):
        width = 1.6 + 9 * min(1.0, bcm / 22)
        m.add(f'<path d="{m.smooth(path)}" fill="none" stroke="#fff" '
              f'stroke-width="{width + 2.5:.1f}" stroke-linecap="round" stroke-opacity="0.8"/>')
        m.add(f'<path d="{m.smooth(path)}" fill="none" stroke="{PIPE}" '
              f'stroke-width="{width:.1f}" stroke-linecap="round" stroke-opacity="0.9" '
              f'marker-end="url(#fa)"><title>{esc(name)}: {bcm:.1f} bcm, {year}</title></path>')
    for name, bcm, year, _ in FLOWS:
        lon, lat, fmt = FLOW_LABEL[name]
        x, y = m.xy(lon, lat)
        text = fmt.format(v=f"{bcm:.1f}" if bcm < 10 else f"{bcm:.0f}")
        m.add(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-size="11" '
              f'font-weight="700" fill="#7a4a22" stroke="#fff" stroke-width="3" '
              f'paint-order="stroke" font-family="Segoe UI, Arial, sans-serif">{esc(text)}'
              f'</text>')
    for lon, lat, name in [(34.0, 39.0, "TURKIYE"), (43.35, 42.25, "GEORGIA"),
                           (44.75, 40.35, "ARMENIA"), (47.9, 41.75, "AZERBAIJAN"),
                           (41.5, 46.4, "RUSSIA"), (49.0, 37.0, "IRAN"),
                           (25.2, 45.6, "ROMANIA"), (25.4, 42.85, "BULGARIA"),
                           (23.0, 39.6, "GREECE"), (38.3, 35.7, "SYRIA")]:
        focus = name.title() in {"Turkiye", "Georgia", "Armenia", "Azerbaijan"}
        m.label(lon, lat, name, size=11 if focus else 10,
                colour=PALETTE["ink"] if focus else "#6f6a61",
                weight="700" if focus else "500", halo=True)
    m.add("</g>")
    legend = html_legend([
        ("Role in gas", [(sw_box(col), name) for name, (col, _) in ROLE.items()]),
        ("Flows, bcm a year", [(sw_line(PIPE, 6), "width by volume")]),
    ])
    cap = ("Flows drawn country to country, not on routes. Into Turkiye: 2024 import "
           "shares (EPDK). Azerbaijani exports: 2025 (Ministry of Energy). Armenia: 2024. "
           "Nakhchivan: its annual need.")
    return imap(svg_of(m), cap, legend)


# ---------------------------------------------------------------------------
# 6. projects and corridors
# ---------------------------------------------------------------------------
PROJECTS = [
    {"name": "TAP expansion", "map": "TAP +", "status": "planned", "bcm": "10 to 20",
     "year": "from 2026", "corridor": "Southern Gas Corridor",
     "anchors": [(26.25, 40.95), (25.4, 41.12), (24.4, 40.95), (22.95, 40.65),
                 (21.3, 40.5)],
     "label": (23.9, 40.25, "middle"), "link": "Kipoi to Italy, through Greece and Albania",
     "stage": "First 1.2 bcm/y step in service, Jan 2026. Next steps by market test.",
     "stake": "Sets how much Caspian gas reaches the EU."},
    {"name": "TANAP expansion", "map": "TANAP +", "status": "planned", "bcm": "16 to 31",
     "year": "n/a", "corridor": "Southern Gas Corridor",
     "anchors": [(42.75, 41.45), (41.27, 39.9), (37.0, 39.75), (30.5, 39.78),
                 (26.6, 40.3), (26.25, 40.95)],
     "label": (36.6, 40.3, "middle"), "link": "Georgian border to Kipoi, compressors",
     "stage": "No FID.", "stake": "Bottleneck for any volume above today's contracts."},
    {"name": "Trans-Caspian Gas Pipeline", "map": "Trans-Caspian", "status": "planned",
     "bcm": "10 to 30", "year": "n/a", "corridor": "Southern Gas Corridor",
     "anchors": [(52.9, 40.0), (51.3, 39.85), (49.47, 40.18)],
     "label": (52.45, 39.35, "end"), "link": "Turkmenbashi to Sangachal, subsea",
     "stage": "Concept. No FID.", "stake": "Turkmen gas for Europe without Russia or Iran."},
    {"name": "Sakarya phase 2", "map": "Sakarya 2", "status": "committed", "bcm": "+3.7",
     "year": "2026", "corridor": "Turkiye domestic supply",
     "anchors": [(31.3, 42.45), (31.7, 42.0), (32.03, 41.56)],
     "label": (30.9, 42.0, "end"), "link": "Black Sea field to Filyos, Osman Gazi FPU",
     "stage": "TPAO. First gas planned Q3 2026.",
     "stake": "Doubles domestic output to about 20 mcm/d."},
    {"name": "Tuz Golu storage expansion", "map": "Tuz Golu", "status": "planned",
     "bcm": "n/a", "year": "n/a", "corridor": "Turkiye domestic supply",
     "point": (33.4, 38.8), "label": (33.9, 38.45, "start"),
     "link": "Salt cavern storage, central Anatolia",
     "stage": "BOTAS tender for the next phase.",
     "stake": "More winter cover. Turkiye targets far larger storage by 2028."},
    {"name": "IGB expansion", "map": "IGB +", "status": "planned", "bcm": "3 to 5",
     "year": "n/a", "corridor": "Turkiye and Greece to the Balkans",
     "anchors": [(25.4, 41.12), (25.55, 41.93), (25.63, 42.42)],
     "label": (25.0, 42.05, "end"), "link": "Komotini to Stara Zagora",
     "stage": "Market test.", "stake": "Moves Caspian and LNG gas north into the Balkans."},
    {"name": "TRIPP", "map": "TRIPP", "status": "planned", "route": "tripp", "bcm": "n/a",
     "year": "n/a", "corridor": "Nakhchivan and TRIPP", "label": (45.75, 38.2, "middle"),
     "link": "Azerbaijan to Nakhchivan through southern Armenia",
     "stage": "Announced Aug 2025, with rail and power. No design.",
     "stake": "Second route to Nakhchivan, after Igdir."},
]


def projects_map(index: CountryIndex) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    pipes = load_json(DIGITAL_DATA / "osm_pipelines.geojson")
    for feat in pipes["features"]:
        if feat["properties"].get("substance") not in ("gas", "natural_gas"):
            continue
        for line in lines_of(feat["geometry"]):
            m.add(f'<path d="{m.path(line, min_px=1.5)}" fill="none" stroke="{PIPE_LIGHT}" '
                  f'stroke-width="0.8"/>')
    railways = load_json(DIGITAL_DATA / "osm_railways.geojson")
    label_countries(m, size=9)
    draw_projects(m, PROJECTS, railways, index)
    m.add("</g>")
    legend = html_legend([
        ("Status", [(sw_line(PALETTE["building"], 3, "7 4"), "committed"),
                    (sw_line(PALETTE["planned"], 3, "1.5 4"), "planned"),
                    (sw_line(PIPE_LIGHT, 2), "existing gas pipelines")]),
    ])
    cap = ("Routes are schematic. TANAP and TAP follow their known corridors. TRIPP: Aras "
           "valley alignment.")
    return imap(svg_of(m), cap, legend) + table(PROJECTS, "Capacity, bcm/y", "bcm")


def draw_projects(m: Map, projects, railways, index, pills=True, pipe=False):
    """Routes, end nodes and pills. pipe draws a thin white core, for the overview."""
    nodes = []
    for p in projects:
        col, dash, _ = STATUS_STYLE[p["status"]]
        width = 3.4 if pipe else 3.0
        if p.get("point"):
            nodes.append((p["point"], col, 5.0))
            continue
        if p.get("route") == "tripp":
            paths = [m.path(line, min_px=0.5) for line in tripp_alignment(railways, index)]
        else:
            paths = [m.smooth(p["anchors"])]
            nodes += [(p["anchors"][0], col, 3.6), (p["anchors"][-1], col, 3.6)]
        for d in paths:
            m.add(f'<path d="{d}" fill="none" stroke="#fff" stroke-width="{width + 2.4}" '
                  f'stroke-linecap="round" stroke-opacity="0.9"/>')
            m.add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{width}" '
                  f'stroke-dasharray="{dash}" stroke-linecap="round"><title>{esc(p["name"])}'
                  f'</title></path>')
            if pipe:
                m.add(f'<path d="{d}" fill="none" stroke="#fff" stroke-width="1" '
                      f'stroke-linecap="round"/>')
    for (lon, lat), col, r in nodes:
        x, y = m.xy(lon, lat)
        m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#fff" stroke="{col}" '
              f'stroke-width="2"/>')
    if pills:
        for p in projects:
            if p.get("map"):
                pill(m, *p["label"], p["map"], p["status"])


def table(projects, cap_head, cap_key) -> str:
    rows = []
    for corridor in dict.fromkeys(p["corridor"] for p in projects):
        group = [p for p in projects if p["corridor"] == corridor]
        for i, p in enumerate(group):
            col, _, word = STATUS_STYLE[p["status"]]
            first = (f'<td rowspan="{len(group)}" class="corr">{esc(corridor)}</td>'
                     if i == 0 else "")
            stage = f"<br><span class='small'>{esc(p['stage'])}</span>" if p["stage"] else ""
            rows.append(
                f"<tr>{first}<td><b>{esc(p['name'])}</b><br><span class='small'>"
                f"{esc(p['link'])}</span></td><td class='n'>{esc(p[cap_key])}</td>"
                f"<td>{esc(p['year'])}</td><td><span class='st' style='border-color:{col};"
                f"color:{col if p['status'] == 'committed' else '#8a6d00'}'>{word}</span>"
                f"{stage}</td><td class='small'>{esc(p['stake'])}</td></tr>"
            )
    return ('<table class="kt pt"><thead><tr><th>Corridor</th><th>Project</th>'
            f'<th class="n">{esc(cap_head)}</th><th>Entry</th><th>Status</th>'
            '<th>What is at stake</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')


# ---------------------------------------------------------------------------
# pane
# ---------------------------------------------------------------------------
def pane(index: CountryIndex, heads) -> str:
    primer_title, map_b, map_c = heads
    return "".join([
        '<details class="primer">',
        f'<summary>{esc(primer_title)}</summary>',
        f'<div class="primer-body">{primer()}</div></details>',
        '<h4 class="mh">Key figures</h4>',
        key_figures(),
        '<h4 class="mh">Background</h4>',
        background(),
        '<h4 class="mh">The region on the ground</h4>',
        ground_map(index),
        f'<h4 class="mh">{esc(map_b)}</h4>',
        structure_map(index),
        f'<h4 class="mh">{esc(map_c)}</h4>',
        projects_map(index),
    ])
