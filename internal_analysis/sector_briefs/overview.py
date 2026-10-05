"""Energy > Overview pane of the Black Sea sector briefs.

Short by design: primary energy and the electricity mix from Our World in Data
(Energy Institute and Ember series), then where power and gas meet. The maps
sit in the Power and Gas panes.
"""

from __future__ import annotations

import csv

from mapkit import PALETTE
from power import GROUPS, HERE, esc, sw_dot

OWID = HERE / "data" / "owid_energy.csv"
STUDY = [("Turkiye", "Turkey"), ("Georgia", "Georgia"), ("Armenia", "Armenia"),
         ("Azerbaijan", "Azerbaijan")]
ELEC = {"Hydro": ["hydro"], "Gas": ["gas"], "Coal": ["coal"], "Solar": ["solar"],
        "Wind": ["wind"], "Nuclear": ["nuclear"], "Oil": ["oil"],
        "Other": ["biofuel", "other_renewables"]}

# Position of each country in energy trade. Short, sourced in the note.
POSITION = {
    "Turkiye": "Net importer. Buys nearly all its gas and oil, mines lignite.",
    "Georgia": "Net importer of fuels. Hydro covers most power, gas fills winter.",
    "Armenia": "Net importer. All gas from Russia and Iran, nuclear fuel from Russia.",
    "Azerbaijan": "Net exporter. Produces about 3.4 times the energy it uses.",
}


def latest(rows, col):
    vals = [(r["year"], float(r[col])) for r in rows if r.get(col)]
    return vals[-1] if vals else (None, None)


def load():
    want = {owid for _, owid in STUDY}
    rows = {}
    with open(OWID, encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["country"] in want:
                rows.setdefault(r["country"], []).append(r)
    out = {}
    for name, owid in STUDY:
        rs = rows[owid]
        year_pe, pe = latest(rs, "primary_energy_consumption")
        _, pc = latest(rs, "energy_per_capita")
        _, gen = latest(rs, "electricity_generation")
        mix = {}
        for g, keys in ELEC.items():
            mix[g] = sum(latest(rs, f"{k}_share_elec")[1] or 0.0 for k in keys)
        out[name] = {"pe": pe, "pe_year": year_pe, "pc": pc / 1000, "gen": gen, "mix": mix,
                     "gas_pe": latest(rs, "gas_share_energy")[1],
                     "prod": sum(latest(rs, f"{k}_production")[1] or 0.0
                                 for k in ("gas", "oil"))
                     if name == "Azerbaijan" else None}
    return out


def mixbar(mix, width=220):
    segs = "".join(
        f'<i style="width:{mix.get(g, 0):.2f}%;background:{col}" '
        f'title="{g} {mix.get(g, 0):.0f}%"></i>'
        for g, col in GROUPS if mix.get(g, 0) > 0.5
    )
    return f'<span class="mixbar" style="width:{width}px">{segs}</span>'


# ---------------------------------------------------------------------------
# 1. primer
# ---------------------------------------------------------------------------
def primer_svg() -> str:
    ink, acc, mut = PALETTE["ink"], PALETTE["accent"], "#6f6a61"
    out = ['<svg viewBox="0 0 860 170" width="100%" font-family="Segoe UI, Arial, sans-serif">',
           '<defs><marker id="oa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
           f'markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="{acc}"/></marker></defs>']

    def box(x, y, w, head, sub, fill="#fff"):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="50" rx="7" fill="{fill}" '
                   f'stroke="{ink}" stroke-width="1.2"/>')
        out.append(f'<text x="{x + w / 2}" y="{y + 22}" text-anchor="middle" font-size="13" '
                   f'font-weight="700" fill="{ink}">{head}</text>')
        out.append(f'<text x="{x + w / 2}" y="{y + 38}" text-anchor="middle" font-size="10.5" '
                   f'fill="{mut}">{sub}</text>')

    def arrow(x1, y1, x2, y2):
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{acc}" '
                   f'stroke-width="2" marker-end="url(#oa)"/>')

    box(12, 12, 170, "Primary energy", "Gas, oil, coal")
    box(12, 102, 170, "Primary energy", "Hydro, nuclear, wind, sun")
    box(290, 57, 170, "Power plants", "Losses, about half for thermal", "#eaf1f8")
    box(568, 12, 170, "Direct use", "Heat, transport, industry")
    box(568, 102, 170, "Electricity", "Homes, industry")
    arrow(184, 37, 566, 37)
    arrow(184, 50, 288, 74)
    arrow(184, 127, 288, 92)
    arrow(462, 92, 566, 122)
    out.append(f'<text x="760" y="42" font-size="10.5" fill="{mut}">Most energy</text>')
    out.append(f'<text x="760" y="56" font-size="10.5" fill="{mut}">never becomes</text>')
    out.append(f'<text x="760" y="70" font-size="10.5" fill="{mut}">electricity.</text>')
    out.append("</svg>")
    return "".join(out)


def primer() -> str:
    bullets = [
        ("Primary energy is the input.",
         "Fuels and natural flows before any conversion. Electricity is one output among "
         "several."),
        ("Thermal plants waste half or more.",
         "A gas plant turns 40 to 60% of its fuel into power. Hydro, wind and sun skip "
         "that loss."),
        ("Gas links the two sectors.",
         "Where gas sets the power price, a gas deal is also a power deal."),
    ]
    items = "".join(f"<li><b>{esc(h)}</b> {esc(t)}</li>" for h, t in bullets)
    return (f'<div class="primer-fig">{primer_svg()}</div>'
            f'<ul class="primer-list">{items}</ul>')


# ---------------------------------------------------------------------------
# 2. key figures
# ---------------------------------------------------------------------------
def key_figures(d) -> str:
    total = sum(v["pe"] for v in d.values())
    tr = d["Turkiye"]["pe"] / total
    az = d["Azerbaijan"]
    tiles = [
        (f"{total:,.0f} TWh", f"Primary energy, four countries. Turkiye uses "
                              f"{100 * tr:.0f}%."),
        (f"{az['prod'] / az['pe']:.1f}x", "Azerbaijan's oil and gas output over its own "
                                          "energy use, 2024."),
        (f"{d['Turkiye']['gas_pe']:.0f}%", "Gas in Turkiye's primary energy, 2024. Oil "
                                           "and coal weigh more."),
        (f"{d['Azerbaijan']['mix']['Gas']:.0f}%", "Gas in Azerbaijan's power, 2025. The "
                                                  "highest of the four."),
    ]
    stats = "".join(f'<div class="stat"><div class="v">{esc(v)}</div>'
                    f'<div class="l">{esc(lab)}</div></div>' for v, lab in tiles)
    legend = "".join(f'<span class="lg">{sw_dot(col)}{g}</span>' for g, col in GROUPS)
    body = "".join(
        f"<tr><td><b>{c}</b></td><td class='n'>{v['pe']:,.0f}"
        f"<span class='small'> ({v['pe_year']})</span></td><td class='n'>{v['pc']:.0f}</td>"
        f"<td class='n'>{v['gen']:,.0f}</td><td>{mixbar(v['mix'])}</td>"
        f"<td class='n'>{v['mix']['Gas']:.0f}%</td><td class='small'>{esc(POSITION[c])}</td>"
        f"</tr>"
        for c, v in d.items()
    )
    table = (
        '<table class="kt"><thead><tr><th>Country</th><th class="n">Primary energy, TWh</th>'
        '<th class="n">MWh per person</th><th class="n">Power output, TWh</th>'
        '<th>Power mix, 2025</th><th class="n">Gas in power</th><th>Energy position</th>'
        f"</tr></thead><tbody>{body}</tbody></table>"
        f'<div class="mixleg">{legend}</div>'
    )
    note = ('<p class="note">Our World in Data: Energy Institute for primary energy, Ember '
            "for power, 2025. Georgia and Armenia: primary energy 2023, no fuel split.</p>")
    return f'<div class="stats">{stats}</div>' + table + note


# ---------------------------------------------------------------------------
# 3. where power and gas meet
# ---------------------------------------------------------------------------
NEXUS = [
    ("Armenia", "Gas for electricity.",
     "Armenia pays for Iranian gas with power, about 3 kWh per cubic metre. The AGIR line "
     "triples the capacity for that swap."),
    ("Azerbaijan", "Every MWh of gas power is gas not exported.",
     "Gas makes 88% of its power. Wind and solar at home free gas for Europe at export "
     "prices."),
    ("Georgia", "Hydro in summer, gas in winter.",
     "Gas plants at Gardabani run on Azerbaijani gas when the rivers are low. Transit fees "
     "paid in gas cover part of it."),
    ("Turkiye", "Gas is priced on imports.",
     "Gas makes 22% of its power but sets the price in many hours. The import bill moves "
     "with Russian, Azerbaijani and LNG contracts."),
]


def nexus() -> str:
    cards = "".join(
        f'<div class="nx"><div class="nx-c">{esc(c)}</div><div class="nx-h">{esc(h)}</div>'
        f'<div class="nx-t">{esc(t)}</div></div>'
        for c, h, t in NEXUS
    )
    style = ('<style>.nxg{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));'
             'gap:12px;margin:6px 0 4px}.nx{border:1px solid #e6e6ea;border-radius:8px;'
             'padding:10px 12px;background:#fff}.nx-c{font-size:11px;font-weight:700;'
             f'letter-spacing:.04em;text-transform:uppercase;color:{PALETTE["accent"]}}}'
             f'.nx-h{{font-weight:700;color:{PALETTE["ink"]};margin:3px 0 4px}}'
             '.nx-t{font-size:13px;color:#403b35;line-height:1.4}</style>')
    links = ('<p class="note">Maps and projects: see the <a href="#energy/power">Power</a> '
             'and <a href="#energy/gas">Gas</a> tabs.</p>')
    return style + f'<div class="nxg">{cards}</div>' + links


# ---------------------------------------------------------------------------
# pane
# ---------------------------------------------------------------------------
def pane(index, heads) -> str:
    primer_title, _, _ = heads
    d = load()
    return "".join([
        '<details class="primer">',
        f'<summary>{esc(primer_title)}</summary>',
        f'<div class="primer-body">{primer()}</div></details>',
        '<h4 class="mh">Key figures</h4>',
        key_figures(d),
        '<h4 class="mh">Where power and gas meet</h4>',
        nexus(),
    ])
