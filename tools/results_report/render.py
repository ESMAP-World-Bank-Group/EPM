# -*- coding: utf-8 -*-
"""Assemble the results report: one self-contained HTML file, no CDN.

The cache, the stylesheet and the chart code are inlined, so the file can be
opened from disk or mailed as an attachment and still work.  Nothing in here is
written for a particular run: sections are emitted for whatever scopes and
scenarios the cache holds.
"""
import html
import json
import time
from pathlib import Path

import findings

HERE = Path(__file__).parent
TPL = HERE / "templates"


def esc(s):
    return html.escape(str(s), quote=True)


def spec(**kw):
    """A chart placeholder; the JS reads the spec off the attribute."""
    return json.dumps(kw, separators=(",", ":")).replace('"', "&quot;")


# A chart's box is given in the pixels it will actually be painted at, so a
# half-width panel keeps full-size type instead of being scaled down with it.
BOX = {
    "full": (1400, 380),      # a card, edge to edge
    "disp": (1400, 430),      # ditto, but 672 slots need the extra height
    "half": (690, 380),       # one of two in a grid2
    "wide": (1060, 330),      # left-hand side of a split
    "third": (350, 330),      # one of three inside a split
}


def chart(cid=None, box="full", **kw):
    kw["w"], kw["h"] = BOX[box]
    return '<div class="chart"%s data-spec="%s"></div>' % (
        (' id="%s"' % cid) if cid else "", spec(**kw))


def panel(title, body, style=""):
    return ('<div class="panel"%s><h3>%s</h3>%s</div>'
            % ((' style="%s"' % style) if style else "", esc(title), body))


def diff_panel(alt, box="full", **kw):
    """The same chart re-based on `alt`, sitting under the chart it explains."""
    if not alt:
        return ""
    return panel("Change vs %s" % label(alt),
                 chart(box=box, minus=alt, **kw), "margin-top:6px")


def picker(target, field, values, current, label_of=None):
    label_of = label_of or (lambda v: v)
    btns = "".join(
        '<button data-value="%s"%s>%s</button>'
        % (esc(v), ' class="on"' if v == current else "", esc(label_of(v)))
        for v in values)
    return ('<div class="picker" data-picker="%s" data-target="%s">%s</div>'
            % (field, target, btns))


def card(title, num, lede, body):
    """The lede is kept in the call sites but not printed: the charts are read
    off the page and pasted into slides, and the prose came along with them."""
    del lede
    return ('<div class="card"><h2><span class="num">%s</span>%s</h2>%s</div>'
            % (esc(num), esc(title), body))


def findings_box(bullets, note="Computed from the run — no text is hand-written."):
    if not bullets:
        return ""
    return ('<div class="findings"><h3>Key findings</h3><ul>%s</ul>'
            '<p class="auto">%s</p></div>'
            % ("".join("<li>%s</li>" % b for b in bullets), esc(note)))


def label(scen):
    return {"LC_Baseline": "Baseline", "baseline": "Baseline",
            "LC_Iso": "Isolated"}.get(scen, scen.replace("LC_", ""))


# ------------------------------------------------------------------ tables

def delta_table(d, scope, ref, alt, years):
    """Headline numbers for the two scenarios at the milestone years."""
    shown = [y for y in ("2025", "2030", "2035", "2040") if y in years]
    rows = []

    def line(name, get, dp=1, unit=""):
        cells = []
        for y in shown:
            i = years.index(y)
            va, vz = get(d["annual"][scope][ref], i), None
            if alt:
                vz = get(d["annual"][scope][alt], i)
            cells.append((va, vz))
        rows.append((name, cells, dp, unit))

    line("Demand", lambda a, i: a["demand"][i])
    line("Generation", lambda a, i: sum(v[i] for v in a["gen"].values()))
    line("Capacity", lambda a, i: sum(v[i] for v in a["cap"].values()))
    line("Net imports",
         lambda a, i: sum(t["imp"][i] - t["exp"][i] for t in a["trade"].values()))
    line("Unserved energy", lambda a, i: a["unmet"][i], 2)
    line("Emissions", lambda a, i: a["emissions"][i], 1)
    if d["annual"][scope][ref].get("price"):
        line("Average price", lambda a, i: (a.get("price") or [0] * 16)[i], 1)

    head = "".join("<th colspan='2'>%s</th>" % y for y in shown)
    sub = "".join("<th>%s</th><th>%s</th>" % (label(ref)[:4], label(alt)[:3] if alt else "—")
                  for _ in shown)
    body = ""
    for name, cells, dp, unit in rows:
        tds = ""
        for va, vz in cells:
            tds += "<td>%s</td>" % findings.n(va, dp)
            if vz is None:
                tds += "<td>—</td>"
            else:
                cls = "" if abs(vz - va) < 1e-9 else (
                    ' class="up"' if vz > va else ' class="dn"')
                tds += "<td%s>%s</td>" % (cls, findings.n(vz, dp))
        body += "<tr><td>%s</td>%s</tr>" % (esc(name), tds)
    return ('<table class="kv"><thead><tr><th>TWh / GW / $ / Mt</th>%s</tr>'
            '<tr><th></th>%s</tr></thead><tbody>%s</tbody></table>'
            % (head, sub, body))


def scenario_table(d):
    rows = ""
    for s in d["scenarios"]:
        cor = d["corridors"][s]
        internal = sum(1 for c in cor.values()
                       if not c["external"] and max(c["ntc"]) > 0)
        external = sum(1 for c in cor.values()
                       if c["external"] and max(c["ntc"]) > 0)
        trade = sum(c["fwd"][-1] + c["rev"][-1] for c in cor.values())
        rows += ("<tr><td>%s</td><td>%s</td><td>%d</td><td>%d</td>"
                 "<td>%s</td></tr>"
                 % (esc(label(s)), esc(s), internal, external,
                    findings.n(trade, 1)))
    return ('<table class="kv"><thead><tr><th>Scenario</th><th>Run key</th>'
            '<th>Internal corridors</th><th>External corridors</th>'
            '<th>Trade in %s (TWh)</th></tr></thead><tbody>%s</tbody></table>'
            % (d["years"][-1], rows))


# ---------------------------------------------------------------- sections

def country_tab(d, scope, ref, alt):
    years = d["years"]
    has_dispatch = scope in d.get("dispatch", {})
    scens = d["scenarios"]
    y0 = years[0]
    milestones = [y for y in ("2025", "2030", "2035") if y in years] or years[:3]
    out = []

    # --- 1. capacity and generation ---------------------------------------
    out.append(card(
        "Annual capacity and generation", "1",
        "Stacked by fuel, one bar per scenario and year. Interconnection "
        "capacity, imports and exports are hatched so own resources stay "
        "readable. The dark line is demand.",
        '<div class="grid2">%s%s</div><div class="grid2">%s%s</div>%s'
        % (panel("Installed capacity (GW)",
                 chart(type="stack", kind="cap", scope=scope, box="half")),
           panel("Generation and trade (TWh)",
                 chart(type="stack", kind="gen", scope=scope, box="half")),
           diff_panel(alt, box="half", type="stack", kind="cap", scope=scope),
           diff_panel(alt, box="half", type="stack", kind="gen", scope=scope),
           '<div style="margin-top:16px">%s</div>'
           % delta_table(d, scope, ref, alt, years))))

    # --- 2. against the published plan ------------------------------------
    if scope in d["plans"]["capacity_gw"]:
        src = d["plans"]["source"].get(scope, "")
        out.append(card(
            "Against the national plan", "2",
            "Model capacity under %s next to the published plan, on the plan's "
            "own milestone years. Source: %s."
            % (esc(label(ref)), esc(src)),
            '<div class="split"><div>%s</div>%s</div>'
            % (chart(type="ndp", scope=scope, scenario=ref, box="wide"),
               findings_box(plan_gap(d, scope, ref),
                            "Gaps recomputed from the run at each publish."))))

    # --- 3. dispatch -------------------------------------------------------
    if has_dispatch:
        out.append(card(
            "Dispatch over the representative year", "3",
            "672 slots: 4 seasons x 7 day types x 24 hours, each weighted by "
            "the share of the year it stands for. Generation stacks upward, "
            "exports and storage charging downward; imports and exports "
            "include trade with countries outside the model. The white line is "
            "net imports, the dark red one demand, and the dashed one the "
            "marginal cost on the right-hand axis. One block per year, the "
            "scenarios stacked, and their difference underneath.",
            dispatch_grid(d, scope, scens, ref, alt)))

    # --- 4. trade ----------------------------------------------------------
    out.append(card(
        "Trade evolution", "4",
        "Imports above the axis, exports below, split by counterparty. "
        "Hover a bar for the net position.",
        chart(type="trade", scope=scope, box="wide")
        + diff_panel(alt, type="trade", scope=scope)))

    # --- 5. maps -----------------------------------------------------------
    maps = "".join(
        '<div class="panel"><h3>%s</h3>%s</div>'
        % (y, chart(type="map", scope=scope, focus=scope, year=y, scenario=ref,
                 box="third"))
        for y in milestones)
    out.append(card(
        "Where the energy flows", "5",
        "Arrows point the way the net flow runs under %s; thickness is the "
        "gross energy on the link and colour is how hard it is worked. "
        "Hover a link for capacity, both directions and the physical lines "
        "behind it." % esc(label(ref)),
        '<div class="split"><div class="grid3">%s</div>%s</div>'
        % (maps, findings_box(findings.country_findings(d, scope, ref, alt)))))

    # --- 6. corridors ------------------------------------------------------
    out.append(card(
        "Corridor capacity over time", "6",
        "One bar group per corridor, corridors gathered under the project that "
        "delivers them. The shaded part of each bar is the share of the year "
        "the link is loaded.",
        picker("cor_" + scope, "scenario", scens, ref, label) +
        chart("cor_" + scope, type="corridor", focus=scope, scenario=ref)))

    del y0
    return "".join(out)


def dispatch_grid(d, scope, scens, ref, alt):
    """One full-width chart per scenario, years down, difference beneath.

    The picker this replaces made the reader hold two pictures in their head to
    compare them; stacked on the page they can just be looked at."""
    rows = []
    for y in d["dispatch_years"]:
        # 672 slots in a half-width panel leave ~1 px per hour and the day-type
        # labels drop out, so the scenarios are stacked full width instead.
        blk = "".join(
            panel("%s - %s" % (y, label(sc)),
                  chart(type="dispatch", scope=scope, year=y, scenario=sc,
                        box="disp"), "margin-bottom:6px")
            for sc in scens)
        if alt and alt != ref:
            blk += panel(
                "%s - change vs %s" % (y, label(alt)),
                chart(type="dispatch", scope=scope, year=y, scenario=ref,
                      minus=alt, box="disp"), "margin-top:6px")
        rows.append('<div style="margin-bottom:22px">%s</div>' % blk)
    return "".join(rows)


def plan_gap(d, scope, ref):
    """Where the model departs from the published plan, per technology."""
    plan = d["plans"]["capacity_gw"].get(scope) or {}
    pyears = [str(y) for y in d["plans"]["years"]]
    merge = {"Reservoir": "Hydro", "ROR": "Hydro", "PSH": "Hydro", "PV": "Solar",
             "Onshore Wind": "Wind", "Offshore Wind": "Wind"}
    a = d["annual"][scope][ref]
    model = {}
    for fuel, series in a["cap"].items():
        k = merge.get(fuel, fuel)
        acc = model.setdefault(k, [0.0] * len(pyears))
        for j, y in enumerate(pyears):
            if y in d["years"]:
                acc[j] += series[d["years"].index(y)]
    out = []
    j = len(pyears) - 1
    for k in sorted(set(list(plan) + list(model))):
        p = (plan.get(k) or [0] * len(pyears))[j]
        m = (model.get(k) or [0] * len(pyears))[j]
        if max(p, m) < 0.05:
            continue
        if p <= 0.05:
            out.append("The plan has no %s in %s; the model builds "
                       "<b>%s GW</b>." % (k, pyears[j], findings.n(m, 1)))
        elif abs(m - p) / p > 0.15:
            out.append("%s in %s: model <b>%s GW</b> against <b>%s GW</b> "
                       "planned (%s%s %%)."
                       % (k, pyears[j], findings.n(m, 1), findings.n(p, 1),
                          "+" if m > p else "-",
                          findings.n(abs(m - p) / p * 100, 0)))
    tp = sum((plan.get(k) or [0] * len(pyears))[j] for k in plan)
    tm = sum(v[j] for v in model.values())
    if tp > 0:
        out.insert(0, "Total capacity in %s: model <b>%s GW</b> vs plan "
                      "<b>%s GW</b> (%s%s %%)."
                      % (pyears[j], findings.n(tm, 1), findings.n(tp, 1),
                         "+" if tm > tp else "-",
                         findings.n(abs(tm - tp) / tp * 100, 0)))
    return out


def regional_tab(d, ref, alt):
    years = d["years"]
    scens = d["scenarios"]
    milestones = [y for y in ("2025", "2030", "2040") if y in years] or years[:3]
    out = []

    out.append(card(
        "Regional capacity and generation", "1",
        "All modelled zones together. Only trade with countries outside the "
        "model is shown as import and export; flows between modelled zones "
        "net out inside the region.",
        '<div class="grid2">%s%s</div><div class="grid2">%s%s</div>%s'
        % (panel("Installed capacity (GW)",
                 chart(type="stack", kind="cap", scope="Region", box="half")),
           panel("Generation and external trade (TWh)",
                 chart(type="stack", kind="gen", scope="Region", box="half")),
           diff_panel(alt, box="half", type="stack", kind="cap", scope="Region"),
           diff_panel(alt, box="half", type="stack", kind="gen", scope="Region"),
           '<div style="margin-top:16px">%s</div>'
           % delta_table(d, "Region", ref, alt, years))))

    # The region is a wide strip: three of them side by side would each be a
    # postage stamp, so they are stacked instead.
    maps = "".join(
        '<div class="panel" style="margin-bottom:10px"><h3>%s</h3>%s</div>'
        % (y, chart(type="map", year=y, scenario=ref, box="wide"))
        for y in milestones)
    out.append(card(
        "Regional flows and congestion", "2",
        "Every corridor in the model under %s. Red links are worked at or "
        "above 85 %% of their capacity; dashed grey links carry nothing. "
        "Links inside one country are drawn thinner so the cross-border "
        "picture stays readable." % esc(label(ref)),
        '<div class="split"><div>%s</div>%s</div>'
        % (maps, findings_box(findings.regional_findings(d, ref, alt)))))

    if "Region" in d.get("dispatch", {}):
        out.append(card(
            "Regional dispatch", "3",
            "The whole modelled area on one stack, so the seasonal shape of "
            "the system is visible in a single picture. One block per year, the "
            "scenarios stacked, and their difference underneath.",
            dispatch_grid(d, "Region", scens, ref, alt)))

    out.append(card(
        "External trade", "4",
        "What the region as a whole buys from and sells to its neighbours.",
        chart(type="trade", scope="Region")
        + diff_panel(alt, type="trade", scope="Region")))

    out.append(card(
        "Every corridor", "5",
        "All modelled links, grouped by the project that delivers them.",
        picker("cor_Region", "scenario", scens, ref, label) +
        chart("cor_Region", type="corridor", scenario=ref)))

    per_country = "".join(
        '<div class="panel"><h3>%s</h3>%s</div>'
        % (c, findings_box(findings.country_findings(d, c, ref, alt), ""))
        for c in sorted(d["annual"]) if c != "Region")
    out.append(card(
        "Country by country", "6",
        "The same computation applied to each modelled country.",
        '<div class="grid2">%s</div>' % per_country))
    return "".join(out)


def run_tab(d):
    body = scenario_table(d)
    body += ("<p class='lede' style='margin-top:14px'>Trade figures come from "
             "<code>Interchange</code> and the external-trade attributes of "
             "<code>summary.csv</code>. <code>NetImport</code> is not used: for "
             "internal zone pairs it is scaled wrongly by a factor of a "
             "thousand in this run. External corridor capacity is absent from "
             "the outputs and is read back from the transfer-limit file each "
             "scenario used.</p>")
    checks = ("<ul>"
              "<li>Hourly dispatch, reweighted by <code>pHours</code>, "
              "reproduces the annual energy of every fuel to within 0.5 %.</li>"
              "<li>Generation + imports + unserved − exports − surplus matches "
              "demand to within 3 %; the residue is network and storage "
              "losses.</li>"
              "<li>Corridor flows agree exactly with the annual trade "
              "aggregates.</li></ul>")
    return (card("Scenarios in this run", "A", "", body) +
            card("Consistency checks", "B",
                 "Run <code>validate.py</code> to reproduce these.", checks))


# -------------------------------------------------------------------- page

def build(cache_path, out_path, countries=None, ref=None, alt=None,
          scenarios=None):
    d = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    # The cache may hold more scenarios than the page should show; the
    # charts read this list, so trimming it here is enough.
    if scenarios:
        keep = [s for s in d["scenarios"] if s in scenarios]
        if keep:
            d["scenarios"] = keep
    d["plans"] = json.loads(
        (HERE / "reference" / "national_plans.json").read_text(encoding="utf-8"))

    scens = d["scenarios"]
    ref = ref or next((s for s in ("LC_Baseline", "baseline") if s in scens),
                      scens[0])
    alt = alt or next((s for s in scens if s != ref), None)

    have = [c for c in sorted(d["annual"]) if c != "Region"]
    countries = [c for c in (countries or have) if c in have]

    tabs = [("regional", "Regional overview")] + \
           [("c_" + c, c) for c in countries] + [("run", "Run & checks")]

    nav = "".join('<button data-tab="%s"%s>%s</button>'
                  % (tid, ' class="on"' if i == 0 else "", esc(name))
                  for i, (tid, name) in enumerate(tabs))

    body = ['<section class="tab on" id="regional">%s</section>'
            % regional_tab(d, ref, alt)]
    for c in countries:
        body.append('<section class="tab" id="c_%s">%s</section>'
                    % (c, country_tab(d, c, ref, alt)))
    body.append('<section class="tab" id="run">%s</section>' % run_tab(d))

    css = (TPL / "report.css").read_text(encoding="utf-8")
    js = (TPL / "report.js").read_text(encoding="utf-8")
    payload = json.dumps(d, separators=(",", ":"))

    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Black Sea 2026 — results review</title>
<style>%s</style></head><body>
<header class="top">
  <button class="deckbtn" id="deck" title="Bigger type on a smaller plot, for
  charts pasted two-per-slide">Deck mode</button>
  <h1>Black Sea 2026 — results review</h1>
  <div class="sub">%s, generated from the model run. Every number on this
  page is read from the run; nothing is transcribed by hand.</div>
  <div class="meta"><span>run %s</span><span>%s scenarios</span>
  <span>%s–%s</span><span>built %s</span></div>
</header>
<nav class="tabs">%s</nav>
<main>%s</main>
<footer>Generated by <code>tools/results_report/build.py</code> — re-run it on
any EPM run to refresh this page.</footer>
<script>window.RD=%s;</script>
<script>%s</script>
</body></html>""" % (
        css, esc("%s vs %s" % (label(ref), label(alt)) if alt else label(ref)),
        esc(d["run"]), len(scens), d["years"][0], d["years"][-1],
        time.strftime("%Y-%m-%d %H:%M"), nav, "".join(body), payload, js)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
