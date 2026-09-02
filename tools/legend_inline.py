# -*- coding: utf-8 -*-
"""Move every chart legend inside its own white chart frame, as a column on the right.

The review pages grew two legend habits: a per-chart legend glued under the <svg>, and a
shared legend sitting above a grid of charts. Both cost vertical space and, in the shared
case, force the reader to look away from the chart to decode it. This rewrites both into
one layout: the <svg> and its legend side by side inside the same white .chart box, legend
stacked vertically on the right.

A shared legend is copied into every real chart it heads and then removed. Sparklines
(no max-width, drawn in .sparkcard) are left alone -- they have no legend and no room.

Charts that draw the peak-demand marker get a matching entry appended, since that series
was only ever explained in the prose above the grid.

The pages are regenerated in part by calibration_review_update.py and ndp_review_update.py,
which swap the <svg> element only, so this migration survives a regeneration. Re-running is
idempotent.

Usage:
    python tools/legend_inline.py [--check]
"""

import argparse
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAGES = [
    os.path.join(ROOT, "Data", "calibration", "calibration_review.html"),
    os.path.join(ROOT, "Data", "calibration", "ndp_review.html"),
]

MARKER = "/* legend-inline */"

CSS = MARKER + """
.chartrow{display:flex;flex-wrap:wrap;align-items:center;gap:10px}
.chartrow>svg{flex:1 1 0;min-width:0;order:0}
.chartrow>.leg,.chartrow>.legend{flex:0 0 auto;order:1;display:flex;flex-direction:column;flex-wrap:nowrap;align-items:flex-start;gap:3px;margin:0;max-width:112px}
.chartrow>.leg>span,.chartrow>.legend>span{display:inline-flex;align-items:flex-start;gap:4px;line-height:1.25;white-space:normal}
.chartrow>.leg i,.chartrow>.legend i{flex:0 0 auto;margin-top:2px}
.chartrow>.pn,.chartrow>.emrow,.chartrow>.expc{flex:1 0 100%;order:9;margin-top:2px}
.lgdem i{background:#c0392b!important;border-radius:1px;transform:rotate(45deg)}
@media(max-width:820px){.chartrow{display:block}.chartrow>.leg,.chartrow>.legend{flex-direction:row;flex-wrap:wrap;max-width:none;margin-top:4px}}
"""

# The peak-demand series is drawn in this colour and explained only in prose.
DEMAND_COLOUR = "#c0392b"
DEMAND_ITEM_LG = ('<span class="lg lgdem"><i></i><span class="lf">Pointe de demande (GW)</span>'
                  '<span class="le">Peak demand (GW)</span></span>')
DEMAND_ITEM_PLAIN = ('<span class="lgdem"><i></i><span class="lf">Pointe de demande (GW)</span>'
                     '<span class="le">Peak demand (GW)</span></span>')


# ------------------------------------------------------------------ html scanning

def match_div(s, i):
    """i points at a '<div'; return the index just past its matching '</div>'."""
    depth = 0
    for m in re.finditer(r"<div\b|</div>", s[i:]):
        depth += -1 if m.group(0) == "</div>" else 1
        if depth == 0:
            return i + m.end()
    return len(s)


def open_divs(s, pos):
    """Start offsets of every <div> still open at pos, innermost first."""
    stack = []
    for m in re.finditer(r"<div\b|</div>", s[:pos]):
        if m.group(0) == "</div>":
            if stack:
                stack.pop()
        else:
            stack.append(m.start())
    return list(reversed(stack))


def wrapped_spans(s):
    """Spans of the .chartrow wrappers an earlier run left behind."""
    return [(m.start(), match_div(s, m.start()))
            for m in re.finditer(r'<div class="chartrow">', s)]


def reach_of(s, hi, first_chart):
    """How far a section legend carries: out to the first ancestor that holds first_chart.

    A legend is sometimes the last child of a small wrapper, with the charts it explains
    sitting in the next wrapper along, so the innermost ancestor is not enough. Ancestor
    ends are resolved lazily -- matching every one of them is what made this slow.
    """
    if first_chart is None:
        return hi
    for i in open_divs(s, hi):
        end = match_div(s, i)
        if end > first_chart:
            return end
    return len(s)


def real_charts(s, lo, hi):
    """Spans of the chart <svg> elements in [lo, hi).

    Charts are viewBox-taller than 100 units; the sparklines drawn in .sparkcard are 54
    and carry no legend, so they are left alone.
    """
    out = []
    for m in re.finditer(r"<svg\b.*?</svg>", s, re.S):
        if m.start() < lo or m.start() >= hi:
            continue
        vb = re.search(r'viewBox="[-\d.]+ [-\d.]+ [-\d.]+ ([-\d.]+)"', m.group(0)[:220])
        if not vb or float(vb.group(1)) < 100:
            continue
        back = s[max(0, m.start() - 300):m.start()]
        if "sparkcard" in " ".join(re.findall(r'<div class="([^"]+)"', back)[-2:]):
            continue
        out.append((m.start(), m.end()))
    return out


def legend_blocks(s):
    """(start, end, class) for every legend div, outermost-first, in document order."""
    out = []
    for cls in ("leg", "legend"):
        for m in re.finditer(r'<div class="%s">' % cls, s):
            out.append((m.start(), match_div(s, m.start()), cls))
    return sorted(out)


def with_demand(legend_html, svg_html, cls):
    """Append the peak-demand entry when the chart actually draws that marker."""
    if DEMAND_COLOUR not in svg_html or "lgdem" in legend_html:
        return legend_html
    item = DEMAND_ITEM_LG if cls == "legend" else DEMAND_ITEM_PLAIN
    return legend_html[: -len("</div>")] + item + "</div>"


# ------------------------------------------------------------------- the rewrite

def rewrite(s, report):
    if MARKER not in s:
        m = re.search(r"</style>", s)
        if not m:
            sys.exit("no <style> block to extend")
        s = s[: m.start()] + CSS + s[m.start():]
        report.append("  CSS .chartrow added")
    else:
        s = re.sub(re.escape(MARKER) + r".*?(?=</style>)", CSS, s, count=1, flags=re.S)
        report.append("  CSS .chartrow already there, refreshed")

    done = wrapped_spans(s)
    fresh = lambda p: not any(a <= p < b for a, b in done)
    legends = [t for t in legend_blocks(s) if fresh(t[0])]
    charts = [t for t in real_charts(s, 0, len(s)) if fresh(t[0])]
    if not legends:
        report.append("  every legend is already framed, nothing to do")
        return s

    # A legend glued straight after an </svg> belongs to that one chart. Everything else
    # heads a group, and reaches every chart after it that is still inside its section.
    starts = [a for a, _ in charts]
    owner = {}          # svg start -> legend index
    glued = {}          # legend index -> svg start
    for k, (lo, hi, _) in enumerate(legends):
        if not s[:lo].rstrip().endswith("</svg>"):
            continue
        prev = [a for a, b in charts if b <= lo]
        if prev and s[prev[-1]:lo].strip().endswith("</svg>"):
            owner[prev[-1]] = k
            glued[k] = prev[-1]

    # A section legend carries to every chart between it and the next section legend.
    groups = [k for k in range(len(legends)) if k not in glued]
    reach = {}
    for n, k in enumerate(groups):
        hi = legends[k][1]
        nxt = legends[groups[n + 1]][0] if n + 1 < len(groups) else len(s)
        after = [a for a in starts if a >= hi and a < nxt]
        reach[k] = min(reach_of(s, hi, after[0] if after else None), nxt)

    for a in starts:
        if a in owner:
            continue
        cands = [k for k in groups if legends[k][1] <= a < reach[k]]
        if cands:
            owner[a] = cands[-1]        # nearest preceding section legend wins

    edits, used, demand = [], set(), 0
    for a, b in charts:
        k = owner.get(a)
        if k is None:
            continue
        lo, hi, cls = legends[k]
        svg_html, legend_html = s[a:b], s[lo:hi]
        if DEMAND_COLOUR in svg_html:
            demand += 1
        body = svg_html + with_demand(legend_html, svg_html, cls)
        # a glued legend sits between the </svg> and hi: swallow it in the same edit
        end = hi if glued.get(k) == a else b
        edits.append((a, end, '<div class="chartrow">' + body + "</div>"))
        used.add(k)

    for k, (lo, hi, cls) in enumerate(legends):
        if k in used and k not in glued:
            edits.append((lo, hi, ""))          # the group original is now redundant
        elif k not in used:
            report.append("  %-6s at offset %d: no chart in range, left in place"
                          % (cls, lo))

    edits.sort(reverse=True)
    for i in range(len(edits) - 1):
        if edits[i][0] < edits[i + 1][1]:
            sys.exit("overlapping edits: %r and %r" % (edits[i][:2], edits[i + 1][:2]))
    for a, b, new in edits:
        s = s[:a] + new + s[b:]

    n_glued = len(set(glued) & used)
    report.append("  %d legend(s) glued to a chart, folded to the right" % n_glued)
    report.append("  %d group legend(s) moved, in %d copies inside the frames"
                  % (len(used) - n_glued, len([e for e in edits if e[2]]) - n_glued))
    report.append("  %d chart(s) get the peak demand entry" % demand)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report without writing")
    args = ap.parse_args()

    for page in PAGES:
        if not os.path.isfile(page):
            print("%s: missing, skipped" % page)
            continue
        s = io.open(page, encoding="utf-8").read()
        report = []
        out = rewrite(s, report)
        print("\n%s" % os.path.basename(page))
        print("\n".join(report))
        print("  %d -> %d bytes, %d .chartrow" % (len(s), len(out), out.count('class="chartrow"')))
        if not args.check:
            io.open(page, "w", encoding="utf-8", newline="\n").write(out)


if __name__ == "__main__":
    main()
