"""
Generate DATA_SOURCES.md and/or DATA_SOURCES.html for an EPM deployment.

Usage:
    python pre-analysis/catalog/generate_docs.py --deployment data_blacksea
    python pre-analysis/catalog/generate_docs.py --deployment data_blacksea --format html
    python pre-analysis/catalog/generate_docs.py --deployment data_blacksea --format both

Each document is written twice: next to the data it describes, and into
pre-analysis/output_catalog/<deployment>/, which is the copy that can be opened
from GitHub. See SHARE_ROOT below.
"""
import argparse
import csv
import re
import yaml
from pathlib import Path
from datetime import date

CATALOG_DIR = Path(__file__).parent / "sources"
PARAMS_FILE = Path(__file__).parent / "epm_parameters.yaml"
REPO_ROOT = Path(__file__).parent.parent.parent

# Where a copy goes so the document has a URL.
#
# epm/input/* is kept out of git (.gitignore), because a deployment folder lives in the R2
# store through DVC, so the copy written next to the data can never be opened from GitHub:
# raw.githubusercontent answers 404 and htmlpreview shows an empty page. This second copy
# sits with the other generated outputs of pre-analysis, which are tracked, and is the one
# to link to. It is not "pre-analysis/catalog/output", which .gitignore drops along with
# every other pre-analysis/**/output.
SHARE_ROOT = REPO_ROOT / "pre-analysis" / "output_catalog"

CONFIDENCE_LABEL = {"high": "[HIGH]", "medium": "[MEDIUM]", "low": "[LOW]"}

# GECO boxes. A geco block under a country and parameter entry of provenance.yaml says that the
# CESI GEC feasibility study departs from our data there. It holds no figure: the CESI aligned file,
# the report reference, the scope and the ids of cesi/cesi_register.yaml (DVC only). Element names
# and main scenario status are read from the register, so the grey box shrinks as the main scenario
# takes CESI values, and disappears once it has taken them all. The GLOBAL_KEY section holds the
# model-wide blocks (pSettings), boxed in every column. validate.py checks the blocks against the files.
# A global block that documents a source (pPlanningReserveMargin, pCarbonPrice) fills every column
# that has no country block of its own, marked "model-wide", and gets its own detail section.
GECO_SOURCE = "cesi_gec_feasibility"
GLOBAL_KEY = "global"
GLOBAL_LABEL = "Model-wide"


def section_label(key):
    return GLOBAL_LABEL if key == GLOBAL_KEY else key


def sections(countries, provenance):
    """Country sections, then the model-wide section when it holds at least one block."""
    g = provenance.get(GLOBAL_KEY)
    return countries + ([GLOBAL_KEY] if isinstance(g, dict) and g else [])


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_catalog():
    sources = {}
    for f in CATALOG_DIR.glob("*.yaml"):
        with open(f, encoding="utf-8") as fh:
            entry = yaml.safe_load(fh)
            sources[entry["id"]] = entry
    return sources


def load_provenance(deployment):
    prov_path = REPO_ROOT / "epm" / "input" / deployment / "provenance.yaml"
    if not prov_path.exists():
        raise FileNotFoundError(f"provenance.yaml not found at {prov_path}")
    with open(prov_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_zcmap(deployment):
    zcmap_path = REPO_ROOT / "epm" / "input" / deployment / "zcmap.csv"
    seen = []
    if zcmap_path.exists():
        with open(zcmap_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                c = row.get("c", "").strip()
                if c and c not in seen:
                    seen.append(c)
    return seen


def load_horizon(deployment):
    """Return (min_year, max_year, step_str) from pDemandForecast year columns."""
    path = REPO_ROOT / "epm" / "input" / deployment / "load" / "pDemandForecast.csv"
    if not path.exists():
        return None, None, None
    with open(path, encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    years = []
    for col in header:
        try:
            years.append(int(col))
        except ValueError:
            pass
    if not years:
        return None, None, None
    years.sort()
    steps = [years[i+1] - years[i] for i in range(len(years)-1)]
    step = max(set(steps), key=steps.count) if steps else 1
    step_str = f"{step} year" + ("s" if step > 1 else "")
    return years[0], years[-1], step_str


def load_params():
    if not PARAMS_FILE.exists():
        return []
    with open(PARAMS_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or []


def load_register(deployment):
    """CESI register entries by id, or {} when the file (DVC only) is absent."""
    path = REPO_ROOT / "epm" / "input" / deployment / "cesi" / "cesi_register.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return {e["id"]: e for e in (yaml.safe_load(fh) or {}).get("entries", [])}


def model_name(deployment, provenance):
    """Derive a human-readable model name."""
    if isinstance(provenance, dict) and "model_name" in provenance:
        return provenance["model_name"]
    name = re.sub(r"^data_", "", deployment).replace("_", " ").title()
    return name


# ── Helpers ────────────────────────────────────────────────────────────────────

def source_short(source_id, catalog):
    if source_id not in catalog:
        return source_id
    s = catalog[source_id]
    name = s.get("name", source_id)
    d = s.get("date", "")
    short = re.split(r" [—–\-] ", name)[0].strip()
    if len(short) > 32:
        short = short[:30] + "…"
    return f"{short} ({d})" if d else short


def source_url(source_id, catalog):
    """Extract a browseable URL from a catalog entry."""
    s = catalog.get(source_id, {})
    url = s.get("url", "")
    if not url:
        note = s.get("access_note", "")
        m = re.search(r'https?://\S+', str(note))
        if m:
            url = m.group(0).rstrip(".,)")
    return url


def source_access(source_id, catalog):
    """Concise 'which file / how accessed' string from a source's access_note."""
    s = catalog.get(source_id, {})
    note = (s.get("access_note") or "").strip()
    if not note:
        return ""
    m = re.search(r'Files?:\s*([^\n]+)', note)          # explicit "File: ..." wins
    if m:
        val = re.split(r'\s+(?:Sheets?:|—)', m.group(1).strip())[0].strip()
        return (val[:160].rstrip() + "…") if len(val) > 160 else val.rstrip(".")
    first = note.split("\n", 1)[0].strip()               # else first line, truncated
    return (first[:200].rstrip() + "…") if len(first) > 200 else first


def sources_display_md(info, catalog):
    """Format primary + secondary sources for MD table cells."""
    proxy_of = info.get("proxy_of", "")
    if proxy_of:
        return f"proxy of {proxy_of}"
    src_id = info.get("source_id")
    secondary = info.get("secondary_source_ids", [])
    parts = []
    if src_id:
        parts.append(source_short(src_id, catalog) + lock(src_id))
    for sid in secondary:
        s = catalog.get(sid, {})
        name = s.get("name", sid)
        short = re.split(r" [—–\-] ", name)[0].strip()
        url = source_url(sid, catalog)
        parts.append((f"[{short}]({url})" if url else short) + lock(sid))
    return " + ".join(parts) if parts else "documented"


def sources_display_html(info, catalog):
    """Format primary + secondary sources for HTML table cells."""
    proxy_of = info.get("proxy_of", "")
    if proxy_of:
        return h(f"proxy of {proxy_of}")
    src_id = info.get("source_id")
    secondary = info.get("secondary_source_ids", [])
    parts = []
    if src_id:
        parts.append(h(source_short(src_id, catalog)) + lock(src_id))
    for sid in secondary:
        s = catalog.get(sid, {})
        name = s.get("name", sid)
        short = re.split(r" [—–\-] ", name)[0].strip()
        url = source_url(sid, catalog)
        if url:
            parts.append(f'<a href="{h(url)}" target="_blank">{h(short)}</a>' + lock(sid))
        else:
            parts.append(h(short) + lock(sid))
    return " + ".join(parts) if parts else "documented"


def render_secondary_sources_md(info, catalog):
    """Render 'Also uses' block for MD detail sections when secondary sources are present."""
    secondary = info.get("secondary_source_ids", [])
    if not secondary:
        return ""
    lines = []
    for sid in secondary:
        s = catalog.get(sid, {})
        name = s.get("name", sid)
        url = source_url(sid, catalog)
        label = f"[{name}]({url})" if url else f"{name} (`{sid}`)"
        lines.append(f"**Also uses**: {label}\n")
    return "\n".join(lines)


def render_secondary_sources_html(info, catalog):
    """Render 'Also uses' block for HTML detail sections when secondary sources are present."""
    secondary = info.get("secondary_source_ids", [])
    if not secondary:
        return ""
    parts = []
    for sid in secondary:
        s = catalog.get(sid, {})
        name = s.get("name", sid)
        url = source_url(sid, catalog)
        if url:
            parts.append(f'<a href="{h(url)}" target="_blank">{h(name)}</a>')
        else:
            parts.append(f'{h(name)} <code>({h(sid)})</code>')
    return f'<p><strong>Also uses</strong>: {", ".join(parts)}</p>'


def get_info(country, param_id, provenance):
    c = provenance.get(country)
    if not isinstance(c, dict):
        return None
    return c.get(param_id)


def anchor(country, resource=None):
    base = re.sub(r"[^a-z0-9]+", "-", country.lower()).strip("-")
    if resource:
        res = re.sub(r"[^a-z0-9]+", "-", resource.lower()).strip("-")
        return f"{base}-{res}"
    return base


def h(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md_cell(text):
    """A markdown table cell survives neither a newline nor a pipe: keep both visible."""
    return str(text).strip().replace("|", "&#124;").replace(chr(10), "<br>")


def render_proxy_note_md(info, catalog):
    """Return a proxy chain line for MD if proxy_of is set."""
    proxy_of = info.get("proxy_of", "")
    if not proxy_of:
        return ""
    src_id = info.get("source_id", "")
    src_name = catalog.get(src_id, {}).get("name", src_id) if src_id else ""
    return f"**Proxied from**: {proxy_of}  \n**Original source**: {src_name}\n"


def render_proxy_note_html(info, catalog):
    """Return HTML proxy chain paragraph if proxy_of is set."""
    proxy_of = info.get("proxy_of", "")
    if not proxy_of:
        return ""
    src_id = info.get("source_id", "")
    src_name = catalog.get(src_id, {}).get("name", src_id) if src_id else ""
    return (
        f'<p><strong>Proxied from</strong>: <code>{h(proxy_of)}</code>'
        f'<br><strong>Original source</strong>: {h(src_name)}'
        f' <code>({h(src_id)})</code></p>'
    )


# ── GECO boxes ────────────────────────────────────────────────────────────────

def lock(source_id):
    """Confidential source whose values the document does not reproduce."""
    return " 🔒" if source_id == GECO_SOURCE else ""


def documented(info):
    """True when the entry documents a source. An entry holding only a geco block does not."""
    return isinstance(info, dict) and any(k != "geco" for k in info)


def geco_files(g):
    f = g.get("file") or []
    return [f] if isinstance(f, str) else list(f)


def geco_open(info, register):
    """[(element, statuses)] of the GECO items the main scenario has not taken. Empty: no box."""
    g = info.get("geco") if isinstance(info, dict) else None
    if not isinstance(g, dict):
        return []
    items = {}
    for rid in g.get("ids") or []:
        e = register.get(rid)
        if e is None:
            items.setdefault(rid, set()).add("n/a")
        elif e.get("main") != "taken":
            items.setdefault(e.get("element", rid), set()).add(e.get("main") or "pending")
    return [(el, sorted(st)) for el, st in items.items()]


def geco_status(items):
    """Main scenario status of the open items, e.g. "pending (4)" or "kept (1), pending (2)"."""
    n = {}
    for _, st in items:
        for s in st:
            n[s] = n.get(s, 0) + 1
    return ", ".join(f"{s} ({c})" for s, c in sorted(n.items()))


def geco_box_md(info, register):
    items = geco_open(info, register)
    if not items:
        return ""
    g = info["geco"]
    files = ", ".join(f"`{f}`" for f in geco_files(g)) or "n/a"
    return (f"> **GECO ≠** {', '.join(el for el, _ in items)}. {g.get('ref', 'n/a')}. "
            f"Scope: {g.get('scope', 'n/a').rstrip('.')}. CESI aligned file: {files}. "
            f"Main scenario: {geco_status(items)}. Values not reproduced: confidential source.\n")


def geco_box_html(info, register):
    items = geco_open(info, register)
    if not items:
        return ""
    g = info["geco"]
    files = ", ".join(f"<code>{h(f)}</code>" for f in geco_files(g)) or "n/a"
    return (f'<div class="geco-detail"><strong>GECO &ne;</strong> {h(", ".join(el for el, _ in items))}. '
            f'{h(g.get("ref", "n/a"))}. Scope: {h(g.get("scope", "n/a").rstrip("."))}.<br>'
            f'CESI aligned file: {files} &middot; Main scenario: {h(geco_status(items))} &middot; '
            f'values not reproduced, confidential source.</div>')


def geco_cell_md(country, pid, provenance, register):
    """Matrix cell mark: the country block, then the model-wide one."""
    names = [el for key in (country, GLOBAL_KEY)
             for el, _ in geco_open(get_info(key, pid, provenance), register)]
    return f"GECO ≠ {', '.join(names)}" if names else ""


def geco_cell_html(country, pid, provenance, register):
    out = []
    for key in (country, GLOBAL_KEY):
        info = get_info(key, pid, provenance)
        items = geco_open(info, register)
        if not items:
            continue
        g = info["geco"]
        label = f'GECO &ne; {h(", ".join(el for el, _ in items))}'
        tip = h(f'{g.get("ref", "n/a")}. {g.get("scope", "n/a").rstrip(".")}. Main scenario: {geco_status(items)}').replace('"', "&quot;")
        if key == country and documented(info):
            out.append(f'<a class="geco-box" href="#{anchor(country, pid)}" title="{tip}">{label}</a>')
        else:
            out.append(f'<span class="geco-box" title="{tip}">{label}</span>')
    return "".join(out)


# ── Markdown ──────────────────────────────────────────────────────────────────

def render_md(deployment, countries, horizon, params, provenance, catalog, register):
    lines = []
    today = date.today()
    mname = model_name(deployment, provenance)
    yr_min, yr_max, step_str = horizon

    lines += [
        f"# Data Sources — EPM — {mname}\n",
        f"*Generated {today}*\n",
        "---\n",
    ]

    # Model overview
    lines += ["## Model overview\n"]
    lines.append(f"**Countries**: {', '.join(countries)}  ")
    if yr_min:
        lines.append(f"**Data horizon**: {yr_min}–{yr_max} · step: {step_str}\n")
    lines.append("")

    # Overview matrix
    header = "| Category | Item | Parameter | Description | " + " | ".join(countries) + " |"
    sep = "|---|---|---|---|" + "---|" * len(countries)
    lines += [header, sep]

    for p in params:
        cat = p.get("category", "")
        pid = p["id"]
        item = p.get("item", pid).replace("|", "\\|")
        desc = p.get("description", "").replace("|", "\\|")
        cells = []
        for country in countries:
            info = get_info(country, pid, provenance)
            ginfo = get_info(GLOBAL_KEY, pid, provenance)
            if documented(info):
                cell = sources_display_md(info, catalog)
                if info.get("needs_review"):
                    cell = f"⚠ {cell}"
            elif documented(ginfo):
                cell = f"{sources_display_md(ginfo, catalog)} (model-wide)"
                if ginfo.get("needs_review"):
                    cell = f"⚠ {cell}"
            else:
                cell = "—"
            box = geco_cell_md(country, pid, provenance, register)
            cells.append(f"{cell}<br>{box}" if box else cell)
        lines.append(f"| {cat} | {item} | `{pid}` | {desc} | " + " | ".join(cells) + " |")

    lines += ["", "---\n"]

    # TOC
    lines += ['<a id="toc"></a>\n', "## Contents\n"]
    for country in sections(countries, provenance):
        cid = anchor(country)
        cdata = provenance.get(country, {})
        if isinstance(cdata, dict) and cdata:
            param_links = " · ".join(
                f"[`{pid}`](#{anchor(country, pid)})"
                for pid in cdata
                if isinstance(cdata[pid], dict)
            )
            lines.append(f"- [{section_label(country)}](#{cid}) — {param_links}")
        else:
            lines.append(f"- [{section_label(country)}](#{cid}) — *not yet documented*")
    lines += ["", "---\n"]

    # Country sections, then the model-wide one
    for country in sections(countries, provenance):
        cid = anchor(country)
        lines += [
            f'<a id="{cid}"></a>\n',
            f"## {section_label(country)}\n",
            "[&#8593; Contents](#toc)\n",
        ]
        cdata = provenance.get(country, {})
        if not isinstance(cdata, dict) or not cdata:
            lines += ["*No data documented yet for this country.*\n", "---\n"]
            continue

        # Recap table
        lines += ["### Summary\n", "| Parameter | Source | Confidence |", "|---|---|---|"]
        for p in params:
            pid = p["id"]
            info = cdata.get(pid)
            if not documented(info):
                continue
            src_display = sources_display_md(info, catalog)
            conf = info.get("confidence", "")
            conf_label = CONFIDENCE_LABEL.get(conf, conf.upper()) if conf else "—"
            if info.get("needs_review"):
                conf_label += " ⚠"
            lines.append(f"| [`{pid}`](#{anchor(country, pid)}) | {src_display} | {conf_label} |")
        lines.append("")

        # Per-parameter detail
        for pid, info in cdata.items():
            if not isinstance(info, dict):
                continue
            rid = anchor(country, pid)
            lines += [
                f'<a id="{rid}"></a>\n',
                f"### `{pid}`\n",
                f"[&#8593; {section_label(country)}](#{cid})\n",
            ]

            proxy_note = render_proxy_note_md(info, catalog)
            if proxy_note:
                lines.append(proxy_note)
            else:
                src_id = info.get("source_id") or (info.get("source_ids") or [None])[0]
                if src_id:
                    s = catalog.get(src_id, {})
                    lines.append(f"**Source**: {s.get('name', src_id)} (`{src_id}`)\n")
                    access = source_access(src_id, catalog)
                    if access:
                        lines.append(f"**Data / file**: {access}\n")
                secondary_note = render_secondary_sources_md(info, catalog)
                if secondary_note:
                    lines.append(secondary_note)

            box = geco_box_md(info, register)
            if box:
                lines.append(box)

            if info.get("needs_review"):
                review_note = info.get("review_note", "Further data collection needed")
                lines.append(f"> ⚠ **Needs review**: {review_note}\n")

            method = info.get("method", "")
            if method:
                lines.append(f"**Method**: {method}\n")

            if "method_table" in info:
                lines += ["| Period | Method | Notes |", "|--------|--------|-------|"]
                for row in info["method_table"]:
                    lines.append(
                        f"| {row.get('period', '')} "
                        f"| `{row.get('method', '')}` "
                        f"| {md_cell(row.get('notes', ''))} |"
                    )
                lines.append("")

            notes = info.get("notes", "")
            if notes:
                lines.append(f"> {notes.strip()}\n")

            meta = []
            if conf := info.get("confidence", ""):
                meta.append(f"Confidence: {CONFIDENCE_LABEL.get(conf, conf.upper())}")
            if last := info.get("last_updated", ""):
                meta.append(f"Last updated: {last}")
            if meta:
                lines.append(f"*{' · '.join(meta)}*\n")

            lines.append("")

        lines.append("---\n")

    return "\n".join(lines)


# ── HTML ──────────────────────────────────────────────────────────────────────

_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  max-width: 1280px; margin: 0 auto; padding: 24px 32px;
  color: #2c3e50; line-height: 1.55; font-size: 14px;
}
h1 { border-bottom: 3px solid #1a5276; padding-bottom: 10px; margin-bottom: 4px; font-size: 1.7em; }
h2 { border-bottom: 1px solid #d5d8dc; margin-top: 48px; padding-bottom: 6px;
     color: #1a5276; font-size: 1.25em; }
h3 { margin-top: 28px; font-size: 1.05em; }
p.meta { color: #888; font-size: 0.82em; margin-top: 4px; margin-bottom: 32px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 0.87em; }
th { background: #2c3e50; color: #fff; padding: 9px 12px; text-align: left;
     font-weight: 600; white-space: nowrap; }
td { padding: 7px 12px; border-bottom: 1px solid #eaecee; vertical-align: top; }
tr:hover td { background: #f8f9fa; }
.cat-row td { background: #eaecee; font-weight: 700; color: #444;
              font-size: 0.78em; letter-spacing: 0.08em; text-transform: uppercase;
              padding: 5px 12px; }
.status-done { background: #edf7ed; }
.status-done a { color: #276327; text-decoration: none; }
.status-done a:hover { text-decoration: underline; }
.status-pending { color: #bbb; }
code { background: #f2f3f4; padding: 1px 5px; border-radius: 3px; font-family: monospace; font-size: 0.9em; }
blockquote { border-left: 3px solid #d5d8dc; margin: 10px 0; padding: 6px 14px;
             color: #666; font-style: italic; }
.back { font-size: 0.78em; color: #aaa; margin: 2px 0 10px; }
.back a { color: #aaa; text-decoration: none; }
.back a:hover { text-decoration: underline; }
.conf { font-size: 0.72em; color: #999; font-weight: normal; margin-left: 6px; }
.toc { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 4px;
       padding: 14px 22px; display: inline-block; min-width: 280px; }
.toc ul { margin: 4px 0; padding-left: 18px; }
.toc li { margin: 4px 0; line-height: 1.4; }
.toc a { color: #1a5276; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
hr { border: none; border-top: 1px solid #e8e8e8; margin: 36px 0; }
.legend { display: flex; gap: 24px; margin: 4px 0 24px; flex-wrap: wrap; }
.legend span { font-size: 0.82em; color: #666; }
.legend .dot-done { color: #276327; font-weight: 700; }
.legend .dot-pending { color: #bbb; font-weight: 700; }
.legend .dot-review { color: #b85c00; font-weight: 700; }
.status-review { background: #fff4e5; }
.status-review a { color: #b85c00; text-decoration: none; }
.status-review a:hover { text-decoration: underline; }
.review-box { background: #fff4e5; border-left: 3px solid #e07b00;
              padding: 8px 14px; margin: 10px 0; font-size: 0.88em; color: #7a3d00; }
.proxy-chain { background: #fafaf2; border-left: 3px solid #e0c840;
               padding: 6px 12px; margin: 8px 0; font-size: 0.88em; }
a.geco-box, span.geco-box { display: block; width: fit-content; margin-top: 4px; padding: 1px 6px;
               border: 1px solid #d6d6d6; border-radius: 3px; background: #f1f1f1; color: #555;
               font-size: 0.82em; line-height: 1.35; text-decoration: none; }
.geco-detail { background: #f3f3f3; border-left: 3px solid #b9b9b9; padding: 6px 12px;
               margin: 8px 0; font-size: 0.88em; color: #444; }
"""


def render_html(deployment, countries, horizon, params, provenance, catalog, register):
    today = date.today()
    mname = model_name(deployment, provenance)
    yr_min, yr_max, step_str = horizon
    country_list = ", ".join(countries)
    horizon_str = f"{yr_min}–{yr_max} · step: {step_str}" if yr_min else "—"
    n_base = 3  # Item | Parameter | Description

    out = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Sources — EPM — {h(mname)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Data Sources — EPM — {h(mname)}</h1>
<p class="meta">
  Generated {today} &nbsp;&middot;&nbsp;
  {h(country_list)} &nbsp;&middot;&nbsp;
  Data horizon: {h(horizon_str)}
</p>
"""]

    out.append('<div class="legend">')
    out.append('<span><span class="dot-done">&#9632;</span> Documented</span>')
    out.append('<span><span class="dot-review">&#9651;</span> Needs review (further data collection needed)</span>')
    out.append('<span><span class="dot-pending">&mdash;</span> Not yet documented</span>')
    out.append('</div>\n')

    # ── Overview table ────────────────────────────────────────────────────────
    out.append('<h2 id="overview">Model overview</h2>\n<table>')
    out.append('<thead><tr><th>Item</th><th>Parameter</th><th>Description</th>')
    for c in countries:
        out.append(f'<th>{h(c)}</th>')
    out.append('</tr></thead>\n<tbody>')

    current_cat = None
    for p in params:
        cat = p.get("category", "")
        pid = p["id"]
        item = p.get("item", pid)
        desc = p.get("description", "")

        if cat != current_cat:
            n_cols = n_base + len(countries)
            out.append(f'<tr class="cat-row"><td colspan="{n_cols}">{h(cat)}</td></tr>')
            current_cat = cat

        out.append('<tr>')
        out.append(f'<td>{h(item)}</td>')
        out.append(f'<td><code>{h(pid)}</code></td>')
        out.append(f'<td>{h(desc)}</td>')
        for country in countries:
            info = get_info(country, pid, provenance)
            box = geco_cell_html(country, pid, provenance, register)
            ginfo = get_info(GLOBAL_KEY, pid, provenance)
            if documented(info) or documented(ginfo):
                if not documented(info):
                    info, link_key, suffix = ginfo, GLOBAL_KEY, " (model-wide)"
                else:
                    link_key, suffix = country, ""
                cell_html = sources_display_html(info, catalog) + suffix
                link = anchor(link_key, pid)
                needs_review = info.get("needs_review", False)
                if needs_review:
                    out.append(f'<td class="status-review"><a href="#{link}">&#9651; {cell_html}</a>{box}</td>')
                else:
                    out.append(f'<td class="status-done"><a href="#{link}">{cell_html}</a>{box}</td>')
            else:
                out.append(f'<td class="status-pending">&mdash;{box}</td>')
        out.append('</tr>')

    out.append('</tbody></table>\n<hr>')

    # ── TOC ───────────────────────────────────────────────────────────────────
    out.append('<h2 id="toc">Contents</h2>\n<div class="toc"><ul>')
    for country in sections(countries, provenance):
        cid = anchor(country)
        cdata = provenance.get(country, {})
        out.append(f'<li><a href="#{cid}"><strong>{h(section_label(country))}</strong></a>')
        if isinstance(cdata, dict) and cdata:
            out.append(' &mdash; ')
            links = ', '.join(
                f'<a href="#{anchor(country, pid)}"><code>{h(pid)}</code></a>'
                for pid in cdata
                if isinstance(cdata[pid], dict)
            )
            out.append(links)
        else:
            out.append(' <em>not yet documented</em>')
        out.append('</li>')
    out.append('</ul></div>\n<hr>')

    # ── Country sections, then the model-wide one ─────────────────────────────
    for country in sections(countries, provenance):
        cid = anchor(country)
        out.append(f'<h2 id="{cid}">{h(section_label(country))}</h2>')
        out.append('<p class="back"><a href="#toc">&#8593; Contents</a></p>')

        cdata = provenance.get(country, {})
        if not isinstance(cdata, dict) or not cdata:
            out.append('<p><em>No data documented yet for this country.</em></p><hr>')
            continue

        # Recap table
        out.append('<h3>Summary</h3>')
        out.append('<table><thead><tr><th>Parameter</th><th>Source</th><th>Confidence</th></tr></thead><tbody>')
        for p in params:
            pid = p["id"]
            info = cdata.get(pid)
            if not documented(info):
                continue
            src_display = sources_display_html(info, catalog)
            conf = info.get("confidence", "")
            conf_html = f'<span class="conf">[{h(conf.upper())}]</span>' if conf else ""
            needs_review = info.get("needs_review", False)
            review_html = ' <span class="conf" style="color:#b85c00">[&#9651; REVIEW]</span>' if needs_review else ""
            link = anchor(country, pid)
            out.append(
                f'<tr><td><a href="#{link}"><code>{h(pid)}</code></a></td>'
                f'<td>{src_display}</td><td>{conf_html}{review_html}</td></tr>'
            )
        out.append('</tbody></table>')

        # Per-parameter detail
        for pid, info in cdata.items():
            if not isinstance(info, dict):
                continue
            rid = anchor(country, pid)
            out.append(f'<h3 id="{rid}"><code>{h(pid)}</code></h3>')
            out.append(f'<p class="back"><a href="#{cid}">&#8593; {h(section_label(country))}</a></p>')

            proxy_note = render_proxy_note_html(info, catalog)
            if proxy_note:
                out.append(f'<div class="proxy-chain">{proxy_note}</div>')
            else:
                src_id = info.get("source_id") or (info.get("source_ids") or [None])[0]
                if src_id:
                    s = catalog.get(src_id, {})
                    out.append(
                        f'<p><strong>Source</strong>: {h(s.get("name", src_id))} '
                        f'<code>({h(src_id)})</code></p>'
                    )
                    access = source_access(src_id, catalog)
                    if access:
                        out.append(f'<p><strong>Data / file</strong>: {h(access)}</p>')
                secondary_note = render_secondary_sources_html(info, catalog)
                if secondary_note:
                    out.append(secondary_note)

            box = geco_box_html(info, register)
            if box:
                out.append(box)

            if info.get("needs_review"):
                review_note = info.get("review_note", "Further data collection needed")
                out.append(f'<div class="review-box">&#9651; <strong>Needs review</strong>: {h(review_note)}</div>')

            method = info.get("method", "")
            if method:
                out.append(f'<p><strong>Method</strong>: {h(method)}</p>')

            if "method_table" in info:
                out.append('<table><thead><tr><th>Period</th><th>Method</th><th>Notes</th></tr></thead><tbody>')
                for row in info["method_table"]:
                    out.append(
                        f'<tr><td>{h(row.get("period", ""))}</td>'
                        f'<td><code>{h(row.get("method", ""))}</code></td>'
                        f'<td>{h(row.get("notes", "")).replace(chr(10), "<br>")}</td></tr>'
                    )
                out.append('</tbody></table>')

            notes = info.get("notes", "")
            if notes:
                out.append(f'<blockquote>{h(notes.strip())}</blockquote>')

            meta = []
            if conf := info.get("confidence", ""):
                meta.append(f'Confidence: {CONFIDENCE_LABEL.get(conf, conf.upper())}')
            if last := info.get("last_updated", ""):
                meta.append(f'Last updated: {last}')
            if meta:
                out.append(f'<p class="back">{h(" · ".join(meta))}</p>')

        out.append('<hr>')

    out.append('</body>\n</html>')
    return '\n'.join(out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate DATA_SOURCES docs for an EPM deployment"
    )
    parser.add_argument("--deployment", required=True, help="e.g. data_blacksea")
    parser.add_argument("--format", choices=["md", "html", "both"], default="both")
    args = parser.parse_args()

    catalog = load_catalog()
    provenance = load_provenance(args.deployment)
    # Country list = zcmap (active) + provenance keys not yet in zcmap (in-progress)
    countries = load_zcmap(args.deployment)
    for c in provenance:
        if c and c not in ("model_name", GLOBAL_KEY) and isinstance(provenance[c], dict) and c not in countries:
            countries.append(c)
    horizon = load_horizon(args.deployment)
    params = load_params()
    register = load_register(args.deployment)

    base = REPO_ROOT / "epm" / "input" / args.deployment
    share = SHARE_ROOT / args.deployment
    share.mkdir(parents=True, exist_ok=True)

    def emit(name, content):
        for out in (base / name, share / name):
            out.write_text(content, encoding="utf-8")
            print(f"Written: {out}")

    if args.format in ("md", "both"):
        emit("DATA_SOURCES.md",
             render_md(args.deployment, countries, horizon, params, provenance, catalog, register))

    if args.format in ("html", "both"):
        emit("DATA_SOURCES.html",
             render_html(args.deployment, countries, horizon, params, provenance, catalog, register))


if __name__ == "__main__":
    main()
