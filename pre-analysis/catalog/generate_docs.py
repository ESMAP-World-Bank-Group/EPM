"""
Generate DATA_SOURCES.md and/or DATA_SOURCES.html for an EPM deployment.

Usage:
    python pre-analysis/catalog/generate_docs.py --deployment data_blacksea
    python pre-analysis/catalog/generate_docs.py --deployment data_blacksea --format html
    python pre-analysis/catalog/generate_docs.py --deployment data_blacksea --format both
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

CONFIDENCE_LABEL = {"high": "[HIGH]", "medium": "[MEDIUM]", "low": "[LOW]"}


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


def sources_display_md(info, catalog):
    """Format primary + secondary sources for MD table cells."""
    proxy_of = info.get("proxy_of", "")
    if proxy_of:
        return f"proxy of {proxy_of}"
    src_id = info.get("source_id")
    secondary = info.get("secondary_source_ids", [])
    parts = []
    if src_id:
        parts.append(source_short(src_id, catalog))
    for sid in secondary:
        s = catalog.get(sid, {})
        name = s.get("name", sid)
        short = re.split(r" [—–\-] ", name)[0].strip()
        url = source_url(sid, catalog)
        parts.append(f"[{short}]({url})" if url else short)
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
        parts.append(h(source_short(src_id, catalog)))
    for sid in secondary:
        s = catalog.get(sid, {})
        name = s.get("name", sid)
        short = re.split(r" [—–\-] ", name)[0].strip()
        url = source_url(sid, catalog)
        if url:
            parts.append(f'<a href="{h(url)}" target="_blank">{h(short)}</a>')
        else:
            parts.append(h(short))
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


# ── Markdown ──────────────────────────────────────────────────────────────────

def render_md(deployment, countries, horizon, params, provenance, catalog):
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
            if info:
                cells.append(sources_display_md(info, catalog))
            else:
                cells.append("—")
        lines.append(f"| {cat} | {item} | `{pid}` | {desc} | " + " | ".join(cells) + " |")

    lines += ["", "---\n"]

    # TOC
    lines += ['<a id="toc"></a>\n', "## Contents\n"]
    for country in countries:
        cid = anchor(country)
        cdata = provenance.get(country, {})
        if isinstance(cdata, dict) and cdata:
            param_links = " · ".join(
                f"[`{pid}`](#{anchor(country, pid)})"
                for pid in cdata
                if isinstance(cdata[pid], dict)
            )
            lines.append(f"- [{country}](#{cid}) — {param_links}")
        else:
            lines.append(f"- [{country}](#{cid}) — *not yet documented*")
    lines += ["", "---\n"]

    # Country sections
    for country in countries:
        cid = anchor(country)
        lines += [
            f'<a id="{cid}"></a>\n',
            f"## {country}\n",
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
            if not info:
                continue
            src_display = sources_display_md(info, catalog)
            conf = info.get("confidence", "")
            conf_label = CONFIDENCE_LABEL.get(conf, conf.upper()) if conf else "—"
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
                f"[&#8593; {country}](#{cid})\n",
            ]

            proxy_note = render_proxy_note_md(info, catalog)
            if proxy_note:
                lines.append(proxy_note)
            else:
                src_id = info.get("source_id") or (info.get("source_ids") or [None])[0]
                if src_id:
                    s = catalog.get(src_id, {})
                    lines.append(f"**Source**: {s.get('name', src_id)} (`{src_id}`)\n")
                secondary_note = render_secondary_sources_md(info, catalog)
                if secondary_note:
                    lines.append(secondary_note)

            method = info.get("method", "")
            if method:
                lines.append(f"**Method**: {method}\n")

            if "method_table" in info:
                lines += ["| Period | Method | Notes |", "|--------|--------|-------|"]
                for row in info["method_table"]:
                    lines.append(
                        f"| {row.get('period', '')} "
                        f"| `{row.get('method', '')}` "
                        f"| {row.get('notes', '')} |"
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
.legend { display: flex; gap: 24px; margin: 4px 0 24px; }
.legend span { font-size: 0.82em; color: #666; }
.legend .dot-done { color: #276327; font-weight: 700; }
.legend .dot-pending { color: #bbb; font-weight: 700; }
.proxy-chain { background: #fafaf2; border-left: 3px solid #e0c840;
               padding: 6px 12px; margin: 8px 0; font-size: 0.88em; }
"""


def render_html(deployment, countries, horizon, params, provenance, catalog):
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
            if info:
                cell_html = sources_display_html(info, catalog)
                link = anchor(country, pid)
                out.append(f'<td class="status-done"><a href="#{link}">{cell_html}</a></td>')
            else:
                out.append('<td class="status-pending">&mdash;</td>')
        out.append('</tr>')

    out.append('</tbody></table>\n<hr>')

    # ── TOC ───────────────────────────────────────────────────────────────────
    out.append('<h2 id="toc">Contents</h2>\n<div class="toc"><ul>')
    for country in countries:
        cid = anchor(country)
        cdata = provenance.get(country, {})
        out.append(f'<li><a href="#{cid}"><strong>{h(country)}</strong></a>')
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

    # ── Country sections ──────────────────────────────────────────────────────
    for country in countries:
        cid = anchor(country)
        out.append(f'<h2 id="{cid}">{h(country)}</h2>')
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
            if not info:
                continue
            src_display = sources_display_html(info, catalog)
            conf = info.get("confidence", "")
            conf_html = f'<span class="conf">[{h(conf.upper())}]</span>' if conf else ""
            link = anchor(country, pid)
            out.append(
                f'<tr><td><a href="#{link}"><code>{h(pid)}</code></a></td>'
                f'<td>{src_display}</td><td>{conf_html}</td></tr>'
            )
        out.append('</tbody></table>')

        # Per-parameter detail
        for pid, info in cdata.items():
            if not isinstance(info, dict):
                continue
            rid = anchor(country, pid)
            out.append(f'<h3 id="{rid}"><code>{h(pid)}</code></h3>')
            out.append(f'<p class="back"><a href="#{cid}">&#8593; {h(country)}</a></p>')

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
                secondary_note = render_secondary_sources_html(info, catalog)
                if secondary_note:
                    out.append(secondary_note)

            method = info.get("method", "")
            if method:
                out.append(f'<p><strong>Method</strong>: {h(method)}</p>')

            if "method_table" in info:
                out.append('<table><thead><tr><th>Period</th><th>Method</th><th>Notes</th></tr></thead><tbody>')
                for row in info["method_table"]:
                    out.append(
                        f'<tr><td>{h(row.get("period", ""))}</td>'
                        f'<td><code>{h(row.get("method", ""))}</code></td>'
                        f'<td>{h(row.get("notes", ""))}</td></tr>'
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
    countries = load_zcmap(args.deployment)
    horizon = load_horizon(args.deployment)
    params = load_params()

    base = REPO_ROOT / "epm" / "input" / args.deployment

    if args.format in ("md", "both"):
        md = render_md(args.deployment, countries, horizon, params, provenance, catalog)
        out = base / "DATA_SOURCES.md"
        out.write_text(md, encoding="utf-8")
        print(f"Written: {out}")

    if args.format in ("html", "both"):
        html_content = render_html(args.deployment, countries, horizon, params, provenance, catalog)
        out = base / "DATA_SOURCES.html"
        out.write_text(html_content, encoding="utf-8")
        print(f"Written: {out}")


if __name__ == "__main__":
    main()
