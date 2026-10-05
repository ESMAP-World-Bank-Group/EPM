"""Build the Black Sea sector briefs: one tab per sector, the same outline in each.

Every sector pane carries six blocks in the same order, so the tabs can be read
side by side and later compared:

    1. How the sector works      collapsible primer with a schematic
    2. Key figures               stat tiles and one mix chart
    3. Background                dated timeline of what shaped the system
    4. The region on the ground  detailed map of the physical system
    5. Structural map            simplified: capacities, zones, interconnections
    6. Projects and corridors    named initiatives, status coded

Panes not yet built render as stubs that name the roadmap step that fills them.

Usage
    python build_brief.py                 # writes beside this script
    python build_brief.py --out <dir>     # writes into another folder as well
"""

from __future__ import annotations

import argparse
import html
import shutil
from datetime import date
from pathlib import Path

import gas
import overview
import power
import transport
import water
import digital
import synergies
import adhoc
from mapkit import BASEMAP, FOCUS, PALETTE, CountryIndex, Map

HERE = Path(__file__).resolve().parent
OUT = HERE / "blacksea_sector_briefs.html"

# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------
TABS = [
    ("energy", "Energy", [("overview", "Overview"), ("power", "Power"), ("gas", "Gas")]),
    ("transport", "Transport", [("overview", "Overview"), ("rail", "Rail and road"),
                                ("ports", "Ports and maritime")]),
    ("digital", "Digital", []),
    ("water", "Water", []),
    ("synergies", "Comparison and synergies", []),
    ("adhoc", "Ad hoc", []),
]

# One entry per sector pane. Titles of blocks 1, 5 and 6 vary by sector, the
# key figure labels preview the tiles each pane will carry, and the step is the
# roadmap step that fills the pane.
PANES = {
    "energy/overview": {
        "primer": "How the energy system fits together",
        "tiles": ["Primary energy supply", "Import dependency",
                  "Energy intensity of GDP", "Share of renewables"],
        "map_b": "Energy flows and dependencies",
        "map_c": "Energy projects and corridors",
        "step": 2,
    },
    "energy/power": {
        "primer": "How a power system works",
        "tiles": ["Installed capacity", "Annual demand", "Peak load",
                  "Cross-border transfer capacity"],
        "map_b": "Capacity, synchronous zones and interconnections",
        "map_c": "Power projects and corridors",
        "step": 1,
    },
    "energy/gas": {
        "primer": "How gas moves from field to consumer",
        "tiles": ["Annual consumption", "Domestic production",
                  "Transit capacity", "Storage capacity"],
        "map_b": "Supply sources, transit and interconnection capacity",
        "map_c": "Gas projects and corridors",
        "step": 2,
    },
    "transport/overview": {
        "primer": "How freight moves across the region",
        "tiles": ["Freight carried", "Transit volume, Middle Corridor",
                  "Main line rail network", "Container throughput"],
        "map_b": "Corridors, modes and border crossings",
        "map_c": "Transport projects and corridors",
        "step": 3,
    },
    "transport/rail": {
        "primer": "How rail and road freight work",
        "tiles": ["Rail network length", "Electrified share",
                  "Rail freight", "Border crossing times"],
        "map_b": "Gauge breaks, electrification and border crossings",
        "map_c": "Rail and road projects",
        "step": 3,
    },
    "transport/ports": {
        "primer": "How ports and short sea shipping work",
        "tiles": ["Port throughput", "Container throughput",
                  "Ferry and Ro-Ro links", "Caspian crossing volume"],
        "map_b": "Port capacity and ferry links",
        "map_c": "Port and maritime projects",
        "step": 3,
    },
    "digital": {
        "primer": "How data moves between countries",
        "tiles": ["Internet users", "Fixed broadband per 100 people",
                  "Submarine cables landing", "Networks at exchanges"],
        "map_b": "Where networks exchange traffic",
        "map_c": "Digital projects and corridors",
        "step": 5,
    },
    "water": {
        "primer": "How water is shared and used",
        "tiles": ["Renewable water resources", "Share from outside the country",
                  "Reservoir storage", "Irrigated area"],
        "map_b": "Basins, storage and transboundary flows",
        "map_c": "Water projects and agreements",
        "step": 4,
    },
}

STUDY = ["Georgia", "Armenia", "Azerbaijan", "Turkiye"]


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------
def stub(text: str, tall: bool = False) -> str:
    cls = "stub tall" if tall else "stub"
    return f'<div class="{cls}">{html.escape(text)}</div>'


def ghost_tiles(labels) -> str:
    tiles = "".join(
        f'<div class="stat ghost"><div class="v">n/a</div>'
        f'<div class="l">{html.escape(label)}</div></div>'
        for label in labels
    )
    return (
        '<div class="headrow">'
        f'<div class="hr-kpi"><div class="stats">{tiles}</div></div>'
        f'<div class="hr-mix">{stub("Mix chart")}</div>'
        "</div>"
    )


# Panes that are built, by key. Each takes the country index and the three
# sector specific block titles.
BUILT = {"energy/overview": overview.pane, "energy/power": power.pane, "energy/gas": gas.pane,
         "transport/overview": transport.overview_pane, "transport/rail": transport.rail_pane,
         "transport/ports": transport.ports_pane, "water": water.pane,
         "digital": digital.pane}


def sector_pane(key: str, index: CountryIndex) -> str:
    spec = PANES[key]
    if key in BUILT:
        return BUILT[key](index, (spec["primer"], spec["map_b"], spec["map_c"]))
    return "".join([
        f'<p class="pending">Not built yet. Roadmap step {spec["step"]}.</p>',
        '<details class="primer">',
        f'<summary>{html.escape(spec["primer"])}</summary>',
        '<div class="primer-body">',
        stub("Schematic of the sector chain, then the operating principles in a few lines."),
        "</div></details>",
        '<h4 class="mh">Key figures</h4>',
        ghost_tiles(spec["tiles"]),
        '<h4 class="mh">Background</h4>',
        stub("Dated timeline of the events that shaped the present system."),
        '<h4 class="mh">The region on the ground</h4>',
        stub("Detailed map of the physical system.", tall=True),
        f'<h4 class="mh">{html.escape(spec["map_b"])}</h4>',
        stub("Simplified structural map.", tall=True),
        f'<h4 class="mh">{html.escape(spec["map_c"])}</h4>',
        stub("Named projects and corridors. Solid in service, dashed under "
             "construction, dotted planned.", tall=True),
    ])


# ---------------------------------------------------------------------------
# scope map
# ---------------------------------------------------------------------------
def scope_map(index: CountryIndex) -> str:
    m = Map(width=560, height=300)
    m.basemap(index)
    places = [
        (43.4, 42.15, "Georgia"), (44.9, 40.25, "Armenia"),
        (48.2, 40.35, "Azerbaijan"), (34.5, 39.0, "Turkiye"),
    ]
    for lon, lat, name in places:
        m.label(lon, lat, name, size=11, colour=PALETTE["ink"], weight="700", halo=True)
    for lon, lat, name in [(34.0, 43.3, "Black Sea"), (50.6, 42.2, "Caspian")]:
        m.label(lon, lat, name, size=10, colour=PALETTE["muted"], weight="400")
    return m.render()


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------
CSS = """
:root{
  --bg:#ffffff; --paper:#ffffff; --ink:#2b2926; --muted:#6f6a61; --faint:#948f84;
  --line:#e6e6ea; --line2:#d9d9df;
  --blue:#256081; --accent:#0277bd; --blue-bg:#eaf1f8; --cream:#f5eeda;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 -apple-system,"Segoe UI",Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 22px}
header.top{padding:30px 0 10px}
h1{font-size:1.7rem;margin:0 0 6px;color:var(--blue);letter-spacing:-.3px}
.sub{color:var(--muted);margin:0 0 14px;font-size:.95rem}
.aidisc{font-size:.72rem;color:var(--faint);margin:0 0 16px}
.scope{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.25fr);gap:22px;
  align-items:center;margin:0 0 10px}
.scope p{margin:0 0 8px;font-size:.88rem;color:var(--muted)}
.scope b{color:var(--ink)}
.scope svg{border:1px solid var(--line);border-radius:6px;display:block}
nav.tabs{position:sticky;top:0;z-index:20;background:var(--paper);
  border-bottom:1px solid var(--line2);box-shadow:0 1px 0 rgba(0,0,0,.03)}
nav.tabs .wrap{display:flex;gap:2px;overflow-x:auto}
nav.tabs button{border:none;background:transparent;padding:14px 15px;font-size:.8rem;
  font-weight:600;color:var(--muted);cursor:pointer;border-bottom:3px solid transparent;
  white-space:nowrap;font-family:inherit;letter-spacing:.2px}
nav.tabs button:hover{color:var(--ink)}
nav.tabs button.active{color:var(--blue);border-bottom-color:var(--blue)}
section.tab{display:none;padding:22px 0 60px}
section.tab.active{display:block}
.subtabs{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 22px;
  border-bottom:2px solid var(--line);padding-bottom:10px}
.subtab{font:600 13px/1 -apple-system,"Segoe UI",sans-serif;color:var(--muted);
  background:var(--line);border:1px solid #e0d8c8;border-radius:20px;padding:7px 14px;
  cursor:pointer;transition:.15s}
.subtab:hover{background:#e7dfce;color:var(--ink)}
.subtab.active{background:var(--blue);border-color:var(--blue);color:#fff}
.subpane{display:none}
.subpane.active{display:block}
.mh{font:700 13px/1 -apple-system,"Segoe UI",sans-serif;color:var(--blue);
  margin:26px 0 10px;text-transform:uppercase;letter-spacing:.6px}
details.primer{border:1px solid var(--line2);border-radius:8px;background:#fbfaf6;
  margin:4px 0 6px}
details.primer summary{cursor:pointer;padding:12px 16px;font-weight:700;color:var(--blue);
  font-size:.92rem;list-style:none}
details.primer summary::-webkit-details-marker{display:none}
details.primer summary:before{content:"+";display:inline-block;width:18px;
  color:var(--accent);font-weight:700}
details.primer[open] summary:before{content:"−"}
.primer-body{padding:0 16px 16px}
.headrow{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(272px,1fr);
  gap:20px;align-items:stretch}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.stat{border:1px solid var(--line);border-radius:8px;padding:14px 16px;background:#fff}
.stat .v{font-size:1.6rem;font-weight:700;color:var(--blue);line-height:1.05;
  letter-spacing:-.5px}
.stat .l{font-size:.74rem;color:var(--muted);margin-top:6px;line-height:1.4}
.stat.ghost .v{color:#c9c4b8}
.stub{border:1.5px dashed var(--line2);border-radius:8px;color:var(--faint);
  font-size:.82rem;padding:18px;text-align:center;background:#fcfcfd;height:100%;
  display:flex;align-items:center;justify-content:center}
.stub.tall{min-height:260px}
.pending{display:inline-block;font-size:.74rem;font-weight:600;color:#8a6d00;
  background:#fbf4d6;border:1px solid #eedf9e;border-radius:4px;padding:3px 9px;
  margin:0 0 14px}
.primer-fig{margin:4px 0 10px}
.primer-list{margin:0;padding-left:18px;font-size:.86rem}
.primer-list li{margin:0 0 5px}
.primer-list b{color:var(--blue)}
.mixcard{border:1px solid var(--line);border-radius:8px;padding:12px 14px;height:100%;
  display:flex;flex-direction:column;align-items:center}
.mixhead{font-size:.74rem;font-weight:700;color:var(--muted);align-self:flex-start;
  margin-bottom:6px}
.mixleg{display:flex;flex-wrap:wrap;gap:4px 10px;justify-content:center;margin-top:6px}
.lg{display:inline-flex;align-items:center;gap:5px;font:11.5px/1.2 -apple-system,"Segoe UI",sans-serif;
  color:#403b35}
.lg i.d{width:11px;height:11px;border-radius:50%;display:inline-block;box-sizing:border-box}
.lg i.d.sq{border-radius:2px}
.lg i.ln{width:18px;height:3px;border-radius:2px;display:inline-block}
.lg .lsw{flex:0 0 auto}
table.kt{width:100%;border-collapse:collapse;font-size:.84rem;margin:16px 0 6px}
table.kt th{text-align:left;font-size:.72rem;color:var(--muted);font-weight:700;
  border-bottom:2px solid var(--line2);padding:6px 8px;text-transform:uppercase;
  letter-spacing:.3px}
table.kt td{border-bottom:1px solid var(--line);padding:7px 8px;vertical-align:top}
table.kt .n{text-align:right;white-space:nowrap}
table.kt .small,.small{font-size:.76rem;color:var(--muted)}
table.kt td.corr{font-weight:700;color:var(--blue);font-size:.78rem;background:#fbfaf6}
.mixbar{display:inline-flex;height:11px;border-radius:3px;overflow:hidden;
  vertical-align:middle;background:#eee}
.mixbar i{display:block;height:100%}
.st{display:inline-block;font-size:.7rem;font-weight:700;border:1.5px solid;
  border-radius:10px;padding:1px 8px;white-space:nowrap}
.msgs{display:grid;gap:10px;margin:4px 0 8px}
.msg{display:flex;gap:12px;align-items:flex-start;border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;padding:12px 14px;line-height:1.5}
.msg-n{flex:0 0 26px;height:26px;border-radius:50%;background:var(--blue);color:#fff;font-weight:700;display:flex;align-items:center;justify-content:center;font-size:.85rem}
.note{font-size:.74rem;color:var(--faint);margin:6px 0 0}
ol.tl{list-style:none;margin:0;padding:0;border-left:2px solid var(--line2)}
ol.tl li{display:grid;grid-template-columns:86px minmax(0,1fr) 150px;gap:12px;
  padding:6px 0 6px 14px;position:relative;font-size:.86rem}
ol.tl li:before{content:"";position:absolute;left:-6px;top:12px;width:10px;height:10px;
  border-radius:50%;background:#fff;border:2px solid var(--accent)}
.tl-y{font-weight:700;color:var(--blue)}
.tl-s{font-size:.72rem;color:var(--faint);text-align:right}
.imap-box{border:1px solid #e6ded0;border-radius:12px;padding:12px;margin:10px 0 18px;
  display:flex;flex-direction:column;gap:12px}
.imap-btns{display:flex;gap:5px;align-items:center;margin-bottom:8px}
.zb{font:600 13px/1 -apple-system,"Segoe UI",sans-serif;min-width:30px;height:28px;
  padding:0 9px;background:#efe9dc;border:1px solid #e0d8c8;border-radius:7px;
  color:#403b35;cursor:pointer}
.psz-w{display:inline-flex;align-items:center;gap:6px;margin-left:auto}
.psz-l{font:11px/1 -apple-system,"Segoe UI",sans-serif;color:#8a857c}
.psz{width:94px;accent-color:#5389AE}
.imap-wrap{width:100%;overflow:hidden;border-radius:8px;border:1px solid #efe9dc}
svg.imap{display:block;width:100%;height:auto;cursor:grab;touch-action:none}
.mapcap{font:11.5px/1.5 -apple-system,"Segoe UI",sans-serif;color:#6f6a61;margin-top:8px}
.mapcap.disc{font-size:11px;color:#8a857c;font-style:italic;margin-top:3px}
.imap-legend{display:flex;flex-direction:column;gap:9px}
.lgrow{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center}
.lgh{font:600 11px/1.2 -apple-system,"Segoe UI",sans-serif;color:#6f6a61;min-width:170px}
footer{color:var(--faint);font-size:.74rem;padding:24px 0 40px;
  border-top:1px solid var(--line)}
@media (max-width:760px){
  .scope,.headrow{grid-template-columns:1fr}
  ol.tl li{grid-template-columns:70px minmax(0,1fr)}
  .tl-s{grid-column:2;text-align:left}
  table.kt{display:block;overflow-x:auto}
}
"""

# Tab and sub-tab state lives in the URL hash, so a link can open a given pane,
# for instance #energy/power.
JS = """
(function(){
  var tabs=document.querySelectorAll('nav.tabs button');
  function show(tab,sub){
    var pane=document.getElementById('t-'+tab);
    if(!pane){tab=tabs[0].dataset.tab;pane=document.getElementById('t-'+tab);}
    tabs.forEach(function(b){b.classList.toggle('active',b.dataset.tab===tab);});
    document.querySelectorAll('section.tab').forEach(function(s){
      s.classList.toggle('active',s===pane);});
    var subs=pane.querySelectorAll('.subtab');
    if(subs.length){
      var names=Array.prototype.map.call(subs,function(b){return b.dataset.sub;});
      if(names.indexOf(sub)<0){sub=names[0];}
      subs.forEach(function(b){b.classList.toggle('active',b.dataset.sub===sub);});
      pane.querySelectorAll('.subpane').forEach(function(p){
        p.classList.toggle('active',p.dataset.sub===sub);});
    }
    var hash='#'+tab+(subs.length?'/'+sub:'');
    if(location.hash!==hash){history.replaceState(null,'',hash);}
  }
  function fromHash(){
    var parts=location.hash.replace('#','').split('/');
    show(parts[0],parts[1]);
  }
  tabs.forEach(function(b){b.addEventListener('click',function(){show(b.dataset.tab);});});
  document.querySelectorAll('.subtab').forEach(function(b){
    b.addEventListener('click',function(){
      show(b.closest('section.tab').id.slice(2),b.dataset.sub);});});
  window.addEventListener('hashchange',fromHash);
  fromHash();
})();
/* Maps zoom by rewriting the viewBox: buttons, wheel and drag. The plant
   slider rescales every plant circle from its stored radius. */
(function(){
  document.querySelectorAll('.imap-box').forEach(function(box){
    var svg=box.querySelector('svg.imap');
    if(!svg){return;}
    var base=svg.getAttribute('viewBox').split(' ').map(Number);
    var vb=base.slice();
    function set(){svg.setAttribute('viewBox',vb.map(function(v){return v.toFixed(1);}).join(' '));}
    function zoom(k,cx,cy){
      var w=vb[2]*k,h=vb[3]*k;
      if(w>base[2]){w=base[2];h=base[3];}
      if(w<base[2]/12){return;}
      cx=(cx===undefined)?vb[0]+vb[2]/2:cx; cy=(cy===undefined)?vb[1]+vb[3]/2:cy;
      var fx=(cx-vb[0])/vb[2], fy=(cy-vb[1])/vb[3];
      vb=[cx-fx*w,cy-fy*h,w,h]; clamp(); set();
    }
    function clamp(){
      vb[0]=Math.min(Math.max(vb[0],base[0]),base[0]+base[2]-vb[2]);
      vb[1]=Math.min(Math.max(vb[1],base[1]),base[1]+base[3]-vb[3]);
    }
    function pt(e){var r=svg.getBoundingClientRect();
      return [vb[0]+(e.clientX-r.left)/r.width*vb[2], vb[1]+(e.clientY-r.top)/r.height*vb[3]];}
    box.querySelectorAll('.zb').forEach(function(b){b.addEventListener('click',function(){
      var z=b.dataset.z;
      if(z==='in'){zoom(0.7);}else if(z==='out'){zoom(1/0.7);}else{vb=base.slice();set();}
    });});
    svg.addEventListener('wheel',function(e){e.preventDefault();var p=pt(e);
      zoom(e.deltaY<0?0.8:1.25,p[0],p[1]);},{passive:false});
    var drag=null;
    svg.addEventListener('pointerdown',function(e){drag=[e.clientX,e.clientY,vb[0],vb[1]];
      svg.setPointerCapture(e.pointerId);svg.style.cursor='grabbing';});
    svg.addEventListener('pointermove',function(e){if(!drag){return;}
      var r=svg.getBoundingClientRect();
      vb[0]=drag[2]-(e.clientX-drag[0])/r.width*vb[2];
      vb[1]=drag[3]-(e.clientY-drag[1])/r.height*vb[3]; clamp(); set();});
    svg.addEventListener('pointerup',function(){drag=null;svg.style.cursor='';});
    var psz=box.querySelector('.psz');
    if(psz){psz.addEventListener('input',function(){var k=parseFloat(psz.value);
      svg.querySelectorAll('circle.pl').forEach(function(c){
        c.setAttribute('r',(parseFloat(c.dataset.r)*k).toFixed(1));});});}
  });
})();
"""


def tab_section(tab_id: str, subs, index: CountryIndex) -> str:
    out = [f'<section class="tab" id="t-{tab_id}">']
    if tab_id == "synergies":
        out.append(synergies.pane(index))
    elif tab_id == "adhoc":
        out.append(adhoc.pane(index))
    elif not subs:
        out.append(sector_pane(tab_id, index))
    else:
        out.append('<div class="subtabs">')
        out += [f'<button class="subtab" data-sub="{s}">{html.escape(label)}</button>'
                for s, label in subs]
        out.append("</div>")
        for s, _ in subs:
            out.append(f'<div class="subpane" data-sub="{s}">'
                       f'{sector_pane(f"{tab_id}/{s}", index)}</div>')
    out.append("</section>")
    return "".join(out)


def page(index: CountryIndex) -> str:
    nav = "".join(
        f'<button data-tab="{tab_id}">{html.escape(label)}</button>'
        for tab_id, label, _ in TABS
    )
    body = "".join(tab_section(tab_id, subs, index) for tab_id, _, subs in TABS)
    study = ", ".join(STUDY[:-1]) + " and " + STUDY[-1]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Black Sea sector briefs</title>
<style>{CSS}</style></head>
<body>
<header class="top"><div class="wrap">
<h1>Black Sea and South Caucasus: sector briefs</h1>
<p class="sub">Energy, transport, digital and water, each on the same outline, then compared.</p>
<p class="aidisc">The production of this page includes AI processes. Double-checking the sources cited is recommended.</p>
<div class="scope">
<div>
<p><b>Study countries.</b> {html.escape(study)}, shaded on the map.</p>
<p><b>Neighbours.</b> Drawn for context where they shape a corridor: Romania,
Bulgaria, Ukraine, Russia, Iran, Kazakhstan.</p>
<p><b>Outline.</b> Every sector pane runs in the same order: how the sector works,
key figures, background, the region on the ground, a structural map, then projects
and corridors.</p>
</div>
{scope_map(index)}
</div>
</div></header>
<nav class="tabs"><div class="wrap">{nav}</div></nav>
<main class="wrap">{body}</main>
<footer class="wrap">Built {date.today().isoformat()} by
internal_analysis/sector_briefs/build_brief.py.</footer>
<script>{JS}</script>
</body></html>
"""


def main(extra_out: Path | None = None) -> None:
    index = CountryIndex(BASEMAP)
    OUT.write_text(page(index), encoding="utf8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    if extra_out:
        target = extra_out / OUT.name
        shutil.copyfile(OUT, target)
        print(f"Copied to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="extra folder to copy the page into")
    args = parser.parse_args()
    main(args.out)
