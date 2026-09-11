"""Cut the results section out of the deck and refresh every chart it carries.

The September deck is a copy of the August one: 77 slides, of which only the
first 18 are results.  The rest is context, country zooms and annexes that the
run does not touch.  This writes a results-only deck and, freed from the August
picture geometry, redraws in place the charts that were pasted as screenshots of
the HTML report, each rendered at the exact size of the box it lands in so the
image is never rescaled by PowerPoint.

    python build_results_deck.py --dry-run     # regenerate and report, no write
    python build_results_deck.py
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                            # blacksea_2026
SRC = ROOT / "BlackSea_regional_power_trade_followup_September2026_results.pptx"
DST = ROOT / "BlackSea_regional_power_trade_followup_September2026_results_only.pptx"
PNGS = ROOT / "Data" / "results" / "slides"

PICTURE = 13
KEEP = 18                 # slides 1..18 are the results section
FLAG = "stale-marker"
ASPECT_TOL = .02

# (slide, picture name) -> (png, generator module, generator arguments).
# The size is read off the deck, so a box that moves takes its chart with it.
SLOTS = [
    (3,  "Picture 2",  "ndp_georgia.png",       "slide_ndp",     ["--scope", "Georgia"]),
    (4,  "Picture 28", "flowmap_georgia.png",   "slide_flowmap", ["--scope", "Georgia"]),
    (5,  "Picture 2",  "ndp_azerbaijan.png",    "slide_ndp",     ["--scope", "Azerbaijan"]),
    (6,  "Picture 8",  "flowmap_azerbaijan.png", "slide_flowmap", ["--scope", "Azerbaijan"]),
    (7,  "Picture 4",  "ndp_armenia.png",       "slide_ndp",     ["--scope", "Armenia"]),
    (8,  "Picture 5",  "flowmap_armenia.png",   "slide_flowmap", ["--scope", "Armenia"]),
    (9,  "Picture 2",  "ndp_turkiye.png",       "slide_ndp",     ["--scope", "Turkiye"]),
    (10, "Picture 3",  "flowmap_turkiye.png",   "slide_flowmap", ["--scope", "Turkiye"]),
    (12, "Picture 3",  "trade_region.png",      "slide_trade",    ["--scope", "Region"]),
    (12, "Picture 5",  "seasonal_region.png",   "slide_seasonal", ["--scope", "Region"]),
    (12, "Picture 8",  "dispatch_region.png",   "slide_dispatch",
     ["--scope", "Region", "--years", "2025,2035"]),
    (13, "Picture 4",  "bssc_map.png",          "slides_regional", ["--chart", "bssc_map"]),
    (14, "Picture 4",  "bssc_impact.png",       "slides_regional", ["--chart", "bssc_impact"]),
    (15, "Picture 4",  "freeexp_map.png",       "slides_regional", ["--chart", "freeexp_map"]),
]


def find(slide, name):
    hits = [sh for sh in slide.shapes
            if sh.shape_type == PICTURE and sh.name == name]
    if len(hits) != 1:
        raise SystemExit("%s: %d matches, expected 1" % (name, len(hits)))
    return hits[0]


def drop_after(prs, keep):
    """Remove every slide past `keep`, and the relationship that carries it."""
    ids = prs.slides._sldIdLst
    for sld in list(ids)[keep:]:
        prs.part.drop_rel(sld.rId)
        ids.remove(sld)


def regenerate(png, module, args, w_in, h_in, dpi=300):
    cmd = [sys.executable, str(HERE / ("%s.py" % module)), *args,
           "--width", "%.3f" % w_in, "--height", "%.3f" % h_in,
           "--dpi", str(dpi), "--out", str(PNGS / png)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    if r.returncode:
        raise SystemExit("%s failed:\n%s" % (png, r.stderr[-2000:]))


def swap(pic, path, slide):
    """Point the picture at a new image, keeping its box, crop and z-order."""
    part, rId = slide.part.get_or_add_image_part(str(path))
    pic._element.blipFill.blip.set(qn("r:embed"), rId)
    return part


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    a = p.parse_args()

    prs = Presentation(str(SRC))
    if len(prs.slides) <= KEEP:
        print("source already trimmed: %d slides" % len(prs.slides))
    else:
        drop_after(prs, KEEP)
    print("results deck: %d slides" % len(prs.slides))

    for n, name, png, module, args in SLOTS:
        slide = prs.slides[n - 1]
        pic = find(slide, name)
        w_in, h_in = pic.width / 914400, pic.height / 914400
        regenerate(png, module, args, w_in, h_in, a.dpi)
        path = PNGS / png
        im = Image.open(path)
        got, want = im.size[0] / im.size[1], w_in / h_in
        if abs(got - want) / want > ASPECT_TOL:
            print("  s%-2d %-12s ASPECT %.2f vs box %.2f, skipped"
                  % (n, name, got, want))
            continue
        if not a.dry_run:
            swap(pic, path, slide)
        print("  s%-2d %-12s <- %-24s %.2f x %.2f in" % (n, name, png, w_in, h_in))

    # Every stale box now carries a real chart, so the warnings come off.
    dropped = 0
    for slide in prs.slides:
        for sh in [s for s in slide.shapes if s.name == FLAG]:
            sh._element.getparent().remove(sh._element)
            dropped += 1
    print("removed %d stale markers" % dropped)

    if a.dry_run:
        print("dry run, nothing written")
        return
    if DST.exists():
        bak = DST.with_suffix(".bak_%s.pptx" % datetime.now().strftime("%Y%m%d_%H%M"))
        shutil.copy2(DST, bak)
        print("backup  %s" % bak.name)
    prs.save(str(DST))
    print("written %s  (%.1f MB)" % (DST.name, DST.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
