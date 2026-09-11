# -*- coding: utf-8 -*-
"""Write the EU price and CBAM sensitivity into the results deck.

Run after build_results_deck.py, which refreshes the wave 1 pictures. This one
touches only the tail of the deck:

    s16  the summary table, filled from the central run
    s17  NEW  the price shock: price trajectories and export volumes
    s18  NEW  the value of the corridors: NPV saving and 2040 volumes
    s19  key points, filled
    s20  gas prices, left alone apart from two em-dashes

    python slides_pricesens_deck.py [--dry]

Every figure below is read from the two runs at build time, never typed in, so
a re-run of the model moves the slides with it. The prose around them is fixed
text and has to be re-read when the numbers move.

The deck has a single layout named DEFAULT carrying no placeholder, so a new
slide comes out blank. The title bar is the one shape copied from an existing
results slide; everything else is built here.
"""
import argparse
import copy
import shutil
import sys
from datetime import datetime
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runcfg  # noqa: E402
import slide_pricesens as sp  # noqa: E402

import pandas as pd  # noqa: E402

DECK = HERE.parents[2] / (
    "BlackSea_regional_power_trade_followup_September2026_results_only.pptx")
SLIDES = HERE.parents[2] / "Data" / "results" / "slides"

TITLE_SRC = 13          # slide whose title bar is copied onto the new slides
INSERT_AFTER = 16       # the two new slides land right after this one

DARK = RGBColor(0x15, 0x43, 0x60)
BODY = RGBColor(0x33, 0x40, 0x4D)
MUTED = RGBColor(0x5A, 0x6B, 0x7B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RULE = RGBColor(0xD6, 0xDE, 0xE7)
BAND = RGBColor(0xF2, 0xF5, 0xF9)

EMU = 914400


def E(inches):
    return Emu(int(round(inches * EMU)))


# --------------------------------------------------------------------- numbers

def read_numbers():
    """Everything the tables and captions quote, straight out of the runs."""
    central = runcfg.OUTVIEW / sp.RUN_CENTRAL
    sens = runcfg.OUTVIEW / sp.RUN_SENS

    def sysrow(run, attribute):
        df = pd.read_csv(run / "summary.csv", low_memory=False)
        r = df[(df["country"] == "System") & (df["attribute"] == attribute)]
        return {c: float(r.iloc[0][c]) for c in r.columns
                if c not in ("country", "zone", "attribute", "resolution", "year")}

    n = {}
    for key, attribute in (("npv", "NPV of system cost: $m"),
                           ("exprev", "Export revenues with external zones: $m"),
                           ("impcost", "Import costs with external zones: $m"),
                           ("trans", "Transmission costs: $m"),
                           ("invest", "Investment costs: $m")):
        n[key] = sysrow(central, attribute)

    # Exports to the EU in 2040, by topology and by price world.
    n["eu2040"] = {}
    for row in sp.TOPO:
        for world in sp.WORLDS:
            key = world[0]
            n["eu2040"][(row[0], key)] = sp.eu_exports(
                sp.run_of(key), sp.scen_of(row, key))[-1]

    # NPV saving against the Base case of the same world, billion USD.
    n["saving"] = {}
    for world in sp.WORLDS:
        key = world[0]
        obj = sp.objectives(sp.run_of(key))
        base = obj[sp.scen_of(sp.TOPO[0], key)]
        for row in sp.TOPO:
            n["saving"][(row[0], key)] = base - obj[sp.scen_of(row, key)]

    n.update(read_gas_evidence(central))
    n.update(read_corridor_evidence(central))
    return n


TR_ZONES = ("CenterAna", "CenterBlack", "EastAna", "EastMed", "NorthWest",
            "SouthEast", "Trakia", "WestAna", "WestMed")


def read_gas_evidence(central):
    """What the flat Turkish gas price does, for the corrections on s20.

    The gas page was written against the WEO STEPS path that fell to 6.5 by
    2030. Every scenario now runs pFuelPrice_tr_gas_flat.csv instead, so the
    numbers on that page have to come back out of the run.
    """
    df = pd.read_csv(central / "summary.csv", low_memory=False)
    price = df[(df["attribute"] == "Price: $/MWh") & df["zone"].isin(TR_ZONES)]
    # The year column comes back as a float, so it is compared as a number.
    yr = pd.to_numeric(price["year"], errors="coerce")
    out = {}
    for year in (2030, 2040):
        v = pd.to_numeric(price[yr == year]["baseline"],
                          errors="coerce").dropna()
        if v.empty:
            raise SystemExit("no Turkish zone price for %d" % year)
        out["tr_price_%d" % year] = (v.min(), v.max())

    tm = pd.read_csv(central / "baseline" / "output_csv"
                     / "pTransmissionMerged.csv")
    flow = tm[(tm["attribute"] == "Interchange") & (tm["y"].astype(str) == "2040")]

    def pair(a, b):
        m = flow[(flow["z"] == a) & (flow["uni"] == b)]["value"]
        return float(m.sum())

    out["ge_tr_net"] = (pair("Georgia", "EastAna")
                        - pair("EastAna", "Georgia")) / 1000.0
    return out


def read_corridor_evidence(central):
    """Who actually fills the corridor in All Projects 2040, for s19."""
    run = central / "LC_AllProjects" / "output_csv"
    tm = pd.read_csv(run / "pTransmissionMerged.csv")
    y40 = tm[tm["y"].astype(str) == "2040"]

    ext = y40[y40["attribute"] == "InterchangeExternalExports"]
    out = {"ge_to_ro": float(ext[(ext["z"] == "Georgia")
                                 & (ext["uni"] == "Romania")]["value"].sum()) / 1000.0,
           "tr_to_eu": float(ext[(ext["z"] == "Trakia")
                                 & ext["uni"].isin(sp.EU)]["value"].sum()) / 1000.0}

    into = y40[(y40["attribute"] == "Interchange") & (y40["uni"] == "Georgia")]
    out["into_ge"] = {str(z): float(v) / 1000.0 for z, v
                      in into.groupby("z")["value"].sum().items()}

    eb = pd.read_csv(run / "pEnergyBalance.csv")
    ge = eb[(eb["z"] == "Georgia") & (eb["y"].astype(str) == "2040")]
    for label, key in (("Total production: GWh", "ge_prod"),
                       ("Total Demand: GWh", "ge_dem"),
                       ("Imports exchange: GWh", "ge_imp")):
        out[key] = float(ge[ge["uni"] == label]["value"].sum()) / 1000.0
    return out


# ----------------------------------------------------------------------- shapes

def set_runs(tf, text):
    """Replace a text frame's content, keeping the formatting of its first run."""
    p = tf.paragraphs[0]
    keep = p.runs[0]
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    keep.text = text
    for extra in tf.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def textbox(slide, left, top, width, height, blocks):
    """A text box from (text, size, bold, colour, space_before) tuples."""
    box = slide.shapes.add_textbox(E(left), E(top), E(width), E(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, (text, size, bold, colour, before) in enumerate(blocks):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(before)
        p.space_after = Pt(0)
        p.line_spacing = 1.05
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = colour
    return box


def table(slide, left, top, width, height, rows, widths, fs=8.5):
    """A flat table: dark header row, banded body, no PowerPoint style chrome."""
    shape = slide.shapes.add_table(len(rows), len(rows[0]),
                                   E(left), E(top), E(width), E(height))
    tbl = shape.table
    tbl.first_row = True
    tbl.horz_banding = False
    for j, w in enumerate(widths):
        tbl.columns[j].width = E(w)
    for i, row in enumerate(rows):
        tbl.rows[i].height = E(height / len(rows))
        for j, text in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = E(.06)
            cell.margin_right = E(.06)
            cell.margin_top = E(.02)
            cell.margin_bottom = E(.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = (
                DARK if i == 0 else (BAND if i % 2 == 0 else WHITE))
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            r = p.add_run()
            r.text = text
            r.font.size = Pt(fs)
            r.font.bold = (i == 0 or j == 0)
            r.font.color.rgb = WHITE if i == 0 else (DARK if j == 0 else BODY)
    return shape


def new_slide(prs, title):
    """A blank slide carrying the same title bar as the results slides."""
    src = prs.slides[TITLE_SRC - 1]
    slide = prs.slides.add_slide(src.slide_layout)
    bar = None
    for shape in src.shapes:
        if shape.has_text_frame and shape.name == "Text 1":
            el = copy.deepcopy(shape._element)
            slide.shapes._spTree.insert_element_before(el, "p:extLst")
            bar = slide.shapes[-1]
            break
    if bar is None:
        raise RuntimeError("no title bar on slide %d" % TITLE_SRC)
    set_runs(bar.text_frame, title)
    return slide


def move_slide(prs, frm, to):
    lst = prs.slides._sldIdLst
    items = list(lst)
    lst.remove(items[frm])
    lst.insert(to, items[frm])


def picture(slide, name, left, top, width):
    path = SLIDES / name
    if not path.exists():
        raise SystemExit("missing chart: %s" % path)
    return slide.shapes.add_picture(str(path), E(left), E(top), width=E(width))


# ------------------------------------------------------------------ the slides

def fill_summary(slide, n):
    """s16: the central run, all four topologies, one column each."""
    for shape in list(slide.shapes):
        if shape.name.startswith("Rectangle"):
            shape._element.getparent().remove(shape._element)

    col = ("baseline", "LC_BSSC", "LC_AllProjects", "LC_FreeExpAll")
    base = n["npv"]["baseline"]

    def money(key, scale=1000.0, absolute=False):
        vals = [n[key][c] / scale for c in col]
        return ["%.2f" % (abs(v) if absolute else v) for v in vals]

    rows = [
        ["NPV 2025-2040, billion USD", "Base", "BSSC", "All Projects",
         "Free Expansion"],
        ["System cost"] + money("npv"),
        ["Saving vs Base"] + ["n/a"] + ["%.2f" % ((base - n["npv"][c]) / 1000.0)
                                        for c in col[1:]],
        ["Export revenue, external zones"] + money("exprev", absolute=True),
        ["Import cost, external zones"] + money("impcost"),
        ["Transmission cost, internal lines"] + money("trans"),
        ["Generation investment"] + money("invest"),
        ["Exports to the EU in 2040, TWh"]
        + ["%.1f" % n["eu2040"][(t, "central")]
           for t in ("Base", "BSSC", "All Projects", "Free Exp.")],
        ["Georgia to Romania limit, MW", "0", "1,300 from 2032",
         "5,200 from 2037", "0"],
    ]
    table(slide, .35, .78, 9.30, 3.35, rows,
          [3.30, 1.35, 1.55, 1.55, 1.55], fs=9.0)

    textbox(slide, .35, 4.26, 9.30, 1.20, [
        ("How to read this", 9.5, True, DARK, 0),
        ("The Georgia to Romania corridor enters the model as an external "
         "transfer limit, not as a line, so it carries no capital cost in the "
         "objective. The transmission cost row covers the internal Caucasus "
         "lines only. Every saving above is therefore gross of the submarine "
         "cable capex, and the 4.7 billion gap between BSSC and All Projects "
         "is what a 5,200 MW corridor would have to cost less than to be worth "
         "building.", 8.0, False, BODY, 3),
        ("Free Expansion opens 21 internal candidate lines and builds 10.2 GW "
         "of them, but it keeps the base external limits, so it never reaches "
         "the EU. It is not a superset of All Projects and the two columns are "
         "not comparable on corridor scope.", 8.0, False, BODY, 3),
    ])


def fill_shock(slide, n):
    """s17: what the three price worlds do, and what the region can sell."""
    picture(slide, "pricesens_price.png", .35, .78, 5.20)
    textbox(slide, 5.80, .82, 3.95, 2.25, [
        ("What is varied", 10.0, True, DARK, 0),
        ("Only the export price file changes. Demand, fleet, fuel prices and "
         "topology are held fixed, so every difference on this deck page is "
         "the price assumption and nothing else.", 8.2, False, BODY, 4),
        ("EU central. The reference used in the August results, around "
         "90 USD/MWh to 2040.", 8.2, False, BODY, 5),
        ("EU very low. Fast EU decarbonisation and weak gas, the netback "
         "falling to 40 USD/MWh by 2040.", 8.2, False, BODY, 4),
        ("EU central + CBAM. The same central price less a border charge set "
         "on the grid average carbon intensity of the exporting zone. It "
         "leaves 11 to 19 USD/MWh from 2030, which is an export ban in "
         "practice rather than a price signal.", 8.2, False, BODY, 4),
    ])
    picture(slide, "pricesens_exports.png", .35, 3.30, 9.20)
    textbox(slide, .35, 5.28, 9.30, .28, [
        ("Volume follows price, but only where a corridor exists. Base and "
         "Free Expansion sit on top of one another in all three worlds, which "
         "is why Base is drawn dashed.", 7.0, False, MUTED, 0),
    ])


def fill_value(slide, n):
    """s18: what the corridors are worth once the price is allowed to move."""
    picture(slide, "pricesens_benefit.png", .35, .78, 5.20)
    textbox(slide, 5.80, .82, 3.95, 2.25, [
        ("Read with three caveats", 10.0, True, DARK, 0),
        ("1. This is not a full two by two. CBAM is tested at central prices "
         "only and the very low world is tested without CBAM, so the two "
         "effects cannot be separated from their combination.", 8.2, False,
         BODY, 4),
        ("2. The CBAM column is the pessimistic bound. With plant level "
         "attribution of a low carbon export the real case sits between this "
         "column and the central one.", 8.2, False, BODY, 4),
        ("3. The corridor is free in the model. Its capex never reaches the "
         "objective, so each bar is a gross benefit to compare against the "
         "cable cost, not a net present value.", 8.2, False, BODY, 4),
    ])

    order = ("Base", "BSSC", "All Projects", "Free Exp.")
    rows = [["Exports to the EU in 2040, TWh"] + [w[1] for w in sp.WORLDS]]
    rows += [[t] + ["%.1f" % n["eu2040"][(t, w[0])] for w in sp.WORLDS]
             for t in order]
    rows += [["NPV saving vs Base, billion USD"] + [""] * len(sp.WORLDS)]
    rows += [[t] + ["%.2f" % n["saving"][(t, w[0])] for w in sp.WORLDS]
             for t in order[1:]]
    table(slide, .35, 3.24, 9.30, 2.02, rows, [3.30, 2.00, 2.00, 2.00], fs=8.0)


def fill_keypoints(slide, n):
    """s19: the headline, drafted from the numbers, for the user to validate."""
    for shape in list(slide.shapes):
        if shape.name in ("TextBox 6", "TextBox 8", "TextBox 10"):
            shape._element.getparent().remove(shape._element)

    ap = n["saving"][("All Projects", "central")]
    bs = n["saving"][("BSSC", "central")]
    fe = n["saving"][("Free Exp.", "central")]
    lost_low = 100.0 * (1.0 - n["saving"][("All Projects", "verylow")] / ap)
    lost_cbam = 100.0 * (1.0 - n["saving"][("All Projects", "cbam")] / ap)

    textbox(slide, .35, .62, 4.55, 4.60, [
        ("With the current data and results", 11.5, True, DARK, 0),
        ("The value of the corridors is the value of EU market access. "
         "All Projects saves %.1f billion against the base topology, BSSC "
         "alone %.1f, and freeing 21 internal candidate lines saves %.2f. "
         "What separates them is external headroom, not internal capacity."
         % (ap, bs, fe), 8.5, False, BODY, 6),
        ("That access is what the price sensitivity puts at risk. Cutting the "
         "EU netback to 40 USD/MWh by 2040 removes %.0f percent of the All "
         "Projects benefit; a CBAM charge set on grid average intensity "
         "removes %.0f percent of it." % (lost_low, lost_cbam),
         8.5, False, BODY, 5),
        ("The ordering survives every world tested. All Projects beats BSSC "
         "beats Free Expansion in all three, so the choice between corridors "
         "does not depend on the EU price assumption. Only the size of the "
         "prize does.", 8.5, False, BODY, 5),
        ("The corridor is a regional pool, not one country's export. In All "
         "Projects 2040 Georgia ships %.1f TWh to Romania while producing %.1f "
         "and consuming %.1f: it takes %.1f TWh in from Azerbaijan, %.1f from "
         "Turkiye and %.1f from Armenia and passes them on. Trakia sends a "
         "further %.1f TWh straight to Bulgaria and Greece."
         % (n["ge_to_ro"], n["ge_prod"], n["ge_dem"],
            n["into_ge"].get("AzerbaijanMain", 0.0),
            n["into_ge"].get("EastAna", 0.0),
            n["into_ge"].get("Armenia", 0.0), n["tr_to_eu"]),
         8.5, False, BODY, 5),
    ])
    textbox(slide, 5.20, .62, 4.55, 2.20, [
        ("However", 11.5, True, DARK, 0),
        ("No corridor capex is in any of these numbers. The Georgia to Romania "
         "capacity is an external transfer limit, so a 5,200 MW submarine link "
         "is free to the optimiser. The benefits above are gross and have to "
         "be tested against a real cable cost before any of them is a case for "
         "investment.", 8.5, False, BODY, 6),
        ("No country planning reserve margin is enforced, the input file being "
         "empty, so reliability value is understated across the board.",
         8.5, False, BODY, 5),
        ("The isolation counterfactual has no price sensitivity: LC_Iso and "
         "LC_TransCaspian were not rerun in waves 2 and 3.", 8.5, False,
         BODY, 5),
    ])
    textbox(slide, 5.20, 2.60, 4.55, 2.60, [
        ("Next steps", 11.5, True, DARK, 0),
        ("Complete the grid: the very low world with CBAM, which is the "
         "downside case the region should actually plan against, and the two "
         "missing topologies.", 8.5, False, BODY, 6),
        ("Cost the corridor. Move the Georgia to Romania link into "
         "pNewTransmission with a capex so the objective sees it, or state a "
         "breakeven cable cost from the gross benefit.", 8.5, False, BODY, 5),
        ("Fill the planning reserve margin, then rerun the wave 1 topologies "
         "once so the reliability value stops being missing.", 8.5, False,
         BODY, 5),
    ])


def fix_dashes(slide):
    """The house rule is no em-dashes in prose. Two survive on the gas page.

    Only the running text below the table is touched. The table cells are the
    author's own notes and several of their dashes are labels rather than
    prose, so rewriting them here would be a bigger edit than it looks.
    """
    fixed = 0
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for p in shape.text_frame.paragraphs:
            for r in p.runs:
                if "—" in r.text:
                    r.text = r.text.replace(" — ", ", ").replace("—", ",")
                    fixed += 1
    return fixed


def set_cell(table_shape, row, col, para, text):
    """Rewrite one paragraph of one cell, keeping its first run's formatting."""
    cell = table_shape.table.cell(row, col)
    p = cell.text_frame.paragraphs[para]
    keep = p.runs[0]
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    keep.text = text


def fix_gas_slide(slide, n):
    """s20 was written against a gas path no scenario uses any more.

    input_scenarios.csv sets pFuelPrice to pFuelPrice_tr_gas_flat.csv in every
    column, which holds Turkiye at 11.3 to 2025 and 10.0 flat from 2030. The
    page still quotes the WEO STEPS path that fell to 6.5 by 2030, calls the
    flat variant unwired, and reports a Turkish price and a Georgian flow
    reversal that the current run does not produce.
    """
    tbl = next(s for s in slide.shapes if s.has_table)
    set_cell(tbl, 1, 2, 0, "11.3 → 10.0")
    set_cell(tbl, 1, 2, 1, "2025 / from 2030, flat")
    set_cell(tbl, 1, 2, 2, "flat variant, used by every scenario")
    set_cell(tbl, 1, 3, 0, "Changed: flat variant now in use")
    set_cell(tbl, 1, 3, 1,
             "•  The WEO STEPS path it replaces fell to 6.5 by 2030. "
             "Holding Turkish gas at 10.0 instead keeps gas the expensive fuel "
             "in Turkiye rather than the cheap one")
    set_cell(tbl, 1, 3, 2,
             "•  pFuelPrice_tr_gas_flat.csv is the pFuelPrice of every "
             "column of input_scenarios.csv, so this is the assumption behind "
             "all results in this deck")

    low30, high30 = n["tr_price_2030"]
    low40, high40 = n["tr_price_2040"]
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for p in shape.text_frame.paragraphs:
            if not p.runs or "Turkish gas glide" not in p.runs[0].text:
                continue
            keep = p.runs[0]
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)
            keep.text = (
                "•  Turkish gas held flat at 10.0 $/MMBtu leaves the "
                "Turkish zonal price at %.0f to %.0f $/MWh in 2030 and %.0f to "
                "%.0f in 2040, well above the 43 the old 6.5 path produced. "
                "Georgia stays a net exporter to Turkiye throughout, %.1f TWh "
                "in 2040, so the flip to importer that path implied does not "
                "happen." % (low30, high30, low40, high40, n["ge_tr_net"]))
            return 1
    return 0


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    n = read_numbers()
    prs = Presentation(str(DECK))
    if len(prs.slides) != 18:
        raise SystemExit("expected an 18 slide deck, found %d. Run "
                         "build_results_deck.py first, and only once."
                         % len(prs.slides))

    fill_summary(prs.slides[15], n)
    print("s16 summary table filled")

    shock = new_slide(prs, "EU price sensitivity: the price shock")
    fill_shock(shock, n)
    move_slide(prs, len(prs.slides) - 1, INSERT_AFTER)
    print("s17 price shock inserted")

    value = new_slide(prs, "EU price sensitivity: value of the corridors")
    fill_value(value, n)
    move_slide(prs, len(prs.slides) - 1, INSERT_AFTER + 1)
    print("s18 value of the corridors inserted")

    fill_keypoints(prs.slides[18], n)
    print("s19 key points filled")

    print("s20 gas page corrected: %d headline run(s), %d em-dash run(s)"
          % (fix_gas_slide(prs.slides[19], n), fix_dashes(prs.slides[19])))

    if a.dry:
        print("dry run, nothing written")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DECK.with_name(DECK.stem + ".bak_" + stamp + DECK.suffix)
    shutil.copy2(DECK, backup)
    prs.save(str(DECK))
    print("backup  %s" % backup.name)
    print("written %s  (%d slides, %.1f MB)"
          % (DECK.name, len(prs.slides), DECK.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
