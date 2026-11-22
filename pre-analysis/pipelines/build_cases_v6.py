"""
Project cases of the v6 input family: the line files behind the rows of the v6 scenario grid.

    python pre-analysis/pipelines/build_cases_v6.py            # write the files, print and save the checks
    python pre-analysis/pipelines/build_cases_v6.py --check    # compare the files on disk with a fresh build, write nothing
    python pre-analysis/pipelines/build_cases_v6.py --force    # rewrite a file that differs from the build

Order of the v6 builds: build_robg_v6.py (edits 1 to 6 and 9), this script (edit 8), then tools/build_ext_inputs_v6.py,
which derives the Ext scope from the files written here. Every file written here is new and carries _v6 in its name.
The v5 files and the files of edits 1 to 6 are read and never modified. A file keeps the column order, the line
ending and the byte order mark of the file it starts from.

A project is a list of capacity steps on a pair of zones. A case is the set of projects forced on top of the line
limits of edit 3: its capacity sits in the limit files from the start year of the project, and its capex stays
outside the objective (net benefit ex post, as in v5). The first model year at or after the start year is the year
the model sees.
  Base         CTN, the committed Georgia to Armenia line. It is in every case but NoCTN.
  NoCTN        the limits of edit 3 alone: no line between Georgia and Armenia (a sensitivity on the Base)
  BSSC         Georgia to Romania subsea cable
  GECO         BSSC plus the three links from the Georgian hub to Romania; the hub links of the Caucasus are in
               the limits of edit 3 already and wake up with the zone map that holds the two hubs
  Onland       Georgia to eastern Türkiye, to north west Türkiye, to Bulgaria
  EWTC         Türkiye to Bulgaria (an internal line of the Int scope) and Türkiye to Greece (an external line)
  TRIPP        segment 1 mainland Azerbaijan to Nakhchivan, segment 2 Nakhchivan to eastern Türkiye
  TCasp        mainland Azerbaijan to Kazakhstan (an external line)
  BSTN         eastern Türkiye to Georgia reinforcement
  ArTur, ArAz  Armenia to eastern Türkiye, Armenia to mainland Azerbaijan
  AllProjects  every project above
  FreeExp      no project forced: every line above is a candidate sized by the model, capex inside the objective

Files, under epm/input/data_blacksea:
  trade/pTransferLimit_robg_<case>_hvdc_v6.csv        limits of edit 3 plus the internal steps of the case
  trade/pExtTransferLimit_robg_<case>_v6.csv          external limits of edit 3 plus the external steps of the case
  trade/pLossFactorInternal_robg_<case>_hvdc_v6.csv   loss file of edit 4 plus the loss rows of the new pairs, taken
                                                      from the v5 case files (written only where a pair is new)
  trade/pNewTransmission_none_v6.csv                  candidate table of the forced cases: the v5 table, with the
                                                      mainland Azerbaijan to Nakhchivan line no longer committed (it
                                                      is segment 1 of TRIPP and sits in the TRIPP limits)
  trade/pNewTransmission_robg_freeexp_v6.csv          candidates of FreeExp: the v3 table, with the Armenian lines on
                                                      the country team table and one Georgia to Armenia line less
                                                      (CTN is in the limits)
  capex_interconnectors_expost_v6.csv                 ex post capex of every forced project
A case writes a file only where it differs from the Base file. The checks are printed and saved to
internal_analysis/scenarios/cases_v6_report.md.

Country team table. The size, the start year and the capex of the two Armenian lines, and the capex of CTN, are
read at build time in the line sheet of the workbook of the Armenia country team (Official Use Only, outside git,
guarded by its checksum in build_robg_v6.py). No figure of that workbook is written in this file or in the report:
they reach the input files only, which are not in git. The size and the year of CTN are a decision of this study
(350 MW from 2032). Since the corrections of 22 November 2026 the limits of edit 3 hold no line in service between
Georgia and Armenia (rule of the study), so the pair carries CTN alone.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PRE = HERE.parents[1]
ROOT = PRE.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE.parent))
import build_ext_inputs as v5              # noqa: E402  Table: a csv kept as text, byte for byte
import build_ext_inputs_v6 as ext6         # noqa: E402  emit: the write, check and force contract
import build_robg_v6 as robg               # noqa: E402  the country team workbook and its checksum

Table = v5.Table
DATA = v5.DATA
REPORT = ROOT / "internal_analysis" / "scenarios" / "cases_v6_report.md"
QUARTERS = ("Q1", "Q2", "Q3", "Q4")
SIDES = ("Import", "Export")
TOL = 1e-9

# sources, never modified
INT_SRC = "trade/pTransferLimit_robg_hvdc_v6.csv"         # edit 3
EXT_SRC = "trade/pExtTransferLimit_robg_v6.csv"           # edit 3
LOSS_SRC = "trade/pLossFactorInternal_robg_hvdc_v6.csv"   # edit 4
NONE_SRC = "trade/pNewTransmission_none.csv"
FREE_SRC = "trade/pNewTransmission_robg_freeexp_v3.csv"
FREE_EXT = "trade/pNewTransmissionExt_robg_freeexp.csv"   # used as it is by FreeExp in the Int scope
CAPEX_SRC = "capex_interconnectors_expost.csv"
V5_INT = "trade/pTransferLimit_robg%s_hvdc.csv"
V5_LOSS = "trade/pLossFactorInternal_robg%s_hvdc.csv"
V5_LOSS_DONOR = "_geco_onland"                             # v5 loss file that holds the rows of every new pair
V5_EWTC = "trade/pExtTransferLimit_ewtc.csv"               # family before Romania and Bulgaria were modelled
V5_TCASP = "trade/pExtTransferLimit_transcaspian.csv"

# files written
INT_OUT = "trade/pTransferLimit_robg_%s_hvdc_v6.csv"
EXT_OUT = "trade/pExtTransferLimit_robg_%s_v6.csv"
LOSS_OUT = "trade/pLossFactorInternal_robg_%s_hvdc_v6.csv"
NONE_OUT = "trade/pNewTransmission_none_v6.csv"
FREE_OUT = "trade/pNewTransmission_robg_freeexp_v6.csv"
CAPEX_OUT = "capex_interconnectors_expost_v6.csv"

NT_HEADER = ["From", "To", "EarliestEntry", "MaximumNumOfLines", "CapacityPerLine", "CostPerLine", "Life", "Status"]
CAPEX_HEADER = ["project", "from", "to", "capacity_mw", "capex_reference_musd", "capex_low_musd", "basis"]

TEAM = "country team table"   # marks a cell read in the workbook at build time
TEAM_SHEET = "pNewTransmission"
TEAM_BLOCK = "Updated"
TEAM_ROWS = {"CTN": ("Georgia", "Armenia", "committed"),        # project: From, To, first word of the comment
             "ArTur": ("Armenia", "EastAna", "candidate"),
             "ArAz": ("Armenia", "Azerbaijan", "candidate")}
TEAM_FREE_LINES = 3           # FreeExp ceiling on each Armenian pair, in lines of the team size (the table gives none)

# Projects. A step is (zone, zone, first year, MW added in both directions). An external step is
# (zone, external zone, first year, MW added to the import and to the export limit).
PROJECTS = {
    "CTN": {"int": [("Armenia", "Georgia", 2032, 350.0)]},
    "BSSC": {"int": [("Georgia", "Romania", 2031, 1300.0)]},
    "GEC": {"int": [("GEC_GE", "Romania", 2032, 1261.0), ("GEC_GE", "Romania", 2036, 1261.0),
                    ("GEC_GE", "Romania", 2040, 1261.0)]},
    "Onland": {"int": [("EastAna", "Georgia", 2031, 2000.0), ("EastAna", "NorthWest", 2031, 2000.0),
                       ("NorthWest", "Bulgaria", 2031, 2000.0)]},
    "EWTC": {"int": [("Bulgaria", "Trakia", 2033, 775.0)], "ext": [("Trakia", "Greece", 2034, 775.0)]},
    "TRIPP": {"int": [("AzerbaijanMain", "Nakhchivan", 2028, 800.0), ("EastAna", "Nakhchivan", 2032, 1000.0)]},
    "TCasp": {"ext": [("AzerbaijanMain", "Kazakhstan", 2035, 1000.0)]},
    "BSTN": {"int": [("EastAna", "Georgia", 2032, 350.0)]},
    "ArTur": {"int": [("Armenia", "EastAna", TEAM, TEAM)]},
    "ArAz": {"int": [("Armenia", "AzerbaijanMain", TEAM, TEAM)]},
}
COMMITTED = "CTN"                                   # in every case but NO_COMMITTED
NO_COMMITTED = "noctn"                              # sensitivity on the Base: the limits of edit 3 alone
CASES = {"base": (), "bssc": ("BSSC",), "geco": ("BSSC", "GEC"), "onland": ("Onland",), "ewtc": ("EWTC",),
         "tripp": ("TRIPP",), "tcasp": ("TCasp",), "bstn": ("BSTN",), "artur": ("ArTur",), "araz": ("ArAz",),
         "allprojects": tuple(p for p in PROJECTS if p != COMMITTED)}
LOSS_TOKEN = {"allprojects": "geco_onland"}         # the loss file of AllProjects also serves FreeExp
V5_TWINS = {"bssc": "_bssc", "geco": "_geco", "onland": "_onland"}   # cases that exist in v5: same steps expected
HUB_ZONES = ("GEC_AZ", "GEC_GE")

# Ex post capex of the projects the v5 table does not hold: the CostPerLine of the candidate row of the same line.
CAPEX_FROM = (("EWTC", FREE_SRC, "Trakia", "Bulgaria"), ("EWTC", FREE_EXT, "Trakia", "Greece"),
              ("TRIPP", FREE_SRC, "AzerbaijanMain", "Nakhchivan"), ("TRIPP", FREE_SRC, "Nakhchivan", "EastAna"),
              ("TCasp", FREE_EXT, "AzerbaijanMain", "Kazakhstan"), ("BSTN", NONE_SRC, "EastAna", "Georgia"))
CAPEX_V5 = {"BSSC": "BSSC", "GEC": "GEC", "Onland": "Onland"}   # project: its row of the v5 table


def fail(msg):
    raise SystemExit("ERROR: " + msg)


def whole(value):
    """A number as the candidate tables write a year, a count or a cost."""
    return "%d" % value if float(value) == int(value) else "%s" % float(value)


# ---------------------------------------------------------------------------- country team table
def team_lines():
    """{project: year, mw, cost, life} from the Updated block of the line sheet of the country team workbook."""
    book_path = robg.AM_BOOK
    if not book_path.exists():
        fail("%s is missing: table of the Armenia country team, Official Use Only, kept out of git" % book_path)
    if robg.sha256_of(book_path) != robg.AM_BOOK_SHA256:
        fail("%s is not the workbook the v6 edits were written on (sha256 differs)" % book_path.name)
    import openpyxl
    book = openpyxl.load_workbook(book_path, data_only=True, read_only=True)
    if TEAM_SHEET not in book.sheetnames:
        fail("%s: no sheet %s" % (book_path.name, TEAM_SHEET))
    grid = [tuple(r) for r in book[TEAM_SHEET].iter_rows(values_only=True)]
    book.close()
    marks = [i for i, r in enumerate(grid) if r and isinstance(r[0], str) and r[0].strip() == TEAM_BLOCK]
    if len(marks) != 1:
        fail("%s, sheet %s: expected one block %s" % (book_path.name, TEAM_SHEET, TEAM_BLOCK))
    head = next((i for i in range(marks[0] + 1, len(grid)) if grid[i] and grid[i][0] == NT_HEADER[0]), None)
    if head is None or [str(h).strip() for h in grid[head][:len(NT_HEADER)]] != NT_HEADER:
        fail("%s, sheet %s: the block %s does not carry the pNewTransmission header" % (book_path.name, TEAM_SHEET, TEAM_BLOCK))
    rows = {}
    for r in grid[head + 1:]:
        cells = list(r) + [None] * (len(NT_HEADER) + 1 - len(r))
        if all(v is None for v in cells):
            break
        rec = dict(zip(NT_HEADER, cells))
        rec["_comment"] = " ".join(str(v).strip() for v in cells[len(NT_HEADER):] if v is not None)
        rows[(str(rec["From"]).strip(), str(rec["To"]).strip())] = rec
    if set(rows) != {v[:2] for v in TEAM_ROWS.values()}:
        fail("%s, sheet %s: the lines of the block %s are not the three this build expects" % (book_path.name, TEAM_SHEET, TEAM_BLOCK))
    team = {}
    for project, (frm, to, word) in TEAM_ROWS.items():
        rec = rows[(frm, to)]
        if not rec["_comment"].lower().startswith(word):
            fail("%s, sheet %s: the line of %s is not marked %s" % (book_path.name, TEAM_SHEET, project, word))
        try:
            team[project] = {"year": int(rec["EarliestEntry"]), "mw": float(rec["CapacityPerLine"]),
                             "cost": float(rec["CostPerLine"]), "life": int(rec["Life"])}
        except (TypeError, ValueError):
            fail("%s, sheet %s: the line of %s misses a year, a size, a cost or a life" % (book_path.name, TEAM_SHEET, project))
    return team


def steps(project, kind, team):
    """The steps of a project with the cells of the country team table filled in."""
    out = []
    for z, z2, year, mw in PROJECTS[project].get(kind, ()):
        if year is TEAM or mw is TEAM:
            year, mw = team[project]["year"], team[project]["mw"]
        out.append((z, z2, year, mw))
    return out


def team_pairs():
    """Pairs whose figures come from the country team table: never printed."""
    return {frozenset(s[:2]) for p, d in PROJECTS.items() for s in d.get("int", ()) if s[2] is TEAM}


# ---------------------------------------------------------------------------- limits
def add_internal(table, a, b, year, mw):
    """Add mw to both directions of the pair from year on. A pair without rows starts at zero."""
    years = table.header[3:]
    for z, z2 in ((a, b), (b, a)):
        rows = [r for r in table.rows if r[0] == z and r[1] == z2]
        if not rows:
            rows = [[z, z2, q] + ["0.0"] * len(years) for q in QUARTERS]
            table.rows.extend(rows)
        if [r[2] for r in rows] != list(QUARTERS):
            fail("pTransferLimit: %s to %s does not hold one row per quarter" % (z, z2))
        for r in rows:
            for j, y in enumerate(years, 3):
                if int(y) >= year:
                    r[j] = "%.1f" % (float(r[j]) + mw)


def add_external(table, z, zext, year, mw):
    """Add mw to the import and export limits of an external border from year on. The border must exist."""
    years = table.header[4:]
    rows = [r for r in table.rows if r[0] == z and r[1] == zext]
    if sorted((r[2], r[3]) for r in rows) != sorted((q, s) for q in QUARTERS for s in SIDES):
        fail("pExtTransferLimit: %s to %s does not hold one import and one export row per quarter" % (z, zext))
    for r in rows:
        dotted = any("." in c for c in r[4:])
        for j, y in enumerate(years, 4):
            if int(y) >= year:
                v = float(r[j]) + mw
                r[j] = "%.1f" % v if dotted or v != int(v) else "%d" % v


def delta(case, base, k):
    """{key: values of the case minus the base} for the rows of the case that are new or differ. No base row may go."""
    ref = {tuple(r[:k]): r[k:] for r in base.rows}
    seen, out = set(), {}
    for r in case.rows:
        key = tuple(r[:k])
        if key in seen:
            fail("a line file holds the row %s twice" % (key,))
        seen.add(key)
        old = ref.get(key)
        d = [float(a) - (float(b) if old else 0.0) for a, b in zip(r[k:], old or r[k:])]
        if old is None or any(abs(v) > TOL for v in d):
            out[key] = d
    gone = [key for key in ref if key not in seen]
    if gone:
        fail("a case file lost rows of its base: %s" % gone[:4])
    return out


def same_delta(a, b):
    return set(a) == set(b) and all(len(a[k]) == len(b[k]) and all(abs(x - y) <= TOL for x, y in zip(a[k], b[k])) for k in a)


def describe(table, d, k, hidden):
    """One line per pair and direction of a delta: the limit of the case, as 'MW from year' at each change."""
    years = table.header[k:]
    rows = {tuple(r[:k]): r[k:] for r in table.rows}
    out = []
    for key in d:
        if key[2] != QUARTERS[0]:
            continue
        name = "%s to %s" % key[:2] + (" (%s)" % key[3].lower() if k == 4 else "")
        if frozenset(key[:2]) in hidden:
            out.append("%s: set by the %s" % (name, TEAM))
            continue
        marks, last = [], None
        for y, v in zip(years, rows[key]):
            if last is None or abs(float(v) - last) > TOL:
                marks.append("%s MW from %s" % (whole(float(v)) if float(v) == int(float(v)) else v, y))
                last = float(v)
        out.append("%s: %s" % (name, ", ".join(marks)))
    return out


def build_limits(log, out, team):
    log.append("Line limits")
    src, ext = Table.read(INT_SRC), Table.read(EXT_SRC)
    if src.header[:3] != ["z", "z2", "q"]:
        fail("unexpected pTransferLimit header %s" % src.header[:4])
    if ext.header[:3] != ["z", "zext", "q"] or ext.header[3] != "":
        fail("unexpected pExtTransferLimit header %s" % ext.header[:5])
    hidden = team_pairs()

    # no line in service on the pair of CTN (rule of the study, correction A1 of build_robg_v6.py): CTN alone
    ctn = steps(COMMITTED, "int", team)
    if len(ctn) != 1:
        fail("%s is one step" % COMMITTED)
    today = {float(c) for r in src.rows if {r[0], r[1]} == set(ctn[0][:2]) for c in r[3:]}
    if today != {0.0}:
        fail("%s: %s holds a limit on the pair before the new line, build edit 3 of build_robg_v6.py" % (COMMITTED, INT_SRC))
    if ctn[0][3] > team[COMMITTED]["mw"] + TOL:
        fail("%s is larger than the total of the %s" % (COMMITTED, TEAM))
    ctn_note = "at" if abs(ctn[0][3] - team[COMMITTED]["mw"]) <= TOL else "below"

    tables = {}
    for token, projects in CASES.items():
        t, x = src.copy(), ext.copy()
        for p in (COMMITTED,) + tuple(projects):
            for z, z2, year, mw in steps(p, "int", team):
                add_internal(t, z, z2, year, mw)
            for z, zext, year, mw in steps(p, "ext", team):
                add_external(x, z, zext, year, mw)
        tables[token] = (t, x)
    base_t, base_x = tables["base"]
    if base_x.rows != ext.rows:
        fail("the Base must keep the external limits of edit 3")
    d0 = delta(base_t, src, 3)
    out[INT_OUT % "base"] = base_t
    log.append("  %s: %d rows, %s plus %s" % (INT_OUT % "base", len(base_t.rows), INT_SRC, COMMITTED))
    for line in describe(base_t, d0, 3, set()):
        log.append("    " + line)
    log.append("  the pair of %s carries the new line alone, %s the total of the %s" % (COMMITTED, ctn_note, TEAM))
    log.append("  external limits of the Base: %s, unchanged (%d rows)" % (EXT_SRC, len(ext.rows)))
    out[INT_OUT % NO_COMMITTED] = src.copy()
    log.append("  case %s: %s, the %d rows of %s (no %s); loss factors and external limits of the Base"
               % (NO_COMMITTED, INT_OUT % NO_COMMITTED, len(src.rows), INT_SRC, COMMITTED))

    deltas = {}
    for token in CASES:
        if token == "base":
            continue
        t, x = tables[token]
        di, dx = delta(t, base_t, 3), delta(x, ext, 4)
        deltas[token] = (di, dx)
        if not di and not dx:
            fail("case %s adds nothing to the Base" % token)
        log.append("  case %s: %s" % (token, " + ".join(CASES[token])))
        if di:
            out[INT_OUT % token] = t
            log.append("    %s: %d rows, %d of the Base changed or added" % (INT_OUT % token, len(t.rows), len(di)))
            for line in describe(t, di, 3, hidden):
                log.append("      " + line)
        else:
            log.append("    internal limits: the Base file")
        if dx:
            out[EXT_OUT % token] = x
            log.append("    %s: %d rows, %d changed" % (EXT_OUT % token, len(x.rows), len(dx)))
            for line in describe(x, dx, 4, hidden):
                log.append("      " + line)

    # checks against the v5 case files: same steps on top of the base
    v5_base = Table.read(V5_INT % "")
    if v5_base.header != src.header:
        fail("the v5 and v6 limit files do not share their header")
    for token, suffix in V5_TWINS.items():
        if not same_delta(deltas[token][0], delta(Table.read(V5_INT % suffix), v5_base, 3)):
            fail("case %s: the steps on top of the Base are not those of %s" % (token, V5_INT % suffix))
    log.append("  check: bssc, geco and onland add to the v6 Base what the v5 case files add to the v5 base")
    single = [t for t in CASES if t not in ("base", "bssc", "allprojects")]   # geco holds bssc
    total = {}
    for token in single:
        for kind in (0, 1):
            for key, d in deltas[token][kind].items():
                acc = total.setdefault((kind, key), [0.0] * len(d))
                total[(kind, key)] = [a + b for a, b in zip(acc, d)]
    for kind in (0, 1):
        if not same_delta(deltas["allprojects"][kind], {k[1]: v for k, v in total.items() if k[0] == kind}):
            fail("allprojects is not the sum of the project cases")
    log.append("  check: allprojects adds the sum of what %s add" % ", ".join(single))

    # external steps against the family that held them before
    old = Table.read(V5_EWTC)
    xi = {tuple(r[:4]): r[4:] for r in tables["ewtc"][1].rows}
    ti = {tuple(r[:3]): r[3:] for r in tables["ewtc"][0].rows}
    oi = {tuple(r[:4]): dict(zip(old.header[4:], r[4:])) for r in old.rows}
    ok = True
    for q in QUARTERS:
        for side in SIDES:
            pairs = [(xi[("Trakia", "Greece", q, side)], ext.header[4:], oi[("Trakia", "Greece", q, side)]),
                     (ti[("Bulgaria", "Trakia", q) if side == "Import" else ("Trakia", "Bulgaria", q)], src.header[3:],
                      oi[("Trakia", "Bulgaria", q, side)])]
            for cells, years, ref in pairs:
                ok = ok and all(abs(float(v) - float(ref[y])) <= TOL for y, v in zip(years, cells) if y in ref)
    if not ok:
        fail("case ewtc: the Bulgarian or the Greek leg is not the limit of %s" % V5_EWTC)
    old = Table.read(V5_TCASP)
    oi = {tuple(r[:4]): dict(zip(old.header[4:], r[4:])) for r in old.rows}
    xi = {tuple(r[:4]): r[4:] for r in tables["tcasp"][1].rows}
    for key in [k for k in xi if k[:2] == ("AzerbaijanMain", "Kazakhstan")]:
        if not all(abs(float(v) - float(oi[key][y])) <= TOL for y, v in zip(ext.header[4:], xi[key]) if y in oi[key]):
            fail("case tcasp: the Kazakh border is not the limit of %s" % V5_TCASP)
    log.append("  check: the two legs of ewtc and the border of tcasp carry the limits of %s and %s" % (V5_EWTC, V5_TCASP))

    # FreeExp: the Base limits, plus the zero rows the v5 file adds for the two subsea pairs
    v5_free = delta(Table.read(V5_INT % "_freeexp"), v5_base, 3)
    if any(abs(v) > TOL for d in v5_free.values() for v in d):
        fail("%s is expected to add zero rows only" % (V5_INT % "_freeexp"))
    free = base_t.copy()
    have = {tuple(r[:3]) for r in free.rows}
    for r in Table.read(V5_INT % "_freeexp").rows:
        if tuple(r[:3]) in v5_free:
            if tuple(r[:3]) in have:
                fail("freeexp: %s is already in the Base" % (tuple(r[:3]),))
            free.rows.append(list(r))
    out[INT_OUT % "freeexp"] = free
    log.append("  case freeexp: %s: %d rows, the Base plus %d rows at zero (%s), as the v5 file"
               % (INT_OUT % "freeexp", len(free.rows), len(v5_free),
                  ", ".join(sorted({"%s to %s" % k[:2] for k in v5_free}))))
    return base_t


# ---------------------------------------------------------------------------- losses
def build_losses(log, out, team):
    log.append("Loss factors")
    src = Table.read(LOSS_SRC)
    v5_base, donor = Table.read(V5_LOSS % ""), Table.read(V5_LOSS % V5_LOSS_DONOR)
    if src.header != donor.header or src.header != v5_base.header:
        fail("the v5 and v6 loss files do not share their header")
    have = {(r[0], r[1]) for r in src.rows}
    new = {(r[0], r[1]): r for r in donor.rows if (r[0], r[1]) not in {(b[0], b[1]) for b in v5_base.rows}}
    for key, r in new.items():
        if key in have:
            fail("loss factors: %s to %s is already in %s" % (key + (LOSS_SRC,)))
        if len(set(r[2:])) != 1:
            fail("loss factors: the v5 row %s to %s is not flat" % key)
    log.append("  %s: %d rows; rows of the new pairs, from %s: %s" % (
        LOSS_SRC, len(src.rows), V5_LOSS % V5_LOSS_DONOR,
        ", ".join("%s to %s %s" % (k[0], k[1], r[2]) for k, r in new.items())))
    written = {}
    for token, projects in CASES.items():
        t = src.copy()
        for p in (COMMITTED,) + tuple(projects):
            for z, z2, _, _ in steps(p, "int", team):
                for key in ((z, z2), (z2, z)):
                    if key in {(r[0], r[1]) for r in t.rows}:
                        continue
                    if key not in new:
                        fail("loss factors: no row for %s to %s, in %s or in the v5 case files" % (key + (LOSS_SRC,)))
                    t.rows.append(list(new[key]))
        if t.rows == src.rows:
            log.append("  case %s: %s" % (token, LOSS_SRC))
            continue
        name = LOSS_OUT % LOSS_TOKEN.get(token, token)
        out[name] = t
        written[token] = t
        log.append("  case %s: %s, %d rows" % (token, name, len(t.rows)))
    for token, suffix in dict(V5_TWINS, allprojects=V5_LOSS_DONOR).items():
        if not same_delta(delta(written[token], src, 2), delta(Table.read(V5_LOSS % suffix), v5_base, 2)):
            fail("loss factors, case %s: the rows added are not those of %s" % (token, V5_LOSS % suffix))
    log.append("  check: each loss file adds to %s the rows its v5 twin adds to the v5 base" % LOSS_SRC)
    return written["allprojects"]


# ---------------------------------------------------------------------------- candidates
def candidate_rows(rel):
    t = Table.read(rel)
    if t.header != NT_HEADER:
        fail("%s: unexpected header %s" % (rel, t.header))
    return t


def build_candidates(log, out, team, loss_all):
    log.append("Candidate tables")
    col = {name: j for j, name in enumerate(NT_HEADER)}

    # forced cases: nothing can be built, and the Nakhchivan line is no longer committed
    none = candidate_rows(NONE_SRC).copy()
    seg1 = PROJECTS["TRIPP"]["int"][0]
    hits = [r for r in none.rows if {r[0], r[1]} == set(seg1[:2])]
    if len(hits) != 1 or hits[0][col["Status"]] != "2" or float(hits[0][col["CapacityPerLine"]]) != seg1[3]:
        fail("%s: expected one committed line between %s and %s at the size of TRIPP segment 1" % ((NONE_SRC,) + seg1[:2]))
    hits[0][col["MaximumNumOfLines"]], hits[0][col["Status"]] = "0", "3"
    if any(r[col["MaximumNumOfLines"]] != "0" or r[col["Status"]] != "3" for r in none.rows):
        fail("%s: a line of the forced cases could still be built" % NONE_OUT)
    out[NONE_OUT] = none
    log.append("  %s: the %d rows of %s; %s to %s is no longer committed (0 lines, status 3): it is segment 1 of TRIPP. "
               "No row can be built" % ((NONE_OUT, len(none.rows), NONE_SRC) + seg1[:2]))

    # FreeExp
    free = candidate_rows(FREE_SRC).copy()
    idx = {(r[0], r[1]): r for r in free.rows}
    if len(idx) != len(free.rows):
        fail("%s: a pair comes twice" % FREE_SRC)
    ctn = steps(COMMITTED, "int", team)[0]
    r = idx.get(("Georgia", "Armenia")) or fail("%s: no Georgia to Armenia candidate" % FREE_SRC)
    if float(r[col["CapacityPerLine"]]) != ctn[3] or int(r[col["MaximumNumOfLines"]]) < 2:
        fail("%s: the Georgia to Armenia candidate is not a set of lines of the size of %s" % (FREE_SRC, COMMITTED))
    r[col["EarliestEntry"]] = whole(ctn[2])
    r[col["MaximumNumOfLines"]] = whole(int(r[col["MaximumNumOfLines"]]) - 1)
    r[col["CostPerLine"]] = whole(team[COMMITTED]["cost"])
    r = idx.get(("AzerbaijanMain", "Nakhchivan")) or fail("%s: no Nakhchivan line" % FREE_SRC)
    if r[col["Status"]] != "2":
        fail("%s: the Nakhchivan line is expected committed" % FREE_SRC)
    r[col["Status"]] = "3"
    for project in ("ArTur", "ArAz"):
        z, z2, year, mw = steps(project, "int", team)[0]
        cells = [z, z2, whole(year), whole(TEAM_FREE_LINES), "%.1f" % mw, whole(team[project]["cost"]),
                 whole(team[project]["life"]), "3"]
        if (z, z2) in idx:
            idx[(z, z2)][:] = cells
        elif (z2, z) in idx:
            fail("%s: %s to %s is written the other way round" % (FREE_SRC, z, z2))
        else:
            free.rows.append(cells)
    if any(r[col["Status"]] != "3" for r in free.rows):
        fail("%s: every line of FreeExp is a candidate" % FREE_OUT)
    pairs = {(r[0], r[1]) for r in loss_all.rows}
    lost = [(r[0], r[1]) for r in free.rows if (r[0], r[1]) not in pairs or (r[1], r[0]) not in pairs]
    if lost:
        fail("FreeExp: no loss factor in both directions for %s" % lost)
    out[FREE_OUT] = free
    log.append("  %s: %d candidates, from the %d rows of %s. Georgia to Armenia: one line less (%s is in the limits), "
               "from %d, capex of the %s. The Nakhchivan line becomes a candidate. The two Armenian lines take the "
               "size, the year and the capex of the %s, up to %d lines each (assumption of this build: the table "
               "gives no ceiling)" % (FREE_OUT, len(free.rows), len(candidate_rows(FREE_SRC).rows), FREE_SRC, COMMITTED,
                                      ctn[2], TEAM, TEAM, TEAM_FREE_LINES))
    hidden = team_pairs() | {frozenset(("Georgia", "Armenia"))}
    for r in free.rows:
        shown = "lines of the %s" % TEAM if frozenset(r[:2]) in hidden else \
            "%s x %s MW from %s, %s MUSD a line" % (r[col["MaximumNumOfLines"]], whole(float(r[col["CapacityPerLine"]])),
                                                   r[col["EarliestEntry"]], r[col["CostPerLine"]])
        log.append("    %s to %s: %s" % (r[0], r[1], shown))
    log.append("    every pair holds a loss factor in both directions in %s" % (LOSS_OUT % LOSS_TOKEN["allprojects"]))
    log.append("    external candidates of FreeExp: %s, as it is" % FREE_EXT)
    return free


# ---------------------------------------------------------------------------- ex post capex
def build_capex(log, out, team):
    log.append("Ex post capex")
    src = Table.read(CAPEX_SRC)
    if src.header != CAPEX_HEADER:
        fail("%s: unexpected header %s" % (CAPEX_SRC, src.header))
    t = src.copy()
    names = [r[0] for r in t.rows]
    if sorted(names) != sorted(CAPEX_V5.values()):
        fail("%s: expected the rows %s" % (CAPEX_SRC, sorted(CAPEX_V5.values())))
    sizes = {}
    for p, d in PROJECTS.items():
        for kind in ("int", "ext"):
            for z, z2, year, mw in steps(p, kind, team):
                sizes[(p, frozenset((z, z2)))] = sizes.get((p, frozenset((z, z2))), 0.0) + mw
    for project, name in CAPEX_V5.items():
        r = t.rows[names.index(name)]
        last = [k for k in sizes if k[0] == project][-1]
        if abs(float(r[3]) - sizes[last]) > TOL:
            fail("%s: the size of %s is not the size forced in the limits" % (CAPEX_SRC, name))
    public = [(r[0], "%s to %s" % (r[1], r[2]), r[3], r[4]) for r in t.rows]
    c = team[COMMITTED]
    t.rows.append([COMMITTED, "Georgia", "Armenia", whole(steps(COMMITTED, "int", team)[0][3]), whole(c["cost"]), whole(c["cost"]),
                   "%s; forced in the Base and in every case, so it cancels in a net benefit; no low basis" % TEAM])
    for project, rel, frm, to in CAPEX_FROM:
        hit = [r for r in candidate_rows(rel).rows if r[0] == frm and r[1] == to]
        if len(hit) != 1:
            fail("%s: expected one row %s to %s" % (rel, frm, to))
        size, cost = float(hit[0][4]), float(hit[0][5])
        if abs(sizes.get((project, frozenset((frm, to))), -1.0) - size) > TOL:
            fail("%s, %s to %s: the candidate is not the size forced by %s" % (rel, frm, to, project))
        t.rows.append([project, frm, to, whole(size), whole(cost), whole(cost),
                       "capex of the candidate row of %s; no low basis" % rel])
        public.append((project, "%s to %s" % (frm, to), whole(size), whole(cost)))
    for project in ("ArTur", "ArAz"):
        z, z2, _, mw = steps(project, "int", team)[0]
        t.rows.append([project, z, z2, whole(mw), whole(team[project]["cost"]), whole(team[project]["cost"]),
                       "%s; no low basis" % TEAM])
    out[CAPEX_OUT] = t
    log.append("  %s: the %d rows of %s, %d rows from the candidate tables, 3 rows from the %s (not printed)"
               % (CAPEX_OUT, len(src.rows), CAPEX_SRC, len(CAPEX_FROM), TEAM))
    for p, pair, size, cost in public:
        log.append("    %s, %s: %s MW, %s MUSD" % (p, pair, size, cost))
    log.append("    sum of the rows printed: %s MUSD" % whole(sum(float(x[3]) for x in public)))


# ---------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare the files with a fresh build, write nothing")
    ap.add_argument("--force", action="store_true", help="rewrite a file that differs from the build")
    args = ap.parse_args(argv)

    sources = [INT_SRC, EXT_SRC, LOSS_SRC, NONE_SRC, FREE_SRC, FREE_EXT, CAPEX_SRC, V5_EWTC, V5_TCASP,
               V5_LOSS % "", V5_LOSS % V5_LOSS_DONOR, V5_INT % "", V5_INT % "_freeexp"] + \
              [V5_INT % s for s in V5_TWINS.values()] + [V5_LOSS % s for s in V5_TWINS.values()]
    log = ["# Project cases of the v6 input family", "",
           "Generated by pre-analysis/pipelines/build_cases_v6.py from %s." % DATA.relative_to(ROOT).as_posix(),
           "Figures of the %s are not printed." % TEAM, ""]
    edited = [INT_SRC, EXT_SRC, LOSS_SRC, NONE_SRC, FREE_SRC, CAPEX_SRC]
    bad = [s for s in edited if not v5.roundtrip(s)]
    if bad:
        fail("the serialisation does not reproduce the bytes of %s" % bad)
    log.append("Sources: %d files, never modified; the %d a file is built from are reproduced byte for byte" % (len(sources), len(edited)))
    team = team_lines()
    out = {}
    build_limits(log, out, team)
    loss_all = build_losses(log, out, team)
    build_candidates(log, out, team, loss_all)
    build_capex(log, out, team)
    clash = [rel for rel in out if rel in sources or "_v6" not in rel]
    if clash:
        fail("a file of this build would overwrite a source or misses the v6 suffix: %s" % clash)
    files = {DATA / rel: table.text().encode("utf-8") for rel, table in out.items()}
    problems = ext6.emit(files, args, log)
    log.append("  %d files, %d problem(s)" % (len(files), problems))
    text = "\n".join(log)
    print(text)
    if not args.check:
        REPORT.write_text(text + "\n", encoding="utf-8")
        print("report: %s" % REPORT.relative_to(ROOT).as_posix())
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
