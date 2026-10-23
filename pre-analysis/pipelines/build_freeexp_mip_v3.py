# -*- coding: utf-8 -*-
"""Free expansion in MIP on the five-year grid (S5_FreeExp_v3_MIP): the MIP twin of S5_FreeExp_v3.
Every output is a NEW file or a NEW scenarios.csv column; nothing read by an existing scenario
changes.

Why a dedicated column
    epm.py --modeltype MIP switches the model type without any data change, but main.gms fixes
    the relative gap at optCR = 0.05: five percent of the objective (about 14 BUSD on 279) is far
    above the interconnection benefits (0.5 to 4 BUSD), so CPLEX would stop at the first rounded
    solution of the root. A CPLEX option file value of epgap takes precedence over optCR, and the
    option file is attached per scenario through the cplexfile cell of scenarios.csv.

File
    cplex/cplex_mip_s5.opt = cplex/cplex_s5.opt (barrier root, 8 threads, 25 GB working memory,
    crossover to a basis) with epgap = 0.0005 (0.05 percent, about 0.14 BUSD), startalg = 4
    (barrier for the root relaxation), nodefileind = 3 (node files to disk when the tree exceeds
    the working memory), mipdisplay = 3 (gap in the log) and without solutiontype = 2 (an LP
    setting; the branch and bound needs a basis).

Column (1)
    S5_FreeExp_v3_MIP = S5_FreeExp_v3 with cplexfile = cplex/cplex_mip_s5.opt; every other cell
    identical. Run with epm.py --modeltype MIP. The integers are the candidate lines (0 to
    MaximumNumOfLines per candidate) and the plants with a UnitSize; the unit commitment and
    economic retirement flags are 0 in pSettings.

Usage: python build_freeexp_mip_v3.py [path to data_blacksea]. Idempotent.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_freeexp_robg import Table  # noqa: E402  (byte preserving csv table, same folder)
import build_costs_v3_robg as v3  # noqa: E402  (ROOT, path, write_table, truncate_columns)

OPT_SRC, OPT_NEW = "cplex/cplex_s5.opt", "cplex/cplex_mip_s5.opt"
NEW, SRC = "S5_FreeExp_v3_MIP", "S5_FreeExp_v3"
LF = chr(10)

DROPPED = ["solutiontype"]
ADDED = [("epgap", "0.0005"), ("startalg", "4"), ("nodefileind", "3"), ("mipdisplay", "3")]
HEADER = """* ------------------------------------------------------------
* CPLEX MIP configuration for the S5 family (5-year grid): the MIP twin
* of cplex_s5.opt. Same barrier settings for the root relaxation (8
* threads, 25 GB working memory), crossover to a basis for the branch
* and bound, node files to disk beyond the working memory. The relative
* gap is set here because main.gms fixes optCR at 0.05: five percent of
* the objective is far above the interconnection benefits. epgap 0.0005
* is about 0.14 BUSD on a 279 BUSD objective; read the final gap in the
* log, not only the solve status. Run with epm.py --modeltype MIP.
* ------------------------------------------------------------
"""


def parse_opt(text):
    """Ordered (key, value) pairs of the non comment lines of a CPLEX option file."""
    pairs = []
    for ln in text.split(LF):
        s = ln.strip()
        if not s or s.startswith("*"):
            continue
        key, value = s.split("=", 1)
        pairs.append((key.strip(), value.strip()))
    return pairs


def build_opt():
    raw = open(v3.path(OPT_SRC), "rb").read()
    assert chr(13).encode() not in raw and raw.endswith(LF.encode()), OPT_SRC
    src = parse_opt(raw.decode("utf-8"))
    keys = [k for k, _ in src]
    assert len(set(keys)) == len(keys), "duplicate keys in " + OPT_SRC
    for k in DROPPED:
        assert k in keys, (OPT_SRC, k)
    for k, _ in ADDED:
        assert k not in keys, (OPT_SRC, k)
    expected = {"lpmethod": "4", "threads": "8", "barcrossalg": "1", "workmem": "25000", "solutiontype": "2"}
    for k, val in expected.items():
        assert dict(src)[k] == val, (OPT_SRC, k, dict(src)[k])
    kept = [(k, val) for k, val in src if k not in DROPPED]
    key_block = ADDED[:3] + [(k, val) for k, val in kept if k not in ("names", "bardisplay", "iis")]
    support = [(k, val) for k, val in kept if k in ("names", "bardisplay", "iis")] + ADDED[3:]
    lines = [HEADER.rstrip(LF), "", "* --- Key parameters ---"]
    lines += ["%s=%s" % kv for kv in key_block]
    lines += ["", "* --- Supporting parameters ---"]
    lines += ["%s=%s" % kv for kv in support]
    text = LF.join(lines) + LF
    with open(v3.path(OPT_NEW), "wb") as f:
        f.write(text.encode("utf-8"))
    back = parse_opt(open(v3.path(OPT_NEW), "rb").read().decode("utf-8"))
    assert dict(back) == dict(kept + ADDED) and len(back) == len(kept) + len(ADDED), OPT_NEW
    assert "solutiontype" not in dict(back)
    print("  %s = %s without %s, plus %s" % (OPT_NEW, OPT_SRC, ", ".join(DROPPED), ", ".join("%s=%s" % kv for kv in ADDED)))


def build_scenarios():
    sc = Table(v3.path("scenarios.csv"))
    H = sc.header
    assert H[0] == "paramNames", H[:3]
    names = [r[0] for r in sc.rows[1:]]
    assert NEW not in H or H[-1] == NEW, "the column exists but is not the last one"
    orig = [c for c in H if c != NEW]
    orig_rows = [list(r[:len(orig)]) for r in sc.rows]
    sc = v3.truncate_columns(sc, len(orig))
    j = orig.index(SRC)
    cells = [r[j] for r in sc.rows[1:]]
    k = names.index("cplexfile")
    assert cells[k] == OPT_SRC, (SRC, cells[k])
    cells[k] = OPT_NEW
    for v in cells:
        if v.endswith(".csv") or v.endswith(".opt"):
            assert os.path.exists(v3.path(v)), (NEW, v)
    sc.append_column(NEW, cells)
    v3.write_table("scenarios.csv", sc)
    back = Table(v3.path("scenarios.csv"))
    assert [r[:len(orig)] for r in back.rows] == orig_rows, "an existing column changed"
    assert back.header[-1] == NEW and [r[-1] for r in back.rows[1:]] == cells, NEW
    changed = [names[i] for i, (a, b) in enumerate(zip([r[j] for r in back.rows[1:]], cells)) if a != b]
    assert changed == ["cplexfile"], changed
    print("scenarios.csv: %d existing columns kept byte for byte, 1 column added: %s = %s with cplexfile = %s" % (len(orig), NEW, SRC, OPT_NEW))


if __name__ == "__main__":
    print("data folder: " + v3.ROOT)
    build_opt()
    build_scenarios()
