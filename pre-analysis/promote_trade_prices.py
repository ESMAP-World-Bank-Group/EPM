# -*- coding: utf-8 -*-
"""P7 - copy the staged trade prices into data_blacksea/trade.

build_trade_prices.py writes only to output_prices/staging and says so on every run,
because promotion into epm/input is a decision and not a side effect. This is that step,
written down instead of done by hand, so a later reader can see exactly which staged file
became which live file.

    staging/<name>/pTradePriceExport.csv  ->  trade/pTradePriceExport_<name>.csv
    staging/<name>/pTradePrice.csv        ->  trade/pTradePrice_<name>.csv

The second line applies only to the five non-CBAM scenarios. CBAM is a levy on what the
EU pays for an imported MWh, so it moves the sell side alone; staging/eu_X_cbam and
staging/eu_X hold a byte-identical pTradePrice.csv, and the live tree keeps one copy under
the plain name. The check below asserts that identity rather than trusting it, since a
silent divergence would put a CBAM scenario on a non-CBAM buy price.

Every file that is overwritten is backed up first. data_blacksea is DVC tracked and
gitignored, so an accidental overwrite here is not recoverable from git.
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE / "output_prices" / "staging"
TRADE = HERE.parent / "epm" / "input" / "data_blacksea" / "trade"
BAK = ".bak_kzprice_20260830"


def plan():
    """[(src, dst)], plus the CBAM pairs whose buy side must be identical."""
    moves, pairs = [], []
    for d in sorted(p for p in STAGING.iterdir() if p.is_dir()):
        moves.append((d / "pTradePriceExport.csv",
                      TRADE / ("pTradePriceExport_%s.csv" % d.name)))
        if d.name.endswith("_cbam"):
            pairs.append((d, STAGING / d.name[:-5]))
        else:
            moves.append((d / "pTradePrice.csv",
                          TRADE / ("pTradePrice_%s.csv" % d.name)))
    return moves, pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)

    moves, pairs = plan()
    for cbam, plain in pairs:
        if not filecmp.cmp(str(cbam / "pTradePrice.csv"),
                           str(plain / "pTradePrice.csv"), shallow=False):
            print("FAIL %s and %s disagree on the buy side" % (cbam.name, plain.name))
            return 1
    print("CBAM buy sides identical to their plain twin: %d pairs" % len(pairs))

    n = 0
    for src, dst in moves:
        if not src.exists():
            print("FAIL missing %s" % src)
            return 1
        same = dst.exists() and filecmp.cmp(str(src), str(dst), shallow=False)
        print("%-8s %-22s -> %s" % ("same" if same else "copy", src.parent.name, dst.name))
        if same or a.dry:
            continue
        if dst.exists():
            shutil.copy2(str(dst), str(dst) + BAK)
        shutil.copy2(str(src), str(dst))
        n += 1
    print("%s %d files (backup suffix %s)" % ("would write" if a.dry else "wrote", n, BAK))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
