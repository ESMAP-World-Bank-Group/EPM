# -*- coding: utf-8 -*-
"""P5 - turn netback.csv into the two EPM trade-price files, in staging.

Writes, per price scenario declared in config/eu_price_scenarios.yaml:

    output_prices/staging/<name>/pTradePrice.csv        buy side  (import)
    output_prices/staging/<name>/pTradePriceExport.csv  sell side (export)

Nothing under epm/input/ is touched. Promotion is P7, and a separate decision.

Built by overwrite, not by regeneration
---------------------------------------
The live pTradePrice.csv is read and only the value columns of the three EU
zones are replaced. Everything else - header bytes, CRLF endings, absence of a
BOM, the 6 720 rows, their order - is carried over untouched. This matters more
than it looks: the file's own row order is not the obvious one. It runs
Q1 d1-d6, Q2 d1-d6, Q3 d1-d6, Q4 d1-d6 and only then Q1-Q4 d7, because the
seventh day-type was added to the representative-day set after the rest. GAMS
reads by label and would not care, but a regenerated file would produce a diff
in which every line has moved and no reviewer could see what actually changed.

Scope
-----
Bulgaria, Greece and Romania take the EU hypothesis. Iran, Iraq, Syria, Russia
and Kazakhstan keep the cost-based values already in the file: no liquid hub, no
TYNDP entry, no ETS. They are copied through, and the DIFF says so explicitly
rather than leaving a reader to notice the absence.

The export file exists because base.gms:686 currently credits exports at the
import price. Until the pTradePriceExport patch lands, only pTradePrice.csv is
read and the sell side is silently overstated by the whole spread - which the
DIFF quantifies so the size of that error is on the record.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_PRE_ANALYSIS = Path(__file__).resolve().parents[1]

HOURS = [f"t{h:02d}" for h in range(1, 25)]
KEY = ["zext", "q", "d", "year"]

# The zones the EU hypothesis speaks for. Everything else in the file is copied.
EU_ZONES = ("Bulgaria", "Greece", "Romania")

# Written precision. Deliberately finer than the two decimals a price needs:
# main.gms:894-895 deletes any hour whose price rounds to zero, in both
# directions at once, so the margin between the 0.01 floor and the rounding
# threshold is a safety property, not formatting taste.
DECIMALS = 3
FLOAT_FMT = f"%.{DECIMALS}f"


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_trade_prices] FAIL: {msg}")


def load_scenarios(path: Path) -> Dict[str, dict]:
    try:
        import yaml
    except ImportError:
        _fail("pyyaml is needed to read eu_price_scenarios.yaml")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    scen = doc.get("scenarios") or {}
    if not scen:
        _fail(f"{path.name}: no scenarios declared")
    ref = doc.get("reference")
    if ref and ref not in scen:
        _fail(f"{path.name}: reference {ref!r} is not among the scenarios")
    for name, spec in scen.items():
        if not {"level", "variant"} <= set(spec or {}):
            _fail(f"{path.name}: {name} needs both 'level' and 'variant'")
    return {"reference": ref, "scenarios": scen}


def read_template(path: Path) -> pd.DataFrame:
    """The live file, with its row order preserved as an explicit column."""
    raw = path.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        _fail(f"{path.name} carries a BOM; the writer here does not emit one")
    df = pd.read_csv(path)
    missing = [c for c in KEY + HOURS if c not in df.columns]
    if missing:
        _fail(f"{path.name}: missing columns {missing}")
    # The live file is all integers, so pandas types the hour columns int64 and
    # assigning a price into them would silently truncate. Cast once, here,
    # rather than at each assignment. The untouched zones keep their value and
    # gain three decimals of formatting - 40 becomes 40.000, which GAMS reads
    # identically.
    df[HOURS] = df[HOURS].astype(float)
    df["_row"] = np.arange(len(df))
    return df


def pivot_netback(nb: pd.DataFrame, level: str, variant: str,
                  direction: str) -> pd.DataFrame:
    """One (zext,q,d,year) x 24h table for a single trajectory and variant."""
    g = nb[(nb["scenario"] == level) & (nb["variant"] == variant)
           & (nb["direction"] == direction)]
    if g.empty:
        _fail(f"netback.csv holds nothing for {level}/{variant}/{direction}")
    out = g.rename(columns={"zone": "zext"})[KEY + HOURS].copy()
    dup = out.duplicated(KEY).sum()
    if dup:
        _fail(f"{level}/{variant}/{direction}: {dup} duplicate keys")
    return out


def apply_prices(template: pd.DataFrame, priced: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the value columns of the EU zones; copy every other row.

    An inner merge on the full key means a day-type or a year present in the
    template but absent from the netback is caught here rather than silently
    left at its old value - which would produce a file that is half hypothesis
    and half legacy placeholder, and looks perfectly fine.
    """
    tmpl = template.copy()
    eu = tmpl[tmpl["zext"].isin(EU_ZONES)]
    if eu.empty:
        _fail(f"none of {EU_ZONES} appear in the template file")

    merged = eu[KEY + ["_row"]].merge(priced, on=KEY, how="left", indicator=True)
    gap = merged[merged["_merge"] != "both"]
    if len(gap):
        s = gap[KEY].head(5).to_string(index=False)
        _fail(f"{len(gap)} template rows have no netback value, e.g.\n{s}")

    tmpl = tmpl.set_index("_row")
    tmpl.loc[merged["_row"].to_numpy(), HOURS] = merged[HOURS].to_numpy()
    return tmpl.reset_index().sort_values("_row").drop(columns="_row")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CRLF, no BOM, trailing newline - the shape the live file already has."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=FLOAT_FMT,
              lineterminator="\r\n", encoding="utf-8")


# ── Gates ───────────────────────────────────────────────────────────────────
def gate_g5a(template: Path, staged: Path) -> dict:
    """Structure identical to the live file: header, keys, order, encoding.

    Compares the first line byte for byte and the key column as a sequence, not
    as a set. Order is checked because a reordered file passes every value test
    and still makes the DIFF unreadable.
    """
    a, b = template.read_bytes(), staged.read_bytes()
    head_a = a.split(b"\n")[0]
    head_b = b.split(b"\n")[0]
    ta, tb = pd.read_csv(template), pd.read_csv(staged)
    key_same = ta[KEY].astype(str).agg("|".join, axis=1).tolist() == \
        tb[KEY].astype(str).agg("|".join, axis=1).tolist()
    return {"header_identical": head_a == head_b,
            "rows": [len(ta), len(tb)],
            "row_count_identical": len(ta) == len(tb),
            "key_sequence_identical": key_same,
            "crlf": b.count(b"\r\n") == len(tb) + 1,
            "no_bom": b[:3] != b"\xef\xbb\xbf",
            "ends_with_newline": b.endswith(b"\n"),
            "pass": bool(head_a == head_b and len(ta) == len(tb) and key_same
                         and b[:3] != b"\xef\xbb\xbf" and b.endswith(b"\n"))}


def gate_g5b(buy: pd.DataFrame, sell: pd.DataFrame) -> dict:
    """The sell price may never exceed the buy price, hour by hour.

    Checked only on the EU zones: the five cost-based zones carry one value in
    both files by construction, so equality there is intended, not a breach.
    """
    b = buy[buy["zext"].isin(EU_ZONES)].set_index(KEY)[HOURS]
    s = sell[sell["zext"].isin(EU_ZONES)].set_index(KEY)[HOURS]
    b, s = b.align(s, join="inner")
    spread = (b - s).to_numpy()
    return {"min_spread_usd": float(spread.min()),
            "mean_spread_usd": float(spread.mean()),
            "n_violations": int((spread < 0).sum()),
            "pass": bool(spread.min() >= 0)}


def gate_g5c(frames: Dict[str, pd.DataFrame]) -> dict:
    """No written value may round to zero at DECIMALS.

    main.gms:894-895 reads a zero as 'no trade this hour' and switches off
    sExportPrice and sImportPrice together. A price that genuinely must be zero
    has to be expressed by removing the transfer limit, never by writing a zero
    into this file.
    """
    out, bad = {}, []
    for name, df in frames.items():
        v = np.round(df[HOURS].to_numpy(dtype=float), DECIMALS)
        n = int((v == 0).sum())
        out[name] = {"n_zero": n, "min_abs": float(np.abs(v).min())}
        if n:
            bad.append(name)
    return {"decimals": DECIMALS, "detail": out, "offending": bad,
            "pass": not bad}


# ── DIFF ────────────────────────────────────────────────────────────────────
def diff_report(template: pd.DataFrame, staged: Dict[str, Dict[str, pd.DataFrame]],
                reference: str | None) -> str:
    """What changes against the flat 70 USD/MWh currently in the model."""
    L: List[str] = []
    L.append("# P5 - staged trade prices vs the live data_blacksea file\n")
    L.append("Live file: every external zone flat across all hours and years - "
             "Bulgaria, Greece, Romania and Kazakhstan at 70 USD/MWh, Iran, "
             "Iraq and Syria at 40, Russia at 35.\n")
    L.append("Only Bulgaria, Greece and Romania are rewritten. The other five "
             "external zones have no liquid hub, no TYNDP entry and no ETS; "
             "they keep their cost-based values and are copied through "
             "unchanged.\n")

    base = {z: float(template[template["zext"] == z][HOURS].to_numpy().mean())
            for z in EU_ZONES}

    L.append("\n## Annual mean, USD 2024 per MWh\n")
    L.append("Buy side (`pTradePrice`), which is the only file the model reads "
             "today.\n")
    L.append("| scenario | zone | live | 2026 | 2030 | 2040 | 2050 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, pair in staged.items():
        for z in EU_ZONES:
            g = pair["buy"]
            g = g[g["zext"] == z]
            cells = []
            for y in (2026, 2030, 2040, 2050):
                v = g[g["year"] == y][HOURS].to_numpy()
                cells.append(f"{v.mean():.1f}" if v.size else "-")
            tag = f"**{name}**" if name == reference else name
            L.append(f"| {tag} | {z} | {base[z]:.0f} | " + " | ".join(cells) + " |")

    L.append("\n## What the sell side costs, if the export patch does not land\n")
    L.append("`base.gms:686` credits exports at the import price. Until "
             "`pTradePriceExport` exists, the model values every exported MWh "
             "at the buy price - the gap below is the per-MWh overstatement of "
             "export revenue.\n")
    L.append("| scenario | zone | mean buy | mean sell | overstatement |")
    L.append("|---|---|---|---|---|")
    for name, pair in staged.items():
        for z in EU_ZONES:
            b = pair["buy"]
            s = pair["sell"]
            bv = b[b["zext"] == z][HOURS].to_numpy().mean()
            sv = s[s["zext"] == z][HOURS].to_numpy().mean()
            L.append(f"| {name} | {z} | {bv:.1f} | {sv:.1f} | "
                     f"**+{bv - sv:.1f}** |")

    L.append("\n## Untouched zones\n")
    L.append("| zone | value | basis |")
    L.append("|---|---|---|")
    for z in sorted(set(template["zext"]) - set(EU_ZONES)):
        v = template[template["zext"] == z][HOURS].to_numpy()
        L.append(f"| {z} | {v.mean():.0f} | cost-based, unchanged |")
    return "\n".join(L) + "\n"


# ── Driver ──────────────────────────────────────────────────────────────────
def run(netback_csv: Path, template_csv: Path, scenarios_yaml: Path,
        out_dir: Path, log=print) -> dict:
    doc = load_scenarios(scenarios_yaml)
    scen, reference = doc["scenarios"], doc["reference"]

    nb = pd.read_csv(netback_csv)
    template = read_template(template_csv)
    log(f"  template {template_csv.name}: {len(template)} rows, "
        f"{template['zext'].nunique()} external zones")
    log(f"  netback: {len(nb)} rows, trajectories "
        f"{sorted(nb['scenario'].unique())}")

    if out_dir.exists():
        shutil.rmtree(out_dir)          # staging is derived; never hand-edited

    staged, flat, g5a = {}, {}, {}
    for name, spec in scen.items():
        lvl, var = spec["level"], spec["variant"]
        buy = apply_prices(template, pivot_netback(nb, lvl, var, "import"))
        sell = apply_prices(template, pivot_netback(nb, lvl, var, "export"))

        d = out_dir / name
        write_csv(buy, d / "pTradePrice.csv")
        write_csv(sell, d / "pTradePriceExport.csv")

        staged[name] = {"buy": buy, "sell": sell}
        flat[f"{name}/pTradePrice"] = buy
        flat[f"{name}/pTradePriceExport"] = sell
        g5a[name] = gate_g5a(template_csv, d / "pTradePrice.csv")

        eu = buy[buy["zext"].isin(EU_ZONES)][HOURS].to_numpy()
        log(f"    {name:20s} {lvl:9s}/{var:4s}  buy mean {eu.mean():6.1f} "
            f"USD/MWh  -> {d.name}/")

    g5b = {n: gate_g5b(p["buy"], p["sell"]) for n, p in staged.items()}
    g5c = gate_g5c(flat)

    (out_dir / "DIFF.md").write_text(
        diff_report(template, staged, reference), encoding="utf-8")

    bad_a = [n for n, g in g5a.items() if not g["pass"]]
    bad_b = [n for n, g in g5b.items() if not g["pass"]]
    log(f"\n  wrote {len(scen)} scenario pairs to {out_dir}")
    log(f"  wrote {out_dir / 'DIFF.md'}")
    log(f"\n  G5a structure identical  "
        f"{'PASS' if not bad_a else 'FAIL ' + ', '.join(bad_a)}")
    log(f"  G5b sell <= buy          "
        f"{'PASS' if not bad_b else 'FAIL ' + ', '.join(bad_b)} "
        f"(min spread {min(g['min_spread_usd'] for g in g5b.values()):+.2f} USD/MWh)")
    log(f"  G5c none rounds to zero  "
        f"{'PASS' if g5c['pass'] else 'FAIL ' + ', '.join(g5c['offending'])} "
        f"(at {DECIMALS} decimals)")

    return {"G5a": g5a, "G5b": g5b, "G5c": g5c,
            "failed": (["G5a"] if bad_a else []) + (["G5b"] if bad_b else [])
                      + ([] if g5c["pass"] else ["G5c"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage pTradePrice / pTradePriceExport from netback.csv.")
    ap.add_argument("--netback", type=Path,
                    default=_PRE_ANALYSIS / "output_prices" / "netback.csv")
    ap.add_argument("--template", type=Path,
                    default=_PRE_ANALYSIS.parent / "epm" / "input"
                    / "data_blacksea" / "trade" / "pTradePrice.csv")
    ap.add_argument("--scenarios", type=Path,
                    default=_PRE_ANALYSIS / "config" / "eu_price_scenarios.yaml")
    ap.add_argument("--out-dir", type=Path,
                    default=_PRE_ANALYSIS / "output_prices" / "staging")
    args = ap.parse_args(argv)

    print("[build_trade_prices] staging only - nothing under epm/input is written")
    qc = run(args.netback, args.template, args.scenarios, args.out_dir)
    if qc["failed"]:
        print("[build_trade_prices] gates not met: " + ", ".join(qc["failed"]))
        return 1
    print("[build_trade_prices] all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
