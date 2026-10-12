# -*- coding: utf-8 -*-
"""BSSC and GECO corridors with Romania internal (RoBg family, step 12).

In the base family Romania is an external zone, so the Black Sea cables are
external lines in pExtTransferLimit_bssc.csv and pExtTransferLimit_geco.csv, and
their loss sits in the Romanian export price. In the RoBg family Romania is
modelled, so the same lines become internal links with an explicit loss factor.

Two scenarios, both copies of LC_Baseline_RoBg:

* LC_BSSC_RoBg: Georgia - Romania, read from pExtTransferLimit_bssc.csv.
* LC_GECO_RoBg: Georgia - Romania and GEC_GE - Romania, read from
  pExtTransferLimit_geco.csv, plus the GEC_AZ and GEC_GE hub zones
  (zcmap_robg_geco.csv = zcmap_robg.csv + the GEC rows of zcmap_geco.csv).
  The hub links inside the Caucasus are already in pTransferLimit_robg.csv.

Capacities are copied year by year and quarter by quarter: the Export row of
(z, Romania) gives z -> Romania, the Import row gives Romania -> z. Loss factor
LOSS on both cables, both directions: the 5% submarine cable value of the EU price
netback (eu_price_netback.py, lambda bareme of 2026-08-19), which carried this
loss in the base family.

Outputs (all new files, read by the two new scenarios only):
  zcmap_robg_geco.csv
  trade/pTransferLimit_robg_bssc.csv, trade/pTransferLimit_robg_geco.csv
  trade/pLossFactorInternal_robg_bssc.csv, trade/pLossFactorInternal_robg_geco.csv
  scenarios.csv: columns LC_BSSC_RoBg and LC_GECO_RoBg appended, every other
  cell left byte for byte.

Idempotent: a second run rewrites the same files and leaves scenarios.csv as is.

Usage (from EPM/):  python pre-analysis/pipelines/build_corridors_robg.py
"""
from __future__ import annotations

import csv
from pathlib import Path

_HERE = Path(__file__).resolve()
_EPM = _HERE.parents[2]
DATA = _EPM / "epm" / "input" / "data_blacksea"
TRADE = DATA / "trade"

BASE_SCENARIO = "LC_Baseline_RoBg"
RO = "Romania"
LOSS = 0.05
CASES = {
    "bssc": dict(scenario="LC_BSSC_RoBg", ext=TRADE / "pExtTransferLimit_bssc.csv",
                 links=["Georgia"], zcmap=None),
    "geco": dict(scenario="LC_GECO_RoBg", ext=TRADE / "pExtTransferLimit_geco.csv",
                 links=["Georgia", "GEC_GE"], zcmap="zcmap_robg_geco.csv"),
}


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_corridors_robg] FAIL: {msg}")


def _read(path: Path) -> tuple[bool, str, list[str]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    eol = "\r\n" if "\r\n" in text else "\n"
    return raw.startswith(b"\xef\xbb\xbf"), eol, text.split(eol)


def _write(path: Path, bom: bool, eol: str, lines: list[str]) -> None:
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + eol.join(lines).encode("utf-8"))


def _body(lines: list[str]) -> tuple[list[str], bool]:
    trailing = bool(lines) and lines[-1] == ""
    return (lines[:-1] if trailing else lines), trailing


def ext_capacity(path: Path, z: str) -> dict[tuple[str, str], dict[str, float]]:
    """(direction, q) -> {year: MW} for the (z, Romania) rows of an external limit file."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    years = [(i, c) for i, c in enumerate(head) if c.strip().isdigit()]
    out = {}
    for r in rows[1:]:
        if len(r) > 3 and r[0] == z and r[1] == RO:
            out[(r[3], r[2])] = {y: float(r[i] or 0) for i, y in years}
    if sorted(k[0] for k in out) != ["Export"] * 4 + ["Import"] * 4:
        _fail(f"{path.name}: expected 4 Export and 4 Import rows for {z},{RO}, got {sorted(out)}")
    return out


def transfer_file(case: dict, out: Path) -> None:
    bom, eol, lines = _read(TRADE / "pTransferLimit_robg.csv")
    body, trailing = _body(lines)
    head = body[0].split(",")
    years = [c for c in head[3:] if c.strip()]
    existing = {tuple(ln.split(",")[:2]) for ln in body[1:]}
    new = []
    for z in case["links"]:
        if (z, RO) in existing or (RO, z) in existing:
            _fail(f"pTransferLimit_robg.csv already holds {z} - {RO}")
        cap = ext_capacity(case["ext"], z)
        for direction, a, b in (("Export", z, RO), ("Import", RO, z)):
            for q in ("Q1", "Q2", "Q3", "Q4"):
                v = cap[(direction, q)]
                if any(y not in v for y in years):
                    _fail(f"{case['ext'].name}: year missing for {z} {direction} {q}")
                new.append(",".join([a, b, q] + [f"{v[y]:.1f}" for y in years]))
    _write(out, bom, eol, body + new + ([""] if trailing else []))
    print(f"  {out.relative_to(_EPM)}: {len(new)} row(s) added to pTransferLimit_robg.csv")


def loss_file(case: dict, out: Path) -> None:
    bom, eol, lines = _read(TRADE / "pLossFactorInternal_robg.csv")
    body, trailing = _body(lines)
    n_years = len(body[0].split(",")) - 2
    new = []
    for z in case["links"]:
        for a, b in ((z, RO), (RO, z)):
            new.append(",".join([a, b] + [str(LOSS)] * n_years))
    _write(out, bom, eol, body + new + ([""] if trailing else []))
    print(f"  {out.relative_to(_EPM)}: {len(new)} row(s) added to pLossFactorInternal_robg.csv")


def zcmap_file(out: Path) -> None:
    bom, eol, lines = _read(DATA / "zcmap_robg.csv")
    body, trailing = _body(lines)
    with open(DATA / "zcmap_geco.csv", newline="", encoding="utf-8-sig") as f:
        gec = [",".join(r) for r in csv.reader(f) if r and r[0].startswith("GEC_")]
    if len(gec) != 2:
        _fail(f"zcmap_geco.csv: expected 2 GEC rows, got {gec}")
    _write(out, bom, eol, body + gec + ([""] if trailing else []))
    print(f"  {out.relative_to(_EPM)}: {len(gec)} row(s) added to zcmap_robg.csv")


def scenarios() -> None:
    path = DATA / "scenarios.csv"
    bom, eol, lines = _read(path)
    body, trailing = _body(lines)
    rows = [ln.split(",") for ln in body]
    head = rows[0]
    if any(len(r) != len(head) for r in rows):
        _fail("scenarios.csv: ragged rows or quoted commas, edit by hand")
    names = [c["scenario"] for c in CASES.values()]
    present = [n for n in names if n in head]
    if present:
        if present != names:
            _fail(f"scenarios.csv: only {present} present")
        print("  scenarios.csv: columns already there")
        return
    i = head.index(BASE_SCENARIO)
    keys = [r[0] for r in rows]
    for k in ("pTransferLimit", "pLossFactorInternal", "zcmap"):
        if k not in keys:
            _fail(f"scenarios.csv: no row '{k}'")
    for key, case in CASES.items():
        over = {"pTransferLimit": f"trade/pTransferLimit_robg_{key}.csv",
                "pLossFactorInternal": f"trade/pLossFactorInternal_robg_{key}.csv"}
        if case["zcmap"]:
            over["zcmap"] = case["zcmap"]
        for r in rows[1:]:
            r.append(over.get(r[0], r[i]))
        head.append(case["scenario"])
    _write(path, bom, eol, [",".join(r) for r in rows] + ([""] if trailing else []))
    print(f"  scenarios.csv: columns {', '.join(names)} appended")


def main() -> int:
    for key, case in CASES.items():
        transfer_file(case, TRADE / f"pTransferLimit_robg_{key}.csv")
        loss_file(case, TRADE / f"pLossFactorInternal_robg_{key}.csv")
        if case["zcmap"]:
            zcmap_file(DATA / case["zcmap"])
    scenarios()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
