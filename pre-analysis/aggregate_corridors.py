"""
aggregate_corridors.py
======================
Aggregate the per-line reference_lines.csv into EPM-ready, corridor-level transmission
inputs — written to a STAGING folder (nothing in data_blacksea is overwritten).

Outputs (pre-analysis/output_transmission/):
  corridors.ref_generated.csv            zone-pair level summary (readable intermediate)
  pTransferLimit.ref_generated.csv        internal existing+committed, directional × quarter × year
  pNewTransmission.ref_generated.csv      internal candidates
  pLossFactorInternal.ref_generated.csv   from loss_pu, directional
  pExtTransferLimit.ref_generated.csv     internal-zone ↔ external-zone (zext), Import/Export
  sTopology.ref_generated.csv             corrected internal topology (TUR-GEO → EastAna)
  DIFF_report.txt                         vs current data_blacksea/trade/ + double-count flags

Anti double-count rule: lines of one corridor whose note says "combined" are a single
border total -> take MAX (not SUM). Genuinely distinct lines -> SUM (and flagged).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parent
_REF = _BASE / "data" / "reference_lines.csv"
_DB = _BASE.parent / "epm" / "input" / "data_blacksea"
_OUT = _BASE / "output_transmission"
_OUT.mkdir(exist_ok=True)

YEARS = [str(y) for y in range(2024, 2054)]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
_DEFAULT_LIFE = 30
_DEFAULT_LOSS = 0.0198


def _num(x):
    try:
        v = float(x)
        return v if v == v else None       # drop NaN
    except (TypeError, ValueError):
        return None


def _year(x):
    m = re.search(r"((?:19|20)\d\d)", str(x))
    return int(m.group(1)) if m else None


def _reduce(values: list[float]) -> float:
    """Sum genuinely distinct lines, but collapse a repeated border total to a
    single value: if all non-zero contributions are identical, it's the same
    border NTC stated on several lines (e.g. AZE-Iran '~600' on 4 lines) -> take it
    once. This is the anti-double-count rule."""
    vals = [v for v in values if v]                  # drop None / 0
    if not vals:
        return 0.0
    if len(vals) > 1 and len({round(v, 3) for v in vals}) == 1:
        return vals[0]                               # identical -> dedup
    return sum(vals)


def _rule(*lists) -> str:
    vals = [v for lst in lists for v in lst if v]
    if len(vals) > 1 and len({round(v, 3) for v in vals}) == 1:
        return "DEDUP (identical values — repeated border total)"
    if len(vals) > 1:
        return "SUM (distinct lines)"
    return "single"


def main() -> int:
    internal = set(pd.read_csv(_DB / "zcmap.csv")["z"].astype(str).str.strip())
    zext = set(pd.read_csv(_DB / "trade" / "zext.csv")["zext"].astype(str).str.strip())
    # Türkiye's 9 internal zones — their intra-country links live in the Türkiye EPM
    # model (not in the cross-border reference_lines), so they are preserved as-is.
    TUR9 = {"WestMed", "WestAna", "EastMed", "EastAna", "SouthEast",
            "NorthWest", "CenterAna", "CenterBlack", "Trakia"}
    df = pd.read_csv(_REF, comment="#")

    corr: dict = {}          # internal undirected corridor (zlo,zhi)
    ext: dict = {}           # external corridor (internal_zone, zext)
    candidates: list = []    # internal candidate lines -> pNewTransmission (Status 3)
    ext_candidates: list = []  # external candidate interconnectors -> separate file
    report: list[str] = []

    for _, r in df.iterrows():
        f, t = str(r.from_zone).strip(), str(r.to_zone).strip()
        fwd, rev = _num(r.mw_fwd), _num(r.mw_rev)
        status = str(r.status).strip().lower()
        combined = "combined" in str(r.get("note", "")).lower()
        loss = _num(r.loss_pu)
        cod = _year(r.get("earliest_entry"))

        if f in internal and t in internal:                    # ── internal corridor
            zlo, zhi = sorted([f, t])
            v_lohi, v_hilo = (fwd, rev) if f == zlo else (rev, fwd)
            c = corr.setdefault((zlo, zhi), dict(
                ex_lohi=[], ex_hilo=[], cm_lohi=[], cm_hilo=[],
                loss=[], combined=False, cod=None, n=0))
            c["combined"] |= combined
            c["n"] += 1
            if loss is not None:
                c["loss"].append(loss)
            if status == "existing":
                c["ex_lohi"].append(v_lohi); c["ex_hilo"].append(v_hilo)
            elif status == "committed":
                c["cm_lohi"].append(v_lohi); c["cm_hilo"].append(v_hilo)
                c["cod"] = cod or c["cod"]
            elif status == "candidate":
                candidates.append(dict(frm=f, to=t, cap=fwd, cod=cod,
                                       project=str(r.get("project", "")).strip()))

        elif (f in internal and t in zext) or (t in internal and f in zext):   # ── external
            zi, ze = (f, t) if f in internal else (t, f)
            exp = fwd if f == zi else rev        # internal -> external = export
            imp = rev if f == zi else fwd        # external -> internal = import
            e = ext.setdefault((zi, ze), dict(exp=[], imp=[], cm_exp=[], cm_imp=[], cod=None, n=0))
            e["n"] += 1
            if status == "existing":
                e["exp"].append(exp); e["imp"].append(imp)
            elif status == "committed":
                e["cm_exp"].append(exp); e["cm_imp"].append(imp); e["cod"] = cod or e["cod"]
            elif status == "candidate":
                # EPM cannot optimise external builds -> separate file for scenarios
                ext_candidates.append(dict(zint=zi, zext=ze, cap=fwd or rev, cod=cod,
                                           project=str(r.get("project", "")).strip()))
        # else: external↔external (RO-UA, BG-RS, …) — not modelled, skipped

    # ── corridors.ref_generated.csv ─────────────────────────────────────────────
    crows = []
    for (zlo, zhi), c in sorted(corr.items()):
        ex_l, ex_h = _reduce(c["ex_lohi"]), _reduce(c["ex_hilo"])
        cm_l, cm_h = _reduce(c["cm_lohi"]), _reduce(c["cm_hilo"])
        rule = _rule(c["ex_lohi"], c["ex_hilo"])
        crows.append(dict(z=zlo, z2=zhi, scope="internal",
                          existing_lohi=ex_l, existing_hilo=ex_h,
                          committed_lohi=cm_l, committed_hilo=cm_h,
                          committed_cod=c["cod"], n_lines=c["n"], agg_rule=rule,
                          loss_pu=round(sum(c["loss"]) / len(c["loss"]), 5) if c["loss"] else _DEFAULT_LOSS))
        if c["n"] > 1:
            report.append(f"[internal] {zlo}-{zhi}: {c['n']} lines -> {rule}  "
                          f"(existing {ex_l}/{ex_h}, committed {cm_l}/{cm_h}@{c['cod']})")
    for (zi, ze), e in sorted(ext.items()):
        exp, imp = _reduce(e["exp"]), _reduce(e["imp"])
        cmx, cmi = _reduce(e["cm_exp"]), _reduce(e["cm_imp"])
        rule = _rule(e["exp"], e["imp"])
        crows.append(dict(z=zi, z2=ze, scope="external",
                          existing_lohi=exp, existing_hilo=imp,
                          committed_lohi=cmx, committed_hilo=cmi, committed_cod=e["cod"],
                          n_lines=e["n"], agg_rule=rule, loss_pu=""))
        if e["n"] > 1:
            report.append(f"[external] {zi}-{ze}: {e['n']} lines -> {rule}  "
                          f"(export {exp}+{cmx}@{e['cod']}, import {imp}+{cmi})")
    pd.DataFrame(crows).to_csv(_OUT / "corridors.ref_generated.csv", index=False)

    # ── pTransferLimit.ref_generated.csv (internal existing+committed) ───────────
    tl = []
    committed_new = []          # committed NEW internal lines -> pNewTransmission Status 2
    for (zlo, zhi), c in sorted(corr.items()):
        ex = dict(lohi=_reduce(c["ex_lohi"]), hilo=_reduce(c["ex_hilo"]))
        cm = dict(lohi=_reduce(c["cm_lohi"]), hilo=_reduce(c["cm_hilo"]))
        cod = c["cod"] or 2024
        ex_tot = ex["lohi"] + ex["hilo"]
        cm_tot = cm["lohi"] + cm["hilo"]
        # committed on a NEW corridor (no existing) = new line -> pNewTransmission (Status 2);
        # committed on an EXISTING corridor = reinforcement -> pTransferLimit ramp.
        committed_is_new = (ex_tot == 0 and cm_tot > 0)
        if committed_is_new:
            committed_new.append(dict(frm=zlo, to=zhi, cap=max(cm["lohi"], cm["hilo"]), cod=c["cod"]))
        for (a, b, key) in [(zlo, zhi, "lohi"), (zhi, zlo, "hilo")]:
            ramp = 0 if committed_is_new else cm[key]   # reinforcement only
            if ex[key] == 0 and ramp == 0:
                continue
            for q in QUARTERS:
                row = {"z": a, "z2": b, "q": q}
                for y in YEARS:
                    row[y] = round(ex[key] + (ramp if int(y) >= cod else 0), 1)
                tl.append(row)
    # MERGE: keep intra-Türkiye links from the current model file (untouched),
    # add the cross-border corridors regenerated from reference_lines.
    cur_tl = pd.read_csv(_DB / "trade" / "pTransferLimit.csv")
    intra = cur_tl[cur_tl.z.isin(TUR9) & cur_tl.z2.isin(TUR9)]
    merged_tl = pd.concat([intra, pd.DataFrame(tl)], ignore_index=True)
    merged_tl.to_csv(_OUT / "pTransferLimit.ref_generated.csv", index=False)

    # ── pNewTransmission.ref_generated.csv (internal: candidates=3, committed-new=2) ──
    nt = [{"From": c["frm"], "To": c["to"], "EarliestEntry": c["cod"] or "",
           "MaximumNumOfLines": 1, "CapacityPerLine": c["cap"] or "",
           "CostPerLine": "", "Life": _DEFAULT_LIFE, "Status": 3}        # candidate
          for c in candidates]
    nt += [{"From": c["frm"], "To": c["to"], "EarliestEntry": c["cod"] or "",
            "MaximumNumOfLines": 1, "CapacityPerLine": c["cap"] or "",
            "CostPerLine": "", "Life": _DEFAULT_LIFE, "Status": 2}        # committed new line
           for c in committed_new]
    pd.DataFrame(nt, columns=["From", "To", "EarliestEntry", "MaximumNumOfLines",
                              "CapacityPerLine", "CostPerLine", "Life", "Status"]
                 ).to_csv(_OUT / "pNewTransmission.ref_generated.csv", index=False)

    # ── pNewTransmissionExt.ref_generated.csv (external candidate interconnectors) ──
    # EPM cannot optimise external builds -> not in base; use to build scenario overlays
    # (ramp pExtTransferLimit from COD in scenarios ③/④).
    nte = [{"From": c["zint"], "To": c["zext"], "EarliestEntry": c["cod"] or "",
            "CapacityPerLine": c["cap"] or "", "Project": c["project"], "Status": 3}
           for c in ext_candidates]
    pd.DataFrame(nte, columns=["From", "To", "EarliestEntry", "CapacityPerLine",
                               "Project", "Status"]
                 ).to_csv(_OUT / "pNewTransmissionExt.ref_generated.csv", index=False)

    # ── pLossFactorInternal.ref_generated.csv ────────────────────────────────────
    lf = []
    for (zlo, zhi), c in sorted(corr.items()):
        loss = round(sum(c["loss"]) / len(c["loss"]), 5) if c["loss"] else _DEFAULT_LOSS
        for a, b in [(zlo, zhi), (zhi, zlo)]:
            lf.append({"z": a, "z2": b, **{y: loss for y in YEARS}})
    cur_lf = pd.read_csv(_DB / "trade" / "pLossFactorInternal.csv")
    intra_lf = cur_lf[cur_lf.z.isin(TUR9) & cur_lf.z2.isin(TUR9)]
    pd.concat([intra_lf, pd.DataFrame(lf)], ignore_index=True
              ).to_csv(_OUT / "pLossFactorInternal.ref_generated.csv", index=False)

    # ── pExtTransferLimit.ref_generated.csv ──────────────────────────────────────
    et = []
    for (zi, ze), e in sorted(ext.items()):
        exp, imp = _reduce(e["exp"]), _reduce(e["imp"])
        cmx, cmi = _reduce(e["cm_exp"]), _reduce(e["cm_imp"])
        cod = e["cod"] or 2024
        for q in QUARTERS:
            et.append({"z": zi, "zext": ze, "q": q, "": "Import",
                       **{y: round(imp + (cmi if int(y) >= cod else 0), 1) for y in YEARS}})
            et.append({"z": zi, "zext": ze, "q": q, "": "Export",
                       **{y: round(exp + (cmx if int(y) >= cod else 0), 1) for y in YEARS}})
    pd.DataFrame(et).to_csv(_OUT / "pExtTransferLimit.ref_generated.csv", index=False)

    # ── sTopology.ref_generated.csv ─────────────────────────────────────────────
    # Patch the CURRENT topology (keep intra-Türkiye + Inter_* pseudo-nodes), only
    # fix the TUR-GEO edge: CenterBlack↔Georgia → EastAna↔Georgia.
    st = pd.read_csv(_DB / "extras" / "sTopology.csv")
    c0, c1 = st.columns[0], st.columns[1]
    fixed = 0
    for i, row in st.iterrows():
        pair = {str(row[c0]).strip(), str(row[c1]).strip()}
        if pair == {"CenterBlack", "Georgia"}:
            other = "Georgia" if str(row[c0]).strip() == "Georgia" else "EastAna"
            st.at[i, c0] = "EastAna" if str(row[c0]).strip() == "CenterBlack" else "Georgia"
            st.at[i, c1] = "Georgia" if str(row[c1]).strip() == "Georgia" else "EastAna"
            fixed += 1
    st = st.drop_duplicates()
    st.to_csv(_OUT / "sTopology.ref_generated.csv", index=False)
    report.append(f"\nsTopology: patched {fixed} CenterBlack-Georgia edge(s) -> EastAna-Georgia "
                  f"(intra-Türkiye & Inter_* nodes preserved).")

    # ── DIFF report ──────────────────────────────────────────────────────────────
    lines = ["=" * 70, "AGGREGATION REPORT (staging — nothing overwritten)", "=" * 70,
             f"internal corridors: {len(corr)} | external corridors: {len(ext)} | "
             f"candidates: {len(candidates)}", "",
             "Multi-line corridors (double-count rule applied):"]
    lines += ["  " + r for r in report] or ["  (none)"]
    # quick diff vs current internal pTransferLimit (Q1, 2025)
    lines += ["", "pTransferLimit Q1-2025 vs current data_blacksea:"]
    cur = cur_tl[cur_tl.q == "Q1"].set_index(["z", "z2"])["2025"].to_dict()
    mq = merged_tl[merged_tl.q == "Q1"]
    new = {(r.z, r.z2): r["2025"] for _, r in mq.iterrows()}
    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    keys = sorted(set(cur) | set(new))
    for k in keys:
        a, b = cur.get(k), new.get(k)
        fa, fb = _f(a), _f(b)
        if fa is None or fb is None or abs(fa - fb) > 0.5:   # ignore float-format noise
            tag = " [NEW]" if k not in cur else (" [removed]" if k not in new else "")
            lines.append(f"  {k[0]}->{k[1]:16s} current={a}  generated={b}{tag}")
    (_OUT / "DIFF_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote 7 files to {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
