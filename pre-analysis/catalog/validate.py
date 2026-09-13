"""
Validate the data source catalog and its links to provenance files.

The catalog convention is stated in catalog/README.md: every file that adds or
updates an EPM CSV must add or update the matching catalog entry in the same
commit. Nothing enforced it, so 21 of 29 entries had drifted off the schema
before this script existed. Run it in CI so the drift cannot come back.

A deployment that carries CESI GEC data (cesi/cesi_register.yaml, DVC only) gets
three more checks, skipped when the register is absent:
- geco: the geco blocks of provenance.yaml match the CESI aligned files. Every
  zone where a cesi/*_cesi.csv differs from its config.csv reference has a
  block, and every block points at a real difference.
- leak: no register value appears in the text that feeds a git tracked file
  (provenance entries citing the study, its catalog entry, DATA_SOURCES).
- traceability: a register entry the main scenario has taken or blended is
  cited in provenance.yaml with a CHANGED line in its method_table.

Usage:
    python pre-analysis/catalog/validate.py            # from the repo root
    python pre-analysis/catalog/validate.py --strict   # warnings are failures

Exit code 0 = clean, 1 = at least one error.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import jsonschema
    import yaml
except ImportError as exc:                                    # pragma: no cover
    sys.exit(f"Missing dependency ({exc.name}). pip install jsonschema pyyaml")

_CATALOG = Path(__file__).resolve().parent
_REPO_ROOT = _CATALOG.parents[1]

GECO_SOURCE = "cesi_gec_feasibility"
GLOBAL_KEY = "global"                  # provenance section of the model-wide geco blocks
# cesi/<param>_cesi.csv -> DATA_SOURCES row that carries its geco block
GECO_ROW = {"pGenDataInputDefault": "pGenDataInput", "pCapexTrajectoriesDefault": "pGenDataInput",
            "pExtTransferLimit": "pTransferLimit", "pTradePriceExport": "pTradePrice"}
ZONE_COLS = ("z", "z2", "zone", "zext")
# Leak check: register numbers with at least this many significant digits. Years and 8760 are
# never distinctive. DATA_SOURCES is long enough for 3 digit coincidences, so it gets 4.
LEAK_SIG, LEAK_SIG_DOCS = 3, 4
LEAK_SKIP = {8760}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_schema(errors: list[str]) -> dict[str, dict]:
    """Every sources/*.yaml validates, and its id matches its filename."""
    schema = json.loads((_CATALOG / "schema" / "source.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(schema)
    entries: dict[str, dict] = {}

    for path in sorted((_CATALOG / "sources").glob("*.yaml")):
        entry = _load_yaml(path)
        for err in sorted(validator.iter_errors(entry), key=lambda e: list(e.path)):
            where = "/".join(map(str, err.path)) or "(root)"
            errors.append(f"{path.name} · {where} — {err.message}")
        if entry.get("id") != path.stem:
            errors.append(
                f"{path.name} — id '{entry.get('id')}' does not match the filename. "
                f"source_id citations resolve by filename."
            )
        entries[path.stem] = entry
    return entries


def find_provenance_files() -> list[Path]:
    return sorted((_REPO_ROOT / "epm" / "input").glob("*/provenance.yaml"))


def check_citations(entries: dict[str, dict], errors: list[str], warnings: list[str]) -> None:
    """Every source_id cited in a provenance file resolves to a catalog entry.

    A citation counts whether it sits in source_id or in secondary_source_ids: a
    source used for one fuel of a block but not the headline one is still used.
    Reading only source_id made twelve entries look orphaned when they were not.
    A geco block cites the CESI study. An entry holding nothing but a geco block
    documents no source of ours, so it needs no source_id.
    """
    cited: dict[str, list[str]] = defaultdict(list)

    for prov_path in find_provenance_files():
        deployment = prov_path.parent.name
        for country, block in _load_yaml(prov_path).items():
            if not isinstance(block, dict):
                continue
            for resource, item in block.items():
                if not isinstance(item, dict):
                    continue
                where = f"{deployment} · {country} · {resource}"
                if "geco" in item:
                    cited[GECO_SOURCE].append(f"{where} (geco)")
                if not set(item) - {"geco"}:
                    continue
                primary = item.get("source_id")
                if not primary:
                    warnings.append(f"{where} — no source_id")
                secondary = item.get("secondary_source_ids") or []
                if isinstance(secondary, str):                # a bare scalar, not a list
                    secondary = [secondary]
                for sid in ([primary] if isinstance(primary, str) else list(primary or [])):
                    cited[sid].append(where)
                for sid in secondary:
                    cited[sid].append(f"{where} (secondary)")

    for sid, where in sorted(cited.items()):
        if sid not in entries:
            errors.append(
                f"source_id '{sid}' cited {len(where)}x but missing from catalog/sources/ "
                f"— e.g. {where[0]}"
            )

    for unused in sorted(set(entries) - set(cited)):
        warnings.append(f"catalog/sources/{unused}.yaml — never cited by any provenance.yaml")


# ── CESI GEC checks ───────────────────────────────────────────────────────────

def _as_list(x) -> list:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _zcmap(dep_dir: Path, errors: list[str] | None = None) -> dict[str, str]:
    """Zone to country over zcmap.csv and its scenario variants (zcmap_*.csv), so zones that only
    some scenarios read, such as the GEC hub, resolve to their country. zcmap.csv wins, and a zone
    mapped to two countries is an error."""
    zc: dict[str, str] = {}
    for path in [dep_dir / "zcmap.csv"] + sorted(dep_dir.glob("zcmap_*.csv")):
        if not path.exists():
            continue
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                z, c = (r.get("z") or "").strip(), (r.get("c") or "").strip()
                if not z:
                    continue
                if z in zc and zc[z] != c:
                    if errors is not None:
                        errors.append(f"{dep_dir.name} · {path.name}: zone {z} maps to {c}, "
                                      f"but to {zc[z]} in an earlier zcmap file")
                    continue
                zc[z] = c
    return zc


def _config(dep_dir: Path) -> dict[str, Path]:
    path = dep_dir / "config.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {r["paramNames"].strip(): dep_dir / r["file"].strip()
                for r in csv.DictReader(fh) if (r.get("paramNames") or "").strip() and (r.get("file") or "").strip()}


def _keys(zone: str, zc: dict[str, str]) -> set[str]:
    """Provenance sections that may hold the entry of a zone: the zone, or its country."""
    return {GLOBAL_KEY} if zone == "all" else {zone, zc.get(zone, zone)}


def _geco_blocks(prov: dict) -> dict[tuple[str, str], dict]:
    return {(key, row): item["geco"]
            for key, block in prov.items() if isinstance(block, dict)
            for row, item in block.items() if isinstance(item, dict) and isinstance(item.get("geco"), dict)}


def _norm(cell: str) -> str:
    cell = cell.strip()
    try:
        return repr(float(cell))
    except ValueError:
        return cell


def diff_zones(ref: Path, new: Path) -> set[str]:
    """Zones of the rows where a CESI aligned file differs from its reference. 'all' if it has no zone column."""
    def rows(p):
        with open(p, encoding="utf-8-sig", newline="") as fh:
            r = list(csv.reader(fh))
        return [c.strip() for c in r[0]], [tuple(_norm(c) for c in x) for x in r[1:] if any(c.strip() for c in x)]

    ha, ra = rows(ref)
    hb, rb = rows(new)
    if ha != hb:
        raise ValueError("header differs from the reference, the file must be a copy of it")
    changed = (Counter(ra) - Counter(rb)) + (Counter(rb) - Counter(ra))
    if not changed:
        return set()
    idx = [i for i, h in enumerate(ha) if h in ZONE_COLS]
    if not idx:
        return {"all"}
    return {row[i] for row in changed for i in idx if i < len(row) and row[i]}


def geco_findings(dep_dir: Path, prov: dict, register: dict, zc: dict, config: dict) -> tuple[list[str], list[str]]:
    """Errors and warnings of the geco blocks of one deployment against its CESI aligned files."""
    errs: list[str] = []
    warns: list[str] = []
    dep = dep_dir.name
    blocks = _geco_blocks(prov)

    diffs: dict[str, set[str]] = {}
    for path in sorted((dep_dir / "cesi").glob("*_cesi.csv")):
        rel = f"cesi/{path.name}"
        stem = path.stem[: -len("_cesi")]
        ref = config.get(stem)
        if ref is None or not ref.exists():
            errs.append(f"{dep} · {rel}: no reference file for {stem} in config.csv")
            continue
        try:
            diffs[rel] = diff_zones(ref, path)
        except ValueError as exc:
            errs.append(f"{dep} · {rel}: {exc}")
            continue
        if not diffs[rel]:
            warns.append(f"{dep} · {rel}: identical to {ref.relative_to(dep_dir).as_posix()}")
        row = GECO_ROW.get(stem, stem)
        for z in sorted(diffs[rel]):
            keys = _keys(z, zc)
            if not any(rel in _as_list(blocks.get((k, row), {}).get("file")) for k in keys):
                errs.append(f"{dep} · {rel}: differs in {z}, but no geco block under "
                            f"{' or '.join(sorted(keys))} · {row} names this file")

    missing = set()
    for (key, row), g in sorted(blocks.items()):
        where = f"{dep} · {key} · {row} · geco"
        files, ids = _as_list(g.get("file")), _as_list(g.get("ids"))
        if not files or not ids or not g.get("ref") or not g.get("scope"):
            errs.append(f"{where}: file, ref, scope and ids are all required")
        for rid in ids:
            e = register.get(rid)
            if e is None:
                errs.append(f"{where}: id {rid} is not in cesi/cesi_register.yaml")
                continue
            zones = _as_list(e.get("zone"))
            if not any(key in _keys(z, zc) for z in zones):
                errs.append(f"{where}: {rid} is a {', '.join(zones)} entry")
            if GECO_ROW.get(e.get("param"), e.get("param")) != row:
                errs.append(f"{where}: {rid} feeds {e.get('param')}, not {row}")
        for f in files:
            if f not in diffs:
                if not (dep_dir / f).exists():
                    missing.add(f)
                continue
            if not any(key in _keys(z, zc) for z in diffs[f]):
                errs.append(f"{where}: {f} does not differ from its reference in {key}")
    if missing:
        warns.append(f"{dep}: {len(missing)} CESI aligned file(s) named by geco blocks not built yet: "
                     f"{', '.join(sorted(missing))}")
    return errs, warns


def _numbers(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _numbers(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _numbers(v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield x


def _sig(v) -> int:
    return len(("%g" % abs(v)).replace(".", "").lstrip("0").rstrip("0"))


def leak_patterns(register: dict, min_sig: int) -> list[tuple[re.Pattern, str]]:
    """One pattern per distinctive register number, in its usual written forms, with the id it comes from."""
    vals: dict[float, str] = {}
    for rid, e in register.items():
        for v in list(_numbers(e.get("cesi_value"))) + list(_numbers(e.get("detail"))):
            if (isinstance(v, int) and 1900 <= v <= 2100) or v in LEAK_SKIP or _sig(v) < min_sig:
                continue
            vals.setdefault(v, rid)
    out = []
    for v, rid in vals.items():
        if float(v).is_integer():
            n = int(v)
            forms = {str(n)} | ({f"{n:,}", f"{n:,}".replace(",", " ")} if n >= 1000 else set())
        else:
            forms = {"%g" % v, ("%g" % v).replace(".", ",")}
        alt = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
        out.append((re.compile(r"(?<![\d.,])(?:" + alt + r")(?![\d]|[.,]\d)"), rid))
    return out


def _cites(item: dict, sid: str) -> bool:
    return sid in _as_list(item.get("source_id")) + _as_list(item.get("secondary_source_ids"))


def leak_findings(dep_dir: Path, prov: dict, register: dict) -> list[str]:
    """Places where a register value reaches text that ends up in a git tracked file."""
    pats, pats_docs = leak_patterns(register, LEAK_SIG), leak_patterns(register, LEAK_SIG_DOCS)
    texts = []
    for key, block in prov.items():
        if not isinstance(block, dict):
            continue
        for row, item in block.items():
            if isinstance(item, dict) and ("geco" in item or _cites(item, GECO_SOURCE)):
                texts.append((f"provenance.yaml · {key} · {row}", json.dumps(item, default=str, ensure_ascii=False), pats))
    cat = _CATALOG / "sources" / f"{GECO_SOURCE}.yaml"
    if cat.exists():
        texts.append((f"catalog/sources/{cat.name}", cat.read_text(encoding="utf-8"), pats))
    for doc in [dep_dir / "DATA_SOURCES.md", dep_dir / "DATA_SOURCES.html",
                _REPO_ROOT / "pre-analysis" / "output_catalog" / dep_dir.name / "DATA_SOURCES.md",
                _REPO_ROOT / "pre-analysis" / "output_catalog" / dep_dir.name / "DATA_SOURCES.html"]:
        if doc.exists():
            texts.append((doc.relative_to(_REPO_ROOT).as_posix(), doc.read_text(encoding="utf-8"), pats_docs))
    errs = []
    for where, text, ps in texts:
        hits = sorted({rid for p, rid in ps if p.search(text)})
        if hits:
            errs.append(f"{dep_dir.name} · {where}: holds a CESI value (register {', '.join(hits)}). "
                        f"CESI figures stay in cesi/, the text cites the report page only")
    return errs


def traceability_findings(prov: dict, register: dict, zc: dict) -> list[str]:
    """A register entry the main scenario takes or blends must be cited, with a CHANGED line."""
    errs = []
    for rid, e in register.items():
        if e.get("main") not in ("taken", "blend"):
            continue
        row = GECO_ROW.get(e.get("param"), e.get("param"))
        for z in _as_list(e.get("zone")):
            keys = sorted(_keys(z, zc))
            items = [prov[k].get(row) for k in keys if isinstance(prov.get(k), dict)]
            if not any(isinstance(i, dict) and _cites(i, GECO_SOURCE) and
                       any(re.search(r"(?<!UN)CHANGED", json.dumps(m, default=str))
                           for m in i.get("method_table") or [])
                       for i in items):
                errs.append(f"register {rid} is '{e['main']}' in the main scenario, but no "
                            f"{' or '.join(keys)} · {row} entry cites {GECO_SOURCE} with a CHANGED "
                            f"line in its method_table")
    return errs


def check_cesi(errors: list[str], warnings: list[str]) -> None:
    for prov_path in find_provenance_files():
        dep_dir = prov_path.parent
        reg_path = dep_dir / "cesi" / "cesi_register.yaml"
        prov = _load_yaml(prov_path)
        if not reg_path.exists():
            if _geco_blocks(prov):
                warnings.append(f"{dep_dir.name}: geco blocks not checked, cesi/cesi_register.yaml "
                                f"is absent (DVC only, run dvc pull)")
            continue
        register = {e["id"]: e for e in (_load_yaml(reg_path).get("entries") or [])}
        zc = _zcmap(dep_dir, errors)
        errs, warns = geco_findings(dep_dir, prov, register, zc, _config(dep_dir))
        errors += errs + leak_findings(dep_dir, prov, register) + traceability_findings(prov, register, zc)
        warnings += warns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    entries = check_schema(errors)
    check_citations(entries, errors, warnings)
    check_cesi(errors, warnings)

    prov_count = len(find_provenance_files())
    print(f"catalog: {len(entries)} entries · {prov_count} provenance file(s)\n")

    for w in warnings:
        print(f"  WARNING  {w}")
    for e in errors:
        print(f"  ERROR    {e}")

    if errors:
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s) — FAILED")
        return 1
    if warnings and args.strict:
        print(f"\n{len(warnings)} warning(s) under --strict — FAILED")
        return 1
    print(f"\nOK — 0 errors, {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
