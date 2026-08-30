"""
Validate the data source catalog and its links to provenance files.

The catalog convention is stated in catalog/README.md: every file that adds or
updates an EPM CSV must add or update the matching catalog entry in the same
commit. Nothing enforced it, so 21 of 29 entries had drifted off the schema
before this script existed. Run it in CI so the drift cannot come back.

Usage:
    python pre-analysis/catalog/validate.py            # from the repo root
    python pre-analysis/catalog/validate.py --strict   # warnings are failures

Exit code 0 = clean, 1 = at least one error.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import jsonschema
    import yaml
except ImportError as exc:                                    # pragma: no cover
    sys.exit(f"Missing dependency ({exc.name}). pip install jsonschema pyyaml")

_CATALOG = Path(__file__).resolve().parent
_REPO_ROOT = _CATALOG.parents[1]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    entries = check_schema(errors)
    check_citations(entries, errors, warnings)

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
