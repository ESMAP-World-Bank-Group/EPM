"""
**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
Description:
    This diagnostic utility inspects the CSV inputs consumed by the
    GAMS-based Electricity Planning Model. It mirrors the structure
    defined in ``input_readers.gms`` and surfaces common data issues
    such as missing files, empty tables, and inconsistent settings
    flags before the model is executed.

Usage example:
    python -m epm.input_diagnostic --folder data_test

Author(s):
    ESMAP Modelling Team

License:
    Creative Commons Zero v1.0 Universal
**********************************************************************
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Core symbols that must exist and contain data for the model to run.
ESSENTIAL_INPUT = {
    "y",
    "pHours",
    "zcmap",
    "pSettings",
    "pGenDataInput",
    "pFuelPrice",
    "pFuelCarbonContent",
    "pTechData",
}

# Optional symbols that still deserve visibility if missing or empty.
IMPORTANT_OPTIONAL_INPUT = {
    "pDemandForecast",
    "pAvailability",
    "pAvailabilityDefault",
    "pVREProfile",
    "pVREgenProfile",
    "pTransferLimit",
    "pExtTransferLimit",
    "pNewTransmission",
    "pPlanningReserveMargin",
    "pSpinningReserveReqCountry",
    "pSpinningReserveReqSystem",
}

# The diagnostic runs in three stages:
#   1. Inspect input_readers.gms to discover which CSV files are expected.
#   2. Load every CSV, highlighting missing or empty data tables.
#   3. Perform targeted content checks (currently focused on pSettings).
# Keeping these steps explicit makes it easy to extend the script with new
# validations as the model evolves.

# Settings that require coherence checks. Values are compared after
# conversion to floats when possible.
SETTINGS_COMPATIBILITY_RULES: Tuple[
    Tuple[str, Callable[[float], bool], str], ...
] = (
    (
        "fApplySystemCo2Constraint",
        lambda value: value == 1.0,
        "activates the system-level CO2 constraint",
    ),
    (
        "fApplyCountryCo2Constraint",
        lambda value: value == 1.0,
        "activates the country-level CO2 constraint",
    ),
    (
        "sMinRenewableSharePct",
        lambda value: value > 0.0,
        "enforces a minimum renewable share",
    ),
    (
        "fApplyCapitalConstraint",
        lambda value: value == 1.0,
        "activates the capital expenditure constraint",
    ),
    (
        "fAllowTransferExpansion",
        lambda value: value == 0.0,
        "prevents internal transfer expansion",
    ),
    (
        "fEnableInternalExchange",
        lambda value: value == 0.0,
        "disables internal exchanges",
    ),
    (
        "fRemoveInternalTransferLimit",
        lambda value: value == 1.0,
        "removes internal transfer limits",
    ),
)


@dataclass
class CSVInput:
    """Metadata for a CSVReader declared in input_readers.gms."""

    name: str
    file_token: str
    resolved_path: Optional[str]
    absolute_path: Optional[Path]
    gams_type: Optional[str]


@dataclass
class CSVTable:
    """Lightweight representation of a CSV table."""

    columns: List[str]
    rows: List[List[str]]

    def row_count(self) -> int:
        """Return the number of data rows (header excluded)."""
        return len(self.rows)

    def has_data(self) -> bool:
        """True when at least one row contains a non-empty value."""
        for row in self.rows:
            if any(cell.strip() for cell in row):
                return True
        return False

    def iter_dicts(self) -> Iterator[Dict[str, str]]:
        """Yield each data row as a dict keyed by column name."""
        for row in self.rows:
            yield {
                column: value
                for column, value in zip_longest(self.columns, row, fillvalue="")
            }

    def get_column(self, name: str) -> List[str]:
        """Return a list with the raw values under the provided column name."""
        index = self._column_index(name)
        if index is None:
            return []
        return [row[index] if index < len(row) else "" for row in self.rows]

    def find_column(self, candidates: Sequence[str]) -> Optional[str]:
        """Locate the first column that matches any of the candidate names."""
        names = {column.lower(): column for column in self.columns}
        for candidate in candidates:
            label = candidate.lower()
            if label in names:
                return names[label]
        return None

    def _column_index(self, name: str) -> Optional[int]:
        lowered = name.lower()
        for position, column in enumerate(self.columns):
            if column.lower() == lowered:
                return position
        return None


@dataclass
class DiagnosticMessage:
    """Simple container for (severity, message) pairs."""

    severity: str
    text: str


def _resolve_macros(raw_value: str, macros: Dict[str, str]) -> str:
    """Replace %MACRO% tokens using the provided dictionary."""

    cleaned = raw_value.strip().strip('"').strip("'")
    pattern = re.compile(r"%(?P<name>[^%]+)%")

    def _replace(match: re.Match[str]) -> str:
        token = match.group("name")
        return macros.get(token, match.group(0))

    return pattern.sub(_replace, cleaned)


def _parse_input_readers(
    gms_path: Path, root_input: str, folder_input: str
) -> Tuple[List[CSVInput], Dict[str, str]]:
    """
    Parse ``input_readers.gms`` to recover CSVReader declarations and the
    macro assignments that determine file locations.

    The parser is intentionally lightweight: rather than depending on a YAML
    library we walk line-by-line, because the embedded code block in GAMS is
    *almost* YAML but not perfectly valid. This approach keeps the diagnostic
    self-contained and easy to adjust when new CSVReader entries are added.
    """

    text = gms_path.read_text(encoding="utf-8")
    macros: Dict[str, str] = {
        "ROOT_INPUT": root_input,
        "FOLDER_INPUT": folder_input,
    }

    csv_inputs: List[CSVInput] = []
    current_block: Optional[Dict[str, str]] = None
    inside_connect = False

    def _flush_current() -> None:
        nonlocal current_block
        if not current_block:
            return
        csv_inputs.append(
            CSVInput(
                name=current_block.get("name", ""),
                file_token=current_block.get("file", ""),
                resolved_path=None,
                absolute_path=None,
                gams_type=current_block.get("type"),
            )
        )
        current_block = None

    assign_pattern = re.compile(
        r"^\$if\s+not\s+set\s+(?P<name>\w+)\s+\$set\s+(?P=name)\s+(?P<value>.+)$",
        re.IGNORECASE,
    )

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        if not stripped:
            continue

        if stripped.startswith("$onEmbeddedCode"):
            inside_connect = True
            continue
        if stripped.startswith("$offEmbeddedCode"):
            inside_connect = False
            _flush_current()
            continue

        assignment = assign_pattern.match(stripped)
        if assignment:
            name = assignment.group("name")
            value = assignment.group("value").strip()
            resolved_value = _resolve_macros(value, macros)
            macros[name] = resolved_value
            continue

        if not inside_connect:
            continue

        if stripped.startswith("- CSVReader"):
            _flush_current()
            current_block = {}
            continue

        if stripped.startswith("- "):
            _flush_current()
            continue

        if current_block is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_block[key.strip()] = value.strip().rstrip(",")

    _flush_current()

    resolved_inputs: List[CSVInput] = []
    for entry in csv_inputs:
        file_token = entry.file_token or ""
        if not file_token:
            resolved_inputs.append(entry)
            continue

        resolved = _resolve_macros(file_token, macros)
        candidate = Path(resolved)
        if not candidate.is_absolute():
            candidate = (gms_path.parent / candidate).resolve()

        resolved_inputs.append(
            CSVInput(
                name=entry.name,
                file_token=file_token,
                resolved_path=resolved,
                absolute_path=candidate,
                gams_type=entry.gams_type,
            )
        )

    return resolved_inputs, macros


def _load_csv_table(path: Path) -> CSVTable:
    """
    Load a CSV file while normalising headers and row lengths.
    Empty leading lines are skipped so the first non-empty row becomes
    the header row, matching how GAMS Connect interprets the files.
    """

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        raw_rows = [[cell.strip() for cell in row] for row in reader]

    header: Optional[List[str]] = None
    data_rows: List[List[str]] = []

    for row in raw_rows:
        if header is None:
            if any(cell for cell in row):
                header = [cell.replace("\ufeff", "").strip() for cell in row]
            continue
        data_rows.append(row)

    if header is None:
        raise ValueError("No header row found.")

    normalised_rows = [
        [row[index] if index < len(row) else "" for index in range(len(header))]
        for row in data_rows
    ]

    return CSVTable(columns=header, rows=normalised_rows)


def _report(severity: str, message: str) -> DiagnosticMessage:
    return DiagnosticMessage(severity=severity, text=message)


def _coerce_numeric(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_messages_by_severity(
    messages: Iterable[DiagnosticMessage],
) -> Dict[str, List[str]]:
    """Group diagnostic messages under their severity for compact printing."""
    ordered: Dict[str, List[str]] = {"ERROR": [], "WARNING": [], "INFO": []}
    for msg in messages:
        ordered.setdefault(msg.severity, []).append(msg.text)
    return ordered


def run_diagnostics(args: argparse.Namespace) -> int:
    """Execute the diagnostics using the provided CLI arguments."""
    gms_path = Path(args.input_readers).resolve()
    if not gms_path.exists():
        print(f"[ERROR] input_readers file not found: {gms_path}")
        return 1

    # Step 1 – discover the expected CSV files by parsing input_readers.gms.
    csv_inputs, macros = _parse_input_readers(gms_path, args.root_input, args.folder)
    messages: List[DiagnosticMessage] = []
    loaded_tables: Dict[str, CSVTable] = {}
    symbols_with_data: set[str] = set()

    # Step 2 – inspect every CSVReader entry and flag issues with individual files.
    for entry in csv_inputs:
        if not entry.name:
            continue

        if not entry.resolved_path or not entry.absolute_path:
            messages.append(
                _report(
                    "WARNING",
                    f"{entry.name}: unable to resolve file path from token "
                    f"{entry.file_token!r}.",
                )
            )
            continue

        target_path = entry.absolute_path
        if not target_path.exists():
            severity = "ERROR" if entry.name in ESSENTIAL_INPUT else "WARNING"
            messages.append(
                _report(
                    severity,
                    f"{entry.name}: missing file {target_path}.",
                )
            )
            continue

        try:
            table = _load_csv_table(target_path)
        except ValueError:
            messages.append(
                _report(
                    "WARNING",
                    f"{entry.name}: {target_path} has no header row (not normal).",
                )
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive
            messages.append(
                _report(
                    "ERROR",
                    f"{entry.name}: failed to load {target_path} ({exc}).",
                )
            )
            continue

        if not table.has_data():
            messages.append(
                _report(
                    "WARNING",
                    f"{entry.name}: {target_path} contains no usable data "
                    "(table is empty; this is not normal).",
                )
            )
        else:
            if args.show_loaded:
                messages.append(
                    _report(
                        "INFO",
                        f"{entry.name}: loaded {table.row_count()} rows from {target_path}.",
                    )
                )
            loaded_tables[entry.name] = table
            symbols_with_data.add(entry.name)

    # Step 3 – confirm that all mandatory model inputs were encountered.
    for symbol in ESSENTIAL_INPUT:
        if symbol not in symbols_with_data:
            messages.append(
                _report(
                    "ERROR",
                    f"{symbol}: required input missing or unreadable.",
                )
            )

    for symbol in IMPORTANT_OPTIONAL_INPUT:
        if symbol not in symbols_with_data:
            messages.append(
                _report(
                    "WARNING",
                    f"{symbol}: optional but important input missing; "
                    "confirm the omission is intentional.",
                )
            )

    # Step 4 – run targeted content checks (currently focused on pSettings).
    if "pSettings" in loaded_tables:
        settings_table = loaded_tables["pSettings"]
        abbrev_col = settings_table.find_column(["abbreviation", "abbr", "code"])
        value_col = settings_table.find_column(["value", "val"])

        if abbrev_col and value_col:
            # Build a mapping from setting abbreviation to its configured value.
            settings_map: Dict[str, str] = {}
            for record in settings_table.iter_dicts():
                key = str(record.get(abbrev_col, "")).strip()
                if not key:
                    continue
                settings_map[key] = record.get(value_col, "")

            if "pSettingsHeader" in loaded_tables:
                header_table = loaded_tables["pSettingsHeader"]
                if header_table.columns:
                    header_col = header_table.columns[0]
                    expected = {
                        str(value).strip()
                        for value in header_table.get_column(header_col)
                        if str(value).strip()
                    }
                    missing = expected.difference(settings_map)
                    if missing:
                        formatted = ", ".join(sorted(missing))
                        messages.append(
                            _report(
                                "WARNING",
                                "pSettings: missing rows for expected abbreviations "
                                f"{formatted}.",
                            )
                        )

            def get_numeric(name: str) -> Optional[float]:
                """Helper to read numeric settings and ignore blank entries."""
                return _coerce_numeric(settings_map.get(name))

            for name, predicate, description in SETTINGS_COMPATIBILITY_RULES:
                numeric_value = get_numeric(name)
                if numeric_value is None:
                    continue
                if predicate(numeric_value):
                    messages.append(
                        _report(
                            "WARNING",
                            f"pSettings: {name}={numeric_value:g} {description}.",
                        )
                    )

            country_spin = get_numeric("fApplyCountrySpinReserveConstraint")
            system_spin = get_numeric("fApplySystemSpinReserveConstraint")
            if country_spin is not None and system_spin is not None:
                if country_spin == 1 and system_spin != 0:
                    messages.append(
                        _report(
                            "ERROR",
                            "pSettings: fApplyCountrySpinReserveConstraint must be 0 "
                            "when fApplySystemSpinReserveConstraint is 1.",
                        )
                    )
                if system_spin == 1 and country_spin != 0:
                    messages.append(
                        _report(
                            "ERROR",
                            "pSettings: fApplySystemSpinReserveConstraint must be 0 "
                            "when fApplyCountrySpinReserveConstraint is 1.",
                        )
                    )
        else:
            messages.append(
                _report(
                    "WARNING",
                    "pSettings: could not locate columns named 'Abbreviation' and "
                    "'Value'; flag consistency checks were skipped.",
                )
            )

    grouped = _collect_messages_by_severity(messages)
    total_errors = len(grouped.get("ERROR", []))

    print(f"Input diagnostics for {macros['ROOT_INPUT']}/{macros['FOLDER_INPUT']}")
    for severity in ("ERROR", "WARNING", "INFO"):
        entries = grouped.get(severity, [])
        if not entries:
            continue
        print(f"\n[{severity}]")
        for text in entries:
            print(f" - {text}")

    if total_errors:
        print(f"\nCompleted with {total_errors} error(s).")
        return 1

    print("\nCompleted without blocking errors.")
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the CSV inputs expected by input_readers.gms "
        "and highlight inconsistencies."
    )
    default_gms = Path(__file__).with_name("input_readers.gms")
    parser.add_argument(
        "--input-readers",
        default=str(default_gms),
        help="Path to the input_readers.gms file (default: %(default)s).",
    )
    parser.add_argument(
        "--root-input",
        dest="root_input",
        default="input",
        help="Value to substitute for %%ROOT_INPUT%% (default: %(default)s).",
    )
    parser.add_argument(
        "--folder",
        default="data_test",
        help="Value to substitute for %%FOLDER_INPUT%% (default: %(default)s).",
    )
    parser.add_argument(
        "--show-loaded",
        action="store_true",
        help="Emit informational messages for successfully loaded inputs.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    return run_diagnostics(args)


if __name__ == "__main__":
    raise SystemExit(main())
