"""Helper functions for Snakemake workflow.

This module contains utility functions extracted from the Snakefile to improve
maintainability and testability.
"""

import os
import shutil
import sys
import traceback
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path

VERBOSE_LOGS = True
PLACEHOLDER_MARKER = "[PLACEHOLDER]"


# ============================================================================
# Path utilities
# ============================================================================

def slug(text):
    """Normalize text into a slug using alphanumeric characters and underscores."""
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def resolve_relative(base, maybe_path):
    """Resolve a path relative to a base when not already absolute."""
    p = Path(maybe_path)
    return p if p.is_absolute() else (base / p)


def resolve_input(path, open_data_dir):
    """Resolve a path under the open-data folder (pre-analysis/dataset).

    Accepts bare filenames (e.g., 'Global-Integrated-Power-April-2025.xlsx'),
    paths already prefixed with 'dataset/' (for backward compatibility),
    or absolute paths.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "dataset":
        p = Path(*p.parts[1:])
    return open_data_dir / p


def resolve_output(path, output_root, output_categories=None, category=None):
    """Write outputs under the workflow root or an optional category subfolder."""
    p = Path(path)
    if p.is_absolute():
        return p
    if category is None:
        base = output_root
    else:
        if output_categories is None:
            raise ValueError("output_categories must be provided when using category")
        base = output_categories.get(category)
        if base is None:
            raise ValueError(f"Unknown output category: {category}")
    return base / p


# ============================================================================
# Output and logging utilities
# ============================================================================

def flatten_outputs(out):
    """Flatten snakemake output structures to plain path strings."""
    paths = []

    def _walk(obj):
        """Recursively collect path strings from heterogeneous output structures."""
        if obj is None:
            return
        if isinstance(obj, (str, os.PathLike)):
            paths.append(str(obj))
            return
        if isinstance(obj, Iterable):
            for item in obj:
                _walk(item)
            return
        try:
            for item in list(obj):
                _walk(item)
        except TypeError:
            pass

    _walk(out)
    return [p for p in paths if p]


def write_placeholder(path, message, verbose=VERBOSE_LOGS):
    """Write a small failure notice to preserve rule outputs on error."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{PLACEHOLDER_MARKER} {message}\n", encoding="utf-8")
    except Exception as exc:
        if verbose:
            print(f"[placeholder] Failed to write placeholder {target}: {exc}")


def write_failure_log(label, failure_log_paths, log_path=None):
    """Append a failure entry for a rule to the shared workflow failure logs."""
    timestamp = datetime.utcnow().isoformat()
    message = f"{timestamp} — Rule {label} failed"
    if log_path:
        message += f" (log: {log_path})"
    for failure_path in failure_log_paths:
        try:
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            with failure_path.open("a", encoding="utf-8") as failure_log:
                failure_log.write(message + "\n")
        except Exception:
            pass


def ensure_list(value):
    """Normalize a value into a list of scalar items."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def find_missing_outputs(outputs):
    """Identify any declared output paths that no longer exist on disk."""
    missing = []
    for path in flatten_outputs(outputs):
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            missing.append(str(p))
    return missing


# ============================================================================
# Country and ISO code utilities
# ============================================================================

@lru_cache(maxsize=None)
def natural_earth_country_lookup():
    """Load Natural Earth country names to ISO code mappings for reuse."""
    try:
        from cartopy.io import shapereader
    except Exception:
        return {}
    try:
        reader = shapereader.Reader(
            shapereader.natural_earth(
                resolution="10m", category="cultural", name="admin_0_countries"
            )
        )
    except Exception:
        return {}

    mapping = {}
    for record in reader.records():
        iso = record.attributes.get("ISO_A2")
        if not iso:
            continue
        iso = iso.upper()
        for key in ("NAME_LONG", "ADMIN", "NAME"):
            name = record.attributes.get(key)
            if isinstance(name, str):
                normalized = name.strip().lower()
                if normalized and normalized not in mapping:
                    mapping[normalized] = iso
    return mapping


def resolve_country_iso_codes(countries):
    """Map a list of configured countries to ISO A2 codes using Natural Earth data."""
    if not countries:
        return [], {}
    lookup = natural_earth_country_lookup()
    if not lookup:
        raise RuntimeError(
            "Unable to map climate countries to ISO codes; ensure cartopy and Natural Earth data are available."
        )

    iso_codes = []
    label_map = {}
    missing = []
    seen = set()
    for country in countries:
        if country is None:
            continue
        normalized = str(country).strip().lower()
        if not normalized:
            continue
        iso = lookup.get(normalized)
        if not iso:
            missing.append(country)
            continue
        if iso not in seen:
            iso_codes.append(iso)
            seen.add(iso)
        label_map[iso] = str(country)

    if missing:
        raise ValueError(
            "Could not derive ISO A2 codes for: "
            + ", ".join(str(item) for item in missing)
        )

    return iso_codes, label_map


def context_countries(config, gap_cfg=None, load_settings=None, climate_iso_codes=None,
                     climate_countries=None, genmap_countries=None, irena_cfg=None):
    """Gather a deduplicated list of countries referenced by the workflow."""
    seen = set()

    def _add_items(value):
        """Add normalized strings from a config entry into the temporary seen set."""
        for item in ensure_list(value):
            seen.add(str(item))

    if gap_cfg:
        _add_items(gap_cfg.get("countries"))
    if load_settings:
        _add_items(load_settings.get("countries"))
        _add_items(load_settings.get("slug_map", {}).values())
    if climate_iso_codes:
        _add_items(climate_iso_codes)
    if climate_countries:
        _add_items(climate_countries)
    if genmap_countries:
        _add_items(genmap_countries)
    if irena_cfg:
        _add_items(irena_cfg.get("countries"))

    return sorted(seen)


# ============================================================================
# Logging and error handling
# ============================================================================

class TeeStream:
    """Write to both the original stream and a log file."""

    def __init__(self, primary, log_file):
        """Initialize a tee stream that mirrors writes to a log file."""
        self._primary = primary
        self._log = log_file

    def write(self, data):
        """Write data to both the primary stream and the captured log."""
        self._primary.write(data)
        self._log.write(data)

    def flush(self):
        """Flush both underlying streams to keep the log in sync."""
        self._primary.flush()
        self._log.flush()

    def __getattr__(self, name):
        """Proxy attribute access to the underlying primary stream."""
        return getattr(self._primary, name)


def best_effort(label, outputs, fn, failure_log_paths, log_path=None, write_placeholders=True,
                context_countries_list=None, verbose=VERBOSE_LOGS):
    """Run fn for the given label/outputs while logging progress and handling failures."""
    start_time = datetime.utcnow()
    countries = context_countries_list or []
    context_note = f"countries={','.join(countries)}" if countries else "countries=unspecified"
    if verbose:
        print(f"[{label}] {start_time.isoformat()} starting ({context_note})")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = None
    log_path_str = None
    if log_path:
        log_target = Path(log_path)
        log_target.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_target.open("w", encoding="utf-8")
        sys.stdout = TeeStream(orig_stdout, log_file)
        sys.stderr = TeeStream(orig_stderr, log_file)
        log_path_str = str(log_target)
    try:
        result = fn()
        if verbose:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            print(f"[{label}] {end_time.isoformat()} completed (duration {duration:.1f}s)")
        missing_outputs = find_missing_outputs(outputs)
        if missing_outputs:
            raise RuntimeError(f"Missing outputs for {label}: {', '.join(missing_outputs)}")
        return result
    except Exception as exc:
        error_time = datetime.utcnow()
        print(f"[{label}] {error_time.isoformat()} ERROR: {exc} ({context_note})")
        write_failure_log(label, failure_log_paths, log_path=log_path_str)
        traceback.print_exc()
        output_paths = flatten_outputs(outputs)
        for path in output_paths:
            if not path:
                continue
            p = Path(path)
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
            except Exception as cleanup_exc:
                if verbose:
                    print(f"[{label}] cleanup warning for {p}: {cleanup_exc}")
        if write_placeholders:
            placeholder = f"Output {label} could not be generated due to: {exc}\n"
            for path in output_paths:
                if not path:
                    continue
                write_placeholder(path, placeholder, verbose=verbose)
            return None
        raise
    finally:
        if log_file:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_file.close()


@contextmanager
def capture_rule_log(log_target):
    """Yield a writable log path derived from the provided target for rule output capture."""
    target = log_target[0] if isinstance(log_target, (list, tuple)) else log_target
    log_path = Path(str(target))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    yield log_path


# ============================================================================
# Configuration validation
# ============================================================================

def validate_config(config, base_dir):
    """Validate configuration structure and required keys.
    
    Raises ValueError with descriptive messages if validation fails.
    """
    errors = []
    
    # Check required top-level sections
    required_sections = ["gap", "rninja", "irena"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required config section: {section}")
    
    # Validate GAP config
    if "gap" in config:
        gap_cfg = config["gap"]
        if "excel" not in gap_cfg:
            errors.append("gap.excel is required")
        if "countries" not in gap_cfg:
            errors.append("gap.countries is required")
        elif not gap_cfg["countries"]:
            errors.append("gap.countries cannot be empty")
        if "tech_types" not in gap_cfg:
            errors.append("gap.tech_types is required")
        elif not gap_cfg["tech_types"]:
            errors.append("gap.tech_types cannot be empty")
    
    # Validate rninja config
    if "rninja" in config:
        rninja_cfg = config["rninja"]
        if "start_year" not in rninja_cfg:
            errors.append("rninja.start_year is required")
        if "end_year" not in rninja_cfg:
            errors.append("rninja.end_year is required")
        elif rninja_cfg.get("start_year", 0) >= rninja_cfg.get("end_year", 0):
            errors.append("rninja.start_year must be < rninja.end_year")
    
    # Validate IRENA config
    if "irena" in config:
        irena_cfg = config["irena"]
        if "countries" not in irena_cfg:
            errors.append("irena.countries is required")
        elif not irena_cfg["countries"]:
            errors.append("irena.countries cannot be empty")
    
    # Validate climate config if enabled
    if config.get("climate_overview", {}).get("enabled", False):
        climate_cfg = config["climate_overview"]
        if "start_year" in climate_cfg and "end_year" in climate_cfg:
            if climate_cfg["start_year"] >= climate_cfg["end_year"]:
                errors.append("climate_overview.start_year must be < end_year")
    
    # Validate load_profile if present
    if "load_profile" in config:
        load_cfg = config["load_profile"]
        if "countries" not in load_cfg and "country" not in load_cfg:
            errors.append("load_profile must have either 'countries' or 'country'")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    return True

