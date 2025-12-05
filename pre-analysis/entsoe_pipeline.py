"""Simple ENSO-E helper for cross-border flows and load exports.

The module ships its own CLI for quick ad-hoc downloads without touching the Snakemake workflow.
It pulls credentials from `config/api_tokens.ini` (or the `API_TOKEN_ENSTOE` env var) and mirrors the
`open_data_config.yaml` block so you can reuse the same defaults when running locally.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import traceback
from configparser import ConfigParser
from pathlib import Path
from typing import Iterable, Sequence, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd
import pycountry
import yaml

if TYPE_CHECKING:
    from entsoe import EntsoePandasClient

try:
    from entsoe import EntsoePandasClient as _EntsoePandasClient
except ImportError:  # pragma: no cover - pip install entsoe-py when using this script.
    _EntsoePandasClient = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "open_data_config.yaml"
TOKEN_SECTION = "api_tokens"
TOKEN_PATH_ENV_VAR = "API_TOKENS_PATH"
DEFAULT_TOKEN_PATH = REPO_ROOT / "config" / "api_tokens.ini"
DEFAULT_TIMEZONE = "Europe/Brussels"
DEFAULT_OUTPUT_SUBDIR = "entsoe"




COUNTRY_NAME_TO_ISO2_OVERRIDES = {
    "bosnia and herzegovina": "BA",
    "croatia": "HR",
    "serbia": "RS",
    "montenegro": "ME",
    "north macedonia": "MK",
    "north macedonia-": "MK",
    "macedonia": "MK",
}


NAME_TO_ISO2: dict[str, str] = {}
for country in pycountry.countries:
    candidate_names = {country.name}
    if hasattr(country, "official_name"):
        candidate_names.add(country.official_name)
    if hasattr(country, "common_name"):
        candidate_names.add(country.common_name)
    candidate_names.add(country.alpha_2)
    candidate_names.add(country.alpha_3)
    for entry in candidate_names:
        if entry:
            NAME_TO_ISO2[str(entry).strip().lower()] = country.alpha_2


def country_name_to_iso2(country_name: str) -> str:
    """Resolve a country name or code to an ISO alpha-2 string, with fuzzy matching."""
    normalized = (country_name or "").strip()
    if not normalized:
        raise ValueError("Country name must not be empty.")

    key = normalized.lower()
    if key in COUNTRY_NAME_TO_ISO2_OVERRIDES:
        return COUNTRY_NAME_TO_ISO2_OVERRIDES[key]
    if key in NAME_TO_ISO2:
        return NAME_TO_ISO2[key]

    closest = difflib.get_close_matches(key, NAME_TO_ISO2.keys(), n=1, cutoff=0.75)
    if closest:
        return NAME_TO_ISO2[closest[0]]

    raise ValueError(f"Unable to resolve '{country_name}' to an ISO alpha-2 code.")


COUNTRY_NEIGHBORS = {
    "AL": ["GR", "ME", "MK"],
    "AT": ["CZ", "DE", "HU", "IT", "CH", "SI", "SK"],
    "BA": ["HR", "ME", "RS"],
    "BG": ["GR", "MK", "RO", "RS", "TR"],
    "HR": ["BA", "HU", "ME", "RS", "SI"],
    "ME": ["AL", "BA", "HR", "RS"],
    "MK": ["AL", "BG", "GR", "RS"],
    "RS": ["AL", "BA", "BG", "HR", "HU", "MK", "RO"],
    "SI": ["AT", "HR", "IT"],
    "IT": ["AT", "FR", "SM", "SI"],
    "GR": ["AL", "BG", "MK"],
    "HU": ["AT", "RO", "RS", "SK", "SI", "UA"],
    "RO": ["BG", "HU", "MD", "RS", "UA"],
}


def find_neighbor_targets(country_iso: str) -> list[str]:
    target = country_iso.strip().upper()
    return COUNTRY_NEIGHBORS.get(target, [])


def build_crossborder_pairs(
    countries: Sequence[str],
    neighbor_lookup: dict[str, Sequence[str]] | Sequence[str] | None = None,
) -> list[Tuple[str, str]]:
    """Build unique cross-border tuples for the provided country list."""
    pairs: list[Tuple[str, str]] = []
    seen = set()
    lookup_dict: dict[str, list[str]] = {}
    default_neighbors: list[str] = []

    if isinstance(neighbor_lookup, dict):
        for key, value in neighbor_lookup.items():
            key_iso = str(key).strip().upper()
            neighbors = [str(item).strip().upper() for item in value if item]
            if neighbors:
                lookup_dict[key_iso] = neighbors
    elif isinstance(neighbor_lookup, (list, tuple)):
        default_neighbors = [str(item).strip().upper() for item in neighbor_lookup if item]

    for country_iso in countries:
        iso = country_iso.strip().upper()
        neighbors = lookup_dict.get(iso, [])
        if not neighbors:
            neighbors = default_neighbors or find_neighbor_targets(iso)
        for neighbor in neighbors:
            pair = (iso, neighbor)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    return pairs


def _slug(value: str) -> str:
    """Return a lowercase slug made of alphanumeric segments separated by underscores."""
    return re.sub(r"[^A-Za-z0-9]+", "_", value or "").strip("_").lower()


def _format_period(timestamp: pd.Timestamp) -> str:
    """Format a timestamp as YYYYMMDD for use in filenames."""
    return timestamp.strftime("%Y%m%d")


def _load_yaml_config(config_path: Path) -> dict:
    """Load the entsoe section from the open-data YAML config, returning an empty dict when missing."""
    if not config_path.exists():
        return {}
    content = config_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content) or {}
    if isinstance(parsed, dict):
        return parsed.get("entsoe", {}) or {}
    return {}

def _to_timestamp(value: str, timezone: str) -> pd.Timestamp:
    """Coerce strings into timezone-aware pandas Timestamps."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        try:
            return timestamp.tz_localize(timezone)
        except TypeError:
            return timestamp.tz_localize(timezone)
    if timezone and timestamp.tzinfo.zone != timezone:
        return timestamp.tz_convert(timezone)
    return timestamp


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_standalone" / DEFAULT_OUTPUT_SUBDIR


def _resolve_output_dir(path: Path | None) -> Path:
    """Resolve user-provided output paths relative to the script directory."""
    if path is None:
        return DEFAULT_OUTPUT_DIR
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def load_api_token(token_name: str, *, config_path: Path | None = None, env_var: str | None = None) -> str:
    """Load an API token from environment variables or a config INI file."""
    env_name = env_var or f"API_TOKEN_{token_name.upper()}"
    token = os.getenv(env_name)
    if token:
        return token

    candidate = Path(config_path or os.getenv(TOKEN_PATH_ENV_VAR) or DEFAULT_TOKEN_PATH)
    if not candidate.exists():
        raise FileNotFoundError(f"API token config not found: {candidate}")

    parser = ConfigParser()
    parser.read(candidate)
    if parser.has_option(TOKEN_SECTION, token_name):
        value = parser.get(TOKEN_SECTION, token_name).strip()
        if value:
            return value

    raise RuntimeError(
        f"Missing API token '{token_name}'. Define it using env var {env_name} or under [{TOKEN_SECTION}] in {candidate}."
    )


def _write_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to CSV under the provided path and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def _read_dataframe(path: Path) -> pd.DataFrame:
    """Load a CSV into a DataFrame with timestamp parsing."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = df.index.name or "timestamp"
    return df


def _resolve_load_column(df: pd.DataFrame, country_code: str) -> str:
    """Pick the appropriate load column from the dataframe."""
    normalized = country_code.strip().lower()
    timestamp_labels = {"timestamp"}
    # Exact match
    if country_code in df.columns:
        return country_code
    # Case-insensitive match
    for col in df.columns:
        if str(col).strip().lower() == normalized:
            return col
    # Common entsoe label
    for col in df.columns:
        if str(col).strip().lower() == "actual load":
            return col
    # Fallback: only one non-timestamp column
    non_ts_cols = [col for col in df.columns if str(col).strip().lower() not in timestamp_labels]
    if len(non_ts_cols) == 1:
        return non_ts_cols[0]
    available = ", ".join(df.columns)
    raise KeyError(f"Missing column for '{country_code}' in load data. Available columns: {available}")


def _format_time_axis(ax, timezone, fmt: str = "%b %d %H:%M") -> None:
    """Apply consistent datetime formatting to an axis."""
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.DateFormatter(fmt, tz=timezone)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def _country_label(country_code: str) -> str:
    """Return a display name for an ISO2 code."""
    record = pycountry.countries.get(alpha_2=country_code.upper())
    return record.name if record else country_code


def _exc_to_str(exc: Exception, verbose: bool) -> str:
    """Return a readable exception string; add class name when message is empty."""
    if verbose:
        return "".join(traceback.format_exception_only(type(exc), exc)).strip()
    message = str(exc).strip()
    return message or f"{exc.__class__.__name__}"


def _should_skip(path: Path, refresh: bool, verbose: bool, label: str) -> bool:
    """Return True when we should reuse an existing file instead of re-querying the API."""
    if refresh:
        return False
    if path.exists():
        if verbose:
            print(f"[entsoe] Using cached {label}: {path}")
        return True
    return False


def _drop_sum_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any column labeled 'sum' (case-insensitive), including MultiIndex entries."""
    def is_sum_label(label: object) -> bool:
        if isinstance(label, tuple):
            return any(is_sum_label(part) for part in label)
        return str(label).strip().lower() == "sum"

    cols_to_drop = [col for col in df.columns if is_sum_label(col)]
    if cols_to_drop:
        return df.drop(columns=cols_to_drop)
    return df


def fetch_crossborder_flows(
    client: "EntsoePandasClient",
    country_from: str,
    country_to: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
    verbose: bool,
    refresh_existing: bool,
) -> tuple[Path, bool]:
    """Download cross-border flows for a country pair and persist them to CSV."""
    filename = f"entsoe_crossborder_{_slug(country_from)}_{_slug(country_to)}_{_format_period(start)}_{_format_period(end)}.csv"
    path = output_dir / "crossborder" / filename
    if _should_skip(path, refresh_existing, verbose, f"cross-border data {country_from}→{country_to}"):
        return path, True

    if verbose:
        print(f"[entsoe] Querying flows {country_from}→{country_to} ({start.isoformat()} → {end.isoformat()})")
    series = client.query_crossborder_flows(
        country_code_from=country_from,
        country_code_to=country_to,
        start=start,
        end=end,
    )
    if isinstance(series, pd.Series):
        df = series.rename("value").to_frame()
    else:
        df = pd.DataFrame(series)
    df.index.name = df.index.name or "timestamp"

    return _write_dataframe(df, path), False


def fetch_physical_crossborder_allborders(
    client: "EntsoePandasClient",
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
    verbose: bool,
    refresh_existing: bool,
) -> tuple[Path, bool]:
    """Download physical cross-border flows for all borders surrounding a country."""
    filename = (
        f"entsoe_physical_crossborder_allborders_{_slug(country_code)}_"
        f"{_format_period(start)}_{_format_period(end)}.csv"
    )
    path = output_dir / "crossborder_allborders" / filename
    if _should_skip(path, refresh_existing, verbose, f"physical cross-border data for {country_code}"):
        return path, True

    if verbose:
        print(
            f"[entsoe] Querying physical cross-border all borders for {country_code} "
            f"({start.isoformat()} → {end.isoformat()})"
        )
    df = client.query_physical_crossborder_allborders(country_code, start=start, end=end, export=True)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df.index.name = df.index.name or "timestamp"
    df = _drop_sum_columns(df)

    return _write_dataframe(df, path), False


def fetch_load(
    client: "EntsoePandasClient",
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
    verbose: bool,
    refresh_existing: bool,
) -> tuple[Path, pd.DataFrame, bool]:
    """Download load series for a country and write it to CSV."""
    filename = f"entsoe_load_{_slug(country_code)}_{_format_period(start)}_{_format_period(end)}.csv"
    path = output_dir / "load" / filename
    if _should_skip(path, refresh_existing, verbose, f"load data for {country_code}"):
        return path, _read_dataframe(path), True

    if verbose:
        print(f"[entsoe] Querying load for {country_code} ({start.isoformat()} → {end.isoformat()})")
    df = client.query_load(country_code, start=start, end=end)
    if isinstance(df, pd.Series):
        df = df.rename(country_code).to_frame()
    elif not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df.index.name = df.index.name or "timestamp"

    _write_dataframe(df, path)
    return path, df, False


def plot_load_by_year(load_df: pd.DataFrame, country_code: str, output_dir: Path) -> Path:
    """Plot hourly load profiles per year and save as PDF."""
    df = load_df.copy()
    df = df.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in df.columns:
        raise ValueError("Load data requires a timestamp index.")
    value_col = _resolve_load_column(df, country_code)
    df["year"] = df["timestamp"].dt.year

    fig, ax = plt.subplots(figsize=(10, 5))
    for year, group in df.groupby("year"):
        ax.plot(group["timestamp"], group[value_col], label=str(year), linewidth=0.8)
    ax.set_title(f"Hourly load profile – {_country_label(country_code)} (MW)")
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _format_time_axis(ax, df["timestamp"].dt.tz)
    ax.set_ylim(bottom=0)
    ax.legend(title="Year", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, linestyle=":", linewidth=0.5)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"load_profile_{country_code}.pdf"
    fig.tight_layout()
    fig.savefig(plot_path, format="pdf")
    plt.close(fig)
    return plot_path


def fetch_installed_generation_capacity_per_unit(
    client: "EntsoePandasClient",
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
    verbose: bool,
    refresh_existing: bool,
) -> tuple[Path, bool]:
    """Download installed generation capacity per unit for a country."""
    filename = (
        f"entsoe_installed_generation_capacity_{_slug(country_code)}_"
        f"{_format_period(start)}_{_format_period(end)}.csv"
    )
    path = output_dir / "installed_capacity" / filename
    if _should_skip(path, refresh_existing, verbose, f"installed capacity for {country_code}"):
        return path, True

    if verbose:
        print(
            f"[entsoe] Querying installed generation capacity per unit for {country_code} "
            f"({start.isoformat()} → {end.isoformat()})"
        )
    df = client.query_installed_generation_capacity_per_unit(country_code, start=start, end=end)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df.index.name = df.index.name or "timestamp"

    return _write_dataframe(df, path), False


def _normalize_country_inputs(inputs: Sequence[str]) -> list[str]:
    """Convert fuzzy country inputs into unique ISO2 codes."""
    normalized: list[str] = []
    for entry in inputs:
        value = str(entry or "").strip()
        if not value:
            continue
        try:
            iso = country_name_to_iso2(value)
        except ValueError as exc:
            raise ValueError(f"{exc}") from exc
        if iso not in normalized:
            normalized.append(iso)
    if not normalized:
        raise ValueError("Country inputs resolved to an empty list.")
    return normalized


def run_entsoe_pipeline(
    start: str | None,
    end: str | None,
    timezone: str,
    country_inputs: Sequence[str],
    output_dir: Path,
    dry_run: bool,
    verbose: bool,
    api_tokens_path: Path | None,
    neighbor_lookup: dict[str, Sequence[str]] | Sequence[str] | None = None,
    variables: Sequence[str] | None = None,
    refresh_existing: bool = False,
) -> int:
    """Logically orchestrate ENTSO-E downloads for the configured countries."""
    if not start or not end:
        print("[entsoe] Both --start and --end are required.")
        return 2

    if not country_inputs:
        print("[entsoe] No load targets configured.")
        return 3

    try:
        countries = _normalize_country_inputs(country_inputs)
    except ValueError as exc:
        print(f"[entsoe] {exc}")
        return 1

    start_ts = _to_timestamp(start, timezone)
    end_ts = _to_timestamp(end, timezone)
    output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_vars = {
        str(var).strip().lower()
        for var in (variables or [])
        if var and str(var).strip()
    }
    if not normalized_vars:
        normalized_vars = {"load", "crossborder", "physical_allborders", "installed_capacity"}

    crossborder_pairs: list[Tuple[str, str]] = []
    if "crossborder" in normalized_vars:
        crossborder_pairs = build_crossborder_pairs(countries, neighbor_lookup=neighbor_lookup)
        if crossborder_pairs:
            if verbose:
                description = ", ".join(f"{frm}→{to}" for frm, to in crossborder_pairs)
                print(f"[entsoe] Resolved cross-border targets: {description}")
        else:
            print("[entsoe] No neighbors resolved; cross-border queries will be skipped.")

    if dry_run:
        print("[entsoe] Dry run summary")
        print(f"  Start/end: {start_ts.isoformat()} → {end_ts.isoformat()} (tz={timezone})")
        if crossborder_pairs:
            print("  Cross-border targets:")
            for frm, to in crossborder_pairs:
                print(f"    - {frm}→{to}")
        if countries:
            print("  Load targets:\n    " + "\n    ".join(countries))
        print(f"  Outputs will land under {output_dir.resolve()}")
        return 0

    token = load_api_token("enstoe", config_path=api_tokens_path)
    if _EntsoePandasClient is None:
        raise ImportError(
            "entsoe-py is required to fetch ENTSO-E data. "
            "Install it via `pip install entsoe-py` or add it to your environment dependencies."
        )
    client = _EntsoePandasClient(api_key=token)

    if "crossborder" in normalized_vars:
        for frm, to in crossborder_pairs:
            try:
                path, cached = fetch_crossborder_flows(
                    client, frm, to, start_ts, end_ts, output_dir, verbose, refresh_existing
                )
                if verbose and not cached:
                    print(f"[entsoe] Saved cross-border data to {path}")
            except Exception as exc:
                print(f"[entsoe] Failed to fetch flows {frm}→{to}: {_exc_to_str(exc, verbose)}")

    for country in countries:
        load_path = None
        load_df = None
        if "load" in normalized_vars:
            try:
                load_path, load_df, cached_load = fetch_load(
                    client, country, start_ts, end_ts, output_dir, verbose, refresh_existing
                )
                if verbose and not cached_load:
                    print(f"[entsoe] Saved load data to {load_path}")
            except Exception as exc:
                print(f"[entsoe] Failed to fetch load data for {country}: {_exc_to_str(exc, verbose)}")
                load_df = None

        if "load" in normalized_vars and load_df is not None:
            try:
                plot_path = plot_load_by_year(load_df, country, output_dir)
                if verbose:
                    print(f"[entsoe] Saved load plot to {plot_path}")
            except Exception as exc:
                print(f"[entsoe] Failed to plot load for {country}: {_exc_to_str(exc, verbose)}")

        if "load" in normalized_vars and load_df is not None:
            try:
                plot_path = plot_load_by_year(load_df, country, output_dir)
                if verbose:
                    print(f"[entsoe] Saved load plot to {plot_path}")
            except Exception as exc:
                print(f"[entsoe] Failed to plot load for {country}: {_exc_to_str(exc, verbose)}")

        if "physical_allborders" in normalized_vars:
            try:
                path, cached_phys = fetch_physical_crossborder_allborders(
                    client, country, start_ts, end_ts, output_dir, verbose, refresh_existing
                )
                if verbose and not cached_phys:
                    print(f"[entsoe] Saved physical cross-border all borders data to {path}")
            except Exception as exc:
                print(f"[entsoe] Failed to fetch physical cross-border all borders for {country}: {_exc_to_str(exc, verbose)}")

        if "installed_capacity" in normalized_vars:
            try:
                path, cached_cap = fetch_installed_generation_capacity_per_unit(
                    client, country, start_ts, end_ts, output_dir, verbose, refresh_existing
                )
                if verbose and not cached_cap:
                    print(f"[entsoe] Saved installed capacity data to {path}")
            except Exception as exc:
                print(f"[entsoe] Failed to fetch installed capacity per unit for {country}: {_exc_to_str(exc, verbose)}")

    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build and parse the CLI arguments for the standalone ENTSO-E helper."""
    parser = argparse.ArgumentParser(description="Fetch ENTSO-E flows and load timeseries.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the open-data config file.")
    parser.add_argument("--start", type=str, help="Start timestamp (e.g. 20240101)." )
    parser.add_argument("--end", type=str, help="End timestamp." )
    parser.add_argument("--timezone", type=str, default=DEFAULT_TIMEZONE, help="Timezone for start/end (defaults to Europe/Brussels)." )
    parser.add_argument(
        "--countries",
        nargs="+",
        type=str,
        help="Fuzzy country names or ISO codes to use for load & cross-border flows (space-separated).",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        type=str,
        help="Data streams to fetch (load, crossborder, physical_allborders, installed_capacity).",
    )
    parser.add_argument("--output-dir", type=Path, help="Override the entsoe output directory.")
    parser.add_argument("--api-tokens-path", type=Path, help="Custom path to the API tokens INI.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without calling ENTSO-E.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress messages.", default=True)
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Force re-download even when the target CSV already exists (default: reuse cached files).",
    )
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = _parse_args()
    config = _load_yaml_config(args.config)
    if not config.get("enabled", True):
        print("[entsoe] Skipped (disabled in configuration).")
        sys.exit(0)

    raw_variables = args.variables or config.get("variables")
    variables = list(raw_variables) if raw_variables else None
    config_output = config.get("output_dir")
    override_output = args.output_dir or (Path(config_output) if config_output else None)
    output_dir = _resolve_output_dir(override_output)
    
    run_entsoe_pipeline(
        start=args.start or config.get("start"),
        end=args.end or config.get("end"),
        timezone=args.timezone or config.get("timezone", DEFAULT_TIMEZONE),
        country_inputs=args.countries or config.get("countries"),
        output_dir=output_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
        api_tokens_path=args.api_tokens_path,
        neighbor_lookup=None,
        variables=variables,
        refresh_existing=args.refresh_existing,
    )
