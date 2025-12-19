"""Load pipeline for Toktarova hourly demand profiles and helper exports.

Main entry points:
- `run_load_workflow`: orchestrates per-country load extraction and optional representative-day export.
- `load_country_load_profile`: read Toktarova CSV and emit per-country time series + diagnostics.
"""

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence, Union

import pandas as pd
import pytz
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOKTAROVA_PATH = (
    BASE_DIR
    / "dataset"
    / "Toktarova2019_paper_LongtermLoadProjectioninHighREsolutionforallCountriesGlobally_ElecPowerEnergySys_supplementary_7_load_all_2020.csv"
)

# --------------------------------------------------------------------------- #
# File + naming utilities
# --------------------------------------------------------------------------- #


def require_file(path, hint=None):
    """Validate that a required external file exists; raise with a helpful message if missing."""
    path = Path(path)
    if not path.exists():
        sharepoint = "https://worldbankgroup.sharepoint.com/:f:/r/teams/PowerSystemPlanning-WBGroup/Shared%20Documents/2.%20Knowledge%20Products/19.%20Databases/EPM%20Prepare%20Data?csf=1&web=1&e=wig2nC"
        extra = f" {hint}" if hint else f" Place the file in the expected directory. All files can be downloaded from WBG SharePoint: {sharepoint}"
        raise FileNotFoundError(f"File not found: {path}.{extra}")
    return path


def _slugify(text):
    """Simple slug for filenames."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def resolve_country_name(input_name, available_names, threshold=0.75, verbose=True, allow_missing=False):
    """Resolve a user-provided country name against available names with fuzzy matching."""
    available = [name for name in available_names if isinstance(name, str)]
    normalized = {name.strip().lower(): name for name in available}
    candidate = input_name.strip().lower()

    if candidate in normalized:
        match = normalized[candidate]
        if verbose:
            print(f"[country-resolver] Exact match: '{input_name}' -> '{match}'")
        return match

    best, best_score = None, 0
    for name in available:
        score = SequenceMatcher(None, candidate, name.strip().lower()).ratio()
        if score > best_score:
            best, best_score = name, score

    if best_score >= threshold:
        if verbose:
            pct = f"{best_score*100:.1f}%"
            print(f"[country-resolver] Using closest match ({pct}): '{input_name}' -> '{best}'")
        return best

    if allow_missing:
        suggestions = ", ".join(available[:5])
        print(
            f"[country-resolver] Missing '{input_name}'. Closest='{best}' ({best_score:.2f}). "
            f"Available examples: {suggestions}. Skipping."
        )
        return None

    suggestions = ", ".join(available[:5])
    raise ValueError(
        f"Could not match country '{input_name}'. Top suggestion: '{best}' "
        f"(score {best_score:.2f}). Available examples: {suggestions}"
    )


# --------------------------------------------------------------------------- #
# Toktarova loaders + exports
# --------------------------------------------------------------------------- #

def load_country_load_profile(
    country: str,
    dataset_path=None,
    output_dir: Union[str, Path] = "output",
    year: int = 2020,
    plot: bool = True,
    generate_heatmap_boxplot: bool = True,
    verbose: bool = True,
):
    """Load a Toktarova hourly load profile, emit CSV + optional plots.

    Data format
    -----------
    Input: Toktarova CSV (semicolon-separated) where row 2 lists country names and hourly values start on row 5.
    Output CSV columns: ``timestamp`` (periods from Jan 1), ``hour`` (0-8759), ``load_mw``.
    """
    data_path = require_file(
        dataset_path or DEFAULT_TOKTAROVA_PATH,
        hint="Place Toktarova2019 load dataset under pre-analysis/dataset/",
    )
    if verbose:
        print(f"[load_profile] Using Toktarova dataset at {data_path}")

    raw = pd.read_csv(data_path, sep=";", decimal=",", header=None, low_memory=False)
    country_row = raw.iloc[2]
    resolved_name = resolve_country_name(country, country_row.tolist(), verbose=True)
    col_idx = country_row[country_row == resolved_name].index[0]
    hourly = raw.iloc[5:, [0, col_idx]].rename(columns={0: "label", col_idx: "load_mw"})
    hourly["hour"] = hourly["label"].astype(str).str.extract(r"(\d+)").astype(int)
    hourly["load_mw"] = pd.to_numeric(hourly["load_mw"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    profile = hourly.dropna(subset=["load_mw"]).sort_values("hour").reset_index(drop=True)
    if verbose:
        print(f"[load_profile] Resolved '{country}' -> '{resolved_name}' with {len(profile)} hourly samples")

    # Build hourly timestamps for the whole year, then drop Feb 29 so the final day is preserved.
    full_year = pd.date_range(
        start=f"{year}-01-01", end=f"{year + 1}-01-01", freq="h", inclusive="left"
    )
    timestamps = full_year[(full_year.month != 2) | (full_year.day != 29)]

    # If the source includes leap-day data (8784 rows), drop the Feb 29 slice before assignment.
    if len(profile) == len(timestamps) + 24:
        leap_day_start = 24 * 59  # first hour of Feb 29 in a leap year
        profile = profile.drop(profile.index[leap_day_start : leap_day_start + 24]).reset_index(drop=True)
        if verbose:
            print("[load_profile] Dropping leap day (Feb 29) to match non-leap timestamp index")
    elif len(profile) != len(timestamps):
        raise ValueError(
            f"Unexpected profile length {len(profile)} for year {year}; expected {len(timestamps)} "
            f"or {len(timestamps) + 24} (with leap day)."
        )

    profile["timestamp"] = timestamps[: len(profile)]
    profile["hour"] = range(len(profile))
    profile = profile[["timestamp", "hour", "load_mw"]]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(country)
    csv_path = out_dir / f"load_profile_{slug}.csv"
    if verbose:
        print(f"[load_profile] Writing load profile CSV to {csv_path}")
    profile.to_csv(csv_path, index=False)

    plot_path = out_dir / f"load_profile_{slug}.pdf"
    heatmap_path = out_dir / f"heatmap_load_{slug}.pdf"
    boxplot_path = out_dir / f"boxplot_load_{slug}.pdf"
    if plot:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        plt.plot(profile["timestamp"], profile["load_mw"], linewidth=0.6, color="#1f78b4")
        plt.xlabel("")
        plt.ylabel("Load (MW)")
        plt.title(f"Hourly load profile - {country}")
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        if verbose:
            print(f"[load_profile] Saved time-series plot to {plot_path}")
        plt.close()
    else:
        plot_path = None

    if generate_heatmap_boxplot:
        from plots import make_boxplot, make_heatmap

        if verbose:
            print(f"[load_profile] Generating diagnostics: {heatmap_path} & {boxplot_path}")

        peak_load = profile["load_mw"].max()
        has_peak = pd.notna(peak_load) and peak_load > 0
        value_col = f"{country}" if has_peak else country
        normalized_load = profile["load_mw"] / peak_load if has_peak else profile["load_mw"]

        df_plot = pd.DataFrame(
            {
                "season": profile["timestamp"].dt.month,
                "day": profile["timestamp"].dt.day,
                "hour": profile["timestamp"].dt.hour,
                value_col: normalized_load.values,
            }
        )

        make_heatmap(df_plot.copy(), tech=f"Load - {country}", path=heatmap_path, vmin=0, vmax=1)
        make_boxplot(
            df_plot,
            tech=f"Load - {country}",
            path=boxplot_path,
            value_label="Daily average load" if has_peak else "Daily average load (MW)",
        )

    return profile, csv_path, plot_path


def export_load_profiles_for_representative_days(
    load_csv_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    slug_map=None,
    value_column: str = "value",
    load_column: str = "load_mw",
):
    """Convert per-country load CSVs into the representative-days raw format.

    Input CSV columns required: ``timestamp`` (datetime), ``hour`` (0-8759), ``load_mw`` (or custom ``load_column``).
    Output columns: ``zone, month, day, hour, <value_column>`` with hour in 0-23 repeating.
    """
    slug_map = slug_map or {}
    records = []
    for path in load_csv_paths:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Load profile not found: {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        if load_column not in df.columns:
            raise ValueError(f"Missing '{load_column}' in {csv_path}")
        if "timestamp" not in df.columns:
            raise ValueError(f"Missing 'timestamp' column in {csv_path}")
        df[load_column] /= max(df[load_column])

        slug = csv_path.stem
        if slug.startswith("load_profile_"):
            slug = slug[len("load_profile_") :]
        zone = slug_map.get(slug, slug)

        subset = df[["timestamp", "hour", load_column]].copy()
        subset["zone"] = zone
        subset["month"] = subset["timestamp"].dt.month
        subset["day"] = subset["timestamp"].dt.day
        subset["hour"] = subset["timestamp"].dt.hour
        subset = subset.rename(columns={load_column: value_column})
        records.append(subset[["zone", "month", "day", "hour", value_column]])

    if not records:
        raise ValueError("No load profile CSVs provided for representative-days export.")

    combined = pd.concat(records, ignore_index=True)
    combined = combined.sort_values(["zone", "month", "day", "hour"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, float_format="%.4f")
    return output_path


# --------------------------------------------------------------------------- #
# Workflow orchestrator
# --------------------------------------------------------------------------- #

def run_load_workflow(
    countries: Union[Sequence[str], str],
    dataset_path=None,
    output_dir: Union[str, Path] = "output",
    year: int = 2020,
    plot: bool = True,
    generate_heatmap_boxplot: bool = True,
    verbose: bool = True,
    export_for_representative_days=None,
    slug_map=None,
    value_column: str = "value",
    load_column: str = "load_mw",
):
    """Execute the Toktarova load pipeline and optional representative-days export.

    Returns dict with per-country CSV paths and optional combined representative-days CSV path.
    """
    if isinstance(countries, str):
        countries_list = [countries]
    else:
        countries_list = list(countries or [])

    if not countries_list:
        raise ValueError("At least one country must be provided to run_load_workflow.")

    slug_map = dict(slug_map or {})
    csv_paths = []
    for country in countries_list:
        _, csv_path, _ = load_country_load_profile(
            country=country,
            dataset_path=dataset_path,
            output_dir=output_dir,
            year=year,
            plot=plot,
            generate_heatmap_boxplot=generate_heatmap_boxplot,
            verbose=verbose,
        )
        csv_paths.append(csv_path)
        slug = _slugify(country)
        slug_map.setdefault(slug, country)

    repr_days_input = None
    if export_for_representative_days:
        repr_days_input = export_load_profiles_for_representative_days(
            load_csv_paths=csv_paths,
            output_path=export_for_representative_days,
            slug_map=slug_map,
            value_column=value_column,
            load_column=load_column,
        )

    return {"csv_paths": csv_paths, "repr_days_input": repr_days_input}


# --------------------------------------------------------------------------- #
# Timezone helpers (currently unused, kept for compatibility)
# --------------------------------------------------------------------------- #

def extract_time_zone(countries: Sequence[str], name_map):
    """Geocode time zones for SPLAT-style country names (currently unused)."""
    standard_names = [name_map.get(c, c) for c in countries]
    tf = TimezoneFinder()
    geolocator = Nominatim(user_agent="splat_timezones")
    country_timezones = {}
    for name, std_name in zip(countries, standard_names):
        try:
            location = geolocator.geocode(std_name, timeout=10)
            if location:
                tz = tf.timezone_at(lat=location.latitude, lng=location.longitude)
                country_timezones[name] = tz
            else:
                print(f"Could not geocode: {std_name}")
        except Exception as exc:
            print(f"Error with {std_name}: {exc}")
    return country_timezones


def convert_to_utc(row, country_timezones):
    """Convert a local timestamp to UTC using a timezone map (currently unused)."""
    ctry = row["CtryName"]
    local_zone = pytz.timezone(country_timezones[ctry])
    local_time = local_zone.localize(row["timestamp"], is_dst=None)
    return local_time.astimezone(pytz.utc)


if __name__ == "__main__":
    output_dir = BASE_DIR / "output_workflow" / "load_standalone"
    
    countries_example = ["Bosnia and Herzegovina", "Croatia"]
    
    run_load_workflow(
        countries=countries_example,
        dataset_path=None,
        output_dir=output_dir,
        year=2020,
        plot=True,
        export_for_representative_days=output_dir / "load_profiles.csv",
    )
    print(f"[load-standalone] Outputs written to {output_dir}")
