"""EPIAS Transparency Platform downloader for Turkey hourly electricity consumption.

Uses the eptr2 package with EPTR2 class (handles TGT auth internally).
Credentials are read from config/api_tokens.ini under [api_tokens]:
  epias_username = ...
  epias_password = ...
"""

from __future__ import annotations

import time
import warnings
from configparser import ConfigParser
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR           = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_PATH = BASE_DIR / "config" / "api_tokens.ini"


def _load_credentials(config_path: Path | None = None) -> tuple[str, str]:
    path = Path(config_path or DEFAULT_TOKEN_PATH)
    if not path.exists():
        raise FileNotFoundError(f"api_tokens.ini not found: {path}")
    cfg = ConfigParser()
    cfg.read(path)
    username = cfg.get("api_tokens", "epias_username", fallback=None)
    password = cfg.get("api_tokens", "epias_password", fallback=None)
    if not username or not password:
        raise RuntimeError("epias_username / epias_password missing in api_tokens.ini")
    return username, password


def download_turkey_load(
    start_year: int,
    end_year: int,
    output_path: Path | None = None,
    refresh: bool = False,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """Download Turkey national hourly load (MW) from EPIAS for start_year..end_year inclusive.

    Parameters
    ----------
    start_year, end_year : int   inclusive year range
    output_path : Path, optional saves CSV and uses as cache on future calls
    refresh : bool               re-download even if cache exists
    config_path : Path, optional path to api_tokens.ini
    """
    if output_path and Path(output_path).exists() and not refresh:
        print(f"[epias] Loading cached Turkey data from {output_path}")
        df = pd.read_csv(output_path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    from eptr2 import EPTR2

    username, password = _load_credentials(config_path)
    client = EPTR2(username=username, password=password, ssl_verify=False)
    print(f"[epias] Authenticated as {username}")

    frames = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            from calendar import monthrange
            last_day = monthrange(year, month)[1]
            start = f"{year}-{month:02d}-01"
            end   = f"{year}-{month:02d}-{last_day:02d}"
            label = f"{year}-{month:02d}"
            try:
                df_m = client.call("rt-cons", start_date=start, end_date=end)
                if df_m is None or df_m.empty:
                    print(f"[epias] {label}: no data")
                    continue
                # Parse timestamp + consumption columns
                ts_col  = next((c for c in df_m.columns if "date" in c.lower()), df_m.columns[0])
                val_col = next((c for c in df_m.columns if "consum" in c.lower() or "tuket" in c.lower()), df_m.columns[-1])
                df_m = df_m[[ts_col, val_col]].copy()
                df_m.columns = ["timestamp", "load_mw"]
                df_m["timestamp"] = pd.to_datetime(df_m["timestamp"], utc=True)
                df_m = df_m.set_index("timestamp")
                frames.append(df_m)
                print(f"[epias] {label}: {len(df_m)} rows")
            except Exception as exc:
                print(f"[epias] {label}: FAILED — {exc}")
            time.sleep(0.5)

    if not frames:
        raise RuntimeError("No Turkey load data downloaded from EPIAS.")

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.resample("h").mean()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path)
        print(f"[epias] Saved {len(combined)} rows → {output_path}")

    return combined


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Turkey hourly load from EPIAS")
    parser.add_argument("--start",   type=int, default=2018)
    parser.add_argument("--end",     type=int, default=2024)
    parser.add_argument("--output",  type=str, default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    default_out = BASE_DIR / "output_workflow" / "blacksea_run1" / "entsoe" / "load" / "turkey_epias_load.csv"
    out = Path(args.output) if args.output else default_out
    df  = download_turkey_load(args.start, args.end, out, refresh=args.refresh)
    print(f"\nDone: {len(df)} hourly rows  {df.index[0]} → {df.index[-1]}")
