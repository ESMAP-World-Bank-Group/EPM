"""Collect the Hungarian data behind edit 4 of the v6 input family (Hungary as a modelled zone).

Network step. It fills a local cache once; build_robg_v6.py --edit 4 then reads the cache
offline, so the build stays reproducible without a connection.

What is fetched
  ENTSO-E Transparency Platform (entsoe-py, token read by entsoe_pipeline.load_api_token):
    load_HU_2023.csv, load_HU_2025.csv      actual total load
    instcap_HU_2025.csv                     installed capacity per production type
    instcap_units_HU_2025.csv               installed capacity per production unit
    gen_HU_hourly_2025.csv                  actual generation per production type, hourly mean
    da_HU_2025.csv                          day-ahead price, EUR/MWh
    flows/flow_<A>_<B>_2025.csv             physical flows on the Hungarian borders
  Renewables Ninja (vre_pipeline.get_renewables_ninja, same call as the other zones):
    ninja_hungary/vre_rninja_hungary_solar.csv and _wind.csv, year 2023

Every file is listed with its size and sha256 in hungary_v6_manifest.csv. A file already in
the cache is not fetched again unless --refresh is given.

Usage (from the EPM folder):
    python -I -X utf8 pre-analysis/pipelines/fetch_hungary_v6.py
    python -I -X utf8 pre-analysis/pipelines/fetch_hungary_v6.py --only load --only flows
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
_PRE_ANALYSIS = _HERE.parents[1]
sys.path.insert(0, str(_HERE.parent))

CACHE = _PRE_ANALYSIS.parents[1] / "Data" / "EU" / "cache_entso_e"
FLEET = CACHE / "fleet"
FLOWS = CACHE / "flows"
NINJA = CACHE / "ninja_hungary"
MANIFEST = FLEET / "hungary_v6_manifest.csv"
GWA = _PRE_ANALYSIS / "data" / "gwa"
GWA_ISO3 = "HUN"
GWA_LAYER = "capacity-factor_IEC2"

AREA = "HU"
TZ = "Europe/Brussels"
LOAD_YEARS = (2023, 2025)
GEN_YEARS = (2023, 2025)   # 2023: solar seen by the TSO in the year of the load shape
YEAR = 2025
NINJA_YEAR = 2023
# Geographic centre of Hungary (Pusztavacs), as (lat, lon). The other zones use their centroid too.
NINJA_POINT = (47.2, 19.5)
# Hungarian borders, ENTSO-E area codes. Romania is already in the cache (edit 3 of the v5 family).
NEIGHBOURS = ("AT", "SK", "HR", "SI", "RS", "UA", "RO")
UKRAINE_CODES = ("UA_IPS", "UA", "UA_BEI")

_SECRETS: list[str] = []


def shown(path: Path) -> str:
    """Name of a file in the manifest: relative to the cache, or to the repository for the atlas raster."""
    try:
        return path.relative_to(CACHE).as_posix()
    except ValueError:
        return "EPM/" + path.relative_to(_PRE_ANALYSIS.parent).as_posix()


def scrub(text: str) -> str:
    """Never let a token reach the console: request URLs can carry it."""
    for secret in _SECRETS:
        if secret:
            text = text.replace(secret, "***")
    return text


def say(msg: str) -> None:
    print(scrub(msg), flush=True)


def window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}0101", tz=TZ), pd.Timestamp(f"{year + 1}0101", tz=TZ)


def client():
    from entsoe import EntsoePandasClient
    from entsoe_pipeline import load_api_token
    token = load_api_token("entsoe")
    _SECRETS.append(token)
    return EntsoePandasClient(api_key=token)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Collector:
    def __init__(self, refresh: bool):
        self.refresh = refresh
        self.records: list[dict] = []
        self.failed: list[str] = []
        self._client = None

    @property
    def api(self):
        if self._client is None:
            self._client = client()
        return self._client

    def get(self, path: Path, query: str, fetch) -> bool:
        """Fetch one file unless cached. fetch() returns a DataFrame or Series already shaped."""
        if fetch is None:
            if not path.exists():
                say(f"  FAILED {path.name}: not written")
                self.failed.append(path.name)
                return False
            status = "written"
        elif path.exists() and not self.refresh:
            status = "cached"
        else:
            try:
                data = fetch()
            except Exception as exc:  # the message may hold the request URL: scrub it
                say(f"  FAILED {path.name}: {type(exc).__name__}: {str(exc)[:300]}")
                self.failed.append(path.name)
                return False
            if data is None or len(data) == 0:
                say(f"  FAILED {path.name}: empty answer")
                self.failed.append(path.name)
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(path)
            status = "fetched"
        self.records.append({
            "file": shown(path), "bytes": path.stat().st_size,
            "sha256": sha256_of(path), "query": query, "status": status,
            "written_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        say(f"  {status:8s} {shown(path)}  {path.stat().st_size:,} bytes")
        return True


def aggregated(gen: pd.DataFrame) -> pd.DataFrame:
    """Keep the 'Actual Aggregated' column of each production type, hourly mean, MW."""
    if isinstance(gen.columns, pd.MultiIndex):
        keep = [c for c in gen.columns if c[1] == "Actual Aggregated"]
        gen = gen[keep]
        gen.columns = [c[0] for c in keep]
    return gen.resample("h").mean()


def step_load(c: Collector) -> None:
    say("[load]")
    for year in LOAD_YEARS:
        start, end = window(year)
        c.get(FLEET / f"load_{AREA}_{year}.csv", f"query_load {AREA} {year}",
              lambda: c.api.query_load(AREA, start=start, end=end))


def step_fleet(c: Collector) -> None:
    say("[fleet]")
    start, end = pd.Timestamp(f"{YEAR}0101", tz=TZ), pd.Timestamp(f"{YEAR}1231", tz=TZ)
    c.get(FLEET / f"instcap_{AREA}_{YEAR}.csv", f"query_installed_generation_capacity {AREA} {YEAR}",
          lambda: c.api.query_installed_generation_capacity(AREA, start=start, end=end))
    c.get(FLEET / f"instcap_units_{AREA}_{YEAR}.csv", f"query_installed_generation_capacity_per_unit {AREA} {YEAR}",
          lambda: per_unit(c))


def per_unit(c: Collector) -> pd.DataFrame:
    """Per unit capacities. The answer for Hungary is empty on some windows: try several, keep the first table."""
    windows = (
        (pd.Timestamp(f"{YEAR}0101", tz=TZ), pd.Timestamp(f"{YEAR}1231", tz=TZ)),
        (pd.Timestamp(f"{YEAR}0101", tz="UTC"), pd.Timestamp(f"{YEAR}0102", tz="UTC")),
        (pd.Timestamp(f"{YEAR - 1}0101", tz=TZ), pd.Timestamp(f"{YEAR - 1}1231", tz=TZ)),
        (pd.Timestamp(f"{YEAR + 1}0101", tz=TZ), pd.Timestamp(f"{YEAR + 1}0102", tz=TZ)),
    )
    last = None
    for start, end in windows:
        try:
            table = units_from_xml(raw_client().query_installed_generation_capacity_per_unit(AREA, start=start, end=end))
        except Exception as exc:  # noqa: BLE001 - the next window is tried, the last error is reported
            last = exc
            say(f"    window {start.date()} to {end.date()}: {type(exc).__name__}")
            continue
        if len(table):
            say(f"    window {start.date()} to {end.date()}: {len(table)} units")
            return table
    raise last if last else RuntimeError("no unit in any window")


def raw_client():
    """Raw client: the pandas parser of entsoe-py fails on Hungarian units that carry no voltage level."""
    from entsoe import EntsoeRawClient
    from entsoe_pipeline import load_api_token
    token = load_api_token("entsoe")
    if token not in _SECRETS:
        _SECRETS.append(token)
    return EntsoeRawClient(api_key=token)


def units_from_xml(text: str) -> pd.DataFrame:
    """One row per production unit of an A71 answer; a missing field stays empty."""
    import xml.etree.ElementTree as ET
    from entsoe.mappings import PSRTYPE_MAPPINGS

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    try:  # the raw client decodes the UTF-8 answer as Latin-1: restore the Hungarian names
        text = text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        pass
    rows = {}
    for series in ET.fromstring(text.encode("utf-8")).iter():
        if local(series.tag) != "TimeSeries":
            continue
        cell = {}
        for node in series.iter():
            name = local(node.tag)
            if name in ("registeredResource.mRID", "registeredResource.name", "psrType", "quantity",
                        "inBiddingZone_Domain.mRID", "start",
                        "production_PowerSystemResources.highVoltageLimit") and name not in cell:
                cell[name] = (node.text or "").strip()
        code = cell.get("registeredResource.mRID")
        if not code:
            continue
        rows[code] = {
            "Bidding Zone": cell.get("inBiddingZone_Domain.mRID", ""),
            "Installed Capacity [MW]": cell.get("quantity", ""),
            "Name": cell.get("registeredResource.name", ""),
            "Production Type": PSRTYPE_MAPPINGS.get(cell.get("psrType", ""), cell.get("psrType", "")),
            "Start": cell.get("start", ""),
            "Voltage Connection Level [kV]": cell.get("production_PowerSystemResources.highVoltageLimit", ""),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def step_generation(c: Collector) -> None:
    say("[generation]")
    for year in GEN_YEARS:
        start, end = window(year)
        c.get(FLEET / f"gen_{AREA}_hourly_{year}.csv", f"query_generation {AREA} {year}, hourly mean of 'Actual Aggregated'",
              lambda start=start, end=end: aggregated(c.api.query_generation(AREA, start=start, end=end)))


def step_price(c: Collector) -> None:
    say("[day-ahead price]")
    start, end = window(YEAR)
    c.get(FLEET / f"da_{AREA}_{YEAR}.csv", f"query_day_ahead_prices {AREA} {YEAR}",
          lambda: c.api.query_day_ahead_prices(AREA, start=start, end=end).rename("eur_mwh"))


def step_flows(c: Collector) -> None:
    say("[flows]")
    start, end = window(YEAR)
    for nb in NEIGHBOURS:
        codes = UKRAINE_CODES if nb == "UA" else (nb,)
        for a_is_hu in (True, False):
            name = f"flow_{AREA}_{nb}_{YEAR}.csv" if a_is_hu else f"flow_{nb}_{AREA}_{YEAR}.csv"
            path = FLOWS / name
            done = False
            for code in codes:
                a, b = (AREA, code) if a_is_hu else (code, AREA)
                before = len(c.failed)
                done = c.get(path, f"query_crossborder_flows {a} to {b} {YEAR}",
                             lambda a=a, b=b: c.api.query_crossborder_flows(a, b, start=start, end=end).rename("mw"))
                if done:
                    break
                del c.failed[before:]
            if not done:
                c.failed.append(name)


def step_atlas(c: Collector) -> None:
    """Global Wind Atlas raster of Hungary, same layer and folder as the four rasters of the wind anchoring."""
    say("[wind atlas]")
    from vre_cf_anchoring import fetch_gwa_raster
    path = GWA / f"{GWA_ISO3}_{GWA_LAYER}.tif"
    if not path.exists() or c.refresh:
        try:
            if path.exists():
                path.unlink()
            fetch_gwa_raster(GWA_ISO3, GWA_LAYER, GWA, log=say)
        except Exception as exc:
            say(f"  FAILED {path.name}: {type(exc).__name__}: {str(exc)[:300]}")
    c.get(path, f"globalwindatlas.info api/gis/country/{GWA_ISO3}/{GWA_LAYER}", None)


def step_ninja(c: Collector) -> None:
    say("[renewables ninja]")
    from vre_pipeline import get_renewables_ninja, load_api_token, rninja_output_filename
    _SECRETS.append(load_api_token("renewables_ninja"))
    NINJA.mkdir(parents=True, exist_ok=True)
    for tech in ("solar", "wind"):
        path = NINJA / rninja_output_filename("hungary", tech)
        if not path.exists() or c.refresh:
            try:
                # Same call as run_blacksea_data.step_ninja: defaults of the pipeline, local_time=True.
                get_renewables_ninja(locations={"Hungary": NINJA_POINT}, tech=tech, start_year=NINJA_YEAR,
                                     end_year=NINJA_YEAR + 1, dataset_label="hungary", output=str(NINJA),
                                     local_time=True)
            except Exception as exc:
                say(f"  FAILED {path.name}: {type(exc).__name__}: {str(exc)[:300]}")
        c.get(path, f"renewables.ninja {tech} lat {NINJA_POINT[0]} lon {NINJA_POINT[1]} {NINJA_YEAR}, merra2, pipeline defaults",
              None)


def summary() -> None:
    """Read the cache back and print the figures the plan quotes for Hungary (H3, H6, H11)."""
    say("\n[summary]  (plan figures in brackets)")
    for year in LOAD_YEARS:
        p = FLEET / f"load_{AREA}_{year}.csv"
        if p.exists():
            s = pd.read_csv(p, index_col=0).iloc[:, 0]
            s.index = pd.to_datetime(s.index, utc=True)
            h = s.resample("h").mean()
            say(f"  load {year}: {h.sum() / 1e6:.2f} TWh, peak {h.max() / 1e3:.2f} GW, {int(h.isna().sum())} missing hours of {len(h)}"
                + ("  [49.2 TWh, 7.4 GW]" if year == YEAR else ""))
    p = FLEET / f"instcap_{AREA}_{YEAR}.csv"
    if p.exists():
        cap = pd.read_csv(p, index_col=0).iloc[0]
        cap = cap[cap > 0]
        say("  installed MW: " + ", ".join(f"{k} {v:,.0f}" for k, v in cap.items())
            + "  [nuclear 2.0 GW, gas 4.2, lignite 1.0, bioenergy 0.6, solar 8.2, wind 0.3]")
    p = FLEET / f"instcap_units_{AREA}_{YEAR}.csv"
    if p.exists():
        u = pd.read_csv(p, index_col=0)
        t = u.groupby("Production Type")["Installed Capacity [MW]"].agg(["count", "sum"])
        say("  units: " + ", ".join(f"{k} {int(r['count'])} units {r['sum']:,.0f} MW" for k, r in t.iterrows()))
    p = FLEET / f"gen_{AREA}_hourly_{YEAR}.csv"
    gen_total = None
    if p.exists():
        g = pd.read_csv(p, index_col=0)
        twh = (g.sum() / 1e6)
        twh = twh[twh.abs() > 0.005]
        gen_total = float(twh.sum())
        say("  generation TWh: " + ", ".join(f"{k} {v:.2f}" for k, v in twh.items())
            + f" | total {gen_total:.2f}  [nuclear 16.1, solar 11.0, gas 8.2]")
    net = 0.0
    parts = []
    for nb in NEIGHBOURS:
        pi, pe = FLOWS / f"flow_{nb}_{AREA}_{YEAR}.csv", FLOWS / f"flow_{AREA}_{nb}_{YEAR}.csv"
        if pi.exists() and pe.exists():
            def twh_of(path: Path) -> float:
                s = pd.read_csv(path, index_col=0)["mw"]
                s.index = pd.to_datetime(s.index, utc=True)
                return float(s.resample("h").mean().sum() / 1e6)
            imp, exp = twh_of(pi), twh_of(pe)
            net += imp - exp
            parts.append(f"{nb} in {imp:.2f} out {exp:.2f}")
        else:
            parts.append(f"{nb} missing")
    say("  flows TWh: " + "; ".join(parts) + f" | net imports {net:.2f}  [9.0]")
    p = FLEET / f"da_{AREA}_{YEAR}.csv"
    if p.exists():
        d = pd.read_csv(p, index_col=0).iloc[:, 0]
        d.index = pd.to_datetime(d.index, utc=True)
        say(f"  day-ahead {YEAR}: mean of hourly means {d.resample('h').mean().mean():.2f} EUR/MWh  [108.5]")
    for tech in ("solar", "wind"):
        p = NINJA / f"vre_rninja_hungary_{tech}.csv"
        if p.exists():
            n = pd.read_csv(p)
            say(f"  ninja {tech} {NINJA_YEAR}: {len(n)} hours, mean {n[str(NINJA_YEAR)].mean():.4f}, max {n[str(NINJA_YEAR)].max():.3f}")


STEPS = {"load": step_load, "fleet": step_fleet, "generation": step_generation, "price": step_price,
         "flows": step_flows, "ninja": step_ninja, "atlas": step_atlas}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", action="append", choices=sorted(STEPS), help="run only this step (repeatable)")
    ap.add_argument("--refresh", action="store_true", help="fetch again even if the file is in the cache")
    ap.add_argument("--summary", action="store_true", help="print the summary from the cache, fetch nothing")
    args = ap.parse_args()
    if args.summary:
        summary()
        return 0
    c = Collector(args.refresh)
    for name in (args.only or list(STEPS)):
        try:
            STEPS[name](c)
        except Exception as exc:
            say(f"  FAILED step {name}: {type(exc).__name__}: {str(exc)[:300]}")
            c.failed.append(name)
    if c.records:
        new = pd.DataFrame(c.records)
        if MANIFEST.exists():
            old = pd.read_csv(MANIFEST)
            new = pd.concat([old[~old.file.isin(new.file)], new], ignore_index=True)
        new.sort_values("file").to_csv(MANIFEST, index=False, lineterminator="\n")
        say(f"manifest: {MANIFEST} ({len(new)} files)")
    summary()
    if c.failed:
        say("FAILED: " + ", ".join(sorted(set(c.failed))))
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # last guard: no raw traceback, it could show a request URL
        say(f"[fetch_hungary_v6] FAIL: {type(exc).__name__}: {str(exc)[:300]}")
        sys.exit(2)
