"""
Build a multi-year (climate-year) dataset of solar PV and wind capacity factors per
EPM zone from the ERA5 reanalysis, for the Monte Carlo 'climate_year' feature and for
any other use of historical weather years (e.g. representative days per weather year).

Source
------
ERA5 hourly point time series (Copernicus Climate Change Service, dataset
'reanalysis-era5-single-levels-timeseries'), downloaded with the CDS API. Requires a CDS
account and API key in ~/.cdsapirc (https://cds.climate.copernicus.eu/how-to-api) and the
ERA5 licence accepted on the dataset page. Contains modified Copernicus Climate Change
Service information.

Conversion (one site per zone, see the sites file)
---------------------------------------------------
- PV: GHI (ssrd / 3600 s) with a cell-temperature correction:
  CF = GHI/1000 * PR * (1 + gamma * (Tcell - 25)), Tcell = T2m + GHI/800 * (NOCT - 20),
  PR = 0.86, gamma = -0.4 %/K, NOCT = 45 C; capped to [0, 1].
- Wind: 100 m wind speed through a generic normalised power curve (IEC class II type,
  cut-in 3 m/s, rated 13 m/s, cut-out 25 m/s), times 0.9 for wake and availability losses.

These simple models are meant for year-to-year *variations* (the Monte Carlo uses the ratio
of a year to the climatology, so absolute levels cancel). They do not replace a site
resource assessment.

Outputs (in --output-dir)
-------------------------
- raw/<zone>.csv            ERA5 hourly series as downloaded (UTC), cached: reruns skip them
- hourly/<zone>.csv.gz      hourly PV and wind capacity factors (UTC)
- vre_monthly_cf.csv        zone, tech (PV / Wind), year, month, cf
- metadata.json             source, years, sites and conversion parameters

Usage
-----
python fetch_climate_years.py --sites zone_sites_sapp.csv --start-year 1994 --end-year 2023 \
    --output-dir output/sapp
"""
import argparse
import glob
import json
import os
import shutil
import tempfile
import time
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

DATASET = 'reanalysis-era5-single-levels-timeseries'
VARIABLES = ['2m_temperature', 'surface_solar_radiation_downwards',
             '100m_u_component_of_wind', '100m_v_component_of_wind']

PV = {'performance_ratio': 0.86, 'gamma': -0.004, 'noct': 45.0}
# Generic normalised power curve: wind speed (m/s) -> output (p.u.); 0 above cut-out.
WIND_CURVE = {'speed': [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25],
              'power': [0, 0, 0.04, 0.11, 0.21, 0.34, 0.49, 0.65, 0.79, 0.90, 0.97, 1.0, 1.0]}
WIND_CUT_OUT = 25.0
WIND_LOSSES = 0.9


def download_site(client, zone, lat, lon, start_year, end_year, raw_dir, retries=3):
    """Download the ERA5 hourly series of one site (cached in raw/<zone>.csv)."""
    target = os.path.join(raw_dir, f'{zone}.csv')
    if os.path.exists(target):
        return target
    request = {
        'variable': VARIABLES,
        'location': {'latitude': float(lat), 'longitude': float(lon)},
        'date': [f'{start_year}-01-01/{end_year}-12-31'],
        'data_format': 'csv',
    }
    for attempt in range(1, retries + 1):
        tmp = tempfile.mkdtemp()
        try:
            archive = os.path.join(tmp, 'series.zip')
            client.retrieve(DATASET, request, archive)
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as z:
                    z.extractall(tmp)
                csvs = glob.glob(os.path.join(tmp, '*.csv'))
                if not csvs:
                    raise RuntimeError('no CSV in the downloaded archive')
                shutil.move(csvs[0], target)
            else:
                shutil.move(archive, target)
            return target
        except Exception as err:  # network / queue errors: retry
            print(f'  {zone}: attempt {attempt} failed: {err}')
            if attempt == retries:
                raise
            time.sleep(30 * attempt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def to_capacity_factors(raw_csv):
    """ERA5 hourly series -> hourly PV and wind capacity factors (UTC)."""
    df = pd.read_csv(raw_csv, parse_dates=['valid_time']).rename(columns={'valid_time': 'time'})
    ghi = (df['ssrd'].clip(lower=0) / 3600.0)  # J/m2 accumulated over the hour -> W/m2
    t_cell = (df['t2m'] - 273.15) + ghi / 800.0 * (PV['noct'] - 20)
    pv = ghi / 1000.0 * PV['performance_ratio'] * (1 + PV['gamma'] * (t_cell - 25))
    speed = np.hypot(df['u100'], df['v100'])
    wind = np.interp(speed, WIND_CURVE['speed'], WIND_CURVE['power']) * WIND_LOSSES
    wind = np.where(speed > WIND_CUT_OUT, 0.0, wind)
    return pd.DataFrame({'time': df['time'], 'PV': pv.clip(0, 1).round(4), 'Wind': np.round(wind, 4)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sites', required=True, help='CSV with columns zone, lat, lon (one site per zone)')
    parser.add_argument('--start-year', type=int, default=1994)
    parser.add_argument('--end-year', type=int, default=2023)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--zones', default=None, help="Optional ';'-separated subset of zones")
    args = parser.parse_args()

    import cdsapi  # imported here so --help works without it

    sites = pd.read_csv(args.sites)
    if args.zones:
        wanted = {z.strip() for z in args.zones.split(';')}
        sites = sites.loc[sites['zone'].isin(wanted)]
    raw_dir = os.path.join(args.output_dir, 'raw')
    hourly_dir = os.path.join(args.output_dir, 'hourly')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(hourly_dir, exist_ok=True)

    client = cdsapi.Client(quiet=True, progress=False)
    monthly, failed = [], []
    for i, site in enumerate(sites.itertuples(), start=1):
        t0 = time.time()
        print(f'[{i}/{len(sites)}] {site.zone} ({site.lat}, {site.lon})', flush=True)
        try:
            raw = download_site(client, site.zone, site.lat, site.lon, args.start_year, args.end_year, raw_dir)
        except Exception as err:
            print(f'  {site.zone}: FAILED ({err})', flush=True)
            failed.append(site.zone)
            continue
        cf = to_capacity_factors(raw)
        cf.to_csv(os.path.join(hourly_dir, f'{site.zone}.csv.gz'), index=False)
        long = cf.melt(id_vars='time', var_name='tech', value_name='cf')
        long['year'], long['month'] = long['time'].dt.year, long['time'].dt.month
        stats = long.groupby(['tech', 'year', 'month'], as_index=False)['cf'].mean()
        stats.insert(0, 'zone', site.zone)
        monthly.append(stats)
        annual = stats.groupby('tech')['cf'].mean().round(3).to_dict()
        print(f'  done in {time.time() - t0:.0f}s, mean CF {annual}', flush=True)

    if monthly:
        out = pd.concat(monthly, ignore_index=True)
        out['cf'] = out['cf'].round(5)
        out.to_csv(os.path.join(args.output_dir, 'vre_monthly_cf.csv'), index=False)
    metadata = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'source': f'ERA5 hourly point time series ({DATASET}), Copernicus Climate Change Service',
        'licence': 'Contains modified Copernicus Climate Change Service information (ERA5 licence)',
        'years': [args.start_year, args.end_year],
        'time_zone': 'UTC',
        'pv_model': PV, 'wind_curve': WIND_CURVE, 'wind_cut_out': WIND_CUT_OUT, 'wind_losses': WIND_LOSSES,
        'sites': sites.to_dict(orient='records'),
        'failed_zones': failed,
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f'Finished: {len(monthly)} zone(s), {len(failed)} failed {failed}', flush=True)


if __name__ == '__main__':
    main()
