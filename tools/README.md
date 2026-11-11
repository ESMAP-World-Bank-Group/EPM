# EPM Tools

This directory contains utility scripts and tools for working with EPM.

## Files

- **`mcp_server.py`** - MCP (Model Context Protocol) server that allows Cursor AI to run EPM directly
- **`test_epm_simple.py`** - Simple test script to run EPM with minimal configuration
- **`aggregate_zones.py`** - Copy an input folder with sub-zones merged into one zone per country (used for `data_sapp_nuc_rsa_moz` -> `data_sapp_nuc_rsa_moz_agg`). See below.
- **Data publishing/sync** (`publish.ps1`, `sync.ps1`/`sync.sh`, `upload_*.py`) - publish model data to / pull it from the private store (R2/S3). See **[DATA_PUBLISH.md](DATA_PUBLISH.md)** / **docs/run/run_data_sync.md**, and the `Publish.bat` / `Sync.bat` launchers at the repo root.

## MCP Server

The MCP server enables running EPM from within Cursor's AI chat. See the main project documentation for setup instructions.

**Location:** `tools/mcp_server.py`

## Aggregate sub-zones

Builds a new input folder in which the sub-zones of a country (for example `Mozambique_South`, `Mozambique_Central`, `Mozambique_North`) become one zone (`Mozambique`). The source folder is not modified.

```bash
python tools/aggregate_zones.py --src data_sapp_nuc_rsa_moz --dst data_sapp_nuc_rsa_moz_agg --force
cd epm && python epm.py --folder_input data_sapp_nuc_rsa_moz_agg --check_inputs
```

1. Units listed in `MERGE_UNITS` are combined into one (sizes summed), e.g. the three Mozambique battery candidates into `New_BESS_MOZ`.
2. Rows that still use the country name as zone are deleted. EPM ignores them already (zone outside `zcmap` or Status 0), and they would become duplicates.
3. Sub-zones are renamed to the country. The duplicates are merged with one rule per file: demand, transfer limits, minimum imports and generation caps are summed; demand profiles are averaged weighted by energy, VRE profiles weighted by VRE capacity; defaults keep the first value; lines between merged sub-zones are dropped.
4. Map layers are regenerated (`geojson_to_epm.csv` with whole-country rows, then `epm/postprocessing/create_geojson.py`).

`--override PARAM=FILE` (repeatable) uses another file for a parameter of `config.csv`, for example the 16-day `pHours`, `pDemandProfile`, `pVREProfile`, `pVREgenProfile` and `pTradePrice` from `pre-analysis/representative_days/output/sapp_2024/epm_16d/` (section 10 of `Run_repdays_SAPP_2024.ipynb`).

Only the files used by `config.csv` and `input_scenarios.csv` are converted. Every deletion and merge is listed in `<dst>/aggregation_log.md`. The zones to merge (`MAP`) and the units to combine (`MERGE_UNITS`) are set at the top of the script.

## Simple Test Script

Quick way to test EPM without MCP:

```bash
python tools/test_epm_simple.py --folder_input data_test --parallel 1
```

