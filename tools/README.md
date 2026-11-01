# EPM Tools

This directory contains utility scripts and tools for working with EPM.

## Files

- **`mcp_server.py`** - MCP (Model Context Protocol) server that allows Cursor AI to run EPM directly
- **`test_epm_simple.py`** - Simple test script to run EPM with minimal configuration
- **`scenarios_build.py`** - builds `epm/input/data_blacksea/scenarios.csv` from `internal_analysis/scenarios/scenario_grid.yaml` (one reference, grids, scopes, cases, sensitivities layered into Scope_Case[_Sens]_Grid columns), writes `scenario_grid_report.md`; `--check` reports drift between the yaml and the csv, `--show NAME` prints one resolved column
- **`scenario_register.py`** - rebuilds `internal_analysis/scenarios/scenario_register.csv`, the run status of every scenario of the grid (local explorer runs under `epm/output_view`, archive twin, `--r2` for what the store serves, `--note NAME TEXT` kept across rebuilds)
- **Data publishing/sync** (`publish.ps1`, `sync.ps1`/`sync.sh`, `upload_*.py`) - publish model data to / pull it from the private store (R2/S3). See **[DATA_PUBLISH.md](DATA_PUBLISH.md)** and the `Publish.bat` / `Sync.bat` double-click launchers at the repo root.

## MCP Server

The MCP server enables running EPM from within Cursor's AI chat. See the main project documentation for setup instructions.

**Location:** `tools/mcp_server.py`

## Simple Test Script

Quick way to test EPM without MCP:

```bash
python tools/test_epm_simple.py --folder_input data_test --cpu 1
```

