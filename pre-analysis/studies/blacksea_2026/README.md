# studies/blacksea_2026 — one-off scripts, unmaintained

Scripts written for **one** study (Black Sea 2026: TUR, ROU, BGR, GEO, ARM, AZE) and
moved here from `pre-analysis/` on 2026-08-14.

**Status: kept for traceability, not for reuse.** They document how a given number
reached `epm/input/data_blacksea/`. Nobody undertakes to maintain them, to run them on
another region, or to fix them if they break.

## The sorting rule

A script lives in `pre-analysis/` if it is **generic**: parameterised by country or
region, called by the `Snakefile`, or imported by another module. Otherwise it lives here.

Concretely, these stayed at the root: `compute_epm_demand.py`, `compute_epm_gendata.py`,
`compute_epm_vre.py` (generic EPM builders), `extract_epm_excel.py`,
`generate_zones_geojson.py`, `run_zoning_study.py`, `export_preferred.py`,
`export_zones_to_explorer.py`, `prepare_explorer_data.py`, `aggregate_corridors.py`,
`build_corridors_from_gpkg.py`, `snakemake_helpers.py`, `_paths.py`.

**A new single-use script is created here directly, not at the root.**

## Path anchoring

Scripts that resolved their paths from their own location were given a header that
restores the two original anchors:

```python
_PRE_ANALYSIS = _Path(__file__).resolve().parents[2]   # pre-analysis/
_REPO_ROOT = _PRE_ANALYSIS.parent                      # repository root
```

The move is therefore behaviour-neutral: the inputs read and the outputs written have
not changed. `_PRE_ANALYSIS` is also added to `sys.path`, so that `from pipelines...`
keeps working.

## What is in here

| Group | Scripts |
|---|---|
| Bulgaria | `build_bulgaria_availability.py`, `build_bulgaria_fuelprice.py`, `build_bulgaria_gendata.py` |
| Romania | `add_romania_zone_polygon.py`, `create_romania_2z.py`, `integrate_romania.py`, `prepare_romania_for_blacksea.py`, `fix_provenance_romania.py` |
| Georgia | `compute_georgia_demand.py`, `compute_georgia_vre_profiles.py`, `generate_georgia_index.py`, `generate_georgia_xborder_excel.py`, `read_georgia_pdfs.py`, `read_georgia_pdfs2.py`, `read_georgia_pdfs3.py`, `read_generation_balance.py`, `extract_monthly_xborder.py`, `extract_exports_2023.py` |
| Armenia / Azerbaijan | `compute_armenia_vre_profiles.py`, `extract_ssc_az.py` |
| Türkiye | `add_existing_zones.py` (existing 9-zone configuration) |
| Regional | `run_blacksea_data.py`, `run_pipelines.py`, `generate_blacksea_zones.py`, `build_reference_lines_from_v7.py`, `make_external_corridors_map.py` |

The three `read_georgia_pdfs*.py` are successive iterations over the same PDF; only the
last one that actually ran is authoritative, the others are kept for the record.

## Deliberately not moved

`pre-analysis/make_transmission_map.py` is Black Sea-specific and belongs here, but it
had uncommitted modifications at the time of the tidy-up. To be moved once that work
is finished.
