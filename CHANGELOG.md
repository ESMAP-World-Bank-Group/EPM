# Changelog

All notable changes to EPM are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versions follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

---

## [Unreleased]

### Fixed
- Post-processing: GAMS special values (`EPS`, `UNDF`, `NA`, `±INF`) are now
  translated to numbers when output CSVs are read, instead of reaching pandas as
  text. A single `EPS` left in a `value` column made pandas infer `object` dtype,
  and `cumsum`/`sum` then concatenated strings instead of adding them: cumulative
  cost files grew quadratically (one run produced a 122 MB `pCostsMerged.csv`) and,
  more importantly, held wrong numbers. `EPS` maps to `0` (a stored zero, not a
  missing value). Runs affected before this fix must be regenerated —
  `pDiscountedWeightedCostsCumulated` values were incorrect, not merely bulky.
- Dashboard: the same translation now applies when the dashboard reads output
  CSVs. It previously used `pd.to_numeric(errors='coerce')`, which avoided the
  string concatenation but mapped `EPS` to `NaN` — and `load_yearly_zone` drops
  `NaN` rows, so stored zeros disappeared from the charts instead of showing as
  zero.

### Added
- `epm/gams_values.py`: single definition of GAMS special-value handling
  (`GAMS_SPECIAL_VALUES`, `coerce_value_column`, `read_output_csv`), shared by
  `epm/output_treatment.py` and `dashboard/data_loader.py`. Duplicating this
  logic is how the bug came back the first time, so it lives in one place.
- `tools/test_output_treatment.py`: regression tests for the above, including a
  guard that fails if an unprotected `pd.read_csv` is added to any of the
  guarded modules (post-processing or dashboard).
- `tools/audit_postprocessing_sync.py`: reports which branches carry the
  post-processing fixes, so drift between study branches stays visible.

---

## [9.0.1-beta] - 2026-05-27

### Fixed
- H2 (hydrogen) module corrections

### Changed
- Data preparation documentation rewritten with phase-based methodology
- Improved diagram visuals and layout in documentation

---

## [9.0.0] - 2026-03-17

### Added
- EPM Dashboard: interactive web interface for results visualization (beta)
- Windows installer (`.exe`) for simplified setup
- Pan-Arab electricity market example dataset (`data_pan_arab`)
- Remote server execution guide
- MCP (Model Context Protocol) integration documentation
- Dispatch-only run mode

### Changed
- Output folder structure reorganized
- Documentation fully overhauled: new structure, improved navigation, expanded guides
- Introduction and case studies pages expanded with regional examples

### Fixed
- Various input data corrections for `data_eapp`

---

## [9.0-beta] - 2024-11-18

### Added
- Initial public release on GitHub
- Python orchestration layer (`epm.py`) wrapping GAMS model
- Full documentation site (MkDocs + GitHub Pages)
- GitHub Actions CI: automated GAMS model test + documentation deployment
- Representative days algorithm documentation and notebook
- Postprocessing pipeline: dispatch graphs, energy mix figures, capacity charts
- East Africa Power Pool (EAPP) example dataset
- `CONTRIBUTING.md`, issue templates, `.gitignore`
- Utility functions (`utils.py`) for visualization
- `colors.csv` for standardized technology colors

---

> **Note:** EPM versions prior to 9.0-beta were internal World Bank tools not tracked on GitHub.  
> For archived releases, see [Zenodo](https://doi.org/10.5281/zenodo.15591290).
