# Versions & Changelog

The full changelog is maintained in [CHANGELOG.md](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/CHANGELOG.md) on GitHub.

!!! info "What \"beta\" means here"
    The optimization core is mature and used in live World Bank studies. What is **not
    frozen** is the input/output schema — file names, folder layout, parameter names — and
    the [Dashboard](../run/run_dashboard.md), which cannot launch a run yet.

    Breaking changes are listed below. Pin an EPM version for the duration of a study.

---

## [9.0.1-beta] - 2026-05-27

### Fixed
- H2 (hydrogen) module corrections

### Changed
- Data preparation documentation rewritten with phase-based methodology
- Improved diagram visuals and layout

---

## [9.0-beta] - 2026-03-17

### Added
- EPM Dashboard: interactive web interface for results visualization (beta)
- Windows installer (`.exe`) for simplified setup
- Pan-Arab electricity market example dataset
- Remote server execution guide
- Dispatch-only run mode

### Changed
- Output folder structure reorganized
- Documentation fully overhauled

### Fixed
- Various input data corrections for EAPP dataset

---

## [9.0-beta] - 2024-11-18

Initial public release on GitHub.

- Python orchestration layer wrapping the GAMS model
- Full documentation site (MkDocs + GitHub Pages)
- GitHub Actions CI: automated model tests and documentation deployment
- East Africa Power Pool (EAPP) example dataset
- Postprocessing pipeline with dispatch, energy mix, and capacity charts

---

> EPM versions prior to 9.0-beta were internal World Bank tools not tracked on GitHub.  
> The last of them, the Excel-driven **v8.5**, is described in [Legacy version](legacy_v8_5.md).  
> Archived releases are available on [Zenodo](https://doi.org/10.5281/zenodo.15591290).
