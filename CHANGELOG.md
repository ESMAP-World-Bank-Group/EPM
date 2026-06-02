# Changelog

All notable changes to EPM are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versions follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

---

## [Unreleased]

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
