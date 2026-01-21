# CLAUDE.md - EPM Electricity Planning Model

## Project Overview

This is the **Electricity Planning Model (EPM)**, a capacity expansion and dispatch optimization model used for World Bank power sector planning studies. The model is implemented in **GAMS** (General Algebraic Modeling System) with a **Python** orchestration layer.

## Critical Rules

### GAMS Files (.gms) - HIGHLY RESTRICTED
- **NEVER modify GAMS files without explicit user approval**
- **Read-only by default** - always ask before making any changes
- Critical files requiring extra caution:
  - `epm/base.gms` - Core mathematical formulation (1,282 lines)
  - `epm/main.gms` - Entry point and orchestration (949 lines)
- Preserve equation naming conventions (e.g., `eq_*`, `def_*`)
- GAMS is case-insensitive but maintain consistent casing for readability

### Python Files (.py) - More Flexible
- Can modify freely as long as tests pass
- Always run full model test after changes
- Match existing code style in the codebase
- Use `pathlib` or `os.path` for cross-platform path handling

### Protected Data - NEVER MODIFY
- `epm/input/data_*/` - All country-specific input folders (data_turkiye, etc.)
- Always ask before editing any data files

## Project Structure

```
EPM_main/
├── epm/                    # Main model code
│   ├── main.gms           # GAMS entry point
│   ├── base.gms           # Core optimization model
│   ├── input_readers.gms  # CSV data loading
│   ├── generate_report.gms # Output extraction
│   ├── epm.py             # Python orchestrator
│   ├── input/             # Input data folders
│   ├── output/            # Generated results
│   └── resources/         # Default parameters
├── pre-analysis/          # Data preprocessing pipelines
├── tools/                 # Utility scripts
├── docs/                  # Jupyter Book documentation
└── config/                # Configuration templates
```

## Environment Setup

```bash
# Python 3.10 with conda
conda create -n esmap_env python=3.10
conda activate esmap_env
pip install -r requirements.txt
```

Required: GAMS installation with valid license (CPLEX solver is default)

## Common Commands

### IMPORTANT: Always use this method to run tests
```bash
conda activate esmap_env
python epm.py --simple
```

**NEVER use `python -m epm.epm`** - always use `python epm.py` directly from the project root.

### Quick Test (Required after changes)
```bash
conda activate esmap_env
python epm.py --simple
```

### Full Model Test
```bash
conda activate esmap_env
python epm.py --folder_input data_test --cpu 4
```

### Input Validation Only
```bash
conda activate esmap_env
python epm.py --folder_input data_test --diagnostic
```

### Build Documentation
```bash
make html  # Generates Jupyter Book in docs/_build/
```

## Key Documentation

- [Run with Python](docs/run/run_python.md) - Main execution guide
- [Input Structure](docs/input/input_structure.md) - Data organization
- [Input Details](docs/input/input_detailed.md) - Parameter documentation
- [Model Formulation](docs/model/model_formulation.md) - Mathematical equations
- [Output Guide](docs/output/postprocessing_output.md) - Results interpretation

## Workflow Guidelines

- **Small changes**: Implement directly, then run full test
- **Large features**: Discuss approach first before implementing
- **GAMS changes**: Always require explicit approval and discussion
- **Python changes**: Implement freely but validate with full model run

## Common Pitfalls

1. **CSV encoding**: Input files must be UTF-8 encoded
2. **Path handling**: Use `pathlib` for cross-platform compatibility (Windows/Mac)
3. **GAMS/Python case sensitivity**: GAMS is case-insensitive, Python is not - be careful with parameter names
4. **Solver configuration**: CPLEX options are in `epm/input/*/cplex/*.opt` - avoid modifying

## File Types

| Extension | Description | Modification Rules |
|-----------|-------------|-------------------|
| `.gms` | GAMS model code | Read-only, ask before edit |
| `.py` | Python code | Can modify with tests |
| `.csv` | Input/output data | Protected in data_* folders |
| `.gdx` | GAMS binary data | Generated, don't edit |
| `.geojson` | Geographic data | Can modify for visualization |

## Architecture Notes

1. **Execution flow**: Python (`epm.py`) → GAMS (`main.gms`) → Results → Postprocessing
2. **Model types**: MIP (mixed-integer) or RMIP (relaxed) optimization
3. **Time representation**: Uses representative days/hours for computational efficiency
4. **Multi-zone**: Supports multiple interconnected zones with transmission constraints
