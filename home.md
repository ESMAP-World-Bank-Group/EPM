# Electricity Planning Model (EPM)

**EPM** is a least-cost capacity expansion and dispatch model for power system planning, developed by the World Bank's [Energy Sector Management Assistance Program (ESMAP)](https://esmap.org). It has been deployed in over 88 countries to inform energy investment and policy decisions.

---

## Where do I start?

| I want to... | Go here |
|---|---|
| Understand what EPM does | [Introduction](docs/introduction/introduction) |
| Install EPM and run my first simulation | [Installation](docs/run/run_installation) |
| Understand how to structure my input data | [Input Overview](docs/input/input_overview) |
| Run multiple scenarios or a sensitivity analysis | [Python — Advanced](docs/run/run_python_advanced) |
| Visualize results in Tableau | [Tableau Dashboards](docs/output/postprocessing_tableau) |
| Read the mathematical formulation | [Model Formulation](docs/model/model_formulation) |

---

## How EPM works

```{mermaid}
flowchart LR
    A("**1. Collect data**\nDemand, generation,\ncosts, policies") -->
    B("**2. Prepare inputs**\nCSV files in\n`input/data_*/`") -->
    C("**3. Configure**\n`config.csv`,\nscenarios, solver") -->
    D("**4. Run EPM**\n`python epm.py`\n--folder_input ...") -->
    E("**5. Optimize**\nGAMS solves\ncapacity + dispatch") -->
    F("**6. Postprocess**\nCSV tables +\nplots generated") -->
    G("**7. Analyze**\nTableau / Python\ndashboards")

    style A fill:#dbeafe,stroke:#3b82f6
    style B fill:#dbeafe,stroke:#3b82f6
    style C fill:#dbeafe,stroke:#3b82f6
    style D fill:#dcfce7,stroke:#16a34a
    style E fill:#dcfce7,stroke:#16a34a
    style F fill:#fef9c3,stroke:#ca8a04
    style G fill:#fef9c3,stroke:#ca8a04
```

**Under the hood:** Python (`epm.py`) orchestrates the run — it reads your configuration, builds scenarios, and launches GAMS in parallel for each one. GAMS solves the optimization and writes results to `epmresults.gdx`. Python then postprocesses everything into CSV tables and charts.

---

## Key capabilities

- **Capacity expansion** — identifies least-cost investment plans across multiple years and zones
- **Dispatch optimization** — co-optimizes generation, reserves, and cross-border trade
- **Scenario analysis** — run hundreds of scenarios in parallel with a single command
- **Sensitivity & Monte Carlo** — built-in support for uncertainty quantification
- **Policy testing** — emissions caps, carbon prices, renewable targets, fuel limits

---

## Quick start

```bash
# 1. Install dependencies
conda create -n epm_env python=3.10
conda activate epm_env
pip install -r requirements.txt

# 2. Run the test dataset (takes ~1 min)
python epm.py --simple

# 3. Run a full solve with 4 CPU cores
python epm.py --folder_input data_test --cpu 4
```

Results are written to `output/simulations_run_<timestamp>/`.

---

## Resources

| Resource | Link |
|---|---|
| Source code | [GitHub — ESMAP-World-Bank-Group/EPM](https://github.com/ESMAP-World-Bank-Group/EPM) |
| Previous versions | [Zenodo archive](https://zenodo.org/communities/esmap-epm) |
| Report an issue | [GitHub Issues](https://github.com/ESMAP-World-Bank-Group/EPM/issues) |
| Cite EPM | See [Introduction](docs/introduction/introduction) |
