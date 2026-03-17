# How EPM Works

EPM is implemented in **GAMS** (General Algebraic Modeling System), with a **Python** orchestration layer that handles input preparation, multi-scenario execution, and postprocessing. No GAMS or Python knowledge is required for standard use, though familiarity with either unlocks deeper customization and flexibility.

This section provides a quick map of the model, the repository, and the typical workflow, so you know where everything lives before diving in.

---

## Typical workflow

<div style="display: flex; align-items: stretch; margin: 1.5rem 0; gap: 0;">

  <div style="flex: 1; background: #fdf8f0; border: 1px solid #d4b87a; border-radius: 8px; padding: 0.65rem 0.7rem; text-align: center;">
    <div style="font-size: 1.3rem; font-weight: 800; color: #1b3a5c; line-height: 1;">1</div>
    <div style="font-size: 0.85rem; font-weight: 700; color: #1b3a5c; margin: 0.2rem 0 0.25rem;">Install</div>
    <div style="font-size: 0.7rem; color: #475569;">Set up Python and GAMS</div>
    <div style="margin-top: 0.4rem;"><a href="../../run/run_installation/" style="font-size: 0.7rem; color: #C8952C; font-weight: 600;">Installation →</a></div>
  </div>

  <div style="display: flex; align-items: center; padding: 0 0.4rem; color: #C8952C; font-size: 1.2rem; font-weight: 300;">→</div>

  <div style="flex: 1; background: #fdf8f0; border: 1px solid #d4b87a; border-radius: 8px; padding: 0.65rem 0.7rem; text-align: center;">
    <div style="font-size: 1.3rem; font-weight: 800; color: #1b3a5c; line-height: 1;">2</div>
    <div style="font-size: 0.85rem; font-weight: 700; color: #1b3a5c; margin: 0.2rem 0 0.25rem;">Prepare inputs</div>
    <div style="font-size: 0.7rem; color: #475569;">Via Dashboard or CSV files</div>
    <div style="margin-top: 0.4rem;"><a href="../../input/input_setup/" style="font-size: 0.7rem; color: #C8952C; font-weight: 600;">Input →</a></div>
  </div>

  <div style="display: flex; align-items: center; padding: 0 0.4rem; color: #C8952C; font-size: 1.2rem; font-weight: 300;">→</div>

  <div style="flex: 1; background: #fdf8f0; border: 1px solid #d4b87a; border-radius: 8px; padding: 0.65rem 0.7rem; text-align: center;">
    <div style="font-size: 1.3rem; font-weight: 800; color: #1b3a5c; line-height: 1;">3</div>
    <div style="font-size: 0.85rem; font-weight: 700; color: #1b3a5c; margin: 0.2rem 0 0.25rem;">Run</div>
    <div style="font-size: 0.7rem; color: #475569;">Via <a href="../../run/run_dashboard/" style="color: #475569;">Dashboard</a>, <a href="../../run/run_python/" style="color: #475569;">Python</a>, or <a href="../../run/run_gams_studio/" style="color: #475569;">GAMS Studio</a></div>
    <div style="margin-top: 0.4rem;"><a href="../../run/run_python/" style="font-size: 0.7rem; color: #C8952C; font-weight: 600;">Run →</a></div>
  </div>

  <div style="display: flex; align-items: center; padding: 0 0.4rem; color: #C8952C; font-size: 1.2rem; font-weight: 300;">→</div>

  <div style="flex: 1; background: #fdf8f0; border: 1px solid #d4b87a; border-radius: 8px; padding: 0.65rem 0.7rem; text-align: center;">
    <div style="font-size: 1.3rem; font-weight: 800; color: #1b3a5c; line-height: 1;">4</div>
    <div style="font-size: 0.85rem; font-weight: 700; color: #1b3a5c; margin: 0.2rem 0 0.25rem;">Analyze results</div>
    <div style="font-size: 0.7rem; color: #475569;">Dashboard or CSV files</div>
    <div style="margin-top: 0.4rem;"><a href="../../output/output_overview/" style="font-size: 0.7rem; color: #C8952C; font-weight: 600;">Output →</a></div>
  </div>

</div>

---

??? "End-to-end architecture"

    EPM can be launched from the **Dashboard** or directly via the Python CLI. Either way, inputs flow through a routing layer (`config.csv`) to GAMS, which solves the optimization and writes a binary results file. Python postprocessing converts that to CSV outputs, which the Dashboard reads for visualization.

    <div class="compact-diagram" markdown="1">
    ```mermaid
    flowchart TD
        subgraph ui ["User interface\n"]
            DASH(["<b>EPM Dashboard</b>\nlaunch · configure · visualize"])
        end

        CLI(["<b>python epm.py</b>"])

        subgraph inputs ["Input layer"]
            CONFIG["<b>config.csv</b>\nRouting table"]
            SCEN["<b>scenarios.csv</b>\nScenario overlays (optional)"]
            CSV["<b>Input CSVs</b>\npSettings · supply/ · load/ · ..."]
        end

        subgraph core ["Core model\n"]
            GAMS["<b>GAMS</b>\nbase.gms · CPLEX"]
        end

        POST["<b>Python postprocessing</b>\nepmresults.gdx → CSV"]
        OUT[("<b>CSV Outputs</b>")]

        DASH -->|launch| CLI
        CLI --> CONFIG
        SCEN -.->|overlays| CONFIG
        CONFIG --> CSV
        CSV --> core
        GAMS --> POST
        POST --> OUT
        OUT -->|results| DASH
    ```
    </div>

??? "Repository structure"

    EPM is fully open-source and hosted on [GitHub](https://github.com/ESMAP-World-Bank-Group/EPM). The repository contains the GAMS model core, the Python orchestration layer, input data examples, pre-analysis tools, and this documentation. Everything you need to run, extend, or contribute to the model is there.

    ```plaintext
    EPM/
    ├── epm/                    # Core model — work here
    │   ├── epm.py              # Python entry point (always run from here)
    │   ├── main.gms            # GAMS orchestration
    │   ├── base.gms            # Core optimization equations
    │   ├── input/              # One subfolder per study (data_test, data_senegal, ...)
    │   ├── postprocessing/     # Output scripts and chart config
    │   ├── resources/          # Shared defaults and column headers
    │   └── output/             # Generated results (auto-created on run)
    ├── pre-analysis/           # Data preprocessing pipelines
    ├── tools/                  # Utility scripts
    ├── docs/                   # This documentation
    └── requirements.txt
    ```

    Each study lives in its own folder inside `epm/input/`. The minimum required files are a `config.csv` (routing table) and the CSV files it points to.

---

## About this documentation

| Tab | What it covers |
|---|---|
| **Installation & Run** | Getting EPM running: install, CLI options, GAMS Studio, remote server |
| **Model** | The math: objective function, constraints, time representation |
| **Input** | Input file formats, parameter catalog, typical values, open data sources |
| **Output** | Output files, charts, Tableau dashboard, Python postprocessing |
| **Contributing** | How to report issues, contribute code or documentation |
| **Resources** | Pre-processing tools, planning process, publications |
