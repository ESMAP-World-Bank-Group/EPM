"""
**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
Description:
    Build input_treated.gdx for one or more scenarios WITHOUT solving the
    model, then run the consistency checks of ``input_treated_check.py``.
    Use it to fix data issues locally before launching a full run.

    GAMS is called on main.gms in compile-only mode (``action=c``): the
    compile-time steps (CSV reading, input_verification.py,
    input_treatment.py and ``$gdxunload input_treated.gdx``) run as in a
    normal launch, nothing is generated or solved. Scenario handling
    (config, scenarios file, -S, --simple, --country) is the one of epm.py.

Usage (from the epm folder, epm_env activated):
    python check_inputs.py --folder_input data_sapp_nuc_rsa_moz
    python check_inputs.py --folder_input data_sapp_nuc_rsa_moz --scenarios -S baseline emi_reduc
    python check_inputs.py --gdx output/simulations_run_xxx/baseline/input_treated.gdx   # check an existing file

Outputs (in output/input_check_<timestamp>/<scenario>/):
    input_treated.gdx, <scenario>_main.log, input_check.csv (one row per finding),
    gams_messages.txt (warnings printed by input_verification / input_treatment)
    and input_check_summary.csv at the run level.

Exit code: 1 if any HIGH finding or a scenario failed to compile, else 0.
**********************************************************************
"""

import argparse
import datetime
import importlib.util
import os
import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MESSAGE_PATTERN = re.compile(r"warning|error|skipped|abort", re.IGNORECASE)


def _load_epm_runner():
    """Load epm.py under the name 'epm_runner'.

    Importing it as 'epm' would shadow the 'epm' package that postprocessing
    imports from. Loaded at import time so multiprocessing workers (spawned on
    Windows) can unpickle launch_epm_multiprocess from 'epm_runner'.
    """
    if "epm_runner" not in sys.modules:
        spec = importlib.util.spec_from_file_location("epm_runner", BASE_DIR / "epm.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["epm_runner"] = module
        spec.loader.exec_module(module)
    return sys.modules["epm_runner"]


epm_runner = _load_epm_runner()


def extract_gams_messages(log_path):
    """Warnings/errors printed between input reading and $gdxunload input_treated.gdx."""
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = next((i for i, l in enumerate(lines) if "INPUT VERIFICATION" in l), 0)
    end = next((i for i, l in enumerate(lines) if "input_treated.gdx" in l and i > start), len(lines))
    seen, out = set(), []
    for line in lines[start:end]:
        line = line.strip()
        if MESSAGE_PATTERN.search(line) and line not in seen:
            seen.add(line)
            out.append(line)
    # GAMS compilation errors after the treatment (e.g. in base.gms) are also worth showing
    out += [l.strip() for l in lines[end:] if l.startswith("***") and "rror" in l]
    return out


def build_treated_inputs(args):
    """Compile main.gms for the selected scenarios; return (run folder, error message or None)."""
    os.chdir(BASE_DIR)  # epm.py resolves GAMS and input paths from the epm folder
    launch_epm_multi_scenarios = epm_runner.launch_epm_multi_scenarios

    label = args.simulation_label or f"input_check_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    folder = BASE_DIR / "output" / label
    error = None
    try:
        launch_epm_multi_scenarios(
            config=args.config,
            scenarios_specification=args.scenarios,
            selected_scenarios=args.selected_scenarios,
            cpu=args.parallel,
            folder_input=os.path.join("input", args.folder_input),
            simple=args.simple,
            simulation_label=label,
            modeltype=args.modeltype,
            country=args.country,
            gams_options=["action=c"],
        )
    except Exception as err:  # keep checking whatever was produced
        error = str(err)
    finally:
        os.chdir(BASE_DIR)
    return folder, error


def check_folder(folder, max_rows):
    from input_treated_check import format_report, run_checks

    summary, any_high = [], False
    scenario_dirs = sorted(p for p in folder.iterdir() if p.is_dir())
    for sdir in scenario_dirs:
        gdx = sdir / "input_treated.gdx"
        log = sdir / f"{sdir.name}_main.log"
        messages = extract_gams_messages(log)
        (sdir / "gams_messages.txt").write_text("\n".join(messages), encoding="utf-8")
        if not gdx.exists():
            any_high = True
            print(f"\n[{sdir.name}] input_treated.gdx was not produced - GAMS stopped earlier. Last messages:")
            print("\n".join("    " + m for m in messages[-15:]) or f"    see {log}")
            summary.append(dict(scenario=sdir.name, status="failed", HIGH=None, MED=None, LOW=None))
            continue
        df = run_checks(gdx)
        df.to_csv(sdir / "input_check.csv", index=False)
        print(format_report(df, title=f"Scenario {sdir.name}: {gdx}", max_rows=max_rows))
        if messages:
            print(f"\n[GAMS] {len(messages)} warning(s) from input_verification / input_treatment "
                  f"(full list: {sdir / 'gams_messages.txt'}):")
            for m in messages[:max_rows]:
                print("    " + (m if len(m) <= 220 else m[:217] + "..."))
        counts = df.severity.value_counts()
        any_high |= counts.get("HIGH", 0) > 0
        summary.append(dict(scenario=sdir.name, status="ok", HIGH=counts.get("HIGH", 0),
                            MED=counts.get("MED", 0), LOW=counts.get("LOW", 0), gams_warnings=len(messages)))
    pd.DataFrame(summary).to_csv(folder / "input_check_summary.csv", index=False)
    return summary, any_high


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build input_treated.gdx (no solve) and check it.")
    parser.add_argument("--folder_input", default="data_test", help="Input folder in epm/input (default: data_test)")
    parser.add_argument("--config", default="config.csv", help="Config file in the input folder (default: config.csv)")
    parser.add_argument("--scenarios", nargs="?", const="scenarios.csv", default=None,
                        help="Scenario file (default when flag given without value: scenarios.csv)")
    parser.add_argument("-S", "--selected_scenarios", nargs="+", default=None, help="Scenarios to check (default: all)")
    parser.add_argument("--simple", nargs="*", default=None, help="Same as epm.py --simple (e.g. y DiscreteCap)")
    parser.add_argument("--country", default=None, help="Same as epm.py --country")
    parser.add_argument("--modeltype", default=None, help="Override the config model type, like epm.py (default: from config)")
    parser.add_argument("--parallel", type=int, default=1, help="Scenarios compiled in parallel (default: 1)")
    parser.add_argument("--simulation_label", default=None, help="Output folder name (default: input_check_<timestamp>)")
    parser.add_argument("--gdx", default=None, help="Skip GAMS: check an existing input_treated.gdx (or a run folder)")
    parser.add_argument("--max_rows", type=int, default=25, help="Rows shown per check in the console (default: 25)")
    args = parser.parse_args(argv)

    if args.gdx:
        target = Path(args.gdx).resolve()
        if target.is_dir():
            _, any_high = check_folder(target if not (target / "input_treated.gdx").exists() else target.parent, args.max_rows)
            return 1 if any_high else 0
        sys.path.insert(0, str(BASE_DIR))
        from input_treated_check import main as check_main
        return check_main([str(target)])

    if args.scenarios is None and args.selected_scenarios not in (None, ["baseline"]):
        parser.error("-S with scenarios other than baseline requires --scenarios")

    folder, error = build_treated_inputs(args)
    if not folder.exists():
        print(f"No output folder created. {error or ''}")
        return 1
    summary, any_high = check_folder(folder, args.max_rows)
    print("\n" + "=" * 100)
    print(pd.DataFrame(summary).to_string(index=False))
    if error:
        print(f"\nGAMS reported an error for at least one scenario:\n{error[:2000]}")
    print(f"\nResults: {folder}")
    return 1 if (any_high or error) else 0


if __name__ == "__main__":
    sys.exit(main())
