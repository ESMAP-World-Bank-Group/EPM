"""
Resolution Advisor -- main CLI.

Manual mode (YAML config):
    python advise.py --config config/blacksea.yaml

Auto mode (open data -- fetches OSM, GPPD, Natural Earth automatically):
    python advise.py --countries TUR ROU BGR GEO ARM AZE --auto

Options:
    --config      Path to YAML config (manual mode)
    --countries   Space-separated ISO_A3 codes (auto mode)
    --auto        Use open data instead of manual YAML parameters
    --output      'table' (default) or 'json'
    --n_zones     Override spatial recommendation for temporal calculation
    --save        Save JSON output to output/{model_name}_recommendation.json
"""
from __future__ import annotations
import argparse
import json
import sys
import os
from pathlib import Path

# allow imports from this directory regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

from schema import AdvisorConfig, CountryConfig, ModelUse, Constraints, RegionConfig
from spatial.recommender import recommend as spatial_recommend
from temporal.recommender import recommend as temporal_recommend


# -- formatting helpers --------------------------------------------------------

def _line(char: str = "-", width: int = 72) -> str:
    return char * width

def _header(title: str, width: int = 72) -> str:
    pad = (width - len(title) - 2) // 2
    bar = "=" * width
    body = "|" + " " * pad + " " + title + " " + " " * (width - pad - len(title) - 3) + "|"
    return f"{bar}\n{body}\n{bar}"

def _box(lines: list[str], width: int = 72) -> str:
    out = ["+" + "-" * (width - 2) + "+"]
    for ln in lines:
        out.append("| " + ln.ljust(width - 3) + "|")
    out.append("+" + "-" * (width - 2) + "+")
    return "\n".join(out)


# -- main report ---------------------------------------------------------------

def run(config_path: str, output_format: str = "table",
        n_zones_override: int | None = None,
        save: bool = False):
    cfg = AdvisorConfig.from_yaml(config_path)
    _run_core(cfg, output_format, n_zones_override, save,
              provenance=None, model_name=Path(config_path).stem)


def run_auto(country_isos: list[str], output_format: str = "table",
             n_zones_override: int | None = None, save: bool = False):
    """Auto mode: fetch open data and compute parameters for each country."""
    from auto import build_country_configs

    print("[auto] Fetching open data and computing parameters...")
    configs, provenance = build_country_configs(country_isos, verbose=True)

    # Build a minimal AdvisorConfig with defaults for model_use + constraints
    region = RegionConfig(name="Auto (" + ", ".join(country_isos) + ")",
                          countries=configs)
    cfg = AdvisorConfig(
        region=region,
        model_use=ModelUse(),
        constraints=Constraints(),
    )
    model_name = "auto_" + "_".join(c.lower() for c in country_isos)
    _run_core(cfg, output_format, n_zones_override, save,
              provenance=provenance, model_name=model_name)


def _run_core(cfg, output_format, n_zones_override, save, provenance, model_name):
    spatial = spatial_recommend(cfg)
    n_zones = n_zones_override if n_zones_override else spatial.recommended
    temporal = temporal_recommend(cfg, n_zones=n_zones)

    if output_format == "json":
        _print_json(cfg, spatial, temporal, n_zones, provenance)
    else:
        _print_table(cfg, spatial, temporal, n_zones, provenance)

    if save:
        _save_json(cfg, spatial, temporal, n_zones, provenance, model_name)


def _print_table(cfg, spatial, temporal, n_zones, provenance=None):
    W = 72
    print()
    print(_header(f"Resolution Advisor -- {cfg.region.name}", W))
    print()

    # -- SPATIAL ------------------------------------------------------------
    print(_line("-", W))
    print("  SPATIAL RESOLUTION")
    print(_line("-", W))
    print()
    print(f"  {'Country':<18} {'Physical min':>14} {'Data cap':>10} {'Capped min':>11}")
    print(f"  {'-'*18} {'-'*14} {'-'*10} {'-'*11}")

    for r in spatial.country_results:
        cap_flag = " [!]" if r.data_cap_active else ""
        print(
            f"  {r.name:<18} {r.floor_physical:>14} {r.floor_capped:>10}{cap_flag}"
        )

    print()
    print(f"  Total floor (conservative):  {spatial.floor_total} zones")
    print(f"  Total ceiling (compute):     {spatial.ceiling_total} zones")
    print(f"  Recommended test points:     N in {spatial.candidates}")
    print()

    if spatial.data_caps_active:
        print(_box([
            "[!] DATA QUALITY CAP ACTIVE",
            f"    Countries where data limits resolution: {', '.join(spatial.data_caps_active)}",
            "    To improve: collect hourly zonal load or use gridflow raster approach.",
        ], W))
        print()

    print("  Per-country drivers:")
    for r in spatial.country_results:
        print(f"  {r.iso} ({r.name})")
        for d in r.drivers:
            print(f"    * {d}")
    print()

    # Auto mode: show data provenance per country
    if provenance:
        print(_line("-", W))
        print("  DATA SOURCES (auto mode)")
        print(_line("-", W))
        print()
        for iso, prov in provenance.items():
            print(f"  {iso}:")
            for param, note in prov.items():
                print(f"    {param:<28} {note}")
        print()

    print(f"  Ceiling logic: {spatial.ceiling_reason}")
    print()

    # -- TEMPORAL -----------------------------------------------------------
    print(_line("-", W))
    print("  TEMPORAL RESOLUTION")
    print(_line("-", W))
    print()
    print(f"  (Based on {n_zones} zones -- adjust with --n_zones)")
    print()
    print(f"  Representative days floor:   {temporal.floor_repr_days}")
    print(f"  Extreme days (added on top): {temporal.floor_extreme_days}")
    print(f"  Total days (floor):          {temporal.recommended_total_days}")
    print(f"  Total hours (floor):         {temporal.total_hours_at_floor}")
    print()
    print(f"  Representative days ceiling: {temporal.ceiling_repr_days}")
    print(f"  Total hours (ceiling):       {temporal.total_hours_at_ceiling}")
    print()
    print(f"  Recommended test points:     N_repr in {temporal.candidates}")
    print()

    print("  Drivers:")
    for d in temporal.drivers:
        print(f"    * {d}")
    print()

    print(f"  Ceiling logic: {temporal.ceiling_reason}")
    print()

    # -- CONVERGENCE GRID ---------------------------------------------------
    print(_line("-", W))
    print("  CONVERGENCE TEST GRID")
    print(_line("-", W))
    print()
    print("  Run EPM on these (n_zones x n_days) combinations:")
    print()
    header = f"  {'N_zones':>10}"
    for nd in temporal.candidates:
        header += f"  {'N_days=' + str(nd):>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for nz in spatial.candidates:
        row = f"  {nz:>10}"
        for nd in temporal.candidates:
            n_vars_est = nz * (nd + temporal.floor_extreme_days) * 24 * cfg.model_use.n_periods
            row += f"  {n_vars_est/1e6:>10.1f}Mv"
        print(row)
    print()
    print("  (Mv = estimated millions of variables -- guide only)")
    print()

    # -- SUMMARY ------------------------------------------------------------
    print(_line("=", W))
    print()
    print(_box([
        "  RECOMMENDATION (conservative starting point)",
        "",
        f"   Zones:  {spatial.recommended} total -- "
        + ", ".join(f"{r.iso}: {r.floor_capped}" for r in spatial.country_results),
        f"   Days:   {temporal.floor_repr_days} representative + "
        f"{temporal.floor_extreme_days} extreme = "
        f"{temporal.recommended_total_days} total ({temporal.total_hours_at_floor}h)",
        "",
        "   Validate by running EPM at 2-3 points in the convergence grid",
        "   and checking that total cost changes < 2% between levels.",
    ], W))
    print()
    print("  Next steps:")
    print("   1. Edit config/blacksea.yaml to adjust country assumptions")
    print("   2. Run: python advise.py --config config/blacksea.yaml")
    print("   3. Once satisfied, run EPM at the recommended configuration")
    print("   4. Compare with +3 zones or +4 days to confirm convergence")
    print()


def _print_json(cfg, spatial, temporal, n_zones, provenance=None):
    out = _build_json(cfg, spatial, temporal, n_zones, provenance)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def _save_json(cfg, spatial, temporal, n_zones, provenance, model_name):
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{model_name}_recommendation.json"
    out = _build_json(cfg, spatial, temporal, n_zones, provenance)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {out_path}")


def _build_json(cfg, spatial, temporal, n_zones, provenance=None) -> dict:
    return {
        "region": cfg.region.name,
        "spatial": {
            "floor_total": spatial.floor_total,
            "ceiling_total": spatial.ceiling_total,
            "candidates": spatial.candidates,
            "recommended": spatial.recommended,
            "data_caps_active": spatial.data_caps_active,
            "countries": [
                {
                    "iso": r.iso,
                    "name": r.name,
                    "floor_physical": r.floor_physical,
                    "floor_capped": r.floor_capped,
                    "data_cap_active": r.data_cap_active,
                    "drivers": r.drivers,
                    "data_provenance": (provenance or {}).get(r.iso, {}),
                }
                for r in spatial.country_results
            ],
        },
        "temporal": {
            "floor_repr_days": temporal.floor_repr_days,
            "floor_extreme_days": temporal.floor_extreme_days,
            "ceiling_repr_days": temporal.ceiling_repr_days,
            "candidates": temporal.candidates,
            "total_hours_at_floor": temporal.total_hours_at_floor,
            "total_hours_at_ceiling": temporal.total_hours_at_ceiling,
            "drivers": temporal.drivers,
        },
        "convergence_grid": {
            "n_zones_candidates": spatial.candidates,
            "n_days_candidates": temporal.candidates,
        },
    }


# -- entry point ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EPM Resolution Advisor -- recommends spatial and temporal granularity"
    )
    # Manual mode
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file (manual mode)")
    # Auto mode
    parser.add_argument("--countries", nargs="+", default=None,
                        metavar="ISO",
                        help="Country ISO_A3 codes for auto mode (e.g. TUR ROU BGR)")
    parser.add_argument("--auto", action="store_true",
                        help="Use open data instead of manual YAML parameters")
    # Common options
    parser.add_argument("--output", choices=["table", "json"], default="table")
    parser.add_argument("--n_zones", type=int, default=None,
                        help="Override spatial recommendation for temporal calculation")
    parser.add_argument("--save", action="store_true",
                        help="Save JSON output to output/<model>_recommendation.json")
    args = parser.parse_args()

    if args.auto or args.countries:
        if not args.countries:
            parser.error("--auto requires --countries ISO1 ISO2 ...")
        run_auto(args.countries, args.output, args.n_zones, args.save)
    elif args.config:
        run(args.config, args.output, args.n_zones, args.save)
    else:
        parser.error("Provide either --config <file> or --countries <ISO...> --auto")
