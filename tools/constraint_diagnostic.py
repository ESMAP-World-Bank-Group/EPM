"""
Constraint Diagnostic Tool for EPM
==================================

Analyzes GAMS optimization results to identify potentially problematic constraints.
This tool reads GDX files and performs statistical analysis to flag:
- Extreme marginal prices (shadow prices)
- Binding constraints with unusual patterns
- Variables hitting bounds
- Slack variables with non-zero values
- Energy balance anomalies

Usage:
    python tools/constraint_diagnostic.py --folder output/simulations_test/baseline
    python tools/constraint_diagnostic.py --folder output/simulations_test/baseline --debug
    python tools/constraint_diagnostic.py --folder output/simulations_test/baseline --output report.txt

Author: EPM Team
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gams.transfer as gt
except ImportError:
    print("ERROR: gams.transfer not available. Please install GAMS Python API.")
    sys.exit(1)


class ConstraintDiagnostic:
    """Analyze EPM optimization results for constraint issues."""

    # Thresholds for flagging issues
    PRICE_EXTREME_LOW = -1000  # $/MWh - very negative price
    PRICE_EXTREME_HIGH = 1000  # $/MWh - very high price
    PRICE_VOLL = 10000  # $/MWh - Value of Lost Load indicator
    MARGINAL_EXTREME = 1e6  # Very large marginal
    ZERO_THRESHOLD = 1e-6  # Consider as zero
    CAPACITY_UTIL_LOW = 0.01  # 1% - suspiciously low utilization
    CAPACITY_UTIL_HIGH = 0.999  # 99.9% - hitting capacity limit

    def __init__(self, folder: str, debug: bool = False):
        """
        Initialize diagnostic tool.

        Parameters
        ----------
        folder : str
            Path to the scenario folder containing GDX files
        debug : bool
            If True, look for PA.gdx with full equation marginals
        """
        self.folder = Path(folder)
        self.debug = debug
        self.issues: List[Dict] = []
        self.warnings: List[Dict] = []
        self.data: Dict[str, pd.DataFrame] = {}

        # Determine which GDX file to use
        self.gdx_file = self._find_gdx_file()

    def _find_gdx_file(self) -> Path:
        """Find the appropriate GDX file to analyze."""
        # Priority: PA.gdx (debug) > epmresults.gdx
        pa_gdx = self.folder / "PA.gdx"
        results_gdx = self.folder / "epmresults.gdx"

        if self.debug and pa_gdx.exists():
            print(f"Using debug GDX file: {pa_gdx}")
            return pa_gdx
        elif results_gdx.exists():
            print(f"Using results GDX file: {results_gdx}")
            return results_gdx
        elif pa_gdx.exists():
            print(f"Using PA.gdx file: {pa_gdx}")
            return pa_gdx
        else:
            raise FileNotFoundError(
                f"No GDX file found in {self.folder}. "
                "Expected PA.gdx or epmresults.gdx"
            )

    def load_data(self) -> None:
        """Load data from GDX file."""
        print(f"\nLoading data from {self.gdx_file}...")

        container = gt.Container(str(self.gdx_file))

        # Load parameters
        for param in container.getParameters():
            try:
                records = container.data[param.name].records
                if records is not None and len(records) > 0:
                    self.data[param.name] = records.copy()
            except Exception as e:
                if self.debug:
                    print(f"  Warning: Could not load parameter {param.name}: {e}")

        # Load variables (for .l levels and bounds)
        for var in container.getVariables():
            try:
                records = container.data[var.name].records
                if records is not None and len(records) > 0:
                    self.data[f"var_{var.name}"] = records.copy()
            except Exception as e:
                if self.debug:
                    print(f"  Warning: Could not load variable {var.name}: {e}")

        # Load equations (for marginals) - only available in PA.gdx
        for eq in container.getEquations():
            try:
                records = container.data[eq.name].records
                if records is not None and len(records) > 0:
                    self.data[f"eq_{eq.name}"] = records.copy()
            except Exception as e:
                if self.debug:
                    print(f"  Warning: Could not load equation {eq.name}: {e}")

        print(f"  Loaded {len(self.data)} data objects")

        # Print available equation marginals if in debug mode
        if self.debug:
            eq_names = [k for k in self.data.keys() if k.startswith("eq_")]
            if eq_names:
                print(f"  Found {len(eq_names)} equations with marginals")

    def _add_issue(self, severity: str, category: str, description: str,
                   details: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        """Add an issue to the findings."""
        issue = {
            "severity": severity,  # "ERROR", "WARNING", "INFO"
            "category": category,
            "description": description,
            "details": details,
            "data": data
        }
        if severity == "ERROR":
            self.issues.append(issue)
        else:
            self.warnings.append(issue)

    def check_prices(self) -> None:
        """Analyze electricity prices for anomalies."""
        print("\n[1/8] Checking electricity prices...")

        if "pPrice" not in self.data:
            print("  Skipping: pPrice not found in GDX")
            return

        df = self.data["pPrice"].copy()

        # Rename value column if needed
        if "value" in df.columns:
            prices = df["value"]
        elif "level" in df.columns:
            prices = df["level"]
        else:
            print("  Skipping: Cannot find price values")
            return

        # Basic statistics
        print(f"  Price statistics:")
        print(f"    Min: {prices.min():.2f} $/MWh")
        print(f"    Max: {prices.max():.2f} $/MWh")
        print(f"    Mean: {prices.mean():.2f} $/MWh")
        print(f"    Std: {prices.std():.2f} $/MWh")

        # Check for extreme negative prices
        neg_prices = df[prices < self.PRICE_EXTREME_LOW]
        if len(neg_prices) > 0:
            self._add_issue(
                "WARNING", "Price",
                f"Found {len(neg_prices)} hours with extremely negative prices (< {self.PRICE_EXTREME_LOW} $/MWh)",
                f"Minimum price: {prices.min():.2f} $/MWh. This may indicate excess generation or curtailment issues.",
                neg_prices.head(10)
            )

        # Check for VOLL prices (unmet demand)
        voll_prices = df[prices >= self.PRICE_VOLL]
        if len(voll_prices) > 0:
            self._add_issue(
                "ERROR", "Price",
                f"Found {len(voll_prices)} hours with VOLL-level prices (>= {self.PRICE_VOLL} $/MWh)",
                "This indicates unmet demand in these hours. Check capacity adequacy.",
                voll_prices.head(10)
            )

        # Check for zero prices
        zero_prices = df[abs(prices) < self.ZERO_THRESHOLD]
        zero_pct = len(zero_prices) / len(df) * 100
        if zero_pct > 50:
            self._add_issue(
                "WARNING", "Price",
                f"{zero_pct:.1f}% of hours have zero or near-zero prices",
                "This may indicate over-capacity or model configuration issues."
            )

        # Check for price volatility by zone
        if "z" in df.columns:
            for zone in df["z"].unique():
                zone_prices = prices[df["z"] == zone]
                cv = zone_prices.std() / zone_prices.mean() if zone_prices.mean() != 0 else 0
                if cv > 2:
                    self._add_issue(
                        "INFO", "Price",
                        f"Zone {zone} has high price volatility (CV={cv:.2f})",
                        "High coefficient of variation may indicate capacity constraints."
                    )

    def check_unmet_demand(self) -> None:
        """Check for unmet demand (load shedding)."""
        print("\n[2/8] Checking unmet demand...")

        # Look for unmet demand variable or parameter
        unmet_vars = ["var_vUnmetDemand", "pUnmetDemand", "var_vUSE"]

        for var_name in unmet_vars:
            if var_name in self.data:
                df = self.data[var_name]
                val_col = "level" if "level" in df.columns else "value"

                if val_col in df.columns:
                    unmet = df[df[val_col] > self.ZERO_THRESHOLD]
                    if len(unmet) > 0:
                        total_unmet = unmet[val_col].sum()
                        self._add_issue(
                            "ERROR", "Unmet Demand",
                            f"Found unmet demand totaling {total_unmet:.2f} MWh",
                            f"Unmet demand in {len(unmet)} time periods. Check generation capacity and constraints.",
                            unmet.head(10)
                        )
                    else:
                        print(f"  No unmet demand found (checked {var_name})")
                    return

        print("  Skipping: Unmet demand variable not found")

    def check_capacity_utilization(self) -> None:
        """Check for suspicious capacity utilization patterns."""
        print("\n[3/8] Checking capacity utilization...")

        # Check dispatch vs capacity
        if "pDispatchPlant" in self.data and "pCapacityPlant" in self.data:
            dispatch = self.data["pDispatchPlant"]
            capacity = self.data["pCapacityPlant"]

            # This would need proper merging based on plant and year
            print("  Dispatch and capacity data available for detailed analysis")

        # Check for generation hitting capacity
        if "var_vPwrOut" in self.data and "var_vCap" in self.data:
            gen = self.data["var_vPwrOut"]
            cap = self.data["var_vCap"]
            print("  Generation and capacity variables available")

        # Alternative: check energy balance
        if "pEnergyBalance" in self.data:
            df = self.data["pEnergyBalance"]
            print(f"  Energy balance data found with {len(df)} records")

    def check_transmission_constraints(self) -> None:
        """Check transmission flow patterns."""
        print("\n[4/8] Checking transmission constraints...")

        flow_vars = ["var_vFlow", "pInterchange", "pFlowMWh"]

        for var_name in flow_vars:
            if var_name in self.data:
                df = self.data[var_name]
                val_col = "level" if "level" in df.columns else "value"

                if val_col in df.columns:
                    # Check for zero flows (may indicate disconnected zones)
                    flows = df[val_col]
                    zero_flows = (abs(flows) < self.ZERO_THRESHOLD).sum()
                    zero_pct = zero_flows / len(flows) * 100

                    print(f"  {var_name}: {zero_pct:.1f}% of flows are zero")

                    if zero_pct > 90:
                        self._add_issue(
                            "WARNING", "Transmission",
                            f">{zero_pct:.0f}% of transmission flows are zero",
                            "Check if transmission topology is correctly defined."
                        )
                return

        print("  Skipping: Flow variables not found")

    def check_storage_operation(self) -> None:
        """Check storage operation patterns."""
        print("\n[5/8] Checking storage operation...")

        storage_vars = ["var_vSOC", "var_vCharge", "var_vCapStor"]
        found = False

        for var_name in storage_vars:
            if var_name in self.data:
                found = True
                df = self.data[var_name]
                val_col = "level" if "level" in df.columns else "value"

                if val_col in df.columns:
                    values = df[val_col]
                    print(f"  {var_name}: min={values.min():.2f}, max={values.max():.2f}")

                    # Check for always-empty storage
                    if values.max() < self.ZERO_THRESHOLD:
                        self._add_issue(
                            "WARNING", "Storage",
                            f"Storage variable {var_name} is always zero",
                            "Storage may not be operating. Check storage constraints and economics."
                        )

        if not found:
            print("  No storage variables found (storage may be disabled)")

    def check_reserve_constraints(self) -> None:
        """Check reserve requirement satisfaction."""
        print("\n[6/8] Checking reserve constraints...")

        reserve_vars = [
            "var_vUnmetCountrySpinningReserve",
            "var_vUnmetSystemSpinningReserve",
            "var_vUnmetCountryPlanningReserve",
            "var_vUnmetSystemPlanningReserve"
        ]

        for var_name in reserve_vars:
            if var_name in self.data:
                df = self.data[var_name]
                val_col = "level" if "level" in df.columns else "value"

                if val_col in df.columns:
                    unmet = df[df[val_col] > self.ZERO_THRESHOLD]
                    if len(unmet) > 0:
                        total_unmet = unmet[val_col].sum()
                        short_name = var_name.replace("var_v", "")
                        self._add_issue(
                            "WARNING", "Reserves",
                            f"Unmet {short_name}: {total_unmet:.2f} MW",
                            f"Reserve shortfall in {len(unmet)} periods.",
                            unmet.head(5)
                        )

        print("  Reserve check complete")

    def check_equation_marginals(self) -> None:
        """Check equation marginals for anomalies (requires PA.gdx)."""
        print("\n[7/8] Checking equation marginals...")

        eq_data = {k: v for k, v in self.data.items() if k.startswith("eq_")}

        if not eq_data:
            print("  Skipping: No equation data found (run with --debug and PA.gdx)")
            return

        print(f"  Found {len(eq_data)} equations to analyze")

        extreme_marginals = []

        for eq_name, df in eq_data.items():
            if "marginal" not in df.columns:
                continue

            marginals = df["marginal"]

            # Skip if all zeros (non-binding)
            if (abs(marginals) < self.ZERO_THRESHOLD).all():
                continue

            # Check for extreme marginals
            max_marginal = marginals.abs().max()
            if max_marginal > self.MARGINAL_EXTREME:
                extreme_marginals.append({
                    "equation": eq_name.replace("eq_", ""),
                    "max_abs_marginal": max_marginal,
                    "binding_count": (abs(marginals) > self.ZERO_THRESHOLD).sum()
                })

        if extreme_marginals:
            # Sort by max marginal
            extreme_marginals.sort(key=lambda x: x["max_abs_marginal"], reverse=True)

            self._add_issue(
                "WARNING", "Marginals",
                f"Found {len(extreme_marginals)} equations with extreme marginals (> {self.MARGINAL_EXTREME})",
                "Large marginals indicate tight constraints that strongly affect the objective.",
                pd.DataFrame(extreme_marginals[:10])
            )

            # Print top offenders
            print("  Top equations with extreme marginals:")
            for item in extreme_marginals[:5]:
                print(f"    {item['equation']}: max_marginal={item['max_abs_marginal']:.2e}")
        else:
            print("  No extreme marginals found")

    def check_cost_components(self) -> None:
        """Check cost components for anomalies."""
        print("\n[8/8] Checking cost components...")

        cost_params = ["pCostsSystem", "pYearlyCostsZone", "pCostsZone"]

        for param_name in cost_params:
            if param_name in self.data:
                df = self.data[param_name]
                val_col = "value" if "value" in df.columns else "level"

                if val_col in df.columns:
                    # Look for negative costs (usually wrong)
                    neg_costs = df[df[val_col] < -self.ZERO_THRESHOLD]
                    if len(neg_costs) > 0:
                        self._add_issue(
                            "WARNING", "Costs",
                            f"Found negative cost components in {param_name}",
                            "Negative costs may indicate model issues unless they're revenues.",
                            neg_costs.head(10)
                        )

                    print(f"  {param_name}: Found {len(df)} cost records")
                return

        print("  Skipping: Cost parameters not found")

    def run_all_checks(self) -> None:
        """Run all diagnostic checks."""
        self.load_data()

        print("\n" + "="*60)
        print("RUNNING DIAGNOSTICS")
        print("="*60)

        self.check_prices()
        self.check_unmet_demand()
        self.check_capacity_utilization()
        self.check_transmission_constraints()
        self.check_storage_operation()
        self.check_reserve_constraints()
        self.check_equation_marginals()
        self.check_cost_components()

    def generate_report(self) -> str:
        """Generate a summary report of findings."""
        lines = []
        lines.append("\n" + "="*60)
        lines.append("DIAGNOSTIC REPORT")
        lines.append("="*60)
        lines.append(f"Folder: {self.folder}")
        lines.append(f"GDX File: {self.gdx_file.name}")
        lines.append("")

        # Summary
        n_errors = len(self.issues)
        n_warnings = len(self.warnings)

        if n_errors == 0 and n_warnings == 0:
            lines.append("✓ No issues found")
        else:
            lines.append(f"Found {n_errors} ERROR(s) and {n_warnings} WARNING(s)")

        # Errors first
        if self.issues:
            lines.append("\n" + "-"*40)
            lines.append("ERRORS")
            lines.append("-"*40)
            for i, issue in enumerate(self.issues, 1):
                lines.append(f"\n[ERROR {i}] {issue['category']}: {issue['description']}")
                if issue['details']:
                    lines.append(f"  → {issue['details']}")
                if issue['data'] is not None:
                    lines.append("  Sample data:")
                    for line in str(issue['data']).split('\n')[:8]:
                        lines.append(f"    {line}")

        # Then warnings
        if self.warnings:
            lines.append("\n" + "-"*40)
            lines.append("WARNINGS")
            lines.append("-"*40)
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"\n[{warning['severity']} {i}] {warning['category']}: {warning['description']}")
                if warning['details']:
                    lines.append(f"  → {warning['details']}")
                if warning['data'] is not None:
                    lines.append("  Sample data:")
                    for line in str(warning['data']).split('\n')[:5]:
                        lines.append(f"    {line}")

        lines.append("\n" + "="*60)
        lines.append("END OF REPORT")
        lines.append("="*60)

        return "\n".join(lines)

    def suggest_investigations(self) -> str:
        """Provide suggestions based on findings."""
        lines = []
        lines.append("\n" + "-"*40)
        lines.append("SUGGESTED INVESTIGATIONS")
        lines.append("-"*40)

        # Group issues by category
        categories = set()
        for issue in self.issues + self.warnings:
            categories.add(issue['category'])

        suggestions = {
            "Price": [
                "• Check pDemandData for demand levels",
                "• Review pGenData for capacity and costs",
                "• Examine pFuelPrice for fuel cost assumptions",
                "• Look at transmission constraints if prices differ by zone"
            ],
            "Unmet Demand": [
                "• Verify sufficient generation capacity exists (pGenData)",
                "• Check if reserve margins are too high (pPlanningReserveMargin)",
                "• Review renewable capacity factors (pGenProfile)",
                "• Examine transmission limits between zones"
            ],
            "Transmission": [
                "• Check pTransmissionCapacity for line limits",
                "• Verify pTopology defines correct connections",
                "• Review pTransmissionLosses",
                "• Look for binding transmission constraints in marginals"
            ],
            "Storage": [
                "• Verify storage is enabled (fEnableStorage)",
                "• Check pStorageDataInput for storage parameters",
                "• Review storage round-trip efficiency",
                "• Compare storage cost vs price arbitrage opportunity"
            ],
            "Reserves": [
                "• Review pSpinningReserveReq settings",
                "• Check pPlanningReserveMargin",
                "• Verify generator reserve capability in pGenData",
                "• Consider if reserve requirements are too stringent"
            ],
            "Marginals": [
                "• Extreme marginals indicate binding constraints",
                "• Consider relaxing constraints with very high marginals",
                "• Check if the constraint definition is correct",
                "• Review if input data is realistic"
            ],
            "Costs": [
                "• Review pGenData capital and O&M costs",
                "• Check pFuelPrice trajectories",
                "• Verify discount rate (pDiscountRate)",
                "• Examine cost breakdown in pCostsPlant"
            ]
        }

        for category in categories:
            if category in suggestions:
                lines.append(f"\n{category}:")
                for suggestion in suggestions[category]:
                    lines.append(f"  {suggestion}")

        if not categories:
            lines.append("No specific suggestions - results look reasonable.")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose EPM optimization results for constraint issues"
    )
    parser.add_argument(
        "--folder", "-f",
        required=True,
        help="Path to scenario folder containing GDX files"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Use PA.gdx for full equation marginals (requires DEBUG=1 run)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for report (default: print to console)"
    )

    args = parser.parse_args()

    # Resolve folder path
    folder = Path(args.folder)
    if not folder.is_absolute():
        # Try relative to current directory and epm/output
        if folder.exists():
            pass
        elif (Path("epm/output") / folder).exists():
            folder = Path("epm/output") / folder
        elif (Path("epm") / folder).exists():
            folder = Path("epm") / folder

    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)

    # Run diagnostics
    diag = ConstraintDiagnostic(str(folder), debug=args.debug)

    try:
        diag.run_all_checks()
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Generate report
    report = diag.generate_report()
    suggestions = diag.suggest_investigations()

    full_report = report + suggestions

    if args.output:
        with open(args.output, "w") as f:
            f.write(full_report)
        print(f"\nReport written to: {args.output}")
    else:
        print(full_report)

    # Exit with error code if issues found
    sys.exit(1 if diag.issues else 0)


if __name__ == "__main__":
    main()
