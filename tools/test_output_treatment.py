#!/usr/bin/env python3
"""
Regression tests for EPM post-processing and dashboard CSV reading.

Why this file exists
--------------------
GAMS writes its special values (EPS, UNDF, NA, +INF, -INF) into output CSVs as
plain text. A single one left in a `value` column makes pandas infer `object`
dtype, and arithmetic on an object column concatenates strings instead of adding
them. A cumulative sum over years then turns into a growing text blob: one run
produced a 122 MB pCostsMerged.csv whose numbers were simply wrong.

The fix is read-time coercion, defined once in epm/gams_values.py and used by
both the pipeline (epm/output_treatment.py) and the viewer
(dashboard/data_loader.py). These tests pin both halves of it:

  1. the behaviour  - EPS reaches the cumulative sum as 0, not as text;
  2. the plumbing   - no unprotected `pd.read_csv` creeps back into either
                      module and re-opens the hole somewhere else.

Test 2 is the one that matters over time. The bug came back once already, on a
study branch that had tried `fillna(0)` instead - which cannot work, because
'EPS' is a string, not a NaN. The dashboard had the mirror-image mistake:
pd.to_numeric(errors='coerce') alone, which maps EPS to NaN and drops a stored
zero out of the charts.

Run directly (exits non-zero on failure) or under pytest:

    python tools/test_output_treatment.py
"""

import ast
import os
import sys
import tempfile

import pandas as pd

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(TOOLS_DIR)
EPM_DIR = os.path.join(BASE_DIR, "epm")
MODULE_PATH = os.path.join(EPM_DIR, "output_treatment.py")

sys.path.insert(0, EPM_DIR)
import output_treatment as ot  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Behaviour: special values must not survive as text
# ---------------------------------------------------------------------------

def test_coerce_translates_gams_special_values():
    """Every documented GAMS special value maps to a number, not to text."""
    df = pd.DataFrame({"value": ["EPS", "1.5", "UNDF", "-INF", "+INF", "NA"]})
    out = ot.coerce_value_column(df.copy(), "value", log_func=lambda m: None)

    assert pd.api.types.is_numeric_dtype(out["value"]), out["value"].dtype
    assert out["value"][0] == 0.0, "EPS is a stored zero, not a missing value"
    assert out["value"][1] == 1.5
    assert pd.isna(out["value"][2]), "UNDF is undefined -> NaN"
    assert out["value"][3] == float("-inf")
    assert out["value"][4] == float("inf")
    assert pd.isna(out["value"][5])


def test_unknown_token_becomes_nan_not_text():
    """An unexpected token degrades to NaN so arithmetic stays numeric."""
    df = pd.DataFrame({"value": ["1", "WAT", "3"]})
    out = ot.coerce_value_column(df.copy(), "value", log_func=lambda m: None)

    assert pd.api.types.is_numeric_dtype(out["value"])
    assert pd.isna(out["value"][1])


def test_cumulative_sum_adds_instead_of_concatenating():
    """The original symptom: a cumsum over a column holding one EPS.

    Without coercion pandas yields ['1', '12', '12EPS', '12EPS4'] - each year
    re-appending the whole history. With it, the running total is arithmetic.
    """
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "pDiscountedWeightedCosts.csv")
    dst = os.path.join(tmp, "pDiscountedWeightedCostsCumulated.csv")

    pd.DataFrame({
        "z": ["Angola_Central"] * 4,
        "uni": ["Carbon costs: $m"] * 4,
        "y": [2025, 2026, 2027, 2028],
        "value": ["1", "2", "EPS", "4"],
    }).to_csv(src, index=False)

    # Not `dtype == object`: pandas 3 reads text columns as `str`, and either way
    # what matters is that the column is not numeric, which is what breaks cumsum.
    assert not pd.api.types.is_numeric_dtype(
        pd.read_csv(src)["value"]
    ), "test data must trip dtype inference"
    assert ot.calculate_cumulative(src, dst, log_func=lambda m: None)

    got = pd.read_csv(dst).sort_values("y")["value"].tolist()
    assert got == [1.0, 3.0, 3.0, 7.0], got


def test_cumulative_output_stays_compact():
    """Guards the blow-up itself: concatenation shows up as absurd field width."""
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "in.csv")
    dst = os.path.join(tmp, "out.csv")

    years = list(range(2025, 2055))
    pd.DataFrame({
        "z": ["Z"] * len(years),
        "uni": ["Fuel costs: $m"] * len(years),
        "y": years,
        "value": ["EPS"] + ["1.2345678901234567"] * (len(years) - 1),
    }).to_csv(src, index=False)

    assert ot.calculate_cumulative(src, dst, log_func=lambda m: None)

    widest = pd.read_csv(dst, dtype=str)["value"].str.len().max()
    assert widest < 30, f"value column blew up to {widest} chars - concatenation is back"


# ---------------------------------------------------------------------------
# 2. Plumbing: every value column must pass through the shared coercion
# ---------------------------------------------------------------------------

# `pd.read_csv` is only legitimate where the value column cannot be corrupted:
# inside the coercing wrapper itself, immediately wrapped by
# coerce_value_column(...), or on a file that has no value column at all.
GUARDED_MODULES = {
    os.path.join(EPM_DIR, "gams_values.py"): {
        # the wrapper that does the coercion
        "functions": {"read_output_csv"},
        "first_args": set(),
    },
    MODULE_PATH: {
        "functions": set(),
        # tech/fuel mapping table - no value column
        "first_args": {"TECHFUEL_PROCESSING_PATH"},
    },
    os.path.join(BASE_DIR, "dashboard", "data_loader.py"): {
        # scenarios.csv maps parameter names to override paths; the two
        # scenario-column helpers read and rewrite input files as raw text
        "functions": {"load_input_scenarios", "add_scenario_column",
                      "save_input_scenarios"},
        "first_args": set(),
    },
}

WRAPPING_CALLS = {"coerce_value_column"}


def _enclosing_function(tree, node):
    """Name of the innermost function containing `node`, or None."""
    best = None
    for candidate in ast.walk(tree):
        if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(candidate, "end_lineno", candidate.lineno)
        if candidate.lineno <= node.lineno <= end:
            if best is None or candidate.lineno > best.lineno:
                best = candidate
    return best.name if best else None


def _wrapped_read_csv_nodes(tree):
    """read_csv calls passed straight into coerce_value_column(...)."""
    wrapped = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else None)
        if name not in WRAPPING_CALLS:
            continue
        for arg in node.args:
            wrapped.add(id(arg))
    return wrapped


def test_no_unprotected_read_csv():
    """Fail if a value column is read without going through the coercion."""
    problems = []

    for path, rules in GUARDED_MODULES.items():
        rel = os.path.relpath(path, BASE_DIR).replace("\\", "/")
        assert os.path.exists(path), f"{rel} is missing"

        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        wrapped = _wrapped_read_csv_nodes(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "read_csv"):
                continue

            if id(node) in wrapped:
                continue
            if _enclosing_function(tree, node) in rules["functions"]:
                continue
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Name) and first.id in rules["first_args"]:
                continue

            problems.append(f"{rel}:{node.lineno}")

    assert not problems, (
        "unprotected pd.read_csv at " + ", ".join(problems) + " - wrap it in "
        "coerce_value_column(...) (or use read_output_csv), otherwise GAMS "
        "special values reach pandas as text and aggregations concatenate. "
        "If the file genuinely has no value column, add its function to "
        "GUARDED_MODULES."
    )


def test_dashboard_reads_eps_as_zero():
    """The viewer must agree with the pipeline: EPS is a stored zero.

    pd.to_numeric(errors='coerce') alone turns EPS into NaN, and load_yearly_zone
    drops NaN rows - so a real result would disappear from the charts.
    """
    loader = os.path.join(BASE_DIR, "dashboard", "data_loader.py")
    with open(loader, encoding="utf-8") as handle:
        source = handle.read()

    assert "gams_values.py" in source, (
        "dashboard/data_loader.py must share epm/gams_values.py, not re-implement it"
    )
    assert 'pd.to_numeric(df["value"], errors="coerce")' not in source, (
        "pd.to_numeric alone maps EPS to NaN; use coerce_value_column"
    )


def test_wrapper_is_actually_used():
    """A sanity check that the wrapper exists and is wired in."""
    assert hasattr(ot, "read_output_csv")
    assert hasattr(ot, "coerce_value_column")
    assert ot.GAMS_SPECIAL_VALUES["EPS"] == 0.0

    with open(MODULE_PATH, encoding="utf-8") as handle:
        source = handle.read()
    assert source.count("read_output_csv(") > 5, "wrapper imported but barely used"

    # The failed fix that let the bug survive on a study branch. It cannot work:
    # 'EPS' is a string, so fillna never sees it and the column stays object.
    assert "fillna(0).cumsum()" not in source, (
        "fillna(0) does not neutralise 'EPS' - it is a string, not a NaN"
    )


# ---------------------------------------------------------------------------

def main():
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {test.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {test.__name__}")

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
