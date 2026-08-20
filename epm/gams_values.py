"""
**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
Description:
    GAMS special-value handling for everything that reads EPM output CSVs.

    GAMS exports its special values as plain text (EPS, UNDF, NA, +INF, -INF).
    A single one left in a value column makes pandas infer 'object' dtype, and
    arithmetic on an object column concatenates strings instead of adding them:
    a cumulative sum over years becomes a growing text blob rather than a
    running total. One run produced a 122 MB pCostsMerged.csv whose numbers
    were simply wrong.

    The two obvious workarounds are both wrong:
      - fillna(0)                  -> 'EPS' is a string, not a NaN; never fires
      - pd.to_numeric(coerce only) -> turns EPS into NaN, i.e. drops a stored
                                      zero that GAMS deliberately kept

    So the translation lives here, once, and every reader goes through it:
    epm/output_treatment.py for the pipeline, dashboard/data_loader.py for the
    viewer. Keeping a single definition is the point of the module - a second
    copy is how the bug came back the first time.

Author(s):
    ESMAP Modelling Team

Organization:
    World Bank

License:
    Creative Commons Zero v1.0 Universal
**********************************************************************
"""

import os
from typing import Callable

import pandas as pd


# EPS is a *stored* zero: GAMS wrote it on purpose to distinguish "computed to
# zero" from "absent". It must arrive as 0.0, never as NaN, or downstream
# dropna() silently removes real results.
GAMS_SPECIAL_VALUES = {
    'EPS': 0.0,
    'UNDF': float('nan'),
    'NA': float('nan'),
    'INF': float('inf'),
    '+INF': float('inf'),
    '-INF': float('-inf'),
}


def _default_log(message: str) -> None:
    """Default logging function that prints to stdout."""
    print(message)


def coerce_value_column(
    df: pd.DataFrame,
    value_col: str = 'value',
    source: str = '',
    log_func: Callable[[str], None] = _default_log
) -> pd.DataFrame:
    """
    Force `value_col` to a numeric dtype, translating GAMS special values.

    Unrecognised non-numeric tokens become NaN and are reported, so a malformed
    export surfaces as a warning instead of silently corrupting later arithmetic.
    Returns the frame unchanged when there is no value column or it is already
    numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to coerce (modified in place and returned)
    value_col : str
        Name of the numeric column (default: 'value')
    source : str
        File name used in log messages
    log_func : callable
        Logging function (default: print)
    """
    if value_col not in df.columns or pd.api.types.is_numeric_dtype(df[value_col]):
        return df

    numeric = pd.to_numeric(df[value_col], errors='coerce')
    unparsed = numeric.isna() & df[value_col].notna()

    if unparsed.any():
        tokens = df[value_col][unparsed].astype(str).str.strip().str.upper()
        numeric.loc[unparsed] = tokens.map(GAMS_SPECIAL_VALUES)

        counts = tokens.value_counts()
        known = {t: n for t, n in counts.items() if t in GAMS_SPECIAL_VALUES}
        unknown = {t: n for t, n in counts.items() if t not in GAMS_SPECIAL_VALUES}
        label = f"{source}: " if source else ""

        if known:
            detail = ', '.join(f"{t}={n}" for t, n in known.items())
            log_func(f"[gams_values]   {label}translated GAMS special values "
                     f"in '{value_col}' ({detail})")
        if unknown:
            detail = ', '.join(f"{t}={n}" for t, n in list(unknown.items())[:5])
            log_func(f"[gams_values]   {label}WARNING - {sum(unknown.values())} "
                     f"non-numeric value(s) in '{value_col}' set to NaN ({detail})")

    df[value_col] = numeric
    return df


def read_output_csv(
    path: str,
    value_col: str = 'value',
    log_func: Callable[[str], None] = _default_log,
    **kwargs
) -> pd.DataFrame:
    """
    Read an EPM output CSV with `value_col` guaranteed numeric.

    Every output CSV read by the post-processing pipeline goes through here, so
    that merges, cumulative sums and group aggregations downstream can rely on
    the value column being a real numeric dtype.
    """
    df = pd.read_csv(path, **kwargs)
    return coerce_value_column(
        df, value_col=value_col, source=os.path.basename(path), log_func=log_func
    )
