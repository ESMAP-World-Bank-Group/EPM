# -*- coding: utf-8 -*-
"""Which EPM run the report is built from.

Every module resolves the run through here, so moving the report onto a new run
is one line (or the BS_RUN environment variable) instead of five hard-coded
paths scattered across the generators.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                          # .../blacksea_2026/EPM
OUTVIEW = ROOT / "epm" / "output_view"
CACHEDIR = HERE / "cache"

DEFAULT_RUN = os.environ.get("BS_RUN", "simulations_run_20260906")


def run_dir(name=None):
    return OUTVIEW / (name or DEFAULT_RUN)


def cache_path(name=None):
    return CACHEDIR / ("%s.json" % (name or DEFAULT_RUN))
