#!/usr/bin/env python3
"""
Report which branches carry the post-processing fixes.

epm/output_treatment.py is a shared tool living in a repository whose branches
are studies. Every study forks the tool and drifts, so a fix landing on main
reaches nobody else until someone notices - which is how the GAMS special-value
bug (EPS read as text, cumsum concatenating strings instead of adding them)
resurfaced months later on a branch that never received it.

This makes the drift visible in one command instead of by accident.

Usage:
    python tools/audit_postprocessing_sync.py            # remote branches
    python tools/audit_postprocessing_sync.py --local    # local branches
    python tools/audit_postprocessing_sync.py --fetch    # git fetch first

Exit code is 0 when every branch carrying output_treatment.py has the fix,
1 otherwise, so it can be used as a check.
"""

import argparse
import subprocess
import sys

MODULE = "epm/output_treatment.py"
TESTS = "tools/test_output_treatment.py"

# What a branch must contain for the fix to be present.
FIX_MARKER = "GAMS_SPECIAL_VALUES"

# The attempt that does not work: 'EPS' is a string, so fillna never sees it,
# the column stays object dtype and cumsum still concatenates.
BROKEN_ATTEMPT = "fillna(0).cumsum()"

FIX_COMMIT = "d08d417d"


def git(*args, check=True):
    """Run a git command and return stdout, or None when it fails."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        if check:
            return None
        return None
    return result.stdout


def list_branches(scope):
    if scope == "local":
        out = git("for-each-ref", "--format=%(refname:short)", "refs/heads") or ""
        return [line for line in out.splitlines() if line]

    # refs/remotes/origin/HEAD is a symref whose short name is plain "origin";
    # it duplicates the default branch, so drop it.
    out = git("for-each-ref", "--format=%(refname:short)\t%(symref)", "refs/remotes/origin") or ""
    branches = []
    for line in out.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        name = parts[0]
        symref = parts[1] if len(parts) > 1 else ""
        if symref or name == "origin":
            continue
        branches.append(name)
    return branches


def audit(scope):
    rows = []
    missing = 0

    for ref in list_branches(scope):
        short = ref[len("origin/"):] if ref.startswith("origin/") else ref
        date = (git("log", "-1", "--format=%ad", "--date=short", ref) or "").strip()

        source = git("show", f"{ref}:{MODULE}")
        if source is None:
            rows.append((short, date, "-", "-", "no post-processing on this branch"))
            continue

        has_fix = FIX_MARKER in source
        has_tests = git("cat-file", "-e", f"{ref}:{TESTS}") is not None

        if not has_fix:
            missing += 1

        if BROKEN_ATTEMPT in source:
            note = "has fillna(0) - ineffective, supersede it"
        elif not has_fix:
            note = "cherry-pick the fix"
        else:
            note = ""

        rows.append((short, date, "yes" if has_fix else "NO",
                     "yes" if has_tests else "no", note))

    return rows, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="audit local branches")
    parser.add_argument("--fetch", action="store_true", help="git fetch --all first")
    args = parser.parse_args()

    if git("rev-parse", "--git-dir") is None:
        print("not inside a git repository", file=sys.stderr)
        return 2

    if args.fetch:
        git("fetch", "--all", "--quiet")

    rows, missing = audit("local" if args.local else "remote")

    header = f"{'BRANCH':<40} {'LAST COMMIT':<12} {'FIX':<5} {'TESTS':<6} NOTE"
    print(header)
    print("-" * 90)
    for short, date, fix, tests, note in rows:
        print(f"{short:<40} {date:<12} {fix:<5} {tests:<6} {note}")

    print()
    if missing == 0:
        print(f"All branches carrying {MODULE} have the GAMS special-value fix.")
        return 0

    print(f"{missing} branch(es) carry {MODULE} without the GAMS special-value fix.")
    print("Runs produced from those branches can hold wrong cumulative cost values.")
    print()
    print("To bring one up to date:")
    print("    git checkout <branch>")
    print(f"    git cherry-pick {FIX_COMMIT}      # read-time coercion in {MODULE}")
    print("    # If it conflicts, it will be in calculate_cumulative: keep main's")
    print("    # version, which replaces the fillna(0) attempt with the coercion.")
    print()
    print("See CHANGELOG.md on main for what the bug was.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
