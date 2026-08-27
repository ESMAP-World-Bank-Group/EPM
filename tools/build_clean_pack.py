"""Build the publishable EPM v8.5 Starter pack from the original archive.

    python tools/build_clean_pack.py path/to/EPM_Starter_pack.zip

Writes EPM_v8.5_Starter_pack.zip next to the input, keeping only material that
needs no clearance decision: the model itself, the Ghana teaching example, and
the original run instructions.

Left out on purpose:
  gamslice.txt          a GAMS licence file -- redistributing it breaks the licence
  *.ppt / *.pptx        training and country decks, one of them named after its author
  the 2023 PDF          never confirmed as published material (the run manuals under
                        Running EPM/ are kept -- they are instructions, not a publication)
  Ghana Initial_model/  a real country study with real assumptions, not a template

The script refuses to write an archive that still contains any of those.
"""

import posixpath
import re
import sys
import zipfile
from pathlib import Path

KEEP = ("generic model/", "ghana example/", "documentation/running epm/")
_DROP = re.compile(
    r"""(?ix)
    gamslice | \.lic$                       # licence files
  | \.pptx?$                                # training and country decks
  | /initial_model/                         # the Ghana country study
    """
)


def excluded(path):
    """True if `path` (lowercased, leading slash) must not be published."""
    if _DROP.search(path):
        return True
    # Drop PDFs, except the run manuals -- those are instructions, not a publication.
    return path.endswith(".pdf") and "/running epm/" not in path
# Flattened so 'Running EPM' does not stay buried under Documentation/.
RENAME = {"documentation/running epm/": "Running EPM/"}


def target(name):
    """Path inside the new archive, or None if the entry is not shipped."""
    rel = name.split("/", 1)[1] if "/" in name else ""
    if not rel or rel.endswith("/"):
        return None
    low = rel.lower()
    if not low.startswith(KEEP) or excluded("/" + low):
        return None
    for prefix, replacement in RENAME.items():
        if low.startswith(prefix):
            return replacement + rel[len(prefix):]
    return rel


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    src = Path(argv[1]).expanduser().resolve()
    if not src.is_file():
        sys.exit(f"no such file: {src}")
    out = src.with_name("EPM_v8.5_Starter_pack.zip")

    root = "EPM_v8.5_Starter_pack/"
    kept, dropped = [], []
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(
        out, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for info in zin.infolist():
            if info.is_dir():
                continue
            dest = target(info.filename)
            if dest is None:
                dropped.append(info.filename)
                continue
            zout.writestr(root + dest, zin.read(info))
            kept.append(dest)

    if not kept:
        out.unlink()
        sys.exit(
            "nothing matched -- the archive layout is not what this script expects.\n"
            "Expected top-level 'Generic model/', 'Ghana example/', 'Documentation/'."
        )

    # Verify: read the archive back rather than trusting the loop above.
    with zipfile.ZipFile(out) as z:
        leaked = [n for n in z.namelist() if excluded("/" + n.lower())]
    if leaked:
        out.unlink()
        sys.exit("REFUSING TO PUBLISH -- these should not be here:\n  " + "\n  ".join(leaked))

    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"\nkept {len(kept)}:")
    for name in sorted(kept):
        print(f"  {name}")
    print(f"\ndropped {len(dropped)}:")
    for name in sorted(dropped):
        print(f"  {posixpath.basename(name.rstrip('/')) or name}")
    print("\nclean: no licence file, no slide deck, no country study")


if __name__ == "__main__":
    main(sys.argv)
